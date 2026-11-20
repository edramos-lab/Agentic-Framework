"""Step 5b — Content Refinement Agent.

Iteratively revises the manuscript based on AgentReview feedback
until the gains stop (or max cycles reached).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..base_agent import BaseAgent
from ..utils.latex import dedupe_floats, disable_unused_bibliography, restrict_citations, strip_fences
from ..models import (
    ManuscriptDraft,
    PipelineState,
    RefinementOutput,
    ReviewScore,
    ReviewVerdict,
)
from .review_agent import ReviewAgent

REFINE_SYSTEM = """\
You are a Content Refinement agent for academic papers.  Given:
1. A LaTeX manuscript
2. Reviewer feedback with specific suggestions

Revise the manuscript to address ALL suggestions.  Improve clarity,
strengthen arguments, fix any issues.  Keep the overall structure intact.

Rules:
- Keep every number, table and equation exactly as given.
- Do not add citations; keep only the \\cite keys already in the manuscript.
- Do not claim results, training or experiments the manuscript does not already report.

Return the COMPLETE revised LaTeX source — not a diff, the full document, from
\\documentclass through \\end{document}, with no markdown fences.
"""


class RefinementAgent(BaseAgent):
    name = "RefinementAgent"

    def __init__(self, llm, work_dir, config, review_agent: ReviewAgent | None = None):
        super().__init__(llm, work_dir, config)
        self.review_agent = review_agent or ReviewAgent(llm, work_dir, config)
        self.max_cycles: int = config.get("max_cycles", 3)
        self.min_improvement: float = config.get("min_improvement", 0.2)

    def execute(self, state: PipelineState) -> PipelineState:
        if not state.draft:
            self.logger.warning("No draft — cannot refine.")
            return state

        current_latex = state.draft.latex_source
        review_history: list[ReviewScore] = []
        prev_score = 0.0
        stopped_reason = "max_cycles"

        for cycle in range(1, self.max_cycles + 1):
            self.logger.info("  Refinement cycle %d / %d", cycle, self.max_cycles)
            self.step(f"cycle{cycle}", f"Cycle {cycle}/{self.max_cycles}: review", "running")

            # Review the current version
            score = self.review_agent.review(current_latex, cycle, prev_score)
            self.step(f"cycle{cycle}", f"Cycle {cycle}/{self.max_cycles}: review", "success",
                      detail=f"score {score.overall_score:.1f}/10 · {score.verdict.value}")
            review_history.append(score)
            self._write(f"review_cycle_{cycle}.json", json.dumps(score.model_dump(), indent=2, default=str))

            self.logger.info(
                "    Score: %.1f | Verdict: %s",
                score.overall_score,
                score.verdict.value,
            )

            # Check stopping conditions
            if cycle > 1 and score.verdict == ReviewVerdict.DEGRADATION:
                self.logger.info("  ⛔ Degradation detected — reverting to previous version.")
                # Revert to previous version (don't apply this cycle's changes)
                stopped_reason = "no_improvement"
                break

            if cycle > 1 and score.verdict == ReviewVerdict.NEUTRAL:
                improvement = score.overall_score - prev_score
                if improvement < self.min_improvement:
                    self.logger.info("  ⏹ Gains stopped (Δ=%.2f < %.2f).", improvement, self.min_improvement)
                    stopped_reason = "no_improvement"
                    break

            # Refine based on feedback
            if cycle < self.max_cycles:  # Don't refine after last review
                current_latex = self._refine(current_latex, score, self._valid_keys(state))
                self._write(f"manuscript_v{cycle + 1}.tex", current_latex)

            prev_score = score.overall_score

        # Compile final PDF
        final_tex_path = self._write("manuscript_final.tex", current_latex)
        pdf_path = self._compile_pdf(Path(final_tex_path))

        state.refinement = RefinementOutput(
            final_latex=current_latex,
            final_pdf_path=pdf_path,
            review_history=review_history,
            total_cycles=len(review_history),
            stopped_reason=stopped_reason,
        )
        return state

    # ----- refinement --------------------------------------------------------

    @staticmethod
    def _valid_keys(state: PipelineState) -> set[str]:
        return {e.cite_key for e in state.literature.bib_entries} if state.literature else set()

    def _refine(self, latex: str, score: ReviewScore, valid_keys: set[str]) -> str:
        """Apply one round of refinement based on review feedback."""
        user_msg = (
            f"# Current Manuscript\n```latex\n{latex}\n```\n\n"
            f"# Reviewer Feedback (score: {score.overall_score}/10)\n"
            f"{score.feedback}\n\n"
            f"# Specific Suggestions\n"
            + "\n".join(f"- {s}" for s in score.suggestions)
        )
        revised = strip_fences(self._prompt(REFINE_SYSTEM, user_msg, max_tokens=16000))
        if "\\end{document}" not in revised:
            self.logger.warning("Refined manuscript is incomplete (no \\end{document}); keeping the previous version.")
            return latex
        revised, removed = restrict_citations(revised, valid_keys)
        if removed:
            self.logger.warning("Removed citations to unknown keys: %s", ", ".join(sorted(set(removed))))
        revised, dupes = dedupe_floats(revised)
        if dupes:
            self.logger.warning("Removed %d duplicated figure/table environment(s)", dupes)
        return disable_unused_bibliography(revised)

    # ----- PDF compilation ---------------------------------------------------

    def _compile_pdf(self, tex_path: Path) -> Path | None:
        """Try to compile LaTeX to PDF using pdflatex or latexmk."""
        import subprocess
        import shutil

        pdf_path = tex_path.with_suffix(".pdf")

        for compiler in ["latexmk -pdf -interaction=nonstopmode", "pdflatex -interaction=nonstopmode"]:
            cmd_parts = compiler.split()
            if not shutil.which(cmd_parts[0]):
                continue
            try:
                subprocess.run(
                    [*cmd_parts, str(tex_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=tex_path.parent,
                )
                if pdf_path.exists():
                    self.logger.info("  PDF compiled: %s", pdf_path)
                    return pdf_path
            except Exception as exc:
                self.logger.warning("  PDF compilation failed with %s: %s", cmd_parts[0], exc)

        self.logger.warning("  No LaTeX compiler available — PDF not generated.")
        return None
