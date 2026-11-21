"""Step 4 — Section Writing Agent.

Merges the outline, generated figures, literature review, and
experiment documentation into a full LaTeX manuscript draft.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..base_agent import BaseAgent
from ..domains import MICROSCOPY, domain_of
from ..utils.metrics import compact_metrics
from ..utils.latex import fix_undefined_refs, close_cut_off, looks_cut_off, dedupe_floats, drop_missing_figures, disable_unused_bibliography, restrict_citations, strip_fences
from ..models import (
    GeneratedFigure,
    ManuscriptDraft,
    PipelineState,
    WritingPlanSection,
)

BASE_PROMPT = """\
You are a scientific manuscript writer.  Given a section plan, relevant
literature paragraphs, figure references, and experiment results, write
the LaTeX content for that section.

Rules:
1. Write fluent, publication-quality academic prose.
2. Use \\cite{key} for citations and \\ref{label} for figure/table cross-refs.
3. Include \\begin{figure} environments where the plan says figures go.
4. For experiment sections, incorporate the metrics and results data.
5. Output ONLY the LaTeX body for this section (no \\documentclass, preamble).
6. Start with \\section{Title} or \\subsection{Title} as appropriate.

Experiment reporting:
- Training ran on Google Colab; edge deployment was benchmarked on a Jetson
  Orin Nano.  Say which hardware each number comes from.
- Report edge results as: batch size 1, input size, number of timed runs after
  warm-up, latency as mean +/- std, and the serving stack (Triton, TensorRT).
- Tables given under "LaTeX table" are generated from the raw results.  Paste
  them unchanged inside a table environment; never retype or round the numbers.
- If the results hold "int8_calibration", the INT8 engine was calibrated with TensorRT's entropy
  calibrator on that many training images (never test images); say so and report its accuracy like
  the other variants.  When "fp16_fallback" is true the engine is mixed INT8/FP16, so call it that and
  do not describe it as pure INT8.  If INT8 was built without a calibration cache, say the INT8 numbers show
  speed only and that accuracy was not validated.
- Do not name a winner when the gap is inside one standard deviation.
"""

SYSTEM_PROMPT = BASE_PROMPT + MICROSCOPY.writer_notes  # the leukemia prompt, unchanged

# Voice rules follow the thesis-voice-coach skill: write like the author, not like a
# polished model.  Equations, tables and reported numbers stay verbatim.
VOICE_RULES = """\

Voice (the author's own, not generic academic prose):
- Imitate the author's writing samples below: their vocabulary, transitions and
  sentence rhythm.  Do not upgrade their vocabulary or smooth their phrasing.
- Vary sentence length.  Mix short sentences with long ones; avoid rows of
  sentences of equal length.
- Avoid parenthetical checklists such as "(e.g., latency, memory, accuracy)";
  write them as a real sentence.
- Avoid formulaic connectives: "It is worth noting that", "A defensible claim is
  therefore", "Consequently, X ... they do not constitute evidence of Y".
- Do not hedge every clause evenly.  Hedge only where the data is weak.
- Do not overclaim.  If differences fall inside overlapping confidence
  intervals, say the results are equivalent rather than naming a winner.
- Keep equations, tables and numbers exactly as given.
- Describe only work the Experiment Results show.  If no training experiment ran, do not
  say the model was trained in this study; say it was a pre-trained model that was benchmarked.
  Report no accuracy figure unless a result gives one.
"""


class WritingAgent(BaseAgent):
    name = "WritingAgent"
    _system_prompt = SYSTEM_PROMPT

    def execute(self, state: PipelineState) -> PipelineState:
        if not state.outline:
            self.logger.warning("No outline — cannot write.")
            return state

        template_text = self._read_template(state.inputs.latex_template_path)
        self._system_prompt = self._build_system_prompt(domain_of(state.inputs.experiment_requirements))

        # Write each section
        section_texts: list[str] = []
        for section_plan in state.outline.writing_plan:
            tex = self._write_section(state, section_plan)
            section_texts.append(tex)
            self.logger.info("  Wrote section: %s (%d chars)", section_plan.title, len(tex))

        # Assemble full document
        full_latex = self._assemble(state, template_text, section_texts)

        valid = {e.cite_key for e in state.literature.bib_entries} if state.literature else set()
        full_latex, removed = restrict_citations(full_latex, valid)
        if removed:
            self.logger.warning("Removed %d citation(s) to keys not in the bibliography: %s",
                                len(removed), ", ".join(sorted(set(removed))))
        full_latex = disable_unused_bibliography(full_latex)
        have = {f.file_path.stem for f in (state.figures or [])}
        full_latex, missing = drop_missing_figures(full_latex, have)
        if missing:
            self.logger.warning("Removed figure(s) with no generated image: %s", ", ".join(missing))
        full_latex, dupes = dedupe_floats(full_latex)
        if dupes:
            self.logger.warning("Removed %d duplicated figure/table environment(s)", dupes)
        full_latex, remapped, unresolved = fix_undefined_refs(full_latex)
        if remapped:
            self.logger.warning("Pointed %d reference(s) at the label that exists: %s", len(remapped),
                                ", ".join(f"{a} -> {b}" for a, b in remapped.items()))
        if unresolved:
            self.logger.warning("References to labels that do not exist: %s", ", ".join(unresolved))

        out_path = self._write("manuscript.tex", full_latex)

        # Also write the .bib next to the manuscript
        if state.literature and state.literature.bib_file_path:
            bib_content = state.literature.bib_file_path.read_text(encoding="utf-8")
            self._write("references.bib", bib_content)

        word_estimate = len(full_latex.split())

        state.draft = ManuscriptDraft(
            latex_source=full_latex,
            output_path=Path(out_path),
            sections_written=[s.title for s in state.outline.writing_plan],
            word_count=word_estimate,
        )
        return state

    # ----- voice -------------------------------------------------------------

    def _build_system_prompt(self, domain=MICROSCOPY) -> str:
        """The base rules plus the domain's reporting rules; append the author's voice rules and writing samples when configured."""
        prompt = BASE_PROMPT + domain.writer_notes
        samples_path = self.config.get("style_samples_path")
        if not samples_path:
            return prompt
        path = Path(samples_path)
        if not path.exists():
            self.logger.warning("style_samples_path not found: %s", path)
            return prompt
        samples = path.read_text(encoding="utf-8")[: self.config.get("style_samples_max_chars", 6000)]
        return f"{prompt}{VOICE_RULES}\n# Author writing samples\n{samples}\n"

    # ----- per-section writing -----------------------------------------------

    def _write_section(self, state: PipelineState, plan: WritingPlanSection) -> str:
        # Build context for this section
        context_parts = [
            f"# Section to write\nTitle: {plan.title}",
            f"Key points: {', '.join(plan.key_points)}",
            f"Target words: {plan.target_words}",
        ]

        # Add figure references
        fig_map = {f.figure_id: f for f in (state.figures or [])}
        for fid in plan.figures:
            fig = fig_map.get(fid)
            if fig:
                context_parts.append(
                    f"Figure: \\label{{{fig.label}}} — {fig.caption}"
                )

        kinds = self._section_kinds(plan)

        entries = state.literature.bib_entries if state.literature else []
        context_parts.append(
            "# Citable references (cite ONLY these keys with \\cite{key}; if none fits a claim, cite nothing)\n"
            + ("\n".join(f"- {e.cite_key}: {e.title} ({e.year})" for e in entries) or "(none available — do not cite)")
        )

        # Add literature if relevant
        if state.literature and kinds & {"intro", "related"}:
            if "intro" in kinds and state.literature.intro_paragraphs:
                context_parts.append(
                    "# Pre-drafted intro paragraphs (refine these):\n"
                    + "\n\n".join(state.literature.intro_paragraphs)
                )
            elif "related" in kinds and state.literature.related_work_paragraphs:
                context_parts.append(
                    "# Pre-drafted related work paragraphs (refine these):\n"
                    + "\n\n".join(state.literature.related_work_paragraphs)
                )

        # Add experiment results for methods/experiments/results sections
        if state.experiment_output and kinds & {"experiments", "discussion", "abstract"}:  # the abstract quotes the headline numbers
            exp_summary = []
            for r in state.experiment_output.results:
                metrics_str = json.dumps(compact_metrics(r.metrics), indent=2) if r.metrics else "N/A"
                exp_summary.append(
                    f"- {r.experiment_id}: status={r.status}, target={r.target}, "
                    f"device={r.device}, duration={r.duration_seconds:.1f}s, metrics={metrics_str}"
                )
            context_parts.append(
                "# Experiment Results\n" + "\n".join(exp_summary)
            )
            if "experiments" in kinds:  # discussion/conclusion cite the numbers but do not repeat the table
                for exp_id, tex in state.experiment_output.latex_tables.items():
                    context_parts.append(f"# LaTeX table for {exp_id} (use verbatim)\n{tex}")

        user_msg = "\n\n".join(context_parts)
        tex = strip_fences(self._prompt(self._system_prompt, user_msg, max_tokens=4096))
        for attempt in (1, 2):  # a section cut off by the output limit has an open table or a dangling \\ref
            problems = looks_cut_off(tex)
            if not problems:
                return tex
            self.logger.warning("  Section '%s' looks cut off (%s); writing it again (attempt %d/2)",
                                plan.title, "; ".join(problems), attempt)
            tex = strip_fences(self._prompt(
                self._system_prompt,
                user_msg + "\n\n# Important\nA previous attempt at this section ended in the middle of a LaTeX environment or "
                "command (" + "; ".join(problems) + "). Write the COMPLETE section again, a little more concisely, and close "
                "every \\begin with its \\end.", max_tokens=8000))
        problems = looks_cut_off(tex)
        if problems:
            self.logger.error("  Section '%s' is still cut off (%s); closing it by hand.", plan.title, "; ".join(problems))
            tex = close_cut_off(tex)
        return tex

    @staticmethod
    def _section_kinds(plan: WritingPlanSection) -> set[str]:
        """Classify a section from its id and title, since the outline LLM picks ids freely."""
        text = f"{plan.section_id} {plan.title}".lower()
        keywords = {
            "intro": ("intro",),
            "related": ("related", "background", "literature"),
            "experiments": ("experiment", "result", "evaluation", "method", "benchmark", "deployment"),
            "discussion": ("discussion", "conclusion", "limitation"),
            "abstract": ("abstract",),
        }
        return {kind for kind, words in keywords.items() if any(w in text for w in words)}

    # ----- assembly ----------------------------------------------------------

    def _assemble(
        self, state: PipelineState, template: str, sections: list[str]
    ) -> str:
        """Insert written sections into the LaTeX template."""
        # If template has a %CONTENT% placeholder, use it
        body = "\n\n".join(sections)

        if "%CONTENT%" in template:
            # A bare %TITLE% would comment out the closing brace of \title{...}
            title = state.outline.title if state.outline else state.inputs.idea.title
            authors = state.inputs.idea.authors or "Anonymous Authors"
            return (template.replace("%TITLE%", title)
                            .replace("%AUTHORS%", authors)
                            .replace("%CONTENT%", body))

        # Otherwise build a minimal document
        title = state.outline.title if state.outline else "Paper"
        return (
            "\\documentclass[conference]{IEEEtran}\n"
            "\\usepackage{graphicx,amsmath,cite,hyperref}\n"
            f"\\title{{{title}}}\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
            f"{body}\n\n"
            "\\bibliographystyle{IEEEtran}\n"
            "\\bibliography{references}\n"
            "\\end{document}\n"
        )

    @staticmethod
    def _read_template(path: Path) -> str:
        path = Path(path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""
