"""Step 5a — AgentReview.

Scores a manuscript revision along multiple dimensions and returns a
verdict: improvement, neutral, or degradation.
"""

from __future__ import annotations

import json

from ..base_agent import BaseAgent
from ..models import PipelineState, ReviewScore, ReviewVerdict

REVIEW_SYSTEM = """\
You are an expert academic paper reviewer (AgentReview).  Given a LaTeX
manuscript, score it on these dimensions (each 1-10):

- **clarity**: writing quality, readability, logical flow
- **novelty**: originality of contributions
- **soundness**: methodological rigor, correctness
- **significance**: impact and relevance to the field
- **presentation**: figures, tables, formatting quality
- **completeness**: are all sections well-developed?

Then provide:
- overall_score (1-10, weighted average)
- verdict: "improvement", "neutral", or "degradation" compared to the
  previous version (use "improvement" for the first review)
- feedback: 2-3 paragraphs of constructive feedback
- suggestions: list of specific, actionable improvements

Return JSON with keys: clarity, novelty, soundness, significance,
presentation, completeness, overall_score, verdict, feedback, suggestions.
Return ONLY valid JSON.
"""


class ReviewAgent(BaseAgent):
    name = "ReviewAgent"

    def review(self, latex_source: str, cycle: int, prev_score: float = 0.0) -> ReviewScore:
        """Score a manuscript and return structured feedback."""
        # the whole manuscript: a cut-off one is reviewed as an incomplete paper
        user_msg = (
            f"Review cycle: {cycle}\n"
            f"Previous overall score: {prev_score}\n\n"
            f"# Manuscript\n```latex\n{latex_source[:150000]}\n```"
        )

        raw = self._prompt_json(REVIEW_SYSTEM, user_msg, max_tokens=4096)

        dimensions = {
            k: float(raw.get(k, 5))
            for k in ("clarity", "novelty", "soundness", "significance",
                       "presentation", "completeness")
        }

        verdict_str = raw.get("verdict", "neutral").lower()
        try:
            verdict = ReviewVerdict(verdict_str)
        except ValueError:
            verdict = ReviewVerdict.NEUTRAL

        return ReviewScore(
            cycle=cycle,
            verdict=verdict,
            overall_score=float(raw.get("overall_score", 5)),
            dimension_scores=dimensions,
            feedback=raw.get("feedback", ""),
            suggestions=raw.get("suggestions", []),
        )

    def execute(self, state: PipelineState) -> PipelineState:
        """Not used directly — the RefinementAgent calls review()."""
        return state
