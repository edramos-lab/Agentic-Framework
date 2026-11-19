"""Step 1 — Outline Agent.

Takes the raw inputs (idea summary, LaTeX template, conference guideline,
experimental requirements) and produces a structured JSON outline with:
- visualization plan
- research graph
- writing plan
- experiment plan
"""

from __future__ import annotations

import json
from pathlib import Path

from ..base_agent import BaseAgent
from ..models import (
    ExperimentPlan,
    Outline,
    PipelineState,
    ResearchNode,
    VisualizationPlan,
    WritingPlanSection,
)

SYSTEM_PROMPT = """\
You are the Outline Architect for an academic paper-writing pipeline.

Given the research idea, conference guidelines, LaTeX template structure,
and experimental requirements, produce a comprehensive JSON outline.

The outline MUST contain exactly these top-level keys:
{
  "title": "...",
  "visualization_plan": [ ... ],
  "research_graph": [ ... ],
  "writing_plan": [ ... ],
  "experiment_plan": [ ... ]
}

Each visualization_plan entry:
  {"figure_id": "fig1", "description": "...", "figure_type": "plot|table|diagram",
   "data_source": "...", "placement_section": "sec_experiments"}

Each research_graph entry:
  {"node_id": "rg1", "label": "...", "query": "search query for Semantic Scholar",
   "importance": "critical|normal|supplementary"}

Each writing_plan entry:
  {"section_id": "sec_intro", "title": "Introduction", "subsections": [...],
   "key_points": [...], "target_words": 800, "figures": ["fig1"]}

Each experiment_plan entry:
  {"experiment_id": "exp1", "name": "...", "description": "...",
   "script": "train.py", "expected_outputs": ["metrics.json"],
   "depends_on": [], "target": "colab|jetson"}

Experiment targets: model TRAINING and any export step run on "colab" (GPU
cloud).  EDGE DEPLOYMENT benchmarks (latency, throughput, size of PyTorch, ONNX
and TensorRT FP32/FP16/INT8 variants) run on "jetson" (Jetson Orin Nano) and
must list the training/export experiment in depends_on, so the trained model
files are carried over.  Only plan "jetson" experiments when the requirements
below say a Jetson is available.

Return ONLY valid JSON — no markdown fences, no commentary.
"""


class OutlineAgent(BaseAgent):
    name = "OutlineAgent"

    def execute(self, state: PipelineState) -> PipelineState:
        # Read the LaTeX template to understand section structure
        template_text = self._read_template(state.inputs.latex_template_path)

        user_msg = self._build_user_message(state, template_text)
        raw = self._prompt_json(SYSTEM_PROMPT, user_msg, max_tokens=8192)

        outline = Outline(
            title=raw.get("title", state.inputs.idea.title),
            visualization_plan=[VisualizationPlan(**v) for v in raw.get("visualization_plan", [])],
            research_graph=[ResearchNode(**r) for r in raw.get("research_graph", [])],
            writing_plan=[WritingPlanSection(**w) for w in raw.get("writing_plan", [])],
            experiment_plan=[ExperimentPlan(**e) for e in raw.get("experiment_plan", [])],
        )

        # Persist the outline
        self._write("outline.json", json.dumps(raw, indent=2, ensure_ascii=False))
        state.outline = outline
        self.logger.info(
            "Outline: %d figures, %d research nodes, %d sections, %d experiments",
            len(outline.visualization_plan),
            len(outline.research_graph),
            len(outline.writing_plan),
            len(outline.experiment_plan),
        )
        return state

    # ----- helpers -----------------------------------------------------------

    @staticmethod
    def _read_template(path: Path) -> str:
        path = Path(path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "(No template file found — use a standard two-column IEEE layout.)"

    def _build_user_message(self, state: PipelineState, template_text: str) -> str:
        idea = state.inputs.idea
        gl = state.inputs.guideline
        exp = state.inputs.experiment_requirements

        parts = [
            f"# Research Idea\nTitle: {idea.title}",
            f"Abstract draft: {idea.abstract_draft}" if idea.abstract_draft else "",
            f"Research questions:\n" + "\n".join(f"- {q}" for q in idea.research_questions),
            f"Contributions:\n" + "\n".join(f"- {c}" for c in idea.contributions),
            f"Methodology sketch: {idea.methodology_sketch}" if idea.methodology_sketch else "",
            f"\n# Conference Guideline\nName: {gl.name}",
            f"Page limit: {gl.page_limit}" if gl.page_limit else "",
            f"Citation style: {gl.citation_style}",
            f"Required sections: {', '.join(gl.sections_required)}" if gl.sections_required else "",
            f"\n# LaTeX Template (excerpt)\n```latex\n{template_text[:3000]}\n```",
        ]

        if exp:
            parts.extend([
                "\n# Experimental Requirements",
                f"Datasets: {', '.join(exp.datasets)}" if exp.datasets else "",
                f"Models: {', '.join(exp.models)}" if exp.models else "",
                f"Metrics: {', '.join(exp.metrics)}" if exp.metrics else "",
                f"Baselines: {', '.join(exp.baselines)}" if exp.baselines else "",
                f"Hardware: {exp.hardware}" if exp.hardware else "",
                f"Scripts directory: {exp.scripts_dir}" if exp.scripts_dir else "",
                "Targets available: colab (training)"
                + (", jetson (edge deployment benchmark)" if exp.jetson.host else ""),
            ])

        return "\n".join(p for p in parts if p)
