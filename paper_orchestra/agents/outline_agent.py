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
import re
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
   "script": "<a filename from the Available scripts list>", "expected_outputs": ["metrics.json"],
   "depends_on": [], "target": "colab|jetson"}

Experiment targets: model TRAINING and any export step run on "colab" (GPU
cloud).  EDGE DEPLOYMENT benchmarks (latency, throughput, size of PyTorch, ONNX
and TensorRT FP32/FP16/INT8 variants) run on "jetson" (Jetson Orin Nano) and
must list the training/export experiment in depends_on, so the trained model
files are carried over.  Only plan "jetson" experiments when the requirements
below say a Jetson is available.

Every experiment's "script" MUST be a filename from the "Available scripts" list
in the requirements below; never invent a script name.  If no listed script covers
an experiment, leave that experiment out instead of planning it.  Leave depends_on
empty when the prerequisite is already done (e.g. a model is already trained).

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

        self._drop_unrunnable(outline, state)

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
    def _script_dirs(exp) -> list[Path]:
        dirs = [exp.scripts_dir] if exp and exp.scripts_dir else []
        return [Path(d) for d in dirs + ["examples/experiments", "experiments"]]

    def _available_scripts(self, exp) -> dict[str, str]:
        """Map script filename -> first docstring line, for every .py in the script folders."""
        found: dict[str, str] = {}
        for d in self._script_dirs(exp):
            for f in sorted(d.glob("*.py")) if d.is_dir() else []:
                if f.name in found:
                    continue
                text = f.read_text(encoding="utf-8", errors="ignore").lstrip()
                m = re.match(r'(?:"""|\'\'\')(.*?)(?:\n|"""|\'\'\')', text)
                found[f.name] = m.group(1).strip() if m else ""
        return found

    @staticmethod
    def _model_ready_note(exp) -> str:
        """Tell the planner when the model is already trained and exported on the Jetson."""
        if exp.jetson.host and exp.jetson.env_vars.get("BENCH_MODEL_DIR"):
            return (
                "The model is ALREADY trained and exported (model.pt, model.onnx) on the Jetson. "
                "Do NOT plan training, export or any colab experiment; plan only the Jetson "
                "benchmark, with empty depends_on. Do not state training results in the paper."
            )
        return ""

    def _drop_unrunnable(self, outline: Outline, state: PipelineState) -> None:
        """Remove experiments whose script does not exist, so none fails on a made-up filename."""
        exp = state.inputs.experiment_requirements
        if not exp:
            return
        available = self._available_scripts(exp)
        kept = [p for p in outline.experiment_plan if p.script in available]
        for p in outline.experiment_plan:
            if p.script not in available:
                self.logger.warning("Dropping experiment %s: script '%s' does not exist", p.experiment_id, p.script)
        ids = {p.experiment_id for p in kept}
        for p in kept:
            p.depends_on = [d for d in p.depends_on if d in ids]
        outline.experiment_plan = kept

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
                "Available scripts (the only valid values for \"script\"):\n"
                + "\n".join(f"- {n}: {d}" for n, d in self._available_scripts(exp).items()),
                self._model_ready_note(exp),
                "Targets available: colab (training)"
                + (", jetson (edge deployment benchmark)" if exp.jetson.host else ""),
            ])

        return "\n".join(p for p in parts if p)
