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
from ..domains import DOMAINS, HELPER_SCRIPTS, MICROSCOPY, domain_of
from ..models import (
    ExperimentPlan,
    Outline,
    PipelineState,
    ResearchNode,
    VisualizationPlan,
    WritingPlanSection,
)

PROMPT_TEMPLATE = """\
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
  {"figure_id": "fig1", "description": "...", "figure_type": "plot|table|diagram|confusion_matrix|roc_curve|morphology_probe|morphology_examples|training_curves",
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

{DOMAIN_SECTION}
Return ONLY valid JSON — no markdown fences, no commentary.
"""


def system_prompt(domain) -> str:
    """The planner's instructions with the domain's figure types and experiment chain."""
    return PROMPT_TEMPLATE.replace("{DOMAIN_SECTION}", domain.planner_section)


SYSTEM_PROMPT = system_prompt(MICROSCOPY)

# The leukemia (microscopy) recipe under its old names, for anything that still imports them
KNOWN_OUTPUTS, KNOWN_TARGETS, CHAIN_NAMES = MICROSCOPY.outputs, MICROSCOPY.targets, MICROSCOPY.names
CHAIN, RESULT_FIGURES = MICROSCOPY.chain, MICROSCOPY.figures


class OutlineAgent(BaseAgent):
    name = "OutlineAgent"

    def execute(self, state: PipelineState) -> PipelineState:
        # Read the LaTeX template to understand section structure
        template_text = self._read_template(state.inputs.latex_template_path)

        user_msg = self._build_user_message(state, template_text)
        exp = state.inputs.experiment_requirements
        raw = self._prompt_json(system_prompt(domain_of(exp)), user_msg, max_tokens=8192)

        outline = Outline(
            title=raw.get("title", state.inputs.idea.title),
            visualization_plan=[VisualizationPlan(**v) for v in raw.get("visualization_plan", [])],
            research_graph=[ResearchNode(**r) for r in raw.get("research_graph", [])],
            writing_plan=[WritingPlanSection(**w) for w in raw.get("writing_plan", [])],
            experiment_plan=[ExperimentPlan(**e) for e in raw.get("experiment_plan", [])],
        )

        self._drop_unrunnable(outline, state)
        self._complete_plan(outline, state)

        # Persist the outline as it will run (after the plan was completed)
        self._write("outline.json", json.dumps(outline.model_dump(mode="json"), indent=2, ensure_ascii=False))
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
                if f.name in found or f.name in HELPER_SCRIPTS:
                    continue
                text = f.read_text(encoding="utf-8", errors="ignore").lstrip()
                m = re.match(r'(?:"""|\'\'\')(.*?)(?:\n|"""|\'\'\')', text)
                found[f.name] = m.group(1).strip() if m else ""
        return found

    def _history_note(self, exp) -> str:
        """Say so when the author supplied per-epoch training curves; the pipeline cannot produce them itself."""
        if any((d / "training_history.json").is_file() for d in self._script_dirs(exp)):
            return ('Available results: "training_history" (per-epoch train/validation loss and accuracy '
                    "recorded by the author in Google Colab).  Plan one training_curves figure from it, "
                    'with data_source "training_history".')
        return ""

    @staticmethod
    def _model_ready_note(exp) -> str:
        """Tell the planner when the model is already trained and exported on the Jetson."""
        if exp.jetson.host and exp.jetson.env_vars.get("BENCH_MODEL_DIR") and not exp.colab.enabled:
            return (
                "The model is ALREADY trained and exported (model.pt, model.onnx) on the Jetson. "
                "Do NOT plan training, export or any colab experiment; plan only the Jetson "
                "experiments (benchmark, accuracy, morphology probe), with empty depends_on. "
                "Do not state training results in the paper."
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
        outputs = domain_of(exp).outputs
        for p in kept:
            p.expected_outputs = outputs.get(p.script, p.expected_outputs)
        ids = {p.experiment_id for p in kept}
        for p in kept:
            p.depends_on = [d for d in p.depends_on if d in ids]
        outline.experiment_plan = kept

    def _complete_plan(self, outline: Outline, state: PipelineState) -> None:
        """Wire the bundled experiments together and make sure their result figures are planned.

        The planner is asked to do this, but a missed step would silently drop a result the paper promises, so the
        known scripts get their target, outputs and dependencies here, the Colab training chain is added in front
        of the Jetson experiments, and each result figure type is planned once.
        """
        exp = state.inputs.experiment_requirements
        if not exp:
            return
        dom = domain_of(exp)
        scripts = self._available_scripts(exp)
        # one plan per bundled script and one per id; the planner sometimes repeats a script or reuses an id
        seen_scripts: set[str] = set()
        seen_ids: set[str] = set()
        unique = []
        kept: dict[str, str] = {}  # script -> id of the plan kept
        remap: dict[str, str] = {}  # id of a dropped duplicate -> id of the plan kept
        for p in outline.experiment_plan:
            if p.script in dom.targets and p.script in seen_scripts:
                remap[p.experiment_id] = kept[p.script]
                continue
            if exp.colab.enabled and p.script == "prepare_models.py" and set(dom.required) <= scripts.keys():
                continue  # Colab trains the model, so there is no pre-trained one to stage
            if p.experiment_id in seen_ids:
                p.experiment_id = f"{p.experiment_id}_{len(seen_ids)}"
            seen_scripts.add(p.script)
            seen_ids.add(p.experiment_id)
            kept[p.script] = p.experiment_id
            unique.append(p)
        outline.experiment_plan = unique
        for p in unique:
            p.depends_on = [d for d in (remap.get(d, d) for d in p.depends_on) if d in seen_ids and d != p.experiment_id]
        for v in outline.visualization_plan:
            v.data_source = remap.get(v.data_source, v.data_source)
        # A table of experiment results is generated by code and handed to the writer as LaTeX; planned as a "figure" it
        # would be written a second time inside a figure environment.
        src_ids = {p.experiment_id.lower() for p in outline.experiment_plan}
        from_results = lambda v: v.figure_type == "table" and (v.data_source.lower() in src_ids or any(w in v.data_source.lower() for w in ("exp", "benchmark", "result")))
        table_figs = {v.figure_id for v in outline.visualization_plan if from_results(v)}
        if table_figs:
            outline.visualization_plan = [v for v in outline.visualization_plan if v.figure_id not in table_figs]
            for w in outline.writing_plan:
                w.figures = [f for f in w.figures if f not in table_figs]
        plans = {p.script: p for p in outline.experiment_plan if p.script in dom.targets}
        edge = [s for s in dom.edge if s in plans]
        chain = [c for c in dom.chain if c in scripts]
        chained = bool(edge) and exp.colab.enabled and set(dom.required) <= set(chain)

        def new_id() -> str:
            used = {p.experiment_id for p in outline.experiment_plan}
            return next(f"exp{i}" for i in range(1, 99) if f"exp{i}" not in used)

        if chained:
            for script in chain:
                if script not in plans:
                    plans[script] = ExperimentPlan(experiment_id=new_id(), name=dom.names[script], script=script)
                    outline.experiment_plan.append(plans[script])
        else:  # no Colab training: the model is already on the Jetson, so the chain has no place
            outline.experiment_plan = [p for p in outline.experiment_plan if p.script not in dom.chain]
            plans = {k: v for k, v in plans.items() if k not in dom.chain}

        for script, plan in plans.items():
            plan.target = (dom.chained_targets.get(script) if chained else None) or dom.targets[script]
            plan.expected_outputs = dom.outputs[script]
        if chained:
            for script, needs in dom.needs.items():  # what each script reads from the experiments before it
                if script in plans:
                    plans[script].depends_on = [plans[n].experiment_id for n in needs if n in plans]
        else:
            for script in edge:
                plans[script].depends_on = []
        ids = {p.experiment_id for p in outline.experiment_plan}
        for p in outline.experiment_plan:  # a step removed above must not leave a dangling dependency
            p.depends_on = [d for d in p.depends_on if d in ids]
        outline.experiment_plan.sort(key=lambda p: list(dom.targets).index(p.script) if p.script in dom.targets else 99)

        for v in outline.visualization_plan:  # a result figure takes its data from the experiment that writes it
            if v.figure_type in dom.figures and dom.figures[v.figure_type][0] in plans:
                v.data_source = plans[dom.figures[v.figure_type][0]].experiment_id

        home = next((w for w in outline.writing_plan if re.search(r"result|experiment", f"{w.section_id} {w.title}", re.I)), None)
        for ftype, (script, caption) in dom.figures.items():
            if script not in plans or any(v.figure_type == ftype for v in outline.visualization_plan):
                continue
            used = {v.figure_id for v in outline.visualization_plan}
            fid = next(f"fig_{ftype}{'' if i == 0 else i}" for i in range(99) if f"fig_{ftype}{'' if i == 0 else i}" not in used)
            outline.visualization_plan.append(VisualizationPlan(
                figure_id=fid, description=caption, figure_type=ftype, data_source=plans[script].experiment_id,
                placement_section=home.section_id if home else ""))
            if home:
                home.figures.append(fid)
        self.logger.info("Experiment plan: %s", ", ".join(f"{p.experiment_id}={p.script}({p.target})" for p in outline.experiment_plan))

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
                f"Dataset: {exp.dataset.id} (source {exp.dataset.source}, domain {exp.dataset.domain}, split {exp.dataset.split})"
                if getattr(exp, "dataset", None) and exp.dataset.domain != "microscopy" else "",
                f"Scripts directory: {exp.scripts_dir}" if exp.scripts_dir else "",
                "Available scripts (the only valid values for \"script\"):\n"
                + "\n".join(f"- {n}: {d}" for n, d in self._available_scripts(exp).items()),
                self._model_ready_note(exp),
                self._history_note(exp),
                "Targets available: colab (training)"
                + (", jetson (edge deployment benchmark)" if exp.jetson.host else ""),
            ])

        return "\n".join(p for p in parts if p)
