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

Result figures are drawn by code from the measurements, so name the experiment that
produces the data in the figure's "data_source" (e.g. "exp1") and use these figure_types:
  confusion_matrix   confusion matrix of every deployed variant      (needs triton_benchmark.py with BENCH_TEST_DIR)
  roc_curve          one-vs-rest ROC curves with AUC                  (same experiment)
  dataset_stats      background statistics per class and a background-only classifier (needs dataset_stats.py)
  augmentation_examples  one image per class re-rendered on several staining protocols (same experiment)
  morphology_probe   accuracy of the baseline and the background-augmented model with the cells masked, on a
                     neutral background and on stain protocols never used in training  (needs morphology_probe.py)
  morphology_examples  image montage of the original, masked and re-backgrounded cells (same experiment)
  training_curves    training/validation loss and accuracy per epoch of the augmented model (train_augmented.py)
  training_curves_baseline   the same for the baseline model (train_baseline.py)
When the matching script is in the Available scripts list, plan it:
  triton_benchmark.py  "jetson"  expected_outputs ["benchmark_results.json"]
If Colab training is available (see the requirements below), the experiments run in this chain:
  prepare_dataset.py   "colab"   downloads the dataset from Kaggle, splits it, packs the Jetson's test and calibration
                       images   expected_outputs ["test_data.zip", "calib_data.zip", "split_info.json"]
  dataset_stats.py     "colab"   statistics of the training backgrounds; runs BEFORE the training
  train_baseline.py    "colab"   trains without background augmentation   depends_on the dataset experiment
  train_augmented.py   "colab"   trains with background/stain augmentation depends_on the dataset experiment
  triton_benchmark.py  "jetson"  benchmarks the AUGMENTED model   depends_on the dataset and train_augmented.py experiments
  morphology_probe.py  "colab"   probes both trained models, with statistics   depends_on the dataset, baseline
                                 and augmented experiments
Without Colab training, morphology_probe.py is a "jetson" experiment on the model already there.
Plan the figures above for them (one of each), placed in the experiments/results sections.  The research question
about whether the model learns cell morphology or the stain/background is answered from dataset_stats.py (what the
backgrounds carry) and morphology_probe.py (how both models behave on held-out stains), never from anything else.

Return ONLY valid JSON — no markdown fences, no commentary.
"""


# What each bundled script writes; the planner's guess at the file names is replaced by these
KNOWN_OUTPUTS = {
    "prepare_dataset.py": ["test_data.zip", "calib_data.zip", "split_info.json"],
    "dataset_stats.py": ["dataset_stats.json", "dataset_stats.png", "augmentation_examples.png"],
    "train_baseline.py": ["training_history_baseline.json", "model_baseline.pt"],
    "train_augmented.py": ["training_history.json", "model.pt", "model.onnx"],
    "triton_benchmark.py": ["benchmark_results.json"],
    "morphology_probe.py": ["probe_results.json", "probe_examples.png"],
}
KNOWN_TARGETS = {
    "prepare_dataset.py": "colab", "dataset_stats.py": "colab", "train_baseline.py": "colab",
    "train_augmented.py": "colab", "triton_benchmark.py": "jetson", "morphology_probe.py": "jetson",
}
CHAIN_NAMES = {
    "prepare_dataset.py": "Download the dataset from Kaggle and split it",
    "dataset_stats.py": "Statistics of the training backgrounds",
    "train_baseline.py": "Train EfficientViT-B0 without background augmentation (baseline)",
    "train_augmented.py": "Train EfficientViT-B0 with background/stain augmentation",
}
CHAIN = ("prepare_dataset.py", "dataset_stats.py", "train_baseline.py", "train_augmented.py")
HELPER_SCRIPTS = {"train_efficientvit.py", "stain_render.py", "int8_calibration.py"}  # imported by the scripts above, never planned
# figure_type -> (script whose experiment supplies the data, caption)
RESULT_FIGURES = {
    "dataset_stats": ("dataset_stats.py", "Background statistics of the training images per class (L*, a*, b*) and the accuracy of a linear classifier given only the background or only the cells."),
    "augmentation_examples": ("dataset_stats.py", "One image per class with its background (and cell colour) re-rendered on the Romanowsky protocols used for augmentation, and on held-out protocols."),
    "training_curves_baseline": ("train_baseline.py", "Training and validation loss and accuracy of the baseline EfficientViT-B0 per epoch, trained on Google Colab."),
    "training_curves": ("train_augmented.py", "Training and validation loss and accuracy of the background-augmented EfficientViT-B0 per epoch, trained on Google Colab."),
    "confusion_matrix": ("triton_benchmark.py", "Confusion matrices of the deployed variants on the held-out test images, from inference on the Jetson Orin Nano."),
    "roc_curve": ("triton_benchmark.py", "One-vs-rest ROC curves of the deployed variants on the held-out test images, from inference on the Jetson Orin Nano."),
    "morphology_probe": ("morphology_probe.py", "Accuracy of the baseline and the background-augmented model with the cells masked, kept on a neutral background, and re-rendered on stain protocols seen and not seen in training."),
    "morphology_examples": ("morphology_probe.py", "Example images for each probe condition: original, cells masked, cells only and re-rendered backgrounds."),
}


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
        for p in kept:
            p.expected_outputs = KNOWN_OUTPUTS.get(p.script, p.expected_outputs)
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
        scripts = self._available_scripts(exp)
        # one plan per bundled script and one per id; the planner sometimes repeats a script or reuses an id
        seen_scripts: set[str] = set()
        seen_ids: set[str] = set()
        unique = []
        kept: dict[str, str] = {}  # script -> id of the plan kept
        remap: dict[str, str] = {}  # id of a dropped duplicate -> id of the plan kept
        for p in outline.experiment_plan:
            if p.script in KNOWN_TARGETS and p.script in seen_scripts:
                remap[p.experiment_id] = kept[p.script]
                continue
            if exp.colab.enabled and p.script == "prepare_models.py" and {"prepare_dataset.py", "train_augmented.py"} <= scripts.keys():
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
        plans = {p.script: p for p in outline.experiment_plan if p.script in KNOWN_TARGETS}
        edge = [s for s in ("triton_benchmark.py", "morphology_probe.py") if s in plans]
        chain = [c for c in CHAIN if c in scripts]
        chained = bool(edge) and exp.colab.enabled and {"prepare_dataset.py", "train_augmented.py"} <= set(chain)

        def new_id() -> str:
            used = {p.experiment_id for p in outline.experiment_plan}
            return next(f"exp{i}" for i in range(1, 99) if f"exp{i}" not in used)

        if chained:
            for script in chain:
                if script not in plans:
                    plans[script] = ExperimentPlan(experiment_id=new_id(), name=CHAIN_NAMES[script], script=script)
                    outline.experiment_plan.append(plans[script])
        else:  # no Colab training: the model is already on the Jetson, so the chain has no place
            outline.experiment_plan = [p for p in outline.experiment_plan if p.script not in CHAIN]
            plans = {k: v for k, v in plans.items() if k not in CHAIN}

        for script, plan in plans.items():
            plan.target = KNOWN_TARGETS[script]
            plan.expected_outputs = KNOWN_OUTPUTS[script]
        if chained:
            prep, aug = plans["prepare_dataset.py"], plans["train_augmented.py"]
            for script in ("dataset_stats.py", "train_baseline.py", "train_augmented.py"):
                if script in plans:  # the statistics come first only through the order below, not as a dependency
                    plans[script].depends_on = [prep.experiment_id]
            if "triton_benchmark.py" in plans:  # the deployed model is the augmented one; the test images come from the dataset step
                plans["triton_benchmark.py"].depends_on = [prep.experiment_id, aug.experiment_id]
            if "morphology_probe.py" in plans:  # probes both trained models on Colab, on the test images packed on the Jetson
                probe = plans["morphology_probe.py"]
                probe.target = "colab"
                probe.depends_on = [prep.experiment_id, aug.experiment_id] + (
                    [plans["train_baseline.py"].experiment_id] if "train_baseline.py" in plans else [])
        else:
            for script in edge:
                plans[script].depends_on = []
        ids = {p.experiment_id for p in outline.experiment_plan}
        for p in outline.experiment_plan:  # a step removed above must not leave a dangling dependency
            p.depends_on = [d for d in p.depends_on if d in ids]
        outline.experiment_plan.sort(key=lambda p: list(KNOWN_TARGETS).index(p.script) if p.script in KNOWN_TARGETS else 99)

        for v in outline.visualization_plan:  # a result figure takes its data from the experiment that writes it
            if v.figure_type in RESULT_FIGURES and RESULT_FIGURES[v.figure_type][0] in plans:
                v.data_source = plans[RESULT_FIGURES[v.figure_type][0]].experiment_id

        home = next((w for w in outline.writing_plan if re.search(r"result|experiment", f"{w.section_id} {w.title}", re.I)), None)
        for ftype, (script, caption) in RESULT_FIGURES.items():
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
                f"Scripts directory: {exp.scripts_dir}" if exp.scripts_dir else "",
                "Available scripts (the only valid values for \"script\"):\n"
                + "\n".join(f"- {n}: {d}" for n, d in self._available_scripts(exp).items()),
                self._model_ready_note(exp),
                self._history_note(exp),
                "Targets available: colab (training)"
                + (", jetson (edge deployment benchmark)" if exp.jetson.host else ""),
            ])

        return "\n".join(p for p in parts if p)
