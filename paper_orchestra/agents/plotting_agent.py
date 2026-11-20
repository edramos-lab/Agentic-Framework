"""Step 2 — Plotting Agent.

Takes the visualization plan from the outline and generates
matplotlib-based plots, tables, and diagrams.

Uses the LLM to generate plotting code for each figure, then
executes it to produce PDF/PNG files.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from ..base_agent import BaseAgent
from ..models import GeneratedFigure, PipelineState, VisualizationPlan

SYSTEM_PROMPT = """\
You are a scientific visualization expert.  Given a figure specification,
write a COMPLETE, self-contained Python script that uses matplotlib (and
optionally seaborn or numpy) to create the figure.

Rules:
1. The script must save the figure to the path given in the variable
   OUTPUT_PATH (already defined in the preamble).
2. Use publication-quality defaults: tight_layout, appropriate font sizes,
   clean color palettes, gridlines where helpful.
3. If the data source is "placeholder", generate realistic synthetic data
   that illustrates the described pattern.
4. Import everything you need.  Do NOT call plt.show().
5. Return ONLY the Python code — no markdown fences.
6. If the spec includes "Experiment data", plot ONLY those numbers, exactly as
   given.  Never invent, round or extrapolate measured values.
7. Write every number into the script as a literal.  Never read any file (no
   open(), json.load or pandas.read_*) — no data file exists when the script runs.
"""

MAX_ATTEMPTS = 3


class PlottingAgent(BaseAgent):
    name = "PlottingAgent"

    def execute(self, state: PipelineState) -> PipelineState:
        if not state.outline:
            self.logger.warning("No outline — skipping plotting.")
            return state

        figures: list[GeneratedFigure] = []
        for spec in state.outline.visualization_plan:
            fig = self._generate_figure(spec, state)
            if fig:
                figures.append(fig)

        state.figures = figures
        self.logger.info("Generated %d / %d figures", len(figures), len(state.outline.visualization_plan))
        return state

    # ----- per-figure logic --------------------------------------------------

    @staticmethod
    def _experiment_data(spec: VisualizationPlan, state: PipelineState) -> tuple[bool, str]:
        """Return (refers_to_experiments, metrics_json) for a figure spec.

        A figure that names an experiment, or says its data comes from the
        experiments, must be drawn from real results — never synthetic data.
        """
        src = spec.data_source.lower()
        results = state.experiment_output.results if state.experiment_output else []
        named = [r for r in results if r.experiment_id.lower() in src]
        if not named and "experiment" not in src and "benchmark" not in src:
            return False, ""
        chosen = named or results
        data = {r.experiment_id: {"target": r.target, "device": r.device, "metrics": r.metrics}
                for r in chosen if r.status == "success" and r.metrics}
        return True, json.dumps(data, indent=2) if data else ""

    def _generate_figure(self, spec: VisualizationPlan, state: PipelineState) -> GeneratedFigure | None:
        out_path = self.agent_dir / f"{spec.figure_id}.pdf"
        uses_results, results_json = self._experiment_data(spec, state)
        if uses_results and not results_json:
            self.logger.warning("Figure %s needs experiment results, none succeeded — skipped.", spec.figure_id)
            return None

        user_msg = (
            f"Figure ID: {spec.figure_id}\n"
            f"Type: {spec.figure_type}\n"
            f"Description: {spec.description}\n"
            f"Data source: {spec.data_source or 'placeholder'}\n"
            f"Section: {spec.placement_section}\n\n"
            f"OUTPUT_PATH = \"{out_path}\"\n"
        )
        if results_json:
            user_msg += f"\nExperiment data (use exactly this):\n{results_json}\n"

        script_path = self.agent_dir / f"{spec.figure_id}_plot.py"
        error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            msg = user_msg
            if error:
                msg += f"\nYour previous script failed with this error; fix it:\n{error[-1500:]}\n"
            code = self._strip_fences(self._prompt(SYSTEM_PROMPT, msg, max_tokens=4096))
            script_path.write_text(f'OUTPUT_PATH = "{out_path}"\n\n' + code, encoding="utf-8")
            try:
                subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    capture_output=True, text=True, timeout=60, check=True,
                )
                break
            except subprocess.CalledProcessError as exc:
                error = exc.stderr or str(exc)
            except subprocess.TimeoutExpired as exc:
                error = str(exc)
            self.logger.warning("Figure %s script failed (attempt %d/%d): %s",
                                spec.figure_id, attempt, MAX_ATTEMPTS, error.strip().splitlines()[-1] if error.strip() else "")
        else:
            return None

        if not out_path.exists():
            self.logger.warning("Figure %s file not created", spec.figure_id)
            return None

        return GeneratedFigure(
            figure_id=spec.figure_id,
            file_path=out_path,
            caption=spec.description,
            label=f"fig:{spec.figure_id}",
        )

    @staticmethod
    def _strip_fences(code: str) -> str:
        lines = code.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
