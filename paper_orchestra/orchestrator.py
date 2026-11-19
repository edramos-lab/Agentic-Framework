"""Pipeline orchestrator — coordinates all agents through the 5-step flow.

Flow
----
Inputs → Step 1 (Outline) → [Experiments ‖ Step 3 (Lit Review)] → Step 2 (Plotting)
       → Step 4 (Section Writing) → Step 5 (Iterative Refinement) → PDF

Plotting runs after the experiments so figures can use the real results.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base_agent import BaseAgent
from .llm import LLMClient
from .models import PipelineInputs, PipelineState

# Agent imports
from .agents.outline_agent import OutlineAgent
from .agents.plotting_agent import PlottingAgent
from .agents.experiment_agent import ExperimentAgent
from .agents.literature_agent import LiteratureAgent
from .agents.writing_agent import WritingAgent
from .agents.review_agent import ReviewAgent
from .agents.refinement_agent import RefinementAgent

logger = logging.getLogger(__name__)


class PaperOrchestra:
    """Main orchestrator that runs the entire pipeline.

    Parameters
    ----------
    llm : LLMClient
        Shared language-model client.
    work_dir : Path
        Root directory for all pipeline outputs.
    config : dict
        Pipeline-wide configuration (agent-level overrides nested by agent name).
    max_parallel : int
        Max threads for the parallel step (Steps 2 + Experiments + 3).
    """

    def __init__(
        self,
        llm: LLMClient,
        work_dir: Path = Path("./output"),
        config: dict[str, Any] | None = None,
        max_parallel: int = 3,
        agent_overrides: dict[str, Any] | None = None,
        on_agent_event: Any | None = None,
        on_agent_step: Any | None = None,
    ):
        self.llm = llm
        self.work_dir = Path(work_dir)
        self.config = config or {}
        if agent_overrides:
            for k, v in agent_overrides.items():
                self.config.setdefault(k, {}).update(v)
        self.max_parallel = max_parallel
        self._on_event = on_agent_event
        self._on_step = on_agent_step

        # Instantiate every agent
        self.outline_agent = OutlineAgent(llm, self.work_dir, self._agent_cfg("outline"))
        self.plotting_agent = PlottingAgent(llm, self.work_dir, self._agent_cfg("plotting"))
        self.experiment_agent = ExperimentAgent(llm, self.work_dir, self._agent_cfg("experiment"))
        self.literature_agent = LiteratureAgent(llm, self.work_dir, self._agent_cfg("literature"))
        self.writing_agent = WritingAgent(llm, self.work_dir, self._agent_cfg("writing"))
        self.review_agent = ReviewAgent(llm, self.work_dir, self._agent_cfg("review"))
        self.refinement_agent = RefinementAgent(
            llm, self.work_dir, self._agent_cfg("refinement"),
            review_agent=self.review_agent,
        )
        for agent in (self.outline_agent, self.plotting_agent, self.experiment_agent,
                      self.literature_agent, self.writing_agent, self.review_agent,
                      self.refinement_agent):
            agent.on_step = self._emit_step

    # ----- public entry point ------------------------------------------------

    def run(self, inputs: PipelineInputs) -> PipelineState:
        """Execute the full pipeline and return the final state."""
        logger.info("═══ PaperOrchestra starting ═══")
        t0 = time.perf_counter()

        state = PipelineState(inputs=inputs, work_dir=self.work_dir)
        state.started_at = datetime.now(timezone.utc)

        # Step 1 — Outline (sequential, everything else depends on it)
        state = self._run_step("Step 1: Outline", self.outline_agent, state)

        # Experiments + Step 3 — run in parallel
        if not inputs.experiment_requirements:
            self._emit("ExperimentAgent", "skipped", "No experiment requirements given")
        state = self._run_parallel(state)

        # Step 2 — Plotting runs after the experiments so figures can use their results
        state = self._run_step("Step 2: Plotting", self.plotting_agent, state)

        # Step 4 — Section Writing (needs everything above)
        state = self._run_step("Step 4: Section Writing", self.writing_agent, state)

        # Step 5 — Iterative Refinement
        state = self._run_step("Step 5: Iterative Refinement", self.refinement_agent, state)

        state.finished_at = datetime.now(timezone.utc)
        elapsed = time.perf_counter() - t0
        logger.info("═══ PaperOrchestra finished in %.1fs ═══", elapsed)
        return state

    # ----- internals ---------------------------------------------------------

    def _agent_cfg(self, name: str) -> dict[str, Any]:
        return self.config.get(name, {})

    def _emit_step(self, step: dict[str, Any]) -> None:
        if self._on_step:
            try:
                self._on_step(step)
            except Exception:
                pass

    def _emit(self, agent_name: str, event: str, detail: str = "") -> None:
        if self._on_event:
            try:
                self._on_event(agent_name, event, detail)
            except Exception:
                pass  # never crash the pipeline for a UI event

    def _run_step(self, label: str, agent: BaseAgent, state: PipelineState) -> PipelineState:
        agent_name = agent.__class__.__name__
        logger.info("── %s ──", label)
        self._emit(agent_name, "running", label)
        try:
            state = agent.run(state)
            self._emit(agent_name, "success")
            return state
        except Exception as exc:
            self._emit(agent_name, "failed", str(exc))
            raise

    def _run_parallel(self, state: PipelineState) -> PipelineState:
        """Run the Experiments and Step 3 (Literature) concurrently.

        Each agent receives a *copy* of the state to avoid races, then we
        merge their outputs back into one state.
        """
        logger.info("── Experiments + Step 3 (parallel) ──")

        agents: list[tuple[str, BaseAgent]] = [
            ("LiteratureAgent", self.literature_agent),
        ]
        if state.inputs.experiment_requirements:
            agents.append(("ExperimentAgent", self.experiment_agent))

        results: dict[str, PipelineState] = {}

        def _run_parallel_agent(name: str, agent: BaseAgent, s: PipelineState) -> PipelineState:
            self._emit(name, "running")
            try:
                result = agent.run(s)
                self._emit(name, "success")
                return result
            except Exception as exc:
                self._emit(name, "failed", str(exc))
                raise

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            future_map = {
                pool.submit(_run_parallel_agent, name, agent, state.model_copy(deep=True)): name
                for name, agent in agents
            }
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    results[name] = future.result()
                    logger.info("  ✔ %s completed", name)
                except Exception:
                    logger.exception("  ✘ %s failed", name)
                    raise

        # Merge parallel results back into the canonical state
        if "LiteratureAgent" in results:
            state.literature = results["LiteratureAgent"].literature
        if "ExperimentAgent" in results:
            state.experiment_output = results["ExperimentAgent"].experiment_output

        return state
