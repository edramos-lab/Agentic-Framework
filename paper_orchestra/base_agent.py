"""Abstract base class for all pipeline agents."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from .llm import LLMClient
from .models import PipelineState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Every agent in the pipeline inherits from this class.

    It provides:
    - access to the shared LLM client
    - structured logging
    - work-directory management
    - timing
    """

    name: str = "BaseAgent"

    def __init__(self, llm: LLMClient, work_dir: Path, config: dict[str, Any] | None = None):
        self.llm = llm
        self.work_dir = work_dir
        self.config = config or {}
        self.logger = logging.getLogger(f"paper_orchestra.{self.name}")
        # Set by the orchestrator so UIs can follow work inside an agent
        self.on_step: Callable[[dict[str, Any]], None] | None = None

    # ----- public entry point ------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Execute the agent and return the updated state."""
        self.logger.info("▶ %s starting", self.name)
        self._ensure_dirs()
        t0 = time.perf_counter()
        try:
            state = self.execute(state)
            elapsed = time.perf_counter() - t0
            self.logger.info("✔ %s finished in %.1fs", self.name, elapsed)
        except Exception:
            elapsed = time.perf_counter() - t0
            self.logger.exception("✘ %s failed after %.1fs", self.name, elapsed)
            raise
        return state

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        """Subclasses implement their logic here."""
        ...

    # ----- helpers -----------------------------------------------------------

    def step(self, step_id: str, label: str, status: str = "running",
             target: str = "", detail: str = "") -> None:
        """Report progress of one sub-task (e.g. one experiment) to the UI."""
        if self.on_step:
            try:
                self.on_step({"agent": self.name, "step_id": step_id, "label": label,
                              "status": status, "target": target, "detail": detail})
            except Exception:
                pass  # a broken UI callback must never stop the pipeline

    @property
    def agent_dir(self) -> Path:
        """Dedicated sub-directory for this agent's outputs."""
        return self.work_dir / self.name.lower().replace(" ", "_")

    def _ensure_dirs(self) -> None:
        self.agent_dir.mkdir(parents=True, exist_ok=True)

    def _prompt(self, system: str, user: str, **kwargs) -> str:
        """Convenience: single-turn chat with system + user message."""
        return self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )

    def _prompt_json(self, system: str, user: str, **kwargs) -> dict:
        """Convenience: single-turn chat returning parsed JSON."""
        return self.llm.chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )

    def _write(self, filename: str, content: str) -> Path:
        """Write a file into this agent's output directory."""
        p = self.agent_dir / filename
        p.write_text(content, encoding="utf-8")
        self.logger.debug("Wrote %s (%d bytes)", p, len(content))
        return p
