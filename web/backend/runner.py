"""Pipeline runner — bridges the web API to PaperOrchestra's core engine.

Runs the pipeline in a background thread and streams agent-level status
updates through an asyncio queue that the WebSocket endpoint consumes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings, vault
from .schemas import AgentStatus, PipelineRunStatus, PipelineStartRequest, StepStatus

logger = logging.getLogger("po.runner")

# ---------------------------------------------------------------------------
# In-memory run registry (swap for a DB in production)
# ---------------------------------------------------------------------------

_runs: dict[str, PipelineRunStatus] = {}
_queues: dict[str, asyncio.Queue] = {}  # run_id → event queue for WS
_loops: dict[str, asyncio.AbstractEventLoop] = {}  # the server loop that owns each queue


def get_run(run_id: str) -> PipelineRunStatus | None:
    return _runs.get(run_id)


def list_runs() -> list[PipelineRunStatus]:
    return sorted(_runs.values(), key=lambda r: r.started_at or datetime.min, reverse=True)


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def _emit(run_id: str, event: dict) -> None:
    """Push an event to the WebSocket queue from a worker thread.

    asyncio.Queue is not thread-safe, so the put must be scheduled on the loop
    that owns the queue; a bare put_nowait from another thread never wakes the
    waiting WebSocket handler.
    """
    q, loop = _queues.get(run_id), _loops.get(run_id)
    if q is None or loop is None or loop.is_closed():
        return
    loop.call_soon_threadsafe(q.put_nowait, event)


def _update_agent(run_id: str, name: str, status: str, message: str = "") -> None:
    run = _runs[run_id]
    for a in run.agents:
        if a.name == name:
            a.status = status
            a.message = message
            if status == "running" and not a.started_at:
                a.started_at = datetime.now(timezone.utc)
            if status in ("success", "failed", "skipped"):
                a.finished_at = datetime.now(timezone.utc)
            break
    _emit(run_id, {
        "type": "agent_update",
        "agent": name,
        "status": status,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _update_step(run_id: str, step: dict) -> None:
    """Record a sub-task of an agent on the run snapshot (polled by the UI)."""
    run = _runs[run_id]
    now = datetime.now(timezone.utc)
    for a in run.agents:
        if a.name != step["agent"]:
            continue
        existing = next((s for s in a.steps if s.step_id == step["step_id"]), None)
        if existing is None:
            existing = StepStatus(step_id=step["step_id"], label=step["label"], started_at=now)
            a.steps.append(existing)
        existing.status = step["status"]
        existing.target = step.get("target", "")
        existing.detail = step.get("detail", "")
        if step["status"] != "running":
            existing.finished_at = now
        break
    _emit(run_id, {"type": "agent_step", **step})


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

AGENT_NAMES = [
    "OutlineAgent",
    "ExperimentAgent",
    "LiteratureAgent",
    "PlottingAgent",
    "WritingAgent",
    "RefinementAgent",  # includes the review cycles, reported as its sub-steps
]


def _run_pipeline(run_id: str, req: PipelineStartRequest) -> None:
    """Execute the full PaperOrchestra pipeline (runs in a worker thread)."""
    run = _runs[run_id]
    run.status = "running"
    run.started_at = datetime.now(timezone.utc)
    _emit(run_id, {"type": "pipeline_started", "run_id": run_id})

    try:
        # --- build core inputs from the web request ---
        from paper_orchestra.models import (
            ColabConfig,
            ConferenceGuideline,
            ExperimentRequirements,
            IdeaSummary,
            JetsonConfig,
            PipelineInputs,
        )
        from paper_orchestra.llm import LLMClient
        from paper_orchestra.orchestrator import PaperOrchestra

        # Inject API keys from the vault into the process env
        for k, v in vault.get_for_env().items():
            os.environ[k] = v

        # LLM client
        api_key = vault.get(
            "OPENAI_API_KEY" if req.llm.provider == "openai" else "ANTHROPIC_API_KEY"
        ) or ""
        llm = LLMClient(
            provider=req.llm.provider,
            model=req.llm.model,
            api_key=api_key,
            temperature=req.llm.temperature,
            max_tokens=req.llm.max_tokens,
        )

        # Map web schemas → core models
        idea = IdeaSummary(
            title=req.idea.title,
            abstract_draft=req.idea.abstract_draft,
            research_questions=req.idea.research_questions,
            contributions=req.idea.contributions,
            methodology_sketch=req.idea.methodology_sketch,
            keywords=req.idea.keywords,
        )

        guideline = ConferenceGuideline(
            name=req.guideline.name,
            page_limit=req.guideline.page_limit,
            format_notes=req.guideline.format_notes,
            citation_style=req.guideline.citation_style,
            abstract_word_limit=req.guideline.abstract_word_limit,
            sections_required=req.guideline.sections_required,
        )

        # Template — look in uploads first, then examples
        template_path = settings.uploads_dir / req.latex_template_filename
        if not template_path.exists():
            template_path = Path("examples/inputs") / req.latex_template_filename

        exp_reqs = None
        if req.experiment_requirements:
            er = req.experiment_requirements
            exp_reqs = ExperimentRequirements(
                datasets=er.datasets,
                models=er.models,
                metrics=er.metrics,
                baselines=er.baselines,
                hardware=er.hardware,
                scripts_dir=er.scripts_dir,
                timeout_minutes=er.timeout_minutes,
                backend=er.backend,
                jetson=JetsonConfig(**er.jetson.model_dump()),
                colab=ColabConfig(
                    enabled=er.colab.enabled,
                    gpu_type=er.colab.gpu_type,
                    tpu_type=er.colab.tpu_type,
                    high_mem=er.colab.high_mem,
                    session_name=er.colab.session_name,
                    keep_session=er.colab.keep_session,
                    env_vars=er.colab.env_vars,
                    pip_packages=er.colab.pip_packages,
                ),
            )

        inputs = PipelineInputs(
            idea=idea,
            latex_template_path=template_path,
            guideline=guideline,
            experiment_requirements=exp_reqs,
        )

        output_dir = settings.runs_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        run.output_dir = str(output_dir)

        # Build the orchestrator with a status callback
        def on_agent_event(agent_name: str, event: str, detail: str = ""):
            _update_agent(run_id, agent_name, event, detail)

        def on_agent_step(step: dict):
            _update_step(run_id, step)

        overrides: dict[str, Any] = {
            "refinement": {
                "max_cycles": req.refinement_max_cycles,
                "min_improvement": req.refinement_min_improvement,
            }
        }
        if req.style_samples_filename:
            samples = settings.uploads_dir / Path(req.style_samples_filename).name
            if samples.exists():
                overrides["writing"] = {"style_samples_path": str(samples)}
            else:
                _emit(run_id, {"type": "log", "message": f"Style samples not found: {samples.name}"})

        orchestra = PaperOrchestra(
            llm=llm,
            work_dir=output_dir,
            max_parallel=req.max_parallel,
            agent_overrides=overrides,
            on_agent_event=on_agent_event,
            on_agent_step=on_agent_step,
        )

        _emit(run_id, {"type": "log", "message": "Pipeline initialized, starting execution..."})

        state = orchestra.run(inputs)

        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        _emit(run_id, {
            "type": "pipeline_completed",
            "run_id": run_id,
            "output_dir": str(output_dir),
        })

    except Exception as exc:
        logger.exception("Pipeline %s failed", run_id)
        run.status = "failed"
        run.error = str(exc)
        run.finished_at = datetime.now(timezone.utc)
        _emit(run_id, {
            "type": "pipeline_failed",
            "run_id": run_id,
            "error": str(exc),
        })
    finally:
        _emit(run_id, {"type": "done"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_pipeline(req: PipelineStartRequest, loop: asyncio.AbstractEventLoop) -> str:
    """Start a pipeline run and return the run_id."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    agents = [AgentStatus(name=n) for n in AGENT_NAMES]
    _runs[run_id] = PipelineRunStatus(run_id=run_id, agents=agents, idea_title=req.idea.title)
    _queues[run_id] = asyncio.Queue()
    _loops[run_id] = loop

    thread = threading.Thread(
        target=_run_pipeline,
        args=(run_id, req),
        name=f"pipeline-{run_id}",
        daemon=True,
    )
    thread.start()
    return run_id


def get_event_queue(run_id: str) -> asyncio.Queue | None:
    return _queues.get(run_id)
