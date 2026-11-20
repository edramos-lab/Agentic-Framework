"""Pydantic schemas for the REST API — decoupled from the core models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------

class ApiKeyIn(BaseModel):
    name: str  # e.g. "OPENAI_API_KEY", "ANTHROPIC_API_KEY"
    value: str


class ApiKeyOut(BaseModel):
    name: str
    is_set: bool


# ---------------------------------------------------------------------------
# Pipeline input (mirrors core models but as flat form data)
# ---------------------------------------------------------------------------

class IdeaInput(BaseModel):
    title: str
    abstract_draft: str = ""
    research_questions: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    methodology_sketch: str = ""
    keywords: list[str] = Field(default_factory=list)


class GuidelineInput(BaseModel):
    name: str = "NeurIPS 2025"
    page_limit: int | None = None
    format_notes: str = ""
    citation_style: str = "ieee"
    abstract_word_limit: int | None = None
    sections_required: list[str] = Field(default_factory=list)


class ColabInput(BaseModel):
    enabled: bool = True
    gpu_type: str = "T4"
    tpu_type: str = ""
    high_mem: bool = False
    session_name: str = "paper-orchestra"
    keep_session: bool = False
    env_vars: dict[str, str] = Field(default_factory=dict)
    pip_packages: list[str] = Field(default_factory=list)


class JetsonInput(BaseModel):
    host: str = ""  # IP or hostname of the Jetson; JETSON_HOST env var overrides it
    user: str = "jetson"
    port: int = 22
    ssh_key: str = "~/.ssh/id_ed25519"
    remote_dir: str = "/home/jetson/paper_orchestra"
    python: str = "python3"
    # Models already on the Jetson, read by examples/experiments/triton_benchmark.py
    env_vars: dict[str, str] = Field(
        default_factory=lambda: {"BENCH_MODEL_DIR": "/home/admin_ot/paper_orchestra/effvit_b0"}
    )
    upload_files: list[str] = Field(default_factory=list)


class ExperimentInput(BaseModel):
    datasets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    hardware: str = ""
    scripts_dir: str = ""
    timeout_minutes: int = 120
    backend: str = "colab"  # default target for plans that do not name one: colab | jetson
    colab: ColabInput = Field(default_factory=ColabInput)
    jetson: JetsonInput = Field(default_factory=JetsonInput)


class LLMInput(BaseModel):
    provider: str = "openai"      # "openai" | "anthropic"
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 4096


class PipelineStartRequest(BaseModel):
    """Full request to kick off a pipeline run."""
    idea: IdeaInput
    guideline: GuidelineInput
    llm: LLMInput = Field(default_factory=LLMInput)
    experiment_requirements: ExperimentInput | None = None
    latex_template_filename: str = "template.tex"
    style_samples_filename: str = ""  # an uploaded file of the author's own prose, for the writing agent
    max_parallel: int = 3
    refinement_max_cycles: int = 3
    refinement_min_improvement: float = 0.2


# ---------------------------------------------------------------------------
# Pipeline status / history
# ---------------------------------------------------------------------------

class StepStatus(BaseModel):
    """One sub-task inside an agent, e.g. a single experiment or review cycle."""
    step_id: str
    label: str
    status: str = "running"  # running | success | failed | skipped
    target: str = ""  # colab | jetson | ""
    detail: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AgentStatus(BaseModel):
    name: str
    status: str = "pending"  # pending | running | success | failed | skipped
    steps: list[StepStatus] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = ""


class PipelineRunStatus(BaseModel):
    run_id: str
    status: str = "pending"  # pending | running | completed | failed
    agents: list[AgentStatus] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str = ""
    output_dir: str = ""
    idea_title: str = ""


class PipelineRunSummary(BaseModel):
    run_id: str
    status: str
    idea_title: str
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunArtifact(BaseModel):
    filename: str  # path relative to the run's output directory
    size: int = 0
    url: str
