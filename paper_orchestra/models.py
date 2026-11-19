"""Data models shared across all agents in the PaperOrchestra pipeline."""

from __future__ import annotations

import enum
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class IdeaSummary(BaseModel):
    """The high-level research idea provided by the user (I)."""
    title: str
    abstract_draft: str = ""
    research_questions: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    methodology_sketch: str = ""
    keywords: list[str] = Field(default_factory=list)
    authors: str = ""  # goes into the template's %AUTHORS% slot; empty prints "Anonymous Authors"


class ConferenceGuideline(BaseModel):
    """Conference or journal formatting requirements (G)."""
    name: str
    page_limit: int | None = None
    format_notes: str = ""
    citation_style: str = "ieee"
    abstract_word_limit: int | None = None
    sections_required: list[str] = Field(default_factory=list)


class ColabConfig(BaseModel):
    """Google Colab CLI execution settings."""
    enabled: bool = True
    gpu_type: str = "T4"  # T4 | L4 | G4 | H100 | A100
    tpu_type: str = ""  # v5e1 | v6e1 — empty means no TPU
    high_mem: bool = False  # requires Colab Pro/Pro+
    session_name: str = "paper-orchestra"
    keep_session: bool = False  # keep VM alive between experiments
    env_vars: dict[str, str] = Field(default_factory=dict)
    pip_packages: list[str] = Field(default_factory=list)
    upload_files: list[str] = Field(default_factory=list)  # local paths to upload before run
    download_patterns: list[str] = Field(default_factory=list)  # remote paths to download after run


class JetsonConfig(BaseModel):
    """SSH settings for running experiments on a Jetson Orin Nano."""
    host: str = ""  # IP or hostname; the JETSON_HOST env var overrides it
    user: str = "jetson"
    port: int = 22
    ssh_key: str = ""  # path to private key; empty uses the ssh-agent / default key
    remote_dir: str = "/home/jetson/paper_orchestra"
    python: str = "python3"
    env_vars: dict[str, str] = Field(default_factory=dict)
    upload_files: list[str] = Field(default_factory=list)  # local paths copied to remote_dir/<experiment>/
    connect_timeout: int = 10


class ExperimentRequirements(BaseModel):
    """Specification for experiments to run (new input)."""
    datasets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    hardware: str = ""
    scripts_dir: str = ""
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    timeout_minutes: int = 120
    backend: str = "colab"  # default target for plans that do not set one: colab | jetson
    colab: ColabConfig = Field(default_factory=ColabConfig)
    jetson: JetsonConfig = Field(default_factory=JetsonConfig)


class PipelineInputs(BaseModel):
    """All raw materials fed into Step 1."""
    idea: IdeaSummary
    latex_template_path: Path
    guideline: ConferenceGuideline
    experiment_requirements: ExperimentRequirements | None = None


# ---------------------------------------------------------------------------
# Step 1 — Outline
# ---------------------------------------------------------------------------

class VisualizationPlan(BaseModel):
    """Which figures/tables to generate and where they go."""
    figure_id: str
    description: str
    figure_type: str = "plot"  # plot | table | diagram | photograph
    data_source: str = ""
    placement_section: str = ""


class ResearchNode(BaseModel):
    """A node in the research graph — a concept or prior-work item to cite."""
    node_id: str
    label: str
    query: str = ""
    importance: str = "normal"  # critical | normal | supplementary


class WritingPlanSection(BaseModel):
    """Plan for a single section of the manuscript."""
    section_id: str
    title: str
    subsections: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    target_words: int = 500
    figures: list[str] = Field(default_factory=list)  # figure_ids


class ExperimentPlan(BaseModel):
    """Plan for how experiments should be executed and documented."""
    experiment_id: str
    name: str
    description: str = ""
    script: str = ""
    expected_outputs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    target: str = ""  # colab (training) | jetson (edge deployment); empty uses the default backend


class Outline(BaseModel):
    """Complete JSON outline produced by Step 1."""
    title: str
    visualization_plan: list[VisualizationPlan] = Field(default_factory=list)
    research_graph: list[ResearchNode] = Field(default_factory=list)
    writing_plan: list[WritingPlanSection] = Field(default_factory=list)
    experiment_plan: list[ExperimentPlan] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 2 — Plotting
# ---------------------------------------------------------------------------

class GeneratedFigure(BaseModel):
    """A figure or table produced by the Plotting Agent."""
    figure_id: str
    file_path: Path
    caption: str = ""
    label: str = ""  # LaTeX label


# ---------------------------------------------------------------------------
# NEW — Experimental Runs
# ---------------------------------------------------------------------------

class ExperimentResult(BaseModel):
    """Result of a single experiment run."""
    experiment_id: str
    status: str = "pending"  # pending | running | success | failed
    log_path: Path | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Path] = Field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    target: str = ""  # colab | jetson
    device: str = ""  # human-readable hardware label, e.g. "Jetson Orin Nano (jetson-orin-02)"
    colab_session: str = ""  # Colab session name used
    colab_notebook_path: Path | None = None  # output .ipynb from Colab


class ExperimentRunsOutput(BaseModel):
    """Aggregate output from the Experimental Runs agent."""
    results: list[ExperimentResult] = Field(default_factory=list)
    summary_table_path: Path | None = None
    documentation_path: Path | None = None
    latex_tables: dict[str, str] = Field(default_factory=dict)  # experiment_id -> LaTeX tabular


# ---------------------------------------------------------------------------
# Step 3 — Literature Review
# ---------------------------------------------------------------------------

class BibEntry(BaseModel):
    """A single bibliography entry."""
    cite_key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    abstract: str = ""
    bibtex: str = ""
    source: str = ""  # openalex | semantic_scholar
    scite: dict[str, Any] = Field(default_factory=dict)  # supporting / contradicting / mentioning counts


class LiteratureReviewOutput(BaseModel):
    """Output from the Literature Review Agent."""
    intro_paragraphs: list[str] = Field(default_factory=list)
    related_work_paragraphs: list[str] = Field(default_factory=list)
    bib_entries: list[BibEntry] = Field(default_factory=list)
    bib_file_path: Path | None = None


# ---------------------------------------------------------------------------
# Step 4 — Section Writing
# ---------------------------------------------------------------------------

class ManuscriptDraft(BaseModel):
    """The full manuscript draft produced by the Section Writing Agent."""
    latex_source: str = ""
    output_path: Path | None = None
    sections_written: list[str] = Field(default_factory=list)
    word_count: int = 0


# ---------------------------------------------------------------------------
# Step 5 — Iterative Refinement
# ---------------------------------------------------------------------------

class ReviewVerdict(str, enum.Enum):
    IMPROVEMENT = "improvement"
    NEUTRAL = "neutral"
    DEGRADATION = "degradation"


class ReviewScore(BaseModel):
    """Score from the AgentReview for a single revision cycle."""
    cycle: int
    verdict: ReviewVerdict
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""
    suggestions: list[str] = Field(default_factory=list)


class RefinementOutput(BaseModel):
    """Final output of the iterative refinement loop."""
    final_latex: str = ""
    final_pdf_path: Path | None = None
    review_history: list[ReviewScore] = Field(default_factory=list)
    total_cycles: int = 0
    stopped_reason: str = ""  # max_cycles | no_improvement | target_reached


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineState(BaseModel):
    """Tracks the full state as it flows through the pipeline."""
    inputs: PipelineInputs
    outline: Outline | None = None
    figures: list[GeneratedFigure] = Field(default_factory=list)
    experiment_output: ExperimentRunsOutput | None = None
    literature: LiteratureReviewOutput | None = None
    draft: ManuscriptDraft | None = None
    refinement: RefinementOutput | None = None
    work_dir: Path = Path("./output")
    started_at: datetime | None = None
    finished_at: datetime | None = None
