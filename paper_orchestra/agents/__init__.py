"""Pipeline agents — one module per step."""

from .outline_agent import OutlineAgent
from .plotting_agent import PlottingAgent
from .experiment_agent import ExperimentAgent
from .literature_agent import LiteratureAgent
from .writing_agent import WritingAgent
from .review_agent import ReviewAgent
from .refinement_agent import RefinementAgent

__all__ = [
    "OutlineAgent",
    "PlottingAgent",
    "ExperimentAgent",
    "LiteratureAgent",
    "WritingAgent",
    "ReviewAgent",
    "RefinementAgent",
]
