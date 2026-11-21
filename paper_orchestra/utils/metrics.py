"""Helpers for passing measured results to an LLM without flooding the prompt."""

from __future__ import annotations

from typing import Any


def compact_metrics(value: Any) -> Any:
    """Copy of a metrics dict without ROC curves or long numeric arrays; AUCs, accuracy and matrices stay."""
    if isinstance(value, dict):
        return {k: compact_metrics(v) for k, v in value.items() if k != "roc"}
    if isinstance(value, list) and len(value) > 12 and all(isinstance(x, (int, float)) for x in value):
        return f"<{len(value)} values, plotted in the figures>"
    return value
