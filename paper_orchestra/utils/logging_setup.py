"""Configure structured logging with Rich."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Set up logging with colored output via Rich (falls back to basic)."""
    numeric = getattr(logging, level.upper(), logging.INFO)

    try:
        from rich.logging import RichHandler

        handler = RichHandler(
            level=numeric,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        fmt = "%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s"

    handler.setLevel(numeric)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("paper_orchestra")
    root.setLevel(numeric)
    root.handlers.clear()
    root.addHandler(handler)
