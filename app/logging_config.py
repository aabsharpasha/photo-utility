"""Structured logging configuration."""

import logging
import sys
from typing import Any

from app.config import get_settings


def configure_logging() -> None:
    """Configure root logger and format."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
        force=True,
    )
    # Reduce noise from third-party libs
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("insightface").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_extra(**kwargs: Any) -> dict[str, Any]:
    """Build dict for structured logging (e.g. request_id, duration_ms)."""
    return {k: v for k, v in kwargs.items() if v is not None}
