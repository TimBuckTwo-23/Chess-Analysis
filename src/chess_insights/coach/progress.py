"""Study-plan targets against the previous report (C4). Owner: llm."""

from __future__ import annotations

from ..models import Report
from .config import CoachConfig


def annotate(report: Report, cfg: CoachConfig) -> None:
    """Fill ``report.coaching.progress`` from ``cfg.previous`` (the last report's JSON)."""
    return None
