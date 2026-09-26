"""Optional Maia-2 move probabilities at your rating (C4, extra [coach]). Owner: llm."""

from __future__ import annotations

from ..models import Report
from .config import CoachConfig


def annotate(report: Report, cfg: CoachConfig) -> None:
    """Set ``Explanation.maia`` and re-rank explanations by drop x P(best move)."""
    return None
