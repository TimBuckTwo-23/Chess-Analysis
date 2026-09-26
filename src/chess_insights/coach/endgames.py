"""Tablebase-checked endgames (C3). Owner: sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import Coaching
from .config import CoachConfig

if TYPE_CHECKING:
    from ..context import AnalysisContext


def annotate(ctx: "AnalysisContext", coaching: Coaching, cfg: CoachConfig) -> None:
    """Fill ``coaching.endgames`` and ``Explanation.tablebase`` for positions with 7 pieces or fewer."""
    return None
