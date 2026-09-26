"""Puzzle packs for your most-missed motifs and your openings, and the review schedule (C2). Owner: drills."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from ..models import Coaching, ModuleResult
from .config import CoachConfig

if TYPE_CHECKING:
    from ..context import AnalysisContext


def annotate(ctx: "AnalysisContext", coaching: Coaching, modules: list[ModuleResult], cfg: CoachConfig,
             today: date) -> None:
    """Fill ``coaching.drills`` (and write the PGN packs next to the report), ``coaching.review`` and
    ``coaching.review_due``."""
    return None
