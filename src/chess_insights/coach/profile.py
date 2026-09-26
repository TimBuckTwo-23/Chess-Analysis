"""Motif profile: which patterns you miss and allow, against your opponents in the same games (C2). Owner: drills."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import Coaching, ModuleResult
from .config import CoachConfig

if TYPE_CHECKING:
    from ..context import AnalysisContext


def annotate(ctx: "AnalysisContext", coaching: Coaching, modules: list[ModuleResult], cfg: CoachConfig) -> None:
    """Fill ``coaching.motif_profile`` / ``motif_chart``, add them to the Engine review section, and add motif
    claims (claim rule at STRICT_ALPHA, BH across motifs) to that section's insights."""
    return None
