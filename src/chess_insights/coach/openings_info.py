"""Opening facts from public sources for explained positions and choice points; theory exit (C3). Owner: sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import Coaching, ModuleResult
from .config import CoachConfig

if TYPE_CHECKING:
    from ..context import AnalysisContext


def annotate(ctx: "AnalysisContext", coaching: Coaching, modules: list[ModuleResult], cfg: CoachConfig) -> None:
    """Set ``Explanation.opening`` for opening positions, ``coaching.theory_exit``, and add facts under the
    openings tables."""
    return None
