"""Turn lines, motifs and concepts into a short explanation and a board (C1). Owner: coach-core."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import CriticalPosition, Explanation
from .config import CoachConfig

if TYPE_CHECKING:
    from ..context import AnalysisContext
    from .deep import DeepResult


def explain_all(
    ctx: "AnalysisContext",
    positions: list[CriticalPosition],
    lines: dict[tuple[str, str], "DeepResult"],
    cfg: CoachConfig,
    notes: list[str],
) -> list[Explanation]:
    """One Explanation per position with lines: motifs, concept deltas, facts, text, diagram and chart."""
    return []
