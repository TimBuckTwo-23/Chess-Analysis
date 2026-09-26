"""Pick the positions worth explaining (C1). Owner: coach-core."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import CriticalPosition, ModuleResult

if TYPE_CHECKING:
    from ..context import AnalysisContext


def select_critical(ctx: "AnalysisContext", modules: list[ModuleResult], max_positions: int) -> list[CriticalPosition]:
    """Repeated mistakes first, then your errors losing at least ``mistakes.MIN_DROP`` win-% points (costliest
    first), then choice points in your main lines; one entry per (EPD, move); at most ``max_positions``."""
    return []
