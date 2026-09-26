"""Re-search critical positions with Stockfish and keep whole lines (C1). Owner: coach-core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ..models import CriticalPosition, Line
from .config import CoachConfig

if TYPE_CHECKING:
    from ..context import AnalysisContext


@dataclass
class DeepResult:
    """Both lines start at the position before your move and are scored from your side.

    ``best_line.moves_uci[0]`` is the engine's move; ``refutation.moves_uci[0]`` is the move you played, followed
    by the engine's best play for both sides.
    """

    epd: str
    played_uci: str
    best_line: Optional[Line]
    refutation: Optional[Line]
    alternatives: list[Line] = field(default_factory=list)  # MultiPV lines 2..N (from your side)
    engine: str = ""
    depth: Optional[int] = None


@dataclass
class ProfileError:
    """One error by either side in an engine-analysed game, with short lines for the motif profile (C2)."""

    game_id: str
    ply: int
    side: str  # "you" | "opponent"
    time_class: str
    fen: str  # before the move
    played_uci: str
    drop: float
    best_line: Optional[Line] = None
    refutation: Optional[Line] = None


def analyse_positions(
    positions: list[CriticalPosition], cfg: CoachConfig, notes: list[str]
) -> dict[tuple[str, str], DeepResult]:
    """(EPD, played UCI) -> DeepResult, cached under ``cfg.cache_dir``/<engine>/<depth>/."""
    return {}


def profile_lines(ctx: "AnalysisContext", cfg: CoachConfig, notes: list[str]) -> list[ProfileError]:
    """Every error (both sides, drop >= mistakes.MIN_DROP) in the engine-analysed games with a best line and a
    refutation at ``cfg.profile_depth`` (single PV), cached like ``analyse_positions``."""
    return []
