"""Tactical motif detectors on python-chess, named with Lichess's puzzle themes (C1). Owner: motifs.

Our own MIT code: Lichess's tagger (ornicar/lichess-puzzler, AGPL-3.0) was read for definitions only.
"""

from __future__ import annotations

from ..models import Line, Motif

CORE_THEMES = ("fork", "pin", "skewer", "hangingPiece", "discoveredAttack", "backRankMate")
THEMES = CORE_THEMES + (
    "discoveredCheck", "doubleCheck", "trappedPiece", "smotheredMate", "mateIn1", "mateIn2", "mateIn3", "mateIn4",
    "mateIn5", "deflection", "attraction", "overloading", "advancedPawn",
)
# Themes whose detector met the precision gate (>= 0.8 against Lichess's labels); the others are reported as
# "a tactic" rather than by name.
GATED_THEMES: frozenset[str] = frozenset()


def training_url(theme: str) -> str:
    return f"https://lichess.org/training/{theme}"


def detect_moves(fen: str, moves_uci: list[str], max_plies: int = 8) -> list[Motif]:
    """Motifs along ``moves_uci`` from ``fen``. ``Motif.side`` is "first" when the side to move at ``fen``
    carries it out and "second" otherwise; ``Motif.line`` is left empty."""
    return []


def detect_line(line: Line, role: str) -> list[Motif]:
    """Motifs along a coaching line. ``role`` is "best" (what you missed: yours are the patterns you carry out)
    or "refutation" (what your move allowed: the opponent's patterns). Sets ``line`` and ``side`` ("you" /
    "opponent")."""
    return []
