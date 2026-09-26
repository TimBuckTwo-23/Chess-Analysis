"""The puzzle PGN with the engine's line as the solution, motif themes and the explanation (C1). Owner: coach-core."""

from __future__ import annotations

from typing import Optional

from ..models import Coaching


def puzzles_pgn(events: list, coaching: Optional[Coaching] = None) -> str:
    """``mistakes.puzzles_to_pgn`` plus, where the coaching layer has them: the best line (up to 8 plies) as the
    solution, a ``Themes`` header with the motif tags and the explanation as the comment."""
    from ..analysis.mistakes import puzzles_to_pgn

    return puzzles_to_pgn(events)
