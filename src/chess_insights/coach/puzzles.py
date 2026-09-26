"""The puzzle PGN with the engine's line as the solution, motif themes and the explanation (C1). Owner: coach-core.

``mistakes.puzzles_to_pgn`` writes one move per puzzle (the engine's best move). Where the coaching layer has more
(``Coaching.puzzle_lines``: your errors' best lines from the deep and profile passes), the solution becomes the
best line (up to ``SOLUTION_PLIES`` plies), a ``Themes`` header carries the motif tags (Lichess theme names, only
those whose detector passed the gate) and the comment is the explanation when one matches the position. Every
other puzzle stays exactly as ``mistakes.puzzles_to_pgn`` writes it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, Optional

import chess

from ..analysis.mistakes import MistakeEvent, collect_errors, puzzles_to_pgn
from ..models import Coaching, Explanation, Line

if TYPE_CHECKING:
    from ..context import AnalysisContext
    from .deep import DeepResult, ProfileError

SOLUTION_PLIES = 8


def puzzle_key(game_id: str, ply: int) -> str:
    """The key of ``Coaching.puzzle_lines`` / ``puzzle_themes``: "<game_id>:<ply>"."""
    return f"{game_id}:{ply}"


def _solution(fen: str, line: Line) -> list[chess.Move]:
    """The line's first ``SOLUTION_PLIES`` moves, from ``fen``, stopping at the first one that isn't legal."""
    board = chess.Board(fen)
    out = []
    for uci in line.moves_uci[:SOLUTION_PLIES]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        out.append(move)
        board.push(move)
    return out


def _explanations(coaching: Coaching) -> dict[tuple[str, str], Explanation]:
    """(EPD, your move as "5...e5") -> its explanation."""
    return {(x.epd, x.played): x for x in coaching.explanations if x.text}


def _one(event: MistakeEvent, number: int) -> str:
    """``mistakes.puzzles_to_pgn`` for one puzzle, numbered ``number`` in the file."""
    return re.sub(r'\[Event "chess-insights puzzle 1"\]', f'[Event "chess-insights puzzle {number}"]',
                  puzzles_to_pgn([event]), count=1)


def _pgn_comment(text: str) -> str:
    return text.replace("}", ")").replace("{", "(")


def puzzles_pgn(events: list, coaching: Optional[Coaching] = None) -> str:
    """``mistakes.puzzles_to_pgn`` plus, where the coaching layer has them: the best line (up to 8 plies) as the
    solution, a ``Themes`` header with the motif tags and the explanation as the comment."""
    if coaching is None or not (coaching.puzzle_lines or coaching.explanations):
        return puzzles_to_pgn(events)
    explained = _explanations(coaching)
    chunks = []
    for number, e in enumerate(events, 1):
        if e.game.rules != "chess":  # the coaching layer reads standard chess only
            chunks.append(_one(e, number))
            continue
        line = coaching.puzzle_lines.get(puzzle_key(e.game.game_id, e.ply))
        moves = _solution(e.fen, line) if line is not None else []
        explanation = explained.get((e.epd, e.move_label))
        if not moves and explanation is not None and explanation.best_line is not None:
            moves = _solution(e.fen, explanation.best_line)
        if moves and moves[0].uci() == e.uci:  # a deeper search that prefers your move is no puzzle solution
            moves = []
        if not moves:
            chunks.append(_one(e, number))
            continue
        base = _one(e, number)
        headers = base.split("\n\n", 1)[0].splitlines()
        themes = [t for t in coaching.puzzle_themes.get(puzzle_key(e.game.game_id, e.ply), []) if t]
        if not themes and explanation is not None:
            themes = list(explanation.drill_themes)
        if themes:
            at = next((i for i, h in enumerate(headers) if h.startswith("[Annotator ")), len(headers))
            headers.insert(at, f'[Themes "{" ".join(dict.fromkeys(themes))}"]')
        if explanation is not None:
            comment = explanation.text
        else:
            comment = (f"In the game you played {e.move_label}, losing {e.drop:.0f} percentage points of winning "
                       f"chances. Find the better move and the line that follows.")
        board = chess.Board(e.fen)
        movetext = board.variation_san(moves)
        chunks.append("\n".join(headers) + f"\n\n{{{_pgn_comment(comment)}}} {movetext} *\n")
    return "\n".join(chunks)


def fill_puzzle_lines(
    ctx: "AnalysisContext",
    coaching: Coaching,
    lines: dict[tuple[str, str], "DeepResult"],
    profile: Iterable["ProfileError"] = (),
) -> None:
    """``coaching.puzzle_lines`` / ``puzzle_themes`` for your errors: the best line (up to ``SOLUTION_PLIES``) from
    the deep pass when the position was re-searched, else from the profile pass; the themes are the motifs you
    carry out along it, named only when their detector passed the gate (``motifs.GATED_THEMES``). An error whose
    deeper search prefers the move you played gets no line (the puzzle keeps the game analysis's move)."""
    from . import motifs
    from .deep import rebase
    from .explain import detect

    gated = frozenset(getattr(motifs, "GATED_THEMES", frozenset()) or ())
    by_ply = {(p.game_id, p.ply): p.best_line for p in profile if p.side == "you" and p.best_line is not None}
    for e in collect_errors(ctx.games, ctx.evals):
        if e.game.rules != "chess" or not e.uci:
            continue
        result = lines.get((e.epd, e.uci))
        line = result.best_line if result is not None else None
        if line is None or not line.moves_uci:
            line = by_ply.get((e.game.game_id, e.ply))
        if line is None or not line.moves_uci or line.moves_uci[0] == e.uci:
            continue
        line = rebase(line, e.fen, SOLUTION_PLIES)
        if not line.moves_uci:
            continue
        key = puzzle_key(e.game.game_id, e.ply)
        coaching.puzzle_lines[key] = line
        themes = [m.theme for m in detect(line, "best") if m.side == "you" and m.theme in gated]
        if themes:
            coaching.puzzle_themes[key] = list(dict.fromkeys(themes))
