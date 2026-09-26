"""The puzzle PGN with the engine's line as the solution, motif themes and the explanation (C1). Owner: coach-core.

``mistakes.puzzles_to_pgn`` writes one move per puzzle (the engine's best move). Where the coaching layer has more
(``Coaching.puzzle_lines``: your errors' best lines from the deep and profile passes), the solution becomes the
best line (up to ``SOLUTION_PLIES`` plies), a ``Themes`` header carries the motif tags (Lichess theme names, only
those whose detector passed the gate) and the explanation, when one matches the position and your move, is the
comment after the solution's first move: the comment before it stays the puzzle's question, since the explanation
names the better move (a Lichess study shows that first comment before you try). Every other puzzle stays exactly
as ``mistakes.puzzles_to_pgn`` writes it.

The ``Themes`` header describes the moves printed, nothing else: the patterns you carry out along that solution
(``Coaching.puzzle_themes`` for the line ``fill_puzzle_lines`` set, or the explanation's best-line motifs when the
solution is the explanation's best line), never what your opponent's refutation of the game move did.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, Optional

import chess

from ..analysis.mistakes import MistakeEvent, build_puzzles, collect_errors, puzzles_to_pgn
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


def _played_uci(x: Explanation) -> str:
    """The move an explanation is about, in UCI (its refutation starts with it; else its label read in its FEN)."""
    if x.refutation is not None and x.refutation.moves_uci:
        return x.refutation.moves_uci[0]
    try:
        board = chess.Board(x.fen)
        return board.parse_san(x.played.split(".")[-1]).uci()
    except ValueError:
        return ""


def _explanations(coaching: Coaching) -> dict[tuple[str, str], Explanation]:
    """(EPD, your move in UCI) -> its explanation (by UCI, so a transposition with other move numbers matches)."""
    return {(x.epd, _played_uci(x)): x for x in coaching.explanations if x.text}


def _one(event: MistakeEvent, number: int) -> str:
    """``mistakes.puzzles_to_pgn`` for one puzzle, numbered ``number`` in the file."""
    return re.sub(r'\[Event "chess-insights puzzle 1"\]', f'[Event "chess-insights puzzle {number}"]',
                  puzzles_to_pgn([event]), count=1)


def _pgn_comment(text: str) -> str:
    return text.replace("}", ")").replace("{", "(")


def solution_themes(explanation: Explanation, plies: int) -> list[str]:
    """The motifs an explanation found along its best line that you carry out within the first ``plies`` moves
    (the solution printed). Not ``drill_themes``: those include what your opponent's refutation did."""
    return list(dict.fromkeys(m.theme for m in explanation.motifs or []
                              if m.line == "best" and m.side == "you" and m.ply < plies))


def puzzles_pgn(events: list, coaching: Optional[Coaching] = None) -> str:
    """``mistakes.puzzles_to_pgn`` plus, where the coaching layer has them: the best line (up to 8 plies) as the
    solution, a ``Themes`` header with the motif tags and the explanation as the comment after the first move."""
    if coaching is None or not (coaching.puzzle_lines or coaching.explanations):
        return puzzles_to_pgn(events)
    explained = _explanations(coaching)
    chunks = []
    for number, e in enumerate(events, 1):
        if e.game.rules != "chess":  # the coaching layer reads standard chess only
            chunks.append(_one(e, number))
            continue
        key = puzzle_key(e.game.game_id, e.ply)
        line = coaching.puzzle_lines.get(key)
        moves = _solution(e.fen, line) if line is not None else []
        explanation = explained.get((e.epd, e.uci))
        from_explanation = False
        if not moves and explanation is not None and explanation.best_line is not None:
            moves = _solution(e.fen, explanation.best_line)
            from_explanation = True
        if moves and moves[0].uci() == e.uci:  # a deeper search that prefers your move is no puzzle solution
            moves = []
        if not moves:
            chunks.append(_one(e, number))
            continue
        base = _one(e, number)
        headers = base.split("\n\n", 1)[0].splitlines()
        if from_explanation:
            themes = solution_themes(explanation, len(moves))
        else:
            themes = [t for t in coaching.puzzle_themes.get(key, []) if t]
        if themes:
            at = next((i for i, h in enumerate(headers) if h.startswith("[Annotator ")), len(headers))
            headers.insert(at, f'[Themes "{" ".join(dict.fromkeys(themes))}"]')
        question = (f"In the game you played {e.move_label}, losing {e.drop:.0f} percentage points of winning "
                    f"chances. Find the better move and the line that follows.")
        board = chess.Board(e.fen)
        if explanation is None:
            movetext = board.variation_san(moves)
        else:  # the explanation right after the solution's first move: "5...Nf6 {After 5...e5, ...} 6.Nxc6 ..."
            movetext = f"{board.variation_san(moves[:1])} {{{_pgn_comment(explanation.text)}}}"
            if len(moves) > 1:
                board.push(moves[0])
                movetext += " " + board.variation_san(moves[1:])
        chunks.append("\n".join(headers) + f"\n\n{{{_pgn_comment(question)}}} {movetext} *\n")
    return "\n".join(chunks)


def puzzle_keys(ctx: "AnalysisContext", coaching: Coaching, errors: Optional[list[MistakeEvent]] = None) -> set[str]:
    """The errors of yours whose best lines and themes ``Coaching.puzzle_lines`` / ``puzzle_themes`` keep: the ones
    the puzzle export can use (``mistakes.build_puzzles``: at most ``PUZZLE_LIMIT``, costliest first, with a known
    better move; standard chess), plus every error in a position an explanation covers (your move there).
    ``errors``: ``mistakes.collect_errors`` of the games, when the caller has it."""
    keys = {puzzle_key(e.game.game_id, e.ply) for e in build_puzzles(ctx.games, ctx.evals) if e.game.rules == "chess"}
    explained = {(x.epd, _played_uci(x)) for x in coaching.explanations}
    if explained:
        errors = collect_errors(ctx.games, ctx.evals) if errors is None else errors
        keys |= {puzzle_key(e.game.game_id, e.ply) for e in errors if (e.epd, e.uci) in explained}
    return keys


def fill_puzzle_lines(
    ctx: "AnalysisContext",
    coaching: Coaching,
    lines: dict[tuple[str, str], "DeepResult"],
    profile: Iterable["ProfileError"] = (),
) -> None:
    """``coaching.puzzle_lines`` / ``puzzle_themes`` for your errors in ``puzzle_keys`` (the puzzle export's and
    the explained ones: the JSON stays small): the best line (up to ``SOLUTION_PLIES``) from the deep pass when the
    position was re-searched, else from the profile pass; the themes are the motifs you carry out along it (read
    with the position before your opponent's last move), named only when their detector passed the gate
    (``motifs.GATED_THEMES``). An error whose deeper search prefers the move you played gets no line (the puzzle
    keeps the game analysis's move). Every key settled here gets its theme list, empty when nothing was found or
    when the deeper search cleared the move, so the motif profile (``profile._puzzle_lines``, which only fills keys
    that have none) never adds its shorter line or tags the key with the patterns of another line."""
    from . import motifs
    from .deep import previous_fens, rebase
    from .explain import detect

    gated = frozenset(getattr(motifs, "GATED_THEMES", frozenset()) or ())
    by_ply = {(p.game_id, p.ply): p.best_line for p in profile if p.side == "you" and p.best_line is not None}
    errors = collect_errors(ctx.games, ctx.evals)
    wanted = puzzle_keys(ctx, coaching, errors)
    events = [e for e in errors if e.game.rules == "chess" and e.uci and puzzle_key(e.game.game_id, e.ply) in wanted]
    plies: dict[str, list[int]] = {}
    for e in events:
        plies.setdefault(e.game.game_id, []).append(e.ply)
    before: dict[str, dict[int, str]] = {}  # game id -> ply -> the position before your opponent's last move
    for e in events:
        key = puzzle_key(e.game.game_id, e.ply)
        result = lines.get((e.epd, e.uci))
        line = result.best_line if result is not None else None
        if line is not None and line.moves_uci and line.moves_uci[0] == e.uci:
            coaching.puzzle_themes[key] = []  # the deeper search prefers your move: settled, with no line
            continue
        if line is None or not line.moves_uci:
            line = by_ply.get((e.game.game_id, e.ply))
        if line is None or not line.moves_uci or line.moves_uci[0] == e.uci:
            continue
        line = rebase(line, e.fen, SOLUTION_PLIES)
        if not line.moves_uci:
            continue
        coaching.puzzle_lines[key] = line
        if e.game.game_id not in before:  # one replay per game
            before[e.game.game_id] = previous_fens(e.game, plies[e.game.game_id])
        previous = before[e.game.game_id].get(e.ply)
        themes = [m.theme for m in detect(line, "best", previous)
                  if m.side == "you" and m.theme in gated and m.ply < len(line.moves_uci)]
        coaching.puzzle_themes[key] = list(dict.fromkeys(themes))
