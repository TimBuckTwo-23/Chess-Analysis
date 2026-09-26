"""Tablebase-checked endgames (C3): wins you let slip and draws you lost, with 7 pieces or fewer on the board.

For each engine-analysed game that reaches 7 pieces or fewer (most recent games first), the Lichess tablebase is
asked about the first such position and about the position before each of your moves from there on. Its answer
lists every legal move with the result it leaves, so one request per move gives both the verdict before your move
and the verdict after it, plus the tablebase's own move. Recorded:

* a won position not won: a tablebase win for you, and your move left a draw or a loss;
* a drawn position lost: a tablebase draw, and your move left a loss.

The ending is labelled by its material at the first such position (pawn, rook, minor piece, queen or mixed).
The result is ``Coaching.endgames`` (a table per ending type, with the most recent slip in each), boards for the
slips, and ``Explanation.tablebase`` for explained positions with 7 pieces or fewer. Observations only: the
conversion claims stay with the engine review's own test. Owner: sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import chess

from ..models import Coaching, Diagram, Game, ModuleResult, Table
from ..visuals import format_text, ordered_formats, position_diagram
from .config import CoachConfig
from .sources import tablebase
from .sources.http import Fetcher

if TYPE_CHECKING:
    from ..context import AnalysisContext

MAX_REQUESTS = 150  # tablebase requests per report (cache hits are free), most recent games first
MAX_PER_GAME = 40  # your moves checked in one ending
MAX_DIAGRAMS = 6  # boards for the most recent slips
TYPES = ("Pawn", "Rook", "Minor piece", "Queen", "Mixed")
RESULT_WORDS = {"win": "a win", "draw": "a draw", "loss": "a loss"}


def endgame_type(board: chess.Board) -> str:
    """'Pawn', 'Rook', 'Minor piece', 'Queen' or 'Mixed' from the pieces besides kings and pawns."""
    kinds = {p.piece_type for p in board.piece_map().values() if p.piece_type not in (chess.KING, chess.PAWN)}
    if not kinds:
        return "Pawn"
    if kinds == {chess.ROOK}:
        return "Rook"
    if kinds <= {chess.KNIGHT, chess.BISHOP}:
        return "Minor piece"
    if kinds == {chess.QUEEN}:
        return "Queen"
    return "Mixed"


@dataclass
class Slip:
    """One of your moves that changed the tablebase result for the worse."""

    game: Game
    ply: int
    fen: str  # before your move
    move: str  # "41.Kd2" / "41...Kd7"
    played_uci: str
    before: str  # "win" | "draw" (for you)
    after: str  # "draw" | "loss" (for you, after your move)
    best: str  # the tablebase's move, numbered like ``move``
    best_uci: str
    ending: str
    dtm: Optional[int] = None


@dataclass
class Ending:
    """One game's ending with 7 pieces or fewer, as far as it was checked."""

    game: Game
    ending: str  # endgame type at the first such position
    entry: Optional[str] = None  # tablebase result for you at the first checked position
    had_win: bool = False  # a tablebase win for you at some point, with you to move
    had_draw: bool = False
    slips: list[Slip] = field(default_factory=list)
    complete: bool = True  # every one of your moves in the ending was checked
    answered: bool = False  # the tablebase answered at least once (False: unavailable, offline or request limit)


def _label(board: chess.Board, move: chess.Move) -> str:
    n = board.fullmove_number
    return f"{n}.{board.san(move)}" if board.turn == chess.WHITE else f"{n}...{board.san(move)}"


def _start_board(game: Game) -> Optional[chess.Board]:
    try:
        return chess.Board(game.initial_fen) if game.initial_fen else chess.Board()
    except ValueError:
        return None


def _check_move(fetcher: Fetcher, board: chess.Board, move: chess.Move, ply: int, ending: Ending) -> bool:
    """Ask the tablebase about the position before your ``move``; record a slip. False when it had no answer."""
    result = tablebase.probe(fetcher, board.fen())
    if result is None:
        return False
    ending.answered = True
    before = result.outcome
    if ending.entry is None:
        ending.entry = before
    mine = result.move(move.uci())
    after = mine.outcome_for_mover if mine else None
    ending.had_win |= before == "win"
    ending.had_draw |= before == "draw"
    worse = (before == "win" and after in ("draw", "loss")) or (before == "draw" and after == "loss")
    best = result.keeping()  # the tablebase's move that keeps the result (its first move, as it sorts them)
    if worse and best is not None:
        try:
            best_move: Optional[chess.Move] = chess.Move.from_uci(best.uci)
        except ValueError:
            best_move = None
        ending.slips.append(
            Slip(
                game=ending.game, ply=ply, fen=board.fen(), move=_label(board, move), played_uci=move.uci(),
                before=before or "", after=after or "",
                best=_label(board, best_move) if best_move in board.legal_moves else best.san,
                best_uci=best.uci, ending=ending.ending, dtm=result.dtm,
            )
        )
    return True


def reaches_tablebase(game: Game) -> bool:
    """Whether a move of ``game`` is played with 7 pieces or fewer on the board (no request)."""
    board = _start_board(game)
    if board is None or game.rules != "chess":
        return False
    for san in game.moves_san:
        if chess.popcount(board.occupied) <= tablebase.MAX_PIECES:
            return True
        try:
            board.push_san(san)
        except ValueError:
            return False
    return False


def check_game(fetcher: Fetcher, game: Game) -> Optional[Ending]:
    """The tablebase view of one game's ending; None when it never reaches 7 pieces. When the tablebase gave no
    answer (unavailable, offline without a cached answer, request limit) the Ending has ``answered`` False."""
    board = _start_board(game)
    if board is None or game.rules != "chess":
        return None
    me = chess.WHITE if game.color == "white" else chess.BLACK
    ending: Optional[Ending] = None
    checked = 0
    for ply, san in enumerate(game.moves_san):
        try:
            move = board.parse_san(san)
        except ValueError:
            break
        if chess.popcount(board.occupied) <= tablebase.MAX_PIECES:
            if ending is None:
                ending = Ending(game=game, ending=endgame_type(board))
                if board.turn != me:  # the first such position, with your opponent to move
                    first = tablebase.probe(fetcher, board.fen())
                    if first is None:
                        ending.complete = False
                        return ending
                    ending.answered = True
                    ending.entry = tablebase.FLIP[first.outcome] if first.outcome else None
            if board.turn == me:
                if checked >= MAX_PER_GAME or not _check_move(fetcher, board, move, ply, ending):
                    ending.complete = False
                    break
                checked += 1
        board.push(move)
    return ending


def _slip_text(s: Slip) -> str:
    before, after = RESULT_WORDS.get(s.before, s.before), RESULT_WORDS.get(s.after, s.after)
    return f"{s.move} ({before} to {after}; tablebase: {s.best})"


def endings_table(
    endings: Sequence[Ending], n_games: int, requests_capped: bool, reached: Optional[int] = None
) -> Optional[Table]:
    """One row per ending type: endings checked, tablebase wins and draws you had, slips, the latest example.

    ``endings`` are the checked ones; ``reached`` counts every analysed game that reached 7 pieces or fewer
    (checked or not), so the note never presents a partial run as the whole picture."""
    if not endings:
        return None
    reached = max(reached if reached is not None else len(endings), len(endings))
    rows = []
    for kind in TYPES:
        es = [e for e in endings if e.ending == kind]
        if not es:
            continue
        slips = sorted((s for e in es for s in e.slips), key=lambda s: s.game.end_time, reverse=True)
        won = [e for e in es if e.had_win]
        drawn = [e for e in es if e.had_draw]
        let_slip = [e for e in won if any(s.before == "win" for s in e.slips)]
        lost_draws = [e for e in es if any(s.before == "draw" for s in e.slips)]
        latest = slips[0] if slips else None
        mix: dict[str, int] = {}
        for e in es:
            mix[e.game.time_class] = mix.get(e.game.time_class, 0) + 1
        rows.append([
            f"{kind} endings", len(es), len(won), len(let_slip), len(drawn), len(lost_draws),
            _slip_text(latest) if latest else "", latest.game.url if latest else "",
            " · ".join(f"{n} {tc}" for tc, n in ordered_formats(mix).items()),
        ])
    formats_mix: dict[str, int] = {}
    for e in endings:
        formats_mix[e.game.time_class] = formats_mix.get(e.game.time_class, 0) + 1
    if reached == len(endings):
        note = (
            f"{len(endings)} of your {n_games} engine-analysed games reached 7 pieces or fewer "
            f"({format_text(formats_mix)}), checked with the Lichess tablebase, which knows the exact result of "
            "every such position."
        )
    else:
        note = (
            f"{reached} of your {n_games} engine-analysed games reached 7 pieces or fewer; {len(endings)} of them "
            f"({format_text(formats_mix)}) were checked with the Lichess tablebase, which knows the exact result "
            "of every such position."
        )
    note += (
        " Had a win / had a draw: games in which the tablebase gave you that result with you to move. Wins let "
        "slip: one of your moves turned a win into a draw or a loss; draws lost: a draw into a loss (50-move rule "
        "included). The example is the most recent slip, with the tablebase's move. Observations, not tested claims."
    )
    if requests_capped or reached > len(endings) or not all(e.complete for e in endings):
        note += (
            " Not every move of every ending could be checked in this run (request limit or the tablebase was "
            "unavailable); later runs continue from the cache."
        )
    return Table(
        title="Endings checked with the tablebase",
        columns=["Ending", "Games", "Had a win", "Wins let slip", "Had a draw", "Draws lost", "Latest slip", "Game",
                 "Formats"],
        rows=rows,
        formats=["text", "int", "int", "int", "int", "int", "text", "url", "text"],
        note=note,
        key_columns=[0, 2, 3, 5],
    )


def slip_diagram(s: Slip) -> Diagram:
    """The position before the slip: your move red, the tablebase's move green."""
    mate = tablebase.mate_in_moves(s.dtm)  # the DTM is in plies; the caption counts moves
    mate_text = f" (mate in {mate} moves with best play)" if mate and mate > 0 and s.before == "win" else ""
    return position_diagram(
        f"{s.ending} ending: {s.move} turned {RESULT_WORDS.get(s.before, s.before)} into "
        f"{RESULT_WORDS.get(s.after, s.after)}",
        s.fen,
        orientation=s.game.color,
        played=s.played_uci,
        best=s.best_uci,
        caption=f"Tablebase: {s.best} keeps {RESULT_WORDS.get(s.before, s.before)}{mate_text}. "
        f"You played {s.move}. {s.game.time_class.capitalize()} game.",
        link=s.game.url,
        time_class=s.game.time_class,
    )


def annotate(
    ctx: "AnalysisContext", coaching: Coaching, cfg: CoachConfig, modules: Optional[list[ModuleResult]] = None
) -> None:
    """Fill ``coaching.endgames`` and ``Explanation.tablebase`` for positions with 7 pieces or fewer.

    Boards for the most recent slips go to ``coaching.endgame_diagrams`` when the contract has that field, else
    to the Engine review section when ``modules`` is given.
    """
    fetcher = Fetcher.from_config(cfg, coaching.notes, max_requests={tablebase.SOURCE: MAX_REQUESTS})

    # explained positions first: they are few and shown in full
    for e in coaching.explanations:
        n = tablebase.pieces(e.fen)
        if n is None or n > tablebase.MAX_PIECES:
            continue
        result = tablebase.probe(fetcher, e.fen)
        if result is None:
            continue
        played = _played_uci(e.fen, e.played)
        e.tablebase = result.as_dict(played)
        src = result.source
        if src is not None and all((s.name, s.url) != (src.name, src.url) for s in e.sources):
            e.sources.append(src)

    by_id = {g.game_id: g for g in ctx.games}
    games = sorted((by_id[gid] for gid in ctx.evals if gid in by_id), key=lambda g: g.end_time, reverse=True)
    endings = []
    reached = 0
    for g in games:
        if tablebase.SOURCE in fetcher.disabled:  # counted, not asked about
            reached += reaches_tablebase(g)
            continue
        ending = check_game(fetcher, g)
        if ending is None:
            continue
        reached += 1
        if ending.answered:
            endings.append(ending)
    capped = fetcher.requests_made.get(tablebase.SOURCE, 0) >= MAX_REQUESTS
    coaching.endgames = endings_table(endings, len(games), capped, reached)
    slips = sorted((s for e in endings for s in e.slips), key=lambda s: s.game.end_time, reverse=True)
    diagrams = [slip_diagram(s) for s in slips[:MAX_DIAGRAMS]]
    if diagrams:
        if hasattr(coaching, "endgame_diagrams"):
            coaching.endgame_diagrams = diagrams  # type: ignore[attr-defined]
        elif modules is not None:
            engine = next((m for m in modules if m.key == "engine"), None)
            if engine is not None:
                engine.diagrams.extend(diagrams)
    coaching.settings["tablebase"] = {
        "reached": reached,
        "endings": len(endings),
        "slips": len(slips),
        "requests": fetcher.requests_made.get(tablebase.SOURCE, 0),
    }


def _played_uci(fen: str, played: str) -> Optional[str]:
    """Your move ('41.Kd2' / 'Kd2' / 'e1d2') as UCI in ``fen``."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    text = (played or "").split(".")[-1].strip()
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text).uci()
        except ValueError:
            continue
    return None


def summary(endings: Sequence[Ending]) -> dict[str, Any]:
    """Counts per ending type (for tests and the JSON)."""
    out: dict[str, Any] = {}
    for e in endings:
        row = out.setdefault(e.ending, {"games": 0, "had_win": 0, "wins_let_slip": 0, "had_draw": 0, "draws_lost": 0})
        row["games"] += 1
        row["had_win"] += e.had_win
        row["had_draw"] += e.had_draw
        row["wins_let_slip"] += any(s.before == "win" for s in e.slips)
        row["draws_lost"] += any(s.before == "draw" for s in e.slips)
    return out
