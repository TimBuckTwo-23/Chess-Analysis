"""Stockfish analysis: per-ply evaluations with Lichess-style win %, accuracy, judgements, phases and tags.

The formulas are ports of Lichess's open-source code (lila / scalachess), so the
numbers can be compared with what a player sees on lichess.org:

* win %            scalachess ``WinPercent.fromCentiPawns`` (cp clamped to +-1000; a mate counts as +-1000)
* move accuracy    lila ``AccuracyPercent.fromWinPercents``
* game accuracy    lila ``AccuracyPercent.gameAccuracy`` (volatility-weighted mean + harmonic mean)
* judgements       lila ``Advice`` (win-chance drops of 5 / 10 / 15 points, plus the mate rules);
                   like Lichess, a move equal to the engine's best move is never judged
* phases           scalachess ``Divider``

Every position of a game is evaluated exactly once (N + 1 searches for N plies), from
the last position backwards like Lichess's fishnet so later positions warm the hash.
The hash is cleared between games, so a game's result does not depend on which game
the engine analysed before it. Results are cached on disk per engine setting
(``<cache_dir>/<cfg.key()>/<game id>.json``) and uncached games are analysed by a pool
of worker processes, each owning one long-lived engine.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import hashlib
import json
import logging
import math
import multiprocessing
import multiprocessing.util
import os
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import chess
import chess.engine

from .models import Game, GameEval, PlyEval

log = logging.getLogger(__name__)

CP_LOSS_CAP = 1000  # largest centipawn loss one move can be charged
MATE_CP = 1000  # a forced mate (and any eval beyond it) counts as this many centipawns: Lichess's eval ceiling
WIN_MULTIPLIER = 0.00368208  # Lichess's cp -> win-chance slope (lila PR #11148)
DEFAULT_DEPTH = 12
MIN_PLIES = 10  # shorter games are not worth an engine run
CACHE_FORMAT = 1
PHASES = ("opening", "middlegame", "endgame")
ANALYSABLE_RULES = ("chess", "chess960")

# Lichess judgement thresholds: drop in the mover's win % (0..100 scale).
JUDGEMENT_DROPS: tuple[tuple[float, str], ...] = ((15.0, "blunder"), (10.0, "mistake"), (5.0, "inaccuracy"))
MISSED_TACTIC_DROP = 15.0  # win-% points lost by not playing a forcing best move
THROWN_WIN_FROM = 80.0  # "thrown_win": the mover was at least this likely to win before the move ...
THROWN_WIN_TO = 50.0  # ... and no more than this after it

_BINARY_NAMES = ("stockfish", "stockfish.exe")
_UNIX_PATHS = (
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "/snap/bin/stockfish",
)
# Stockfish for Windows ships as a zip holding e.g. stockfish\stockfish-windows-x86-64-avx2.exe.
_WINDOWS_ROOTS = ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
_WINDOWS_PATTERNS = (
    "Stockfish*/stockfish*.exe",
    "Stockfish*/*/stockfish*.exe",
    "Programs/Stockfish*/stockfish*.exe",
    "Programs/Stockfish*/*/stockfish*.exe",
)
_WINDOWS_USER_PATTERNS = ("Downloads/stockfish*/stockfish*.exe", "Downloads/stockfish*/*/stockfish*.exe")

_ENGINE_NAMES: dict[str, str] = {}


# --------------------------------------------------------------------------- configuration
@dataclass
class EngineConfig:
    """How hard Stockfish looks at each position, and how many engines run at once."""

    path: Optional[str] = None
    depth: Optional[int] = DEFAULT_DEPTH
    nodes: Optional[int] = None
    movetime: Optional[float] = None  # seconds per position
    threads: int = 1  # per engine; 1 keeps results reproducible
    hash_mb: int = 64
    workers: int = 0  # engine processes; 0 = auto: max(1, cpu_count - 1)

    def key(self) -> str:
        """Cache key for the search limit, e.g. "sf-d12" or "sf-n200000"."""
        parts = []
        if self.depth:
            parts.append(f"d{self.depth}")
        if self.nodes:
            parts.append(f"n{self.nodes}")
        if self.movetime:
            parts.append(f"t{self.movetime:g}")
        return "sf-" + "-".join(parts or [f"d{DEFAULT_DEPTH}"])

    def limit(self) -> chess.engine.Limit:
        if not (self.depth or self.nodes or self.movetime):
            return chess.engine.Limit(depth=DEFAULT_DEPTH)
        return chess.engine.Limit(depth=self.depth or None, nodes=self.nodes or None, time=self.movetime or None)

    def worker_count(self) -> int:
        return self.workers if self.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    @property
    def effective_depth(self) -> Optional[int]:
        """Depth recorded in GameEval: the configured depth, or the default when no limit is set."""
        if self.depth:
            return self.depth
        return None if (self.nodes or self.movetime) else DEFAULT_DEPTH


# --------------------------------------------------------------------------- finding and starting Stockfish
def _is_executable(path: Path) -> bool:
    return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))


def _resolve(candidate: str) -> Optional[str]:
    """A file path or a command name on PATH -> the executable's path, or None."""
    path = Path(candidate).expanduser()
    if _is_executable(path):
        return str(path)
    return shutil.which(candidate)


def windows_candidates(env: Optional[dict[str, str]] = None) -> list[str]:
    """Stockfish executables in the usual Windows install and download folders, in search order."""
    env = dict(os.environ) if env is None else env
    found: list[str] = []
    for var in _WINDOWS_ROOTS:
        root = env.get(var)
        if root:
            for pattern in _WINDOWS_PATTERNS:
                found.extend(sorted(glob.glob(os.path.join(root, pattern))))
    home = env.get("USERPROFILE")
    if home:
        for pattern in _WINDOWS_USER_PATTERNS:
            found.extend(sorted(glob.glob(os.path.join(home, pattern))))
    return [p for p in found if Path(p).is_file()]


def find_stockfish(explicit: Optional[str] = None) -> Optional[str]:
    """Path of a usable Stockfish binary, or None.

    An explicit path (``--stockfish``) is the only candidate when given. Otherwise:
    ``$CHESS_INSIGHTS_STOCKFISH``, ``stockfish`` on PATH, the usual Linux / macOS
    locations, then the usual Windows install and download folders.
    """
    if explicit:
        return _resolve(explicit)
    env = os.environ.get("CHESS_INSIGHTS_STOCKFISH")
    if env:
        found = _resolve(env)
        if found:
            return found
        log.warning("CHESS_INSIGHTS_STOCKFISH=%r is not an executable file; searching elsewhere", env)
    for name in _BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return found
    for cand in _UNIX_PATHS:
        if _is_executable(Path(cand)):
            return cand
    if os.name == "nt":
        cands = windows_candidates()
        if cands:
            return cands[0]
    return None


def open_engine(cfg: EngineConfig) -> chess.engine.SimpleEngine:
    """Start Stockfish with the configured threads and hash size."""
    if not cfg.path:
        raise FileNotFoundError("no Stockfish path configured")
    engine = chess.engine.SimpleEngine.popen_uci(cfg.path)
    try:
        wanted = {"Threads": cfg.threads, "Hash": cfg.hash_mb}
        options = {name: value for name, value in wanted.items() if name in engine.options}
        if options:
            engine.configure(options)
    except BaseException:
        _close_engine(engine)
        raise
    return engine


def _close_engine(engine: chess.engine.SimpleEngine) -> None:
    """Stop the engine process and its background thread (never raises)."""
    with contextlib.suppress(Exception):
        engine.close()


def engine_name(path: str) -> str:
    """"Stockfish 16" from the engine's UCI id (cached per path); the file name if it can't be started."""
    if path in _ENGINE_NAMES:
        return _ENGINE_NAMES[path]
    name: Optional[str] = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            name = engine.id.get("name")
        finally:
            _close_engine(engine)
    except Exception as exc:  # noqa: BLE001 — any start-up failure just means "use the file name"
        log.debug("could not read the engine name from %s: %s", path, exc)
    _ENGINE_NAMES[path] = name or Path(path).name
    return _ENGINE_NAMES[path]


# --------------------------------------------------------------------------- Lichess formulas
def win_percent(cp: Optional[int], mate: Optional[int] = None) -> float:
    """Winning chances 0..100 from the point of view of the side the score belongs to.

    ``50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)`` with cp clamped to +-1000.
    A mate counts as +-1000 cp by its sign, like Lichess's server-side analysis (97.5 / 2.5):
    the distance to mate is ignored. ``mate == 0`` means this side is checkmated. No score
    at all is an even position (50).
    """
    if mate is not None:
        cp = MATE_CP if mate > 0 else -MATE_CP
    if cp is None:
        return 50.0
    cp = max(-MATE_CP, min(MATE_CP, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-WIN_MULTIPLIER * cp)) - 1.0)


def move_accuracy(win_before: float, win_after: float) -> float:
    """Lichess per-move accuracy (0..100) from the mover's win % before and after the move."""
    if win_after >= win_before:
        return 100.0
    drop = win_before - win_after
    raw = 103.1668100711649 * math.exp(-0.04354415386753951 * drop) - 3.166924740191411
    return max(0.0, min(100.0, raw + 1.0))  # +1: Lichess's bonus for imperfect analysis


def _pstdev(xs: Sequence[float]) -> float:
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))


def game_accuracy(
    white_pov_win_percents: Sequence[float], color: str, white_moves_first: bool = True
) -> Optional[float]:
    """Lichess game accuracy (0..100) for ``color``; None if that side made no move.

    ``white_pov_win_percents`` holds White's win % for every position of the game: the
    start position, then the position after each ply (N + 1 values for N plies). Each
    move's accuracy is weighted by how volatile the game was around it (standard deviation
    of win % in a sliding window, clamped to 0.5..12); the result is the average of that
    weighted mean and the harmonic mean, so a few bad moves pull it down hard.
    """
    wp = [float(w) for w in white_pov_win_percents]
    if len(wp) < 2:
        return None
    size = max(2, min(8, (len(wp) - 1) // 10))
    # The first moves reuse the first window so that every move gets a weight (Scala's
    # List.fill(size - 2)(take(size)) ::: sliding(size)).
    windows = [wp[:size]] * max(0, min(size, len(wp)) - 2)
    windows += [wp[i : i + size] for i in range(len(wp) - size + 1)] if len(wp) >= size else [wp]
    want_white = color == "white"
    pairs: list[tuple[float, float]] = []
    for i, window in zip(range(len(wp) - 1), windows):
        white_moved = (i % 2 == 0) == white_moves_first
        if white_moved != want_white:
            continue
        acc = move_accuracy(wp[i], wp[i + 1]) if white_moved else move_accuracy(wp[i + 1], wp[i])
        pairs.append((acc, max(0.5, min(12.0, _pstdev(window)))))
    if not pairs:
        return None
    weighted = sum(a * w for a, w in pairs) / sum(w for _, w in pairs)
    harmonic = len(pairs) / sum(1.0 / max(1.0, a) for a, _ in pairs)
    return (weighted + harmonic) / 2.0


# Lichess's mate rules compare centipawns (-999 / -700 / 700 / 999); win % is strictly monotonic
# in cp, so the same cut-offs are applied on the win % scale.
_WIN_AT_MINUS_999 = win_percent(-999)
_WIN_AT_MINUS_700 = win_percent(-700)
_WIN_AT_700 = win_percent(700)
_WIN_AT_999 = win_percent(999)


def judge(
    win_before: float,
    win_after: float,
    *,
    mate_before: Optional[int] = None,
    mate_after: Optional[int] = None,
) -> Optional[str]:
    """"inaccuracy" | "mistake" | "blunder" | None, following Lichess's ``Advice``.

    Win % and the optional mate-in-N values are from the MOVER's point of view (a positive
    mate means the mover mates). Without mates: a drop of >= 5 / 10 / 15 win-% points.
    With a mate on either side of the move, only the mate rules apply:

    * a mate against the mover appears (it wasn't there before): blunder, or only a
      mistake / inaccuracy if the mover was already lost (below -700 / -999 cp);
    * the mover had a forced mate and lost it: blunder, or a mistake / inaccuracy if the
      position is still crushing (above +700 / +999 cp).
    """
    if mate_before is None and mate_after is None:
        drop = win_before - win_after
        return next((name for threshold, name in JUDGEMENT_DROPS if drop >= threshold), None)
    if mate_before is None and mate_after is not None and mate_after < 0:  # mate created against the mover
        if win_before < _WIN_AT_MINUS_999:
            return "inaccuracy"
        return "mistake" if win_before < _WIN_AT_MINUS_700 else "blunder"
    if mate_before is not None and mate_before > 0 and (mate_after is None or mate_after < 0):  # mate lost
        if mate_after is None and win_after > _WIN_AT_999:
            return "inaccuracy"
        return "mistake" if mate_after is None and win_after > _WIN_AT_700 else "blunder"
    return None


# --------------------------------------------------------------------------- phases (scalachess Divider)
_RANK_1 = chess.BB_RANK_1
_RANK_8 = chess.BB_RANK_8
_REGIONS = [(0x0303 << (x + 8 * y), y + 1) for y in range(7) for x in range(7)]  # every 2x2 square block


def _majors_and_minors(board: chess.Board) -> int:
    return chess.popcount(board.occupied & ~(board.kings | board.pawns))


def _backrank_sparse(board: chess.Board) -> bool:
    return (
        chess.popcount(_RANK_1 & board.occupied_co[chess.WHITE]) < 4
        or chess.popcount(_RANK_8 & board.occupied_co[chess.BLACK]) < 4
    )


def _region_score(y: int, white: int, black: int) -> int:
    """scalachess Divider ``score``: how interlocked the two armies are in one 2x2 block on rank ``y``."""
    if white == 0:
        if black == 1:
            return 1 + y
        if black == 2:
            return 2 + (6 - y) if y < 6 else 0
        if black in (3, 4):
            return 3 + (7 - y) if y < 7 else 0
        return 0
    if white == 1:
        return {0: 1 + (8 - y), 1: 5 + abs(4 - y), 2: 4 + (7 - y), 3: 5 + (7 - y)}.get(black, 0)
    if white == 2:
        if black == 0:
            return 2 + (y - 2) if y > 2 else 0
        return {1: 4 + (y - 1), 2: 7}.get(black, 0)
    if white == 3:
        if black == 0:
            return 3 + (y - 1) if y > 1 else 0
        return 5 + (y - 1) if black == 1 else 0
    if white == 4:
        return 3 + (y - 1) if black == 0 and y > 1 else 0
    return 0


def _mixedness(board: chess.Board) -> int:
    white, black = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
    return sum(_region_score(y, chess.popcount(white & r), chess.popcount(black & r)) for r, y in _REGIONS)


def divide(boards: Sequence[chess.Board]) -> tuple[Optional[int], Optional[int]]:
    """scalachess ``Divider``: (index of the first middlegame board, index of the first endgame board).

    Middlegame: at most 10 queens/rooks/bishops/knights left, a sparse back rank, or
    interlocked armies (mixedness > 150). Endgame: at most 6 such pieces. Like
    scalachess, the middlegame index is None when it coincides with the endgame.
    """
    mid = next(
        (i for i, b in enumerate(boards) if _majors_and_minors(b) <= 10 or _backrank_sparse(b) or _mixedness(b) > 150),
        None,
    )
    end = next((i for i, b in enumerate(boards) if _majors_and_minors(b) <= 6), None) if mid is not None else None
    middle = mid if mid is not None and (end is None or mid < end) else None
    return middle, end


def _phases_from_boards(boards_before: Sequence[chess.Board]) -> list[str]:
    opening, middlegame, endgame = PHASES
    middle, end = divide(boards_before)
    out = []
    for i in range(len(boards_before)):
        if end is not None and i >= end:
            out.append(endgame)  # also when the middlegame was skipped (Lichess would say "opening")
        elif middle is not None and i >= middle:
            out.append(middlegame)
        else:
            out.append(opening)
    return out


# --------------------------------------------------------------------------- replay helpers
def _start_board(game: Game) -> chess.Board:
    return chess.Board(game.initial_fen or chess.STARTING_FEN, chess960=game.rules == "chess960")


def _replay(game: Game) -> Optional[tuple[list[chess.Board], list[chess.Move]]]:
    """(positions before each ply plus the final one, moves), or None if a move is illegal."""
    try:
        board = _start_board(game)
    except ValueError:
        return None
    boards = [board.copy()]
    moves: list[chess.Move] = []
    for san in game.moves_san:
        try:
            move = board.parse_san(san)
        except ValueError:
            return None
        if not move:  # "--" null move: not a real chess move
            return None
        board.push(move)
        moves.append(move)
        boards.append(board.copy())
    return boards, moves


def game_phases(game: Game) -> list[str]:
    """Phase ("opening" | "middlegame" | "endgame") of every ply, judged on the position before it.

    Uses scalachess's Divider. Plies after an illegal move (a corrupt record) keep the phase
    of the last legal position; an unreadable start position counts as the opening.
    """
    try:
        board = _start_board(game)
    except ValueError:
        return ["opening"] * game.plies
    boards = []
    for san in game.moves_san:
        boards.append(board.copy(stack=False))
        try:
            move = board.parse_san(san)
        except ValueError:
            break
        if not move:
            break
        board.push(move)
    phases = _phases_from_boards(boards)
    fill = phases[-1] if phases else "opening"
    return phases + [fill] * (game.plies - len(phases))


def _time_spent(game: Game) -> list[Optional[float]]:
    """Seconds the mover spent on each ply: previous clock of the same mover (the starting clock for their
    first move) minus the clock after the move, plus the increment, floored at 0. None without clocks."""
    n = game.plies
    if game.time_class == "daily" or not game.base_seconds:
        return [None] * n
    clocks, base, inc = game.clocks, float(game.base_seconds), float(game.increment or 0)
    out: list[Optional[float]] = []
    for i in range(n):
        after = clocks[i] if i < len(clocks) else None
        before = base if i < 2 else (clocks[i - 2] if i - 2 < len(clocks) else None)
        out.append(None if after is None or before is None else max(0.0, before - after + inc))
    return out


# --------------------------------------------------------------------------- analysing one game
@dataclass
class _PositionEval:
    cp: int  # White's POV, clamped to +-MATE_CP
    mate: Optional[int]  # White's POV mate-in-N (positive: White mates), None if no forced mate
    best: Optional[chess.Move]  # engine's first PV move; None in terminal positions
    checkmate: bool = False


def _evaluate(
    engine: chess.engine.SimpleEngine, board: chess.Board, limit: chess.engine.Limit, session: object
) -> _PositionEval:
    if board.is_checkmate():
        return _PositionEval(-MATE_CP if board.turn == chess.WHITE else MATE_CP, None, None, checkmate=True)
    if board.is_stalemate() or board.is_insufficient_material():
        return _PositionEval(0, None, None)
    info = engine.analyse(board, limit, game=session, info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
    pov = info.get("score")
    if pov is None:
        raise chess.engine.EngineError(f"no score for {board.fen()}")
    score = pov.white()
    mate = score.mate()
    if mate == 0:  # "mated" reported for a live position: trust the side to move is lost
        mate, cp = None, (-MATE_CP if board.turn == chess.WHITE else MATE_CP)
    elif mate is not None:
        cp = MATE_CP if mate > 0 else -MATE_CP
    else:
        cp = max(-MATE_CP, min(MATE_CP, score.score() or 0))
    pv = info.get("pv") or []
    best = pv[0] if pv and board.is_legal(pv[0]) else None
    return _PositionEval(cp, mate, best)


def _tags(
    board: chess.Board,
    move: chess.Move,
    before: _PositionEval,
    after: _PositionEval,
    after_board: chess.Board,
    mover_mate_before: Optional[int],
    mover_mate_after: Optional[int],
    win_before: float,
    win_after: float,
    judgement: Optional[str],
) -> list[str]:
    tags = []
    if mover_mate_before is not None and mover_mate_before > 0 and not after.checkmate:
        if mover_mate_after is None or mover_mate_after < 0:
            tags.append("missed_mate")
    already_mated = mover_mate_before is not None and mover_mate_before < 0
    if mover_mate_after is not None and mover_mate_after < 0 and not already_mated:
        tags.append("allowed_mate")
    if judgement in ("mistake", "blunder") and after.best is not None and after_board.is_capture(after.best):
        tags.append("hung_material")
    best = before.best
    if (
        best is not None
        and best != move
        and (board.is_capture(best) or board.gives_check(best) or best.promotion is not None)
        and win_before - win_after >= MISSED_TACTIC_DROP
    ):
        tags.append("missed_tactic")
    if win_before >= THROWN_WIN_FROM and win_after <= THROWN_WIN_TO:
        tags.append("thrown_win")
    return tags


def analyze_game(game: Game, engine: chess.engine.SimpleEngine, cfg: EngineConfig) -> Optional[GameEval]:
    """Evaluate every position of ``game`` once and build its PlyEvals; None if the moves don't replay."""
    replay = _replay(game)
    if replay is None:
        return None
    boards, moves = replay
    limit = cfg.limit()
    session = object()  # a fresh "game" for python-chess: sends ucinewgame, so the hash starts empty
    # Backwards, like fishnet: the hash entries of later positions help the earlier searches.
    pos = [_evaluate(engine, b, limit, session) for b in reversed(boards)][::-1]
    white_wp = [win_percent(e.cp, e.mate) for e in pos]
    phases = _phases_from_boards(boards[:-1])
    spent = _time_spent(game)
    plies: list[PlyEval] = []
    for i, move in enumerate(moves):
        board, before, after = boards[i], pos[i], pos[i + 1]
        white_moved = board.turn == chess.WHITE
        sign = 1 if white_moved else -1
        win_before = white_wp[i] if white_moved else 100.0 - white_wp[i]
        win_after = white_wp[i + 1] if white_moved else 100.0 - white_wp[i + 1]
        mover_mate_before = None if before.mate is None else before.mate * sign
        mover_mate_after = None if after.mate is None else after.mate * sign
        played_best = before.best is not None and move == before.best
        judgement = (
            None
            if after.checkmate or played_best
            else judge(win_before, win_after, mate_before=mover_mate_before, mate_after=mover_mate_after)
        )
        mover = "white" if white_moved else "black"
        plies.append(
            PlyEval(
                ply=i,
                mover=mover,
                is_user=mover == game.color,
                san=game.moves_san[i],
                best_san=board.san(before.best) if before.best is not None else None,
                cp_before=before.cp,
                cp_after=after.cp,
                mate_before=before.mate,
                mate_after=after.mate,
                win_before=win_before,
                win_after=win_after,
                accuracy=move_accuracy(win_before, win_after),
                cp_loss=min(CP_LOSS_CAP, max(0, sign * (before.cp - after.cp))),
                judgement=judgement,
                phase=phases[i],
                clock_after=game.clocks[i] if i < len(game.clocks) else None,
                time_spent=spent[i],
                tags=_tags(
                    board,
                    move,
                    before,
                    after,
                    boards[i + 1],
                    mover_mate_before,
                    mover_mate_after,
                    win_before,
                    win_after,
                    judgement,
                ),
            )
        )
    white_first = boards[0].turn == chess.WHITE
    opp_color = "black" if game.color == "white" else "white"
    return GameEval(
        game_id=game.game_id,
        engine=engine.id.get("name") or "Stockfish",
        depth=cfg.effective_depth,
        plies=plies,
        my_accuracy=game_accuracy(white_wp, game.color, white_first),
        opp_accuracy=game_accuracy(white_wp, opp_color, white_first),
    )


# --------------------------------------------------------------------------- (de)serialisation and cache
def game_eval_to_dict(ev: GameEval) -> dict[str, Any]:
    return dataclasses.asdict(ev)


def game_eval_from_dict(d: dict[str, Any]) -> GameEval:
    fields = {f.name for f in dataclasses.fields(PlyEval)}
    plies = [PlyEval(**{k: v for k, v in p.items() if k in fields}) for p in d.get("plies", [])]
    for p in plies:
        p.tags = list(p.tags or [])
    return GameEval(
        game_id=d["game_id"],
        engine=d.get("engine") or "Stockfish",
        depth=d.get("depth"),
        plies=plies,
        my_accuracy=d.get("my_accuracy"),
        opp_accuracy=d.get("opp_accuracy"),
    )


def cache_file_name(game_id: str) -> str:
    """Filesystem-safe, collision-free file name for a game id (uuids are kept as they are)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", game_id).strip("._") or "game"
    if safe != game_id or len(safe) > 100:
        safe = f"{safe[:80]}-{hashlib.sha1(game_id.encode('utf-8')).hexdigest()[:12]}"
    return f"{safe}.json"


def _cache_path(cache_dir: Path, cfg: EngineConfig, game: Game) -> Path:
    return cache_dir / cfg.key() / cache_file_name(game.game_id)


def _load_cached(path: Path, game: Game) -> Optional[GameEval]:
    """The cached analysis of ``game``, or None if missing, unreadable or made for other moves."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("format") != CACHE_FORMAT
            or data.get("moves") != game.moves_san
            or data.get("initial_fen") != game.initial_fen
        ):
            return None
        ev = game_eval_from_dict(data["eval"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if data.get("color") != game.color:  # same game analysed for the other player: swap perspective
        ev.my_accuracy, ev.opp_accuracy = ev.opp_accuracy, ev.my_accuracy
        for p in ev.plies:
            p.is_user = p.mover == game.color
    return ev


def _write_cached(path: Path, game: Game, ev: GameEval) -> None:
    """Atomic write (temp file + rename): an interrupted run never leaves a half-written cache file."""
    payload = {
        "format": CACHE_FORMAT,
        "moves": game.moves_san,
        "initial_fen": game.initial_fen,
        "color": game.color,
        "eval": game_eval_to_dict(ev),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except OSError as exc:
        log.warning("could not cache the engine analysis of %s: %s", game.game_id, exc)


# --------------------------------------------------------------------------- running many games
class _Analyzer:
    """One long-lived engine; restarted (and the game retried once) if the engine process dies."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.engine: Optional[chess.engine.SimpleEngine] = None

    def analyze(self, game: Game) -> Optional[GameEval]:
        try:
            return analyze_game(game, self._engine(), self.cfg)
        except chess.engine.EngineTerminatedError:
            self.close()  # the engine process died: start a new one and give this game one more try
            return analyze_game(game, self._engine(), self.cfg)

    def _engine(self) -> chess.engine.SimpleEngine:
        if self.engine is None:
            self.engine = open_engine(self.cfg)
        return self.engine

    def close(self) -> None:
        if self.engine is not None:
            engine, self.engine = self.engine, None
            _close_engine(engine)


_WORKER: dict[str, _Analyzer] = {}


def _worker_init(cfg: EngineConfig) -> None:
    """Pool initializer: start this worker's engine and make sure it is closed when the worker exits.

    python-chess runs each engine on a non-daemon thread, so a worker whose engine is left
    open would hang on exit; the Finalize hook runs in every worker's normal shutdown path.
    """
    analyzer = _Analyzer(cfg)
    _WORKER["analyzer"] = analyzer
    multiprocessing.util.Finalize(None, analyzer.close, exitpriority=100)
    try:
        analyzer.engine = open_engine(cfg)
    except Exception as exc:  # noqa: BLE001 — retried per game, which then reports the error
        log.warning("engine failed to start in worker %d: %s", os.getpid(), exc)


def _worker_analyze(game: Game) -> Optional[GameEval]:
    return _WORKER["analyzer"].analyze(game)


OnResult = Callable[[Game, Optional[GameEval]], None]


def _analyze_in_process(games: list[Game], cfg: EngineConfig, on_result: OnResult) -> None:
    """One engine in this process; it is closed however the loop ends (error, KeyboardInterrupt)."""
    analyzer = _Analyzer(cfg)
    try:
        for game in games:
            try:
                ev = analyzer.analyze(game)
            except Exception as exc:  # noqa: BLE001 — one bad game must not stop the run
                log.warning("engine analysis of %s failed: %s", game.game_id, exc)
                ev = None
            on_result(game, ev)
    finally:
        analyzer.close()


def _analyze_uncached(games: list[Game], cfg: EngineConfig, on_result: OnResult) -> None:
    """Analyse ``games`` (one in-process engine, or a pool of worker processes) and report each result.

    If the worker processes can't start or die (e.g. a script without an ``if __name__ ==
    "__main__":`` guard, which "spawn" requires), the games they didn't finish are analysed
    in this process instead.
    """
    workers = min(cfg.worker_count(), len(games))
    if workers <= 1:
        _analyze_in_process(games, cfg, on_result)
        return

    # "spawn" (not fork): each worker starts clean, without copies of the parent's threads or engines,
    # and behaves the same on Linux, macOS and Windows.
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init, initargs=(cfg,)
    )
    unfinished: list[Game] = []
    finished = False
    try:
        futures = {pool.submit(_worker_analyze, g): g for g in games}
        for fut in as_completed(futures):
            game = futures[fut]
            try:
                ev = fut.result()
            except BrokenProcessPool:
                unfinished.append(game)
                continue
            except Exception as exc:  # noqa: BLE001 — one bad game must not stop the run
                log.warning("engine analysis of %s failed: %s", game.game_id, exc)
                ev = None
            on_result(game, ev)
        finished = True
    finally:
        # On an interrupt or error: drop queued games and don't wait; workers close their engines on exit.
        pool.shutdown(wait=finished, cancel_futures=True)
    if unfinished:
        log.warning(
            "engine worker processes failed; analysing the remaining %d games in this process", len(unfinished)
        )
        _analyze_in_process(unfinished, cfg, on_result)


def _is_candidate(game: Game) -> bool:
    return game.rules in ANALYSABLE_RULES and game.plies >= MIN_PLIES


def analyze_games(
    games: list[Game],
    cfg: EngineConfig,
    cache_dir: Path | str | None = None,
    max_games: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, GameEval]:
    """Engine analysis of the most recent ``max_games`` standard / Chess960 games with at least 10 plies.

    Cached analyses are loaded instead of recomputed; new ones are cached as they finish.
    ``progress(done, total)`` is called as games complete (cached games count as done
    up front). Games whose moves don't replay legally, or whose analysis fails, are left
    out of the result. Raises FileNotFoundError when games need analysing and no
    Stockfish binary can be found.
    """
    by_id = {g.game_id: g for g in games if _is_candidate(g)}  # a repeated id keeps its last copy
    candidates = sorted(by_id.values(), key=lambda g: g.end_time)
    if max_games is not None:
        candidates = candidates[-max_games:] if max_games > 0 else []
    cache = Path(cache_dir) if cache_dir is not None else None

    results: dict[str, GameEval] = {}
    todo: list[Game] = []
    skipped = 0
    for game in candidates:
        cached = _load_cached(_cache_path(cache, cfg, game), game) if cache is not None else None
        if cached is not None:
            results[game.game_id] = cached
        elif _replay(game) is None:  # checked here so a corrupt record never costs an engine start
            log.warning("skipping %s: its moves don't replay legally", game.game_id)
            skipped += 1
        else:
            todo.append(game)
    total, done = len(candidates), len(results) + skipped
    if progress and done:
        progress(done, total)

    if todo:
        path = cfg.path or find_stockfish()
        if not path:
            raise FileNotFoundError("Stockfish not found: install it or pass its path (--stockfish PATH)")
        run_cfg = dataclasses.replace(cfg, path=path)

        def on_result(game: Game, ev: Optional[GameEval]) -> None:
            nonlocal done
            done += 1
            if ev is not None:
                results[game.game_id] = ev
                _ENGINE_NAMES.setdefault(path, ev.engine)
                if cache is not None:
                    _write_cached(_cache_path(cache, run_cfg, game), game, ev)
            if progress:
                progress(done, total)

        _analyze_uncached(todo, run_cfg, on_result)

    return {g.game_id: results[g.game_id] for g in candidates if g.game_id in results}
