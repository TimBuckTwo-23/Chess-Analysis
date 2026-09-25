"""Stockfish analysis: per-ply evaluations with Lichess-style win %, accuracy, judgements, phases and tags.

The formulas are ports of Lichess's open-source code (lila / scalachess), so the
numbers can be compared with what a player sees on lichess.org. tests/test_engine_parity.py
checks them vector by vector against lila's and scalachess's own test cases:

* win %            scalachess ``WinPercent.fromCentiPawns`` (cp clamped to +-1000; a mate counts as +-1000)
* move accuracy    lila ``AccuracyPercent.fromWinPercents``
* game accuracy    lila ``AccuracyPercent.gameAccuracy`` (volatility-weighted mean + harmonic mean); like
                   Lichess, the checkmate position is left out, so a mating move doesn't count
* judgements       lila ``Advice`` (win-chance drops of 5 / 10 / 15 points on the UNclamped centipawns,
                   plus the mate rules); a move equal to the engine's best move is never judged or tagged
* phases           scalachess ``Divider``; a move's phase is that of the position after it (lila insight)

Deliberate differences from Lichess, each pinned down by a ``test_deviation_*`` test:

* the start position is evaluated by Stockfish (Lichess assumes +15 cp);
* one move's centipawn loss is capped at ``CP_LOSS_CAP`` (Lichess: up to 2000);
* a game that goes straight from the opening into an endgame has "endgame" moves (Lichess: "opening");
* a side's accuracy is reported even when the other side never moved (Lichess: neither).

Every position of a game is evaluated exactly once (N + 1 searches for N plies), from
the last position backwards like Lichess's fishnet so later positions warm the hash.
The hash is cleared between games, so a game's result does not depend on which game
the engine analysed before it.

**Cache.** What Stockfish said about each position (White's score, mate distance, best
move) is cached per engine version and search setting, in
``<cache_dir>/<engine>/<limit>/<game id>.json`` (e.g. ``stockfish-16/sf-d12/``).
Everything derived from it (win %, judgements, tags, phases, clocks, whose side you were
on) is rebuilt from the game when the cache is read, so a fixed formula or a re-parsed
game never serves stale numbers, and a new Stockfish version starts a fresh cache.

**Parallelism.** Several engines run at once, each driven by a worker thread of this
process. The searches run in the engine processes, so threads scale like a process pool,
but they need no ``if __name__ == "__main__":`` guard on Windows and macOS, and every
engine process is gone when ``analyze_games`` returns or raises, Ctrl+C included.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import glob
import hashlib
import json
import logging
import math
import os
import queue
import re
import shutil
import tempfile
import threading
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
DEFAULT_HASH_MB = 64
MIN_PLIES = 10  # shorter games are not worth an engine run
CACHE_FORMAT = 2  # 2: raw engine output per position (1: derived PlyEvals, no engine version in the path)
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
# File names Windows reserves for devices, with or without an extension.
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(10)}
)
# Start positions Stockfish can't search (it may crash on them).
_UNPLAYABLE = (
    chess.STATUS_EMPTY
    | chess.STATUS_NO_WHITE_KING
    | chess.STATUS_NO_BLACK_KING
    | chess.STATUS_TOO_MANY_KINGS
    | chess.STATUS_PAWNS_ON_BACKRANK
    | chess.STATUS_OPPOSITE_CHECK
)

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
    hash_mb: int = DEFAULT_HASH_MB
    workers: int = 0  # engines running at once; 0 = auto: max(1, cpu_count - 1)

    def key(self) -> str:
        """Cache key for the settings that change Stockfish's numbers, e.g. "sf-d12" or "sf-n200000-th2"."""
        parts = []
        if self.depth:
            parts.append(f"d{self.depth}")
        if self.nodes:
            parts.append(f"n{self.nodes}")
        if self.movetime:
            parts.append(f"t{self.movetime:g}")
        key = "sf-" + "-".join(parts or [f"d{DEFAULT_DEPTH}"])
        if self.threads != 1:
            key += f"-th{self.threads}"
        if self.hash_mb != DEFAULT_HASH_MB:
            key += f"-h{self.hash_mb}"
        return key

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


def _popen(path: str) -> chess.engine.SimpleEngine:
    # setpgrp: the engine gets its own process group, so Ctrl+C reaches only Python, which then closes the
    # engines itself (instead of every engine dying mid-search at the same moment).
    return chess.engine.SimpleEngine.popen_uci(path, setpgrp=True)


def open_engine(cfg: EngineConfig) -> chess.engine.SimpleEngine:
    """Start Stockfish with the configured threads and hash size."""
    if not cfg.path:
        raise FileNotFoundError("no Stockfish path configured")
    engine = _popen(cfg.path)
    try:
        wanted = {"Threads": cfg.threads, "Hash": cfg.hash_mb}
        options = {name: value for name, value in wanted.items() if name in engine.options}
        if options:
            engine.configure(options)
    except BaseException:
        _close_engine(engine)
        raise
    return engine


def _ignore_engine_terminated(loop: Any, context: dict[str, Any]) -> None:
    """Event-loop error handler for an engine we are killing on purpose: a search it was running fails with
    EngineTerminatedError, which asyncio would otherwise print as "Future exception was never retrieved"."""
    if not isinstance(context.get("exception"), chess.engine.EngineTerminatedError):
        loop.default_exception_handler(context)


def _close_engine(engine: Any, wait: bool = True) -> None:
    """Kill the engine process (never raises); with ``wait``, until it has exited (at most a few seconds)."""
    loop = getattr(getattr(engine, "protocol", None), "loop", None)
    if loop is not None:
        with contextlib.suppress(Exception):
            loop.set_exception_handler(_ignore_engine_terminated)
    with contextlib.suppress(Exception):
        engine.close()
    if wait:
        _wait_exited(engine)


def _wait_exited(engine: Any) -> None:
    returncode = getattr(engine, "returncode", None)  # SimpleEngine: set once the process has exited
    if isinstance(returncode, concurrent.futures.Future):
        with contextlib.suppress(Exception):
            returncode.result(timeout=5)


def engine_name(path: str) -> str:
    """"Stockfish 16" from the engine's UCI id (cached per path); the file name if it can't be started."""
    if path in _ENGINE_NAMES:
        return _ENGINE_NAMES[path]
    name: Optional[str] = None
    try:
        engine = _popen(path)
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


def _unclamped_win_percent(cp: int) -> float:
    """Win % without the +-1000 clamp: Lichess's Advice (judgements) compares these."""
    cp = max(-20_000, min(20_000, cp))  # beyond that the chances are 0 / 100 to double precision; avoids overflow
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
    mate means the mover mates). Without mates: a drop of >= 5 / 10 / 15 win-% points
    (Lichess computes these from the unclamped centipawns: see :func:`judge_positions`).
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


# --------------------------------------------------------------------------- one position
@dataclass(frozen=True)
class PositionEval:
    """What Stockfish said about one position, from WHITE's point of view (this is what gets cached).

    ``cp`` is the raw centipawn score (not clamped), or +-MATE_CP for a position where the side to
    move is checkmated; ``mate`` is mate-in-N (positive: White mates) and replaces ``cp``; ``best``
    is the engine's best move in UCI notation.
    """

    cp: Optional[int] = None
    mate: Optional[int] = None
    best: Optional[str] = None
    checkmate: bool = False  # the side to move is checkmated (no search needed)

    @property
    def white_cp(self) -> int:
        """White's score in centipawns, clamped to +-MATE_CP (a mate counts as the clamp)."""
        if self.mate is not None:
            return MATE_CP if self.mate > 0 else -MATE_CP
        return 0 if self.cp is None else max(-MATE_CP, min(MATE_CP, self.cp))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.cp is not None:
            out["cp"] = self.cp
        if self.mate is not None:
            out["mate"] = self.mate
        if self.best:
            out["best"] = self.best
        if self.checkmate:
            out["checkmate"] = True
        return out

    @classmethod
    def from_json(cls, data: Any) -> "PositionEval":
        if not isinstance(data, dict):
            raise TypeError(f"not a position evaluation: {data!r}")
        cp, mate, best = data.get("cp"), data.get("mate"), data.get("best")
        for value in (cp, mate):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"not a score: {value!r}")
        if best is not None and not isinstance(best, str):
            raise TypeError(f"not a move: {best!r}")
        return cls(cp=cp, mate=mate, best=best, checkmate=bool(data.get("checkmate", False)))


def evaluate_position(
    engine: chess.engine.SimpleEngine, board: chess.Board, limit: chess.engine.Limit, session: object
) -> PositionEval:
    """One search (none for a finished position). ``session`` identifies the game, so the hash is cleared
    (``ucinewgame``) whenever a new game starts."""
    if board.is_checkmate():
        return PositionEval(cp=-MATE_CP if board.turn == chess.WHITE else MATE_CP, checkmate=True)
    if board.is_stalemate() or board.is_insufficient_material():
        return PositionEval(cp=0)
    info = engine.analyse(board, limit, game=session, info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
    pov = info.get("score")
    if pov is None:
        raise chess.engine.EngineError(f"no score for {board.fen()}")
    score = pov.white()
    pv = info.get("pv") or []
    best = board.uci(pv[0]) if pv and board.is_legal(pv[0]) else None
    mate = score.mate()
    if mate == 0:
        # python-chess reports a finished game as Mate(-0) for the side to move, and PovScore.white() of it can be
        # MateGiven: mate() is 0 either way and carries no sign. The side to move is the one that is mated.
        return PositionEval(cp=-MATE_CP if board.turn == chess.WHITE else MATE_CP, best=best)
    if mate is not None:
        return PositionEval(mate=mate, best=best)
    return PositionEval(cp=score.score(), best=best)


def judge_positions(before: PositionEval, after: PositionEval, white_moved: bool) -> Optional[str]:
    """Lichess's judgement of a move from the White-POV evaluations of the positions before and after it.

    Like lila's ``Advice``: winning chances of the UNclamped centipawns (+20 -> +4.5 pawns is a blunder,
    although both clamp to about +10), the mate rules when either side is a forced mate, and nothing for
    the mating move itself.
    """
    if after.checkmate:
        return None
    sign = 1 if white_moved else -1

    def mover_win(pos: PositionEval) -> float:
        if pos.mate is not None:
            return win_percent(None, mate=sign * pos.mate)
        return _unclamped_win_percent(sign * (pos.cp or 0))

    return judge(
        mover_win(before),
        mover_win(after),
        mate_before=None if before.mate is None else sign * before.mate,
        mate_after=None if after.mate is None else sign * after.mate,
    )


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


def _move_phases(boards: Sequence[chess.Board]) -> list[str]:
    """Phase of each move from ``boards`` = the start position and the position after every move.

    Like lila's insight ``Phase.of``, a move belongs to the phase of the position it creates (the move
    that trades into an endgame is an endgame move). When the middlegame is skipped (it would start on
    the same position as the endgame), moves from that position on are "endgame" (lila: "opening").
    """
    opening, middlegame, endgame = PHASES
    middle, end = divide(boards)
    out = []
    for after in range(1, len(boards)):
        if end is not None and after >= end:
            out.append(endgame)
        elif middle is not None and after >= middle:
            out.append(middlegame)
        else:
            out.append(opening)
    return out


# --------------------------------------------------------------------------- replay helpers
def _start_board(game: Game) -> chess.Board:
    board = chess.Board(game.initial_fen or chess.STARTING_FEN, chess960=game.rules == "chess960")
    if board.status() & _UNPLAYABLE:
        raise ValueError(f"not a playable start position: {game.initial_fen}")
    return board


def _replay(game: Game, stack: bool = True) -> Optional[tuple[list[chess.Board], list[chess.Move]]]:
    """(the start position and the position after every ply, moves), or None if a move is illegal.

    With ``stack`` each board carries the moves that led to it (so the engine sees repetitions).
    """
    try:
        board = _start_board(game)
    except ValueError:
        return None
    boards = [board.copy(stack=stack)]
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
        boards.append(board.copy(stack=stack))
    return boards, moves


def game_phases(game: Game) -> list[str]:
    """Phase ("opening" | "middlegame" | "endgame") of every ply: that of the position after it (Lichess).

    Uses scalachess's Divider. Plies after an illegal move (a corrupt record) keep the phase
    of the last legal move; an unreadable start position counts as the opening.
    """
    try:
        board = _start_board(game)
    except ValueError:
        return ["opening"] * game.plies
    boards = [board.copy(stack=False)]
    for san in game.moves_san:
        try:
            move = board.parse_san(san)
        except ValueError:
            break
        if not move:
            break
        board.push(move)
        boards.append(board.copy(stack=False))
    phases = _move_phases(boards)
    fill = phases[-1] if phases else "opening"
    return phases + [fill] * (game.plies - len(phases))


def _time_spent(game: Game) -> list[Optional[float]]:
    """Seconds the mover spent on each ply: previous clock of the same mover (the starting clock for their
    first move) minus the clock after the move, plus the increment, floored at 0. None without clocks.

    chess.com's clock after a move already includes the increment (``180+2``: ``1. e4 {[%clk 0:03:02]}``).
    """
    n = game.plies
    if game.time_class == "daily" or not game.base_seconds:
        return [None] * n
    clocks, base, inc = list(game.clocks or []), float(game.base_seconds), float(game.increment or 0)
    out: list[Optional[float]] = []
    for i in range(n):
        after = clocks[i] if i < len(clocks) else None
        before = base if i < 2 else (clocks[i - 2] if i - 2 < len(clocks) else None)
        out.append(None if after is None or before is None else max(0.0, before - after + inc))
    return out


# --------------------------------------------------------------------------- one game
def evaluate_game(
    game: Game, engine: chess.engine.SimpleEngine, cfg: EngineConfig
) -> Optional[list[PositionEval]]:
    """Stockfish's verdict on each of the N + 1 positions of ``game``; None if its moves don't replay."""
    replay = _replay(game, stack=True)
    if replay is None:
        return None
    boards, _ = replay
    limit = cfg.limit()
    session = object()  # a fresh "game" for python-chess: sends ucinewgame, so the hash starts empty
    # Backwards, like fishnet: the hash entries of later positions help the earlier searches.
    return [evaluate_position(engine, b, limit, session) for b in reversed(boards)][::-1]


def _legal_uci(board: chess.Board, uci: Optional[str]) -> Optional[chess.Move]:
    if not uci:
        return None
    try:
        return board.parse_uci(uci)
    except ValueError:
        return None


def _tags(
    board: chess.Board,
    move: chess.Move,
    best: Optional[chess.Move],
    reply: Optional[chess.Move],
    after_board: chess.Board,
    mover_mate_before: Optional[int],
    mover_mate_after: Optional[int],
    win_before: float,
    win_after: float,
    judgement: Optional[str],
) -> list[str]:
    """Error tags of a move that is not the engine's best (``reply``: the engine's best answer to it)."""
    tags = []
    if mover_mate_before is not None and mover_mate_before > 0:
        if mover_mate_after is None or mover_mate_after < 0:
            tags.append("missed_mate")
    already_mated = mover_mate_before is not None and mover_mate_before < 0
    if mover_mate_after is not None and mover_mate_after < 0 and not already_mated:
        tags.append("allowed_mate")
    if judgement in ("mistake", "blunder") and reply is not None and after_board.is_capture(reply):
        tags.append("hung_material")
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


def build_game_eval(
    game: Game, positions: Sequence[PositionEval], engine: str = "Stockfish", depth: Optional[int] = None
) -> Optional[GameEval]:
    """The game's PlyEvals from the engine's verdicts on its N + 1 positions (no engine needed).

    Returns None if the moves don't replay or ``positions`` doesn't have one entry per position.
    Engine best moves that aren't legal in their position are ignored.
    """
    replay = _replay(game, stack=False)
    if replay is None or len(positions) != len(replay[0]):
        return None
    boards, moves = replay
    white_wp = [win_percent(p.white_cp) for p in positions]
    phases = _move_phases(boards)
    spent = _time_spent(game)
    clocks = list(game.clocks or [])
    plies: list[PlyEval] = []
    for i, move in enumerate(moves):
        board, before, after = boards[i], positions[i], positions[i + 1]
        white_moved = board.turn == chess.WHITE
        sign = 1 if white_moved else -1
        win_before = white_wp[i] if white_moved else 100.0 - white_wp[i]
        win_after = white_wp[i + 1] if white_moved else 100.0 - white_wp[i + 1]
        best = _legal_uci(board, before.best)
        if after.checkmate or best == move:  # the mating move, or the engine's own choice: never an error
            judgement, tags = None, []
        else:
            judgement = judge_positions(before, after, white_moved)
            tags = _tags(
                board,
                move,
                best,
                _legal_uci(boards[i + 1], after.best),
                boards[i + 1],
                None if before.mate is None else sign * before.mate,
                None if after.mate is None else sign * after.mate,
                win_before,
                win_after,
                judgement,
            )
        mover = "white" if white_moved else "black"
        plies.append(
            PlyEval(
                ply=i,
                mover=mover,
                is_user=mover == game.color,
                san=game.moves_san[i],
                best_san=board.san(best) if best is not None else None,
                cp_before=before.white_cp,
                cp_after=after.white_cp,
                mate_before=before.mate,
                mate_after=after.mate,
                win_before=win_before,
                win_after=win_after,
                accuracy=move_accuracy(win_before, win_after),
                cp_loss=min(CP_LOSS_CAP, max(0, sign * (before.white_cp - after.white_cp))),
                judgement=judgement,
                phase=phases[i],
                clock_after=clocks[i] if i < len(clocks) else None,
                time_spent=spent[i],
                tags=tags,
            )
        )
    # Like Lichess (fishnet drops the checkmate position), the mating move takes no part in the accuracy.
    accuracy_wp = white_wp[:-1] if positions[-1].checkmate else white_wp
    white_first = boards[0].turn == chess.WHITE
    opp_color = "black" if game.color == "white" else "white"
    return GameEval(
        game_id=game.game_id,
        engine=engine,
        depth=depth,
        plies=plies,
        my_accuracy=game_accuracy(accuracy_wp, game.color, white_first),
        opp_accuracy=game_accuracy(accuracy_wp, opp_color, white_first),
    )


def analyze_game(game: Game, engine: chess.engine.SimpleEngine, cfg: EngineConfig) -> Optional[GameEval]:
    """Evaluate every position of ``game`` once and build its PlyEvals; None if the moves don't replay."""
    positions = evaluate_game(game, engine, cfg)
    if positions is None:
        return None
    return build_game_eval(game, positions, engine.id.get("name") or "Stockfish", cfg.effective_depth)


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


def _is_reserved_on_windows(name: str) -> bool:
    return name.split(".")[0].rstrip(" ").upper() in _WINDOWS_RESERVED


def cache_file_name(game_id: str) -> str:
    """File name for a game id that is safe and distinct on Windows, macOS and Linux (uuids are kept as they are).

    Anything other than lowercase ids of letters, digits, ``.``, ``_`` and ``-`` gets a hash of the exact id
    appended, so ids that differ only in case or in replaced characters never share a file on a
    case-insensitive file system; names Windows reserves (``CON``, ``COM1`` ...) are prefixed.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", game_id).strip("._") or "game"
    if _is_reserved_on_windows(safe):
        safe = "_" + safe
    if safe != game_id or len(safe) > 100 or safe != safe.lower():
        safe = f"{safe[:80]}-{hashlib.sha1(game_id.encode('utf-8')).hexdigest()[:12]}"
    return f"{safe}.json"


def _dir_name(label: str) -> str:
    """Directory name for an engine version: "Stockfish 16" -> "stockfish-16"."""
    name = re.sub(r"[^a-z0-9._-]+", "-", label.lower()).strip("-._")[:60] or "engine"
    return "_" + name if _is_reserved_on_windows(name) else name


def cache_root(cache_dir: Path | str, cfg: EngineConfig, engine_label: str) -> Path:
    """Where analyses by this engine version with these settings are cached."""
    return Path(cache_dir) / _dir_name(engine_label) / cfg.key()


def _cache_dirs(cache: Path, cfg: EngineConfig, engine_label: Optional[str]) -> list[Path]:
    if engine_label is not None:
        return [cache_root(cache, cfg, engine_label)]
    # Stockfish isn't installed here: any engine's analyses with these settings, most recently written first.
    found = []
    for d in glob.glob(os.path.join(glob.escape(str(cache)), "*", glob.escape(cfg.key()))):
        with contextlib.suppress(OSError):
            found.append((os.path.getmtime(d), Path(d)))
    return [d for _, d in sorted(found, reverse=True)]


def _load_cached(path: Path, game: Game) -> Optional[GameEval]:
    """The cached analysis of ``game``, or None if missing, unreadable, or made for other moves / positions."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("format") != CACHE_FORMAT
            or data.get("moves") != game.moves_san
            or data.get("initial_fen") != game.initial_fen
            or bool(data.get("chess960")) != (game.rules == "chess960")
        ):
            return None
        positions = [PositionEval.from_json(p) for p in data["positions"]]
        depth = data.get("depth")
        return build_game_eval(
            game, positions, str(data.get("engine") or "Stockfish"), depth if isinstance(depth, int) else None
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _write_cached(path: Path, game: Game, positions: Sequence[PositionEval], engine: str, depth: Optional[int]) -> None:
    """Atomic write (temp file + rename): an interrupted run never leaves a half-written cache file."""
    payload = {
        "format": CACHE_FORMAT,
        "engine": engine,
        "depth": depth,
        "moves": game.moves_san,
        "initial_fen": game.initial_fen,
        "chess960": game.rules == "chess960",
        "positions": [p.to_json() for p in positions],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
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
class _Cancelled(Exception):
    """The run is being shut down (error or Ctrl+C in the main thread)."""


class _EngineUnavailable(RuntimeError):
    """Stockfish could not be started; later games fail with this at once instead of retrying."""


class _Analyzer:
    """One long-lived engine, restarted (and the game retried once) if its process dies.

    ``close`` may be called from another thread: it kills the engine (an ongoing search then fails
    at once) and no new engine is started afterwards.
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._closed = False
        self._closing: list[chess.engine.SimpleEngine] = []
        self._start_error: Optional[_EngineUnavailable] = None

    def _get(self) -> chess.engine.SimpleEngine:
        with self._lock:
            if self._closed:
                raise _Cancelled()
            if self._start_error is not None:
                raise self._start_error
            if self._engine is None:
                try:
                    self._engine = open_engine(self.cfg)
                except Exception as exc:  # noqa: BLE001 — reported per game by the caller
                    self._start_error = _EngineUnavailable(f"Stockfish could not be started: {exc}")
                    raise self._start_error from exc
            return self._engine

    def _discard(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
            closed = self._closed
        if engine is not None:
            _close_engine(engine)
        if closed:
            raise _Cancelled()

    def evaluate(self, game: Game) -> Optional[list[PositionEval]]:
        try:
            return evaluate_game(game, self._get(), self.cfg)
        except chess.engine.EngineTerminatedError:
            self._discard()  # the engine process died: start a new one and give this game one more try
            return evaluate_game(game, self._get(), self.cfg)

    def close(self) -> None:
        """Kill the engine (without waiting) and refuse to start another one."""
        with self._lock:
            self._closed = True
            engine, self._engine = self._engine, None
        if engine is not None:
            _close_engine(engine, wait=False)
            self._closing.append(engine)

    def wait(self) -> None:
        """Wait until the engine closed by ``close`` has exited."""
        for engine in self._closing:
            _wait_exited(engine)
        self._closing.clear()


class _Failures:
    """Logs failed games: each distinct error once as a warning, repeats at debug level."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def report(self, game: Game, exc: BaseException) -> None:
        text = str(exc) or type(exc).__name__
        level = logging.DEBUG if text in self.seen else logging.WARNING
        self.seen.add(text)
        log.log(level, "engine analysis of %s failed: %s", game.game_id, text)


OnResult = Callable[[Game, Optional[list[PositionEval]]], None]


def _work(analyzer: _Analyzer, todo: "queue.SimpleQueue[Game]", done: "queue.SimpleQueue[tuple]") -> None:
    """Worker thread: analyse games from ``todo`` until it is empty or the run is cancelled."""
    while True:
        try:
            game = todo.get_nowait()
        except queue.Empty:
            return
        try:
            done.put((game, analyzer.evaluate(game), None))
        except _Cancelled:
            return
        except Exception as exc:  # noqa: BLE001 — one bad game must not stop the run
            done.put((game, None, exc))


def _analyze_uncached(games: list[Game], cfg: EngineConfig, on_result: OnResult) -> None:
    """Analyse ``games`` with up to ``cfg.worker_count()`` engines and report each result (in this thread).

    Every engine is killed before this returns or raises (an error, or Ctrl+C in this thread).
    """
    games = sorted(games, key=lambda g: -g.plies)  # longest first, so no long game is left running alone at the end
    workers = max(1, min(cfg.worker_count(), len(games)))
    analyzers = [_Analyzer(cfg) for _ in range(workers)]
    failures = _Failures()
    threads: list[threading.Thread] = []
    try:
        if workers == 1:
            for game in games:
                try:
                    positions = analyzers[0].evaluate(game)
                except Exception as exc:  # noqa: BLE001 — one bad game must not stop the run
                    failures.report(game, exc)
                    positions = None
                on_result(game, positions)
            return
        todo: queue.SimpleQueue[Game] = queue.SimpleQueue()
        done: queue.SimpleQueue[tuple] = queue.SimpleQueue()
        for game in games:
            todo.put(game)
        threads = [
            threading.Thread(target=_work, args=(a, todo, done), name=f"stockfish-worker-{i + 1}", daemon=True)
            for i, a in enumerate(analyzers)
        ]
        for t in threads:
            t.start()
        for _ in games:
            while True:
                try:  # a timed wait, so Ctrl+C is noticed promptly on Windows too
                    game, positions, exc = done.get(timeout=0.2)
                    break
                except queue.Empty:
                    if not any(t.is_alive() for t in threads) and done.empty():
                        log.warning("engine workers stopped early; some games were not analysed")
                        return
            if exc is not None:
                failures.report(game, exc)
            on_result(game, positions)
    finally:
        for a in analyzers:
            a.close()
        for a in analyzers:
            a.wait()
        for t in threads:
            t.join(timeout=5)


def _is_candidate(game: Game) -> bool:
    return game.rules in ANALYSABLE_RULES and game.plies >= MIN_PLIES


def analyze_games(
    games: list[Game],
    cfg: EngineConfig,
    cache_dir: Path | str | None = None,
    max_games: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, GameEval]:
    """Engine analysis of the ``max_games`` most recent standard / Chess960 games with at least 10 plies.

    Games whose moves don't replay legally are skipped (and don't count towards ``max_games``).
    Cached analyses by the same engine version and settings are loaded instead of recomputed; new
    ones are cached as they finish. ``progress(done, total)`` is called as games complete (cached
    games count as done up front). Games whose analysis fails are left out of the result, which
    is ordered oldest first. Raises FileNotFoundError when games need analysing and no Stockfish
    binary can be found.
    """
    if max_games is not None and max_games <= 0:
        return {}
    by_id = {g.game_id: g for g in games if _is_candidate(g)}  # a repeated id keeps its last copy
    newest_first = sorted(by_id.values(), key=lambda g: g.end_time, reverse=True)
    cache = Path(cache_dir) if cache_dir is not None else None
    path = cfg.path or find_stockfish()
    label = engine_name(path) if path else None
    dirs = _cache_dirs(cache, cfg, label) if cache is not None else []

    selected: list[Game] = []
    results: dict[str, GameEval] = {}
    todo: list[Game] = []
    for game in newest_first:
        if max_games is not None and len(selected) >= max_games:
            break
        name = cache_file_name(game.game_id)
        cached = next((ev for d in dirs if (ev := _load_cached(d / name, game)) is not None), None)
        if cached is not None:
            results[game.game_id] = cached
        elif _replay(game, stack=False) is None:  # checked here so a corrupt record never costs an engine start
            log.warning("skipping %s: its moves don't replay legally", game.game_id)
            continue
        else:
            todo.append(game)
        selected.append(game)
    total, done = len(selected), len(results)
    if progress and done:
        progress(done, total)

    if todo:
        if not path or label is None:
            raise FileNotFoundError("Stockfish not found: install it or pass its path (--stockfish PATH)")
        run_cfg = dataclasses.replace(cfg, path=path)
        depth = cfg.effective_depth
        out_dir = cache_root(cache, cfg, label) if cache is not None else None
        engine_label = label

        def on_result(game: Game, positions: Optional[list[PositionEval]]) -> None:
            nonlocal done
            done += 1
            ev = build_game_eval(game, positions, engine_label, depth) if positions is not None else None
            if ev is not None:
                results[game.game_id] = ev
                if out_dir is not None:
                    _write_cached(out_dir / cache_file_name(game.game_id), game, positions, engine_label, depth)
            if progress:
                progress(done, total)

        _analyze_uncached(todo, run_cfg, on_result)

    return {g.game_id: results[g.game_id] for g in sorted(selected, key=lambda g: g.end_time) if g.game_id in results}
