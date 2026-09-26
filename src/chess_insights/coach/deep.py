"""Re-search critical positions with Stockfish and keep whole lines (C1). Owner: coach-core.

The game analysis (engine.py) keeps one number and one move per position. Explaining a move needs lines: the
engine's best line from the position before your move (MultiPV, so the alternatives come too) and its answer to
the move you played (the refutation). Both start at the position before your move and are scored from your side.

* ``analyse_positions``: the critical positions at ``cfg.depth`` (default 20), MultiPV ``cfg.multipv``.
* ``profile_lines``: every error by either side in the engine-analysed games at ``cfg.profile_depth`` (single PV),
  for the motif profile (C2) and the puzzle export. Small searches: a real run has about 2,500 of them.

Each search is capped in time (``SEARCH_SECONDS`` ...), so a run stays within minutes on four CPUs whatever the
positions; a result that stopped short of the depth says so (``Line.depth``) and the notes count them.

Workers: one Stockfish per thread, like ``engine._analyze_uncached`` (the same ``engine._Analyzer`` lifecycle: an
engine that dies is restarted and the position retried once; every engine is gone when a run returns or raises,
Ctrl+C included). Cache: one JSON file per (EPD, move) under ``cfg.cache_dir/<engine>/d<depth>/``, separate from
the game-analysis cache (``engine.CACHE_FORMAT`` stays valid). The hash is cleared before each position, so a
result never depends on which position an engine searched before it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import queue
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import chess
import chess.engine

from .. import engine as sf
from ..analysis import mistakes
from ..models import CriticalPosition, Line
from .config import CoachConfig

if TYPE_CHECKING:
    from ..context import AnalysisContext

log = logging.getLogger(__name__)

CACHE_FORMAT = 1  # deep-line cache files (independent of engine.CACHE_FORMAT)
# Time caps per search, in seconds (None = none). Measured with Stockfish 16 on one thread: MultiPV 3 at depth 20
# takes 4 to 7 s in opening and middlegame positions, a single line at depth 20 about 1.3 s, depth 10 about 0.03 s.
SEARCH_SECONDS: Optional[float] = 8.0  # the MultiPV search before your move
REFUTATION_SECONDS: Optional[float] = 4.0  # the single line after your move
PROFILE_SECONDS: Optional[float] = 0.5  # each profile search
PROGRESS_EVERY = 30.0  # seconds between progress lines in the log


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


# --------------------------------------------------------------------------- one position
@dataclass(frozen=True)
class _Task:
    """One position to search: the MultiPV search before the move and the single line after it."""

    fen: str
    played_uci: str
    depth: int
    multipv: int
    seconds: Optional[float]  # cap for the search before the move
    refutation_seconds: Optional[float]

    @property
    def epd(self) -> str:
        return " ".join(self.fen.split()[:4])

    @property
    def key(self) -> tuple[str, str]:
        return (self.epd, self.played_uci)


def _limit(depth: int, seconds: Optional[float]) -> chess.engine.Limit:
    return chess.engine.Limit(depth=depth, time=seconds or None)


def _score(pov: chess.engine.PovScore, color: chess.Color) -> tuple[Optional[int], Optional[int]]:
    """(centipawns, mate-in-N) from ``color``'s side; a mate counts as +-engine.MATE_CP centipawns too."""
    score = pov.pov(color)
    mate = score.mate()
    if mate is not None:
        if mate == 0:  # the side to move is mated (python-chess gives no sign)
            mated = pov.turn
            return (-sf.MATE_CP if mated == color else sf.MATE_CP), 0
        return (sf.MATE_CP if mate > 0 else -sf.MATE_CP), mate
    return score.score(), None


def make_line(fen: str, moves: Sequence[chess.Move], cp: Optional[int], mate: Optional[int],
              depth: Optional[int]) -> Line:
    """A Line from ``fen`` along ``moves`` (stopping at the first illegal one), with SAN and the end position."""
    board = chess.Board(fen)
    uci, san = [], []
    for move in moves:
        if move not in board.legal_moves:
            break
        san.append(board.san(move))
        uci.append(move.uci())
        board.push(move)
    return Line(fen=fen, moves_uci=uci, moves_san=san, cp_end=cp, mate_end=mate, fen_end=board.fen(), depth=depth)


def rebase(line: Line, fen: Optional[str] = None, plies: Optional[int] = None) -> Line:
    """``line`` replayed from ``fen`` (the same position with other move counters: the cache is keyed by EPD, so
    a line may have been found from another game) and/or cut to its first ``plies`` moves. The verdict stays the
    engine's verdict on the whole line."""
    moves = line.moves_uci if plies is None else line.moves_uci[:plies]
    return make_line(fen or line.fen, [chess.Move.from_uci(u) for u in moves], line.cp_end, line.mate_end,
                     line.depth)


def _finished(board: chess.Board, color: chess.Color) -> tuple[Optional[int], Optional[int]]:
    """Score of a finished game from ``color``'s side (mate: +-MATE_CP with mate 0; a draw: 0)."""
    if board.is_checkmate():
        return (-sf.MATE_CP if board.turn == color else sf.MATE_CP), 0
    return 0, None


def search(eng: chess.engine.SimpleEngine, task: _Task, engine_label: str = "") -> "DeepResult":
    """The MultiPV search before the move and the single-PV search after it (one engine, hash cleared first)."""
    board = chess.Board(task.fen)
    me = board.turn
    played = chess.Move.from_uci(task.played_uci)
    if played not in board.legal_moves:
        raise ValueError(f"{task.played_uci} is not legal in {task.fen}")
    session = object()  # a new "game" for python-chess: ucinewgame, so the hash starts empty
    infos = eng.analyse(board, _limit(task.depth, task.seconds), multipv=max(1, task.multipv), game=session,
                        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
    if isinstance(infos, dict):
        infos = [infos]
    lines = []
    for info in infos:
        pv, score = info.get("pv") or [], info.get("score")
        if pv and score is not None:
            lines.append(make_line(task.fen, pv, *_score(score, me), info.get("depth")))
    after = board.copy(stack=False)
    after.push(played)
    if after.is_game_over():
        refutation = make_line(task.fen, [played], *_finished(after, me), None)
    else:
        info = eng.analyse(after, _limit(task.depth, task.refutation_seconds), game=session,
                           info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
        score = info.get("score")
        cp, mate = _score(score, me) if score is not None else (None, None)
        refutation = make_line(task.fen, [played, *(info.get("pv") or [])], cp, mate, info.get("depth"))
    return DeepResult(
        epd=task.epd,
        played_uci=task.played_uci,
        best_line=lines[0] if lines else None,
        refutation=refutation,
        alternatives=lines[1:],
        engine=engine_label,
        depth=task.depth,
    )


# --------------------------------------------------------------------------- cache
def cache_dir(root: Path, engine_label: str, depth: int) -> Path:
    """<cache_dir>/<engine>/d<depth>/ (e.g. .../coach/stockfish-16/d20/)."""
    return Path(root) / sf._dir_name(engine_label) / f"d{depth}"


def cache_file(directory: Path, epd: str, played_uci: str) -> Path:
    return directory / (hashlib.sha1(f"{epd} {played_uci}".encode("utf-8")).hexdigest()[:24] + ".json")


def _line_from_json(data: Any) -> Optional[Line]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise TypeError("not a line")
    names = {f.name for f in dataclasses.fields(Line)}
    line = Line(**{k: v for k, v in data.items() if k in names})
    if not isinstance(line.fen, str) or not isinstance(line.moves_uci, list):
        raise TypeError("not a line")
    return line


def _covers(stored: Optional[float], wanted: Optional[float]) -> bool:
    """A search capped at ``stored`` seconds is as good as one capped at ``wanted`` (None = no cap)."""
    return stored is None or (wanted is not None and stored >= wanted)


def _load(path: Path, task: _Task, engine_label: str) -> Optional[DeepResult]:
    """The cached result for ``task``, or None if missing, unreadable, or from a smaller search."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("format") != CACHE_FORMAT
            or data.get("epd") != task.epd
            or data.get("played") != task.played_uci
            or data.get("depth") != task.depth
            or int(data.get("multipv", 0)) < task.multipv
            or not _covers(data.get("seconds"), task.seconds)
            or not _covers(data.get("refutation_seconds"), task.refutation_seconds)
        ):
            return None
        lines = [_line_from_json(x) for x in data.get("lines", [])]
        return DeepResult(
            epd=task.epd,
            played_uci=task.played_uci,
            best_line=lines[0] if lines else None,
            refutation=_line_from_json(data.get("refutation")),
            alternatives=[x for x in lines[1 : task.multipv] if x is not None],
            engine=str(data.get("engine") or engine_label),
            depth=task.depth,
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _save(path: Path, task: _Task, result: DeepResult) -> Optional[OSError]:
    """Atomic write (temp file + rename); returns the error when the file could not be written."""
    payload = {
        "format": CACHE_FORMAT,
        "epd": task.epd,
        "played": task.played_uci,
        "engine": result.engine,
        "depth": task.depth,
        "multipv": task.multipv,
        "seconds": task.seconds,
        "refutation_seconds": task.refutation_seconds,
        "lines": [dataclasses.asdict(x) for x in [result.best_line, *result.alternatives] if x is not None],
        "refutation": dataclasses.asdict(result.refutation) if result.refutation is not None else None,
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
        return exc
    return None


# --------------------------------------------------------------------------- running many positions
class _Worker(sf._Analyzer):
    """One long-lived engine for positions (``engine._Analyzer``'s lifecycle: restarted once if it dies)."""

    def __init__(self, cfg: sf.EngineConfig, engine_label: str) -> None:
        super().__init__(cfg)
        self.label = engine_label

    def evaluate(self, task: _Task) -> DeepResult:  # type: ignore[override]
        try:
            return search(self._get(), task, self.label)
        except chess.engine.EngineTerminatedError:
            self._discard()  # the engine process died: start a new one and give this position one more try
            return search(self._get(), task, self.label)


OnResult = Callable[[_Task, Optional[DeepResult]], None]


def _run(tasks: list[_Task], ecfg: sf.EngineConfig, engine_label: str, on_result: OnResult) -> None:
    """Search ``tasks`` with up to ``ecfg.worker_count()`` engines, reporting each result in this thread.

    The pattern of ``engine._analyze_uncached``: worker threads (``engine._work``) take tasks from a queue; this
    thread waits with a timeout (so Ctrl+C is noticed promptly) and kills every engine before returning or raising.
    """
    workers = max(1, min(ecfg.worker_count(), len(tasks)))
    analyzers = [_Worker(ecfg, engine_label) for _ in range(workers)]
    seen_errors: set[str] = set()
    threads: list[threading.Thread] = []

    def failed(task: _Task, exc: BaseException) -> None:
        text = str(exc) or type(exc).__name__
        log.log(logging.DEBUG if text in seen_errors else logging.WARNING, "deep analysis of %s failed: %s",
                task.fen, text)
        seen_errors.add(text)

    try:
        if workers == 1:
            for task in tasks:
                try:
                    result: Optional[DeepResult] = analyzers[0].evaluate(task)
                except sf._Cancelled:
                    return
                except Exception as exc:  # noqa: BLE001 — one bad position must not stop the run
                    failed(task, exc)
                    result = None
                on_result(task, result)
            return
        todo: queue.SimpleQueue = queue.SimpleQueue()
        done: queue.SimpleQueue = queue.SimpleQueue()
        for task in tasks:
            todo.put(task)
        threads = [
            threading.Thread(target=sf._work, args=(a, todo, done), name=f"coach-stockfish-{i + 1}", daemon=True)
            for i, a in enumerate(analyzers)
        ]
        for t in threads:
            t.start()
        for _ in tasks:
            while True:
                try:  # a timed wait, so Ctrl+C is noticed promptly on Windows too
                    task, result, exc = done.get(timeout=0.2)
                    break
                except queue.Empty:
                    if not any(t.is_alive() for t in threads) and done.empty():
                        log.warning("deep analysis workers stopped early; some positions were not analysed")
                        return
            if exc is not None:
                failed(task, exc)
            on_result(task, result)
    finally:
        for a in analyzers:
            a.close()
        for a in analyzers:
            a.wait()
        for t in threads:
            t.join(timeout=5)


def _progress(what: str, total: int) -> Callable[[int], None]:
    """A logger of "<what>: 40/150 positions" lines, at most every ``PROGRESS_EVERY`` seconds, and at the end."""
    started = last = time.monotonic()

    def report(done: int) -> None:
        nonlocal last
        now = time.monotonic()
        if done == total or now - last >= PROGRESS_EVERY:
            last = now
            log.info("%s: %d/%d positions (%.0f s)", what, done, total, now - started)

    return report


def _engine(cfg: CoachConfig) -> tuple[Optional[str], Optional[str]]:
    """(Stockfish path, its name) or (None, None)."""
    path = sf.find_stockfish(cfg.stockfish)
    return (path, sf.engine_name(path)) if path else (None, None)


def run_tasks(
    tasks: Sequence[_Task], cfg: CoachConfig, path: str, engine_label: str, what: str
) -> dict[tuple[str, str], DeepResult]:
    """Results for ``tasks`` (one per (EPD, move)): cached ones loaded, the rest searched in parallel and cached."""
    unique = list({t.key: t for t in tasks}.values())
    results: dict[tuple[str, str], DeepResult] = {}
    todo: list[_Task] = []
    root = Path(cfg.cache_dir) if cfg.cache_dir is not None else None
    for task in unique:
        cached = None
        if root is not None:
            cached = _load(cache_file(cache_dir(root, engine_label, task.depth), *task.key), task, engine_label)
        if cached is not None:
            results[task.key] = cached
        else:
            todo.append(task)
    if not todo:
        return results
    ecfg = sf.EngineConfig(path=path, depth=todo[0].depth, threads=1, hash_mb=cfg.hash_mb, workers=cfg.workers)
    log.info("%s: %d positions (%d cached) at depth %d with %d engines", what, len(unique), len(results),
             todo[0].depth, max(1, min(ecfg.worker_count(), len(todo))))
    report = _progress(what, len(unique))
    done = len(results)
    cache_errors = 0

    def on_result(task: _Task, result: Optional[DeepResult]) -> None:
        nonlocal done, cache_errors
        done += 1
        if result is not None:
            results[task.key] = result
            if root is not None:
                error = _save(cache_file(cache_dir(root, engine_label, task.depth), *task.key), task, result)
                if error is not None:
                    cache_errors += 1
                    log.log(logging.WARNING if cache_errors == 1 else logging.DEBUG,
                            "could not cache the deep analysis in %s: %s", root, error)
        report(done)

    _run(todo, ecfg, engine_label, on_result)
    return results


def _short(results: dict[tuple[str, str], DeepResult], depth: int) -> int:
    """Results whose search stopped short of ``depth`` (the time cap)."""
    return sum(
        1 for r in results.values()
        if any(x is not None and x.depth is not None and x.depth < depth for x in (r.best_line, r.refutation))
    )


def _config_seconds(cfg: CoachConfig) -> tuple[Optional[float], Optional[float]]:
    """(MultiPV cap, refutation cap). ``cfg.search_seconds`` when the config has one: a number caps the MultiPV
    search (the refutation gets half), 0 or less means no cap; None or no such setting: the defaults above."""
    cap = getattr(cfg, "search_seconds", None)
    if cap is None:
        return SEARCH_SECONDS, REFUTATION_SECONDS
    return (float(cap), float(cap) / 2) if cap > 0 else (None, None)


# --------------------------------------------------------------------------- public API
def analyse_positions(
    positions: list[CriticalPosition], cfg: CoachConfig, notes: list[str]
) -> dict[tuple[str, str], DeepResult]:
    """(EPD, played UCI) -> DeepResult, cached under ``cfg.cache_dir``/<engine>/d<depth>/.

    MultiPV ``cfg.multipv`` at ``cfg.depth`` before your move, one line after it. Without Stockfish: a note and {}.
    """
    if not positions:
        return {}
    path, label = _engine(cfg)
    if not path or not label:
        notes.append("Deep analysis skipped: Stockfish not found (install it or pass --stockfish).")
        return {}
    seconds, refutation_seconds = _config_seconds(cfg)
    tasks = [
        _Task(p.fen, p.played_uci, cfg.depth, max(1, cfg.multipv), seconds, refutation_seconds) for p in positions
    ]
    results = run_tasks(tasks, cfg, path, label, "Deep analysis")
    missing = len({t.key for t in tasks}) - len(results)
    if missing:
        notes.append(f"Deep analysis failed for {missing} of {len({t.key for t in tasks})} positions.")
    short = _short(results, cfg.depth)
    if short:
        caps = (f" (the time cap: {seconds:g} s before your move, {refutation_seconds:g} s after it)"
                if seconds and refutation_seconds else "")
        notes.append(f"{short} of {len(results)} positions were searched to less than depth {cfg.depth}{caps}.")
    return results


_PROFILE_MEMO: dict[str, Any] = {}  # the last profile_lines result, so a second caller in the same run pays nothing


def _profile_key(ctx: "AnalysisContext", cfg: CoachConfig) -> tuple:
    return (id(ctx), len(ctx.evals), len(ctx.games), cfg.profile_depth, str(cfg.cache_dir), cfg.stockfish,
            getattr(cfg, "search_seconds", None))


def profile_errors(ctx: "AnalysisContext") -> list[ProfileError]:
    """Every error by either side (drop >= ``mistakes.MIN_DROP``, not the engine's own move) in the analysed
    standard games, without lines yet; in game order."""
    out: list[ProfileError] = []
    by_id = ctx.games_by_id
    for gid, ev in ctx.evals.items():
        game = by_id.get(gid)
        if game is None or game.rules != "chess":
            continue
        wanted = {
            p.ply: p for p in ev.plies
            if 0 <= p.ply < game.plies and mistakes._is_error(p, game.moves_san[p.ply], mistakes.MIN_DROP)
        }
        if not wanted:
            continue
        try:
            board = chess.Board(game.initial_fen or chess.STARTING_FEN)
            for ply, san in enumerate(game.moves_san[: max(wanted) + 1]):
                move = board.parse_san(san)
                p = wanted.get(ply)
                if p is not None:
                    out.append(ProfileError(
                        game_id=gid, ply=ply, side="you" if p.is_user else "opponent", time_class=game.time_class,
                        fen=board.fen(), played_uci=move.uci(), drop=float(p.win_before - p.win_after),
                    ))
                board.push(move)
        except ValueError:  # an illegal move in a corrupt record: keep what we have
            continue
    return out


def profile_lines(ctx: "AnalysisContext", cfg: CoachConfig, notes: list[str]) -> list[ProfileError]:
    """Every error (both sides, drop >= mistakes.MIN_DROP) in the engine-analysed games with a best line and a
    refutation at ``cfg.profile_depth`` (single PV), cached like ``analyse_positions``.

    Errors whose search failed keep ``best_line`` / ``refutation`` None. Without Stockfish: a note and [].
    The result is kept for the rest of the run, so the motif profile and the puzzle export share one pass.
    """
    key = _profile_key(ctx, cfg)
    memo = _PROFILE_MEMO.get("last")
    if memo is not None and memo[0] == key and memo[1]() is ctx:
        return list(memo[2])
    errors = profile_errors(ctx)
    if not errors:
        return []
    path, label = _engine(cfg)
    if not path or not label:
        if not any("Stockfish not found" in n for n in notes):  # said once is enough
            notes.append("Motif lines skipped: Stockfish not found (install it or pass --stockfish).")
        return []
    depth = cfg.profile_depth
    tasks = [_Task(e.fen, e.played_uci, depth, 1, PROFILE_SECONDS, PROFILE_SECONDS) for e in errors]
    results = run_tasks(tasks, cfg, path, label, "Motif lines")
    for e in errors:
        r = results.get((" ".join(e.fen.split()[:4]), e.played_uci))
        if r is not None:
            e.best_line, e.refutation = (
                None if x is None else (x if x.fen == e.fen else rebase(x, e.fen)) for x in (r.best_line, r.refutation)
            )
    missing = sum(1 for e in errors if e.refutation is None)
    if missing:
        notes.append(f"Motif lines missing for {missing} of {len(errors)} errors (the engine failed on them).")
    try:
        _PROFILE_MEMO["last"] = (key, weakref.ref(ctx), errors)
    except TypeError:  # a context type without weak references: no memo
        pass
    return list(errors)
