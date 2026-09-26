"""Optional Maia-2 move probabilities at your rating (C4, extra [maia]). Owner: llm.

Maia-2 (CSSLab, MIT, ``pip install maia2``) predicts which move a human of a given Lichess rating plays. For each
explained position this adds P(best move) and P(played move) at your Lichess-equivalent rating, using the blitz
model for blitz and bullet games and the rapid model for rapid (and daily) games, and puts the explanations in
order of drop x P(best move): the misses that players at your level usually find come first.

Nothing here changes a claim; without the package (or with ``--offline``, since Maia-2 downloads its weights on
first use) the step leaves a note and the explanations keep their order.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import chess

from ..models import Explanation, Report
from .config import CoachConfig
from .packet import user_ratings

log = logging.getLogger(__name__)

# Maia-2 model per time class: the blitz model for blitz and bullet, rapid for the slower formats.
MODEL_FOR = {"bullet": "blitz", "blitz": "blitz", "rapid": "rapid", "daily": "rapid"}
# Fallback chess.com -> Lichess conversion when coach.rating_map is not available: the ChessGoals blitz table
# (updated July 2026: chess.com 900 is about Lichess 1360, 1000 about 1425), extended in a straight line and used
# for every format. ``CoachConfig.rating_map`` may give other points per time class: {"rapid": [[800, 1250], ...]}.
FALLBACK_POINTS = ((900, 1360), (1000, 1425))
MAX_FAILURES = 3  # stop after this many failed predictions in a row with none working

# (fen, lichess_rating, model_type) -> {uci: probability}
Predictor = Callable[[str, int, str], dict[str, float]]


def _interpolate(rating: float, points: Any) -> Optional[int]:
    """Piecewise-linear through (chess.com, Lichess) ``points``, extended in a straight line past both ends."""
    try:
        pts = sorted((float(a), float(b)) for a, b in points)
    except (TypeError, ValueError):
        return None
    if len(pts) < 2:
        return None
    if rating <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif rating >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        (x0, y0), (x1, y1) = next((a, b) for a, b in zip(pts, pts[1:]) if a[0] <= rating <= b[0])
    if x1 == x0:
        return int(round(y0))
    return int(round(y0 + (rating - x0) * (y1 - y0) / (x1 - x0)))


def lichess_rating(rating: int, time_class: str, cfg: CoachConfig) -> tuple[int, str]:
    """Your Lichess-equivalent rating and where the conversion came from.

    Uses ``coach.rating_map`` when another part of the layer provides it (``lichess_equivalent(rating,
    time_class)``), else the fallback table above (or ``cfg.rating_map`` points for the time class).
    """
    try:
        from . import rating_map  # type: ignore[attr-defined]
    except ImportError:
        rating_map = None
    fn = getattr(rating_map, "lichess_equivalent", None) if rating_map is not None else None
    if callable(fn):
        try:
            value = fn(rating, time_class)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(round(value)), "coach.rating_map"
        except Exception as exc:  # noqa: BLE001 — fall back to the local table
            log.debug("rating_map.lichess_equivalent failed: %s", exc)
    custom = (cfg.rating_map or {}).get(time_class)
    value = _interpolate(rating, custom) if custom else None
    if value is not None:
        return value, "your rating map"
    return _interpolate(rating, FALLBACK_POINTS) or int(rating), "ChessGoals blitz table (approximate)"


def _mirror(uci: str) -> str:
    """The same move with the board flipped (a2a4 -> a7a5): Maia-2 works from the side to move's view."""
    try:
        m = chess.Move.from_uci(uci)
    except ValueError:
        return uci
    promo = chess.piece_symbol(m.promotion) if m.promotion else ""
    return "".join(chess.square_name(chess.square_mirror(sq)) for sq in (m.from_square, m.to_square)) + promo


def _uci(board: chess.Board, move: Optional[str]) -> Optional[str]:
    text = str(move or "").replace("…", "...").split(".")[-1].strip()
    if not text:
        return None
    try:
        return board.parse_san(text).uci()
    except ValueError:
        try:
            m = chess.Move.from_uci(text)
        except ValueError:
            return None
        return m.uci() if m in board.legal_moves else None


def move_ucis(exp: Explanation) -> tuple[Optional[str], Optional[str]]:
    """(best, played) in UCI for the position before your move."""
    try:
        board = chess.Board(exp.fen)
    except ValueError:
        return None, None
    epd = " ".join(exp.fen.split()[:4])
    best = played = None
    if exp.best_line and exp.best_line.moves_uci and " ".join((exp.best_line.fen or exp.fen).split()[:4]) == epd:
        best = exp.best_line.moves_uci[0]
    if exp.refutation and exp.refutation.moves_uci and " ".join((exp.refutation.fen or exp.fen).split()[:4]) == epd:
        played = exp.refutation.moves_uci[0]
    return best or _uci(board, exp.best), played or _uci(board, exp.played)


def board_view(probs: dict[str, float], fen: str) -> dict[str, float]:
    """``probs`` keyed by moves on the real board. A predictor that answers from the side to move's view (Black's
    moves flipped, a7a5 given as a2a4) is recognised by its keys: fewer of them are legal as they stand than
    flipped."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return dict(probs)
    legal = {m.uci() for m in board.legal_moves}
    as_is = sum(1 for k in probs if k in legal)
    flipped = sum(1 for k in probs if _mirror(k) in legal)
    return {_mirror(k): v for k, v in probs.items()} if flipped > as_is else dict(probs)


def _probability(probs: dict[str, float], uci: str) -> float:
    return float(probs.get(uci, 0.0))


def maia2_predictor() -> Optional[Predictor]:
    """A predictor backed by the maia2 package (models loaded on first use, on the CPU); None if not installed."""
    try:
        from maia2 import inference, model  # type: ignore[import-not-found]
    except ImportError:
        return None
    models: dict[str, Any] = {}
    prepared: list[Any] = []

    def predict(fen: str, rating: int, model_type: str) -> dict[str, float]:
        if model_type not in models:
            models[model_type] = model.from_pretrained(type=model_type, device="cpu")
        if not prepared:
            prepared.append(inference.prepare())
        move_probs, _ = inference.inference_each(models[model_type], prepared[0], fen, rating, rating)
        return {str(k): float(v) for k, v in dict(move_probs).items()}

    return predict


def annotate(report: Report, cfg: CoachConfig, predictor: Optional[Predictor] = None) -> None:
    """Set ``Explanation.maia`` and re-rank explanations by drop x P(best move).

    ``predictor`` is for tests; normally Maia-2 is loaded from the optional package. Explanations without a
    prediction (no rating for their format, no usable move) go after the ranked ones, in their old order.
    """
    coaching = report.coaching
    if coaching is None or not coaching.explanations:
        return
    if predictor is None:
        if cfg.offline:
            coaching.notes.append("Maia-2 skipped (--offline: it downloads its model weights on first use).")
            return
        predictor = maia2_predictor()
        if predictor is None:
            coaching.notes.append("Maia-2 skipped: the maia2 package is not installed (pip install maia2).")
            return
    ratings = user_ratings(report)
    scored, failures, last_error = 0, 0, ""
    used: dict[str, dict[str, Any]] = {}
    for exp in coaching.explanations:
        model_type = MODEL_FOR.get(exp.time_class)
        rating = ratings.get(exp.time_class)
        best, played = move_ucis(exp)
        if model_type is None or rating is None or not best or not played:
            continue
        lichess, source = lichess_rating(rating, exp.time_class, cfg)
        try:
            probs = board_view(predictor(exp.fen, lichess, model_type), exp.fen)
        except Exception as exc:  # noqa: BLE001 — one position failing must not stop the rest
            failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            if not scored and failures >= MAX_FAILURES:
                break
            continue
        exp.maia = {
            "rating": lichess,
            "chesscom_rating": rating,
            "model": model_type,
            "p_best": round(_probability(probs, best), 3),
            "p_played": round(_probability(probs, played), 3),
        }
        used.setdefault(exp.time_class, {"chesscom": rating, "lichess": lichess, "model": model_type, "source": source})
        scored += 1
    if failures:
        coaching.notes.append(f"Maia-2 failed on {failures} position(s) ({last_error}).")
    if not scored:
        return
    order = {id(e): i for i, e in enumerate(coaching.explanations)}

    def key(e: Explanation) -> tuple[int, float, int]:
        if e.maia and isinstance(e.maia.get("p_best"), (int, float)):
            return 0, -(float(e.drop or 0.0) * float(e.maia["p_best"])), order[id(e)]
        return 1, 0.0, order[id(e)]

    coaching.explanations.sort(key=key)
    coaching.settings["maia"] = {
        "positions": scored, "order": "win-% lost x P(best move) at your rating", "ratings": used
    }
