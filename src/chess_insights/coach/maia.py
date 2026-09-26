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
from . import rating_map
from .config import CoachConfig
from .packet import user_ratings

log = logging.getLogger(__name__)

# Maia-2 model per time class: the blitz model for blitz and bullet, rapid for the slower formats.
MODEL_FOR = {"bullet": "blitz", "blitz": "blitz", "rapid": "rapid", "daily": "rapid"}
MAX_FAILURES = 3  # stop after this many failed predictions in a row with none working

# (fen, lichess_rating, model_type) -> {uci: probability}
Predictor = Callable[[str, int, str], dict[str, float]]


def lichess_rating(rating: int, time_class: str, cfg: CoachConfig) -> tuple[int, str]:
    """Your Lichess-equivalent rating and where the conversion came from: ``rating_map.lichess_equivalent`` with
    ``cfg.rating_map`` (per-format rows that replace the built-in table, such as ``{"rapid": [[800, 1250], ...]}``),
    the same conversion the opening explorer's rating groups use."""
    overrides = cfg.rating_map or None
    return (rating_map.lichess_equivalent(rating, time_class, overrides),
            rating_map.source_of(rating, time_class, overrides))


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
