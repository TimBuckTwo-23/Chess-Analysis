"""Lichess cloud eval: up to five engine lines for positions other people have analysed on Lichess (no key).

``https://lichess.org/api/cloud-eval?fen=...&multiPv=3``. A 404 means the position is not in the database (common
past the opening). Evaluations in the answer are from White's side; the ``Line`` objects returned here are from
the side to move, as ``models.Line`` asks. Answers are cached for 30 days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import chess

from ...models import Line, Source
from .http import Fetcher, normalize_fen, quote_fen

SOURCE = "cloud_eval"
URL = "https://lichess.org/api/cloud-eval"
MULTIPV = 3


@dataclass
class CloudEval:
    fen: str
    found: bool  # False: not in the cloud database (HTTP 404)
    lines: list[Line] = field(default_factory=list)  # best first
    depth: Optional[int] = None
    knodes: Optional[int] = None
    source: Optional[Source] = None


def url(fen: str, multipv: int = MULTIPV) -> str:
    return f"{URL}?fen={quote_fen(fen)}&multiPv={multipv}"


def _line(fen: str, pv: dict[str, Any], depth: Optional[int]) -> Optional[Line]:
    """One principal variation -> Line (SAN for the legal prefix of the moves; score for the side to move)."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    sign = 1 if board.turn == chess.WHITE else -1
    ucis, sans = [], []
    for uci in str(pv.get("moves") or "").split():
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        sans.append(board.san(move))
        ucis.append(uci)
        board.push(move)
    if not ucis:
        return None
    cp = pv.get("cp")
    mate = pv.get("mate")
    return Line(
        fen=fen,
        moves_uci=ucis,
        moves_san=sans,
        cp_end=sign * int(cp) if isinstance(cp, (int, float)) else None,
        mate_end=sign * int(mate) if isinstance(mate, (int, float)) else None,
        fen_end=board.fen(),
        depth=depth,
    )


def parse(body: Any, fen: str) -> CloudEval:
    """A cloud-eval answer -> CloudEval."""
    if not isinstance(body, dict):
        return CloudEval(fen=fen, found=False)
    depth = body.get("depth") if isinstance(body.get("depth"), int) else None
    lines = [ln for pv in body.get("pvs") or [] if isinstance(pv, dict) for ln in [_line(fen, pv, depth)] if ln]
    knodes = body.get("knodes") if isinstance(body.get("knodes"), int) else None
    return CloudEval(fen=fen, found=bool(lines), lines=lines, depth=depth, knodes=knodes)


def evaluate(fetcher: Fetcher, fen: str, multipv: int = MULTIPV) -> Optional[CloudEval]:
    """The cloud lines for ``fen``; ``found=False`` when Lichess has none; None when the service is unavailable."""
    norm = normalize_fen(fen)
    if norm is None:
        return None
    got = fetcher.get(SOURCE, url(norm, multipv))
    if got is None:
        return None
    if got.status == 404:
        return CloudEval(fen=norm, found=False)
    result = parse(got.data, norm)
    depth = f", depth {result.depth}" if result.depth else ""
    result.source = Source(
        name=f"Lichess cloud eval{depth}",
        url="https://lichess.org/analysis/standard/" + norm.replace(" ", "_"),
        retrieved=got.retrieved,
    )
    return result
