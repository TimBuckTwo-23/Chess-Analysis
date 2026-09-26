"""The Lichess tablebase: the exact result of positions with 7 pieces or fewer (no key).

``https://tablebase.lichess.org/standard?fen=...`` answers with a ``category`` for the side to move (win, draw,
loss, and the 50-move-rule cases cursed-win / blessed-loss), the distance to zeroing (DTZ) and to mate (DTM),
and every legal move with the category it leaves *for the opponent*, best first. Tablebase answers never change,
so they are cached for good.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import chess

from ...models import Source
from .http import Fetcher, normalize_fen, quote_fen

SOURCE = "tablebase"
URL = "https://tablebase.lichess.org/standard"
MAX_PIECES = 7

# category -> the practical result for the side to move under the 50-move rule. "maybe-*" answers (the DTZ was
# rounded and the halfmove clock decides) and "unknown" stay None: the tablebase is not sure, so neither are we.
OUTCOME = {
    "win": "win",
    "syzygy-win": "win",
    "cursed-win": "draw",  # a win on the board, but not within the 50-move rule
    "draw": "draw",
    "blessed-loss": "draw",
    "syzygy-loss": "loss",
    "loss": "loss",
}
FLIP = {"win": "loss", "draw": "draw", "loss": "win"}


def outcome(category: Optional[str]) -> Optional[str]:
    """'win' / 'draw' / 'loss' for the side to move, or None when the tablebase is unsure."""
    return OUTCOME.get(str(category or ""))


@dataclass
class TablebaseMove:
    uci: str
    san: str
    category: str  # for the side to move AFTER this move (the opponent)
    dtz: Optional[int] = None
    dtm: Optional[int] = None
    checkmate: bool = False
    stalemate: bool = False

    @property
    def outcome_for_mover(self) -> Optional[str]:
        """The result this move leaves for the side that played it."""
        o = outcome(self.category)
        return FLIP[o] if o else None


@dataclass
class TablebaseResult:
    fen: str
    category: str
    dtz: Optional[int] = None
    dtm: Optional[int] = None
    checkmate: bool = False
    stalemate: bool = False
    insufficient_material: bool = False
    moves: list[TablebaseMove] = field(default_factory=list)  # best first for the side to move
    source: Optional[Source] = None

    @property
    def outcome(self) -> Optional[str]:
        return outcome(self.category)

    @property
    def best(self) -> Optional[TablebaseMove]:
        return self.moves[0] if self.moves else None

    def move(self, move: str) -> Optional[TablebaseMove]:
        """The entry for ``move`` (UCI or SAN), or None."""
        for m in self.moves:
            if move in (m.uci, m.san):
                return m
        return None

    def as_dict(self, played: Optional[str] = None) -> dict[str, Any]:
        """The facts for ``Explanation.tablebase``: result, the tablebase's move, what ``played`` left."""
        out: dict[str, Any] = {"category": self.category, "result": self.outcome, "dtz": self.dtz, "dtm": self.dtm}
        if self.best is not None:
            out.update(best=self.best.san, best_uci=self.best.uci, best_result=self.best.outcome_for_mover)
        mine = self.move(played) if played else None
        if mine is not None:
            out.update(played=mine.san, played_result=mine.outcome_for_mover, played_category=mine.category)
        if self.source is not None:
            out.update(source=self.source.name, url=self.source.url, retrieved=self.source.retrieved)
        return out


def _opt_int(x: Any) -> Optional[int]:
    return int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def parse(body: Any, fen: str) -> Optional[TablebaseResult]:
    """A tablebase answer -> TablebaseResult (None when it isn't one)."""
    if not isinstance(body, dict) or "category" not in body:
        return None
    moves = [
        TablebaseMove(
            uci=str(m.get("uci") or ""),
            san=str(m.get("san") or ""),
            category=str(m.get("category") or ""),
            dtz=_opt_int(m.get("dtz")),
            dtm=_opt_int(m.get("dtm")),
            checkmate=bool(m.get("checkmate")),
            stalemate=bool(m.get("stalemate")),
        )
        for m in body.get("moves") or []
        if isinstance(m, dict) and m.get("uci")
    ]
    return TablebaseResult(
        fen=fen,
        category=str(body.get("category") or ""),
        dtz=_opt_int(body.get("dtz")),
        dtm=_opt_int(body.get("dtm")),
        checkmate=bool(body.get("checkmate")),
        stalemate=bool(body.get("stalemate")),
        insufficient_material=bool(body.get("insufficient_material")),
        moves=moves,
    )


def pieces(fen: str) -> Optional[int]:
    """Pieces on the board, kings and pawns included; None for an invalid FEN."""
    try:
        return chess.popcount(chess.Board(fen).occupied)
    except ValueError:
        return None


def url(fen: str) -> str:
    return f"{URL}?fen={quote_fen(fen)}"


def probe(fetcher: Fetcher, fen: str) -> Optional[TablebaseResult]:
    """The tablebase verdict for ``fen``; None with more than 7 pieces, a finished game or an unavailable service."""
    norm = normalize_fen(fen)
    if norm is None:
        return None
    board = chess.Board(norm)
    if chess.popcount(board.occupied) > MAX_PIECES or board.is_game_over():
        return None
    got = fetcher.get(SOURCE, url(norm))
    if got is None or not got.ok:
        return None
    result = parse(got.data, norm)
    if result is None:
        return None
    result.source = Source(
        name="Lichess tablebase",
        url="https://lichess.org/analysis/standard/" + norm.replace(" ", "_"),
        retrieved=got.retrieved,
    )
    return result
