"""The Lichess opening explorer: which moves masters, and players one or two rating groups above you, choose.

Endpoints (https://github.com/lichess-org/api, tag "Opening Explorer"):

* ``https://explorer.lichess.org/masters?fen=...``: over-the-board games of 2200+ players;
* ``https://explorer.lichess.org/lichess?fen=...&speeds=blitz,rapid&ratings=1200,1400,1600``: Lichess games.

Both need a Lichess personal access token (``Authorization: Bearer``; the explorer answers 401 without one since
2026). Without a token only answers cached by earlier runs are used, and the notes say how to add one.
Answers are cached for 30 days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import chess

from ...models import MoveStat, Source
from .http import TOKEN_NOTE, Fetcher, normalize_fen, quote_fen

SOURCE = "explorer"
MASTERS_URL = "https://explorer.lichess.org/masters"
LICHESS_URL = "https://explorer.lichess.org/lichess"
SPEEDS = ("blitz", "rapid")
TOP_GAMES = 2
GAME_URL = "https://lichess.org/{id}"


@dataclass
class ExplorerResult:
    """Every move played in the position, most played first, with a top game and the explorer's opening name."""

    fen: str
    database: str  # "masters" | "lichess"
    moves: list[MoveStat] = field(default_factory=list)
    total: int = 0  # games in the position
    top_game: str = ""  # URL of the first top game
    eco: str = ""
    name: str = ""
    ratings: list[int] = field(default_factory=list)  # rating groups asked for (lichess database)
    source: Optional[Source] = None

    def top(self, n: int = 3) -> list[MoveStat]:
        return self.moves[:n]

    def rank(self, move: str) -> Optional[int]:
        """1-based rank of ``move`` (SAN or UCI) among the moves played here; None when nobody played it."""
        for i, m in enumerate(self.moves, start=1):
            if move in (m.san, m.uci):
                return i
        return None


def masters_url(fen: str) -> str:
    return f"{MASTERS_URL}?fen={quote_fen(fen)}&topGames={TOP_GAMES}"


def lichess_url(fen: str, ratings: Sequence[int], speeds: Sequence[str] = SPEEDS) -> str:
    return (
        f"{LICHESS_URL}?fen={quote_fen(fen)}&speeds={','.join(speeds)}"
        f"&ratings={','.join(str(r) for r in ratings)}&topGames={TOP_GAMES}"
    )


def _int(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


def parse(body: Any, fen: str, database: str) -> ExplorerResult:
    """An explorer answer -> ExplorerResult. Scores are for the side to move at ``fen``."""
    result = ExplorerResult(fen=fen, database=database)
    if not isinstance(body, dict):
        return result
    try:
        white_to_move = chess.Board(fen).turn == chess.WHITE
    except ValueError:
        white_to_move = True
    moves = []
    for m in body.get("moves") or []:
        if not isinstance(m, dict):
            continue
        w, d, b = _int(m.get("white")), _int(m.get("draws")), _int(m.get("black"))
        n = w + d + b
        mine = w if white_to_move else b
        moves.append(
            MoveStat(
                san=str(m.get("san") or ""),
                uci=str(m.get("uci") or ""),
                games=n,
                score=(mine + 0.5 * d) / n if n else None,
                avg_rating=_int(m.get("averageRating")) or None,
            )
        )
    total = _int(body.get("white")) + _int(body.get("draws")) + _int(body.get("black")) or sum(m.games for m in moves)
    for m in moves:
        m.share = m.games / total if total else 0.0
    moves.sort(key=lambda m: -m.games)  # the explorer already sorts this way; keep it stable if it ever doesn't
    result.moves = moves
    result.total = total
    games = [g for g in (body.get("topGames") or []) if isinstance(g, dict) and g.get("id")]
    if games:
        result.top_game = GAME_URL.format(id=games[0]["id"])
    opening = body.get("opening")
    if isinstance(opening, dict):
        result.eco = str(opening.get("eco") or "")
        result.name = str(opening.get("name") or "")
    return result


def _query(fetcher: Fetcher, fen: str, url: str, database: str, label: str) -> Optional[ExplorerResult]:
    headers = {"Authorization": f"Bearer {fetcher.lichess_token}"} if fetcher.lichess_token else {}
    got = fetcher.get(SOURCE, url, headers=headers, cache_only=not fetcher.lichess_token, miss_note=TOKEN_NOTE)
    if got is None or not got.ok:
        return None
    result = parse(got.data, fen, database)
    # the API needs a token, so the credit links the analysis board, where the explorer opens without one
    result.source = Source(name=label, url=analysis_url(fen), retrieved=got.retrieved)
    return result


def masters(fetcher: Fetcher, fen: str) -> Optional[ExplorerResult]:
    """What masters play in ``fen``; None when the explorer is unavailable (no token, offline, errors)."""
    norm = normalize_fen(fen)
    if norm is None:
        return None
    return _query(fetcher, norm, masters_url(norm), "masters", "Lichess opening explorer (masters)")


def lichess(
    fetcher: Fetcher, fen: str, ratings: Sequence[int], speeds: Sequence[str] = SPEEDS
) -> Optional[ExplorerResult]:
    """What Lichess players in the rating groups ``ratings`` play in ``fen`` (blitz and rapid games)."""
    norm = normalize_fen(fen)
    if norm is None or not ratings:
        return None
    groups = ", ".join(str(r) for r in ratings)
    result = _query(
        fetcher, norm, lichess_url(norm, ratings, speeds), "lichess",
        f"Lichess opening explorer ({'/'.join(speeds)}, rating groups {groups})",
    )
    if result is not None:
        result.ratings = list(ratings)
    return result


def analysis_url(fen: str) -> str:
    """A Lichess analysis board for ``fen`` with the explorer one click away (no token needed to open it)."""
    return "https://lichess.org/analysis/standard/" + fen.replace(" ", "_")
