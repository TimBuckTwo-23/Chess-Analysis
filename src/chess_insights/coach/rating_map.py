"""chess.com ratings on the Lichess scale, and the Lichess rating groups the opening explorer filters by.

The two sites' ratings differ a lot at club level (Lichess starts players at 1500 and uses Glicko-2 with other
pools), so "players one or two groups above you" on the Lichess explorer needs a conversion first. The table is
deliberately small and editable: change the rows below, or pass ``CoachConfig.rating_map`` such as
``{"blitz": [[900, 1360], [1000, 1425]]}`` to replace one format's rows for a run.

Source: ChessGoals rating comparison, https://chessgoals.com/rating-comparison/ (July 2026 update), which fits
the ratings of players with accounts on both sites. Only the rows marked in ``CHECKED`` were compared with that
page when this table was written (chess.com blitz 900 is about Lichess blitz 1360, 1000 about 1425); the other
rows are rough estimates in the same spirit, to be refreshed from the page (ChessGoals notes the gap moves by
about 30 points a year). Numbers come out rounded ("about 1390"), never as exact equivalents.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, Optional, Sequence

SOURCE_NAME = "ChessGoals rating comparison, July 2026"
SOURCE_URL = "https://chessgoals.com/rating-comparison/"

# chess.com rating -> Lichess rating in the same format, ascending. Edit freely; keep at least two rows per format.
TABLES: dict[str, list[tuple[int, int]]] = {
    "blitz": [
        (400, 1035), (600, 1165), (800, 1295), (900, 1360), (1000, 1425), (1200, 1555), (1400, 1690),
        (1600, 1830), (1800, 1970), (2000, 2110), (2200, 2260), (2400, 2410), (2600, 2570),
    ],
    "rapid": [
        (400, 1100), (600, 1250), (800, 1390), (1000, 1520), (1200, 1640), (1400, 1760), (1600, 1880),
        (1800, 2000), (2000, 2120), (2200, 2250), (2400, 2380),
    ],
    "bullet": [
        (400, 900), (600, 1080), (800, 1250), (1000, 1400), (1200, 1540), (1400, 1680), (1600, 1810),
        (1800, 1940), (2000, 2070), (2200, 2200), (2400, 2340),
    ],
}
# rows compared with the source page (format -> chess.com ratings); everything else is an estimate
CHECKED: dict[str, frozenset[int]] = {"blitz": frozenset({900, 1000}), "rapid": frozenset(), "bullet": frozenset()}

# The Lichess opening explorer's rating groups: a group holds the ratings from its value up to the next one.
LICHESS_GROUPS = (0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500)
LICHESS_MIN, LICHESS_MAX = 400, 3300


def _rows(time_class: str, overrides: Optional[dict[str, Any]] = None) -> Optional[list[tuple[int, int]]]:
    rows: Any = (overrides or {}).get(time_class, TABLES.get(time_class))
    if isinstance(rows, dict):  # {"900": 1360, ...}
        rows = list(rows.items())
    try:
        pairs = sorted((int(a), int(b)) for a, b in rows or [])
    except (TypeError, ValueError):
        return None
    return pairs if len(pairs) >= 2 else None


def to_lichess(
    rating: Optional[float], time_class: str = "blitz", overrides: Optional[dict[str, Any]] = None
) -> Optional[int]:
    """A chess.com rating in ``time_class`` on the Lichess scale of the same format (piecewise linear between the
    table's rows, straight on past its ends, rounded to 5); None without a rating or a table for the format."""
    if rating is None:
        return None
    rows = _rows(time_class, overrides)
    if rows is None:
        return None
    xs = [a for a, _ in rows]
    i = min(max(bisect_right(xs, rating) - 1, 0), len(rows) - 2)
    (x0, y0), (x1, y1) = rows[i], rows[i + 1]
    value = y0 + (y1 - y0) * (float(rating) - x0) / (x1 - x0) if x1 != x0 else float(y0)
    value = min(max(value, LICHESS_MIN), LICHESS_MAX)
    return int(5 * round(value / 5))


def rating_group(lichess_rating: float) -> int:
    """The explorer group that holds ``lichess_rating``: 1390 -> 1200, 1400 -> 1400, 2600 -> 2500."""
    return LICHESS_GROUPS[max(0, bisect_right(LICHESS_GROUPS, lichess_rating) - 1)]


def peer_groups(lichess_rating: float, above: int = 2) -> list[int]:
    """Your group and the ``above`` groups over it: 1390 -> [1200, 1400, 1600] (fewer at the top of the scale)."""
    i = LICHESS_GROUPS.index(rating_group(lichess_rating))
    return list(LICHESS_GROUPS[i : i + above + 1])


def groups_for(
    chesscom_rating: Optional[float], time_class: str = "blitz", overrides: Optional[dict[str, Any]] = None,
    above: int = 2,
) -> list[int]:
    """Lichess explorer groups for a chess.com rating: blitz 949 -> about 1390 -> [1200, 1400, 1600]."""
    lichess = to_lichess(chesscom_rating, time_class, overrides)
    return peer_groups(lichess, above) if lichess is not None else []


def describe(chesscom_rating: float, time_class: str = "blitz", overrides: Optional[dict[str, Any]] = None) -> str:
    """'your chess.com blitz 949 is about 1390 on Lichess (ChessGoals rating comparison, July 2026)'."""
    lichess = to_lichess(chesscom_rating, time_class, overrides)
    if lichess is None:
        return ""
    return f"your chess.com {time_class} {int(round(chesscom_rating))} is about {lichess} on Lichess ({SOURCE_NAME})"


def group_label(groups: Sequence[int]) -> str:
    """'1200–1799' for [1200, 1400, 1600]; '2500+' for [2500]."""
    if not groups:
        return ""
    lo, hi = min(groups), max(groups)
    i = LICHESS_GROUPS.index(hi) if hi in LICHESS_GROUPS else -1
    if i < 0 or i == len(LICHESS_GROUPS) - 1:
        return f"{lo}+"
    return f"{lo}–{LICHESS_GROUPS[i + 1] - 1}"
