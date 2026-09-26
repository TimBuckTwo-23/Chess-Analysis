"""Study-plan targets against the previous report (C4). Owner: llm.

Each study-plan item carries today's number for its target (``StudyItem.baseline``: {"metric", "value"}). When the
previous report's JSON is available (``cfg.previous``), the same metric in its study plan gives the "before", and
the item gets a line that says what was compared. A verdict (``ProgressItem.improved``: the "Improved" / "Not yet"
badge) comes only from a test; everything else is shown without one.

* Count metrics (``insights.target_for``: "rated games started 23:00–03:00", "quick losses in the French Defense",
  "games with 5...e5 in this position") are counts over all the report's games. When this report's games are the
  previous report's games plus newer ones (the workflow analyses every game each time), the newer games' count is
  the difference, and the share of the games since the previous report (its ``date_to``) is compared with the share
  before: "rated games started 23:00–03:00: 31% of your 600 games before 12 Sep → 7% of your 100 games since".
  The test is a two-proportion z-test (``stats.two_proportion_test``) on each period's effective number of games
  (``effective_units``): games in one session share their time of day and much else, so they are not independent,
  and the test counts a period's games as if every session's games all went the same way, weighting long sessions
  by their size (Kish's effective sample size of the sessions). It needs at least ``MIN_NEW_GAMES`` new games, and
  the p-values of the tested lines are adjusted together (Benjamini-Hochberg) and judged at ``PROGRESS_ALPHA``. A
  line that passes gets ``improved`` True (the change is for the better) or False (for the worse); one that doesn't
  says "no clear change yet" and gets no badge.
* Every other metric (the share of the clock, the score against your rating, errors per 100 moves ...) is computed
  over all the report's games, most of them in both reports, so its two numbers are shown side by side with the
  number of new games, and no verdict.

Which way is better depends on the metric (``direction``): less clock used, fewer blunders per 100 moves and a
higher score against your rating are all better.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from ..models import Game, ProgressItem, Report
from ..stats import MINUS, bh_adjust, pct, per100, two_proportion_test
from .config import CoachConfig

# Which way is better, by words in the metric (insights.target_for). Checked in this order: the first match wins.
_DIRECTIONS: tuple[tuple[str, int], ...] = (
    ("score vs rating", +1),
    ("converted", +1),
    ("engine eval", +1),
    ("per 100 moves", -1),  # blunders, mistakes, missed motifs
    ("share of the clock", -1),
    ("time trouble", -1),
    ("losses on time", -1),
    ("rated games started", -1),
    ("quick losses", -1),
    ("games with ", -1),  # games with the same wrong move in a position
)

# Metrics that count games among all the report's games (insights.target_for): the only ones progress can test,
# as their newer games' count is the difference between the two reports.
COUNT_METRICS = ("rated games started", "quick losses in", "games with ")
MIN_NEW_GAMES = 20  # new games a count metric needs before it is compared at all
PROGRESS_ALPHA = 0.05  # after Benjamini-Hochberg across the tested lines
SESSION_GAP_MIN = 30.0  # a longer pause between games starts a new session (as in the habits section)


def direction(metric: str) -> Optional[int]:
    """+1 when a higher value is better, -1 when lower is better, None when the metric is unknown."""
    m = str(metric or "").lower()
    return next((d for words, d in _DIRECTIONS if words in m), None)


def is_count(metric: str) -> bool:
    """A metric that counts games among all the report's games ("rated games started 23:00–03:00")."""
    m = str(metric or "").lower()
    return any(m.startswith(words) for words in COUNT_METRICS)


def format_value(metric: str, value: float, whole: Optional[bool] = None) -> str:
    """The value in the report's units: percentages, points per 100 games, pawns, counts.

    ``whole``: show a plain number without decimals (default: when it has none).
    """
    m = str(metric or "").lower()
    if "centipawn" in m:
        return f"{value / 100:+.1f}".replace("-", MINUS)
    if "per game" in m and ("score vs rating" in m or "losses on time" in m):
        return f"{per100(value)} per 100 games"
    if m.startswith("share of"):
        return pct(value)
    if "per 100 moves" in m:
        return f"{value:.1f}"
    if whole is None:
        whole = float(value).is_integer()
    if whole:
        return str(int(round(value)))
    return f"{value:.2f}".replace("-", MINUS)


PER_100 = " per 100 games"


def _label(metric: str, per_100: bool = False) -> str:
    """The metric as shown: evaluations in pawns, not centipawns; "per game" left out when the numbers say "per
    100 games"."""
    label = re.sub(r"\s*\(centipawns\)", " (pawns)", str(metric))
    return re.sub(r",?\s+per game\b", "", label) if per_100 else label


def _key(metric: Any) -> str:
    return re.sub(r"\s+", " ", str(metric or "")).strip().lower()


def _num(value: Any) -> Optional[float]:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and value == value
    return float(value) if ok else None


def compare(metric: str, before: float, now: float) -> ProgressItem:
    """One progress line with the two numbers side by side ("50% → 38%"), and no verdict (``title`` is left for
    the caller)."""
    whole = float(before).is_integer() and float(now).is_integer()  # both numbers in the same form
    a, b = format_value(metric, before, whole), format_value(metric, now, whole)
    if a.endswith(PER_100) and b.endswith(PER_100):  # the unit once: "−16 → −4 per 100 games"
        text = f"{_label(metric, True)}: {a[: -len(PER_100)]} → {b}"
    else:
        text = f"{_label(metric)}: {a} → {b}"
    return ProgressItem(title="", metric=metric, before=before, now=now, text=text, improved=None)


# --------------------------------------------------------------------------- the games since the previous report
def _utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _when(value: Any) -> Optional[datetime]:
    """An ISO date-time from the JSON (or a datetime) as an aware UTC datetime; None when unreadable."""
    if isinstance(value, datetime):
        return _utc(value)
    try:
        return _utc(datetime.fromisoformat(str(value)))
    except (TypeError, ValueError):
        return None


def _day(moment: datetime, year: Optional[int] = None) -> str:
    """"12 Sep" (with the year when it differs from ``year``)."""
    return f"{moment.day} {moment:%b}" + (f" {moment.year}" if year is not None and moment.year != year else "")


def session_sizes(games: Sequence[Game], gap_min: float = SESSION_GAP_MIN) -> list[int]:
    """Games per session: live games with pauses of at most ``gap_min`` minutes between them share one (as in the
    habits section); each daily game is its own."""
    from ..analysis.habits import build_timeline

    sizes = Counter(t.session for t in build_timeline(games, session_gap_min=gap_min))
    return list(sizes.values()) + [1] * sum(1 for g in games if g.time_class == "daily")


def effective_units(games: Sequence[Game], gap_min: float = SESSION_GAP_MIN) -> float:
    """How many independent games ``games`` are worth when each session's games may all go the same way: Kish's
    effective sample size of the sessions, (sum of sizes)² / (sum of squared sizes). The number of sessions when
    they are all the same size, fewer when a few long sessions hold most of the games."""
    sizes = session_sizes(games, gap_min)
    squares = sum(n * n for n in sizes)
    return sum(sizes) ** 2 / squares if squares else 0.0


@dataclass
class Periods:
    """The previous report's games against the ones since (``since``: the day of its last game, in words)."""

    since: str
    before: int  # games in the previous report
    new: int  # games since
    before_units: float = 0.0  # effective_units of each period's games
    new_units: float = 0.0
    # the report's games are exactly the previous report's plus newer ones, so a count's difference is the newer
    # games' count (known only with the games)
    comparable: bool = False


def periods(report: Report, previous: dict[str, Any], games: Optional[Sequence[Game]]) -> Optional[Periods]:
    """The two periods, from the previous report's ``n_games`` and ``date_to``; None when it has neither."""
    prev_n, prev_to = _num(previous.get("n_games")), _when(previous.get("date_to"))
    if prev_n is None or prev_to is None:
        return None
    now_to = _when(report.date_to) if report.date_to else None
    since = _day(prev_to, now_to.year if now_to else None)
    if games is None:
        return Periods(since, int(prev_n), max(0, int(report.n_games) - int(prev_n)))
    earlier = [g for g in games if _utc(g.end_time) <= prev_to]
    later = [g for g in games if _utc(g.end_time) > prev_to]
    prev_from = _when(previous.get("date_from"))
    first = min((_utc(g.end_time) for g in earlier), default=None)
    same_player = str(previous.get("username") or report.username).lower() == str(report.username).lower()
    comparable = (
        same_player
        and len(earlier) == int(prev_n)
        and len(earlier) + len(later) == int(report.n_games)
        and (prev_from is None or first is None or abs((first - prev_from).total_seconds()) < 1)
    )
    return Periods(since, len(earlier), len(later), effective_units(earlier), effective_units(later), comparable)


@dataclass
class _Tested:
    item: ProgressItem
    p_value: float
    change: float  # the rate since minus the rate before


def _count_line(metric: str, before: float, now: float, p: Periods) -> tuple[ProgressItem, Optional[_Tested]]:
    """A count metric's line: the share of the new games against the share before, tested when it can be."""
    label = _label(metric)
    fresh = now - before
    valid = p.comparable and 0 <= before <= p.before and 0 <= fresh <= p.new
    if not valid or not p.new:
        line = compare(metric, before, now)
        line.text += f" in all your games ({_new_games(p)})"
        return line, None
    rate_before = before / p.before if p.before else 0.0
    if p.new < MIN_NEW_GAMES:
        text = (f"{label}: {int(fresh)} of your {p.new} {_games(p.new)} since {p.since}, {pct(rate_before)} of the "
                f"{p.before} before (too few new games to compare yet)")
        return ProgressItem("", metric, before, now, text), None
    rate_new = fresh / p.new
    text = (f"{label}: {pct(rate_before)} of your {p.before} {_games(p.before)} before {p.since} → "
            f"{pct(rate_new)} of your {p.new} {_games(p.new)} since")
    item = ProgressItem("", metric, before, now, text)
    if not p.before or not p.before_units or not p.new_units or direction(metric) is None:
        item.text += " (not compared)"
        return item, None
    # each period's counts scaled to its effective number of games (sessions, weighted by size)
    test = two_proportion_test(rate_new * p.new_units, p.new_units, rate_before * p.before_units, p.before_units)
    return item, _Tested(item, test.p_value, rate_new - rate_before)


def _games(n: int) -> str:
    return "game" if n == 1 else "games"


def _new_games(p: Optional[Periods]) -> str:
    if p is None:
        return ""
    if p.new <= 0:
        return f"no new games since {p.since}"
    return f"{p.new} new {_games(p.new)} since {p.since}"


def _side_by_side(metric: str, before: float, now: float, p: Optional[Periods]) -> ProgressItem:
    """Any other metric: its value in the previous report and now, over all the games of each, no verdict."""
    line = compare(metric, before, now)
    if p is not None:
        line.text += f" (all your games; {_new_games(p)})"
    return line


def annotate(report: Report, cfg: CoachConfig, games: Optional[Sequence[Game]] = None) -> None:
    """Fill ``report.coaching.progress`` from ``cfg.previous`` (the last report's JSON). ``games``: the report's
    games, needed to test a count metric (without them every line is shown without a verdict)."""
    coaching = report.coaching
    previous = cfg.previous
    if coaching is None or not isinstance(previous, dict):
        return
    before: dict[str, float] = {}
    for item in previous.get("study_plan") or []:
        if not isinstance(item, dict):
            continue
        baseline = item.get("baseline") or {}
        value = _num(baseline.get("value"))
        if baseline.get("metric") and value is not None:
            before.setdefault(_key(baseline["metric"]), value)
    span = periods(report, previous, games)
    items: list[ProgressItem] = []
    tested: list[_Tested] = []
    for item in report.study_plan:
        metric = str((item.baseline or {}).get("metric") or "")
        now = _num((item.baseline or {}).get("value"))
        if not metric or now is None or _key(metric) not in before:
            continue
        if is_count(metric) and span is not None:
            line, test = _count_line(metric, before[_key(metric)], now, span)
            if test is not None:
                tested.append(test)
        else:
            line = _side_by_side(metric, before[_key(metric)], now, span)
        line.title = item.title
        items.append(line)
    for t, p_adjusted in zip(tested, bh_adjust([t.p_value for t in tested])):
        d = direction(t.item.metric)
        if p_adjusted <= PROGRESS_ALPHA and d is not None and t.change:
            t.item.improved = t.change * d > 0
        else:
            t.item.text += " (no clear change yet)"
    coaching.progress = items
    since = str(previous.get("generated_at") or "")[:10]
    if items and since:
        coaching.settings["progress_since"] = since
    if items and span is not None:
        coaching.settings["progress_new_games"] = span.new
