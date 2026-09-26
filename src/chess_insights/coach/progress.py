"""Study-plan targets against the previous report (C4). Owner: llm.

Each study-plan item carries today's number for its target (``StudyItem.baseline``: {"metric", "value"}). When the
previous report's JSON is available (``cfg.previous``), the same metric in its study plan gives the "before", and
the item gets a line such as "share of the clock used on the first 15 moves (blitz): 50% → 38%". Whether that is
an improvement depends on the metric: less clock used, fewer blunders per 100 moves and a higher score against
your rating are all better (the metrics are the ones ``insights.target_for`` writes).

This is an observation next to the plan: the two reports usually share most of their games, so a change here is
not a test of anything.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..models import ProgressItem, Report
from ..stats import MINUS, pct, per100
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


def direction(metric: str) -> Optional[int]:
    """+1 when a higher value is better, -1 when lower is better, None when the metric is unknown."""
    m = str(metric or "").lower()
    return next((d for words, d in _DIRECTIONS if words in m), None)


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
    """One progress line (``title`` is left for the caller)."""
    whole = float(before).is_integer() and float(now).is_integer()  # both numbers in the same form
    a, b = format_value(metric, before, whole), format_value(metric, now, whole)
    d = direction(metric)
    improved = None if d is None or a == b else (now - before) * d > 0
    if a.endswith(PER_100) and b.endswith(PER_100):  # the unit once: "−16 → −4 per 100 games"
        text = f"{_label(metric, True)}: {a[: -len(PER_100)]} → {b}"
    else:
        text = f"{_label(metric)}: {a} → {b}"
    return ProgressItem(title="", metric=metric, before=before, now=now, text=text, improved=improved)


def annotate(report: Report, cfg: CoachConfig) -> None:
    """Fill ``report.coaching.progress`` from ``cfg.previous`` (the last report's JSON)."""
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
    items: list[ProgressItem] = []
    for item in report.study_plan:
        metric = str((item.baseline or {}).get("metric") or "")
        now = _num((item.baseline or {}).get("value"))
        if not metric or now is None or _key(metric) not in before:
            continue
        line = compare(metric, before[_key(metric)], now)
        line.title = item.title
        items.append(line)
    coaching.progress = items
    since = str(previous.get("generated_at") or "")[:10]
    if items and since:
        coaching.settings["progress_since"] = since
