"""Self-contained HTML report.

One file with inline CSS, hand-written SVG charts and a tiny optional tooltip
script: it opens offline, works at phone width, follows the viewer's light/dark
theme, and can be published as an Artifact fragment (``standalone=False``).

Everything that comes from data is escaped, and only http(s) URLs ever become
links. The value-formatting helpers at the top are shared with the Markdown
renderer, so both outputs show identical numbers.
"""

from __future__ import annotations

import html as _html
import itertools
import math
import numbers
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Optional, Sequence
from urllib.parse import urlsplit
from xml.etree import ElementTree

from ..insights import STUDY_LIBRARY
from ..models import VALUE_FORMATS, Chart, Diagram, Insight, Kpi, ModuleResult, Report, StudyItem, Table

try:  # numpy comes with pandas; it is only needed to recognise its time scalars
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

# =========================================================================== value formatting
MISSING = "—"  # em dash for None / NaN
MINUS = "−"  # real minus sign
NUMERIC_FORMATS = frozenset(f for f in VALUE_FORMATS if f not in ("text", "url"))
_PCT_FORMATS = frozenset({"pct", "signed_pct"})
_SIGNED_FORMATS = frozenset({"signed_pct", "signed_int"})
_URL_RE = re.compile(r"^https?://", re.I)
HUGE = 1e15  # beyond this, numbers are shown in scientific notation (1.5e+18) instead of 19 grouped digits


def _np_time(value: Any) -> Optional[str]:
    """"timedelta64" / "datetime64" for numpy time scalars (timedelta64 even passes as an integer), else None."""
    if _np is None:
        return None
    if isinstance(value, _np.timedelta64):
        return "timedelta64"
    if isinstance(value, _np.datetime64):
        return "datetime64"
    return None


def _duration_seconds(value: Any) -> Optional[float]:
    """Seconds in a timedelta / numpy timedelta64; None for NaT and calendar units (months, years)."""
    if isinstance(value, timedelta):
        return value.total_seconds()
    try:
        x = float(value / _np.timedelta64(1, "s"))
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None
    return x if math.isfinite(x) else None


def _np_datetime(value: Any) -> Any:
    """A numpy datetime64 as a Python datetime when it is in range, else its ISO text."""
    try:
        out = value.astype("datetime64[us]").item()
    except (ValueError, OverflowError, TypeError):
        out = None
    return out if isinstance(out, datetime) else str(_np.datetime_as_string(value))


def is_missing(value: Any) -> bool:
    """None, NaN, +-inf, pandas NA/NaT, numpy NaT, Decimal NaN/inf and blank strings all render as a dash."""
    if value is None:
        return True
    if type(value).__name__ in ("NAType", "NaTType"):
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Decimal):
        return not value.is_finite()
    if _np_time(value):
        return bool(_np.isnat(value))
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            return not math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def to_number(value: Any) -> Optional[float]:
    """Finite float for numbers, Decimals, durations (in seconds) and numeric strings; None otherwise
    (bools and dates are not numbers here)."""
    if isinstance(value, bool) or is_missing(value):
        return None
    kind = _np_time(value)
    if kind == "datetime64":
        return None
    if kind == "timedelta64" or isinstance(value, timedelta):
        x = _duration_seconds(value)
        if x is None:
            return None
    elif isinstance(value, Decimal):
        x = float(value)  # finite: is_missing() rejected NaN / sNaN / inf
    elif isinstance(value, numbers.Real):
        try:
            x = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(value, str):
        try:
            x = float(value.strip().replace(",", "").replace(MINUS, "-"))
        except ValueError:
            return None
    else:
        return None
    return x if math.isfinite(x) else None


def infer_format(value: Any) -> str:
    """Best-guess VALUE_FORMATS entry for a value whose column has no declared format."""
    if isinstance(value, bool):
        return "text"
    kind = _np_time(value)
    if kind == "timedelta64" or isinstance(value, timedelta):
        return "seconds"
    if kind == "datetime64":
        return "text"
    if isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, (numbers.Real, Decimal)):
        x = to_number(value)
        return "int" if x is not None and x.is_integer() else "float2"
    if isinstance(value, str) and _URL_RE.match(value.strip()):
        return "url"
    return "text"


def _quantize(x: float | Decimal, digits: int) -> Decimal:
    """Round half up (7.25 -> 7.3), exact in decimal so 0.535 does not become 0.53."""
    d = x if isinstance(x, Decimal) else Decimal(repr(float(x)))
    try:
        return d.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    except InvalidOperation:  # beyond Decimal precision: astronomically large values
        return d


def _number_text(d: Decimal, digits: int, *, grouping: bool = True, signed: bool = False) -> str:
    if abs(d) >= HUGE:
        body = format(float(abs(d)), ".3g")
    else:
        body = format(abs(d), f"{',' if grouping else ''}.{digits}f")
    if d < 0:
        return MINUS + body
    if signed and d > 0:
        return "+" + body
    return body


def _percent(x: float, *, signed: bool) -> str:
    scaled = Decimal(repr(x)) * 100
    if signed:  # whole percent; one decimal only when it would otherwise read as 0
        digits = 1 if x != 0 and _quantize(scaled, 0) == 0 else 0
    else:  # one decimal below 10%
        digits = 1 if x != 0 and abs(x) < 0.1 else 0
    d = _quantize(scaled, digits)
    if digits and abs(d) >= 10:  # 9.96% rounds up to 10.0%: whole percent like every other value >= 10%
        digits, d = 0, _quantize(scaled, 0)
    if d == 0:
        # a real but tiny rate (a blunder rate of 0.04%) is not "0%"
        return "<0.1%" if x > 0 and not signed else "0%"
    return _number_text(d, digits, signed=signed) + "%"


def _clock(seconds: float) -> str:
    """Whole seconds as m:ss or h:mm:ss (sign kept)."""
    sign = MINUS if seconds < 0 else ""
    if abs(seconds) >= HUGE:
        return sign + format(abs(seconds), ".3g") + "s"
    total = int(_quantize(abs(seconds), 0))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{sign}{hours}:{minutes:02d}:{secs:02d}" if hours else f"{sign}{minutes}:{secs:02d}"


def _seconds(x: float) -> str:
    tenths = _quantize(abs(x), 1)
    if tenths < 60:
        return f"{MINUS if x < 0 and tenths else ''}{tenths:.1f}s"
    return _clock(x)


def _plain_number(x: float) -> str:
    """A float as plain text: up to 4 decimals, no trailing zeros, scientific notation when huge,
    a real minus sign, and never "-0"."""
    a = abs(x)
    if a >= HUGE:
        body = format(a, ".3g")
    elif a.is_integer():
        body = str(int(a))
    else:
        body = format(round(a, 4), "f").rstrip("0").rstrip(".") or "0"
    return MINUS + body if x < 0 and body != "0" else body


def _plain_text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    kind = _np_time(value)
    if kind == "datetime64":
        value = MISSING if is_missing(value) else _np_datetime(value)
    elif kind == "timedelta64" or isinstance(value, timedelta):
        if is_missing(value):
            return MISSING
        seconds = _duration_seconds(value)
        return str(value) if seconds is None else _seconds(seconds)
    if isinstance(value, datetime):
        if (value.hour, value.minute, value.second) == (0, 0, 0):
            return f"{value:%Y-%m-%d}"
        return f"{value:%Y-%m-%d %H:%M}"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, numbers.Integral):
        n = int(value)
        return (MINUS if n < 0 else "") + str(abs(n))  # exact digits: an id stays an id
    if isinstance(value, (numbers.Real, Decimal)):
        x = to_number(value)
        return MISSING if x is None else _plain_number(x)
    if isinstance(value, (list, tuple, set, frozenset)):
        return ", ".join(_plain_text(v) for v in value)
    return str(value)


def format_value(value: Any, fmt: Optional[str] = None) -> str:
    """Plain-text rendering of one value in a ``models.VALUE_FORMATS`` format.

    ``fmt=None`` (or an unknown format) infers one from the value. Missing values
    give an em dash; negative numbers use a real minus sign. ``url`` returns the
    URL itself: turning it into a link is the renderer's job.
    """
    if is_missing(value):
        return MISSING
    if fmt not in VALUE_FORMATS:
        fmt = infer_format(value)
    if fmt == "url":
        return str(value).strip()
    if fmt == "text":
        return _plain_text(value)
    x = to_number(value)
    if x is None:  # text in a numeric column: show it as it is
        return _plain_text(value)
    if fmt == "int":
        return _number_text(_quantize(x, 0), 0)
    if fmt == "rating":
        return _number_text(_quantize(x, 0), 0, grouping=False)
    if fmt == "signed_int":
        return _number_text(_quantize(x, 0), 0, signed=True)
    if fmt == "float1":
        return _number_text(_quantize(x, 1), 1)
    if fmt == "float2":
        return _number_text(_quantize(x, 2), 2)
    if fmt in _PCT_FORMATS:
        return _percent(x, signed=fmt == "signed_pct")
    return _seconds(x)


def is_numeric_format(fmt: Optional[str], sample: Iterable[Any] = ()) -> bool:
    """Whether a column should be right-aligned: declared numeric, or (undeclared) all values numeric."""
    if fmt in NUMERIC_FORMATS:
        return True
    if fmt in VALUE_FORMATS:
        return False
    present = [v for v in sample if not is_missing(v)]
    return bool(present) and all(infer_format(v) in NUMERIC_FORMATS for v in present)


def safe_url(value: Any) -> Optional[str]:
    """The URL if it is a plain http(s) URL that is safe to put in an href, else None."""
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url or len(url) > 2048 or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    return url


def url_link_text(url: str) -> str:
    """Short link text for a URL table cell: "game" for chess.com games, else "view"."""
    lowered = url.lower()
    return "game" if "chess.com" in lowered and "/game/" in lowered else "view"


# =========================================================================== shared report wording
KIND_LABELS = {"strength": "Strength", "weakness": "Weakness", "observation": "Observation"}
GLYPH_MEANINGS = {
    "??": "Serious weakness",
    "?": "Weakness",
    "?!": "Minor weakness",
    "!!": "Major strength",
    "!": "Strength",
    "=": "Observation",
}
METHOD_NOTES = (
    "Every score is compared with the Elo expected score for the rating gap in each game, not with 50%. "
    "Beating lower-rated players is expected, so it is not counted as a strength.",
    "Strengths and weaknesses need a minimum number of games (about 8 for an opening with one colour, "
    "20 for splits such as games after a loss). Smaller samples appear as observations or are left out.",
    "Priority is how big the effect is multiplied by how sure we are of it. The confidence label reflects "
    "the number of games and how consistent the pattern is.",
    "chess.com accuracy (Game Review) and this report's engine accuracy use different formulas. "
    "Compare each only with itself.",
)


def insight_kind(ins: Insight) -> str:
    return ins.kind if ins.kind in KIND_LABELS else "observation"


def _priority(ins: Insight) -> float:
    try:
        p = float(ins.priority)
    except (TypeError, ValueError):
        return 0.0
    return p if math.isfinite(p) else 0.0


def glyph_for(ins: Insight) -> str:
    """Chess annotation glyph for an insight: ??, ?, ?! for weaknesses, !!, ! for strengths, = otherwise."""
    kind, p = insight_kind(ins), _priority(ins)
    if kind == "weakness":
        return "??" if p >= 0.5 else "?" if p >= 0.25 else "?!"
    if kind == "strength":
        return "!!" if p >= 0.5 else "!"
    return "="


def confidence_label(confidence: Any) -> str:
    c = to_number(confidence) or 0.0
    return "High confidence" if c >= 0.7 else "Medium confidence" if c >= 0.4 else "Low confidence"


def category_label(category: Any) -> str:
    lib = STUDY_LIBRARY.get(str(category)) if category else None
    if lib and lib.get("label"):
        return str(lib["label"])
    text = str(category or "").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "General"


def format_date(dt: Any) -> str:
    if not isinstance(dt, date):
        return ""
    if isinstance(dt, datetime) and dt.tzinfo:
        dt = dt.astimezone(timezone.utc)
    return f"{dt.day} {dt:%b %Y}"


def date_range_text(start: Any, end: Any) -> str:
    a, b = format_date(start), format_date(end)
    if a and b and a != b:
        return f"{a} – {b}"
    return a or b


def format_generated(dt: Any) -> str:
    if not isinstance(dt, datetime):
        return format_date(dt)
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc)
        return f"{dt.day} {dt:%b %Y}, {dt:%H:%M} UTC"
    return f"{dt.day} {dt:%b %Y}, {dt:%H:%M}"


def games_text(n: Any) -> str:
    count = int(to_number(n) or 0)
    return f"{count:,} game" + ("" if count == 1 else "s")


def report_title(report: Report) -> str:
    name = str(report.username or "").strip()
    return f"{name} · Chess Insights" if name else "Chess Insights"


def text_or_empty(value: Any) -> str:
    return "" if value is None else str(value).strip()


def plan_titles(report: Report) -> set[str]:
    """Titles already covered by the study plan (their study steps are shown there)."""
    return {text_or_empty(item.title) for item in report.study_plan or []}


def valid_urls(urls: Iterable[Any], limit: int = 5) -> list[str]:
    out = []
    for url in urls or []:
        safe = safe_url(url)
        if safe and safe not in out:
            out.append(safe)
        if len(out) >= limit:
            break
    return out


# =========================================================================== chart data
@dataclass
class ChartData:
    """A chart normalised for drawing: every series has exactly one finite value or None per label."""

    labels: list[str]
    series: list[tuple[str, list[Optional[float]]]]
    fmt: Optional[str]  # a numeric VALUE_FORMAT, or None to infer per value
    reference: Optional[float]  # never 0: a reference at zero is the emphasised zero baseline instead
    emphasise_zero: bool = False

    @property
    def has_values(self) -> bool:
        return bool(self.labels) and any(v is not None for _, vals in self.series for v in vals)


def chart_data(chart: Chart) -> ChartData:
    series_in = [s for s in (chart.series or []) if s is not None]
    labels = ["" if is_missing(label) else _plain_text(label) for label in (chart.labels or [])]
    n = max([len(labels)] + [len(s.values or []) for s in series_in])
    # a series longer than the labels keeps its extra values: they are labelled by position, never dropped
    labels += [str(i + 1) for i in range(len(labels), n)]
    series = []
    for i, s in enumerate(series_in):
        values = list(s.values or [])[:n]
        values += [None] * (n - len(values))
        name = _plain_text(s.name) if not is_missing(s.name) else ("Value" if len(series_in) == 1 else f"Series {i + 1}")
        series.append((name, [to_number(v) for v in values]))
    roles = result_roles([name for name, _ in series])
    if roles:  # wins, then draws, then losses (stable within each role)
        order = sorted(range(len(series)), key=lambda k: _ROLE_ORDER[roles[k]])
        series = [series[k] for k in order]
    fmt = chart.value_format if chart.value_format in NUMERIC_FORMATS else None
    reference = to_number(chart.reference)
    if reference == 0:  # the zero line is already drawn: emphasise it rather than dashing over it
        return ChartData(labels, series, fmt, None, emphasise_zero=True)
    return ChartData(labels, series, fmt, reference)


# ---- series colours: by identity, never by position -------------------------------------------
# Results (wins / draws / losses) get a diverging scheme: a green arm, a neutral grey draw and a
# red arm, strongest at the ends ("Won by checkmate", "Lost by checkmate"), lighter towards the
# draws. Up to RESULT_STEPS series per arm; anything else uses the categorical slots 1-8.
RESULT_STEPS = 4
# Result colours from different arms (or an arm and the draw grey) that are too alike to touch:
# normal-vision OKLab dE < 15 or colour-blind (min protan/deutan, Machado 2009) dE < 6 in the light or
# dark theme. Steps within one arm are a lightness ramp and may touch. test_report recomputes this.
_RESULT_CLASH = frozenset(
    frozenset(pair)
    for pair in (
        ("w2", "d"), ("w3", "d"), ("d", "l3"), ("w1", "l2"), ("w2", "l3"),
        ("w3", "l1"), ("w3", "l3"), ("w3", "l4"), ("w4", "l2"), ("w4", "l4"),
    )
)
_ROLE_ORDER = {"win": 0, "draw": 1, "loss": 2}
_ROLE_RES = (
    ("win", re.compile(r"^(?:wins?|won|victor(?:y|ies))\b", re.I)),
    ("draw", re.compile(r"^(?:draws?|drawn)\b", re.I)),
    ("loss", re.compile(r"^(?:loss(?:es)?|lost|lose|defeats?)\b", re.I)),
)
# Well-known series that recur across charts and reports keep one colour wherever they appear,
# so a filter that removes bullet does not repaint blitz. Slots 2 (orange) and 4 (yellow) are too
# alike to share a chart (dE 13.7 light; colour-blind dE 4.8 dark), so they go to the pair least
# often played together, bullet and daily; every other pair and trio clears the floors.
IDENTITY_SLOTS = {"blitz": "1", "bullet": "2", "rapid": "3", "daily": "4"}
_CLASHING_SLOTS = frozenset({frozenset({"2", "4"})})  # among slots 1-4; test_report recomputes this


def result_roles(names: Sequence[str]) -> Optional[list[str]]:
    """["win", "draw", "loss", ...] when every series is a result series (and there are 2+ roles), else None."""
    roles = []
    for name in names:
        role = next((r for r, rx in _ROLE_RES if rx.match(str(name).strip())), None)
        if role is None:
            return None
        roles.append(role)
    if len(set(roles)) < 2 or roles.count("draw") > 1:
        return None
    if roles.count("win") > RESULT_STEPS or roles.count("loss") > RESULT_STEPS:
        return None
    return roles


def _arm_options(k: int) -> list[tuple[int, ...]]:
    """Increasing colour steps for an arm of ``k`` result series, preferred first: widest spacing, then strongest."""
    combos = list(itertools.combinations(range(1, RESULT_STEPS + 1), k))
    return sorted(combos, key=lambda c: (-min((b - a for a, b in zip(c, c[1:])), default=0), c))


def _result_slots(roles: Sequence[str], stacks: Optional[Sequence[Sequence[int]]]) -> list[str]:
    """Result colour per series. The series drawn anywhere share out the arm's steps (strongest
    first), choosing the most preferred steps for which no two touching marks of different roles
    clash. Marks touch when they are neighbours in a stack (``stacks``: the series indices of each
    drawn stack, bottom up) or, without stacks, neighbours in series order."""
    if stacks is None:
        stacks = [list(range(len(roles)))]
    drawn = {j for stack in stacks for j in stack}
    touching = {(a, b) for stack in stacks for a, b in zip(stack, stack[1:])}
    win_idx = [j for j, r in enumerate(roles) if r == "win" and j in drawn]
    loss_idx = [j for j, r in enumerate(roles) if r == "loss" and j in drawn]  # lightest (next to the draws) first
    wins, losses = _arm_options(len(win_idx)), _arm_options(len(loss_idx))

    def assign(w: tuple[int, ...], lo: tuple[int, ...]) -> list[str]:
        # a series with nothing to draw (all zero) is left out of the legend: any colour of its role
        slots = [{"win": "w1", "draw": "d", "loss": "l1"}[r] for r in roles]
        for j, k in zip(win_idx, w):
            slots[j] = f"w{k}"
        for j, k in zip(loss_idx, reversed(lo)):
            slots[j] = f"l{k}"
        return slots

    for _, w, lo in sorted(
        ((wi + li, wi), w, lo) for (wi, w), (li, lo) in itertools.product(enumerate(wins), enumerate(losses))
    ):
        slots = assign(w, lo)
        if not any(slots[a][0] != slots[b][0] and frozenset((slots[a], slots[b])) in _RESULT_CLASH for a, b in touching):
            return slots
    # e.g. 4 wins stacked directly on 4 losses: no clash-free choice, keep the full ramps
    return assign(wins[0], losses[0])


def _slots_clash(slots: Sequence[str]) -> bool:
    return any(frozenset(pair) in _CLASHING_SLOTS for pair in itertools.combinations(slots, 2))


def series_slots(names: Sequence[str], stacks: Optional[Sequence[Sequence[int]]] = None) -> list[str]:
    """Colour slot per series: "w1".."w4" / "d" / "l1".."l4" for results (wins run strongest ->
    lightest towards the draws, losses lightest -> strongest away from them; ``stacks`` as in
    :func:`_result_slots`), fixed slots for known series (time classes), else "1".."8" in order."""
    roles = result_roles(names)
    if roles:
        return _result_slots(roles, stacks)
    # in order, counting only series that are drawn (all-zero stacked series take no colour)
    drawn = sorted({j for stack in stacks or [] for j in stack}) or list(range(len(names)))
    positional = ["1"] * len(names)
    for k, j in enumerate(drawn):
        positional[j] = str(k % MAX_SERIES + 1)
    known = [IDENTITY_SLOTS.get(str(name).strip().lower()) for name in names]
    if names and all(known) and len(set(known)) == len(known):
        if not _slots_clash(known) or _slots_clash(positional):  # e.g. bullet + daily alone: in order instead
            return [str(k) for k in known]
    return positional


def _cls(prefix: str, slot: str) -> str:
    """CSS class for a series slot: f1 / k1 / sw1 for categorical slots, f-w1 / sw-d for result slots."""
    return f"{prefix}{slot}" if slot.isdigit() else f"{prefix}-{slot}"


TABLE_MAX_SERIES_COLUMNS = 4


def chart_table(chart: Chart) -> Table:
    """The chart's numbers as a table: the accessible twin of every chart (also used by Markdown).

    One row per category and one column per series; when there are more than
    ``TABLE_MAX_SERIES_COLUMNS`` series and fewer categories (9 result types over 3 time
    controls), rows and columns swap so the table stays narrow enough for a phone.
    """
    data = chart_data(chart)
    fmt = data.fmt or "auto"
    names = [name for name, _ in data.series]
    if len(names) > TABLE_MAX_SERIES_COLUMNS and len(data.labels) < len(names):
        return Table(
            title=text_or_empty(chart.title),
            columns=[""] + list(data.labels),
            rows=[[name] + list(vals) for name, vals in data.series],
            formats=["text"] + [fmt] * len(data.labels),
        )
    return Table(
        title=text_or_empty(chart.title),
        columns=[""] + names,
        rows=[[label] + [vals[i] for _, vals in data.series] for i, label in enumerate(data.labels)],
        formats=["text"] + [fmt] * len(names),
    )


# =========================================================================== SVG charts
VB_W = 400.0  # viewBox width; charts scale to their container
PLOT_H = 170.0
LABEL_PX = 13.0
TICK_PX = 12.0
# Glyph advances in em for fitting labels: the wider of DejaVu Sans (what system-ui resolves to
# on Linux) and Liberation Sans/Arial, so a label that fits here also fits in IBM Plex Sans,
# San Francisco and Roboto. An average-width guess clipped "−20%" and "+10%" at chart edges.
_ADVANCE = {
    ch: w
    for chars, w in (
        ("'ijl", 0.28), ("I", 0.3), (" ,.", 0.32), ("/:;\\|", 0.34), ("-f", 0.36), ("()[]", 0.39), ("t", 0.4),
        ("!r", 0.41), ('"', 0.46), ("*J`", 0.5), ("sz", 0.53), ("c", 0.55), ("?L_", 0.56), ("k", 0.58), ("vxy", 0.6),
        ("FT", 0.61), ("aeo", 0.62), ("$0123456789bdghnpqu{}", 0.64), ("EKPSY", 0.67), ("ABVXZ", 0.69), ("CR", 0.73),
        ("U", 0.74), ("N", 0.75), ("H", 0.76), ("D", 0.77), ("&G", 0.78), ("OQ", 0.79), ("w", 0.82),
        ("#+<=>^~−±≤≥×↗", 0.84), ("M", 0.87), ("%", 0.95), ("m", 0.98), ("W", 0.99), ("@—…→", 1.02),
        ("–", 0.56), ("·’", 0.34), ("“”", 0.52),
    )
    for ch in chars
}
BAR_MAX = 20.0  # <= 24 CSS px at typical desktop scale
GAP = 2.0  # surface gap between touching marks
RADIUS = 4.0
MAX_SERIES = 8
_ROT = math.radians(40)
_SECONDS_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 43200, 86400)


def _esc(value: Any) -> str:
    return _html.escape(text_or_empty(value), quote=True)


def _n(x: float) -> str:
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _advance(ch: str) -> float:
    w = _ADVANCE.get(ch)
    if w is not None:
        return w
    if unicodedata.combining(ch):
        return 0.0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 1.0
    return 0.84 if 0x2000 <= ord(ch) < 0x2E80 else 0.7  # symbols / other letters


def _tw(text: str, size: float = LABEL_PX) -> float:
    """Estimated rendered width of ``text`` at ``size`` (viewBox units == px at 1:1)."""
    return size * sum(_advance(ch) for ch in text)


def _fit(text: str, max_w: float, size: float) -> str:
    """The longest prefix of ``text`` that fits in ``max_w`` (at least one character)."""
    acc, out = 0.0, []
    for ch in text:
        acc += size * _advance(ch)
        if acc > max_w and out:
            break
        out.append(ch)
    return "".join(out)


def _ellipsize(text: str, max_w: float, size: float = LABEL_PX) -> str:
    if _tw(text, size) <= max_w:
        return text
    return _fit(text, max_w - _tw("…", size), size).rstrip() + "…"


def _shorten(text: str, max_w: float, size: float = LABEL_PX) -> str:
    """Cut from the middle, keeping the last words: labels that differ only at the end
    ("... Attack 3 (Black)" / "... Attack 4 (White)") stay distinguishable. The full label is in the tooltip."""
    if _tw(text, size) <= max_w:
        return text
    words = text.split()
    tail: list[str] = []
    while len(tail) < len(words) - 1 and _tw("… " + " ".join([words[-len(tail) - 1]] + tail), size) <= max_w * 0.6:
        tail.insert(0, words[-len(tail) - 1])
    if not tail:
        return _ellipsize(text, max_w, size)
    end = "… " + " ".join(tail)
    head = " ".join(words[: len(words) - len(tail)])
    return _fit(head, max_w - _tw(end, size), size).rstrip() + end


def _wrap(text: str, max_w: float, size: float = LABEL_PX, max_lines: int = 2) -> list[str]:
    """Greedy word wrap into at most ``max_lines`` lines; if needed, the last line is cut in the
    middle so it keeps the label's last words (see :func:`_shorten`)."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        while _tw(word, size) > max_w and len(word) > 1:
            if cur:
                lines.append(cur)
                cur = ""
            head = _fit(word, max_w, size)
            lines.append(head)
            word = word[len(head) :]
        candidate = f"{cur} {word}" if cur else word
        if _tw(candidate, size) <= max_w:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        return [""]
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [_shorten(" ".join(lines[max_lines - 1 :]), max_w, size)]
    return lines


def _wrap_exact(text: str, max_w: float, size: float = LABEL_PX, max_lines: int = 4) -> Optional[list[str]]:
    """Word wrap that never cuts: lines break at spaces or after a dash/slash ("00:00–" / "04:00").
    None when the text needs more than ``max_lines`` lines or a single piece is wider than ``max_w``."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        for k, piece in enumerate(re.split(r"(?<=[–—/])(?=.)", word)):
            if _tw(piece, size) > max_w:
                return None
            candidate = cur + ("" if k else " ") + piece if cur else piece
            if _tw(candidate, size) <= max_w:
                cur = candidate
            else:
                lines.append(cur)
                cur = piece
    if cur:
        lines.append(cur)
    return lines if 0 < len(lines) <= max_lines else None


def _nice_step(raw: float, integer: bool) -> float:
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if integer and not float(step).is_integer():
            continue
        if raw <= step * (1 + 1e-9):
            return step
    return 10 * mag


def nice_ticks(lo: float, hi: float, fmt: Optional[str] = None, target: int = 5) -> list[float]:
    """Round axis ticks covering [lo, hi]: steps of 1/2/2.5/5 x 10^k (clock-friendly for seconds)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if lo > hi:
        lo, hi = hi, lo
    if hi - lo < 1e-12:
        if lo == 0:
            lo, hi = 0.0, (4.0 if fmt in ("int", "rating", "signed_int") else 1.0)
        else:
            pad = abs(lo) * 0.1
            lo, hi = lo - pad, hi + pad
    integer = fmt in ("int", "rating", "signed_int")
    raw = (hi - lo) / max(target - 1, 1)
    if fmt == "seconds" and raw >= 20:
        step = float(next((s for s in _SECONDS_STEPS if s >= raw), _nice_step(raw, True)))
    else:
        step = _nice_step(raw, integer)
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    count = max(1, int(round((end - start) / step)))
    return [round(start + i * step, 10) for i in range(count + 1)]


def _decimals(step: float) -> int:
    for d in range(4):
        if abs(round(step, d) - step) < 1e-9 * max(1.0, abs(step)):
            return d
    return 4


def _tick_text(v: float, fmt: Optional[str], step: float, span: float) -> str:
    if fmt in _PCT_FORMATS:
        d = _decimals(step * 100)
        q = _quantize(v * 100, d)
        return "0%" if q == 0 else _number_text(q, d, signed=fmt == "signed_pct") + "%"
    if fmt == "seconds":
        if span >= 60 and step >= 1:
            return _clock(v)
        d = _decimals(step)
        q = _quantize(v, d)
        return ("0" if q == 0 else _number_text(q, d)) + "s"
    d = _decimals(step)
    q = _quantize(v, d)
    return _number_text(q, d, grouping=fmt != "rating", signed=fmt == "signed_int")


def _linear(d0: float, d1: float, r0: float, r1: float) -> Callable[[float], float]:
    if d1 == d0:
        return lambda v: (r0 + r1) / 2
    k = (r1 - r0) / (d1 - d0)
    return lambda v: r0 + (v - d0) * k


def _axis_ticks(values: Sequence[float], fmt: Optional[str], *, include_zero: bool, extra: Optional[float], target: int = 5):
    pts = list(values) + ([extra] if extra is not None else [])
    lo, hi = (min(pts), max(pts)) if pts else (0.0, 1.0)
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    ticks = nice_ticks(lo, hi, fmt, target=target)
    step = ticks[1] - ticks[0]
    span = max(abs(ticks[0]), abs(ticks[-1]))
    return ticks, [_tick_text(t, fmt, step, span) for t in ticks]


def _vbar(x: float, w: float, y_base: float, y_end: float, rounded: bool = True) -> str:
    """Column path: square at the baseline, 4px rounded corners at the data end."""
    h = abs(y_base - y_end)
    if h < 0.75:  # a real zero: hairline stub, so it reads differently from a gap
        return f"M{_n(x)},{_n(y_base - 0.75)}h{_n(w)}v1.5h{_n(-w)}Z"
    r = min(RADIUS, h, w / 2) if rounded else 0.0
    if y_end < y_base:
        t = y_end
        return (
            f"M{_n(x)},{_n(y_base)}V{_n(t + r)}A{_n(r)},{_n(r)} 0 0 1 {_n(x + r)},{_n(t)}"
            f"H{_n(x + w - r)}A{_n(r)},{_n(r)} 0 0 1 {_n(x + w)},{_n(t + r)}V{_n(y_base)}Z"
        )
    b = y_end
    return (
        f"M{_n(x)},{_n(y_base)}V{_n(b - r)}A{_n(r)},{_n(r)} 0 0 0 {_n(x + r)},{_n(b)}"
        f"H{_n(x + w - r)}A{_n(r)},{_n(r)} 0 0 0 {_n(x + w)},{_n(b - r)}V{_n(y_base)}Z"
    )


def _hbar(y: float, h: float, x_base: float, x_end: float) -> str:
    """Horizontal bar path: square at the baseline, rounded at the data end."""
    w = abs(x_end - x_base)
    if w < 0.75:
        return f"M{_n(x_base - 0.75)},{_n(y)}h1.5v{_n(h)}h-1.5Z"
    r = min(RADIUS, w, h / 2)
    if x_end > x_base:
        e = x_end
        return (
            f"M{_n(x_base)},{_n(y)}H{_n(e - r)}A{_n(r)},{_n(r)} 0 0 1 {_n(e)},{_n(y + r)}"
            f"V{_n(y + h - r)}A{_n(r)},{_n(r)} 0 0 1 {_n(e - r)},{_n(y + h)}H{_n(x_base)}Z"
        )
    e = x_end
    return (
        f"M{_n(x_base)},{_n(y)}H{_n(e + r)}A{_n(r)},{_n(r)} 0 0 0 {_n(e)},{_n(y + r)}"
        f"V{_n(y + h - r)}A{_n(r)},{_n(r)} 0 0 0 {_n(e + r)},{_n(y + h)}H{_n(x_base)}Z"
    )


def _tip(label: str, items: Sequence[tuple[str, Optional[float]]], fmt: Optional[str]) -> str:
    """Tooltip text: category first, then one "series: value" line per series."""
    return "\n".join([label] + [f"{name}: {format_value(v, fmt)}" for name, v in items])


def _svg_open(height: float, title: str) -> str:
    return (
        f'<svg viewBox="0 0 {_n(VB_W)} {_n(height)}" role="img" aria-label="{_esc(title or "Chart")}" '
        f'focusable="false" preserveAspectRatio="xMidYMid meet">'
    )


def _hline(cls: str, x1: float, x2: float, y: float) -> str:
    return f'<line class="{cls}" x1="{_n(x1)}" x2="{_n(x2)}" y1="{_n(y)}" y2="{_n(y)}"/>'


def _vline(cls: str, x: float, y1: float, y2: float) -> str:
    return f'<line class="{cls}" x1="{_n(x)}" x2="{_n(x)}" y1="{_n(y1)}" y2="{_n(y2)}"/>'


def reference_text(ref: float, fmt: Optional[str]) -> str:
    return f"Reference {format_value(ref, fmt)}"


def _zero_class(data: ChartData, values: Sequence[float]) -> str:
    return "zero" if data.emphasise_zero or data.fmt in _SIGNED_FORMATS or any(v < 0 for v in values) else "base"


def _svg_columns(
    data: ChartData,
    series: list[tuple[str, list[Optional[float]]]],
    slots: Sequence[str],
    title: str,
    stacked: bool,
    allow_horizontal: bool = False,
) -> Optional[str]:
    """Vertical bars: grouped when there are several series, or stacked.

    With ``allow_horizontal``, returns None instead of rotating long category labels: the caller
    then draws horizontal bars, where labels have room to wrap.
    """
    fmt, labels, n, ref = data.fmt, data.labels, len(data.labels), data.reference
    if stacked:
        extent = []
        for i in range(n):
            col = [vals[i] for _, vals in series if vals[i] is not None]
            extent += [sum(v for v in col if v > 0), sum(v for v in col if v < 0)]
    else:
        extent = [v for _, vals in series for v in vals if v is not None]
    ticks, tick_txt = _axis_ticks(extent, fmt, include_zero=True, extra=ref)
    has_neg = any(v < 0 for v in extent)

    left = max(_tw(t, TICK_PX) for t in tick_txt) + 10
    right = 6.0
    band = (VB_W - left - right) / n
    widest = max(_tw(label) for label in labels)
    shown = labels
    thin = 1
    rotate = widest > band - 4
    line_h = LABEL_PX * 1.2
    multiline: Optional[list[list[str]]] = None
    if rotate and n >= 10 and widest <= 60:  # e.g. hours 00-23: every k-th label, upright
        thin, rotate = math.ceil((widest + 8) / band), False
    if rotate:  # upright on up to four lines reads better than 40° text on a phone, if nothing is cut
        wrapped = [_wrap_exact(label, band - 6) for label in labels]
        if all(w is not None for w in wrapped):
            multiline, rotate = wrapped, False  # type: ignore[assignment]
    if rotate and allow_horizontal:
        return None
    if rotate:
        shown = [_shorten(label, 175) for label in labels]
        max_w = max(_tw(t) for t in shown)
        # the first label runs down-left from its band centre; its glyph height adds LABEL_PX * sin
        left = max(left, max_w * math.cos(_ROT) + LABEL_PX * math.sin(_ROT) - band / 2 + 2)
        band = (VB_W - left - right) / n
        thin = max(1, math.ceil(LABEL_PX * 1.3 / band))
        x_band = 14 + max_w * math.sin(_ROT) + LABEL_PX * math.cos(_ROT)
    elif multiline:
        x_band = 12 + max(len(lines) for lines in multiline) * line_h
    else:
        x_band = 26.0

    single = len(series) == 1 and not stacked
    cap_txt: list[str] = []
    if single and n <= 16:
        cap_txt = [format_value(v, fmt) if v is not None else "" for v in series[0][1]]
        if any(_tw(t, TICK_PX) > band - 2 for t in cap_txt):  # neighbouring bars of similar height would collide
            cap_txt = []
    neg_room = 16.0 if cap_txt and has_neg else 0.0
    top = 20.0
    height = top + PLOT_H + neg_room + x_band
    y = _linear(ticks[0], ticks[-1], top + PLOT_H, top)
    x_right = VB_W - right

    out = [_svg_open(height, title)]
    for t, txt in zip(ticks, tick_txt):
        if t != 0:
            out.append(_hline("grid", left, x_right, y(t)))
        out.append(f'<text class="t-tick" x="{_n(left - 8)}" y="{_n(y(t))}" dy=".32em" text-anchor="end">{_esc(txt)}</text>')

    base_y = y(0.0)
    for i in range(n):
        x0 = left + i * band
        items = [(name, vals[i]) for name, vals in series]
        out.append(f'<g class="mk"><title>{_esc(_tip(labels[i], items, fmt))}</title>')
        out.append(f'<rect class="hit" x="{_n(x0)}" y="{_n(top)}" width="{_n(band)}" height="{_n(PLOT_H)}"/>')
        if stacked:
            bw = min(BAR_MAX, band * 0.6)
            bx = x0 + (band - bw) / 2
            segs = [(j, v) for j, (_, vals) in enumerate(series) if (v := vals[i]) is not None and v != 0]
            last_pos = max((k for k, (_, v) in enumerate(segs) if v > 0), default=-1)
            last_neg = max((k for k, (_, v) in enumerate(segs) if v < 0), default=-1)
            pos_acc = neg_acc = 0.0
            first_pos = first_neg = True
            for k, (j, v) in enumerate(segs):
                if v > 0:
                    y_start, y_end = y(pos_acc) - (0 if first_pos else GAP), y(pos_acc + v)
                    pos_acc += v
                    first_pos, is_end = False, k == last_pos
                else:
                    y_start, y_end = y(neg_acc) + (0 if first_neg else GAP), y(neg_acc + v)
                    neg_acc += v
                    first_neg, is_end = False, k == last_neg
                if abs(y_start - y_end) >= 0.5 and (y_end < y_start) == (v > 0):
                    out.append(f'<path class="bar {_cls("f", slots[j])}" d="{_vbar(bx, bw, y_start, y_end, is_end)}"/>')
        else:
            s = len(series)
            gw = min(band * 0.72, s * BAR_MAX + (s - 1) * GAP)
            bw = (gw - (s - 1) * GAP) / s
            bx = x0 + (band - gw) / 2
            for j, (_, vals) in enumerate(series):
                v = vals[i]
                if v is None:
                    continue
                out.append(f'<path class="bar {_cls("f", slots[j])}" d="{_vbar(bx + j * (bw + GAP), bw, base_y, y(v))}"/>')
        out.append("</g>")
        if cap_txt and cap_txt[i]:
            v = series[0][1][i] or 0.0
            ty = y(v) - 6 if v >= 0 else y(v) + 15
            out.append(f'<text class="t-val" x="{_n(x0 + band / 2)}" y="{_n(ty)}" text-anchor="middle">{_esc(cap_txt[i])}</text>')

    out.append(_hline(_zero_class(data, extent), left, x_right, base_y))
    if ref is not None:
        out.append(_hline("ref", left, x_right, y(ref)))

    label_y = top + PLOT_H + neg_room + (12 if rotate else 18)
    for i, text in enumerate(shown):
        if i % thin:
            continue
        cx = left + (i + 0.5) * band
        if rotate:
            out.append(
                f'<text class="t-x" x="{_n(cx)}" y="{_n(label_y)}" dy=".32em" text-anchor="end" '
                f'transform="rotate(-40 {_n(cx)} {_n(label_y)})">{_esc(text)}</text>'
            )
        elif multiline:
            spans = "".join(
                f'<tspan class="t-x" x="{_n(cx)}" y="{_n(label_y + k * line_h)}">{_esc(line)}</tspan>'
                for k, line in enumerate(multiline[i])
            )
            out.append(f'<text class="t-x" text-anchor="middle">{spans}</text>')
        else:
            out.append(f'<text class="t-x" x="{_n(cx)}" y="{_n(label_y)}" text-anchor="middle">{_esc(text)}</text>')
    out.append("</svg>")
    return "".join(out)


def _tick_labels_fit(ticks: Sequence[float], texts: Sequence[str], x: Callable[[float], float], pad: float = 6.0) -> bool:
    """Whether centred x-axis tick labels leave at least ``pad`` between neighbours."""
    return all(
        x(b) - x(a) >= (_tw(ta, TICK_PX) + _tw(tb, TICK_PX)) / 2 + pad
        for (a, ta), (b, tb) in zip(zip(ticks, texts), zip(ticks[1:], texts[1:]))
    )


def _svg_hbar(data: ChartData, series: list[tuple[str, list[Optional[float]]]], slots: Sequence[str], title: str) -> str:
    """Horizontal bars, labels on the left (long opening names wrap to two lines)."""
    fmt, labels, n, ref = data.fmt, data.labels, len(data.labels), data.reference
    values = [v for _, vals in series for v in vals if v is not None]
    has_neg = any(v < 0 for v in values)
    single = len(series) == 1

    label_max = VB_W * 0.42
    wrapped = [_wrap(label, label_max) for label in labels]
    label_w = min(label_max, max((_tw(line) for lines in wrapped for line in lines), default=0.0))
    plot_l = label_w + 12

    tick_txt: list[str] = []
    ticks: list[float] = []
    val_txt: list[str] = []
    if single:
        vals = series[0][1]
        val_txt = [format_value(v, fmt) for v in vals]
        pos_w = max((_tw(t, TICK_PX) for t, v in zip(val_txt, vals) if v is None or v >= 0), default=0.0) + 8
        neg_w = max((_tw(t, TICK_PX) for t, v in zip(val_txt, vals) if v is not None and v < 0), default=0.0) + 8
        pts = values + [0.0] + ([ref] if ref is not None else [])
        lo, hi = min(pts), max(pts)
        if lo == hi:
            hi = lo + 1.0
        x_from = plot_l + (neg_w if has_neg else 0.0)
        x_to = max(VB_W - pos_w, x_from + 60)
    else:
        # x-axis tick labels sit side by side: fewer ticks when wide labels ("100,000") would collide
        for target in (5, 4, 3):
            ticks, tick_txt = _axis_ticks(values, fmt, include_zero=True, extra=ref, target=target)
            lo, hi = ticks[0], ticks[-1]
            x_from = max(plot_l + 4, _tw(tick_txt[0], TICK_PX) / 2 + 2)
            x_to = VB_W - max(_tw(tick_txt[-1], TICK_PX) / 2, 4.0)
            if _tick_labels_fit(ticks, tick_txt, _linear(lo, hi, x_from, x_to)):
                break
    x = _linear(lo, hi, x_from, x_to)

    s = len(series)
    bh = 16.0 if single else max(6.0, min(12.0, (30.0 - (s - 1) * GAP) / s))
    group = s * bh + (s - 1) * GAP
    line_h = LABEL_PX * 1.2
    pitch = max(group + 12, max(len(w) for w in wrapped) * line_h + 8)
    top = 8.0
    rows_bottom = top + n * pitch
    height = rows_bottom + (26.0 if not single else 6.0)

    out = [_svg_open(height, title)]
    every = 1  # still crowded at 3 ticks: label every other one (gridlines stay)
    while every < len(ticks) and not _tick_labels_fit(ticks[::every], tick_txt[::every], x):
        every += 1
    for k, (t, txt) in enumerate(zip(ticks, tick_txt)):
        if t != 0:
            out.append(_vline("grid", x(t), top, rows_bottom))
        if k % every == 0:
            out.append(f'<text class="t-tick" x="{_n(x(t))}" y="{_n(rows_bottom + 17)}" text-anchor="middle">{_esc(txt)}</text>')

    for i, label in enumerate(labels):
        row_y = top + i * pitch
        cy = row_y + pitch / 2
        items = [(name, vals[i]) for name, vals in series]
        out.append(f'<g class="mk"><title>{_esc(_tip(label, items, fmt))}</title>')
        out.append(f'<rect class="hit" x="0" y="{_n(row_y)}" width="{_n(VB_W)}" height="{_n(pitch)}"/>')
        lines = wrapped[i]
        first_y = cy - (len(lines) - 1) * line_h / 2
        spans = "".join(
            f'<tspan x="{_n(label_w)}" y="{_n(first_y + k * line_h)}" dy=".32em">{_esc(line)}</tspan>'
            for k, line in enumerate(lines)
        )
        out.append(f'<text class="t-y" text-anchor="end">{spans}</text>')
        for j, (_, vals) in enumerate(series):
            v = vals[i]
            if v is None:
                continue
            by = cy - group / 2 + j * (bh + GAP)
            out.append(f'<path class="bar {_cls("f", slots[j])}" d="{_hbar(by, bh, x(0.0), x(v))}"/>')
        out.append("</g>")
        if single:
            v = series[0][1][i]
            if v is not None and v < 0:
                out.append(
                    f'<text class="t-val" x="{_n(x(v) - 6)}" y="{_n(cy)}" dy=".32em" text-anchor="end">'
                    f"{_esc(val_txt[i])}</text>"
                )
            else:
                tx = x(v if v is not None else 0.0) + 6
                cls = "t-val" if v is not None else "t-tick"
                out.append(f'<text class="{cls}" x="{_n(tx)}" y="{_n(cy)}" dy=".32em">{_esc(val_txt[i])}</text>')

    out.append(_vline(_zero_class(data, values), x(0.0), top - 2, rows_bottom))
    if ref is not None:
        out.append(_vline("ref", x(ref), top - 4, rows_bottom))
    out.append("</svg>")
    return "".join(out)


_MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
# On a month axis, points up to this many calendar months apart are joined by the solid line (a
# month or two without games is normal); a longer break is drawn as a dashed bridge.
MONTH_LINE_MAX_STEP = 3


def month_positions(labels: Sequence[str]) -> Optional[list[int]]:
    """Calendar month numbers when every label is "YYYY-MM" in increasing order, else None."""
    out = []
    for label in labels:
        m = _MONTH_RE.match(label.strip())
        if not m:
            return None
        out.append(int(m.group(1)) * 12 + int(m.group(2)) - 1)
    if len(out) < 2 or any(b <= a for a, b in zip(out, out[1:])):
        return None
    return out


def _svg_line(data: ChartData, series: list[tuple[str, list[Optional[float]]]], slots: Sequence[str], title: str) -> str:
    """Lines over a category x-axis. "YYYY-MM" labels get a calendar scale, because monthly series
    only list months with games. On a month axis the line runs on across up to two months without
    games and breaks at longer ones; on other axes a missing value breaks it. A dashed bridge in the
    series colour keeps the series readable across each break."""
    fmt, labels, n, ref = data.fmt, data.labels, len(data.labels), data.reference
    values = [v for _, vals in series for v in vals if v is not None]
    include_zero = data.emphasise_zero or fmt in _SIGNED_FORMATS or (min(values) < 0 < max(values))
    ticks, tick_txt = _axis_ticks(values, fmt, include_zero=include_zero, extra=ref)
    left = max(_tw(t, TICK_PX) for t in tick_txt) + 10
    single = len(series) == 1
    last_idx = [max((i for i, v in enumerate(vals) if v is not None), default=-1) for _, vals in series]
    end_txt = format_value(series[0][1][last_idx[0]], fmt) if single and last_idx[0] >= 0 else ""
    right = _tw(end_txt, TICK_PX) + 16 if end_txt else 10.0
    x_from, x_to = left + 8, VB_W - right
    width = x_to - x_from
    months = month_positions(labels)
    if n == 1:
        xs = [(x_from + x_to) / 2]
    elif months:
        xs = [x_from + (m - months[0]) * width / (months[-1] - months[0]) for m in months]
    else:
        xs = [x_from + i * width / (n - 1) for i in range(n)]
    shown_txt = [_shorten(label, max(width / 2, 40.0)) for label in labels]

    top = 14.0
    height = top + PLOT_H + 28
    y = _linear(ticks[0], ticks[-1], top + PLOT_H, top)

    out = [_svg_open(height, title)]
    zero_drawn = False
    for t, txt in zip(ticks, tick_txt):
        if t == 0:
            zero_drawn = True
        else:
            out.append(_hline("grid", left, VB_W - 4, y(t)))
        out.append(f'<text class="t-tick" x="{_n(left - 8)}" y="{_n(y(t))}" dy=".32em" text-anchor="end">{_esc(txt)}</text>')
    if zero_drawn:
        out.append(_hline(_zero_class(data, values), left, VB_W - 4, y(0.0)))
    if ref is not None:
        out.append(_hline("ref", left, VB_W - 4, y(ref)))

    for j, (_, vals) in enumerate(series):
        slot = slots[j]
        segments: list[list[int]] = []
        for i, v in enumerate(vals):
            if v is None:
                continue
            prev = segments[-1][-1] if segments else None
            if prev is not None and (months[i] - months[prev] <= MONTH_LINE_MAX_STEP if months else prev == i - 1):
                segments[-1].append(i)
            else:
                segments.append([i])
        pt = lambda i: f"{_n(xs[i])},{_n(y(vals[i]))}"  # noqa: E731
        bridges = " ".join(f"M{pt(a[-1])} L{pt(b[0])}" for a, b in zip(segments, segments[1:]))
        if bridges:
            out.append(f'<path class="ln ln--gap {_cls("k", slot)}" d="{bridges}"/>')
        d = " ".join("M" + " L".join(pt(i) for i in seg) for seg in segments)
        if d:
            out.append(f'<path class="ln {_cls("k", slot)}" d="{d}"/>')
        dots = {seg[0] for seg in segments if len(seg) == 1} | ({last_idx[j]} if last_idx[j] >= 0 else set())
        for i in sorted(dots):
            v = vals[i]
            if v is None:
                continue
            out.append(f'<circle class="ring" cx="{_n(xs[i])}" cy="{_n(y(v))}" r="7"/>')
            out.append(f'<circle class="dot {_cls("f", slot)}" cx="{_n(xs[i])}" cy="{_n(y(v))}" r="5"/>')

    for i in range(n):
        lo_x = left if i == 0 else (xs[i - 1] + xs[i]) / 2
        hi_x = VB_W if i == n - 1 else (xs[i] + xs[i + 1]) / 2
        items = [(name, vals[i]) for name, vals in series]
        out.append(f'<g class="mk band"><title>{_esc(_tip(labels[i], items, fmt))}</title>')
        out.append(f'<rect class="hit" x="{_n(lo_x)}" y="{_n(top)}" width="{_n(hi_x - lo_x)}" height="{_n(PLOT_H)}"/>')
        out.append(_vline("xh", xs[i], top, top + PLOT_H))
        for j, (_, vals) in enumerate(series):
            v = vals[i]
            if v is not None:
                out.append(f'<circle class="hd ring" cx="{_n(xs[i])}" cy="{_n(y(v))}" r="6"/>')
                out.append(f'<circle class="hd {_cls("f", slots[j])}" cx="{_n(xs[i])}" cy="{_n(y(v))}" r="4"/>')
        out.append("</g>")

    if end_txt:
        i = last_idx[0]
        out.append(
            f'<text class="t-val" x="{_n(xs[i] + 10)}" y="{_n(y(series[0][1][i] or 0.0))}" dy=".32em">{_esc(end_txt)}</text>'
        )
    placed: list[tuple[int, float, float, float]] = []  # (index, centre, left, right)
    for i in [n - 1, 0] + list(range(n - 2, 0, -1)):  # the most recent and the first point, then any with room
        half = _tw(shown_txt[i]) / 2
        cx = min(max(xs[i], half + 2), VB_W - half - 2)  # edge labels are pulled inside the chart
        if i not in {p[0] for p in placed} and all(cx + half + 6 <= lo or cx - half - 6 >= hi for _, _, lo, hi in placed):
            placed.append((i, cx, cx - half, cx + half))
    for i, cx, _, _ in sorted(placed):
        out.append(f'<text class="t-x" x="{_n(cx)}" y="{_n(top + PLOT_H + 20)}" text-anchor="middle">{_esc(shown_txt[i])}</text>')
    out.append("</svg>")
    return "".join(out)


def _fold_series(series: list[tuple[str, list[Optional[float]]]]) -> list[tuple[str, list[Optional[float]]]]:
    """Stacks: keep 7 series and sum the rest into "Other" (never more than 8 colours)."""
    keep, rest = series[: MAX_SERIES - 1], series[MAX_SERIES - 1 :]
    n = len(series[0][1])
    other: list[Optional[float]] = []
    for i in range(n):
        vals = [vals[i] for _, vals in rest if vals[i] is not None]
        other.append(sum(vals) if vals else None)
    return keep + [("Other", other)]


# =========================================================================== HTML blocks
def _link(url: str, text: str, *, cls: str = "") -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return (
        f'<a{class_attr} href="{_esc(url)}" target="_blank" rel="noopener noreferrer" title="{_esc(url)}">'
        f'{_esc(text)}<span aria-hidden="true"> ↗</span></a>'
    )


_ECO_RE = re.compile(r"^[A-E]\d\d$")
_MOVES_RE = re.compile(r"^\d+\.(\.\.)?\s*(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]|O-O)")


def _cell_html(value: Any, fmt: Optional[str]) -> tuple[str, str]:
    """(inner html, extra css class) for one table cell."""
    if not is_missing(value) and (fmt == "url" or (fmt not in VALUE_FORMATS and infer_format(value) == "url")):
        url = safe_url(value)
        if url:
            return _link(url, url_link_text(url)), ""
        return _esc(value), ""
    text = format_value(value, fmt)
    if isinstance(value, str) and (_ECO_RE.match(text) or _MOVES_RE.match(text)):
        return _esc(text), "mono"
    return _esc(text), "wide" if len(text) > 24 else ""


def _table_html(table: Table, *, show_title: bool = True) -> str:
    columns = [text_or_empty(c) for c in (table.columns or [])]
    rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in (table.rows or [])]
    ncol = max([len(columns)] + [len(r) for r in rows])
    title = text_or_empty(table.title)
    parts = ['<figure class="table-block">']
    if show_title and title:
        parts.append(f'<figcaption class="block-title">{_esc(title)}</figcaption>')
    if ncol == 0 or not rows:
        parts.append('<p class="empty">No rows yet.</p>')
    else:
        columns += [""] * (ncol - len(columns))
        rows = [r + [None] * (ncol - len(r)) for r in rows]
        fmts = list(table.formats or [])
        fmts = [f if f else None for f in fmts] + [None] * (ncol - len(fmts))
        numeric = [is_numeric_format(fmts[c], (r[c] for r in rows)) for c in range(ncol)]
        label = f' aria-label="{_esc(title)}"' if title else ""
        parts.append(f'<div class="table-wrap" role="region" tabindex="0"{label}><table><thead><tr>')
        for c, name in enumerate(columns):
            cls = ' class="num"' if numeric[c] else ""
            parts.append(f'<th scope="col"{cls}>{_esc(name)}</th>')
        parts.append("</tr></thead><tbody>")
        for r in rows:
            parts.append("<tr>")
            for c, value in enumerate(r):
                inner, extra = _cell_html(value, fmts[c])
                classes = " ".join(x for x in ("num" if numeric[c] else "", extra) if x)
                parts.append(f'<td class="{classes}">{inner}</td>' if classes else f"<td>{inner}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
    if text_or_empty(table.note):
        parts.append(f'<p class="note">{_esc(table.note)}</p>')
    parts.append("</figure>")
    return "".join(parts)


def _legend(
    series: list[tuple[str, list[Optional[float]]]], slots: Sequence[str], kind: str, reference: str, *, always: bool = False
) -> str:
    """Series keys (when there are 2+ series, or ``always``) plus the dashed reference-line key."""
    key = "key key--line" if kind == "line" else "key"
    items = [
        f'<li><span class="{key} {_cls("sw", slots[j])}" aria-hidden="true"></span>{_esc(name)}</li>'
        for j, (name, _) in enumerate(series)
    ] if len(series) >= 2 or always else []
    if reference:
        items.append(f'<li><span class="key key--ref" aria-hidden="true"></span>{_esc(reference)}</li>')
    return f'<ul class="legend" aria-label="Legend">{"".join(items)}</ul>'


def _chart_html(chart: Chart) -> str:
    title = text_or_empty(chart.title)
    kind = chart.kind if chart.kind in ("bar", "hbar", "line", "stacked_bar") else "table"
    data = chart_data(chart)
    parts = [f'<figure class="ci-chart" data-kind="{_esc(kind)}">']
    if title:
        parts.append(f'<figcaption class="block-title">{_esc(title)}</figcaption>')
    if not data.has_values:
        parts.append('<p class="empty">No data to chart yet.</p>')
        if text_or_empty(chart.note):
            parts.append(f'<p class="note">{_esc(chart.note)}</p>')
        parts.append("</figure>")
        return "".join(parts)

    series = data.series
    notes = [text_or_empty(chart.note)]
    if len(series) > MAX_SERIES and not result_roles([name for name, _ in series]):
        if kind == "stacked_bar":
            series = _fold_series(series)
        else:
            notes.append(f"Showing the first {MAX_SERIES} of {len(series)} series; the data table lists all of them.")
            series = series[:MAX_SERIES]
    if not any(v is not None for _, vals in series for v in vals):
        kind = "table"  # only the hidden series carry numbers: show them as a table
    if kind == "table":
        parts.append(_table_html(chart_table(chart), show_title=False))
    else:
        stacks: Optional[list[list[int]]] = None
        keyed = list(range(len(series)))
        if kind == "stacked_bar":  # which segments touch: zeros are not drawn, so neighbours can be far apart
            stacks = []
            for i in range(len(data.labels)):
                segs = [(j, v) for j, (_, vals) in enumerate(series) if (v := vals[i]) is not None and v != 0]
                stacks += [[j for j, v in segs if v > 0], [j for j, v in segs if v < 0]]
            keyed = sorted({j for stack in stacks for j in stack}) or keyed  # nothing drawn: no legend needed
        slots = series_slots([name for name, _ in series], stacks)
        ref_label = reference_text(data.reference, data.fmt) if data.reference is not None else ""
        if len(series) >= 2 or ref_label:
            # a stacked series with nothing to draw is left out of the key (the data table lists it)
            parts.append(_legend([series[j] for j in keyed], [slots[j] for j in keyed], kind, ref_label, always=len(series) >= 2))
        if kind == "hbar":
            parts.append(_svg_hbar(data, series, slots, title))
        elif kind == "line":
            parts.append(_svg_line(data, series, slots, title))
        else:
            svg = _svg_columns(
                data, series, slots, title, stacked=kind == "stacked_bar", allow_horizontal=kind == "bar" and len(data.labels) <= 12
            )
            parts.append(svg if svg is not None else _svg_hbar(data, series, slots, title))
    for note in notes:
        if note:
            parts.append(f'<p class="note">{_esc(note)}</p>')
    if kind != "table":
        parts.append(
            '<details class="chart-data"><summary>Show the numbers</summary>'
            f"{_table_html(chart_table(chart), show_title=False)}</details>"
        )
    parts.append("</figure>")
    return "".join(parts)


def _kpi_html(kpi: Kpi) -> str:
    fmt = kpi.format if kpi.format in VALUE_FORMATS else None
    url = safe_url(kpi.value) if (fmt == "url" or (fmt is None and infer_format(kpi.value) == "url")) else None
    if url:
        value_html, cls = _link(url, url_link_text(url)), "kpi-value kpi-value--text"
    else:
        text = format_value(kpi.value, fmt)
        is_number = (fmt in NUMERIC_FORMATS or (fmt is None and infer_format(kpi.value) in NUMERIC_FORMATS)) and (
            to_number(kpi.value) is not None
        )
        value_html = _esc(text)
        cls = "kpi-value" if is_number or text == MISSING else "kpi-value kpi-value--text"
    hint = f'<dd class="kpi-hint">{_esc(kpi.hint)}</dd>' if text_or_empty(kpi.hint) else ""
    return f'<div class="kpi"><dt>{_esc(kpi.label)}</dt><dd class="{cls}">{value_html}</dd>{hint}</div>'


def _games_html(urls: Iterable[Any], label: str) -> str:
    links = valid_urls(urls)
    if not links:
        return ""
    items = "".join(_link(u, f"Game {i}") for i, u in enumerate(links, 1))
    return f'<div class="games"><span class="games-label">{_esc(label)}</span>{items}</div>'


def _confidence_html(confidence: Any) -> str:
    text = confidence_label(confidence)
    level = text.split()[0].lower()
    return (
        f'<span class="pill conf conf--{level}"><span class="conf-bars" aria-hidden="true">'
        f"<i></i><i></i><i></i></span>{_esc(text)}</span>"
    )


def _insight_html(ins: Insight, *, compact: bool, show_study: bool) -> str:
    kind = insight_kind(ins)
    h = "h4" if compact else "h3"
    glyph = glyph_for(ins)
    meaning = GLYPH_MEANINGS[glyph]
    kicker = f"{KIND_LABELS[kind]} · {category_label(ins.category)}"
    parts = [f'<article class="card insight insight--{kind}{" insight--compact" if compact else ""}">']
    parts.append(
        f'<div class="insight-head"><span class="glyph glyph--{kind}" role="img" aria-label="{_esc(meaning)}" '
        f'title="{_esc(meaning)}">{_esc(glyph)}</span><div class="insight-heading">'
        f'<p class="kicker">{_esc(kicker)}</p><{h} class="insight-title">{_esc(ins.title)}</{h}></div></div>'
    )
    if text_or_empty(ins.detail):
        parts.append(f'<p class="insight-detail">{_esc(ins.detail)}</p>')
    steps = [text_or_empty(s) for s in (ins.study or []) if text_or_empty(s)]
    if show_study and steps:
        items = "".join(f"<li>{_esc(s)}</li>" for s in steps)
        if compact:
            parts.append(
                f'<details class="study"><summary>What to do ({len(steps)})</summary><ul class="checklist">{items}</ul></details>'
            )
        else:
            parts.append(f'<div class="study"><p class="mini-label">What to do</p><ul class="checklist">{items}</ul></div>')
    parts.append(f'<div class="insight-meta">{_confidence_html(ins.confidence)}{_games_html(ins.example_games, "Review")}</div>')
    parts.append("</article>")
    return "".join(parts)


def _study_html(items: Sequence[StudyItem]) -> str:
    head = (
        '<div class="section-head"><h2 id="study-h">Study plan</h2>'
        '<p class="section-sub">Most important first. Work through it from the top.</p></div>'
    )
    if not items:
        body = '<p class="empty">Nothing to study yet: no weakness stood out clearly enough in these games.</p>'
    else:
        lis = []
        for i, item in enumerate(items, 1):
            actions = [text_or_empty(a) for a in (item.actions or []) if text_or_empty(a)]
            checklist = (
                '<ul class="checklist">' + "".join(f"<li>{_esc(a)}</li>" for a in actions) + "</ul>" if actions else ""
            )
            kicker = f'<p class="kicker">{_esc(item.category)}</p>' if text_or_empty(item.category) else ""
            why = f'<p class="plan-why">{_esc(item.why)}</p>' if text_or_empty(item.why) else ""
            lis.append(
                f'<li class="plan-item"><span class="plan-num" aria-hidden="true">{i}.</span><div class="plan-body">'
                f'{kicker}<h3 class="plan-title">{_esc(item.title)}</h3>{why}{checklist}'
                f"{_games_html(item.games, 'Review these games')}</div></li>"
            )
        body = '<ol class="plan">' + "".join(lis) + "</ol>"
    return f'<section class="section" id="study" aria-labelledby="study-h">{head}{body}</section>'


def _sw_html(report: Report) -> str:
    covered = plan_titles(report)

    def column(key: str, title: str, glyph: str, insights: Sequence[Insight], empty: str) -> str:
        cards = "".join(
            _insight_html(i, compact=False, show_study=text_or_empty(i.title) not in covered) for i in insights
        )
        body = f'<div class="cards">{cards}</div>' if cards else f'<p class="empty">{_esc(empty)}</p>'
        kind = "strength" if key == "strengths" else "weakness"
        return (
            f'<section class="sw-col" id="{key}" aria-labelledby="{key}-h"><div class="sw-head">'
            f'<span class="glyph glyph--{kind} glyph--sm" aria-hidden="true">{glyph}</span>'
            f'<h2 id="{key}-h">{title}</h2></div>{body}</section>'
        )

    key = (
        '<p class="glyph-key">Marks follow chess annotation: <b>??</b> serious, <b>?</b> clear, <b>?!</b> minor weakness; '
        "<b>!!</b> major, <b>!</b> solid strength; <b>=</b> observation.</p>"
    )
    more = "More games make patterns reliable enough to report."
    return (
        '<div class="section sw">'
        + column("strengths", "Strengths", "!", report.strengths or [], f"No clear strengths yet. {more}")
        + column("weaknesses", "Weaknesses", "?", report.weaknesses or [], f"No clear weaknesses yet. {more}")
        + key
        + "</div>"
    )


def lichess_analysis_url(fen: str) -> Optional[str]:
    """Link that opens the position on the Lichess analysis board."""
    fen = (fen or "").strip()
    if not fen or not re.fullmatch(r"[1-8KQRBNPkqrbnp/]+ [wb] [KQkqA-Ha-h-]+ [a-h1-8-]+( \d+ \d+)?", fen):
        return None
    return "https://lichess.org/analysis/" + fen.replace(" ", "_")


# --------------------------------------------------------------------------- board diagrams
# Diagram.svg is meant to be python-chess output, but it is still data: rather than trusting it,
# it is parsed and re-written from an allowlist of the elements and attributes python-chess
# draws with. Anything else (scripts, event handlers, foreignObject, external references,
# CSS url(), comments/CDATA that HTML and XML parse differently) rejects the whole SVG, and the
# board is redrawn from the FEN instead. Ids get a per-diagram prefix so several boards on one
# page never share ids (python-chess reuses "white-pawn", "check_gradient", ...).
_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"
_SVG_MAX_CHARS = 400_000
_SVG_MAX_DEPTH = 24
_SVG_MAX_ATTR = 20_000  # the longest python-chess path data is a few thousand characters
_SVG_TAGS = frozenset(
    {"svg", "defs", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "use",
     "radialGradient", "linearGradient", "stop", "text", "tspan", "title"}
)
_SVG_TEXT_TAGS = frozenset({"text", "tspan", "title"})
_SVG_DROP = frozenset({"desc", "metadata"})  # python-chess puts an ASCII board (<pre>) in <desc>
_SVG_NUMS = re.compile(r"^[\d\s.,eE+\-%]{1,200}$")
_SVG_PATH = re.compile(r"^[\d\s.,eE+\-MmLlHhVvCcSsQqTtAaZz]{0,20000}$")
# One way to match any input (a function name must follow the separators), with bounded repeats:
# a nested "(?:\s*...\s*,?)*" backtracked exponentially on "scale(1)  scale(1)  ... !".
_SVG_TRANSFORM = re.compile(
    r"^\s*(?:(?:matrix|translate|scale|rotate|skewX|skewY)\s*\([\d\s.,eE+\-]{0,120}\)[\s,]*){0,8}$"
)
_SVG_PAINT = re.compile(r"^(?:#[0-9a-fA-F]{3,8}|[a-zA-Z]{1,20}|rgba?\([\d\s.,%]+\)|url\(#[A-Za-z][\w.-]{0,63}\))$")
_SVG_KEYWORD = re.compile(r"^[a-zA-Z-]{1,24}$")
_SVG_ID = re.compile(r"^[A-Za-z][\w.-]{0,63}$")
_SVG_CLASS = re.compile(r"^[\w\s-]{0,120}$")
_SVG_REF = re.compile(r"^#([A-Za-z][\w.-]{0,63})$")
_SVG_ATTRS: dict[str, re.Pattern[str]] = {
    **{a: _SVG_NUMS for a in (
        "viewBox", "width", "height", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "fx", "fy",
        "offset", "opacity", "fill-opacity", "stroke-opacity", "stop-opacity", "stroke-width", "stroke-dasharray",
        "stroke-miterlimit", "font-size", "points", "dx", "dy")},
    **{a: _SVG_PAINT for a in ("fill", "stroke", "stop-color")},
    **{a: _SVG_KEYWORD for a in (
        "fill-rule", "clip-rule", "stroke-linecap", "stroke-linejoin", "text-anchor", "dominant-baseline",
        "alignment-baseline", "font-weight", "font-family", "gradientUnits", "visibility", "version")},
    "d": _SVG_PATH,
    "transform": _SVG_TRANSFORM,
    "gradientTransform": _SVG_TRANSFORM,
    "class": _SVG_CLASS,
}
_SVG_STYLE_PROPS = {k: v for k, v in _SVG_ATTRS.items() if k not in ("d", "class", "points", "viewBox", "version")}


class _UnsafeSvg(ValueError):
    pass


def _svg_local(name: str) -> tuple[str, str]:
    if name.startswith("{"):
        ns, _, local = name[1:].partition("}")
        return ns, local
    return "", name


def _svg_attr(name: str, value: str, prefix: str) -> tuple[str, str]:
    ns, local = _svg_local(name)
    if ns == _XLINK_NS and local == "href":
        name = "xlink:href"
    elif ns:
        raise _UnsafeSvg(name)
    else:
        name = local
    value = value.strip()
    if len(value) > _SVG_MAX_ATTR:
        raise _UnsafeSvg(f"{name} is {len(value)} characters long")
    if name in ("href", "xlink:href"):
        ref = _SVG_REF.match(value)
        if not ref:
            raise _UnsafeSvg(f"{name}={value[:40]!r}")
        return name, f"#{prefix}{ref.group(1)}"
    if name == "id":
        if not _SVG_ID.match(value):
            raise _UnsafeSvg(f"id={value[:40]!r}")
        return name, prefix + value
    if name == "style":
        decls = []
        for decl in value.split(";"):
            if not decl.strip():
                continue
            prop, sep, val = decl.partition(":")
            prop, val = prop.strip().lower(), val.strip()
            rule = _SVG_STYLE_PROPS.get(prop)
            if not sep or rule is None or not rule.match(val):
                raise _UnsafeSvg(f"style {decl[:40]!r}")
            decls.append(f"{prop}:{_svg_prefix_urls(val, prefix)}")
        return name, ";".join(decls)
    rule = _SVG_ATTRS.get(name)
    if rule is None or not rule.match(value):
        raise _UnsafeSvg(f"{name}={value[:40]!r}")
    return name, _svg_prefix_urls(value, prefix)


def _svg_prefix_urls(value: str, prefix: str) -> str:
    return re.sub(r"url\(#([\w.-]+)\)", lambda m: f"url(#{prefix}{m.group(1)})", value)


def _svg_write(el: ElementTree.Element, prefix: str, out: list[str], depth: int) -> None:
    ns, tag = _svg_local(el.tag if isinstance(el.tag, str) else "")
    if ns not in ("", _SVG_NS) or depth > _SVG_MAX_DEPTH:
        raise _UnsafeSvg(tag)
    if tag in _SVG_DROP:
        return
    if tag not in _SVG_TAGS:
        raise _UnsafeSvg(tag)
    attrs = "".join(f' {k}="{_esc(v)}"' for k, v in (_svg_attr(n, v, prefix) for n, v in el.attrib.items()))
    if depth == 0:
        attrs = f' xmlns="{_SVG_NS}" xmlns:xlink="{_XLINK_NS}"' + attrs
    out.append(f"<{tag}{attrs}>")
    if el.text and el.text.strip():
        if tag not in _SVG_TEXT_TAGS:
            raise _UnsafeSvg(f"text in <{tag}>")
        out.append(_esc(el.text))
    for child in el:
        _svg_write(child, prefix, out, depth + 1)
        if child.tail and child.tail.strip():
            raise _UnsafeSvg("stray text")
    out.append(f"</{tag}>")


def sanitize_board_svg(svg: Any, prefix: str = "") -> str:
    """Re-serialise a board SVG through the allowlist; "" if it contains anything else."""
    if not isinstance(svg, str):
        return ""
    text = svg.strip()
    if text.startswith("<?xml"):
        text = text[text.find("?>") + 2 :].lstrip()
    # no comments, CDATA, doctypes/entities or processing instructions: HTML and XML disagree on them
    if not text.startswith("<svg") or len(text) > _SVG_MAX_CHARS or "<!" in text or "<?" in text:
        return ""
    try:
        root = ElementTree.fromstring(text)
        if _svg_local(root.tag)[1] != "svg":
            return ""
        out: list[str] = []
        _svg_write(root, prefix, out, 0)
    except (ElementTree.ParseError, _UnsafeSvg, ValueError):
        return ""
    return "".join(out)


def _board_from_fen(fen: Any) -> str:
    """A plain python-chess board for a FEN (used when the supplied SVG is rejected)."""
    try:
        import chess
        import chess.svg

        board = chess.Board(str(fen).strip())
        return chess.svg.board(board, size=280)
    except Exception:  # noqa: BLE001 — invalid FEN (or python-chess missing): no board
        return ""


def _diagram_html(d: Diagram, id_prefix: str = "b-") -> str:
    svg = sanitize_board_svg(d.svg, id_prefix) or sanitize_board_svg(_board_from_fen(d.fen), id_prefix)
    links = []
    game = safe_url(d.link)
    if game:
        links.append(_link(game, "Open the game"))
    analysis = lichess_analysis_url(d.fen)
    if analysis:
        links.append(_link(analysis, "Analyse on Lichess"))
    board = f'<div class="board-svg" role="img" aria-label="{_esc(d.title)}">{svg}</div>' if svg else ""
    return (
        '<figure class="board-fig">'
        f"{board}"
        f'<figcaption><span class="block-title">{_esc(d.title)}</span>'
        + (f'<span class="board-cap">{_esc(d.caption)}</span>' if text_or_empty(d.caption) else "")
        + (f'<span class="board-links">{" ".join(links)}</span>' if links else "")
        + "</figcaption></figure>"
    )


def _module_html(module: ModuleResult, anchor: str) -> str:
    title = text_or_empty(module.title) or text_or_empty(module.key) or "Section"
    parts = [f'<section class="section module" id="{anchor}" aria-labelledby="{anchor}-h">']
    parts.append(f'<div class="section-head"><h2 id="{anchor}-h">{_esc(title)}</h2>')
    if text_or_empty(module.summary):
        parts.append(f'<p class="summary">{_esc(module.summary)}</p>')
    parts.append("</div>")
    if module.kpis:
        parts.append('<dl class="kpis">' + "".join(_kpi_html(k) for k in module.kpis) + "</dl>")
    if module.charts:
        parts.append('<div class="charts">' + "".join(_chart_html(c) for c in module.charts) + "</div>")
    if getattr(module, "diagrams", None):
        parts.append(
            '<div class="boards">'
            + "".join(_diagram_html(d, f"{anchor}-b{i}-") for i, d in enumerate(module.diagrams, 1))
            + "</div>"
        )
    for table in module.tables or []:
        parts.append(_table_html(table))
    if module.insights:
        ranked = sorted(module.insights, key=lambda i: -_priority(i))
        parts.append(
            f'<div class="findings"><h3 class="sub-head">Findings <span class="count">{len(ranked)}</span></h3>'
            '<div class="cards cards--grid">'
            + "".join(_insight_html(i, compact=True, show_study=True) for i in ranked)
            + "</div></div>"
        )
    parts.append("</section>")
    return "".join(parts)


# ids the page itself uses: the fixed sections and their headings ("study" / "study-h")
_RESERVED_IDS = {"study", "strengths", "weaknesses", "top", "main"}
_RESERVED_IDS |= {f"{i}-h" for i in _RESERVED_IDS}


def module_anchors(modules: Sequence[ModuleResult]) -> list[str]:
    """Unique, URL-safe section ids: the module key when possible.

    Each module section also uses ``<anchor>-h`` for its heading, so neither id may repeat
    another section's id, heading id or a reserved page id (compared case-insensitively).
    """
    taken = set(_RESERVED_IDS)
    out = []
    for i, m in enumerate(modules, 1):
        base = re.sub(r"[^A-Za-z0-9_-]+", "-", text_or_empty(m.key)).strip("-") or f"module-{i}"
        if not base[0].isalpha():
            base = f"m-{base}"
        if base.lower() in _RESERVED_IDS:
            base = f"{base}-section"
        anchor, k = base, 2
        while anchor.lower() in taken or f"{anchor}-h".lower() in taken:
            anchor, k = f"{base}-{k}", k + 1
        taken |= {anchor.lower(), f"{anchor}-h".lower()}
        out.append(anchor)
    return out


def _masthead_html(report: Report) -> str:
    meta = [f"<span>{_esc(games_text(report.n_games))}</span>"]
    span = date_range_text(report.date_from, report.date_to)
    if span:
        meta.append(f"<span>{_esc(span)}</span>")
    if text_or_empty(report.filters):
        meta.append(f"<span>{_esc(report.filters)}</span>")
    sep = '<span class="sep" aria-hidden="true">·</span>'
    engine = f'<p class="engine-note">{_esc(report.engine_note)}</p>' if text_or_empty(report.engine_note) else ""
    name = text_or_empty(report.username) or "Your games"
    return (
        '<header class="masthead"><p class="brand"><span class="board-mark" aria-hidden="true"></span>Chess Insights</p>'
        f'<h1 class="player">{_esc(name)}</h1><p class="meta">{sep.join(meta)}</p>{engine}</header>'
    )


def _headline(report: Report) -> str:
    text = text_or_empty(report.headline)
    if text:
        return text
    if not report.n_games:
        return "No games matched the filters, so there is nothing to analyse yet."
    return "No clear patterns yet. Play more games, or widen the filters, for reliable insights."


def _nav_html(report: Report, anchors: Sequence[str]) -> str:
    def chip(href: str, text: str, count: Optional[int] = None) -> str:
        badge = f'<span class="count">{count}</span>' if count is not None else ""
        return f'<a href="#{href}">{_esc(text)}{badge}</a>'

    chips = [
        chip("study", "Study plan", len(report.study_plan or [])),
        chip("strengths", "Strengths", len(report.strengths or [])),
        chip("weaknesses", "Weaknesses", len(report.weaknesses or [])),
    ]
    chips += [chip(a, text_or_empty(m.title) or a) for m, a in zip(report.modules or [], anchors)]
    return f'<nav class="chips" aria-label="Sections">{"".join(chips)}</nav>'


def _footer_html(report: Report) -> str:
    notes = "".join(f"<li>{_esc(n)}</li>" for n in METHOD_NOTES)
    generated = format_generated(report.generated_at)
    gen = f"<p>Generated {_esc(generated)} by chess-insights.</p>" if generated else ""
    return f'<footer class="foot"><h2>How this report works</h2><ul>{notes}</ul>{gen}</footer>'


def _body_html(report: Report) -> str:
    modules = list(report.modules or [])
    anchors = module_anchors(modules)
    return "".join(
        [
            '<div class="ci" lang="en">',
            _masthead_html(report),
            f'<p class="lede">{_esc(_headline(report))}</p>',
            _nav_html(report, anchors),
            '<main class="sections">',
            _study_html(report.study_plan or []),
            _sw_html(report),
            *(_module_html(m, a) for m, a in zip(modules, anchors)),
            "</main>",
            _footer_html(report),
            "</div>",
        ]
    )


# =========================================================================== CSS / JS
_FONT_LINK = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600'
    "&amp;family=IBM+Plex+Sans+Condensed:wght@500;600&amp;family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400"
    '&amp;display=swap">'
)

_LIGHT = {
    "paper": "#F6F6F3",
    "surface": "#FCFCFB",
    "ink": "#16181D",
    "ink-2": "#4E535C",
    "muted": "#676B73",  # >= 4.5:1 on paper and surface: notes, hints and ticks are text too
    "hairline": "#E1E0D9",
    "axis": "#C3C2B7",
    "accent": "#3E6B4F",
    "accent-wash": "#E6EDE7",
    "board-light": "#E9E4D4",
    "board-dark": "#7D8F6E",
    "good": "#1F7A3A",
    "good-wash": "#E2EFE4",
    "bad": "#B83535",
    "bad-wash": "#F7E4E2",
    "neutral": "#4E535C",
    "neutral-wash": "#ECECE7",
    "tip-bg": "#16181D",
    "tip-ink": "#F2F2EF",
    "scroll-shadow": "rgba(22,24,29,.22)",
    "s1": "#2a78d6",
    "s2": "#eb6834",
    "s3": "#1baf7a",
    "s4": "#eda100",
    "s5": "#e87ba4",
    "s6": "#008300",
    "s7": "#4a3aa7",
    "s8": "#e34948",
    # results: diverging green / grey / red, validated for colour-vision deficiency on adjacent pairs
    "s-w1": "#016d5e",
    "s-w2": "#008874",
    "s-w3": "#3e9f8c",
    "s-w4": "#63baa8",
    "s-d": "#808284",
    "s-l1": "#b6322d",
    "s-l2": "#ca564d",
    "s-l3": "#db786e",
    "s-l4": "#e8978d",
}
_DARK = {
    "paper": "#0F1113",
    "surface": "#181A1D",
    "ink": "#F2F2EF",
    "ink-2": "#C3C2B7",
    "muted": "#8A8E96",
    "hairline": "#2C2C2A",
    "axis": "#4A4A45",
    "accent": "#7FB08F",
    "accent-wash": "#1E2A22",
    "board-light": "#4A5043",
    "board-dark": "#6F8263",
    "good": "#5CC27A",
    "good-wash": "#18291D",
    "bad": "#E66767",
    "bad-wash": "#351C1C",
    "neutral": "#C3C2B7",
    "neutral-wash": "#25272A",
    "tip-bg": "#F2F2EF",
    "tip-ink": "#16181D",
    "scroll-shadow": "rgba(242,242,239,.2)",
    "s1": "#3987e5",
    "s2": "#d95926",
    "s3": "#199e70",
    "s4": "#c98500",
    "s5": "#d55181",
    "s6": "#008300",
    "s7": "#9085e9",
    "s8": "#e66767",
    "s-w1": "#2ac8ae",
    "s-w2": "#15ad95",
    "s-w3": "#1f8f7c",
    "s-w4": "#047463",
    "s-d": "#86888a",
    "s-l1": "#fc7b6f",
    "s-l2": "#d55f56",
    "s-l3": "#aa4941",
    "s-l4": "#81332d",
}

_BASE_CSS = """
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font-sans);font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.ci{max-width:1080px;margin-inline:auto;padding-left:max(clamp(16px,4vw,40px),env(safe-area-inset-left,0px));padding-right:max(clamp(16px,4vw,40px),env(safe-area-inset-right,0px));padding-block:28px 56px}
:where(.ci) :where(h1,h2,h3,h4){margin:0;text-wrap:balance}
:where(.ci) :where(p,dl,dd,figure){margin:0}
:where(.ci) :where(ul,ol){margin:0;padding:0}
.ci a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
.ci a:hover{text-decoration-thickness:2px}
.ci :focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
.masthead{display:grid;gap:6px}
.brand{display:flex;align-items:center;gap:10px;font:600 13px/1.2 var(--font-cond);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2)}
.board-mark{flex:none;width:20px;height:20px;border-radius:3px;background:conic-gradient(var(--board-dark) 0 25%,var(--board-light) 0 50%,var(--board-dark) 0 75%,var(--board-light) 0) 0 0/10px 10px}
.player{font:600 32px/1.15 var(--font-sans);letter-spacing:-.01em;overflow-wrap:anywhere}
.meta{color:var(--ink-2)}
.meta .sep{color:var(--muted);padding-inline:.4em}
.engine-note{color:var(--muted);font-size:13px}
.lede{margin-top:20px;max-width:62ch;font-size:18px;line-height:1.45;color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.chips a{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border:1px solid var(--hairline);border-radius:999px;background:var(--surface);color:var(--ink-2);font-size:13px;line-height:1.4;text-decoration:none}
.chips a:hover{border-color:var(--accent);color:var(--accent)}
.count{font:500 12px/1 var(--font-mono);color:var(--muted)}
.sections{display:flex;flex-direction:column;gap:44px;margin-top:36px}
.section{display:flex;flex-direction:column;gap:20px;min-width:0;border-top:1px solid var(--hairline);padding-top:28px}
.section-head{display:grid;gap:6px;max-width:72ch}
.section h2,.sw h2{font:600 24px/1.2 var(--font-cond);letter-spacing:.005em}
.section-sub,.summary{color:var(--ink-2)}
.kicker{font:600 13px/1.3 var(--font-cond);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.mini-label{font:600 13px/1.3 var(--font-cond);letter-spacing:.04em;color:var(--ink-2)}
.study{display:grid;gap:6px}
.empty{color:var(--muted)}
.note{font-size:13px;color:var(--muted)}
.plan{list-style:none;display:flex;flex-direction:column}
.plan-item{display:grid;grid-template-columns:2.6rem minmax(0,1fr);gap:12px;padding-block:18px;border-bottom:1px solid var(--hairline)}
.plan-item:first-child{padding-top:2px}
.plan-item:last-child{border-bottom:0;padding-bottom:0}
.plan-num{font:500 24px/1.15 var(--font-mono);color:var(--accent)}
.plan-body{display:grid;gap:8px;max-width:72ch;min-width:0}
.plan-title{font:600 18px/1.3 var(--font-sans)}
.plan-why{color:var(--ink-2)}
.checklist{list-style:none;display:grid;gap:6px}
.checklist li{position:relative;padding-left:26px}
.checklist li::before{content:"";position:absolute;left:1px;top:.3em;width:13px;height:13px;border:1.5px solid var(--board-dark);border-radius:3px}
.games{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;font-size:13px}
.games-label{color:var(--muted);font-weight:500}
.games a{white-space:nowrap}
.sw{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:36px 32px}
.sw-col{display:flex;flex-direction:column;gap:14px;min-width:0}
.sw-head{display:flex;align-items:center;gap:10px}
.glyph-key{grid-column:1/-1;font-size:13px;color:var(--muted)}
.glyph-key b{font:600 13px var(--font-mono);color:var(--ink-2)}
.cards{display:grid;gap:12px;align-content:start}
.cards--grid{grid-template-columns:repeat(auto-fill,minmax(min(100%,300px),1fr))}
.card{background:var(--surface);border:1px solid var(--hairline);border-radius:8px;min-width:0}
.insight{display:grid;gap:10px;padding:14px 16px;align-content:start}
.insight-head{display:grid;grid-template-columns:auto minmax(0,1fr);gap:12px;align-items:start}
.insight-heading{display:grid;gap:2px;min-width:0}
.insight-title{font:600 15px/1.35 var(--font-sans);overflow-wrap:anywhere}
.insight-detail{color:var(--ink-2);overflow-wrap:anywhere}
.insight-meta{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px}
.glyph{display:inline-grid;place-items:center;flex:none;width:34px;height:34px;border-radius:6px;font:600 15px/1 var(--font-mono);letter-spacing:-.04em}
.glyph--sm{width:28px;height:28px;font-size:14px}
.insight--compact .glyph{width:28px;height:28px;font-size:14px}
.glyph--weakness{background:var(--bad-wash);color:var(--bad)}
.glyph--strength{background:var(--good-wash);color:var(--good)}
.glyph--observation{background:var(--neutral-wash);color:var(--neutral)}
.pill{display:inline-flex;align-items:center;gap:7px;padding:2px 10px;border:1px solid var(--hairline);border-radius:999px;color:var(--ink-2);font-size:13px;line-height:1.5;white-space:nowrap}
.conf-bars{display:inline-flex;align-items:flex-end;gap:2px;height:10px}
.conf-bars i{display:block;width:3px;border-radius:1px;background:var(--hairline)}
.conf-bars i:nth-child(1){height:5px}.conf-bars i:nth-child(2){height:7.5px}.conf-bars i:nth-child(3){height:10px}
.conf--low .conf-bars i:nth-child(-n+1),.conf--medium .conf-bars i:nth-child(-n+2),.conf--high .conf-bars i{background:var(--ink-2)}
details.study summary,.chart-data summary{cursor:pointer;font-size:13px;color:var(--ink-2);width:max-content;max-width:100%}
details.study{display:block}
details.study[open] summary{margin-bottom:8px}
.study .checklist{font-size:15px}
.findings{display:flex;flex-direction:column;gap:12px}
.sub-head{display:flex;align-items:baseline;gap:8px;font:600 18px/1.3 var(--font-cond)}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,150px),1fr));gap:10px}
.kpi{display:flex;flex-direction:column;gap:4px;min-width:0;padding:12px 14px;background:var(--surface);border:1px solid var(--hairline);border-radius:8px}
.kpi dt{font-size:13px;line-height:1.3;color:var(--ink-2)}
.kpi-value{font:500 24px/1.15 var(--font-mono);color:var(--ink);overflow-wrap:anywhere}
.kpi-value--text{font:600 15px/1.35 var(--font-sans)}
.kpi-hint{font-size:13px;line-height:1.35;color:var(--muted)}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,340px),1fr));gap:20px}
.boards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,240px),1fr));gap:20px}
.board-fig{display:flex;flex-direction:column;gap:10px;min-width:0;margin:0}
.board-svg svg{display:block;width:100%;max-width:320px;height:auto;border-radius:6px}
.board-fig figcaption{display:flex;flex-direction:column;gap:4px;font-size:14px;color:var(--ink-2)}
.board-cap{line-height:1.45}
.board-links{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px}
.ci-chart{display:flex;flex-direction:column;gap:10px;min-width:0;padding:14px 14px 12px;background:var(--surface);border-radius:8px}
.block-title{font:600 15px/1.3 var(--font-cond);color:var(--ink)}
.legend{list-style:none;display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px;color:var(--ink-2)}
.legend li{display:inline-flex;align-items:center;gap:6px}
.key{display:inline-block;width:10px;height:10px;border-radius:2px}
.key--line{width:16px;height:2px;border-radius:1px}
.key--ref{width:18px;height:0;border-radius:0;border-top:2px dashed var(--ink-2)}
.ci-chart svg{display:block;width:100%;height:auto;overflow:visible}
.ci-chart svg text{font-family:var(--font-sans);font-size:13px;fill:var(--ink-2)}
.ci-chart svg line,.ci-chart svg text{pointer-events:none}
.ci-chart .t-tick{font-size:12px;fill:var(--muted);font-variant-numeric:tabular-nums}
.ci-chart .t-val{font-size:12px;fill:var(--ink);font-variant-numeric:tabular-nums}
.ci-chart .t-val{paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
.ci-chart .grid{stroke:var(--hairline);stroke-width:1px;vector-effect:non-scaling-stroke}
.ci-chart .base{stroke:var(--axis);stroke-width:1px;vector-effect:non-scaling-stroke}
.ci-chart .zero{stroke:var(--ink-2);stroke-width:1.5px;vector-effect:non-scaling-stroke}
.ci-chart .ref{stroke:var(--ink-2);stroke-width:1.5px;stroke-dasharray:5 4;vector-effect:non-scaling-stroke}
.ci-chart .hit{fill:transparent}
.ci-chart .ln{fill:none;stroke-width:2px;stroke-linejoin:round;stroke-linecap:round;vector-effect:non-scaling-stroke}
.ci-chart .ln--gap{stroke-width:1.5px;stroke-dasharray:5 4;stroke-linecap:butt}
.ci-chart .ring{fill:var(--surface)}
.ci-chart .xh{stroke:var(--axis);stroke-width:1px;vector-effect:non-scaling-stroke;opacity:0}
.ci-chart .hd{opacity:0}
.ci-chart .band:hover .xh,.ci-chart .band:hover .hd{opacity:1}
.ci-chart .mk:hover .bar{opacity:.78}
.chart-data .table-wrap{margin-top:8px}
.table-block{display:grid;gap:8px;min-width:0}
.table-wrap{--wrap-bg:var(--paper);overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%;background:linear-gradient(to right,var(--wrap-bg) 40%,transparent) left center/28px 100% no-repeat local,linear-gradient(to left,var(--wrap-bg) 40%,transparent) right center/28px 100% no-repeat local,radial-gradient(farthest-side at 0 50%,var(--scroll-shadow),transparent) left center/10px 100% no-repeat scroll,radial-gradient(farthest-side at 100% 50%,var(--scroll-shadow),transparent) right center/10px 100% no-repeat scroll}
.ci-chart .table-wrap{--wrap-bg:var(--surface)}
.ci table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.4;font-variant-numeric:tabular-nums}
.ci th,.ci td{padding:7px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--hairline)}
.ci th{font:600 13px/1.3 var(--font-cond);letter-spacing:.03em;color:var(--ink-2);border-bottom-color:var(--axis);vertical-align:bottom}
.ci th:first-child,.ci td:first-child{padding-left:0}
.ci th:last-child,.ci td:last-child{padding-right:0}
.ci td{min-width:6ch}
.ci td.wide{min-width:18ch}
.ci td:first-child:not(.num){min-width:13ch}
.ci .num{text-align:right;white-space:nowrap}
.ci td.mono{font-family:var(--font-mono);font-size:13px}
.ci tbody tr:hover{background:var(--accent-wash)}
.foot{display:grid;gap:10px;margin-top:48px;padding-top:24px;border-top:1px solid var(--hairline);font-size:13px;color:var(--ink-2)}
.foot>*{max-width:80ch}
.foot h2{font:600 15px/1.3 var(--font-cond);color:var(--ink)}
.foot ul{padding-left:1.1em;display:grid;gap:6px}
.ci-tip{position:fixed;z-index:20;pointer-events:none;max-width:260px;padding:8px 10px;border-radius:6px;background:var(--tip-bg);color:var(--tip-ink);font:13px/1.4 var(--font-sans);box-shadow:0 6px 20px rgba(0,0,0,.2)}
.ci-tip[hidden]{display:none}
.ci-tip-head{opacity:.72;margin-bottom:2px}
.ci-tip strong{font-family:var(--font-mono);font-weight:600;margin-right:8px}
@media (max-width:560px){
  .player{font-size:28px}
  .lede{font-size:17px}
  .ci-chart{padding:12px 8px 10px}
  .insight{padding:12px 14px}
  .plan-item{grid-template-columns:2rem minmax(0,1fr);gap:8px}
  .plan-num{font-size:18px;line-height:1.35}
}
@media (prefers-reduced-motion:no-preference){.chips a,.ci-chart .bar{transition:opacity .12s,border-color .12s,color .12s}}
"""

_SCRIPT = """<script>
(function () {
  if (!document.querySelector || !window.addEventListener) return;
  var tip = document.createElement("div");
  tip.className = "ci-tip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  document.body.appendChild(tip);
  var titles = document.querySelectorAll(".ci-chart svg .mk > title");
  for (var i = 0; i < titles.length; i++) {
    var t = titles[i];
    t.parentNode.setAttribute("data-tip", t.textContent);
    t.parentNode.removeChild(t);
  }
  function show(el, x, y) {
    var lines = el.getAttribute("data-tip").split("\\n");
    tip.textContent = "";
    lines.forEach(function (line, k) {
      var row = document.createElement("div");
      var cut = line.lastIndexOf(": ");
      if (k === 0 && lines.length > 1) { row.className = "ci-tip-head"; row.textContent = line; }
      else if (cut > 0) {
        var value = document.createElement("strong");
        value.textContent = line.slice(cut + 2);
        row.appendChild(value);
        row.appendChild(document.createTextNode(line.slice(0, cut)));
      } else { row.textContent = line; }
      tip.appendChild(row);
    });
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var vw = document.documentElement.clientWidth, vh = window.innerHeight;
    var left = Math.min(x + 14, vw - w - 8), top = y + 16;
    if (top + h > vh - 8) top = y - h - 12;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  function track(e) {
    var el = e.target && e.target.closest ? e.target.closest(".ci-chart [data-tip]") : null;
    if (el) show(el, e.clientX, e.clientY); else tip.hidden = true;
  }
  document.addEventListener("pointermove", track);
  document.addEventListener("pointerdown", track);
  window.addEventListener("scroll", function () { tip.hidden = true; }, { passive: true });
})();
</script>"""


def _tokens(values: dict[str, str]) -> str:
    return "".join(f"--{k}:{v};" for k, v in values.items())


def build_css(standalone: bool = True) -> str:
    fonts = (
        '--font-sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;'
        '--font-cond:"IBM Plex Sans Condensed","Roboto Condensed","Arial Narrow","IBM Plex Sans",system-ui,sans-serif;'
        '--font-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;'
    )
    slots = [str(i) for i in range(1, MAX_SERIES + 1)] + [
        f"{role}{k}" for role in ("w", "l") for k in range(1, RESULT_STEPS + 1)
    ] + ["d"]
    series = "".join(
        f".ci-chart .{_cls('f', s)}{{fill:var(--s{'' if s.isdigit() else '-'}{s})}}"
        f".ci-chart .{_cls('k', s)}{{stroke:var(--s{'' if s.isdigit() else '-'}{s})}}"
        f".{_cls('sw', s)}{{background:var(--s{'' if s.isdigit() else '-'}{s})}}"
        for s in slots
    )
    dark = _tokens(_DARK) + "color-scheme:dark;"
    safe_area = (
        ":root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}" if standalone else ""
    )
    return (
        f":root{{{_tokens(_LIGHT)}{fonts}color-scheme:light}}"
        f'@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{{dark}}}}}'
        f':root[data-theme="dark"]{{{dark}}}'
        f"{safe_area}{_BASE_CSS}{series}"
    )


# =========================================================================== entry point
def render_html(report: Report, *, standalone: bool = True) -> str:
    """The whole report as HTML.

    ``standalone=True`` gives a complete document to open locally. ``standalone=False``
    gives a fragment for publishing as an Artifact: it starts with ``<title>`` and
    ``<style>`` and has no doctype/html/head/body tags (the host wraps it).
    """
    title = f"<title>{_esc(report_title(report))}</title>"
    style = f"<style>{build_css(standalone)}</style>"
    body = _body_html(report)
    if not standalone:
        return f"{title}\n{style}\n{_FONT_LINK}\n{body}\n{_SCRIPT}\n"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{title}\n{_FONT_LINK}\n{style}\n</head>\n<body>\n{body}\n{_SCRIPT}\n</body>\n</html>\n"
    )
