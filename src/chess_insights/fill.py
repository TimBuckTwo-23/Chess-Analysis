"""Fallbacks that give every finding what the report shows next to it: its formats and a picture.

The analysis modules set ``Insight.formats``, ``Insight.chart`` and ``Insight.diagram`` where they know best;
these fallbacks only fill what a module left empty, from facts it already recorded:

* ``fill_formats``: the time classes of the games the module analysed (the engine-analysed games for the engine
  sections), narrowed to the format(s) a finding names in its evidence ("time_class", "time_classes", a rating
  pool such as "Blitz").
* ``fill_visuals``: a small chart built from the finding's evidence (your score against the rating's
  expectation, your rate against your opponents' in the same games ...), or the board for a finding about one
  position. Evidence that fits none of the patterns below leaves the finding without a picture rather than
  with one that says something the numbers don't.

Neither changes which findings exist, their kind, severity, confidence or wording. ``pipeline.build_report``
calls both for every report, the per-format views included.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence

from .models import TIME_CLASSES, Chart, Diagram, Game, Insight, ModuleResult
from .stats import MINUS, per100_games
from .visuals import FORMAT_NAMES, comparison_chart, format_counts, ordered_formats, position_diagram

if TYPE_CHECKING:
    from .context import AnalysisContext

log = logging.getLogger(__name__)

# Sections whose findings come from the engine-analysed games only.
ENGINE_MODULES = frozenset({"engine_stats", "mistakes"})

YOU, OPPONENTS = "You", "Your opponents"


# --------------------------------------------------------------------------- formats
def module_games(module_key: str, ctx: "AnalysisContext") -> list[Game]:
    """The games a module's findings rest on: the engine-analysed ones for the engine sections, else all."""
    if module_key in ENGINE_MODULES:
        return [g for g in ctx.games if g.game_id in ctx.evals]
    return list(ctx.games)


def _time_class(value: Any) -> Optional[str]:
    """"blitz", "Blitz", "Blitz (chess960)" -> "blitz"; anything else -> None."""
    if not isinstance(value, str):
        return None
    words = value.strip().lower().replace("(", " ").split()
    return words[0] if words and words[0] in TIME_CLASSES else None


def named_formats(evidence: dict[str, Any]) -> list[str]:
    """The time classes a finding's evidence is about ("time_class", "time_classes", a rating "pool"), in order."""
    names: list[str] = []
    candidates: list[Any] = [evidence.get("time_class")]
    if isinstance(evidence.get("time_classes"), (list, tuple)):
        candidates += list(evidence["time_classes"])
    pool = evidence.get("pool")
    candidates.append(pool.get("pool") if isinstance(pool, dict) else pool)
    for value in candidates:
        tc = _time_class(value)
        if tc and tc not in names:
            names.append(tc)
    return names


def _count(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _games_behind(evidence: dict[str, Any], tc: str, single: bool) -> Optional[int]:
    """The number of games the evidence itself gives for format ``tc``, if any."""
    nested = evidence.get(tc)
    if isinstance(nested, dict) and _count(nested.get("n")):
        return _count(nested.get("n"))
    pool = evidence.get("pool")
    if isinstance(pool, dict) and _time_class(pool.get("pool")) == tc:
        return _count(pool.get("n"))
    return _count(evidence.get("n")) if single else None


def fill_formats(modules: Iterable[ModuleResult], ctx: "AnalysisContext") -> None:
    """Give every finding without ``formats`` the format mix behind it (see the module docstring)."""
    for m in modules:
        if not m.insights:
            continue
        counts = format_counts(module_games(m.key, ctx))
        for ins in m.insights:
            if ins.formats:
                continue
            named = named_formats(ins.evidence or {})
            formats = {}
            for tc in named:
                n = _games_behind(ins.evidence, tc, single=len(named) == 1) or counts.get(tc)
                if n:
                    formats[tc] = n
            ins.formats = ordered_formats(formats) if formats else dict(counts)


# --------------------------------------------------------------------------- visuals
def _num(value: Any) -> Optional[float]:
    """A finite number (numpy scalars included); None for anything else, text and booleans too."""
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _share(value: Any) -> Optional[float]:
    x = _num(value)
    return x if x is not None and 0.0 <= x <= 1.0 else None


def _ratio(part: Any, whole: Any) -> Optional[float]:
    p, w = _num(part), _num(whole)
    return p / w if p is not None and w and w > 0 and 0 <= p <= w else None


def _games_note(n: Any, extra: str = "") -> str:
    count = _count(n)
    parts = [f"{count} games" if count else "", extra]
    return "; ".join(p for p in parts if p)


def _expected_label(evidence: dict[str, Any]) -> str:
    if evidence.get("expected_kind") == "attenuated":
        return "Expected (allowing for rating noise)"
    return "Expected from ratings"


def _score_of(d: dict[str, Any], prefix: str = "") -> Optional[float]:
    return _share(d.get(f"{prefix}rated_score")) if f"{prefix}rated_score" in d else _share(d.get(f"{prefix}score"))


# evidence key prefix -> the games it is about (for the chart title)
_SCORE_PREFIXES = {"": "", "after_two_losses_": "After two losses in a row: "}


def _score_chart(ins: Insight) -> Optional[Chart]:
    """score vs expected (also ``after_two_losses_score`` ...): "You" vs "Expected from ratings"."""
    ev = ins.evidence
    for prefix, games in _SCORE_PREFIXES.items():
        score, expected = _score_of(ev, prefix), _share(ev.get(f"{prefix}expected"))
        if score is None or expected is None:
            continue
        n = ev.get(f"{prefix}n_rated", ev.get(f"{prefix}n"))
        rest = _num(ev.get("rest_delta"))
        extra = f"your other games: {per100_games(rest)} vs your rating" if rest is not None and not prefix else ""
        return comparison_chart(
            f"{games}your score vs what your ratings predict" if games else "Your score vs what your ratings predict",
            [YOU, _expected_label(ev)],
            [("Score", [score, expected])],
            value_format="pct",
            note=_games_note(n, extra),
        )
    return None


_GROUP_LABELS = {"white": "As White", "black": "As Black", "rest": "Your other games"}


def _grouped_score_chart(ins: Insight) -> Optional[Chart]:
    """Nested score summaries ({"white": {...}, "black": {...}} or {"pool": {...}, "rest": {...}})."""
    labels, mine, expected, counts = [], [], [], []
    for key, d in ins.evidence.items():
        if not isinstance(d, dict):
            continue
        score, exp = _score_of(d), _share(d.get("expected"))
        if score is None or exp is None:
            continue
        if key == "pool":
            label = str(d.get("pool") or "These games")
        elif key == "rest" and isinstance(ins.evidence.get("pool"), dict):
            label = "Your other time controls"
        else:
            label = _GROUP_LABELS.get(key, str(key).replace("_", " ").capitalize())
        labels.append(label)
        mine.append(score)
        expected.append(exp)
        if (n := _count(d.get("n_rated", d.get("n")))) is not None:
            counts.append(f"{label}: {n} games")
    if not labels:
        return None
    return comparison_chart(
        "Your score vs what your ratings predict",
        labels,
        [(YOU, mine), (_expected_label(ins.evidence), expected)],
        value_format="pct",
        note="; ".join(counts),
    )


# (your key, your opponents' key, value format): the same measure for both players in the same games.
_PAIRS = {
    "per100": ("per100", "opp_per100", "float1"),
    "rest_per100": ("rest_per100", "opp_rest_per100", "float1"),
    "other_errors": ("other_errors_per100", "opp_other_errors_per100", "float1"),
    "per100_low": ("per100_low", "opp_per100_low", "float1"),
    "per100_ok": ("per100_ok", "opp_per100_ok", "float1"),
    "rate": ("rate", "opp_rate", "pct"),
    "trouble": ("trouble_rate", "opponent_trouble_rate", "pct"),
    "opening_share": ("your_share", "opponent_share", "pct"),
    "missed_mates": ("missed_mates", "opp_missed_mates", "int"),
    "allowed_mates": ("allowed_mates", "opp_allowed_mates", "int"),
    "abandoned": ("abandoned", "opponent_abandoned", "int"),
}
# (chart title, [(pair row, bar label)]): the first chart whose rows are all in the evidence is drawn.
# "{phase}" is the evidence's game phase.
_PAIR_CHARTS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Blunders per 100 moves, you vs your opponents", [("per100_low", "Short of time"), ("per100_ok", "With time left")]),
    ("Mistakes and blunders per 100 moves, you vs your opponents", [("per100", "{phase}"), ("rest_per100", "Other phases")]),
    ("Errors per 100 moves, you vs your opponents", [("per100", "This kind of error"), ("other_errors", "Other errors")]),
    ("Blunders per 100 moves, you vs your opponents", [("per100", "Blunders per 100 moves")]),
    ("Converting winning positions, you vs your opponents", [("rate", "Winning positions won")]),
    ("Time trouble, you vs your opponents", [("trouble", "Games in time trouble")]),
    ("Opening pace, you vs your opponents", [("opening_share", "Clock used on the first moves")]),
    ("Mates, you vs your opponents", [("missed_mates", "Mates missed"), ("allowed_mates", "Mates allowed")]),
    ("Abandoned games, you vs your opponents", [("abandoned", "Games abandoned")]),
]


def _pair_values(ev: dict[str, Any], row: str) -> Optional[tuple[float, float, str]]:
    mine_key, theirs_key, fmt = _PAIRS[row]
    convert = _share if fmt == "pct" else _num
    mine, theirs = convert(ev.get(mine_key)), convert(ev.get(theirs_key))
    if mine is None or theirs is None:
        return None
    return mine, theirs, fmt


def _pair_chart(ins: Insight) -> Optional[Chart]:
    """Your rate against your opponents' in the same games (grouped where the evidence has several)."""
    ev = ins.evidence
    for title, rows in _PAIR_CHARTS:
        values = [_pair_values(ev, row) for row, _ in rows]
        if any(v is None for v in values):
            continue
        phase = str(ev.get("phase") or "This phase").capitalize()
        return comparison_chart(
            title,
            [label.format(phase=phase) for _, label in rows],
            [(YOU, [v[0] for v in values]), (OPPONENTS, [v[1] for v in values])],  # type: ignore[index]
            value_format=values[0][2],  # type: ignore[index]
            note=_games_note(ev.get("games", ev.get("n"))),
        )
    return None


def _per_format_pair_chart(ins: Insight) -> Optional[Chart]:
    """One finding over several formats ({"time_classes": [...], "blitz": {...}, "rapid": {...}}): a bar each."""
    ev = ins.evidence
    classes = [tc for tc in ev.get("time_classes") or [] if isinstance(ev.get(tc), dict)]
    if not classes:
        return None
    labels = [FORMAT_NAMES.get(tc, tc) for tc in classes]
    note = ", ".join(f"{FORMAT_NAMES.get(tc, tc)}: {n} games" for tc in classes if (n := _count(ev[tc].get("n"))))
    for title, rows in _PAIR_CHARTS:
        if len(rows) != 1:
            continue
        row, label = rows[0]
        values = [_pair_values(ev[tc], row) for tc in classes]
        if any(v is None for v in values):
            continue
        return comparison_chart(
            f"{label} by format, you vs your opponents",
            labels,
            [(YOU, [v[0] for v in values]), (OPPONENTS, [v[1] for v in values])],  # type: ignore[index]
            value_format=values[0][2],  # type: ignore[index]
            note=note,
        )
    scores = [(_score_of(ev[tc]), _share(ev[tc].get("expected"))) for tc in classes]
    if all(s is not None and e is not None for s, e in scores):
        return comparison_chart(
            "Your score vs what your ratings predict, by format",
            labels,
            [(YOU, [s for s, _ in scores]), ("Expected from ratings", [e for _, e in scores])],
            value_format="pct",
            note=note,
        )
    return None


def _delta_chart(ins: Insight) -> Optional[Chart]:
    """Score vs your rating in these games and in your other games (evidence "delta" and "earlier_delta")."""
    ev = ins.evidence
    delta, other = _num(ev.get("delta")), _num(ev.get("earlier_delta"))
    if delta is None or other is None:
        return None
    first = _count(ev.get("from_game"))
    labels = [f"From game {first} of a session on", "Earlier games"] if first else ["These games", "Your other games"]
    return comparison_chart(
        "Score vs your rating",
        labels,
        [("Score vs your rating", [delta, other])],
        value_format="signed_pct",
        reference=0.0,
        note=_games_note(ev.get("n"), "points per 100 games above (+) or below (−) what your ratings predict"),
    )


def _share_chart(ins: Insight) -> Optional[Chart]:
    """Two shares the finding's test compares (quick losses vs quick wins, mated in losses vs mating in wins ...)."""
    ev = ins.evidence
    specs = [
        ("Games ending in checkmate", "Ended in checkmate", ("mated", "losses"), ("mating", "wins"),
         ("Your losses", "Your wins")),
        ("Games over quickly", "Over quickly", ("short_losses", "losses"), ("short_wins", "wins"),
         ("Your losses", "Your wins")),
    ]
    for title, series, (a, a_of), (b, b_of), labels in specs:
        first, second = _ratio(ev.get(a), ev.get(a_of)), _ratio(ev.get(b), ev.get(b_of))
        if first is not None and second is not None:
            note = f"{int(_num(ev[a_of]) or 0)} losses, {int(_num(ev[b_of]) or 0)} wins"
            return comparison_chart(title, list(labels), [(series, [first, second])], value_format="pct", note=note)
    moves, blunders = _ratio(ev.get("moves"), ev.get("total_moves")), _ratio(ev.get("blunders"), ev.get("total_blunders"))
    if ev.get("piece") and moves is not None and blunders is not None:
        return comparison_chart(
            f"{str(ev['piece']).capitalize()} moves",
            ["Share of your moves", "Share of your blunders"],
            [("Share", [moves, blunders])],
            value_format="pct",
            note=_games_note(None, f"{int(_num(ev['total_blunders']) or 0)} blunders in {int(_num(ev['total_moves']) or 0)} moves"),
        )
    return None


def _flag_chart(ins: Insight) -> Optional[Chart]:
    """Games lost on time against games won on time (the flag balance the clock findings test)."""
    ev = ins.evidence
    lost, won = _num(ev.get("lost_on_time")), _num(ev.get("won_on_time"))
    if lost is None or won is None:
        return None
    return comparison_chart(
        "Games decided on time",
        ["You lost on time", "You won on time"],
        [("Games", [lost, won])],
        value_format="int",
        note=_games_note(ev.get("n")),
    )


def _accuracy_chart(ins: Insight) -> Optional[Chart]:
    """Engine accuracy in wins / draws / losses (a line at your opponents' average), or chess.com's accuracy."""
    ev = ins.evidence
    by_result = [(label, _num(ev.get(key))) for key, label in (("win", "Wins"), ("draw", "Draws"), ("loss", "Losses"))]
    by_result = [(label, v) for label, v in by_result if v is not None]
    if by_result and "opponents" in ev:
        opp = _num(ev.get("opponents"))
        return comparison_chart(
            "Your engine accuracy by result",
            [label for label, _ in by_result],
            [("Your accuracy", [v for _, v in by_result])],
            value_format="float1",
            reference=opp,
            note=_games_note(ev.get("games"), "the line is your opponents' average" if opp is not None else ""),
        )
    mine, opp = _num(ev.get("mine")), _num(ev.get("opponents"))
    if mine is not None and opp is not None:
        rows = [(YOU, mine)]
        rows += [(label, v) for key, label in (("wins", "In your wins"), ("losses", "In your losses"))
                 if (v := _num(ev.get(key))) is not None]
        rows.append((OPPONENTS, opp))
        return comparison_chart(
            "chess.com accuracy in reviewed games",
            [label for label, _ in rows],
            [("Accuracy", [v for _, v in rows])],
            value_format="float1",
            note=_games_note(ev.get("n")),
        )
    return None


def _by_format_rate_chart(ins: Insight) -> Optional[Chart]:
    """A rate per format ({"by_time_class": {"blitz": 0.08, ...}}), with the overall rate as a line."""
    ev = ins.evidence
    by = ev.get("by_time_class")
    if not isinstance(by, dict):
        return None
    rates = {tc: v for tc, v in ((tc, _share(v)) for tc, v in by.items()) if v is not None}
    if not rates:
        return None
    order = list(ordered_formats({tc: 1 for tc in rates}))
    what = "Draw rate" if ins.category == "endings" and "draws" in ev else "Rate"
    several = len(order) > 1
    return comparison_chart(
        f"{what} by format" if several else what,
        [FORMAT_NAMES.get(tc, tc) for tc in order],
        [(what, [rates[tc] for tc in order])],
        value_format="pct",
        reference=_share(ev.get("rate")) if several else None,
        note=_games_note(ev.get("n"), "the line is your rate over all formats" if several else ""),
    )


def _opening_eval_chart(ins: Insight) -> Optional[Chart]:
    """Stockfish's average evaluation after the opening, in pawns from your side (0 = equal)."""
    ev = ins.evidence
    avg = _num(ev.get("avg_cp"))
    if avg is None or not ev.get("family"):
        return None
    return comparison_chart(
        "Evaluation after move 10",
        [str(ev["family"])],
        [("Pawns, from your side", [avg / 100.0])],
        value_format="signed_float2",
        reference=0.0,
        note=_games_note(ev.get("games"), "0 is an equal position"),
    )


def _rating_chart(ins: Insight) -> Optional[Chart]:
    """A rating trend: your rating at the start and at the end of the period (evidence "first" / "last")."""
    ev = ins.evidence
    first, last, pool = _num(ev.get("first")), _num(ev.get("last")), ev.get("pool")
    if first is None or last is None or not isinstance(pool, str):
        return None
    change = _num(ev.get("change"))
    trend = f"the fitted trend is {change:+.0f} points".replace("-", MINUS) if change is not None else ""
    return comparison_chart(
        f"Your {pool} rating over the period",
        ["First game", "Latest game"],
        [("Rating", [first, last])],
        value_format="rating",
        note=_games_note(ev.get("n"), trend),
    )


def _breadth_chart(ins: Insight) -> Optional[Chart]:
    """Repertoire breadth: how many different moves make up most of your games, per position."""
    ev = ins.evidence
    ks = ev.get("choices_for_coverage")
    if not isinstance(ks, dict) or not ks:
        return None
    rows = [(k, _count(v)) for k, v in ks.items()]
    rows = [(k, v) for k, v in rows if v]
    if not rows:
        return None
    labels = ["Your first move" if k in ("", "first move") else f"Against 1.{k}" for k, _ in rows]
    return comparison_chart(
        "Different moves that make up most of your games",
        labels,
        [("Moves", [float(v) for _, v in rows])],
        value_format="int",
        note=_games_note(ev.get("games"), "fewer moves = a narrower repertoire"),
    )


_CHARTS: Sequence[Callable[[Insight], Optional[Chart]]] = (
    _per_format_pair_chart,
    _grouped_score_chart,
    _score_chart,
    _delta_chart,
    _share_chart,
    _pair_chart,
    _flag_chart,
    _accuracy_chart,
    _by_format_rate_chart,
    _opening_eval_chart,
    _rating_chart,
    _breadth_chart,
)


def chart_from_evidence(ins: Insight) -> Optional[Chart]:
    """The first chart pattern the finding's evidence fits, or None."""
    if not isinstance(ins.evidence, dict):
        return None
    for build in _CHARTS:
        chart = build(ins)
        if chart is not None:
            return chart
    return None


def diagram_from_evidence(ins: Insight) -> Optional[Diagram]:
    """The board for a finding about one position (evidence "fen" and your "move"; the engine's "best" in green)."""
    ev = ins.evidence
    fen, move = ev.get("fen"), ev.get("move")
    if not isinstance(fen, str) or not fen or not isinstance(move, str) or not move:
        return None
    parts = fen.split()
    orientation = "black" if len(parts) > 1 and parts[1] == "b" else "white"
    best = ev.get("best") if isinstance(ev.get("best"), str) else None
    diagram = position_diagram(
        f"You played {move} here",
        fen,
        orientation=orientation,
        played=move,
        best=best,
        caption=f"Red: your move {move}." + (f" Green: Stockfish's {best}." if best else ""),
        link=ins.example_games[0] if ins.example_games else "",
    )
    return diagram if diagram.arrows else None


def fill_visuals(modules: Iterable[ModuleResult]) -> None:
    """Give every finding with neither a chart nor a diagram one built from its evidence (when one fits)."""
    for m in modules:
        for ins in m.insights or []:
            if ins.chart is not None or ins.diagram is not None:
                continue
            try:
                chart = chart_from_evidence(ins)
                if chart is not None:
                    ins.chart = chart
                else:
                    ins.diagram = diagram_from_evidence(ins)
            except Exception as exc:  # noqa: BLE001 — a picture is never worth a failed report
                log.warning("no fallback picture for %s: %s", ins.id, exc)


def fill(modules: Sequence[ModuleResult], ctx: "AnalysisContext") -> None:
    """Both fallbacks, each isolated: a failure logs a warning and leaves the findings as they were."""
    for name, step in (("formats", lambda: fill_formats(modules, ctx)), ("visuals", lambda: fill_visuals(modules))):
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            log.warning("filling in the findings' %s failed: %s", name, exc)
