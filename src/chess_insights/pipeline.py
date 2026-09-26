"""Run every analysis module and assemble the Report.

    games + evals ─ run_modules ─ coach.build_coaching ─ build_report (fill, rank, plan) ─ coach.finish_coaching
                                                                     └─ the same per format: Report.format_reports

Every step is isolated: a failing module, coaching step or format view logs a warning and the report still builds.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, get_args

from .context import AnalysisContext
from .fill import fill
from .insights import build_study_plan, headline, practice_actions, rank_insights, summary_lines
from .models import Coaching, Game, GameEval, Insight, InsightKind, ModuleResult, Report
from .visuals import FORMAT_ORDER, format_counts, format_text

if TYPE_CHECKING:
    from .coach import CoachConfig

log = logging.getLogger(__name__)

# A format gets its own view of the report (Report.format_reports) from this many games, when the games cover two
# or more formats. Below it a view would have too few games for most tests to say anything (the modules need 8
# games per opening and colour, 25 per split), and its sections would be mostly "not enough data".
# (visuals.MIN_FORMAT_GAMES is the much smaller bar for a format's own bar in a chart split by format.)
MIN_FORMAT_GAMES = 60

# (module path under chess_insights.analysis, section title used if the module crashes)
MODULES: list[tuple[str, str]] = [
    ("results", "Results & rating"),
    ("openings", "Openings"),
    ("time_mgmt", "Clock & time management"),
    ("endings", "How your games end"),
    ("habits", "Habits & tilt"),
    ("engine_stats", "Engine review"),
    ("mistakes", "Positions you keep getting wrong"),
]


_KINDS = frozenset(get_args(InsightKind))


def _finite(value: Any) -> float:
    """``value`` as a finite float; ValueError for None, pd.NA, NaN, inf, strings ..."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"not a number: {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"not a finite number: {value!r}")
    return number


def _check_insight(ins: Any) -> Insight:
    """Validate (and normalise in place) one insight so ranking and rendering can't crash on it.

    severity / confidence become plain floats (a numpy or pandas scalar is fine; None, pd.NA, NaN
    are not: NaN would even rank first, since min(1.0, nan) is 1.0).
    """
    if not isinstance(ins, Insight):
        raise ValueError(f"not an Insight: {type(ins).__name__}")
    if ins.kind not in _KINDS:
        raise ValueError(f"unknown kind {ins.kind!r}")
    for name in ("id", "category", "title"):
        if not isinstance(getattr(ins, name), str) or not getattr(ins, name):
            raise ValueError(f"{name} must be a non-empty string, got {getattr(ins, name)!r}")
    ins.severity = _finite(ins.severity)
    ins.confidence = _finite(ins.confidence)
    ins.detail = "" if ins.detail is None else str(ins.detail)
    ins.study = [str(a) for a in (ins.study or []) if a]
    ins.example_games = [str(u) for u in (ins.example_games or []) if u]
    if not isinstance(ins.evidence, dict):
        ins.evidence = {}
    return ins


def _clean_insights(result: ModuleResult, name: str) -> None:
    """Drop (with a warning) the insights of one module that would break ranking or rendering."""
    kept = []
    for ins in result.insights or []:
        try:
            kept.append(_check_insight(ins))
        except ValueError as exc:
            log.warning("analysis module %s: dropped insight %r: %s", name, getattr(ins, "id", ins), exc)
    result.insights = kept


def run_modules(ctx: AnalysisContext, modules: Optional[list[tuple[str, str]]] = None) -> list[ModuleResult]:
    """Run each module in isolation: one failing module never sinks the whole report."""
    results: list[ModuleResult] = []
    for name, title in modules or MODULES:
        try:
            mod = importlib.import_module(f"chess_insights.analysis.{name}")
            result = mod.analyze(ctx)
            if not isinstance(result, ModuleResult):
                raise TypeError(f"analyze() returned {type(result).__name__}, not ModuleResult")
            _clean_insights(result, name)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 — isolate module bugs
            log.warning("analysis module %s failed: %s", name, exc)
            log.debug("%s", traceback.format_exc())
            results.append(
                ModuleResult(
                    key=name,
                    title=title,
                    summary=f"This section could not be computed ({type(exc).__name__}: {exc}).",
                    stats={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
    return results


def _time_control_text(g: Game) -> str:
    if g.time_class == "daily":
        return "daily"
    if not g.base_seconds:
        return g.time_class
    base = g.base_seconds / 60
    return f"{base:g}+{g.increment}"


def game_label(g: Game, year: Optional[int] = None) -> str:
    """Link text for a game: 'Loss · 5+0 · vs 1512 · 12 Aug' (the year only when it differs from ``year``)."""
    result = {"win": "Win", "draw": "Draw", "loss": "Loss"}.get(g.outcome, g.outcome)
    opponent = f"vs {g.opp_rating}" if g.opp_rating else (f"vs {g.opponent}" if g.opponent else "")
    end = g.end_time.astimezone(timezone.utc) if g.end_time.tzinfo else g.end_time
    day = f"{end.day} {end:%b}" + (f" {end.year}" if year is not None and end.year != year else "")
    return " · ".join(p for p in (result, _time_control_text(g), opponent, day) if p)


def _diagram_links(d: Any) -> Iterator[Any]:
    """The game a board links to, and any links its strips or their frames carry."""
    if d is None:
        return
    yield getattr(d, "link", None)
    for strip in getattr(d, "strips", None) or []:
        yield getattr(strip, "link", None)
        for frame in getattr(strip, "frames", None) or []:
            yield getattr(frame, "link", None)


def _table_cells(table: Any) -> Iterator[Any]:
    for row in getattr(table, "rows", None) or []:
        if isinstance(row, (list, tuple)):
            yield from row


def linked_games(
    modules: Iterable[ModuleResult], coaching: Optional[Coaching] = None, extra: Iterable[str] = ()
) -> set[str]:
    """Every link the report shows that could be a game: findings (their games and board), each section's boards
    and tables, the coaching's explanations (their game, the games they repeat in, their board), endgame boards and
    tables, and ``extra`` (the study plan's games). Other links are harmless: only games get a label."""
    wanted: list[Any] = list(extra)
    for m in modules:
        for ins in m.insights or []:
            wanted += ins.example_games or []
            wanted += _diagram_links(getattr(ins, "diagram", None))
            wanted += _table_cells(getattr(getattr(ins, "chart", None), "table", None))
        for d in getattr(m, "diagrams", None) or []:
            wanted += _diagram_links(d)
        for table in [*(m.tables or []), *(getattr(c, "table", None) for c in m.charts or [])]:
            wanted += _table_cells(table)
    if coaching is not None:
        for e in getattr(coaching, "explanations", None) or []:
            wanted += [getattr(e, "game_url", None), *(getattr(e, "games", None) or [])]
            wanted += _diagram_links(getattr(e, "diagram", None))
        for d in getattr(coaching, "endgame_diagrams", None) or []:
            wanted += _diagram_links(d)
        for name in ("endgames", "theory_exit", "motif_profile"):
            wanted += _table_cells(getattr(coaching, name, None))
    return {u for u in wanted if isinstance(u, str) and u}


def game_labels(
    games: list[Game],
    modules: list[ModuleResult],
    extra: Iterable[str] = (),
    *,
    coaching: Optional[Coaching] = None,
) -> dict[str, str]:
    """Labels for every game the report links to (``linked_games``): "Loss · 5+0 · vs 1512 · 12 Aug"."""
    wanted = linked_games(modules, coaching, extra)
    year = max((g.end_time.year for g in games), default=None)
    return {g.url: game_label(g, year) for g in games if g.url and g.url in wanted}


def _practice(modules: list[ModuleResult], coaching: Optional[Coaching]) -> list[str]:
    """``insights.practice_actions``, with the coaching when it takes it (its drills become plan actions)."""
    try:
        params = inspect.signature(practice_actions).parameters
    except (TypeError, ValueError):  # a builtin or a mock without a signature
        params = {}  # type: ignore[assignment]
    if "coaching" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return practice_actions(modules, coaching=coaching)
    return practice_actions(modules)


def build_report(
    ctx: AnalysisContext,
    modules: list[ModuleResult],
    *,
    filters: str = "",
    engine_note: str = "",
    top_n: Optional[int] = None,
    coaching: Optional[Coaching] = None,
    practice_coaching: Optional[Coaching] = None,
    time_class: str = "",
) -> Report:
    """``top_n``: how many strengths and weaknesses the report lists (None: every one that passed the rule).

    Every finding first gets its formats and a picture where its module gave none (``fill``). ``coaching`` goes
    on the report; the study plan's practice actions come from ``practice_coaching`` (default: ``coaching``), so a
    format view can point at drills that live on the main report. ``time_class`` marks a one-format view.
    """
    fill(modules, ctx)
    strengths, weaknesses = rank_insights(modules, top_n=top_n)
    practice = _practice(modules, practice_coaching if practice_coaching is not None else coaching)
    plan = build_study_plan(weaknesses, practice=practice)
    games = ctx.games
    return Report(
        username=ctx.username,
        generated_at=datetime.now(timezone.utc),
        filters=filters,
        n_games=len(games),
        date_from=games[0].end_time if games else None,
        date_to=games[-1].end_time if games else None,
        modules=modules,
        strengths=strengths,
        weaknesses=weaknesses,
        study_plan=plan,
        engine_note=engine_note,
        headline=headline(strengths, weaknesses, plan),
        summary_lines=summary_lines(modules, plan, strengths, weaknesses),
        game_labels=game_labels(games, modules, (u for item in plan for u in item.games), coaching=coaching),
        demo=bool(ctx.opt("demo", False)),
        time_class=time_class,
        formats=format_counts(games),
        engine_formats=format_counts(g for g in games if g.game_id in ctx.evals),
        coaching=coaching,
    )


# --------------------------------------------------------------------------- coaching
def _coaching_failed(exc: Exception, what: str) -> str:
    log.warning("%s failed: %s", what, exc)
    log.debug("%s", traceback.format_exc())
    return f"The coaching could not be computed ({type(exc).__name__}: {exc})."


def build_coaching(ctx: AnalysisContext, modules: list[ModuleResult], cfg: "CoachConfig") -> Coaching:
    """``coach.build_coaching``, isolated like a module: a failure leaves a Coaching with a note, never an error.

    The coaching may add findings to a section (only motif findings that passed the claim rule); they are
    checked like every module's before ranking.
    """
    try:
        from . import coach

        coaching = coach.build_coaching(ctx, modules, cfg)
        if not isinstance(coaching, Coaching):
            raise TypeError(f"build_coaching() returned {type(coaching).__name__}, not Coaching")
    except Exception as exc:  # noqa: BLE001 — the report must build without its coaching
        coaching = Coaching(notes=[_coaching_failed(exc, "coaching")])
    for m in modules:
        _clean_insights(m, m.key)
    return coaching


def finish_coaching(report: Report, cfg: "CoachConfig") -> None:
    """``coach.finish_coaching`` (progress, Maia, the LLM coach), isolated: a failure becomes a note."""
    try:
        from . import coach

        coach.finish_coaching(report, cfg)
    except Exception as exc:  # noqa: BLE001
        note = _coaching_failed(exc, "finishing the coaching")
        if report.coaching is not None:
            report.coaching.notes.append(note)


def view_coaching(coaching: Optional[Coaching], time_class: str) -> Optional[Coaching]:
    """A format view's copy of the coaching: that format's explanations only. Drills, the motif profile and the
    review schedule stay on the main report (they are practice for every format)."""
    if coaching is None:
        return None
    return Coaching(
        explanations=[e for e in coaching.explanations if e.time_class == time_class],
        notes=list(coaching.notes),
        settings=dict(coaching.settings),
        llm=dict(coaching.llm),
    )


# --------------------------------------------------------------------------- format views
def view_filters(time_class: str, filters: str) -> str:
    """"blitz only, rated, standard chess" from the report's filter text (its time-control part replaced)."""

    def is_time_part(part: str) -> bool:
        return part == "all time controls" or all(p in FORMAT_ORDER for p in part.split("+"))

    parts = [p for p in (filters or "").split(", ") if p and not is_time_part(p)]
    return ", ".join([format_text({time_class: 1}), *parts])


def view_engine_note(time_class: str, view: AnalysisContext, whole: AnalysisContext, engine_note: str) -> str:
    """The engine note for one format's view: how many of its games Stockfish analysed (or why none)."""
    if not whole.evals:
        return engine_note  # not run, or skipped: the same reason holds for every format
    if not view.evals:
        sample = engine_note.strip().rstrip(".")
        note = f"None of your {time_class} games were in the engine sample" + (f" ({sample})." if sample else ".")
        listed = [tc for tc in whole.opt("engine_time_classes", None) or [] if isinstance(tc, str)]
        if listed and time_class not in {tc.lower() for tc in listed}:
            sent = format_text({tc.lower(): 1 for tc in listed}).removesuffix(" only")
            note += f" Only {sent} games were sent to Stockfish."
        elif whole.opt("engine_sample", "recent") != "balanced":
            note += (" A balanced engine sample (--engine-sample balanced, the default on GitHub) takes games from "
                     "every format.")
        elif time_class == "daily":
            note += " A balanced sample takes daily games only when asked to (--engine-time-class)."
        return note
    first = next(iter(view.evals.values()))
    depth = f" at depth {first.depth}" if first.depth else ""
    k, n = len(view.evals), len(view.games)
    if k >= n:
        return f"{first.engine}{depth} on all {n} of your {time_class} games."
    # both samples take a format's games newest first
    return f"{first.engine}{depth} on {k} of your {n} {time_class} games (the most recent)."


def build_format_views(
    ctx: AnalysisContext,
    modules: Optional[list[tuple[str, str]]] = None,
    *,
    filters: str = "",
    engine_note: str = "",
    coaching: Optional[Coaching] = None,
    min_games: int = MIN_FORMAT_GAMES,
) -> dict[str, Report]:
    """The same analysis on each format's games alone, for every format with at least ``min_games`` games.

    Only when the games cover two or more formats (a one-format report is already its own view). Each view is a
    full Report: its claims are tested on that format's games only (its own false-claim budget), its engine
    sections use that format's engine-analysed games, and it has no views of its own. Keys in format order.
    """
    counts = format_counts(ctx.games)
    if len(counts) < 2:
        return {}
    views: dict[str, Report] = {}
    for tc, n in counts.items():
        if n < min_games:
            continue
        games = [g for g in ctx.games if g.time_class == tc]
        ids = {g.game_id for g in games}
        sub = AnalysisContext(
            username=ctx.username,
            games=games,
            evals={gid: ev for gid, ev in ctx.evals.items() if gid in ids},
            options=dict(ctx.options),
        )
        try:
            views[tc] = build_report(
                sub,
                run_modules(sub, modules),
                filters=view_filters(tc, filters),
                engine_note=view_engine_note(tc, sub, ctx, engine_note),
                coaching=view_coaching(coaching, tc),
                practice_coaching=coaching,
                time_class=tc,
            )
        except Exception as exc:  # noqa: BLE001 — a view is extra; the main report stands without it
            log.warning("the %s view could not be built: %s", tc, exc)
            log.debug("%s", traceback.format_exc())
    return views


def run_analysis(
    games: list[Game],
    username: str,
    *,
    evals: Optional[dict[str, GameEval]] = None,
    options: Optional[dict[str, Any]] = None,
    filters: str = "",
    engine_note: str = "",
    modules: Optional[list[tuple[str, str]]] = None,
    coach: Optional["CoachConfig"] = None,
) -> Report:
    """Games (already filtered) -> Report.

    ``coach``: run the coaching layer (explanations, drills ...) with these settings; None skips it.
    ``options["format_views"]``: False skips the per-format views (default: built whenever two or more formats
    are present, each from ``MIN_FORMAT_GAMES`` games).
    """
    ctx = AnalysisContext(
        username=username,
        games=sorted(games, key=lambda g: (g.end_time, g.game_id)),  # ties: same order whatever the source order
        evals=evals or {},
        options=options or {},
    )
    results = run_modules(ctx, modules)
    coaching = build_coaching(ctx, results, coach) if coach is not None else None
    report = build_report(ctx, results, filters=filters, engine_note=engine_note, coaching=coaching)
    if coach is not None:
        finish_coaching(report, coach)
    if ctx.opt("format_views", True):
        report.format_reports = build_format_views(
            ctx, modules, filters=filters, engine_note=engine_note, coaching=report.coaching
        )
    return report
