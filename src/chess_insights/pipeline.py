"""Run every analysis module and assemble the Report."""

from __future__ import annotations

import importlib
import logging
import math
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, get_args

from .context import AnalysisContext
from .insights import build_study_plan, headline, practice_actions, rank_insights, summary_lines
from .models import Game, GameEval, Insight, InsightKind, ModuleResult, Report

log = logging.getLogger(__name__)

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


def game_labels(games: list[Game], modules: list[ModuleResult], extra: Iterable[str] = ()) -> dict[str, str]:
    """Labels for every game a finding, a diagram or the study plan links to."""
    wanted = set(extra)
    for m in modules:
        for ins in m.insights or []:
            wanted.update(ins.example_games or [])
        for d in getattr(m, "diagrams", None) or []:
            if d.link:
                wanted.add(d.link)
    year = max((g.end_time.year for g in games), default=None)
    return {g.url: game_label(g, year) for g in games if g.url and g.url in wanted}


def build_report(
    ctx: AnalysisContext,
    modules: list[ModuleResult],
    *,
    filters: str = "",
    engine_note: str = "",
    top_n: Optional[int] = None,
) -> Report:
    """``top_n``: how many strengths and weaknesses the report lists (None: every one that passed the rule)."""
    strengths, weaknesses = rank_insights(modules, top_n=top_n)
    plan = build_study_plan(weaknesses, practice=practice_actions(modules))
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
        game_labels=game_labels(games, modules, (u for item in plan for u in item.games)),
        demo=bool(ctx.opt("demo", False)),
    )


def run_analysis(
    games: list[Game],
    username: str,
    *,
    evals: Optional[dict[str, GameEval]] = None,
    options: Optional[dict[str, Any]] = None,
    filters: str = "",
    engine_note: str = "",
    modules: Optional[list[tuple[str, str]]] = None,
) -> Report:
    """Games (already filtered) -> Report."""
    ctx = AnalysisContext(
        username=username,
        games=sorted(games, key=lambda g: (g.end_time, g.game_id)),  # ties: same order whatever the source order
        evals=evals or {},
        options=options or {},
    )
    return build_report(ctx, run_modules(ctx, modules), filters=filters, engine_note=engine_note)
