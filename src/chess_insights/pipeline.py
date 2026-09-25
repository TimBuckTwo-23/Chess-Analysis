"""Run every analysis module and assemble the Report."""

from __future__ import annotations

import importlib
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

from .context import AnalysisContext
from .insights import build_study_plan, headline, rank_insights
from .models import Game, GameEval, ModuleResult, Report

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


def run_modules(ctx: AnalysisContext, modules: Optional[list[tuple[str, str]]] = None) -> list[ModuleResult]:
    """Run each module in isolation: one failing module never sinks the whole report."""
    results: list[ModuleResult] = []
    for name, title in modules or MODULES:
        try:
            mod = importlib.import_module(f"chess_insights.analysis.{name}")
            results.append(mod.analyze(ctx))
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


def build_report(
    ctx: AnalysisContext,
    modules: list[ModuleResult],
    *,
    filters: str = "",
    engine_note: str = "",
    top_n: int = 5,
) -> Report:
    strengths, weaknesses = rank_insights(modules, top_n=top_n)
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
        study_plan=build_study_plan(weaknesses),
        engine_note=engine_note,
        headline=headline(strengths, weaknesses),
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
        games=sorted(games, key=lambda g: g.end_time),
        evals=evals or {},
        options=options or {},
    )
    return build_report(ctx, run_modules(ctx, modules), filters=filters, engine_note=engine_note)
