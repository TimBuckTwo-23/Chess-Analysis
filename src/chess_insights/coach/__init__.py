"""The coaching layer: explain positions and route to practice, without adding claims.

    games + evals + ranked findings
        │
        ├─ critical.select_critical   positions worth explaining (your errors, repeated mistakes, choice points)
        ├─ deep.analyse_positions     Stockfish re-search: best line + the refutation of your move (own cache)
        ├─ motifs.detect_line         tactical patterns along both lines (Lichess theme names)
        ├─ concepts                   Stockfish 16 classical eval terms at the two line ends, board facts
        ├─ sources/*, openings_info   opening explorer, cloud eval, tablebase, opening names, Wikibooks (C3)
        ├─ explain                    two or three plain sentences + the board with both lines
        ├─ deep.profile_lines         short lines for every error, both sides (motif profile, puzzle export)
        ├─ puzzles.fill_puzzle_lines  your errors' best lines and motif themes for the --puzzles PGN
        ├─ profile, drills            motif profile (you vs opponents), puzzle packs, review schedule (C2)
        └─ finish_coaching            after ranking: progress vs the last report, Maia, the LLM coach (C4)

Rules: never create or change an Insight here (the one exception, motif claims, goes through the claim rule in
``profile.motif_claims`` and is added to the Engine review section); every step may fail on its own and leaves
a note instead; the report builds with ``--offline`` and no keys.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Optional

from ..models import Coaching, ModuleResult, Report
from .config import CoachConfig

if TYPE_CHECKING:
    from ..context import AnalysisContext

log = logging.getLogger(__name__)

__all__ = ["CoachConfig", "build_coaching", "finish_coaching"]


def _step(coaching: Coaching, name: str, fn: Callable[[], object]) -> object:
    """Run one step; a failure becomes a note (and a debug trace), never an exception."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — one failing source must not sink the coaching
        log.warning("coaching step %s failed: %s", name, exc)
        log.debug("%s", traceback.format_exc())
        coaching.notes.append(f"{name} skipped ({type(exc).__name__}: {exc}).")
        return None


def build_coaching(ctx: "AnalysisContext", modules: list[ModuleResult], cfg: Optional[CoachConfig] = None) -> Coaching:
    """Explanations, drills and the review schedule for the analysed games. Runs before ranking.

    May append tables, charts and diagrams to ``modules`` (the motif profile goes into the Engine review section,
    opening facts under the openings tables) and, through ``profile.motif_claims`` only, motif findings to the
    Engine review section. Never raises.
    """
    from . import critical, deep, drills, endgames, explain, openings_info, profile, puzzles

    cfg = cfg or CoachConfig()
    coaching = Coaching(settings={"depth": cfg.depth, "max_positions": cfg.max_positions, "multipv": cfg.multipv})
    today = cfg.today or datetime.now(timezone.utc).date()
    if not ctx.evals:
        coaching.notes.append("Coaching needs the engine analysis (--engine): nothing to explain yet.")
        # what needs no engine: opening names, theory exit and database facts, and the openings drill pack
        _step(coaching, "Opening facts", lambda: openings_info.annotate(ctx, coaching, modules, cfg))
        _step(coaching, "Drills", lambda: drills.annotate(ctx, coaching, modules, cfg, today))
        return coaching

    positions = _step(
        coaching, "Critical positions", lambda: critical.select_critical(ctx, modules, cfg.max_positions)
    ) or []
    coaching.settings["positions"] = len(positions)
    lines = _step(coaching, "Deep analysis", lambda: deep.analyse_positions(positions, cfg, coaching.notes)) or {}
    engine_name = next((r.engine for r in lines.values() if r.engine), "")
    if engine_name:
        coaching.settings["engine"] = engine_name
    explanations = _step(
        coaching, "Explanations", lambda: explain.explain_all(ctx, positions, lines, cfg, coaching.notes)
    ) or []
    coaching.explanations = list(explanations)
    coaching.settings["explained"] = len(coaching.explanations)

    # Your errors' best lines for the puzzle export: from the deep pass, else the profile pass (which the motif
    # profile below reuses: deep.profile_lines keeps its result for the run).
    profile_errors = (_step(coaching, "Motif lines", lambda: deep.profile_lines(ctx, cfg, coaching.notes)) or []
                      if cfg.profile else [])
    _step(coaching, "Puzzle lines", lambda: puzzles.fill_puzzle_lines(ctx, coaching, lines, profile_errors))

    _step(coaching, "Opening facts", lambda: openings_info.annotate(ctx, coaching, modules, cfg))
    _step(coaching, "Endgames", lambda: endgames.annotate(ctx, coaching, cfg))

    if cfg.profile:
        _step(coaching, "Motif profile", lambda: profile.annotate(ctx, coaching, modules, cfg))
    _step(coaching, "Drills", lambda: drills.annotate(ctx, coaching, modules, cfg, today))
    return coaching


def finish_coaching(report: Report, cfg: Optional[CoachConfig] = None) -> Report:
    """After ranking: progress against the previous report, Maia move probabilities, the LLM coach. Never raises."""
    from . import llm, maia, progress

    cfg = cfg or CoachConfig()
    coaching = report.coaching
    if coaching is None:
        return report
    _step(coaching, "Progress", lambda: progress.annotate(report, cfg))
    if cfg.maia:
        _step(coaching, "Maia", lambda: maia.annotate(report, cfg))
    if cfg.llm:
        _step(coaching, "LLM coach", lambda: llm.annotate(report, cfg))
    return report
