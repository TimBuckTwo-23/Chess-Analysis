"""Pick the positions worth explaining (C1). Owner: coach-core.

Three kinds, in this order:

* **repeated**: the same wrong move in the same position in several of your games (``mistakes.find_repeated`` and
  ``mistakes.assess``, exactly as the "Positions you keep getting wrong" section finds them), most costly first
  (games x average win chance lost);
* **error**: every other move of yours that lost at least ``mistakes.MIN_DROP`` win-% points, costliest first;
* **choice**: your usual move at each choice point of your main lines (``openings.choice_positions``, the positions
  the openings section draws), so the report can show the engine's view of it. It counts as a mistake only where
  the openings section says so (``engine_mistake``): in the opening several moves are often about equally good.

Nothing here is tested or claimed: the list only decides which positions the deep analysis looks at.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Iterable, Sequence

import chess

from ..analysis import mistakes, openings
from ..models import CriticalPosition, Game, ModuleResult

if TYPE_CHECKING:
    from ..context import AnalysisContext

log = logging.getLogger(__name__)

CHOICE_SHARE = 0.2  # choice points get up to this share of the slots the repeated mistakes leave free
FAMILY_SHARE = 0.7  # a choice point is named after an opening family only when this share of its games has it


def _time_class(games: Iterable[Game]) -> str:
    """The format of the games when they all share one ("" = several formats, as in ``Diagram.time_class``)."""
    classes = {g.time_class for g in games}
    return classes.pop() if len(classes) == 1 else ""


def _standard(game: Game) -> bool:
    """Standard chess only: Chess960 castling differs, and the deep analysis reads plain FENs."""
    return game.rules == "chess"


def _insight_ids(modules: Sequence[ModuleResult]) -> dict[tuple[str, str], str]:
    """(EPD, UCI) -> the id of the "Positions you keep getting wrong" finding about that move in that position."""
    out: dict[tuple[str, str], str] = {}
    for m in modules:
        if m.key != mistakes.KEY:
            continue
        for ins in m.insights or []:
            evidence = ins.evidence or {}
            fen, san = evidence.get("fen"), evidence.get("move")
            if not fen or not san:
                continue
            try:
                board = chess.Board(fen)
                move = board.parse_san(san)
            except ValueError:
                continue
            out[(board.epd(), move.uci())] = ins.id
    return out


def _family(games: Sequence[Game]) -> str:
    """The opening family most of ``games`` share ("" when they spread over several: after 1.e4, 1...e5 leads to
    the Italian, the Ruy Lopez and the Scotch, and naming one of them would be wrong)."""
    counts = Counter(g.opening_family for g in games if g.opening_family)
    if not counts:
        return ""
    family, n = counts.most_common(1)[0]
    return family if n >= FAMILY_SHARE * len(games) else ""


def _repeated(
    ctx: "AnalysisContext", events: list[mistakes.MistakeEvent], ids: dict[tuple[str, str], str]
) -> list[CriticalPosition]:
    repeated = mistakes.assess(
        mistakes.find_repeated(events, ctx.games, ctx.evals), *mistakes.error_rate(ctx.games, ctx.evals)
    )
    # A follow-up folded into an earlier habit is described by that habit's finding, so it gets the same id.
    _, folded = mistakes.fold_follow_ups(repeated)
    for root in repeated:
        root_id = ids.get((root.epd, root.uci))
        for follow in folded.get(id(root), []) if root_id else []:
            ids.setdefault((follow.epd, follow.uci), root_id)
    out = []
    for r in sorted(repeated, key=lambda r: (-r.weight, r.epd, r.uci)):
        e = r.first
        if not _standard(e.game):
            continue
        out.append(
            CriticalPosition(
                game_id=e.game.game_id,
                url=e.game.url,
                ply=e.ply,
                fen=e.fen,
                played_uci=r.uci,
                played_san=r.san,
                move_label=mistakes.move_label(e.fen, r.san),
                best_san=r.best_san,
                drop=float(r.avg_drop),
                time_class=_time_class(r.played_in) if r.played_in else e.game.time_class,
                color=e.game.color,
                kind="repeated",
                repeats=r.errors,
                games=[g.url for g in r.played_in if g.url],  # most recent first
                insight_id=ids.get((r.epd, r.uci)),
                opening_family=e.game.opening_family or "",
                moves_before=list(e.game.moves_san[: e.ply]),
            )
        )
    return out


def _errors(events: list[mistakes.MistakeEvent], ids: dict[tuple[str, str], str]) -> list[CriticalPosition]:
    """Your single errors, costliest first (the most recent first on ties)."""
    out = []
    for e in sorted(events, key=lambda e: (-e.drop, -e.game.end_time.timestamp(), e.game.game_id, e.ply)):
        if not _standard(e.game) or not e.uci:
            continue
        out.append(
            CriticalPosition(
                game_id=e.game.game_id,
                url=e.game.url,
                ply=e.ply,
                fen=e.fen,
                played_uci=e.uci,
                played_san=e.san,
                move_label=e.move_label,
                best_san=e.best_san,
                drop=float(e.drop),
                time_class=e.game.time_class,
                color=e.game.color,
                kind="error",
                repeats=1,
                games=[e.game.url] if e.game.url else [],
                insight_id=ids.get((e.epd, e.uci)),
                opening_family=e.game.opening_family or "",
                moves_before=list(e.game.moves_san[: e.ply]),
            )
        )
    return out


def engine_mistake(pos: CriticalPosition, min_drop: float = openings.Thresholds.engine_min_drop) -> bool:
    """Whether the game analysis calls your move at a choice point an engine mistake, as the openings section does
    (``openings.ChoicePosition.best_by == "engine"`` with a better move than yours): Stockfish prefers another
    move and yours lost at least ``min_drop`` win-% points on average in the analysed games
    (``openings.Thresholds.engine_min_drop``). Below that several opening moves are about equally good, and the
    choice is not explained as an error. Errors and repeated mistakes are always mistakes."""
    if pos.kind != "choice":
        return True
    return bool(pos.best_san) and pos.best_san != pos.played_san and pos.drop > 0 and pos.drop >= min_drop


def _choices(ctx: "AnalysisContext") -> list[CriticalPosition]:
    """One entry per choice point of your main lines (``openings.choice_positions``, the positions the openings
    section draws), for the move you usually play there.

    ``drop`` is the win-% points Stockfish says that move lost on average in the analysed games (0.0 without
    them: no verdict), ``best_san`` Stockfish's first choice (None without evals); ``engine_mistake`` says
    whether that makes the move a mistake."""
    by_id = ctx.games_by_id
    out = []
    for pos in openings.choice_positions(ctx):
        usual = pos.usual
        if not usual.uci or not usual.game_ids:
            continue
        games = [by_id[gid] for gid in usual.game_ids if gid in by_id]  # most recent first
        family = pos.opening_family if pos.opening_family and _family(games) == pos.opening_family else ""
        out.append(
            CriticalPosition(
                game_id=usual.game_ids[0],
                url=usual.urls[0] if usual.urls else "",
                ply=pos.ply,
                fen=pos.fen,
                played_uci=usual.uci,
                played_san=usual.san,
                move_label=mistakes.move_label(pos.fen, usual.san),
                best_san=pos.engine_best,
                drop=float(usual.engine_drop) if usual.engine_drop is not None else 0.0,
                time_class=next(iter(usual.formats)) if len(usual.formats) == 1 else "",
                color=pos.color,
                kind="choice",
                repeats=usual.games,
                games=list(usual.urls),
                opening_family=family,
                moves_before=list(pos.moves_before),
            )
        )
    return out


def select_critical(ctx: "AnalysisContext", modules: list[ModuleResult], max_positions: int) -> list[CriticalPosition]:
    """Repeated mistakes first, then your errors losing at least ``mistakes.MIN_DROP`` win-% points (costliest
    first), then choice points in your main lines; one entry per (EPD, move); at most ``max_positions``.

    Choice points keep a place when there are more errors than slots: they get up to ``CHOICE_SHARE`` of the slots
    the repeated mistakes leave free (there are few of them: at most ``openings.Thresholds.max_choice_diagrams``
    positions, one move each). Standard chess only.
    """
    if max_positions <= 0 or not ctx.evals:
        return []
    ids = _insight_ids(modules)
    events = mistakes.collect_errors(ctx.games, ctx.evals)
    groups = [_repeated(ctx, events, ids) if events else [], _errors(events, ids)]
    try:
        groups.append(_choices(ctx))
    except Exception as exc:  # noqa: BLE001 — choice points are a bonus: errors and habits come first
        log.warning("choice points skipped in the coaching: %s", exc)
        groups.append([])

    seen: set[tuple[str, str]] = set()
    repeated, errors, choices = [], [], []
    for group, kept in zip(groups, (repeated, errors, choices)):
        for c in group:
            key = (c.epd, c.played_uci)
            if key not in seen:
                seen.add(key)
                kept.append(c)
    repeated = repeated[:max_positions]
    free = max_positions - len(repeated)
    n_choices = min(len(choices), int(free * CHOICE_SHARE)) if free > 0 else 0
    n_errors = min(len(errors), free - n_choices)
    n_choices = min(len(choices), free - n_errors)  # the slots the errors did not need
    return repeated + errors[:n_errors] + choices[:n_choices]
