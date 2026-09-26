"""Pick the positions worth explaining (C1). Owner: coach-core.

Three kinds, in this order:

* **repeated**: the same wrong move in the same position in several of your games (``mistakes.find_repeated`` and
  ``mistakes.assess``, exactly as the "Positions you keep getting wrong" section finds them), most costly first
  (games x average win chance lost);
* **error**: every other move of yours that lost at least ``mistakes.MIN_DROP`` win-% points, costliest first;
* **choice**: the moves you play often at your choice points in the opening (``openings.choice_points``), so the
  report can show the engine's view of each branch of your main lines.

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


def _choices(ctx: "AnalysisContext") -> list[CriticalPosition]:
    """One entry per move you play often at a choice point of your main lines (``openings.choice_points``)."""
    th = openings.Thresholds.from_ctx(ctx)
    by_colour: dict = {"white": [], "black": []}
    for g in ctx.games:
        if openings.is_eligible(g):
            by_colour[g.color].append(g)
    out = []
    for point in openings.choice_points(by_colour, th):
        board = chess.Board()
        try:
            for san in point.prefix:
                board.push_san(san)
        except ValueError:
            continue
        ply, fen = len(point.prefix), board.fen()
        for option in point.options:
            games = sorted(option.games, key=lambda g: g.end_time, reverse=True)
            if not games:
                continue
            san = games[0].moves_san[ply]
            try:
                move = board.parse_san(san)
            except ValueError:
                continue
            # The engine's view of this move in the analysed games: win chance lost, and its preferred move.
            drops: list[float] = []
            bests: Counter = Counter()
            for g in games:
                ev = ctx.evals.get(g.game_id)
                p = next((p for p in ev.plies if p.ply == ply), None) if ev is not None else None
                if p is None:
                    continue
                drops.append(0.0 if p.best_san == san else max(0.0, p.win_before - p.win_after))
                if p.best_san:
                    bests[p.best_san] += 1
            out.append(
                CriticalPosition(
                    game_id=games[0].game_id,
                    url=games[0].url,
                    ply=ply,
                    fen=fen,
                    played_uci=move.uci(),
                    played_san=san,
                    move_label=mistakes.move_label(fen, san),
                    best_san=bests.most_common(1)[0][0] if bests else None,
                    drop=sum(drops) / len(drops) if drops else 0.0,
                    time_class=_time_class(games),
                    color=point.color,
                    kind="choice",
                    repeats=len(games),
                    games=[g.url for g in games if g.url],
                    opening_family=_family(games),
                    moves_before=list(point.prefix),
                )
            )
    return out


def select_critical(ctx: "AnalysisContext", modules: list[ModuleResult], max_positions: int) -> list[CriticalPosition]:
    """Repeated mistakes first, then your errors losing at least ``mistakes.MIN_DROP`` win-% points (costliest
    first), then choice points in your main lines; one entry per (EPD, move); at most ``max_positions``.

    Choice points keep a place when there are more errors than slots: they get up to ``CHOICE_SHARE`` of the slots
    the repeated mistakes leave free (there are few of them: at most ``openings.Thresholds.max_choice_points``
    positions with two or three moves each). Standard chess only.
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
