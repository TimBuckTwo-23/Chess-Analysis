"""Habits & tilt: sessions, playing on after losses, fatigue, streaks and time of day.

Live games only (a daily game is played over days, so it has no session). Games
are put on a timeline by start time (chess.com's, or the end time minus the time
both clocks used) and grouped into sessions: the next game starting at most
30 minutes after the previous one ended stays in the same session.

Every comparison is score minus the Elo expected score. "After a loss" is judged
against all your other games (a difference test), so a player whose rating lags
behind their strength isn't flagged just for scoring above expectation elsewhere.

Claims follow the project-wide rule (``stats.significance``). Two hypotheses are
fixed in advance, with their direction, and tested on their own at ``stats.ALPHA``
with a one-sided p-value: scoring worse straight after a loss, and scoring worse late
at night (games started 23:00-03:00 in the player's time zone, the one time-of-day
effect with a strong prior). The late-night window only means late night when the
player's time zone is known (``ctx.options["tz"]``, an IANA name; no ``--tz``, and
``--tz UTC`` / ``GMT`` which the command line turns into "UTC", mean "not given"); otherwise the same UTC window is tested like any other
block of the day, at ``stats.STRICT_ALPHA``, and titled by its UTC hours. Everything
else is an exploratory scan at ``stats.STRICT_ALPHA``: the six 4-hour blocks of the
day (Benjamini-Hochberg adjusted because six are tested at once) and session length.
A session-length finding must survive leaving out the games played straight after a
loss when that is a finding too (later games in a session follow losses more often).
Games with fewer than 4 plies stay on the timeline (they still end a session or
a streak) but are not scored.
"""

from __future__ import annotations

import calendar
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..context import AnalysisContext
from ..models import Chart, Game, Insight, Kpi, ModuleResult, Series, Table
from ..stats import (
    ALPHA,
    STRICT_ALPHA,
    MeanTest,
    ScoreSummary,
    bh_adjust,
    clamp,
    difference_test,
    one_sided,
    pct,
    per100_games,
    sample_confidence,
    severity_from_points,
    shrink_effect,
    significance,
    summarize,
    vs_rating,
)
from .results import MAX_EXAMPLES, recent_urls
from .time_mgmt import time_spent

KEY = "habits"
TITLE = "Habits & tilt"

MIN_PLIES = 4
# Without clocks, assume each player uses this share of (base + 40 x increment) over an 80-ply game.
CLOCK_USAGE = 0.7
FATIGUE_BUCKETS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("1st game", 1, 1),
    ("2nd–3rd", 2, 3),
    ("4th–6th", 4, 6),
    ("7th+", 7, None),
)
SESSION_SIZES: tuple[tuple[str, int, Optional[int]], ...] = (
    ("1", 1, 1),
    ("2–3", 2, 3),
    ("4–6", 4, 6),
    ("7+", 7, None),
)
# (label, name used in titles) for the six 4-hour blocks starting at 00:00
DAY_PARTS: tuple[tuple[str, str], ...] = (
    ("00:00–04:00", "late-night"),
    ("04:00–08:00", "early-morning"),
    ("08:00–12:00", "morning"),
    ("12:00–16:00", "afternoon"),
    ("16:00–20:00", "evening"),
    ("20:00–24:00", "late-evening"),
)
# The pre-specified late-night window (hours in the player's time zone): 23:00-03:00.
LATE_NIGHT = ("23:00", "03:00")
LATE_NIGHT_HOURS = frozenset({23, 0, 1, 2})
# Day parts that overlap the late-night window: a claim on one of them and on the window is one finding.
LATE_NIGHT_PARTS = frozenset({0, 5})
# Time-zone option values that mean "not given": no --tz at all (None), or --tz UTC / GMT / Z, which the
# command line turns into "UTC" (they say nothing about when the player's night is; pass e.g. "Etc/UTC" or
# "Europe/London" to mean it).
UNKNOWN_TZ = frozenset({"", "UTC"})
SCORE_COLUMNS = ["Games", "W/D/L", "Score", "Rating predicts", "vs rating"]
SCORE_FORMATS = ["int", "text", "pct", "pct", "signed_pct"]


@dataclass(frozen=True)
class Thresholds:
    """Minimum sample sizes and effect sizes; override with ``ctx.options["habits.<name>"]``."""

    min_games: int = 20  # live games below which the summary says there is not enough data
    session_gap_min: float = 30.0  # a longer pause between games starts a new session
    tilt_window_min: float = 15.0  # "right after" a game
    min_split_games: int = 25  # rated games in the after-a-loss / late-in-session group
    min_bucket_games: int = 25  # rated games in a time-of-day block
    min_effect: float = 0.06  # points per game
    alpha: float = ALPHA  # after a loss, late night: pre-specified primary claims (one-sided)
    strict_alpha: float = STRICT_ALPHA  # time-of-day blocks, session length: exploratory scans
    fatigue_splits: tuple[int, ...] = (4, 7)  # "late in a session" = from this game number on
    min_rematches: int = 5
    min_streak: int = 3  # longest losing streak worth mentioning
    min_chart_games: int = 10  # rated games for a bar in the score charts (fewer: left blank)

    @classmethod
    def from_ctx(cls, ctx: AnalysisContext) -> "Thresholds":
        return cls(**{f.name: ctx.opt(f"{KEY}.{f.name}", f.default) for f in dataclasses.fields(cls)})


# --------------------------------------------------------------------------- timeline
def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def estimated_duration(game: Game) -> float:
    """Seconds the game lasted: both clocks' used time when known, else an estimate from the time control."""
    spent = [s for s in time_spent(game) if s is not None]
    if spent:
        return sum(spent)
    if not game.base_seconds or game.time_class == "daily":
        return 0.0
    per_player = game.base_seconds + 40 * game.increment
    return 2.0 * per_player * CLOCK_USAGE * min(1.0, game.plies / 80.0)


def game_start(game: Game) -> datetime:
    """When the game started (UTC): ``start_time`` if plausible, else end time minus the estimated duration."""
    end = _utc(game.end_time)
    if game.start_time is not None and _utc(game.start_time) <= end:
        return _utc(game.start_time)
    return end - timedelta(seconds=estimated_duration(game))


@dataclass
class TimelineGame:
    """One live game on the timeline."""

    game: Game
    start: datetime  # UTC
    session: int  # 0-based session number
    position: int  # 1-based game number within the session
    gap: Optional[float]  # seconds from the previous game's end to this game's start; None for the first game
    prev: Optional[Game]
    situation: str  # after_loss | after_win | after_draw | short_break | break
    loss_run: int  # consecutive losses just before this game, in the same session

    @property
    def scored(self) -> bool:
        return self.game.plies >= MIN_PLIES


def build_timeline(
    games: Sequence[Game], session_gap_min: float = 30.0, tilt_window_min: float = 15.0
) -> list[TimelineGame]:
    """Live games in order, grouped into sessions, each labelled with what happened just before it."""
    live = sorted((g for g in games if g.time_class != "daily"), key=lambda g: _utc(g.end_time))
    session_gap, window = session_gap_min * 60.0, tilt_window_min * 60.0
    out: list[TimelineGame] = []
    prev: Optional[Game] = None
    session, position, loss_run = -1, 0, 0
    for g in live:
        start = game_start(g)
        gap = (start - _utc(prev.end_time)).total_seconds() if prev is not None else None
        if gap is None or gap > session_gap:
            session, position, loss_run = session + 1, 1, 0
            situation = "break"
        else:
            position += 1
            situation = f"after_{prev.outcome}" if gap <= window and prev is not None else "short_break"
        out.append(TimelineGame(g, start, session, position, gap, prev, situation, loss_run))
        loss_run = loss_run + 1 if g.outcome == "loss" else 0
        prev = g
    return out


def longest_losing_streak(timeline: Sequence[TimelineGame]) -> list[Game]:
    """The longest run of consecutive losses (the most recent one on ties)."""
    best: list[Game] = []
    run: list[Game] = []
    for t in timeline:
        if t.game.outcome != "loss":
            run = []
            continue
        run.append(t.game)
        if len(run) >= len(best):
            best = list(run)
    return best


def resolve_tz(name: Any) -> tuple[tzinfo, str]:
    """(time zone, its name) for an IANA name; UTC when the name is missing or unknown."""
    try:
        return ZoneInfo(str(name)), str(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError, OSError):
        return timezone.utc, "UTC"


def tz_label(name: str) -> str:
    """How a time zone is named in the report: 'Etc/UTC' and friends as plain 'UTC'."""
    if name.upper() in ("UTC", "ETC/UTC", "ETC/UCT", "ETC/UNIVERSAL", "ETC/ZULU", "UCT", "UNIVERSAL", "ZULU"):
        return "UTC"
    if name.upper() in ("ETC/GMT", "ETC/GMT0", "ETC/GMT+0", "ETC/GMT-0", "ETC/GREENWICH", "GMT", "GREENWICH"):
        return "GMT"
    return name


def local_time_known(name: Any) -> bool:
    """Whether ``name`` (the "tz" option) says which time zone the player lives in (see UNKNOWN_TZ)."""
    return name is not None and str(name) not in UNKNOWN_TZ and resolve_tz(name)[1] == str(name)


def day_part(moment: datetime, tz: tzinfo) -> int:
    """Index into DAY_PARTS of ``moment`` in time zone ``tz``."""
    return _utc(moment).astimezone(tz).hour // 4


# --------------------------------------------------------------------------- tables & charts
def _score_cells(s: ScoreSummary) -> list[Any]:
    return [s.n, s.wdl, s.rated_score, s.expected, s.delta]


def _score_table(title: str, first: str, rows: Sequence[tuple[str, ScoreSummary]], note: str) -> Table:
    return Table(
        title=title,
        columns=[first, *SCORE_COLUMNS],
        rows=[[label, *_score_cells(s)] for label, s in rows],
        formats=["text", *SCORE_FORMATS],
        note=note,
    )


def _delta_chart(
    title: str, rows: Sequence[tuple[str, ScoreSummary]], note: str, min_games: int = 0, table: Optional[Table] = None
) -> Chart:
    """Score vs rating per group; groups with fewer than ``min_games`` rated games are left blank (a bar from
    3 games would stand out most while meaning least)."""
    small = [label for label, s in rows if 0 < s.n_rated < min_games]
    if small:
        note += f" Blank: fewer than {min_games} rated games ({', '.join(small)}); the numbers are in the table."
    return Chart(
        kind="bar",
        title=title,
        labels=[label for label, _ in rows],
        series=[Series("Score vs rating", [s.delta if s.n_rated >= min_games else None for _, s in rows])],
        value_format="signed_pct",
        note=note,
        reference=0.0,
        table=table,
    )


def _situations(th: Thresholds) -> tuple[tuple[str, str], ...]:
    w, gap = f"{th.tilt_window_min:g}", f"{th.session_gap_min:g}"
    return (
        ("after_loss", f"Within {w} min of a loss"),
        ("after_win", f"Within {w} min of a win"),
        ("after_draw", f"Within {w} min of a draw"),
        ("short_break", f"{w}–{gap} min after the previous game"),
        ("break", f"After a break (> {gap} min) or first game"),
    )


def session_sizes(timeline: Sequence[TimelineGame]) -> dict[int, int]:
    """Games per session, by session number."""
    sizes: dict[int, int] = {}
    for t in timeline:
        sizes[t.session] = max(sizes.get(t.session, 0), t.position)
    return sizes


def session_table(sizes: dict[int, int]) -> Table:
    rows = []
    for label, lo, hi in SESSION_SIZES:
        matching = [n for n in sizes.values() if n >= lo and (hi is None or n <= hi)]
        rows.append([label, len(matching), sum(matching)])
    return Table(
        title="Session lengths",
        columns=["Games in the session", "Sessions", "Games"],
        rows=rows,
        formats=["text", "int", "int"],
    )


# --------------------------------------------------------------------------- insights
def tilt_insight(
    groups: dict[str, list[Game]], scored: Sequence[Game], th: Thresholds
) -> tuple[Optional[Insight], MeanTest, ScoreSummary, ScoreSummary]:
    """After a loss (within the tilt window) vs every other game."""
    after = groups["after_loss"]
    after_ids = {id(g) for g in after}
    rest = [g for g in scored if id(g) not in after_ids]
    a, r = summarize(after), summarize(rest)
    test = difference_test(a.test, r.test)
    if a.n_rated < th.min_split_games or r.n_rated < th.min_split_games:
        return None, test, a, r
    n = min(a.n_rated, r.n_rated)
    significant, confidence = significance(one_sided(test, -1), th.min_split_games, alpha=th.alpha, n=n)
    if test.mean > -th.min_effect or a.test.mean >= 0 or not significant:
        return None, test, a, r
    w = f"{th.tilt_window_min:g}"
    losses = [g for g in after if g.outcome == "loss"]
    return (
        Insight(
            id=f"{KEY}.weakness.after-a-loss",
            kind="weakness",
            category="habits",
            title="You play worse right after a loss",
            detail=(
                f"In {a.n_rated} games started within {w} minutes of a loss you scored {pct(a.rated_score)} where your "
                f"rating predicts {pct(a.expected)}: {vs_rating(a.test.mean)}. Your other games: "
                f"{per100_games(r.test.mean)} vs your rating."
            ),
            severity=severity_from_points(shrink_effect(test.mean, test.se)),
            confidence=confidence,
            evidence={
                "n": a.n_rated,
                "score": a.rated_score,
                "expected": a.expected,
                "delta": a.delta,
                "rest_delta": r.delta,
                "rest_n": r.n_rated,
                "gap": test.mean,
                "p_value": test.p_value,
                "p_value_one_sided": one_sided(test, -1).p_value,
            },
            study=[
                f"Stop rule: after any loss, wait at least {w} minutes before starting another game.",
                "Use the break to find the move where the lost game turned; start the next game only once you "
                "know what went wrong.",
                "If you catch yourself pressing 'New game' seconds after a loss, switch to puzzles instead.",
            ],
            example_games=recent_urls(losses),
        ),
        test,
        a,
        r,
    )


def _fatigue_tests(
    scored_timeline: Sequence[TimelineGame], splits: Sequence[int], leave_out: frozenset[int] = frozenset()
) -> list[tuple[int, list[Game], ScoreSummary, ScoreSummary, MeanTest]]:
    """(first late game number, late games, late summary, earlier summary, late-minus-earlier test) per split."""
    out = []
    for first_late in splits:
        tl = [t for t in scored_timeline if id(t.game) not in leave_out]
        late = [t.game for t in tl if t.position >= first_late]
        early = [t.game for t in tl if t.position < first_late]
        lt, er = summarize(late), summarize(early)
        out.append((first_late, late, lt, er, difference_test(lt.test, er.test)))
    return out


def fatigue_insight(
    scored_timeline: Sequence[TimelineGame], th: Thresholds, after_loss: Optional[Sequence[Game]] = None
) -> tuple[Optional[Insight], dict[str, Any]]:
    """Late-in-session games vs earlier ones, for each split point; the stronger qualifying one wins.

    ``after_loss``: the games played straight after a loss, when that is a finding of its own.
    Games later in a session follow a loss more often (a session's first game never does), so a
    session-length claim must then also hold with those games left out; otherwise it is the
    after-a-loss finding seen again.
    """
    splits = _fatigue_tests(scored_timeline, th.fatigue_splits)
    adjusted = bh_adjust([t[4].p_value for t in splits])
    without = None
    if after_loss:
        without = _fatigue_tests(scored_timeline, th.fatigue_splits, frozenset(id(g) for g in after_loss))
        without_adjusted = bh_adjust([t[4].p_value for t in without])
    stats: dict[str, Any] = {}
    best: Optional[Insight] = None
    for i, ((first_late, late, lt, er, test), p_adj) in enumerate(zip(splits, adjusted, strict=True)):
        stats[f"from_game_{first_late}"] = {
            "n": lt.n_rated,
            "delta": lt.delta,
            "earlier_delta": er.delta,
            "gap": test.mean,
            "p_adjusted": p_adj,
        }
        if lt.n_rated < th.min_split_games or er.n_rated < th.min_split_games:
            continue
        n = min(lt.n_rated, er.n_rated)
        significant, confidence = significance(test, th.min_split_games, p_adj, alpha=th.strict_alpha, n=n)
        if test.mean > -th.min_effect or lt.test.mean >= 0 or not significant:
            continue
        if without is not None:
            _, _, lt2, er2, test2 = without[i]
            holds, _ = significance(
                test2, th.min_split_games, without_adjusted[i], alpha=th.strict_alpha, n=min(lt2.n_rated, er2.n_rated)
            )
            stats[f"from_game_{first_late}"]["without_after_loss_gap"] = test2.mean
            if not holds or test2.mean > -th.min_effect:
                continue
        losses = [g for g in late if g.outcome == "loss"]
        candidate = Insight(
            id=f"{KEY}.weakness.long-sessions",
            kind="weakness",
            category="habits",
            title=f"You score worse from the {_ordinal(first_late)} game of a session on",
            detail=(
                f"From the {_ordinal(first_late)} game of a session on you scored {pct(lt.rated_score)} in "
                f"{lt.n_rated} games where your rating predicts {pct(lt.expected)}: {vs_rating(lt.test.mean)}. "
                f"In the first {first_late - 1} games: {per100_games(er.test.mean)} vs your rating."
            ),
            severity=severity_from_points(shrink_effect(test.mean, test.se)),
            confidence=confidence,
            evidence={"from_game": first_late, **stats[f"from_game_{first_late}"]},
            study=[
                f"Cap your sessions at {first_late - 1} games, then stop or switch to puzzles.",
                "Take a 10-minute break away from the screen every three games.",
                "End the session after the first careless loss instead of playing 'one more'.",
            ],
            example_games=recent_urls(losses),
        )
        if best is None or candidate.priority > best.priority:
            best = candidate
    return best, stats


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def time_of_day_insights(
    parts: list[list[Game]], scored: Sequence[Game], tz_name: str, th: Thresholds, local: bool = True
) -> tuple[list[Insight], list[Optional[float]]]:
    """Each 4-hour block vs all other games; BH-adjusted over the blocks with enough games, strict level.

    ``local``: the blocks are in the player's own time zone, so they can be named (late night,
    morning...); otherwise they are titled by their hours in ``tz_name``.
    """
    summaries = [summarize(p) for p in parts]
    tests: dict[int, tuple[MeanTest, ScoreSummary, ScoreSummary]] = {}
    for i, part in enumerate(parts):
        if summaries[i].n_rated < th.min_bucket_games:
            continue
        ids = {id(g) for g in part}
        rest = summarize([g for g in scored if id(g) not in ids])
        if rest.n_rated < th.min_bucket_games:
            continue
        tests[i] = (difference_test(summaries[i].test, rest.test), summaries[i], rest)
    order = list(tests)
    adjusted = dict(zip(order, bh_adjust([tests[i][0].p_value for i in order]), strict=True))
    qualifying: list[tuple[int, float, MeanTest, ScoreSummary, ScoreSummary]] = []
    for i in order:
        test, s, rest = tests[i]
        n = min(s.n_rated, rest.n_rated)
        significant, confidence = significance(test, th.min_bucket_games, adjusted[i], alpha=th.strict_alpha, n=n)
        if test.mean <= -th.min_effect and s.test.mean < 0 and significant:
            qualifying.append((i, confidence, test, s, rest))
    out = []
    for i, confidence, test, s, rest in qualifying:
        label, name = DAY_PARTS[i]
        start, end = label.split("–")
        if not local:
            title = f"You score below your rating in games started {label} {tz_name}"
        else:
            title = f"You score below your rating in {name} games ({label})"
        losses = [g for g in parts[i] if g.outcome == "loss"]
        out.append(
            Insight(
                id=f"{KEY}.weakness.time-of-day-{start[:2]}-{end[:2]}",
                kind="weakness",
                category="habits",
                title=title,
                detail=(
                    f"In {s.n_rated} games started between {start} and {end} ({tz_name}) you scored "
                    f"{pct(s.rated_score)} where your rating predicts {pct(s.expected)}: {vs_rating(s.test.mean)}. "
                    f"All your other games: {per100_games(rest.test.mean)} vs your rating."
                ),
                severity=severity_from_points(shrink_effect(test.mean, test.se)),
                confidence=confidence,
                evidence={
                    "block": label,
                    "time_zone": tz_name,
                    "n": s.n_rated,
                    "score": s.rated_score,
                    "expected": s.expected,
                    "delta": s.delta,
                    "rest_delta": rest.delta,
                    "gap": test.mean,
                    "p_value": test.p_value,
                    "p_adjusted": adjusted[i],
                },
                study=[
                    f"Avoid rated games between {start} and {end}; play puzzles or unrated games then instead.",
                    f"Replay your last {min(MAX_EXAMPLES, len(losses)) or 'few'} losses from this time of day and "
                    "check whether they came from careless blunders, a sign of tiredness.",
                    "If you do play then, choose a slower time control and stop after the first loss.",
                ],
                example_games=recent_urls(losses),
            )
        )
    return out, [adjusted.get(i) for i in range(len(parts))]


def is_late_night(moment: datetime, tz: tzinfo) -> bool:
    """Started 23:00-03:00 in time zone ``tz``."""
    return _utc(moment).astimezone(tz).hour in LATE_NIGHT_HOURS


def late_night_insight(
    late: list[Game], scored: Sequence[Game], tz_name: str, th: Thresholds, local: bool = True
) -> tuple[Optional[Insight], dict[str, Any]]:
    """Games started 23:00-03:00 vs all other games: a pre-specified hypothesis tested alone.

    At ``th.alpha`` when the window is in the player's own time zone (``local``); otherwise it
    is just another 4-hour window of the day in ``tz_name``, tested at ``th.strict_alpha`` and
    titled by its hours.
    """
    s = summarize(late)
    ids = {id(g) for g in late}
    rest = summarize([g for g in scored if id(g) not in ids])
    test = difference_test(s.test, rest.test)
    stats = {"n": s.n_rated, "delta": s.delta, "rest_delta": rest.delta, "gap": test.mean, "p_value": test.p_value}
    if s.n_rated < th.min_bucket_games or rest.n_rated < th.min_bucket_games:
        return None, stats
    directional = one_sided(test, -1)
    stats["p_value_one_sided"] = directional.p_value
    stats["local_time"] = local
    alpha = th.alpha if local else th.strict_alpha
    n = min(s.n_rated, rest.n_rated)
    significant, confidence = significance(directional, th.min_bucket_games, alpha=alpha, n=n)
    if test.mean > -th.min_effect or s.test.mean >= 0 or not significant:
        return None, stats
    start, end = LATE_NIGHT
    losses = [g for g in late if g.outcome == "loss"]
    title = (
        f"You score below your rating in late-night games ({start}–{end})"
        if local
        else f"You score below your rating in games started {start}–{end} {tz_name}"
    )
    return (
        Insight(
            id=f"{KEY}.weakness.late-night",
            kind="weakness",
            category="habits",
            title=title,
            detail=(
                f"In {s.n_rated} games started between {start} and {end} ({tz_name}) you scored {pct(s.rated_score)} "
                f"where your rating predicts {pct(s.expected)}: {vs_rating(s.test.mean)}. All your other games: "
                f"{per100_games(rest.test.mean)} vs your rating."
            ),
            severity=severity_from_points(shrink_effect(test.mean, test.se)),
            confidence=confidence,
            evidence={
                "window": f"{start}–{end}",
                "time_zone": tz_name,
                "local_time": local,
                "n": s.n_rated,
                "score": s.rated_score,
                "expected": s.expected,
                "delta": s.delta,
                "rest_delta": rest.delta,
                "gap": test.mean,
                "p_value": test.p_value,
                "p_value_one_sided": directional.p_value,
            },
            study=[
                f"Stop starting rated games after {start}{'' if local else ' ' + tz_name}; play puzzles or unrated "
                "games instead.",
                f"Replay your last {min(MAX_EXAMPLES, len(losses)) or 'few'} late-night losses and check whether "
                "they came from careless blunders or the clock, both signs of tiredness.",
                "If you do play late, choose a slower time control and stop after the first loss.",
            ],
            example_games=recent_urls(losses),
        ),
        stats,
    )


def blunder_rate(games: Sequence[Game], evals: dict) -> tuple[int, int, int]:
    """(your blunders, your moves, analysed games) over the engine-analysed ``games``."""
    blunders = moves = n = 0
    for g in games:
        ev = evals.get(g.game_id)
        if ev is None:
            continue
        n += 1
        for p in ev.plies:
            if p.is_user:
                moves += 1
                blunders += p.judgement == "blunder"
    return blunders, moves, n


def add_blunder_rates(ins: Insight, window: Sequence[Game], scored: Sequence[Game], evals: dict, min_games: int = 10) -> None:
    """Answer "were those games lost to tired blunders?" with Stockfish's count, when it analysed enough of them.

    Plain numbers for context, not a claim: nothing is tested here."""
    ids = {id(g) for g in window}
    b1, m1, n1 = blunder_rate(window, evals)
    b2, m2, n2 = blunder_rate([g for g in scored if id(g) not in ids], evals)
    if n1 < min_games or n2 < min_games or not m1 or not m2:
        return
    r1, r2 = 100.0 * b1 / m1, 100.0 * b2 / m2
    ins.detail += (
        f" Stockfish analysed {n1} of these games: you blundered {r1:.1f} times per 100 moves in them, "
        f"{r2:.1f} in {n2} analysed games at other times."
    )
    ins.evidence.update(blunders_per100=r1, other_blunders_per100=r2, analysed_games=n1)
    if r1 > r2:
        ins.study = [
            s if "careless blunders" not in s else
            "Before each move in a late game, take one extra second to check what your opponent's last move "
            f"threatens: you blunder more then ({r1:.1f} vs {r2:.1f} per 100 moves)."
            for s in ins.study
        ]
    else:
        ins.study = [
            s if "careless blunders" not in s else
            f"You don't blunder more at this hour ({r1:.1f} vs {r2:.1f} per 100 moves), so replay your recent "
            "losses from it and look for the clock or the opening instead."
            for s in ins.study
        ]


def merge_late_night(late: Optional[Insight], blocks: list[Insight]) -> list[Insight]:
    """One finding when the late-night window and an overlapping 4-hour block are both flagged.

    The one with the stronger evidence (smaller unadjusted p-value) is kept; on a tie (the same
    games, e.g. every late game started after midnight) the 4-hour block.
    """
    if late is None:
        return blocks
    labels = {DAY_PARTS[i][0] for i in LATE_NIGHT_PARTS}
    overlapping = [b for b in blocks if b.evidence.get("block") in labels]
    if overlapping and min(b.evidence["p_value"] for b in overlapping) <= late.evidence["p_value"]:
        return blocks
    return [late, *(b for b in blocks if b not in overlapping)]


def rematch_insight(
    scored_timeline: Sequence[TimelineGame], th: Thresholds
) -> tuple[Optional[Insight], dict[str, Any]]:
    """Immediate rematches against the opponent who just beat you."""
    games = [
        t.game
        for t in scored_timeline
        if t.situation == "after_loss" and t.prev is not None and t.prev.opponent.lower() == t.game.opponent.lower()
    ]
    s = summarize(games)
    stats = {"n": s.n, "score": s.score, "expected": s.expected, "delta": s.delta}
    if s.n < th.min_rematches:
        return None, stats
    detail = f"You played an immediate rematch after {s.n} losses and scored {pct(s.score)} ({s.wdl} W/D/L)"
    detail += f" where your rating predicts {pct(s.expected)} ({per100_games(s.test.mean)})." if s.n_rated else "."
    return (
        Insight(
            id=f"{KEY}.observation.rematches",
            kind="observation",
            category="habits",
            title=f"You took an immediate rematch after {s.n} losses and scored {pct(s.score)}",
            detail=detail,
            severity=severity_from_points(s.test.mean) if s.n_rated else 0.0,
            confidence=sample_confidence(s.n_rated),
            evidence=stats,
            study=[
                "Before clicking Rematch, ask whether you want a good game or revenge; if it's revenge, take a break "
                "first.",
                "Spend two minutes on the lost game before a rematch: what did this opponent do that worked?",
            ],
            example_games=recent_urls(games, "loss"),
        ),
        stats,
    )


def _streak_advice(after_two: ScoreSummary) -> str:
    """Stop after two losses only when the games after two losses actually went badly."""
    if after_two.n_rated and after_two.test.mean < 0:
        return "Stop after two losses in a row: in your games that is when you score worst."
    if after_two.n_rated:
        return (
            "After two losses in a row you still played about as well as your rating predicts, so the streak itself "
            "is not the problem: look at what the lost games had in common."
        )
    return "Take a short break after a loss before starting the next game."


def streak_insight(streak: list[Game], after_two: ScoreSummary, n_games: int, th: Thresholds) -> Optional[Insight]:
    if len(streak) < th.min_streak:
        return None
    first, last = f"{_utc(streak[0].end_time):%Y-%m-%d}", f"{_utc(streak[-1].end_time):%Y-%m-%d}"
    when = f"on {first}" if first == last else f"from {first} to {last}"
    detail = f"Your longest run of consecutive losses was {len(streak)} games, {when}."
    if after_two.n:
        detail += (
            f" In the same session after two or more losses in a row you scored {pct(after_two.score)} "
            f"in {after_two.n} games"
        )
        detail += (
            f" where your rating predicts {pct(after_two.expected)} ({per100_games(after_two.test.mean)})."
            if after_two.n_rated
            else "."
        )
    return Insight(
        id=f"{KEY}.observation.losing-streak",
        kind="observation",
        category="habits",
        title=f"Your longest losing streak was {len(streak)} games",
        detail=detail,
        severity=clamp(len(streak) / 10.0),
        confidence=sample_confidence(n_games),
        evidence={
            "longest": len(streak),
            "after_two_losses_n": after_two.n,
            "after_two_losses_score": after_two.score,
            "after_two_losses_expected": after_two.expected,
            "after_two_losses_delta": after_two.delta,
        },
        study=[
            _streak_advice(after_two),
            "Replay the games of your longest streak and look for a shared cause: the same opening, the clock, "
            "or hurried moves.",
        ],
        example_games=[g.url for g in reversed(streak) if g.url][:MAX_EXAMPLES],
    )


# --------------------------------------------------------------------------- module entry point
def _summary(
    n: int,
    n_sessions: int,
    avg: float,
    longest: int,
    a: ScoreSummary,
    r: ScoreSummary,
    insights: list[Insight],
    th: Thresholds,
) -> str:
    text = (
        f"You played {n} live game{'s' if n != 1 else ''} in {n_sessions} session{'s' if n_sessions != 1 else ''} "
        f"(on average {avg:.1f} games, at most {longest})."
    )
    if a.n_rated and r.n_rated:
        text += (
            f" Right after a loss you score {per100_games(a.test.mean)} vs your rating, "
            f"otherwise {per100_games(r.test.mean)}."
        )
    if n < th.min_games:
        games_text = f"{n} live game{'s' if n != 1 else ''}"
        return f"Not enough data yet: only {games_text}, so treat these numbers as a snapshot. {text}"
    ranked = sorted((i for i in insights if i.kind != "observation"), key=lambda i: -i.priority)
    if ranked:
        text += f" Key finding: {ranked[0].title}."
    return text


def analyze(ctx: AnalysisContext) -> ModuleResult:
    """Sessions, tilt after losses, rematches, fatigue, losing streaks and time of day (live games)."""
    th = Thresholds.from_ctx(ctx)
    timeline = build_timeline(ctx.games, th.session_gap_min, th.tilt_window_min)
    if not timeline:
        return ModuleResult(
            key=KEY,
            title=TITLE,
            summary="Not enough data: there are no live games to analyse (daily games are left out of this section).",
            stats={"n": 0},
        )
    tz_option = ctx.opt("tz", None)
    tz, tz_raw = resolve_tz(tz_option if tz_option is not None else "UTC")
    tz_name = tz_label(tz_raw)
    local = local_time_known(tz_option)
    scored_tl = [t for t in timeline if t.scored]
    scored = [t.game for t in scored_tl]

    # sessions
    sizes = session_sizes(timeline)
    n_sessions, longest = len(sizes), max(sizes.values())
    avg = len(timeline) / n_sessions

    # tilt
    situations = _situations(th)
    groups: dict[str, list[Game]] = {key: [] for key, _ in situations}
    for t in scored_tl:
        groups[t.situation].append(t.game)
    tilt, tilt_test, after_loss, rest = tilt_insight(groups, scored, th)
    situation_rows = [(label, summarize(groups[key])) for key, label in situations]

    # fatigue
    fatigue_rows = [
        (label, summarize([t.game for t in scored_tl if t.position >= lo and (hi is None or t.position <= hi)]))
        for label, lo, hi in FATIGUE_BUCKETS
    ]
    fatigue, fatigue_stats = fatigue_insight(scored_tl, th, groups["after_loss"] if tilt else None)

    # streaks and rematches
    streak = longest_losing_streak(timeline)
    after_two = summarize([t.game for t in scored_tl if t.loss_run >= 2])
    rematch, rematch_stats = rematch_insight(scored_tl, th)

    # time of day / weekday (in the player's time zone), by start time as chess.com reports it
    parts: list[list[Game]] = [[] for _ in DAY_PARTS]
    weekdays: list[list[Game]] = [[] for _ in range(7)]
    late: list[Game] = []
    for g in scored:
        moment = g.start_time if g.start_time is not None else g.end_time
        parts[day_part(moment, tz)].append(g)
        weekdays[_utc(moment).astimezone(tz).weekday()].append(g)
        if is_late_night(moment, tz):
            late.append(g)
    tod_insights, tod_p = time_of_day_insights(parts, scored, tz_name, th, local)
    late_insight, late_stats = late_night_insight(late, scored, tz_name, th, local)
    tod_insights = merge_late_night(late_insight, tod_insights)
    if ctx.evals:
        labels = [label for label, _ in DAY_PARTS]
        for ins in tod_insights:
            block = ins.evidence.get("block")
            window = late if block is None else parts[labels.index(block)] if block in labels else []
            add_blunder_rates(ins, window, scored, ctx.evals)
    part_rows = [(label, summarize(p)) for (label, _), p in zip(DAY_PARTS, parts, strict=True)]
    weekday_rows = [(calendar.day_name[i], summarize(weekdays[i])) for i in range(7)]

    insights = [i for i in (tilt, fatigue) if i] + tod_insights
    insights += [i for i in (rematch, streak_insight(streak, after_two, len(timeline), th)) if i]
    if len(timeline) < th.min_games:
        insights = []

    diff_note = "vs rating = your score minus what your rating predicts, per game (+5% = 5 points per 100 games)."
    paired = [
        _delta_chart(
            "Score vs rating after a loss, a win or a break",
            situation_rows,
            "Per game, by when the game started relative to the previous one.",
            th.min_chart_games,
            _score_table(
                "After a loss, a win or a break",
                "Game started",
                situation_rows,
                f"When each game started relative to the end of your previous live game. {diff_note}",
            ),
        ),
        _delta_chart(
            "Score vs rating by game number in a session",
            fatigue_rows,
            "Per game, against what your rating predicts.",
            th.min_chart_games,
            _score_table(
                "By game number in a session",
                "Game in the session",
                fatigue_rows,
                f"A session ends when you pause for more than {th.session_gap_min:g} minutes. {diff_note}",
            ),
        ),
        _delta_chart(
            f"Score vs rating by time of day ({tz_name})",
            part_rows,
            "Per game, by start time.",
            th.min_chart_games,
            _score_table(f"By time of day ({tz_name})", "Started", part_rows, f"Start times in {tz_name}. {diff_note}"),
        ),
    ]
    # a chart with no bar to draw gives way to its table
    charts = [c for c in paired if any(v is not None for v in c.series[0].values)]
    tables = [session_table(sizes)] + [c.table for c in paired if c not in charts and c.table is not None]
    tables.append(_score_table("By weekday", "Day", weekday_rows, f"Days in {tz_name}. {diff_note}"))

    kpis = [
        Kpi("Sessions", n_sessions, "int", hint=f"a pause of more than {th.session_gap_min:g} min starts a new one"),
        Kpi("Games per session", avg, "float1", hint=f"longest: {longest}"),
        Kpi("After a loss", after_loss.delta, "signed_pct", hint=f"score vs rating, {after_loss.n_rated} games"),
        Kpi("All other games", rest.delta, "signed_pct", hint=f"score vs rating, {rest.n_rated} games"),
        Kpi("Longest losing streak", len(streak), "int"),
    ]

    return ModuleResult(
        key=KEY,
        title=TITLE,
        summary=_summary(len(timeline), n_sessions, avg, longest, after_loss, rest, insights, th),
        kpis=kpis,
        tables=tables,
        charts=charts,
        insights=insights,
        stats={
            "n": len(timeline),
            "n_scored": len(scored),
            "sessions": n_sessions,
            "games_per_session": avg,
            "longest_session": longest,
            "time_zone": tz_raw,
            "local_time": local,
            "situations": {key: _stats(s) for (key, _), (_, s) in zip(situations, situation_rows, strict=True)},
            "after_loss_gap": {"gap": tilt_test.mean, "p_value": tilt_test.p_value},
            "fatigue": {label: _stats(s) for label, s in fatigue_rows},
            "fatigue_tests": fatigue_stats,
            "longest_losing_streak": len(streak),
            "after_two_losses": _stats(after_two),
            "rematches": rematch_stats,
            "time_of_day": {
                label: {**_stats(s), "p_adjusted": p} for (label, s), p in zip(part_rows, tod_p, strict=True)
            },
            "late_night": late_stats,
            "weekday": {label: _stats(s) for label, s in weekday_rows},
        },
    )


def _stats(s: ScoreSummary) -> dict[str, Any]:
    return {"n": s.n, "n_rated": s.n_rated, "score": s.score, "expected": s.expected, "delta": s.delta}
