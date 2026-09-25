"""Habits & tilt: sessions, playing on after losses, fatigue, streaks and time of day.

Live games only (a daily game is played over days, so it has no session). Games
are put on a timeline by start time (chess.com's, or the end time minus the time
both clocks used) and grouped into sessions: the next game starting at most
30 minutes after the previous one ended stays in the same session.

Every comparison is score minus the Elo expected score. "After a loss" is judged
against all your other games (a difference test), so a player whose rating lags
behind their strength isn't flagged just for scoring above expectation elsewhere.
Time-of-day buckets are Benjamini-Hochberg adjusted because six are tested at once.
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
from ..stats import MeanTest, bh_adjust, clamp, combined_confidence, pct, sample_confidence, severity_from_points
from .results import MAX_EXAMPLES, ScoreSummary, difference_test, fmt_points, recent_urls, summarize, with_p_value
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
SCORE_COLUMNS = ["Games", "W/D/L", "Score", "Expected", "Difference"]
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
    min_confidence: float = 0.4  # combined_confidence needed for a weakness
    max_p_value: float = 0.05  # ... and the (multiple-testing adjusted) test must be significant at this level
    fatigue_splits: tuple[int, ...] = (4, 7)  # "late in a session" = from this game number on
    min_rematches: int = 5
    min_streak: int = 3  # longest losing streak worth mentioning

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


def _delta_chart(title: str, rows: Sequence[tuple[str, ScoreSummary]], note: str) -> Chart:
    return Chart(
        kind="bar",
        title=title,
        labels=[label for label, _ in rows],
        series=[Series("Score − expected", [s.delta for _, s in rows])],
        value_format="signed_pct",
        note=note,
        reference=0.0,
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
    confidence = combined_confidence(test, th.min_split_games)
    significant = confidence >= th.min_confidence and test.p_value <= th.max_p_value
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
                f"In {a.n_rated} games started within {w} minutes of a loss you scored {pct(a.rated_score)} where "
                f"{pct(a.expected)} was expected ({fmt_points(a.test.mean)} points per game); in your other games "
                f"{fmt_points(r.test.mean)} per game. The gap is {fmt_points(test.mean)} points per game."
            ),
            severity=severity_from_points(test.mean),
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
            },
            study=[
                f"Adopt a stop rule: after a loss, wait at least {w} minutes before the next game, and stop for "
                "the day after two losses in a row.",
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


def fatigue_insight(
    scored_timeline: Sequence[TimelineGame], th: Thresholds
) -> tuple[Optional[Insight], dict[str, Any]]:
    """Late-in-session games vs earlier ones, for each split point; the stronger qualifying one wins."""
    splits = []
    for first_late in th.fatigue_splits:
        late = [t.game for t in scored_timeline if t.position >= first_late]
        early = [t.game for t in scored_timeline if t.position < first_late]
        splits.append((first_late, late, summarize(late), summarize(early)))
    tests = [difference_test(lt.test, er.test) for _, _, lt, er in splits]
    adjusted = bh_adjust([t.p_value for t in tests])
    stats: dict[str, Any] = {}
    best: Optional[Insight] = None
    for (first_late, late, lt, er), test, p_adj in zip(splits, tests, adjusted, strict=True):
        stats[f"from_game_{first_late}"] = {
            "n": lt.n_rated,
            "delta": lt.delta,
            "earlier_delta": er.delta,
            "gap": test.mean,
            "p_adjusted": p_adj,
        }
        if lt.n_rated < th.min_split_games or er.n_rated < th.min_split_games:
            continue
        confidence = combined_confidence(with_p_value(test, p_adj), th.min_split_games)
        if test.mean > -th.min_effect or lt.test.mean >= 0 or confidence < th.min_confidence or p_adj > th.max_p_value:
            continue
        losses = [g for g in late if g.outcome == "loss"]
        candidate = Insight(
            id=f"{KEY}.weakness.long-sessions",
            kind="weakness",
            category="habits",
            title="Your results drop the longer you play in one sitting",
            detail=(
                f"From the {_ordinal(first_late)} game of a session on you scored {pct(lt.rated_score)} in "
                f"{lt.n_rated} games where {pct(lt.expected)} was expected ({fmt_points(lt.test.mean)} points per "
                f"game), against {fmt_points(er.test.mean)} per game in the first {first_late - 1} games."
            ),
            severity=severity_from_points(test.mean),
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
    parts: list[list[Game]], scored: Sequence[Game], tz_name: str, th: Thresholds
) -> tuple[list[Insight], list[Optional[float]]]:
    """Each 4-hour block vs all other games; BH-adjusted over the blocks with enough games."""
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
        confidence = combined_confidence(with_p_value(test, adjusted[i]), th.min_bucket_games)
        significant = confidence >= th.min_confidence and adjusted[i] <= th.max_p_value
        if test.mean <= -th.min_effect and s.test.mean < 0 and significant:
            qualifying.append((i, confidence, test, s, rest))
    worst = min(qualifying, key=lambda q: q[3].test.mean)[0] if qualifying else None
    out = []
    for i, confidence, test, s, rest in qualifying:
        label, name = DAY_PARTS[i]
        start, end = label.split("–")
        title = (
            f"Your {name} games ({label}) are your worst"
            if i == worst
            else f"You score below your rating in {name} games ({label})"
        )
        losses = [g for g in parts[i] if g.outcome == "loss"]
        out.append(
            Insight(
                id=f"{KEY}.weakness.time-of-day-{start[:2]}-{end[:2]}",
                kind="weakness",
                category="habits",
                title=title,
                detail=(
                    f"In {s.n_rated} games started between {start} and {end} ({tz_name}) you scored "
                    f"{pct(s.rated_score)} where {pct(s.expected)} was expected ({fmt_points(s.test.mean)} points per "
                    f"game); in all your other games {fmt_points(rest.test.mean)} per game."
                ),
                severity=severity_from_points(test.mean),
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
    detail += f" where {pct(s.expected)} was expected ({fmt_points(s.test.mean)} per game)." if s.n_rated else "."
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
            f" where {pct(after_two.expected)} was expected ({fmt_points(after_two.test.mean)} per game)."
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
            "Stop after two losses in a row: a losing streak is the most expensive time to keep playing.",
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
        f"You played {n} live games in {n_sessions} session{'s' if n_sessions != 1 else ''} "
        f"(on average {avg:.1f} games, at most {longest})."
    )
    if a.n_rated and r.n_rated:
        text += (
            f" Right after a loss you score {fmt_points(a.test.mean)} points per game against your rating, "
            f"otherwise {fmt_points(r.test.mean)}."
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
    tz, tz_name = resolve_tz(ctx.opt("tz", "UTC"))
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
    fatigue, fatigue_stats = fatigue_insight(scored_tl, th)

    # streaks and rematches
    streak = longest_losing_streak(timeline)
    after_two = summarize([t.game for t in scored_tl if t.loss_run >= 2])
    rematch, rematch_stats = rematch_insight(scored_tl, th)

    # time of day / weekday (in the player's time zone), by start time as chess.com reports it
    parts: list[list[Game]] = [[] for _ in DAY_PARTS]
    weekdays: list[list[Game]] = [[] for _ in range(7)]
    for g in scored:
        moment = g.start_time if g.start_time is not None else g.end_time
        parts[day_part(moment, tz)].append(g)
        weekdays[_utc(moment).astimezone(tz).weekday()].append(g)
    tod_insights, tod_p = time_of_day_insights(parts, scored, tz_name, th)
    part_rows = [(label, summarize(p)) for (label, _), p in zip(DAY_PARTS, parts, strict=True)]
    weekday_rows = [(calendar.day_name[i], summarize(weekdays[i])) for i in range(7)]

    insights = [i for i in (tilt, fatigue) if i] + tod_insights
    insights += [i for i in (rematch, streak_insight(streak, after_two, len(timeline), th)) if i]
    if len(timeline) < th.min_games:
        insights = []

    diff_note = "Difference = your score minus the Elo expected score, per game."
    tables = [
        session_table(sizes),
        _score_table(
            "After a loss, a win or a break",
            "Game started",
            situation_rows,
            f"When each game started relative to the end of your previous live game. {diff_note}",
        ),
        _score_table(
            "By game number in a session",
            "Game in the session",
            fatigue_rows,
            f"A session ends when you pause for more than {th.session_gap_min:g} minutes. {diff_note}",
        ),
        _score_table(
            f"By time of day ({tz_name})",
            "Started",
            part_rows,
            f"Start times in {tz_name}. {diff_note}",
        ),
        _score_table("By weekday", "Day", weekday_rows, f"Days in {tz_name}. {diff_note}"),
    ]
    charts = [
        _delta_chart(
            "Score vs expected after a loss, a win or a break",
            situation_rows,
            "Per game, by when the game started relative to the previous one.",
        ),
        _delta_chart(
            "Score vs expected by game number in a session", fatigue_rows, "Per game, against the Elo expectation."
        ),
        _delta_chart(f"Score vs expected by time of day ({tz_name})", part_rows, "Per game, by start time."),
    ]
    charts = [c for c in charts if any(v is not None for v in c.series[0].values)]

    kpis = [
        Kpi("Sessions", n_sessions, "int", hint=f"a pause of more than {th.session_gap_min:g} min starts a new one"),
        Kpi("Games per session", avg, "float1", hint=f"longest: {longest}"),
        Kpi("After a loss", after_loss.delta, "signed_pct", hint=f"score vs expected, {after_loss.n_rated} games"),
        Kpi("Otherwise", rest.delta, "signed_pct", hint=f"score vs expected, {rest.n_rated} games"),
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
            "time_zone": tz_name,
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
            "weekday": {label: _stats(s) for label, s in weekday_rows},
        },
    )


def _stats(s: ScoreSummary) -> dict[str, Any]:
    return {"n": s.n, "n_rated": s.n_rated, "score": s.score, "expected": s.expected, "delta": s.delta}
