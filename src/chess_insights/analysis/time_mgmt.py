"""Clock & time management: time trouble, flagging, opening pace and think time.

Only live games with clock data count (daily games have no running clock), and
every clock metric is reported per time class, because 1+0 and 15+10 clocks
cannot be compared. Wherever possible the benchmark is *your opponents in the
same games* (rating-matched players on the same clock):

* time trouble: your clock after one of your moves fell below
  max(10% of the starting clock, 5 s), or you flagged;
* flagging: losses on time vs wins on time (a sign test on who flags whom);
* opening pace: the share of the starting clock used on the first 15 moves;
* clock balance: who is ahead on the clock at moves 20 and 30;
* think time: average seconds per move by move number.

Each claim is one test per time class, BH-adjusted over the time classes, under the
project-wide rule ``stats.significance`` at ``stats.STRICT_ALPHA``: the paired tests
against your opponents in the same games have plenty of power for clock habits that
matter, so the strict level costs little.

``time_spent`` is the shared definition of the seconds spent on one ply
(``engine.py`` computes the same thing independently).
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..context import AnalysisContext
from ..models import TIME_CLASSES, Chart, Color, Game, Insight, Kpi, ModuleResult, Series, Table
from ..stats import (
    STRICT_ALPHA,
    MeanTest,
    ScoreSummary,
    bh_adjust,
    clamp,
    mean_test,
    normal_cdf,
    pct,
    severity_from_points,
    significance,
    summarize,
)
from .results import MAX_EXAMPLES, fmt_points

KEY = "time"
TITLE = "Clock & time management"

MIN_PLIES = 4  # shorter games are aborted starts, not clock handling
TROUBLE_FRACTION = 0.10  # time trouble: below this share of the starting clock ...
TROUBLE_FLOOR = 5.0  # ... or below this many seconds, whichever is larger
OPENING_MOVES = 15
CHECKPOINTS = (20, 30)
# Per-game noise floor (share of the starting clock) for paired clock comparisons, so a
# handful of near-identical games can't produce a zero standard error.
CLOCK_NOISE = 0.02
MOVE_BUCKETS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("1–10", 1, 10),
    ("11–20", 11, 20),
    ("21–30", 21, 30),
    ("31–40", 31, 40),
    ("41+", 41, None),
)


@dataclass(frozen=True)
class Thresholds:
    """Minimum sample sizes and effect sizes; override with ``ctx.options["time.<name>"]``."""

    min_games: int = 15  # live games with clocks in a time class before it is reported
    strict_alpha: float = STRICT_ALPHA  # every clock claim (see the module docstring)
    min_trouble_gap: float = 0.10  # your time-trouble rate minus your opponents' in the same games
    min_losses: int = 20  # losses in a time class before judging how many were on time
    min_timeouts: int = 10  # games decided on time (either way) before comparing who flags whom
    min_flag_share: float = 0.15  # share of your losses that were on time, for a weakness
    min_flag_balance: float = 0.20  # (lost on time - won on time) / games decided on time
    slow_opening_ratio: float = 1.5  # your opening clock use divided by your opponents'
    min_opening_gap: float = 0.05  # ... and at least this much more of the starting clock
    min_ahead_share: float = 0.60  # clock strength: ahead at move 20 in at least this share of games

    @classmethod
    def from_ctx(cls, ctx: AnalysisContext) -> "Thresholds":
        return cls(**{f.name: ctx.opt(f"{KEY}.{f.name}", f.default) for f in dataclasses.fields(cls)})


# --------------------------------------------------------------------------- per-game clock facts
def time_spent(game: Game) -> list[Optional[float]]:
    """Seconds the mover spent on each ply (aligned with ``moves_san``); None where clocks are missing.

    For the mover's k-th move: the same mover's previous clock (``base_seconds`` for their
    first move) minus the clock after the move, plus the increment, floored at 0.
    """
    clocks = game.clocks
    base = float(game.base_seconds) if game.base_seconds else None
    inc = float(game.increment or 0)
    out: list[Optional[float]] = []
    for i in range(game.plies):
        after = clocks[i] if i < len(clocks) else None
        before = base if i < 2 else (clocks[i - 2] if i - 2 < len(clocks) else None)
        out.append(None if after is None or before is None else max(0.0, before - after + inc))
    return out


def trouble_threshold(base_seconds: float) -> float:
    """Clock level (seconds) below which a player counts as in time trouble."""
    return max(TROUBLE_FRACTION * base_seconds, TROUBLE_FLOOR)


def flagged_side(game: Game) -> Optional[Color]:
    """Colour that ran out of time, if the game ended on the clock."""
    other: Color = "black" if game.color == "white" else "white"
    if game.termination == "timeout":
        return {"loss": game.color, "win": other}.get(game.outcome)
    if game.termination == "timeout_vs_insufficient":
        return game.mover(game.plies)  # only the side to move can run out of time
    return None


@dataclass
class GameClock:
    """Clock facts for one live game, for both sides."""

    game: Game
    threshold: float  # time-trouble line in seconds
    my_trouble: bool
    opp_trouble: bool
    my_opening: Optional[float]  # share of the starting clock used on the first OPENING_MOVES moves
    opp_opening: Optional[float]
    balance: dict[int, float]  # checkpoint move -> (your clock - opponent's) / starting clock
    my_spent: list[Optional[float]]  # seconds per own move, in move order
    opp_spent: list[Optional[float]]


def _opening_share(spent: Sequence[Optional[float]], base: float) -> Optional[float]:
    first = spent[:OPENING_MOVES]
    if len(first) < OPENING_MOVES or any(s is None for s in first):
        return None
    return sum(s for s in first if s is not None) / base


def game_clock(game: Game) -> Optional[GameClock]:
    """Clock facts for a live game with clock data for both sides; None for any other game."""
    if game.time_class == "daily" or not game.base_seconds or game.plies < MIN_PLIES:
        return None
    other: Color = "black" if game.color == "white" else "white"
    mine = [i for i in range(game.plies) if game.mover(i) == game.color]
    theirs = [i for i in range(game.plies) if game.mover(i) == other]

    def clock(i: int) -> Optional[float]:
        return game.clocks[i] if i < len(game.clocks) else None

    my_clocks = [c for i in mine if (c := clock(i)) is not None]
    opp_clocks = [c for i in theirs if (c := clock(i)) is not None]
    if not my_clocks or not opp_clocks:
        return None
    base = float(game.base_seconds)
    threshold = trouble_threshold(base)
    flagged = flagged_side(game)
    spent = time_spent(game)
    my_spent, opp_spent = [spent[i] for i in mine], [spent[i] for i in theirs]
    balance: dict[int, float] = {}
    for move in CHECKPOINTS:
        if len(mine) >= move and len(theirs) >= move:
            a, b = clock(mine[move - 1]), clock(theirs[move - 1])
            if a is not None and b is not None:
                balance[move] = (a - b) / base
    return GameClock(
        game=game,
        threshold=threshold,
        my_trouble=min(my_clocks) < threshold or flagged == game.color,
        opp_trouble=min(opp_clocks) < threshold or flagged == other,
        my_opening=_opening_share(my_spent, base),
        opp_opening=_opening_share(opp_spent, base),
        balance=balance,
        my_spent=my_spent,
        opp_spent=opp_spent,
    )


# --------------------------------------------------------------------------- tests on paired data
def paired_rate_test(mine: Sequence[bool], theirs: Sequence[bool]) -> MeanTest:
    """McNemar z-test of rate(mine) - rate(theirs) for yes/no outcomes paired by game."""
    n = len(mine)
    only_me = sum(1 for a, b in zip(mine, theirs, strict=True) if a and not b)
    only_them = sum(1 for a, b in zip(mine, theirs, strict=True) if b and not a)
    discordant = only_me + only_them
    if n == 0 or discordant == 0:
        return MeanTest(n, 0.0, float("inf"), 0.0, 1.0)
    se = math.sqrt(discordant) / n
    diff = (only_me - only_them) / n
    z = diff / se
    return MeanTest(n, diff, se, z, 2.0 * (1.0 - normal_cdf(abs(z))))


def sign_test(plus: int, minus: int) -> MeanTest:
    """Sign test: are ``plus`` and ``minus`` a fair coin? ``mean`` = (plus - minus) / (plus + minus)."""
    k = plus + minus
    if k == 0:
        return MeanTest(0, 0.0, float("inf"), 0.0, 1.0)
    mean = (plus - minus) / k
    se = 1.0 / math.sqrt(k)
    z = mean / se
    return MeanTest(k, mean, se, z, 2.0 * (1.0 - normal_cdf(abs(z))))


def _paired_mean_test(diffs: Sequence[float]) -> MeanTest:
    return mean_test(diffs, min_se=CLOCK_NOISE / math.sqrt(len(diffs))) if diffs else mean_test([])


# --------------------------------------------------------------------------- per time class
@dataclass
class Checkpoint:
    move: int
    n: int  # games where both players made this many moves (with clocks)
    behind: int  # ... and you had less time than your opponent
    ahead: int
    test: MeanTest  # mean of (your clock - opponent's) / starting clock

    @property
    def behind_rate(self) -> Optional[float]:
        return self.behind / self.n if self.n else None

    @property
    def ahead_rate(self) -> Optional[float]:
        return self.ahead / self.n if self.n else None


@dataclass
class ClassStats:
    """Clock statistics for one time class."""

    time_class: str
    clocks: list[GameClock]
    trouble_rate: float
    opp_trouble_rate: float
    trouble_test: MeanTest  # paired: your time-trouble rate minus your opponents'
    in_trouble: ScoreSummary
    not_in_trouble: ScoreSummary
    losses: int
    lost_on_time: int
    won_on_time: int
    flag_test: MeanTest  # sign test, mean = (lost on time - won on time) / games decided on time
    opening_n: int
    my_opening: Optional[float]
    opp_opening: Optional[float]
    opening_test: MeanTest  # paired: your opening share minus your opponent's
    checkpoints: dict[int, Checkpoint]

    @property
    def n(self) -> int:
        return len(self.clocks)

    @property
    def games(self) -> list[Game]:
        return [c.game for c in self.clocks]

    @property
    def flag_loss_share(self) -> Optional[float]:
        return self.lost_on_time / self.losses if self.losses else None

    @property
    def opening_ratio(self) -> Optional[float]:
        if self.my_opening is None or not self.opp_opening:
            return None
        return self.my_opening / self.opp_opening


def class_stats(time_class: str, clocks: Sequence[GameClock]) -> ClassStats:
    n = len(clocks)
    games = [c.game for c in clocks]
    mine, theirs = [c.my_trouble for c in clocks], [c.opp_trouble for c in clocks]
    lost = sum(1 for g in games if g.outcome == "loss" and g.termination == "timeout")
    won = sum(1 for g in games if g.outcome == "win" and g.termination == "timeout")
    opening: list[tuple[float, float]] = [
        (c.my_opening, c.opp_opening) for c in clocks if c.my_opening is not None and c.opp_opening is not None
    ]
    checkpoints = {}
    for move in CHECKPOINTS:
        diffs = [c.balance[move] for c in clocks if move in c.balance]
        checkpoints[move] = Checkpoint(
            move=move,
            n=len(diffs),
            behind=sum(1 for d in diffs if d < 0),
            ahead=sum(1 for d in diffs if d > 0),
            test=_paired_mean_test(diffs),
        )
    return ClassStats(
        time_class=time_class,
        clocks=list(clocks),
        trouble_rate=sum(mine) / n if n else 0.0,
        opp_trouble_rate=sum(theirs) / n if n else 0.0,
        trouble_test=paired_rate_test(mine, theirs),
        in_trouble=summarize([c.game for c in clocks if c.my_trouble]),
        not_in_trouble=summarize([c.game for c in clocks if not c.my_trouble]),
        losses=sum(1 for g in games if g.outcome == "loss"),
        lost_on_time=lost,
        won_on_time=won,
        flag_test=sign_test(lost, won),
        opening_n=len(opening),
        my_opening=sum(a for a, _ in opening) / len(opening) if opening else None,
        opp_opening=sum(b for _, b in opening) / len(opening) if opening else None,
        opening_test=_paired_mean_test([a - b for a, b in opening]),
        checkpoints=checkpoints,
    )


def _in_bucket(spent: Sequence[Optional[float]], lo: int, hi: Optional[int]) -> list[float]:
    """Known think times of moves number ``lo``..``hi`` (1-based, ``hi`` None = no upper limit)."""
    return [s for k, s in enumerate(spent, 1) if s is not None and k >= lo and (hi is None or k <= hi)]


def think_profile(clocks: Sequence[GameClock]) -> list[tuple[str, Optional[float], Optional[float], int, int]]:
    """(bucket label, your avg seconds per move, opponents' avg, your moves, their moves) per move bucket."""
    rows = []
    for label, lo, hi in MOVE_BUCKETS:
        mine = [s for c in clocks for s in _in_bucket(c.my_spent, lo, hi)]
        theirs = [s for c in clocks for s in _in_bucket(c.opp_spent, lo, hi)]
        rows.append(
            (
                label,
                sum(mine) / len(mine) if mine else None,
                sum(theirs) / len(theirs) if theirs else None,
                len(mine),
                len(theirs),
            )
        )
    return rows


def _class_order(time_class: str, n: int) -> tuple[int, int, str]:
    rank = TIME_CLASSES.index(time_class) if time_class in TIME_CLASSES else len(TIME_CLASSES)
    return (-n, rank, time_class)


# --------------------------------------------------------------------------- insights
def _score_phrase(s: ScoreSummary) -> str:
    """'54% where 49% was expected (+0.05 per game, 40 games)' (plain score without ratings)."""
    if s.n_rated == 0 or s.expected is None:
        return f"{pct(s.score)} in {s.n} games"
    return (
        f"{pct(s.rated_score)} where {pct(s.expected)} was expected "
        f"({fmt_points(s.test.mean)} per game, {s.n_rated} games)"
    )


def _slowest_first(clocks: Sequence[GameClock]) -> list[str]:
    """URLs of the losses with the slowest openings first, then other games, slowest first."""
    ranked = sorted(
        clocks, key=lambda c: (c.game.outcome != "loss", -(c.my_opening or 0.0), -c.game.end_time.timestamp())
    )
    return [c.game.url for c in ranked if c.game.url][:MAX_EXAMPLES]


def trouble_insight(cs: ClassStats, p_adj: float, th: Thresholds) -> Optional[Insight]:
    significant, confidence = significance(cs.trouble_test, th.min_games, p_adj, alpha=th.strict_alpha)
    gap = cs.trouble_rate - cs.opp_trouble_rate
    if cs.n < th.min_games or gap < th.min_trouble_gap or not significant:
        return None
    tc = cs.time_class
    trouble_games = [c.game for c in cs.clocks if c.my_trouble]
    trouble_losses = [g for g in trouble_games if g.outcome == "loss"]
    k = min(MAX_EXAMPLES, len(trouble_losses)) or "few"
    detail = (
        f"In {cs.n} {tc} games you ran low on time (below 10% of your starting clock, at least 5 s, or flagged) in "
        f"{pct(cs.trouble_rate)} of them; your opponents did in {pct(cs.opp_trouble_rate)} of the same games. "
        f"In those time-trouble games you scored {_score_phrase(cs.in_trouble)}"
    )
    detail += f"; in the rest {_score_phrase(cs.not_in_trouble)}." if cs.not_in_trouble.n else "."
    return Insight(
        id=f"{KEY}.weakness.time-trouble-{tc}",
        kind="weakness",
        category="time",
        title=f"You get into time trouble far more often than your opponents in {tc}",
        detail=detail,
        severity=clamp(gap / 0.30),
        confidence=confidence,
        evidence={
            "time_class": tc,
            "n": cs.n,
            "trouble_rate": cs.trouble_rate,
            "opponent_trouble_rate": cs.opp_trouble_rate,
            "p_value": cs.trouble_test.p_value,
            "p_adjusted": p_adj,
            "score_in_trouble": cs.in_trouble.rated_score,
            "expected_in_trouble": cs.in_trouble.expected,
            "score_otherwise": cs.not_in_trouble.rated_score,
            "expected_otherwise": cs.not_in_trouble.expected,
        },
        study=[
            f"Set clock checkpoints for {tc}: reach move 20 with at least half of your starting time left.",
            f"Replay your last {k} time-trouble losses and find the one or two moves you spent longest on: "
            "was that think worth it?",
            "Move quickly in simple positions (recaptures, forced replies, obvious developing moves) and save your "
            "time for the critical moments.",
        ],
        example_games=[g.url for g in sorted(trouble_losses, key=lambda g: g.end_time, reverse=True) if g.url][
            :MAX_EXAMPLES
        ],
    )


def flag_insight(cs: ClassStats, p_adj: float, th: Thresholds) -> Optional[Insight]:
    if cs.losses < th.min_losses:
        return None
    significant, confidence = significance(cs.flag_test, th.min_timeouts, p_adj, alpha=th.strict_alpha)
    share = cs.flag_loss_share or 0.0
    if cs.flag_test.mean < th.min_flag_balance or share < th.min_flag_share or not significant:
        return None
    tc = cs.time_class
    net = cs.lost_on_time - cs.won_on_time
    timeouts = [g for g in cs.games if g.outcome == "loss" and g.termination == "timeout"]
    return Insight(
        id=f"{KEY}.weakness.lost-on-time-{tc}",
        kind="weakness",
        category="time",
        title=f"You lose on time too often in {tc}",
        detail=(
            f"{cs.lost_on_time} of your {cs.losses} {tc} losses ({pct(share)}) were on time, while you won on time "
            f"only {cs.won_on_time} times in the same {cs.n} games: {net} more flags against you than for you."
        ),
        severity=severity_from_points(net / cs.n, full_at=0.10),
        confidence=confidence,
        evidence={
            "time_class": tc,
            "n": cs.n,
            "losses": cs.losses,
            "lost_on_time": cs.lost_on_time,
            "won_on_time": cs.won_on_time,
            "share_of_losses": share,
            "p_value": cs.flag_test.p_value,
            "p_adjusted": p_adj,
        },
        study=[
            f"Replay your {min(MAX_EXAMPLES, len(timeouts))} most recent losses on time and note how much time you had "
            "left at moves 20 and 30: that is where the flag was really lost.",
            "Once you are low on time, switch to safe, fast moves: keep every piece protected and avoid "
            "complications.",
            "When you are ahead on the board but behind on the clock, trade pieces early so the position becomes "
            "easy to play quickly.",
        ],
        example_games=[g.url for g in sorted(timeouts, key=lambda g: g.end_time, reverse=True) if g.url][
            :MAX_EXAMPLES
        ],
    )


def opening_insight(cs: ClassStats, p_adj: float, th: Thresholds) -> Optional[Insight]:
    ratio = cs.opening_ratio
    if cs.opening_n < th.min_games or ratio is None or cs.my_opening is None or cs.opp_opening is None:
        return None
    significant, confidence = significance(cs.opening_test, th.min_games, p_adj, alpha=th.strict_alpha)
    gap = cs.my_opening - cs.opp_opening
    if ratio < th.slow_opening_ratio or gap < th.min_opening_gap or not significant:
        return None
    tc = cs.time_class
    slow = [c for c in cs.clocks if c.my_opening is not None and c.opp_opening is not None]
    families = Counter(c.game.opening_family for c in slow if c.game.opening_family and not c.game.initial_fen)
    cp = cs.checkpoints.get(CHECKPOINTS[0])
    detail = (
        f"In {cs.opening_n} {tc} games that reached move {OPENING_MOVES}, you used {pct(cs.my_opening)} of your "
        f"starting clock on your first {OPENING_MOVES} moves, {ratio:.1f}× as much as your opponents in the same "
        f"games ({pct(cs.opp_opening)})."
    )
    if cp and cp.n:
        detail += f" You were behind on the clock at move {cp.move} in {pct(cp.behind_rate)} of {cp.n} games."
    (base, inc), _ = Counter((c.game.base_seconds or 0, c.game.increment) for c in slow).most_common(1)[0]
    per_move = max(1, round(cs.opp_opening * base / OPENING_MOVES))
    study = [
        f"Budget about {per_move} seconds per move for your first {OPENING_MOVES} moves in your {base / 60:g}+{inc} "
        "games (your opponents' pace) and keep the time you save for the middlegame.",
        "When you leave your preparation, play on principles (develop, castle, fight for the centre) instead of "
        "calculating long lines in the first moves.",
    ]
    if families:
        fam, count = max(families.items(), key=lambda kv: (kv[1], kv[0]))
        study.insert(
            0,
            f"Your most common opening in these games is the {fam} ({count} games): learn its first 10-12 moves "
            "and typical plans well enough to play them in a few seconds each.",
        )
    return Insight(
        id=f"{KEY}.weakness.slow-opening-{tc}",
        kind="weakness",
        category="time",
        title=f"You spend too long in the opening in {tc}",
        detail=detail,
        severity=clamp((ratio - 1.0) / 1.0),
        confidence=confidence,
        evidence={
            "time_class": tc,
            "n": cs.opening_n,
            "your_share": cs.my_opening,
            "opponent_share": cs.opp_opening,
            "ratio": ratio,
            "p_value": cs.opening_test.p_value,
            "p_adjusted": p_adj,
        },
        study=study[:4],
        example_games=_slowest_first(slow),
    )


def clock_strength_insight(cs: ClassStats, p_flags: float, p_balance: float, th: Thresholds) -> Optional[Insight]:
    cp = cs.checkpoints.get(CHECKPOINTS[0])
    if cp is None or cp.n < th.min_games or (cp.ahead_rate or 0.0) < th.min_ahead_share:
        return None
    balance_sig, balance_conf = significance(cp.test, th.min_games, p_balance, alpha=th.strict_alpha)
    flag_sig, flag_conf = significance(cs.flag_test, th.min_timeouts, p_flags, alpha=th.strict_alpha)
    confidence = min(balance_conf, flag_conf)
    if cp.test.mean <= 0 or cs.flag_test.mean > -th.min_flag_balance or not (balance_sig and flag_sig):
        return None
    tc = cs.time_class
    wins_on_time = [g for g in cs.games if g.outcome == "win" and g.termination == "timeout"]
    return Insight(
        id=f"{KEY}.strength.clock-handling-{tc}",
        kind="strength",
        category="time",
        title=f"You manage the clock well in {tc}",
        detail=(
            f"At move {cp.move} you had more time than your opponent in {pct(cp.ahead_rate)} of {cp.n} {tc} games "
            f"(on average {pct(cp.test.mean)} of the starting clock more), and you won on time {cs.won_on_time} "
            f"times while losing on time {cs.lost_on_time} times."
        ),
        severity=severity_from_points((cs.won_on_time - cs.lost_on_time) / cs.n, full_at=0.10),
        confidence=confidence,
        evidence={
            "time_class": tc,
            "n": cp.n,
            "ahead_rate": cp.ahead_rate,
            "mean_lead": cp.test.mean,
            "won_on_time": cs.won_on_time,
            "lost_on_time": cs.lost_on_time,
            "p_value_balance": cp.test.p_value,
            "p_value_flags": cs.flag_test.p_value,
        },
        study=[
            "Keep your pace: in objectively worse but complicated positions, play on and make your opponent spend "
            "their time.",
            "Use your time lead: in the critical middlegame moments you can afford an extra 20-30 seconds to "
            "calculate.",
        ],
        example_games=[g.url for g in sorted(wins_on_time, key=lambda g: g.end_time, reverse=True) if g.url][
            :MAX_EXAMPLES
        ],
    )


# --------------------------------------------------------------------------- tables & charts
def trouble_table(classes: Sequence[ClassStats]) -> Table:
    rows = [
        [
            cs.time_class.capitalize(),
            cs.n,
            cs.trouble_rate,
            cs.opp_trouble_rate,
            cs.in_trouble.rated_score,
            cs.in_trouble.delta,
            cs.not_in_trouble.rated_score,
            cs.not_in_trouble.delta,
        ]
        for cs in classes
    ]
    return Table(
        title="Time trouble",
        columns=[
            "Time control",
            "Games",
            "You in time trouble",
            "Opponents in time trouble",
            "Score in time trouble",
            "Difference in time trouble",
            "Score otherwise",
            "Difference otherwise",
        ],
        rows=rows,
        formats=["text", "int", "pct", "pct", "pct", "signed_pct", "pct", "signed_pct"],
        note=(
            "Time trouble: after one of your moves your clock was below 10% of the starting time (at least 5 s), "
            "or you ran out of time. Opponents are measured in the same games. Scores cover games with ratings; "
            "Difference is your score minus the Elo expected score, per game."
        ),
    )


def flag_table(classes: Sequence[ClassStats]) -> Table:
    rows = [
        [
            cs.time_class.capitalize(),
            cs.n,
            cs.losses,
            cs.lost_on_time,
            cs.flag_loss_share,
            cs.won_on_time,
            cs.won_on_time - cs.lost_on_time,
        ]
        for cs in classes
    ]
    return Table(
        title="Winning and losing on time",
        columns=["Time control", "Games", "Losses", "Lost on time", "Share of losses", "Won on time", "Net"],
        rows=rows,
        formats=["text", "int", "int", "int", "pct", "int", "signed_int"],
        note="Net = games you won on time minus games you lost on time.",
    )


def pace_table(classes: Sequence[ClassStats]) -> Table:
    rows = []
    for cs in classes:
        row: list[Any] = [cs.time_class.capitalize(), cs.opening_n, cs.my_opening, cs.opp_opening]
        for move in CHECKPOINTS:
            cp = cs.checkpoints[move]
            row += [cp.n, cp.behind_rate]
        rows.append(row)
    columns = ["Time control", f"Games to move {OPENING_MOVES}", "Clock used: you", "Clock used: opponents"]
    for move in CHECKPOINTS:
        columns += [f"Games to move {move}", f"Behind at move {move}"]
    return Table(
        title="Opening pace and clock balance",
        columns=columns,
        rows=rows,
        formats=["text", "int", "pct", "pct"] + ["int", "pct"] * len(CHECKPOINTS),
        note=(
            f"Clock used = share of the starting time spent on the first {OPENING_MOVES} moves (increments "
            "included), in games where both players got that far. Behind = you had less time left than your "
            "opponent after each of you made that many moves."
        ),
    )


def think_table(time_class: str, profile: Sequence[tuple[str, Optional[float], Optional[float], int, int]]) -> Table:
    return Table(
        title=f"Think time by move number ({time_class})",
        columns=["Moves", "You (avg per move)", "Opponents (avg per move)", "Your moves", "Opponent moves"],
        rows=[list(row) for row in profile],
        formats=["text", "seconds", "seconds", "int", "int"],
        note="Average seconds spent per move, with the increment added back (time actually used).",
    )


def trouble_chart(classes: Sequence[ClassStats]) -> Chart:
    return Chart(
        kind="bar",
        title="Games in time trouble",
        labels=[cs.time_class.capitalize() for cs in classes],
        series=[
            Series("You", [cs.trouble_rate for cs in classes]),
            Series("Opponents", [cs.opp_trouble_rate for cs in classes]),
        ],
        value_format="pct",
        note="Share of games in which each side's clock fell below 10% of the starting time (at least 5 s).",
    )


def think_chart(time_class: str, profile: Sequence[tuple[str, Optional[float], Optional[float], int, int]]) -> Chart:
    return Chart(
        kind="bar",
        title=f"Average think time per move ({time_class})",
        labels=[row[0] for row in profile],
        series=[Series("You", [row[1] for row in profile]), Series("Opponents", [row[2] for row in profile])],
        value_format="seconds",
        note="By move number. Later buckets only include games that lasted that long.",
    )


# --------------------------------------------------------------------------- module entry point
def _kpis(main: ClassStats) -> list[Kpi]:
    tc = main.time_class
    kpis = [
        Kpi(
            f"Time-trouble games ({tc})",
            main.trouble_rate,
            "pct",
            hint=f"opponents: {pct(main.opp_trouble_rate)} of the same {main.n} games",
        ),
        Kpi(
            f"Lost on time ({tc})",
            main.lost_on_time,
            "int",
            hint=f"{pct(main.flag_loss_share)} of your losses; you won on time {main.won_on_time}",
        ),
    ]
    if main.opening_n:
        kpis.append(
            Kpi(
                f"Clock used by move {OPENING_MOVES} ({tc})",
                main.my_opening,
                "pct",
                hint=f"opponents: {pct(main.opp_opening)}",
            )
        )
    cp = main.checkpoints[CHECKPOINTS[0]]
    if cp.n:
        kpis.append(
            Kpi(f"Behind on the clock at move {cp.move}", cp.behind_rate, "pct", hint=f"{cp.n} {tc} games")
        )
    return kpis


def _stats(cs: ClassStats) -> dict[str, Any]:
    return {
        "n": cs.n,
        "trouble_rate": cs.trouble_rate,
        "opponent_trouble_rate": cs.opp_trouble_rate,
        "trouble_p_value": cs.trouble_test.p_value,
        "score_in_trouble": cs.in_trouble.rated_score,
        "delta_in_trouble": cs.in_trouble.delta,
        "score_otherwise": cs.not_in_trouble.rated_score,
        "delta_otherwise": cs.not_in_trouble.delta,
        "losses": cs.losses,
        "lost_on_time": cs.lost_on_time,
        "won_on_time": cs.won_on_time,
        "flag_p_value": cs.flag_test.p_value,
        "opening_n": cs.opening_n,
        "opening_share": cs.my_opening,
        "opponent_opening_share": cs.opp_opening,
        "opening_ratio": cs.opening_ratio,
        "checkpoints": {
            str(m): {"n": cp.n, "behind": cp.behind, "ahead": cp.ahead, "mean_lead": cp.test.mean if cp.n else None}
            for m, cp in cs.checkpoints.items()
        },
    }


def _not_enough(summary: str, **stats: Any) -> ModuleResult:
    return ModuleResult(key=KEY, title=TITLE, summary=f"Not enough data: {summary}", stats=stats)


def analyze(ctx: AnalysisContext) -> ModuleResult:
    """Time trouble, flagging, opening pace, clock balance and think time, per live time class."""
    th = Thresholds.from_ctx(ctx)
    live = [g for g in ctx.games if g.time_class != "daily"]
    if not live:
        return _not_enough("there are no live games to analyse (daily games have no running clock).", n=0)
    clocked = [gc for g in sorted(live, key=lambda g: g.end_time) if (gc := game_clock(g))]
    if not clocked:
        return _not_enough(
            f"none of your {len(live)} live games has usable clock data (clocks for both players and at least "
            f"{MIN_PLIES} plies).",
            n=0,
            n_live=len(live),
        )

    by_class: dict[str, list[GameClock]] = {}
    for gc in clocked:
        by_class.setdefault(gc.game.time_class, []).append(gc)
    ordered = sorted(by_class, key=lambda tc: _class_order(tc, len(by_class[tc])))
    classes = [class_stats(tc, by_class[tc]) for tc in ordered if len(by_class[tc]) >= th.min_games]
    if not classes:
        counts = ", ".join(f"{len(by_class[tc])} {tc}" for tc in ordered)
        return _not_enough(
            f"clock statistics need at least {th.min_games} live games with clocks in one time control "
            f"(you have {counts}).",
            n=len(clocked),
            n_live=len(live),
        )

    # one test per time class for each kind of claim: adjust for testing several classes at once
    p_trouble = bh_adjust([cs.trouble_test.p_value for cs in classes])
    p_flags = bh_adjust([cs.flag_test.p_value for cs in classes])
    p_opening = bh_adjust([cs.opening_test.p_value for cs in classes])
    p_balance = bh_adjust([cs.checkpoints[CHECKPOINTS[0]].test.p_value for cs in classes])
    insights: list[Insight] = []
    for i, cs in enumerate(classes):
        found = (
            trouble_insight(cs, p_trouble[i], th),
            flag_insight(cs, p_flags[i], th),
            opening_insight(cs, p_opening[i], th),
            clock_strength_insight(cs, p_flags[i], p_balance[i], th),
        )
        insights.extend(ins for ins in found if ins)

    main = classes[0]
    profile = think_profile(main.clocks)
    tables = [trouble_table(classes), flag_table(classes), pace_table(classes), think_table(main.time_class, profile)]
    charts = [trouble_chart(classes)]
    if any(row[1] is not None for row in profile):
        charts.append(think_chart(main.time_class, profile))

    summary = (
        f"In {main.time_class} ({main.n} games with clocks) you got into time trouble in {pct(main.trouble_rate)} "
        f"of games (your opponents in {pct(main.opp_trouble_rate)}) and lost {main.lost_on_time} on time while "
        f"winning {main.won_on_time} on time."
    )
    ranked = sorted((i for i in insights if i.kind != "observation"), key=lambda i: -i.priority)
    if ranked:
        summary += f" Key finding: {ranked[0].title}."
    skipped = [tc for tc in ordered if len(by_class[tc]) < th.min_games]
    return ModuleResult(
        key=KEY,
        title=TITLE,
        summary=summary,
        kpis=_kpis(main),
        tables=tables,
        charts=charts,
        insights=insights,
        stats={
            "n": len(clocked),
            "n_live": len(live),
            "main_time_class": main.time_class,
            "by_time_class": {cs.time_class: _stats(cs) for cs in classes},
            "skipped_time_classes": {tc: len(by_class[tc]) for tc in skipped},
            "think_time": [
                {"moves": label, "you": mine, "opponents": theirs, "your_moves": n_me, "opponent_moves": n_them}
                for label, mine, theirs, n_me, n_them in profile
            ],
        },
    )
