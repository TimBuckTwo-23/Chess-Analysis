"""Engine review: accuracy, blunder rates, weak game phases, conversion, tactics and opening outcomes.

Needs Stockfish analysis (``ctx.evals``, produced by ``engine.py``). Every rate is
benchmarked against the OPPONENTS' moves in the same games: they are rating-matched
players facing the same positions on the same clock, which controls for position
difficulty and playing strength far better than any fixed "good accuracy" number.

Games, not moves, are the unit of every significance test. Errors cluster within games
(one bad game holds several blunders), so move-level rates are compared with a
cluster-robust test over per-game counts (:func:`clustered_rate_test`); game-level
measures (conversion, opening evaluations) are tested per game directly. Claims follow
the project-wide rule (``stats.significance``), and comparisons repeated over groups
(phases, time classes, openings, tactic types) are Benjamini-Hochberg adjusted.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from ..context import AnalysisContext
from ..engine import PHASES
from ..models import TIME_CLASSES, Chart, Game, GameEval, Insight, Kpi, ModuleResult, PlyEval, Series, Table
from ..stats import (
    STRICT_ALPHA,
    MeanTest,
    bh_adjust,
    clamp,
    mean_test,
    normal_cdf,
    pct,
    sample_confidence,
    significance,
    two_proportion_test,
)
from .results import MAX_EXAMPLES, fmt_signed, slugify

KEY = "engine"
TITLE = "Engine review"

MIN_PLIES = 4  # shorter games are aborted starts, not skill
OPENING_PLY = 20  # opening outcome = eval after both players' 10th move, relative to the start position
OPENING_CP_CAP = 500  # per-game cap on that eval, so one early blunder doesn't dominate an average
OPENING_EVAL_SD = 150.0  # floor on the per-game spread of the move-10 eval (cp) for the significance test
PIECES = (("K", "King"), ("Q", "Queen"), ("R", "Rook"), ("B", "Bishop"), ("N", "Knight"), ("P", "Pawn"))
MOVE_BUCKETS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("1–10", 1, 10),
    ("11–20", 11, 20),
    ("21–30", 21, 30),
    ("31–40", 31, 40),
    ("41+", 41, None),
)


@dataclass(frozen=True)
class Thresholds:
    """Minimum sample sizes and effect sizes; override with ``ctx.options["engine.<name>"]``."""

    min_games: int = 10  # analysed games before any strength / weakness is claimed (the tests' unit)
    min_confidence: float = 0.5  # confidence needed for a strength / weakness (insights.MIN_CONFIDENCE)
    max_p_value: float = STRICT_ALPHA  # ... and the (adjusted) test must be significant at this level (stats policy)
    min_moves: int = 300  # your analysed moves before overall blunder / tactics comparisons
    blunder_ratio: float = 1.25  # your blunder rate vs your opponents' (or the inverse, for a strength)
    min_phase_moves: int = 150  # your moves in a phase before judging it
    phase_ratio: float = 1.3  # your mistake+blunder rate in a phase vs your opponents'
    pressure_fraction: float = 0.15  # "short of time": clock before the move below this share of the start
    min_pressure_moves: int = 30  # your moves while short of time, per time class
    pressure_ratio: float = 2.0  # blunder rate short of time vs otherwise
    pressure_peer_ratio: float = 1.3  # ... and vs your opponents, relative to the same with time left (per phase)
    win_threshold: float = 85.0  # win % that counts as a winning position (your point of view)
    loss_threshold: float = 15.0  # win % that counts as a lost position
    min_conversion_games: int = 10  # games that reached a winning (or lost) position, for each side
    min_conversion_gap: float = 0.10  # your conversion rate vs your opponents'
    tactic_ratio: float = 1.3  # your missed-tactic / hung-material rate vs your opponents'
    min_missed_mates: int = 3  # missed forced mates before they are mentioned
    min_anatomy_blunders: int = 10  # blunders before describing which pieces / moves they come from
    min_opening_games: int = 5  # analysed games for an opening row
    min_opening_insight_games: int = 8  # ... and for an opening strength / weakness
    opening_eval_threshold: float = 60.0  # average eval gained / lost by move 10 (cp): a clearly good / bad opening
    min_month_games: int = 3  # analysed games for a point on the accuracy trend

    @classmethod
    def from_ctx(cls, ctx: AnalysisContext) -> "Thresholds":
        return cls(**{f.name: ctx.opt(f"{KEY}.{f.name}", f.default) for f in dataclasses.fields(cls)})


# --------------------------------------------------------------------------- counting
@dataclass
class Tally:
    """Move counts for one side (you or your opponents) over some set of positions."""

    moves: int = 0
    inaccuracies: int = 0
    mistakes: int = 0
    blunders: int = 0
    cp_loss: int = 0
    missed_tactics: int = 0
    hung: int = 0
    missed_mates: int = 0
    allowed_mates: int = 0

    def add(self, p: PlyEval) -> None:
        self.moves += 1
        self.inaccuracies += int(p.judgement == "inaccuracy")
        self.mistakes += int(p.judgement == "mistake")
        self.blunders += int(p.judgement == "blunder")
        self.cp_loss += max(0, int(p.cp_loss or 0))
        tags = p.tags or ()
        self.missed_tactics += int("missed_tactic" in tags)
        self.hung += int("hung_material" in tags)
        self.missed_mates += int("missed_mate" in tags)
        self.allowed_mates += int("allowed_mate" in tags)

    @property
    def errors(self) -> int:
        """Mistakes plus blunders."""
        return self.mistakes + self.blunders

    def per100(self, count: int) -> Optional[float]:
        return 100.0 * count / self.moves if self.moves else None

    @property
    def acpl(self) -> Optional[float]:
        return self.cp_loss / self.moves if self.moves else None

    def __add__(self, other: "Tally") -> "Tally":
        return Tally(*(getattr(self, f.name) + getattr(other, f.name) for f in dataclasses.fields(Tally)))


@dataclass
class Analysed:
    """One game joined with its engine analysis (plies sorted, out-of-range plies dropped)."""

    game: Game
    ev: GameEval
    plies: list[PlyEval]

    def mine(self) -> list[PlyEval]:
        return [p for p in self.plies if p.is_user]

    def pov_win(self) -> list[float]:
        """Your win % in every analysed position (before each ply, and after the last one)."""
        out = []
        for p in self.plies:
            out.append(p.win_before if p.is_user else 100.0 - p.win_before)
            out.append(p.win_after if p.is_user else 100.0 - p.win_after)
        return out

    def count(self, pred: Callable[[PlyEval], bool]) -> int:
        return sum(1 for p in self.plies if p.is_user and pred(p))


def join(ctx: AnalysisContext) -> list[Analysed]:
    """Analysed games in the current selection, oldest first."""
    out = []
    for g in sorted(ctx.games, key=lambda g: g.end_time):
        ev = ctx.evals.get(g.game_id)
        if ev is None or g.plies < MIN_PLIES:
            continue
        plies = sorted((p for p in ev.plies if 0 <= p.ply < g.plies), key=lambda p: p.ply)
        if plies:
            out.append(Analysed(g, ev, plies))
    return out


def tally(
    records: Iterable[Analysed], mine: bool, pred: Callable[[Analysed, PlyEval], bool] = lambda r, p: True
) -> Tally:
    """Moves by you (``mine``) or your opponents that satisfy ``pred``."""
    t = Tally()
    for r in records:
        for p in r.plies:
            if p.is_user == mine and pred(r, p):
                t.add(p)
    return t


def per_game(
    records: Iterable[Analysed], mine: bool, pred: Callable[[Analysed, PlyEval], bool] = lambda r, p: True
) -> list[Tally]:
    """Like :func:`tally`, but one Tally per game (in ``records`` order): the input of the significance tests."""
    return [tally([r], mine, pred) for r in records]


def total(tallies: Iterable[Tally]) -> Tally:
    return sum(tallies, Tally())


def clustered_rate_test(a: Sequence[Tally], b: Sequence[Tally], count: Callable[[Tally], int]) -> MeanTest:
    """z-test of rate(a) - rate(b), where rate = sum(count) / sum(moves), with GAMES as the unit.

    ``a[i]`` and ``b[i]`` are the same game (you and your opponent in it, or your moves short of time and
    your other moves). Moves within a game are not independent observations (one bad game produces
    several blunders), so the variance is the cluster-robust (sandwich) variance of the two ratio
    estimators, from per-game residuals; pairing the two sides of each game also removes what the game
    itself contributes to both. It is never taken below the plain binomial variance. The result's ``n``
    is the number of games with moves on either side.
    """
    return clustered_share_test([(count(x), x.moves) for x in a], [(count(y), y.moves) for y in b])


def clustered_share_test(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]) -> MeanTest:
    """:func:`clustered_rate_test` on per-game ``(events, out of)`` pairs, e.g. (missed tactics, errors)."""
    games = [(x1, m1, x2, m2) for (x1, m1), (x2, m2) in zip(a, b) if m1 or m2]
    n = len(games)
    x1, n1 = sum(g[0] for g in games), sum(g[1] for g in games)
    x2, n2 = sum(g[2] for g in games), sum(g[3] for g in games)
    if n < 2 or n1 == 0 or n2 == 0:
        return MeanTest(n, 0.0, float("inf"), 0.0, 1.0)
    p1, p2 = x1 / n1, x2 / n2
    residuals = [(c1 - p1 * m1) / n1 - (c2 - p2 * m2) / n2 for c1, m1, c2, m2 in games]
    variance = n / (n - 1) * sum(r * r for r in residuals)
    pooled = (x1 + x2) / (n1 + n2)
    variance = max(variance, pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if variance <= 0.0:
        return MeanTest(n, p1 - p2, float("inf"), 0.0, 1.0)
    se = math.sqrt(variance)
    z = (p1 - p2) / se
    return MeanTest(n, p1 - p2, se, z, 2.0 * (1.0 - normal_cdf(abs(z))))


def _claim(test: MeanTest, th: "Thresholds", p_adjusted: Optional[float] = None, min_n: Optional[int] = None):
    """(significant, confidence) under the project-wide rule, with this module's alpha and confidence floor."""
    significant, confidence = significance(test, th.min_games if min_n is None else min_n, p_adjusted,
                                           alpha=th.max_p_value)
    return significant and confidence >= th.min_confidence, confidence


def _clearly_higher(a: Optional[float], b: Optional[float], ratio: float) -> bool:
    """a is at least ``ratio`` times b (and strictly above it; any positive a beats b = 0)."""
    return a is not None and b is not None and a > b and a >= ratio * b


def _rate_ratio(x1: int, n1: int, x2: int, n2: int) -> Optional[float]:
    """(x1 / n1) / (x2 / n2) with half an event added to each count, so a zero doesn't make it 0 or infinite."""
    if n1 <= 0 or n2 <= 0:
        return None
    return ((x1 + 0.5) / n1) / ((x2 + 0.5) / n2)


def _stands_out(ratio: Optional[float], rest: Optional[float], needed: float) -> float:
    """How much a specific you-vs-opponents ratio (one phase, one error type ...) exceeds the same ratio for the
    rest of your play: ``ratio / max(1, rest)``, or 0 unless that is at least ``needed``.

    Twice your opponents' error rate everywhere is one finding (you make more errors), not one per phase:
    a part of your game is only singled out when it is clearly worse (or better) than the rest.
    """
    if ratio is None:
        return 0.0
    excess = ratio / max(1.0, rest if rest is not None else 1.0)
    return excess if excess >= needed else 0.0


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def _f1(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def _urls(records: Iterable[Analysed], key: Callable[[Analysed], Any]) -> list[str]:
    """URLs of ``records`` sorted by ``key`` (most instructive first), distinct, at most MAX_EXAMPLES."""
    out: list[str] = []
    for r in sorted(records, key=key):
        if r.game.url and r.game.url not in out:
            out.append(r.game.url)
    return out[:MAX_EXAMPLES]


def _recent(r: Analysed) -> float:
    return -r.game.end_time.timestamp()


def _clock_before(g: Game, ply: int) -> Optional[float]:
    """The mover's clock before ``ply``: their previous clock, or the starting clock for their first move.

    None when the game has no clock for the move (a PGN without ``%clk``): the starting clock only counts
    for a first move whose clock was recorded.
    """
    if ply < 2:
        recorded = ply < len(g.clocks) and g.clocks[ply] is not None
        return float(g.base_seconds) if g.base_seconds and recorded else None
    return g.clocks[ply - 2] if ply - 2 < len(g.clocks) else None


def piece_of(san: str) -> str:
    """Piece letter moved by a SAN move (castling is a king move; anything else unmarked is a pawn)."""
    s = (san or "").strip()
    if s.startswith(("O-O", "0-0")):
        return "K"
    return s[0] if s and s[0] in "KQRBN" else "P"


def move_bucket(ply: int) -> str:
    move_no = ply // 2 + 1
    return next(label for label, lo, hi in MOVE_BUCKETS if move_no >= lo and (hi is None or move_no <= hi))


# --------------------------------------------------------------------------- overview
def _time_class_order(tc: str) -> tuple[int, str]:
    return (TIME_CLASSES.index(tc) if tc in TIME_CLASSES else len(TIME_CLASSES), tc)


def time_class_table(records: Sequence[Analysed]) -> tuple[Table, dict[str, Any]]:
    groups: dict[str, list[Analysed]] = defaultdict(list)
    for r in records:
        groups[r.game.time_class].append(r)
    rows, stats = [], {}
    for tc in sorted(groups, key=_time_class_order):
        rs = groups[tc]
        me, opp = tally(rs, True), tally(rs, False)
        mine_acc, opp_acc = _avg(r.ev.my_accuracy for r in rs), _avg(r.ev.opp_accuracy for r in rs)
        row = [
            tc.capitalize(), len(rs), mine_acc, opp_acc, me.acpl, opp.acpl,
            me.per100(me.blunders), opp.per100(opp.blunders),
        ]
        rows.append(row)
        stats[tc] = dict(zip(("games", "accuracy", "opp_accuracy", "acpl", "opp_acpl", "blunders_per100",
                              "opp_blunders_per100"), row[1:]))
    table = Table(
        title="By time control",
        columns=[
            "Time control", "Games", "Your accuracy", "Opponents' accuracy", "Your avg cp loss",
            "Opponents' avg cp loss", "Your blunders /100 moves", "Opponents' blunders /100 moves",
        ],
        rows=rows,
        formats=["text", "int", "float1", "float1", "int", "int", "float1", "float1"],
        note="Accuracy is Lichess-style (0-100). Average centipawn loss: how much of a pawn each move gave away "
        "on average (100 = one pawn).",
    )
    return table, stats


def blunder_insight(
    records: Sequence[Analysed], me_games: Sequence[Tally], opp_games: Sequence[Tally], th: Thresholds
) -> Optional[Insight]:
    """Overall blunder rate vs your opponents in the same games."""
    me, opp = total(me_games), total(opp_games)
    if len(records) < th.min_games or me.moves < th.min_moves or opp.moves < th.min_moves:
        return None
    mine, theirs = me.per100(me.blunders), opp.per100(opp.blunders)
    test = clustered_rate_test(me_games, opp_games, lambda t: t.blunders)
    significant, confidence = _claim(test, th)
    if not significant:
        return None
    detail = (
        f"Stockfish found {me.blunders} blunders in your {me.moves} moves over {len(records)} games "
        f"({_f1(mine)} per 100 moves); your opponents made {_f1(theirs)} per 100 moves in the same games "
        f"({opp.blunders} in {opp.moves} moves)."
    )
    evidence = {
        "games": len(records), "moves": me.moves, "blunders": me.blunders, "per100": mine,
        "opp_moves": opp.moves, "opp_blunders": opp.blunders, "opp_per100": theirs, "p_value": test.p_value,
    }
    if _clearly_higher(mine, theirs, th.blunder_ratio):
        worst = [r for r in records if r.count(lambda p: p.judgement == "blunder")]
        return Insight(
            id=f"{KEY}.weakness.blunder-rate",
            kind="weakness",
            category="blunders",
            title="You blunder more often than your opponents",
            detail=detail,
            severity=clamp((mine / theirs - 1.0) if theirs else 1.0),
            confidence=confidence,
            evidence=evidence,
            study=[
                "Before every move, do a blunder check: after my move, what are all of my opponent's checks, "
                "captures and threats?",
                "Replay the linked games at each blunder and write down in one line what you missed "
                "(a loose piece, a fork, a back-rank mate ...).",
                "Play some slower games (15+10 or longer) to build the checking habit without clock pressure.",
            ],
            example_games=_urls(
                worst, lambda r: (r.game.outcome != "loss", -r.count(lambda p: p.judgement == "blunder"), _recent(r))
            ),
        )
    if _clearly_higher(theirs, mine, th.blunder_ratio):
        return Insight(
            id=f"{KEY}.strength.blunder-rate",
            kind="strength",
            category="blunders",
            title="You blunder less often than your opponents",
            detail=detail,
            severity=clamp((theirs / mine - 1.0) if mine else 1.0),
            confidence=confidence,
            evidence=evidence,
            study=[
                "Keep your blunder-check routine: it is winning you games. Spend the saved effort on deeper plans.",
                "Look for positions where you played safe but passive moves: safe play is a base to build "
                "ambition on.",
            ],
            example_games=_urls(
                [r for r in records if r.game.outcome == "win"],
                lambda r: (r.count(lambda p: p.judgement == "blunder"), _recent(r)),
            ),
        )
    return None


def accuracy_by_result(records: Sequence[Analysed], th: Thresholds) -> tuple[Optional[Insight], dict[str, Any]]:
    """Observation: your accuracy in wins, draws and losses."""
    by: dict[str, list[float]] = {o: [] for o in ("win", "draw", "loss")}
    for r in records:
        if r.ev.my_accuracy is not None:
            by[r.game.outcome].append(r.ev.my_accuracy)
    stats = {o: {"games": len(v), "accuracy": _avg(v)} for o, v in by.items()}
    n = sum(len(v) for v in by.values())
    if n < th.min_games:
        return None, stats
    labels = {"win": "wins", "draw": "draws", "loss": "losses"}
    parts = [f"{_avg(v):.0f} in {labels[o]}" for o, v in by.items() if v]
    detail_parts = [f"{_avg(v):.1f} in {len(v)} {labels[o]}" for o, v in by.items() if v]
    opp = _avg(r.ev.opp_accuracy for r in records)
    detail = f"Your average accuracy was {', '.join(detail_parts)}."
    if opp is not None:
        detail += f" Your opponents averaged {opp:.1f} over the same games."
    wins, losses = _avg(by["win"]), _avg(by["loss"])
    losses_by_acc = [r for r in records if r.game.outcome == "loss" and r.ev.my_accuracy is not None]
    insight = Insight(
        id=f"{KEY}.observation.accuracy-by-result",
        kind="observation",
        category="accuracy",
        title=f"Your accuracy: {', '.join(parts)}",
        detail=detail,
        severity=clamp(abs(wins - losses) / 40.0) if wins is not None and losses is not None else 0.1,
        confidence=sample_confidence(n),
        evidence={"games": n, **{o: s["accuracy"] for o, s in stats.items()}, "opponents": opp},
        study=[
            "Compare your least accurate losses (linked): were they lost to one big blunder or to many small "
            "inaccuracies? The fix is different.",
            "In each linked loss, find the first move that lost 10% or more of your winning chances and name "
            "what you should have checked.",
        ],
        example_games=_urls(losses_by_acc, lambda r: (r.ev.my_accuracy, _recent(r))),
    )
    return insight, stats


# --------------------------------------------------------------------------- phases
PHASE_STUDY = {
    "opening": [
        "Replay the first 12 moves of the linked games next to an opening explorer and note where you left "
        "known theory.",
        "For each opening you play, learn its 2-3 standard plans and typical traps, not just the move order.",
        "Until move 12, check every capture and check your opponent gets after your intended move.",
    ],
    "middlegame": [
        "Do 15-20 minutes of mixed tactics puzzles a day: most middlegame errors are missed tactics.",
        "In the linked games, find your first mistake after the opening and write down the threat you overlooked.",
        "Study a few annotated master games in the pawn structures your openings lead to.",
    ],
    "endgame": [
        "Work through the essential endgames: king-and-pawn opposition, Lucena and Philidor rook endings.",
        "Play out the endgames from the linked games against an engine until you can win or hold them.",
        "In endgames, activate your king early and count tempi before pushing passed pawns.",
    ],
}


def phase_section(records: Sequence[Analysed], th: Thresholds) -> tuple[Table, Chart, list[Insight], dict[str, Any]]:
    me_games = {ph: per_game(records, True, lambda r, p, ph=ph: p.phase == ph) for ph in PHASES}
    opp_games = {ph: per_game(records, False, lambda r, p, ph=ph: p.phase == ph) for ph in PHASES}
    me = {ph: total(me_games[ph]) for ph in PHASES}
    opp = {ph: total(opp_games[ph]) for ph in PHASES}
    rows, stats = [], {}
    for ph in PHASES:
        m, o = me[ph], opp[ph]
        rows.append([ph.capitalize(), m.moves, m.acpl, m.per100(m.errors), o.moves, o.acpl, o.per100(o.errors)])
        stats[ph] = {
            "moves": m.moves, "acpl": m.acpl, "errors_per100": m.per100(m.errors),
            "opp_moves": o.moves, "opp_acpl": o.acpl, "opp_errors_per100": o.per100(o.errors),
        }
    table = Table(
        title="Game phases",
        columns=[
            "Phase", "Your moves", "Your avg cp loss", "Your mistakes+blunders /100",
            "Opponents' moves", "Opponents' avg cp loss", "Opponents' mistakes+blunders /100",
        ],
        rows=rows,
        formats=["text", "int", "int", "float1", "int", "int", "float1"],
        note="Phases follow Lichess's rules: the middlegame starts once pieces come off or the back ranks empty; "
        "the endgame once at most 6 queens, rooks, bishops and knights remain.",
    )
    chart = Chart(
        kind="bar",
        title="Mistakes and blunders per 100 moves, by phase",
        labels=[ph.capitalize() for ph in PHASES],
        series=[
            Series("You", [me[ph].per100(me[ph].errors) for ph in PHASES]),
            Series("Opponents", [opp[ph].per100(opp[ph].errors) for ph in PHASES]),
        ],
        value_format="float1",
    )
    if len(records) < th.min_games:
        return table, chart, [], stats

    judged = [ph for ph in PHASES if me[ph].moves >= th.min_phase_moves and opp[ph].moves >= th.min_phase_moves]
    tests = {ph: clustered_rate_test(me_games[ph], opp_games[ph], lambda t: t.errors) for ph in judged}
    adjusted = dict(zip(judged, bh_adjust([tests[ph].p_value for ph in judged])))
    # A phase is singled out only if it also holds a larger (smaller) share of your errors than of your
    # opponents': making more errors everywhere is the blunder-rate finding, not a phase one.
    my_errors, opp_errors = per_game(records, True), per_game(records, False)
    shares = {
        ph: clustered_share_test(
            [(t.errors, e.errors) for t, e in zip(me_games[ph], my_errors)],
            [(t.errors, e.errors) for t, e in zip(opp_games[ph], opp_errors)],
        )
        for ph in judged
    }
    share_adjusted = dict(zip(judged, bh_adjust([shares[ph].p_value for ph in judged])))
    weak, strong = [], []
    for ph in judged:
        m, o = me[ph], opp[ph]
        m_rest = total(me[q] for q in PHASES if q != ph)
        o_rest = total(opp[q] for q in PHASES if q != ph)
        significant, confidence = _claim(tests[ph], th, adjusted[ph])
        stands_out, share_confidence = _claim(shares[ph], th, share_adjusted[ph])
        stats[ph].update(p_value=tests[ph].p_value, p_adjusted=adjusted[ph], games=tests[ph].n,
                         share_p_adjusted=share_adjusted[ph])
        if not (significant and stands_out):
            continue
        confidence = min(confidence, share_confidence)
        ratio, rest = _rate_ratio(m.errors, m.moves, o.errors, o.moves), _rate_ratio(
            m_rest.errors, m_rest.moves, o_rest.errors, o_rest.moves
        )
        inverse = None if ratio is None else 1.0 / ratio
        inverse_rest = None if rest is None else 1.0 / rest
        more = m.per100(m.errors) > o.per100(o.errors) and shares[ph].mean > 0
        fewer = o.per100(o.errors) > m.per100(m.errors) and shares[ph].mean < 0
        if more and (excess := _stands_out(ratio, rest, th.phase_ratio)):
            weak.append((excess, ph, confidence))
        elif fewer and (excess := _stands_out(inverse, inverse_rest, th.phase_ratio)):
            strong.append((excess, ph, confidence))
    insights = []
    if weak:
        excess, ph, confidence = max(weak)
        insights.append(_phase_insight(records, ph, me, opp, confidence, adjusted[ph], "weakness", excess))
    if strong:
        excess, ph, confidence = max(strong)
        insights.append(_phase_insight(records, ph, me, opp, confidence, adjusted[ph], "strength", excess))
    return table, chart, insights, stats


def _phase_insight(
    records: Sequence[Analysed],
    ph: str,
    me: dict[str, Tally],
    opp: dict[str, Tally],
    confidence: float,
    p_adj: float,
    kind: str,
    excess: float,
) -> Insight:
    m, o = me[ph], opp[ph]
    m_rest, o_rest = total(me[q] for q in PHASES if q != ph), total(opp[q] for q in PHASES if q != ph)
    mine, theirs = m.per100(m.errors) or 0.0, o.per100(o.errors) or 0.0
    rest_mine, rest_theirs = m_rest.per100(m_rest.errors), o_rest.per100(o_rest.errors)
    detail = (
        f"In the {ph} you made {m.errors} mistakes and blunders in {m.moves} moves ({mine:.1f} per 100 moves); "
        f"your opponents made {theirs:.1f} per 100 moves in the same games. In the rest of the game: "
        f"{_f1(rest_mine)} for you vs {_f1(rest_theirs)} for them. Average centipawn loss in the {ph}: "
        f"{m.acpl or 0:.0f} for you vs {o.acpl or 0:.0f} for them."
    )
    evidence = {
        "phase": ph, "moves": m.moves, "errors": m.errors, "per100": mine, "acpl": m.acpl,
        "opp_moves": o.moves, "opp_errors": o.errors, "opp_per100": theirs, "opp_acpl": o.acpl, "p_adjusted": p_adj,
        "rest_per100": rest_mine, "opp_rest_per100": rest_theirs, "relative_to_rest": excess,
    }

    def errors_in_phase(r: Analysed) -> int:
        return r.count(lambda p: p.phase == ph and p.judgement in ("mistake", "blunder"))

    if kind == "weakness":
        return Insight(
            id=f"{KEY}.weakness.phase-{ph}",
            kind="weakness",
            category="phases",
            title=f"You make more mistakes than your opponents in the {ph}",
            detail=detail,
            severity=clamp(excess - 1.0),
            confidence=confidence,
            evidence=evidence,
            study=PHASE_STUDY[ph],
            example_games=_urls(
                [r for r in records if errors_in_phase(r)],
                lambda r: (-errors_in_phase(r), r.game.outcome != "loss", _recent(r)),
            ),
        )
    return Insight(
        id=f"{KEY}.strength.phase-{ph}",
        kind="strength",
        category="phases",
        title=f"You play the {ph} better than your opponents",
        detail=detail,
        severity=clamp(excess - 1.0),
        confidence=confidence,
        evidence=evidence,
        study=[
            f"Steer games toward the {ph}: it is where you outplay people at your level.",
            f"Replay the linked wins to see which {ph} skills earn you points, and keep using them.",
        ],
        example_games=_urls(
            [r for r in records if r.game.outcome == "win"], lambda r: (errors_in_phase(r), _recent(r))
        ),
    )


# --------------------------------------------------------------------------- time pressure
PRESSURE_KEYS = ("me_low", "me_ok", "opp_low", "opp_ok")
Cells = dict[tuple[str, str], Tally]  # (PRESSURE_KEYS entry, phase) -> moves


def _pressure_excess(totals: dict[tuple[str, str], tuple[int, int]], phases: Sequence[str] = PHASES) -> Optional[float]:
    """How much more time pressure raises your blunder rate than your opponents', comparing like with like, phase
    by phase: 1 = no more than theirs.

    ``totals`` maps (key, phase) to (blunders, moves). In each phase with moves in all four cells the ratio of
    ratios (your short-of-time rate / theirs) / (your rate with time left / theirs) is taken, and the phases are
    pooled on the log scale with inverse-variance weights (Woolf; half an event is added to each cell of a
    phase with an empty cell). Low-clock moves are mostly endgame moves, so an unstratified comparison would
    blame the clock for an endgame weakness. None if no phase can be compared.
    """
    weighted = total_weight = 0.0
    for ph in phases:
        cells = [totals.get((k, ph), (0, 0)) for k in PRESSURE_KEYS]
        if min(m for _, m in cells) <= 0:
            continue
        pad = 0.5 if min(x for x, _ in cells) == 0 else 0.0
        (a, na), (c, nc), (b, nb), (d, nd) = ((x + pad, m) for x, m in cells)
        log_ratio = math.log(a / na) - math.log(b / nb) - math.log(c / nc) + math.log(d / nd)
        weight = 1.0 / (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
        weighted += weight * log_ratio
        total_weight += weight
    return math.exp(weighted / total_weight) if total_weight > 0 else None


def pressure_excess_test(per_game: Sequence[Cells]) -> tuple[Optional[float], MeanTest]:
    """(:func:`_pressure_excess`, one-sided test that it exceeds 1) with GAMES as the unit.

    The standard error of log(excess) is the delete-one-game jackknife, so it carries the noise of every
    ingredient (your short-of-time blunders, your opponents' rates, your ratio to them with time left) and
    the clustering of errors within games. ``n`` is the number of games.
    """
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for cells in per_game:
        for key, t in cells.items():
            totals[key][0] += t.blunders
            totals[key][1] += t.moves
    n = len(per_game)
    full = _pressure_excess({k: (x, m) for k, (x, m) in totals.items()})
    if full is None:
        return None, MeanTest(n, 0.0, float("inf"), 0.0, 1.0)
    phases = [ph for ph in PHASES if min(totals.get((k, ph), [0, 0])[1] for k in PRESSURE_KEYS) > 0]
    replicates = []
    for cells in per_game:
        loo = {
            k: (x - (cells[k].blunders if k in cells else 0), m - (cells[k].moves if k in cells else 0))
            for k, (x, m) in totals.items()
        }
        value = _pressure_excess(loo, phases)
        if value is not None:
            replicates.append(math.log(value))
    estimate = math.log(full)
    if len(replicates) < 2:
        return full, MeanTest(n, estimate, float("inf"), 0.0, 1.0)
    centre = sum(replicates) / len(replicates)
    k = len(replicates)
    se = math.sqrt((k - 1) / k * sum((x - centre) ** 2 for x in replicates))
    if se <= 0.0:
        return full, MeanTest(n, estimate, float("inf"), 0.0, 1.0)
    z = estimate / se
    return full, MeanTest(n, estimate, se, z, clamp(1.0 - normal_cdf(z)))


def pressure_section(
    records: Sequence[Analysed], th: Thresholds
) -> tuple[Optional[Table], list[Insight], dict[str, Any]]:
    """Your blunder rate on moves made while short of time vs other moves, per time class (live games with clocks)."""
    keys = PRESSURE_KEYS
    by_game: dict[str, list[dict[str, Tally]]] = defaultdict(list)  # time class -> one {key: Tally} per game
    by_phase: dict[str, list[Cells]] = defaultdict(list)  # the same moves split by phase, one Cells per game
    low_games: dict[str, list[Analysed]] = defaultdict(list)
    for r in records:
        g = r.game
        if g.time_class == "daily" or not g.base_seconds or not any(c is not None for c in g.clocks or ()):
            continue
        limit = th.pressure_fraction * float(g.base_seconds)
        counts = {k: Tally() for k in keys}
        cells: Cells = defaultdict(Tally)
        low_blunder = False
        for p in r.plies:
            before = _clock_before(g, p.ply)
            if before is None:
                continue
            low = before < limit
            key = ("me" if p.is_user else "opp") + ("_low" if low else "_ok")
            counts[key].add(p)
            cells[(key, p.phase)].add(p)
            low_blunder |= low and p.is_user and p.judgement == "blunder"
        if any(t.moves for t in counts.values()):
            by_game[g.time_class].append(counts)
            by_phase[g.time_class].append(cells)
        if low_blunder:
            low_games[g.time_class].append(r)
    if not by_game:
        return None, [], {}
    groups = {tc: {k: total(c[k] for c in games) for k in keys} for tc, games in by_game.items()}

    def column(tc: str, key: str) -> list[Tally]:
        return [c[key] for c in by_game[tc]]

    order = sorted(groups, key=_time_class_order)
    rows, stats = [], {}
    for tc in order:
        t = groups[tc]
        row = [
            tc.capitalize(), t["me_low"].moves, t["me_low"].per100(t["me_low"].blunders),
            t["me_ok"].per100(t["me_ok"].blunders), t["opp_low"].moves,
            t["opp_low"].per100(t["opp_low"].blunders), t["opp_ok"].per100(t["opp_ok"].blunders),
        ]
        rows.append(row)
        stats[tc] = dict(zip(("moves_low", "blunders_per100_low", "blunders_per100_ok", "opp_moves_low",
                              "opp_blunders_per100_low", "opp_blunders_per100_ok"), row[1:]))
    share = pct(th.pressure_fraction)
    table = Table(
        title="Blunders when short of time",
        columns=[
            "Time control", "Your moves short of time", "Your blunders /100 (short of time)",
            "Your blunders /100 (otherwise)", "Opponents' moves short of time",
            "Opponents' blunders /100 (short of time)", "Opponents' blunders /100 (otherwise)",
        ],
        rows=rows,
        formats=["text", "int", "float1", "float1", "int", "float1", "float1"],
        note=f"Short of time: less than {share} of the starting clock left before the move. Live games with clock "
        "data only.",
    )
    if len(records) < th.min_games:
        return table, [], stats
    judged = [tc for tc in order if groups[tc]["me_low"].moves >= th.min_pressure_moves and groups[tc]["me_ok"].moves]
    # Everyone blunders more when short of time, so the claim needs two things: your own rate jumps (short of
    # time vs otherwise), and it jumps further than your opponents' does (your short-of-time rate vs theirs).
    def blunders(t: Tally) -> int:
        return t.blunders

    own = {tc: clustered_rate_test(column(tc, "me_low"), column(tc, "me_ok"), blunders) for tc in judged}
    benched = [tc for tc in judged if groups[tc]["opp_low"].moves >= th.min_pressure_moves]
    bench = {tc: clustered_rate_test(column(tc, "me_low"), column(tc, "opp_low"), blunders) for tc in benched}
    # ... and that jump must be larger than your opponents' in the same phases of the game (see _pressure_excess)
    excess_tests = {tc: pressure_excess_test(by_phase[tc]) for tc in benched}
    own_adj = dict(zip(judged, bh_adjust([own[tc].p_value for tc in judged])))
    bench_adj = dict(zip(benched, bh_adjust([bench[tc].p_value for tc in benched])))
    excess_adj = dict(zip(benched, bh_adjust([excess_tests[tc][1].p_value for tc in benched])))
    insights = []
    for tc in judged:
        t = groups[tc]
        low, ok = t["me_low"].per100(t["me_low"].blunders), t["me_ok"].per100(t["me_ok"].blunders)
        opp_low, opp_ok = t["opp_low"].per100(t["opp_low"].blunders), t["opp_ok"].per100(t["opp_ok"].blunders)
        significant, confidence = _claim(own[tc], th, own_adj[tc])
        stats[tc].update(p_value=own[tc].p_value, p_adjusted=own_adj[tc], games=own[tc].n)
        if not significant:
            continue
        if not _clearly_higher(low, ok, th.pressure_ratio):
            continue
        worse_than_peers, excess = False, 0.0
        if tc in bench:
            bench_significant, bench_conf = _claim(bench[tc], th, bench_adj[tc])
            stats[tc].update(benchmark_p_value=bench[tc].p_value, benchmark_p_adjusted=bench_adj[tc])
            # Short of time you must be worse off against your opponents than you are with time on the clock, in
            # the same phase: blundering twice as often as them in every situation is the blunder-rate finding,
            # and twice as often in the endgame (where most short-of-time moves are) is the endgame one.
            ratio, excess_test = excess_tests[tc]
            excess_significant, excess_conf = _claim(excess_test, th, excess_adj[tc])
            stats[tc].update(phase_adjusted_excess=ratio, excess_p_adjusted=excess_adj[tc])
            excess = ratio if ratio is not None and ratio >= th.pressure_peer_ratio else 0.0
            worse_than_peers = bench_significant and excess_significant and (low or 0) > (opp_low or 0) and excess > 0
            confidence = min(confidence, bench_conf, excess_conf)
        detail = (
            f"In {tc}, with less than {share} of your starting clock left you blundered {_f1(low)} times per 100 "
            f"moves ({t['me_low'].blunders} in {t['me_low'].moves} moves), against {_f1(ok)} per 100 moves with "
            "more time."
        )
        if t["opp_low"].moves:
            detail += (
                f" Your opponents in the same games: {_f1(opp_low)} per 100 moves when short of time "
                f"({t['opp_low'].moves} moves), {_f1(opp_ok)} otherwise."
            )
        else:
            detail += " Your opponents were never that short of time in these games."
        evidence = {
            "time_class": tc, "moves_low": t["me_low"].moves, "blunders_low": t["me_low"].blunders,
            "per100_low": low, "per100_ok": ok, "opp_moves_low": t["opp_low"].moves, "opp_per100_low": opp_low,
            "opp_per100_ok": opp_ok, "p_adjusted": own_adj[tc],
            "phase_adjusted_excess": excess_tests[tc][0] if tc in excess_tests else None,
        }
        examples = _urls(low_games[tc], lambda r: (r.game.outcome != "loss", _recent(r)))
        if worse_than_peers:
            insights.append(
                Insight(
                    id=f"{KEY}.weakness.time-pressure-blunders-{tc}",
                    kind="weakness",
                    category="time",
                    title=f"You blunder much more than your opponents when short of time in {tc}",
                    detail=detail,
                    severity=clamp(excess - 1.0),
                    confidence=confidence,
                    evidence=evidence | {"benchmark_p_adjusted": bench_adj[tc]},
                    study=[
                        f"Budget your clock in {tc}: reach move 30 with at least a third of your starting time.",
                        "When short of time, play safe moves: keep pieces protected and your king covered; don't "
                        "start complications.",
                        "Practise with increment (e.g. 3+2 or 10+5) so the last phase of the game stops being a "
                        "scramble.",
                    ],
                    example_games=examples,
                )
            )
        else:
            insights.append(
                Insight(
                    id=f"{KEY}.observation.time-pressure-blunders-{tc}",
                    kind="observation",
                    category="time",
                    title=f"You blunder much more when short of time in {tc}",
                    detail=detail + " Most players do; the fix is to reach the end of the game with time left.",
                    severity=clamp((low / ok - 1.0) / 3.0 if ok else 1.0),
                    confidence=confidence,
                    evidence=evidence,
                    study=[
                        f"Budget your clock in {tc}: reach move 30 with at least a third of your starting time.",
                        "When short of time, play safe moves: keep pieces protected and your king covered.",
                    ],
                    example_games=examples,
                )
            )
    return table, insights, stats


# --------------------------------------------------------------------------- conversion
@dataclass
class Conversion:
    """Games split by who reached a winning position FIRST (each game counts once, so the groups are independent)."""

    reached: list[Analysed] = field(default_factory=list)  # you got a winning position first
    opp_reached: list[Analysed] = field(default_factory=list)  # your opponent did

    @property
    def converted(self) -> int:
        return sum(1 for r in self.reached if r.game.outcome == "win")

    @property
    def opp_converted(self) -> int:
        return sum(1 for r in self.opp_reached if r.game.outcome == "loss")

    @property
    def saved(self) -> int:
        """Games you drew or won after being lost."""
        return len(self.opp_reached) - self.opp_converted

    @property
    def rate(self) -> Optional[float]:
        return self.converted / len(self.reached) if self.reached else None

    @property
    def opp_rate(self) -> Optional[float]:
        return self.opp_converted / len(self.opp_reached) if self.opp_reached else None


def conversion_section(records: Sequence[Analysed], th: Thresholds) -> tuple[Table, list[Insight], dict[str, Any]]:
    conv = Conversion()
    peak: dict[str, float] = {}
    for r in records:
        wins = r.pov_win()
        peak[r.game.game_id] = max(wins)
        first = next((w for w in wins if w >= th.win_threshold or w <= th.loss_threshold), None)
        if first is not None:
            (conv.reached if first >= th.win_threshold else conv.opp_reached).append(r)

    def wdl(rs: list[Analysed]) -> list[int]:
        c = Counter(r.game.outcome for r in rs)
        return [len(rs), c["win"], c["draw"], c["loss"]]

    rows = [
        [f"You got a winning position first (≥ {th.win_threshold:.0f}%)", *wdl(conv.reached), conv.rate],
        [f"Your opponent got one first (you ≤ {th.loss_threshold:.0f}%)", *wdl(conv.opp_reached),
         conv.saved / len(conv.opp_reached) if conv.opp_reached else None],
    ]
    table = Table(
        title="Winning and losing positions",
        columns=["Situation", "Games", "Won", "Drawn", "Lost", "Converted / saved"],
        rows=rows,
        formats=["text", "int", "int", "int", "int", "pct"],
        note=f"Winning position: Stockfish gave that side at least a {th.win_threshold:.0f}% chance to win. Each "
        "game counts once, for the side that got there first. Converted = you won those games; saved = you drew "
        "or won anyway.",
    )
    stats = {
        "reached": len(conv.reached), "converted": conv.converted, "rate": conv.rate,
        "opp_reached": len(conv.opp_reached), "opp_converted": conv.opp_converted, "opp_rate": conv.opp_rate,
        "saved": conv.saved,
    }
    insights: list[Insight] = []
    enough = (
        len(records) >= th.min_games
        and len(conv.reached) >= th.min_conversion_games
        and len(conv.opp_reached) >= th.min_conversion_games
    )
    if not enough:
        return table, insights, stats

    test = two_proportion_test(conv.converted, len(conv.reached), conv.opp_converted, len(conv.opp_reached))
    significant, confidence = _claim(test, th, min_n=2 * th.min_conversion_games)  # n: games, each counted once
    stats.update(p_value=test.p_value)
    rate, opp_rate = conv.rate or 0.0, conv.opp_rate or 0.0
    base = (
        f"You were first to reach a winning position (Stockfish: at least {th.win_threshold:.0f}% winning "
        f"chances) in {len(conv.reached)} games and won {conv.converted} of them ({pct(rate)}). When your "
        f"opponents got there first, they won {conv.opp_converted} of {len(conv.opp_reached)} ({pct(opp_rate)})."
    )
    evidence = {k: stats[k] for k in ("reached", "converted", "rate", "opp_reached", "opp_converted", "opp_rate")}
    if significant and rate <= opp_rate - th.min_conversion_gap:
        thrown = [r for r in conv.reached if r.game.outcome != "win"]
        insights.append(
            Insight(
                id=f"{KEY}.weakness.conversion",
                kind="weakness",
                category="conversion",
                title="You let winning positions slip more often than your opponents do",
                detail=base,
                severity=clamp((opp_rate - rate) / 0.30),
                confidence=confidence,
                evidence=evidence | {"p_value": test.p_value},
                study=[
                    "Replay the linked games from the moment you were winning and find the move where the advantage "
                    "started to slip.",
                    "When winning: trade pieces (not pawns), remove your opponent's counterplay first, and don't rush.",
                    "Practise converting: set up a winning position from one of your games and play it out against "
                    "an engine.",
                ],
                example_games=_urls(
                    thrown, lambda r: (r.game.outcome != "loss", -peak[r.game.game_id], _recent(r))
                ),
            )
        )
    elif significant and rate >= opp_rate + th.min_conversion_gap:
        insights.append(
            Insight(
                id=f"{KEY}.strength.conversion",
                kind="strength",
                category="conversion",
                title="You convert winning positions better than your opponents",
                detail=base,
                severity=clamp((rate - opp_rate) / 0.30),
                confidence=confidence,
                evidence=evidence | {"p_value": test.p_value},
                study=[
                    "Your technique in won positions is an asset: aim for positions with a small, safe edge "
                    "and grind.",
                    "Replay the linked wins to note the simplifying decisions that worked, so you repeat them.",
                ],
                example_games=_urls([r for r in conv.reached if r.game.outcome == "win"], _recent),
            )
        )
    saved = [r for r in conv.opp_reached if r.game.outcome != "loss"]
    save_rate = conv.saved / len(conv.opp_reached)
    if saved:
        review = "Replay the linked comebacks to see which defensive ideas worked for you."
        examples = _urls(saved, lambda r: (r.game.outcome != "win", _recent(r)))
    else:
        review = (
            "Replay the linked losses from the moment you were lost and look for moves that would have set "
            "your opponent harder problems."
        )
        examples = _urls(conv.opp_reached, _recent)
    insights.append(
        Insight(
            id=f"{KEY}.observation.resilience",
            kind="observation",
            category="conversion",
            title=f"You saved {conv.saved} of {len(conv.opp_reached)} lost positions ({pct(save_rate)})",
            detail=(
                f"In {len(conv.opp_reached)} games your opponent got a winning position first (you had at most "
                f"{th.loss_threshold:.0f}% winning chances); you still drew or won {conv.saved} of them. Your "
                f"opponents saved {len(conv.reached) - conv.converted} of the {len(conv.reached)} games where you "
                f"got there first ({pct(1.0 - rate)})."
            ),
            severity=clamp(abs(save_rate - (1.0 - rate)) / 0.30),
            confidence=sample_confidence(len(conv.opp_reached)),
            evidence={"lost_positions": len(conv.opp_reached), "saved": conv.saved, "save_rate": save_rate},
            study=[
                "In a lost position, set practical problems: keep pieces on, create threats and make every "
                "move matter for your opponent.",
                review,
            ],
            example_games=examples,
        )
    )
    return table, insights, stats


# --------------------------------------------------------------------------- tactics
def _tag_shares(records: Sequence[Analysed], mine: bool, tag: str) -> list[tuple[int, int]]:
    """Per game: (mistakes and blunders carrying ``tag``, all mistakes and blunders) by you or your opponents."""
    out = []
    for r in records:
        errors = [p for p in r.plies if p.is_user == mine and p.judgement in ("mistake", "blunder")]
        out.append((sum(1 for p in errors if tag in (p.tags or ())), len(errors)))
    return out


def _errors_without(records: Sequence[Analysed], mine: bool, tag: str) -> int:
    """Mistakes and blunders by you (``mine``) or your opponents that don't carry ``tag``."""
    return sum(
        1
        for r in records
        for p in r.plies
        if p.is_user == mine and p.judgement in ("mistake", "blunder") and tag not in (p.tags or ())
    )


def _costliest(records: Sequence[Analysed], tag: str) -> list[str]:
    """Games with your costliest move carrying ``tag`` (largest win-% drop) first."""

    def drop(r: Analysed) -> float:
        return max((p.win_before - p.win_after for p in r.plies if p.is_user and tag in (p.tags or ())), default=0.0)

    return _urls([r for r in records if drop(r) > 0], lambda r: (-drop(r), _recent(r)))


TACTIC_TEXT = {
    "missed_tactic": dict(
        attr="missed_tactics",
        slug="missed-tactics",
        weak_title="You miss more tactical shots than your opponents",
        strong_title="You miss fewer tactical shots than your opponents",
        what="missed a winning capture, check or promotion (costing at least 15% of your winning chances)",
        study=[
            "Solve 20 minutes of puzzles a day, focused on forks, pins and discovered attacks.",
            "Before each move, list every check, capture and threat for both sides.",
            "Replay the linked games at the moment of the miss and find the shot yourself before looking at "
            "the engine's move.",
        ],
    ),
    "hung_material": dict(
        attr="hung",
        slug="hung-material",
        weak_title="You leave material hanging more often than your opponents",
        strong_title="You leave material hanging less often than your opponents",
        what="made a mistake or blunder that let your opponent win material with a capture",
        study=[
            "Before letting go of a piece, ask: after this move, what can my opponent capture, and is it defended?",
            "Drill 'hanging piece' and 'loose piece' puzzle themes until you spot undefended pieces instantly.",
            "Replay the linked games at the moment you lost material and name the capture you allowed.",
        ],
    ),
}


def tactics_section(
    records: Sequence[Analysed], me_games: Sequence[Tally], opp_games: Sequence[Tally], th: Thresholds
) -> tuple[Table, list[Insight], dict[str, Any]]:
    me, opp = total(me_games), total(opp_games)
    rows = [
        ["Missed a tactical shot", me.missed_tactics, me.per100(me.missed_tactics),
         opp.missed_tactics, opp.per100(opp.missed_tactics)],
        ["Left material hanging", me.hung, me.per100(me.hung), opp.hung, opp.per100(opp.hung)],
        ["Missed a forced mate", me.missed_mates, me.per100(me.missed_mates),
         opp.missed_mates, opp.per100(opp.missed_mates)],
        ["Allowed a forced mate", me.allowed_mates, me.per100(me.allowed_mates),
         opp.allowed_mates, opp.per100(opp.allowed_mates)],
    ]
    table = Table(
        title="Tactics",
        columns=["Pattern", "You", "You /100 moves", "Opponents", "Opponents /100 moves"],
        rows=rows,
        formats=["text", "int", "float2", "int", "float2"],
        note="Missed tactical shot: the engine's best move was a capture, check or promotion, you played something "
        "else and lost at least 15% of your winning chances. Left material hanging: a mistake or blunder the "
        "opponent could punish with a capture that wins material (recapturing in a trade doesn't count).",
    )
    stats: dict[str, Any] = {
        "missed_tactics": me.missed_tactics, "opp_missed_tactics": opp.missed_tactics,
        "hung": me.hung, "opp_hung": opp.hung,
        "missed_mates": me.missed_mates, "opp_missed_mates": opp.missed_mates,
        "allowed_mates": me.allowed_mates, "opp_allowed_mates": opp.allowed_mates,
    }
    insights: list[Insight] = []
    if len(records) >= th.min_games and me.moves >= th.min_moves and opp.moves >= th.min_moves:
        tags = list(TACTIC_TEXT)
        tests = {
            tag: clustered_rate_test(me_games, opp_games, lambda t, attr=TACTIC_TEXT[tag]["attr"]: getattr(t, attr))
            for tag in tags
        }
        adjusted = dict(zip(tags, bh_adjust([tests[t].p_value for t in tags])))
        # ... and the error type must make up a larger (smaller) share of your errors than of your opponents'.
        shares = {tag: clustered_share_test(_tag_shares(records, True, tag), _tag_shares(records, False, tag))
                  for tag in tags}
        share_adjusted = dict(zip(tags, bh_adjust([shares[t].p_value for t in tags])))
        for tag in tags:
            text = TACTIC_TEXT[tag]
            mine_n, theirs_n = getattr(me, text["attr"]), getattr(opp, text["attr"])
            mine, theirs = me.per100(mine_n), opp.per100(theirs_n)
            significant, confidence = _claim(tests[tag], th, adjusted[tag])
            stands_out, share_confidence = _claim(shares[tag], th, share_adjusted[tag])
            stats[f"{tag}_p_adjusted"] = adjusted[tag]
            stats[f"{tag}_share_p_adjusted"] = share_adjusted[tag]
            if not (significant and stands_out):
                continue
            confidence = min(confidence, share_confidence)
            # Compared with your other mistakes and blunders: more errors of every kind is the blunder-rate finding.
            other, opp_other = _errors_without(records, True, tag), _errors_without(records, False, tag)
            ratio = _rate_ratio(mine_n, me.moves, theirs_n, opp.moves)
            rest = _rate_ratio(other, me.moves, opp_other, opp.moves)
            more = shares[tag].mean > 0 and (mine or 0) > (theirs or 0)
            fewer = shares[tag].mean < 0 and (theirs or 0) > (mine or 0)
            weak_excess = _stands_out(ratio, rest, th.tactic_ratio) if more else 0.0
            strong_excess = _stands_out(1.0 / ratio, 1.0 / rest, th.tactic_ratio) if fewer and ratio and rest else 0.0
            detail = (
                f"In {len(records)} analysed games you {text['what']} {mine_n} times ({_f1(mine)} per 100 moves); "
                f"your opponents did so {_f1(theirs)} times per 100 moves in the same games. Your other mistakes "
                f"and blunders: {_f1(me.per100(other))} per 100 moves vs {_f1(opp.per100(opp_other))} for them."
            )
            evidence = {"count": mine_n, "per100": mine, "opp_count": theirs_n, "opp_per100": theirs,
                        "p_adjusted": adjusted[tag], "other_errors_per100": me.per100(other),
                        "opp_other_errors_per100": opp.per100(opp_other)}
            if weak_excess:
                insights.append(
                    Insight(
                        id=f"{KEY}.weakness.{text['slug']}",
                        kind="weakness",
                        category="tactics",
                        title=text["weak_title"],
                        detail=detail,
                        severity=clamp(weak_excess - 1.0),
                        confidence=confidence,
                        evidence=evidence,
                        study=text["study"],
                        example_games=_costliest(records, tag),
                    )
                )
            elif strong_excess:
                insights.append(
                    Insight(
                        id=f"{KEY}.strength.{text['slug']}",
                        kind="strength",
                        category="tactics",
                        title=text["strong_title"],
                        detail=detail,
                        severity=clamp(strong_excess - 1.0),
                        confidence=confidence,
                        evidence=evidence,
                        study=[
                            "Your tactical alertness is an asset: choose sharp, open positions where it counts most.",
                            "Keep a daily puzzle habit so the edge doesn't fade.",
                        ],
                        example_games=_urls([r for r in records if r.game.outcome == "win"], _recent),
                    )
                )
    if me.missed_mates >= th.min_missed_mates:
        insights.append(
            Insight(
                id=f"{KEY}.observation.missed-mates",
                kind="observation",
                category="tactics",
                title=f"You missed {me.missed_mates} forced checkmates",
                detail=(
                    f"Stockfish saw a forced mate for you {me.missed_mates} times that you didn't follow through "
                    f"(your opponents: {opp.missed_mates}); you also allowed {me.allowed_mates} forced mates "
                    f"against you (your opponents: {opp.allowed_mates})."
                ),
                severity=clamp(me.missed_mates / max(1, len(records)) * 5.0),
                confidence=sample_confidence(me.missed_mates, half_at=5.0),
                evidence={"missed_mates": me.missed_mates, "opp_missed_mates": opp.missed_mates,
                          "allowed_mates": me.allowed_mates, "opp_allowed_mates": opp.allowed_mates},
                study=[
                    "Solve mate-in-2 and mate-in-3 puzzles until the common patterns (back rank, smothered, "
                    "Anastasia's) are automatic.",
                    "When your opponent's king is exposed, look at every check first, even ones that sacrifice "
                    "material.",
                ],
                example_games=_costliest(records, "missed_mate"),
            )
        )
    return table, insights, stats


# --------------------------------------------------------------------------- blunder anatomy
def anatomy_section(
    records: Sequence[Analysed], th: Thresholds
) -> tuple[Table, Table, Chart, list[Insight], dict[str, Any]]:
    moves: Counter = Counter()
    blunders: Counter = Counter()
    for r in records:
        for p in r.mine():
            piece = piece_of(p.san)
            moves[piece] += 1
            blunders[piece] += int(p.judgement == "blunder")
    total_moves, total_blunders = sum(moves.values()), sum(blunders.values())
    piece_rows = [
        [name, moves[k], blunders[k], 100.0 * blunders[k] / moves[k] if moves[k] else None,
         blunders[k] / total_blunders if total_blunders else None]
        for k, name in PIECES
    ]
    piece_table = Table(
        title="Your blunders by piece moved",
        columns=["Piece moved", "Your moves", "Your blunders", "Blunders /100 moves", "Share of your blunders"],
        rows=piece_rows,
        formats=["text", "int", "int", "float1", "pct"],
        note="Castling counts as a king move.",
    )
    labels = [label for label, _, _ in MOVE_BUCKETS]
    me_b = {label: tally(records, True, lambda r, p, label=label: move_bucket(p.ply) == label) for label in labels}
    opp_b = {label: tally(records, False, lambda r, p, label=label: move_bucket(p.ply) == label) for label in labels}
    bucket_rows = [
        [label, me_b[label].moves, me_b[label].per100(me_b[label].blunders),
         opp_b[label].moves, opp_b[label].per100(opp_b[label].blunders)]
        for label in labels
    ]
    bucket_table = Table(
        title="Blunders by move number",
        columns=["Move numbers", "Your moves", "Your blunders /100", "Opponents' moves", "Opponents' blunders /100"],
        rows=bucket_rows,
        formats=["text", "int", "float1", "int", "float1"],
    )
    chart = Chart(
        kind="bar",
        title="Blunders per 100 moves, by move number",
        labels=labels,
        series=[Series("You", [row[2] for row in bucket_rows]), Series("Opponents", [row[4] for row in bucket_rows])],
        value_format="float1",
    )
    stats = {
        "by_piece": {name: {"moves": moves[k], "blunders": blunders[k]} for k, name in PIECES},
        "by_move": {
            label: {
                "moves": me_b[label].moves, "blunders": me_b[label].blunders,
                "opp_moves": opp_b[label].moves, "opp_blunders": opp_b[label].blunders,
            }
            for label in labels
        },
    }
    if total_blunders < th.min_anatomy_blunders or not total_moves:
        return piece_table, bucket_table, chart, [], stats

    def lift(k: str) -> float:
        return (blunders[k] / total_blunders) / (moves[k] / total_moves) if moves[k] else 0.0

    piece = max((k for k, _ in PIECES if blunders[k] >= 3), key=lambda k: (lift(k), blunders[k]), default=None)
    worst_bucket = max(
        (label for label in labels if me_b[label].moves >= 30),
        key=lambda label: me_b[label].per100(me_b[label].blunders) or 0.0,
        default=None,
    )
    if piece is None:
        return piece_table, bucket_table, chart, [], stats
    name = dict(PIECES)[piece].lower()
    detail = (
        f"{name.capitalize()} moves were {pct(moves[piece] / total_moves)} of your moves but "
        f"{pct(blunders[piece] / total_blunders)} of your {total_blunders} blunders ({blunders[piece]})."
    )
    study = [
        f"Before every {name} move, check which squares and pieces it stops guarding and whether it can be "
        "attacked or trapped on its new square.",
        "Keep a blunder log: for each blunder note the piece, the move number and what you missed; patterns repeat.",
    ]
    if worst_bucket is not None:
        b = me_b[worst_bucket]
        detail += f" Your blunder rate peaks at moves {worst_bucket} ({_f1(b.per100(b.blunders))} per 100 moves)."
        study.append(f"Around moves {worst_bucket}, slow down and double-check every capture and check for both sides.")
    blundered = [r for r in records if r.count(lambda p: p.judgement == "blunder" and piece_of(p.san) == piece)]
    insight = Insight(
        id=f"{KEY}.observation.blunder-anatomy",
        kind="observation",
        category="blunders",
        title=f"{name.capitalize()} moves account for {pct(blunders[piece] / total_blunders)} of your blunders",
        detail=detail,
        severity=clamp((lift(piece) - 1.0) / 2.0),
        confidence=sample_confidence(total_blunders),
        evidence={"piece": name, "blunders": blunders[piece], "moves": moves[piece], "total_blunders": total_blunders,
                  "total_moves": total_moves, "worst_moves": worst_bucket},
        study=study,
        example_games=_urls(blundered, _recent),
    )
    return piece_table, bucket_table, chart, [insight], stats


# --------------------------------------------------------------------------- opening outcome
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                          -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-12:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_test(test: MeanTest) -> MeanTest:
    """``test`` (a one-sample :func:`stats.mean_test`) with its two-sided p-value from Student's t (n - 1 degrees
    of freedom) instead of the normal distribution, which is too optimistic for a handful of games."""
    if test.n < 2 or not math.isfinite(test.se):
        return dataclasses.replace(test, p_value=1.0)
    df = test.n - 1
    return dataclasses.replace(test, p_value=clamp(_betainc(df / 2.0, 0.5, df / (df + test.z * test.z))))


def opening_eval(r: Analysed) -> Optional[float]:
    """Eval (cp, your side) gained from the start position to after ply OPENING_PLY; None if the game ended earlier.

    Measured from Stockfish's own evaluation of the start position, so White's normal first-move edge (about
    +0.3) counts as neither a good opening for White nor a bad one for Black. Capped at +-OPENING_CP_CAP.
    """
    p = next((p for p in r.plies if p.ply == OPENING_PLY - 1), None)
    if p is None:
        return None
    first = next((q for q in r.plies if q.ply == 0), None)
    gained = p.cp_after - (first.cp_before if first is not None else 0)
    cp = gained if r.game.color == "white" else -gained
    return float(max(-OPENING_CP_CAP, min(OPENING_CP_CAP, cp)))


def opening_section(
    records: Sequence[Analysed], th: Thresholds
) -> tuple[Optional[Table], list[Insight], dict[str, Any]]:
    groups: dict[tuple[str, str], list[tuple[Analysed, float]]] = defaultdict(list)
    for r in records:
        g = r.game
        if g.initial_fen or g.rules != "chess" or not g.opening_family:
            continue
        cp = opening_eval(r)
        if cp is not None:
            groups[(g.opening_family, g.color)].append((r, cp))
    shown = sorted(
        (k for k, v in groups.items() if len(v) >= th.min_opening_games), key=lambda k: (-len(groups[k]), k)
    )
    if not shown:
        return None, [], {}
    rows, stats = [], {}
    tests: dict[tuple[str, str], MeanTest] = {}
    for key in shown:
        family, colour = key
        items = groups[key]
        evals = [cp for _, cp in items]
        tests[key] = student_t_test(mean_test(evals, min_sd=OPENING_EVAL_SD))  # as few as 8 games: t, not z
        avg = sum(evals) / len(evals)
        worse = sum(1 for cp in evals if cp <= -100)
        score = sum(r.game.score for r, _ in items) / len(items)
        rows.append([family, colour.capitalize(), len(items), avg, worse, score])
        stats[f"{family} ({colour})"] = {"games": len(items), "avg_cp": avg, "worse_by_a_pawn": worse, "score": score}
    table = Table(
        title="Where your openings leave you (engine eval after move 10)",
        columns=["Opening", "Colour", "Games", "Avg eval after move 10", "Games a pawn or more worse", "Score"],
        rows=rows,
        formats=["text", "text", "int", "signed_int", "int", "pct"],
        note="Stockfish's evaluation from your side after both players' 10th move, in centipawns (100 = one pawn), "
        "compared with its evaluation of the start position, so White's usual small edge counts as 0; capped at "
        "±5 pawns per game. Games from set-up positions are left out.",
    )
    insights: list[Insight] = []
    if len(records) < th.min_games:
        return table, insights, stats
    judged = [k for k in shown if len(groups[k]) >= th.min_opening_insight_games]
    adjusted = dict(zip(judged, bh_adjust([tests[k].p_value for k in judged])))
    for key in judged:
        family, colour = key
        significant, confidence = _claim(tests[key], th, adjusted[key], min_n=th.min_opening_insight_games)
        avg = tests[key].mean
        if not significant or abs(avg) < th.opening_eval_threshold:
            continue
        items = groups[key]
        n = len(items)
        worse = sum(1 for _, cp in items if cp <= -100)
        better = sum(1 for _, cp in items if cp >= 100)
        slug = f"openings-{colour}-{slugify(family)}"
        side = colour.capitalize()
        evidence = {"family": family, "colour": colour, "games": n, "avg_cp": avg, "p_adjusted": adjusted[key]}
        pawns = f"{avg / 100:+.1f}".replace("-", "−")
        if avg < 0:
            insights.append(
                Insight(
                    id=f"{KEY}.weakness.{slug}",
                    kind="weakness",
                    category="openings",
                    title=f"You come out of the {family} worse off as {side}",
                    detail=(
                        f"In {n} analysed games as {side} in the {family}, Stockfish rated your position after "
                        f"move 10 at {fmt_signed(avg)} centipawns on average compared with the start (about {pawns} "
                        f"pawns); {worse} of them were already a pawn or more worse."
                    ),
                    severity=clamp(abs(avg) / 150.0),
                    confidence=confidence,
                    evidence=evidence,
                    study=[
                        "Replay the first 10 moves of the linked games next to an opening explorer and find the "
                        "move where your position started to slip.",
                        f"Pick one main line of the {family} and learn its plans from 2-3 annotated master games.",
                        "Check the engine's first inaccuracy in each linked game: it is often the same move or idea.",
                    ],
                    example_games=_urls([r for r, _ in items], lambda r: (opening_eval(r), _recent(r))),
                )
            )
        else:
            insights.append(
                Insight(
                    id=f"{KEY}.strength.{slug}",
                    kind="strength",
                    category="openings",
                    title=f"The {family} gives you good positions as {side}",
                    detail=(
                        f"In {n} analysed games as {side} in the {family}, Stockfish rated your position after "
                        f"move 10 at {fmt_signed(avg)} centipawns on average compared with the start (about {pawns} "
                        f"pawns); {better} of them were already a pawn or more better."
                    ),
                    severity=clamp(abs(avg) / 150.0),
                    confidence=confidence,
                    evidence=evidence,
                    study=[
                        f"Keep the {family} in your repertoire and deepen it with one new sideline a month.",
                        "Review the linked games from move 10 onward: are you turning the early edge into wins?",
                    ],
                    example_games=_urls([r for r, _ in items], lambda r: (-(opening_eval(r) or 0.0), _recent(r))),
                )
            )
    return table, insights, stats


# --------------------------------------------------------------------------- accuracy trend
def accuracy_trend(records: Sequence[Analysed], th: Thresholds) -> tuple[Optional[Chart], dict[str, Any]]:
    months: dict[str, list[Analysed]] = defaultdict(list)
    for r in records:
        months[r.game.end_time.strftime("%Y-%m")].append(r)
    labels = sorted(months)
    mine, theirs, stats = [], [], {}
    for m in labels:
        rs = [r for r in months[m] if r.ev.my_accuracy is not None]
        ok = len(rs) >= th.min_month_games
        a, b = _avg(r.ev.my_accuracy for r in rs), _avg(r.ev.opp_accuracy for r in rs)
        mine.append(a if ok else None)
        theirs.append(b if ok else None)
        stats[m] = {"games": len(rs), "accuracy": a, "opp_accuracy": b}
    if sum(v is not None for v in mine) < 2:
        return None, stats
    chart = Chart(
        kind="line",
        title="Accuracy by month",
        labels=labels,
        series=[Series("You", mine), Series("Opponents", theirs)],
        value_format="float1",
        note=f"Months with fewer than {th.min_month_games} analysed games are left blank.",
    )
    return chart, stats


# --------------------------------------------------------------------------- module entry point
def _engine_label(records: Sequence[Analysed]) -> str:
    (engine, depth), _ = Counter((r.ev.engine, r.ev.depth) for r in records).most_common(1)[0]
    return f"{engine}, depth {depth}" if depth else engine


def _kpis(records: Sequence[Analysed], me: Tally, opp: Tally) -> list[Kpi]:
    my_acc, opp_acc = _avg(r.ev.my_accuracy for r in records), _avg(r.ev.opp_accuracy for r in records)
    return [
        Kpi("Games analysed", len(records), "int", hint=_engine_label(records)),
        Kpi("Your accuracy", my_acc, "float1", hint=f"opponents {_f1(opp_acc)}"),
        Kpi("Opponents' accuracy", opp_acc, "float1", hint="same games"),
        Kpi("Blunders /100 moves", me.per100(me.blunders), "float1", hint=f"opponents {_f1(opp.per100(opp.blunders))}"),
        Kpi("Mistakes /100 moves", me.per100(me.mistakes), "float1", hint=f"opponents {_f1(opp.per100(opp.mistakes))}"),
        Kpi(
            "Inaccuracies /100 moves",
            me.per100(me.inaccuracies),
            "float1",
            hint=f"opponents {_f1(opp.per100(opp.inaccuracies))}",
        ),
        Kpi(
            "Average centipawn loss",
            round(me.acpl) if me.acpl is not None else None,
            "int",
            hint=f"opponents {round(opp.acpl) if opp.acpl is not None else 'n/a'}",
        ),
    ]


def _summary(records: Sequence[Analysed], me: Tally, opp: Tally, insights: list[Insight], th: Thresholds) -> str:
    my_acc, opp_acc = _avg(r.ev.my_accuracy for r in records), _avg(r.ev.opp_accuracy for r in records)
    n = len(records)
    text = f"Stockfish reviewed {n} of your games"
    if my_acc is not None and opp_acc is not None:
        text += f": your accuracy averaged {my_acc:.1f} against {opp_acc:.1f} for your opponents"
    text += (
        f", and you blundered {_f1(me.per100(me.blunders))} times per 100 moves "
        f"(opponents {_f1(opp.per100(opp.blunders))})."
    )
    if n < th.min_games:
        return f"Not enough data for firm conclusions: only {n} analysed game{'s' if n != 1 else ''}. {text}"
    ranked = sorted((i for i in insights if i.kind != "observation"), key=lambda i: -i.priority)
    if ranked:
        text += f" Key finding: {ranked[0].title}."
    return text


def analyze(ctx: AnalysisContext) -> ModuleResult:
    """Accuracy, blunder rates, phases, time pressure, conversion, tactics and opening outcomes vs your opponents."""
    th = Thresholds.from_ctx(ctx)
    if not ctx.evals:
        return ModuleResult(
            key=KEY,
            title=TITLE,
            summary="Engine analysis was not run, so accuracy, blunder, game-phase and tactics insights are missing. "
            "Re-run the report with --engine (Stockfish must be installed; use --stockfish PATH if it isn't found "
            "automatically).",
            stats={"games": 0, "engine": False},
        )
    records = join(ctx)
    if not records:
        return ModuleResult(
            key=KEY,
            title=TITLE,
            summary="Not enough data: none of the engine-analysed games are in the current selection "
            "(or they are too short to judge).",
            stats={"games": 0, "engine": True},
        )

    me_games, opp_games = per_game(records, True), per_game(records, False)
    me, opp = total(me_games), total(opp_games)
    tables: list[Table] = []
    charts: list[Chart] = []
    insights: list[Insight] = []

    tc_table, tc_stats = time_class_table(records)
    tables.append(tc_table)
    blunders = blunder_insight(records, me_games, opp_games, th)
    if blunders:
        insights.append(blunders)
    by_result, result_stats = accuracy_by_result(records, th)
    if by_result:
        insights.append(by_result)

    phase_table, phase_chart, phase_insights, phase_stats = phase_section(records, th)
    tables.append(phase_table)
    charts.append(phase_chart)
    insights += phase_insights

    pressure_table, pressure_insights, pressure_stats = pressure_section(records, th)
    if pressure_table:
        tables.append(pressure_table)
    insights += pressure_insights

    conv_table, conv_insights, conv_stats = conversion_section(records, th)
    tables.append(conv_table)
    insights += conv_insights

    tactic_table, tactic_insights, tactic_stats = tactics_section(records, me_games, opp_games, th)
    tables.append(tactic_table)
    insights += tactic_insights

    piece_table, bucket_table, bucket_chart, anatomy_insights, anatomy_stats = anatomy_section(records, th)
    tables += [piece_table, bucket_table]
    charts.append(bucket_chart)
    insights += anatomy_insights

    opening_table, opening_insights, opening_stats = opening_section(records, th)
    if opening_table:
        tables.append(opening_table)
    insights += opening_insights

    trend_chart, trend_stats = accuracy_trend(records, th)
    if trend_chart:
        charts.append(trend_chart)

    def side(t: Tally, acc: Optional[float]) -> dict[str, Any]:
        return {
            "moves": t.moves, "accuracy": acc, "acpl": t.acpl,
            "blunders_per100": t.per100(t.blunders), "mistakes_per100": t.per100(t.mistakes),
            "inaccuracies_per100": t.per100(t.inaccuracies),
        }

    stats = {
        "games": len(records),
        "engine": _engine_label(records),
        "player": side(me, _avg(r.ev.my_accuracy for r in records)),
        "opponents": side(opp, _avg(r.ev.opp_accuracy for r in records)),
        "by_time_class": tc_stats,
        "accuracy_by_result": result_stats,
        "phases": phase_stats,
        "time_pressure": pressure_stats,
        "conversion": conv_stats,
        "tactics": tactic_stats,
        "blunder_anatomy": anatomy_stats,
        "openings": opening_stats,
        "accuracy_by_month": trend_stats,
    }
    return ModuleResult(
        key=KEY,
        title=TITLE,
        summary=_summary(records, me, opp, insights, th),
        kpis=_kpis(records, me, opp),
        tables=tables,
        charts=charts,
        insights=insights,
        stats=stats,
    )
