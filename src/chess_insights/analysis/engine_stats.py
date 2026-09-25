"""Engine review: accuracy, blunder rates, weak game phases, conversion, tactics and opening outcomes.

Needs Stockfish analysis (``ctx.evals``, produced by ``engine.py``). Every rate is
benchmarked against the OPPONENTS' moves in the same games: they are rating-matched
players facing the same positions on the same clock, which controls for position
difficulty and playing strength far better than any fixed "good accuracy" number.

Moves within one game are not independent (a bad game produces several errors), so
move-level rate comparisons count each move as ``1 / MOVE_DESIGN_EFFECT`` of an
observation before testing. Comparisons repeated over groups (phases, time classes,
openings) are Benjamini-Hochberg adjusted.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from ..context import AnalysisContext
from ..engine import PHASES
from ..models import TIME_CLASSES, Chart, Game, GameEval, Insight, Kpi, ModuleResult, PlyEval, Series, Table
from ..stats import (
    MeanTest,
    bh_adjust,
    clamp,
    combined_confidence,
    mean_test,
    pct,
    sample_confidence,
    two_proportion_test,
)
from .results import MAX_EXAMPLES, fmt_signed, slugify, with_p_value

KEY = "engine"
TITLE = "Engine review"

MIN_PLIES = 4  # shorter games are aborted starts, not skill
MOVE_DESIGN_EFFECT = 2.0  # errors cluster within games: count each move as half an independent observation
OPENING_PLY = 20  # opening outcome = eval after both players' 10th move
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

    min_games: int = 10  # analysed games before any strength / weakness is claimed
    min_confidence: float = 0.4  # combined_confidence needed for a strength / weakness
    max_p_value: float = 0.05  # ... and the (adjusted) test must be significant at this level
    min_moves: int = 300  # your analysed moves before overall blunder / tactics comparisons
    blunder_ratio: float = 1.25  # your blunder rate vs your opponents' (or the inverse, for a strength)
    min_phase_moves: int = 150  # your moves in a phase before judging it
    phase_ratio: float = 1.3  # your mistake+blunder rate in a phase vs your opponents'
    pressure_fraction: float = 0.15  # "short of time": clock before the move below this share of the start
    min_pressure_moves: int = 30  # your moves while short of time, per time class
    pressure_ratio: float = 2.0  # blunder rate short of time vs otherwise
    win_threshold: float = 85.0  # win % that counts as a winning position (your point of view)
    loss_threshold: float = 15.0  # win % that counts as a lost position
    min_conversion_games: int = 10  # games that reached a winning (or lost) position, for each side
    min_conversion_gap: float = 0.10  # your conversion rate vs your opponents'
    tactic_ratio: float = 1.3  # your missed-tactic / hung-material rate vs your opponents'
    min_missed_mates: int = 3  # missed forced mates before they are mentioned
    min_anatomy_blunders: int = 10  # blunders before describing which pieces / moves they come from
    min_opening_games: int = 5  # analysed games for an opening row
    min_opening_insight_games: int = 8  # ... and for an opening strength / weakness
    opening_eval_threshold: float = 60.0  # average move-10 eval (cp) that counts as a clearly good / bad opening
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


def rate_test(x1: int, n1: int, x2: int, n2: int) -> MeanTest:
    """two_proportion_test on move counts, deflated by MOVE_DESIGN_EFFECT (moves cluster within games)."""
    d = MOVE_DESIGN_EFFECT
    return two_proportion_test(x1 / d, round(n1 / d), x2 / d, round(n2 / d))


def _clearly_higher(a: Optional[float], b: Optional[float], ratio: float) -> bool:
    """a is at least ``ratio`` times b (and strictly above it; any positive a beats b = 0)."""
    return a is not None and b is not None and a > b and a >= ratio * b


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
    """The mover's clock before ``ply``: their previous clock, or the starting clock for their first move."""
    if ply < 2:
        return float(g.base_seconds) if g.base_seconds else None
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


def blunder_insight(records: Sequence[Analysed], me: Tally, opp: Tally, th: Thresholds) -> Optional[Insight]:
    """Overall blunder rate vs your opponents in the same games."""
    if len(records) < th.min_games or me.moves < th.min_moves or opp.moves < th.min_moves:
        return None
    mine, theirs = me.per100(me.blunders), opp.per100(opp.blunders)
    test = rate_test(me.blunders, me.moves, opp.blunders, opp.moves)
    confidence = combined_confidence(test, th.min_moves)
    if test.p_value > th.max_p_value or confidence < th.min_confidence:
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
    me = {ph: tally(records, True, lambda r, p, ph=ph: p.phase == ph) for ph in PHASES}
    opp = {ph: tally(records, False, lambda r, p, ph=ph: p.phase == ph) for ph in PHASES}
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
    tests = {ph: rate_test(me[ph].errors, me[ph].moves, opp[ph].errors, opp[ph].moves) for ph in judged}
    adjusted = dict(zip(judged, bh_adjust([tests[ph].p_value for ph in judged])))
    weak, strong = [], []
    for ph in judged:
        mine, theirs = me[ph].per100(me[ph].errors), opp[ph].per100(opp[ph].errors)
        confidence = combined_confidence(with_p_value(tests[ph], adjusted[ph]), th.min_phase_moves)
        stats[ph].update(p_value=tests[ph].p_value, p_adjusted=adjusted[ph])
        if adjusted[ph] > th.max_p_value or confidence < th.min_confidence:
            continue
        if _clearly_higher(mine, theirs, th.phase_ratio):
            weak.append((mine - theirs, ph, confidence))
        elif _clearly_higher(theirs, mine, th.phase_ratio):
            strong.append((theirs - mine, ph, confidence))
    insights = []
    if weak:
        _, ph, confidence = max(weak)
        insights.append(_phase_insight(records, ph, me[ph], opp[ph], confidence, adjusted[ph], "weakness"))
    if strong:
        _, ph, confidence = max(strong)
        insights.append(_phase_insight(records, ph, me[ph], opp[ph], confidence, adjusted[ph], "strength"))
    return table, chart, insights, stats


def _phase_insight(
    records: Sequence[Analysed], ph: str, m: Tally, o: Tally, confidence: float, p_adj: float, kind: str
) -> Insight:
    mine, theirs = m.per100(m.errors) or 0.0, o.per100(o.errors) or 0.0
    detail = (
        f"In the {ph} you made {m.errors} mistakes and blunders in {m.moves} moves ({mine:.1f} per 100 moves); "
        f"your opponents made {theirs:.1f} per 100 moves in the same games. Average centipawn loss there: "
        f"{m.acpl or 0:.0f} for you vs {o.acpl or 0:.0f} for them."
    )
    evidence = {
        "phase": ph, "moves": m.moves, "errors": m.errors, "per100": mine, "acpl": m.acpl,
        "opp_moves": o.moves, "opp_errors": o.errors, "opp_per100": theirs, "opp_acpl": o.acpl, "p_adjusted": p_adj,
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
            severity=clamp(mine / theirs - 1.0 if theirs else 1.0),
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
        severity=clamp(theirs / mine - 1.0 if mine else 1.0),
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
def pressure_section(
    records: Sequence[Analysed], th: Thresholds
) -> tuple[Optional[Table], list[Insight], dict[str, Any]]:
    """Your blunder rate on moves made while short of time vs other moves, per time class (live games)."""
    groups: dict[str, dict[str, Tally]] = defaultdict(
        lambda: {k: Tally() for k in ("me_low", "me_ok", "opp_low", "opp_ok")}
    )
    low_games: dict[str, list[Analysed]] = defaultdict(list)
    for r in records:
        g = r.game
        if g.time_class == "daily" or not g.base_seconds:
            continue
        limit = th.pressure_fraction * float(g.base_seconds)
        low_blunder = False
        for p in r.plies:
            before = _clock_before(g, p.ply)
            if before is None:
                continue
            low = before < limit
            groups[g.time_class][("me" if p.is_user else "opp") + ("_low" if low else "_ok")].add(p)
            low_blunder |= low and p.is_user and p.judgement == "blunder"
        if low_blunder:
            low_games[g.time_class].append(r)
    if not groups:
        return None, [], {}
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
    tests = {
        tc: rate_test(groups[tc]["me_low"].blunders, groups[tc]["me_low"].moves,
                      groups[tc]["me_ok"].blunders, groups[tc]["me_ok"].moves)
        for tc in judged
    }
    adjusted = dict(zip(judged, bh_adjust([tests[tc].p_value for tc in judged])))
    insights = []
    for tc in judged:
        t = groups[tc]
        low, ok = t["me_low"].per100(t["me_low"].blunders), t["me_ok"].per100(t["me_ok"].blunders)
        confidence = combined_confidence(with_p_value(tests[tc], adjusted[tc]), th.min_pressure_moves)
        stats[tc].update(p_value=tests[tc].p_value, p_adjusted=adjusted[tc])
        if adjusted[tc] > th.max_p_value or confidence < th.min_confidence:
            continue
        if not _clearly_higher(low, ok, th.pressure_ratio):
            continue
        opp_low, opp_ok = t["opp_low"].per100(t["opp_low"].blunders), t["opp_ok"].per100(t["opp_ok"].blunders)
        detail = (
            f"In {tc}, with less than {share} of your starting clock left you blundered {_f1(low)} times per 100 "
            f"moves ({t['me_low'].blunders} in {t['me_low'].moves} moves), against {_f1(ok)} per 100 moves with "
            f"more time. Your opponents in the same games: {_f1(opp_low)} vs {_f1(opp_ok)}."
        )
        insights.append(
            Insight(
                id=f"{KEY}.weakness.time-pressure-blunders-{tc}",
                kind="weakness",
                category="time",
                title=f"You blunder much more when short of time in {tc}",
                detail=detail,
                severity=clamp((low / ok - 1.0) / 3.0 if ok else 1.0),
                confidence=confidence,
                evidence={
                    "time_class": tc, "moves_low": t["me_low"].moves, "blunders_low": t["me_low"].blunders,
                    "per100_low": low, "per100_ok": ok, "opp_per100_low": opp_low, "opp_per100_ok": opp_ok,
                    "p_adjusted": adjusted[tc],
                },
                study=[
                    f"Budget your clock in {tc}: reach move 30 with at least a third of your starting time.",
                    "When short of time, play safe moves: keep pieces protected and your king covered; don't "
                    "start complications.",
                    "Practise with increment (e.g. 3+2 or 10+5) so the last phase of the game stops being a scramble.",
                ],
                example_games=_urls(low_games[tc], lambda r: (r.game.outcome != "loss", _recent(r))),
            )
        )
    return table, insights, stats


# --------------------------------------------------------------------------- conversion
@dataclass
class Conversion:
    reached: list[Analysed] = field(default_factory=list)  # you reached a winning position
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
        if max(wins) >= th.win_threshold:
            conv.reached.append(r)
        if min(wins) <= th.loss_threshold:
            conv.opp_reached.append(r)

    def wdl(rs: list[Analysed]) -> list[int]:
        c = Counter(r.game.outcome for r in rs)
        return [len(rs), c["win"], c["draw"], c["loss"]]

    rows = [
        [f"You reached a winning position (≥ {th.win_threshold:.0f}%)", *wdl(conv.reached), conv.rate],
        [f"Your opponent reached a winning position (you ≤ {th.loss_threshold:.0f}%)", *wdl(conv.opp_reached),
         conv.saved / len(conv.opp_reached) if conv.opp_reached else None],
    ]
    table = Table(
        title="Winning and losing positions",
        columns=["Situation", "Games", "Won", "Drawn", "Lost", "Converted / saved"],
        rows=rows,
        formats=["text", "int", "int", "int", "int", "pct"],
        note="Winning position: Stockfish gave that side at least an 85% chance to win at some point. Converted = "
        "you won those games; saved = you drew or won after being lost.",
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
    confidence = combined_confidence(test, 2 * th.min_conversion_games)
    stats.update(p_value=test.p_value)
    rate, opp_rate = conv.rate or 0.0, conv.opp_rate or 0.0
    base = (
        f"You reached a winning position (Stockfish: at least {th.win_threshold:.0f}% winning chances) in "
        f"{len(conv.reached)} games and won {conv.converted} of them ({pct(rate)}). Your opponents converted "
        f"{conv.opp_converted} of the {len(conv.opp_reached)} winning positions they reached against you "
        f"({pct(opp_rate)})."
    )
    evidence = {k: stats[k] for k in ("reached", "converted", "rate", "opp_reached", "opp_converted", "opp_rate")}
    significant = test.p_value <= th.max_p_value and confidence >= th.min_confidence
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
    insights.append(
        Insight(
            id=f"{KEY}.observation.resilience",
            kind="observation",
            category="conversion",
            title=f"You saved {conv.saved} of {len(conv.opp_reached)} lost positions ({pct(save_rate)})",
            detail=(
                f"In {len(conv.opp_reached)} games Stockfish gave you at most {th.loss_threshold:.0f}% winning "
                f"chances at some point; you still drew or won {conv.saved} of them. Your opponents saved "
                f"{len(conv.reached) - conv.converted} of their {len(conv.reached)} lost positions "
                f"({pct(1.0 - rate)})."
            ),
            severity=clamp(abs(save_rate - (1.0 - rate)) / 0.30),
            confidence=sample_confidence(len(conv.opp_reached)),
            evidence={"lost_positions": len(conv.opp_reached), "saved": conv.saved, "save_rate": save_rate},
            study=[
                "In a lost position, set practical problems: keep pieces on, create threats and make every "
                "move matter for your opponent.",
                "Replay the linked comebacks to see which defensive ideas worked for you.",
            ],
            example_games=_urls(saved, lambda r: (r.game.outcome != "win", _recent(r))),
        )
    )
    return table, insights, stats


# --------------------------------------------------------------------------- tactics
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
    records: Sequence[Analysed], me: Tally, opp: Tally, th: Thresholds
) -> tuple[Table, list[Insight], dict[str, Any]]:
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
        "opponent could punish with a capture.",
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
            tag: rate_test(
                getattr(me, TACTIC_TEXT[tag]["attr"]), me.moves, getattr(opp, TACTIC_TEXT[tag]["attr"]), opp.moves
            )
            for tag in tags
        }
        adjusted = dict(zip(tags, bh_adjust([tests[t].p_value for t in tags])))
        for tag in tags:
            text = TACTIC_TEXT[tag]
            mine_n, theirs_n = getattr(me, text["attr"]), getattr(opp, text["attr"])
            mine, theirs = me.per100(mine_n), opp.per100(theirs_n)
            confidence = combined_confidence(with_p_value(tests[tag], adjusted[tag]), th.min_moves)
            stats[f"{tag}_p_adjusted"] = adjusted[tag]
            if adjusted[tag] > th.max_p_value or confidence < th.min_confidence:
                continue
            detail = (
                f"In {len(records)} analysed games you {text['what']} {mine_n} times ({_f1(mine)} per 100 moves); "
                f"your opponents did so {_f1(theirs)} times per 100 moves in the same games."
            )
            evidence = {"count": mine_n, "per100": mine, "opp_count": theirs_n, "opp_per100": theirs,
                        "p_adjusted": adjusted[tag]}
            if _clearly_higher(mine, theirs, th.tactic_ratio):
                insights.append(
                    Insight(
                        id=f"{KEY}.weakness.{text['slug']}",
                        kind="weakness",
                        category="tactics",
                        title=text["weak_title"],
                        detail=detail,
                        severity=clamp(mine / theirs - 1.0 if theirs else 1.0),
                        confidence=confidence,
                        evidence=evidence,
                        study=text["study"],
                        example_games=_costliest(records, tag),
                    )
                )
            elif _clearly_higher(theirs, mine, th.tactic_ratio):
                insights.append(
                    Insight(
                        id=f"{KEY}.strength.{text['slug']}",
                        kind="strength",
                        category="tactics",
                        title=text["strong_title"],
                        detail=detail,
                        severity=clamp(theirs / mine - 1.0 if mine else 1.0),
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
def opening_eval(r: Analysed) -> Optional[float]:
    """Your eval (cp, capped at +-OPENING_CP_CAP) after ply OPENING_PLY; None if the game didn't get there."""
    p = next((p for p in r.plies if p.ply == OPENING_PLY - 1), None)
    if p is None:
        return None
    cp = p.cp_after if r.game.color == "white" else -p.cp_after
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
        tests[key] = mean_test(evals, min_sd=OPENING_EVAL_SD)
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
        "capped at ±5 pawns per game. Games from set-up positions are left out.",
    )
    insights: list[Insight] = []
    if len(records) < th.min_games:
        return table, insights, stats
    judged = [k for k in shown if len(groups[k]) >= th.min_opening_insight_games]
    adjusted = dict(zip(judged, bh_adjust([tests[k].p_value for k in judged])))
    for key in judged:
        family, colour = key
        test = with_p_value(tests[key], adjusted[key])
        confidence = combined_confidence(test, th.min_opening_insight_games)
        avg = test.mean
        if adjusted[key] > th.max_p_value or confidence < th.min_confidence or abs(avg) < th.opening_eval_threshold:
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
                        f"move 10 at {fmt_signed(avg)} centipawns on average (about {pawns} pawns); {worse} of "
                        f"them were already a pawn or more worse."
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
                        f"move 10 at {fmt_signed(avg)} centipawns on average (about {pawns} pawns); {better} of "
                        f"them were already a pawn or more better."
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

    me, opp = tally(records, True), tally(records, False)
    tables: list[Table] = []
    charts: list[Chart] = []
    insights: list[Insight] = []

    tc_table, tc_stats = time_class_table(records)
    tables.append(tc_table)
    blunders = blunder_insight(records, me, opp, th)
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

    tactic_table, tactic_insights, tactic_stats = tactics_section(records, me, opp, th)
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
