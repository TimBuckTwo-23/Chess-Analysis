"""The motif profile (coach/profile.py): counts of the patterns you and your opponents miss and allow, and the only
claims the coaching layer makes, which must be as honest as every other finding (few false claims on data without
a real difference, and real differences found)."""

import dataclasses
import math
import os
import random

import chess
import pytest

from chess_insights.coach import CoachConfig, deep, motifs, profile
from chess_insights.coach.deep import ProfileError
from chess_insights.coach.profile import GameCounts, motif_claims
from chess_insights.context import AnalysisContext
from chess_insights.models import Coaching, Line, ModuleResult, Motif
from factories import make_game, make_game_eval, make_ply_eval

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
GATED = frozenset({"fork", "pin", "skewer", "hangingPiece", "discoveredAttack", "backRankMate"})

# Per mistake or blunder: how often its refutation carries each pattern (what it allows the other side). All 17
# patterns the detectors name, as in production (motifs.GATED_THEMES), so the tests face as many tests per report.
ALLOWED = {"hangingPiece": 0.35, "fork": 0.10, "pin": 0.05, "skewer": 0.03, "discoveredAttack": 0.04,
           "backRankMate": 0.03, "discoveredCheck": 0.02, "doubleCheck": 0.005, "trappedPiece": 0.02,
           "smotheredMate": 0.002, "mateIn1": 0.06, "mateIn2": 0.04, "mateIn3": 0.02, "mateIn4": 0.01,
           "deflection": 0.03, "attraction": 0.02, "advancedPawn": 0.04}
MISSED = {k: 0.9 * v for k, v in ALLOWED.items()}  # patterns in a best line with no chance before it
THEMES = frozenset(ALLOWED)
TACTICAL = 0.6  # share of errors whose lines carry patterns at all (patterns come together)
FORMATS = (("bullet", 1.4), ("blitz", 1.0), ("rapid", 0.7))  # (format, how error-prone)
RUNS = int(os.environ.get("MOTIF_RUNS", "0"))  # > 0: re-measure the rates below over that many worlds each


def _poisson(rng: random.Random, lam: float) -> int:
    if lam > 30:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _patterns(rng: random.Random, base: dict, rich: dict, scale: float = 1.0) -> tuple:
    """The patterns along one line: only tactical errors carry any, and they come together."""
    if rng.random() > TACTICAL:
        return ()
    return tuple(t for t, p in base.items() if rng.random() < min(1.0, p * scale / TACTICAL * rich[t]))


def _add(gc: GameCounts, kind: str, theme: str, side: int) -> None:
    gc.counts.setdefault((kind, theme), [0, 0])[side] += 1


def simulate(n_games: int, seed: int, *, boost: float = 1.0, q: float = 0.5, planted_q=None, planted_allowed=None,
             punish: float = 1.0, indep_share: float = 0.1, followup_allows: float = 0.3) -> list[GameCounts]:
    """Engine-analysed games in which a missed pattern mostly exists because the other side had just allowed it.

    Each side makes errors of its own at its base rate (yours times ``boost``), with heavy clustering: a game-level
    error propensity shared by both sides (wild games), a per-side factor and a per-game richness in each pattern
    shared by both sides (positions rich in forks). Each error's refutation carries patterns with the same
    per-error probabilities for both sides (``ALLOWED``). Every error is a chance for the other side, which misses
    it with probability ``q``, the same for both sides: an error on the next move whose best line carries the
    chance's patterns. Such a follow-up allows patterns too (``followup_allows`` of the usual rate) and is a chance
    in turn. A share ``indep_share`` of the own errors also carries a missed pattern with no chance before it.

    So a side that errs more hands the other side more patterns to miss: the reviewer's coupled world, in which
    the old per-move "missed" test made claims out of error rates. Planted differences: ``planted_q`` {theme: q}
    (you miss the chances carrying it with that probability), ``planted_allowed`` {theme: probability per error of
    yours}, ``punish`` (you miss every chance ``punish`` times as often: worse at punishing everything).
    """
    rng = random.Random(seed)
    planted_q, planted_allowed = planted_q or {}, planted_allowed or {}
    yours = {**ALLOWED, **planted_allowed}
    games = []
    for i in range(n_games):
        tc, prone = rng.choice(FORMATS)
        moves = rng.randint(15, 60)
        wild = rng.gammavariate(2.0, 0.5)
        rich = {t: rng.gammavariate(1.0, 1.0) for t in ALLOWED}
        gc = GameCounts(f"g{i}", tc, (moves, moves - rng.randint(0, 1)))
        chances = []  # (side that erred, the patterns it left the other side)
        for side in (0, 1):
            own = rng.gammavariate(4.0, 0.25) * (boost if side == 0 else 1.0)
            for _ in range(_poisson(rng, moves * 0.09 * prone * wild * own)):
                left = _patterns(rng, yours if side == 0 else ALLOWED, rich)
                gc.errors[side] += 1
                gc.chances[1 - side] += 1
                gc.pattern_chances[1 - side] += bool(left)
                for t in left:
                    _add(gc, "allowed", t, side)
                    _add(gc, profile.OWN_ALLOWED, t, side)
                if rng.random() < indep_share:
                    for t in _patterns(rng, MISSED, rich):
                        _add(gc, "missed", t, side)
                chances.append((side, left))
        while chances:
            side, left = chances.pop()
            other = 1 - side
            p = q if other == 1 else max([q * punish] + [planted_q[t] for t in left if t in planted_q])
            if rng.random() >= min(1.0, p):
                continue
            gc.errors[other] += 1
            gc.followups[other] += 1
            gc.pattern_followups[other] += bool(left)
            for t in left:
                _add(gc, profile.CHANCE_MISSED, t, other)
                _add(gc, "missed", t, other)
            again = _patterns(rng, yours if other == 0 else ALLOWED, rich, followup_allows)
            gc.chances[side] += 1
            gc.pattern_chances[side] += bool(again)
            for t in again:
                _add(gc, "allowed", t, other)
            chances.append((other, again))
        games.append(gc)
    return games


def _formats(games) -> dict[str, int]:
    return {tc: sum(g.time_class == tc for g in games) for tc, _ in FORMATS}


def _claims(n_games: int, seeds, gated=THEMES, **world) -> list[list[str]]:
    out = []
    for seed in seeds:
        games = simulate(n_games, seed, **world)
        out.append([i.id for i in motif_claims(games, gated, formats=_formats(games))[0]])
    return out


def _error_ratio(games) -> float:
    """Your mistakes and blunders per move over your opponents'."""
    e1, e2 = sum(g.errors[0] for g in games), sum(g.errors[1] for g in games)
    m1, m2 = sum(g.moves[0] for g in games), sum(g.moves[1] for g in games)
    return (e1 / m1) / (e2 / m2)


# --------------------------------------------------------------------------- honesty of the claims
# Measured over 200 worlds of 300 analysed games and 60 of 1,000 in each world below: 0 to 0.033 false claims per
# report. The old tests ("missed" per move, "allowed" among all errors) made 0.9 to 12 per report in the coupled
# worlds at 300 games and 5 to 26 at 1,000 ("you miss hanging pieces less often than your opponents" beside a
# higher blunder rate). MOTIF_RUNS=<n> re-measures with n worlds per size.
def test_no_false_motif_claims_when_both_sides_share_the_same_rates():
    claims = _claims(300, range(RUNS or 60))
    mean = sum(len(c) for c in claims) / len(claims)
    assert mean <= 0.05, [c for c in claims if c]


# The coupled worlds of the review: a side that errs more hands the other side more patterns to miss, and both
# sides miss a pattern they were left with the same probability. {your mistakes and blunders per move over your
# opponents': [(your base error rate multiplier, q, share of own errors with a missed pattern of their own)]}:
# with a few such misses (10%), and fully coupled (none, q 0.6: the review's worst case).
COUPLED = {0.8: [(0.5, 0.5, 0.1), (0.35, 0.6, 0.0)], 1.25: [(2.1, 0.5, 0.1), (2.6, 0.6, 0.0)]}


@pytest.mark.parametrize("ratio", list(COUPLED))
def test_different_error_rates_are_not_motif_findings_when_missed_patterns_follow_allowed_ones(ratio):
    claims = []
    for boost, q, indep in COUPLED[ratio]:
        world = dict(boost=boost, q=q, indep_share=indep)
        assert _error_ratio(simulate(1000, 0, **world)) == pytest.approx(ratio, abs=0.05)  # the world it says
        claims += _claims(300, range(RUNS or 15), **world) + _claims(1000, range(1000, 1000 + (RUNS or 5)), **world)
    mean = sum(len(c) for c in claims) / len(claims)
    assert mean <= 0.05, (mean, [c for c in claims if c])


@pytest.mark.parametrize("punish", [0.75, 1.3])
def test_missing_more_chances_of_every_kind_is_not_a_motif_finding(punish):
    # You miss every chance your opponents leave you 30% more often (or 25% less often) than they miss yours: a
    # difference in punishing mistakes in general, not "you miss forks more often" once per pattern.
    claims = _claims(300, range(RUNS or 20), punish=punish) + _claims(1000, range(RUNS or 5), punish=punish)
    assert sum(len(c) for c in claims) / len(claims) <= 0.05, [c for c in claims if c]


def test_a_real_difference_is_found():
    # you miss forks twice as often as your opponents, per chance (0.7 of the forks they leave you, against 0.35):
    # found in 30 of 40 worlds of 300 analysed games, 39 of 40 at 600 and 40 of 40 at 1,000 (1.5 times as often:
    # 14, 26 and 35 of 40)
    found, opposite = 0, 0
    for seed in range(20):
        games = simulate(300, 100 + seed, q=0.35, planted_q={"fork": 0.7})
        ids = [i.id for i in motif_claims(games, THEMES, formats=_formats(games))[0]]
        found += "tactics.weakness.motif-missed.fork" in ids
        opposite += "tactics.strength.motif-missed.fork" in ids
    assert found >= 14, found
    assert opposite == 0


def test_a_real_difference_in_what_you_allow_is_found():
    # your mistakes and blunders allow a fork twice as often as your opponents' (0.2 of them, against 0.1): found
    # in 38 of 40 worlds of 300 analysed games, 40 of 40 at 600 and 1,000
    claims = _claims(300, range(200, 220), planted_allowed={"fork": 0.2})
    assert sum("tactics.weakness.motif-allowed.fork" in c for c in claims) >= 16
    assert not any("tactics.strength.motif-allowed.fork" in c for c in claims)


def test_a_claim_carries_formats_a_chart_split_by_format_and_a_drill():
    games = simulate(300, 101, q=0.35, planted_q={"fork": 0.8})
    formats = _formats(games)
    claims, stats = motif_claims(games, THEMES, formats=formats)
    [fork] = [i for i in claims if i.id == "tactics.weakness.motif-missed.fork"]
    assert fork.kind == "weakness" and fork.category == "tactics"
    assert fork.title == "You miss forks more often than your opponents"
    assert fork.confidence >= 0.5 and 0 < fork.severity <= 1
    assert fork.evidence["p_adjusted"] <= 0.01 and fork.evidence["share_p_adjusted"] <= 0.01
    # per chance: of the forks your opponents' errors left you, the share you missed, against theirs
    ev = fork.evidence
    assert ev["chances"] == sum(g.count("allowed", "fork")[1] for g in games) and ev["rate"] > 1.5 * ev["opp_rate"]
    assert ev["count"] == sum(g.count(profile.CHANCE_MISSED, "fork")[0] for g in games)
    assert fork.formats == formats
    assert fork.detail.startswith("In 300 engine-analysed games (bullet, blitz and rapid), your opponents' mistakes "
                                  f"and blunders left you a fork {ev['chances']} times and you missed {ev['count']}")
    chart = fork.chart
    assert chart.labels == ["All games", "Bullet", "Blitz", "Rapid"] and chart.value_format == "pct"
    assert [s.name for s in chart.series] == ["You", "Opponents"]
    you, them = chart.series[0].values, chart.series[1].values
    assert you[0] == pytest.approx(ev["rate"]) and you[0] > them[0] > 0
    assert chart.table is not None and len(chart.table.rows) == 4
    assert any("lichess.org/training/fork" in a for a in fork.study)
    assert stats["fork:missed"]["claim"] == "weakness"


def test_the_allowed_share_leaves_out_the_missed_chances():
    # Your errors allow forks at the same rate as your opponents', but half of their errors are chances they
    # missed (which allow nothing): among all errors your share looks twice theirs, among their own errors not.
    games = []
    for i in range(200):
        gc = GameCounts(f"g{i}", "blitz", (40, 40), [4, 8], {("allowed", "fork"): [1, 1],
                                                            (profile.OWN_ALLOWED, "fork"): [1, 1]})
        gc.followups = [0, 4]
        games.append(gc)
    [test] = [t for t in profile.motif_tests(games, {"fork"}) if t.kind == "allowed"]
    assert test.share.mean == pytest.approx(0.0) and test.rate.mean == pytest.approx(0.0)
    assert motif_claims(games, {"fork"})[0] == []


def test_minimum_samples_before_any_test():
    games = simulate(40, 7, planted_q={"backRankMate": 0.9})
    tests = {(t.theme, t.kind) for t in profile.motif_tests(games, THEMES)}
    for t in profile.motif_tests(games, THEMES):
        assert t.games_with >= profile.MIN_MOTIF_GAMES and t.events >= profile.MIN_MOTIF_EVENTS
    assert ("backRankMate", "missed") not in tests  # rare: never enough games
    assert profile.motif_tests(games, frozenset()) == []  # nothing gated, nothing tested


def test_only_gated_patterns_are_named():
    games = simulate(300, 3, q=0.35, planted_q={"fork": 0.9})
    assert "tactics.weakness.motif-missed.fork" in [i.id for i in motif_claims(games, THEMES)[0]]
    assert motif_claims(games, frozenset({"pin"}))[0] == []  # the fork detector did not pass the gate
    table = profile.profile_table(games, {"pin"}, {"blitz": 300})
    assert [r[0] for r in table.rows] == ["Pin"]


# --------------------------------------------------------------------------- counting
def _records_ctx(n=3):
    games, evals = [], {}
    for k in range(n):
        g = make_game(color="white" if k % 2 == 0 else "black", time_class=["blitz", "rapid", "bullet"][k % 3])
        plies = [make_ply_eval(ply=i, mover="white" if i % 2 == 0 else "black", is_user=g.is_my_ply(i),
                               san=g.moves_san[i]) for i in range(g.plies)]
        games.append(g)
        evals[g.game_id] = make_game_eval(g.game_id, plies)
    return AnalysisContext(username="tester", games=games, evals=evals)


def _line(fen=START, moves=("e2e4", "e7e5", "g1f3")):
    return Line(fen=fen, moves_uci=list(moves))


def test_collect_counts_each_error_once_per_pattern_and_side():
    ctx = _records_ctx(2)
    from chess_insights.analysis.engine_stats import join

    records = join(ctx)
    g0, g1 = ctx.games
    best, refute = _line(), _line(moves=("d2d4", "d7d5"))
    found = {
        id(best): [Motif("fork", "best", 0, ["e4"], "you"), Motif("fork", "best", 2, ["f3"], "you"),
                   Motif("pin", "best", 1, ["e5"], "opponent")],  # the opponent's pin is not what you missed
        id(refute): [Motif("skewer", "refutation", 1, [], "opponent"), Motif("fork", "refutation", 0, [], "you")],
    }
    errors = [
        ProfileError(g0.game_id, 4, "you", "blitz", START, "e2e4", 30.0, best, refute),
        ProfileError(g0.game_id, 5, "opponent", "blitz", START, "e2e4", 20.0, best, None),
        ProfileError(g0.game_id, 5, "opponent", "blitz", START, "e2e4", 20.0, best, None),  # duplicate
        ProfileError("not-analysed", 3, "you", "blitz", START, "e2e4", 20.0, best, refute),
        ProfileError(g1.game_id, 6, "you", "rapid", START, "e2e4", 20.0, None, refute),
    ]
    games, events = profile.collect(records, errors, lambda line, role: found.get(id(line), []))
    c0, c1 = games
    assert c0.errors == [1, 1] and c1.errors == [1, 0]
    assert c0.count("missed", "fork") == (1, 1) and c0.count("missed", "pin") == (0, 0)
    assert c0.count("allowed", "skewer") == (1, 0) and c0.count("allowed", "fork") == (0, 0)
    assert c1.count("allowed", "skewer") == (1, 0) and c1.count("missed", "fork") == (0, 0)
    assert c0.moves == (10, 10)
    assert len(events) == 4


def test_collect_links_each_chance_to_the_next_move():
    """An error with a refutation is a chance for the other side; an error of theirs on the next ply missed it."""
    ctx = _records_ctx(1)
    from chess_insights.analysis.engine_stats import join

    [g] = ctx.games  # you are White: your moves are the even plies
    lines = {name: _line() for name in ("fork-left", "pin-left", "quiet", "missed-fork", "other", "none")}
    found = {
        id(lines["fork-left"]): [Motif("fork", "refutation", 0, [], "opponent"),
                                 Motif("hangingPiece", "refutation", 0, [], "opponent")],
        id(lines["pin-left"]): [Motif("pin", "refutation", 0, [], "opponent")],
        id(lines["missed-fork"]): [Motif("fork", "best", 0, ["e4"], "you")],
    }
    errors = [
        # your error at ply 4 leaves a fork and a hanging piece; your opponent errs at ply 5: both chances missed
        ProfileError(g.game_id, 4, "you", "blitz", START, "e2e4", 30.0, lines["none"], lines["fork-left"]),
        ProfileError(g.game_id, 5, "opponent", "blitz", START, "e2e4", 20.0, lines["missed-fork"], lines["quiet"]),
        # ... which leaves you a chance with no pattern: your error at ply 6 is a missed chance, of no pattern
        ProfileError(g.game_id, 6, "you", "blitz", START, "e2e4", 20.0, lines["other"], lines["pin-left"]),
        # your pin at ply 6 is taken (no error at ply 7); your error at ply 10 follows no error of theirs
        ProfileError(g.game_id, 10, "you", "blitz", START, "e2e4", 20.0, lines["missed-fork"], None),
    ]
    [c], events = profile.collect(join(ctx), errors, lambda line, role: found.get(id(line), []), gated={"fork", "pin"})
    assert c.errors == [3, 1] and c.chances == [1, 2] and c.followups == [1, 1]
    assert c.pattern_chances == [0, 2] and c.pattern_followups == [0, 1]  # the quiet chance has no named pattern
    assert c.count(profile.CHANCE_MISSED, "fork") == (0, 1) and c.count(profile.CHANCE_MISSED, "hangingPiece") == (0, 1)
    assert c.count("allowed", "pin") == (1, 0) and c.count(profile.OWN_ALLOWED, "pin") == (0, 0)  # ply 6: a followup
    assert c.count(profile.OWN_ALLOWED, "fork") == (1, 0) and c.own_errors(0) == 2 and c.own_errors(1) == 0
    # the fork your opponent missed at ply 5 came right after you left it; yours at ply 10 followed no chance
    missed = {(e.error.ply, e.after_chance) for e in events if e.kind == "missed"}
    assert missed == {(5, True), (10, False)}
    assert c.count("missed", "fork") == (1, 1)  # the table still counts every missed fork
    # without a gate every named pattern counts as a tactical chance
    [c], _ = profile.collect(join(ctx), errors, lambda line, role: found.get(id(line), []))
    assert c.pattern_chances == [0, 2]


def test_the_contrast_test_asks_whether_a_share_stands_out_more_for_you():
    # you miss 60% of your fork chances and 30% of the others; your opponents 30% and 30%
    rows = [(6, 10, 3, 10, 3, 10, 3, 10)] * 40
    t = profile.ratio_contrast_test(rows)
    assert t.mean == pytest.approx(math.log(2.0), abs=0.01) and t.p_value < 0.001  # log(60% / 30%) - log(1)
    # you miss everything more often than your opponents, forks no more than the rest: no contrast
    even = profile.ratio_contrast_test([(6, 10, 6, 10, 3, 10, 3, 10)] * 40)
    assert even.mean == pytest.approx(0.0) and even.p_value > 0.5
    assert profile.ratio_contrast_test([]).p_value == 1.0
    assert profile.ratio_contrast_test([(1, 1, 0, 0, 1, 1, 0, 0)] * 5).p_value == 1.0  # no other chances


def test_a_line_the_detectors_fail_on_counts_without_a_pattern():
    ctx = _records_ctx(1)
    from chess_insights.analysis.engine_stats import join

    [g] = ctx.games
    good, broken = _line(), _line(moves=("d2d4",))

    def detect(line, role):
        if line is broken:
            raise ValueError("odd line")
        return [Motif("fork", role, 0, ["e4"], "you" if role == "best" else "opponent")]

    errors = [ProfileError(g.game_id, 4, "you", "blitz", START, "e2e4", 30.0, good, broken)]
    notes = []
    [c], events = profile.collect(join(ctx), errors, detect, notes)
    assert c.count("missed", "fork") == (1, 0) and c.count("allowed", "fork") == (0, 0) and len(events) == 1
    assert notes == ["Motif profile: the pattern detectors failed on 1 engine line (counted without a pattern)."]


def test_table_and_chart_are_per_100_moves_with_practice_links():
    g = GameCounts("a", "blitz", (50, 50), [4, 2], {("missed", "fork"): [2, 1], ("allowed", "pin"): [1, 0],
                                                     ("missed", "mateIn2"): [3, 3]})
    h = GameCounts("b", "blitz", (50, 50), [1, 1], {("missed", "fork"): [1, 0]})
    table = profile.profile_table([g, h], GATED, {"blitz": 2})
    assert table.columns[:5] == ["Pattern", "You missed", "Opponents missed", "You allowed", "Opponents allowed"]
    assert table.rows[0] == ["Fork", 3.0, 1.0, 0.0, 0.0, "https://lichess.org/training/fork"]
    assert [r[0] for r in table.rows] == ["Fork", "Pin"]  # mateIn2 is not gated here
    assert "2 engine-analysed games (2 blitz)" in table.note
    # untested: the gaps between you and your opponents here are observations, not findings
    assert table.note.endswith(profile.UNTESTED_NOTE) and "not findings" in profile.UNTESTED_NOTE
    chart = profile.profile_chart([g, h], GATED, {"blitz": 2})
    assert chart.note.endswith(profile.UNTESTED_NOTE)
    assert chart.labels == ["Fork · missed", "Fork · allowed", "Pin · missed", "Pin · allowed"]
    assert chart.series[0].values == [3.0, 0.0, 0.0, 1.0] and chart.series[1].values == [1.0, 0.0, 0.0, 0.0]
    assert chart.table.rows[0][:3] == ["Fork", "missed", 3]
    assert profile.profile_table([g, h], frozenset(), {}) is None


def test_names_of_patterns():
    assert profile.motif_name("fork", plural=True) == "forks"
    assert profile.motif_name("hangingPiece") == "hanging piece"
    assert profile.motif_name("someNewTheme", plural=True) == "some new themes"


# --------------------------------------------------------------------------- the coaching step
def _planted_world(n_games=90, seed=5, p_you=0.8, p_opp=0.15, p_pin=0.3):
    """Games, evals and profiled errors where you miss the forks your opponents' errors leave you far more often
    than they miss yours (by default).

    Each game has four chances, one error each at the plies below: your opponent's error leaves you a fork, and
    your next move is an error too (the fork missed) with probability ``p_you``; your error leaves them a fork,
    missed with probability ``p_opp``; and the same with a pin (missed with ``p_pin`` on both sides), so there are
    other tactical chances to compare with."""
    rng = random.Random(seed)
    games, evals, errors, registry = [], {}, [], {}
    for k in range(n_games):
        g = make_game(color="white" if k % 2 else "black", time_class=["blitz", "rapid", "bullet"][k % 3],
                      game_id=f"pg{k}", url=f"https://www.chess.com/game/live/{9000 + k}")
        plies = [make_ply_eval(ply=i, mover="white" if i % 2 == 0 else "black", is_user=g.is_my_ply(i),
                               san=g.moves_san[i]) for i in range(g.plies)]
        o = 5 if g.color == "white" else 4  # your opponent's first error
        board, fens = chess.Board(), []
        for san in g.moves_san:
            fens.append(board.fen())
            board.push_san(san)
        mine = []
        for start, first, second, theme, p_miss in ((o, "opponent", "you", "fork", p_you),
                                                     (o + 3, "you", "opponent", "fork", p_opp),
                                                     (o + 6, "opponent", "you", "pin", p_pin),
                                                     (o + 9, "you", "opponent", "pin", p_pin)):
            for ply, side, role in ((start, first, "refutation"), (start + 1, second, "best")):
                if role == "best" and rng.random() >= p_miss:
                    continue  # the chance was taken: no error
                fen = fens[ply]
                best = Line(fen=fen, moves_uci=["h2h4"] if chess.Board(fen).turn == chess.WHITE else ["h7h5"])
                refutation = Line(fen=fens[ply + 1], moves_uci=[])
                if role == "refutation":
                    registry[id(refutation)] = [Motif(theme, "refutation", 0, ["e4", "e5"], "opponent")]
                else:
                    registry[id(best)] = [Motif(theme, "best", 0, ["f3", "e5", "d4"], "you")]
                played = chess.Board(fen).parse_san(g.moves_san[ply]).uci()
                errors.append(ProfileError(g.game_id, ply, side, g.time_class, fen, played, 20.0 + k % 7, best,
                                           refutation))
                if side == "you":
                    mine.append(ply)
        for i in mine:  # your errors are errors in the game analysis too (the puzzle export's selection)
            plies[i] = dataclasses.replace(plies[i], win_before=60.0, win_after=40.0,
                                           best_san="h4" if g.moves_san[i] != "h4" else "a4")
        games.append(g)
        evals[g.game_id] = make_game_eval(g.game_id, plies)
    ctx = AnalysisContext(username="tester", games=games, evals=evals)
    return ctx, errors, registry


def test_annotate_adds_the_profile_and_claims_to_the_engine_review(monkeypatch):
    ctx, errors, registry = _planted_world()
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: errors)
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: registry.get(id(line), []))
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork", "pin"}))
    engine = ModuleResult(key="engine", title="Engine review", summary="")
    coaching = Coaching()
    profile.annotate(ctx, coaching, [engine], CoachConfig())

    assert coaching.motif_profile is not None and coaching.motif_profile in engine.tables
    assert coaching.motif_chart is not None and coaching.motif_chart in engine.charts
    assert "(30 bullet, 30 blitz, 30 rapid)" in coaching.motif_profile.note  # the formats analysed, in order
    assert profile.UNTESTED_NOTE in coaching.motif_profile.note and profile.UNTESTED_NOTE in coaching.motif_chart.note
    [claim] = [i for i in engine.insights if i.id == "tactics.weakness.motif-missed.fork"]
    assert [i.id for i in engine.insights] == ["tactics.weakness.motif-missed.fork"]  # the pins: no difference
    assert claim.formats == {"bullet": 30, "blitz": 30, "rapid": 30}
    assert claim.example_games and all(u.startswith("https://www.chess.com/game/live/") for u in claim.example_games)
    diagram = claim.diagram
    assert diagram is not None and diagram.fen in {e.fen for e in errors if e.side == "you"}
    assert [a.kind for a in diagram.arrows] == ["played", "best"]
    assert diagram.strips and diagram.strips[0].frames[0].marks[0].kind == "attacker"
    assert diagram.orientation in ("white", "black") and diagram.link == claim.example_games[0]
    counts = coaching.settings["motif_profile"]["counts"]
    assert counts["fork"]["you_missed"] > 3 * counts["fork"]["opp_missed"] > 0
    assert counts["fork"]["you_chances"] == counts["fork"]["opp_allowed"] == 90
    assert counts["fork"]["you_missed_chances"] == counts["fork"]["you_missed"]
    assert engine.stats["motif_profile"]["tests"]["fork:missed"]["claim"] == "weakness"
    assert coaching.settings["motif_profile"]["findings"] == [claim.id]
    # the board shows one of the forks your opponent had just left you
    assert claim.diagram.title.startswith("The fork you missed")
    # the puzzle export gets your errors' best lines and their named patterns
    assert set(coaching.puzzle_lines) == {f"{e.game_id}:{e.ply}" for e in errors if e.side == "you"}
    assert any(v == ["fork"] for v in coaching.puzzle_themes.values())


def test_annotate_names_nothing_before_the_detectors_pass_the_gate(monkeypatch):
    ctx, errors, registry = _planted_world(40)
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: errors)
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: registry.get(id(line), []))
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset())
    engine = ModuleResult(key="engine", title="Engine review", summary="")
    coaching = Coaching()
    profile.annotate(ctx, coaching, [engine], CoachConfig())
    assert coaching.motif_profile is None and not engine.tables and not engine.insights
    assert any("precision check" in n for n in coaching.notes)
    assert coaching.settings["motif_profile"]["unnamed_themes"] == ["fork", "pin"]


def test_annotate_without_lines_or_engine_games_leaves_a_note(monkeypatch):
    engine = ModuleResult(key="engine", title="Engine review", summary="")
    coaching = Coaching()
    profile.annotate(AnalysisContext(username="x", games=[]), coaching, [engine], CoachConfig())
    assert coaching.notes and not engine.tables
    ctx, _, _ = _planted_world(5)
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: [])
    coaching = Coaching()
    profile.annotate(ctx, coaching, [engine], CoachConfig())  # the stub on this branch returns []
    assert any("no engine lines" in n for n in coaching.notes) and coaching.motif_profile is None


@pytest.mark.parametrize("kind", ["missed", "allowed"])
def test_event_diagram_shows_the_right_line(kind):
    g = make_game(color="white")
    best, refutation = Line(START, ["g1f3", "b8c6"]), Line(START, ["d2d4", "d7d5", "c2c4"])
    e = ProfileError(g.game_id, 0, "you", "blitz", START, "d2d4", 25.0, best, refutation)
    ev = profile.MotifEvent(g, e, kind, Motif("fork", "best" if kind == "missed" else "refutation", 1, ["c6"], ""))
    d = profile.event_diagram(ev)
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("d2", "d4", "played"), ("g1", "f3", "best")]
    frames = d.strips[0].frames
    assert [f.move for f in frames] == (["1.Nf3", "1...Nc6"] if kind == "missed" else ["1.d4", "1...d5", "2.c4"])
    assert frames[1].marks and frames[1].marks[0].square == "c6"
    assert "It cost you 25 percentage points" in d.caption and d.time_class == "blitz"
    assert d.caption.startswith("You played 1.d4 (red); 1.Nf3 (green) was better.")


def test_event_diagram_of_an_opponents_miss_keeps_your_side_at_the_bottom():
    g = make_game(color="black")
    best, refutation = Line(START, ["g1f3", "b8c6"]), Line(START, ["d2d4", "d7d5"])
    e = ProfileError(g.game_id, 0, "opponent", "rapid", START, "d2d4", 18.0, best, refutation)
    d = profile.event_diagram(profile.MotifEvent(g, e, "missed", Motif("fork", "best", 0, ["f3"], "you")))
    assert d.title == "A fork your opponent missed: 1.Nf3 instead of 1.d4"
    assert d.caption.startswith("Your opponent played 1.d4 (red); 1.Nf3 (green) was better. It cost your opponent 18")
    assert d.orientation == "black"
    d = profile.event_diagram(profile.MotifEvent(g, e, "allowed", Motif("fork", "refutation", 1, [], "opponent")))
    assert d.title == "The fork your opponent allowed after 1.d4"


def test_the_strip_reaches_the_pattern():
    g = make_game(color="white")
    moves = ["g1f3", "b8c6", "b1c3", "g8f6", "e2e4", "e7e5", "f3e5", "c6e5"]
    e = ProfileError(g.game_id, 0, "you", "blitz", START, "a2a3", 20.0, Line(START, moves), None)
    d = profile.event_diagram(profile.MotifEvent(g, e, "missed", Motif("fork", "best", 6, ["e5", "f7", "c6"], "you")))
    frames = d.strips[0].frames
    assert len(frames) == 7 and frames[6].move == "4.Nxe5" and [m.square for m in frames[6].marks] == ["e5", "f7", "c6"]
    short = profile.event_diagram(profile.MotifEvent(g, e, "missed", Motif("fork", "best", 0, ["f3"], "you")))
    assert len(short.strips[0].frames) == 4  # a few moves of context after an early pattern


def test_a_strength_shows_one_of_your_opponents_misses(monkeypatch):
    ctx, errors, registry = _planted_world(p_you=0.15, p_opp=0.8)
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: errors)
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: registry.get(id(line), []))
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork", "pin"}))
    engine = ModuleResult(key="engine", title="Engine review", summary="")
    profile.annotate(ctx, Coaching(), [engine], CoachConfig())
    [claim] = [i for i in engine.insights if i.id == "tactics.strength.motif-missed.fork"]
    assert claim.kind == "strength" and claim.title == "You miss forks less often than your opponents"
    assert claim.chart is not None and claim.diagram is not None
    assert claim.diagram.title.startswith("A fork your opponent missed")
    assert claim.diagram.link in claim.example_games


# --------------------------------------------------------------------------- the previous position and the puzzle lines
def _one_game_with_errors(errors_at=(4, 6, 8)):
    """One game of yours (White, the default moves) whose moves at ``errors_at`` lost 20 win-% points."""
    import chess

    g = make_game(color="white", game_id="prev1", url="https://www.chess.com/game/live/77")
    plies = []
    for i, san in enumerate(g.moves_san):
        worse = i in errors_at
        plies.append(make_ply_eval(ply=i, mover="white" if i % 2 == 0 else "black", is_user=g.is_my_ply(i), san=san,
                                   best_san="h4" if worse else san, win_before=60.0 if worse else 51.8,
                                   win_after=40.0 if worse else 51.8))
    board, fens = chess.Board(), []
    for san in g.moves_san:
        fens.append(board.fen())
        board.push_san(san)
    ctx = AnalysisContext(username="tester", games=[g], evals={g.game_id: make_game_eval(g.game_id, plies)})
    return ctx, g, fens


def test_best_lines_are_read_with_the_position_before_the_last_move():
    ctx, g, fens = _one_game_with_errors()
    records = profile.join(ctx)
    best = Line(fen=fens[4], moves_uci=["f1c4"])
    refutation = Line(fen=fens[4], moves_uci=["b1c3", "d7d5"])
    calls = []

    def three(line, role, previous_fen=None):
        calls.append((role, previous_fen))
        return []

    # computed from the game (one replay) when the error does not carry it ...
    profile.collect(records, [ProfileError(g.game_id, 4, "you", "blitz", fens[4], "b1c3", 20.0, best, refutation)],
                    three)
    assert calls == [("best", fens[3]), ("refutation", None)]  # the position before 2...Nc6, the opponent's move
    # ... or taken from it (deep.profile_errors sets it)
    calls.clear()
    carried = ProfileError(g.game_id, 4, "you", "blitz", fens[4], "b1c3", 20.0, best, refutation, previous_fen="x")
    profile.collect(records, [carried], three)
    assert calls[0] == ("best", "x")
    # an opponent's error: the position before your last move
    calls.clear()
    profile.collect(records, [ProfileError(g.game_id, 5, "opponent", "blitz", fens[5], "f8c5", 20.0,
                                           Line(fen=fens[5], moves_uci=["g8f6"]), None)], three)
    assert calls == [("best", fens[4])]
    # a detector that takes only (line, role) still works
    two_calls = []
    profile.collect(records, [ProfileError(g.game_id, 4, "you", "blitz", fens[4], "b1c3", 20.0, best, refutation)],
                    lambda line, role: two_calls.append(role) or [])
    assert two_calls == ["best", "refutation"]


def test_deep_records_the_previous_position_of_every_error():
    ctx, g, fens = _one_game_with_errors()
    found = deep.profile_errors(ctx)
    assert [(e.ply, e.previous_fen) for e in found] == [(4, fens[3]), (6, fens[5]), (8, fens[7])]
    assert deep.previous_fens(g, [0, 1, 4, 999]) == {1: fens[0], 4: fens[3]}


def test_puzzle_lines_are_kept_only_for_the_puzzle_exports_errors(monkeypatch):
    """Not every profiled error of yours (about 1,500 in a real run): the ones mistakes.build_puzzles exports, plus
    the explained positions."""
    from chess_insights.coach import puzzles

    ctx, g, fens = _one_game_with_errors(errors_at=(4, 6, 8))
    errors = [ProfileError(g.game_id, ply, "you", "blitz", fens[ply], "a2a3", 20.0,
                           Line(fen=fens[ply], moves_uci=["h2h4"]), Line(fen=fens[ply], moves_uci=["a2a3"]))
              for ply in (2, 4, 6, 8)]  # ply 2 is no error in the game analysis: no puzzle
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: errors)
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: [])
    coaching = Coaching()
    profile.annotate(ctx, coaching, [], CoachConfig())
    assert set(coaching.puzzle_lines) == {f"{g.game_id}:{p}" for p in (4, 6, 8)}
    # at most PUZZLE_LIMIT puzzles, costliest first: here one
    real = puzzles.build_puzzles
    monkeypatch.setattr(puzzles, "build_puzzles", lambda games, evals: real(games, evals, limit=1))
    coaching = Coaching()
    profile.annotate(ctx, coaching, [], CoachConfig())
    assert len(coaching.puzzle_lines) == 1


def test_puzzle_themes_come_only_with_the_line_they_describe():
    """The profile pass adds a Themes entry only for a line it supplies, and only patterns within the moves kept."""
    g = make_game(color="white", game_id="pz")
    long = Line(START, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d4", "e5d4", "e1g1", "f6e4"])
    errors = [ProfileError("pz", ply, "you", "blitz", START, "a2a3", 20.0, long, None) for ply in (4, 6, 8, 10, 12)]
    events = [
        profile.MotifEvent(g, errors[0], "missed", Motif("fork", "best", 2, [], "you")),
        profile.MotifEvent(g, errors[1], "missed", Motif("pin", "best", 1, [], "you")),
        profile.MotifEvent(g, errors[2], "missed", Motif("skewer", "best", 9, [], "you")),  # beyond the 8 plies kept
        profile.MotifEvent(g, errors[3], "missed", Motif("pin", "best", 1, [], "you")),
    ]
    errors[4].played_uci = "e2e4"  # the line starts with the move played: no puzzle
    coaching = Coaching()
    deeper = Line(START, ["d2d4"])
    coaching.puzzle_lines["pz:6"] = deeper  # the deeper coach line (fill_puzzle_lines) found no pattern on it
    coaching.puzzle_themes["pz:10"] = []  # the puzzle pass settled this one without a line
    profile._puzzle_lines(coaching, errors, events, frozenset({"fork", "pin", "skewer"}))
    assert coaching.puzzle_lines["pz:6"] is deeper and "pz:6" not in coaching.puzzle_themes  # not the profile's pin
    assert coaching.puzzle_themes == {"pz:4": ["fork"], "pz:10": []}
    assert set(coaching.puzzle_lines) == {"pz:4", "pz:6", "pz:8"}
    assert len(coaching.puzzle_lines["pz:8"].moves_uci) == profile.PUZZLE_LINE_PLIES  # the skewer is not in it
