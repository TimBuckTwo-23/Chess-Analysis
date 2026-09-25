"""analysis/time_mgmt.py: time spent per move, time trouble, flagging, opening pace, clock balance."""

import json
import math
import random

import pytest

from chess_insights.analysis import time_mgmt
from chess_insights.context import AnalysisContext
from chess_insights.models import CATEGORIES, VALUE_FORMATS, ModuleResult
from factories import make_game

MOVES = ["Nf3", "Nf6", "Ng1", "Ng8"]
BLACK_FIRST_FEN = "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1"


# --------------------------------------------------------------------------- helpers
def assert_consistent(mr: ModuleResult) -> None:
    for chart in mr.charts:
        assert chart.value_format in VALUE_FORMATS
        assert chart.series and all(len(s.values) == len(chart.labels) for s in chart.series), chart.title
    for table in mr.tables:
        assert table.formats is None or len(table.formats) == len(table.columns)
        assert table.formats is None or set(table.formats) <= set(VALUE_FORMATS)
        assert all(len(row) == len(table.columns) for row in table.rows), table.title
        assert len(set(table.columns)) == len(table.columns), table.title
    assert all(k.format in VALUE_FORMATS for k in mr.kpis)
    for ins in mr.insights:
        assert ins.id.startswith(f"{mr.key}.{ins.kind}.") and ins.category in CATEGORIES
        assert 0.0 <= ins.severity <= 1.0 and 0.0 <= ins.confidence <= 1.0
        assert 2 <= len(ins.study) <= 4 and ins.title and ins.detail
        assert len(ins.example_games) <= 5 and all(u.startswith("https://") for u in ins.example_games)
    json.dumps(mr.stats, default=str, allow_nan=False)


def run(games, **options) -> ModuleResult:
    mr = time_mgmt.analyze(AnalysisContext("tester", sorted(games, key=lambda g: g.end_time), options=options))
    assert (mr.key, mr.title) == ("time", "Clock & time management")
    assert_consistent(mr)
    return mr


def clocks_from(white_spent, black_spent, base, inc=0, white_first=True):
    """Clock after each ply (mover's remaining seconds, increment included) from seconds spent per move."""
    remaining = {"w": float(base), "b": float(base)}
    order = [("w", white_spent), ("b", black_spent)] if white_first else [("b", black_spent), ("w", white_spent)]
    clocks = []
    for k in range(max(len(white_spent), len(black_spent))):
        for side, spent in order:
            if k < len(spent):
                remaining[side] = remaining[side] - spent[k] + inc
                clocks.append(round(remaining[side], 1))
    return clocks


def timed_game(my_spent, opp_spent, *, color="white", base=300, inc=0, **kw):
    white, black = (my_spent, opp_spent) if color == "white" else (opp_spent, my_spent)
    clocks = clocks_from(white, black, base, inc)
    kw.setdefault("time_control", f"{base}+{inc}" if inc else str(base))
    return make_game(
        color=color,
        base_seconds=base,
        increment=inc,
        moves_san=(MOVES * 100)[: len(clocks)],
        clocks=clocks,
        **kw,
    )


def insight(mr, id_):
    return next((i for i in mr.insights if i.id == id_), None)


def claims(mr):
    return [i.id for i in mr.insights if i.kind != "observation"]


def table(mr, title):
    return next(t for t in mr.tables if t.title.startswith(title))


def row(mr, title, first):
    t = table(mr, title)
    return dict(zip(t.columns, next(r for r in t.rows if r[0] == first)))


# --------------------------------------------------------------------------- time spent
def test_time_spent_first_move_uses_base_and_adds_increment():
    g = make_game(base_seconds=180, increment=2, moves_san=MOVES, clocks=[181.0, 180.0, 175.0, 170.0])
    assert time_mgmt.time_spent(g) == pytest.approx([1.0, 2.0, 8.0, 12.0])


def test_time_spent_is_floored_at_zero_and_handles_gaps():
    # White's clock goes up by more than the increment (e.g. time added by the opponent): spent is 0, not negative.
    clocks = [60.0, 59.0, 65.0, None, 64.0, 50.0]
    g = make_game(base_seconds=60, increment=1, moves_san=MOVES + MOVES[:2], clocks=clocks)
    assert time_mgmt.time_spent(g) == pytest.approx([1.0, 2.0, 0.0, None, 2.0, None])
    assert time_mgmt.time_spent(make_game(base_seconds=None, moves_san=MOVES, clocks=[50.0, 50.0, 40.0, 40.0])) == [
        None,
        None,
        pytest.approx(10.0),
        pytest.approx(10.0),
    ]


def test_time_spent_daily_and_short_clock_lists():
    daily = make_game(time_class="daily", time_control="1/86400", base_seconds=86400, moves_san=MOVES)
    assert time_mgmt.time_spent(daily) == [None] * 4
    assert time_mgmt.time_spent(make_game(moves_san=MOVES, clocks=[299.0])) == [pytest.approx(1.0), None, None, None]
    assert time_mgmt.time_spent(make_game(moves_san=[], clocks=[])) == []


# --------------------------------------------------------------------------- per-game clock facts
def test_time_trouble_threshold_boundaries():
    assert time_mgmt.trouble_threshold(300) == 30.0
    assert time_mgmt.trouble_threshold(30) == 5.0  # the 5-second floor
    at_line = timed_game([5] * 13 + [205], [5] * 14)  # my clock ends exactly at 30.0 s
    below = timed_game([5] * 13 + [205.1], [5] * 14)
    assert at_line.clocks[-2] == 30.0 and not time_mgmt.game_clock(at_line).my_trouble
    assert time_mgmt.game_clock(below).my_trouble
    assert not time_mgmt.game_clock(below).opp_trouble


def test_flagging_counts_as_time_trouble_even_with_comfortable_recorded_clocks():
    flagged = timed_game([5] * 10, [5] * 10, outcome="loss", termination="timeout", my_result_code="timeout")
    gc = time_mgmt.game_clock(flagged)
    assert gc.my_trouble and not gc.opp_trouble
    won = timed_game([5] * 10, [5] * 10, outcome="win", termination="timeout", opp_result_code="timeout")
    assert time_mgmt.game_clock(won).opp_trouble and not time_mgmt.game_clock(won).my_trouble
    # timeout vs insufficient material: the side to move after the last ply ran out of time
    draw = timed_game([5] * 10, [5] * 9, outcome="draw", termination="timeout_vs_insufficient")
    assert time_mgmt.flagged_side(draw) == "black"
    assert time_mgmt.game_clock(draw).opp_trouble and not time_mgmt.game_clock(draw).my_trouble


def test_black_to_move_first_positions_attribute_plies_correctly():
    # set-up position with Black to move: ply 0 is Black's (the player's) first move
    clocks = clocks_from([5] * 10, [50] + [5] * 9, base=300, white_first=False)
    g = make_game(color="black", initial_fen=BLACK_FIRST_FEN, moves_san=(MOVES * 5)[:20], clocks=clocks)
    gc = time_mgmt.game_clock(g)
    assert gc.my_spent[0] == pytest.approx(50.0) and gc.opp_spent[0] == pytest.approx(5.0)


def test_games_without_usable_clocks_are_skipped():
    assert time_mgmt.game_clock(make_game(with_clocks=False)) is None
    assert time_mgmt.game_clock(make_game(moves_san=["e4", "e5", "Nf3"])) is None  # fewer than 4 plies
    assert time_mgmt.game_clock(make_game(time_class="daily", base_seconds=86400)) is None
    assert time_mgmt.game_clock(make_game(base_seconds=None, clocks=[250.0] * 20)) is None
    only_mine = make_game(moves_san=MOVES, clocks=[290.0, None, 280.0, None])
    assert time_mgmt.game_clock(only_mine) is None


def test_opening_share_needs_fifteen_moves_each():
    short = time_mgmt.game_clock(timed_game([6] * 14, [3] * 14))
    assert short.my_opening is None and short.opp_opening is None
    full = time_mgmt.game_clock(timed_game([6] * 15 + [1] * 5, [3] * 15 + [1] * 5, inc=2))
    assert full.my_opening == pytest.approx(90 / 300) and full.opp_opening == pytest.approx(45 / 300)


def test_clock_balance_at_checkpoints():
    gc = time_mgmt.game_clock(timed_game([4] * 30, [6] * 30, base=600))
    assert gc.balance[20] == pytest.approx(40 / 600) and gc.balance[30] == pytest.approx(60 / 600)
    short = time_mgmt.game_clock(timed_game([4] * 25, [6] * 24, base=600))
    assert 20 in short.balance and 30 not in short.balance


# --------------------------------------------------------------------------- statistics helpers
def test_paired_rate_and_sign_tests():
    t = time_mgmt.paired_rate_test([True] * 10 + [False] * 30, [False] * 8 + [True] * 2 + [False] * 30)
    # 8 games only the player was in trouble, 0 only the opponent: diff = 8/40, z = 8 / sqrt(8)
    assert t.n == 40 and t.mean == pytest.approx(0.2) and t.z == pytest.approx(math.sqrt(8))
    assert time_mgmt.paired_rate_test([True, False], [True, False]).p_value == 1.0
    assert time_mgmt.paired_rate_test([], []).n == 0
    s = time_mgmt.sign_test(12, 4)
    assert s.n == 16 and s.mean == pytest.approx(0.5) and s.z == pytest.approx(2.0)
    assert time_mgmt.sign_test(0, 0).p_value == 1.0


# --------------------------------------------------------------------------- module: empty / small input
def test_empty_daily_only_and_clockless_input():
    for games in (
        [],
        [make_game(time_class="daily", time_control="1/86400", base_seconds=86400) for _ in range(30)],
        [make_game(with_clocks=False) for _ in range(30)],
        [make_game() for _ in range(14)],
    ):
        mr = run(games)
        assert mr.summary.lower().startswith("not enough data") and mr.insights == []


def test_classes_ordered_by_games_and_small_classes_skipped():
    games = (
        [make_game(time_class="rapid", time_control="600", base_seconds=600) for _ in range(20)]
        + [make_game() for _ in range(16)]
        + [make_game(time_class="bullet", time_control="60", base_seconds=60) for _ in range(5)]
    )
    mr = run(games)
    assert [r[0] for r in table(mr, "Time trouble").rows] == ["Rapid", "Blitz"]
    assert mr.stats["main_time_class"] == "rapid" and mr.stats["skipped_time_classes"] == {"bullet": 5}
    assert "not enough data" not in mr.summary.lower()


# --------------------------------------------------------------------------- module: planted effects
def test_time_trouble_weakness_against_opponents_in_same_games():
    trouble = [timed_game([5] * 14 + [250], [5] * 15, outcome="loss") for _ in range(20)]  # 20 s left: < 30 s
    opp_trouble = [timed_game([5] * 15, [5] * 14 + [250], outcome="win") for _ in range(4)]
    calm = [timed_game([5] * 15, [5] * 15, outcome=o) for o in ["win", "draw", "loss", "win"] * 4]
    mr = run(trouble + opp_trouble + calm)
    r = row(mr, "Time trouble", "Blitz")
    assert r["Games"] == 40 and r["You in time trouble"] == pytest.approx(0.5)
    assert r["Opponents in time trouble"] == pytest.approx(0.1)
    assert r["Score in time trouble"] == 0.0 and r["Difference in time trouble"] == pytest.approx(-0.5)
    ins = insight(mr, "time.weakness.time-trouble-blitz")
    assert ins and ins.category == "time" and ins.kind == "weakness"
    assert "50%" in ins.detail and "10%" in ins.detail
    assert len(ins.example_games) == 5 and ins.example_games[0] == trouble[-1].url  # most recent first


def test_equal_time_trouble_is_not_a_weakness():
    both = [timed_game([5] * 14 + [250], [5] * 14 + [250], outcome=o) for o in ["win", "loss"] * 12]
    calm = [timed_game([5] * 15, [5] * 15, outcome=o) for o in ["win", "loss"] * 8]
    mr = run(both + calm)
    assert row(mr, "Time trouble", "Blitz")["You in time trouble"] == pytest.approx(0.6)
    assert insight(mr, "time.weakness.time-trouble-blitz") is None


def test_flag_counting_and_lost_on_time_weakness():
    lost_on_time = [
        timed_game([5] * 12, [5] * 12, outcome="loss", termination="timeout", my_result_code="timeout")
        for _ in range(12)
    ]
    other_losses = [timed_game([5] * 12, [5] * 12, outcome="loss", termination="resignation") for _ in range(13)]
    won_on_time = [
        timed_game([5] * 12, [5] * 12, outcome="win", termination="timeout", opp_result_code="timeout")
    ]
    wins = [timed_game([5] * 12, [5] * 12, outcome="win") for _ in range(14)]
    mr = run(lost_on_time + other_losses + won_on_time + wins)
    r = row(mr, "Winning and losing on time", "Blitz")
    assert (r["Games"], r["Losses"], r["Lost on time"], r["Won on time"], r["Net"]) == (40, 25, 12, 1, -11)
    assert r["Share of losses"] == pytest.approx(12 / 25)
    ins = insight(mr, "time.weakness.lost-on-time-blitz")
    assert ins and "12 of your 25 blitz losses" in ins.detail
    assert set(ins.example_games) <= {g.url for g in lost_on_time}
    assert mr.stats["by_time_class"]["blitz"]["lost_on_time"] == 12


def test_flags_below_minimum_losses_are_not_judged():
    games = [
        timed_game([5] * 12, [5] * 12, outcome="loss", termination="timeout", my_result_code="timeout")
        for _ in range(8)
    ] + [timed_game([5] * 12, [5] * 12, outcome="win") for _ in range(30)]
    assert insight(run(games), "time.weakness.lost-on-time-blitz") is None  # only 8 losses


def test_slow_opening_weakness_and_its_benchmark():
    slow = [timed_game([12] * 15 + [2] * 5, [5] * 15 + [2] * 5, outcome=o) for o in ["win", "loss"] * 10]
    mr = run(slow)
    r = row(mr, "Opening pace", "Blitz")
    assert r["Clock used: you"] == pytest.approx(0.6) and r["Clock used: opponents"] == pytest.approx(0.25)
    assert r["Behind at move 20"] == 1.0
    ins = insight(mr, "time.weakness.slow-opening-blitz")
    assert ins and ins.title == "You spend too long in the opening in blitz"
    assert "2.4×" in ins.detail and "first 15 moves" in ins.detail
    assert any("5 seconds per move" in s for s in ins.study)  # opponents' pace: 25% of 300 s over 15 moves

    similar = [timed_game([6] * 15 + [2] * 5, [5] * 15 + [2] * 5, outcome=o) for o in ["win", "loss"] * 10]
    assert insight(run(similar), "time.weakness.slow-opening-blitz") is None  # 1.2x is normal variation


def test_clock_handling_strength():
    flags = [
        timed_game([3] * 25, [6] * 25, outcome="win", termination="timeout", opp_result_code="timeout")
        for _ in range(14)
    ]
    rest = [timed_game([3] * 25, [6] * 25, outcome=o) for o in ["win", "loss", "draw"] * 5]
    rest.append(timed_game([3] * 25, [6] * 25, outcome="loss", termination="timeout", my_result_code="timeout"))
    mr = run(flags + rest)
    ins = insight(mr, "time.strength.clock-handling-blitz")
    assert ins and ins.kind == "strength" and "won on time 14 times" in ins.detail
    assert row(mr, "Opening pace", "Blitz")["Behind at move 20"] == 0.0


def test_think_time_profile_by_move_bucket():
    my = [float(k) for k in range(1, 46)]  # k seconds on move k
    games = [timed_game(my, [2.0] * 45, base=3600, time_class="rapid") for _ in range(15)]
    mr = run(games)
    t = table(mr, "Think time by move number")
    assert [r[0] for r in t.rows] == ["1–10", "11–20", "21–30", "31–40", "41+"]
    assert [r[1] for r in t.rows] == pytest.approx([5.5, 15.5, 25.5, 35.5, 43.0])
    assert [r[2] for r in t.rows] == pytest.approx([2.0] * 5)
    assert [r[3] for r in t.rows] == [150, 150, 150, 150, 75]
    chart = next(c for c in mr.charts if c.title.startswith("Average think time"))
    assert chart.labels == [r[0] for r in t.rows] and chart.value_format == "seconds"
    assert chart.series[0].values == pytest.approx([r[1] for r in t.rows])


def noise_games(seed: int, n_games: int = 80) -> list:
    """Both sides drawn from the same clock habits; results and flags independent of the clock."""
    rng = random.Random(seed)
    games = []
    for _ in range(n_games):
        pace = (rng.uniform(0.5, 1.8), rng.uniform(0.5, 1.8))
        n = rng.randint(10, 35)
        my = [rng.uniform(1, 10) * pace[0] for _ in range(n)]
        opp = [rng.uniform(1, 10) * pace[1] for _ in range(n)]
        outcome = rng.choice(["win", "loss", "draw"])
        kw = {"outcome": outcome}
        if outcome != "draw" and rng.random() < 0.25:
            kw.update(termination="timeout")
        games.append(timed_game(my, opp, base=180, opp_rating=1500 + rng.randint(-150, 150), **kw))
    return games


def test_symmetric_noise_produces_no_claims():
    assert claims(run(noise_games(0))) == []


def test_noise_false_claim_rate_matches_the_significance_bar():
    # Each claim needs an adjusted p <= 0.05, so on pure noise only a few runs in 40 may show one.
    runs_with_claims = sum(1 for seed in range(40) if claims(run(noise_games(seed))))
    assert runs_with_claims <= 4
