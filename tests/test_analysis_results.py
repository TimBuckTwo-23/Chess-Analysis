"""analysis/results.py: score vs expectation, colour, opponent strength, time controls, rating history."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from chess_insights import parse
from chess_insights.analysis import results
from chess_insights.context import AnalysisContext
from chess_insights.models import CATEGORIES, VALUE_FORMATS, ModuleResult
from chess_insights.stats import attenuated_expected, expected_score
from factories import make_game

T0 = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)


def at(days: float) -> datetime:
    return T0 + timedelta(days=days)


def run(games, **options) -> ModuleResult:
    mr = results.analyze(AnalysisContext("tester", sorted(games, key=lambda g: g.end_time), options=options))
    assert_consistent(mr)
    return mr


def assert_consistent(mr: ModuleResult) -> None:
    """Every chart/table/KPI/insight is well formed and the stats are strict JSON."""
    for chart in mr.charts:
        assert chart.value_format in VALUE_FORMATS
        assert chart.series and all(len(s.values) == len(chart.labels) for s in chart.series), chart.title
    for table in mr.tables:
        if table.formats is not None:
            assert len(table.formats) == len(table.columns) and set(table.formats) <= set(VALUE_FORMATS)
        assert all(len(row) == len(table.columns) for row in table.rows), table.title
    assert all(k.format in VALUE_FORMATS for k in mr.kpis)
    for ins in mr.insights:
        assert ins.id.startswith(f"{mr.key}.{ins.kind}.")
        assert ins.kind in ("strength", "weakness", "observation") and ins.category in CATEGORIES
        assert 0.0 <= ins.severity <= 1.0 and 0.0 <= ins.confidence <= 1.0
        assert 2 <= len(ins.study) <= 4 and ins.title and ins.detail
        assert len(ins.example_games) <= 5 and all(u.startswith("https://") for u in ins.example_games)
    json.dumps(mr.stats, default=str, allow_nan=False)


def games_by(n_wins: int, n_losses: int, n_draws: int = 0, **kw) -> list:
    outcomes = ["win"] * n_wins + ["loss"] * n_losses + ["draw"] * n_draws
    # interleave the outcomes in time (stepping by 7 permutes any length not divisible by 7)
    outcomes = [outcomes[(i * 7) % len(outcomes)] for i in range(len(outcomes))] if len(outcomes) % 7 else outcomes
    return [make_game(outcome=o, **kw) for o in outcomes]


def ids(mr: ModuleResult) -> set[str]:
    return {i.id for i in mr.insights}


# --------------------------------------------------------------------------- empty / tiny / missing data
def test_empty_input_says_not_enough_data():
    mr = run([])
    assert (mr.key, mr.title) == ("results", "Results & rating")
    assert "not enough data" in mr.summary.lower()
    assert mr.insights == [] and mr.stats["n"] == 0


def test_tiny_input_has_numbers_but_no_claims():
    mr = run([make_game(outcome="win"), make_game(outcome="loss"), make_game(outcome="draw")])
    assert "not enough data" in mr.summary.lower()
    assert mr.insights == []
    kpis = {k.label: k.value for k in mr.kpis}
    assert kpis["Games"] == 3 and kpis["Score"] == pytest.approx(0.5)


def test_games_without_ratings_do_not_break_anything():
    games = [make_game(outcome="win" if i % 3 else "loss", my_rating=None, opp_rating=None) for i in range(40)]
    mr = run(games)
    kpis = {k.label: k.value for k in mr.kpis}
    assert kpis["Rating predicts"] is None and kpis["Score vs rating"] is None and kpis["Rating equivalent"] is None
    assert "no ratings" in mr.summary
    assert not [i for i in mr.insights if i.kind != "observation"]
    buckets = next(t for t in mr.tables if t.title == "By opponent strength")
    assert [row[1] for row in buckets.rows] == [0] * 5
    assert "40 game(s) without ratings" in buckets.note
    assert not any(c.kind == "line" for c in mr.charts)


def test_daily_and_zero_move_games_count_in_totals():
    daily = make_game(time_class="daily", time_control="1/86400", base_seconds=86400, with_clocks=False, outcome="win")
    abandoned = make_game(moves_san=[], outcome="loss", termination="abandoned", my_result_code="abandoned")
    mr = run([daily, abandoned])
    assert mr.stats["n"] == 2 and (mr.stats["wins"], mr.stats["losses"]) == (1, 1)
    table = next(t for t in mr.tables if t.title == "By time control")
    assert [row[0] for row in table.rows] == ["Blitz", "Daily"]


# --------------------------------------------------------------------------- core maths
def test_summarize_expected_score_maths():
    g1 = make_game(my_rating=1500, opp_rating=1700, outcome="win")
    g2 = make_game(my_rating=1500, opp_rating=1300, outcome="loss")
    g3 = make_game(my_rating=1500, opp_rating=1500, outcome="win")
    g4 = make_game(my_rating=None, opp_rating=1500, outcome="draw")  # no expectation
    s = results.summarize([g1, g2, g3, g4])
    assert (s.n, s.wins, s.draws, s.losses, s.n_rated) == (4, 2, 1, 1, 3)
    assert s.score == pytest.approx(2.5 / 4)
    assert s.rated_score == pytest.approx(2 / 3)
    e = [expected_score(1500, 1700), expected_score(1500, 1300), 0.5]
    assert e[0] == pytest.approx(0.2402530733)
    assert s.expected == pytest.approx(sum(e) / 3)
    assert s.delta == pytest.approx(((1 - e[0]) + (0 - e[1]) + (1 - e[2])) / 3)
    att = results.summarize([g1, g2, g3, g4], attenuate=True)
    assert att.expected == pytest.approx(sum(attenuated_expected(x) for x in e) / 3)
    assert s.wdl == "2/1/1" and s.half_width == pytest.approx(1.96 * s.test.se)


def test_summarize_uses_pregame_ratings_and_handles_empty():
    g = make_game(my_rating=1510, opp_rating=1490, my_rating_before=1500, opp_rating_before=1500, outcome="win")
    assert results.summarize([g]).expected == pytest.approx(0.5)
    empty = results.summarize([])
    assert (empty.n, empty.score, empty.expected, empty.delta, empty.half_width, empty.shrunk()) == (
        0,
        None,
        None,
        None,
        None,
        None,
    )


def test_standard_error_is_floored_at_the_play_to_your_rating_variance():
    losses = results.summarize(games_by(0, 8))  # zero sample variance
    assert losses.test.se == pytest.approx((8 * 0.25) ** 0.5 / 8)
    assert losses.test.p_value == pytest.approx(0.0047, abs=5e-4)
    streak = results.summarize(games_by(8, 1))  # sample SD 0.33 < binomial 0.5
    assert streak.test.se == pytest.approx(0.5 / 3)
    with_draws = results.summarize(games_by(4, 4, 2))  # draws lower the variance: e(1-e) - d/4
    floor = ((10 * (0.25 - 0.2 / 4)) ** 0.5) / 10
    assert with_draws.test.se >= floor - 1e-12
    varied = results.summarize(games_by(20, 20))  # sample variance above the floor: left alone
    assert varied.test.se == pytest.approx((0.25 * 40 / 39) ** 0.5 / 40**0.5)


def test_difference_test_is_welch_style():
    a = results.summarize(games_by(30, 10)).test
    b = results.summarize(games_by(10, 30)).test
    d = results.difference_test(a, b, offset=0.04)
    assert d.mean == pytest.approx(a.mean - b.mean - 0.04)
    assert d.se == pytest.approx((a.se**2 + b.se**2) ** 0.5)
    assert d.p_value < 0.001
    tiny = results.difference_test(results.summarize(games_by(1, 0)).test, b)
    assert tiny.p_value == 1.0


@pytest.mark.parametrize(
    "diff, bucket",
    [
        (-400, "much_lower"),
        (-151, "much_lower"),
        (-150, "lower"),
        (-51, "lower"),
        (-50, "similar"),
        (0, "similar"),
        (50, "similar"),
        (51, "higher"),
        (150, "higher"),
        (151, "much_higher"),
        (None, None),
        (float("nan"), None),
    ],
)
def test_opponent_bucket_boundaries(diff, bucket):
    assert results.opponent_bucket(diff) == bucket


def test_opponent_table_assigns_boundary_games():
    diffs = [-151, -150, -51, -50, 0, 50, 51, 150, 151]
    games = [make_game(my_rating=1500, opp_rating=1500 + d) for d in diffs] + [make_game(opp_rating=None)]
    mr = run(games)
    table = next(t for t in mr.tables if t.title == "By opponent strength")
    assert [row[0] for row in table.rows] == [label for _, label in results.OPPONENT_BUCKETS]
    assert [row[1] for row in table.rows] == [1, 2, 3, 2, 1]
    much_lower = table.rows[0]
    e = expected_score(1500, 1349)
    assert much_lower[4] == pytest.approx(e) and much_lower[5] == pytest.approx(attenuated_expected(e))
    assert much_lower[6] == pytest.approx(1.0 - attenuated_expected(e))  # the default factory game is a win
    assert "1 game(s) without ratings" in table.note
    chart = next(c for c in mr.charts if c.title.startswith("Score vs realistic"))
    assert chart.labels == [label for _, label in results.OPPONENT_BUCKETS] and chart.reference == 0.0


# --------------------------------------------------------------------------- time controls & rating
def test_time_control_table_and_pools():
    blitz = [
        make_game(outcome="win", my_rating=1510, opp_rating=1500, end_time=at(1)),
        make_game(
            outcome="win",
            my_rating=1525,
            opp_rating=1500,
            my_rating_before=1510,
            opp_rating_before=1515,
            end_time=at(2),
        ),
        make_game(
            outcome="draw",
            my_rating=1524,
            opp_rating=1530,
            my_rating_before=1525,
            opp_rating_before=1529,
            end_time=at(3),
        ),
        make_game(
            outcome="loss",
            my_rating=1512,
            opp_rating=1480,
            my_rating_before=1524,
            opp_rating_before=1468,
            end_time=at(4),
        ),
    ]
    casual = make_game(
        outcome="win", rated=False, my_rating=1600, opp_rating=1600, end_time=at(5)
    )  # not a rating point
    rapid = [
        make_game(
            time_class="rapid", time_control="600", base_seconds=600, my_rating=1700, opp_rating=1650, end_time=at(2)
        )
    ]
    c960 = [
        make_game(
            rules="chess960", initial_fen="bnrqkrnb/pppppppp/8/8/8/8/PPPPPPPP/BNRQKRNB w KQkq - 0 1", end_time=at(3)
        )
    ]
    daily = [
        make_game(time_class="daily", time_control="1/86400", base_seconds=86400, with_clocks=False, end_time=at(6))
    ]
    mr = run(blitz + [casual] + rapid + c960 + daily)
    table = next(t for t in mr.tables if t.title == "By time control")
    assert [row[0] for row in table.rows] == ["Blitz", "Blitz (chess960)", "Rapid", "Daily"]
    row = dict(zip(table.columns, table.rows[0]))
    blitz_all = blitz + [casual]
    assert (row["Games"], row["W/D/L"]) == (5, "3/1/1")
    assert row["Score"] == pytest.approx(3.5 / 5)
    exp = [g.expected_score for g in blitz_all]
    assert row["Rating predicts"] == pytest.approx(sum(exp) / 5)
    assert row["vs rating"] == pytest.approx(sum(g.score - e for g, e in zip(blitz_all, exp)) / 5)
    # rating history uses rated games only: first game has no pre-game rating, so it starts from 1510
    assert (row["Current rating"], row["Change"], row["Peak"]) == (1512, 2, 1525)
    assert len(table.formats) == len(table.columns)


def test_rating_chart_months_union_with_gaps():
    blitz = [make_game(my_rating=1400 + i, end_time=at(i)) for i in range(5)]  # January
    blitz += [make_game(my_rating=1450 + i, end_time=at(60 + i)) for i in range(5)]  # March
    rapid = [make_game(time_class="rapid", my_rating=1600 + i, end_time=at(35 + i)) for i in range(6)]  # February
    rapid += [make_game(time_class="rapid", my_rating=1650 + i, end_time=at(62 + i)) for i in range(4)]  # March
    bullet = [
        make_game(time_class="bullet", my_rating=1200, end_time=at(10 + i)) for i in range(3)
    ]  # too few for a line
    mr = run(blitz + rapid + bullet)
    chart = next(c for c in mr.charts if c.kind == "line")
    assert chart.labels == ["2024-01", "2024-02", "2024-03"]
    series = {s.name: s.values for s in chart.series}
    assert series == {"Blitz": [1404.0, None, 1454.0], "Rapid": [None, 1605.0, 1653.0]}
    kpis = {k.label: k for k in mr.kpis}
    assert kpis["Blitz rating"].value == 1454 and kpis["Rapid rating"].value == 1653
    assert "Bullet rating" not in kpis  # fewer than 10 games
    assert kpis["Peak blitz rating"].value == 1454
    assert "rated games" in kpis["Blitz rating"].hint


def test_time_control_that_stands_out_is_one_claim():
    # Two pools make one comparison seen from both ends: one claim, for the pool further from its own
    # rating (rapid +0.25 vs blitz -0.15). The old code also called blitz "your weakest time control".
    rapid = games_by(30, 10, time_class="rapid", time_control="600", base_seconds=600)
    blitz = games_by(14, 26)
    mr = run(rapid + blitz)
    claims = [i for i in mr.insights if "time-control" in i.id]
    assert [i.id for i in claims] == ["results.strength.time-control-rapid"]
    best = claims[0]
    assert best.title == "You outperform your rapid rating more than your other ratings" and best.category == "results"
    assert "75%" in best.detail and "35%" in best.detail and "lags behind" in best.detail
    assert set(best.example_games) <= {g.url for g in rapid if g.outcome == "win"}
    assert mr.stats["time_control_comparison"]["best"] == "Rapid"
    assert mr.stats["time_control_comparison"]["claimed"] == "Rapid"
    # the weaker side of the same comparison, when it is the pool that is off its rating
    blitz_only_off = games_by(20, 20, time_class="rapid", time_control="600", base_seconds=600) + games_by(8, 32)
    worst = [i for i in run(blitz_only_off).insights if "time-control" in i.id]
    assert [i.id for i in worst] == ["results.weakness.time-control-blitz"]
    assert any("increment" in s for s in worst[0].study)
    bar = next(c for c in mr.charts if c.title == "Score vs rating by time control")
    assert bar.labels == ["Blitz", "Rapid"] and bar.series[0].values == pytest.approx([14 / 40 - 0.5, 0.25])


def test_single_time_control_or_small_samples_make_no_comparison():
    assert not [i for i in run(games_by(30, 10)).insights if "time-control" in i.id]
    small = games_by(15, 5, time_class="rapid") + games_by(5, 15)  # 20 < 25 games each
    assert not [i for i in run(small).insights if "time-control" in i.id]


def test_rating_trend_rising_is_an_observation():
    # A rating is a random walk: a rising line is reported, but it is not evidence of a strength
    # (the old expectation, kind == "strength", fired on null data with no real effect).
    games = []
    for i in range(30):  # 30 games over the last 60 days, rating climbing 3 points a game, mostly wins
        rating = 1500 + 3 * i
        games.append(
            make_game(outcome="win" if i % 4 else "loss", my_rating=rating, opp_rating=rating, end_time=at(40 + 2 * i))
        )
    mr = run(games)
    trend = next(i for i in mr.insights if "trend-blitz" in i.id)
    assert trend.kind == "observation" and trend.id == "results.observation.trend-blitz"
    # the title gives the actual change, first to last; the trend line (slope 1.5/day x 90 days) is in the detail
    assert trend.title == "Your blitz rating went from 1500 to 1587 in the last 90 days (+87)"
    assert "a straight line through all those games says +135" in trend.detail
    assert mr.stats["trends"]["Blitz"]["change"] == pytest.approx(135.0)
    assert "Keep doing what works" not in " ".join(trend.study)


def test_rating_trend_falling_and_stable_are_observations():
    falling = [make_game(outcome="loss", my_rating=1600 - 2 * i, opp_rating=1600, end_time=at(i)) for i in range(20)]
    ins = next(i for i in run(falling).insights if "trend" in i.id)
    assert ins.kind == "observation" and ins.title.endswith("(−38)")
    stable = [
        make_game(outcome="win" if i % 2 else "loss", my_rating=1500 + (i % 2), end_time=at(i)) for i in range(20)
    ]
    ins = next(i for i in run(stable).insights if "trend" in i.id)
    assert ins.kind == "observation" and ins.title.endswith("(+1)") and "plateau" in ins.study[0]
    old = [make_game(my_rating=1500, end_time=at(i)) for i in range(20)] + [make_game(end_time=at(400))]
    assert not [i for i in run(old).insights if "trend" in i.id]  # nothing in the last 90 days


# --------------------------------------------------------------------------- colour
def test_black_weakness_detected_and_named():
    white = games_by(28, 12, color="white")
    black = games_by(12, 28, color="black")
    mr = run(white + black)
    ins = next(i for i in mr.insights if i.category == "color")
    assert ins.id == "results.weakness.colour-black" and ins.kind == "weakness"
    assert "black" in ins.title.lower()
    assert all("Black" in s or "1.e4" in s for s in ins.study)
    recent_black_losses = sorted((g for g in black if g.outcome == "loss"), key=lambda g: g.end_time, reverse=True)
    assert ins.example_games == [g.url for g in recent_black_losses[:5]]
    table = next(t for t in mr.tables if t.title == "By colour")
    assert [row[:3] for row in table.rows] == [["White", 40, "28/0/12"], ["Black", 40, "12/0/28"]]


def test_colour_gap_from_one_opening_is_not_a_second_claim():
    # Black is weak only in the Caro-Kann (15/150), which the openings module names; without those games
    # White and Black are normal. The colour finding points to the opening instead of repeating it.
    def fam(color, family, wins, losses):
        return games_by(wins, losses, color=color, opening_family=family, opening=family)

    black = fam("black", "Caro-Kann Defense", 15, 135) + fam("black", "French Defense", 50, 50)
    black += fam("black", "King's Indian Defense", 50, 50)
    white = fam("white", "Italian Game", 100, 100) + fam("white", "London System", 100, 100)
    mr = run(black + white)
    ins = next(i for i in mr.insights if i.category == "color")
    assert ins.kind == "observation" and ins.id == "results.observation.colour-black"
    assert "Caro-Kann Defense" in ins.title and ins.evidence["explained_by"] == ["black:Caro-Kann Defense"]
    # the same gap spread over every Black opening is a colour claim
    spread = fam("black", "Caro-Kann Defense", 38, 78) + fam("black", "French Defense", 38, 78)
    spread += fam("black", "King's Indian Defense", 38, 78)
    ins = next(i for i in run(spread + white).insights if i.category == "color")
    assert ins.kind == "weakness" and ins.id == "results.weakness.colour-black"


def test_white_weakness_detected():
    mr = run(games_by(12, 28, color="white") + games_by(28, 12, color="black"))
    assert "results.weakness.colour-white" in ids(mr)


def test_normal_white_edge_and_small_samples_are_not_flagged():
    # White +0.02, Black -0.02 per game: exactly the usual colour gap, even over 800 games
    mr = run(games_by(208, 192, color="white") + games_by(192, 208, color="black"))
    assert not [i for i in mr.insights if i.category == "color"]
    assert mr.stats["colour_gap"]["excess"] == pytest.approx(0.0, abs=1e-9)
    few = run(games_by(15, 0, color="white") + games_by(0, 15, color="black"))
    assert not [i for i in few.insights if i.category == "color"]


# --------------------------------------------------------------------------- opponents
def test_drop_points_against_lower_rated_players():
    weaker = games_by(12, 18, my_rating=1500, opp_rating=1350)
    stronger = games_by(20, 10, my_rating=1500, opp_rating=1650)
    mr = run(weaker + stronger)
    weak = next(i for i in mr.insights if i.id == "results.weakness.lower-rated-opponents")
    assert weak.category == "opponents" and weak.title == "You drop points against lower-rated players"
    assert "40%" in weak.detail and "30 games" in weak.detail
    strong = next(i for i in mr.insights if i.id == "results.strength.higher-rated-opponents")
    assert strong.title == "You punch above your weight against stronger players"
    assert strong.evidence["expected_kind"] == "attenuated"
    assert strong.evidence["expected"] == pytest.approx(attenuated_expected(expected_score(1500, 1650)))


def test_opponent_groups_need_enough_games():
    mr = run(games_by(2, 8, my_rating=1500, opp_rating=1300) + games_by(30, 30))
    assert not [i for i in mr.insights if i.category == "opponents"]


# --------------------------------------------------------------------------- chess.com accuracy
def test_accuracy_observation_needs_twenty_reviewed_games():
    games = [make_game(outcome="win", my_accuracy=85.0, opp_accuracy=70.0) for _ in range(12)]
    games += [make_game(outcome="loss", my_accuracy=65.0, opp_accuracy=80.0) for _ in range(8)]
    mr = run(games)
    ins = next(i for i in mr.insights if i.id == "results.observation.chesscom-accuracy")
    assert ins.kind == "observation" and "85.0 in wins, 65.0 in losses" in ins.title
    assert ins.example_games and set(ins.example_games) <= {g.url for g in games if g.outcome == "loss"}
    assert "chess.com Game Review accuracy" in ins.title
    table = next(t for t in mr.tables if t.title.startswith("chess.com Game Review accuracy"))
    assert table.rows == [["Blitz", 20, pytest.approx(77.0), 85.0, 65.0, pytest.approx(74.0)]]
    fewer = run(games[1:])
    assert "results.observation.chesscom-accuracy" not in ids(fewer)
    assert any(t.title.startswith("chess.com Game Review accuracy") for t in fewer.tables)


# --------------------------------------------------------------------------- options & real data
def test_thresholds_can_be_overridden_through_options():
    ctx = AnalysisContext("tester", [], options={"results.min_effect": 0.1, "results.min_color_games": 5})
    th = results.Thresholds.from_ctx(ctx)
    assert (th.min_effect, th.min_color_games, th.min_tc_games) == (0.1, 5, 25)


def test_real_chesscom_games_every_player(fixtures_dir):
    raws = json.loads((fixtures_dir / "real_chesscom_games.json").read_text())["games"]
    users = {raw[side]["username"] for raw in raws for side in ("white", "black") if raw.get(side, {}).get("username")}
    for user in sorted(users):
        games = parse.parse_games(raws, user)  # unfiltered: variants, chess960, daily, casual all included
        mr = run(games)
        assert mr.stats["n"] == len(games)


# --------------------------------------------------------------------------- calibration: claims that used to fire on noise
def test_opponent_claim_must_hold_for_the_plain_elo_formula_too():
    # 85% against opponents 300 points weaker: exactly what the plain Elo formula predicts (85%), but
    # well above the pessimistic attenuated expectation (76%). Rating noise alone can explain that, so
    # no "you reliably beat lower-rated players" (the old attenuated-only test claimed it).
    games = games_by(850, 150, my_rating=1500, opp_rating=1200) + games_by(400, 400)
    mr = run(games)
    lower = mr.stats["opponent_groups"]["lower"]
    assert lower["gap"] > 0.07 and lower["p_adjusted"] < 0.001  # significant against the attenuated expectation
    assert abs(lower["elo_gap"]) < 0.01  # ... but not against plain Elo
    assert not [i for i in mr.insights if i.category == "opponents"]


def test_underrated_player_has_no_opponent_strength_claims():
    # 150 Elo stronger than the rating says, against every kind of opponent: the old test (each group
    # against its expectation) said "you reliably beat lower-rated players"; compared with the
    # player's other games, nothing about opponent strength stands out.
    games = games_by(170, 30, my_rating=1500, opp_rating=1350)  # true 85%, rating says 70%
    games += games_by(140, 60, my_rating=1500, opp_rating=1500)  # true 70%, rating says 50%
    games += games_by(100, 100, my_rating=1500, opp_rating=1650)  # true 50%, rating says 30%
    mr = run(games)
    assert mr.stats["opponent_groups"]["lower"]["delta"] > 0.15  # far above its expectation ...
    assert not [i for i in mr.insights if i.category == "opponents"]  # ... like every other game


def test_time_controls_are_compared_one_against_the_rest_not_best_against_worst():
    # Three pools, each near its own expectation; rapid looks a bit better. The old best-vs-worst
    # comparison (selection-biased) called rapid the best and blitz the weakest time control.
    rapid = games_by(16, 9, time_class="rapid", time_control="600", base_seconds=600)
    blitz = games_by(12, 13)
    bullet = games_by(13, 12, time_class="bullet", time_control="60", base_seconds=60)
    mr = run(rapid + blitz + bullet)
    assert not [i for i in mr.insights if "time-control" in i.id]
    tests = mr.stats["time_control_comparison"]["tests"]
    assert set(tests) == {"Rapid", "Blitz", "Bullet"} and all(t["p_adjusted"] >= t["p_value"] - 1e-12 for t in tests.values())
    # a real gap against the other two still counts
    strong = games_by(34, 6, time_class="rapid", time_control="600", base_seconds=600)
    mr = run(strong + games_by(20, 20) + bullet)
    ins = next(i for i in mr.insights if i.id == "results.strength.time-control-rapid")
    assert ins.kind == "strength" and "other time controls" in ins.detail
