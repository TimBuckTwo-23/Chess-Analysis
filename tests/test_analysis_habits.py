"""analysis/habits.py: sessions, tilt after losses, rematches, fatigue, streaks, time of day."""

import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from chess_insights.analysis import habits
from chess_insights.context import AnalysisContext
from chess_insights.models import CATEGORIES, VALUE_FORMATS, ModuleResult
from factories import make_game, all_tables

T0 = datetime(2024, 3, 4, 18, 0, tzinfo=timezone.utc)  # a Monday


# --------------------------------------------------------------------------- helpers
def assert_consistent(mr: ModuleResult) -> None:
    for chart in mr.charts:
        assert chart.value_format in VALUE_FORMATS
        assert chart.series and all(len(s.values) == len(chart.labels) for s in chart.series), chart.title
    for table in mr.tables:
        assert table.formats is None or len(table.formats) == len(table.columns)
        assert table.formats is None or set(table.formats) <= set(VALUE_FORMATS)
        assert all(len(row) == len(table.columns) for row in table.rows), table.title
    assert all(k.format in VALUE_FORMATS for k in mr.kpis)
    for ins in mr.insights:
        assert ins.id.startswith(f"{mr.key}.{ins.kind}.") and ins.category in CATEGORIES
        assert 0.0 <= ins.severity <= 1.0 and 0.0 <= ins.confidence <= 1.0
        assert 2 <= len(ins.study) <= 4 and ins.title and ins.detail
        assert len(ins.example_games) <= 5 and all(u.startswith("https://") for u in ins.example_games)
    json.dumps(mr.stats, default=str, allow_nan=False)


def run(games, **options) -> ModuleResult:
    mr = habits.analyze(AnalysisContext("tester", sorted(games, key=lambda g: g.end_time), options=options))
    assert (mr.key, mr.title) == ("habits", "Habits & tilt")
    assert_consistent(mr)
    return mr


def game_at(start: datetime, minutes: float = 8.0, **kw):
    return make_game(start_time=start, end_time=start + timedelta(minutes=minutes), **kw)


def sequence(start: datetime, outcomes: list[str], gap_min: float = 2.0, minutes: float = 8.0, **kw) -> list:
    """Back-to-back games: each starts ``gap_min`` minutes after the previous one ended."""
    games, t = [], start
    for o in outcomes:
        games.append(game_at(t, minutes, outcome=o, **kw))
        t += timedelta(minutes=minutes + gap_min)
    return games


def insight(mr, id_):
    return next((i for i in mr.insights if i.id == id_), None)


def claims(mr):
    return [i.id for i in mr.insights if i.kind != "observation"]


def table(mr, title):
    return next(t for t in all_tables(mr) if t.title.startswith(title))


def rows_by_first(t):
    return {r[0]: dict(zip(t.columns, r)) for r in t.rows}


# --------------------------------------------------------------------------- timeline
def test_session_split_at_the_30_minute_boundary():
    a = game_at(T0)
    b = game_at(a.end_time + timedelta(minutes=30))  # exactly 30 min: same session
    c = game_at(b.end_time + timedelta(minutes=30, seconds=1))  # just over: new session
    tl = habits.build_timeline([c, a, b])
    assert [t.game for t in tl] == [a, b, c]
    assert [(t.session, t.position) for t in tl] == [(0, 1), (0, 2), (1, 1)]
    assert tl[1].gap == 1800.0 and tl[0].gap is None
    assert [t.situation for t in tl] == ["break", "short_break", "break"]


def test_after_loss_classification_at_the_15_minute_boundary():
    loss = game_at(T0, outcome="loss")
    at_15 = game_at(loss.end_time + timedelta(minutes=15), outcome="win")
    after_16 = game_at(at_15.end_time + timedelta(minutes=15, seconds=1), outcome="draw")
    quick = game_at(after_16.end_time + timedelta(minutes=1), outcome="loss")
    tl = habits.build_timeline([loss, at_15, after_16, quick])
    assert [t.situation for t in tl] == ["break", "after_loss", "short_break", "after_draw"]
    assert habits.build_timeline([loss, at_15], tilt_window_min=10)[1].situation == "short_break"


def test_daily_games_are_left_out_and_losing_runs_reset_per_session():
    daily = make_game(
        time_class="daily", time_control="1/86400", base_seconds=86400, end_time=T0 + timedelta(minutes=3)
    )
    first = sequence(T0, ["loss", "loss", "loss", "win"])
    later = sequence(T0 + timedelta(hours=5), ["loss", "win"])
    tl = habits.build_timeline(first + later + [daily])
    assert daily not in [t.game for t in tl]
    assert [t.loss_run for t in tl] == [0, 1, 2, 3, 0, 1]
    assert [len(habits.longest_losing_streak(tl))] == [3]


def test_start_time_estimated_from_clocks_or_time_control():
    end = T0 + timedelta(hours=1)
    with_clocks = make_game(start_time=None, end_time=end)  # factory clocks: 5 s per ply, 20 plies
    assert habits.game_start(with_clocks) == end - timedelta(seconds=100)
    no_clocks = make_game(start_time=None, end_time=end, with_clocks=False)  # 300 s base, 20 plies
    assert habits.game_start(no_clocks) == end - timedelta(seconds=2 * 300 * habits.CLOCK_USAGE * 20 / 80)
    later_than_end = make_game(start_time=end + timedelta(minutes=5), end_time=end, with_clocks=False)
    assert habits.game_start(later_than_end) < end
    naive = make_game(start_time=datetime(2024, 3, 4, 17, 0), end_time=datetime(2024, 3, 4, 17, 8))
    assert habits.game_start(naive).tzinfo is not None


def test_time_zone_bucketing():
    ny, name = habits.resolve_tz("America/New_York")
    assert name == "America/New_York"
    moment = datetime(2024, 6, 1, 3, 30, tzinfo=timezone.utc)  # 23:30 on Friday 31 May in New York (UTC-4)
    assert habits.day_part(moment, ny) == 5 and habits.day_part(moment, timezone.utc) == 0
    winter = datetime(2024, 1, 15, 4, 30, tzinfo=timezone.utc)  # 23:30 EST (UTC-5)
    assert habits.day_part(winter, ny) == 5
    assert habits.resolve_tz("Not/AZone") == (timezone.utc, "UTC")
    assert habits.resolve_tz("") == (timezone.utc, "UTC") and habits.resolve_tz(None)[1] == "UTC"

    games = [game_at(moment + timedelta(days=7 * i)) for i in range(3)]  # all Saturdays 03:30 UTC
    mr = run(games, tz="America/New_York")
    assert rows_by_first(table(mr, "By time of day"))["20:00–24:00"]["Games"] == 3
    assert rows_by_first(table(mr, "By weekday"))["Friday"]["Games"] == 3
    utc = run(games, tz="Mars/Olympus")
    assert rows_by_first(table(utc, "By time of day"))["00:00–04:00"]["Games"] == 3
    assert utc.stats["time_zone"] == "UTC"


# --------------------------------------------------------------------------- module: empty / small input
def test_empty_and_daily_only_input():
    assert run([]).summary.lower().startswith("not enough data")
    daily = [make_game(time_class="daily", time_control="1/86400", base_seconds=86400) for _ in range(30)]
    mr = run(daily)
    assert mr.summary.lower().startswith("not enough data") and mr.insights == []


def test_small_input_has_tables_but_no_claims():
    games = sequence(T0, ["loss", "win", "loss", "loss", "draw"])
    games.append(game_at(T0 + timedelta(days=1), outcome="win", moves_san=["e4"]))  # 1 ply: timeline only
    mr = run(games)
    assert "not enough data" in mr.summary.lower() and mr.insights == []
    assert mr.stats["n"] == 6 and mr.stats["n_scored"] == 5 and mr.stats["sessions"] == 2
    sizes = rows_by_first(table(mr, "Session lengths"))
    assert sizes["1"]["Sessions"] == 1 and sizes["4–6"]["Games"] == 5
    assert mr.stats["longest_losing_streak"] == 2


# --------------------------------------------------------------------------- module: planted effects
def tilt_games(after_loss_losses: int, rng: random.Random) -> list:
    """120 two-game sessions: the first game alternates W/L; the second starts 5 min after it."""
    games = []
    after_loss_outcomes = ["loss"] * after_loss_losses + ["win"] * (60 - after_loss_losses)
    rng.shuffle(after_loss_outcomes)
    after_win_outcomes = ["loss", "win"] * 30
    for i in range(120):
        start = T0 + timedelta(hours=7 * i)
        first = "loss" if i % 2 else "win"
        second = after_loss_outcomes.pop() if first == "loss" else after_win_outcomes.pop()
        games += sequence(start, [first, second], gap_min=5)
    return games


def test_playing_worse_after_a_loss_is_found():
    mr = run(tilt_games(45, random.Random(1)))
    ins = insight(mr, "habits.weakness.after-a-loss")
    assert ins and ins.title == "You play worse right after a loss" and ins.category == "habits"
    assert "In 60 games started within 15 minutes of a loss you scored 25%" in ins.detail
    assert any("stop rule" in s.lower() for s in ins.study)
    rows = rows_by_first(table(mr, "After a loss, a win or a break"))
    assert rows["Within 15 min of a loss"]["Games"] == 60 and rows["Within 15 min of a win"]["Games"] == 60
    assert rows["After a break (> 30 min) or first game"]["Games"] == 120
    assert mr.stats["sessions"] == 120 and "Key finding" in mr.summary


def test_no_tilt_when_results_after_losses_are_normal():
    mr = run(tilt_games(30, random.Random(2)))
    assert insight(mr, "habits.weakness.after-a-loss") is None


def test_rematch_observation():
    games = []
    for i in range(6):
        start = T0 + timedelta(days=i)
        games += [
            game_at(start, outcome="loss", opponent="Rival"),
            game_at(start + timedelta(minutes=10), outcome="win" if i % 2 else "loss", opponent="rival"),
        ]
    games += sequence(T0 + timedelta(days=10), ["win", "loss", "draw"] * 5)
    mr = run(games)
    ins = insight(mr, "habits.observation.rematches")
    assert ins and ins.kind == "observation" and "after 6 losses" in ins.detail and "3/0/3" in ins.detail
    assert mr.stats["rematches"]["n"] == 6


def test_fatigue_buckets_and_late_session_weakness():
    games = []
    for s in range(15):
        outcomes = ["win", "loss", "win", "loss", "loss", "loss", "loss", "win" if s % 3 == 0 else "loss"]
        games += sequence(T0 + timedelta(days=s), outcomes, gap_min=20)  # 20 min gaps: no tilt window
    mr = run(games)
    rows = rows_by_first(table(mr, "By game number in a session"))
    assert [rows[k]["Games"] for k in ("1st game", "2nd–3rd", "4th–6th", "7th+")] == [15, 30, 45, 30]
    assert rows["1st game"]["Score"] == 1.0 and rows["4th–6th"]["Score"] == pytest.approx(0.0)
    ins = insight(mr, "habits.weakness.long-sessions")
    assert ins and "Cap your sessions at 3 games" in ins.study[0]
    chart = next(c for c in mr.charts if "game number" in c.title)
    assert chart.labels == ["1st game", "2nd–3rd", "4th–6th", "7th+"]


def test_session_length_is_not_a_second_tilt_finding():
    # Games straight after a loss are worse (tilt) and, since a session's first game never follows a
    # loss, later games follow losses more often. Leaving the after-loss games out, later games are as
    # good as earlier ones: no separate "long sessions" finding.
    rng = random.Random(3)
    games = []
    for s in range(60):
        outcomes, prev = [], None
        for _ in range(8):
            prev = "loss" if rng.random() < (0.8 if prev == "loss" else 0.4) else "win"
            outcomes.append(prev)
        games += sequence(T0 + timedelta(days=s), outcomes, gap_min=3)
    mr = run(games)
    assert claims(mr) == ["habits.weakness.after-a-loss"]
    split = mr.stats["fatigue_tests"]["from_game_4"]
    assert split["p_adjusted"] < 0.001 and abs(split["without_after_loss_gap"]) < 0.06
    scored = [t for t in habits.build_timeline(games) if t.scored]
    assert habits.fatigue_insight(scored, habits.Thresholds())[0] is not None  # what it would say on its own


def test_losing_streak_observation():
    outcomes = ["win", "loss", "loss", "loss", "loss", "draw", "loss", "loss", "win", "loss"]
    games = sequence(T0, outcomes) + sequence(T0 + timedelta(days=1), ["win", "draw"] * 6)
    mr = run(games)
    ins = insight(mr, "habits.observation.losing-streak")
    assert ins and ins.title == "Your longest losing streak was 4 games" and "on 2024-03-04" in ins.detail
    assert len(ins.example_games) == 4
    # after 2+ losses in a row (same session): the 3rd and 4th loss, the draw, and the win after "loss, loss"
    assert mr.stats["after_two_losses"]["n"] == 4 and mr.stats["after_two_losses"]["score"] == pytest.approx(0.375)
    # below expectation after two losses: the stop rule is backed by the player's own games
    assert ins.study[0].startswith("Stop after two losses in a row")


def test_streak_advice_follows_the_players_own_results():
    from chess_insights.stats import summarize

    fine = summarize([make_game(outcome=o) for o in ("win", "win", "draw", "loss", "win")])  # above expectation
    assert "Stop after two losses" not in habits._streak_advice(fine)
    assert "about as well as your rating predicts" in habits._streak_advice(fine)
    poor = summarize([make_game(outcome=o) for o in ("loss", "loss", "draw", "loss")])
    assert habits._streak_advice(poor).startswith("Stop after two losses in a row")


def test_late_night_finding_answers_the_blunder_question_from_engine_data():
    from factories import make_game_eval, make_ply_eval
    from chess_insights.models import Insight

    late = [make_game(game_id=f"l{i}") for i in range(12)]
    other = [make_game(game_id=f"o{i}") for i in range(12)]

    def ev(g, blunders):
        plies = [make_ply_eval(ply=2 * k, judgement="blunder" if k < blunders else None) for k in range(10)]
        return make_game_eval(g.game_id, plies)

    evals = {g.game_id: ev(g, 2) for g in late} | {g.game_id: ev(g, 1) for g in other}
    ins = Insight("habits.weakness.late-night", "weakness", "habits", "t", "Detail.", 0.5, 0.9,
                  study=["Avoid.", "Replay your last 5 late-night losses and check whether they came from careless "
                         "blunders or the clock, both signs of tiredness.", "Slower."])
    habits.add_blunder_rates(ins, late, late + other, evals)
    assert "you blundered 20.0 times per 100 moves in them, 10.0 in 12 analysed games at other times" in ins.detail
    assert "you blunder more then (20.0 vs 10.0 per 100 moves)" in ins.study[1]
    unchanged = Insight("x", "weakness", "habits", "t", "Detail.", 0.5, 0.9, study=["a", "b"])
    habits.add_blunder_rates(unchanged, late[:3], late + other, evals)  # too few analysed games: no numbers
    assert unchanged.detail == "Detail."


def test_late_night_block_is_found_in_the_players_time_zone():
    rng = random.Random(5)
    games = []
    for d in range(60):
        day = T0 + timedelta(days=d)
        games.append(game_at(day.replace(hour=5, minute=30), outcome="loss" if rng.random() < 0.8 else "win"))
        for hour in (14, 18, 22):  # UTC
            games.append(game_at(day.replace(hour=hour), outcome=rng.choice(["win", "loss"])))
    utc = run(games)  # no time zone given: the blocks are UTC hours, not named times of day
    ins = insight(utc, "habits.weakness.time-of-day-04-08")
    assert ins and ins.title == "You score below your rating in games started 04:00–08:00 UTC"
    ny = run(games, tz="America/New_York")  # 05:30 UTC = 00:30 / 01:30 in New York
    ins = insight(ny, "habits.weakness.time-of-day-00-04")
    assert ins and ins.title == "You score below your rating in late-night games (00:00–04:00)"
    assert "(America/New_York)" in ins.detail and len(claims(ny)) == 1


def noise_games(seed: int, n_games: int = 300) -> list:
    """Sessions at random times; each result drawn from the Elo expectation, independent of context."""
    rng = random.Random(seed)
    games, t = [], T0
    while len(games) < n_games:
        t += timedelta(hours=rng.uniform(3, 30))
        cur = t
        for _ in range(rng.randint(1, 7)):
            opp = 1500 + rng.randint(-200, 200)
            e = 1 / (1 + 10 ** ((opp - 1500) / 400))
            u = rng.random()
            outcome = "win" if u < e - 0.03 else ("draw" if u < e + 0.03 else "loss")
            game = game_at(cur, rng.uniform(4, 12), outcome=outcome, opp_rating=opp)
            games.append(game)
            cur = game.end_time + timedelta(minutes=rng.choice([rng.uniform(0.2, 14), rng.uniform(16, 29), 45]))
    return games


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_noise_produces_no_claims(seed):
    mr = run(noise_games(seed), tz="Europe/Berlin")
    assert claims(mr) == []
    assert len(mr.charts) == 3 and all(len(c.series[0].values) == len(c.labels) for c in mr.charts)


def test_noise_false_claim_rate_matches_the_significance_bar():
    # Every claim needs an adjusted p <= 0.05; with three kinds of test, a handful of runs in 60 may show one.
    runs_with_claims = sum(1 for seed in range(60) if claims(run(noise_games(100 + seed))))
    assert runs_with_claims <= 8


def test_late_night_window_spanning_two_blocks_is_one_finding():
    # Bad results at 23:30 and 01:30 UTC: the 23:00-03:00 window catches both, while each 4-hour block
    # (20-24, 00-04) only holds half of them. One late-night finding, not a block claim as well.
    rng = random.Random(8)
    games = []
    for d in range(60):
        day = T0 + timedelta(days=d)
        for hour in (14, 18, 20):
            games.append(game_at(day.replace(hour=hour), outcome=rng.choice(["win", "loss"])))
        games.append(game_at(day.replace(hour=23, minute=30), outcome="loss" if rng.random() < 0.72 else "win"))
        games.append(game_at(day.replace(hour=1, minute=30) + timedelta(days=1), outcome="loss" if rng.random() < 0.72 else "win"))
    mr = run(games, tz="Etc/UTC")  # the player lives in UTC
    ins = insight(mr, "habits.weakness.late-night")
    assert ins and ins.title == "You score below your rating in late-night games (23:00–03:00)"
    assert "In 120 games started between 23:00 and 03:00 (UTC)" in ins.detail  # Etc/UTC reads as UTC
    assert claims(mr) == ["habits.weakness.late-night"]
    assert mr.stats["late_night"]["n"] == 120 and mr.stats["late_night"]["local_time"]
    # without a time zone (the command line's default "UTC") 23:00-03:00 UTC is just another window
    # of the day for most players: tested at the strict level and titled by its UTC hours
    default = run(games, tz="UTC")
    ins = insight(default, "habits.weakness.late-night")
    assert ins and ins.title == "You score below your rating in games started 23:00–03:00 UTC"
    assert not default.stats["late_night"]["local_time"] and not default.stats["local_time"]
    # in a time zone where those games fall in the afternoon there is no late-night finding
    assert insight(run(games, tz="Asia/Tokyo"), "habits.weakness.late-night") is None
