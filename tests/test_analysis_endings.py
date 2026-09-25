"""analysis/endings.py: result mix, loss profile, abandoned games, game length and draw rates."""

import json
import random

import pytest

from chess_insights.analysis import endings
from chess_insights.context import AnalysisContext
from chess_insights.models import CATEGORIES, VALUE_FORMATS, ModuleResult
from factories import make_game, all_tables

C960_FEN = "bnrqkrnb/pppppppp/8/8/8/8/PPPPPPPP/BNRQKRNB w KQkq - 0 1"


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
    mr = endings.analyze(AnalysisContext("tester", sorted(games, key=lambda g: g.end_time), options=options))
    assert (mr.key, mr.title) == ("endings", "How your games end")
    assert_consistent(mr)
    return mr


def moves(n_full: int) -> list[str]:
    """A move list of exactly ``n_full`` full moves (2 * n_full plies)."""
    return (["Nf3", "Nf6", "Ng1", "Ng8"] * n_full)[: 2 * n_full]


LOSER_CODES = {"checkmate": "checkmated", "timeout": "timeout", "abandoned": "abandoned"}


def g(outcome: str, termination: str, **kw):
    """A game with consistent result codes for ``outcome`` and ``termination``."""
    loser = LOSER_CODES.get(termination, "resigned")
    codes = {
        "win": {"my_result_code": "win", "opp_result_code": loser},
        "loss": {"my_result_code": loser, "opp_result_code": "win"},
        "draw": {"my_result_code": "agreed", "opp_result_code": "agreed"},
    }[outcome]
    return make_game(outcome=outcome, termination=termination, **{**codes, **kw})


def insight(mr, id_):
    return next((i for i in mr.insights if i.id == id_), None)


def claims(mr):
    return [i.id for i in mr.insights if i.kind != "observation"]


def table(mr, title):
    return next(t for t in all_tables(mr) if t.title == title)


def rows_by_first(t):
    return {r[0]: dict(zip(t.columns, r)) for r in t.rows}


# --------------------------------------------------------------------------- classification
def test_length_buckets_at_boundaries():
    cases = {0: "short", 1: "short", 20: "short", 21: "early", 30: "early", 31: "middle", 45: "middle"}
    cases.update({46: "long", 60: "long", 61: "very_long", 200: "very_long"})
    for full_moves, key in cases.items():
        assert endings.length_bucket(full_moves) == key, full_moves
    # a 41-ply game has 21 full moves
    assert endings.length_bucket(make_game(moves_san=moves(21)[:41]).full_moves) == "early"
    assert endings.length_bucket(make_game(moves_san=moves(20)).full_moves) == "short"


def test_mix_keys_and_methods():
    assert endings.mix_key(g("win", "checkmate")) == "win_checkmate"
    assert endings.mix_key(g("loss", "abandoned")) == "loss_other"
    assert endings.mix_key(g("loss", "variant")) == "loss_other"
    assert endings.mix_key(g("draw", "repetition")) == "draw"
    assert endings.method(g("loss", "abandoned")) == "abandoned"
    assert endings.method(g("win", "other")) == "other"
    assert endings.draw_type(g("draw", "fifty_move")) == "fifty_move"
    assert endings.draw_type(g("draw", "weird")) == "other"


# --------------------------------------------------------------------------- empty / small input
def test_empty_input():
    mr = run([])
    assert mr.summary.lower().startswith("not enough data") and mr.insights == []


def test_tiny_and_daily_only_input_has_tables_but_no_claims():
    mr = run([g("win", "resignation", time_class="daily", time_control="1/86400", base_seconds=86400)])
    assert "not enough data" in mr.summary.lower() and "only game was a win" in mr.summary
    mix = table(mr, "Result mix by time control")
    assert mr.insights == [] and mix.columns == ["Result", "Daily (1 game)"] and mix.rows == [["Won by resignation", 1.0]]


def test_zero_move_games_and_missing_ratings_do_not_break_anything():
    games = [g("loss", "abandoned", moves_san=[], my_rating=None, opp_rating=None) for _ in range(5)]
    games += [g("win", "resignation", my_rating=None, opp_rating=None) for _ in range(10)]
    mr = run(games)
    assert mr.stats["n_played"] == 10 and mr.stats["abandoned"]["abandoned"] == 5
    length = rows_by_first(table(mr, "Score by game length"))
    assert length["≤ 20"]["Games"] == 10 and length["≤ 20"]["vs rating"] is None
    assert all(len(c.labels) for c in mr.charts) and not any(c.title.startswith("Score vs") for c in mr.charts)
    assert "5 abandoned or very short game(s)" in table(mr, "Score by game length").note


# --------------------------------------------------------------------------- result mix / profile
def test_result_mix_table_and_stacked_chart():
    blitz = [g("win", "checkmate")] * 2 + [g("win", "timeout"), g("draw", "repetition"), g("loss", "resignation")]
    rapid = [
        g(o, t, time_class="rapid", time_control="600", base_seconds=600)
        for o, t in [("loss", "checkmate"), ("loss", "abandoned"), ("win", "resignation")]
    ]
    mr = run(blitz + rapid)
    t = table(mr, "Result mix by time control")
    assert t.columns == ["Result", "Blitz (5 games)", "Rapid (3 games)", "All (8 games)"]
    assert len(t.formats) == len(t.columns) and all(len(r) == len(t.columns) for r in t.rows)
    rows = rows_by_first(t)
    assert rows["Won by checkmate"]["Blitz (5 games)"] == pytest.approx(0.4)
    assert rows["Drawn"]["Blitz (5 games)"] == pytest.approx(0.2)
    assert rows["Lost: abandoned / other"]["Rapid (3 games)"] == pytest.approx(1 / 3)
    assert "Won: abandoned / other" not in rows  # no game ended that way
    for col in range(1, len(t.columns)):
        assert sum(r[col] for r in t.rows) == pytest.approx(1.0)
    chart = next(c for c in mr.charts if c.kind == "stacked_bar")
    assert chart.labels == ["Blitz", "Rapid"] and len(chart.series) == len(endings.MIX)
    blitz_by_label = {s.name: s.values[0] for s in chart.series}
    assert all(blitz_by_label[label] == pytest.approx(row["Blitz (5 games)"]) for label, row in rows.items())


def test_loss_profile_counts_and_shares():
    games = [g("loss", "checkmate")] * 3 + [g("loss", "timeout")] + [g("win", "checkmate"), g("win", "resignation")]
    games.append(g("loss", "checkmate", moves_san=["f3", "e5", "g4"]))  # 3 plies: left out of skill metrics
    mr = run(games)
    rows = rows_by_first(table(mr, "How your losses and wins end"))
    assert rows["Checkmate"]["Losses"] == 3 and rows["Checkmate"]["Share of losses"] == pytest.approx(0.75)
    assert rows["Checkmate"]["Share of wins"] == pytest.approx(0.5) and rows["On time"]["Losses"] == 1
    assert "1 game(s) with fewer than 4 plies" in table(mr, "How your losses and wins end").note


def test_checkmated_more_than_mating_is_an_observation():
    # Never a weakness: a player who plays on in lost positions (a resignation becomes a mate) gets
    # exactly this pattern with no king-safety problem. The old weakness fired in every report of a
    # null world where the player rarely resigns.
    games = [g("loss", "checkmate") for _ in range(16)] + [g("loss", "resignation") for _ in range(14)]
    games += [g("win", "checkmate") for _ in range(3)] + [g("win", "resignation") for _ in range(27)]
    mr = run(games)
    ins = insight(mr, "endings.observation.checkmated-often")
    assert ins and ins.kind == "observation" and ins.category == "endings"
    assert ins.title == "53% of your losses end in checkmate, against 10% of your wins"
    assert "16 of your 30 losses" in ins.detail and "playing on in lost positions" in ins.detail
    assert set(ins.example_games) <= {x.url for x in games[:16]}
    assert any("mating-pattern" in s for s in ins.study)
    assert claims(mr) == []


def test_mate_observation_compares_shares_not_counts():
    # Mated 25 times in 60 losses, mating 30 times in 150 wins: more mates *given* than taken, so the old
    # title "You get checkmated much more often than you checkmate" was false in plain counts.
    games = [g("loss", "checkmate") for _ in range(25)] + [g("loss", "resignation") for _ in range(35)]
    games += [g("win", "checkmate") for _ in range(30)] + [g("win", "resignation") for _ in range(120)]
    ins = insight(run(games), "endings.observation.checkmated-often")
    assert ins.title == "42% of your losses end in checkmate, against 20% of your wins"


def test_similar_mate_shares_are_not_reported():
    games = [g("loss", "checkmate") for _ in range(8)] + [g("loss", "resignation") for _ in range(22)]
    games += [g("win", "checkmate") for _ in range(7)] + [g("win", "resignation") for _ in range(23)]
    assert insight(run(games), "endings.observation.checkmated-often") is None
    few = [g("loss", "checkmate") for _ in range(15)] + [g("win", "resignation") for _ in range(15)]
    assert insight(run(few), "endings.observation.checkmated-often") is None  # fewer than 20 losses


def test_abandoned_games_observation_and_timeouts_stay_descriptive():
    games = [g("loss", "abandoned", moves_san=[]), g("loss", "abandoned")]
    games += [g("loss", "timeout") for _ in range(20)] + [g("win", "resignation") for _ in range(20)]
    games.append(g("win", "abandoned"))
    mr = run(games)
    ins = insight(mr, "endings.observation.abandoned-games")
    assert ins and ins.kind == "observation" and "abandoned 2 of your 43 games" in ins.detail
    assert "opponents abandoned 1" in ins.detail and "1.0 points" in ins.detail  # two games expected at 50%
    assert not [i for i in mr.insights if "time" in i.id]  # timeouts belong to the clock module
    assert insight(run(games[1:]), "endings.observation.abandoned-games") is None  # a single abandonment


# --------------------------------------------------------------------------- game length
def test_length_table_rows():
    games = [g("win", "resignation", moves_san=moves(20)), g("loss", "resignation", moves_san=moves(21))]
    games += [g("draw", "agreement", moves_san=moves(61), opp_rating=1600)]
    mr = run(games)
    rows = rows_by_first(table(mr, "Score by game length"))
    assert [r[0] for r in table(mr, "Score by game length").rows] == ["≤ 20", "21–30", "31–45", "46–60", "61+"]
    assert rows["≤ 20"]["W/D/L"] == "1/0/0" and rows["21–30"]["W/D/L"] == "0/0/1"
    e = 1 / (1 + 10 ** (100 / 400))
    assert rows["61+"]["Rating predicts"] == pytest.approx(e) and rows["61+"]["vs rating"] == pytest.approx(0.5 - e)
    assert rows["31–45"]["Games"] == 0 and rows["31–45"]["Score"] is None
    chart = next(c for c in mr.charts if c.title.startswith("Score vs rating by game length"))
    assert chart.series[0].values[2] is None and chart.series[0].values[0] == pytest.approx(0.5)


def test_short_and_long_games_that_stand_out_are_observations():
    rng = random.Random(3)
    short = [g("loss" if i % 4 else "win", "resignation", moves_san=moves(12 + i % 8)) for i in range(40)]
    long_ = [g("win" if i % 4 else "loss", "resignation", moves_san=moves(47 + i % 10)) for i in range(40)]
    middle = [g(rng.choice(["win", "loss"]), "resignation", moves_san=moves(35)) for _ in range(30)]
    mr = run(short + long_ + middle)
    weak = insight(mr, "endings.observation.length-short")
    assert weak and weak.kind == "observation"
    assert weak.title == "Your games of ≤ 20 moves score below your games of other lengths"
    assert "25%" in weak.detail and "40 games" in weak.detail and "when you and your opponents resign" in weak.detail
    shortest = sorted((x for x in short if x.outcome == "loss"), key=lambda x: (x.plies, -x.end_time.timestamp()))
    assert weak.example_games == [x.url for x in shortest[:5]]
    strong = insight(mr, "endings.observation.length-long")
    assert strong and strong.category == "endings" and "score above" in strong.title
    assert mr.stats["by_length"]["≤ 20"]["p_adjusted"] is not None
    assert claims(mr) == []


def test_underrated_player_has_no_length_findings():
    # Every length band 20 points above its expectation (a rating that lags the player's strength):
    # the old per-band test called several bands strengths; compared with each other, none stands out.
    rng = random.Random(5)
    games = [
        g("win" if rng.random() < 0.7 else "loss", "resignation", moves_san=moves(n))
        for n in (15, 25, 38, 52, 70)
        for _ in range(60)
    ]
    mr = run(games)
    assert mr.stats["by_length"]["61+"]["delta"] > 0.1
    assert not [i for i in mr.insights if ".length-" in i.id]


def test_a_player_who_never_resigns_gets_no_endings_claims():
    # Results drawn from the Elo expectation; every loss is played on to mate in a long game while
    # opponents resign normally. The old tests reported "checkmated often", "you underperform in very
    # long games" and "strong in short games" as weaknesses and strengths.
    rng = random.Random(7)
    games = []
    for _ in range(300):
        if rng.random() < 0.5:
            games.append(g("win", "resignation", moves_san=moves(rng.randint(15, 45))))
        else:
            games.append(g("loss", "checkmate", moves_san=moves(rng.randint(40, 80))))
    mr = run(games)
    assert claims(mr) == []
    assert insight(mr, "endings.observation.checkmated-often") is not None


def noise_games(seed: int, n_games: int = 200) -> list:
    """Results drawn from the Elo expectation, independent of game length and of how games end."""
    rng = random.Random(seed)
    games = []
    for _ in range(n_games):
        opp = 1500 + rng.randint(-200, 200)
        e = 1 / (1 + 10 ** ((opp - 1500) / 400))
        outcome = "win" if rng.random() < e else "loss"
        how = rng.choice(["resignation", "checkmate"])
        games.append(g(outcome, how, moves_san=moves(rng.randint(8, 70)), opp_rating=opp))
    return games


def test_length_noise_produces_no_claims():
    assert claims(run(noise_games(11))) == []


def test_noise_false_claim_rate_matches_the_significance_bar():
    # Every claim needs an adjusted p <= 0.05; with two kinds of test, a handful of runs in 60 may show one.
    runs_with_claims = sum(1 for seed in range(60) if claims(run(noise_games(seed))))
    assert runs_with_claims <= 8


def test_length_is_not_a_finding_when_nearly_every_game_has_that_length():
    games = [g("win", "resignation", moves_san=moves(15)) for _ in range(40)]
    games += [g("loss", "resignation", moves_san=moves(50)) for _ in range(5)]
    mr = run(games)
    assert rows_by_first(table(mr, "Score by game length"))["≤ 20"]["vs rating"] == pytest.approx(0.5)
    assert not [i for i in mr.insights if ".length-" in i.id]


def test_chess960_games_are_counted_like_any_other():
    games = [g("win", "checkmate", rules="chess960", initial_fen=C960_FEN) for _ in range(12)]
    mr = run(games)
    assert rows_by_first(table(mr, "How your losses and wins end"))["Checkmate"]["Wins"] == 12


# --------------------------------------------------------------------------- draws
def test_draw_table_and_draw_rate_observation():
    games = [g("draw", t) for t in ["agreement", "repetition", "repetition", "stalemate", "timeout_vs_insufficient"]]
    games += [g("win", "resignation") for _ in range(10)] + [g("loss", "resignation") for _ in range(5)]
    games += [g("draw", "fifty_move", time_class="rapid", time_control="600", base_seconds=600)]
    mr = run(games)
    rows = rows_by_first(table(mr, "Draws by time control"))
    assert rows["Blitz"]["Draws"] == 5 and rows["Blitz"]["Draw rate"] == pytest.approx(0.25)
    assert rows["Blitz"]["Repetition"] == 2 and rows["Blitz"]["Flag vs insufficient material"] == 1
    assert rows["Rapid"]["50-move rule"] == 1
    ins = insight(mr, "endings.observation.draw-rate")
    assert ins and ins.kind == "observation" and "25% in blitz" in ins.detail and "rapid" not in ins.detail
    assert "repetition (2 of 6)" in ins.detail
