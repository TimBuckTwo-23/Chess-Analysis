"""analysis/openings.py: repertoire performance by first move, family and line, per colour."""

import json

import pytest

from chess_insights import parse
from chess_insights.analysis import openings
from chess_insights.context import AnalysisContext
from chess_insights.models import CATEGORIES, VALUE_FORMATS, ModuleResult
from chess_insights.stats import shrink
from factories import make_game

CARO = ["e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6", "Be2", "c5"]
FRENCH = ["e4", "e6", "d4", "d5", "Nc3", "Bb4", "e5", "c5"]
ITALIAN = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6"]
SICILIAN = ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"]
QGD = ["d4", "d5", "c4", "e6", "Nc3", "Nf6"]
LONDON = ["d4", "d5", "Bf4", "Nf6", "e3", "e6"]
C960_FEN = "bnrqkrnb/pppppppp/8/8/8/8/PPPPPPPP/BNRQKRNB w KQkq - 0 1"

OPENING_KW = {
    "caro": dict(
        color="black",
        opening_family="Caro-Kann Defense",
        opening="Caro-Kann Defense: Advance Variation",
        eco="B12",
        moves_san=CARO,
    ),
    "french": dict(
        color="black",
        opening_family="French Defense",
        opening="French Defense: Winawer Variation",
        eco="C15",
        moves_san=FRENCH,
    ),
    "qgd": dict(
        color="black",
        opening_family="Queen's Gambit Declined",
        opening="Queen's Gambit Declined",
        eco="D30",
        moves_san=QGD,
    ),
    "italian": dict(
        color="white",
        opening_family="Italian Game",
        opening="Italian Game: Giuoco Pianissimo",
        eco="C50",
        moves_san=ITALIAN,
    ),
    "london": dict(color="white", opening_family="London System", opening="London System", eco="D02", moves_san=LONDON),
    "sicilian_w": dict(
        color="white",
        opening_family="Sicilian Defense",
        opening="Sicilian Defense: Open",
        eco="B90",
        moves_san=SICILIAN,
    ),
}


def run(games, **options) -> ModuleResult:
    mr = openings.analyze(AnalysisContext("tester", sorted(games, key=lambda g: g.end_time), options=options))
    assert_consistent(mr)
    return mr


def assert_consistent(mr: ModuleResult) -> None:
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


def play(key: str, wins: int, losses: int, draws: int = 0, **kw) -> list:
    """Games in one opening; outcomes interleaved in time (stepping by 7 permutes lengths not divisible by 7)."""
    outcomes = ["win"] * wins + ["loss"] * losses + ["draw"] * draws
    if len(outcomes) % 7:
        outcomes = [outcomes[(i * 7) % len(outcomes)] for i in range(len(outcomes))]
    return [make_game(outcome=o, **{**OPENING_KW[key], **kw}) for o in outcomes]


def insight(mr: ModuleResult, id_: str):
    return next((i for i in mr.insights if i.id == id_), None)


def table(mr: ModuleResult, title: str):
    return next(t for t in mr.tables if t.title == title)


# --------------------------------------------------------------------------- empty / exclusions / missing data
def test_empty_input_says_not_enough_data():
    mr = run([])
    assert (mr.key, mr.title) == ("openings", "Openings")
    assert "not enough data" in mr.summary.lower() and mr.insights == []


def test_chess960_variants_and_tiny_games_are_excluded():
    standard = play("italian", 6, 4)
    c960 = [make_game(rules="chess960", initial_fen=C960_FEN, opening_family=None, opening=None) for _ in range(3)]
    custom = [make_game(initial_fen="8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")]  # set-up position in standard chess
    koth = [make_game(rules="kingofthehill")]
    tiny = [make_game(moves_san=["e4"]), make_game(moves_san=[], outcome="loss", termination="abandoned")]
    mr = run(standard + c960 + custom + koth + tiny)
    assert (mr.stats["n"], mr.stats["excluded_custom"], mr.stats["excluded_short"]) == (10, 5, 2)
    white = table(mr, "As White")
    assert sum(row[1] for row in white.rows) == 10
    assert "7 game(s) left out" in white.note
    assert "not enough data" not in mr.summary.lower()
    only_excluded = run(c960 + tiny)
    assert "not enough data" in only_excluded.summary.lower() and only_excluded.tables == []


def test_games_without_ratings():
    games = play("caro", 2, 20, my_rating=None, opp_rating=None) + play("italian", 20, 2, my_rating=None)
    mr = run(games)
    assert not [i for i in mr.insights if i.kind != "observation"]
    row = dict(zip(table(mr, "As Black").columns, table(mr, "As Black").rows[0]))
    assert row["Games"] == 22 and row["Score"] == pytest.approx(2 / 22)
    assert row["Expected"] is None and row["Difference"] is None and row["Adjusted"] is None and row["± (95%)"] is None
    assert mr.charts == []


def test_tiny_sample_has_tables_but_no_claims():
    mr = run(play("caro", 0, 6))
    assert "not enough data" in mr.summary.lower() and mr.insights == []
    assert table(mr, "As Black").rows[0][:2] == ["Caro-Kann Defense", 6]


# --------------------------------------------------------------------------- grouping
def test_first_move_groups_for_each_colour():
    white = [make_game(color="white", moves_san=[m, "e5"]) for m in ("e4", "e4", "d4", "c4", "Nf3", "g3", "b3")]
    black = [make_game(color="black", moves_san=[m, "c6"]) for m in ("e4", "d4", "d4", "c4", "f4")]
    mr = run(white + black)
    rows = {(r[0], r[1]): r for r in table(mr, "By first move").rows}
    assert list(rows) == [
        ("White", "1.e4"),
        ("White", "1.d4"),
        ("White", "1.c4"),
        ("White", "1.Nf3"),
        ("White", "Other first moves"),
        ("Black", "vs 1.e4"),
        ("Black", "vs 1.d4"),
        ("Black", "vs other"),
    ]
    assert [r[2] for r in rows.values()] == [2, 1, 1, 1, 2, 1, 2, 2]
    assert rows[("White", "1.e4")][3] == pytest.approx(2 / 7)  # share of White games
    assert rows[("Black", "vs other")][3] == pytest.approx(2 / 5)


def test_group_row_values():
    games = play("caro", 3, 4, 1, opp_rating=1600)  # 8 games, 3.5 points, expected 0.36 each
    short = make_game(
        outcome="loss", termination="checkmate", opp_rating=1600, **{**OPENING_KW["caro"], "moves_san": CARO * 2}
    )
    flagged = make_game(
        outcome="loss", termination="timeout", opp_rating=1600, **{**OPENING_KW["caro"], "moves_san": CARO}
    )
    mr = run(games + [short, flagged])
    row = dict(zip(table(mr, "As Black").columns, table(mr, "As Black").rows[0]))
    all_games = games + [short, flagged]
    e = 1 / (1 + 10 ** (100 / 400))
    assert row["Games"] == 10 and row["W/D/L"] == "3/1/6" and row["Share"] == 1.0
    assert row["Score"] == pytest.approx(3.5 / 10) and row["Expected"] == pytest.approx(e)
    assert row["Difference"] == pytest.approx(3.5 / 10 - e)
    assert row["Adjusted"] == pytest.approx(shrink(3.5 / 10 - e, 10, 0.0, 10.0))
    assert row["Avg moves"] == pytest.approx(sum(g.full_moves for g in all_games) / 10)
    # the default factory loss is a 20-ply resignation (short); the timeout doesn't count
    assert row["Short losses"] == 5


def test_short_loss_definition():
    th = openings.Thresholds()
    assert openings.is_short_loss(make_game(outcome="loss", termination="resignation"), th.short_loss_moves)
    assert not openings.is_short_loss(make_game(outcome="loss", termination="timeout"), th.short_loss_moves)
    assert not openings.is_short_loss(make_game(outcome="loss", termination="abandoned"), th.short_loss_moves)
    assert not openings.is_short_loss(make_game(outcome="loss", moves_san=["e4", "e5"]), th.short_loss_moves)
    assert not openings.is_short_loss(make_game(outcome="loss", moves_san=ITALIAN * 7), th.short_loss_moves)  # 28 moves
    assert not openings.is_short_loss(make_game(outcome="win"), th.short_loss_moves)


def test_family_table_top_twelve_then_other_row():
    games = []
    for i in range(15):
        games += [
            make_game(color="white", opening_family=f"Family {i:02d} Opening", opening=None) for _ in range(20 - i)
        ]
    games += [make_game(color="white", opening_family=None, opening=None) for _ in range(4)]
    mr = run(games)
    white = table(mr, "As White")
    assert len(white.rows) == 13
    assert [r[0] for r in white.rows[:12]] == [f"Family {i:02d} Opening" for i in range(12)]
    other = white.rows[-1]
    assert other[0] == "Other (3 more openings + unnamed)"
    assert other[1] == (20 - 12) + (20 - 13) + (20 - 14) + 4
    assert sum(r[1] for r in white.rows) == len(games)
    assert sum(r[2] for r in white.rows) == pytest.approx(1.0)


def test_most_played_lines_needs_five_games():
    games = play("caro", 3, 3) + play("italian", 2, 2)
    mr = run(games)
    lines = table(mr, "Most played lines")
    assert [(r[0], r[1], r[2]) for r in lines.rows] == [("Caro-Kann Defense: Advance Variation", "Black", 6)]


# --------------------------------------------------------------------------- shrinkage & chart
def test_shrinkage_pulls_small_samples_harder():
    big = play("caro", 12, 28)  # 40 games at 30%
    small = play("french", 3, 7)  # 10 games at 30%
    mr = run(big + small)
    fams = mr.stats["families"]
    caro, french = fams["black:Caro-Kann Defense"], fams["black:French Defense"]
    assert caro["delta"] == pytest.approx(french["delta"]) == pytest.approx(-0.2)
    assert caro["shrunk_delta"] == pytest.approx(shrink(-0.2, 40, 0.0, 10.0)) == pytest.approx(-0.16)
    assert french["shrunk_delta"] == pytest.approx(-0.1)
    assert caro["shrunk_delta"] < french["shrunk_delta"] < 0
    assert caro["half_width"] < french["half_width"]
    chart = mr.charts[0]
    assert chart.kind == "hbar" and chart.value_format == "signed_pct" and chart.reference == 0.0
    assert chart.labels == ["Caro-Kann Defense (Black)", "French Defense (Black)"]
    assert chart.series[0].values == pytest.approx([-0.16, -0.1])


def test_chart_needs_eight_rated_games():
    mr = run(play("caro", 3, 4) + play("italian", 20, 20))
    assert mr.charts[0].labels == ["Italian Game (White)"]


# --------------------------------------------------------------------------- insights
def test_bad_opening_with_enough_games_is_a_weakness():
    # 6/30 (20%) against 15/30 in the player's other Black opening: gap -0.30, p ~ 0.02. The old
    # 8/30 (27%) had a BH-adjusted p of 0.06 and was only flagged under the old lax rule, which
    # also flagged openings in pure noise.
    caro = play("caro", 6, 24)
    others = play("qgd", 15, 15) + play("italian", 15, 15)
    mr = run(caro + others)
    ins = insight(mr, "openings.weakness.black.caro-kann-defense")
    assert ins is not None and ins.category == "openings"
    assert ins.title == "You score less with the Caro-Kann Defense than with your other Black openings"
    assert ins.detail.startswith(
        "You score 20% in 30 games where 50% was expected (−0.30 points per game against your rating); "
        "in your other 30 games as Black, 0.00. The gap is −0.30 points per game"
    )
    recent_losses = sorted((g for g in caro if g.outcome == "loss"), key=lambda g: g.end_time, reverse=True)
    assert ins.example_games == [g.url for g in recent_losses[:5]]
    assert any("Advance Variation" in s for s in ins.study)
    assert ins.evidence["shrunk_delta"] == pytest.approx(shrink(6 / 30 - 0.5, 30, 0.0, 10.0))
    assert ins.evidence["gap"] == pytest.approx(-0.3) and ins.evidence["rest_delta"] == pytest.approx(0.0)
    assert ins.evidence["effect"] == pytest.approx(shrink(-0.3, 30, 0.0, 10.0))
    assert mr.stats["family_tests"]["black:Caro-Kann Defense"]["p_adjusted"] <= 0.05
    # two groups with one colour are one comparison: the QGD is not also called a strength
    assert "black:Queen's Gambit Declined" not in mr.stats["family_tests"]
    assert not [i for i in mr.insights if i.kind == "strength"]
    assert "Caro-Kann" in mr.summary
    assert any(k.label == "Most costly opening" and k.value == "Caro-Kann Defense (Black)" for k in mr.kpis)


def test_small_or_mild_samples_are_not_flagged():
    assert (
        insight(run(play("caro", 0, 5) + play("italian", 10, 10)), "openings.weakness.black.caro-kann-defense") is None
    )
    # 8 games at 37.5%: raw -0.125 but shrunk to -0.056, under the 0.06 bar
    mr = run(play("caro", 3, 5) + play("italian", 10, 10))
    assert mr.stats["families"]["black:Caro-Kann Defense"]["shrunk_delta"] == pytest.approx(-0.125 * 8 / 18)
    assert insight(mr, "openings.weakness.black.caro-kann-defense") is None


def test_good_opening_is_a_strength_with_recent_wins():
    italian = play("italian", 24, 6)
    mr = run(italian + play("london", 15, 15) + play("caro", 15, 15))
    ins = insight(mr, "openings.strength.white.italian-game")
    assert ins is not None and ins.title == "You score more with the Italian Game than with your other White openings"
    recent_wins = sorted((g for g in italian if g.outcome == "win"), key=lambda g: g.end_time, reverse=True)
    assert ins.example_games == [g.url for g in recent_wins[:5]]


def test_openings_chosen_by_the_opponent_are_worded_as_against():
    assert openings.chosen_by("Sicilian Defense") == "black"
    assert openings.chosen_by("Albin Countergambit") == "black"
    assert openings.chosen_by("Englund Gambit") == "black"
    assert openings.chosen_by("Italian Game") == "white"
    assert openings.chosen_by("Queen's Gambit Declined") == "white"
    mr = run(play("sicilian_w", 6, 24) + play("italian", 15, 15) + play("caro", 15, 15))
    ins = insight(mr, "openings.weakness.white.sicilian-defense")
    assert ins.title == "You score less against the Sicilian Defense than against other openings as White"
    assert any("answer to the Sicilian Defense" in s for s in ins.study)


def test_early_disasters_flagged_when_quick_losses_stand_out():
    long_loss = CARO * 6  # 30 moves
    french = play("french", 16, 2, draws=4, moves_san=long_loss)
    french += [make_game(outcome="loss", termination="checkmate", **OPENING_KW["french"]) for _ in range(6)]  # 4 moves
    caro = play("caro", 20, 20, moves_san=long_loss)
    mr = run(french + caro)
    assert insight(mr, "openings.weakness.black.french-defense") is None  # scores fine overall
    ins = insight(mr, "openings.weakness.early-losses.black.french-defense")
    assert ins is not None and ins.title == "You often lose quickly in the French Defense as Black"
    assert "6 of your 8 losses" in ins.detail and "0%" in ins.detail
    shortest = sorted(
        (g for g in french if g.outcome == "loss" and g.plies <= 8), key=lambda g: (g.plies, -g.end_time.timestamp())
    )
    assert ins.example_games == [g.url for g in shortest[:5]]


def test_early_losses_not_repeated_for_a_family_already_flagged():
    caro = [make_game(outcome="loss", termination="resignation", **OPENING_KW["caro"]) for _ in range(20)]
    caro += play("caro", 4, 0)
    others = play("french", 12, 12, moves_san=FRENCH * 5) + play("italian", 20, 20, moves_san=ITALIAN * 5)
    mr = run(caro + others)
    weak = insight(mr, "openings.weakness.black.caro-kann-defense")
    assert weak is not None and "20 of your 20 losses in it were over within 25 moves" in weak.detail
    assert not [i for i in mr.insights if "early-losses" in i.id]


def test_repertoire_breadth_observations():
    white = []
    for i in range(10):
        white += [make_game(color="white", opening_family=f"System {i}", opening=None) for _ in range(3)]
    black = play("caro", 12, 12)
    mr = run(white + black)
    broad = insight(mr, "openings.observation.breadth-white")
    assert broad.kind == "observation" and "very broad" in broad.title and "8 openings" in broad.title
    compact = insight(mr, "openings.observation.breadth-black")
    assert "compact" in compact.title and "1 opening cover" in compact.title
    assert mr.stats["breadth"] == {"white": 8, "black": 1}


def test_bh_adjustment_is_recorded_for_every_tested_family():
    black = play("caro", 8, 22) + play("french", 5, 5) + play("qgd", 10, 10)
    white = play("italian", 16, 14) + play("london", 15, 15)  # two White groups: one comparison, kept for the Italian
    mr = run(black + white)
    tests = mr.stats["family_tests"]
    assert set(tests) == {"black:Caro-Kann Defense", "black:French Defense", "black:Queen's Gambit Declined",
                          "white:Italian Game"}
    assert all(t["p_adjusted"] >= t["p_value"] for t in tests.values())


def test_thresholds_can_be_overridden_through_options():
    # 7 straight losses against 10/20 in the other Black opening (p ~ 0.02); the old 0/5 was not significant.
    games = play("caro", 0, 7) + play("french", 10, 10) + play("italian", 10, 10)
    assert insight(run(games), "openings.weakness.black.caro-kann-defense") is None  # 7 < 8 games
    mr = run(games, **{"openings.min_family_games": 5})
    assert insight(mr, "openings.weakness.black.caro-kann-defense") is not None


def test_real_chesscom_games_every_player(fixtures_dir):
    raws = json.loads((fixtures_dir / "real_chesscom_games.json").read_text())["games"]
    users = {raw[side]["username"] for raw in raws for side in ("white", "black") if raw.get(side, {}).get("username")}
    for user in sorted(users):
        games = parse.parse_games(raws, user)  # unfiltered: variants, chess960, daily, casual all included
        mr = run(games)
        assert mr.stats["n"] + mr.stats["excluded_custom"] + mr.stats["excluded_short"] == len(games)


# --------------------------------------------------------------------------- calibration: claims that used to fire on noise
def test_sharp_openings_are_not_early_loss_weaknesses():
    # Quick losses AND quick wins in the French: a sharp opening, not a weakness. The old test compared
    # the French losses with your *other* losses (all long here) and flagged it.
    long_moves = CARO * 6  # 30 moves
    french = [make_game(outcome=o, termination="resignation", **OPENING_KW["french"]) for o in ["win", "loss"] * 12]
    caro = play("caro", 12, 12, moves_san=long_moves)
    mr = run(french + caro)
    assert not [i for i in mr.insights if "early-losses" in i.id and i.kind != "observation"]
    tests = mr.stats["early_loss_tests"]["black:French Defense"]
    assert tests["short_loss_share"] == 1.0 and tests["short_win_share"] == 1.0 and not tests["significant"]


def test_openings_are_compared_with_your_other_openings_of_the_same_colour():
    # A player who is underrated (scores 70% where 50% is expected in every opening) has no opening
    # strengths: the old test against the Elo expectation called all five openings strengths.
    fams = {"black": ("caro", "french", "qgd"), "white": ("italian", "london")}
    underrated = [g for keys in fams.values() for key in keys for g in play(key, 35, 15)]
    mr = run(underrated)
    assert not [i for i in mr.insights if i.kind in ("strength", "weakness")]
    tests = mr.stats["family_tests"]
    assert tests["black:Caro-Kann Defense"]["gap"] == pytest.approx(0.0)
    assert tests["black:Caro-Kann Defense"]["rest_delta"] == pytest.approx(0.2)
    # A weak Black as a whole (44% vs White's 56%) is the results module's colour question, not a
    # weakness of each Black opening (nor a strength of each White one).
    colour_gap = [g for key in fams["black"] for g in play(key, 22, 28)]
    colour_gap += [g for key in fams["white"] for g in play(key, 28, 22)]
    assert not [i for i in run(colour_gap).insights if i.kind in ("strength", "weakness")]


def test_one_opening_that_is_almost_all_of_a_colour_is_not_claimed():
    # 30 Caro-Kann games at 20% and nothing else as Black: the Caro-Kann can't be told apart from Black.
    mr = run(play("caro", 6, 24) + play("italian", 15, 15) + play("london", 15, 15))
    assert "black:Caro-Kann Defense" not in mr.stats["family_tests"]
    assert insight(mr, "openings.weakness.black.caro-kann-defense") is None


def test_quick_resigner_has_no_early_loss_weaknesses():
    # Every loss is a quick resignation and every win a long game, in every opening: a resignation
    # habit, not a weakness of one opening (the mirror test alone flagged each of them).
    long_moves = CARO * 6  # 30 moves
    games = []
    for key in ("caro", "french", "qgd", "italian", "london"):
        games += play(key, 12, 0, moves_san=long_moves) + play(key, 0, 12)  # 10-ply losses by resignation
    mr = run(games)
    assert not [i for i in mr.insights if "early-losses" in i.id]
    tests = mr.stats["early_loss_tests"]
    assert tests["black:French Defense"]["short_loss_share"] == 1.0 and tests["black:French Defense"]["gap"] == 0.0
    assert all(not t["significant"] for t in tests.values())
