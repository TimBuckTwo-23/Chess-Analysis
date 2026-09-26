"""analysis/structure.py: castling, early queen moves, development and pawn moves, you vs your opponents."""

import dataclasses
import hashlib
import json

import chess
import pytest

from chess_insights import insights, pipeline
from chess_insights.analysis import structure
from chess_insights.context import AnalysisContext
from chess_insights.models import CATEGORIES, VALUE_FORMATS, ModuleResult
from chess_insights.stats import MeanTest
from factories import DEFAULT_MOVES, all_tables, make_game
from null_world import OPENINGS, null_games

STRUCTURE = ("structure", "Development & king safety")
MODULES = pipeline.MODULES if STRUCTURE in pipeline.MODULES else pipeline.MODULES + [STRUCTURE]

# Both sides develop all four knights and bishops and make four pawn moves; only one of them castles.
WHITE_UNCASTLED = "e4 e5 Nf3 Nc6 Bc4 Bc5 Nc3 Nf6 d3 d6 Be3 O-O h3 Be6 a3 a6 Rb1 h6 Ra1 Re8".split()  # 6...O-O
BLACK_UNCASTLED = "e4 e5 Nf3 Nc6 Bc4 Bc5 Nc3 Nf6 d3 d6 Be3 Be6 O-O h6 a3 a6 h3 Rb8 Re1 Ra8".split()  # 7.O-O
# 2...Qxd5 is a capture (not counted), 3...Qa5 and 8.Qe2 are early queen moves; 9.O-O-O, 10...O-O
SCANDINAVIAN = "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 Qe2 Bb4 O-O-O Nbd7 a3 O-O".split()
# The same with 8.O-O: only Black moves the queen early; White makes four pawn moves (exd5 included), Black three
QUEEN_EARLY = "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 O-O Bb4 a3 Nbd7 Re1 O-O".split()
# 2.Qh5 is an early queen move (chased by 3...g6); Black castles on move 6, White not by move 10
QH5 = "e4 e5 Qh5 Nc6 Bc4 g6 Qf3 Nf6 Ne2 Bg7 Nbc3 O-O d3 d6 Bg5 Be6 h3 a6 a3 Rb8".split()
# White: eight pawn moves in ten, knights out late (9.Ne2, 10.Nd2), bishops at home; Black: four pawn moves, castled
PAWNY = "e4 e5 d4 Nc6 c3 Nf6 f3 d6 g3 Be7 h3 O-O a3 Re8 b3 Bf8 Ne2 h6 Nd2 a6".split()
FORMATS = ["blitz", "blitz", "rapid", "bullet"]


# --------------------------------------------------------------------------- helpers
def assert_consistent(mr: ModuleResult) -> None:
    for chart in mr.charts + [i.chart for i in mr.insights if i.chart]:
        assert chart.value_format in VALUE_FORMATS
        assert chart.series and all(len(s.values) == len(chart.labels) for s in chart.series), chart.title
    for table in all_tables(mr) + [i.chart.table for i in mr.insights if i.chart and i.chart.table]:
        assert table.formats is None or len(table.formats) == len(table.columns)
        assert table.formats is None or set(table.formats) <= set(VALUE_FORMATS)
        assert all(len(row) == len(table.columns) for row in table.rows), table.title
    assert all(k.format in VALUE_FORMATS for k in mr.kpis)
    for ins in mr.insights:
        assert ins.id.startswith(f"{mr.key}.{ins.kind}.") and ins.category in CATEGORIES
        assert ins.kind in ("strength", "weakness")
        assert 0.0 <= ins.severity <= 1.0 and 0.0 <= ins.confidence <= 1.0
        assert 2 <= len(ins.study) <= 4 and ins.title and ins.detail
        assert len(ins.example_games) <= 5 and all(u.startswith("https://") for u in ins.example_games)
        assert ins.formats and ins.chart is not None
    json.dumps(mr.stats, default=str, allow_nan=False)


def run(games, **options) -> ModuleResult:
    mr = structure.analyze(AnalysisContext("tester", sorted(games, key=lambda g: g.end_time), options=options))
    assert (mr.key, mr.title) == STRUCTURE
    assert_consistent(mr)
    return mr


def uncastled_games(n: int, **kw) -> list:
    """``n`` games in which you never castle by move 10 and your opponent does, alternating colours and formats."""
    games = []
    for k in range(n):
        color = "white" if k % 2 == 0 else "black"
        moves = WHITE_UNCASTLED if color == "white" else BLACK_UNCASTLED
        games.append(make_game(moves_san=moves, color=color, time_class=FORMATS[k % 4], outcome="loss" if k % 3 else "win", **kw))
    return games


def castled_games(n: int) -> list:
    """The same games with the colours swapped: you castle, your opponent doesn't."""
    return [
        make_game(moves_san=WHITE_UNCASTLED if k % 2 else BLACK_UNCASTLED, color="black" if k % 2 else "white",
                  time_class=FORMATS[k % 4], outcome="win" if k % 3 else "loss")
        for k in range(n)
    ]


# --------------------------------------------------------------------------- reading the first moves
def test_read_opening_counts_each_habit_for_both_players():
    hs = {h.key: h for h in structure.habits(structure.Thresholds())}
    counts = lambda o: {k: (h.count(o.me), h.count(o.opp)) for k, h in hs.items()}  # noqa: E731
    italian = structure.read_opening(make_game(moves_san=DEFAULT_MOVES, color="white"), 20)
    # both castle on move 6; the c-bishops stay at home (Bb3 and Ba7 are second moves of the same bishops)
    assert counts(italian) == {"castled": (1, 1), "early_queen": (0, 0), "minors": (3, 3), "pawns": (4, 4)}
    assert italian.me.castle_move == italian.opp.castle_move == 6
    scandi = structure.read_opening(make_game(moves_san=SCANDINAVIAN, color="black"), 20)
    assert (scandi.me.queen_move, scandi.me.queen_ply) == (3, 5)  # 3...Qa5, not the capture 2...Qxd5
    assert scandi.opp.queen_move == 8 and scandi.opp.castle_move == 9  # 8.Qe2, 9.O-O-O
    assert counts(scandi) == {"castled": (1, 1), "early_queen": (1, 1), "minors": (4, 4), "pawns": (3, 4)}


def test_a_piece_captured_at_home_was_never_developed():
    # 3.Bxb7 and 4.Bxc8: Black's c8-bishop is taken before it moved; White's bishop counts once however often it moves
    o = structure.read_opening(make_game(moves_san="g3 e6 Bg2 Ba3 Bxb7 Nc6 Bxc8 Qxc8".split(), color="black"), 20)
    assert (o.me.minors_by(4), o.opp.minors_by(4)) == (2, 1)
    assert o.me.queen_move is None  # 4...Qxc8 is a capture


def test_unreadable_moves_and_other_start_positions_are_left_out():
    chess960 = make_game(rules="chess960", initial_fen="bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9")
    setup = make_game(initial_fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", moves_san=["e4"])
    standard_fen = make_game(initial_fen=chess.STARTING_FEN)
    broken = make_game(moves_san=["e4", "e5", "Nf9"] + DEFAULT_MOVES[3:])
    assert structure.is_standard_start(standard_fen) and not structure.is_standard_start(chess960)
    assert not structure.is_standard_start(setup)
    mr = run([chess960, setup, standard_fen, broken])
    assert mr.stats["skipped_variant_or_setup"] == 2 and mr.stats["unreadable"] == 1
    assert mr.stats["habits"]["castled"]["n"] == 1


def test_a_game_counts_for_a_habit_only_when_both_players_reached_its_window():
    short = [make_game(moves_san=SCANDINAVIAN[:16]), make_game(moves_san=SCANDINAVIAN[:19]), make_game(moves_san=["e4", "e5", "Qh5"])]
    mr = run(short + [make_game(moves_san=SCANDINAVIAN)])
    habits = mr.stats["habits"]
    assert habits["early_queen"]["n"] == 3  # 16 plies or more: both players made 8 moves
    assert habits["castled"]["n"] == habits["minors"]["n"] == habits["pawns"]["n"] == 1  # 20 plies


# --------------------------------------------------------------------------- the test
def test_paired_share_test_weights_white_and_black_games_equally():
    # White always castles by move 10 and Black never; you play White in 80% of your games. Pooled, you castle
    # 60 points more often than your opponents; with the colours weighted equally the gap is 0.
    pairs = [(1, 0, "white")] * 80 + [(0, 1, "black")] * 20
    test = structure.paired_share_test(pairs, 1)
    assert test.mean == pytest.approx(0.0) and test.p_value > 0.5
    # a real gap in both colours is found, as a difference in shares
    pairs = [(1, 0, "white")] * 30 + [(0, 0, "white")] * 20 + [(1, 0, "black")] * 25 + [(1, 1, "black")] * 25
    test = structure.paired_share_test(pairs, 1)
    assert test.mean == pytest.approx(0.5 * 30 / 50 + 0.5 * 25 / 50) and test.p_value < 1e-6 and test.n == 100
    # minor pieces as a share of four; no variation at all is no evidence
    assert structure.paired_share_test([(4, 2, "white"), (4, 2, "black")] * 10, 4).mean == pytest.approx(0.5)
    assert structure.paired_share_test([(0, 0, "white")] * 50, 1).p_value == 1.0
    assert structure.paired_share_test([], 1).p_value == 1.0


def test_castling_later_than_your_opponents_is_a_weakness_with_chart_formats_and_board():
    mr = run(uncastled_games(60))
    [ins] = mr.insights
    assert ins.id == "structure.weakness.castled" and ins.category == "structure"
    assert ins.title == "You castle later than your opponents"
    assert ins.detail.startswith(
        "By move 10 you had castled in 0% of games, your opponents in 100% of the same games (60 games in which "
        "both of you made at least 10 moves)."
    )
    assert "By format, you vs them: bullet 0% vs 100%, blitz 0% vs 100%, rapid 0% vs 100%." in ins.detail
    assert ins.formats == {"bullet": 15, "blitz": 30, "rapid": 15}
    assert ins.evidence["gap"] == pytest.approx(-1.0) and ins.evidence["p_adjusted"] <= 0.01
    # the chart: you vs your opponents per format, then all games, with the numbers underneath
    assert ins.chart.labels == ["Bullet", "Blitz", "Rapid", "All games"]
    assert [s.name for s in ins.chart.series] == ["You", "Opponents"]
    assert ins.chart.series[1].values == [1.0, 1.0, 1.0, 1.0] and ins.chart.value_format == "pct"
    assert ins.chart.table.rows[-1] == ["All games", 60, 0.0, 1.0]
    # a board from the first example game (a loss): your king still at home after move 10, marked
    d = ins.diagram
    assert d is not None and d.link == ins.example_games[0] and d.time_class
    board = chess.Board(d.fen)
    my = chess.WHITE if d.orientation == "white" else chess.BLACK
    assert [m.square for m in d.marks] == [chess.square_name(board.king(my))] and d.marks[0].kind == "weak"
    assert board.fullmove_number == 11 and d.last_move == chess.Move.from_uci(d.last_move).uci()
    assert "after move 10 you hadn't castled (king on e" in d.caption and "your opponent castled on move" in d.caption
    # the study plan's target carries today's number
    assert ins.evidence["target"] == "Castle by move 10 in at least 100% of your games, like your opponents (now 0%)."
    assert insights.target_for(ins) == (ins.evidence["target"], {"metric": "share of games castled by move 10", "value": 0.0})


def test_castling_earlier_is_a_strength():
    mr = run(castled_games(60))
    assert [i.id for i in mr.insights] == ["structure.strength.castled"]
    strength = mr.insights[0]
    assert strength.title == "You castle earlier than your opponents"
    assert strength.example_games and strength.evidence["gap"] == pytest.approx(1.0)
    # a strength gets a board too: the same moment on your opponent's side, their uncastled king marked
    d = strength.diagram
    assert d.title == "Move 10: your opponent hasn't castled yet" and d.link == strength.example_games[0]
    board = chess.Board(d.fen)
    opponent = chess.BLACK if d.orientation == "white" else chess.WHITE
    assert [(m.square, m.kind) for m in d.marks] == [(chess.square_name(board.king(opponent)), "target")]
    assert d.caption.startswith("Win, ") and "you castled on move " in d.caption
    assert "after move 10 your opponent hadn't (king on e" in d.caption


def test_early_queen_strip_marks_the_queen_when_it_is_chased():
    mr = run([make_game(moves_san=QH5, color="white", time_class="blitz") for _ in range(40)])
    queen = next(i for i in mr.insights if i.id == "structure.weakness.early-queen")
    d = queen.diagram
    assert d.title == "Your queen comes out on move 2" and d.orientation == "white"
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("d1", "h5", "played")] and d.last_move == "e7e5"
    [strip] = d.strips
    assert [f.move for f in strip.frames] == ["2.Qh5", "2...Nc6", "3.Bc4", "3...g6"]
    assert [f.caption for f in strip.frames] == ["", "", "", "Queen attacked"]
    assert [(m.square, m.kind) for m in strip.frames[3].marks] == [("h5", "target"), ("g6", "attacker")]
    # the same games from Black's side: your opponent's early queen move is a grey arrow, and the strip shows
    # your pawn chasing it
    mr = run([make_game(moves_san=QH5, color="black", time_class="blitz") for _ in range(40)])
    strength = next(i for i in mr.insights if i.id == "structure.strength.early-queen")
    d = strength.diagram
    assert d.title == "Your opponent's queen comes out on move 2" and d.orientation == "black"
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("d1", "h5", "neutral")]
    assert "your opponent played 2.Qh5; you made no early queen move of your own" in d.caption
    assert d.strips[0].frames[3].caption == "Queen attacked"


def test_pawn_moves_and_development_boards_mark_the_squares_they_are_about():
    mr = run([make_game(moves_san=PAWNY, color="white", time_class="rapid") for _ in range(40)])
    by = {i.id: i for i in mr.insights}
    assert set(by) == {"structure.weakness.castled", "structure.weakness.minors", "structure.weakness.pawns"}
    pawns = by["structure.weakness.pawns"].diagram
    assert pawns.title == "Move 10: many pawn moves"
    assert sorted(m.square for m in pawns.marks) == ["a3", "b3", "c3", "d4", "e4", "f3", "g3", "h3"]
    assert {m.kind for m in pawns.marks} == {"focus"}
    assert "8 of your first 10 moves were pawn moves; your opponent's 4." in pawns.caption
    minors = by["structure.weakness.minors"].diagram
    assert sorted((m.square, m.kind) for m in minors.marks) == [("c1", "weak"), ("f1", "weak")]
    assert "you had 2 of 4 knights and bishops out; your opponent 3." in minors.caption


def test_a_piece_that_went_out_and_came_back_is_not_marked_at_home():
    moves = "e4 e5 Nf3 Nc6 Ng1 Nf6".split()  # the g1 knight went out and came back
    o = structure.read_opening(make_game(moves_san=moves), 20)
    board = o.board_after(5)
    assert o.me.minors_by(3) == 1
    assert structure.pieces_at_home(board, chess.WHITE, o.ucis) == [chess.B1, chess.C1, chess.F1]
    assert structure.pieces_at_home(board, chess.BLACK, o.ucis) == [chess.C8, chess.F8]


def test_games_without_a_link_still_get_a_board():
    mr = run(uncastled_games(40, url=""))
    [ins] = mr.insights
    assert ins.example_games == [] and ins.diagram is not None and ins.diagram.link == ""


def test_early_queen_weakness_shows_the_queen_move_as_a_red_arrow():
    mr = run([make_game(moves_san=QUEEN_EARLY, color="black", time_class="rapid") for _ in range(60)])
    queen = next(i for i in mr.insights if i.id == "structure.weakness.early-queen")
    assert queen.title == "You bring your queen out early more often than your opponents"
    assert queen.formats == {"rapid": 60} and queen.chart.labels == ["Rapid"]  # one format: no "All games" bar
    d = queen.diagram
    assert d.title == "Your queen comes out on move 3" and d.orientation == "black" and d.time_class == "rapid"
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("d5", "a5", "played")]
    assert d.last_move == "b1c3" and sorted(m.square for m in d.marks) == ["b8", "c8", "f8", "g8"]
    assert "you played 3...Qa5, with 4 of your knights and bishops still at home" in d.caption
    # you also make fewer pawn moves than your opponents (3 against 4): described, never called a strength
    pawns = mr.stats["habits"]["pawns"]
    assert (pawns["you"], pawns["opponents"]) == (3, 4) and pawns["p_adjusted"] <= 0.01
    assert [i.id for i in mr.insights] == ["structure.weakness.early-queen"]


def test_small_gaps_and_small_samples_are_not_claims():
    th = structure.Thresholds()
    castled = structure.habits(th)[0]
    openings = [structure.read_opening(make_game(), 20) for _ in range(40)]
    # overwhelming evidence for a 5-point gap: too small to matter (the bar is 10 points)
    small = structure.HabitResult(castled, openings, MeanTest(40, -0.05, 0.004, -12.5, 1e-30), p_adjusted=1e-30)
    assert structure.habit_insight(small, th) is None
    big = dataclasses.replace(small, test=MeanTest(40, -0.2, 0.016, -12.5, 1e-30))
    assert structure.habit_insight(big, th).id == "structure.weakness.castled"
    assert structure.habit_insight(dataclasses.replace(big, p_adjusted=0.02), th) is None  # the strict alpha
    games = [make_game(moves_san=DEFAULT_MOVES, color=c) for c in ("white", "black") * 30]
    assert run(games).insights == []  # the same habits on both sides
    few = run(uncastled_games(20))  # a 100-point gap, but fewer games than the minimum
    assert few.insights == [] and few.summary.startswith("Not enough data yet: only 20 games reached move 10")
    assert run(uncastled_games(20), **{"structure.min_games": 10}).insights  # thresholds are options


def test_module_section_overview_kpis_and_summary():
    mr = run(uncastled_games(60))
    [chart] = mr.charts
    assert chart.title == "Your first moves against your opponents'" and chart.value_format == "pct"
    assert chart.labels == ["Castled by move 10", "Queen out early (moves 1–8)", "Knights and bishops out by move 10",
                            "Pawn moves in moves 1–10"]
    assert chart.series[0].values == [0.0, 0.0, 1.0, 0.4] and chart.series[1].values == [1.0, 0.0, 1.0, 0.4]
    rows = chart.table.rows  # by format, in the habit's own units
    assert ["Castled by move 10", "Blitz", 30, "0%", "100%"] in rows
    assert ["Knights and bishops out by move 10", "All games", 60, "4.0 of 4", "4.0 of 4"] in rows
    assert chart.table.key_columns == [0, 1, 3, 4]
    assert [k.label for k in mr.kpis] == ["Castled by move 10", "Knights and bishops out", "Queen out early",
                                          "Early pawn moves"]
    assert mr.kpis[0].value == 0.0 and mr.kpis[0].hint == "opponents 100%"
    assert mr.kpis[1].hint == "of 4 by move 10; opponents 4.0"
    assert mr.summary.startswith("In the 60 standard games that reached move 10, you had castled by then in 0% (your "
                                 "opponents in the same games: 100%), had 4.0 of your 4 knights and bishops out (4.0)")
    assert mr.summary.endswith("Key finding: You castle later than your opponents.")


@pytest.mark.parametrize("games", [[], [make_game(moves_san=["e4"])], [make_game(rules="chess960")]])
def test_no_usable_games_is_not_an_error(games):
    mr = run(games)
    assert mr.insights == [] and mr.kpis == [] and mr.charts == []
    assert mr.summary.startswith("Not enough data: none of your standard games reached move 10")


def test_pipeline_lists_the_finding_and_plans_it_with_a_target():
    report = pipeline.run_analysis(uncastled_games(60), "tester", modules=MODULES)
    section = next(m for m in report.modules if m.key == "structure")
    assert "error" not in section.stats and section.title == "Development & king safety"
    assert "structure.weakness.castled" in [i.id for i in report.weaknesses]
    item = next(i for i in report.study_plan if "structure.weakness.castled" in i.insight_ids)
    assert item.category == "Your first ten moves · practise it in your next games"
    assert item.target.startswith("Castle by move 10 in at least 100% of your games")
    assert item.baseline["metric"] == "share of games castled by move 10" and 2 <= len(item.actions) <= 3
    assert insights.STUDY_LIBRARY["structure"]["label"] == "Development & king safety"


# --------------------------------------------------------------------------- the null world
def _fingerprint(games) -> str:
    """Everything about the null world's games except the moves."""
    rows = [
        (g.game_id, g.url, g.color, g.opponent, g.outcome, g.my_result_code, g.opp_result_code, g.termination,
         g.time_class, g.time_control, g.end_time.isoformat(), g.start_time.isoformat(), g.my_rating, g.opp_rating,
         g.my_rating_before, g.opp_rating_before, g.opening_family, g.eco, g.plies, g.clocks)
        for g in games
    ]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def test_null_world_opening_habits_change_nothing_but_the_moves():
    # fingerprints of the null world before it had opening habits: results, clocks, lengths, openings and
    # schedules are exactly as they were, planted effects included (so the other modules' numbers don't move)
    assert _fingerprint(null_games(120, seed=3)) == "bde8102657d09e46"
    assert _fingerprint(null_games(120, seed=3, planted={"play_on": 0.8, "flagging": 0.3})) == "d0036b3f31282ad4"
    planted = {"short_losses": ("white", "Italian Game", 0.4), "time_trouble": (0.35, -0.1), "tilt_after_loss": -0.12}
    assert _fingerprint(null_games(120, seed=3, planted=planted)) == "307c660f187cb50e"
    games = null_games(120, seed=3)
    lines = {(color, family): line for color in OPENINGS for line, family, _ in OPENINGS[color]}
    assert all(g.moves_san[: len(lines[g.color, g.opening_family])] == lines[g.color, g.opening_family] for g in games)
    for g in games[:40]:  # every move is legal
        board = chess.Board()
        for san in g.moves_san:
            board.push_san(san)


def test_null_world_players_share_the_same_opening_habits():
    mr = structure.analyze(AnalysisContext("p", null_games(600, seed=11)))
    habits = mr.stats["habits"]
    assert 0.35 < habits["castled"]["you"] < 0.65 and 2.5 < habits["minors"]["you"] < 3.5  # realistic levels
    for key, tolerance in (("castled", 0.06), ("early_queen", 0.04), ("minors", 0.15), ("pawns", 0.2)):
        assert abs(habits[key]["you"] - habits[key]["opponents"]) < tolerance, (key, habits[key])
    assert mr.insights == []


def test_planted_structure_effects_change_only_the_players_habits():
    n, seed = 160, 5

    def habits(planted=None):
        games = null_games(n, seed=seed, planted=planted)
        return games, structure.analyze(AnalysisContext("p", games)).stats["habits"]

    base_games, base = habits()
    late_games, late = habits({"late_castling": 1.0})
    assert _fingerprint(late_games) == _fingerprint(base_games)  # results, clocks, lengths ...
    assert late["castled"]["you"] < 0.05 and abs(late["castled"]["opponents"] - base["castled"]["opponents"]) < 0.06
    _, queen = habits({"early_queen": 1.0})
    assert queen["early_queen"]["you"] > 0.9 and queen["early_queen"]["opponents"] < 0.25
    _, slow = habits({"slow_development": 1.0})
    assert slow["minors"]["you"] < slow["minors"]["opponents"] - 0.6
    assert abs(slow["minors"]["opponents"] - base["minors"]["opponents"]) < 0.2
