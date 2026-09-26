"""Formats and pictures on the openings and habits findings: every finding says which formats it is about and
carries a chart (and, for openings, a board); the choice points become boards and a reusable list."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import chess
import pytest

from chess_insights import parse, synth
from chess_insights.analysis import habits, openings
from chess_insights.context import AnalysisContext
from chess_insights.models import ARROW_KINDS, TIME_CLASSES, VALUE_FORMATS, Chart, Diagram, ModuleResult
from chess_insights.stats import per100_games, pct, vs_rating
from factories import make_game, make_game_eval, make_ply_eval
from null_world import null_games

T0 = datetime(2024, 3, 4, 18, 0, tzinfo=timezone.utc)
CARO = ["e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6", "Be2", "c5"]
ITALIAN = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6"]
RUY = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]
CARO_KW = dict(color="black", opening_family="Caro-Kann Defense", opening="Caro-Kann Defense: Advance Variation")


def analyse(module, games, **options) -> ModuleResult:
    return module.analyze(AnalysisContext("tester", sorted(games, key=lambda g: g.end_time), options=options))


# --------------------------------------------------------------------------- checks shared by every test
def check_chart(chart: Chart, finding: bool = True) -> None:
    assert chart.title and chart.value_format in VALUE_FORMATS
    assert chart.labels and chart.series
    if finding:  # small enough for a phone, next to the finding
        assert len(chart.labels) <= 6 and len(chart.series) <= 3, chart.title
    assert all(len(s.values) == len(chart.labels) for s in chart.series), chart.title
    assert any(v is not None for s in chart.series for v in s.values), chart.title
    if chart.table is not None:
        t = chart.table
        assert t.formats is None or (len(t.formats) == len(t.columns) and set(t.formats) <= set(VALUE_FORMATS))
        assert all(len(row) == len(t.columns) for row in t.rows)
        assert t.key_columns is None or all(0 <= c < len(t.columns) for c in t.key_columns)


def check_board(fen: str, arrows=(), last_move: str = "") -> chess.Board:
    board = chess.Board(fen)
    for a in arrows:
        assert a.kind in ARROW_KINDS and chess.parse_square(a.start) is not None and chess.parse_square(a.end) is not None
    if last_move:
        chess.Move.from_uci(last_move)
    return board


def check_diagram(d: Diagram) -> None:
    assert d.title and d.orientation in ("white", "black")
    board = check_board(d.fen, d.arrows, d.last_move)
    for a in d.arrows:  # every arrow is a legal move in the position
        move = chess.Move(chess.parse_square(a.start), chess.parse_square(a.end))
        legal = {(m.from_square, m.to_square) for m in board.legal_moves}
        castles = {(m.from_square, chess.square(6 if board.is_kingside_castling(m) else 2, chess.square_rank(m.from_square)))
                   for m in board.legal_moves if board.is_castling(m)}
        assert (move.from_square, move.to_square) in legal | castles, (d.title, a)
    assert d.time_class in ("", *TIME_CLASSES)
    for strip in d.strips:
        assert strip.title and strip.frames
        for frame in strip.frames:
            check_board(frame.fen, frame.arrows, frame.last_move)


def check_visuals(mr: ModuleResult, n_games: int) -> None:
    """Every finding says which formats it is about and has a picture; the module's own boards are valid."""
    for ins in mr.insights:
        assert ins.formats, ins.id
        assert set(ins.formats) <= set(TIME_CLASSES) and all(n > 0 for n in ins.formats.values()), ins.id
        assert sum(ins.formats.values()) <= n_games, ins.id
        assert ins.chart is not None or ins.diagram is not None, ins.id
        if ins.chart is not None:
            check_chart(ins.chart)
        if ins.diagram is not None:
            check_diagram(ins.diagram)
    for chart in mr.charts:
        check_chart(chart, finding=False)
    for d in mr.diagrams:
        check_diagram(d)


def find(mr: ModuleResult, id_: str):
    return next((i for i in mr.insights if i.id == id_), None)


# --------------------------------------------------------------------------- realistic worlds
OPENING_WORLD = {"opening": ("black", "Caro-Kann Defense", -0.15), "short_losses": ("white", "Italian Game", 0.40)}
HABIT_WORLD = {"tilt_after_loss": -0.12, "late_night": -0.10}


@pytest.fixture(scope="module")
def opening_world():
    return null_games(1200, seed=3, planted=OPENING_WORLD)


@pytest.fixture(scope="module")
def habit_world():
    return null_games(1200, seed=5, planted=HABIT_WORLD)


@pytest.fixture(scope="module")
def demo_games():
    """The synthetic demo player (``chess-insights demo``), played by the fast heuristic mover."""
    archives = synth.generate_archives(300, seed=7, engine_path=None, workers=1)
    return parse.parse_games([g for month in archives.values() for g in month], synth.DEFAULT_PERSONA.username)


def test_every_finding_has_formats_and_a_picture_in_realistic_worlds(opening_world, habit_world):
    mr = analyse(openings, opening_world)
    assert {"openings.weakness.black.caro-kann-defense", "openings.weakness.early-losses.white.italian-game",
            "openings.observation.breadth-white", "openings.observation.breadth-black"} <= {i.id for i in mr.insights}
    check_visuals(mr, len(opening_world))
    assert all(i.diagram is not None for i in mr.insights if i.kind != "observation")  # opening findings: a board
    for tz in ("Etc/UTC", "America/New_York", None):
        mr = analyse(habits, habit_world, **({"tz": tz} if tz else {}))
        check_visuals(mr, len(habit_world))
    assert {"habits.weakness.after-a-loss", "habits.weakness.late-night"} <= {
        i.id for i in analyse(habits, habit_world, tz="Etc/UTC").insights
    }


def test_every_finding_has_formats_and_a_picture_for_the_demo_player(demo_games):
    for module in (openings, habits):
        for tz in ("Etc/UTC", "Europe/Berlin"):
            mr = analyse(module, demo_games, tz=tz)
            assert mr.insights, module.KEY
            check_visuals(mr, len(demo_games))
    mr = analyse(openings, demo_games)
    assert mr.diagrams and len(mr.diagrams) <= openings.Thresholds().max_choice_diagrams
    assert all(len(i.formats) > 1 for i in mr.insights)  # the demo player plays several formats


# --------------------------------------------------------------------------- the chart says what the text says
def first_values(ins) -> list:
    return [s.values[0] for s in ins.chart.series]


def test_opening_chart_numbers_agree_with_the_finding(opening_world):
    mr = analyse(openings, opening_world)
    caro = find(mr, "openings.weakness.black.caro-kann-defense")
    you, rest = first_values(caro)
    assert you == pytest.approx(caro.evidence["delta"]) and rest == pytest.approx(caro.evidence["rest_delta"])
    assert vs_rating(you) in caro.detail
    assert caro.chart.labels[0] == "All games" and set(caro.chart.labels[1:]) <= {"Bullet", "Blitz", "Rapid"}
    first = caro.chart.table.rows[0]
    assert first[2] == caro.evidence["n_rated"] and f"You score {pct(first[3])} in {first[2]} games" in caro.detail
    assert sum(caro.formats.values()) == caro.evidence["n_rated"]
    for label, value in zip(caro.chart.labels[1:], caro.chart.series[0].values[1:]):  # each format's own games
        games = [g for g in opening_world if g.color == "black" and g.opening_family == "Caro-Kann Defense"
                 and g.time_class == label.lower() and g.expected_score is not None]
        assert value == pytest.approx(sum(g.score - g.expected_score for g in games) / len(games))

    quick = find(mr, "openings.weakness.early-losses.white.italian-game")
    losses, wins = first_values(quick)
    ev = quick.evidence
    assert losses == pytest.approx(ev["short_losses"] / ev["losses"]) and wins == pytest.approx(ev["short_wins"] / ev["wins"])
    assert f"({pct(losses)})" in quick.detail and f"({pct(wins)})" in quick.detail
    assert quick.chart.labels[-1] == "Other openings"
    other_losses, other_wins = quick.chart.series[0].values[-1], quick.chart.series[1].values[-1]
    assert abs(other_losses - other_wins) == pytest.approx(abs(ev["rest_gap"]))
    assert sum(quick.formats.values()) == ev["losses"] + ev["wins"]

    for colour in ("white", "black"):
        breadth = find(mr, f"openings.observation.breadth-{colour}")
        assert first_values(breadth) == list(breadth.evidence["choices_for_coverage"].values())


def test_habit_chart_numbers_agree_with_the_finding(habit_world):
    mr = analyse(habits, habit_world, tz="Etc/UTC")
    for id_ in ("habits.weakness.after-a-loss", "habits.weakness.late-night"):
        ins = find(mr, id_)
        you, rest = first_values(ins)
        assert you == pytest.approx(ins.evidence["delta"]) and rest == pytest.approx(ins.evidence["rest_delta"])
        assert vs_rating(you) in ins.detail and per100_games(rest) in ins.detail
        assert sum(ins.formats.values()) == ins.evidence["n"]
        assert ins.chart.value_format == "signed_pct" and ins.chart.reference == 0.0
    streak = find(mr, "habits.observation.losing-streak")
    assert first_values(streak) == [streak.evidence["longest"]]
    timeline = habits.build_timeline(habit_world)
    for label, value in zip(streak.chart.labels[1:], streak.chart.series[0].values[1:]):
        # a format's own games in order: a win in another format in between doesn't end its streak
        assert value == len(habits.longest_losing_streak([t for t in timeline if t.game.time_class == label.lower()]))


def test_fatigue_and_rematch_charts_agree_with_the_finding():
    games = []
    for s in range(15):
        outcomes = ["win", "loss", "win", "loss", "loss", "loss", "loss", "win" if s % 3 == 0 else "loss"]
        games += sequence(T0 + timedelta(days=s), outcomes, gap_min=20)
    mr = analyse(habits, games)
    ins = find(mr, "habits.weakness.long-sessions")
    assert first_values(ins) == pytest.approx([ins.evidence["delta"], ins.evidence["earlier_delta"]])
    assert ins.chart.labels == ["Blitz"] and ins.formats == {"blitz": ins.evidence["n"]}

    games = []
    for i in range(6):
        start = T0 + timedelta(days=i)
        games += [game_at(start, outcome="loss", opponent="Rival"),
                  game_at(start + timedelta(minutes=10), outcome="win" if i % 2 else "loss", opponent="rival")]
    games += sequence(T0 + timedelta(days=10), ["win", "loss", "draw"] * 5)
    rematch = find(analyse(habits, games), "habits.observation.rematches")
    you, predicted = first_values(rematch)
    assert f"scored {pct(you)}" in rematch.detail and f"predicts {pct(predicted)}" in rematch.detail
    assert rematch.chart.series[0].name == "You" and rematch.formats == {"blitz": 6}


# --------------------------------------------------------------------------- split by format
def game_at(start: datetime, minutes: float = 8.0, **kw):
    return make_game(start_time=start, end_time=start + timedelta(minutes=minutes), **kw)


def sequence(start: datetime, outcomes: list[str], gap_min: float = 2.0, minutes: float = 8.0, **kw) -> list:
    games, t = [], start
    for o in outcomes:
        games.append(game_at(t, minutes, outcome=o, **kw))
        t += timedelta(minutes=minutes + gap_min)
    return games


def tilt_games(after_loss_losses: int, time_class: str, day0: int, seed: int) -> list:
    """120 two-game sessions: the first game alternates W/L; the second starts 5 min after it."""
    rng = random.Random(seed)
    after_loss = ["loss"] * after_loss_losses + ["win"] * (60 - after_loss_losses)
    rng.shuffle(after_loss)
    after_win = ["loss", "win"] * 30
    base = {"blitz": 300, "rapid": 600, "bullet": 60}[time_class]
    games = []
    for i in range(120):
        first = "loss" if i % 2 else "win"
        second = after_loss.pop() if first == "loss" else after_win.pop()
        games += sequence(T0 + timedelta(days=day0, hours=7 * i), [first, second], gap_min=5,
                          time_class=time_class, time_control=str(base), base_seconds=base)
    return games


def test_tilt_chart_shows_which_format_it_holds_in():
    # after a loss: 50 losses in 60 blitz games, 30 in 60 rapid games (no tilt in rapid)
    games = tilt_games(50, "blitz", 0, 1) + tilt_games(30, "rapid", 100, 2)
    ins = find(analyse(habits, games), "habits.weakness.after-a-loss")
    assert ins is not None and ins.formats == {"blitz": 60, "rapid": 60}
    chart = ins.chart
    assert chart.labels == ["All games", "Blitz", "Rapid"]
    after, other = (dict(zip(chart.labels, s.values)) for s in chart.series)
    assert after["Blitz"] == pytest.approx(10 / 60 - 0.5) and after["Rapid"] == pytest.approx(0.0)
    assert after["All games"] == pytest.approx(40 / 120 - 0.5) and other["Blitz"] == pytest.approx(0.0)
    assert "Blitz 60 · Rapid 60 rated games" in chart.note
    rows = {(r[0], r[1]): r for r in chart.table.rows}
    assert rows[("Blitz", "After a loss")][2] == 60 and rows[("Rapid", "Other games")][2] == 180


def test_single_format_findings_have_one_bar_named_after_the_format():
    ins = find(analyse(habits, tilt_games(45, "bullet", 0, 1)), "habits.weakness.after-a-loss")
    assert ins.chart.labels == ["Bullet"] and ins.formats == {"bullet": 60}
    assert "Bullet only (60 rated games)" in ins.chart.note


def test_small_formats_get_no_bar_of_their_own():
    games = tilt_games(45, "blitz", 0, 1)
    games += sequence(T0 + timedelta(days=200), ["loss", "loss", "loss"], gap_min=3, time_class="rapid",
                      time_control="600", base_seconds=600)  # 2 rapid games after a loss
    ins = find(analyse(habits, games), "habits.weakness.after-a-loss")
    assert ins.chart.labels == ["All games", "Blitz"] and ins.formats == {"blitz": 60, "rapid": 2}
    assert "Rapid: fewer than 10 rated games, no bar of its own" in ins.chart.note


@pytest.mark.parametrize("n", [0, 1, 3, 12, 40])
def test_tiny_and_odd_inputs_do_not_crash(n):
    rng = random.Random(n)
    games = [
        make_game(outcome=rng.choice(["win", "loss", "draw"]), time_class=rng.choice(["blitz", "rapid"]),
                  color=rng.choice(["white", "black"]), moves_san=rng.choice([CARO, ITALIAN, RUY * 3, ["e4"], []]),
                  opening_family=rng.choice([None, "Caro-Kann Defense", "Italian Game"]),
                  my_rating=rng.choice([1500, None]))
        for _ in range(n)
    ]
    for module in (openings, habits):
        mr = analyse(module, games, tz="Europe/Paris")
        check_visuals(mr, n)
    assert isinstance(openings.choice_positions(AnalysisContext("t", games)), list)


# --------------------------------------------------------------------------- opening boards
def caro_games(moves_by_game: list[list[str]], losses: int, **kw) -> list:
    return [
        make_game(outcome="loss" if i < losses else "win", moves_san=moves, **{**CARO_KW, **kw})
        for i, moves in enumerate(moves_by_game)
    ]


def weak_caro_world(caro_moves: list[list[str]]) -> list:
    """Caro-Kann games scoring badly next to other Black games scoring as expected: a Caro-Kann weakness."""
    caro = caro_games(caro_moves, losses=int(len(caro_moves) * 0.8))
    others = [make_game(color="black", opening_family="French Defense", opening="French Defense",
                        moves_san=["e4", "e6", "d4", "d5"], outcome="win" if i % 2 else "loss") for i in range(30)]
    whites = [make_game(color="white", outcome="win" if i % 2 else "loss") for i in range(30)]
    return caro + others + whites


def test_opening_finding_shows_the_usual_position_of_your_games():
    mr = analyse(openings, weak_caro_world([CARO] * 30))
    ins = find(mr, "openings.weakness.black.caro-kann-defense")
    d = ins.diagram
    board = chess.Board()
    for san in CARO:
        board.push_san(san)
    assert d.fen == board.fen() and d.orientation == "black" and d.last_move == "c6c5"
    assert d.title == "Caro-Kann Defense as Black: your usual position after 5...c5"
    assert d.caption.startswith("30 of your 30 games in the Caro-Kann Defense as Black reached this position")
    strip = d.strips[0]
    assert strip.title == "Your main line: 1.e4 c6 2.d4 d5 3.e5 Bf5 4.Nf3 e6 5.Be2 c5"
    assert [f.move for f in strip.frames] == ["3.e5", "3...Bf5", "4.Nf3", "4...e6", "5.Be2", "5...c5"]
    assert strip.frames[-1].fen == d.fen
    assert d.link in ins.example_games  # a recent loss in the opening


def test_usual_position_stops_where_your_games_part_ways():
    branches = [
        ["e4", "c6", "d4", "d5", "e5", "Bf5"],
        ["e4", "c6", "d4", "d5", "Nc3", "dxe4"],
        ["e4", "c6", "d4", "d5", "exd5", "cxd5"],
    ]
    assert openings.main_line([make_game(moves_san=m) for m in branches * 10]) == ["e4", "c6", "d4", "d5"]
    # 12 of 30 on one branch: no majority after 2...d5; the board shows the position after 2...d5
    mr = analyse(openings, weak_caro_world(branches * 10))
    d = find(mr, "openings.weakness.black.caro-kann-defense").diagram
    assert d.last_move == "d7d5" and "after 2...d5" in d.title
    assert [a.kind for a in d.arrows] == ["line"]  # the most common next move (ties: alphabetical)
    # nothing shared by most games: the most played path while a quarter of the games follow it
    three = [make_game(moves_san=m) for m in (["e4", "e5"], ["d4", "d5"], ["c4", "e5"])] * 3
    assert openings.main_line(three) == [] and openings.usual_line(three) == ["c4", "e5"]  # ties: alphabetical
    five = [make_game(moves_san=[m, "e5"]) for m in ("e4", "d4", "c4", "Nf3", "g3")] * 2
    assert openings.usual_line(five) == []
    assert openings.usual_line(
        [make_game(moves_san=m) for m in [["e4", "e5"]] * 4 + [["d4", "d5"]] * 3 + [["c4", "e5"]] * 3]
    ) == ["e4", "e5"]


# --------------------------------------------------------------------------- choice points
def choice_world() -> list:
    """After 1.e4 e5 2.Nf3 Nc6 you play 3.Bc4 (the Italian, scoring well) or 3.Bb5 (the Ruy Lopez, badly)."""
    italian = [make_game(color="white", moves_san=ITALIAN, outcome="win" if i % 5 else "loss") for i in range(30)]
    ruy = [
        make_game(color="white", opening_family="Ruy Lopez Opening", opening="Ruy Lopez Opening: Morphy Defense",
                  moves_san=RUY, outcome="win" if i % 5 == 0 else "loss", time_class="rapid" if i % 2 else "blitz")
        for i in range(30)
    ]
    return italian + ruy


def test_choice_points_are_a_reusable_list_of_positions():
    games = choice_world()
    positions = openings.choice_positions(AnalysisContext("t", games))
    pos = next(p for p in positions if p.moves_before == ["e4", "e5", "Nf3", "Nc6"])
    board = chess.Board()
    for san in pos.moves_before:
        board.push_san(san)
    assert pos.fen == board.fen() and pos.ply == 4 and pos.color == "white" and pos.last_move == "b8c6"
    assert pos.where == "After 1.e4 e5 2.Nf3 Nc6" and pos.total == 60
    assert [(m.san, m.uci, m.label, m.games) for m in pos.moves] == [("Bb5", "f1b5", "3.Bb5", 30),
                                                                     ("Bc4", "f1c4", "3.Bc4", 30)]
    bb5, bc4 = pos.moves
    assert bb5.score == pytest.approx(0.2) and bc4.score == pytest.approx(0.8)
    assert bb5.delta == pytest.approx(-0.3) and bc4.fair > bb5.fair
    assert bb5.formats == {"blitz": 15, "rapid": 15} and pos.formats == {"blitz": 45, "rapid": 15}
    assert len(bb5.urls) == 30 and bb5.urls[0] == max((g for g in games if g.moves_san[4] == "Bb5"),
                                                         key=lambda g: g.end_time).url
    assert pos.best == "Bc4" and pos.engine_best is None and pos.opening_family == "Italian Game"
    assert pos.epd == " ".join(board.fen().split()[:4])
    assert openings.choice_positions(AnalysisContext("t", games), max_points=1) == positions[:1]


def test_choice_points_become_boards_with_your_moves_as_arrows():
    mr = analyse(openings, choice_world())
    d = next(d for d in mr.diagrams if "after 1.e4 e5 2.Nf3 Nc6 (White)" in d.title)
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("f1", "b5", "played"), ("f1", "c4", "best")]
    assert d.caption.startswith(
        "3.Bb5: 30 games, 20% (−30 per 100 games vs your rating); 3.Bc4: 30 games, 80% (+30 per 100 games vs your "
        "rating). 3.Bc4 has scored best for you so far (fair estimate)."
    )
    assert "Blitz 45 · Rapid 15 games" in d.caption and d.orientation == "white" and d.last_move == "b8c6"
    table = next(t for t in mr.tables if t.title == "Your choices at key moves")
    assert "drawn as a board" in table.note and "Formats: Blitz 45 · Rapid 15 games." in table.note
    # the Ruy Lopez finding's advice is this choice: its board carries the choice with both moves
    ins = find(mr, "openings.weakness.white.ruy-lopez-opening")
    assert ins.study[0].startswith("After 1.e4 e5 2.Nf3 Nc6 you have a choice")
    assert ins.diagram.fen != d.fen  # the main board: the Ruy Lopez position your games reach
    choice = ins.diagram.strips[-1]
    assert choice.title == "Your choice after 1.e4 e5 2.Nf3 Nc6: 3.Bc4 or 3.Bb5"
    frame = choice.frames[0]
    assert frame.fen == d.fen and frame.move == "2...Nc6" and frame.caption == "3.Bc4 80% in 30 games, 3.Bb5 20% in 30"
    assert [(a.start, a.end, a.kind) for a in frame.arrows] == [("f1", "b5", "played"), ("f1", "c4", "best")]


def test_the_engine_preferred_move_is_the_green_arrow():
    games = choice_world()
    evals = {
        g.game_id: make_game_eval(g.game_id, [make_ply_eval(ply=4, san=g.moves_san[4], best_san="d4")]) for g in games[:40]
    }
    ctx = AnalysisContext("t", sorted(games, key=lambda g: g.end_time), evals=evals)
    pos = openings.choice_positions(ctx)[0]
    assert pos.engine_best == "d4" and pos.best == "d4"
    d = openings.choice_diagram(pos)
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("f1", "b5", "played"), ("d2", "d4", "best"),
                                                            ("f1", "c4", "neutral")]
    assert "Stockfish prefers 3.d4." in d.caption


def test_when_your_usual_move_scores_best_it_is_green_not_red():
    games = choice_world()
    for g in list(games):  # the Italian becomes the more played move
        if g.moves_san[4] == "Bc4":
            games.append(make_game(color="white", moves_san=ITALIAN, outcome="win", end_time=g.end_time))
    pos = openings.choice_positions(AnalysisContext("t", sorted(games, key=lambda g: g.end_time)))[0]
    assert pos.usual.san == "Bc4" and pos.best == "Bc4"
    assert [a.kind for a in openings.choice_diagram(pos).arrows] == ["best", "neutral"]


# --------------------------------------------------------------------------- module tables and charts name their formats
def test_module_tables_and_charts_say_which_formats_they_mix(opening_world, habit_world):
    mr = analyse(openings, opening_world)
    for t in mr.tables:
        assert "Formats: Bullet" in t.note and "Blitz" in t.note and "Rapid" in t.note, t.title
    assert "Formats: Bullet" in mr.charts[0].note
    per_format = next(c for c in mr.charts if c.title == "Score vs rating by opening, per format")
    assert [s.name for s in per_format.series] == ["Bullet", "Blitz", "Rapid"]
    assert mr.stats["formats"] == {"bullet": 184, "blitz": 721, "rapid": 295}

    mr = analyse(habits, habit_world, tz="Etc/UTC")
    for c in mr.charts:
        assert "Formats: Bullet" in c.note
        assert {"Bullet vs rating", "Blitz vs rating", "Rapid vs rating"} <= set(c.table.columns)
    weekday = next(t for t in mr.tables if t.title == "By weekday")
    rows = {r[0]: dict(zip(weekday.columns, r)) for r in weekday.rows}
    monday = [g for g in habit_world if g.time_class == "rapid" and g.plies >= 4 and g.expected_score is not None
              and (g.start_time or g.end_time).weekday() == 0]
    assert rows["Monday"]["Rapid vs rating"] == pytest.approx(
        sum(g.score - g.expected_score for g in monday) / len(monday))
    assert "Formats:" in next(t for t in mr.tables if t.title == "Session lengths").note


def test_one_format_needs_no_format_notes():
    games = [g for g in null_games(400, seed=9) if g.time_class == "blitz"]
    mr = analyse(openings, games)
    assert not any("Formats:" in t.note for t in mr.tables) and len(mr.charts) == 1
    mr = analyse(habits, games)
    assert not any("Formats:" in c.note for c in mr.charts)
    assert all("Blitz vs rating" not in c.table.columns for c in mr.charts)
