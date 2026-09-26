"""Formats and pictures on every finding of results, time_mgmt and endings.

Every finding carries the games per format behind it and a small chart (and, for how games end, a
board of a real final position); the chart's numbers are the ones in the finding's text. Pictures
never change which findings are made.
"""

from __future__ import annotations

import dataclasses
import random
from datetime import datetime, timedelta, timezone

import chess
import pytest

from chess_insights import parse, synth, visuals
from chess_insights.analysis import endings, results, time_mgmt
from chess_insights.context import AnalysisContext
from chess_insights.models import TIME_CLASSES, VALUE_FORMATS
from chess_insights.stats import pct, per100_games
from factories import make_game

CLOCKS = {"bullet": 60, "blitz": 300, "rapid": 600}
KNIGHTS = ["Nf3", "Nf6", "Ng1", "Ng8"]  # legal for as long as you like
FOOLS_MATE = ["f3", "e5", "g4", "Qh4#"]  # White is mated
SCHOLARS_MATE = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]  # White mates


def fmt(tc: str) -> dict:
    """Keyword arguments for a game in time class ``tc``."""
    return {"time_class": tc, "time_control": str(CLOCKS[tc]), "base_seconds": CLOCKS[tc]}


def run(module, games, **options):
    return module.analyze(AnalysisContext("tester", sorted(games, key=lambda g: g.end_time), options=options))


def games_by(n_wins: int, n_losses: int, n_draws: int = 0, **kw) -> list:
    outcomes = ["win"] * n_wins + ["loss"] * n_losses + ["draw"] * n_draws
    outcomes = [outcomes[(i * 7) % len(outcomes)] for i in range(len(outcomes))] if len(outcomes) % 7 else outcomes
    return [make_game(outcome=o, **kw) for o in outcomes]


def finding(mr, prefix: str):
    return next(i for i in mr.insights if i.id.startswith(prefix))


def series(chart, name: str) -> list:
    return next(s.values for s in chart.series if s.name == name)


def check_diagram(d, ins) -> None:
    board = chess.Board(d.fen)  # a legal position
    assert d.title and d.caption.startswith("Final position:") and d.link in ins.example_games
    assert d.time_class in TIME_CLASSES and d.orientation in ("white", "black")
    last = chess.Move.from_uci(d.last_move)
    assert board.piece_at(last.to_square) is not None  # the highlighted move landed there


def check_visuals(mr) -> None:
    """Every finding names its formats and has a small, well-formed chart or board."""
    for ins in mr.insights:
        assert ins.formats and all(n > 0 for n in ins.formats.values()), ins.id
        assert set(ins.formats) <= set(TIME_CLASSES) and list(ins.formats) == list(visuals.ordered_formats(ins.formats))
        assert ins.chart is not None or ins.diagram is not None, ins.id
        c = ins.chart
        if c is not None:
            assert c.title and c.note and c.value_format in VALUE_FORMATS, ins.id
            assert 1 <= len(c.labels) <= 6 and 1 <= len(c.series) <= 3, (ins.id, c.labels, len(c.series))
            assert all(len(s.values) == len(c.labels) for s in c.series)
            assert any(v is not None for s in c.series for v in s.values), ins.id
            if c.table is not None:
                assert all(len(r) == len(c.table.columns) for r in c.table.rows)
                assert c.table.formats is None or len(c.table.formats) == len(c.table.columns)
        if ins.diagram is not None:
            check_diagram(ins.diagram, ins)
    for chart in mr.charts:
        assert chart.series and all(len(s.values) == len(chart.labels) for s in chart.series), chart.title
        if chart.table is not None:
            assert all(len(r) == len(chart.table.columns) for r in chart.table.rows), chart.table.title


# --------------------------------------------------------------------------- results
def test_colour_finding_is_split_by_format_and_matches_its_text():
    games = []
    for tc in ("bullet", "blitz", "rapid"):
        games += games_by(10, 4, color="white", **fmt(tc)) + games_by(4, 10, color="black", **fmt(tc))
    mr = run(results, games)
    check_visuals(mr)
    ins = finding(mr, "results.weakness.colour-black")
    assert ins.formats == {"bullet": 28, "blitz": 28, "rapid": 28}
    chart = ins.chart
    assert chart.labels == ["All formats", "Bullet", "Blitz", "Rapid"] and chart.value_format == "signed_pct"
    white, black = series(chart, "White"), series(chart, "Black")
    assert white[0] == pytest.approx(ins.evidence["white"]["delta"])
    assert black[0] == pytest.approx(ins.evidence["black"]["delta"])
    assert per100_games(white[0]) in ins.detail and per100_games(black[0]) in ins.detail
    assert white[1:] == pytest.approx([10 / 14 - 0.5] * 3) and black[1:] == pytest.approx([4 / 14 - 0.5] * 3)
    assert chart.table.rows[1][:2] == ["Bullet", 14]  # the games behind each bar
    # the section's own colour chart says the same, per format
    module_chart = next(c for c in mr.charts if c.title == "Score vs rating with White and Black")
    assert module_chart.labels == chart.labels
    assert "Bullet 28 · Blitz 28 · Rapid 28 games" in next(t for t in mr.tables if t.title == "By colour").note


def test_opponent_findings_compare_with_your_other_games_per_format():
    games = []
    for tc, higher in (("bullet", (7, 3)), ("blitz", (7, 3)), ("rapid", (6, 4))):
        games += games_by(4, 6, my_rating=1500, opp_rating=1350, **fmt(tc))
        games += games_by(*higher, my_rating=1500, opp_rating=1650, **fmt(tc))
    mr = run(results, games)
    check_visuals(mr)
    weak = finding(mr, "results.weakness.lower-rated-opponents")
    assert weak.formats == {"bullet": 20, "blitz": 20, "rapid": 20}
    chart = weak.chart
    assert chart.labels == ["All formats", "Bullet", "Blitz", "Rapid"]
    group, rest = series(chart, "Opponents 50+ below you"), series(chart, "Your other games")
    assert group[0] == pytest.approx(weak.evidence["delta"]) and rest[0] == pytest.approx(weak.evidence["rest_delta"])
    assert per100_games(group[0]) in weak.detail and per100_games(rest[0]) in weak.detail
    strong = finding(mr, "results.strength.higher-rated-opponents")
    assert series(strong.chart, "Opponents 50+ above you")[0] == pytest.approx(strong.evidence["delta"])
    # the section's opponent-strength chart gets one series per format
    buckets = next(c for c in mr.charts if c.title.startswith("Score vs realistic"))
    assert [s.name for s in buckets.series] == ["All formats", "Bullet", "Blitz", "Rapid"]
    assert buckets.table.columns[:3] == ["Opponent vs you", "Bullet: games", "Bullet: vs rating"]


def test_time_control_finding_charts_every_time_control():
    rapid = games_by(30, 10, **fmt("rapid"))
    blitz = games_by(14, 26, **fmt("blitz"))
    mr = run(results, rapid + blitz)
    check_visuals(mr)
    ins = finding(mr, "results.strength.time-control-rapid")
    assert ins.formats == {"blitz": 40, "rapid": 40}
    assert ins.chart.labels == ["Rapid", "Blitz"]
    you, predicted = series(ins.chart, "You"), series(ins.chart, "Rating predicts")
    assert you == pytest.approx([0.75, 0.35]) and predicted == pytest.approx([0.5, 0.5])
    assert all(pct(v) in ins.detail for v in you + predicted)


def test_rating_trend_finding_is_a_small_line_of_that_format_only():
    start = datetime(2024, 3, 1, tzinfo=timezone.utc)
    games = [
        make_game(
            outcome="win" if i % 4 else "loss",
            my_rating=1500 + 3 * i,
            opp_rating=1500 + 3 * i,
            end_time=start + timedelta(days=2 * i),
        )
        for i in range(30)
    ]
    games += games_by(5, 5, **fmt("rapid"), end_time=start)
    mr = run(results, games)
    check_visuals(mr)
    trend = finding(mr, "results.observation.trend-blitz")
    assert trend.formats == {"blitz": 30}
    chart = trend.chart
    assert chart.kind == "line" and len(chart.labels) == results.TREND_POINTS
    actual, line = series(chart, "Your rating"), series(chart, "Straight line through your games")
    assert (actual[0], actual[-1]) == (trend.evidence["first"], trend.evidence["last"]) == (1500, 1587)
    assert line[-1] - line[0] == pytest.approx(trend.evidence["change"], abs=0.2)
    assert "Blitz only: 30 rated games" in chart.note


def test_accuracy_finding_numbers_match_the_title():
    games = []
    for tc in ("bullet", "blitz"):
        games += [make_game(outcome="win", my_accuracy=85.0, opp_accuracy=70.0, **fmt(tc)) for _ in range(6)]
        games += [make_game(outcome="loss", my_accuracy=65.0, opp_accuracy=80.0, **fmt(tc)) for _ in range(4)]
    mr = run(results, games)
    check_visuals(mr)
    ins = finding(mr, "results.observation.chesscom-accuracy")
    assert ins.formats == {"bullet": 10, "blitz": 10}
    assert ins.chart.labels == ["All formats", "Bullet", "Blitz"]
    firsts = [s.values[0] for s in ins.chart.series]
    assert firsts == pytest.approx([77.0, 85.0, 65.0])
    assert all(f"{v:.1f}" in ins.title for v in firsts)


def test_section_charts_and_kpis_say_which_formats_they_cover():
    games = games_by(20, 20, **fmt("blitz")) + games_by(12, 8, **fmt("rapid"))
    mr = run(results, games)
    check_visuals(mr)
    assert next(k for k in mr.kpis if k.label == "Games").hint.endswith("Blitz 40 · Rapid 20 games")
    delta = next(c for c in mr.charts if c.title == "Score vs rating by time control")
    assert delta.labels == ["All formats", "Blitz", "Rapid"] and "Blitz 40 · Rapid 20 rated games" in delta.note
    rating = next(c for c in mr.charts if c.kind == "line")
    assert [s.name for s in rating.series] == ["Blitz", "Rapid"]  # one rating line per format


# --------------------------------------------------------------------------- time_mgmt
def timed_game(my_spent, opp_spent, *, color="white", base=300, inc=0, **kw):
    white, black = (my_spent, opp_spent) if color == "white" else (opp_spent, my_spent)
    remaining, clocks = {"w": float(base), "b": float(base)}, []
    for k in range(max(len(white), len(black))):
        for side, spent in (("w", white), ("b", black)):
            if k < len(spent):
                remaining[side] -= spent[k] - inc
                clocks.append(round(remaining[side], 1))
    kw.setdefault("time_control", f"{base}+{inc}" if inc else str(base))
    return make_game(color=color, base_seconds=base, increment=inc, moves_san=(KNIGHTS * 100)[: len(clocks)],
                     clocks=clocks, **kw)


def flag_player(base: int = 300, **kw) -> list:
    """15 losses on time, 10 other losses and 15 wins."""
    lost = [
        timed_game([5] * 12, [5] * 12, outcome="loss", termination="timeout", my_result_code="timeout", base=base, **kw)
        for _ in range(15)
    ]
    others = [timed_game([5] * 12, [5] * 12, outcome="loss", base=base, **kw) for _ in range(10)]
    return lost + others + [timed_game([5] * 12, [5] * 12, outcome="win", base=base, **kw) for _ in range(15)]


def test_time_trouble_finding_names_its_format_and_shows_the_others():
    trouble = [timed_game([5] * 14 + [250], [5] * 15, outcome="loss") for _ in range(20)]
    opp_trouble = [timed_game([5] * 15, [5] * 14 + [250], outcome="win") for _ in range(4)]
    calm = [timed_game([5] * 15, [5] * 15, outcome=o) for o in ["win", "draw", "loss", "win"] * 4]
    rapid = [timed_game([5] * 15, [5] * 15, base=600, time_class="rapid", outcome=o) for o in ["win", "loss"] * 10]
    mr = run(time_mgmt, trouble + opp_trouble + calm + rapid)
    check_visuals(mr)
    ins = finding(mr, "time.weakness.time-trouble-blitz")
    assert ins.formats == {"blitz": 40}
    assert ins.chart.labels == ["Blitz", "Rapid"]
    you, opp = series(ins.chart, "You"), series(ins.chart, "Opponents")
    assert you == pytest.approx([0.5, 0.0]) and opp == pytest.approx([0.1, 0.0])
    assert pct(you[0]) in ins.detail and pct(opp[0]) in ins.detail
    assert "about blitz" in ins.chart.note and "Blitz 40 · Rapid 20 games with clocks" in ins.chart.note


def test_losing_on_time_in_two_formats_is_one_finding_with_a_board():
    mr = run(time_mgmt, flag_player() + flag_player(base=600, time_class="rapid"))
    check_visuals(mr)
    ins = finding(mr, "time.weakness.lost-on-time-blitz-rapid")
    assert ins.formats == {"blitz": 40, "rapid": 40}
    assert ins.chart.labels == ["Blitz", "Rapid"]
    assert series(ins.chart, "Lost on time") == [15, 15] and series(ins.chart, "Won on time") == [0, 0]
    assert "15 of your 25 blitz losses" in ins.detail and "15 of your 25 rapid losses" in ins.detail
    board = ins.diagram
    assert board is not None and board.link == ins.example_games[0]
    assert "you lost on time after 12 moves" in board.caption and "Material was level." in board.caption
    assert board.time_class in ("blitz", "rapid") and board.last_move == "f6g8"


def test_opening_pace_and_clock_handling_findings():
    slow = [timed_game([12] * 15 + [2] * 5, [5] * 15 + [2] * 5, outcome=o) for o in ["win", "loss"] * 10]
    mr = run(time_mgmt, slow)
    check_visuals(mr)
    ins = finding(mr, "time.weakness.slow-opening-blitz")
    assert ins.formats == {"blitz": 20} and ins.chart.labels == ["Blitz"]
    you, opp = series(ins.chart, "You"), series(ins.chart, "Opponents")
    assert you == pytest.approx([0.6]) and opp == pytest.approx([0.25])
    assert pct(you[0]) in ins.detail and pct(opp[0]) in ins.detail
    pace = next(c for c in mr.charts if c.title.startswith("Clock used on the first"))
    assert pace.table is not None and pace.table.title == "Opening pace and clock balance"

    flags = [
        timed_game([3] * 25, [6] * 25, outcome="win", termination="timeout", opp_result_code="timeout")
        for _ in range(14)
    ]
    rest = [timed_game([3] * 25, [6] * 25, outcome=o) for o in ["win", "loss", "draw"] * 5]
    rest.append(timed_game([3] * 25, [6] * 25, outcome="loss", termination="timeout", my_result_code="timeout"))
    mr = run(time_mgmt, flags + rest)
    check_visuals(mr)
    ins = finding(mr, "time.strength.clock-handling-blitz")
    assert ins.formats == {"blitz": 30}
    ahead = series(ins.chart, "You ahead")
    assert ahead == pytest.approx([1.0]) and f"{pct(ahead[0])} of 30 blitz games" in ins.detail


# --------------------------------------------------------------------------- endings
def mate_player(formats=("bullet", "blitz", "rapid")) -> list:
    """Mated in about half the losses, mating in a tenth of the wins; spread over ``formats``."""
    games = []
    for i, tc in enumerate(formats):
        k = len(formats)
        games += [make_game(outcome="loss", termination="checkmate", my_result_code="checkmated",
                            moves_san=FOOLS_MATE, **fmt(tc)) for _ in range(16 // k + (i < 16 % k))]
        games += [make_game(outcome="loss", **fmt(tc)) for _ in range(14 // k + (i < 14 % k))]
        games += [make_game(outcome="win", termination="checkmate", opp_result_code="checkmated",
                            moves_san=SCHOLARS_MATE, **fmt(tc)) for _ in range(3 // k + (i < 3 % k))]
        games += [make_game(outcome="win", **fmt(tc)) for _ in range(27 // k + (i < 27 % k))]
    return games


def test_checkmated_often_has_a_split_chart_and_a_mated_board():
    games = mate_player()
    mr = run(endings, games)
    check_visuals(mr)
    ins = finding(mr, "endings.observation.checkmated-often")
    assert ins.formats == visuals.format_counts(games) and sum(ins.formats.values()) == 60  # every decisive game
    losses, wins = series(ins.chart, "Your losses"), series(ins.chart, "Your wins")
    assert ins.chart.labels == ["All formats", "Bullet", "Blitz", "Rapid"]
    assert (losses[0], wins[0]) == pytest.approx((16 / 30, 3 / 30))
    assert ins.title == f"{pct(losses[0])} of your losses end in checkmate, against {pct(wins[0])} of your wins"
    board = ins.diagram
    assert chess.Board(board.fen).is_checkmate() and board.last_move == "d8h4" and board.orientation == "white"
    assert [(m.square, m.kind) for m in board.marks] == [("e1", "check")]
    assert "you were checkmated after 2 moves" in board.caption and board.link == ins.example_games[0]


def test_abandoned_draw_and_length_findings():
    games = [make_game(outcome="loss", termination="abandoned", my_result_code="abandoned", **fmt("blitz"))
             for _ in range(2)]
    games += [make_game(outcome="loss", termination="abandoned", my_result_code="abandoned", moves_san=[],
                        **fmt("rapid"))]
    games += games_by(20, 18, 4, **fmt("blitz")) + games_by(10, 10, 2, **fmt("rapid"))
    mr = run(endings, games)
    check_visuals(mr)
    ins = finding(mr, "endings.observation.abandoned-games")
    assert ins.formats == {"blitz": 44, "rapid": 23}
    mine = series(ins.chart, "You abandoned")
    assert ins.chart.labels == ["All formats", "Blitz", "Rapid"] and mine == [3, 2, 1]
    assert f"You abandoned {int(mine[0])} games" in ins.title
    assert ins.diagram is not None and "you abandoned the game" in ins.diagram.caption  # the 0-move one has no board
    draws = finding(mr, "endings.observation.draw-rate")
    rate = series(draws.chart, "Draw rate")
    assert ins.chart and pct(rate[0]) in draws.title and rate[1:] == pytest.approx([4 / 44, 2 / 23])
    assert draws.diagram is not None and "you agreed a draw" in draws.diagram.caption

    rng = random.Random(3)
    moves = lambda n: (KNIGHTS * n)[: 2 * n]  # noqa: E731
    tcs = ("blitz", "rapid")
    short = [
        make_game(outcome="loss" if i % 4 else "win", moves_san=moves(12 + i % 8), **fmt(tcs[i % 2])) for i in range(40)
    ]
    long_ = [
        make_game(outcome="win" if i % 4 else "loss", moves_san=moves(47 + i % 10), **fmt(tcs[i % 2])) for i in range(40)
    ]
    middle = [make_game(outcome=rng.choice(["win", "loss"]), moves_san=moves(35)) for _ in range(30)]
    mr = run(endings, short + long_ + middle)
    check_visuals(mr)
    ins = finding(mr, "endings.observation.length-short")
    assert ins.formats == {"blitz": 70, "rapid": 40}
    band, rest = series(ins.chart, "Games of ≤ 20 moves"), series(ins.chart, "Other lengths")
    assert band[0] == pytest.approx(ins.evidence["delta"]) and rest[0] == pytest.approx(ins.evidence["rest_delta"])
    assert per100_games(band[0]) in ins.detail and per100_games(rest[0]) in ins.detail
    length = next(c for c in mr.charts if c.title.startswith("Score vs rating by game length"))
    assert [s.name for s in length.series] == ["All formats", "Blitz", "Rapid"]
    assert length.table.columns[-2:] == ["vs rating: Blitz", "vs rating: Rapid"]


# --------------------------------------------------------------------------- small, single-format and odd inputs
@pytest.mark.parametrize("module", [results, time_mgmt, endings])
def test_tiny_inputs_do_not_crash(module):
    daily = make_game(time_class="daily", time_control="1/86400", base_seconds=86400)
    for games in ([], [make_game()], [make_game(outcome=o) for o in ("win", "loss", "draw")], [daily]):
        check_visuals(run(module, games))


def test_single_format_findings_name_the_format_instead_of_all_formats():
    games = games_by(28, 12, color="white") + games_by(12, 28, color="black")
    ins = finding(run(results, games), "results.weakness.colour-black")
    assert ins.formats == {"blitz": 80} and ins.chart.labels == ["Blitz"]
    mr = run(endings, mate_player(("blitz",)))
    check_visuals(mr)
    assert finding(mr, "endings.observation.checkmated-often").chart.labels == ["Blitz"]


def test_a_game_that_cannot_be_replayed_gets_no_board():
    broken = make_game(outcome="loss", termination="checkmate", moves_san=["e4", "e4", "e4", "e4"])
    assert endings.final_position_diagram(broken, "x") is None
    assert endings.final_position_diagram(make_game(moves_san=["e4", "e5"]), "x") is None  # too short to say much
    mate = make_game(color="black", outcome="win", termination="checkmate", moves_san=FOOLS_MATE)
    board = endings.final_position_diagram(mate, "x")
    assert board.orientation == "black" and "you checkmated your opponent" in board.caption


def claims_of(mr) -> list[tuple]:
    return [(i.id, i.kind, i.title, i.detail, i.severity, i.confidence, i.example_games) for i in mr.insights]


def test_pictures_never_change_the_findings(monkeypatch):
    scenarios = [
        (results, games_by(28, 12, color="white", **fmt("rapid")) + games_by(12, 28, color="black")),
        (time_mgmt, flag_player() + flag_player(base=600, time_class="rapid")),
        (endings, mate_player()),
    ]
    for module, games in scenarios:
        with_pictures = run(module, games)
        with monkeypatch.context() as m:
            def broken(*args, **kwargs):
                raise RuntimeError("no pictures today")

            for name in ("split_chart", "split_series"):
                m.setattr(results, name, broken)
            m.setattr(endings, "split_chart", broken)
            m.setattr(endings, "first_diagram", broken)
            m.setattr(time_mgmt, "finding_chart", broken)
            m.setattr(time_mgmt, "first_diagram", broken)
            m.setattr(results, "trend_chart", broken)
            without = run(module, games)
        assert claims_of(without) == claims_of(with_pictures) and with_pictures.insights
        assert all(ins.formats for ins in without.insights)  # the formats never depend on a picture


# --------------------------------------------------------------------------- the synthetic demo player
@pytest.fixture(scope="module")
def demo_games():
    archives = synth.generate_archives(260, seed=7, engine_path=None, workers=2)
    raw = [g for month in archives.values() for g in month]
    return sorted(parse.parse_games(raw, synth.DEFAULT_PERSONA.username), key=lambda g: g.end_time)


def test_demo_player_every_finding_has_formats_and_a_picture(demo_games):
    found = []
    for module in (results, time_mgmt, endings):
        mr = run(module, demo_games)
        check_visuals(mr)
        found += mr.insights
    assert len(found) >= 4
    assert any(ins.diagram is not None for ins in found)  # at least one board of a real final position
    everything = visuals.format_counts(demo_games)
    for ins in found:
        assert all(ins.formats[tc] <= everything[tc] for tc in ins.formats), ins.id


def test_demo_player_per_format_views(demo_games):
    for tc, n in visuals.format_counts(demo_games).items():
        subset = [g for g in demo_games if g.time_class == tc]
        for module in (results, time_mgmt, endings):
            mr = run(module, subset)
            check_visuals(mr)
            for ins in mr.insights:
                assert list(ins.formats) == [tc], (tc, ins.id)
                if ins.chart is not None and ins.id.split(".")[0] != "time":
                    assert "All formats" not in ins.chart.labels


def test_merged_clock_finding_keeps_the_formats_of_every_class():
    blitz, rapid = flag_player(), flag_player(base=600, time_class="rapid")
    ins = finding(run(time_mgmt, blitz + rapid), "time.weakness.lost-on-time-")
    assert dataclasses.asdict(ins)["formats"] == {"blitz": 40, "rapid": 40}
