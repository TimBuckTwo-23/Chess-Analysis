"""Pictures and formats of the engine review and the repeated-mistakes section (engine_stats, mistakes).

Every finding must say which formats its games come from (``Insight.formats``) and carry a small chart of its
numbers and/or a board of the position it is about. Most tests use a cheap stand-in for Stockfish (material plus
the best capture, and mate in one) run through ``engine.build_game_eval``, so the judgements, tags, phases and
accuracies come from the real code without an engine; one small test runs the real thing.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Optional

import chess
import pytest

from chess_insights import engine, parse, synth, visuals
from chess_insights.analysis import engine_stats, mistakes
from chess_insights.context import AnalysisContext
from chess_insights.engine import MATE_CP, PositionEval, build_game_eval, material_gain
from chess_insights.models import ARROW_KINDS, TIME_CLASSES, VALUE_FORMATS, Diagram, Game, Insight, ModuleResult
from factories import make_game, make_game_eval, make_ply_eval

VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
CLOCKS = {"bullet": (60, "60"), "blitz": (180, "180"), "rapid": (600, "600")}


# --------------------------------------------------------------------------- a stand-in for Stockfish
def _material(board: chess.Board) -> int:
    return sum(VALUES[p.piece_type] * (1 if p.color == chess.WHITE else -1) for p in board.piece_map().values())


def pseudo_position(board: chess.Board) -> PositionEval:
    """White-POV "engine" verdict: mate in one if there is one, else material plus 90% of the best capture for the
    side to move; the best move is that capture (or the first legal move in UCI order)."""
    if board.is_checkmate():
        return PositionEval(cp=-MATE_CP if board.turn == chess.WHITE else MATE_CP, checkmate=True)
    if board.is_stalemate() or board.is_insufficient_material():
        return PositionEval(cp=0)
    sign = 1 if board.turn == chess.WHITE else -1
    best, gain = None, 0
    for move in board.legal_moves:
        if board.gives_check(move):
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return PositionEval(mate=sign, best=move.uci())
        g = 100 * material_gain(board, move)
        if g > gain:
            best, gain = move, g
    if best is None:
        best = min(board.legal_moves, key=lambda m: m.uci())
    return PositionEval(cp=_material(board) + sign * int(0.9 * gain), best=best.uci())


def pseudo_eval(game: Game):
    board = chess.Board(game.initial_fen or chess.STARTING_FEN)
    positions = [pseudo_position(board)]
    for san in game.moves_san:
        board.push_san(san)
        positions.append(pseudo_position(board))
    return build_game_eval(game, positions, "Stockfish 16", 12)


def random_moves(rng: random.Random, plies: int, careless: chess.Color) -> tuple[list[str], int]:
    """A legal game: the ``careless`` side plays random moves, the other takes the most valuable capture when it can.
    Returns the SAN moves and the final material balance for the careless side."""
    board = chess.Board()
    sans = []
    for _ in range(plies):
        moves = sorted(board.legal_moves, key=lambda m: m.uci())
        if not moves:
            break
        captures = [m for m in moves if board.is_capture(m)]
        if board.turn != careless and captures:
            move = max(captures, key=lambda m: (material_gain(board, m), m.uci()))
        else:
            move = rng.choice(moves)
        sans.append(board.san(move))
        board.push(move)
    balance = _material(board) * (1 if careless == chess.WHITE else -1)
    if board.is_checkmate():
        balance = -MATE_CP if board.turn == careless else MATE_CP
    return sans, balance


def realistic_pairs(seed: int = 5, n: int = 60, formats=("bullet", "blitz", "rapid")) -> list[tuple[Game, object]]:
    """Legal games across ``formats`` (in turn) in which you play carelessly against greedy opponents."""
    rng = random.Random(seed)
    pairs = []
    for k in range(n):
        tc = formats[k % len(formats)]
        base, control = CLOCKS.get(tc, (86400, "1/86400"))
        color = "white" if k % 2 == 0 else "black"
        sans, balance = random_moves(rng, rng.randint(40, 80), chess.WHITE if color == "white" else chess.BLACK)
        outcome = "win" if balance > 200 else "loss" if balance < -200 else "draw"
        game = make_game(color=color, moves_san=sans, time_class=tc, time_control=control, base_seconds=base,
                         outcome=outcome, opening_family=None, opening=None, eco=None)
        pairs.append((game, pseudo_eval(game)))
    return pairs


def analyse(pairs, games=None, **options) -> tuple[ModuleResult, ModuleResult]:
    games = sorted(games if games is not None else [g for g, _ in pairs], key=lambda g: g.end_time)
    ctx = AnalysisContext("tester", games, evals={g.game_id: ev for g, ev in pairs if ev is not None},
                          options=options)
    return engine_stats.analyze(ctx), mistakes.analyze(ctx)


# --------------------------------------------------------------------------- checks
def check_chart(chart) -> None:
    assert chart.title and chart.value_format in VALUE_FORMATS and chart.kind in ("bar", "hbar", "line", "stacked_bar")
    assert 1 <= len(chart.labels) <= 6 and 1 <= len(chart.series) <= 3, (chart.title, chart.labels)
    assert all(len(s.values) == len(chart.labels) for s in chart.series), chart.title
    assert all(v is None or math.isfinite(v) for s in chart.series for v in s.values), chart.title


def check_diagram(d: Diagram, games: dict[str, Game]) -> None:
    """The board is the game's position before one of your moves: that move red, from your side, what followed."""
    assert d.svg == "" and d.title and d.caption
    game = games[d.link]
    assert d.time_class == game.time_class and d.orientation == game.color
    board = chess.Board(game.initial_fen or chess.STARTING_FEN)
    for ply, san in enumerate(game.moves_san):
        move = board.parse_san(san)
        if board.fen() == d.fen and game.is_my_ply(ply):
            break
        board.push(move)
    else:
        raise AssertionError(f"{d.fen} is not a position of {d.link} with you to move")
    played = visuals.move_arrow(board, san, "played")
    assert (d.arrows[0].start, d.arrows[0].end, d.arrows[0].kind) == (played.start, played.end, "played")
    assert all(a.kind in ARROW_KINDS for a in d.arrows) and len({(a.start, a.end) for a in d.arrows}) == len(d.arrows)
    if len(d.arrows) > 1:
        assert d.arrows[1].kind == "best"
    [strip] = d.strips
    assert 1 <= len(strip.frames) <= mistakes.STRIP_PLIES
    assert strip.frames[0].move == visuals.move_label(board, move) and strip.frames[0].last_move == move.uci()
    assert len(strip.frames) == min(mistakes.STRIP_PLIES, game.plies - ply)  # the moves actually played


def check_numbers(ins: Insight) -> None:
    """The chart's first bar (every game, or the one format) shows the numbers of the finding's text."""
    ev, chart = ins.evidence, ins.chart
    values = {s.name: s.values for s in chart.series}
    first = {name: v[0] for name, v in values.items()}
    kind = ins.id.split(".", 2)[2]
    if ins.id.startswith("mistakes."):
        assert first[f"Played {mistakes.move_label(ev['fen'], ev['move'])}"] == ev["errors"]
        assert first["Another move"] == ev["reached"] - ev["errors"]
        assert f"{ev['errors']} of" in ins.title
    elif kind in ("blunder-rate", "missed-tactics", "hung-material") or kind.startswith("phase-"):
        assert first["You"] == pytest.approx(ev["per100"]) and first["Opponents"] == pytest.approx(ev["opp_per100"])
        assert f"{ev['per100']:.1f}" in ins.detail
    elif kind.startswith("time-pressure-blunders-"):
        assert values["You"] == [pytest.approx(ev["per100_low"]), pytest.approx(ev["per100_ok"])]
        assert chart.labels == ["Short of time", "More time"] and ins.formats.keys() == {ev["time_class"]}
    elif kind == "conversion":
        assert first["You"] == pytest.approx(ev["rate"]) and first["Opponents"] == pytest.approx(ev["opp_rate"])
    elif kind == "resilience":
        assert first["You"] == pytest.approx(ev["save_rate"])
        assert sum(ins.formats.values()) >= ev["lost_positions"]
    elif kind == "missed-mates":
        assert first["You"] == ev["missed_mates"] and first["Opponents"] == ev["opp_missed_mates"]
    elif kind == "blunder-anatomy":
        piece = chart.labels.index(ev["piece"].capitalize())
        assert values["Share of your blunders"][piece] == pytest.approx(ev["blunders"] / ev["total_blunders"])
        assert values["Share of your moves"][piece] == pytest.approx(ev["moves"] / ev["total_moves"])
    elif kind.startswith("openings-"):
        assert chart.series[0].values[0] == pytest.approx(ev["avg_cp"] / 100.0)
        assert sum(ins.formats.values()) == ev["games"]
    elif kind == "accuracy-by-result":
        named = {"Wins": "win", "Draws": "draw", "Losses": "loss"}
        assert all(first[name] == pytest.approx(ev[named[name]]) for name in values)
        assert sum(ins.formats.values()) == ev["games"]
    else:  # a new finding: add its check here
        raise AssertionError(f"no number check for {ins.id}")


def check_module(mr: ModuleResult, games: list[Game], analysed: int) -> None:
    by_url = {g.url: g for g in games}
    for ins in mr.insights:
        assert ins.formats, ins.id
        assert all(tc in TIME_CLASSES and n > 0 for tc, n in ins.formats.items()), ins.id
        assert list(ins.formats) == list(visuals.ordered_formats(ins.formats)), ins.id  # report's format order
        assert ins.chart is not None or ins.diagram is not None, ins.id
        if ins.chart is not None:
            check_chart(ins.chart)
            check_numbers(ins)
        if ins.diagram is not None:
            check_diagram(ins.diagram, by_url)
        if not ins.id.startswith("mistakes."):  # engine findings: the analysed games they used
            assert sum(ins.formats.values()) <= analysed, ins.id
    for chart in mr.charts:
        assert chart.series and all(len(s.values) == len(chart.labels) for s in chart.series), chart.title
    for d in mr.diagrams:
        check_diagram(d, by_url)


# --------------------------------------------------------------------------- realistic factory games
@pytest.fixture(scope="module")
def mixed():
    pairs = realistic_pairs()
    return pairs, analyse(pairs)


def test_every_finding_has_formats_and_a_picture_on_realistic_games(mixed):
    pairs, (eng, mis) = mixed
    games = [g for g, _ in pairs]
    ids = [i.id for i in eng.insights]
    # careless moves against greedy opponents: blunders, hanging pieces, missed captures, bad positions
    assert "engine.weakness.blunder-rate" in ids and any("hung-material" in i or "missed-tactics" in i for i in ids)
    check_module(eng, games, len(pairs))
    check_module(mis, games, len(pairs))
    blunders = next(i for i in eng.insights if i.id == "engine.weakness.blunder-rate")
    assert blunders.formats == {"bullet": 20, "blitz": 20, "rapid": 20}
    assert blunders.chart.labels == ["All games", "Bullet", "Blitz", "Rapid"]
    assert blunders.diagram is not None and blunders.diagram.title.startswith("Your costliest blunder: ")
    for ins in eng.insights:
        if ins.id.endswith(("missed-tactics", "hung-material", "missed-mates")) and ins.kind != "strength":
            assert ins.diagram is not None, ins.id


def test_format_bars_are_computed_on_each_formats_games(mixed):
    pairs, (eng, _) = mixed
    blunders = next(i for i in eng.insights if i.id == "engine.weakness.blunder-rate")
    you = dict(zip(blunders.chart.labels, blunders.chart.series[0].values))
    them = dict(zip(blunders.chart.labels, blunders.chart.series[1].values))
    for tc in ("bullet", "blitz", "rapid"):
        plies = [p for g, ev in pairs if g.time_class == tc for p in ev.plies]
        mine = [p for p in plies if p.is_user]
        theirs = [p for p in plies if not p.is_user]
        name = visuals.FORMAT_NAMES[tc]
        assert you[name] == pytest.approx(100 * sum(p.judgement == "blunder" for p in mine) / len(mine))
        assert them[name] == pytest.approx(100 * sum(p.judgement == "blunder" for p in theirs) / len(theirs))
    # the accuracy chart's line is the opponents' average over every game; the note gives it per format too
    accuracy = next(i for i in eng.insights if i.id == "engine.observation.accuracy-by-result")
    for tc in ("bullet", "blitz", "rapid"):
        theirs = [ev.opp_accuracy for g, ev in pairs if g.time_class == tc]
        assert f"{tc} {sum(theirs) / len(theirs):.1f}" in accuracy.chart.note


def test_costliest_example_board_is_the_largest_drop(mixed):
    pairs, (eng, _) = mixed
    blunders = next(i for i in eng.insights if i.id == "engine.weakness.blunder-rate")
    worst = max(
        ((p.win_before - p.win_after, g, p) for g, ev in pairs for p in ev.plies if p.is_user and p.judgement == "blunder"),
        key=lambda x: (x[0], x[1].end_time),
    )
    drop, game, ply = worst
    assert blunders.diagram.link == game.url
    assert f"from {ply.win_before:.0f}% to {ply.win_after:.0f}%" in blunders.diagram.caption
    frames = blunders.diagram.strips[0].frames
    assert frames[0].caption == f"Your chances: {ply.win_after:.0f}%"  # the engine's verdict after your move


def test_module_tables_and_charts_say_which_formats_they_cover(mixed):
    _, (eng, mis) = mixed
    by_title = {c.title: c for c in eng.charts}
    assert by_title["Engine accuracy by format"].labels == ["Bullet", "Blitz", "Rapid"]
    assert by_title["Engine accuracy by format"].table.title == "By time control"
    for title in ("Mistakes and blunders per 100 moves, by phase", "Blunders per 100 moves, by move number"):
        assert "Bullet 20 · Blitz 20 · Rapid 20 games" in by_title[title].note
    for table in eng.tables:
        if table.title != "Blunders when short of time":  # one row per time class already
            assert "Formats: " in table.note, table.title
    assert "(20 bullet, 20 blitz and 20 rapid)" in eng.summary and "Bullet 20 · Blitz 20 · Rapid 20" in eng.kpis[0].hint
    assert eng.stats["formats"] == {"bullet": 20, "blitz": 20, "rapid": 20}
    puzzles = next(t for t in mis.tables if t.title == "Your costliest mistakes (puzzles)")
    assert {row[1] for row in puzzles.rows} <= {"Bullet", "Blitz", "Rapid"}
    assert "These 10: " in puzzles.note and "(20 bullet, 20 blitz and 20 rapid)" in mis.summary


def test_the_costliest_puzzles_get_a_board_each_in_table_order(mixed):
    pairs, (_, mis) = mixed
    puzzles = next(t for t in mis.tables if t.title == "Your costliest mistakes (puzzles)")
    boards = [d for d in mis.diagrams if d.title.startswith("Puzzle ")]
    assert len(boards) == len(puzzles.rows) == mistakes.PUZZLE_TABLE_ROWS
    assert [d.link for d in boards] == [row[6] for row in puzzles.rows]
    # the Lichess board opens from your side, as the report's board is drawn
    assert [d.fen.replace(" ", "_") + ("?color=black" if d.orientation == "black" else "") for d in boards] == [
        row[5].removeprefix("https://lichess.org/analysis/") for row in puzzles.rows
    ]
    assert any(d.orientation == "black" for d in boards) and any(d.orientation == "white" for d in boards)
    for d, row in zip(boards, puzzles.rows):
        assert row[2] in d.caption and row[3] in d.caption and visuals.FORMAT_NAMES[d.time_class] == row[1]
        assert [a.kind for a in d.arrows] == ["played", "best"]
        assert d.caption.endswith(f"{row[3]} (green) was better.")  # the green arrow shows it: no "find it first"
    events = mistakes.build_puzzles([g for g, _ in pairs], {g.game_id: ev for g, ev in pairs})
    assert [e.game.url for e in events[: len(boards)]] == [d.link for d in boards]  # the export is unchanged


# --------------------------------------------------------------------------- repeated mistakes across formats
LINE = ["e4", "e5", "Nf3", "Nc6", "Bc4"]


def _error_game(n: int, time_class: str, reply: str = "Nd4", error: bool = True):
    base, control = CLOCKS[time_class]
    game = make_game(color="black", moves_san=LINE + [reply, "Nxe5", "Qg5", "Nxf7", "Qxg2"], game_id=f"r{n}",
                     url=f"https://www.chess.com/game/live/9{n}", time_class=time_class, time_control=control,
                     base_seconds=base)
    plies = [make_ply_eval(ply=i, mover="white" if i % 2 == 0 else "black", is_user=i % 2 == 1, san=s, best_san=s,
                           win_before=50, win_after=50 - i) for i, s in enumerate(game.moves_san)]
    if error:
        plies[5] = make_ply_eval(ply=5, mover="black", is_user=True, san=reply, best_san="Nf6", win_before=48.0,
                                 win_after=30.0, judgement="blunder")
    return game, make_game_eval(game.game_id, plies)


def test_a_repeated_mistake_is_split_by_format():
    specs = [_error_game(n, tc) for n, tc in enumerate(["blitz"] * 4 + ["rapid"] * 2 + ["bullet"], 1)]
    specs += [_error_game(n, tc, "Nf6", error=False) for n, tc in enumerate(["rapid", "rapid", "bullet"], 20)]
    games = [g for g, _ in specs]
    eng, mis = analyse(specs)
    [ins] = mis.insights
    assert ins.formats == {"bullet": 2, "blitz": 4, "rapid": 4}
    assert ins.chart.kind == "stacked_bar" and ins.chart.labels == ["All games", "Bullet", "Blitz", "Rapid"]
    assert [s.values for s in ins.chart.series] == [[7, 1, 4, 2], [3, 1, 0, 2]]
    assert "in 7 of 10 games" in ins.title and "Bullet 2 · Blitz 4 · Rapid 4" in ins.chart.note
    check_module(mis, games, len(specs))
    board = ins.diagram
    assert board is mis.diagrams[0] and board.orientation == "black"
    assert [f.move for f in board.strips[0].frames] == ["3...Nd4", "4.Nxe5", "4...Qg5", "5.Nxf7"]
    assert board.strips[0].frames[0].caption == "Your chances: 30%"  # your move, from your side
    assert board.strips[0].frames[1].caption == "Your chances: 56%"  # 100 - the opponent's 44% after 4.Nxe5
    [table, _] = mis.tables
    assert table.rows[0][4] == "Bullet 1 · Blitz 4 · Rapid 2"  # the games in which you played it, by format


# --------------------------------------------------------------------------- the synthetic demo player
@pytest.fixture(scope="module")
def demo():
    archives = synth.generate_archives(90, seed=11, engine_path=None, workers=1)
    games = parse.parse_games([g for month in archives.values() for g in month], synth.DEFAULT_PERSONA.username)
    pairs = [(g, pseudo_eval(g)) for g in games]
    return games, pairs


def test_every_finding_of_the_demo_player_has_formats_and_a_picture(demo):
    games, pairs = demo
    eng, mis = analyse(pairs, games)
    assert eng.insights and mis.insights and mis.diagrams
    assert Counter(g.time_class for g in games).keys() >= {"bullet", "blitz", "rapid"}
    check_module(eng, games, len(pairs))
    check_module(mis, games, len(pairs))
    assert all(i.diagram is not None for i in mis.insights)


def test_each_format_of_the_demo_player_on_its_own(demo):
    """A per-format report runs the modules on one format's games: single-format labels, nothing crashes."""
    games, pairs = demo
    for tc in ("bullet", "blitz", "rapid"):
        subset = [(g, ev) for g, ev in pairs if g.time_class == tc]
        eng, mis = analyse(subset)
        for mr in (eng, mis):
            check_module(mr, [g for g, _ in subset], len(subset))
            for ins in mr.insights:
                assert ins.formats.keys() == {tc}, ins.id
                if ins.chart is not None and ins.chart.labels[0] != "Short of time" and ins.id.split(".")[2] not in (
                        "blunder-anatomy",):
                    assert ins.chart.labels == [visuals.FORMAT_NAMES[tc]], (ins.id, ins.chart.labels)


# --------------------------------------------------------------------------- tiny and awkward inputs
def test_tiny_single_format_and_awkward_inputs_do_not_crash():
    pairs = realistic_pairs(seed=9, n=3, formats=("bullet",))
    for chunk in (pairs[:1], pairs):
        eng, mis = analyse(chunk)
        check_module(eng, [g for g, _ in chunk], len(chunk))
        check_module(mis, [g for g, _ in chunk], len(chunk))
    daily = realistic_pairs(seed=3, n=12, formats=("daily",))
    for g, _ in daily:
        g.clocks = [None] * g.plies
    eng, mis = analyse(daily, **{"engine.min_games": 2, "engine.min_moves": 10})
    check_module(eng, [g for g, _ in daily], len(daily))
    assert all(i.formats == {"daily": 12} for i in eng.insights if not i.id.startswith("engine.observation.time"))


def test_boards_are_skipped_when_the_moves_do_not_replay():
    good = realistic_pairs(seed=4, n=30, formats=("blitz", "rapid"))
    for g, _ in good:
        g.moves_san[1] = "Ke4"  # corrupt record: nothing replays past the first move
    eng, mis = analyse(good)
    assert eng.insights and all(i.diagram is None and i.chart is not None for i in eng.insights)
    assert all(i.formats for i in eng.insights) and not mis.diagrams


def test_game_diagram_edge_cases():
    game = make_game(moves_san=["e4", "e5", "Nf3"])
    describe = lambda played, best: (played, best or "")  # noqa: E731
    assert mistakes.game_diagram(game, 3, describe) is None  # no such move
    assert mistakes.game_diagram(game, -1, describe) is None
    last = mistakes.game_diagram(game, 2, describe, best_san="Nc3")
    assert [f.move for f in last.strips[0].frames] == ["2.Nf3"] and last.last_move == "e7e5"
    assert last.strips[0].frames[0].caption == ""  # no engine verdicts given
    assert (last.title, last.caption) == ("2.Nf3", "2.Nc3")
    broken = make_game(initial_fen="not a fen", moves_san=["e4"])
    assert mistakes.game_diagram(broken, 0, describe) is None
    assert mistakes.formats_note({}) == "" and mistakes.formats_words({}) == ""
    assert mistakes.formats_note({"rapid": 2, "bullet": 3}) == "Bullet 3 · Rapid 2 games"
    assert mistakes.formats_words({"blitz": 5}) == "all blitz"
    # a "better move" that isn't legal in the position (an eval from another record) gets no arrow and no mention
    stale = mistakes.game_diagram(game, 2, describe, best_san="Qxf7#")
    assert [a.kind for a in stale.arrows] == ["played"] and stale.caption == ""


def test_chess960_boards_castle_in_the_arrows_and_the_strip():
    fen = "rk5r/pppppppp/8/8/8/8/PPPPPPPP/RK5R w HAha - 0 1"
    game = make_game(rules="chess960", initial_fen=fen, moves_san=["a3", "a6", "O-O", "O-O-O", "h3"])
    board = mistakes.game_diagram(game, 2, lambda played, best: (played, best or ""), best_san="h3")
    assert [(a.start, a.end, a.kind) for a in board.arrows] == [("b1", "g1", "played"), ("h2", "h3", "best")]
    [strip] = board.strips
    assert [f.move for f in strip.frames] == ["2.O-O", "2...O-O-O", "3.h3"]  # the whole rest of the game
    after = chess.Board(strip.frames[1].fen, chess960=True)
    assert after.king(chess.WHITE) == chess.G1 and after.king(chess.BLACK) == chess.C8


def test_board_titles_name_the_move_you_missed():
    p = make_ply_eval(ply=38, mover="white", is_user=True, san="Qd2", best_san="Qh7#", mate_before=1,
                      win_before=100.0, win_after=60.0, tags=["missed_mate"])
    record = engine_stats.Analysed(make_game(), make_game_eval("g", [p]), [p])
    title, caption = engine_stats._mate_caption(record, p)("20.Qd2", "20.Qh7#")
    assert title == "Your costliest missed mate: 20.Qh7#"
    assert caption == "Stockfish saw a forced mate in 1 here, starting with 20.Qh7# (green); you played 20.Qd2 (red)."
    assert engine_stats._mate_caption(record, p)("20.Qd2", None)[0] == "Your costliest missed mate, at 20.Qd2"
    black = make_ply_eval(ply=39, mover="black", is_user=True, san="Kg8", mate_before=-3, tags=["missed_mate"])
    assert "a forced mate in 3 here" in engine_stats._mate_caption(record, black)("20...Kg8", "20...Qh2+")[1]
    shot = engine_stats._tactic_caption("missed_tactic")(record, p)
    assert shot("20.Qd2", "20.Qh7#")[0] == "Your costliest missed shot: 20.Qh7#"
    assert shot("20.Qd2", None)[0] == "Your costliest missed shot, at 20.Qd2"


# --------------------------------------------------------------------------- real Stockfish
@pytest.mark.engine
def test_boards_from_real_stockfish_analysis(stockfish_path, demo):
    games, _ = demo
    picked = [next(g for g in games if g.time_class == tc and g.plies >= 30) for tc in ("bullet", "blitz", "rapid")]
    cfg = engine.EngineConfig(path=stockfish_path, depth=6, hash_mb=16, workers=1)
    eng_process = engine.open_engine(cfg)
    try:
        pairs = [(g, engine.analyze_game(g, eng_process, cfg)) for g in picked]
    finally:
        eng_process.quit()
    eng, mis = analyse(pairs, **{"engine.min_games": 2, "engine.min_moves": 20})
    check_module(eng, picked, len(pairs))
    check_module(mis, picked, len(pairs))
    assert mis.diagrams  # three real games always hold some costly mistake
    for d in mis.diagrams:
        board = chess.Board(d.fen)
        for arrow in d.arrows:
            move = chess.Move(chess.parse_square(arrow.start), chess.parse_square(arrow.end))
            assert board.piece_at(move.from_square).color == board.turn
