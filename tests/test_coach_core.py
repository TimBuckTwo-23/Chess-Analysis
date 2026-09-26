"""The coaching layer's core (C1): critical positions, explanations with boards and charts, the puzzle PGN, and
build_coaching's orchestration. No Stockfish needed: engines, lines and motifs are faked."""

from __future__ import annotations

import dataclasses
import io
import re

import chess
import chess.engine
import chess.pgn
import pytest

from chess_insights import engine as sf
from chess_insights.analysis import mistakes
from chess_insights.coach import build_coaching, concepts, critical, deep, explain, motifs, puzzles
from chess_insights.coach.config import CoachConfig
from chess_insights.coach.deep import DeepResult, ProfileError, make_line
from chess_insights.coach.explain import Comparison
from chess_insights.context import AnalysisContext
from chess_insights.models import Coaching, ConceptDelta, CriticalPosition, Explanation, Motif

from factories import DEFAULT_MOVES, make_game, make_game_eval, make_ply_eval

SICILIAN = ["e4", "c5", "Nf3", "Nc6", "Nc3", "e6", "d4", "cxd4", "Nxd4", "e5", "Ndb5", "a6", "Nd6+", "Bxd6", "Qxd6",
            "Qe7"]
OPEN_GAME = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6", "O-O", "O-O"]
OTHER_ITALIAN = OPEN_GAME + ["h3", "a6", "a3", "Ba7", "Re1", "h6", "Nbd2", "Re8"]


def _evals(game, errors: dict[int, tuple[float, float, str]]):
    """A GameEval with no drop anywhere except ``errors``: ply -> (win before, win after, engine's move)."""
    plies = []
    for i, san in enumerate(game.moves_san):
        mover = "white" if i % 2 == 0 else "black"
        if i in errors:
            before, after, best = errors[i]
            plies.append(make_ply_eval(ply=i, mover=mover, is_user=mover == game.color, san=san, best_san=best,
                                       win_before=before, win_after=after, judgement="blunder"))
        else:
            plies.append(make_ply_eval(ply=i, mover=mover, is_user=mover == game.color, san=san, best_san=san))
    return make_game_eval(game.game_id, plies)


@pytest.fixture
def player():
    """Four games with the 5...e5 habit (three blitz, one bullet), two single errors, and two games of 1...e5, so
    that 1.e4 is a choice point (with the choice thresholds lowered)."""
    games, evals = [], {}
    for i, tc in enumerate(["blitz", "blitz", "bullet", "blitz"]):
        g = make_game(moves_san=list(SICILIAN), color="black", time_class=tc, opening="Sicilian Defense: Open",
                      opening_family="Sicilian Defense", outcome="loss", minutes_ago=100 - i)
        games.append(g)
        evals[g.game_id] = _evals(g, {9: (45.0, 25.0, "Nf6")})
    big = make_game(moves_san=list(DEFAULT_MOVES), color="white", time_class="rapid", minutes_ago=50)
    small = make_game(moves_san=list(OTHER_ITALIAN), color="white", time_class="bullet", minutes_ago=40)
    games += [big, small]
    evals[big.game_id] = _evals(big, {12: (60.0, 20.0, "h3")})  # 7.Re1 lost 40 points
    evals[small.game_id] = _evals(small, {14: (55.0, 43.0, "d4")})  # 8.a3 lost 12
    for i in range(2):
        g = make_game(moves_san=list(OPEN_GAME), color="black", minutes_ago=30 - i)
        games.append(g)
    games.sort(key=lambda g: g.end_time)
    ctx = AnalysisContext("tester", games, evals=evals, options={"openings.min_choice_games": 2})
    return ctx, [mistakes.analyze(ctx)]


# --------------------------------------------------------------------------- critical positions
def test_repeated_mistakes_first_then_costliest_errors_then_choice_points(player):
    ctx, modules = player
    picked = critical.select_critical(ctx, modules, 150)
    assert [p.kind for p in picked] == ["repeated", "error", "error", "choice", "choice"]
    rep = picked[0]
    sic = [g for g in ctx.games if g.moves_san == SICILIAN]
    assert rep.move_label == "5...e5" and rep.played_san == "e5" and rep.played_uci == "e6e5"
    assert rep.best_san == "Nf6" and rep.drop == pytest.approx(20.0) and rep.repeats == 4
    assert rep.games == [g.url for g in sorted(sic, key=lambda g: g.end_time, reverse=True)]  # most recent first
    assert rep.time_class == ""  # blitz and bullet: several formats
    assert rep.color == "black" and rep.opening_family == "Sicilian Defense" and rep.moves_before == SICILIAN[:9]
    assert rep.ply == 9 and rep.url == rep.games[0] and chess.Board(rep.fen).epd() == rep.epd
    insight = modules[0].insights[0]
    assert rep.insight_id == insight.id  # the finding the mistakes section made about this position
    # single errors, costliest first, each in its own format; no second entry for the habit's (EPD, move)
    assert [(p.move_label, round(p.drop), p.time_class, p.repeats) for p in picked[1:3]] == [
        ("7.Re1", 40, "rapid", 1), ("8.a3", 12, "bullet", 1)]
    assert picked[1].insight_id is None and picked[1].best_san == "h3"
    # the choice point after 1.e4: the moves you play there (4 x 1...c5, 2 x 1...e5), from the opening tree
    assert [(p.move_label, p.repeats, p.moves_before) for p in picked[3:]] == [("1...c5", 4, ["e4"]),
                                                                             ("1...e5", 2, ["e4"])]
    # the engine's view where the games were analysed (the 1...c5 games), nothing otherwise
    assert [(p.drop, p.best_san) for p in picked[3:]] == [(0.0, "c5"), (0.0, None)]
    assert len({(p.epd, p.played_uci) for p in picked}) == len(picked)


def test_the_cap_keeps_habits_and_errors_first(player):
    ctx, modules = player
    assert [p.kind for p in critical.select_critical(ctx, modules, 1)] == ["repeated"]
    assert [p.kind for p in critical.select_critical(ctx, modules, 3)] == ["repeated", "error", "error"]
    assert critical.select_critical(ctx, modules, 0) == []
    assert critical.select_critical(dataclasses.replace(ctx, evals={}), modules, 10) == []
    # without the mistakes module's findings: the same positions, no insight ids
    assert all(p.insight_id is None for p in critical.select_critical(ctx, [], 150))


def test_chess960_games_are_left_out(player):
    ctx, modules = player
    games = [dataclasses.replace(g, rules="chess960") for g in ctx.games]
    assert critical.select_critical(dataclasses.replace(ctx, games=games), [], 150) == []


# --------------------------------------------------------------------------- explanations
def _position(kind: str = "repeated") -> CriticalPosition:
    board = chess.Board()
    for san in SICILIAN[:9]:
        board.push_san(san)
    return CriticalPosition(
        game_id="g1", url="https://www.chess.com/game/live/1", ply=9, fen=board.fen(), played_uci="e6e5",
        played_san="e5", move_label="5...e5", best_san="Nf6", drop=20.0, time_class="blitz", color="black",
        kind=kind, repeats=3, games=["https://www.chess.com/game/live/1", "https://www.chess.com/game/live/2"],
        insight_id="mistakes.weakness.abc", opening_family="Sicilian Defense", moves_before=SICILIAN[:9],
    )


def _result(pos: CriticalPosition, best=("g8f6", "d4c6", "b7c6", "e4e5", "f6d5", "c3e4"),
            ref=("e6e5", "d4b5", "a7a6", "b5d6", "f8d6", "d1d6", "d8e7"), best_cp=-39, ref_cp=-157,
            ref_mate=None) -> DeepResult:
    moves = lambda ucis: [chess.Move.from_uci(u) for u in ucis]  # noqa: E731
    return DeepResult(
        epd=pos.epd, played_uci=pos.played_uci,
        best_line=make_line(pos.fen, moves(best), best_cp, None, 20),
        refutation=make_line(pos.fen, moves(ref), ref_cp, ref_mate, 20),
        alternatives=[], engine="Stockfish 16", depth=20,
    )


@pytest.fixture
def motif_hooks(monkeypatch):
    """Hand-made motifs instead of the detectors (another engineer's): returns the dict to fill per role."""
    found: dict[str, list] = {"best": [], "refutation": []}
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: list(found[role]))
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset())
    return found


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if s]


def test_explanation_of_a_habit_position(motif_hooks):
    pos = _position()
    ex = explain.explain_position(explain._Games(AnalysisContext("t", [])), pos, _result(pos), None, frozenset())
    assert isinstance(ex, Explanation)
    assert (ex.epd, ex.played, ex.best, ex.kind, ex.drop, ex.time_class, ex.color, ex.repeats) == (
        pos.epd, "5...e5", "5...Nf6", "repeated", 20.0, "blitz", "black", 3)
    assert ex.insight_id == pos.insight_id and ex.game_url == pos.url and ex.games == pos.games
    assert ex.best_line.moves_san[:2] == ["Nf6", "Nxc6"] and ex.refutation.moves_san[:3] == ["e5", "Ndb5", "a6"]
    assert ex.text.startswith("After 5...e5, Stockfish's answer is 6.Ndb5 a6 7.Nd6+ Bxd6.")
    assert "−1.57 for you, against −0.39 after 5...Nf6" in ex.text
    assert len(_sentences(ex.text)) <= 3 and "centipawn" not in ex.text
    assert [s.name for s in ex.sources] == ["Stockfish 16, depth 20"]
    assert ex.motifs == [] and ex.drill_themes == [] and ex.chart is None  # no concepts without Stockfish 16
    d = ex.diagram
    assert d.fen == pos.fen and d.orientation == "black" and d.time_class == "blitz" and d.link == pos.url
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("e6", "e5", "played"), ("g8", "f6", "best")]
    assert d.title == "Sicilian Defense: 5...e5" and "in 3 games" in d.caption
    assert [s.title for s in d.strips] == ["After your 5...e5", "The better 5...Nf6"]
    ref, best = d.strips
    assert [f.move for f in ref.frames] == ["6.Ndb5", "6...a6", "7.Nd6+", "7...Bxd6"]
    assert [f.caption for f in ref.frames] == ["", "", "", "−1.57 for you"]
    assert [f.move for f in best.frames] == ["5...Nf6", "6.Nxc6", "6...bxc6", "7.e5"]
    assert best.frames[-1].caption == "−0.39 for you" and best.frames[0].last_move == "g8f6"


def test_a_motif_is_named_only_when_its_detector_passed_the_gate(motif_hooks, monkeypatch):
    pos = _position()
    fork = Motif(theme="fork", line="refutation", ply=3, squares=["d6", "e8", "b7"], side="opponent")
    mine = Motif(theme="pin", line="refutation", ply=2, squares=["a1"], side="you")  # not the opponent's: ignored
    motif_hooks["refutation"] = [fork, mine]
    games = explain._Games(AnalysisContext("t", []))
    ex = explain.explain_position(games, pos, _result(pos), None, frozenset(motifs.GATED_THEMES))
    assert "White has a tactic: 6.Ndb5" in ex.text and "fork" not in ex.text
    assert ex.motifs == [] and ex.drill_themes == [] and ex.diagram.marks == []
    assert "check, capture and threat" in ex.text
    # once the fork detector passes the gate, the explanation names it, marks it and routes to fork drills
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork"}))
    out = explain.explain_all(AnalysisContext("t", []), [pos], {(pos.epd, pos.played_uci): _result(pos)},
                              CoachConfig(stockfish="/nonexistent/stockfish"), [])
    ex = out[0]
    assert "White has a fork: 6.Ndb5" in ex.text and "attack at once" in ex.text
    assert ex.motifs == [fork] and ex.drill_themes == ["fork"]
    assert [(m.square, m.kind) for m in ex.diagram.marks] == [("d6", "attacker"), ("e8", "target"), ("b7", "target")]
    frames = ex.diagram.strips[0].frames
    assert frames[2].marks and frames[2].marks[0].square == "d6" and not frames[0].marks  # ply 3 = third frame


def test_missed_motif_and_mate_texts(motif_hooks, monkeypatch):
    pos = _position(kind="error")
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork"}))
    motif_hooks["best"] = [Motif(theme="fork", line="best", ply=0, squares=["f6"], side="you"),
                           Motif(theme="pin", line="best", ply=1, squares=["c6"], side="opponent")]
    games = explain._Games(AnalysisContext("t", []))
    ex = explain.explain_position(games, pos, _result(pos), None, frozenset({"fork"}))
    assert ex.text.startswith("You had a fork: 5...Nf6 6.Nxc6 bxc6 7.e5 Nd5.")
    assert "your own checks, captures and threats" in ex.text and ex.drill_themes == ["fork"]
    motif_hooks["best"] = []
    mated = _result(pos, ref=("e6e5", "d4b5"), ref_cp=-1000, ref_mate=-2)
    ex = explain.explain_position(games, pos, mated, None, frozenset())
    assert ex.text.startswith("After 5...e5, White mates in 2: 6.Ndb5.")
    assert "mate in 2 for White" in ex.text and ex.diagram.strips[0].frames[-1].caption == "mate in 2 for White"


def test_a_deeper_search_that_prefers_your_move_says_so(motif_hooks):
    pos = _position(kind="error")
    same = _result(pos, best=("e6e5", "d4b5"), best_cp=-20, ref_cp=-20)
    ex = explain.explain_position(explain._Games(AnalysisContext("t", [])), pos, same, None, frozenset())
    assert "Stockfish's own first choice" in ex.text and "does not hold up" in ex.text
    assert [a.kind for a in ex.diagram.arrows] == ["played"] and len(ex.diagram.strips) == 1
    assert len(_sentences(ex.text)) <= 3


def test_choice_point_text(motif_hooks):
    pos = _position(kind="choice")
    ex = explain.explain_position(explain._Games(AnalysisContext("t", [])), pos, _result(pos), None, frozenset())
    assert ex.text.startswith("Stockfish prefers 5...Nf6 (−0.39 for you) to your 5...e5 (−1.57 for you)")
    assert ex.diagram.strips[1].title == "Stockfish's 5...Nf6"


def test_what_it_wins_material_or_concepts():
    deltas = [ConceptDelta("King safety", -0.85, label="king safety"), ConceptDelta("Mobility", -0.38,
              label="piece activity"), ConceptDelta("Material", -0.5, label="material"),
              ConceptDelta("Threats", 0.3, label="threats")]
    text = explain._what_it_wins(Comparison(deltas=deltas, facts=["you lose the bishop pair"]), gap=1.3)
    assert text == ("a few moves later you stand 0.85 pawns worse on king safety and 0.38 on piece activity, "
                    "and you lose the bishop pair")
    assert explain._what_it_wins(Comparison(deltas=deltas, material_lost=1), gap=1.2) == "you lose a pawn"
    # the engine sees compensation (a gap far below the material): talk about the concepts instead
    assert explain._what_it_wins(Comparison(deltas=deltas, material_lost=3), gap=0.5).startswith("a few moves")
    assert explain._what_it_wins(Comparison(material_lost=3), gap=3.0) == "you lose 3 pawns' worth of material"
    assert explain._what_it_wins(Comparison(), gap=1.0) == ""


def test_concepts_chart_and_sources_with_a_stockfish_16_table(motif_hooks, monkeypatch, fixtures_dir):
    """The comparison reads Stockfish 16's table at both line ends (faked here from a recorded table)."""
    table = concepts.parse_eval_table((fixtures_dir / "coach" / "sf16_eval_sicilian_5e5.txt").read_text())
    worse = {**table, "Mobility": (0.80, 0.90), "King safety": (0.60, 0.10)}  # White's side: worse for Black
    pos = _position()
    result = _result(pos)
    ref_end = concepts.comparison_point(pos.fen, result.refutation.moves_uci)
    monkeypatch.setattr(concepts.ClassicalEval, "table",
                        lambda self, fen: worse if fen == ref_end.fen() else table)
    notes: list[str] = []
    out = explain.explain_all(AnalysisContext("t", []), [pos], {(pos.epd, pos.played_uci): result},
                              CoachConfig(stockfish="/nonexistent/stockfish"), notes)
    ex = out[0]
    assert [c.term for c in ex.concepts][:2] == ["Mobility", "King safety"] and all(c.value < 0 for c in ex.concepts)
    assert "pawns worse on piece activity" in ex.text and "you lose the bishop pair" in ex.facts
    chart = ex.chart
    assert chart.kind == "hbar" and chart.value_format == "signed_float2" and chart.reference == 0.0
    assert chart.labels[:2] == ["Piece activity", "King safety"] and chart.series[0].values[0] < 0
    assert "pawns" in chart.note and [s.name for s in ex.sources][-1] == "Stockfish 16 classical evaluation terms"


def test_explain_all_skips_positions_without_lines_and_notes_missing_concepts(motif_hooks, tmp_path):
    pos = _position()
    other = dataclasses.replace(_position(), played_uci="d7d6", played_san="d6", move_label="5...d6")
    bad = _result(pos)
    notes: list[str] = []
    out = explain.explain_all(AnalysisContext("t", []), [pos, other],
                              {(pos.epd, pos.played_uci): bad, (other.epd, other.played_uci): bad},
                              CoachConfig(stockfish=str(tmp_path / "missing")), notes)
    assert [x.played for x in out] == ["5...e5"]  # the other's refutation doesn't start with its move
    assert any("Concept terms skipped" in n for n in notes)
    assert explain.explain_all(AnalysisContext("t", []), [pos], {}, CoachConfig(), []) == []


def test_a_crashing_motif_detector_costs_nothing(monkeypatch):
    def boom(line, role):
        raise RuntimeError("detector bug")

    monkeypatch.setattr(motifs, "detect_line", boom)
    pos = _position()
    ex = explain.explain_position(explain._Games(AnalysisContext("t", [])), pos, _result(pos), None, frozenset())
    assert ex is not None and ex.motifs == []


# --------------------------------------------------------------------------- the puzzle PGN
def _read_all(pgn: str) -> list[chess.pgn.Game]:
    stream, games = io.StringIO(pgn), []
    while (game := chess.pgn.read_game(stream)) is not None:
        assert not game.errors, game.errors
        games.append(game)
    return games


def test_puzzle_pgn_with_lines_themes_and_explanations(player):
    ctx, _ = player
    events = mistakes.build_puzzles(ctx.games, ctx.evals)
    assert puzzles.puzzles_pgn(events) == mistakes.puzzles_to_pgn(events)
    assert puzzles.puzzles_pgn(events, Coaching()) == mistakes.puzzles_to_pgn(events)
    first, second, last = events[0], events[1], events[-1]
    assert (first.move_label, second.move_label, last.move_label) == ("7.Re1", "5...e5", "8.a3")
    board = chess.Board(first.fen)
    line = []
    for _ in range(10):  # a 10-ply line: the solution keeps 8
        move = min(board.legal_moves, key=lambda m: m.uci())
        line.append(move)
        board.push(move)
    coaching = Coaching()
    coaching.puzzle_lines[puzzles.puzzle_key(first.game.game_id, first.ply)] = make_line(first.fen, line, 120, None, 10)
    coaching.puzzle_themes[puzzles.puzzle_key(first.game.game_id, first.ply)] = ["fork", "hangingPiece"]
    pos = _position()
    res = _result(pos)
    coaching.explanations.append(Explanation(epd=second.epd, fen=second.fen, played=second.move_label, best="5...Nf6",
                                             best_line=res.best_line, refutation=res.refutation,
                                             text="After 5...e5, White plays {6.Ndb5}.", drill_themes=["pin"]))
    pgn = puzzles.puzzles_pgn(events, coaching)
    games = _read_all(pgn)
    assert len(games) == len(events)
    for i, (g, e) in enumerate(zip(games, events), 1):
        assert g.headers["Event"] == f"chess-insights puzzle {i}" and g.headers["FEN"] == e.fen
        b = g.board()
        for move in g.mainline_moves():
            assert move in b.legal_moves
            b.push(move)
    assert len(list(games[0].mainline_moves())) == puzzles.SOLUTION_PLIES
    assert games[0].headers["Themes"] == "fork hangingPiece"
    assert games[0].comment.startswith(f"In the game you played {first.move_label}")
    # the explanation matches the second puzzle's position and move: its text (braces made safe) and themes
    assert [m.uci() for m in games[1].mainline_moves()] == res.best_line.moves_uci[: puzzles.SOLUTION_PLIES]
    assert games[1].comment == "After 5...e5, White plays (6.Ndb5)." and games[1].headers["Themes"] == "pin"
    # nothing for the last one: exactly the single-move puzzle
    single = mistakes.puzzles_to_pgn([last]).replace("puzzle 1", f"puzzle {len(events)}")
    assert pgn.endswith(single) and "Themes" not in games[-1].headers


def test_fill_puzzle_lines_from_deep_and_profile_results(player, monkeypatch):
    ctx, _ = player
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork"}))
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: [
        Motif("fork", role, 0, ["d5"], "you"), Motif("pin", role, 1, ["c6"], "you"),
        Motif("skewer", role, 1, ["c6"], "opponent")])
    events = mistakes.collect_errors(ctx.games, ctx.evals)
    sic = next(e for e in events if e.move_label == "5...e5")
    rapid = next(e for e in events if e.move_label == "7.Re1")
    bullet = next(e for e in events if e.move_label == "8.a3")
    board = chess.Board(sic.fen)
    deep_best = make_line(sic.fen, [board.parse_san(s) for s in ("Nf6",)], -39, None, 20)
    lines = {(sic.epd, sic.uci): DeepResult(sic.epd, sic.uci, deep_best, None)}
    b2 = chess.Board(rapid.fen)
    prof_line = make_line(rapid.fen, [b2.parse_san("h3")], 40, None, 10)
    b3 = chess.Board(bullet.fen)
    own_move = make_line(bullet.fen, [b3.parse_san(bullet.san)], 0, None, 10)  # the profile search prefers yours
    profile = [ProfileError(rapid.game.game_id, rapid.ply, "you", "rapid", rapid.fen, rapid.uci, 40.0, prof_line),
               ProfileError(bullet.game.game_id, bullet.ply, "you", "bullet", bullet.fen, bullet.uci, 12.0, own_move)]
    coaching = Coaching()
    puzzles.fill_puzzle_lines(ctx, coaching, lines, profile)
    sic_keys = {puzzles.puzzle_key(e.game.game_id, e.ply) for e in events if e.move_label == "5...e5"}
    assert sic_keys | {puzzles.puzzle_key(rapid.game.game_id, rapid.ply)} == set(coaching.puzzle_lines)
    assert all(coaching.puzzle_lines[k].moves_san == ["Nf6"] for k in sic_keys)
    assert all(v == ["fork"] for v in coaching.puzzle_themes.values())  # gated, and carried out by you
    assert len(coaching.puzzle_themes) == len(coaching.puzzle_lines)


# --------------------------------------------------------------------------- build_coaching
class _Fake:
    """A scripted engine: the first legal move in UCI order, +30 cp for the side to move."""

    def __init__(self):
        self.id, self.options = {"name": "Fake 1"}, {}

    def analyse(self, board, limit, multipv=None, game=None, info=None):
        infos = []
        for i, move in enumerate(sorted(board.legal_moves, key=lambda m: m.uci())[: multipv or 1]):
            infos.append({"score": chess.engine.PovScore(chess.engine.Cp(30 - 20 * i), board.turn), "pv": [move],
                          "depth": limit.depth})
        return infos if multipv is not None else infos[0]

    def configure(self, options):
        pass

    def close(self):
        pass


def test_build_coaching_end_to_end_with_a_fake_engine(player, monkeypatch, tmp_path):
    ctx, modules = player
    monkeypatch.setattr(sf, "open_engine", lambda cfg: _Fake())
    monkeypatch.setattr(sf, "find_stockfish", lambda explicit=None: str(tmp_path / "not-a-binary"))
    monkeypatch.setattr(sf, "engine_name", lambda path: "Fake 1")
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: [])
    deep._PROFILE_MEMO.clear()
    coaching = build_coaching(ctx, modules, CoachConfig(workers=2, cache_dir=tmp_path / "coach"))
    assert coaching.settings["positions"] == 5 and coaching.settings["engine"] == "Fake 1"
    assert coaching.settings["explained"] == len(coaching.explanations) >= 3
    kinds = [x.kind for x in coaching.explanations]
    assert kinds[0] == "repeated" and "choice" in kinds
    assert all(x.diagram is not None and x.diagram.strips for x in coaching.explanations)
    assert any("Concept terms skipped" in n for n in coaching.notes)  # the fake path is no Stockfish 16
    assert coaching.puzzle_lines and all(len(x.moves_uci) <= puzzles.SOLUTION_PLIES
                                         for x in coaching.puzzle_lines.values())
    assert (tmp_path / "coach" / "fake-1" / "d20").is_dir()
    assert (tmp_path / "coach" / "fake-1" / f"d{CoachConfig().profile_depth}").is_dir()


def test_build_coaching_without_evals_or_stockfish_leaves_notes(player, monkeypatch):
    ctx, modules = player
    empty = build_coaching(dataclasses.replace(ctx, evals={}), modules, CoachConfig())
    assert empty.explanations == [] and "needs the engine analysis" in empty.notes[0]
    monkeypatch.setattr(sf, "find_stockfish", lambda explicit=None: None)
    deep._PROFILE_MEMO.clear()
    coaching = build_coaching(ctx, modules, CoachConfig())
    assert coaching.explanations == [] and coaching.puzzle_lines == {}
    assert sum("Stockfish not found" in n for n in coaching.notes) == 1  # said once
    assert any("Deep analysis skipped" in n for n in coaching.notes)
    assert coaching.settings["positions"] == 5
