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
    assert [p.kind for p in picked] == ["repeated", "error", "error", "choice"]
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
    # the choice point after 1.e4 (4 x 1...c5, 2 x 1...e5, from openings.choice_positions): your usual move there
    assert [(p.move_label, p.repeats, p.moves_before) for p in picked[3:]] == [("1...c5", 4, ["e4"])]
    # the engine's view in the analysed games: Stockfish's first choice is your move, which lost nothing
    choice = picked[3]
    assert (choice.drop, choice.best_san, choice.time_class) == (0.0, "c5", "")  # blitz and bullet games
    assert choice.games == [g.url for g in sorted(sic, key=lambda g: g.end_time, reverse=True)]
    assert choice.url == choice.games[0] and not critical.engine_mistake(choice)
    assert len({(p.epd, p.played_uci) for p in picked}) == len(picked)


def test_the_cap_keeps_habits_and_errors_first(player):
    ctx, modules = player
    assert [p.kind for p in critical.select_critical(ctx, modules, 1)] == ["repeated"]
    assert [p.kind for p in critical.select_critical(ctx, modules, 3)] == ["repeated", "error", "error"]
    assert critical.select_critical(ctx, modules, 0) == []
    assert critical.select_critical(dataclasses.replace(ctx, evals={}), modules, 10) == []
    # without the mistakes module's findings: the same positions, no insight ids
    assert all(p.insight_id is None for p in critical.select_critical(ctx, [], 150))


def test_a_choice_point_is_named_after_an_opening_only_when_most_of_its_games_share_it(player):
    ctx, modules = player
    choices = [p for p in critical.select_critical(ctx, modules, 150) if p.kind == "choice"]
    assert [(p.move_label, p.opening_family) for p in choices] == [("1...c5", "Sicilian Defense")]
    # when the games of your usual move spread over several openings, the position gets no opening name
    spread = {g.game_id: f for g, f in zip([g for g in ctx.games if g.moves_san == SICILIAN],
                                           ["Sicilian Defense", "Sicilian Defense", "Ruy Lopez", "Other"])}
    games = [dataclasses.replace(g, opening_family=spread.get(g.game_id, g.opening_family)) for g in ctx.games]
    choices = [p for p in critical.select_critical(dataclasses.replace(ctx, games=games), [], 150)
               if p.kind == "choice"]
    assert [(p.move_label, p.opening_family) for p in choices] == [("1...c5", "")]
    games = [g for g in ctx.games if g.moves_san == SICILIAN]
    assert critical._family(games) == "Sicilian Defense"
    mixed = games[:2] + [dataclasses.replace(g, opening_family=f) for g, f in zip(games[2:], ["Ruy Lopez", "Other"])]
    assert critical._family(mixed) == ""  # 1...e5 leads to several openings: naming one would be wrong
    assert critical._family([dataclasses.replace(g, opening_family=None) for g in games]) == ""


def _choice_ctx(drop: float, formats=("blitz",) * 4):
    """Four games of 1...c5 (the formats given) and two of 1...e5 after 1.e4; in the 1...c5 games Stockfish prefers
    1...e5 and says 1...c5 lost ``drop`` win-% points."""
    games, evals = [], {}
    for i, tc in enumerate(formats):
        g = make_game(moves_san=list(SICILIAN), color="black", time_class=tc, opening_family="Sicilian Defense",
                      minutes_ago=100 - i)
        games.append(g)
        evals[g.game_id] = _evals(g, {1: (50.0, 50.0 - drop, "e5")})
    for i in range(2):
        games.append(make_game(moves_san=list(OPEN_GAME), color="black", minutes_ago=30 - i))
    games.sort(key=lambda g: g.end_time)
    return AnalysisContext("tester", games, evals=evals, options={"openings.min_choice_games": 2})


@pytest.mark.parametrize("drop", [3.0, 6.0])  # 8 and up: an error, listed as one
def test_choice_points_come_from_the_openings_sections_positions(drop):
    """Your usual move at each of openings.choice_positions; a mistake exactly when the openings section draws
    Stockfish's move as the better one (it lost at least engine_min_drop win-% points on average)."""
    from chess_insights.analysis import openings

    ctx = _choice_ctx(drop)
    (pos,) = openings.choice_positions(ctx)
    (choice,) = [p for p in critical.select_critical(ctx, [], 150) if p.kind == "choice"]
    assert (choice.played_san, choice.played_uci) == (pos.usual.san, pos.usual.uci) == ("c5", "c7c5")
    assert (choice.ply, choice.fen, choice.moves_before, choice.color) == (pos.ply, pos.fen, pos.moves_before, "black")
    assert choice.repeats == pos.usual.games == 4 and choice.games == pos.usual.urls  # newest first
    assert choice.game_id == pos.usual.game_ids[0] and choice.url == pos.usual.urls[0]
    assert choice.time_class == "blitz" and choice.opening_family == "Sicilian Defense"
    assert choice.best_san == pos.engine_best == "e5" and choice.drop == pytest.approx(drop)
    assert critical.engine_mistake(choice) == (pos.best_by == "engine" and pos.best != pos.usual.san) == (drop >= 5)
    # games in two formats: no single format
    (mixed,) = [p for p in critical.select_critical(_choice_ctx(drop, ("blitz", "bullet") * 2), [], 150)
                if p.kind == "choice"]
    assert mixed.time_class == ""


def test_engine_mistake_needs_a_verdict_from_the_game_analysis():
    pos = _position(kind="choice")
    assert critical.engine_mistake(pos)  # 20 points and Stockfish prefers 5...Nf6
    assert not critical.engine_mistake(dataclasses.replace(pos, drop=4.9))
    assert not critical.engine_mistake(dataclasses.replace(pos, drop=0.0), min_drop=0.0)  # 0.0: no verdict at all
    assert not critical.engine_mistake(dataclasses.replace(pos, best_san="e5"))  # Stockfish's own move
    assert not critical.engine_mistake(dataclasses.replace(pos, best_san=None))
    assert critical.engine_mistake(dataclasses.replace(pos, kind="error", drop=0.0))  # errors are errors


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
    assert d.last_move == "f3d4"  # 5.Nxd4, the move that led to the position, is highlighted
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
    # the forking knight is not on d6 before your move: the fork is marked on the strip frame where it appears, not
    # on the main board (where the tint would land on an empty square)
    assert ex.diagram.marks == []
    frames = ex.diagram.strips[0].frames
    assert [(m.square, m.kind) for m in frames[2].marks] == [("d6", "attacker"), ("e8", "target"), ("b7", "target")]
    assert not frames[0].marks  # ply 3 = third frame
    # the text and the strip reach the move where the fork cashes in (ply 3 + 2: 8.Qxd6)
    assert "White has a fork: 6.Ndb5 a6 7.Nd6+ Bxd6 8.Qxd6." in ex.text
    assert [f.move for f in ex.diagram.strips[0].frames][-1] == "8.Qxd6"


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
    # the mate is the whole story: the better move and its verdict, no material or concepts on the way
    assert "Stockfish prefers 5...Nf6 (−0.39 for you)." in ex.text and "lose" not in ex.text
    assert ex.diagram.strips[0].frames[-1].caption == "mate in 2 for White"


def test_a_deeper_search_that_prefers_your_move_says_so(motif_hooks):
    pos = _position(kind="error")
    same = _result(pos, best=("e6e5", "d4b5"), best_cp=-20, ref_cp=-20)
    ex = explain.explain_position(explain._Games(AnalysisContext("t", [])), pos, same, None, frozenset())
    assert "Stockfish's own first choice" in ex.text and "does not hold up" in ex.text
    assert [a.kind for a in ex.diagram.arrows] == ["played"] and len(ex.diagram.strips) == 1
    assert len(_sentences(ex.text)) <= 3
    # the MultiPV line and its number, as the other positions' texts quote this move (not the second search's)
    same = _result(pos, best=("e6e5", "d4b5", "a7a6"), best_cp=-20, ref_cp=-35)
    ex = explain.explain_position(explain._Games(AnalysisContext("t", [])), pos, same, None, frozenset())
    assert "(−0.20 for you)" in ex.text and "−0.35" not in ex.text
    assert ex.diagram.strips[0].frames[-1].caption == "−0.20 for you"


class _NoConcepts:
    """A concept tool that must not be asked."""

    def compare(self, *args):
        raise AssertionError("no concept comparison expected")


def test_a_move_the_deeper_search_finds_close_to_its_choice_is_not_explained_as_an_error(motif_hooks, monkeypatch):
    """Less than an inaccuracy behind (CLOSE_DROP win-% points): say so, with no motifs, drills or concepts."""
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork"}))
    motif_hooks["refutation"] = [Motif(theme="fork", line="refutation", ply=1, squares=["b5"], side="opponent")]
    games = explain._Games(AnalysisContext("t", []))
    pos = _position(kind="error")
    close = _result(pos, best_cp=-39, ref_cp=-60)  # 46.4 against 43.5 win-%: 2.9 points
    assert explain.win_drop(close.best_line, close.refutation) < explain.CLOSE_DROP
    ex = explain.explain_position(games, pos, close, _NoConcepts(), frozenset({"fork"}))
    assert ex.text == ("A deeper look at depth 20 finds your 5...e5 (−0.60 for you) close to Stockfish's first choice, "
                       "5...Nf6 (−0.39), so the quicker game analysis was too harsh on it; the main line after 5...e5 "
                       "is 6.Ndb5 a6 7.Nd6+ Bxd6.")
    assert ex.motifs == [] and ex.drill_themes == [] and ex.chart is None and ex.diagram.marks == []
    assert [s.title for s in ex.diagram.strips] == ["After your 5...e5", "Stockfish's 5...Nf6"]
    # at a choice point: close to Stockfish's first choice, without a comparison of concepts
    ex = explain.explain_position(games, _position(kind="choice"), close, _NoConcepts(), frozenset({"fork"}))
    assert ex.text == ("Your 5...e5 (−0.60 for you) is close to Stockfish's first choice, 5...Nf6 (−0.39); its main "
                       "line after 5...e5 is 6.Ndb5 a6 7.Nd6+ Bxd6.")
    # a lost position where your move costs 4 pawns but few win-% points is still explained as an error
    lopsided = _result(pos, best_cp=-800, ref_cp=-1200)
    assert explain.win_drop(lopsided.best_line, lopsided.refutation) < explain.CLOSE_DROP
    assert not explain.is_close(lopsided.best_line, lopsided.refutation)
    assert "close to" not in explain.explain_position(games, pos, lopsided, None, frozenset()).text
    # a real error (10.5 points) is explained as one
    ex = explain.explain_position(games, pos, _result(pos), None, frozenset({"fork"}))
    assert "White has a fork" in ex.text and ex.drill_themes == ["fork"]


def test_a_mate_in_either_line_leaves_material_and_concepts_out(motif_hooks):
    pos = _position(kind="error")
    games = explain._Games(AnalysisContext("t", []))
    mated = _result(pos, ref=("e6e5", "d4b5"), ref_cp=-1000, ref_mate=-2)
    assert explain.explain_position(games, pos, mated, _NoConcepts(), frozenset()).chart is None
    mating = _result(pos, best_cp=1000)
    mating.best_line.mate_end = 3
    ex = explain.explain_position(games, pos, mating, _NoConcepts(), frozenset())
    assert "against mate in 3 for you after 5...Nf6." in ex.text and "lose" not in ex.text
    assert "your own checks, captures and threats" in ex.text


def test_material_is_blamed_on_your_move_only_against_both_the_start_and_the_better_line():
    tool = explain._Concepts(None)  # no Stockfish: board facts and material only
    # 1.Kb2 lets Black take the rook on h1; 1.Rxh5+ would have won the queen
    fen = "7k/8/8/7q/8/8/8/K6R w - - 0 1"
    best = make_line(fen, [chess.Move.from_uci(u) for u in ("h1h5", "h8g7")], 900, None, 20)
    ref = make_line(fen, [chess.Move.from_uci(u) for u in ("a1b2", "h5h1")], -1000, None, 20)
    cmp = tool.compare(fen, best, ref, "white")
    assert (cmp.material_lost, cmp.material_missed) == (5, 9)
    assert explain._what_it_wins(cmp, gap=19.0, best_label="1.Rxh5+") == (  # what changes hands, not a count
        "you lose a rook, and 1.Rxh5+ would have won the queen")
    # the e4 pawn falls whatever you play: it is not what your move lost
    fen = "4r2k/8/8/8/4P3/8/8/K7 w - - 0 1"
    best = make_line(fen, [chess.Move.from_uci(u) for u in ("a1b2", "e8e4")], -500, None, 20)
    ref = make_line(fen, [chess.Move.from_uci(u) for u in ("a1b1", "e8e4")], -560, None, 20)
    cmp = tool.compare(fen, best, ref, "white")
    assert (cmp.material_lost, cmp.material_missed) == (0, 0)
    assert explain._what_it_wins(cmp, gap=0.6) == ""
    # without the engine's backing (half a pawn of evaluation per pawn) the material is not named: 4.e4 loses the d4
    # pawn, but the knight 4.d5 wins a few moves in comes back as pawns (+0.75 against -0.55)
    assert explain._what_it_wins(Comparison(material_lost=1, material_missed=3), gap=1.3) == "you lose a pawn"
    assert explain._what_it_wins(Comparison(material_lost=1, material_missed=3), gap=0.4) == ""
    assert explain._what_it_wins(Comparison(material_missed=3), gap=1.5, best_label="4.d5") == (
        "4.d5 would have won 3 pawns' worth of material")
    tool.close()


def test_choice_point_text(motif_hooks):
    pos = _position(kind="choice")
    ex = explain.explain_position(explain._Games(AnalysisContext("t", [])), pos, _result(pos), None, frozenset())
    assert ex.text.startswith("Stockfish prefers 5...Nf6 (−0.39 for you) to your 5...e5 (−1.57 for you)")
    assert ex.diagram.strips[1].title == "Stockfish's 5...Nf6"


def test_a_choice_point_below_the_openings_bar_is_not_explained_as_an_error(motif_hooks, monkeypatch):
    """Under engine_min_drop win-% points on average in the analysed games, Stockfish's move and yours stand side by
    side: no "Stockfish prefers", no comparison of what it wins, no motifs, drills or concepts."""
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork"}))
    motif_hooks["refutation"] = [Motif(theme="fork", line="refutation", ply=1, squares=["b5"], side="opponent")]
    games = explain._Games(AnalysisContext("t", []))
    pos = dataclasses.replace(_position(kind="choice"), drop=2.3)
    ex = explain.explain_position(games, pos, _result(pos), _NoConcepts(), frozenset({"fork"}))
    assert ex.text == ("Stockfish's first choice here is 5...Nf6 (−0.39 for you); after your 5...e5 (−1.57 for you) "
                       "its main line is 6.Ndb5 a6 7.Nd6+ Bxd6. In the games it analysed, your 5...e5 lost 2.3 "
                       "percentage points of winning chances on average, less than an inaccuracy: in the opening "
                       "several moves are often about equally good.")
    assert "prefers" not in ex.text and "Compared with" not in ex.text
    assert ex.motifs == [] and ex.drill_themes == [] and ex.chart is None and ex.diagram.marks == []
    assert ex.diagram.strips[1].title == "Stockfish's 5...Nf6"
    # no analysed game with the move (drop 0.0): no average is quoted, and still no verdict
    pos = dataclasses.replace(pos, drop=0.0)
    ex = explain.explain_position(games, pos, _result(pos), _NoConcepts(), frozenset({"fork"}))
    assert ex.text.startswith("Stockfish's first choice here is 5...Nf6") and "percentage points" not in ex.text
    # the run's own bar (openings.engine_min_drop) decides: at 2 points this is a mistake
    pos = dataclasses.replace(pos, drop=2.3)
    ex = explain.explain_position(games, pos, _result(pos), None, frozenset({"fork"}), choice_min_drop=2.0)
    assert ex.text.startswith("Stockfish prefers 5...Nf6") and ex.drill_themes == ["fork"]


def test_evaluations_past_ten_pawns_are_named_not_printed(motif_hooks):
    pos = _position(kind="error")
    games = explain._Games(AnalysisContext("t", []))
    lost = _result(pos, best_cp=-120, ref_cp=-3468)
    ex = explain.explain_position(games, pos, lost, None, frozenset())
    assert "34.68" not in ex.text and "Stockfish rates the line as a lost position for you, against −1.20" in ex.text
    assert ex.diagram.strips[0].frames[-1].caption == "a lost position for you"
    won = _result(pos, best_cp=2150, ref_cp=150)
    ex = explain.explain_position(games, pos, won, None, frozenset())
    assert "21.50" not in ex.text and "against a winning position for you after 5...Nf6" in ex.text
    assert explain.verdict(won.best_line, "black", short=True) == "a winning position for you"
    assert explain.verdict(won.refutation, "black") == "+1.50 for you"
    assert explain.verdict(make_line(pos.fen, [], 1000, None, 20), "black") == "+10.00 for you"  # 10 pawns: a number
    assert explain.verdict(make_line(pos.fen, [], -1000, -3, 20), "black") == "mate in 3 for White"
    deltas = [ConceptDelta("King safety", -12.4, label="king safety"), ConceptDelta("Mobility", -0.38,
              label="piece activity")]
    assert explain._what_it_wins(Comparison(deltas=deltas), gap=5.0) == (
        "a few moves later you stand far worse on king safety and 0.38 on piece activity")
    deltas = [ConceptDelta("King safety", -12.4, label="king safety"), ConceptDelta("Mobility", -10.5,
              label="piece activity")]
    assert explain._what_it_wins(Comparison(deltas=deltas), gap=5.0) == (
        "a few moves later you stand far worse on king safety and far worse on piece activity")


def test_the_explanation_carries_the_format_of_its_games():
    """The format views keep an explanation by its time class: set from the games whenever they share one."""
    blitz = [make_game(time_class="blitz") for _ in range(2)]
    bullet = make_game(time_class="bullet")
    games = explain._Games(AnalysisContext("t", blitz + [bullet]))
    pos = dataclasses.replace(_position(), time_class="", games=[g.url for g in blitz], url=blitz[0].url,
                              game_id=blitz[0].game_id)
    assert games.single_format(pos) == "blitz"
    ex = explain.explain_position(games, pos, _result(pos), None, frozenset())
    assert ex.time_class == "blitz" and ex.diagram.time_class == "blitz"
    assert games.single_format(dataclasses.replace(pos, games=[blitz[0].url, bullet.url])) == ""
    one = dataclasses.replace(pos, games=[], url="", game_id=bullet.game_id, repeats=1)
    assert games.single_format(one) == "bullet"
    # games the context does not hold: the position's own format
    assert games.single_format(dataclasses.replace(pos, games=["https://elsewhere/1"], time_class="rapid")) == "rapid"


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
    assert "pawns" in chart.note and "Stockfish 16 classical evaluation terms" in [s.name for s in ex.sources]


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
    # the explanation matches the second puzzle's position and move: its text (braces made safe) and themes. It names
    # the better move, so it comes after the solution's first move; before it the puzzle still only asks
    assert [m.uci() for m in games[1].mainline_moves()] == res.best_line.moves_uci[: puzzles.SOLUTION_PLIES]
    assert games[1].comment.startswith("In the game you played 5...e5") and "Nf6" not in games[1].comment
    first_move = games[1].next()
    assert first_move.move.uci() == "g8f6" and first_move.comment == "After 5...e5, White plays (6.Ndb5)."
    assert "Themes" not in games[1].headers  # the explanation has no motif along its best line
    # an explanation is matched by position and move, not by its label: "6...e5" after a transposition still counts
    coaching.explanations[0].played = "6...e5"
    again = _read_all(puzzles.puzzles_pgn(events, coaching))
    assert again[1].next().comment == "After 5...e5, White plays (6.Ndb5)."
    # nothing for the last one: exactly the single-move puzzle
    single = mistakes.puzzles_to_pgn([last]).replace("puzzle 1", f"puzzle {len(events)}")
    assert pgn.endswith(single) and "Themes" not in games[-1].headers


def test_puzzle_themes_describe_the_printed_solution_never_the_opponents_tactic(player, monkeypatch):
    """The Themes header of a puzzle names the patterns you carry out along the moves printed: not the opponent's
    tactic from the refutation (a "mateIn1" tag on a puzzle whose side to move has no mate), and not the patterns of
    another line (the motif profile's shorter line)."""
    ctx, _ = player
    events = mistakes.build_puzzles(ctx.games, ctx.evals)
    second = events[1]  # 5...e5, explained below
    res = _result(_position())
    theirs = Motif("fork", "refutation", 3, ["d6", "e8", "b7"], "opponent")
    x = Explanation(epd=second.epd, fen=second.fen, played=second.move_label, best="5...Nf6", best_line=res.best_line,
                    refutation=res.refutation, text="After 5...e5 ...", motifs=[theirs], drill_themes=["fork"])
    coaching = Coaching(explanations=[x])
    assert "Themes" not in _read_all(puzzles.puzzles_pgn(events, coaching))[1].headers
    x.motifs = [theirs, Motif("pin", "best", 1, ["c6"], "you"), Motif("skewer", "best", 9, ["a1"], "you")]
    assert _read_all(puzzles.puzzles_pgn(events, coaching))[1].headers["Themes"] == "pin"  # the skewer comes later
    # fill_puzzle_lines records an empty list for a line it sets without a pattern, so the motif profile (which
    # only fills keys without one) cannot tag it with its own line's patterns
    monkeypatch.setattr(motifs, "detect_line", lambda line, role, previous_fen=None: [])
    coaching = Coaching()
    board = chess.Board(second.fen)
    deep_best = make_line(second.fen, [board.parse_san("Nf6")], -39, None, 20)
    puzzles.fill_puzzle_lines(ctx, coaching, {(second.epd, second.uci): DeepResult(second.epd, second.uci,
                                                                                    deep_best, None)})
    key = puzzles.puzzle_key(second.game.game_id, second.ply)
    assert coaching.puzzle_lines[key].moves_san == ["Nf6"] and coaching.puzzle_themes[key] == []
    coaching.puzzle_themes.setdefault(key, ["skewer"])  # the motif profile leaves a settled key alone
    assert coaching.puzzle_themes[key] == []
    assert "Themes" not in _read_all(puzzles.puzzles_pgn(events, coaching))[1].headers
    # a deeper search that prefers the move played settles the key with no line: the puzzle stays the game
    # analysis's single move, and the motif profile adds no line of its own
    coaching = Coaching()
    own = make_line(second.fen, [chess.Move.from_uci(second.uci)], -20, None, 20)
    puzzles.fill_puzzle_lines(ctx, coaching, {(second.epd, second.uci): DeepResult(second.epd, second.uci, own, None)})
    assert key not in coaching.puzzle_lines and coaching.puzzle_themes[key] == []


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


def test_fill_puzzle_lines_keeps_the_exports_errors_and_the_explained_ones(player, monkeypatch):
    """Only errors the puzzle export can use (build_puzzles: at most PUZZLE_LIMIT, costliest, with a better move)
    and errors in explained positions get a line; best lines are read with the position before the last move."""
    ctx, _ = player
    seen = []

    def detect(line, role, previous_fen=None):
        seen.append((line.fen, previous_fen))
        return [Motif("fork", role, 0, ["d5"], "you")]

    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork"}))
    monkeypatch.setattr(motifs, "detect_line", detect)
    events = mistakes.collect_errors(ctx.games, ctx.evals)
    profile = [ProfileError(e.game.game_id, e.ply, "you", e.game.time_class, e.fen, e.uci, e.drop,
                            make_line(e.fen, [chess.Move.from_uci(next(m.uci() for m in chess.Board(e.fen).legal_moves
                                                                       if m.uci() != e.uci))], 0, None, 10))
               for e in events]
    real = puzzles.build_puzzles
    monkeypatch.setattr(puzzles, "build_puzzles", lambda games, evals: real(games, evals, limit=1))
    coaching = Coaching()
    puzzles.fill_puzzle_lines(ctx, coaching, {}, profile)
    rapid = next(e for e in events if e.move_label == "7.Re1")  # the costliest (40 points)
    assert set(coaching.puzzle_lines) == {puzzles.puzzle_key(rapid.game.game_id, rapid.ply)}
    assert set(coaching.puzzle_themes) == set(coaching.puzzle_lines)
    before = chess.Board()
    for san in rapid.game.moves_san[: rapid.ply - 1]:
        before.push_san(san)
    assert seen == [(rapid.fen, before.fen())]  # the position before 6...O-O, your opponent's last move
    # an explained position adds its errors (the 5...e5 habit, in four games)
    sic = next(e for e in events if e.move_label == "5...e5")
    coaching = Coaching(explanations=[Explanation(sic.epd, sic.fen, "5...e5", "5...Nf6", None,
                                                  make_line(sic.fen, [chess.Move.from_uci(sic.uci)], -150, None, 20),
                                                  text="After 5...e5 ...")])
    puzzles.fill_puzzle_lines(ctx, coaching, {}, profile)
    sic_keys = {puzzles.puzzle_key(e.game.game_id, e.ply) for e in events if e.move_label == "5...e5"}
    assert set(coaching.puzzle_lines) == sic_keys | {puzzles.puzzle_key(rapid.game.game_id, rapid.ply)}


# --------------------------------------------------------------------------- build_coaching
class _Fake:
    """A scripted engine: the first legal move in UCI order, +60 cp for the side to move (so every move but
    the first is an error of about 13 win-% points)."""

    def __init__(self):
        self.id, self.options = {"name": "Fake 1"}, {}

    def analyse(self, board, limit, multipv=None, game=None, info=None):
        infos = []
        for i, move in enumerate(sorted(board.legal_moves, key=lambda m: m.uci())[: multipv or 1]):
            infos.append({"score": chess.engine.PovScore(chess.engine.Cp(60 - 20 * i), board.turn), "pv": [move],
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
    assert coaching.settings["positions"] == 4 and coaching.settings["engine"] == "Fake 1"
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
    assert coaching.settings["positions"] == 4


def test_without_engine_data_coaching_still_runs_the_engine_free_steps(monkeypatch):
    from chess_insights import coach
    from chess_insights.coach import drills, openings_info
    from chess_insights.context import AnalysisContext
    from factories import make_game

    ran = []
    monkeypatch.setattr(openings_info, "annotate", lambda ctx, coaching, modules, cfg: ran.append("openings"))
    monkeypatch.setattr(drills, "annotate", lambda ctx, coaching, modules, cfg, today: ran.append("drills"))
    coaching = coach.build_coaching(AnalysisContext("me", [make_game()]), [], coach.CoachConfig())
    assert ran == ["openings", "drills"]
    assert coaching.explanations == [] and "--engine" in coaching.notes[0]


def test_concept_note_for_the_largest_difference_and_previous_position():
    from chess_insights.coach.explain import concept_note_for, previous_position
    from chess_insights.models import ConceptDelta

    note, cites = concept_note_for([ConceptDelta("Mobility", -0.3, label="piece activity"),
                                    ConceptDelta("King safety", -1.17, label="king safety")])
    assert note["label"] and note["text"] and "gutenberg.org" in note["url"]
    assert "king" in note["label"].lower()
    assert cites and all("Project Gutenberg" in s.name for s in cites)
    assert concept_note_for([ConceptDelta("Material", 1.0)]) == ({}, [])
    assert concept_note_for([]) == ({}, [])
    import chess

    board = chess.Board()
    for san in ["e4", "c5", "d4"]:
        board.push_san(san)
    before_d4 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    assert previous_position(None, ["e4", "c5", "d4"], board.epd()) == before_d4
    assert previous_position(None, ["e4", "c5", "d4"], chess.Board().epd()) is None
    assert previous_position(None, [], board.epd()) is None
