"""Tablebase-checked endings (coach/endgames): slips found on recorded tablebase answers, the table per ending type,
boards for the slips and the facts for explained endgame positions. No real network requests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import chess
import pytest

from chess_insights.coach import endgames
from chess_insights.coach.config import CoachConfig
from chess_insights.coach.sources import http, tablebase
from chess_insights.coach.sources.http import Fetcher
from chess_insights.context import AnalysisContext
from chess_insights.models import Coaching, Explanation, ModuleResult
from factories import make_game, make_game_eval, make_ply_eval

_spec = importlib.util.spec_from_file_location(
    "recorded_http", Path(__file__).parent / "fixtures" / "sources" / "recorded_http.py"
)
recorded = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recorded)  # type: ignore[union-attr]

ROOK_DRAW = "8/5k2/8/3R4/5P2/5K2/8/r7 w - - 0 1"  # recorded: a draw; Rd1 and Ra5 lose
KPK_LOST = "4k3/8/4K3/4P3/8/8/8/8 b - - 0 1"  # recorded: Black to move loses
KRK_WON = "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1"  # recorded: every move wins
KQK = "8/8/8/3k4/8/2K5/8/7Q w - - 0 1"
# Constructed (no such answer was recorded): a queen ending where one move keeps the win and one hangs the queen.
KQK_ANSWER = {
    "url": tablebase.url(KQK), "status": 200, "content_type": "application/json",
    "body": {
        "category": "win", "dtz": 13, "dtm": 13, "checkmate": False, "stalemate": False,
        "moves": [
            {"uci": "h1h5", "san": "Qh5+", "category": "loss", "dtz": -12, "dtm": -12},
            {"uci": "h1e4", "san": "Qe4+", "category": "draw", "dtz": 0, "dtm": 0},
        ],
    },
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = recorded.block_network(monkeypatch)
    monkeypatch.setattr(http, "default_sleep", lambda s: None)
    http.reset_throttle()
    yield
    assert not attempts, f"a test tried to reach the network: {attempts}"


@pytest.fixture
def session(monkeypatch):
    s = recorded.RecordedSession("tablebase_rook_ending", "tablebase_kpk_lost", "tablebase_krk").add(KQK_ANSWER)
    monkeypatch.setattr(http, "session_factory", lambda: s)
    return s


def ending_game(fen: str, color: str, moves: list[str], **kw):
    return make_game(initial_fen=fen, color=color, moves_san=moves, time_class=kw.pop("time_class", "blitz"), **kw)


def evals_for(games):
    return {g.game_id: make_game_eval(g.game_id, [make_ply_eval(ply=0)]) for g in games}


def test_endgame_type_from_material():
    assert endgames.endgame_type(chess.Board("8/8/8/8/8/4k3/4P3/4K3 w - - 0 1")) == "Pawn"
    assert endgames.endgame_type(chess.Board(ROOK_DRAW)) == "Rook"
    assert endgames.endgame_type(chess.Board("8/8/8/4k3/8/2B5/p7/4K2N w - - 0 1")) == "Minor piece"
    assert endgames.endgame_type(chess.Board("8/8/8/4k3/8/8/1q6/Q3K3 w - - 0 1")) == "Queen"
    assert endgames.endgame_type(chess.Board("8/8/8/4k3/8/2b5/8/R3K3 w - - 0 1")) == "Mixed"


def test_a_drawn_rook_ending_lost_is_recorded_with_the_tablebase_move(tmp_path, session):
    f = Fetcher(tmp_path / "sources", sleep=lambda s: None)
    game = ending_game(ROOK_DRAW, "white", ["Rd1"])
    ending = endgames.check_game(f, game)
    assert ending.ending == "Rook" and ending.entry == "draw" and ending.had_draw and not ending.had_win
    (slip,) = ending.slips
    assert (slip.move, slip.before, slip.after, slip.best) == ("1.Rd1", "draw", "loss", "1.f5")
    assert slip.played_uci == "d5d1" and slip.best_uci == "f4f5"


def test_lost_and_won_endings_without_slips(tmp_path, session):
    f = Fetcher(tmp_path / "sources", sleep=lambda s: None)
    lost = endgames.check_game(f, ending_game(KPK_LOST, "black", ["Kd8"]))
    assert lost.ending == "Pawn" and lost.entry == "loss" and lost.slips == []
    won = endgames.check_game(f, ending_game(KRK_WON, "white", ["Ra5+"]))
    assert won.entry == "win" and won.had_win and won.slips == []


def test_the_first_small_position_with_the_opponent_to_move(tmp_path, session):
    """You as White, Black to move in a lost pawn ending: the first position is asked about too (win for you)."""
    f = Fetcher(tmp_path / "sources", sleep=lambda s: None)
    ending = endgames.check_game(f, ending_game(KPK_LOST, "white", ["Kd8"]))
    assert ending.entry == "win" and ending.slips == [] and len(session.calls) == 1


def test_games_with_more_than_7_pieces_are_not_asked_about(tmp_path, session):
    f = Fetcher(tmp_path / "sources", sleep=lambda s: None)
    assert endgames.check_game(f, make_game()) is None
    assert session.calls == []


def test_annotate_builds_the_table_boards_and_explanation_facts(tmp_path, session):
    games = [
        ending_game(KPK_LOST, "black", ["Kd8"], time_class="rapid"),
        ending_game(KRK_WON, "white", ["Ra5+"]),
        ending_game(KQK, "white", ["Qe4+"], time_class="bullet"),
        ending_game(ROOK_DRAW, "white", ["Rd1"]),  # the most recent
        make_game(),  # never reaches 7 pieces
    ]
    unanalysed = ending_game(ROOK_DRAW, "white", ["Ra5"])  # no engine analysis: not checked
    ctx = AnalysisContext("tester", games + [unanalysed], evals=evals_for(games))
    explained = Explanation(epd=" ".join(ROOK_DRAW.split()[:4]), fen=ROOK_DRAW, played="1.Rd1", best="1.f5",
                            best_line=None, refutation=None, time_class="blitz", color="white")
    opening = Explanation(epd="", fen=chess.STARTING_FEN, played="1.e4", best="1.d4", best_line=None, refutation=None)
    coaching = Coaching(explanations=[explained, opening])
    engine = ModuleResult(key="engine", title="Engine review", summary="")
    endgames.annotate(ctx, coaching, CoachConfig(sources_cache=tmp_path / "sources"), modules=[engine])

    assert explained.tablebase["result"] == "draw" and explained.tablebase["best"] == "f5"
    assert explained.tablebase["played"] == "Rd1" and explained.tablebase["played_result"] == "loss"
    assert any(s.name == "Lichess tablebase" for s in explained.sources)
    assert opening.tablebase == {}

    table = coaching.endgames
    assert table.columns[:6] == ["Ending", "Games", "Had a win", "Wins let slip", "Had a draw", "Draws lost"]
    rows = {r[0]: r for r in table.rows}
    assert list(rows) == ["Pawn endings", "Rook endings", "Queen endings"]
    assert rows["Rook endings"][1:6] == [2, 1, 0, 1, 1]
    assert rows["Rook endings"][6] == "1.Rd1 (a draw to a loss; tablebase: 1.f5)"
    assert rows["Rook endings"][7] == games[3].url
    assert rows["Queen endings"][1:6] == [1, 1, 1, 0, 0]
    assert rows["Pawn endings"][1:6] == [1, 0, 0, 0, 0] and rows["Pawn endings"][6] == ""
    assert rows["Rook endings"][8] == "2 blitz" and rows["Queen endings"][8] == "1 bullet"
    assert "4 of your 5 engine-analysed games" in table.note and "bullet, blitz and rapid" in table.note

    # boards for the slips, most recent first: your move red, the tablebase's green
    rook, queen = engine.diagrams
    assert rook.title == "Rook ending: 1.Rd1 turned a draw into a loss" and rook.link == games[3].url
    assert [(a.start, a.end, a.kind) for a in rook.arrows] == [("d5", "d1", "played"), ("f4", "f5", "best")]
    # the tablebase counts DTM in plies: 13 plies is mate in 7 moves
    assert queen.time_class == "bullet" and "Tablebase: 1.Qh5+ keeps a win (mate in 7 moves" in queen.caption
    assert coaching.settings["tablebase"] == {"reached": 4, "endings": 4, "slips": 2, "requests": 4}
    assert not any("Ra5" in url for url, _ in session.calls)


def test_request_cap_and_unavailable_tablebase(tmp_path, monkeypatch, session):
    monkeypatch.setattr(endgames, "MAX_REQUESTS", 1)
    games = [ending_game(KRK_WON, "white", ["Ra5+"]), ending_game(ROOK_DRAW, "white", ["Rd1"])]
    ctx = AnalysisContext("tester", games, evals=evals_for(games))
    coaching = Coaching()
    endgames.annotate(ctx, coaching, CoachConfig(sources_cache=tmp_path / "sources"))
    assert [r[0] for r in coaching.endgames.rows] == ["Rook endings"] and coaching.endgames.rows[0][1] == 1
    assert "request limit" in coaching.endgames.note
    # the note says both games reached the tablebase but only one was checked, not "1 of your 2 reached"
    assert "2 of your 2 engine-analysed games reached 7 pieces or fewer; 1 of them (blitz only)" in (
        coaching.endgames.note
    )
    assert coaching.settings["tablebase"]["reached"] == 2 and coaching.settings["tablebase"]["endings"] == 1
    assert any("request limit" in n for n in coaching.notes)

    # a timeout: no table, one note, and the run goes on
    timeout = recorded.TimeoutSession()
    monkeypatch.setattr(http, "session_factory", lambda: timeout)
    coaching = Coaching()
    endgames.annotate(ctx, coaching, CoachConfig(sources_cache=tmp_path / "other"))
    assert coaching.endgames is None and len(timeout.calls) == 1
    assert coaching.notes == ["Lichess tablebase did not answer in time; skipped for the rest of this report."]


def test_offline_uses_only_cached_answers(tmp_path, session):
    games = [ending_game(ROOK_DRAW, "white", ["Rd1"])]
    ctx = AnalysisContext("tester", games, evals=evals_for(games))
    endgames.annotate(ctx, Coaching(), CoachConfig(sources_cache=tmp_path / "sources"))
    calls = len(session.calls)
    coaching = Coaching()
    endgames.annotate(ctx, coaching, CoachConfig(sources_cache=tmp_path / "sources", offline=True))
    assert len(session.calls) == calls and coaching.endgames.rows[0][5] == 1
    coaching = Coaching()
    endgames.annotate(ctx, coaching, CoachConfig(sources_cache=tmp_path / "empty", offline=True))
    assert coaching.endgames is None and any("offline" in n for n in coaching.notes)


def test_a_slip_names_a_move_that_keeps_the_result(tmp_path, monkeypatch):
    """An answer whose moves are not sorted best first: the slip's move is one that keeps the draw."""
    doc = recorded.load("tablebase_rook_ending")
    shuffled = dict(doc, body=dict(doc["body"], moves=list(reversed(doc["body"]["moves"]))))
    s = recorded.RecordedSession().add(shuffled)
    monkeypatch.setattr(http, "session_factory", lambda: s)
    ending = endgames.check_game(Fetcher(tmp_path / "sources", sleep=lambda s: None),
                                 ending_game(ROOK_DRAW, "white", ["Rd1"]))
    (slip,) = ending.slips
    kept = tablebase.parse(doc["body"], ROOK_DRAW).move(slip.best_uci)
    assert slip.best != "1.Rd1" and kept.outcome_for_mover == "draw"


def test_games_counted_when_the_tablebase_is_out(tmp_path, monkeypatch):
    """After a timeout the other endings are counted as reached but never asked about."""
    games = [ending_game(ROOK_DRAW, "white", ["Rd1"]), ending_game(KRK_WON, "white", ["Ra5+"]), make_game()]
    ctx = AnalysisContext("tester", games, evals=evals_for(games))
    timeout = recorded.TimeoutSession()
    monkeypatch.setattr(http, "session_factory", lambda: timeout)
    coaching = Coaching()
    endgames.annotate(ctx, coaching, CoachConfig(sources_cache=tmp_path / "sources"))
    assert len(timeout.calls) == 1 and coaching.endgames is None
    assert coaching.settings["tablebase"]["reached"] == 2 and coaching.settings["tablebase"]["endings"] == 0
    assert endgames.reaches_tablebase(games[0]) and not endgames.reaches_tablebase(games[2])
