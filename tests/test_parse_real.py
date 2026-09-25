"""Parser checks against real, recorded chess.com API responses (tests/fixtures/real_chesscom_games.json)."""

import json

import chess
import pytest

from chess_insights import parse
from chess_insights.dataset import filter_games


@pytest.fixture(scope="module")
def raws(fixtures_dir):
    return json.loads((fixtures_dir / "real_chesscom_games.json").read_text())["games"]


def players(raw):
    return [raw[s]["username"] for s in ("white", "black") if raw.get(s, {}).get("username")]


def test_every_real_game_parses_from_both_sides(raws):
    for raw in raws:
        for user in players(raw):
            g = parse.parse_game(raw, user)
            assert g is not None
            res = parse.parse_pgn_headers(raw.get("pgn") or "").get("Result")
            if res in ("1-0", "0-1", "1/2-1/2"):
                expected = "draw" if res == "1/2-1/2" else ("win" if (res == "1-0") == (g.color == "white") else "loss")
                assert g.outcome == expected, g.url


def test_real_games_replay_to_the_reported_final_position(raws):
    checked = 0
    for raw in raws:
        g = parse.parse_game(raw, raw["white"]["username"])
        if g.rules not in ("chess", "chess960") or not raw.get("pgn"):
            continue
        board = chess.Board(g.initial_fen or chess.STARTING_FEN, chess960=g.rules == "chess960")
        for san in g.moves_san:
            board.push_san(san)
        # compare placement, side to move and castling (chess.com always writes an e.p. square after a double push)
        assert board.fen().split()[:3] == raw["fen"].split()[:3], g.url
        checked += 1
    assert checked >= 20


def test_caruana_archive_month(raws):
    caruana = parse.parse_games(raws[:17], "fabianocaruana")
    assert len(caruana) == 17
    g = next(g for g in caruana if g.url.endswith("4861120901"))
    assert (g.color, g.outcome, g.termination, g.time_class) == ("black", "win", "resignation", "blitz")
    assert (g.my_accuracy, g.opp_accuracy) == pytest.approx((89.3367519583493, 85.16453042379537))
    assert g.opening == "English Opening: Four Knights Quiet Line"
    assert g.eco == "A28"
    assert len(g.clocks) == g.plies == 38 and all(c is not None for c in g.clocks)
    assert g.clocks[:4] == [179.5, 179.0, 178.6, 178.4]  # per ply, both sides: 1. c4 {0:02:59.5} 1... e5 {0:02:59} ...
    assert g.initial_fen is None  # initial_setup "" means the standard position


def test_chess960_uses_pgn_fen(raws):
    raw = next(r for r in raws if r.get("rules") == "chess960")
    g = parse.parse_game(raw, raw["white"]["username"])
    headers = parse.parse_pgn_headers(raw["pgn"])
    assert g.initial_fen == headers["FEN"]


def test_daily_games_have_no_clocks_and_a_start_time(raws):
    dailies = [parse.parse_game(r, r["white"]["username"]) for r in raws if r.get("time_class") == "daily"]
    assert dailies
    for g in dailies:
        assert all(c is None for c in g.clocks)
        assert g.start_time is not None and g.start_time <= g.end_time


def test_variants_parse_but_are_filtered_by_default(raws):
    variants = [parse.parse_game(r, r["white"]["username"]) for r in raws if r.get("rules") in ("oddschess", "bughouse")]
    assert {g.rules for g in variants} == {"oddschess", "bughouse"}
    bughouse = next(g for g in variants if g.rules == "bughouse")
    assert bughouse.plies == 0  # bughouse games have no PGN
    assert filter_games(variants) == []
