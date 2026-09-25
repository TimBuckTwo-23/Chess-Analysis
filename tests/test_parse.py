import json
import time
from datetime import datetime, timezone

import chess
import pytest

from chess_insights import parse


@pytest.fixture(scope="module")
def raw_games(fixtures_dir):
    with open(fixtures_dir / "chesscom_archive_sample.json") as fh:
        return json.load(fh)["games"]


@pytest.fixture(scope="module")
def games(raw_games):
    return parse.parse_games(raw_games, "TesterBob")


def by_url(games, suffix):
    return next(g for g in games if g.url.endswith(suffix))


def test_skips_foreign_games_and_dedupes(games, raw_games):
    assert len(raw_games) == 10
    assert len(games) == 8  # one game without the player, one exact duplicate
    assert [g.end_time for g in games] == sorted(g.end_time for g in games)


def test_white_win_by_resignation(games):
    g = by_url(games, "105000000001")
    assert (g.color, g.outcome, g.termination) == ("white", "win", "resignation")
    assert (g.my_rating, g.opp_rating, g.opponent) == (1512, 1498, "knightrider77")
    assert (g.time_class, g.base_seconds, g.increment, g.rated) == ("blitz", 300, 0, True)
    assert g.eco == "C50"
    assert g.opening_family == "Italian Game"
    assert g.opening == "Italian Game: Giuoco Pianissimo Italian Four Knights Variation"
    assert (g.my_accuracy, g.opp_accuracy) == (88.4, 71.9)
    assert g.moves_san[:5] == ["e4", "e5", "Nf3", "Nc6", "Bc4"]
    assert len(g.moves_san) == len(g.clocks) == 32
    assert g.clocks[0] == pytest.approx(296.9)
    assert g.start_time == datetime(2024, 3, 2, 18, 2, tzinfo=timezone.utc)
    assert g.end_time.tzinfo is not None


def test_black_timeout_loss_case_insensitive_username(games):
    g = by_url(games, "105000000002")
    assert (g.color, g.outcome, g.termination, g.my_result_code) == ("black", "loss", "timeout", "timeout")
    assert (g.base_seconds, g.increment) == (180, 2)
    assert g.opening == "Caro-Kann Defense: Advance Variation"
    assert g.opening_family == "Caro-Kann Defense"
    assert g.username == "testerbob"


def test_draw_by_repetition(games):
    g = by_url(games, "105000000003")
    assert (g.outcome, g.termination, g.score) == ("draw", "repetition", 0.5)
    assert g.opening_family == "Queen's Gambit"
    assert g.opening == "Queen's Gambit: Declined Queen's Knight Variation"


def test_checkmated_bullet(games):
    g = by_url(games, "105000000004")
    assert (g.color, g.outcome, g.termination, g.time_class) == ("white", "loss", "checkmate", "bullet")
    assert g.moves_san == ["f3", "e5", "g4", "Qh4#"]


def test_daily_game(games):
    g = by_url(games, "605000000005")
    assert (g.time_class, g.base_seconds, g.increment) == ("daily", 86400, 0)
    assert g.outcome == "win" and g.termination == "checkmate"
    assert g.start_time == datetime(2024, 3, 7, 9, 0, tzinfo=timezone.utc)
    assert g.moves_san[-1] == "Qxf7#"


def test_chess960_keeps_initial_fen(games):
    g = by_url(games, "105000000006")
    assert g.rules == "chess960"
    assert g.initial_fen.startswith("bqnbrknr/")
    assert g.opening is None and g.opening_family is None


def test_abandoned_zero_move_game(games):
    g = by_url(games, "105000000007")
    assert (g.outcome, g.termination, g.plies) == ("loss", "abandoned", 0)


def test_casual_agreed_draw_with_increment(games):
    g = by_url(games, "105000000009")
    assert (g.rated, g.outcome, g.termination) == (False, "draw", "agreement")
    assert (g.time_class, g.base_seconds, g.increment) == ("rapid", 900, 10)
    assert g.opening_family == "Petrov's Defense"


def test_moves_are_legal_when_replayed(games):
    for g in games:
        board = chess.Board(g.initial_fen or chess.STARTING_FEN, chess960=g.rules == "chess960")
        for san in g.moves_san:
            board.push_san(san)


@pytest.mark.parametrize(
    "tc,expected",
    [("600", (600, 0)), ("180+2", (180, 2)), ("1/86400", (86400, 0)), ("-", (None, 0)), ("", (None, 0)), ("abc", (None, 0))],
)
def test_parse_time_control(tc, expected):
    assert parse.parse_time_control(tc) == expected


@pytest.mark.parametrize(
    "tc,cls",
    [("60", "bullet"), ("120+1", "bullet"), ("180", "blitz"), ("300+5", "blitz"), ("600", "rapid"), ("1/259200", "daily")],
)
def test_time_class_for(tc, cls):
    assert parse.time_class_for(tc) == cls


@pytest.mark.parametrize(
    "url,opening,family",
    [
        ("https://www.chess.com/openings/Sicilian-Defense-Open-2...d6-3.d4-cxd4-4.Nxd4-Nf6-5.Nc3", "Sicilian Defense: Open", "Sicilian Defense"),
        ("https://www.chess.com/openings/Queens-Pawn-Opening-Accelerated-London-System", "Queen's Pawn Opening: Accelerated London System", "Queen's Pawn Opening"),
        ("https://www.chess.com/openings/Kings-Indian-Defense", "King's Indian Defense", "King's Indian Defense"),
        ("https://www.chess.com/openings/Ruy-Lopez-Opening-Morphy-Defense", "Ruy Lopez Opening: Morphy Defense", "Ruy Lopez Opening"),
        ("https://www.chess.com/openings/Nimzo-Indian-Defense-Rubinstein-Variation", "Nimzo-Indian Defense: Rubinstein Variation", "Nimzo-Indian Defense"),
        ("https://www.chess.com/openings/Van-Geet-Opening", "Van Geet Opening", "Van Geet Opening"),
        (None, None, None),
        ("https://www.chess.com/openings/", None, None),
    ],
)
def test_opening_from_eco_url(url, opening, family):
    assert parse.opening_from_eco_url(url) == (opening, family)


def test_movetext_ignores_variations_nags_and_annotations():
    pgn = '[Event "x"]\n\n1. e4! {[%clk 0:05:00]} 1... e5?! $2 (1... c5 2. Nf3) 2. Nf3 {a comment} Nc6 3. Bb5 a6 0-1'
    moves, clocks = parse.parse_movetext(pgn)
    assert moves == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    assert clocks[0] == 300.0 and clocks[1] is None


def test_parse_clock_handles_long_daily_clocks():
    assert parse.parse_clock("0:02:59.9") == pytest.approx(179.9)
    assert parse.parse_clock("71:59:40") == 71 * 3600 + 59 * 60 + 40


def test_termination_and_outcome_mapping():
    assert parse.normalize_termination("win", "checkmated") == "checkmate"
    assert parse.normalize_termination("timeout", "win") == "timeout"
    assert parse.normalize_termination("timevsinsufficient", "timevsinsufficient") == "timeout_vs_insufficient"
    assert parse.normalize_termination("50move", "50move") == "fifty_move"
    assert parse.outcome_for("abandoned", "win") == "loss"
    assert parse.outcome_for("mystery", "mystery", "0-1", "black") == "win"


def test_pgn_import_matches_json_import(fixtures_dir, games):
    text = (fixtures_dir / "sample_games.pgn").read_text()
    pgn_games = parse.games_from_pgn(text, "testerbob")
    assert len(pgn_games) == 5
    json_by_url = {g.url: g for g in games}
    for g in pgn_games:
        ref = json_by_url[g.url]
        assert (g.color, g.outcome, g.termination, g.time_class) == (ref.color, ref.outcome, ref.termination, ref.time_class)
        assert g.moves_san == ref.moves_san
        assert g.my_rating == ref.my_rating
        assert g.opening == ref.opening


def test_malformed_records_are_skipped():
    raws = [{"white": {"username": "tester"}, "black": {"username": "x"}, "end_time": "not-a-number"}, {}, {"white": None}]
    assert parse.parse_games(raws, "tester") == []


def test_parsing_is_fast(raw_games):
    many = []
    for i in range(3000):
        g = dict(raw_games[i % 3])
        g["uuid"] = f"u{i}"
        many.append(g)
    t0 = time.perf_counter()
    parsed = parse.parse_games(many, "testerbob")
    assert len(parsed) == 3000
    assert time.perf_counter() - t0 < 6.0
