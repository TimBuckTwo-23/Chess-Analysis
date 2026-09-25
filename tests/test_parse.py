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


def test_daily_clocks_are_dropped(games):
    g = by_url(games, "605000000005")
    assert g.clocks == [None] * g.plies  # archived daily [%clk] = time spent / 10, not remaining time


def test_pregame_ratings_are_reconstructed():
    from chess_insights.parse import fill_pregame_ratings
    from factories import make_game

    g1 = make_game(my_rating=1500, opp_rating=1480)
    g2 = make_game(my_rating=1508, opp_rating=1600)  # won +8 -> before: me 1500, opp 1608
    g3 = make_game(my_rating=1400, opp_rating=1300)  # -108 swing: keep opponent's post-game rating
    g4 = make_game(my_rating=1400, opp_rating=1450, rated=False)
    g5 = make_game(my_rating=1300, opp_rating=1300, time_class="rapid")  # separate pool
    fill_pregame_ratings([g1, g2, g3, g4, g5])
    assert (g1.my_rating_before, g1.opp_rating_before, g1.rating_diff) == (None, None, -20)
    assert (g2.my_rating_before, g2.opp_rating_before, g2.rating_diff) == (1500, 1608, 108)
    assert (g3.my_rating_before, g3.opp_rating_before) == (1508, 1300)
    assert (g4.my_rating_before, g4.opp_rating_before) == (1400, 1450)
    assert g5.my_rating_before is None


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
        g["url"] = f"https://www.chess.com/game/live/{900000000 + i}"  # distinct games have distinct URLs
        many.append(g)
    t0 = time.perf_counter()
    parsed = parse.parse_games(many, "testerbob")
    assert len(parsed) == 3000
    assert time.perf_counter() - t0 < 6.0


def test_pgn_with_bom_and_windows_line_endings(fixtures_dir):
    text = (fixtures_dir / "sample_games.pgn").read_text(encoding="utf-8")
    plain = parse.games_from_pgn(text, "testerbob")
    windows = parse.games_from_pgn("\ufeff" + text.replace("\n", "\r\n"), "testerbob")
    assert [(g.game_id, g.moves_san, g.clocks, g.outcome, g.opening) for g in windows] == [
        (g.game_id, g.moves_san, g.clocks, g.outcome, g.opening) for g in plain
    ]
    first = parse.split_pgn("\ufeff" + text)[0]
    assert parse.parse_pgn_headers(first)["Event"] == "Live Chess"


def test_pgn_import_recognises_variants_and_odds_games():
    def pgn(extra, event="Live Chess"):
        return (
            f'[Event "{event}"]\n[Site "Chess.com"]\n[White "me"]\n[Black "you"]\n[Result "1-0"]\n{extra}'
            '[TimeControl "180"]\n[Termination "me won by resignation"]\n[Link "https://www.chess.com/game/live/9"]\n\n1. e4 e5 1-0\n'
        )

    odds = pgn('[SetUp "1"]\n[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R2QKB1R w KQkq - 0 1"]\n', "Live Chess - Odds Chess")
    assert parse.games_from_pgn(odds, "me")[0].rules == "oddschess"
    koth = pgn('[Variant "King of the Hill"]\n', "Live Chess - King of the Hill")
    assert parse.games_from_pgn(koth, "me")[0].rules == "kingofthehill"
    c960 = pgn('[Variant "Chess960"]\n[SetUp "1"]\n[FEN "bqnbrknr/pppppppp/8/8/8/8/PPPPPPPP/BQNBRKNR w HEhe - 0 1"]\n')
    assert parse.games_from_pgn(c960, "me")[0].rules == "chess960"
    assert parse.games_from_pgn(pgn(""), "me")[0].rules == "chess"


def test_parse_games_survives_any_malformed_record():
    raws = [
        {"white": "https://api.chess.com/pub/player/tester", "black": "https://api.chess.com/pub/player/x"},
        {"white": {"username": "tester", "result": "win"}, "black": {"username": "x", "result": "resigned"},
         "end_time": 1700000000, "accuracies": "n/a"},
        {"white": {"username": "tester", "result": "win"}, "black": {"username": "x", "result": "resigned"},
         "end_time": 1700000000, "pgn": 12345},
        None,
        "not a game",
    ]
    games = parse.parse_games(raws, "tester")
    assert all(g.outcome == "win" for g in games)
