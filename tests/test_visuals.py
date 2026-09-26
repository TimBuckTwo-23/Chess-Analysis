import chess

from chess_insights import visuals
from chess_insights.models import Mark
from factories import make_game


def test_format_counts_text_and_order():
    games = [make_game(time_class=tc) for tc in ["rapid", "bullet", "blitz", "bullet"]]
    assert list(visuals.format_counts(games).items()) == [("bullet", 2), ("blitz", 1), ("rapid", 1)]
    assert visuals.format_text({"blitz": 3}) == "blitz only"
    assert visuals.format_text({"rapid": 1, "bullet": 2}) == "bullet and rapid"
    assert visuals.format_text({"rapid": 1, "bullet": 2, "blitz": 1}) == "bullet, blitz and rapid"
    assert visuals.format_text({}) == ""


def test_split_by_format_skips_small_formats():
    games = [make_game(time_class="blitz", outcome="win")] * 12 + [make_game(time_class="rapid")] * 3
    labels, values, counts = visuals.split_by_format(games, lambda gs: sum(g.score for g in gs) / len(gs))
    assert labels == ["Blitz"] and values == [1.0] and counts == [12]


def test_position_diagram_arrows_and_castling():
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    d = visuals.position_diagram("x", fen, played="O-O", best="d3", others=[("Nc3", "neutral"), ("zz", "line")])
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("e1", "g1", "played"), ("d2", "d3", "best"),
                                                            ("b1", "c3", "neutral")]
    assert visuals.position_diagram("x", fen, played="e1g1").arrows[0].end == "g1"
    assert visuals.position_diagram("x", "not a fen", played="e4").arrows == []


def test_line_strip_numbers_moves_and_stops_at_illegal():
    start = chess.Board().fen()
    strip = visuals.line_strip("t", start, ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4"], max_frames=4,
                               captions=["a"], marks=[[Mark("e4")]])
    assert [f.move for f in strip.frames] == ["1.e4", "1...c5", "2.Nf3", "2...Nc6"]
    assert strip.frames[0].caption == "a" and strip.frames[0].marks[0].square == "e4"
    assert strip.frames[-1].last_move == "b8c6"
    assert len(visuals.line_strip("t", start, ["e2e4", "e2e4"]).frames) == 1


def test_numbered_moves_threats_and_shredder_fens():
    sicilian = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
    d = visuals.position_diagram("x", sicilian, played="5...e5", best="5... a6", others=[("Ndb5", "threat")])
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("e6", "e5", "played"), ("a7", "a6", "best"),
                                                            ("d4", "b5", "threat")]
    assert visuals.position_diagram("x", chess.Board().fen(), played="1.e4").arrows[0].end == "e4"
    # a threat is not read for the side to move when it isn't a threat
    assert visuals.position_diagram("x", sicilian, others=[("Ndb5", "neutral")]).arrows == []
    shredder = "4k3/8/8/8/8/8/8/R3K2R w HA - 0 1"
    castle = visuals.position_diagram("x", shredder, played="O-O").arrows
    assert castle and castle[0].start == "e1"
