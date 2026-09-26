"""The inline-SVG board renderer (report/boards.py): pieces from one sprite, arrows, marks, words, bad input."""

from __future__ import annotations

import re

import chess
import pytest

from chess_insights import visuals
from chess_insights.models import Arrow, Mark
from chess_insights.report import boards

SICILIAN = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
ITALIAN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def _uses(svg: str) -> list[tuple[str, int, int]]:
    return [(p, int(x), int(y)) for p, x, y in re.findall(r'<use href="#ci-pc-(\w\w)" x="(\d+)" y="(\d+)"/>', svg)]


def test_board_draws_every_piece_from_the_sprite_with_arrows_and_words():
    d = visuals.position_diagram("5...e5", SICILIAN, orientation="black", played="e5", best="a6")
    svg = boards.board_svg(d.fen, orientation=d.orientation, arrows=d.arrows, label="Open Sicilian")
    assert svg.startswith('<svg class="cb" viewBox="0 0 360 360" role="img" aria-label="')
    assert len(_uses(svg)) == len(chess.Board(SICILIAN).piece_map())  # one <use> per piece, no inline drawings
    assert '<use href="#ci-dk" class="d"/>' in svg  # the dark squares come from the sprite too
    assert svg.count('class="a a-played"') == 1 and svg.count('class="a a-best"') == 1
    label = re.search(r'aria-label="([^"]*)"', svg).group(1)
    assert label == ("Open Sicilian. Black to move (you). Your move e5 (red), better a6 (green). "
                     "Board shown from Black&#x27;s side.")
    assert svg.count('class="co ') == 16  # a-h and 1-8 on the main board


def test_orientation_puts_the_players_side_at_the_bottom():
    start = chess.STARTING_FEN
    white = dict(((x, y), p) for p, x, y in _uses(boards.board_svg(start)))
    black = dict(((x, y), p) for p, x, y in _uses(boards.board_svg(start, orientation="black")))
    assert white[(0, 315)] == "wR" and white[(180, 315)] == "wK"  # a1 bottom-left, king on e1
    assert black[(315, 0)] == "wR" and black[(135, 0)] == "wK"  # from Black's side: a1 top-right
    assert black[(135, 315)] == "bK"  # e8 at the bottom, fourth from the right
    # coordinates follow the orientation: files run h..a from Black's side
    files = re.findall(r'<text class="co on-\w" x="\d+" y="357" text-anchor="end">(\w)</text>', boards.board_svg(start, orientation="black"))
    assert "".join(files) == "hgfedcba"


def test_mini_board_has_no_coordinates_and_marks_last_move():
    svg = boards.board_svg(SICILIAN, orientation="black", last_move="e6e5", mini=True, label="After 5...e5")
    assert 'class="cb cb--mini"' in svg and 'class="co' not in svg
    assert svg.count('class="lm"') == 2
    assert "aria-label=\"After 5...e5. Black to move (you). Board shown from Black&#x27;s side.\"" in svg
    assert "Black to move (your opponent)" in boards.board_svg(SICILIAN, mini=True)  # White at the bottom: you are White


def test_marks_arrows_kinds_and_bad_annotations():
    arrows = [Arrow("e2", "e4", "played"), Arrow("e2", "e4", "best"), Arrow("d8", "d4", "threat"),
              Arrow("g1", "f3", "line"), Arrow("b1", "c3", "zigzag"), Arrow("x9", "a1", "played"), Arrow("d4", "d4", "best")]
    marks = [Mark("d4", "target"), Mark("c6", "attacker"), Mark("f7", "weak"), Mark("e4", "focus"), Mark("z0", "focus")]
    svg = boards.board_svg(chess.STARTING_FEN, arrows=arrows, marks=marks)
    assert svg.count('class="a a-played"') == 1  # the duplicate e2-e4 is dropped
    assert 'class="a a-threat"' in svg and 'class="a a-line"' in svg
    assert 'class="a a-neutral"' in svg  # an unknown kind is drawn grey
    assert re.search(r'class="a a-best"><circle', svg)  # start == end: a ring
    assert 'class="rg rg-target"' in svg and 'class="rg rg-focus"' in svg
    assert 'class="tn"' in svg and 'class="wk"' in svg
    # the most important arrows are drawn last, on top
    order = re.findall(r'class="a a-(\w+)"', svg)
    assert order.index("played") > order.index("threat") > order.index("neutral")


def test_check_is_tinted_and_mate_is_described():
    mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # fool's mate
    svg = boards.board_svg(mate)
    assert 'fill="url(#ci-chk)"' in svg
    assert "White is checkmated" in svg


@pytest.mark.parametrize("fen", [None, "", "not a fen", 12, "8/8/8/8/8/8/8/8 w - - 0 1 `**x**` <b>", "rnbqkbnr/pppppppp/9/8 w"])
def test_bad_fen_renders_nothing(fen):
    assert boards.board_svg(fen, arrows=[Arrow("e2", "e4", "played")]) == ""


def test_labels_are_escaped():
    svg = boards.board_svg(chess.STARTING_FEN, label='<script>alert("x")</script>')
    assert "<script>" not in svg and "&lt;script&gt;" in svg


def test_sprite_is_hidden_without_display_none():
    sprite = boards.sprite_svg()
    assert 'width="0" height="0"' in sprite and 'aria-hidden="true"' in sprite and "position:absolute" in sprite
    assert "display:none" not in sprite
    ids = re.findall(r'id="([^"]+)"', sprite)
    assert len(ids) == len(set(ids)) == 14  # 12 pieces, the dark squares and the check glow
    assert all(i.startswith(boards.SPRITE_PREFIX) for i in ids)


def test_arrow_words_for_castling_and_either_side():
    board = chess.Board(ITALIAN)
    assert boards.arrows_text(board, [Arrow("e1", "g1", "played")]) == "Your move O-O (red)"
    # an arrow for the side not to move (a threat) is still named as a move
    assert boards.arrows_text(board, [Arrow("f6", "e4", "threat")]) == "Threat Nxe4 (orange)"
    assert boards.arrows_text(board, [Arrow("a1", "h8", "line")]) == "A1 to h8 (blue)"


def test_line_helpers():
    assert boards.numbered_moves(SICILIAN, ["e6e5", "d4b5", "a7a6", "b5d6"]) == "5...e5 6.Ndb5 a6 7.Nd6+"
    assert boards.numbered_moves(SICILIAN, [], ["e5", "Ndb5"]) == "5...e5 6.Ndb5"  # SAN works too
    assert boards.numbered_moves(SICILIAN, ["e6e5", "e2e4"]) == "5...e5"  # stops at the first illegal move
    assert boards.numbered_moves("bad", [], ["e5", "Nf3"]) == "e5 Nf3"
    assert boards.compact_labels(["6.Ndb5", "6...a6", "7.Nd6+", "7...Bxd6"]) == "6.Ndb5 a6 7.Nd6+ Bxd6"
    assert boards.compact_labels(["5...e5", "6.Ndb5"]) == "5...e5 6.Ndb5"
    url = "https://lichess.org/analysis/" + SICILIAN.replace(" ", "_")
    assert boards.fen_from_analysis_url(url) == SICILIAN
    assert boards.fen_from_analysis_url("https://evil.example/analysis/" + SICILIAN.replace(" ", "_")) == ""
    assert boards.epd(SICILIAN) == "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq -"


def test_one_square_arrow_keeps_a_visible_shaft():
    svg = boards.board_svg(SICILIAN, orientation="black", arrows=[Arrow("e6", "e5", "played")])
    x1, y1, x2, y2 = map(float, re.search(r'<line x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"/>', svg).groups())
    assert x1 == x2 and abs(y2 - y1) >= 15  # python-chess's full-size head would leave 0.15 of a square
    # a long arrow keeps python-chess's head (0.75 of a square long)
    long = boards.board_svg(chess.STARTING_FEN, arrows=[Arrow("d1", "h5", "played")])
    tip, *_ = re.search(r'<polygon points="([^"]+)"', long).group(1).split()
    lx, ly = map(float, re.search(r'<line [^>]*x2="([\d.]+)" y2="([\d.]+)"', long).groups())
    tx, ty = map(float, tip.split(","))
    assert abs(((tx - lx) ** 2 + (ty - ly) ** 2) ** 0.5 - 0.75 * boards.SQ) < 0.2


def test_promotion_arrows_never_name_a_piece_they_do_not_know():
    board = chess.Board("3r4/4P1k1/8/8/8/8/5K2/8 w - - 0 1")
    assert boards.arrows_text(board, [Arrow("e7", "e8", "played")]) == "Your move e8 (red)"  # e8=N or e8=Q?
    assert boards.arrows_text(board, [Arrow("e7", "d8", "best")]) == "Better exd8 (green)"


def test_chess960_castling_arrow_is_named_as_castling():
    fen = "4k3/8/8/8/8/8/8/1R2K1R1 w GB - 0 1"  # Shredder castling letters: read as Chess960
    board = boards.parse_board(fen)
    assert board is not None and board.chess960
    assert boards.arrows_text(board, [Arrow("e1", "g1", "played"), Arrow("e1", "c1", "best")]) == (
        "Your move O-O (red), better O-O-O (green)")
    assert boards.board_svg(fen, arrows=[Arrow("e1", "g1", "played")]).count('class="a a-played"') == 1
