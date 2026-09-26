"""Chess boards as compact inline SVG, drawn from a FEN plus annotations.

Every board on a page references ONE piece sprite (:func:`sprite_svg`, python-chess's piece drawings) with
``<use href="#ci-pc-wN">``, so a report with hundreds of positions stays small. Squares, arrows, marks and the
last-move tint are coloured by CSS classes whose custom properties (light and dark values) live in the page's
stylesheet (``BOARD_LIGHT`` / ``BOARD_DARK`` here, merged into the theme tokens by ``report/html.py``).

Arrows follow one colour code everywhere in the report: your move red, the engine's better move green, a threat
orange, a line of play (or the move that has scored best for you) blue, anything else grey. A FEN python-chess cannot read renders nothing (never an exception).

The text helpers at the bottom (the board described in words, a line of play in numbered SAN, EPD keys) are
shared with the Markdown renderer.
"""

from __future__ import annotations

import html as _html
import math
import re
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import unquote, urlsplit

import chess
import chess.svg

from ..models import ARROW_KINDS, MARK_KINDS, Arrow, Mark

SQ = 45  # python-chess draws each piece in a 45 x 45 box: one square
SIZE = 8 * SQ
SPRITE_PREFIX = "ci-"  # every id the sprite defines starts with this
DARK_SQUARES_ID = "ci-dk"
CHECK_ID = "ci-chk"

# Arrow kinds in drawing order: the arrows that matter most are drawn last (on top).
_ARROW_ORDER = {kind: i for i, kind in enumerate(("neutral", "line", "threat", "played", "best"))}
ARROW_COLOUR_WORDS = {"played": "red", "best": "green", "threat": "orange", "line": "blue", "neutral": "grey"}
_RING_MARKS = frozenset({"focus", "target"})  # drawn as a ring over the piece; the other kinds tint the square

# Colour tokens (CSS custom properties) for the boards, light and dark theme. Pieces keep python-chess's white
# and black; the dark theme dims the squares a little but keeps enough contrast for black pieces.
BOARD_LIGHT = {
    "sq-light": "#E9E4D4",
    "sq-dark": "#8FA37E",
    "sq-last": "rgba(214,176,40,.42)",
    "sq-tint": "rgba(235,120,20,.38)",
    "sq-weak": "rgba(90,90,100,.34)",
    "ar-played": "#C62F2F",
    "ar-best": "#16833D",
    "ar-threat": "#E07000",
    "ar-line": "#2A66C9",
    "ar-neutral": "#5F636B",
    "ring-focus": "#2A66C9",
    "ring-target": "#C62F2F",
    "chk": "#E0301E",
}
BOARD_DARK = {
    "sq-light": "#C9C4B3",
    "sq-dark": "#768A65",
    "sq-last": "rgba(230,190,50,.45)",
    "sq-tint": "rgba(250,140,40,.42)",
    "sq-weak": "rgba(40,40,50,.42)",
    "ar-played": "#D93A3A",
    "ar-best": "#118A3E",
    "ar-threat": "#EA7A0C",
    "ar-line": "#2F6ED6",
    "ar-neutral": "#555A63",
    "ring-focus": "#2F6ED6",
    "ring-target": "#D93A3A",
    "chk": "#F0402A",
}

BOARD_CSS = """
.cb{display:block;width:100%;max-width:300px;height:auto;aspect-ratio:1;border-radius:4px;overflow:hidden;flex:none}
.cb--mini{max-width:110px;border-radius:3px}
.cb .l{fill:var(--sq-light)}
.cb .d{fill:var(--sq-dark)}
.cb .lm{fill:var(--sq-last)}
.cb .tn{fill:var(--sq-tint)}
.cb .wk{fill:var(--sq-weak)}
.cb .co{font:600 12px var(--font-sans);pointer-events:none}
.cb .co.on-l{fill:var(--sq-dark)}
.cb .co.on-d{fill:var(--sq-light)}
.cb .a{opacity:.86}
.cb .a line{stroke-width:9}
.cb .a circle{fill:none;stroke-width:4.5}
.cb .a-played line,.cb .a-played circle{stroke:var(--ar-played)}.cb .a-played polygon{fill:var(--ar-played)}
.cb .a-best line,.cb .a-best circle{stroke:var(--ar-best)}.cb .a-best polygon{fill:var(--ar-best)}
.cb .a-threat line,.cb .a-threat circle{stroke:var(--ar-threat)}.cb .a-threat polygon{fill:var(--ar-threat)}
.cb .a-line line,.cb .a-line circle{stroke:var(--ar-line)}.cb .a-line polygon{fill:var(--ar-line)}
.cb .a-neutral line,.cb .a-neutral circle{stroke:var(--ar-neutral)}.cb .a-neutral polygon{fill:var(--ar-neutral)}
.cb .rg{fill:none;stroke-width:4}
.cb .rg-focus{stroke:var(--ring-focus)}
.cb .rg-target{stroke:var(--ring-target)}
.ci-sprite .c0{stop-color:var(--chk);stop-opacity:1}
.ci-sprite .c1{stop-color:var(--chk);stop-opacity:.55}
.ci-sprite .c2{stop-color:var(--chk);stop-opacity:0}
"""


# --------------------------------------------------------------------------- the page-level sprite
def _piece_symbol(symbol: str) -> str:
    """python-chess's drawing of one piece, with our id and without its generic class names."""
    g = chess.svg.PIECES[symbol]
    g = re.sub(r'\s(?:id|class)="[^"]*"', "", g, count=2)
    colour = "w" if symbol.isupper() else "b"
    return g.replace("<g", f'<g id="{SPRITE_PREFIX}pc-{colour}{symbol.upper()}"', 1)


def _dark_squares_path() -> str:
    """Every dark square, with the top-left square light (true for both orientations: a8 and h1 are light)."""
    return "".join(f"M{c * SQ} {r * SQ}h{SQ}v{SQ}h-{SQ}z" for r in range(8) for c in range(8) if (r + c) % 2)


def sprite_svg() -> str:
    """The one hidden SVG every board refers to: 12 pieces, the dark squares and the check glow.

    Not ``display:none``: Chrome does not render ``<use>`` references into an SVG that is not rendered.
    """
    pieces = "".join(_piece_symbol(s) for s in "KQRBNPkqrbnp")
    return (
        '<svg class="ci-sprite" width="0" height="0" aria-hidden="true" focusable="false" '
        'style="position:absolute;width:0;height:0;overflow:hidden">'
        f'<defs><path id="{DARK_SQUARES_ID}" d="{_dark_squares_path()}"/>'
        f'<radialGradient id="{CHECK_ID}"><stop offset="0%" class="c0"/><stop offset="50%" class="c1"/>'
        f'<stop offset="100%" class="c2"/></radialGradient>{pieces}</defs></svg>'
    )


# --------------------------------------------------------------------------- one board
def parse_board(fen: Any) -> Optional[chess.Board]:
    """The position for a FEN (standard first, then Chess960; Chess960 first for Shredder castling letters such
    as "HFhf"); None when python-chess cannot read it."""
    if not isinstance(fen, str) or not fen.strip():
        return None
    text = " ".join(fen.split())
    fields = text.split(" ")
    shredder = len(fields) > 2 and bool(re.search(r"[A-Ha-h]", fields[2]))
    for chess960 in ((True, False) if shredder else (False, True)):
        try:
            return chess.Board(text, chess960=chess960)
        except ValueError:
            continue
    return None


def _square(name: Any) -> Optional[int]:
    try:
        return chess.parse_square(str(name).strip().lower())
    except (ValueError, AttributeError):
        return None


def _xy(square: int, flipped: bool) -> tuple[int, int]:
    """Top-left corner of ``square`` on the drawn board (``flipped``: Black at the bottom)."""
    f, r = chess.square_file(square), chess.square_rank(square)
    return ((7 - f) * SQ, r * SQ) if flipped else (f * SQ, (7 - r) * SQ)


def _n(x: float) -> str:
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _rect(cls: str, x: int, y: int) -> str:
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{SQ}" height="{SQ}"/>'


def _arrow_svg(start: int, end: int, kind: str, flipped: bool) -> str:
    """An arrow drawn like python-chess's: a shaft and a triangular head (a ring when start == end).

    A one-square move gets a shorter head, so its shaft still shows between the two squares (python-chess's
    full-size head covers all of it, and the arrow reads as a lone triangle on a small board).
    """
    x0, y0 = (v + SQ / 2 for v in _xy(start, flipped))
    x1, y1 = (v + SQ / 2 for v in _xy(end, flipped))
    if start == end:
        return f'<g class="a a-{kind}"><circle cx="{_n(x1)}" cy="{_n(y1)}" r="{_n(SQ * 0.45)}"/></g>'
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    head, margin = min(0.75 * SQ, 0.5 * length), 0.1 * SQ
    width = max(head, 0.6 * SQ)
    sx, sy = x1 - dx * (head + margin) / length, y1 - dy * (head + margin) / length
    tx, ty = x1 - dx * margin / length, y1 - dy * margin / length
    wx, wy = dy * 0.5 * width / length, dx * 0.5 * width / length
    points = f"{_n(tx)},{_n(ty)} {_n(sx + wx)},{_n(sy - wy)} {_n(sx - wx)},{_n(sy + wy)}"
    return (
        f'<g class="a a-{kind}"><line x1="{_n(x0)}" y1="{_n(y0)}" x2="{_n(sx)}" y2="{_n(sy)}"/>'
        f'<polygon points="{points}"/></g>'
    )


def _uci_squares(uci: Any) -> Optional[tuple[int, int]]:
    text = str(uci or "").strip().lower()
    if not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", text):
        return None
    return chess.parse_square(text[:2]), chess.parse_square(text[2:4])


def board_svg(
    fen: Any,
    *,
    orientation: str = "white",
    arrows: Iterable[Arrow] = (),
    marks: Iterable[Mark] = (),
    last_move: str = "",
    mini: bool = False,
    label: str = "",
    you: Optional[str] = None,
) -> str:
    """One board as inline SVG ("" for a FEN python-chess cannot read).

    ``orientation`` is the side at the bottom; ``mini`` draws a small strip frame (no coordinates). ``label``
    starts the board's accessible description; ``you`` ("white" / "black") names the analysed player's side
    in it (by default the side at the bottom).
    """
    board = parse_board(fen)
    if board is None:
        return ""
    flipped = str(orientation).lower() == "black"
    out = [f'<rect class="l" width="{SIZE}" height="{SIZE}"/><use href="#{DARK_SQUARES_ID}" class="d"/>']

    moved = _uci_squares(last_move)
    if moved:
        out += [_rect("lm", *_xy(sq, flipped)) for sq in dict.fromkeys(moved)]

    clean_marks = [m for m in marks or () if isinstance(m, Mark) and _square(m.square) is not None]
    check_squares = {_square(m.square) for m in clean_marks if m.kind == "check"}
    if board.is_check():
        king = board.king(board.turn)
        if king is not None:
            check_squares.add(king)
    for sq in sorted(s for s in check_squares if s is not None):
        x, y = _xy(sq, flipped)
        out.append(f'<rect x="{x}" y="{y}" width="{SQ}" height="{SQ}" fill="url(#{CHECK_ID})"/>')
    for m in clean_marks:
        if m.kind in ("attacker", "weak"):
            out.append(_rect("tn" if m.kind == "attacker" else "wk", *_xy(_square(m.square), flipped)))  # type: ignore[arg-type]

    if not mini:  # coordinates inside the edge squares, in the colour of the other square shade
        for i in range(8):
            file_char = "hgfedcba"[i] if flipped else "abcdefgh"[i]
            bottom_light = i % 2 == 1  # a1 (or h8 from Black's side) is dark
            out.append(
                f'<text class="co on-{"l" if bottom_light else "d"}" x="{i * SQ + SQ - 3}" y="{SIZE - 3}" '
                f'text-anchor="end">{file_char}</text>'
            )
            rank_char = str(i + 1) if flipped else str(8 - i)
            left_light = i % 2 == 0
            out.append(f'<text class="co on-{"l" if left_light else "d"}" x="3" y="{i * SQ + 12}">{rank_char}</text>')

    for sq, piece in sorted(board.piece_map().items()):
        x, y = _xy(sq, flipped)
        colour = "w" if piece.color == chess.WHITE else "b"
        out.append(f'<use href="#{SPRITE_PREFIX}pc-{colour}{piece.symbol().upper()}" x="{x}" y="{y}"/>')

    for m in clean_marks:
        if m.kind in _RING_MARKS:
            x, y = _xy(_square(m.square), flipped)  # type: ignore[arg-type]
            out.append(f'<circle class="rg rg-{m.kind}" cx="{x + SQ / 2:g}" cy="{y + SQ / 2:g}" r="{SQ / 2 - 3:g}"/>')

    clean_arrows = _clean_arrows(arrows)
    for a, start, end in drawing_order(clean_arrows):
        out.append(_arrow_svg(start, end, a.kind, flipped))

    words = describe_board(board, orientation="black" if flipped else "white", arrows=[a for a, _, _ in clean_arrows],
                           marks=clean_marks, label=label, you=you)
    cls = "cb cb--mini" if mini else "cb"
    return (
        f'<svg class="{cls}" viewBox="0 0 {SIZE} {SIZE}" role="img" aria-label="{_html.escape(words, quote=True)}" '
        f'focusable="false">{"".join(out)}</svg>'
    )


def _direction(start: int, end: int) -> tuple[int, int]:
    """An arrow's step reduced to lowest terms: arrows along the same line from one square share it."""
    df = chess.square_file(end) - chess.square_file(start)
    dr = chess.square_rank(end) - chess.square_rank(start)
    g = math.gcd(df, dr) or 1
    return df // g, dr // g


def drawing_order(arrows: Sequence[tuple[Arrow, int, int]]) -> list[tuple[Arrow, int, int]]:
    """The order to draw arrows in: the ones that matter most last, on top (``_ARROW_ORDER``). Arrows that leave
    the same square in the same direction (your d2-d3 and the engine's d2-d4) are drawn together, longest first:
    the shorter one on top keeps its shaft visible and the longer one's head sticks out beyond it, whatever their
    colours (drawn by kind alone, the green d2-d4 would cover the red d2-d3 but for a sliver of its head)."""
    groups: dict[tuple[int, tuple[int, int]], list[int]] = {}
    for i, (_, start, end) in enumerate(arrows):
        groups.setdefault((start, _direction(start, end)), []).append(i)
    keys = []
    for i, (a, start, end) in enumerate(arrows):
        group = groups[(start, _direction(start, end))]
        rank = max(_ARROW_ORDER[arrows[j][0].kind] for j in group)  # the group goes where its top arrow goes
        keys.append((rank, group[0], -chess.square_distance(start, end), i))
    return [arrows[i] for i in sorted(range(len(arrows)), key=keys.__getitem__)]


def _clean_arrows(arrows: Iterable[Arrow]) -> list[tuple[Arrow, int, int]]:
    """Arrows with readable squares, an allowed kind (unknown kinds become neutral) and no duplicates."""
    out: list[tuple[Arrow, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for a in arrows or ():
        if not isinstance(a, Arrow):
            continue
        start, end = _square(a.start), _square(a.end)
        if start is None or end is None or (start, end) in seen:
            continue
        seen.add((start, end))
        kind = a.kind if a.kind in ARROW_KINDS else "neutral"
        out.append((Arrow(chess.square_name(start), chess.square_name(end), kind), start, end))
    return out


# --------------------------------------------------------------------------- the board in words
def _find_move(board: chess.Board, start: int, end: int) -> Optional[chess.Move]:
    """The legal move an arrow start -> end stands for; castling arrows point at the king's destination (in
    Chess960 too, where python-chess encodes castling as king-takes-rook)."""
    try:
        return board.find_move(start, end)
    except (ValueError, AssertionError):
        pass
    for move in board.legal_moves:
        if move.from_square == start and board.is_castling(move):
            dest = chess.square(2 if board.is_queenside_castling(move) else 6, chess.square_rank(start))
            if dest == end:
                return move
    return None


def arrow_move_text(board: chess.Board, arrow: Arrow) -> str:
    """The arrow as a move in SAN ("Nf3", for either side), or "g1 to f3" when it is not a legal move.

    An arrow does not say which piece a pawn promotes to, so a promotion is named by its squares only ("e8",
    "exd8"), never as a queen it may not have been."""
    start, end = _square(arrow.start), _square(arrow.end)
    if start is None or end is None:
        return ""
    if start == end:
        return chess.square_name(start)
    for turn in (board.turn, not board.turn):
        b = board.copy(stack=False)
        if b.turn != turn:
            b.turn = turn
            b.ep_square = None
        move = _find_move(b, start, end)
        if move is None:
            continue
        san = b.san(move)
        return re.sub(r"=[QRBN][+#]?$", "", san) if move.promotion else san
    return f"{chess.square_name(start)} to {chess.square_name(end)}"


ARROW_WORDS = {"played": "your move", "best": "better", "threat": "threat"}  # before the move, by kind


def arrow_phrase(board: Optional[chess.Board], arrow: Arrow, words: Optional[dict[str, str]] = None) -> str:
    """One arrow in words, without its colour: "your move d5", "better Nf3", "threat Qxf7", "Nc3". ``words``
    replaces the words before the move for some kinds ({"line": "best for you so far"})."""
    kind = arrow.kind if arrow.kind in ARROW_KINDS else "neutral"
    move = arrow_move_text(board, arrow) if board is not None else f"{arrow.start} to {arrow.end}"
    if not move:
        return ""
    prefix = {**ARROW_WORDS, **(words or {})}.get(kind, "")
    return f"{prefix} {move}" if prefix else move


def arrows_text(board: Optional[chess.Board], arrows: Iterable[Arrow], words: Optional[dict[str, str]] = None) -> str:
    """"Your move d5 (red), better Nf3 (green)": the arrows in words, in the report's colour code."""
    parts = []
    for a in arrows or ():
        if not isinstance(a, Arrow):
            continue
        phrase = arrow_phrase(board, a, words)
        if phrase:
            parts.append(f"{phrase} ({ARROW_COLOUR_WORDS.get(a.kind, 'grey')})")
    text = ", ".join(parts)
    return text[:1].upper() + text[1:]


MARK_WORDS = {"focus": "look at", "target": "target", "attacker": "attacker", "weak": "weak square", "check": "check"}


def describe_board(
    board: chess.Board,
    *,
    orientation: str = "white",
    arrows: Sequence[Arrow] = (),
    marks: Sequence[Mark] = (),
    label: str = "",
    you: Optional[str] = None,
) -> str:
    """The position in words for screen readers: whose move, what the arrows and marked squares are."""
    side = "White" if board.turn == chess.WHITE else "Black"
    sentences = []
    if label.strip():
        sentences.append(label.strip().rstrip("."))
    mover = f"{side} to move"
    you = (you or orientation or "").lower()
    if you in ("white", "black"):
        mover += " (you)" if (you == "white") == (board.turn == chess.WHITE) else " (your opponent)"
    if board.is_checkmate():
        mover = f"{side} is checkmated"
    elif board.is_check():
        mover += ", in check"
    sentences.append(mover)
    described = arrows_text(board, arrows)
    if described:
        sentences.append(described)
    marked = [f"{MARK_WORDS.get(m.kind, m.kind)} {m.square}" for m in marks or () if m.kind in MARK_KINDS and m.kind != "check"]
    if marked:
        sentences.append("Marked: " + ", ".join(marked))
    if orientation == "black":
        sentences.append("Board shown from Black's side")
    return ". ".join(sentences) + "."


# --------------------------------------------------------------------------- lines of play and position keys
def epd(fen: Any) -> str:
    """The first four FEN fields (placement, side to move, castling, en passant): how positions are matched."""
    return " ".join(str(fen or "").split()[:4])


def side_to_move(fen: Any) -> str:
    """"white" / "black" for a FEN ("white" when unreadable)."""
    parts = str(fen or "").split()
    return "black" if len(parts) > 1 and parts[1] == "b" else "white"


def numbered_moves(fen: Any, moves_uci: Sequence[str] = (), moves_san: Sequence[str] = (), limit: int = 16) -> str:
    """A line in numbered SAN from ``fen``: "5...e5 6.Ndb5 a6 7.Nd6+". Replays ``moves_uci`` (or ``moves_san``)
    and stops at the first move that is not legal; without a readable FEN, the SAN moves as given."""
    board = parse_board(fen)
    moves = list(moves_uci or ()) or list(moves_san or ())
    if board is None:
        return " ".join(str(m) for m in list(moves_san or ())[:limit])
    out: list[str] = []
    for i, text in enumerate(moves[:limit]):
        move = None
        for parse in (board.parse_uci, board.parse_san):
            try:
                move = parse(str(text).strip())
                break
            except ValueError:
                continue
        if move is None or move not in board.legal_moves:
            break
        san = board.san(move)
        n = board.fullmove_number
        if board.turn == chess.WHITE:
            out.append(f"{n}.{san}")
        else:
            out.append(f"{n}...{san}" if i == 0 else san)
        board.push(move)
    return " ".join(out)


def compact_labels(labels: Sequence[str]) -> str:
    """Frame labels ("6.Ndb5", "6...a6", "7.Nd6+") as one line: "6.Ndb5 a6 7.Nd6+"."""
    out: list[str] = []
    prev_white = ""
    for label in labels:
        text = str(label or "").strip()
        if not text:
            continue
        m = re.match(r"^(\d+)\.\.\.(.+)$", text)
        if m and prev_white == m.group(1):
            text = m.group(2)
        w = re.match(r"^(\d+)\.(?!\.)", text)
        prev_white = w.group(1) if w else ""
        out.append(text)
    return " ".join(out)


def fen_from_analysis_url(url: Any) -> str:
    """The FEN in a Lichess analysis-board URL ("https://lichess.org/analysis/<fen with _>"), else ""."""
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return ""
    host = (parts.hostname or "").lower()
    if host not in ("lichess.org", "www.lichess.org"):
        return ""
    m = re.match(r"^/analysis/(?:standard/)?(.+)$", parts.path)
    if not m:
        return ""
    fen = unquote(m.group(1)).replace("_", " ").strip()
    return fen if parse_board(fen) is not None else ""
