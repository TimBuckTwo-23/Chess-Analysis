"""The fact check for everything the LLM coach writes (C4). Owner: llm.

An LLM text reaches the report only if every checkable statement in it is in the coaching packet:

* **Moves.** Every move the text names ("Nf3", "5...e5", "5... e5", "O-O", "exd5", "e8=Q+") is replayed with
  python-chess from the position's FEN along the lines the packet gives for it (the best line, the refutation of
  your move, cloud-eval lines, opening-database moves, the tablebase move). A move that is not one of those
  moves, or is written with the wrong move number, rejects the text. Moves written one after another ("6.Ndb5 a6
  7.Nd6+ Bxd6") must follow each other in one of those lines, and a bare square right after a White move is read
  as Black's reply. Anywhere else a bare square ("d6") is a square; a pawn move is recognised as a move when it
  has a move number, a capture, a promotion or a check sign. Computer notation ("e6e5") is refused.
* **Material.** "wins your queen", "you lose a knight": some line of the position must capture a piece of that
  kind (one of yours, when the text says "your").
* **Numbers.** Every number must be one the packet holds (within 0.1, or the rounding of the digits written):
  percentages are compared with shares (31% = 0.31) and win-% points, evaluations in pawns. A number written with
  a sign ("+1.5", "−16") must have that sign in the packet, unless its sentence names White or Black. Move numbers,
  squares and small counting words ("two", "twice") are not numbers here; "move 7" must be a move of the lines.
* **Claims.** An insight id in ``claim_ids`` must be one of the report's claims; wording that calls something your
  strength, weakness or habit must be backed by a claim of that kind on the same subject.
* **Formats.** A sentence may name the format of its game (bullet, blitz, rapid) and those of the claims it relies
  on; another format only next to your own number for it (your rating or your games in that format).
* **Style.** At most three sentences for a position, pawns rather than centipawns, plain text without markup or
  internal ids.

A rejected explanation keeps its template text; a weekly-plan entry that names no study-plan item is dropped.
* **Direction.** "you stand 1.5 pawns better" must point the way the engine's evaluation does (from your side).

What this cannot check: evaluations in words without a number ("you were winning"), whose move a piece move is
when it has no number ("White plays Bxd6"), and chess ideas said in words only ("the knight beats the bishop").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any, Iterable, Optional, Sequence

import chess

from ..models import PlanEntry
from .packet import claim_index

TOLERANCE = 0.1
MAX_SENTENCES = 3
CONSTANTS = (0.0, 100.0)  # "per 100 games", "0"
FIGURINES = str.maketrans(dict(zip("♔♕♖♗♘♚♛♜♝♞", "KQRBNKQRBN")))  # figurine notation -> letters

# A move, optionally numbered ("5.", "5...", "5. ...", "5…") or marked as Black's ("...e5"). Group ``san`` is the
# move itself.
MOVE_RE = re.compile(
    r"(?<![\w.\-])"
    r"(?:(?P<num>\d{1,3})\s*(?P<dots>\.\s*\.\.\.|\.{3}|…|\.)\s*|(?P<black>\.{3}|…)\s?)?"
    r"(?P<san>O-O-O|O-O|0-0-0|0-0|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]"
    r"|[a-h]x[a-h][1-8](?:=?[QRBN])?|[a-h][1-8](?:=?[QRBN])?)"
    r"(?P<suffix>[+#]?)(?:[!?]{1,2})?"
    r"(?![\w\-])"
)
UCI_RE = re.compile(r"(?<![\w\-/=])[a-h][1-8][a-h][1-8][qrbn]?(?![\w\-])")  # "e6e5": computer notation
ID_RE = re.compile(r"\b[a-z][a-z_]*\.(?:strength|weakness|observation)\.[a-z0-9][a-z0-9_.\-]*[a-z0-9]")
URL_RE = re.compile(r"https?://[^\s<>()\"']+")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER_RE = re.compile(
    r"(?<![\w.,])(?P<sign>[+\-−±])?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?(?![\d])"
    r"(?P<pct>\s?%|\s?per\s?cent\b|\s?percent\b)?"
)
MOVE_REF_RE = re.compile(r"\bmoves?\s+(?:number\s+|no\.\s*)?$", re.IGNORECASE)  # "on move 7", before the number
NUMBER_WORDS = {
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}  # "one", "two", "three", "once", "twice", "both" ... are small counting words and not checked
NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
CENTIPAWN_RE = re.compile(r"\bcenti-?pawns?\b|\bcp\b|\d\s*cp\b", re.IGNORECASE)
MARKUP_RE = re.compile(r"</?[a-zA-Z][^>]*>|\*\*|__|^#+\s", re.MULTILINE)
RANK_RE = re.compile(r"\b[1-8](?:st|nd|rd|th)[\s-]+ranks?\b|\branks?\s+[1-8]\b", re.IGNORECASE)  # geometry, not facts
FEN_RE = re.compile(r"^\s*[1-8pnbrqkPNBRQK]+(?:/[1-8pnbrqkPNBRQK]+){7}\s+[wb]\b")
# An evaluation given for a side ("+1.5 for White", "White is 1.5 pawns up") may carry the other sign than the
# packet's (which are from your side); "as Black" only says which side you play.
SIDE_RE = re.compile(
    r"\bfor\s+(?:white|black)\b|\b(?:white|black)(?:'s)?\s+(?:is|stands|would|will|leads|keeps|gets|has|had|was|"
    r"advantage|edge|lead)\b",
    re.IGNORECASE,
)
# "you stand 1.5 pawns worse", "you are a pawn up": an evaluation in words, whose direction the lines must share
DIRECTION_RE = re.compile(
    r"(?<![\w.])(?P<num>\d+(?:\.\d+)?)\s+pawns?(?:'s?)?\s+(?:of\s+\w+\s+)?(?P<dir>better|worse|up|down|ahead|behind)\b",
    re.IGNORECASE,
)
YOU_RE = re.compile(r"\byou\b", re.IGNORECASE)  # a sentence about you ("you stand", "leaves you")

# "wins your queen", "you lose a knight", "takes the dark-squared bishop": a capture the lines must contain.
CAPTURE_RE = re.compile(
    r"\b(?:wins?|winning|won|takes?|taking|took|captures?|capturing|captured|loses?|losing|lost|drops?|dropping|"
    r"dropped|hangs?|hanging|hung|grabs?|grabbing|grabbed|nets?|netting|netted|costs?\s+you|picks?\s+(?:up|off)|"
    r"picked\s+(?:up|off)|snaps?\s+(?:up|off)|snapped\s+(?:up|off))\s+"
    r"(?P<det>(?:(?:your|my|the|a|an|his|her|their|its|that|this|one|another|extra|whole|free|loose|undefended|"
    r"pinned|trapped|stranded|poor|last|opponent's|[a-z]+-squared)\s+){0,3})"
    r"(?P<piece>queen|rook|bishop|knight|pawn)s?\b(?!\s+pair)",
    re.IGNORECASE,
)
PIECE_TYPES = {"queen": chess.QUEEN, "rook": chess.ROOK, "bishop": chess.BISHOP, "knight": chess.KNIGHT,
               "pawn": chess.PAWN}

# Wording that says something about you as a player: needs a claim of that kind behind it. Chess uses
# "weakness" for squares too ("a weakness on d6", "your d6 weakness"), so only personal phrasings count and
# BOARD_WORDS inside the phrase mark it as being about the position.
WEAKNESS_RE = re.compile(
    r"\byour\s+(?:\w+\s+){0,2}?(?:weakness(?:es)?|weak\s+(?:spot|point|side|area)s?|problems?|flaws?|habits?)\b"
    r"|\bweakness(?:es)?\s+(?:of\s+yours|in\s+your\s+(?:game|play|chess))\b"
    r"|\bweak\s+(?:spot|point|side|area)s?\s+(?:of|in)\s+your\s+(?:game|play|chess)\b"
    r"|\b(?:a|this|that)\s+habit\s+of\s+(?:yours|playing)\b|\byou\s+have\s+(?:a|this|that)\s+habit\b"
    r"|\byou(?:'re|\s+are)\s+(?:\w+\s+)?(?:weak|bad|poor|worse)\s+(?:at|in|with)\b"
    r"|\byou\s+(?:\w+\s+)?struggle\b|\bcost(?:s|ing)?\s+you\s+(?:points|games|rating)\b",
    re.IGNORECASE,
)
STRENGTH_RE = re.compile(
    r"\byour\s+(?:\w+\s+){0,2}?(?:strengths?|strong\s+(?:suit|point|side)s?|assets?)\b"
    r"|\bstrengths?\s+(?:of\s+yours|in\s+your\s+(?:game|play|chess))\b"
    r"|\byou(?:'re|\s+are)\s+(?:\w+\s+)?(?:strong|good|great|excellent|better)\s+(?:at|in|with)\b"
    r"|\byou\s+(?:\w+\s+)?excel\b",
    re.IGNORECASE,
)
BOARD_WORDS = re.compile(
    r"\b[a-h][1-8]\b|\b[a-h]-(?:pawn|file)\b|pawn|square|king|queen|rook|bishop|knight|piece|rank|file|diagonal|"
    r"structure|colou?r|back-rank|castl",
    re.IGNORECASE,
)
_STOP = frozenset(
    """about above after again against also always because been before being below better between both could does
    doing down during each even every from further have having here into just like made make more most much must
    never only other ours over same should some such than that their them then there these they this those through
    under until very were what when where which while will with would your yours you're really often usually
    strength strengths weakness weaknesses weak strong problem problems struggle excel asset assets costing cost
    costs points point games game play chess player move moves position positions rating ratings good great poor
    worse played plays playing well times time habit habits""".split()
)

# "bullet" and "blitz" name a format wherever they appear; "rapid" and "daily" only where they read as one ("in
# rapid", "your rapid games", "blitz and rapid", "Rapid is ..."), not in "your rapid development", "in rapid
# succession" or "practise daily".
_FORMAT_WORD = r"(?:bullet|blitz|rapid|daily)"
FORMAT_RE = re.compile(
    r"\b(bullet|blitz)\b"
    r"|\b(?:in|at|and|or|vs|versus)\s+(rapid|daily)\b(?![\s-]+(?:succession|development|fire|attack|advances?|"
    r"growth|expansion|order|tempo|mobili[sz]ation|progress|pace|decline|response|reaction|practice|training)\b)"
    r"|\b(rapid|daily)(?=\s+(?:games?|chess|time\s+controls?|ratings?|format|pool|results?|view|only|is|was|suits)\b"
    rf"|\s*(?:,|and|or|vs\.?|versus)\s+{_FORMAT_WORD}\b)"
    r"|(?<=,\s)(rapid|daily)\b",
    re.IGNORECASE,
)


@dataclass
class Check:
    """The verifier's verdict on one text: ``ok`` when ``problems`` is empty."""

    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


# --------------------------------------------------------------------------- moves
@dataclass(frozen=True)
class NamedMove:
    raw: str
    number: Optional[int]
    white: Optional[bool]  # None when neither the text nor the move before it says whose move it is
    san: str  # as written, suffixes dropped
    start: int
    end: int


@dataclass(frozen=True)
class _Ply:
    number: int
    white: bool
    san: str  # normalised
    move: chess.Move
    board: chess.Board  # the position before the move


def normalize_san(san: str) -> str:
    """SAN without check / mate / annotation signs, "=" in promotions, and zeros in castling."""
    s = str(san or "").translate(FIGURINES).strip()
    s = s.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    s = re.sub(r"[+#!?]+$", "", s)
    return s.replace("=", "")


def _tokens(text: str) -> list[tuple[NamedMove, bool]]:
    """Every MOVE_RE match, and whether it reads as a move on its own (a bare square does not)."""
    out = []
    for m in MOVE_RE.finditer(str(text or "").translate(FIGURINES)):
        san, num, black = m.group("san"), m.group("num"), m.group("black")
        is_move = bool(num or black) or san[0] in "KQRBNO0" or "x" in san or bool(m.group("suffix")) or len(san) > 2
        white = False if black else None if not num else m.group("dots").strip() == "."
        mv = NamedMove(m.group(0), int(num) if num else None, white, normalize_san(san), m.start(), m.end())
        out.append((mv, is_move))
    return out


def named_moves(text: str) -> list[NamedMove]:
    """The moves a text names. A bare square ("d6", "e5") without a move number is a square, not a move."""
    return [mv for mv, is_move in _tokens(text) if is_move]


def _next_ply(mv: NamedMove) -> tuple[Optional[int], Optional[bool]]:
    """(number, White's move?) of the move after ``mv`` in a line."""
    if mv.white is None:
        return None, None
    if mv.white:
        return mv.number, False
    return (mv.number + 1 if mv.number is not None else None), True


def move_runs(text: str) -> list[list[NamedMove]]:
    """The moves a text names, grouped into runs written one after another ("6.Ndb5 a6 7.Nd6+ Bxd6").

    Inside a run an unnumbered move takes its number from the move before it, and a bare square right after a
    White move is Black's reply ("6.Ndb5 a6"); anywhere else a bare square stays a square.
    """
    text = str(text or "").translate(FIGURINES)
    runs: list[list[NamedMove]] = []
    run: list[NamedMove] = []
    for mv, is_move in _tokens(text):
        if run and text[run[-1].end : mv.start].strip():  # words in between: a new run
            runs.append(run)
            run = []
        if not is_move and not (run and run[-1].white is True):
            if run:
                runs.append(run)
                run = []
            continue
        if run and mv.number is None:
            number, white = _next_ply(run[-1])
            if mv.white is None or mv.white == white:
                mv = replace(mv, number=number, white=white)
        run.append(mv)
    if run:
        runs.append(run)
    return runs


def _board(fen: str) -> Optional[chess.Board]:
    try:
        return chess.Board(fen)
    except ValueError:
        try:
            return chess.Board(fen, chess960=True)
        except ValueError:
            return None


def _parse(board: chess.Board, move: str) -> Optional[chess.Move]:
    text = str(move or "").strip().replace("…", "...")
    try:
        m = chess.Move.from_uci(text)
        if m in board.legal_moves:
            return m
    except ValueError:
        pass
    try:
        return board.parse_san(normalize_san(text.split(".")[-1].strip()) or "?")
    except ValueError:
        return None


def replay(fen: str, moves: Sequence[str]) -> tuple[list[_Ply], list[chess.Board]]:
    """The plies of a line (numbered SAN or UCI) from ``fen`` and the board before each; stops at an illegal move."""
    board = _board(fen)
    plies: list[_Ply] = []
    boards: list[chess.Board] = []
    if board is None:
        return plies, boards
    for move in moves:
        m = _parse(board, move)
        if m is None:
            break
        before = board.copy(stack=False)
        boards.append(before)
        plies.append(_Ply(board.fullmove_number, board.turn == chess.WHITE, normalize_san(board.san(m)), m, before))
        board.push(m)
    boards.append(board)
    return plies, boards


def position_lines(position: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Every line the packet gives for a position, as (start FEN, moves): the moves an explanation may name."""
    fen = str(position.get("fen") or "")
    lines: list[tuple[str, list[str]]] = []

    def add(line: Any) -> None:
        if isinstance(line, dict) and line.get("moves"):
            lines.append((str(line.get("fen") or fen), [str(m) for m in line["moves"]]))

    add(position.get("best_line"))
    add(position.get("refutation"))
    for key in ("played", "best"):
        if position.get(key):
            lines.append((fen, [str(position[key])]))
    opening = position.get("opening") or {}
    for line in opening.get("cloud_lines") or []:
        add(line)
    for stat in list(opening.get("masters") or []) + list(opening.get("peers") or []):
        if isinstance(stat, dict) and stat.get("move"):
            lines.append((fen, [str(stat["move"])]))
    tb = position.get("tablebase") or {}
    if isinstance(tb, dict) and tb.get("best"):
        lines.append((fen, [str(tb["best"])]))
    return lines


def _replayed(positions: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], list[_Ply], list[chess.Board]]]:
    """(position, plies, boards) for every line of every position."""
    out = []
    for position in positions:
        for fen, moves in position_lines(position):
            plies, boards = replay(fen, moves)
            out.append((position, plies, boards))
    return out


def _same_move(mv: NamedMove, ply: _Ply) -> bool:
    """The text's move is the line's move: the same SAN, or SAN that python-chess reads as it ("Nd4b5", "e8Q")."""
    if mv.san == ply.san:
        return True
    try:
        return ply.board.parse_san(mv.san) == ply.move
    except ValueError:
        return False


def _fits(mv: NamedMove, ply: _Ply) -> bool:
    if mv.number is not None and mv.number != ply.number:
        return False
    if mv.white is not None and mv.white != ply.white:
        return False
    return _same_move(mv, ply)


def _run_in_line(run: Sequence[NamedMove], plies: Sequence[_Ply]) -> bool:
    """The moves of ``run`` are consecutive plies of the line."""
    return any(all(_fits(mv, plies[j + k]) for k, mv in enumerate(run)) for j in range(len(plies) - len(run) + 1))


def check_moves(text: str, positions: Iterable[dict[str, Any]]) -> list[str]:
    """Problems with the moves ``text`` names, against the lines of ``positions`` (empty list = all fine)."""
    text = str(text or "")
    lines = _replayed(positions)
    all_plies = [p for _, plies, _ in lines for p in plies]
    boards = [b for _, _, bs in lines for b in bs]
    problems = []
    for run in move_runs(text):
        if any(_run_in_line(run, plies) for _, plies, _ in lines):
            continue
        alone = [mv for mv in run if not any(_fits(mv, p) for p in all_plies)]
        for mv in alone:
            if any(_parse(b, mv.san) is not None for b in boards) or any(_same_move(mv, p) for p in all_plies):
                problems.append(f"names {mv.raw.strip()}, which is not in the position's lines")
            else:
                problems.append(f"names {mv.raw.strip()}, which is not a legal move in the position's lines")
        if not alone:
            written = " ".join(mv.raw.strip() for mv in run)
            problems.append(f"names {written}, but these moves do not follow each other in the position's lines")
    for m in UCI_RE.finditer(_mask(text, [(u.start(), u.end()) for u in URL_RE.finditer(text)])):
        problems.append(f"writes {m.group(0)} in computer notation, not in standard notation")
    return problems


def move_numbers(positions: Iterable[dict[str, Any]]) -> set[int]:
    """The move numbers of every ply in the positions' lines ("on move 7" must be one of them)."""
    return {p.number for _, plies, _ in _replayed(positions) for p in plies}


# --------------------------------------------------------------------------- material
def _captures(positions: Iterable[dict[str, Any]]) -> set[tuple[int, str]]:
    """(piece type, "you" | "opponent") of every piece captured along the positions' lines."""
    out: set[tuple[int, str]] = set()
    for position, plies, _ in _replayed(positions):
        you = chess.BLACK if str(position.get("you_play") or "") == "black" else chess.WHITE
        for p in plies:
            if not p.board.is_capture(p.move):
                continue
            if p.board.is_en_passant(p.move):
                piece_type, color = chess.PAWN, not p.board.turn
            else:
                piece = p.board.piece_at(p.move.to_square)
                if piece is None or piece.color == p.board.turn:  # Chess960 castling "takes" its own rook
                    continue
                piece_type, color = piece.piece_type, piece.color
            out.add((piece_type, "you" if color == you else "opponent"))
    return out


def check_material(text: str, positions: Iterable[dict[str, Any]]) -> list[str]:
    """"wins your queen", "you lose a knight": a piece the text says is won or lost must be captured in a line."""
    positions = list(positions)
    captured: Optional[set[tuple[int, str]]] = None
    problems = []
    for m in CAPTURE_RE.finditer(str(text or "").replace("’", "'")):
        if captured is None:
            captured = _captures(positions)
        det = set(m.group("det").lower().split())
        whose = "you" if det & {"your", "my"} else "opponent" if det & {"his", "her", "their", "opponent's"} else None
        piece_type = PIECE_TYPES[m.group("piece").lower()]
        if not any(pt == piece_type and whose in (None, side) for pt, side in captured):
            problems.append(f'says "{m.group(0).strip()}", but no line of the position captures such a piece')
    return problems


# --------------------------------------------------------------------------- numbers
@dataclass(frozen=True)
class NamedNumber:
    raw: str
    value: float  # the magnitude as written
    decimals: int
    percent: bool
    sign: int = 0  # +1 / -1 when written with a sign ("+1.5", "−16"), 0 otherwise
    move_ref: bool = False  # "move 7": a move number, checked against the lines


def _mask(text: str, spans: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


def _strip_non_numbers(text: str) -> str:
    """``text`` with moves, URLs, dates and insight ids blanked out, so their digits don't count as numbers."""
    text = str(text or "").translate(FIGURINES)
    spans = [(m.start(), m.end()) for m in URL_RE.finditer(text)]
    spans += [(m.start(), m.end()) for m in DATE_RE.finditer(text)]
    spans += [(m.start(), m.end()) for m in ID_RE.finditer(text)]
    spans += [(m.start(), m.end()) for m in RANK_RE.finditer(text)]
    spans += [(m.start(), m.end()) for m in UCI_RE.finditer(text)]
    text = _mask(text, spans)
    return _mask(text, [(m.start, m.end) for m in named_moves(text)])


def named_numbers(text: str) -> list[NamedNumber]:
    """The numbers a text states (digits and the larger number words), move numbers and squares excluded."""
    clean = _strip_non_numbers(text)
    out = []
    for m in NUMBER_RE.finditer(clean):
        whole = m.group("int").replace(",", "")
        dec = m.group("dec") or ""
        value = float(f"{whole}.{dec}" if dec else whole)
        sign = {"+": 1, "-": -1, "−": -1}.get(m.group("sign") or "", 0)
        move_ref = not dec and bool(MOVE_REF_RE.search(clean[max(0, m.start() - 16) : m.start()]))
        out.append(NamedNumber(m.group(0).strip(), value, len(dec), bool(m.group("pct")), sign, move_ref))
    for m in NUMBER_WORD_RE.finditer(clean):
        out.append(NamedNumber(m.group(0), float(NUMBER_WORDS[m.group(1).lower()]), 0, False))
    return out


def packet_numbers(obj: Any) -> list[float]:
    """Every number in a packet part (values, and numbers written inside its strings), with its sign.

    A number written in a string without a sign counts as positive. Shares also count as percentages (0.31 and
    31), and percentages as shares.
    """
    found: set[float] = set()

    def walk(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            found.add(float(x))
        elif isinstance(x, str):
            if FEN_RE.match(x):  # a FEN's digits are empty squares and counters, not facts
                return
            found.update(float(k) for k in re.findall(r"mateIn(\d+)", x))
            for n in named_numbers(x):
                value = -n.value if n.sign < 0 else n.value
                found.add(value / 100.0 if n.percent else value)
                found.add(value)
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(obj)
    found.update(CONSTANTS)
    return sorted(found | {v * 100.0 for v in found if abs(v) <= 1.0})


def _matches(n: NamedNumber, allowed: Sequence[float], either_sign: bool = False) -> bool:
    tol = max(TOLERANCE, 0.5 * 10 ** -n.decimals)
    candidates = [(n.value, tol)]
    if n.percent:
        candidates.append((n.value / 100.0, tol / 100.0))
    if n.sign == 0 or either_sign:  # a size: "1.5 pawns down", "31%", or "+1.5 for White"
        return any(abs(v - abs(a)) <= t + 1e-9 for v, t in candidates for a in allowed)
    return any(abs(n.sign * v - a) <= t + 1e-9 for v, t in candidates for a in allowed)


def check_numbers(text: str, allowed: Sequence[float], moves: Iterable[int] = ()) -> list[str]:
    """Numbers ``text`` states that are not in ``allowed``; "move N" may also be a move number of the lines.

    A signed number must match with its sign, unless its sentence names White or Black (the packet's evaluations
    are from your side; "+1.5 for White" may be "−1.5 for you").
    """
    moves = set(moves)
    problems = []
    for sentence in _sentences(str(text or "")):
        either = bool(SIDE_RE.search(sentence))
        for n in named_numbers(sentence):
            if n.move_ref and int(n.value) in moves:
                continue
            if not _matches(n, allowed, either):
                problems.append(f"states {n.raw}, which is not a number in the report")
    return problems


def _evaluations(positions: Iterable[dict[str, Any]]) -> list[float]:
    """The signed evaluations of the positions, from your side: the lines' ends, the concept differences, and the
    pawns your move lost (either way: "1 pawn worse than 5...a6" and "5...a6 is 1 pawn better")."""
    out: list[float] = []
    for p in positions:
        for key in ("best_line", "refutation"):
            v = (p.get(key) or {}).get("eval_pawns")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
        out += [float(c["pawns"]) for c in p.get("concepts") or [] if isinstance(c.get("pawns"), (int, float))]
        lost = p.get("pawns_lost")
        if isinstance(lost, (int, float)) and not isinstance(lost, bool):
            out += [float(lost), -float(lost)]
    return out


def check_direction(text: str, positions: Iterable[dict[str, Any]]) -> list[str]:
    """"you stand 1.5 pawns better": an evaluation said in words must point the way the engine's does (from your
    side, in sentences about you that name neither White nor Black)."""
    evals: Optional[list[float]] = None
    problems = []
    for sentence in _sentences(str(text or "")):
        if SIDE_RE.search(sentence) or not YOU_RE.search(sentence):
            continue
        for m in DIRECTION_RE.finditer(_mask(sentence, [(x.start, x.end) for x in named_moves(sentence)])):
            if evals is None:
                evals = _evaluations(positions)
            value = float(m.group("num"))
            sign = 1 if m.group("dir").lower() in ("better", "up", "ahead") else -1
            decimals = len(m.group("num").partition(".")[2])
            tol = max(TOLERANCE, 0.5 * 10 ** -decimals)
            if not any(abs(sign * value - e) <= tol + 1e-9 for e in evals):
                problems.append(f'says "{m.group(0)}", but the engine\'s evaluations point the other way')
    return problems


def _packet_strings(obj: Any) -> list[str]:
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(obj)
    return out


def _url(url: str) -> str:
    return url.rstrip(".,;:!?)").rstrip("/")


def check_links(text: str, scope: Any) -> list[str]:
    """URLs and ISO dates in ``text`` must appear in the packet (a URL whole, not a part of one)."""
    strings = _packet_strings(scope)
    urls = {_url(m.group(0)) for s in strings for m in URL_RE.finditer(s)}
    problems = []
    for m in URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?)")
        if _url(url) not in urls:
            problems.append(f"links {url}, which is not in the report")
    for m in DATE_RE.finditer(text):
        if not any(m.group(0) in s for s in strings):
            problems.append(f"gives the date {m.group(0)}, which is not in the report")
    return problems


# --------------------------------------------------------------------------- claims
def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-']+", str(text or "").lower()) if len(w) >= 4 and w not in _STOP}


def subject_words(text: str) -> set[str]:
    """What a text is about: its content words and the moves it names ("5...e5" -> "e5")."""
    return content_words(text) | {mv.san.lower() for mv in named_moves(text)}


def claim_words(claim: dict[str, Any]) -> set[str]:
    """What a claim is about: the words of its title and detail, the names in its evidence and the moves in them.

    Neither the formats behind a claim ("bullet" is not what the Sicilian claim is about, unless its title names
    it) nor its category ("openings" is not what one opening's claim is about) count.
    """
    parts = [claim.get("title"), claim.get("detail")]
    parts += [v for v in (claim.get("evidence") or {}).values() if isinstance(v, str) and not FEN_RE.match(v)]
    return subject_words(" ".join(str(p) for p in parts if p))


def _sentences(text: str) -> list[str]:
    masked = _mask(text, [(m.start, m.end) for m in named_moves(text)])
    masked = re.sub(r"\d\.\d", "0x0", masked)
    masked = re.sub(r"\b(e\.g|i\.e|vs|etc|approx)\.", lambda m: m.group(0).replace(".", " "), masked, flags=re.I)
    spans, start = [], 0
    for m in re.finditer(r"[.!?]+(?=\s|$)", masked):
        spans.append(text[start : m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        spans.append(tail)
    return [s for s in spans if re.search(r"\w", s)]


def check_claims(
    text: str, claims: dict[str, dict[str, Any]], cited: Sequence[str] = (), own: Optional[str] = None
) -> list[str]:
    """Insight ids must be claims; personal strength / weakness / habit wording needs a claim of that kind on its
    subject.

    ``cited`` are the claim ids the LLM says the text relies on; ``own`` is the claim the position belongs to.
    """
    text = str(text or "").replace("’", "'")  # you’re -> you're
    problems = []
    for cid in list(cited) + ID_RE.findall(text):
        if cid not in claims:
            problems.append(f"cites {cid}, which is not one of the report's claims")
    backing = [claims[c] for c in dict.fromkeys([*cited, *([own] if own else []), *ID_RE.findall(text)]) if c in claims]
    for sentence in _sentences(text):
        for pattern, kind in ((WEAKNESS_RE, "weakness"), (STRENGTH_RE, "strength")):
            if not any(not BOARD_WORDS.search(m.group(0)) for m in pattern.finditer(sentence)):
                continue
            words = subject_words(pattern.sub(" ", sentence))
            if not any(c.get("kind") == kind and (words & claim_words(c)) for c in backing):
                problems.append(f'calls something your {kind} that is not one of the report\'s claims: "{sentence}"')
    return problems


# --------------------------------------------------------------------------- formats
def named_formats(text: str) -> list[str]:
    """The formats a text names, lower case, in order (repeats kept)."""
    return [next(g for g in m.groups() if g).lower() for m in FORMAT_RE.finditer(str(text or ""))]


def format_numbers(packet: dict[str, Any]) -> dict[str, list[float]]:
    """Your own numbers per format (rating, games, engine-analysed games): a sentence that gives one of them may
    name its format."""
    player = packet.get("player") or {}
    out: dict[str, list[float]] = {}
    for key in ("ratings", "games_per_format", "engine_games_per_format"):
        for tc, n in (player.get(key) or {}).items():
            if isinstance(n, (int, float)) and not isinstance(n, bool):
                out.setdefault(str(tc).lower(), []).append(float(n))
    return out


def claim_formats(claim: dict[str, Any]) -> set[str]:
    """The formats behind a claim: its game counts, its per-format view and the formats its title names."""
    out = {str(tc).lower() for tc in claim.get("formats") or {}}
    if claim.get("view"):
        out.add(str(claim["view"]).lower())
    return out | set(named_formats(str(claim.get("title") or "")))


def check_formats(text: str, allowed: Iterable[str], numbers: Optional[dict[str, list[float]]] = None) -> list[str]:
    """A format a sentence names (bullet, blitz, rapid, daily) must be one the facts behind it are about, or come
    with your own number for that format ("your 949 blitz rating")."""
    ok = {str(tc).lower() for tc in allowed if tc}
    about = ", ".join(sorted(ok)) or "no format"
    problems = []
    for sentence in _sentences(str(text or "")):
        stated = named_numbers(sentence)
        for tc in dict.fromkeys(named_formats(sentence)):
            if tc in ok:
                continue
            own = (numbers or {}).get(tc) or []
            if any(n.sign == 0 and abs(n.value - v) < 0.5 for n in stated for v in own):
                continue
            problems.append(f"says {tc}, but the facts behind it are about {about}")
    return problems


# --------------------------------------------------------------------------- style
def check_style(text: str, max_sentences: Optional[int] = MAX_SENTENCES) -> list[str]:
    problems = []
    if not str(text or "").strip():
        return ["is empty"]
    if CENTIPAWN_RE.search(text):
        problems.append("uses centipawns (the report gives evaluations in pawns)")
    if MARKUP_RE.search(text):
        problems.append("contains markup")
    if ID_RE.search(text):
        problems.append("shows an internal insight id")
    n = len(_sentences(text))
    if max_sentences is not None and n > max_sentences:
        problems.append(f"has {n} sentences (at most {max_sentences})")
    return problems


# --------------------------------------------------------------------------- whole texts
def _scope(packet: dict[str, Any], position: Optional[dict[str, Any]], claims: Iterable[dict[str, Any]]) -> list[Any]:
    parts: list[Any] = [packet.get("player") or {}, list(claims)]
    if position is not None:
        parts.append(position)
    return parts


def verify_explanation(
    text: str, position: dict[str, Any], packet: dict[str, Any], claim_ids: Sequence[str] = ()
) -> Check:
    """Check an LLM explanation of one packet position (moves, material, numbers, claims, formats, style)."""
    claims = claim_index(packet)
    own = position.get("insight_id") if position.get("insight_id") in claims else None
    related = [claims[c] for c in dict.fromkeys([*claim_ids, *([own] if own else [])]) if c in claims]
    scope = _scope(packet, position, related)
    problems = check_style(text)
    problems += check_moves(text, [position])
    problems += check_material(text, [position])
    problems += check_numbers(text, packet_numbers(scope), move_numbers([position]))
    problems += check_direction(text, [position])
    problems += check_links(text, scope)
    problems += check_claims(text, claims, claim_ids, own)
    # the game's format, and the formats of the claims it relies on
    formats = {str(position.get("time_class") or "")} | {tc for c in related for tc in claim_formats(c)}
    problems += check_formats(text, formats - {""}, format_numbers(packet))
    return Check(problems)


def verify_answer(
    text: str,
    packet: dict[str, Any],
    epds: Sequence[str] = (),
    claim_ids: Sequence[str] = (),
    max_sentences: Optional[int] = 8,
    question: str = "",
) -> Check:
    """Check an answer to a question about the whole report (``chess-insights ask``).

    Moves are replayed in the positions the answer names by EPD (every packet position when it names none);
    numbers may come from anywhere in the packet. A format may be named when the question names it, when a cited
    claim or position is about it, or next to your own number for it.
    """
    claims = claim_index(packet)
    named = [p for p in packet.get("positions") or [] if p.get("epd") in set(epds)]
    positions = named if epds else list(packet.get("positions") or [])
    known = {p.get("epd") for p in named}
    problems = [f"refers to a position ({e}) that is not in the report" for e in epds if e not in known]
    problems += check_style(text, max_sentences)
    problems += check_moves(text, positions)
    problems += check_material(text, positions)
    problems += check_numbers(text, packet_numbers(packet), move_numbers(positions))
    problems += check_direction(text, positions)
    problems += check_links(text, packet)
    problems += check_claims(text, claims, claim_ids)
    formats = set(named_formats(question)) | {str(p.get("time_class") or "") for p in named}
    formats |= {tc for c in claim_ids if c in claims for tc in claim_formats(claims[c])}
    problems += check_formats(text, formats - {""}, format_numbers(packet))
    return Check(problems)


def _norm_title(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def verify_plan(
    entries: Sequence[dict[str, Any]], packet: dict[str, Any], practice_minutes: int, today: date
) -> tuple[list[PlanEntry], list[str]]:
    """Weekly-plan entries that name a study-plan item, trimmed to the practice budget; and why others were dropped.

    ``item`` may be the item's title or its packet id ("plan-2"). Recheck dates outside the next 12 weeks become
    four weeks from ``today``. When the minutes add up to more than ``practice_minutes`` a day, every entry is
    scaled down (in steps of 5 minutes, at least 5) and the last ones are dropped while it still does not fit.
    """
    plan = {**{_norm_title(p.get("title")): p for p in packet.get("study_plan") or []},
            **{_norm_title(p.get("id")): p for p in packet.get("study_plan") or []}}
    kept: list[PlanEntry] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        item = plan.get(_norm_title(e.get("item")))
        if item is None:
            dropped.append(f'"{e.get("item")}" is not an item of the study plan')
            continue
        title = str(item.get("title") or "")
        minutes = e.get("minutes_per_week")
        if title in seen or not isinstance(minutes, (int, float)) or isinstance(minutes, bool) or minutes <= 0:
            dropped.append(f'"{title}" is repeated or has no minutes')
            continue
        seen.add(title)
        try:
            recheck = date.fromisoformat(str(e.get("recheck_date") or ""))
        except ValueError:
            recheck = None
        if recheck is None or not today < recheck <= today + timedelta(weeks=12):
            recheck = today + timedelta(weeks=4)
        kept.append(PlanEntry(item=title, minutes_per_week=int(round(minutes)), recheck_date=recheck.isoformat(),
                              note=str(item.get("target") or "")))
    budget = max(0, int(practice_minutes)) * 7
    if kept and budget < 5:  # no practice time at all: nothing fits
        dropped += [f'"{p.item}" does not fit {practice_minutes} minutes a day' for p in kept]
        return [], dropped
    total = sum(p.minutes_per_week for p in kept)
    if kept and total > budget:
        for p in kept:
            p.minutes_per_week = max(5, int(p.minutes_per_week * budget / total) // 5 * 5)
        while sum(p.minutes_per_week for p in kept) > budget and len(kept) > 1:
            dropped.append(f'"{kept[-1].item}" does not fit {practice_minutes} minutes a day')
            kept.pop()
    return kept, dropped
