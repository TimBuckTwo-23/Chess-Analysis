"""The fact check for everything the LLM coach writes (C4). Owner: llm.

An LLM text reaches the report only if every checkable statement in it is in the coaching packet:

* **Moves.** Every move the text names ("Nf3", "5...e5", "5... e5", "O-O", "exd5", "e8=Q+") is replayed with
  python-chess from the position's FEN along the lines the packet gives for it (the best line, the refutation of
  your move, cloud-eval lines, opening-database moves, the tablebase move). A move that is not one of those
  moves, or is written with the wrong move number, rejects the text. A bare square ("d6") is read as a square;
  a pawn move is recognised as a move when it has a move number, a capture, a promotion or a check sign.
* **Numbers.** Every number must be one the packet holds (within 0.1, or the rounding of the digits written):
  percentages are compared with shares (31% = 0.31) and win-% points, evaluations in pawns. Move numbers,
  squares and small counting words ("two", "twice") are not numbers here.
* **Claims.** An insight id in the text or in ``claim_ids`` must be one of the report's claims; wording that calls
  something your strength or weakness must be backed by a claim of that kind on the same subject.
* **Formats.** An explanation may name the format of its game (bullet, blitz, rapid) and those of the claims it
  relies on, no other.
* **Style.** At most three sentences for a position, pawns rather than centipawns, plain text.

A rejected explanation keeps its template text; a weekly-plan entry that names no study-plan item is dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
ID_RE = re.compile(r"\b[a-z][a-z_]*\.(?:strength|weakness|observation)\.[a-z0-9][a-z0-9_.\-]*[a-z0-9]")
URL_RE = re.compile(r"https?://[^\s<>()\"']+")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER_RE = re.compile(
    r"(?<![\w.,])[+\-−±]?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?(?![\d])"
    r"(?P<pct>\s?%|\s?per\s?cent\b|\s?percent\b)?"
)
NUMBER_WORDS = {
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}  # "one", "two", "three", "once", "twice", "both" ... are small counting words and not checked
NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
CENTIPAWN_RE = re.compile(r"\bcenti-?pawns?\b|\bcp\b|\d\s*cp\b", re.IGNORECASE)
MARKUP_RE = re.compile(r"</?[a-zA-Z][^>]*>|\*\*|__|^#+\s", re.MULTILINE)
RANK_RE = re.compile(r"\b[1-8](?:st|nd|rd|th)[\s-]+ranks?\b|\branks?\s+[1-8]\b", re.IGNORECASE)  # geometry, not facts
FEN_RE = re.compile(r"^\s*[1-8pnbrqkPNBRQK]+(?:/[1-8pnbrqkPNBRQK]+){7}\s+[wb]\b")

# Wording that says something about you as a player: needs a claim of that kind behind it. Chess uses
# "weakness" for squares too ("a weakness on d6", "your d6 weakness"), so only personal phrasings count and
# BOARD_WORDS inside the phrase mark it as being about the position.
WEAKNESS_RE = re.compile(
    r"\byour\s+(?:\w+\s+){0,2}?(?:weakness(?:es)?|weak\s+(?:spot|point|side|area)s?|problems?|flaws?)\b"
    r"|\bweakness(?:es)?\s+(?:of\s+yours|in\s+your\s+(?:game|play|chess))\b"
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
    worse played plays playing well times time""".split()
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
    white: Optional[bool]  # None when the text gives no move number
    san: str  # as written, suffixes dropped
    start: int
    end: int


@dataclass(frozen=True)
class _Ply:
    number: int
    white: bool
    san: str  # normalised


def normalize_san(san: str) -> str:
    """SAN without check / mate / annotation signs, "=" in promotions, and zeros in castling."""
    s = str(san or "").translate(FIGURINES).strip()
    s = s.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    s = re.sub(r"[+#!?]+$", "", s)
    return s.replace("=", "")


def named_moves(text: str) -> list[NamedMove]:
    """The moves a text names. A bare square ("d6", "e5") without a move number is a square, not a move."""
    out = []
    for m in MOVE_RE.finditer(str(text or "").translate(FIGURINES)):
        san, num, black = m.group("san"), m.group("num"), m.group("black")
        is_move = bool(num or black) or san[0] in "KQRBNO0" or "x" in san or bool(m.group("suffix")) or len(san) > 2
        if not is_move:
            continue
        white = False if black else None if not num else m.group("dots").strip() == "."
        out.append(NamedMove(m.group(0), int(num) if num else None, white, normalize_san(san), m.start(), m.end()))
    return out


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
        boards.append(board.copy(stack=False))
        plies.append(_Ply(board.fullmove_number, board.turn == chess.WHITE, normalize_san(board.san(m))))
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


def check_moves(text: str, positions: Iterable[dict[str, Any]]) -> list[str]:
    """Problems with the moves ``text`` names, against the lines of ``positions`` (empty list = all fine)."""
    plies: list[_Ply] = []
    boards: list[chess.Board] = []
    for position in positions:
        for fen, moves in position_lines(position):
            p, b = replay(fen, moves)
            plies += p
            boards += b
    problems = []
    for mv in named_moves(text):
        same = [p for p in plies if p.san == mv.san]
        if mv.number is not None:
            same = [p for p in same if p.number == mv.number]
        if mv.white is not None:
            same = [p for p in same if p.white == mv.white]
        if same:
            continue
        legal_somewhere = any(_parse(b, mv.san) is not None for b in boards)
        if legal_somewhere or any(p.san == mv.san for p in plies):
            problems.append(f"names {mv.raw.strip()}, which is not in the position's lines")
        else:
            problems.append(f"names {mv.raw.strip()}, which is not a legal move in the position's lines")
    return problems


# --------------------------------------------------------------------------- numbers
@dataclass(frozen=True)
class NamedNumber:
    raw: str
    value: float
    decimals: int
    percent: bool


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
        out.append(NamedNumber(m.group(0).strip(), value, len(dec), bool(m.group("pct"))))
    for m in NUMBER_WORD_RE.finditer(clean):
        out.append(NamedNumber(m.group(0), float(NUMBER_WORDS[m.group(1).lower()]), 0, False))
    return out


def packet_numbers(obj: Any) -> list[float]:
    """Every number in a packet part (values, and numbers written inside its strings), as magnitudes."""
    found: set[float] = set()

    def walk(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            found.add(abs(float(x)))
        elif isinstance(x, str):
            if FEN_RE.match(x):  # a FEN's digits are empty squares and counters, not facts
                return
            found.update(float(k) for k in re.findall(r"mateIn(\d+)", x))
            for n in named_numbers(x):
                found.add(n.value / 100.0 if n.percent else n.value)
                found.add(n.value)
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(obj)
    found.update(CONSTANTS)
    return sorted(found | {v * 100.0 for v in found if v <= 1.0})


def _matches(n: NamedNumber, allowed: Sequence[float]) -> bool:
    tol = max(TOLERANCE, 0.5 * 10 ** -n.decimals)
    candidates = [(n.value, tol)]
    if n.percent:
        candidates.append((n.value / 100.0, tol / 100.0))
    return any(abs(v - a) <= t + 1e-9 for v, t in candidates for a in allowed)


def check_numbers(text: str, allowed: Sequence[float]) -> list[str]:
    return [
        f"states {n.raw}, which is not a number in the report" for n in named_numbers(text) if not _matches(n, allowed)
    ]


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


def check_links(text: str, scope: Any) -> list[str]:
    """URLs and ISO dates in ``text`` must appear in the packet."""
    strings = _packet_strings(scope)
    problems = []
    for m in URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?)")
        if not any(url in s for s in strings):
            problems.append(f"links {url}, which is not in the report")
    for m in DATE_RE.finditer(text):
        if not any(m.group(0) in s for s in strings):
            problems.append(f"gives the date {m.group(0)}, which is not in the report")
    return problems


# --------------------------------------------------------------------------- claims
def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-']+", str(text or "").lower()) if len(w) >= 4 and w not in _STOP}


def claim_words(claim: dict[str, Any]) -> set[str]:
    """What a claim is about, in words: its title, detail, category and the names in its evidence.

    The formats behind a claim are not its subject ("bullet" is not what the Sicilian claim is about), unless its
    title names them.
    """
    parts = [claim.get("title"), claim.get("detail"), claim.get("category")]
    parts += [v for v in (claim.get("evidence") or {}).values() if isinstance(v, str)]
    return content_words(" ".join(str(p) for p in parts if p))


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
    """Insight ids must be claims; personal strength / weakness wording needs a claim of that kind on its subject.

    ``cited`` are the claim ids the LLM says the text relies on; ``own`` is the claim the position belongs to.
    """
    text = str(text or "").replace("\u2019", "'")  # you’re -> you're
    problems = []
    for cid in list(cited) + ID_RE.findall(text):
        if cid not in claims:
            problems.append(f"cites {cid}, which is not one of the report's claims")
    backing = [claims[c] for c in dict.fromkeys([*cited, *([own] if own else []), *ID_RE.findall(text)]) if c in claims]
    for sentence in _sentences(text):
        for pattern, kind in ((WEAKNESS_RE, "weakness"), (STRENGTH_RE, "strength")):
            if not any(not BOARD_WORDS.search(m.group(0)) for m in pattern.finditer(sentence)):
                continue
            words = content_words(pattern.sub(" ", sentence))
            if not any(c.get("kind") == kind and (words & claim_words(c)) for c in backing):
                problems.append(f'calls something your {kind} that is not one of the report\'s claims: "{sentence}"')
    return problems


# --------------------------------------------------------------------------- formats
FORMAT_RE = re.compile(r"\b(bullet|blitz|rapid|daily)\b", re.IGNORECASE)


def check_formats(text: str, allowed: Iterable[str]) -> list[str]:
    """A format the text names (bullet, blitz, rapid, daily) must be one the facts behind it are about."""
    ok = {str(tc).lower() for tc in allowed}
    named = dict.fromkeys(m.group(1).lower() for m in FORMAT_RE.finditer(str(text or "")))
    about = ", ".join(sorted(ok)) or "no format"
    return [f"says {tc}, but the facts behind it are about {about}" for tc in named if tc not in ok]


# --------------------------------------------------------------------------- style
def check_style(text: str, max_sentences: Optional[int] = MAX_SENTENCES) -> list[str]:
    problems = []
    if not str(text or "").strip():
        return ["is empty"]
    if CENTIPAWN_RE.search(text):
        problems.append("uses centipawns (the report gives evaluations in pawns)")
    if MARKUP_RE.search(text):
        problems.append("contains markup")
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
    """Check an LLM explanation of one packet position (moves, numbers, claims, three sentences at most)."""
    claims = claim_index(packet)
    own = position.get("insight_id") if position.get("insight_id") in claims else None
    related = [claims[c] for c in dict.fromkeys([*claim_ids, *([own] if own else [])]) if c in claims]
    scope = _scope(packet, position, related)
    problems = check_style(text)
    problems += check_moves(text, [position])
    problems += check_numbers(text, packet_numbers(scope))
    problems += check_links(text, scope)
    problems += check_claims(text, claims, claim_ids, own)
    # the game's format, and the formats (and title words) of the claims it relies on
    formats = {str(position.get("time_class") or "")} | {tc for c in related for tc in (c.get("formats") or {})}
    formats |= {m.group(1).lower() for c in related for m in FORMAT_RE.finditer(str(c.get("title") or ""))}
    problems += check_formats(text, formats - {""})
    return Check(problems)


def verify_answer(
    text: str,
    packet: dict[str, Any],
    epds: Sequence[str] = (),
    claim_ids: Sequence[str] = (),
    max_sentences: Optional[int] = 8,
) -> Check:
    """Check an answer to a question about the whole report (``chess-insights ask``).

    Moves are replayed in the positions the answer names by EPD (every packet position when it names none);
    numbers may come from anywhere in the packet.
    """
    claims = claim_index(packet)
    positions = [p for p in packet.get("positions") or [] if not epds or p.get("epd") in set(epds)]
    known = {p.get("epd") for p in positions}
    problems = [f"refers to a position ({e}) that is not in the report" for e in epds if e not in known]
    problems += check_style(text, max_sentences)
    problems += check_moves(text, positions)
    problems += check_numbers(text, packet_numbers(packet))
    problems += check_links(text, packet)
    problems += check_claims(text, claims, claim_ids)
    return Check(problems)


def _norm_title(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def verify_plan(
    entries: Sequence[dict[str, Any]], packet: dict[str, Any], practice_minutes: int, today: date
) -> tuple[list[PlanEntry], list[str]]:
    """Weekly-plan entries that name a study-plan item, trimmed to the practice budget; and why others were dropped.

    ``item`` may be the item's title or its packet id ("plan-2"). Recheck dates outside the next 12 weeks become
    four weeks from ``today``. When the minutes add up to more than ``practice_minutes`` a day, every entry is
    scaled down (in steps of 5 minutes, at least 5).
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
    total = sum(p.minutes_per_week for p in kept)
    if kept and budget and total > budget:
        for p in kept:
            p.minutes_per_week = max(5, int(p.minutes_per_week * budget / total) // 5 * 5)
        while sum(p.minutes_per_week for p in kept) > budget and len(kept) > 1:
            dropped.append(f'"{kept[-1].item}" does not fit {practice_minutes} minutes a day')
            kept.pop()
    return kept, dropped
