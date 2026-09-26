"""Turn lines, motifs and concepts into a short explanation and a board (C1). Owner: coach-core.

For each critical position with engine lines (``deep.analyse_positions``):

* motifs along both lines (``motifs.detect_line``): the refutation's patterns carried out by your opponent (what
  your move allowed) and the best line's patterns carried out by you (what you missed). A motif is named only when
  its detector passed the precision gate (``motifs.GATED_THEMES``, read at call time); otherwise the text says
  "a tactic" and the board marks nothing;
* what is new: a pattern your opponent has along the better line too was not allowed by your move, and one you
  carry out after your move anyway was not missed; a pin that stood before your move is left out by the detector;
* concept deltas and board facts where the two lines have settled (``concepts.comparison_ply``: no capture left
  that wins material), and the material that changes hands up to there, counted from before your opponent's last
  capture (so taking back a piece is a recapture, not a win);
* at most three plain sentences: what the refutation does, what it wins (what changes hands, the point of the line
  or a concept), and what to check next time; evaluations in pawns. A forced mate for you leads ("You had mate in
  3"); a line's point is named when the board shows it: a capture of a pawn or piece that more enemy pieces attack
  than yours defend, a piece of yours chased by a pawn so that it has to move again, an enemy piece landing on a
  square no pawn of yours can cover;
* the board before your move (your move red, the better move green, your side at the bottom) with two strips of
  small boards: the refutation after your move and the better line, each long enough to show the named pattern
  cashing in. A pattern is marked on the main board only when all its pieces already stand there; otherwise on
  the strip frame where it appears;
* a chart of the concept differences, and ``Explanation.verdict``: "fine" (the deeper search's own first choice),
  "close" (not even an inaccuracy behind it, or a choice point under the openings section's bar) or "error".

Everything is deterministic: the same lines give the same words.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import chess

from .. import engine as sf
from .. import visuals
from ..analysis import openings
from ..analysis.mistakes import format_line
from ..models import Chart, ConceptDelta, CriticalPosition, Diagram, Explanation, Line, Mark, Motif, Source, Strip
from . import concepts, motifs
from .critical import engine_mistake
from .config import CoachConfig
from .deep import rebase

if TYPE_CHECKING:
    from ..context import AnalysisContext
    from .deep import DeepResult

log = logging.getLogger(__name__)

STRIP_FRAMES = 4  # small boards per line
# Below this many win-% points the deeper search does not even call your move an inaccuracy (engine.JUDGEMENT_DROPS):
# the text then says your move is close to Stockfish's choice instead of explaining an error.
CLOSE_DROP = min(drop for drop, _ in sf.JUDGEMENT_DROPS)
CLOSE_GAP = 1.0  # ... and less than this many pawns: in a lopsided position -10 against -6 is a few win-% points
# At a choice point your usual move is a mistake only when it lost at least this many win-% points on average in the
# analysed games (the openings section's bar, ``openings.Thresholds.engine_min_drop``; ``explain_all`` reads the run's
# setting).
CHOICE_MIN_DROP = openings.Thresholds.engine_min_drop
TEXT_PLIES = 4  # moves of a line quoted in the text ...
MAX_TEXT_PLIES = 8  # ... or up to this many, to reach the move where the named pattern cashes in
MAX_STRIP_FRAMES = 8  # the strips follow the text up to this many boards
LEAD_MATE_PLIES = 9  # a forced mate for you is quoted to the end when it takes at most this many plies
# Past this many centipawns (10 pawns) the text names the result ("a winning position for you") instead of a number
# such as −34.68: the size of a lopsided evaluation means nothing to a reader.
BIG_EVAL = 1000

# theme -> (the pattern in words, what to check next time when your move allowed it)
MOTIF_WORDS: dict[str, tuple[str, str]] = {
    "fork": ("a fork", "check which two of your pieces one enemy piece could attack at once"),
    "pin": ("a pin", "check which of your pieces stand in line with your king or queen"),
    "skewer": ("a skewer", "check for two of your pieces on one line with the more valuable one in front"),
    "discoveredAttack": ("a discovered attack", "check what an enemy piece hits when the piece in front of it moves"),
    "discoveredCheck": ("a discovered check", "check what an enemy piece hits when the piece in front of it moves"),
    "doubleCheck": ("a double check", "check every line that leads to your king"),
    "hangingPiece": ("an undefended piece to take", "count the attackers and defenders of each piece before you move"),
    "trappedPiece": ("a trapped piece", "check that every piece of yours still has a safe square to go to"),
    "backRankMate": ("a back-rank mate", "make sure your king has an escape square off the back rank"),
    "smotheredMate": ("a smothered mate", "watch knight checks when your own pieces box in your king"),
    "deflection": ("a deflection", "check which of your pieces is doing two jobs at once"),
    "attraction": ("a sacrifice that drags a piece onto a bad square",
                   "check what a sacrifice next to your king would do"),
    "overloading": ("an overloaded defender", "check which of your pieces is doing two jobs at once"),
    "advancedPawn": ("a far-advanced pawn", "count how many moves an enemy pawn needs to promote"),
}
for _n in range(1, 6):
    MOTIF_WORDS[f"mateIn{_n}"] = (f"mate in {_n}", "look at every check your opponent has after your move")

CONCEPT_CHECKS = {
    "king safety": "ask what your move does to the pawns and pieces around your king",
    "piece activity": "ask which of your pieces your move leaves with fewer squares",
    "pawn structure": "ask which squares a pawn move gives up for good",
    "material": "count the attackers and defenders of every pawn and piece your move touches",
    "piece placement": "ask where each piece is headed before you move it",
    "knight placement": "ask where each piece is headed before you move it",
    "bishop placement": "ask where each piece is headed before you move it",
    "rook placement": "ask where each piece is headed before you move it",
    "queen placement": "ask where each piece is headed before you move it",
    "threats": "look at what your opponent threatens after your move",
    "space": "ask whether your move gives your opponent room to expand",
    "passed pawns": "count how many moves each passed pawn needs to promote",
}
GENERIC_CHECK = "look at every check, capture and threat your opponent has after your move"
MISSED_CHECK = "look at your own checks, captures and threats first"
COUNT_CHECK = "count the attackers and defenders of every pawn and piece before you move"
TEMPO_CHECK = "before you put a piece on a square, check whether a pawn can chase it away with a gain of time"
INVASION_CHECK = "ask which squares in your camp no pawn of yours can cover any more, and which enemy piece can reach them"
PIECE_WORDS = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
               chess.QUEEN: "queen", chess.KING: "king"}
MAX_TRADE_KINDS = 4  # "your queen and two pawns for a knight and a bishop"; with more kinds of piece, the amount
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


# --------------------------------------------------------------------------- small helpers
def pawns(cp: float) -> str:
    """"+0.40" / "−1.52" (pawns, a real minus sign), as the report writes evaluations."""
    return f"{cp / 100:+.2f}".replace("-", "−")


def verdict(line: Optional[Line], color: str, short: bool = False) -> str:
    """The engine's verdict on a line from your side: "−1.52 for you", "mate in 3 for White", "mate in 2 for you";
    past ``BIG_EVAL`` "a winning position for you" / "a lost position for you" instead of a number; ``short`` leaves
    "for you" off a number ("−0.39")."""
    if line is None:
        return "unclear"
    if line.mate_end is not None:
        n = abs(line.mate_end)
        if line.mate_end > 0:
            return f"mate in {n} for you" if n else "checkmate for you"
        winner = "Black" if color == "white" else "White"
        return f"mate in {n} for {winner}" if n else f"checkmate for {winner}"
    cp = line.cp_end or 0
    if abs(cp) > BIG_EVAL:
        return "a winning position for you" if cp > 0 else "a lost position for you"
    return pawns(cp) + ("" if short else " for you")


def _is_number(text: str) -> bool:
    """A verdict given as a number ("−1.52 for you"), not in words."""
    return text[:1] in "+−-" and text[1:2].isdigit()


def _amount(value: float) -> str:
    """'0.30 pawns' for a concept difference; 'far' past ``BIG_EVAL`` (a number that size means nothing)."""
    return "far" if abs(value) * 100 > BIG_EVAL else f"{abs(value):.2f} pawns"


def _label(fen: str, uci: str) -> str:
    """"5...a6" for a UCI move in ``fen``; "" if it isn't legal there."""
    board = chess.Board(fen)
    move = visuals.parse_move(board, uci)
    return visuals.move_label(board, move) if move is not None else ""


def _after(fen: str, uci: str) -> str:
    board = chess.Board(fen)
    board.push_uci(uci)
    return board.fen()


def _line_text(line: Line, start: int, plies: int) -> str:
    """Moves ``start`` .. ``start + plies`` of a line, numbered: "6.Ndb5 a6 7.Nd6+ Bxd6"."""
    board = chess.Board(line.fen)
    for uci in line.moves_uci[:start]:
        board.push_uci(uci)
    return format_line(line.moves_san[start : start + plies], board.fen())


def _chances(drop: float) -> str:
    """'12 percentage points' / '2.3 percentage points' (of winning chances), as the openings section writes them."""
    return f"{drop:.0f} percentage points" if drop >= 9.95 else f"{drop:.1f} percentage points"


def _material_words(n: int) -> str:
    return "a pawn" if n == 1 else f"{n} pawns' worth of material"


def _pieces_words(types: Sequence[int], yours: bool) -> str:
    """"your queen" / "the queen" / "a rook and a knight" / "two pawns" for captured piece types, dearest first."""
    counts = Counter(types)
    parts = []
    for t in sorted(counts, key=lambda t: (-concepts.PIECE_VALUES.get(t, 0), t)):
        n, name = counts[t], PIECE_WORDS[t]
        if n == 1:
            parts.append(("your " if yours else "the ") + name if t == chess.QUEEN else f"a {name}")
        else:
            parts.append(f"{NUMBER_WORDS.get(n, str(n))} {name}s")
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def trade_words(given: Sequence[int], taken: Sequence[int], net: int, yours: bool) -> str:
    """What changes hands, from the side that comes out behind (``yours``: you, else your opponent): "your queen for
    a rook and a knight", "a knight for two pawns", "a pawn". ``given`` are that side's pieces taken, ``taken`` the
    other side's. Equal values cancel (a knight for a bishop is a trade). "" when the pieces don't add up to ``net``
    pawns (a promotion on the way) or nothing is left after the trades."""
    value = concepts.PIECE_VALUES
    given, taken = list(given), list(taken)
    for v in sorted({value.get(t, 0) for t in given} & {value.get(t, 0) for t in taken}):
        k = min(sum(value.get(t, 0) == v for t in given), sum(value.get(t, 0) == v for t in taken))
        for bucket in (given, taken):
            for _ in range(k):
                bucket.remove(next(t for t in bucket if value.get(t, 0) == v))
    if not given or sum(value.get(t, 0) for t in given) - sum(value.get(t, 0) for t in taken) != net:
        return ""
    if len(set(given)) + len(set(taken)) > MAX_TRADE_KINDS:  # too long to read: the amount says it better
        return ""
    words = _pieces_words(given, yours)
    return words + (f" for {_pieces_words(taken, not yours)}" if taken else "")


def _down_words(cmp: "Comparison", lost: int) -> str:
    """"a knight down" / "down 4 pawns' worth of material" for what your move leaves you without."""
    words = cmp.lost_words
    if words and " for " not in words and " and " not in words:
        return words.replace("your ", "a ") + " down"
    return f"down {_material_words(lost)}"


def _more_words(n: int) -> str:
    """"a pawn more" / "2 pawns' worth more"."""
    return "a pawn more" if n == 1 else f"{n} pawns' worth more"


def _count_words(n: int) -> str:
    return NUMBER_WORDS.get(n, str(n))


def _motif_name(m: Motif, gated: frozenset) -> str:
    return MOTIF_WORDS.get(m.theme, (m.theme, ""))[0] if m.theme in gated else "a tactic"


class _Games:
    """What the explanations look up about games: the format by URL and the start position by id."""

    def __init__(self, ctx: "AnalysisContext") -> None:
        self.time_class = {g.url: g.time_class for g in ctx.games if g.url}
        self.by_id = {g.game_id: g.time_class for g in ctx.games}
        self.start = {g.game_id: g.initial_fen for g in ctx.games}

    def format_mix(self, urls: Sequence[str]) -> dict[str, int]:
        """Games per time class behind a position, in the report's format order."""
        return visuals.ordered_formats(Counter(self.time_class[u] for u in urls if u in self.time_class))

    def single_format(self, pos: CriticalPosition) -> str:
        """The position's format when all its games share one ("" for several): read from the games themselves (the
        format views keep an explanation by it), else ``pos.time_class`` when a game is not in the context."""
        urls = list(pos.games) or ([pos.url] if pos.url else [])
        formats = {self.time_class.get(u) for u in urls}
        if not urls and pos.game_id in self.by_id:
            formats = {self.by_id[pos.game_id]}
        if not formats or None in formats or "" in formats:
            return pos.time_class
        return formats.pop() if len(formats) == 1 else ""


# --------------------------------------------------------------------------- motifs and concepts
def detect(line: Optional[Line], role: str, previous_fen: Optional[str] = None) -> list[Motif]:
    """``motifs.detect_line`` made safe: any list is accepted, odd entries dropped, a failure is no motif."""
    if line is None or not line.moves_uci:
        return []
    try:
        try:
            found = (motifs.detect_line(line, role, previous_fen) if previous_fen
                     else motifs.detect_line(line, role)) or []
        except TypeError:  # a detector without the optional previous_fen argument
            found = motifs.detect_line(line, role) or []
    except Exception as exc:  # noqa: BLE001 — a detector bug must not cost the position its explanation
        log.debug("motif detection failed on %s: %s", line.fen, exc)
        return []
    return [m for m in found if isinstance(m, Motif)]


def _other_line(line: Optional[Line], side: str) -> list[Motif]:
    """``motifs.carried_by`` made safe (a detector without it, or a failure: nothing)."""
    carried = getattr(motifs, "carried_by", None)
    if carried is None or line is None or not line.moves_uci:
        return []
    try:
        return [m for m in carried(line, side) or [] if isinstance(m, Motif)]
    except Exception as exc:  # noqa: BLE001
        log.debug("motif detection failed on %s: %s", line.fen, exc)
        return []


def _same_pattern(a: Motif, b: Motif) -> bool:
    """The same theme on (mostly) the same squares: at least two in common, or all of a shorter list."""
    if a.theme != b.theme:
        return False
    common = set(a.squares) & set(b.squares)
    return len(common) >= min(2, len(set(a.squares)), len(set(b.squares))) and bool(common or not a.squares)


def split_motifs(best: Optional[Line], refutation: Optional[Line],
                 previous_fen: Optional[str] = None) -> tuple[list[Motif], list[Motif]]:
    """(what your move allowed: the opponent's patterns along the refutation, what you missed: yours along the
    best line). ``previous_fen``: the position before your opponent's last move, which tells the best line's
    first capture of a free piece from the end of a trade.

    Only what differs between the lines: a pattern your opponent also has along the better line was not allowed
    by your move (13.gxf3 Nxd4, when 13.Bxf7+ Kxf7 ... Nxd4 comes too), and one you carry out after your move anyway
    was not missed."""
    allowed = [m for m in detect(refutation, "refutation") if m.side == "opponent"]
    missed = [m for m in detect(best, "best", previous_fen) if m.side == "you"]
    if allowed:
        anyway = _other_line(best, "second")
        allowed = [m for m in allowed if not any(_same_pattern(m, o) for o in anyway)]
    if missed:
        anyway = _other_line(refutation, "first")
        missed = [m for m in missed if not any(_same_pattern(m, o) for o in anyway)]
    return allowed, missed


def previous_position(start: Optional[str], moves_before: Sequence[str], epd: str) -> Optional[str]:
    """The FEN before the last move of ``moves_before`` (your opponent's move into the position), or None when
    the moves don't lead to ``epd``."""
    if not moves_before:
        return None
    try:
        board = chess.Board(start or chess.STARTING_FEN)
        for san in moves_before[:-1]:
            board.push_san(san)
        previous = board.fen()
        board.push_san(moves_before[-1])
    except ValueError:
        return None
    return previous if board.epd() == epd else None


NOTE_SKIP = frozenset({"Material", "Imbalance", "Winnable"})  # not concepts a classic's note should teach here


def concept_note_for(deltas: Sequence[ConceptDelta], king: bool = True) -> tuple[dict, list[Source]]:
    """The note on the concept your move made worst (the most negative difference with a note; never Material,
    Imbalance or Winnable), and its citations ({}, [] when none covers it). A concept on which your move did better
    gets no note: the note would teach the opposite of what the line shows. ``king`` False: your move did not touch
    your king's shelter, so no note on king safety (castle early ...) either."""
    worse = [d for d in deltas if d.value < 0 and d.term not in NOTE_SKIP and (king or d.term != "King safety")]
    if not worse:
        return {}, []
    try:
        from .sources.concept_notes import concept_note, note_sources
    except ImportError:  # the notes ship with the package; be safe in a trimmed install
        return {}, []
    for delta in sorted(worse, key=lambda d: (d.value, d.term)):
        note = concept_note(delta.term) or concept_note(delta.label)
        if note:
            return ({"label": note.get("label", ""), "text": note.get("text", ""), "credit": note.get("credit", ""),
                     "url": note.get("url", "")}, note_sources(note))
    return {}, []


@dataclass
class Comparison:
    """What the two lines leave behind, read at their settled comparison points (``concepts.comparison_ply``)."""

    deltas: list[ConceptDelta] = field(default_factory=list)  # refutation end minus best end, your side
    facts: list[str] = field(default_factory=list)  # what is worse for you at the refutation end, in words
    # Material (pawn 1, knight and bishop 3, rook 5, queen 9) the refutation takes from you: against the position
    # before your move (before your opponent's last move when that was a capture: a piece taken and not taken back is
    # lost) AND against the better line (the smaller of the two, so a pawn you lose either way is not blamed on your
    # move) ...
    material_lost: int = 0
    # ... and what the better line wins for you, against the same starting count and against the refutation (54.gxh6
    # takes a rook that 54.d6 leaves alone). A recapture wins nothing; neither does a pawn taken in a line that
    # Stockfish still rates more than a pawn below the material (a gambit: the other side has play for it).
    material_missed: int = 0
    lost_words: str = ""  # what changes hands along the refutation: "your queen for a rook and a knight"
    missed_words: str = ""  # what the better line wins: "a knight", "the queen for a rook"
    lost_relative: bool = False  # the better line loses material too: the loss is counted against it
    missed_relative: bool = False  # your move's line wins material too: the win is counted against it
    # What the refutation takes from you against the starting count alone, when its own verdict shows it (a pawn of
    # evaluation per two of material): named even when the better line loses material later on (33...Bg6 loses the
    # queen; 33...Kh7 is bad too, so the difference between the lines is small).
    lost_abs: int = 0
    recapture: str = ""  # "pawn" / "piece": the better move takes back what your opponent has just taken
    refutation_ply: Optional[int] = None  # the comparison points: moves played along each line
    best_ply: Optional[int] = None


def last_move(previous_fen: Optional[str], fen: str) -> Optional[tuple[chess.Board, chess.Move]]:
    """(the position before it, the move) that leads from ``previous_fen`` to ``fen``; None when none does."""
    if not previous_fen:
        return None
    try:
        before = chess.Board(previous_fen)
        target = chess.Board(fen)
    except ValueError:
        return None
    for move in before.legal_moves:
        before.push(move)
        same = before.board_fen() == target.board_fen() and before.turn == target.turn
        before.pop()
        if same:
            return before, move
    return None


def _line_board(fen: str, moves_uci: Sequence[str], n: int) -> chess.Board:
    board = chess.Board(fen)
    for uci in moves_uci[:n]:
        board.push_uci(uci)
    return board


def _pawns_of(line: Line) -> float:
    """The engine's verdict on a line in pawns (a mate counts as +-MATE_CP)."""
    return (line.cp_end or 0) / 100.0


def _reach(fen: str, moves_uci: Sequence[str]) -> int:
    """``motifs.forcing_reach`` for your moves along a line (the whole line when the detector has none)."""
    reach = getattr(motifs, "forcing_reach", None)
    try:
        return reach(fen, list(moves_uci), 0) if reach is not None else len(moves_uci)
    except Exception:  # noqa: BLE001
        return len(moves_uci)


class _Concepts:
    """Concept deltas and board facts at the comparison points of two lines, with one Stockfish 16 process."""

    def __init__(self, path: Optional[str]) -> None:
        self.eval = concepts.ClassicalEval(path)

    def close(self) -> None:
        self.eval.close()

    def compare(self, fen: str, best: Line, refutation: Line, color: str,
                previous_fen: Optional[str] = None) -> Comparison:
        b_n = concepts.comparison_ply(fen, best.moves_uci)
        r_n = concepts.comparison_ply(fen, refutation.moves_uci)
        if b_n is None or r_n is None:
            return Comparison()
        b_end, r_end = _line_board(fen, best.moves_uci, b_n), _line_board(fen, refutation.moves_uci, r_n)
        me = chess.WHITE if color == "white" else chess.BLACK
        b_facts, r_facts = concepts.board_facts(b_end, color), concepts.board_facts(r_end, color)  # type: ignore[arg-type]
        # a king that castles a move or two after the comparison point is not left in the centre
        if concepts.castles_within(fen, refutation.moves_uci, r_n, color):  # type: ignore[arg-type]
            r_facts.king_in_centre = r_facts.king_central = False
        if concepts.castles_within(fen, best.moves_uci, b_n, color):  # type: ignore[arg-type]
            b_facts.king_central = False
        start = chess.Board(fen)
        base = concepts.material(start, me)
        # Your opponent's last move took something: count from before it, so that taking back is no win and not
        # taking back is a loss.
        last = last_move(previous_fen, fen)
        taken: Optional[int] = None
        if last is not None and (last[0].is_capture(last[1])):
            before, move = last
            base = concepts.material(before, me)
            taken = chess.PAWN if before.is_en_passant(move) else before.piece_type_at(move.to_square)
        b_mat, r_mat = b_facts.material, r_facts.material
        extra = [taken] if taken else []
        value = concepts.PIECE_VALUES
        # What the better line wins. Material taken by force (your captures while each of your moves is a check or a
        # capture, ``motifs.forcing_reach``) counts; material the other side gives up later counts only when the
        # engine's verdict shows it (at most a pawn below the material): a pawn taken in a gambit line that
        # Stockfish still rates in the other side's favour is not "won" (2...cxd4 3.Nf3 e5 4.c3 Nf6 5.cxd4 Nxe4).
        b_given, b_got = concepts.captures(fen, best.moves_uci[:b_n], color)  # type: ignore[arg-type]
        missed_abs = b_mat - base
        mating = best.mate_end is not None and best.mate_end > 0
        if missed_abs > 0 and not mating and _pawns_of(best) < b_mat - 1:
            reach = _reach(fen, best.moves_uci)
            forced = concepts.captures(fen, best.moves_uci[: min(b_n, reach + 1)], color)[1]  # type: ignore[arg-type]
            forced_abs = sum(value.get(t, 0) for t in forced) - sum(value.get(t, 0) for t in b_given + extra)
            if forced_abs < missed_abs:
                missed_abs, b_got = forced_abs, forced
        lost_abs, gap = base - r_mat, b_mat - r_mat
        lost, missed = max(0, min(lost_abs, gap)), max(0, min(missed_abs, gap))
        out = Comparison(
            facts=concepts.fact_differences(b_facts, r_facts, start=concepts.board_facts(start, color)),  # type: ignore[arg-type]
            material_lost=lost,
            material_missed=missed,
            refutation_ply=r_n,
            best_ply=b_n,
        )
        if lost_abs > 0:
            given, got = concepts.captures(fen, refutation.moves_uci[:r_n], color)  # type: ignore[arg-type]
            out.lost_words = trade_words(given + extra, got, lost_abs, yours=True)
            out.lost_relative = 0 < lost < lost_abs
            if _pawns_of(refutation) <= -lost_abs / 2:  # the line's own verdict shows the loss
                out.lost_abs = lost_abs
        if missed:
            out.missed_words = trade_words(b_got, b_given + extra, missed_abs, yours=False)
            out.missed_relative = missed < missed_abs
        if last is not None and taken and best.moves_uci:
            first = chess.Move.from_uci(best.moves_uci[0])
            if first.to_square == last[1].to_square and start.is_capture(first):
                out.recapture = "pawn" if taken == chess.PAWN else "piece"
        b_table = self.eval.table(b_end.fen())
        r_table = self.eval.table(r_end.fen()) if b_table is not None else None
        if b_table is not None and r_table is not None:
            out.deltas = concepts.concept_deltas(
                b_table, r_table, color, concepts.material_phase(b_end), concepts.material_phase(r_end),  # type: ignore[arg-type]
                material_level=b_mat == r_mat,
            )
        return out


def _gap(best: Line, refutation: Line) -> float:
    """How much better the best line is than the refutation for you, in pawns (mates count as 10)."""
    return ((best.cp_end or 0) - (refutation.cp_end or 0)) / 100.0


def win_drop(best: Line, refutation: Line) -> float:
    """Win-% points your move loses in the deeper search (the game analysis's measure of an error)."""
    return sf.win_percent(best.cp_end, best.mate_end) - sf.win_percent(refutation.cp_end, refutation.mate_end)


def is_close(best: Line, refutation: Line) -> bool:
    """The deeper search hardly minds your move: under ``CLOSE_DROP`` win-% points and ``CLOSE_GAP`` pawns behind
    its first choice (a different move)."""
    return (bool(best.moves_uci) and bool(refutation.moves_uci) and best.moves_uci[0] != refutation.moves_uci[0]
            and win_drop(best, refutation) < CLOSE_DROP and _gap(best, refutation) < CLOSE_GAP)


def shown_line(best: Line, refutation: Line) -> Line:
    """The line after your move that the text and the strip show. When the deeper search's first choice is your
    own move, its MultiPV line (so your move reads with the same number as in the other positions' texts), unless
    that line stops at your move."""
    if best.moves_uci and refutation.moves_uci and best.moves_uci[0] == refutation.moves_uci[0] \
            and len(best.moves_uci) > 1:
        return best
    return refutation


def _mate_line(best: Line, refutation: Line) -> bool:
    """A forced mate decides one of the lines: then material and concepts along the way say nothing."""
    return (refutation.mate_end is not None and refutation.mate_end < 0) or (
        best.mate_end is not None and best.mate_end > 0)


# --------------------------------------------------------------------------- the words
def _material_backed(cmp: Comparison, gap: float) -> tuple[int, int]:
    """(material lost, material missed) that the text may name: each backed by the engine's verdict, at least half a
    pawn of evaluation per pawn of material named (the loss first). Material counted in the middle of a line can be
    passing (a knight won for two pawns that come back later), and the verdict is what tells."""
    lost = cmp.material_lost if cmp.material_lost >= 1 and gap >= cmp.material_lost / 2 else 0
    missed = cmp.material_missed if cmp.material_missed >= 1 and gap >= (lost + cmp.material_missed) / 2 else 0
    return lost, missed


def _plain_loss(cmp: Comparison, gap: float) -> int:
    """A loss of a piece or more that the refutation's own verdict shows, when the difference between the lines
    does not back it (the better line loses material later on): named as what changes hands, without a comparison."""
    lost, _ = _material_backed(cmp, gap)
    return cmp.lost_abs if not lost and cmp.lost_abs >= 3 and cmp.lost_words else 0


@dataclass
class Point:
    """The point of a refutation that the board shows (a pawn counted short of defenders, a piece chased by a pawn,
    an enemy piece on a hole), in words, and what to check next time."""

    kind: str  # "count" | "tempo" | "invasion"
    text: str
    check: str
    covers_material: bool = False  # the words already say what you lose


def _opponent(color: str) -> str:
    return "Black" if color == "white" else "White"


def _count_point(pos: CriticalPosition, refutation: Line, cmp: Comparison, lost: int) -> Optional[Point]:
    """Your opponent's first reply takes a pawn or piece of yours that more of their pieces attacked than yours
    defended, and the capture wins material by the static count (4.e4 Qxd4: two attackers on d4, one defender)."""
    if len(refutation.moves_uci) < 2:
        return None
    before = chess.Board(pos.fen)
    mine = chess.Move.from_uci(refutation.moves_uci[0])
    board = before.copy(stack=False)
    board.push(mine)
    reply = chess.Move.from_uci(refutation.moves_uci[1])
    if reply not in board.legal_moves or not board.is_capture(reply) or board.is_en_passant(reply):
        return None
    victim = board.piece_type_at(reply.to_square)
    if victim is None or victim == chess.KING:
        return None
    if mine.to_square == reply.to_square and before.is_capture(mine):
        if concepts.see(before, mine) >= 0:  # your capture was a fair trade (9...Bxf3 10.Bxf3)
            return None
    elif concepts.see(board, reply) <= 0:
        return None
    attackers = len(board.attackers(board.turn, reply.to_square))
    defenders = len(board.attackers(not board.turn, reply.to_square))
    if attackers <= defenders:
        return None
    them = _opponent(pos.color).lower()
    attack = f"one {them} piece attacks" if attackers == 1 else f"{_count_words(attackers)} {them} pieces attack"
    defend = ("none of yours defends" if not defenders else
              f"only {_count_words(defenders)} of yours {'defends' if defenders == 1 else 'defend'}")
    name = PIECE_WORDS[victim]
    text = (f"{visuals.move_label(board, reply)} takes the {name} on {chess.square_name(reply.to_square)}, which "
            f"{attack} and {defend}")
    covers = not cmp.lost_relative and lost == concepts.PIECE_VALUES.get(victim, 0)
    return Point("count", text, COUNT_CHECK, covers_material=covers)


def _tempo_point(pos: CriticalPosition, best: Line, refutation: Line, best_label: str) -> Optional[Point]:
    """The piece you just moved is hit by a pawn and has to move again (2...Nc6 3.d5 Nb8); when the better move
    takes the chasing pawn first (2...cxd4), say so."""
    if len(refutation.moves_uci) < 3:
        return None
    board = chess.Board(pos.fen)
    mine = chess.Move.from_uci(refutation.moves_uci[0])
    piece = board.piece_type_at(mine.from_square)
    if piece in (None, chess.PAWN, chess.KING) or board.is_castling(mine):
        return None
    board.push(mine)
    reply = chess.Move.from_uci(refutation.moves_uci[1])
    if reply not in board.legal_moves or board.piece_type_at(reply.from_square) != chess.PAWN:
        return None
    reply_label = visuals.move_label(board, reply)
    board.push(reply)
    again = chess.Move.from_uci(refutation.moves_uci[2])
    if not board.attacks_mask(reply.to_square) & chess.BB_SQUARES[mine.to_square] or again.from_square != mine.to_square \
            or again not in board.legal_moves:
        return None
    text = (f"{reply_label} hits your {PIECE_WORDS[piece]} with a pawn and it has to move again "
            f"({visuals.move_label(board, again)}), so you lose time")
    first = chess.Move.from_uci(best.moves_uci[0]) if best.moves_uci else None
    if first is not None and first.to_square == reply.from_square and chess.Board(pos.fen).is_capture(first):
        text += f"; {best_label} first takes the pawn that chases it"
    return Point("tempo", text, TEMPO_CHECK)


def _invasion_point(pos: CriticalPosition, best: Line, refutation: Line, plies: int) -> Optional[Point]:
    """An enemy piece lands on a hole in your camp within the moves the text quotes (5...e5 6.Ndb5 a6 7.Nd6+), and
    not on the same square along the better line."""
    found = concepts.hole_invasion(pos.fen, refutation.moves_uci, pos.color, plies)  # type: ignore[arg-type]
    if found is None:
        return None
    other = concepts.hole_invasion(pos.fen, best.moves_uci, pos.color, plies)  # type: ignore[arg-type]
    if other is not None and other.square == found.square:
        return None
    text = (f"{_opponent(pos.color)}'s {found.piece} gets to {found.square} ({found.san}), a square no pawn of yours "
            f"can cover")
    return Point("invasion", text, INVASION_CHECK)


def _recapture_point(pos: CriticalPosition, cmp: Comparison, best_label: str) -> Optional[Point]:
    """The better move takes back what your opponent has just taken, and your move does not (5.exd6 Nf6?: 5...Qxd6
    takes back the pawn)."""
    if not cmp.recapture:
        return None
    return Point("recapture", f"{best_label} takes back the {cmp.recapture} {_opponent(pos.color)} has just taken",
                 MISSED_CHECK)


def find_point(pos: CriticalPosition, best: Line, refutation: Line, cmp: Comparison, gap: float,
               best_label: str) -> Optional[Point]:
    """The first point the board shows: a pawn or piece counted short of defenders, and, when the text names no
    material, a recapture you skipped, a piece chased by a pawn, or an enemy piece on a hole in your camp."""
    lost, missed = _material_backed(cmp, gap)
    point = _count_point(pos, refutation, cmp, lost)
    if point is not None or lost or missed or _plain_loss(cmp, gap):
        return point
    return (_recapture_point(pos, cmp, best_label) or _tempo_point(pos, best, refutation, best_label)
            or _invasion_point(pos, best, refutation, TEXT_PLIES + 1))


def _concept_words(deltas: Sequence[ConceptDelta], n: int = 2) -> str:
    worse = [c for c in deltas if c.value < 0 and c.term not in concepts.MATERIAL_TERMS][:n]
    if not worse:
        return ""
    text = f"a few moves later you stand {_amount(worse[0].value)} worse on {worse[0].label}"
    if len(worse) > 1:
        second = _amount(worse[1].value)
        text += f" and {'far worse' if second == 'far' else second.replace(' pawns', '')} on {worse[1].label}"
    return text


def _what_it_wins(cmp: Comparison, gap: float, best_label: str = "", played: str = "",
                  point: Optional[Point] = None) -> str:
    """"you lose a knight for two pawns" / "5.Nxe5 would have won the queen" (when the engine's verdict backs it;
    what changes hands, else the amount; "more than after ..." when the other line moves material too; a recapture
    "takes back the piece"), else the two concepts the refutation leaves you worst on; plus the first board fact.
    ``point``: the point of the line (``find_point``) comes first, with at most one more item."""
    parts = [point.text] if point is not None else []
    lost, missed = _material_backed(cmp, gap)
    plain = _plain_loss(cmp, gap)
    better = best_label or "the better move"
    yours = f"your {played}" if played else "your move"
    material = []
    covered = point is not None and point.covers_material
    if lost and not covered:
        if cmp.recapture and not missed:
            material.append(f"{better} takes back the {cmp.recapture}, and {yours} leaves you "
                            f"{_down_words(cmp, lost)}")
        elif cmp.lost_relative:
            material.append(f"you lose {cmp.lost_words} ({_more_words(lost)} than after {better})"
                            if cmp.lost_words else f"you lose {_material_words(lost)} more than after {better}")
        else:
            material.append(f"you lose {cmp.lost_words or _material_words(lost)}")
    elif plain and not covered:
        material.append(f"you lose {cmp.lost_words}")
    if missed:
        if cmp.missed_relative:
            material.append(f"{better} would have won {cmp.missed_words} ({_more_words(missed)} than {yours})"
                            if cmp.missed_words else
                            f"{better} would have won {_material_words(missed)} more than {yours}")
        else:
            material.append(f"{better} would have won {cmp.missed_words or _material_words(missed)}")
    if material:
        parts.append(", and ".join(material))
    elif point is None or not (point.covers_material or ";" in point.text):
        concept = _concept_words(cmp.deltas, 1 if point is not None else 2)
        if concept:
            parts.append(concept)
    # a third item would make the sentence too long
    if cmp.facts and len(parts) < 2 and not any(", and " in p or ";" in p for p in parts):
        parts.append(cmp.facts[0])
    parts = parts[:2]
    if len(parts) == 2:
        return f"{parts[0]}, and {parts[1]}"
    return parts[0] if parts else ""


def _near_king(pos: CriticalPosition) -> bool:
    """Whether your move touches your king's shelter: a king move or castling, a pawn on the king's file or next to
    it, or a piece leaving a square next to the king. (A knight developing on the other wing does not.)"""
    try:
        board = chess.Board(pos.fen)
        move = chess.Move.from_uci(pos.played_uci)
    except ValueError:
        return True
    me = board.turn
    king = board.king(me)
    piece = board.piece_type_at(move.from_square)
    if king is None or piece is None or piece == chess.KING or board.is_castling(move):
        return True
    if piece == chess.PAWN:
        return abs(chess.square_file(move.from_square) - chess.square_file(king)) <= 1
    return chess.square_distance(move.from_square, king) <= 1


def _check_next_time(allowed: Sequence[Motif], missed: Sequence[Motif], gated: frozenset, cmp: Comparison,
                     gap: float, mated: bool, mating: bool = False, point: Optional[Point] = None,
                     near_king: bool = True) -> str:
    """What to look at before a move like this one, from the motif, else the point of the line, else the material,
    else the worst concept (king safety only when your move touched the king's shelter)."""
    if mated:
        return MOTIF_WORDS["mateIn1"][1]
    named = [m for m in allowed if m.theme in gated and m.theme in MOTIF_WORDS]
    if named:
        return MOTIF_WORDS[named[0].theme][1]
    if allowed:
        return GENERIC_CHECK
    if missed or mating:
        return MISSED_CHECK
    if point is not None:
        return point.check
    lost, gained = _material_backed(cmp, gap)
    if lost or gained:
        return CONCEPT_CHECKS["material"] if lost and not cmp.recapture else MISSED_CHECK
    if _plain_loss(cmp, gap):
        return CONCEPT_CHECKS["material"]
    worse = [c for c in cmp.deltas if c.value < 0 and c.label in CONCEPT_CHECKS and c.term not in concepts.MATERIAL_TERMS
             and (near_king or c.label != "king safety")]
    if worse:
        return CONCEPT_CHECKS[worse[0].label]
    return GENERIC_CHECK


def _leads(m: Motif, line: Line) -> bool:
    """Whether a motif may lead the text. A far-advanced pawn does only when it goes on to promote in the line
    (6.dxe7 Nxe7 is a pawn taken back, not a pawn about to queen): otherwise it is a chip at most."""
    if m.theme != "advancedPawn":
        return True
    square = m.squares[0] if m.squares else ""
    return any(len(u) == 5 and (u[:2] == square or k == m.ply)
               for k, u in enumerate(line.moves_uci[m.ply : m.ply + 4], start=m.ply))


def _mate_name(missed: Sequence[Motif], gated: frozenset, n: int) -> str:
    """"a back-rank mate in 3" / "a smothered mate in 2" / "mate in 3"."""
    themes = {m.theme for m in missed if m.theme in gated}
    for theme, words in (("backRankMate", "a back-rank mate"), ("smotheredMate", "a smothered mate")):
        if theme in themes:
            return f"{words} in {n}"
    return f"mate in {n}"


def shown_plies(best: Line, refutation: Line, allowed: Sequence[Motif], missed: Sequence[Motif],
                cmp: Optional[Comparison] = None) -> tuple[int, int]:
    """(moves of the refutation after your move, moves of the better line) that the text quotes and the strips
    draw: ``TEXT_PLIES`` (and one more for the better line), or up to the move where the leading pattern cashes in
    (``Motif.ply + 2``) or where the material the text names has changed hands (the comparison point), or a forced
    mate for you to the end; at most ``MAX_TEXT_PLIES``."""
    ref_plies, best_plies = TEXT_PLIES, TEXT_PLIES + 1
    leading = [m for m in allowed if _leads(m, refutation)]
    if leading:
        ref_plies = min(MAX_TEXT_PLIES, max(ref_plies, leading[0].ply + 2))
    if cmp is not None:
        gap = _gap(best, refutation)
        lost, missed_material = _material_backed(cmp, gap)
        if (lost or _plain_loss(cmp, gap)) and cmp.refutation_ply:
            ref_plies = max(ref_plies, min(MAX_TEXT_PLIES, cmp.refutation_ply - 1))
        if missed_material and cmp.best_ply:
            best_plies = max(best_plies, min(MAX_TEXT_PLIES + 1, cmp.best_ply))
    mated = refutation.mate_end is not None and refutation.mate_end < 0
    if best.mate_end is not None and best.mate_end > 0 and not mated:
        if 2 * best.mate_end - 1 <= LEAD_MATE_PLIES:
            best_plies = max(best_plies, 2 * best.mate_end - 1)
    else:
        lead = [m for m in missed if _leads(m, best)]
        if lead:
            best_plies = min(MAX_TEXT_PLIES + 1, max(best_plies, lead[0].ply + 3))
    return ref_plies, best_plies


def explanation_text(
    pos: CriticalPosition,
    best: Line,
    refutation: Line,
    allowed: Sequence[Motif],
    missed: Sequence[Motif],
    cmp: Comparison,
    gated: frozenset,
    depth: Optional[int] = None,
    choice_min_drop: float = CHOICE_MIN_DROP,
) -> str:
    """At most three sentences: what the refutation does, what it wins, what to check next time.

    When the deeper search finds your move as good as its own (the same move) or close to it (``is_close``: not even
    an inaccuracy), the text says so instead of explaining an error that is not one. At a choice point the text
    says Stockfish prefers another move only when the game analysis calls yours a mistake, as the openings section
    does (``critical.engine_mistake`` with ``choice_min_drop``); otherwise it gives Stockfish's first choice and its
    line after your move side by side, without a verdict.

    For an error: a forced mate against you leads, then a forced mate for you ("You had mate in 3"), then what your
    move allowed ("After 15.Qd1, Black has an undefended piece to take: ..."), then what you missed ("You had a
    fork: ..."; the next sentence then names your move and its evaluation), else Stockfish's answer and the point
    of the line (``find_point``).
    """
    played = pos.move_label
    best_label = _label(pos.fen, best.moves_uci[0]) if best.moves_uci else ""
    opponent = _opponent(pos.color)
    same = bool(best.moves_uci) and best.moves_uci[0] == pos.played_uci
    refutation = shown_line(best, refutation)
    ref_v, best_v = verdict(refutation, pos.color), verdict(best, pos.color, short=True)
    gap = _gap(best, refutation)
    answer = _line_text(refutation, 1, TEXT_PLIES)
    close = not same and is_close(best, refutation)
    at = f" at depth {depth}" if depth else ""

    if pos.kind == "choice":
        mistake = not (same or close) and engine_mistake(pos, choice_min_drop)
        if same:
            first = f"Stockfish agrees with your {played} ({ref_v})"
        elif close:
            first = f"Your {played} ({ref_v}) is close to Stockfish's first choice, {best_label} ({best_v})"
        elif mistake:
            first = f"Stockfish prefers {best_label} ({verdict(best, pos.color)}) to your {played} ({ref_v})"
        else:  # below the openings section's bar: two good moves side by side, not an error
            first = f"Stockfish's first choice here is {best_label} ({verdict(best, pos.color)})"
            first += f"; after your {played} ({ref_v}) its main line is {answer}." if answer else \
                f", and it rates your {played} {ref_v if _is_number(ref_v) else 'as ' + ref_v}."
            if pos.drop > 0:  # 0.0: no analysed game with the move, so no average to quote
                bar = "less than an inaccuracy" if pos.drop < CLOSE_DROP else \
                    f"under the {choice_min_drop:g} this report counts as a mistake at a choice point"
                first += (f" In the games it analysed, your {played} lost {_chances(pos.drop)} of winning chances on "
                          f"average, {bar}: in the opening several moves are often about equally good.")
            return first
        first += f"; its main line after {played} is {answer}." if answer else "."
        point = find_point(pos, best, refutation, cmp, gap, best_label) if mistake else None
        wins = _what_it_wins(cmp, gap, best_label, played, point) if mistake else ""
        return first + (f" Compared with {best_label}, {wins}." if wins else "")

    if same:
        return (f"A deeper look{at} makes your {played} Stockfish's own first choice ({ref_v}), so the verdict of the "
                f"quicker game analysis on it does not hold up" + (f"; the main line is {answer}." if answer else "."))
    if close:
        return (f"A deeper look{at} finds your {played} ({ref_v}) close to Stockfish's first choice, {best_label} "
                f"({best_v}), so the quicker game analysis was too harsh on it"
                + (f"; the main line after {played} is {answer}." if answer else "."))

    mated = refutation.mate_end is not None and refutation.mate_end < 0
    mating = not mated and best.mate_end is not None and best.mate_end > 0
    leading = [m for m in allowed if _leads(m, refutation)]
    lead_missed = [m for m in missed if _leads(m, best)]
    ref_plies, best_plies = shown_plies(best, refutation, allowed, missed, cmp)
    shown = _line_text(refutation, 1, ref_plies)
    point = None
    if not (mated or mating or leading or lead_missed):
        point = find_point(pos, best, refutation, cmp, gap, best_label)
    your_side = False  # the second sentence names your move ("After your 50...Nd4, Stockfish rates the position")
    if mated:
        first = f"After {played}, {opponent} mates in {abs(refutation.mate_end)}" + (f": {answer}." if answer else ".")
    elif mating:
        first = f"You had {_mate_name(missed, gated, best.mate_end)}: {_line_text(best, 0, best_plies)}."
        your_side = True
    elif leading:
        first = f"After {played}, {opponent} has {_motif_name(leading[0], gated)}" + (f": {shown}." if shown else ".")
    elif lead_missed:
        first = f"You had {_motif_name(lead_missed[0], gated)}: {_line_text(best, 0, best_plies)}."
        your_side = True
    elif shown:
        first = f"After {played}, Stockfish's answer is {shown}."
    else:
        first = f"Stockfish prefers {best_label} to {played}."
    if mated:  # the mate is the whole story: no material or concepts on the way to it
        second = f"Stockfish prefers {best_label} ({verdict(best, pos.color)})."
    else:
        wins = "" if mating else _what_it_wins(cmp, gap, best_label, played, point)
        rated = ref_v if _is_number(ref_v) else "as " + ref_v
        second = (f"After your {played}, Stockfish rates the position {rated}" if your_side
                  else f"Stockfish rates the line {rated}") + f", against {best_v} after {best_label}"
        second += f": {wins}." if wins else "."
    check = _check_next_time(leading, lead_missed, gated, cmp, gap, mated, mating, point, _near_king(pos))
    return f"{first} {second} Next time, {check}."


# --------------------------------------------------------------------------- the pictures
CHECK_THEMES = frozenset({"discoveredCheck", "doubleCheck"})


def motif_marks(m: Motif, board: chess.Board, carrier: chess.Color,
                reference: Optional[chess.Board] = None) -> list[Mark]:
    """Marks for one motif on ``board`` (the position right after ``Motif.ply``): a piece of the side carrying the
    pattern out is an "attacker", a piece of the other side a "target"; for a discovered or double check, the
    checking pieces are attackers and the king the target, nothing else. Empty squares get no mark.

    ``reference``: another board the marks go on (the position before your move). The motif is marked there only
    when every piece it marks stands on the same square in both; otherwise it gets no mark on that board."""
    checkers = board.checkers() if m.theme in CHECK_THEMES else chess.SquareSet()
    out: list[Mark] = []
    for name in m.squares or []:
        if name not in chess.SQUARE_NAMES:
            continue
        square = chess.parse_square(name)
        piece = board.piece_at(square)
        if piece is None:
            continue
        if reference is not None and reference.piece_at(square) != piece:
            return []
        if m.theme in CHECK_THEMES:
            if piece.piece_type == chess.KING and piece.color != carrier:
                kind = "target"
            elif square in checkers:
                kind = "attacker"
            else:
                continue
        else:
            kind = "attacker" if piece.color == carrier else "target"
        if all(x.square != name for x in out):
            out.append(Mark(name, kind))
    return out


def _carrier(m: Motif, color: str) -> chess.Color:
    me = chess.WHITE if color == "white" else chess.BLACK
    return me if m.side == "you" else not me


def _boards(line: Line, n: int) -> list[chess.Board]:
    """The positions after each of the first ``n`` moves of ``line`` (fewer when it stops)."""
    board = chess.Board(line.fen)
    out = []
    for uci in line.moves_uci[:n]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        board.push(move)
        out.append(board.copy(stack=False))
    return out


def _add(marks: list[Mark], new: Sequence[Mark]) -> None:
    for mark in new:
        if all(x.square != mark.square for x in marks):
            marks.append(mark)


def _board_marks(pos: CriticalPosition, best: Line, refutation: Line, ms: Sequence[Motif],
                 gated: frozenset) -> list[Mark]:
    """Marks on the board before your move: only patterns whose pieces all stand there already (a pin your move
    creates by stepping off a line); the rest are marked on the strip frame where they appear."""
    reference = chess.Board(pos.fen)
    out: list[Mark] = []
    for m in ms:
        if m.theme not in gated:
            continue
        boards = _boards(best if m.line == "best" else refutation, m.ply + 1)
        if len(boards) == m.ply + 1:
            _add(out, motif_marks(m, boards[-1], _carrier(m, pos.color), reference))
    return out


def _frame_marks(ms: Sequence[Motif], gated: frozenset, line: Line, offset: int, frames: int,
                 color: str) -> list[list[Mark]]:
    """Marks per strip frame: a motif at line ply ``p`` shows on frame ``p - offset`` (the position after that
    move), one mark per square."""
    out: list[list[Mark]] = [[] for _ in range(frames)]
    boards = _boards(line, offset + frames)[offset:]
    for m in ms:
        i = m.ply - offset
        if m.theme in gated and 0 <= i < min(frames, len(boards)):
            _add(out[i], motif_marks(m, boards[i], _carrier(m, color)))
    return out


def _strip(title: str, fen: str, moves: Sequence[str], last_caption: str, marks: list[list[Mark]],
           frames: int = STRIP_FRAMES) -> Strip:
    n = min(frames, len(moves))
    captions = [""] * (n - 1) + [last_caption] if n else []
    return visuals.line_strip(title, fen, list(moves), max_frames=frames, captions=captions, marks=marks)


def _last_move(start: Optional[str], moves_before: Sequence[str], epd: str) -> str:
    """UCI of the move that led to the position (highlighted on the board); "" when the moves don't lead there."""
    if not moves_before:
        return ""
    try:
        board = chess.Board(start or chess.STARTING_FEN)
        for san in moves_before:
            board.push_san(san)
    except ValueError:
        return ""
    return board.peek().uci() if board.epd() == epd else ""


def build_diagram(games: _Games, pos: CriticalPosition, best: Line, refutation: Line,
                  allowed: Sequence[Motif], missed: Sequence[Motif], gated: frozenset, close: bool = False,
                  cmp: Optional[Comparison] = None) -> Diagram:
    """The board before your move (your move red, the better one green, your side at the bottom, the move before it
    highlighted; a motif marked only when all its pieces stand there already) with two strips: the refutation after
    your move and the better line, each as long as the text quotes it (``shown_plies``: to where the named pattern
    cashes in, at most ``MAX_STRIP_FRAMES`` boards), marked on the frame where the pattern appears and ending on the
    verdict. ``close``: the better line is only slightly better (its strip is "Stockfish's")."""
    played = pos.move_label
    best_uci = best.moves_uci[0] if best.moves_uci else None
    best_label = _label(pos.fen, best_uci) if best_uci else ""
    same = best_uci == pos.played_uci
    where = f"{pos.opening_family}: " if pos.opening_family else ""
    before = format_line(pos.moves_before, games.start.get(pos.game_id), last_n=6) if pos.moves_before else ""
    caption = (f"After {before}. " if before else "") + f"Red: your {played}"
    mix = games.format_mix(pos.games) if pos.repeats > 1 else {}
    if pos.repeats > 1:
        caption += f", in {pos.repeats} games" + (
            f" ({', '.join(f'{n} {tc}' for tc, n in mix.items())})" if len(mix) > 1 else "")
    caption += "." if same or not best_label else f". Green: {best_label}, Stockfish's choice."
    diagram = visuals.position_diagram(
        f"{where}{played}",
        pos.fen,
        orientation=pos.color,
        played=pos.played_uci,
        best=None if same else best_uci,
        caption=caption,
        link=pos.url,
        time_class=games.single_format(pos),
        last_move=_last_move(games.start.get(pos.game_id), pos.moves_before, pos.epd),
        marks=_board_marks(pos, best, refutation, list(allowed) + list(missed), gated),
    )
    ref_plies, best_plies = shown_plies(best, refutation, allowed, missed, cmp)
    mating = best.mate_end is not None and best.mate_end > 0 and not (
        refutation.mate_end is not None and refutation.mate_end < 0)
    after = _after(pos.fen, pos.played_uci)
    ref_moves = refutation.moves_uci[1:]
    strips = []
    if ref_moves:
        n = min(STRIP_FRAMES if ref_plies <= TEXT_PLIES else min(ref_plies, MAX_STRIP_FRAMES), len(ref_moves))
        strips.append(_strip(f"After your {played}", after, ref_moves, verdict(refutation, pos.color),
                             _frame_marks(allowed, gated, refutation, 1, n, pos.color), n))
    if best.moves_uci and not same:
        # the default strip is a move shorter than the text; a mate or a pattern the text follows to the end is
        # drawn to the end
        n = min(STRIP_FRAMES if best_plies <= TEXT_PLIES + 1 and not mating else min(best_plies, MAX_STRIP_FRAMES),
                len(best.moves_uci))
        title = f"Stockfish's {best_label}" if pos.kind == "choice" or close else f"The better {best_label}"
        strips.append(_strip(title, pos.fen, best.moves_uci, verdict(best, pos.color),
                             _frame_marks(missed, gated, best, 0, n, pos.color), n))
    diagram.strips = [s for s in strips if s.frames]
    return diagram


def concept_chart(pos: CriticalPosition, best_label: str, deltas: Sequence[ConceptDelta]) -> Optional[Chart]:
    if not deltas:
        return None
    return visuals.comparison_chart(
        f"{pos.move_label} against {best_label}, in pawns",
        [c.label[:1].upper() + c.label[1:] for c in deltas],
        [("Difference for you", [c.value for c in deltas])],
        value_format="signed_float2",
        kind="hbar",
        reference=0.0,
        note=(f"Stockfish 16's classical evaluation terms about {concepts.COMPARE_PLIES // 2} moves into each line "
              f"(at the first position where no capture wins material): the line after your {pos.move_label} minus "
              f"the line after {best_label}, from your side, in pawns. Below zero: worse for you. Only differences "
              f"of at least {concepts.MIN_DELTA:.2f} are shown; material is left out when the two lines end with "
              f"different material (the text says what changes hands)."),
    )


# --------------------------------------------------------------------------- one position
def verdict_of(pos: CriticalPosition, best: Line, refutation: Line,
               choice_min_drop: float = CHOICE_MIN_DROP) -> str:
    """``Explanation.verdict``: "fine" when the deeper search's first choice is the move played, "close" when it
    finds the move close to its choice (``is_close``) or at a choice point under the openings section's bar,
    else "error"."""
    if best.moves_uci and best.moves_uci[0] == pos.played_uci:
        return "fine"
    if is_close(best, refutation) or not engine_mistake(pos, choice_min_drop):
        return "close"
    return "error"


def explain_position(
    games: _Games,
    pos: CriticalPosition,
    result: "DeepResult",
    concept_tool: Optional[_Concepts],
    gated: frozenset,
    choice_min_drop: float = CHOICE_MIN_DROP,
) -> Optional[Explanation]:
    """The Explanation of one position from its deep result (None without both lines)."""
    if result.best_line is None or result.refutation is None or not result.best_line.moves_uci:
        return None
    best = rebase(result.best_line, pos.fen) if result.best_line.fen != pos.fen else result.best_line
    refutation = rebase(result.refutation, pos.fen) if result.refutation.fen != pos.fen else result.refutation
    if not refutation.moves_uci or refutation.moves_uci[0] != pos.played_uci:
        return None
    same = best.moves_uci[0] == pos.played_uci
    close = not same and is_close(best, refutation)  # the deeper search hardly minds your move
    # a choice point the game analysis does not call a mistake: two good moves side by side, not an error
    fine = same or close or not engine_mistake(pos, choice_min_drop)
    shown = shown_line(best, refutation)  # the line after your move in the text and the strip
    # No motifs, drill themes or concept comparison for a move that is not an error, and no concepts when a forced
    # mate decides a line (material and structure on the way to it say nothing).
    previous = previous_position(games.start.get(pos.game_id), pos.moves_before, pos.epd)
    allowed, missed = ([], []) if fine else split_motifs(best, refutation, previous)
    cmp = Comparison()
    if concept_tool is not None and not (fine or _mate_line(best, refutation)):
        cmp = concept_tool.compare(pos.fen, best, refutation, pos.color, previous)
    depths = [x.depth for x in (best, refutation) if x.depth]
    depth = min(depths) if depths else result.depth
    engine_name = result.engine or "Stockfish"
    sources = [Source(name=f"{engine_name}, depth {depth}" if depth else engine_name)]
    if cmp.deltas:
        sources.append(Source(name="Stockfish 16 classical evaluation terms"))
    note, note_cites = concept_note_for(cmp.deltas, king=_near_king(pos))
    sources += note_cites
    kept = [m for m in list(allowed) + list(missed) if m.theme in gated]
    best_label = _label(pos.fen, best.moves_uci[0])
    return Explanation(
        epd=pos.epd,
        fen=pos.fen,
        played=pos.move_label,
        best=best_label or None,
        best_line=best,
        refutation=refutation,
        motifs=kept,
        concepts=cmp.deltas,
        facts=cmp.facts,
        text=explanation_text(pos, best, refutation, allowed, missed, cmp, gated, depth, choice_min_drop),
        sources=sources,
        insight_id=pos.insight_id,
        kind=pos.kind,
        drop=pos.drop,
        time_class=games.single_format(pos),
        color=pos.color,
        verdict=verdict_of(pos, best, refutation, choice_min_drop),
        game_url=pos.url,
        games=list(pos.games),
        repeats=pos.repeats,
        diagram=build_diagram(games, pos, best, shown, allowed, missed, gated, close, cmp),
        chart=concept_chart(pos, best_label, cmp.deltas),
        drill_themes=list(dict.fromkeys(m.theme for m in kept)),
        concept_note=note,
    )


def explain_all(
    ctx: "AnalysisContext",
    positions: list[CriticalPosition],
    lines: dict[tuple[str, str], "DeepResult"],
    cfg: CoachConfig,
    notes: list[str],
) -> list[Explanation]:
    """One Explanation per position with lines: motifs, concept deltas, facts, text, diagram and chart."""
    todo = [(p, lines[(p.epd, p.played_uci)]) for p in positions if (p.epd, p.played_uci) in lines]
    if not todo:
        return []
    gated = frozenset(getattr(motifs, "GATED_THEMES", frozenset()) or ())
    games = _Games(ctx)
    min_drop = openings.Thresholds.from_ctx(ctx).engine_min_drop
    tool = _Concepts(sf.find_stockfish(cfg.stockfish))
    out: list[Explanation] = []
    failed = 0
    try:
        for pos, result in todo:
            try:
                explanation = explain_position(games, pos, result, tool, gated, min_drop)
            except Exception as exc:  # noqa: BLE001 — one odd position must not cost the others their explanation
                failed += 1
                log.warning("could not explain %s %s: %s", pos.move_label, pos.fen, exc)
                continue
            if explanation is not None:
                out.append(explanation)
    finally:
        tool.close()
    if tool.eval.note:
        notes.append(tool.eval.note)
    elif tool.eval.broken:
        notes.append("Concept terms skipped: Stockfish's eval command failed, so the explanations use board facts "
                     "only.")
    elif tool.eval.path is None:
        notes.append(f"Concept terms skipped: {concepts.NEED_SF16}, and no Stockfish was found.")
    if failed:
        notes.append(f"{failed} positions could not be explained.")
    return out
