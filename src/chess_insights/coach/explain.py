"""Turn lines, motifs and concepts into a short explanation and a board (C1). Owner: coach-core.

For each critical position with engine lines (``deep.analyse_positions``):

* motifs along both lines (``motifs.detect_line``): the refutation's patterns carried out by your opponent (what
  your move allowed) and the best line's patterns carried out by you (what you missed). A motif is named only when
  its detector passed the precision gate (``motifs.GATED_THEMES``, read at call time); otherwise the text says
  "a tactic" and the board marks nothing;
* concept deltas and board facts where the two lines have played out (``concepts``);
* at most three plain sentences: what the refutation does, what it wins (material or a concept), and what to check
  next time; evaluations in pawns;
* the board before your move (your move red, the better move green, your side at the bottom) with two strips of
  small boards: the refutation after your move and the better line;
* a chart of the concept differences.

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
from ..analysis.mistakes import format_line
from ..models import Chart, ConceptDelta, CriticalPosition, Diagram, Explanation, Line, Mark, Motif, Source, Strip
from . import concepts, motifs
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
TEXT_PLIES = 4  # moves of a line quoted in the text

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


# --------------------------------------------------------------------------- small helpers
def pawns(cp: float) -> str:
    """"+0.40" / "−1.52" (pawns, a real minus sign), as the report writes evaluations."""
    return f"{cp / 100:+.2f}".replace("-", "−")


def verdict(line: Optional[Line], color: str, short: bool = False) -> str:
    """The engine's verdict on a line from your side: "−1.52 for you", "mate in 3 for White", "mate in 2 for you";
    ``short`` leaves "for you" off a number ("−0.39")."""
    if line is None:
        return "unclear"
    if line.mate_end is not None:
        n = abs(line.mate_end)
        if line.mate_end > 0:
            return f"mate in {n} for you" if n else "checkmate for you"
        winner = "Black" if color == "white" else "White"
        return f"mate in {n} for {winner}" if n else f"checkmate for {winner}"
    return pawns(line.cp_end or 0) + ("" if short else " for you")


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


def _material_words(n: int) -> str:
    return "a pawn" if n == 1 else f"{n} pawns' worth of material"


def _motif_name(m: Motif, gated: frozenset) -> str:
    return MOTIF_WORDS.get(m.theme, (m.theme, ""))[0] if m.theme in gated else "a tactic"


class _Games:
    """What the explanations look up about games: the format by URL and the start position by id."""

    def __init__(self, ctx: "AnalysisContext") -> None:
        self.time_class = {g.url: g.time_class for g in ctx.games if g.url}
        self.start = {g.game_id: g.initial_fen for g in ctx.games}

    def format_mix(self, urls: Sequence[str]) -> dict[str, int]:
        """Games per time class behind a position, in the report's format order."""
        return visuals.ordered_formats(Counter(self.time_class[u] for u in urls if u in self.time_class))


# --------------------------------------------------------------------------- motifs and concepts
def detect(line: Optional[Line], role: str) -> list[Motif]:
    """``motifs.detect_line`` made safe: any list is accepted, odd entries dropped, a failure is no motif."""
    if line is None or not line.moves_uci:
        return []
    try:
        found = motifs.detect_line(line, role) or []
    except Exception as exc:  # noqa: BLE001 — a detector bug must not cost the position its explanation
        log.debug("motif detection failed on %s: %s", line.fen, exc)
        return []
    return [m for m in found if isinstance(m, Motif)]


def split_motifs(best: Optional[Line], refutation: Optional[Line]) -> tuple[list[Motif], list[Motif]]:
    """(what your move allowed: the opponent's patterns along the refutation, what you missed: yours along the
    best line)."""
    allowed = [m for m in detect(refutation, "refutation") if m.side == "opponent"]
    missed = [m for m in detect(best, "best") if m.side == "you"]
    return allowed, missed


@dataclass
class Comparison:
    """What the two lines leave behind, read at their comparison points (``concepts.comparison_point``)."""

    deltas: list[ConceptDelta] = field(default_factory=list)  # refutation end minus best end, your side
    facts: list[str] = field(default_factory=list)  # what is worse for you at the refutation end, in words
    # Material (pawn 1, knight and bishop 3, rook 5, queen 9) the refutation takes from you: against the position
    # before your move AND against the better line (the smaller of the two, so a pawn you lose either way is not
    # blamed on your move) ...
    material_lost: int = 0
    # ... and what the better line wins for you, against the position before your move and against the refutation
    # (54.gxh6 takes a rook that 54.d6 leaves alone).
    material_missed: int = 0


class _Concepts:
    """Concept deltas and board facts at the comparison points of two lines, with one Stockfish 16 process."""

    def __init__(self, path: Optional[str]) -> None:
        self.eval = concepts.ClassicalEval(path)

    def close(self) -> None:
        self.eval.close()

    def compare(self, fen: str, best: Line, refutation: Line, color: str) -> Comparison:
        b_end = concepts.comparison_point(fen, best.moves_uci)
        r_end = concepts.comparison_point(fen, refutation.moves_uci)
        if b_end is None or r_end is None:
            return Comparison()
        me = chess.WHITE if color == "white" else chess.BLACK
        b_facts, r_facts = concepts.board_facts(b_end, color), concepts.board_facts(r_end, color)  # type: ignore[arg-type]
        before = concepts.material(chess.Board(fen), me)
        out = Comparison(
            facts=concepts.fact_differences(b_facts, r_facts),
            material_lost=max(0, min(before, b_facts.material) - r_facts.material),
            material_missed=max(0, b_facts.material - max(before, r_facts.material)),
        )
        b_table = self.eval.table(b_end.fen())
        r_table = self.eval.table(r_end.fen()) if b_table is not None else None
        if b_table is not None and r_table is not None:
            out.deltas = concepts.concept_deltas(
                b_table, r_table, color, concepts.material_phase(b_end), concepts.material_phase(r_end),  # type: ignore[arg-type]
                material_level=b_facts.material == r_facts.material,
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


def _what_it_wins(cmp: Comparison, gap: float, best_label: str = "") -> str:
    """"you lose a pawn" / "5.Nxe5 would have won 3 pawns' worth of material" (when the engine's verdict backs it),
    else the two concepts the refutation leaves you worst on; plus the first board fact."""
    parts = []
    lost, missed = _material_backed(cmp, gap)
    if lost or missed:
        material = []
        if lost:
            material.append(f"you lose {_material_words(lost)}")
        if missed:
            material.append(f"{best_label or 'the better move'} would have won {_material_words(missed)}")
        parts.append(", and ".join(material))
    else:
        worse = [c for c in cmp.deltas if c.value < 0 and c.term != "Material"][:2]
        if worse:
            text = f"a few moves later you stand {abs(worse[0].value):.2f} pawns worse on {worse[0].label}"
            if len(worse) > 1:
                text += f" and {abs(worse[1].value):.2f} on {worse[1].label}"
            parts.append(text)
    if cmp.facts and not (lost and missed):  # a third item would make the sentence too long
        parts.append(cmp.facts[0])
    if len(parts) == 2:
        return f"{parts[0]}, and {parts[1]}"
    return parts[0] if parts else ""


def _check_next_time(allowed: Sequence[Motif], missed: Sequence[Motif], gated: frozenset, cmp: Comparison,
                     gap: float, mated: bool, mating: bool = False) -> str:
    """What to look at before a move like this one, from the motif, else the material, else the worst concept."""
    if mated:
        return MOTIF_WORDS["mateIn1"][1]
    named = [m for m in allowed if m.theme in gated and m.theme in MOTIF_WORDS]
    if named:
        return MOTIF_WORDS[named[0].theme][1]
    if allowed:
        return GENERIC_CHECK
    if missed or mating:
        return MISSED_CHECK
    lost, gained = _material_backed(cmp, gap)
    if lost or gained:
        return CONCEPT_CHECKS["material"] if lost else MISSED_CHECK
    worse = [c for c in cmp.deltas if c.value < 0 and c.label in CONCEPT_CHECKS and c.term != "Material"]
    if worse:
        return CONCEPT_CHECKS[worse[0].label]
    return GENERIC_CHECK


def explanation_text(
    pos: CriticalPosition,
    best: Line,
    refutation: Line,
    allowed: Sequence[Motif],
    missed: Sequence[Motif],
    cmp: Comparison,
    gated: frozenset,
    depth: Optional[int] = None,
) -> str:
    """At most three sentences: what the refutation does, what it wins, what to check next time.

    When the deeper search finds your move as good as its own (the same move) or close to it (``is_close``: not even
    an inaccuracy), the text says so instead of explaining an error that is not one.
    """
    played = pos.move_label
    best_label = _label(pos.fen, best.moves_uci[0]) if best.moves_uci else ""
    opponent = "Black" if pos.color == "white" else "White"
    same = bool(best.moves_uci) and best.moves_uci[0] == pos.played_uci
    refutation = shown_line(best, refutation)
    ref_v, best_v = verdict(refutation, pos.color), verdict(best, pos.color, short=True)
    gap = _gap(best, refutation)
    answer = _line_text(refutation, 1, TEXT_PLIES)
    close = not same and is_close(best, refutation)
    at = f" at depth {depth}" if depth else ""

    if pos.kind == "choice":
        if same:
            first = f"Stockfish agrees with your {played} ({ref_v})"
        elif close:
            first = f"Your {played} ({ref_v}) is close to Stockfish's first choice, {best_label} ({best_v})"
        else:
            first = f"Stockfish prefers {best_label} ({verdict(best, pos.color)}) to your {played} ({ref_v})"
        first += f"; its main line after {played} is {answer}." if answer else "."
        wins = _what_it_wins(cmp, gap, best_label) if not (same or close) else ""
        return first + (f" Compared with {best_label}, {wins}." if wins else "")

    if same:
        return (f"A deeper look{at} makes your {played} Stockfish's own first choice ({ref_v}), so the verdict of the "
                f"quicker game analysis on it does not hold up" + (f"; the main line is {answer}." if answer else "."))
    if close:
        return (f"A deeper look{at} finds your {played} ({ref_v}) close to Stockfish's first choice, {best_label} "
                f"({best_v}), so the quicker game analysis was too harsh on it"
                + (f"; the main line after {played} is {answer}." if answer else "."))

    mated = refutation.mate_end is not None and refutation.mate_end < 0
    mating = best.mate_end is not None and best.mate_end > 0
    if mated:
        first = f"After {played}, {opponent} mates in {abs(refutation.mate_end)}" + (f": {answer}." if answer else ".")
    elif allowed:
        first = f"After {played}, {opponent} has {_motif_name(allowed[0], gated)}" + (f": {answer}." if answer else ".")
    elif missed:
        first = f"You had {_motif_name(missed[0], gated)}: {_line_text(best, 0, TEXT_PLIES + 1)}."
    elif answer:
        first = f"After {played}, Stockfish's answer is {answer}."
    else:
        first = f"Stockfish prefers {best_label} to {played}."
    if mated:  # the mate is the whole story: no material or concepts on the way to it
        second = f"Stockfish prefers {best_label} ({verdict(best, pos.color)})."
    else:
        wins = "" if mating else _what_it_wins(cmp, gap, best_label)
        second = f"Stockfish rates the line {ref_v}, against {best_v} after {best_label}" + (
            f": {wins}." if wins else ".")
    check = _check_next_time(allowed, missed, gated, cmp, gap, mated, mating)
    return f"{first} {second} Next time, {check}."


# --------------------------------------------------------------------------- the pictures
def _marks(ms: Sequence[Motif], gated: frozenset) -> list[Mark]:
    out: list[Mark] = []
    for m in ms:
        if m.theme not in gated:
            continue
        for i, sq in enumerate(m.squares or []):
            if sq in chess.SQUARE_NAMES and all(x.square != sq for x in out):
                out.append(Mark(sq, "attacker" if i == 0 else "target"))
    return out


def _frame_marks(ms: Sequence[Motif], gated: frozenset, offset: int, frames: int) -> list[list[Mark]]:
    """Marks per strip frame: a motif at line ply ``p`` shows on frame ``p - offset``."""
    out: list[list[Mark]] = [[] for _ in range(frames)]
    for m in ms:
        i = m.ply - offset
        if m.theme in gated and 0 <= i < frames:
            out[i].extend(_marks([m], gated))
    return out


def _strip(title: str, fen: str, moves: Sequence[str], last_caption: str, marks: list[list[Mark]]) -> Strip:
    n = min(STRIP_FRAMES, len(moves))
    captions = [""] * (n - 1) + [last_caption] if n else []
    return visuals.line_strip(title, fen, list(moves), max_frames=STRIP_FRAMES, captions=captions, marks=marks)


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
                  allowed: Sequence[Motif], missed: Sequence[Motif], gated: frozenset, close: bool = False) -> Diagram:
    """The board before your move (your move red, the better one green, your side at the bottom, motif squares
    marked, the move before it highlighted) with two strips: the refutation after your move and the better line,
    each ending on the verdict. ``close``: the better line is only slightly better (its strip is "Stockfish's")."""
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
        time_class=pos.time_class,
        last_move=_last_move(games.start.get(pos.game_id), pos.moves_before, pos.epd),
        marks=_marks(list(allowed) + list(missed), gated),
    )
    after = _after(pos.fen, pos.played_uci)
    ref_moves = refutation.moves_uci[1:]
    strips = []
    if ref_moves:
        n = min(STRIP_FRAMES, len(ref_moves))
        strips.append(_strip(f"After your {played}", after, ref_moves, verdict(refutation, pos.color),
                             _frame_marks(allowed, gated, 1, n)))
    if best.moves_uci and not same:
        n = min(STRIP_FRAMES, len(best.moves_uci))
        title = f"Stockfish's {best_label}" if pos.kind == "choice" or close else f"The better {best_label}"
        strips.append(_strip(title, pos.fen, best.moves_uci, verdict(best, pos.color),
                             _frame_marks(missed, gated, 0, n)))
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
              f"(at the first quiet position): the line after your {pos.move_label} minus the line after "
              f"{best_label}, from your side, in pawns. Below zero: worse for you. Only differences of at least "
              f"{concepts.MIN_DELTA:.2f} are shown."),
    )


# --------------------------------------------------------------------------- one position
def explain_position(
    games: _Games,
    pos: CriticalPosition,
    result: "DeepResult",
    concept_tool: Optional[_Concepts],
    gated: frozenset,
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
    shown = shown_line(best, refutation)  # the line after your move in the text and the strip
    # No motifs, drill themes or concept comparison for a move the deeper search does not call an error, and no
    # concepts when a forced mate decides a line (material and structure on the way to it say nothing).
    allowed, missed = ([], []) if same or close else split_motifs(best, refutation)
    cmp = Comparison()
    if concept_tool is not None and not (same or close or _mate_line(best, refutation)):
        cmp = concept_tool.compare(pos.fen, best, refutation, pos.color)
    depths = [x.depth for x in (best, refutation) if x.depth]
    depth = min(depths) if depths else result.depth
    engine_name = result.engine or "Stockfish"
    sources = [Source(name=f"{engine_name}, depth {depth}" if depth else engine_name)]
    if cmp.deltas:
        sources.append(Source(name="Stockfish 16 classical evaluation terms"))
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
        text=explanation_text(pos, best, refutation, allowed, missed, cmp, gated, depth),
        sources=sources,
        insight_id=pos.insight_id,
        kind=pos.kind,
        drop=pos.drop,
        time_class=pos.time_class,
        color=pos.color,
        game_url=pos.url,
        games=list(pos.games),
        repeats=pos.repeats,
        diagram=build_diagram(games, pos, best, shown, allowed, missed, gated, close),
        chart=concept_chart(pos, best_label, cmp.deltas),
        drill_themes=list(dict.fromkeys(m.theme for m in kept)),
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
    tool = _Concepts(sf.find_stockfish(cfg.stockfish))
    out: list[Explanation] = []
    failed = 0
    try:
        for pos, result in todo:
            try:
                explanation = explain_position(games, pos, result, tool, gated)
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
