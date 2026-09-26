"""Positional concepts at the ends of the two lines: Stockfish 16's classical eval terms and board facts (C1).
Owner: coach-core.

Stockfish 16 is the last release that prints its classical evaluation term by term (``eval``: material, pawns,
mobility, king safety ... for middlegame and endgame, from White's side). The terms read right after a move are
misleading: right after 5...e5 Stockfish counts the pawn's attack on the d4 knight as a plus for Black. So the
concepts compare the two lines where they have played out: the refutation of your move against the engine's best
line, each read at ``COMPARE_PLIES`` plies from the position before your move (or the nearest quiet position:
not in check, nothing left to recapture). The engine's PV runs 15 to 25 plies, but its far end is speculative, and
the three habit positions of the coaching plan read true at this distance (5...e5: piece activity, king safety and
pawn structure all worse than after 5...a6).

Board facts (python-chess, any Stockfish or none) describe the same two positions in words: the bishop pair,
castling, a king left in the centre, isolated, doubled and backward pawns, and holes.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional, Sequence

import chess

from ..models import Color, ConceptDelta

log = logging.getLogger(__name__)

TERMS = ("Material", "Imbalance", "Pawns", "Knights", "Bishops", "Rooks", "Queens", "Mobility", "King safety",
         "Threats", "Passed", "Space", "Winnable")
LABELS = {
    "Material": "material", "Imbalance": "piece balance", "Pawns": "pawn structure", "Knights": "knights",
    "Bishops": "bishops", "Rooks": "rooks", "Queens": "queen", "Mobility": "piece activity",
    "King safety": "king safety", "Threats": "threats", "Passed": "passed pawns", "Space": "space",
    "Winnable": "winning chances",
}
MIN_DELTA = 0.15  # pawns
COMPARE_PLIES = 6  # where the two lines are compared: this many plies from the position before your move ...
QUIET_EXTRA = 4  # ... or up to this many plies later, at the first quiet position
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
MAX_PHASE = 24  # all the pieces of the start position
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
CENTRE_AFTER_MOVE = 12  # a king still on the d- or e-file after this move is "in the centre"
EVAL_TIMEOUT = 10.0  # seconds for one ``eval`` answer before the helper gives up
NEED_SF16 = "concepts need Stockfish 16"

Table = dict[str, tuple[float, float]]

_ROW = re.compile(r"^\|\s*(?P<term>[A-Za-z ]+?)\s*\|(?P<white>[^|]*)\|(?P<black>[^|]*)\|(?P<total>[^|]*)\|\s*$")


# --------------------------------------------------------------------------- Stockfish 16's term table
def _cell(text: str) -> Optional[float]:
    text = text.strip()
    if not text or set(text) <= {"-"}:  # "----": no value for this side
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_eval_table(text: str) -> Optional[dict[str, tuple[float, float]]]:
    """Stockfish 16 ``eval`` output -> {term: (MG, EG)} totals from White's side; None without the table.

    Reads the Total column of "Contributing terms for the classical eval" (pawns). A "----" cell counts as 0 when
    its partner has a value; a term with no value at all is left out. In check Stockfish prints no table.
    """
    start = text.find("Contributing terms for the classical eval")
    if start < 0:
        return None
    out: Table = {}
    for raw in text[start:].splitlines()[1:]:
        line = raw.strip()
        if not line:
            if out:
                break
            continue
        m = _ROW.match(line)
        if m is None:
            continue
        term = m.group("term").strip()
        if term == "Total":
            break
        if term not in TERMS:
            continue
        cells = m.group("total").split()
        if len(cells) != 2:
            continue
        mg, eg = _cell(cells[0]), _cell(cells[1])
        if mg is None and eg is None:
            continue
        out[term] = (mg or 0.0, eg or 0.0)
    return out or None


def material_phase(board: chess.Board) -> int:
    """0 (bare kings and pawns) .. 24 (every piece): knights and bishops 1, rooks 2, queens 4."""
    phase = sum(w * chess.popcount(board.pieces_mask(pt, chess.WHITE) | board.pieces_mask(pt, chess.BLACK))
                for pt, w in PHASE_WEIGHTS.items())
    return min(MAX_PHASE, phase)


def blend(mg: float, eg: float, phase: int) -> float:
    """Middlegame and endgame values mixed by the material phase, as a tapered evaluation does."""
    return (mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE


def concept_deltas(
    best: Table, refutation: Table, color: Color, best_phase: int, refutation_phase: int,
    *, material_level: bool = False,
) -> list[ConceptDelta]:
    """Refutation-line end minus best-line end, from ``color``'s side, per term; |value| >= ``MIN_DELTA`` only.

    Each end is blended with its own material phase (what the term is worth in that position); ``mg`` / ``eg`` are
    the raw differences. ``material_level``: both ends have the same material, so a Material difference is piece
    placement (Stockfish's Material term includes its piece-square tables). Worst for you first.
    """
    sign = 1.0 if color == "white" else -1.0
    out = []
    for term in TERMS:
        if term not in best or term not in refutation:
            continue
        (bmg, beg), (rmg, reg) = best[term], refutation[term]
        value = sign * (blend(rmg, reg, refutation_phase) - blend(bmg, beg, best_phase))
        if abs(value) < MIN_DELTA:
            continue
        label = "piece placement" if term == "Material" and material_level else LABELS[term]
        out.append(ConceptDelta(term=term, value=round(value, 2), mg=round(sign * (rmg - bmg), 2),
                                eg=round(sign * (reg - beg), 2), label=label, source="stockfish16"))
    return sorted(out, key=lambda c: (c.value, c.term))


# --------------------------------------------------------------------------- where to compare the lines
def _quiet(board: chess.Board, last: Optional[chess.Move]) -> bool:
    """Not in check, and nothing can take on the square the last move went to (no exchange half done)."""
    if board.is_check():
        return False
    if last is None:
        return True
    return not any(m.to_square == last.to_square for m in board.generate_legal_captures())


def comparison_point(fen: str, moves_uci: Sequence[str], plies: int = COMPARE_PLIES) -> Optional[chess.Board]:
    """The position of a line where concepts are read: after ``plies`` moves, or the first quiet one up to
    ``QUIET_EXTRA`` plies later, or else the last quiet one before it (never the start). None if there is none."""
    board = chess.Board(fen)
    boards: list[tuple[chess.Board, Optional[chess.Move]]] = []
    for uci in moves_uci[: plies + QUIET_EXTRA]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        board.push(move)
        boards.append((board.copy(stack=False), move))
    if not boards:
        return None
    target = min(plies, len(boards)) - 1
    for i in list(range(target, len(boards))) + list(range(target - 1, -1, -1)):
        b, last = boards[i]
        if _quiet(b, last):
            return b
    return None


# --------------------------------------------------------------------------- Stockfish 16 "eval" helper
class ClassicalEval:
    """Runs ``eval`` on FENs with one Stockfish process (started on first use; ``close`` or ``with`` ends it).

    ``table(fen)`` is None in check, and always None once a position not in check came back without the table
    (a Stockfish newer than 16): then ``supported`` is False and ``note`` says so.
    """

    def __init__(self, path: Optional[str | Sequence[str]], timeout: float = EVAL_TIMEOUT) -> None:
        self.path = path  # the Stockfish binary (or a command line: [program, args ...])
        self.timeout = timeout
        self.name = ""
        self.supported: Optional[bool] = None if path else False
        self.broken = False
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @property
    def note(self) -> str:
        if self.supported is False and self.path and not self.broken:
            return (f"{NEED_SF16[:1].upper()}{NEED_SF16[1:]} (the last version that reports its classical eval "
                    f"terms; this is {self.name or 'another version'}), so the explanations use board facts only.")
        return ""

    def __enter__(self) -> "ClassicalEval":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _start(self) -> subprocess.Popen:
        if self._proc is None:
            # Its own process group (like engine._popen), so Ctrl+C reaches only Python, which then closes it.
            extra: dict = {"start_new_session": True} if os.name != "nt" else {
                "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
            command = [self.path] if isinstance(self.path, str) else list(self.path or [])
            self._proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True, bufsize=1, **extra)
            for line in self._ask("uci", "uciok"):
                if line.startswith("id name "):
                    self.name = line[len("id name "):].strip()
        return self._proc

    def _ask(self, commands: str, until: str) -> list[str]:
        """Send ``commands`` and read lines up to ``until``; kills the process when it takes over ``timeout``."""
        proc = self._proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        timer = threading.Timer(self.timeout, proc.kill)
        timer.daemon = True
        timer.start()
        try:
            proc.stdin.write(commands + "\n")
            proc.stdin.flush()
            lines = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("Stockfish stopped answering")
                line = line.rstrip("\r\n")
                if line.strip() == until:
                    return lines
                lines.append(line)
        finally:
            timer.cancel()

    def table(self, fen: str) -> Optional[Table]:
        """{term: (MG, EG)} from White's side for ``fen``; None in check, without Stockfish 16, or on failure."""
        if self.supported is False or self.broken or not self.path:
            return None
        with self._lock:
            try:
                self._start()
                text = "\n".join(self._ask(f"position fen {fen}\neval\nisready", "readyok"))
            except (OSError, RuntimeError, ValueError) as exc:
                log.warning("Stockfish eval failed: %s", exc)
                self.broken = True
                self.close()
                return None
        parsed = parse_eval_table(text)
        if parsed is None and "in check" not in text and self.supported is None:
            self.supported = False
            self.close()
        elif parsed is not None:
            self.supported = True
        return parsed

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001 — closing must never raise
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- python-chess board facts
@dataclass
class BoardFacts:
    """Structural facts about one position from one side."""

    bishop_pair: bool
    opp_bishop_pair: bool
    can_castle: bool  # still has a castling right
    castled: bool  # the king stands where castling puts it
    king_in_centre: bool  # after move CENTRE_AFTER_MOVE, on the d- or e-file, with the opponent's queen on
    material: int  # yours minus your opponent's (pawn 1, knight and bishop 3, rook 5, queen 9)
    isolated: list[str] = field(default_factory=list)
    doubled_files: list[str] = field(default_factory=list)
    backward: list[str] = field(default_factory=list)
    holes: list[str] = field(default_factory=list)


def _rel_rank(square: int, color: chess.Color) -> int:
    """1..8 from ``color``'s side."""
    rank = chess.square_rank(square)
    return rank + 1 if color == chess.WHITE else 8 - rank


def material(board: chess.Board, color: chess.Color) -> int:
    return sum(v * (len(board.pieces(pt, color)) - len(board.pieces(pt, not color))) for pt, v in PIECE_VALUES.items())


def board_facts(board: chess.Board, color: Color) -> BoardFacts:
    me = chess.WHITE if color == "white" else chess.BLACK
    pawns = board.pieces(chess.PAWN, me)
    files: dict[int, list[int]] = {}
    for sq in pawns:
        files.setdefault(chess.square_file(sq), []).append(sq)

    def neighbours(f: int) -> list[int]:
        return [sq for nf in (f - 1, f + 1) for sq in files.get(nf, [])]

    isolated = sorted(chess.square_name(sq) for sq in pawns if not neighbours(chess.square_file(sq)))
    doubled = sorted(f"{chess.FILE_NAMES[f]}" for f, sqs in files.items() if len(sqs) > 1)
    backward = []
    for sq in pawns:
        f, r = chess.square_file(sq), _rel_rank(sq, me)
        near = neighbours(f)
        if not near or any(_rel_rank(n, me) <= r for n in near) or r >= 7:
            continue
        stop = sq + (8 if me == chess.WHITE else -8)
        if board.attackers_mask(not me, stop) & board.pieces_mask(chess.PAWN, not me):
            backward.append(chess.square_name(sq))
    holes = []
    for f in range(1, 7):  # b..g
        for rel in (3, 4):
            sq = chess.square(f, rel - 1 if me == chess.WHITE else 8 - rel)
            if sq in pawns:
                continue
            if not any(_rel_rank(n, me) < rel for n in neighbours(f)):
                holes.append(chess.square_name(sq))
    king = board.king(me)
    home = chess.BB_RANK_1 if me == chess.WHITE else chess.BB_RANK_8
    castled = king is not None and bool(chess.BB_SQUARES[king] & home) and chess.square_file(king) in (1, 2, 6, 7)
    centre = (
        king is not None
        and board.fullmove_number > CENTRE_AFTER_MOVE
        and chess.square_file(king) in (3, 4)
        and bool(board.pieces(chess.QUEEN, not me))
    )
    return BoardFacts(
        bishop_pair=len(board.pieces(chess.BISHOP, me)) >= 2,
        opp_bishop_pair=len(board.pieces(chess.BISHOP, not me)) >= 2,
        can_castle=board.has_castling_rights(me),
        castled=castled,
        king_in_centre=centre,
        material=material(board, me),
        isolated=isolated,
        doubled_files=doubled,
        backward=sorted(backward),
        holes=sorted(holes),
    )


def _squares(names: Sequence[str]) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def fact_differences(best: BoardFacts, refutation: BoardFacts, limit: int = 3) -> list[str]:
    """What is worse for you at the end of the refutation than at the end of the best line, in words.

    Only differences, and only those that go against you: "you lose the bishop pair", "your king stays in the
    centre", "d5 becomes a hole in your camp". At most ``limit``, the most important first.
    """
    out = []
    if best.bishop_pair and not refutation.bishop_pair:
        out.append("you lose the bishop pair")
    elif refutation.opp_bishop_pair and not best.opp_bishop_pair:
        out.append("your opponent gets the bishop pair")
    if (best.can_castle or best.castled) and not (refutation.can_castle or refutation.castled):
        out.append("you lose the right to castle")
    if refutation.king_in_centre and not best.king_in_centre:
        out.append("your king stays in the centre")
    new_holes = [h for h in refutation.holes if h not in best.holes]
    if len(refutation.holes) > len(best.holes) and new_holes:
        out.append(f"{_squares(new_holes)} {'becomes a hole' if len(new_holes) == 1 else 'become holes'} in your "
                   f"camp (no pawn of yours can cover {'it' if len(new_holes) == 1 else 'them'})")
    if len(refutation.isolated) > len(best.isolated):
        new = [s for s in refutation.isolated if s not in best.isolated] or refutation.isolated
        out.append(f"you get {'an isolated pawn' if len(new) == 1 else 'isolated pawns'} on {_squares(new)}")
    if len(refutation.doubled_files) > len(best.doubled_files):
        new = [f for f in refutation.doubled_files if f not in best.doubled_files] or refutation.doubled_files
        out.append(f"you get doubled pawns on the {_squares(new)}-file" if len(new) == 1
                   else f"you get doubled pawns on the {_squares(new)} files")
    if len(refutation.backward) > len(best.backward):
        new = [s for s in refutation.backward if s not in best.backward] or refutation.backward
        out.append(f"your pawn on {new[0]} is left backward" if len(new) == 1
                   else f"your pawns on {_squares(new)} are left backward")
    return out[:limit]
