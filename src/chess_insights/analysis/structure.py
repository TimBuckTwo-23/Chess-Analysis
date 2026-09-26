"""Development & king safety: your first ten moves against your opponent's in the same game.

Four habits, measured for both players in every standard game:

* castled by move 10 (each side's own tenth move);
* moved the queen in the first 8 moves other than to capture (recapturing with the queen is often forced);
* knights and bishops that had left their starting squares by move 10, of 4;
* pawn moves among the first 10 moves.

Every habit is compared with your opponent's in the same game, so the format, the clock, the rating gap and
the result are the same for both sides. Games are the unit of the tests (a paired test on the per-game
difference, :func:`paired_share_test`), with White and Black games weighted equally because White moves
first. The four tests are one family: Benjamini-Hochberg adjusted, judged at ``stats.STRICT_ALPHA`` under the
project-wide rule (``stats.significance``), and a finding also needs a gap big enough to matter
(``Thresholds``). Findings are associations ("you castle later than your opponents"), never causes.

The opening is the same game for both sides but not the same side of it: you choose your side of an asymmetric
line (in the Caro-Kann Black usually castles later than White, in the Ruy Lopez earlier), so your repertoire
alone can open a gap. The fold rule (as the colour claim's in ``results``) guards against it: for each would-be
finding, the (colour, opening family) group that contributes most to the gap is left out and the habit tested
again (the family's other p-values unchanged, BH again); when the finding no longer passes, it is an observation
worded "Mostly your <family> games: ...", with that group against your other games, never a strength or weakness.

A game counts for a habit only when both players made the moves it looks at (20 plies for the habits
measured by move 10, 16 for the queen), so a game that ended on move 7 is not "not castled" for either side;
this also leaves out games under 4 plies. Chess960 and games from a set-up position are left out: their
pieces don't start on the usual squares.

Every finding carries the formats behind it, a chart of you against your opponents by format, and a board from
one of its games (:func:`habit_diagram`): your side of it for a weakness, your opponent's for a strength.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import chess

from ..context import AnalysisContext
from ..models import Chart, Diagram, Game, Insight, Kpi, Mark, ModuleResult, Strip, Table
from ..stats import (
    STRICT_ALPHA,
    MeanTest,
    bh_adjust,
    clamp,
    normal_cdf,
    pct,
    shrink_effect,
    significance,
)
from ..visuals import (
    FORMAT_NAMES,
    MIN_FORMAT_GAMES,
    comparison_chart,
    format_counts,
    line_strip,
    move_label,
    position_diagram,
)
from .results import MAX_EXAMPLES

KEY = "structure"
TITLE = "Development & king safety"

MIN_PLIES = 4  # shorter games are aborted starts, not skill (the window rule below leaves them out anyway)
MINOR_HOMES = {
    chess.WHITE: (chess.B1, chess.G1, chess.C1, chess.F1),
    chess.BLACK: (chess.B8, chess.G8, chess.C8, chess.F8),
}
COLOURS = {"white": chess.WHITE, "black": chess.BLACK}
OTHER_FAMILY = "unnamed opening"  # games without an opening name form one group
FOLD_GROUPS = 2  # opening groups the fold rule leaves out, one after the other (a main line with each colour)
MIN_GROUP_GAMES = 5  # opening groups listed in a chart's table
MAX_GROUP_ROWS = 6


@dataclass(frozen=True)
class Thresholds:
    """Move windows, minimum sample sizes and effect sizes; override with ``ctx.options["structure.<name>"]``."""

    castle_by: int = 10  # "castled by move N"
    queen_by: int = 8  # "queen moved (not capturing) in the first N moves"
    develop_by: int = 10  # "knights and bishops out by move N"
    pawn_window: int = 10  # "pawn moves among the first N moves"
    min_games: int = 30  # games in which both players reached the window, before a habit is judged
    min_castle_gap: float = 0.10  # share of games (10 points)
    min_queen_gap: float = 0.08  # share of games
    min_minor_gap: float = 0.30  # knights and bishops out, per game
    min_pawn_gap: float = 0.50  # pawn moves, per game
    alpha: float = STRICT_ALPHA  # an exploratory family (stats policy)
    min_confidence: float = 0.5  # insights.MIN_CONFIDENCE

    @classmethod
    def from_ctx(cls, ctx: AnalysisContext) -> "Thresholds":
        return cls(**{f.name: ctx.opt(f"{KEY}.{f.name}", f.default) for f in dataclasses.fields(cls)})


# --------------------------------------------------------------------------- reading the first moves
@dataclass
class Side:
    """One player's first moves in one game."""

    castle_move: Optional[int] = None  # own move number on which the side castled (within the moves read)
    queen_ply: Optional[int] = None  # ply index of the side's first queen move that is not a capture
    queen_move: Optional[int] = None  # ... and its own move number
    minors: list[int] = field(default_factory=list)  # minors[k - 1]: knights and bishops moved by own move k
    pawns: list[int] = field(default_factory=list)  # pawns[k - 1]: pawn moves among own moves 1..k

    def minors_by(self, move: int) -> int:
        """Knights and bishops that had left their starting squares by the end of own move ``move``."""
        return self.minors[min(move, len(self.minors)) - 1] if self.minors and move > 0 else 0

    def pawns_by(self, move: int) -> int:
        return self.pawns[min(move, len(self.pawns)) - 1] if self.pawns and move > 0 else 0


@dataclass
class Opening:
    """The first moves of one standard game, for both players."""

    game: Game
    me: Side
    opp: Side
    ucis: list[str]  # the plies read, UCI

    @property
    def plies(self) -> int:
        return self.game.plies

    @property
    def my_color(self) -> chess.Color:
        return COLOURS[self.game.color]

    @property
    def group(self) -> tuple[str, str]:
        """(your colour, opening family): the repertoire group of the fold rule."""
        return self.game.color, self.game.opening_family or self.game.opening or OTHER_FAMILY

    def board_after(self, ply: int) -> chess.Board:
        """The position after ply index ``ply`` (the start position for -1)."""
        board = chess.Board()
        for uci in self.ucis[: ply + 1]:
            board.push(chess.Move.from_uci(uci))
        return board


def is_standard_start(g: Game) -> bool:
    """Standard chess from the usual start position (a set-up FEN equal to the start counts as the start)."""
    if (g.rules or "chess") != "chess":
        return False
    if not g.initial_fen:
        return True
    return g.initial_fen.split()[:4] == chess.STARTING_FEN.split()[:4]


def read_opening(g: Game, max_plies: int) -> Optional[Opening]:
    """Both players' first ``max_plies`` plies of a standard game; None when a move can't be read."""
    board = chess.Board()
    sides = {chess.WHITE: Side(), chess.BLACK: Side()}
    origin = {sq: sq for color in MINOR_HOMES for sq in MINOR_HOMES[color]}  # where each knight and bishop is now
    moved = {chess.WHITE: 0, chess.BLACK: 0}
    pawns = {chess.WHITE: 0, chess.BLACK: 0}
    ucis = []
    for ply, san in enumerate(g.moves_san[:max_plies]):
        color = board.turn
        number = ply // 2 + 1  # own move number (White moves first from the standard start)
        try:
            move = board.parse_san(san)
        except ValueError:
            return None
        side = sides[color]
        piece = board.piece_type_at(move.from_square)
        if board.is_castling(move) and side.castle_move is None:
            side.castle_move = number
        if piece == chess.QUEEN and not board.is_capture(move) and side.queen_ply is None:
            side.queen_ply, side.queen_move = ply, number
        if piece == chess.PAWN:
            pawns[color] += 1
        if move.to_square in origin and board.piece_at(move.to_square) is not None:
            del origin[move.to_square]  # a knight or bishop captured, at home or after moving
        if move.from_square in origin:
            if origin.pop(move.from_square) == move.from_square:  # its first move
                moved[color] += 1
            origin[move.to_square] = -1  # tracked as "already moved"
        side.minors.append(moved[color])
        side.pawns.append(pawns[color])
        ucis.append(move.uci())
        board.push(move)
    me = COLOURS[g.color]
    return Opening(game=g, me=sides[me], opp=sides[not me], ucis=ucis)


def pieces_at_home(board: chess.Board, color: chess.Color, ucis: Sequence[str] = ()) -> list[int]:
    """Starting squares whose own knight or bishop has not moved yet.

    ``ucis`` are the moves that led to ``board`` from the start: a square no move has left or entered still holds
    its original piece, so a knight that went out and came back home is not "still at home".
    """
    rank = 0 if color == chess.WHITE else 7
    homes = ((1, chess.KNIGHT), (2, chess.BISHOP), (5, chess.BISHOP), (6, chess.KNIGHT))  # (file, piece)
    touched = {sq for move in map(chess.Move.from_uci, ucis) for sq in (move.from_square, move.to_square)}
    return [
        chess.square(file, rank)
        for file, kind in homes
        if board.piece_at(chess.square(file, rank)) == chess.Piece(kind, color)
        and chess.square(file, rank) not in touched
    ]


def moved_pawns(board: chess.Board, color: chess.Color) -> list[int]:
    """Squares of ``color``'s pawns that have left their starting rank."""
    start = 1 if color == chess.WHITE else 6
    return [sq for sq in board.pieces(chess.PAWN, color) if chess.square_rank(sq) != start]


# --------------------------------------------------------------------------- habits
@dataclass(frozen=True)
class Habit:
    """One habit: how to count it per side, its window and which direction is the good one."""

    key: str
    label: str  # "Castled by move 10"
    by: int  # own moves both players must have made for a game to count
    scale: int  # the count's maximum: 1 for yes / no, 4 minor pieces, 10 moves
    good: int  # +1: more is better (castling, development); -1: less is better
    min_gap: float  # in the habit's own units (share of games, pieces, pawn moves)
    full: float  # a gap this big (own units) is severity 1
    strength: bool = True  # whether doing better than your opponents is claimed as a strength

    def count(self, s: Side) -> int:
        if self.key == "castled":
            return int(s.castle_move is not None and s.castle_move <= self.by)
        if self.key == "early_queen":
            return int(s.queen_move is not None and s.queen_move <= self.by)
        if self.key == "minors":
            return s.minors_by(self.by)
        return s.pawns_by(self.by)

    @property
    def binary(self) -> bool:
        return self.scale == 1

    @property
    def unit(self) -> int:
        """Own units per share: 1 for a yes / no habit (a share of games), else the count's maximum."""
        return 1 if self.binary else self.scale

    @property
    def value_format(self) -> str:
        return "pct" if self.binary else "float1"

    def number(self, x: Optional[float]) -> str:
        """A per-game average: '38%' for a yes / no habit, '2.9' for a count."""
        if x is None:
            return "n/a"
        return pct(x) if self.binary else f"{x:.1f}"

    def value_text(self, x: Optional[float]) -> str:
        """'38%' / '2.9 of 4' / '3.4 of 10'."""
        return self.number(x) if self.binary or x is None else f"{self.number(x)} of {self.scale}"


def habits(th: Thresholds) -> tuple[Habit, ...]:
    return (
        Habit("castled", f"Castled by move {th.castle_by}", th.castle_by, 1, +1, th.min_castle_gap, full=0.40),
        Habit("early_queen", f"Queen out early (moves 1–{th.queen_by})", th.queen_by, 1, -1, th.min_queen_gap,
              full=0.30),
        Habit("minors", f"Knights and bishops out by move {th.develop_by}", th.develop_by, 4, +1, th.min_minor_gap,
              full=1.0),
        Habit("pawns", f"Pawn moves in moves 1–{th.pawn_window}", th.pawn_window, th.pawn_window, -1,
              th.min_pawn_gap, full=2.0, strength=False),
    )


def paired_share_test(pairs: Sequence[tuple[int, int, str]], scale: int) -> MeanTest:
    """Test of your share minus your opponent's, with games as the unit and White and Black weighted equally.

    ``pairs`` holds (your count, your opponent's count, your colour) per game; a share is count / ``scale``
    (1 for a yes / no habit, 4 for minor pieces, 10 for pawn moves among ten moves). Each game gives one
    difference, so whatever the game brings to both sides (opening, clock, result) cancels. White moves first,
    which changes how soon either side castles or develops, so the differences are averaged within each colour
    first and the two colours count equally: a player with more White games is not judged on a colour mix.
    Within a colour, the variance never drops below that of two independent binomial shares (as in
    ``engine_stats.clustered_share_test``). The result's ``mean`` is the balanced difference in shares.
    """
    strata: dict[str, list[tuple[float, float]]] = {}
    for mine, theirs, color in pairs:
        strata.setdefault(color, []).append((mine / scale, theirs / scale))
    used = [xs for _, xs in sorted(strata.items()) if len(xs) >= 2]
    n = len(pairs)
    if not used:
        return MeanTest(n, 0.0, float("inf"), 0.0, 1.0)
    k = len(used)
    mean = variance = 0.0
    for xs in used:
        m = len(xs)
        diffs = [a - b for a, b in xs]
        d = sum(diffs) / m
        s2 = sum((x - d) ** 2 for x in diffs) / (m - 1)
        p = sum(a + b for a, b in xs) / (2 * m)
        floor = 2.0 * p * (1.0 - p) / scale
        mean += d / k
        variance += max(s2, floor) / m / k**2
    if variance <= 0.0:
        return MeanTest(n, mean, float("inf"), 0.0, 1.0)
    se = math.sqrt(variance)
    z = mean / se
    return MeanTest(n, mean, se, z, 2.0 * (1.0 - normal_cdf(abs(z))))


@dataclass
class HabitResult:
    habit: Habit
    games: list[Opening]  # games in which both players reached the window
    test: MeanTest
    p_adjusted: float = 1.0

    @property
    def n(self) -> int:
        return len(self.games)

    def average(self, mine: bool, games: Optional[Sequence[Opening]] = None) -> Optional[float]:
        """Your (or your opponents') average count per game: a share of games for a yes / no habit."""
        gs = self.games if games is None else games
        if not gs:
            return None
        return sum(self.habit.count(o.me if mine else o.opp) for o in gs) / len(gs)

    @property
    def gap(self) -> float:
        """Colour-balanced gap, you minus your opponents, in the habit's own units (share, pieces, moves)."""
        return self.test.mean * self.habit.unit

    def by_format(self) -> list[tuple[str, list[Opening]]]:
        """(time class, games) for every format with games, in the report's order."""
        counts = format_counts(o.game for o in self.games)
        return [(tc, [o for o in self.games if o.game.time_class == tc]) for tc in counts]

    def format_stats(self) -> dict[str, dict[str, Any]]:
        return {tc: {"n": len(gs), "you": self.average(True, gs), "opponents": self.average(False, gs)}
                for tc, gs in self.by_format()}

    def by_group(self) -> dict[tuple[str, str], list[Opening]]:
        """(your colour, opening family) -> games, most games first."""
        groups: dict[tuple[str, str], list[Opening]] = {}
        for o in self.games:
            groups.setdefault(o.group, []).append(o)
        return dict(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    def group_stats(self, limit: int = MAX_GROUP_ROWS) -> list[dict[str, Any]]:
        """The biggest opening groups (at least ``MIN_GROUP_GAMES`` games): colour, family, games, you, opponents."""
        return [
            {"colour": c, "family": f, "n": len(gs), "you": self.average(True, gs),
             "opponents": self.average(False, gs)}
            for (c, f), gs in list(self.by_group().items())[:limit]
            if len(gs) >= MIN_GROUP_GAMES
        ]

    def contributions(self) -> dict[tuple[str, str], float]:
        """Each opening group's part of the colour-balanced gap (``test.mean``, in shares): the sum of its games'
        differences over its colour's games, over the number of colours (as :func:`paired_share_test` weighs)."""
        colours: dict[str, int] = {}
        for o in self.games:
            colours[o.game.color] = colours.get(o.game.color, 0) + 1
        used = {c: n for c, n in colours.items() if n >= 2}
        out: dict[tuple[str, str], float] = {}
        for key, gs in self.by_group().items():
            if key[0] in used:
                diff = sum(self.habit.count(o.me) - self.habit.count(o.opp) for o in gs) / self.habit.scale
                out[key] = diff / used[key[0]] / len(used)
        return out


@dataclass
class Fold:
    """The fold rule's check of one would-be finding: the opening groups left out, one after the other, each the
    biggest part of the gap left, and the habit tested again without them (``holds``: the finding still passes)."""

    groups: list[tuple[str, str, list[Opening]]]  # (your colour, opening family, its games)
    without: HabitResult  # the habit on the other games, with its BH-adjusted p-value
    holds: bool
    too_few: bool = False  # the other games are fewer than the minimum: no habit can be told from the opening

    @property
    def games(self) -> list[Opening]:
        return [o for _, _, gs in self.groups for o in gs]

    def names(self) -> str:
        """'Caro-Kann Defense games with Black' / 'Sicilian Defense games with White and King's Indian Defense
        games with Black' / 'Caro-Kann Defense and French Defense games with Black'."""
        colours = {c for c, _, _ in self.groups}
        if len(colours) == 1:
            return f"{' and '.join(f for _, f, _ in self.groups)} games with {self.groups[0][0].capitalize()}"
        return " and ".join(f"{f} games with {c.capitalize()}" for c, f, _ in self.groups)


# --------------------------------------------------------------------------- charts
_UNIT_NOTE = {
    "castled": "share of games",
    "early_queen": "share of games (a queen move that is not a capture)",
    "minors": "of your 4 knights and bishops, per game",
    "pawns": "per game",
}


def habit_chart(r: HabitResult) -> Chart:
    """You vs your opponents on one habit, split by format (formats with too few games get no bar of their own)."""
    h = r.habit
    rows = [(FORMAT_NAMES.get(tc, tc), gs) for tc, gs in r.by_format() if len(gs) >= MIN_FORMAT_GAMES]
    if len(rows) != 1 or len(rows[0][1]) != r.n:
        rows.append(("All games", r.games))
    you = [r.average(True, gs) for _, gs in rows]
    them = [r.average(False, gs) for _, gs in rows]
    openings = [[_group_label(g["colour"], g["family"]), g["n"], g["you"], g["opponents"]] for g in r.group_stats()]
    table = Table(
        title=f"{h.label}: you vs your opponents",
        columns=["Games", "Count", "You", "Opponents"],
        rows=[[label, len(gs), a, b] for (label, gs), a, b in zip(rows, you, them, strict=True)] + openings,
        formats=["text", "int", h.value_format, h.value_format],
        note=f"Games in which both players made at least {h.by} moves; you and your opponent in the same games."
        + (" Then your most-played openings: the line you choose can set when each side castles and develops."
           if openings else ""),
    )
    return comparison_chart(
        f"{h.label}: you vs your opponents",
        [label for label, _ in rows],
        [("You", you), ("Opponents", them)],
        value_format=h.value_format,
        note=f"{h.label}, {_UNIT_NOTE[h.key]}. Formats with fewer than {MIN_FORMAT_GAMES} games have no bar of "
        "their own.",
        kind="bar",
        table=table,
    )


def _group_label(colour: str, family: str) -> str:
    """'Caro-Kann Defense (Black)'."""
    return f"{family} ({colour.capitalize()})"


def fold_chart(r: HabitResult, fold: Fold) -> Chart:
    """The opening groups that make most of the gap against your other games and all games: you vs your
    opponents."""
    h = r.habit
    rows = [(_group_label(c, f), gs) for c, f, gs in fold.groups]
    rows += [("Your other games", fold.without.games), ("All games", r.games)]
    you = [r.average(True, gs) for _, gs in rows]
    them = [r.average(False, gs) for _, gs in rows]
    families = " and ".join(f for _, f, _ in fold.groups)
    return comparison_chart(
        f"{h.label}: your {families} games against the rest",
        [label for label, _ in rows],
        [("You", you), ("Opponents", them)],
        value_format=h.value_format,
        note=f"{h.label}, {_UNIT_NOTE[h.key]}. You and your opponent in the same games.",
        kind="bar",
        table=Table(
            title=f"{h.label}: your {families} games against the rest",
            columns=["Games", "Count", "You", "Opponents"],
            rows=[[label, len(gs), a, b] for (label, gs), a, b in zip(rows, you, them, strict=True)],
            formats=["text", "int", h.value_format, h.value_format],
        ),
    )


def overview_chart(results: Sequence[HabitResult], th: Thresholds) -> Optional[Chart]:
    """Every habit as a share (you vs your opponents, all games), with the numbers by format underneath."""
    shown = [r for r in results if r.n]
    if not shown:
        return None
    rows = []
    for r in shown:
        formats = [(FORMAT_NAMES.get(tc, tc), gs) for tc, gs in r.by_format()]
        if len(formats) > 1:
            formats.append(("All games", r.games))
        for label, gs in formats:
            rows.append([r.habit.label, label, len(gs), r.habit.value_text(r.average(True, gs)),
                         r.habit.value_text(r.average(False, gs))])
    table = Table(
        title="Your first moves against your opponents', by format",
        columns=["Habit", "Format", "Games", "You", "Opponents"],
        rows=rows,
        formats=["text", "text", "int", "text", "text"],
        key_columns=[0, 1, 3, 4],
        note="You and your opponent in the same games; a game counts once both players made the moves the habit "
        "looks at. Standard chess only.",
    )
    return comparison_chart(
        "Your first moves against your opponents'",
        [r.habit.label for r in shown],
        [("You", [r.average(True) / r.habit.scale for r in shown]),
         ("Opponents", [r.average(False) / r.habit.scale for r in shown])],
        value_format="pct",
        note=f"Castling and the queen: share of games. Knights and bishops: share of the four out by move "
        f"{th.develop_by}. Pawn moves: share of the first {th.pawn_window} moves.",
        kind="hbar",
        table=table,
    )


# --------------------------------------------------------------------------- findings
_TEXT = {  # (habit, kind) -> (title, study actions)
    ("castled", "weakness"): (
        "You castle later than your opponents",
        [
            "Know the move by which you normally castle in each of your openings, and aim to castle by move 8–10 "
            "unless there is a concrete reason to wait.",
            "Replay the linked games at move 10: what kept your king in the centre (a bishop or knight still at "
            "home, an early queen or pawn move)?",
            "While your king is still in the centre, check your opponent's checks and captures along the e-file "
            "and the diagonals to it before every move.",
        ],
    ),
    ("castled", "strength"): (
        "You castle earlier than your opponents",
        [
            "Keep castling early: it lets you play the middlegame without worrying about your king.",
            "Use the time it buys you: when your opponent's king is still in the centre, look for moves that open "
            "the centre (a d- or e-pawn break).",
        ],
    ),
    ("early_queen", "weakness"): (
        "You bring your queen out early more often than your opponents",
        [
            "Before moving your queen in the first moves, ask whether a knight or bishop could do the job: an "
            "early queen gets chased and your opponent develops with tempo.",
            "Replay the linked games at your early queen move and count how many of your opponent's next moves "
            "attacked it.",
        ],
    ),
    ("early_queen", "strength"): (
        "You bring your queen out early less often than your opponents",
        [
            "Keep developing knights and bishops before the queen.",
            "When your opponent's queen comes out early, look for developing moves that attack it: each one "
            "gains you a move.",
        ],
    ),
    ("minors", "weakness"): (
        "You develop your knights and bishops more slowly than your opponents",
        [
            "Aim to have all four knights and bishops out by move 10: before each opening move, ask which piece "
            "is not in the game yet.",
            "Replay the linked games at move 10 and compare your pieces still at home with your opponent's.",
            "Knights before bishops, and no piece moved twice before the others are out unless it wins something "
            "(Capablanca's rules for the opening).",
        ],
    ),
    ("minors", "strength"): (
        "You develop your knights and bishops faster than your opponents",
        [
            "Keep developing quickly, and use the lead: open the centre before your opponent catches up.",
            "When you are ahead in development, look for pawn breaks that open lines toward the uncastled king.",
        ],
    ),
    ("pawns", "weakness"): (
        "You make more pawn moves early on than your opponents",
        [
            "In your first 10 moves, make only the pawn moves that free your pieces or fight for the centre; use "
            "the rest to bring out pieces and castle.",
            "Replay the linked games and mark each early pawn move: which of them could have been a piece move?",
        ],
    ),
}


def _detail(r: HabitResult) -> str:
    h = r.habit
    mine, theirs = h.number(r.average(True)), h.number(r.average(False))
    games = f"{r.n} game{'s' if r.n != 1 else ''} in which both of you made at least {h.by} moves"
    if h.key == "castled":
        text = f"By move {h.by} you had castled in {mine} of games, your opponents in {theirs} of the same games ({games})."
    elif h.key == "early_queen":
        text = (
            f"You moved your queen in the first {h.by} moves, other than to capture, in {mine} of games; your "
            f"opponents in {theirs} of the same games ({games})."
        )
    elif h.key == "minors":
        text = (
            f"By move {h.by} you had on average {mine} of your 4 knights and bishops off their starting squares, "
            f"your opponents {theirs} in the same games ({games})."
        )
    else:
        text = (
            f"On average {mine} of your first {h.by} moves were pawn moves, against your opponents' {theirs} in the "
            f"same games ({games})."
        )
    split = [
        f"{FORMAT_NAMES.get(tc, tc).lower()} {h.number(r.average(True, gs))} vs {h.number(r.average(False, gs))}"
        for tc, gs in r.by_format()
        if len(gs) >= MIN_FORMAT_GAMES
    ]
    if len(split) > 1:
        text += " By format, you vs them: " + ", ".join(split) + "."
    return text


def _target(r: HabitResult) -> tuple[str, str]:
    """(target sentence, metric) for the study plan: your opponents' number, with today's."""
    h = r.habit
    mine, theirs = h.number(r.average(True)), h.number(r.average(False))
    if h.key == "castled":
        return (f"Castle by move {h.by} in at least {theirs} of your games, like your opponents (now {mine}).",
                f"share of games castled by move {h.by}")
    if h.key == "early_queen":
        return (f"Move your queen in the first {h.by} moves (other than to capture) in at most {theirs} of your "
                f"games, like your opponents (now {mine}).", f"share of games with an early queen move (moves 1–{h.by})")
    if h.key == "minors":
        return (f"Have {theirs} of your knights and bishops out by move {h.by} on average, like your opponents "
                f"(now {mine}).", f"knights and bishops out by move {h.by}, per game")
    return (f"Make at most {theirs} pawn moves in your first {h.by} moves on average, like your opponents "
            f"(now {mine}).", f"pawn moves in the first {h.by} moves, per game")


def _examples(r: HabitResult, kind: str) -> list[Opening]:
    """Games that show the finding: you did it and your opponent didn't (by the most), losses first for a
    weakness and wins first for a strength, then the most recent; games with a link before games without one."""
    h = r.habit
    sign = 1 if (kind == "weakness") == (h.good < 0) else -1  # +1: you did more of it than your opponent

    def edge(o: Opening) -> int:
        return sign * (h.count(o.me) - h.count(o.opp))

    wanted = "loss" if kind == "weakness" else "win"
    picked = [o for o in r.games if edge(o) > 0]
    picked.sort(key=lambda o: (not o.game.url, o.game.outcome != wanted, -edge(o), -o.game.end_time.timestamp()))
    return picked


def queen_strip(o: Opening, ply: int, title: str) -> Strip:
    """The early queen move at ply index ``ply`` and the three plies after it, as small boards; in each, the queen
    and the pieces attacking it are marked while it is under attack."""
    board = o.board_after(ply - 1)
    fen, owner = board.fen(), board.turn
    moves = o.ucis[ply : ply + 4]
    marks: list[list[Mark]] = []
    captions: list[str] = []
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
        frame: list[Mark] = []
        for sq in board.pieces(chess.QUEEN, owner):
            attackers = board.attackers(not owner, sq)
            if attackers:
                frame += [Mark(chess.square_name(sq), "target")]
                frame += [Mark(chess.square_name(a), "attacker") for a in attackers]
        marks.append(frame)
        captions.append("Queen attacked" if frame else "")
    return line_strip(title, fen, moves, max_frames=4, captions=captions, marks=marks)


def habit_diagram(r: HabitResult, o: Opening, kind: str = "weakness") -> Optional[Diagram]:
    """The board that shows a finding in one of its example games, from your side of the board.

    A weakness shows your side: your early queen move as a red arrow (and what followed as a strip), or the
    position after move N with your uncastled king, your knights and bishops still at home or your moved pawns
    marked. A strength shows the same moment on your opponent's side (their king, their pieces at home, their
    early queen move): what you did better, and what there was to play against.
    """
    h, g = r.habit, o.game
    mine = kind == "weakness"
    color = o.my_color if mine else not o.my_color
    side, other = (o.me, o.opp) if mine else (o.opp, o.me)
    game = f"{g.outcome.capitalize()}, {g.time_class}"
    if h.key == "early_queen":
        ply = side.queen_ply
        if ply is None or ply < 1:
            return None
        board = o.board_after(ply - 1)
        move = chess.Move.from_uci(o.ucis[ply])
        home = pieces_at_home(board, color, o.ucis[:ply])
        label = move_label(board, move)
        if mine:
            title = f"Your queen comes out on move {side.queen_move}"
            still = f", with {len(home)} of your knights and bishops still at home" if home else ""
            caption = f"{game}: you played {label}{still}."
        else:
            title = f"Your opponent's queen comes out on move {side.queen_move}"
            caption = (f"{game}: your opponent played {label}; you made no early queen move of your own "
                       f"(moves 1–{h.by}, captures aside).")
        diagram = position_diagram(
            title,
            board.fen(),
            orientation=g.color,
            caption=caption,
            link=g.url,
            time_class=g.time_class,
            last_move=o.ucis[ply - 1],
            played=move.uci() if mine else None,  # your move red; your opponent's grey
            others=() if mine else [(move.uci(), "neutral")],
            marks=[Mark(chess.square_name(sq), "weak" if mine else "target") for sq in home],
        )
        strip = queen_strip(o, ply, "What happened next")
        if strip.frames:
            diagram.strips = [strip]
        return diagram
    last = 2 * h.by - 1
    board = o.board_after(last)
    mark = "weak" if mine else "target"
    if h.key == "castled":
        king = board.king(color)
        where = f" (king on {chess.square_name(king)})" if king is not None else ""
        marks = [Mark(chess.square_name(king), mark)] if king is not None else []
        if mine:
            theirs = (f"your opponent castled on move {other.castle_move}" if other.castle_move
                      else "your opponent hadn't castled either")
            title = f"Move {h.by}: you haven't castled yet"
            caption = f"{game}: after move {h.by} you hadn't castled{where}; {theirs}."
        else:
            title = f"Move {h.by}: your opponent hasn't castled yet"
            caption = f"{game}: you castled on move {other.castle_move}; after move {h.by} your opponent hadn't{where}."
    elif h.key == "minors":
        marks = [Mark(chess.square_name(sq), mark) for sq in pieces_at_home(board, color, o.ucis[: last + 1])]
        title = f"Move {h.by}: pieces still at home" if mine else f"Move {h.by}: your opponent's pieces still at home"
        caption = (
            f"{game}: after move {h.by} you had {o.me.minors_by(h.by)} of 4 knights and bishops out; your opponent "
            f"{o.opp.minors_by(h.by)}."
        )
    else:
        marks = [Mark(chess.square_name(sq), "focus") for sq in moved_pawns(board, color)]
        title = f"Move {h.by}: many pawn moves"
        caption = (
            f"{game}: {o.me.pawns_by(h.by)} of your first {h.by} moves were pawn moves; your opponent's "
            f"{o.opp.pawns_by(h.by)}."
        )
    return position_diagram(
        title, board.fen(), orientation=g.color, caption=caption, link=g.url, time_class=g.time_class,
        last_move=o.ucis[last] if last < len(o.ucis) else "", marks=marks,
    )


def _passes(r: HabitResult, th: Thresholds) -> Optional[tuple[str, float]]:
    """(kind, confidence) when the gap passes the claim rule and is big enough to matter; None otherwise."""
    h = r.habit
    if r.n < th.min_games:
        return None
    significant, confidence = significance(r.test, th.min_games, r.p_adjusted, alpha=th.alpha)
    if not significant or confidence < th.min_confidence or abs(r.gap) < h.min_gap:
        return None
    kind = "strength" if (r.gap > 0) == (h.good > 0) else "weakness"
    if kind == "strength" and not h.strength:
        return None
    return kind, confidence


def fold_check(r: HabitResult, th: Thresholds, family: dict[str, float], steps: int = FOLD_GROUPS) -> Optional[Fold]:
    """The fold rule for a habit that passes the claim rule (None for any other): leave out the (colour, opening
    family) group that contributes most to the gap and test the habit again; while it still passes, do the same
    with the biggest part left, up to ``steps`` groups (your main line with each colour, say). ``family`` holds
    the raw p-values of the habits judged together; each re-test's replaces this habit's, BH-adjusted again."""
    passed = _passes(r, th)
    if passed is None:
        return None
    h, sign = r.habit, 1.0 if r.gap > 0 else -1.0
    current, groups = r, []
    for _ in range(steps):
        shares = current.contributions()
        key = max(shares, key=lambda k: sign * shares[k]) if shares else None
        if key is None or sign * shares[key] <= 0.0:
            break
        groups.append((key[0], key[1], [o for o in current.games if o.group == key]))
        rest = [o for o in current.games if o.group != key]
        test = paired_share_test([(h.count(o.me), h.count(o.opp), o.game.color) for o in rest], h.scale)
        raw = {**family, h.key: test.p_value}
        names = sorted(raw)
        adjusted = dict(zip(names, bh_adjust([raw[k] for k in names]), strict=True))
        current = HabitResult(h, rest, test, adjusted[h.key])
        again = _passes(current, th)
        if again is None or again[0] != passed[0]:
            return Fold(groups, current, holds=False, too_few=current.n < th.min_games)
    return Fold(groups, current, holds=True, too_few=current.n < th.min_games) if groups else None


def _gap_text(h: Habit, gap: float) -> str:
    """'4 points' / '0.2 pieces' / '0.3 pawn moves'."""
    if h.binary:
        return f"{abs(gap) * 100:.0f} points"
    return f"{abs(gap):.1f} " + ("pieces" if h.key == "minors" else "pawn moves")


def _group_text(r: HabitResult, games: Sequence[Opening]) -> str:
    """'you castled by move 10 in 2% of them, your opponents in 47%' (the games of one opening group)."""
    h = r.habit
    you, them = h.number(r.average(True, games)), h.number(r.average(False, games))
    if h.key == "castled":
        return f"you castled by move {h.by} in {you} of them, your opponents in {them}"
    if h.key == "early_queen":
        return f"you brought your queen out early in {you} of them, your opponents in {them}"
    if h.key == "minors":
        return f"you had {you} of 4 knights and bishops out by move {h.by}, your opponents {them}"
    return f"you made {you} pawn moves in your first {h.by}, your opponents {them}"


_BOOK = {  # (habit, kind the gap would have been) -> what the opening's main lines would do too
    ("castled", "weakness"): "castle as late",
    ("castled", "strength"): "castle as early",
    ("early_queen", "weakness"): "bring the queen out as early",
    ("early_queen", "strength"): "keep the queen at home as long",
    ("minors", "weakness"): "develop the knights and bishops as slowly",
    ("minors", "strength"): "develop the knights and bishops as fast",
    ("pawns", "weakness"): "make as many early pawn moves",
}


def fold_observation(r: HabitResult, fold: Fold, kind: str, confidence: float) -> Insight:
    """A would-be finding that the fold rule traced to one or two opening groups: an observation about them."""
    h = r.habit
    held = HabitResult(h, fold.games, r.test)
    shown = _examples(held, kind)
    rest = fold.without
    one = len(fold.groups) == 1
    usual = f"the way {'that opening is' if one else 'those openings are'} usually played"
    if not rest.n:
        after = f"All of them are {'that opening' if one else 'those openings'}, so a habit of yours can't be told " \
                f"apart from {usual}."
    elif fold.too_few:
        after = f"Your other {rest.n} games are too few to tell a habit of yours from {usual}."
    else:
        after = (
            f"Without them the gap is {_gap_text(h, rest.gap)} ({h.value_text(r.average(True, rest.games))} "
            f"against {h.value_text(r.average(False, rest.games))} in {rest.n} games), too small or unclear to call "
            f"a {kind}: {usual} is the likelier reason, rather than a habit of yours."
        )
    parts = " and ".join(
        f"your {f} games with {c.capitalize()} ({len(gs)} game{'s' if len(gs) != 1 else ''}: "
        f"{h.value_text(r.average(True, gs))} against {h.value_text(r.average(False, gs))})"
        for c, f, gs in fold.groups
    )
    colours = {c for c, _, _ in fold.groups}
    if len(colours) == 1:
        colour = fold.groups[0][0].capitalize()
        title = (f"Mostly your {' and '.join(f for _, f, _ in fold.groups)} games: with {colour} "
                 f"{_group_text(r, fold.games)}")
        lines = f"the opening's main lines for {colour}"
    else:
        title = f"Mostly your {fold.names()}: {_group_text(r, fold.games)}"
        lines = "those openings' main lines"
    shrunk = shrink_effect(r.test.mean, r.test.se, prior_sd=h.full / h.unit / 2.0) * h.unit
    return Insight(
        id=f"{KEY}.observation.{h.key.replace('_', '-')}",
        kind="observation",
        category="structure",
        title=title,
        detail=f"{_detail(r)} Most of that gap comes from {parts}. {after}",
        severity=clamp(abs(shrunk) / h.full),
        confidence=confidence,
        evidence={
            "habit": h.key,
            "n": r.n,
            "you": r.average(True),
            "opponents": r.average(False),
            "gap": r.gap,
            "p_value": r.test.p_value,
            "p_adjusted": r.p_adjusted,
            "would_be": kind,
            "groups": [{"colour": c, "family": f, "n": len(gs), "you": r.average(True, gs),
                        "opponents": r.average(False, gs)} for c, f, gs in fold.groups],
            "without": {"n": rest.n, "you": r.average(True, rest.games), "opponents": r.average(False, rest.games),
                        "gap": rest.gap if rest.n else None, "p_value": rest.test.p_value,
                        "p_adjusted": rest.p_adjusted},
            "by_opening": r.group_stats(),
            "by_format": r.format_stats(),
        },
        study=[
            f"Compare your {fold.names()} with {lines}: if those {_BOOK[(h.key, kind)]} as your games do, the "
            "gap comes from the opening, not from a habit of yours.",
            f"Replay one of the linked games at move {h.by} and check your moves against the main line.",
        ],
        example_games=[o.game.url for o in shown if o.game.url][:MAX_EXAMPLES],
        formats=format_counts(o.game for o in fold.games),
        chart=fold_chart(r, fold),
        diagram=habit_diagram(r, shown[0], kind) if shown else None,
    )


def habit_insight(r: HabitResult, th: Thresholds, fold: Optional[Fold] = None) -> Optional[Insight]:
    """A strength or weakness when the gap passes the claim rule and is big enough to matter; an observation
    about one opening group when the fold rule (``fold``, from :func:`fold_check`) traces the gap to it."""
    passed = _passes(r, th)
    if passed is None:
        return None
    kind, confidence = passed
    if fold is not None and not fold.holds:
        return fold_observation(r, fold, kind, confidence)
    h = r.habit
    title, study = _TEXT[(h.key, kind)]
    # severity from the gap pulled toward 0 by its noise (winner's curse), in the habit's own units
    shrunk = shrink_effect(r.test.mean, r.test.se, prior_sd=h.full / h.unit / 2.0) * h.unit
    shown = _examples(r, kind)
    target, metric = _target(r)
    evidence: dict[str, Any] = {
        "habit": h.key,
        "n": r.n,
        "you": r.average(True),
        "opponents": r.average(False),
        "gap": r.gap,
        "p_value": r.test.p_value,
        "p_adjusted": r.p_adjusted,
        "by_format": r.format_stats(),
        "by_opening": r.group_stats(),
        "metric": metric,
        "target": target,
    }
    if fold is not None:  # the gap holds without the opening groups that contribute most to it
        evidence["fold"] = _fold_stats(fold)
    return Insight(
        id=f"{KEY}.{kind}.{h.key.replace('_', '-')}",
        kind=kind,  # type: ignore[arg-type]
        category="structure",
        title=title,
        detail=_detail(r),
        severity=clamp(abs(shrunk) / h.full),
        confidence=confidence,
        evidence=evidence,
        study=study,
        example_games=[o.game.url for o in shown if o.game.url][:MAX_EXAMPLES],
        formats=format_counts(o.game for o in r.games),
        chart=habit_chart(r),
        diagram=habit_diagram(r, shown[0], kind) if shown else None,
    )


# --------------------------------------------------------------------------- module entry point
def _summary(by: dict[str, HabitResult], insights: Sequence[Insight], skipped: int, th: Thresholds) -> str:
    main = by["castled"]
    if not main.n:
        note = " (Chess960 and games from a set-up position are left out)" if skipped else ""
        return f"Not enough data: none of your standard games reached move {th.castle_by}{note}."

    def both(key: str) -> tuple[str, str]:
        r = by[key]
        return r.habit.number(r.average(True)), r.habit.number(r.average(False))

    castle, queen, minors, pawns = both("castled"), both("early_queen"), both("minors"), both("pawns")
    text = (
        f"In the {main.n} standard games that reached move {th.castle_by}, you had castled by then in {castle[0]} "
        f"(your opponents in the same games: {castle[1]}), had {minors[0]} of your 4 knights and bishops out "
        f"({minors[1]}), moved your queen early in {queen[0]} ({queen[1]}) and made {pawns[0]} pawn moves in your "
        f"first {th.pawn_window} ({pawns[1]})."
    )
    if main.n < th.min_games:
        return (f"Not enough data yet: only {main.n} games reached move {th.castle_by}, so treat these numbers as a "
                f"snapshot. {text}")
    ranked = sorted((i for i in insights if i.kind != "observation"), key=lambda i: -i.priority)
    if ranked:
        return f"{text} Key finding: {ranked[0].title}."
    if any(i.kind == "observation" for i in insights):
        return (f"{text} None of the gaps is a habit of yours: the clearest comes mostly from one of your openings "
                "(below).")
    return f"{text} None of the gaps is big and clear enough to call a strength or weakness."


def _kpis(by: dict[str, HabitResult], th: Thresholds) -> list[Kpi]:
    rows = (
        ("castled", f"Castled by move {th.castle_by}", ""),
        ("minors", "Knights and bishops out", f"of 4 by move {th.develop_by}; "),
        ("early_queen", "Queen out early", f"moves 1–{th.queen_by}, not capturing; "),
        ("pawns", "Early pawn moves", f"in moves 1–{th.pawn_window}; "),
    )
    kpis = []
    for key, label, prefix in rows:
        r = by[key]
        theirs = r.average(False)
        hint = prefix + (f"opponents {r.habit.number(theirs)}" if theirs is not None else "")
        kpis.append(Kpi(label, r.average(True), r.habit.value_format, hint=hint.rstrip("; ")))
    return kpis


def _fold_stats(fold: Optional[Fold]) -> Optional[dict[str, Any]]:
    if fold is None:
        return None
    return {"groups": [{"colour": c, "family": f, "n": len(gs)} for c, f, gs in fold.groups], "holds": fold.holds,
            "n_without": fold.without.n, "gap_without": fold.without.gap if fold.without.n else None,
            "p_value_without": fold.without.test.p_value, "p_adjusted_without": fold.without.p_adjusted}


def analyze(ctx: AnalysisContext) -> ModuleResult:
    """Castling, early queen moves, development and early pawn moves: you against your opponents in the same games."""
    th = Thresholds.from_ctx(ctx)
    hs = habits(th)
    window = 2 * max(h.by for h in hs)
    standard = [g for g in ctx.games if is_standard_start(g)]
    skipped = len(ctx.games) - len(standard)
    openings: list[Opening] = []
    unreadable = 0
    for g in standard:
        if g.plies < MIN_PLIES:
            continue
        o = read_opening(g, window)
        if o is None:
            unreadable += 1
        else:
            openings.append(o)

    results = []
    for h in hs:
        games = [o for o in openings if o.plies >= 2 * h.by]
        pairs = [(h.count(o.me), h.count(o.opp), o.game.color) for o in games]
        results.append(HabitResult(h, games, paired_share_test(pairs, h.scale)))
    judged = [r for r in results if r.n >= th.min_games]
    for r, p in zip(judged, bh_adjust([r.test.p_value for r in judged]), strict=True):
        r.p_adjusted = p
    family = {r.habit.key: r.test.p_value for r in judged}
    folds = {r.habit.key: fold_check(r, th, family) for r in results}
    insights = [i for i in (habit_insight(r, th, folds[r.habit.key]) for r in results) if i]
    by = {r.habit.key: r for r in results}
    overview = overview_chart(results, th)

    return ModuleResult(
        key=KEY,
        title=TITLE,
        summary=_summary(by, insights, skipped, th),
        kpis=_kpis(by, th) if by["castled"].n else [],
        charts=[overview] if overview else [],
        insights=insights,
        stats={
            "n": len(ctx.games),
            "n_standard": len(standard),
            "skipped_variant_or_setup": skipped,
            "unreadable": unreadable,
            "habits": {
                r.habit.key: {
                    "label": r.habit.label,
                    "n": r.n,
                    "you": r.average(True),
                    "opponents": r.average(False),
                    "gap": r.gap if r.n else None,
                    "p_value": r.test.p_value,
                    "p_adjusted": r.p_adjusted,
                    "by_format": r.format_stats(),
                    "by_opening": r.group_stats(),
                    "fold": _fold_stats(folds[r.habit.key]),
                }
                for r in results
            },
        },
    )
