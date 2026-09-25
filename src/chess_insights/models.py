"""Core data contracts shared by every module.

Everything downstream of the API client works on these types:

    chess.com JSON --parse.py--> Game --analysis/*--> ModuleResult --insights.py--> Report --report/*--> HTML / Markdown / JSON

Games are always normalised to the *analysed player's* point of view
(``color``, ``outcome``, ``my_rating`` ...), so analysis code never has to
ask "which side was I?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

Color = Literal["white", "black"]
Outcome = Literal["win", "draw", "loss"]
InsightKind = Literal["strength", "weakness", "observation"]

# ---------------------------------------------------------------------------
# chess.com result codes (per player, in game["white"]["result"] / ["black"]["result"])
# ---------------------------------------------------------------------------
WIN_CODES = frozenset({"win"})
DRAW_CODES = frozenset(
    {"agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"}
)
LOSS_CODES = frozenset(
    {
        "checkmated",
        "timeout",
        "resigned",
        "lose",
        "abandoned",
        "kingofthehill",
        "threecheck",
        "bughousepartnerlose",
    }
)

# Normalised "how did the game end" values (Game.termination). Derived from the
# loser's code for decisive games, or the draw code for draws.
TERMINATIONS = (
    "checkmate",
    "resignation",
    "timeout",
    "abandoned",
    "agreement",
    "repetition",
    "stalemate",
    "insufficient",
    "fifty_move",
    "timeout_vs_insufficient",
    "variant",  # kingofthehill / threecheck / bughouse partner loss
    "other",
)

TIME_CLASSES = ("bullet", "blitz", "rapid", "daily")

# Insight categories (Insight.category). Keep in sync with insights.STUDY_LIBRARY.
CATEGORIES = (
    "results",  # overall score, rating trend, formats
    "color",  # white vs black
    "openings",  # repertoire performance
    "opponents",  # performance vs stronger/weaker opponents
    "time",  # clock usage, time trouble, flagging
    "endings",  # how games end (resign/timeout/mate), game length
    "habits",  # tilt, sessions, time of day
    "accuracy",  # engine accuracy / ACPL
    "blunders",  # engine blunder/mistake rates and types
    "phases",  # opening / middlegame / endgame quality
    "conversion",  # winning won positions, saving lost ones
    "tactics",  # missed tactical shots / missed mates
)


@dataclass
class Game:
    """One chess.com game, seen from the analysed player's side."""

    # identity
    game_id: str  # chess.com uuid when present, else the game URL
    url: str

    # perspective
    username: str  # analysed player, as spelled in the game
    color: Color
    opponent: str
    outcome: Outcome
    my_result_code: str  # raw chess.com code for the player, e.g. "resigned"
    opp_result_code: str
    termination: str  # one of TERMINATIONS

    # format
    time_class: str  # bullet | blitz | rapid | daily
    time_control: str  # raw chess.com string: "600", "180+2", "1/86400"
    base_seconds: Optional[int]  # live: starting clock; daily: seconds per move; None if unknown
    increment: int  # live increment in seconds; 0 for none and for daily
    rules: str  # chess | chess960 | bughouse | ...
    rated: bool

    # time (tz-aware UTC)
    end_time: datetime
    start_time: Optional[datetime]

    # ratings (after-game rating as reported by chess.com)
    my_rating: Optional[int]
    opp_rating: Optional[int]

    # opening
    eco: Optional[str]  # "C50" (PGN ECO header)
    opening: Optional[str]  # "Italian Game: Giuoco Piano" (from ECOUrl slug)
    opening_family: Optional[str]  # "Italian Game"

    # chess.com accuracy (present only for reviewed games)
    my_accuracy: Optional[float]
    opp_accuracy: Optional[float]

    # moves
    initial_fen: Optional[str]  # None for the standard start position
    moves_san: list[str]  # every ply, SAN
    clocks: list[Optional[float]]  # mover's remaining seconds AFTER each ply (increment included); all None for daily games
    pgn: str

    # chess.com reports ratings AFTER the game. parse.parse_games reconstructs the
    # pre-game ratings (previous rated game in the same pool) so that expected scores
    # aren't biased by the game's own result. None when unknown (first game in a pool).
    my_rating_before: Optional[int] = None
    opp_rating_before: Optional[int] = None

    # ---- derived helpers -------------------------------------------------
    @property
    def plies(self) -> int:
        return len(self.moves_san)

    @property
    def full_moves(self) -> int:
        """Number of moves in chess notation (a 41-ply game has 21 moves)."""
        return (self.plies + 1) // 2

    @property
    def score(self) -> float:
        return {"win": 1.0, "draw": 0.5, "loss": 0.0}[self.outcome]

    @property
    def rating_diff(self) -> Optional[int]:
        """Opponent rating minus my rating before the game (positive = opponent stronger).

        Uses the reconstructed pre-game ratings when available, else the post-game ones.
        """
        if self.my_rating_before is not None and self.opp_rating_before is not None:
            return self.opp_rating_before - self.my_rating_before
        if self.my_rating is None or self.opp_rating is None:
            return None
        return self.opp_rating - self.my_rating

    @property
    def expected_score(self) -> Optional[float]:
        """Elo expected score for the analysed player."""
        diff = self.rating_diff
        if diff is None:
            return None
        return 1.0 / (1.0 + 10 ** (diff / 400.0))

    @property
    def white_to_move_first(self) -> bool:
        if not self.initial_fen:
            return True
        parts = self.initial_fen.split()
        return len(parts) < 2 or parts[1] == "w"

    def mover(self, ply: int) -> Color:
        """Colour that played ply index ``ply`` (0-based)."""
        white_first = self.white_to_move_first
        return "white" if (ply % 2 == 0) == white_first else "black"

    def is_my_ply(self, ply: int) -> bool:
        return self.mover(ply) == self.color

    @property
    def my_ply_indices(self) -> list[int]:
        return [i for i in range(self.plies) if self.is_my_ply(i)]


# ---------------------------------------------------------------------------
# Engine analysis (produced by engine.py, consumed by analysis/engine_stats.py)
# ---------------------------------------------------------------------------
@dataclass
class PlyEval:
    """Engine verdict on one ply. Win percentages are from the MOVER's point of view."""

    ply: int  # 0-based index into Game.moves_san
    mover: Color
    is_user: bool
    san: str
    best_san: Optional[str]  # engine's preferred move in the position before this ply
    cp_before: int  # eval before the move, WHITE's POV, centipawns (mate mapped to +-MATE_CP)
    cp_after: int  # eval after the move, WHITE's POV
    mate_before: Optional[int]  # mate-in-N before the move, WHITE's POV sign; None if no mate seen
    mate_after: Optional[int]
    win_before: float  # 0..100, mover's POV
    win_after: float  # 0..100, mover's POV
    accuracy: float  # 0..100 Lichess-style move accuracy for the mover
    cp_loss: int  # max(0, mover-POV cp drop), capped (see engine.CP_LOSS_CAP)
    judgement: Optional[str]  # None | "inaccuracy" | "mistake" | "blunder"
    phase: str  # "opening" | "middlegame" | "endgame"
    clock_after: Optional[float]  # mover's remaining seconds after the move
    time_spent: Optional[float]  # seconds the mover spent on this ply (increment-adjusted), if known
    tags: list[str] = field(default_factory=list)  # missed_mate, allowed_mate, hung_material, missed_tactic, thrown_win


@dataclass
class GameEval:
    game_id: str
    engine: str  # e.g. "Stockfish 16"
    depth: Optional[int]
    plies: list[PlyEval]
    my_accuracy: Optional[float]  # Lichess-style game accuracy, 0..100
    opp_accuracy: Optional[float]


# ---------------------------------------------------------------------------
# Analysis output contracts
# ---------------------------------------------------------------------------
@dataclass
class Insight:
    """One finding about the player, ranked by ``priority``.

    severity   0..1 — how much it matters (size of the effect in points/rating/rates)
    confidence 0..1 — how sure we are (sample size, interval width)
    """

    id: str  # stable, e.g. "openings.weakness.black.caro-kann-defense"
    kind: InsightKind
    category: str  # one of CATEGORIES
    title: str  # one plain-language line: "The Caro-Kann is costing you points as Black"
    detail: str  # 1-3 sentences with the numbers behind it
    severity: float
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)  # n, score, expected, ...
    study: list[str] = field(default_factory=list)  # concrete actions
    example_games: list[str] = field(default_factory=list)  # chess.com URLs, most instructive first, <=5

    @property
    def priority(self) -> float:
        return max(0.0, min(1.0, self.severity)) * max(0.0, min(1.0, self.confidence))


# Value formats understood by the renderers (Table.formats / Chart.value_format / Kpi.format)
VALUE_FORMATS = ("text", "int", "float1", "float2", "pct", "signed_pct", "signed_int", "rating", "url", "seconds")


@dataclass
class Table:
    title: str
    columns: list[str]
    rows: list[list[Any]]
    formats: Optional[list[str]] = None  # one of VALUE_FORMATS per column; None = auto
    note: str = ""


@dataclass
class Series:
    name: str
    values: list[Optional[float]]


@dataclass
class Chart:
    kind: Literal["bar", "hbar", "line", "stacked_bar"]
    title: str
    labels: list[str]  # category / x-axis labels, one per value in each series
    series: list[Series]
    value_format: str = "int"  # one of VALUE_FORMATS
    note: str = ""
    reference: Optional[float] = None  # optional horizontal reference line (e.g. 0.5 score)


@dataclass
class Kpi:
    label: str
    value: Any
    format: str = "text"  # one of VALUE_FORMATS
    hint: str = ""


@dataclass
class Diagram:
    """A chess position to show in the report (e.g. a position you keep getting wrong)."""

    title: str
    fen: str
    svg: str  # rendered by chess.svg from ``fen`` in our own code — trusted markup, never user text
    caption: str = ""
    link: str = ""  # chess.com game URL


@dataclass
class ModuleResult:
    key: str  # "results", "openings", ...
    title: str  # section heading
    summary: str  # 1-2 plain-language sentences
    kpis: list[Kpi] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)  # machine-readable numbers (JSON export)
    diagrams: list[Diagram] = field(default_factory=list)


@dataclass
class StudyItem:
    title: str  # "Fix your Caro-Kann (Black)"
    why: str  # links back to the insight numbers
    actions: list[str]  # concrete steps
    games: list[str] = field(default_factory=list)  # URLs to review
    category: str = ""
    priority: float = 0.0


@dataclass
class Report:
    username: str
    generated_at: datetime
    filters: str  # human description of the game filter, e.g. "rated blitz+rapid, standard chess, since 2024-01"
    n_games: int
    date_from: Optional[datetime]
    date_to: Optional[datetime]
    modules: list[ModuleResult]
    strengths: list[Insight]
    weaknesses: list[Insight]
    study_plan: list[StudyItem]
    engine_note: str = ""  # e.g. "Stockfish 16 depth 12 on 150 most recent games" or why engine analysis was skipped
    headline: str = ""  # one or two sentences summarising the report
