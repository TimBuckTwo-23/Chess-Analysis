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
    "positions",  # the same wrong move in the same position, game after game
    "structure",  # castling, development, early queen moves: your habits against your opponents' in the same games
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
    # Which formats the finding is about: games (or engine-analysed games) behind it per time class, e.g.
    # {"bullet": 210, "blitz": 88}. Every finding carries it, so the report can say "blitz and rapid" or
    # "bullet only"; fill.fill_formats sets it from the module's games when a module leaves it empty.
    formats: dict[str, int] = field(default_factory=dict)
    # Every finding carries a picture: a small chart of the numbers behind it (you vs the benchmark, split by
    # format where the data allows) and/or the position it is about. fill.fill_visuals adds a chart from
    # ``evidence`` when a module sets neither.
    chart: Optional["Chart"] = None
    diagram: Optional["Diagram"] = None

    @property
    def priority(self) -> float:
        return max(0.0, min(1.0, self.severity)) * max(0.0, min(1.0, self.confidence))


# Value formats understood by the renderers (Table.formats / Chart.value_format / Kpi.format)
VALUE_FORMATS = (
    "text", "int", "float1", "float2", "pct", "signed_pct", "signed_int", "signed_float2", "rating", "url", "seconds",
)


@dataclass
class Table:
    title: str
    columns: list[str]
    rows: list[list[Any]]
    formats: Optional[list[str]] = None  # one of VALUE_FORMATS per column; None = auto
    note: str = ""
    key_columns: Optional[list[int]] = None  # columns a phone shows by default in a wide table; None = a guess


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
    # the full numbers behind the chart (more columns than the series): shown under "Show the numbers"
    # instead of the bare series, so the section need not list the same numbers again as a table
    table: Optional[Table] = None


@dataclass
class Kpi:
    label: str
    value: Any
    format: str = "text"  # one of VALUE_FORMATS
    hint: str = ""


# Board annotations. Renderers draw every board themselves from the FEN (one shared piece sprite per
# page), so a Diagram is plain data: position, arrows, marked squares and optional move strips.
ARROW_KINDS = ("played", "best", "threat", "line", "neutral")  # red, green, orange, blue, grey
MARK_KINDS = ("focus", "target", "attacker", "weak", "check")


@dataclass
class Arrow:
    start: str  # square name, e.g. "e2"
    end: str  # e.g. "e4" (castling: the king's destination)
    kind: str = "neutral"  # one of ARROW_KINDS


@dataclass
class Mark:
    square: str  # e.g. "d6"
    kind: str = "focus"  # one of MARK_KINDS


@dataclass
class Frame:
    """One small board in a strip: the position after ``move``."""

    fen: str
    move: str = ""  # the move that led here, numbered: "6.Ndb5" / "6...a6"
    last_move: str = ""  # its UCI ("b5d6"), highlighted on the board
    caption: str = ""  # e.g. "+1.5 for White" or "Fork: d6 hits king and b7"
    arrows: list[Arrow] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)


@dataclass
class Strip:
    """A line of play shown as a row of small boards ("What happens after 5...e5")."""

    title: str
    frames: list[Frame] = field(default_factory=list)


@dataclass
class Diagram:
    """A chess position to show in the report (a position you keep getting wrong, a choice point, a missed shot)."""

    title: str
    fen: str
    svg: str = ""  # legacy: a pre-rendered board. Renderers draw from ``fen`` + annotations; leave empty
    caption: str = ""
    link: str = ""  # chess.com game URL
    orientation: Color = "white"  # the analysed player's side at the bottom
    arrows: list[Arrow] = field(default_factory=list)  # played move red, better move green, ...
    marks: list[Mark] = field(default_factory=list)
    last_move: str = ""  # UCI of the move that led to ``fen`` (highlighted)
    strips: list[Strip] = field(default_factory=list)  # optional lines of play after the position
    time_class: str = ""  # format of the game the position comes from ("" = several / not a game)


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
    insight_ids: list[str] = field(default_factory=list)  # the findings this item works on, lead finding first
    findings: list[str] = field(default_factory=list)  # their titles, in the same order
    target: str = ""  # what to aim for by the next report, with today's number
    baseline: dict[str, Any] = field(default_factory=dict)  # {"metric", "value", "insight"}: today's number, for the next report


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
    summary_lines: list[str] = field(default_factory=list)  # the headline as short lines: ratings, first steps, a strength
    game_labels: dict[str, str] = field(default_factory=dict)  # game URL -> "Loss · 5+0 · vs 1512 · 12 Aug"
    demo: bool = False  # a synthetic demo player: games, opponents and links are made up (links are not shown)
    time_class: str = ""  # "" = every format in ``filters``; "blitz" etc. for one format's view
    formats: dict[str, int] = field(default_factory=dict)  # games per time class in this report
    engine_formats: dict[str, int] = field(default_factory=dict)  # engine-analysed games per time class
    # The same analysis run separately on each format with enough games ("bullet", "blitz", "rapid", "daily"
    # order). Each is a full Report with its own claims (tested on that format's games only), its
    # ``time_class`` set, and no format_reports of its own.
    format_reports: dict[str, "Report"] = field(default_factory=dict)
    coaching: Optional["Coaching"] = None  # explanations, drills, schedule (coach/); None when not run


# ---------------------------------------------------------------------------
# Coaching layer (chess_insights/coach). Explanation and practice material only: it never creates a
# claim. Every Explanation points at a position (EPD) or an insight id that already exists.
# ---------------------------------------------------------------------------
@dataclass
class Line:
    """A sequence of moves from ``fen``, with the engine's verdict at its end.

    ``cp_end`` / ``mate_end`` are from the point of view of the side to move at ``fen`` (the side whose
    choice it was), so a refutation of your move shows up negative for you.
    """

    fen: str
    moves_uci: list[str] = field(default_factory=list)
    moves_san: list[str] = field(default_factory=list)
    cp_end: Optional[int] = None
    mate_end: Optional[int] = None
    fen_end: str = ""
    depth: Optional[int] = None


@dataclass
class Motif:
    """A tactical pattern found along a line, named with Lichess's puzzle-theme names."""

    theme: str  # "fork", "pin", "skewer", "discoveredAttack", "hangingPiece", "backRankMate", "mateIn2" ...
    line: str  # "refutation" (what your move allowed) | "best" (what you missed)
    ply: int = 0  # index into the line's moves where the pattern appears
    squares: list[str] = field(default_factory=list)  # squares involved, the active piece first (for the board)
    side: str = ""  # "you" | "opponent": who carries out the pattern


@dataclass
class ConceptDelta:
    """How a positional term differs between the ends of the refutation and the best line, from your side.

    Negative = the refutation line ends worse for you on this term. Pawns (100 cp = 1.0).
    """

    term: str  # Stockfish 16 classical term ("Mobility", "King safety", ...) or a python-chess fact key
    value: float  # phase-blended difference, pawns
    mg: Optional[float] = None
    eg: Optional[float] = None
    label: str = ""  # plain words: "piece activity", "king safety", "pawn structure"
    source: str = "stockfish16"  # "stockfish16" | "python-chess"


@dataclass
class Source:
    """Where an external fact came from, shown with the fact."""

    name: str  # "Lichess opening explorer (masters)", "Wikibooks, CC BY-SA 4.0", "Stockfish 16, depth 20"
    url: str = ""
    retrieved: str = ""  # ISO date
    license: str = ""


@dataclass
class MoveStat:
    """One move's share and score in an opening database (from the side to move's point of view)."""

    san: str
    uci: str = ""
    games: int = 0
    share: float = 0.0  # of all games in the position
    score: Optional[float] = None  # 0..1 for the side to move
    avg_rating: Optional[int] = None


@dataclass
class OpeningFacts:
    """What the databases say about a position (C3). Every list may be empty when a source is unavailable."""

    eco: str = ""
    name: str = ""  # chess-openings name of the position (or the nearest named one before it)
    masters: list[MoveStat] = field(default_factory=list)
    peers: list[MoveStat] = field(default_factory=list)  # players one or two rating groups above you
    peer_groups: list[int] = field(default_factory=list)  # Lichess rating groups used for ``peers``
    played_rank_masters: Optional[int] = None  # 1-based rank of your move among the masters' moves
    played_rank_peers: Optional[int] = None
    master_game: str = ""  # a master game URL from the position
    cloud_lines: list[Line] = field(default_factory=list)
    wiki_text: str = ""
    wiki_url: str = ""
    sources: list[Source] = field(default_factory=list)


@dataclass
class CriticalPosition:
    """A position worth explaining: one of your errors, a repeated mistake or a choice point in a main line."""

    game_id: str
    url: str
    ply: int  # index of your move in Game.moves_san
    fen: str  # the position before your move
    played_uci: str
    played_san: str
    move_label: str  # "5...e5"
    best_san: Optional[str]
    drop: float  # win-% points your move lost (engine)
    time_class: str
    color: Color  # your colour
    kind: str = "error"  # "error" | "repeated" | "choice"
    repeats: int = 1  # games in which you played it here
    games: list[str] = field(default_factory=list)  # URLs of those games, most recent first
    insight_id: Optional[str] = None
    opening_family: str = ""
    moves_before: list[str] = field(default_factory=list)  # SAN from the start (for the caption and sources)

    @property
    def epd(self) -> str:
        return " ".join(self.fen.split()[:4])


@dataclass
class Explanation:
    """Why a move was wrong (or what the better choice is), from the engine's lines."""

    epd: str
    fen: str
    played: str  # "5...e5"
    best: Optional[str]  # "5...a6"
    best_line: Optional[Line]
    refutation: Optional[Line]  # the engine's line after your move
    motifs: list[Motif] = field(default_factory=list)
    concepts: list[ConceptDelta] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)  # board facts in words ("you lose the bishop pair")
    text: str = ""  # two or three plain sentences (templates, or a verified LLM rewrite)
    text_source: str = "template"  # "template" | "llm"
    sources: list[Source] = field(default_factory=list)
    insight_id: Optional[str] = None
    kind: str = "error"  # as CriticalPosition.kind
    drop: float = 0.0
    time_class: str = ""
    color: Color = "white"
    # What the deeper search says about the move played: "error" (it confirms the mistake), "close" (under an
    # inaccuracy and under a pawn behind its first choice) or "fine" (its own first choice). A repeated-mistake
    # finding whose move the deeper search clears is not kept as a weakness (pipeline.apply_deep_verdicts).
    verdict: str = "error"
    game_url: str = ""
    games: list[str] = field(default_factory=list)
    repeats: int = 1
    opening: Optional[OpeningFacts] = None
    tablebase: dict[str, Any] = field(default_factory=dict)  # {"category": "win", "best": "Kd6", ...} when <= 7 pieces
    maia: dict[str, Any] = field(default_factory=dict)  # {"rating": 1400, "p_best": 0.31, "p_played": 0.22}
    diagram: Optional[Diagram] = None  # the board with arrows and the two lines as strips
    chart: Optional[Chart] = None  # concept differences as bars
    drill_themes: list[str] = field(default_factory=list)  # Lichess training themes to practise this
    # A short note on the concept behind the largest concept difference, from a public-domain classic:
    # {"label", "text", "credit", "url"} (coach.sources.concept_notes); {} when none applies
    concept_note: dict[str, Any] = field(default_factory=dict)


@dataclass
class DrillPuzzle:
    puzzle_id: str  # Lichess PuzzleId
    fen: str  # the puzzle position (after the opponent's first move)
    solution_uci: list[str]
    solution_san: list[str]
    rating: int
    themes: list[str] = field(default_factory=list)
    url: str = ""  # https://lichess.org/training/<PuzzleId>
    opening_tags: list[str] = field(default_factory=list)


@dataclass
class Drill:
    """A pack of practice puzzles for one theme (or for your openings)."""

    theme: str  # Lichess theme name, or "openings"
    title: str  # "30 fork puzzles"
    link: str  # https://lichess.org/training/<theme>
    file: str = ""  # PGN written next to the report ("" when not written)
    rating_range: tuple[int, int] = (1200, 1600)
    puzzles: list[DrillPuzzle] = field(default_factory=list)
    reason: str = ""  # "you missed 23 forks in 300 games (opponents: 15)"


@dataclass
class ReviewItem:
    """A puzzle scheduled for spaced repetition (1, 3, 7 and 21 days after the report)."""

    item_id: str  # "own:<game_id>:<ply>" or "lichess:<PuzzleId>"
    kind: str  # "own" | "drill"
    title: str
    fen: str
    url: str
    due: str  # ISO date
    step: int = 0  # 0..3 -> 1, 3, 7, 21 days


@dataclass
class PlanEntry:
    """One line of the weekly plan (C4): minutes per week and when to check again."""

    item: str
    minutes_per_week: int
    recheck_date: str = ""  # ISO date
    note: str = ""


@dataclass
class ProgressItem:
    """A study-plan target next to the previous report's number."""

    title: str
    metric: str
    before: Optional[float]
    now: Optional[float]
    text: str  # "clock used by move 15: 50% -> 38%"
    improved: Optional[bool] = None


@dataclass
class Coaching:
    explanations: list[Explanation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # what was skipped and why (no Stockfish 16, offline, ...)
    settings: dict[str, Any] = field(default_factory=dict)  # engine, depth, positions, ...
    motif_profile: Optional[Table] = None  # motif, you missed, they missed, you allowed, they allowed (per 100 moves)
    motif_chart: Optional[Chart] = None
    drills: list[Drill] = field(default_factory=list)
    review: list[ReviewItem] = field(default_factory=list)  # scheduled by this report
    review_due: list[ReviewItem] = field(default_factory=list)  # due now, from earlier reports
    theory_exit: Optional[Table] = None  # where you leave named opening theory, per line (C3)
    endgames: Optional[Table] = None  # tablebase-checked endings (C3)
    endgame_diagrams: list[Diagram] = field(default_factory=list)  # boards for those endings
    weekly_plan: list[PlanEntry] = field(default_factory=list)  # C4
    progress: list[ProgressItem] = field(default_factory=list)  # C4
    llm: dict[str, Any] = field(default_factory=dict)  # {"model", "accepted", "rejected"} when the LLM ran
    # For the puzzle export (--puzzles): your errors' best lines (up to 8 plies) and gated motif themes, keyed
    # "<game_id>:<ply>", only for the errors that export can use (mistakes.build_puzzles: at most 300, costliest
    # first, standard chess) plus your errors in the explained positions (coach.puzzles.puzzle_keys), so the JSON
    # stays small. From the deep pass, else the profile pass; empty when neither ran.
    puzzle_lines: dict[str, Line] = field(default_factory=dict)
    puzzle_themes: dict[str, list[str]] = field(default_factory=dict)
