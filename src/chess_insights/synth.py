"""Synthetic chess.com archives for a fictional player with planted strengths and weaknesses.

``chess-insights demo`` analyses these games and checks that the report finds the
planted traits again (:data:`PLANTED_TRAITS`), so the generator doubles as an
end-to-end recovery test. It works in three passes:

1. **Plan** (sequential, one RNG). A calendar of playing sessions, time controls,
   colours, openings, opponents' rating offsets and each game's *intended*
   result, drawn from the Elo expectation shifted by the persona's traits
   (Caro-Kann, Italian Game, tilt after losses, late-night play, rapid vs blitz).
2. **Play** (parallel, one RNG and a freshly reset engine per game, so the output
   does not depend on the worker count). Every game is played move by move with
   python-chess: shallow Stockfish (or a fast material heuristic) proposes moves
   and each side errs with a probability that depends on its role in the intended
   result, the game phase and its clock. Games end only in ways that are true on
   the board or the clock (mate, resignation in a lost position, flag, repetition,
   stalemate, insufficient material, 50-move rule, agreed draw in a level
   position). When a game drifts from its plan, the board decides.
3. **Assemble** (sequential). Real timestamps from the played durations, Elo
   ratings (K=16) from the actual results, and chess.com JSON with PGN.
"""

from __future__ import annotations

import math
import multiprocessing
import multiprocessing.util
import os
import random
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import chess
import chess.engine

from .fetch import GameStore, month_end, parse_month
from .stats import WHITE_EDGE

# --------------------------------------------------------------------------- opening book


@dataclass(frozen=True)
class OpeningLine:
    """A few moves of real theory plus the ECO code and chess.com ECOUrl slug they get."""

    eco: str
    slug: str  # path after https://www.chess.com/openings/
    moves: tuple[str, ...]  # SAN, from the initial position

    @property
    def url(self) -> str:
        return f"https://www.chess.com/openings/{self.slug}"


def _line(eco: str, slug: str, moves: str) -> OpeningLine:
    return OpeningLine(eco, slug, tuple(moves.split()))


OPENINGS: dict[str, OpeningLine] = {
    # 1.e4 e5 2.Nf3 Nc6 systems (the player's White repertoire; also met as Black)
    "italian_pianissimo": _line(
        "C54",
        "Italian-Game-Giuoco-Pianissimo-Normal",
        "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O",
    ),
    "italian_two_knights": _line(
        "C55",
        "Italian-Game-Two-Knights-Defense-Modern-Bishops-Opening",
        "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Be7 O-O O-O Re1 d6",
    ),
    "italian_anti_fried_liver": _line(
        "C50",
        "Italian-Game-Anti-Fried-Liver-Defense",
        "e4 e5 Nf3 Nc6 Bc4 h6 O-O Nf6 d3 Bc5 c3 d6",
    ),
    "italian_knight_attack": _line(
        "C58",
        "Italian-Game-Two-Knights-Defense-Knight-Attack-Normal-Variation",
        "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6",
    ),
    "ruy_closed": _line(
        "C84",
        "Ruy-Lopez-Opening-Morphy-Defense-Closed-Variations",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1",
    ),
    "ruy_exchange": _line(
        "C69",
        "Ruy-Lopez-Opening-Exchange-Variation-5.O-O-f6-6.d4",
        "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4",
    ),
    "ruy_berlin": _line(
        "C65",
        "Ruy-Lopez-Opening-Berlin-Defense-4.d3-Bc5",
        "e4 e5 Nf3 Nc6 Bb5 Nf6 d3 Bc5 c3 O-O O-O d6",
    ),
    "scotch_classical": _line(
        "C45",
        "Scotch-Game-Classical-Variation-5.Be3-Qf6-6.c3-Nge7",
        "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7",
    ),
    "scotch_schmidt": _line(
        "C45",
        "Scotch-Game-Schmidt-Variation-5.Nxc6-bxc6-6.e5-Qe7",
        "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7",
    ),
    "four_knights_scotch": _line(
        "C47",
        "Four-Knights-Game-Scotch-Variation",
        "e4 e5 Nf3 Nc6 Nc3 Nf6 d4 exd4 Nxd4 Bb4 Nxc6 bxc6",
    ),
    "vienna": _line(
        "C28",
        "Vienna-Game-Stanley-Variation-Three-Knights-Variation",
        "e4 e5 Nc3 Nf6 Bc4 Nc6 d3 Bc5 f4 d6 Nf3",
    ),
    # other replies to 1.e4
    "sicilian_alapin": _line("B22", "Sicilian-Defense-Alapin-Variation-2...Nf6", "e4 c5 c3 Nf6 e5 Nd5 d4 cxd4 Nf3 Nc6"),
    "french_advance": _line(
        "C02",
        "French-Defense-Advance-Variation-3...c5-4.c3-Nc6",
        "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6",
    ),
    "scandinavian": _line(
        "B01",
        "Scandinavian-Defense-Mieses-Kotrc-Variation",
        "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6",
    ),
    "petrov": _line("C42", "Petrovs-Defense-Classical-Attack", "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3"),
    "pirc": _line("B08", "Pirc-Defense-Classical-Variation", "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O"),
    # the player's Caro-Kann (as Black)
    "caro_advance_short": _line(
        "B12",
        "Caro-Kann-Defense-Advance-Variation-Short-Variation",
        "e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 O-O Nc6",
    ),
    "caro_advance_c5": _line(
        "B12",
        "Caro-Kann-Defense-Advance-Variation-3...c5",
        "e4 c6 d4 d5 e5 c5 dxc5 e6 Be3 Nd7 Nf3 Bxc5",
    ),
    "caro_exchange": _line(
        "B13",
        "Caro-Kann-Defense-Exchange-Variation",
        "e4 c6 d4 d5 exd5 cxd5 Bd3 Nc6 c3 Nf6 Bf4 Bg4",
    ),
    "caro_classical": _line(
        "B18",
        "Caro-Kann-Defense-Classical-Variation",
        "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 Nf3 Nd7",
    ),
    "caro_two_knights": _line("B11", "Caro-Kann-Defense-Two-Knights-Attack", "e4 c6 Nc3 d5 Nf3 Bg4 h3 Bxf3 Qxf3 e6"),
    # the player's answers to 1.d4 / 1.c4 / 1.Nf3
    "qgd_harrwitz": _line(
        "D37",
        "Queens-Gambit-Declined-Queens-Knight-Variation-4.Nf3-Be7-5.Bf4",
        "d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7 Bf4 O-O e3 c5",
    ),
    "qgd_exchange": _line(
        "D35",
        "Queens-Gambit-Declined-Exchange-Variation",
        "d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O",
    ),
    "london": _line(
        "D00",
        "Queens-Pawn-Opening-Accelerated-London-System",
        "d4 d5 Bf4 Nf6 e3 c5 c3 Nc6 Nd2 e6 Ngf3 Bd6",
    ),
    "english_kings": _line(
        "A29",
        "English-Opening-Kings-English-Variation-Four-Knights-Variation",
        "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6",
    ),
    "kings_indian_attack": _line("A07", "Kings-Indian-Attack", "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O"),
}

ITALIAN_LINES = ("italian_pianissimo", "italian_two_knights", "italian_anti_fried_liver", "italian_knight_attack")
CARO_KANN_LINES = ("caro_advance_short", "caro_advance_c5", "caro_exchange", "caro_classical", "caro_two_knights")

# Player as White (always 1.e4): (line, weight). Italian 45%, Ruy Lopez 20%, Scotch 10%, the rest is
# whatever Black answers instead of 1...e5 2...Nc6. ``Persona.italian_share`` rescales the Italian block.
_WHITE_REPERTOIRE: tuple[tuple[str, float], ...] = (
    ("italian_pianissimo", 0.20),
    ("italian_two_knights", 0.12),
    ("italian_anti_fried_liver", 0.06),
    ("italian_knight_attack", 0.07),
    ("ruy_closed", 0.08),
    ("ruy_exchange", 0.06),
    ("ruy_berlin", 0.06),
    ("scotch_classical", 0.06),
    ("scotch_schmidt", 0.04),
    ("sicilian_alapin", 0.09),
    ("french_advance", 0.06),
    ("scandinavian", 0.04),
    ("petrov", 0.03),
    ("pirc", 0.03),
)
# Player as Black: the opponent's first move, then the line.
_OPPONENT_FIRST_MOVE = (("e4", 0.60), ("d4", 0.28), ("c4", 0.06), ("Nf3", 0.06))
_BLACK_VS_E5 = (  # after 1.e4 e5 the opponent picks the system
    ("italian_pianissimo", 0.14),
    ("italian_two_knights", 0.12),
    ("italian_anti_fried_liver", 0.04),
    ("italian_knight_attack", 0.08),
    ("ruy_closed", 0.08),
    ("ruy_exchange", 0.06),
    ("ruy_berlin", 0.06),
    ("scotch_classical", 0.07),
    ("scotch_schmidt", 0.05),
    ("four_knights_scotch", 0.15),
    ("vienna", 0.15),
)
_CARO_KANN_MIX = (
    ("caro_advance_short", 0.35),
    ("caro_advance_c5", 0.15),
    ("caro_exchange", 0.20),
    ("caro_classical", 0.20),
    ("caro_two_knights", 0.10),
)
_BLACK_VS_D4 = (("qgd_harrwitz", 0.35), ("qgd_exchange", 0.30), ("london", 0.35))

# --------------------------------------------------------------------------- persona and planted traits


@dataclass
class Persona:
    """A fictional chess.com player and the traits planted in their games.

    ``*_shift`` values are points per game added to the Elo expectation when a game's intended
    result is drawn. They add up where contexts overlap (a late-night Caro-Kann straight after a
    loss gets three), and games with none of the planted contexts get ``baseline_shift``: a real
    player's rating already prices in their weaknesses, so their ordinary games sit a little above
    expectation and the rating stays roughly stable. The defaults are calibrated so the planted
    subsets land near: Caro-Kann -0.18, Italian +0.12, after a loss -0.12, late night -0.13,
    rapid +0.09 (blitz -0.03, bullet -0.04).
    """

    username: str = "demo_player"
    start_ratings: dict[str, int] = field(
        default_factory=lambda: {"blitz": 1450, "rapid": 1520, "bullet": 1300, "daily": 1410}
    )
    time_class_mix: dict[str, float] = field(
        default_factory=lambda: {"blitz": 0.60, "rapid": 0.28, "bullet": 0.10, "daily": 0.02}
    )
    # T8: stronger at rapid than at the fast time controls
    time_class_shift: dict[str, float] = field(
        default_factory=lambda: {"blitz": 0.0, "rapid": 0.115, "bullet": -0.02, "daily": 0.0}
    )
    baseline_shift: float = 0.09
    # T1: Caro-Kann as Black against 1.e4
    caro_kann_share: float = 0.38
    caro_kann_shift: float = -0.15
    # T2: Italian Game as White (share of all White games; White always opens 1.e4)
    italian_share: float = 0.45
    italian_shift: float = 0.18
    # T3: blitz clock handling
    blitz_opening_time_factor: float = 2.1  # time used on moves 1-15 relative to a typical opponent
    blitz_flag_share: float = 0.28  # share of planned blitz losses that end on the clock
    low_clock_error_factor: float = 2.2  # error-rate multiplier with less than 10% of the clock left
    # T4: tilt
    tilt_shift: float = -0.125
    tilt_window_min: float = 15.0
    quick_requeue_after_loss: float = 0.8
    # T5: late-night play (games starting 23:00-03:00 UTC)
    late_night_shift: float = -0.16
    late_session_share: float = 0.28
    # T6: error-rate multiplier once the position is an endgame (opponents: 1.0)
    endgame_error_factor: float = 3.0
    # T7: share of non-wins in which the player first gets a clearly winning position (opponents: lower)
    throw_win_share: float = 0.50
    opponent_throw_win_share: float = 0.05


DEFAULT_PERSONA = Persona()

PLANTED_TRAITS: list[dict[str, Any]] = [
    {
        "id": "T1-caro-kann",
        "category": "openings",
        "expect": "weakness",
        "description": "Scores well below their rating with the Caro-Kann Defense as Black",
        "keywords": ["caro-kann", "caro kann"],
    },
    {
        "id": "T2-italian",
        "category": "openings",
        "expect": "strength",
        "description": "Scores above their rating with the Italian Game as White",
        # White-side phrasings only: doing well *against* the Italian as Black is not this trait
        "keywords": [
            "white.italian",
            "italian game as white",
            "italian game is working for you as white",
            "as white in the italian",
            "as white with the italian",
            "italian game (white)",
            "white: italian",
        ],
    },
    {
        "id": "T3-blitz-clock",
        "category": "time",
        "expect": "weakness",
        "description": "Spends far too long on the opening in blitz, gets into time trouble and loses on time",
        "keywords": [
            "time trouble",
            "on time",
            "flag",
            "timeout",
            "time pressure",
            "low on time",
            "first 15",
            "opening moves",
            "slow",
        ],
    },
    {
        "id": "T4-tilt",
        "category": "habits",
        "expect": "weakness",
        "description": "Scores worse in games started within 15 minutes of a loss (tilt)",
        "keywords": [
            "after a loss",
            "after losing",
            "after losses",
            "tilt",
            "right after",
            "straight after",
            "losing streak",
        ],
    },
    {
        "id": "T5-late-night",
        "category": "habits",
        "expect": "weakness",
        "description": "Scores worse in games started late at night (23:00-03:00 UTC)",
        "keywords": ["night", "midnight", "23:00", "late hours", "after 11"],
    },
    {
        "id": "T6-endgame-errors",
        "category": "phases",
        "expect": "weakness",
        "description": "Makes far more mistakes and blunders in endgames than in the middlegame",
        "keywords": ["endgame", "ending"],
    },
    {
        "id": "T7-conversion",
        "category": "conversion",
        "expect": "weakness",
        "description": "Often fails to win clearly winning positions (+3 or better)",
        "keywords": ["convert", "winning position", "won position", "winning advantage", "let the win slip", "slip"],
    },
    {
        "id": "T8-rapid",
        "category": "results",
        "expect": "strength",
        "description": "Performs better at rapid than at blitz relative to their rating",
        "keywords": ["rapid"],
    },
]

# --------------------------------------------------------------------------- tuning constants

MATE_CP = 10_000
K_FACTOR = 16.0
OPPONENT_RATING_SD = 110.0
DEFAULT_END = datetime(2026, 9, 1, tzinfo=timezone.utc)
LIVE_ID_BASE = 140_000_000_000
DAILY_ID_BASE = 900_000_000

TIME_CONTROLS: dict[str, tuple[tuple[str, float], ...]] = {
    "bullet": (("60", 0.65), ("120+1", 0.35)),
    "blitz": (("180", 0.22), ("180+2", 0.23), ("300", 0.45), ("300+5", 0.10)),
    "rapid": (("600", 0.70), ("900+10", 0.20), ("1800", 0.10)),
    "daily": (("1/86400", 1.0),),
}
_DRAW_RATE = {"bullet": 0.05, "blitz": 0.06, "rapid": 0.08, "daily": 0.10}
# P(a decisive game ends on the clock | player loses), P(... | player wins); blitz losses use the persona.
_FLAG_SHARE = {"bullet": (0.35, 0.30), "blitz": (0.28, 0.08), "rapid": (0.07, 0.05), "daily": (0.0, 0.0)}
_RESIGN_P = {"bullet": 0.20, "blitz": 0.30, "rapid": 0.40, "daily": 0.45}
# Share of the starting clock a typical player uses on moves 1-15.
_OPENING_TIME_FRACTION = {"bullet": 0.22, "blitz": 0.20, "rapid": 0.24}
_PLAYER_OPENING_FACTOR = {"bullet": 1.25, "rapid": 1.25}  # blitz comes from the persona (T3)
_SESSION_SIZE_WEIGHTS = (15, 20, 20, 15, 10, 8, 6, 6)  # 1..8 games
_BASE_ERROR_RATES = (0.030, 0.050, 0.080)  # blunder, mistake, inaccuracy per move (typical club player)
# Win-percentage drop bands (Lichess judgement thresholds) for each kind of error.
_ERROR_BANDS = {"blunder": (30.0, 101.0), "mistake": (20.0, 30.0), "inaccuracy": (10.0, 20.0)}
_ENGINE_NODES = 2500
_ENGINE_MULTIPV = 4
_ENGINE_ERROR_DEPTH = 3


# --------------------------------------------------------------------------- small helpers
def _win_pct(cp: float) -> float:
    """Lichess win percentage (0..100) for a centipawn score from the mover's point of view."""
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def _move_accuracy(win_drop: float) -> float:
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * max(0.0, win_drop)) - 3.1669))


def _expected(offset: float) -> float:
    """Elo expected score against an opponent rated ``offset`` points higher."""
    return 1.0 / (1.0 + 10 ** (offset / 400.0))


def _weighted(rng: random.Random, options: Sequence[tuple[str, float]]) -> str:
    return rng.choices([o for o, _ in options], [w for _, w in options])[0]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _parse_tc(tc: str) -> tuple[int, int]:
    if "/" in tc:
        return int(tc.split("/", 1)[1]), 0
    base, _, inc = tc.partition("+")
    return int(base), int(inc or 0)


def _months_before(end: datetime, months: int) -> datetime:
    y, m = end.year, end.month - months
    while m <= 0:
        y, m = y - 1, m + 12
    return end.replace(year=y, month=m, day=min(end.day, 28))


def _expected_duration(base: int, inc: int) -> float:
    """Rough seconds a live game lasts, used only to lay out the calendar before games are played."""
    return base * 1.3 + inc * 50.0


def _find_stockfish() -> Optional[str]:
    for cand in (
        os.environ.get("CHESS_INSIGHTS_STOCKFISH"),
        shutil.which("stockfish"),
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
    ):
        if cand and Path(cand).is_file():
            return cand
    return None


# --------------------------------------------------------------------------- opponent handles
_NAME_WORDS_A = (
    "pawn", "rook", "bishop", "queen", "castle", "gambit", "tactic", "blitz", "check", "zugzwang", "fianchetto",
    "silent", "lucky", "dark", "brave", "lazy", "crazy", "quiet", "shadow", "iron", "golden", "red", "blue",
    "wild", "clever", "grumpy", "sleepy", "cosmic", "frosty", "sneaky", "humble", "rusty", "mighty", "sunny",
)
_NAME_WORDS_B = (
    "hunter", "master", "wizard", "slayer", "storm", "rider", "fan", "machine", "panda", "tiger", "fox", "owl",
    "wolf", "bear", "dragon", "falcon", "ninja", "pirate", "monk", "baron", "chef", "coach", "dad", "kid",
    "otter", "badger", "comet", "pilot", "viking", "sparrow", "moose", "raven",
)
_FIRST_NAMES = (
    "alex", "maria", "juan", "li", "anna", "omar", "nina", "igor", "sven", "priya", "tom", "lucas", "sofia",
    "mateo", "yuki", "ahmed", "elena", "david", "sara", "marco", "chen", "olga", "ravi", "emma", "noah", "luca",
    "ivan", "fatima", "jonas", "leo", "hana", "pedro", "kofi", "mila", "arjun", "zoe",
)
_SURNAMES = ("smith", "garcia", "novak", "kim", "rossi", "silva", "petrov", "muller", "khan", "tanaka", "berg", "costa")


def _handle(rng: random.Random) -> str:
    style = rng.random()
    a, b = rng.choice(_NAME_WORDS_A), rng.choice(_NAME_WORDS_B)
    first, last = rng.choice(_FIRST_NAMES), rng.choice(_SURNAMES)
    if style < 0.22:
        return f"{a.capitalize()}{b.capitalize()}{rng.randint(1, 99)}"
    if style < 0.40:
        return f"{first}_{b}"
    if style < 0.58:
        return f"{first}{rng.randint(1970, 2012)}"
    if style < 0.72:
        return f"{a}_{b}_{rng.randint(1, 9)}"
    if style < 0.86:
        return f"{first.capitalize()}{last.capitalize()}{rng.choice(['', str(rng.randint(1, 999))])}"
    return f"{first[0]}{last}{rng.randint(10, 99)}"


class _OpponentPool:
    """Plausible, mostly distinct opponent handles with the occasional familiar face."""

    def __init__(self, rng: random.Random, exclude: str) -> None:
        self.rng = rng
        self.seen: list[str] = []
        self.used = {exclude.lower()}

    def pick(self) -> str:
        if self.seen and self.rng.random() < 0.04:
            return self.rng.choice(self.seen)
        for _ in range(50):
            name = _handle(self.rng)
            if name.lower() not in self.used:
                break
        else:  # the name space is huge; this only guards against pathological seeds
            name = f"player_{len(self.seen) + 1}"
        self.used.add(name.lower())
        self.seen.append(name)
        return name


# --------------------------------------------------------------------------- planning
@dataclass
class _Plan:
    index: int
    game_seed: int
    session: int  # -1 for daily games
    time_class: str
    time_control: str
    base: int  # live: starting clock (s); daily: seconds per move
    inc: int
    color: str  # the player's colour
    opponent: str
    opening: str  # key into OPENINGS
    rated: bool
    offset: int  # opponent's pre-game rating minus the player's
    expected: float
    shift: float
    target: str  # intended result for the player: win | draw | loss
    finish: str  # "board" or "flag" (a decisive game meant to end on the clock)
    swing: bool  # the eventual non-winner first gets a clearly winning position
    reviewed: bool  # chess.com accuracies available
    planned_start: datetime
    gap_after: float  # seconds before the next game of the session
    tilt: bool
    late: bool
    keys: tuple[str, ...] = ()  # planted subsets the game belongs to: its time class plus trait contexts
    mu: float = 0.5  # intended expected score (Elo expectation + shift)
    max_duration: Optional[float] = None  # daily games must finish inside the window


@dataclass
class _Session:
    time_class: str
    size: int
    start: datetime


class _QuotaPicker:
    """Weighted choices whose running frequencies stay close to the weights.

    A real repertoire is stable: someone who answers 1.e4 with the Caro-Kann 38% of the time
    does not drift to 23% over a few hundred games the way independent draws can. Options that
    fall behind their share get proportionally more weight, so the order stays random while
    the frequencies track the persona.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.counts: dict[str, dict[str, int]] = {}

    def pick(self, decision: str, options: Sequence[tuple[str, float]]) -> str:
        counts = self.counts.setdefault(decision, {})
        n = sum(counts.values()) + 1
        total = sum(w for _, w in options) or 1.0
        choice = _weighted(self.rng, [(o, max(0.02, w / total * n - counts.get(o, 0))) for o, w in options])
        counts[choice] = counts.get(choice, 0) + 1
        return choice


def _pick_opening(picker: _QuotaPicker, color: str, persona: Persona) -> str:
    if color == "white":
        italian_w = sum(w for k, w in _WHITE_REPERTOIRE if k in ITALIAN_LINES)
        rest_w = 1.0 - italian_w
        scale_it = persona.italian_share / italian_w if italian_w else 0.0
        scale_rest = (1.0 - persona.italian_share) / rest_w if rest_w else 0.0
        weights = [(k, w * (scale_it if k in ITALIAN_LINES else scale_rest)) for k, w in _WHITE_REPERTOIRE]
        return picker.pick("white", weights)
    first = picker.pick("first-move", _OPPONENT_FIRST_MOVE)
    if first == "d4":
        return picker.pick("vs-d4", _BLACK_VS_D4)
    if first == "c4":
        return "english_kings"
    if first == "Nf3":
        return "kings_indian_attack"
    rest = 1.0 - persona.caro_kann_share
    reply = picker.pick("vs-e4", [("caro", persona.caro_kann_share), ("e5", rest * 0.8), ("scandinavian", rest * 0.2)])
    if reply == "caro":
        return picker.pick("caro-kann", _CARO_KANN_MIX)
    if reply == "e5":
        return picker.pick("vs-e5", _BLACK_VS_E5)
    return "scandinavian"


def _session_start(rng: random.Random, day: datetime, persona: Persona) -> datetime:
    """Evening-heavy start times (UTC) with a share of late-night sessions."""
    r = rng.random()
    late = persona.late_session_share
    if r < late:
        minute = rng.uniform(23 * 60 + 40, 25 * 60 + 50)  # 23:40 - 01:50 next day
    elif r < late + 0.55:
        minute = rng.uniform(17 * 60, 21 * 60 + 30)
    elif r < late + 0.70:
        minute = rng.uniform(12 * 60, 16 * 60 + 30)
    else:
        minute = rng.uniform(7 * 60, 11 * 60)
    return day + timedelta(minutes=minute)


class _OutcomeBalancer:
    """Keeps each planted subset's intended results close to what its shift promises.

    With independent draws a 45-game subset (say, the Caro-Kann games) wanders about +-0.07
    points per game from its planned shift, enough to hide a trait in an unlucky seed. A negative
    feedback on each subset's running surplus keeps it much closer. Trait contexts are spread over
    months, so they take a firm gain; time classes, whose games follow each other within a
    session, get a gentle one so consecutive games stay nearly independent.
    """

    GAIN = {"bullet": 0.04, "blitz": 0.04, "rapid": 0.04, "daily": 0.04}
    CONTEXT_GAIN = 0.12

    def __init__(self) -> None:
        self.surplus: dict[str, float] = {}

    def adjust(self, keys: Sequence[str], mu: float) -> float:
        return mu - sum(self.GAIN.get(k, self.CONTEXT_GAIN) * self.surplus.get(k, 0.0) for k in keys)

    def record(self, keys: Sequence[str], score: float, mu: float) -> None:
        for k in keys:
            self.surplus[k] = self.surplus.get(k, 0.0) + score - mu


_SCORE = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def _sample_target(
    rng: random.Random, expected: float, shift: float, time_class: str, balancer: _OutcomeBalancer, keys: Sequence[str]
) -> str:
    mu = _clamp(expected + shift, 0.03, 0.97)
    p_score = _clamp(balancer.adjust(keys, mu), 0.02, 0.98)
    draw = _DRAW_RATE[time_class] * (1.0 - abs(expected - 0.5))
    draw = min(draw, 2 * p_score, 2 * (1 - p_score))
    u = rng.random()
    target = "win" if u < p_score - draw / 2 else "draw" if u < p_score + draw / 2 else "loss"
    balancer.record(keys, _SCORE[target], mu)
    return target


def _plan_one(
    rng: random.Random,
    persona: Persona,
    pool: _OpponentPool,
    balancer: _OutcomeBalancer,
    picker: _QuotaPicker,
    *,
    time_class: str,
    time_control: str,
    planned_start: datetime,
    tilt: bool,
    late: bool,
    session: int,
    prev_opponent: Optional[str],
) -> _Plan:
    base, inc = _parse_tc(time_control)
    color = picker.pick("colour", (("white", 0.5), ("black", 0.5)))
    opening = _pick_opening(picker, color, persona)
    offset = int(_clamp(round(rng.gauss(0.0, OPPONENT_RATING_SD)), -400, 400))
    expected = _expected(offset)

    # The first-move advantage (White scores about WHITE_EDGE more than Black at equal ratings) is part
    # of every game's expectation; the planted shifts come on top, as the analyses measure them.
    expected = _clamp(expected + (WHITE_EDGE / 2 if color == "white" else -WHITE_EDGE / 2), 0.01, 0.99)
    shift = persona.time_class_shift.get(time_class, 0.0)
    contexts = []
    if color == "black" and opening in CARO_KANN_LINES:
        shift += persona.caro_kann_shift
        contexts.append("caro-kann")
    if color == "white" and opening in ITALIAN_LINES:
        shift += persona.italian_shift
        contexts.append("italian")
    if tilt:
        shift += persona.tilt_shift
        contexts.append("tilt")
    if late:
        shift += persona.late_night_shift
        contexts.append("late")
    if not contexts:
        shift += persona.baseline_shift

    target = _sample_target(rng, expected, shift, time_class, balancer, [time_class, *contexts])
    # how games end is a habit too: flags and thrown wins keep a steady share (quota picks)
    finish = "board"
    if target != "draw" and time_class != "daily":
        p_loss, p_win = _FLAG_SHARE[time_class]
        if time_class == "blitz":
            p_loss = persona.blitz_flag_share
        p_flag = p_loss if target == "loss" else p_win
        finish = picker.pick(f"finish:{time_class}:{target}", (("flag", p_flag), ("board", 1.0 - p_flag)))
    swing = False
    if finish == "board":
        share = persona.throw_win_share if target != "win" else persona.opponent_throw_win_share
        swing = picker.pick(f"swing:{target}", (("yes", share), ("no", 1.0 - share))) == "yes"
    rematch = prev_opponent is not None and rng.random() < 0.08
    return _Plan(
        index=-1,
        game_seed=0,
        session=session,
        time_class=time_class,
        time_control=time_control,
        base=base,
        inc=inc,
        color=color,
        opponent=prev_opponent if rematch and prev_opponent else pool.pick(),
        opening=opening,
        rated=rng.random() < 0.97,
        offset=offset,
        expected=expected,
        shift=shift,
        target=target,
        finish=finish,
        swing=swing,
        reviewed=rng.random() < 0.35,
        planned_start=planned_start,
        gap_after=0.0,
        tilt=tilt,
        late=late,
        keys=(time_class, *contexts),
        mu=_clamp(expected + shift, 0.03, 0.97),
    )


def _gap_after(rng: random.Random, target: str, persona: Persona) -> float:
    """Seconds between the end of a game and the start of the next one in the same session."""
    if target == "loss":
        if rng.random() < persona.quick_requeue_after_loss:
            return rng.uniform(10, min(10 * 60, persona.tilt_window_min * 60 - 30))
        return rng.uniform(persona.tilt_window_min * 60 + 60, 45 * 60)
    if rng.random() < 0.7:
        return rng.uniform(15, 10 * 60)
    return rng.uniform(11 * 60, 40 * 60)


def _settle_subsets(plans: list[_Plan], persona: Persona, rng: random.Random) -> None:
    """Flip a few intended results so every planted subset scores what its shifts promise.

    The balancer keeps subsets close while planning, but it cannot correct a subset's last games.
    This pass closes the remaining gap (usually one to three games per subset) by turning wins
    into losses or back. It only touches games whose result cannot change the next game's
    context: the last game of a session, one followed by a long pause, or a daily game.
    Trait subsets are settled with games that carry no other trait; time classes afterwards
    with ordinary games, so fixing one subset leaves the others alone.
    """
    last_of_session: dict[int, _Plan] = {}
    for p in sorted(plans, key=lambda p: p.planned_start):
        if p.session >= 0:
            last_of_session[p.session] = p
    window = persona.tilt_window_min * 60

    def free(p: _Plan) -> bool:
        return p.session < 0 or p.gap_after > window or last_of_session[p.session] is p

    for key in ("caro-kann", "italian", "tilt", "late", *TIME_CONTROLS):
        members = [p for p in plans if key in p.keys]
        surplus = sum(_SCORE[p.target] - p.mu for p in members)
        own_only = 1 if key in TIME_CONTROLS else 2  # the time class plus, for a trait, the trait itself
        candidates = [p for p in members if free(p) and p.target != "draw" and len(p.keys) == own_only]
        rng.shuffle(candidates)
        for p in candidates:
            if abs(surplus) <= 0.5:
                break
            if surplus > 0 and p.target == "win":
                p.target, surplus = "loss", surplus - 1.0
            elif surplus < 0 and p.target == "loss":
                p.target, surplus = "win", surplus + 1.0


def _plan_games(n_games: int, seed: int, persona: Persona, start: datetime, end: datetime) -> list[_Plan]:
    rng = random.Random(f"chess-insights-synth:{seed}")
    pool = _OpponentPool(rng, persona.username)
    balancer = _OutcomeBalancer()
    picker = _QuotaPicker(rng)
    mix = {tc: max(0.0, w) for tc, w in persona.time_class_mix.items() if tc in TIME_CONTROLS}
    total_w = sum(mix.values()) or 1.0
    mix = {tc: w / total_w for tc, w in mix.items()} or {"blitz": 1.0}
    n_daily = int(round(n_games * mix.get("daily", 0.0)))
    n_live = n_games - n_daily
    live_mix = {tc: w for tc, w in mix.items() if tc != "daily" and w > 0} or {"blitz": 1.0}
    live_total = sum(live_mix.values())

    # Sessions: sizes first, then a time class that keeps the running mix close to the target.
    sessions: list[_Session] = []
    assigned = {tc: 0 for tc in live_mix}
    remaining = n_live
    while remaining > 0:
        size = min(remaining, rng.choices(range(1, 9), _SESSION_SIZE_WEIGHTS)[0])
        done = sum(assigned.values()) + size
        deficits = [(tc, max(0.02, w / live_total * done - assigned[tc])) for tc, w in live_mix.items()]
        tc = _weighted(rng, deficits)
        assigned[tc] += size
        sessions.append(_Session(tc, size, start))
        remaining -= size

    # Calendar: weekday/weekend and month-to-month activity; evening-heavy hours.
    day0 = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    n_days = max(1, (end - day0).days - 1)
    intensity: dict[tuple[int, int], float] = {}
    day_weights = []
    for d in range(n_days):
        day = day0 + timedelta(days=d)
        f = intensity.setdefault((day.year, day.month), rng.uniform(0.55, 1.45))
        day_weights.append(f * (1.6 if day.weekday() >= 5 else 1.0))
    play_time = n_live * 600.0 + len(sessions) * 2700.0
    crowded = play_time > 0.35 * (end - start).total_seconds()
    for i, s in enumerate(sessions):
        if crowded:  # too many games for the window: spread sessions evenly instead of by preferred hour
            s.start = start + (end - start) * (i / max(1, len(sessions)))
        else:
            day = day0 + timedelta(days=rng.choices(range(n_days), day_weights)[0])
            s.start = max(start, _session_start(rng, day, persona))
    sessions.sort(key=lambda s: s.start)

    plans: list[_Plan] = []
    free_at = start
    for sid, s in enumerate(sessions):
        t = max(s.start, free_at)
        session_tc = _weighted(rng, TIME_CONTROLS[s.time_class])
        prev: Optional[_Plan] = None
        for _ in range(s.size):
            tc_str = session_tc if rng.random() < 0.85 else _weighted(rng, TIME_CONTROLS[s.time_class])
            tilt = prev is not None and prev.target == "loss" and prev.gap_after <= persona.tilt_window_min * 60
            plan = _plan_one(
                rng,
                persona,
                pool,
                balancer,
                picker,
                time_class=s.time_class,
                time_control=tc_str,
                planned_start=t,
                tilt=tilt,
                late=t.hour in (23, 0, 1, 2),
                session=sid,
                prev_opponent=prev.opponent if prev else None,
            )
            plan.gap_after = _gap_after(rng, plan.target, persona)
            t += timedelta(seconds=_expected_duration(plan.base, plan.inc) + plan.gap_after)
            plans.append(plan)
            prev = plan
        free_at = t + timedelta(minutes=45)

    # Daily games run in the background; each has its own start and must finish inside the window.
    latest = max(start, end - max(timedelta(days=21), (end - start) * 0.1))
    for _ in range(n_daily):
        started = (start + (latest - start) * rng.random()).replace(microsecond=0)
        plan = _plan_one(
            rng,
            persona,
            pool,
            balancer,
            picker,
            time_class="daily",
            time_control=_weighted(rng, TIME_CONTROLS["daily"]),
            planned_start=started,
            tilt=False,
            late=False,
            session=-1,
            prev_opponent=None,
        )
        plan.max_duration = (end - started).total_seconds() - 3600
        plans.append(plan)

    _settle_subsets(plans, persona, rng)
    plans.sort(key=lambda p: p.planned_start)
    for i, p in enumerate(plans):
        p.index = i
        p.game_seed = random.Random(f"chess-insights-synth:{seed}:{i}").getrandbits(63)
    return plans


# --------------------------------------------------------------------------- move proposers
Candidate = tuple[chess.Move, int]  # move and its score (cp, mover's point of view, mates folded to +-MATE_CP)


class _Mover(Protocol):
    exhaustive: bool  # top() already scores every legal move

    def new_game(self, key: object) -> None: ...

    def top(self, board: chess.Board) -> list[Candidate]: ...

    def full(self, board: chess.Board) -> list[Candidate]: ...

    def close(self) -> None: ...


class _EngineMover:
    """Shallow, node-limited Stockfish. The hash is cleared per game so results never depend on game order."""

    exhaustive = False

    def __init__(self, path: str) -> None:
        self.path = path
        self.engine = self._open()
        self.key: object = None

    def _open(self) -> chess.engine.SimpleEngine:
        engine = chess.engine.SimpleEngine.popen_uci(self.path)
        engine.configure({"Threads": 1, "Hash": 16})
        return engine

    def restart(self) -> None:
        try:
            self.engine.quit()
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, OSError):
            pass
        self.engine = self._open()

    def new_game(self, key: object) -> None:
        self.key = key

    def _analyse(self, board: chess.Board, limit: chess.engine.Limit, multipv: int) -> list[Candidate]:
        infos = self.engine.analyse(board, limit, multipv=multipv, game=self.key)
        out: list[Candidate] = []
        for info in infos:
            pv, score = info.get("pv"), info.get("score")
            if pv and score is not None:
                out.append((pv[0], int(score.pov(board.turn).score(mate_score=MATE_CP))))
        if not out:  # never expected; keep the game legal regardless
            out = [(next(iter(board.legal_moves)), 0)]
        out.sort(key=lambda c: -c[1])
        return out

    def top(self, board: chess.Board) -> list[Candidate]:
        return self._analyse(board, chess.engine.Limit(nodes=_ENGINE_NODES), _ENGINE_MULTIPV)

    def full(self, board: chess.Board) -> list[Candidate]:
        return self._analyse(board, chess.engine.Limit(depth=_ENGINE_ERROR_DEPTH), 500)

    def close(self) -> None:
        try:
            self.engine.quit()
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, OSError):
            pass


_VALUES = (0, 100, 310, 325, 500, 900, 0)  # indexed by chess.PieceType
_CENTER = tuple(
    3 - max(abs(2 * chess.square_file(sq) - 7), abs(2 * chess.square_rank(sq) - 7)) // 2 for sq in chess.SQUARES
)  # 3 in the centre, 0 on the rim


class _HeuristicMover:
    """Fast one-ply material mover: captures, hanging pieces, mate-in-one and a little positional sense."""

    exhaustive = True

    def new_game(self, key: object) -> None:
        pass

    def top(self, board: chess.Board) -> list[Candidate]:
        return self.full(board)

    def close(self) -> None:
        pass

    @staticmethod
    def _cheapest(board: chess.Board, mask: int) -> int:
        return min(_VALUES[board.piece_type_at(sq) or 0] or 10_000 for sq in chess.scan_forward(mask))

    def full(self, board: chess.Board) -> list[Candidate]:
        us, them = board.turn, not board.turn
        material = sum(
            (1 if p.color == us else -1) * _VALUES[p.piece_type] for p in board.piece_map().values()
        )
        endgame = _is_endgame(board)
        opening = board.fullmove_number <= 10
        hanging: list[tuple[int, int]] = []
        for sq in chess.scan_forward(board.occupied_co[us] & ~board.kings):
            att = board.attackers_mask(them, sq)
            if att:
                value = _VALUES[board.piece_type_at(sq) or 0]
                loss = value if not board.attackers_mask(us, sq) else max(0, value - self._cheapest(board, att))
                if loss:
                    hanging.append((sq, loss))
        their_king = board.king(them)
        our_king = board.king(us)
        out: list[Candidate] = []
        for mv in board.legal_moves:
            piece = board.piece_type_at(mv.from_square) or chess.PAWN
            if board.is_en_passant(mv):
                gain = 100
            else:
                gain = _VALUES[board.piece_type_at(mv.to_square) or 0]
            if mv.promotion:
                gain += _VALUES[mv.promotion] - 100
            moved = _VALUES[mv.promotion or piece]
            risk = 0
            if piece != chess.KING:
                att = board.attackers_mask(them, mv.to_square)
                if att:
                    defenders = board.attackers_mask(us, mv.to_square) & ~chess.BB_SQUARES[mv.from_square]
                    risk = moved if not defenders else max(0, moved - self._cheapest(board, att))
            left_hanging = max((loss for sq, loss in hanging if sq != mv.from_square and sq != mv.to_square), default=0)
            score = material + gain - max(risk, left_hanging * 9 // 10)
            score += self._positional(board, mv, piece, us, endgame, opening, material, our_king, their_king)
            if board.gives_check(mv):
                score += 25
                board.push(mv)
                if board.is_checkmate():
                    score = MATE_CP - 1
                board.pop()
            elif endgame and material > 200 and len(board.piece_map()) <= 6:
                board.push(mv)
                if board.is_stalemate():
                    score = 0
                board.pop()
            out.append((mv, score))
        out.sort(key=lambda c: -c[1])
        return out

    @staticmethod
    def _positional(
        board: chess.Board,
        mv: chess.Move,
        piece: int,
        us: bool,
        endgame: bool,
        opening: bool,
        material: int,
        our_king: Optional[int],
        their_king: Optional[int],
    ) -> int:
        frm, to = mv.from_square, mv.to_square
        if board.is_castling(mv):
            return 45
        if piece in (chess.KNIGHT, chess.BISHOP):
            bonus = 8 * (_CENTER[to] - _CENTER[frm])
            home_rank = 0 if us == chess.WHITE else 7
            if opening and chess.square_rank(frm) == home_rank:
                bonus += 12
            return bonus
        if piece == chess.PAWN:
            advance = abs(chess.square_rank(to) - chess.square_rank(frm))
            if endgame:
                return 12 * advance
            if opening and chess.square_file(frm) in (3, 4):
                return 10
            return 2 * advance
        if piece == chess.QUEEN:
            return -20 if opening else 4 * (_CENTER[to] - _CENTER[frm])
        if piece == chess.KING:
            if not endgame:
                return -25
            bonus = 6 * (_CENTER[to] - _CENTER[frm])
            if material >= 300 and their_king is not None and our_king is not None:  # walk the king in to help mate
                bonus += 10 * (chess.square_distance(our_king, their_king) - chess.square_distance(to, their_king))
            return bonus
        return 3 * (_CENTER[to] - _CENTER[frm])  # rooks


def _is_endgame(board: chess.Board) -> bool:
    """Lichess-style: six or fewer queens, rooks, bishops and knights left on the board."""
    return chess.popcount(board.occupied & ~board.pawns & ~board.kings) <= 6


# --------------------------------------------------------------------------- playing one game
@dataclass
class _Played:
    index: int
    sans: list[str]
    clocks: list[int]  # [%clk] in tenths: live = mover's clock after the ply; daily = time spent / 10
    think: list[int]  # tenths of a second spent on each ply
    final_think: int  # tenths spent by the side to move before the game ended (flag / resignation / agreement)
    end_kind: str  # a key of _TERMINATION_TEXT
    winner: Optional[str]  # "white" / "black" / None
    fen: str
    tcn: str
    accuracy: Optional[tuple[float, float]]  # (white, black)


class _Director:
    """Decides how error-prone each side is, given the intended result and how the game is going."""

    def __init__(self, plan: _Plan, player: chess.Color) -> None:
        self.plan = plan
        self.player = player
        opp = not player
        self.winner: Optional[chess.Color] = {"win": player, "loss": opp}.get(plan.target)
        # In a swing game the side that will NOT win first gets a clearly winning position.
        self.leader = opp if plan.target == "win" else player
        self.swing_phase = 1 if plan.swing else 2
        self._swing_hits = 0

    def observe(self, side: chess.Color, score: int, move_no: int) -> None:
        if self.swing_phase == 1:
            if side == self.leader:  # a clearly winning position: about 88% winning chances
                self._swing_hits = self._swing_hits + 1 if score >= 550 else 0
            if self._swing_hits >= 2 or move_no > 45:
                self.swing_phase = 2

    def role(self, side: chess.Color) -> str:
        if self.swing_phase == 1:
            return "winner" if side == self.leader else "loser"
        if self.winner is None:
            return "drawer"
        return "winner" if side == self.winner else "loser"

    def error_multiplier(self, side: chess.Color, score: int, move_no: int, endgame: bool) -> tuple[float, float]:
        """(blunder multiplier, mistake/inaccuracy multiplier) for ``side``, whose eval is ``score``."""
        role = self.role(side)
        if self.plan.finish == "flag" and self.swing_phase == 2:
            if role == "winner":
                return 0.2, 0.35
            m = 2.0 if score >= 150 else 0.6  # the side that will flag keeps the board level
            return m, m
        if role == "winner":  # rarely throws the game away; ordinary mistakes still happen
            return (0.2, 0.5) if score < 150 else (0.15, 0.3)
        if role == "loser":
            ramp = _clamp((move_no - 12) / 28.0, 0.0, 1.0)
            if self.swing_phase == 1:
                ramp = 0.4 + 0.6 * ramp  # the side destined to win first hands over a clear advantage
            if side == self.player and not endgame:
                ramp *= 0.6  # the player tends to hold the middlegame and go wrong later (T6)
            elif side != self.player and endgame:
                ramp *= 0.3  # opponents mostly go wrong before the endgame
            if score >= 150:  # ahead against the script: gives it back
                m = 2.5 + 2.5 * ramp
            elif score > -250:
                m = 1.0 + 2.5 * ramp
            else:
                m = 1.0
            return m, m
        if score >= 100:  # a draw is on the cards: the side that got ahead lets it go again
            return 3.0, 3.0
        return (0.15, 0.3) if score <= -100 else (0.2, 0.35)

    def error_floor(self, side: chess.Color) -> Optional[int]:
        """Lowest score (cp, own view) an error may leave ``side`` with: the side meant to win, or
        both sides in a game meant to be drawn, never blunder into a clearly lost position."""
        return -100 if self.role(side) in ("winner", "drawer") else None  # margin for the shallow search

    def is_intended_winner(self, side: chess.Color) -> bool:
        return self.winner is not None and side == self.winner

    @property
    def wants_draw(self) -> bool:
        return self.winner is None and self.swing_phase == 2


def _board_termination(board: chess.Board) -> Optional[tuple[str, Optional[chess.Color]]]:
    """chess.com's automatic endings (checked after every move)."""
    if board.is_checkmate():
        return "checkmate", not board.turn
    if board.is_stalemate():
        return "stalemate", None
    if board.is_insufficient_material():
        return "insufficient", None
    if board.halfmove_clock >= 100:
        return "50move", None
    if board.is_repetition(3):
        return "repetition", None
    return None


class _GamePlayer:
    def __init__(self, plan: _Plan, persona: Persona, mover: _Mover) -> None:
        self.plan = plan
        self.persona = persona
        self.mover = mover
        self.rng = random.Random(plan.game_seed)
        self.player = chess.WHITE if plan.color == "white" else chess.BLACK
        self.book = OPENINGS[plan.opening].moves
        self.live = plan.time_class != "daily"
        self.director = _Director(plan, self.player)
        self.caro = plan.color == "black" and plan.opening in CARO_KANN_LINES

    # ---- clock -----------------------------------------------------------
    def _think(self, side: chess.Color, move_no: int, in_book: bool, clock: int) -> int:
        """Tenths of a second spent on this move."""
        rng, plan = self.rng, self.plan
        if not self.live:
            hours = 0.2 if move_no == 1 else 6.0 * math.exp(rng.gauss(-0.5, 1.0))
            return int(_clamp(hours * 3600.0, 60.0, 23.8 * 3600.0) * 10)
        base, inc = float(plan.base), float(plan.inc)
        is_player = side == self.player
        director = self.director
        flagger = plan.finish == "flag" and director.swing_phase == 2 and not director.is_intended_winner(side)
        seconds_left = clock / 10.0
        if move_no == 1:
            mean = min(2.0, 0.4 + base / 300.0)
        elif move_no <= 15:
            factor = 1.0
            if is_player:
                factor = (
                    self.persona.blitz_opening_time_factor
                    if plan.time_class == "blitz"
                    else _PLAYER_OPENING_FACTOR.get(plan.time_class, 1.0)
                )
            if flagger:
                factor = max(factor, 1.6)
            frac = _OPENING_TIME_FRACTION.get(plan.time_class, 0.2) * factor
            mean = frac * base * (0.35 if in_book else 1.0) / 10.0 + 0.7 * inc
        else:
            mean = seconds_left / max(8.0, 40.0 - move_no) + 0.8 * inc
            if flagger:  # does not speed up: keeps thinking as if there were time to spare
                mean = max(3.0 * mean, 0.025 * base + 1.1 * inc)
            elif seconds_left < 0.15 * base:  # time scramble: move fast
                mean = min(mean, max(0.3, seconds_left / 20.0 + 0.6 * inc))
        sigma = 0.55
        t = mean * math.exp(rng.gauss(-sigma * sigma / 2.0, sigma))
        return max(1, int(round(t * 10.0)))

    # ---- errors ----------------------------------------------------------
    def _error_kind(
        self, side: chess.Color, score: int, move_no: int, ply: int, board: chess.Board, clock: int
    ) -> Optional[str]:
        """Draw this move's quality: None (a good move), "inaccuracy", "mistake" or "blunder"."""
        is_player = side == self.player
        endgame = _is_endgame(board)
        blunder_role, other_role = self.director.error_multiplier(side, score, move_no, endgame)
        context = 1.0  # phase, clock and opening effects shared by all error kinds
        if ply < len(self.book) + 12:
            context *= 0.5
        elif endgame and is_player:
            context *= self.persona.endgame_error_factor
            blunder_role = max(blunder_role, 0.5)  # no "safe mode" in endgames, even when winning (T6, T7)
        if self.live:
            seconds_left, base = clock / 10.0, float(self.plan.base)
            low = self.persona.low_clock_error_factor if is_player else 1.3
            if seconds_left < max(10.0, 0.10 * base):
                context *= low
            elif seconds_left < 0.20 * base:
                context *= 1.0 + 0.4 * (low - 1.0)
        if self.caro and is_player and ply < len(self.book) + 16:
            context *= 1.6  # comes out of the Caro-Kann worse than it should
        base_b, base_m, base_i = _BASE_ERROR_RATES
        pb = base_b * blunder_role * context
        pm = base_m * other_role * context
        pi = base_i * math.sqrt(other_role * context)
        total = pb + pm + pi
        if total > 0.8:
            pb, pm, pi = (x * 0.8 / total for x in (pb, pm, pi))
        u = self.rng.random()
        if u < pb:
            return "blunder"
        if u < pb + pm:
            return "mistake"
        if u < pb + pm + pi:
            return "inaccuracy"
        return None

    def _choose(
        self, board: chess.Board, cands: list[Candidate], kind: Optional[str], floor: Optional[int]
    ) -> tuple[chess.Move, float]:
        """Pick a move of the requested quality; returns it with its win-percentage drop.

        ``floor``: an erring side never picks a move that leaves it below this score (cp).
        """
        rng = self.rng
        best_wp = _win_pct(cands[0][1])
        if self.director.role(board.turn) == "loser" and cands[0][1] >= MATE_CP - 100 and rng.random() < 0.85:
            # the side that is meant to lose this game misses the mate, the classic way to let a win slip
            pool = [c for c in cands if c[1] < MATE_CP - 100]
            if not pool and not self.mover.exhaustive:
                pool = [c for c in self.mover.full(board) if c[1] < MATE_CP - 100]
            if pool:
                return pool[0][0], best_wp - _win_pct(pool[0][1])
        if kind is None:
            ok = [(m, best_wp - _win_pct(s)) for m, s in cands if best_wp - _win_pct(s) < 4.0]
            if self.director.wants_draw and board.fullmove_number >= 15 and rng.random() < 0.6:
                for m, drop in ok:  # steer towards a repetition
                    if self._repeats(board, m):
                        return m, drop
            elif self.director.is_intended_winner(board.turn) and board.halfmove_clock >= 4 and len(ok) > 1:
                fresh = [c for c in ok if not self._repeats(board, c[0])]  # the side playing to win avoids repeating
                ok = fresh or ok
            if len(ok) == 1 or rng.random() < 0.72:
                return ok[0]
            return rng.choice(ok[1:])
        kinds = ("blunder", "mistake", "inaccuracy")
        pool = cands
        for attempt in range(2):
            ref = _win_pct(pool[0][1])
            for k in kinds[kinds.index(kind) :]:
                lo, hi = _ERROR_BANDS[k]
                band = [
                    (m, ref - _win_pct(s))
                    for m, s in pool
                    if lo <= ref - _win_pct(s) < hi and (floor is None or s >= floor)
                ]
                if band:
                    return rng.choice(band)
            if attempt == 0:
                if self.mover.exhaustive:
                    break
                pool = self.mover.full(board)
        return cands[0][0], 0.0

    @staticmethod
    def _allows_mate_in_one(board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        try:
            for reply in board.legal_moves:
                if board.gives_check(reply):
                    board.push(reply)
                    mate = board.is_checkmate()
                    board.pop()
                    if mate:
                        return True
            return False
        finally:
            board.pop()

    def _avoid_mate_in_one(
        self, board: chess.Board, cands: list[Candidate], move: chess.Move, drop: float
    ) -> tuple[chess.Move, float]:
        """The one-ply heuristic cannot see a mate coming; a side that must not lose looks once."""
        if not self._allows_mate_in_one(board, move):
            return move, drop
        best_wp = _win_pct(cands[0][1])
        for m, sc in cands:
            if m != move and not self._allows_mate_in_one(board, m):
                return m, best_wp - _win_pct(sc)
        return move, drop

    @staticmethod
    def _repeats(board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        try:
            return board.is_repetition(2)
        finally:
            board.pop()

    def _resigns(self, side: chess.Color, score: int, prev: Optional[int], move_no: int) -> bool:
        if self.plan.finish == "flag" and self.director.swing_phase == 2:
            return False  # a time scramble: both sides play on, one of them hoping for the flag
        if self.plan.swing and self.director.is_intended_winner(side):
            return False  # hangs on in a lost position; the other side is about to let the win slip
        mated_soon = score <= -(MATE_CP - 100)
        lost = mated_soon or (score <= -500 and prev is not None and prev <= -500)
        p = _RESIGN_P[self.plan.time_class]
        if not lost:
            if move_no > 40 and score <= -300 and prev is not None and prev <= -300:
                p = 0.15  # a long, joyless defence
            else:
                return False
        if mated_soon:
            p *= 0.4  # many club players let the mate happen
        if self.director.is_intended_winner(side) or self.director.wants_draw:
            p *= 0.2  # stubborn: keeps hoping for a swindle
        return self.rng.random() < p

    def _agrees_draw(self, score: int, move_no: int) -> bool:
        if self.director.wants_draw and move_no >= 25 and abs(score) <= 60:
            return self.rng.random() < 0.22
        if move_no >= 100 and abs(score) <= 150:
            return self.rng.random() < 0.3
        return False

    # ---- main loop -------------------------------------------------------
    def play(self) -> _Played:
        plan, rng, board = self.plan, self.rng, chess.Board()
        self.mover.new_game(("synth", plan.game_seed))
        start_clock = plan.base * 10
        clock = {chess.WHITE: start_clock, chess.BLACK: start_clock}
        sans: list[str] = []
        moves: list[chess.Move] = []
        clocks: list[int] = []
        thinks: list[int] = []
        drops: dict[chess.Color, list[float]] = {chess.WHITE: [], chess.BLACK: []}
        last_score: dict[chess.Color, Optional[int]] = {chess.WHITE: None, chess.BLACK: None}
        end_kind, winner, final_think = "", None, 0

        while True:
            term = _board_termination(board)
            if term:
                end_kind, winner = term
                break
            side = board.turn
            ply = len(sans)
            move_no = ply // 2 + 1
            in_book = ply < len(self.book)
            think = self._think(side, move_no, in_book, clock[side])
            if self.live and think >= clock[side]:
                final_think = clock[side]
                if board.has_insufficient_material(not side):
                    end_kind, winner = "timevsinsufficient", None
                else:
                    end_kind, winner = "timeout", not side
                break
            if in_book:
                move, drop = board.parse_san(self.book[ply]), 0.0
            else:
                cands = self.mover.top(board)
                score = cands[0][1]
                self.director.observe(side, score, move_no)
                if self._resigns(side, score, last_score[side], move_no) or (move_no > 150 and score <= -200):
                    end_kind, winner, final_think = "resignation", not side, think
                    break
                if self._agrees_draw(score, move_no) or (move_no > 150 and abs(score) <= 200):
                    end_kind, winner, final_think = "agreement", None, think
                    break
                last_score[side] = score
                kind = self._error_kind(side, score, move_no, ply, board, clock[side])
                move, drop = self._choose(board, cands, kind, self.director.error_floor(side))
                if self.mover.exhaustive and self.director.error_floor(side) is not None:
                    move, drop = self._avoid_mate_in_one(board, cands, move, drop)
            sans.append(board.san(move))
            moves.append(move)
            board.push(move)
            drops[side].append(drop)
            thinks.append(think)
            if self.live:
                clock[side] = clock[side] - think + plan.inc * 10
                clocks.append(clock[side])
            else:
                clocks.append(think // 10)  # archived daily games store time spent / 10 in [%clk]

        if not self.live and plan.max_duration is not None:
            total = sum(thinks) + final_think
            budget = int(plan.max_duration * 10)
            if total > budget > 0:  # squeeze a long correspondence game into the window
                scale = budget / total
                thinks = [max(1, int(t * scale)) for t in thinks]
                final_think = int(final_think * scale)
                clocks = [t // 10 for t in thinks]

        accuracy = None
        if plan.reviewed and drops[chess.WHITE] and drops[chess.BLACK]:
            accuracy = (_game_accuracy(drops[chess.WHITE], rng), _game_accuracy(drops[chess.BLACK], rng))
        return _Played(
            index=plan.index,
            sans=sans,
            clocks=clocks,
            think=thinks,
            final_think=final_think,
            end_kind=end_kind,
            winner=None if winner is None else ("white" if winner == chess.WHITE else "black"),
            fen=board.fen(en_passant="fen"),
            tcn=encode_tcn(moves),
            accuracy=accuracy,
        )


def _game_accuracy(drops: list[float], rng: random.Random) -> float:
    """chess.com-like accuracy: blend of the arithmetic and harmonic mean of per-move accuracies."""
    accs = [_move_accuracy(d) for d in drops]
    arith = sum(accs) / len(accs)
    harmonic = len(accs) / sum(1.0 / max(a, 1.0) for a in accs)
    return round(_clamp((arith + harmonic) / 2.0 + rng.uniform(-1.5, 1.5), 5.0, 99.8), 2)


# --------------------------------------------------------------------------- TCN (chess.com move encoding)
_TCN_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?{~}(^)[_]@#$,./&-*++="
_TCN_PROMOTIONS = "qnrbkp"


def encode_tcn(moves: Sequence[chess.Move]) -> str:
    """chess.com's compact move list: two characters per move (from, to), promotions folded into 'to'."""
    out = []
    for mv in moves:
        to = mv.to_square
        if mv.promotion:
            file_step = chess.square_file(mv.to_square) - chess.square_file(mv.from_square)
            to = 64 + 3 * _TCN_PROMOTIONS.index(chess.piece_symbol(mv.promotion)) + file_step + 1
        out.append(_TCN_CHARS[mv.from_square] + _TCN_CHARS[to])
    return "".join(out)


# --------------------------------------------------------------------------- worker plumbing
_WORKER_MOVER: Optional[_Mover] = None
_WORKER_PERSONA: Optional[Persona] = None


def _make_mover(engine_path: Optional[str]) -> _Mover:
    return _EngineMover(engine_path) if engine_path else _HeuristicMover()


def _play_with(mover: _Mover, plan: _Plan, persona: Persona) -> _Played:
    try:
        return _GamePlayer(plan, persona, mover).play()
    except chess.engine.EngineTerminatedError:
        if not isinstance(mover, _EngineMover):
            raise
        mover.restart()  # a crashed engine is restarted once; the game replays identically
        return _GamePlayer(plan, persona, mover).play()


def _init_worker(engine_path: Optional[str], persona: Persona) -> None:
    global _WORKER_MOVER, _WORKER_PERSONA
    _WORKER_MOVER = _make_mover(engine_path)
    _WORKER_PERSONA = persona
    multiprocessing.util.Finalize(None, _WORKER_MOVER.close, exitpriority=10)


def _play_in_worker(plan: _Plan) -> _Played:
    assert _WORKER_MOVER is not None and _WORKER_PERSONA is not None
    return _play_with(_WORKER_MOVER, plan, _WORKER_PERSONA)


def _mp_context() -> Any:
    """fork where it is safe (Linux, single-threaded caller): fast and no re-import of ``__main__``."""
    if sys.platform.startswith("linux") and threading.active_count() == 1:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _play_all(
    plans: list[_Plan],
    persona: Persona,
    engine_path: Optional[str],
    workers: int,
    progress: Optional[Callable[[int, int], None]],
) -> dict[int, _Played]:
    total = len(plans)
    results: dict[int, _Played] = {}
    if workers <= 1 or total < 4:
        mover = _make_mover(engine_path)
        try:
            for plan in plans:
                results[plan.index] = _play_with(mover, plan, persona)
                if progress:
                    progress(len(results), total)
        finally:
            mover.close()
        return results
    ctx = _mp_context()
    pool = ctx.Pool(min(workers, total), initializer=_init_worker, initargs=(engine_path, persona))
    try:
        # longest-running (rapid) games first keeps the workers evenly loaded
        order = sorted(plans, key=lambda p: -_expected_duration(p.base if p.time_class != "daily" else 900, p.inc))
        for played in pool.imap_unordered(_play_in_worker, order, chunksize=1):
            results[played.index] = played
            if progress:
                progress(len(results), total)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    return results


# --------------------------------------------------------------------------- assembly (JSON + PGN)
_TERMINATION_TEXT = {
    "checkmate": "{w} won by checkmate",
    "resignation": "{w} won by resignation",
    "timeout": "{w} won on time",
    "agreement": "Game drawn by agreement",
    "repetition": "Game drawn by repetition",
    "stalemate": "Game drawn by stalemate",
    "insufficient": "Game drawn by insufficient material",
    "50move": "Game drawn by 50-move rule",
    "timevsinsufficient": "Game drawn by timeout vs insufficient material",
}
_LOSER_CODE = {"checkmate": "checkmated", "resignation": "resigned", "timeout": "timeout"}
_DRAW_CODE = {
    "agreement": "agreed",
    "repetition": "repetition",
    "stalemate": "stalemate",
    "insufficient": "insufficient",
    "50move": "50move",
    "timevsinsufficient": "timevsinsufficient",
}
_UUID_NODE = 0x5E1D3A9F0B17
_UUID_EPOCH_OFFSET = 12_219_292_800  # seconds between 1582-10-15 and 1970-01-01


def _uuid1_like(ts: float, rng: random.Random, node: int = _UUID_NODE) -> str:
    """A version-1 style UUID for timestamp ``ts``, like chess.com's game and player ids."""
    t = int((ts + _UUID_EPOCH_OFFSET) * 10_000_000) + rng.randrange(10_000_000)
    seq = rng.getrandbits(14)
    time_hi_version = ((t >> 48) & 0x0FFF) | 0x1000
    fields = (t & 0xFFFFFFFF, (t >> 32) & 0xFFFF, time_hi_version, (seq >> 8) | 0x80, seq & 0xFF, node)
    return str(uuid.UUID(fields=fields))


def _player_uuid(username: str) -> str:
    rng = random.Random(f"chess-insights-player:{username.lower()}")
    joined = datetime(2012, 1, 1, tzinfo=timezone.utc).timestamp() + rng.uniform(0, 12 * 365 * 86400)
    return _uuid1_like(joined, rng, node=rng.getrandbits(48))


def _fmt_clock(tenths: int) -> str:
    """'0:04:56.9' / '0:04:02' like chess.com's [%clk] (tenths shown only when non-zero)."""
    tenths = max(0, int(tenths))
    h, rem = divmod(tenths, 36_000)
    m, rem = divmod(rem, 600)
    s, t = divmod(rem, 10)
    return f"{h}:{m:02d}:{s:02d}" + (f".{t}" if t else "")


def _movetext(sans: list[str], clocks: list[int], result: str) -> str:
    parts = []
    for i, (san, clk) in enumerate(zip(sans, clocks)):
        number = i // 2 + 1
        prefix = f"{number}." if i % 2 == 0 else f"{number}..."
        parts.append(f"{prefix} {san} {{[%clk {_fmt_clock(clk)}]}}")
    parts.append(result)
    return " ".join(parts)


def _player_json(name: str, rating: int, result: str) -> dict[str, Any]:
    return {
        "rating": rating,
        "result": result,
        "@id": f"https://api.chess.com/pub/player/{name.lower()}",
        "username": name,
        "uuid": _player_uuid(name),
    }


def _game_json(
    plan: _Plan,
    played: _Played,
    persona: Persona,
    *,
    start: datetime,
    end: datetime,
    url: str,
    ratings: tuple[int, int],
) -> dict[str, Any]:
    white, black = (persona.username, plan.opponent) if plan.color == "white" else (plan.opponent, persona.username)
    if played.winner is None:
        result = "1/2-1/2"
        codes = (_DRAW_CODE[played.end_kind],) * 2
        termination = _TERMINATION_TEXT[played.end_kind]
    else:
        result = "1-0" if played.winner == "white" else "0-1"
        loser = _LOSER_CODE[played.end_kind]
        codes = ("win", loser) if played.winner == "white" else (loser, "win")
        termination = _TERMINATION_TEXT[played.end_kind].format(w=white if played.winner == "white" else black)
    opening = OPENINGS[plan.opening]
    daily = plan.time_class == "daily"
    headers = [
        ("Event", "Let's Play!" if daily else "Live Chess"),
        ("Site", "Chess.com"),
        ("Date", f"{start:%Y.%m.%d}"),
        ("Round", "-"),
        ("White", white),
        ("Black", black),
        ("Result", result),
        ("CurrentPosition", played.fen),
        ("Timezone", "UTC"),
        ("ECO", opening.eco),
        ("ECOUrl", opening.url),
        ("UTCDate", f"{start:%Y.%m.%d}"),
        ("UTCTime", f"{start:%H:%M:%S}"),
        ("WhiteElo", str(ratings[0])),
        ("BlackElo", str(ratings[1])),
        ("TimeControl", plan.time_control),
        ("Termination", termination),
        ("StartTime", f"{start:%H:%M:%S}"),
        ("EndDate", f"{end:%Y.%m.%d}"),
        ("EndTime", f"{end:%H:%M:%S}"),
        ("Link", url),
    ]
    pgn = "\n".join(f'[{k} "{v}"]' for k, v in headers) + "\n\n" + _movetext(played.sans, played.clocks, result) + "\n"
    game: dict[str, Any] = {
        "url": url,
        "pgn": pgn,
        "time_control": plan.time_control,
        "end_time": int(end.timestamp()),
        "rated": plan.rated,
    }
    if played.accuracy is not None:
        game["accuracies"] = {"white": played.accuracy[0], "black": played.accuracy[1]}
    game.update(
        {
            "tcn": played.tcn,
            "uuid": _uuid1_like(start.timestamp(), random.Random(plan.game_seed ^ 0x5EED)),
            "initial_setup": chess.STARTING_FEN,
            "fen": played.fen,
        }
    )
    if daily:
        game["start_time"] = int(start.timestamp())
    game.update(
        {
            "time_class": plan.time_class,
            "rules": "chess",
            "white": _player_json(white, ratings[0], codes[0]),
            "black": _player_json(black, ratings[1], codes[1]),
            "eco": opening.url,
        }
    )
    return game


def _duration_seconds(played: _Played) -> int:
    return max(1, math.ceil((sum(played.think) + played.final_think) / 10.0))


def _assemble(
    plans: list[_Plan], played: dict[int, _Played], persona: Persona, window_start: datetime
) -> list[dict[str, Any]]:
    # 1. timeline: games of a session follow each other; sessions never overlap
    times: dict[int, tuple[datetime, datetime]] = {}
    sessions: dict[int, list[_Plan]] = {}
    for p in plans:
        if p.session >= 0:
            sessions.setdefault(p.session, []).append(p)
        else:
            end = p.planned_start + timedelta(seconds=_duration_seconds(played[p.index]))
            times[p.index] = (p.planned_start, end)
    free_at: Optional[datetime] = None
    for sid in sorted(sessions, key=lambda s: sessions[s][0].planned_start):
        games = sessions[sid]
        t = games[0].planned_start
        if free_at is not None:
            t = max(t, free_at + timedelta(minutes=10))
        t = t.replace(microsecond=0)
        for p in games:
            end = t + timedelta(seconds=_duration_seconds(played[p.index]))
            times[p.index] = (t, end)
            t = end + timedelta(seconds=int(p.gap_after))
        free_at = times[games[-1].index][1]

    # 2. ratings evolve with the actual results, in the order games finished (one pool per time class).
    # chess.com shows post-game ratings; changes are symmetric so the pre-game ratings can be recovered.
    fallback = int(sum(persona.start_ratings.values()) / len(persona.start_ratings)) if persona.start_ratings else 1500
    rating = {tc: int(persona.start_ratings.get(tc, fallback)) for tc in TIME_CONTROLS}
    order = sorted(plans, key=lambda p: (times[p.index][1], p.index))
    shown: dict[int, tuple[int, int]] = {}
    for p in order:
        rec = played[p.index]
        mine = rating[p.time_class]
        theirs = max(100, mine + p.offset)
        score = 0.5 if rec.winner is None else float(rec.winner == p.color)
        delta = int(round(K_FACTOR * (score - _expected(theirs - mine)))) if p.rated else 0
        delta = min(max(delta, 100 - mine), theirs - 100)  # chess.com ratings never drop below 100
        mine_after, theirs_after = mine + delta, theirs - delta
        rating[p.time_class] = mine_after
        shown[p.index] = (mine_after, theirs_after) if p.color == "white" else (theirs_after, mine_after)

    # 3. ids increase with start time, like chess.com's
    base_ts = window_start.timestamp()
    games = []
    for p in plans:
        start, end = times[p.index]
        offset = int(start.timestamp() - base_ts)
        if p.time_class == "daily":
            url = f"https://www.chess.com/game/daily/{DAILY_ID_BASE + offset // 60 * 7 + p.index % 7}"
        else:
            url = f"https://www.chess.com/game/live/{LIVE_ID_BASE + offset * 12 + p.index % 12}"
        games.append(_game_json(p, played[p.index], persona, start=start, end=end, url=url, ratings=shown[p.index]))
    return games


# --------------------------------------------------------------------------- public API
def generate_archives(
    n_games: int = 400,
    *,
    seed: int = 7,
    persona: Persona = DEFAULT_PERSONA,
    engine_path: Optional[str] = "auto",
    workers: int = 0,
    end: Optional[datetime] = None,
    months: int = 12,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Synthetic monthly archives: ``{"YYYY-MM": [chess.com game objects sorted by end_time]}``.

    ``engine_path``: "auto" finds Stockfish (falling back to the heuristic mover when there is
    none), None always uses the fast heuristic mover, anything else is a Stockfish binary.
    ``workers``: parallel processes, 0 = CPUs - 1. The output depends only on the arguments
    (seed, persona, engine or not, window), never on the worker count.
    """
    if n_games <= 0:
        return {}
    end = end or DEFAULT_END
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    end = end.astimezone(timezone.utc)
    start = _months_before(end, max(1, months))
    if engine_path == "auto":
        engine_path = _find_stockfish()
    elif engine_path is not None and not Path(engine_path).is_file():
        raise FileNotFoundError(f"Stockfish binary not found: {engine_path}")
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)

    plans = _plan_games(n_games, seed, persona, start, end)
    played = _play_all(plans, persona, engine_path, workers, progress)
    games = _assemble(plans, played, persona, start)

    archives: dict[str, list[dict[str, Any]]] = {}
    for game in sorted(games, key=lambda g: (g["end_time"], g["url"])):
        key = datetime.fromtimestamp(game["end_time"], tz=timezone.utc).strftime("%Y-%m")
        archives.setdefault(key, []).append(game)
    return dict(sorted(archives.items()))


def write_archives(archives: dict[str, list[dict[str, Any]]], cache_dir: Path | str, username: str) -> Path:
    """Store the archives in the same on-disk cache ``fetch.sync`` fills; returns the player's cache root."""
    store = GameStore(cache_dir, username)
    months = sorted(parse_month(key) for key in archives)
    fetched_at = month_end(months[-1]) if months else datetime.now(timezone.utc)
    for ym in months:
        key = f"{ym[0]:04d}-{ym[1]:02d}"
        store.save_month(ym, archives[key], etag=None, fetched_at=fetched_at, complete=True)
    store.root.mkdir(parents=True, exist_ok=True)
    return store.root
