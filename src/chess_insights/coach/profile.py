"""Motif profile: which patterns you miss and allow, against your opponents in the same games (C2). Owner: drills.

Every mistake or blunder in the engine-analysed games (both sides, from ``deep.profile_lines``) comes with a short
best line and the engine's refutation of the move played. The motif detectors (``motifs.detect_line``) name the
patterns along them:

* **missed**: a pattern in the best line that the player who went wrong would have carried out (a fork they
  had and didn't play);
* **allowed**: a pattern in the refutation that the other side carries out (the fork their move let in).

Each error counts once per pattern, and the counts are compared per 100 moves of each side in the same games, so
your opponents (rating-matched, same positions, same clock) are the benchmark, as everywhere in the Engine review.

The table and chart are observations. The only claims are "you miss (allow) forks more (less) often than your
opponents", and they are built so that making more errors than your opponents cannot turn into a pattern claim:

* A pattern is mostly there to be missed because the other side's last move allowed it (a piece left hanging, a
  fork let in), so a player who blunders more hands their opponents more patterns to miss. "Missed" claims are
  therefore judged **per chance**: of the X's your opponents' mistakes and blunders left you (their errors whose
  refutation carries X), the share you missed (your next move was a mistake or blunder too), against the same
  share for your opponents, judged on the smaller side's games with such a chance. The share must also stand out
  against the share of your other tactical chances you missed, more than it does for your opponents
  (:func:`ratio_contrast_test`): missing more chances of every kind, or a pattern that merely comes with another
  one, is not one claim per pattern.
* "Allowed" claims need the pattern to be more common per move *and* a larger share of your own mistakes and
  blunders than of theirs, where the errors made right after one of the other side's (a missed chance, whose
  number depends on the other side's errors) are left out of both.

Each test has games as the unit (the clustered tests of ``analysis.engine_stats``, pairing you and your opponent
in each game), is Benjamini-Hochberg adjusted across every pattern and both kinds, and must pass
``stats.significance`` at ``stats.STRICT_ALPHA`` with minimum samples and a ratio of at least ``MOTIF_RATIO``
(the comparison with the rest is not truncated at 1). Only patterns whose detector passed the precision gate
(``motifs.GATED_THEMES``) are named at all.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence

import chess

from ..analysis.engine_stats import Analysed, clustered_share_test, join
from ..models import Chart, Coaching, Diagram, Game, Insight, Line, Mark, ModuleResult, Motif, Table
from ..stats import STRICT_ALPHA, MeanTest, bh_adjust, clamp, normal_cdf, pct, significance
from ..visuals import (
    FORMAT_NAMES,
    MIN_FORMAT_GAMES,
    comparison_chart,
    format_counts,
    format_text,
    line_strip,
    move_label,
    parse_move,
    position_diagram,
)
from .config import CoachConfig

if TYPE_CHECKING:
    from ..context import AnalysisContext
    from .deep import ProfileError

KEYS = ("engine", "engine_stats")  # the Engine review section's ModuleResult.key
SIDES = ("you", "opponent")
KINDS = ("missed", "allowed")
MIN_MOTIF_GAMES = 30  # games in which either side had the pattern (of that kind), before a claim
MIN_MOTIF_EVENTS = 20  # times it turned up, both sides together, before a claim
MOTIF_RATIO = 1.3  # your rate vs your opponents' (the inverse for a strength), as engine_stats' tactic_ratio
MIN_CONFIDENCE = 0.5  # insights.MIN_CONFIDENCE
CHART_THEMES = 6  # patterns in the overview chart (the table lists every named one)
MAX_EXAMPLES = 5
PUZZLE_LINE_PLIES = 8  # best lines kept for the puzzle export
MAX_STRIP_FRAMES = 8  # small boards in a pattern's strip (at least 4, and up to the pattern itself)

# Lichess theme -> (singular, plural), in the report's words
MOTIF_NAMES: dict[str, tuple[str, str]] = {
    "fork": ("fork", "forks"),
    "pin": ("pin", "pins"),
    "skewer": ("skewer", "skewers"),
    "hangingPiece": ("hanging piece", "hanging pieces"),
    "discoveredAttack": ("discovered attack", "discovered attacks"),
    "discoveredCheck": ("discovered check", "discovered checks"),
    "doubleCheck": ("double check", "double checks"),
    "trappedPiece": ("trapped piece", "trapped pieces"),
    "backRankMate": ("back-rank mate", "back-rank mates"),
    "smotheredMate": ("smothered mate", "smothered mates"),
    "mate": ("forced mate", "forced mates"),
    "mateIn1": ("mate in 1", "mates in 1"),
    "mateIn2": ("mate in 2", "mates in 2"),
    "mateIn3": ("mate in 3", "mates in 3"),
    "mateIn4": ("mate in 4", "mates in 4"),
    "mateIn5": ("mate in 5", "mates in 5"),
    "deflection": ("deflection", "deflections"),
    "attraction": ("attraction (decoy)", "attractions (decoys)"),
    "overloading": ("overloaded defender", "overloaded defenders"),
    "advancedPawn": ("advanced-pawn tactic", "advanced-pawn tactics"),
}

# What to look for before each move, per pattern (the fallback covers the rest)
HABITS: dict[str, str] = {
    "fork": "Before each move, look for a square where one piece attacks two targets at once (king, queen, a loose "
    "piece), for you and for your opponent.",
    "pin": "Before each move, look along every line through a king or queen: a piece in front of it can't move "
    "freely.",
    "skewer": "Before each move, look along every line through your king and queen: a check or attack there can win "
    "what stands behind them.",
    "hangingPiece": "Before letting go of a piece, count what attacks and defends every piece you leave behind.",
    "discoveredAttack": "Before each move, look at the pieces standing in front of a rook, bishop or queen: moving "
    "them uncovers an attack.",
    "backRankMate": "Keep an escape square for your king (a pawn move such as h3), and check your opponent's back "
    "rank for mates.",
}
_DEFAULT_HABIT = "Before each move, list every check, capture and threat for both sides."


def motif_name(theme: str, plural: bool = False) -> str:
    """'fork' / 'forks', 'discovered attack' ...; unknown themes are split from camelCase."""
    if theme in MOTIF_NAMES:
        return MOTIF_NAMES[theme][1 if plural else 0]
    words = re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", theme).lower()
    return f"{words}s" if plural else words


def training_url(theme: str) -> str:
    return f"https://lichess.org/training/{theme}"


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


# --------------------------------------------------------------------------- counting
CHANCE_MISSED = "chance-missed"  # count kind: errors right after the other side's error that left the pattern
OWN_ALLOWED = "own-allowed"  # count kind: "allowed" among the errors that did not follow one of the other side's


@dataclass
class GameCounts:
    """One engine-analysed game: each side's moves, profiled errors and pattern counts (the unit of every test).

    Pairs are (yours, your opponent's). ``counts`` has, per (kind, theme):

    * ``"missed"`` / ``"allowed"``: errors whose best line / refutation carries the pattern (the table and chart);
    * ``CHANCE_MISSED``: errors made right after one of the other side's errors whose refutation carried the
      pattern: chances left by the other side and missed (a side's ``"allowed"`` count is the *other* side's
      chances of that pattern);
    * ``OWN_ALLOWED``: ``"allowed"`` among the errors that are not such follow-ups.

    ``chances``: the other side's errors with a refutation line (the chances it left you, and you left it);
    ``followups``: errors made right after one of those (a chance missed, whatever the pattern);
    ``pattern_chances`` / ``pattern_followups``: the same for the chances whose refutation carries a named pattern
    (the tactical chances each pattern's miss rate is compared with).
    """

    game_id: str
    time_class: str
    moves: tuple[int, int]
    errors: list[int] = field(default_factory=lambda: [0, 0])  # mistakes and blunders with engine lines
    counts: dict[tuple[str, str], list[int]] = field(default_factory=dict)  # (kind, theme) -> errors with it
    url: str = ""
    chances: list[int] = field(default_factory=lambda: [0, 0])
    followups: list[int] = field(default_factory=lambda: [0, 0])
    pattern_chances: list[int] = field(default_factory=lambda: [0, 0])
    pattern_followups: list[int] = field(default_factory=lambda: [0, 0])

    def count(self, kind: str, theme: str) -> tuple[int, int]:
        pair = self.counts.get((kind, theme))
        return (pair[0], pair[1]) if pair else (0, 0)

    def own_errors(self, side: int) -> int:
        """Errors of ``side`` (0 = you) that did not come right after a chance the other side left."""
        return self.errors[side] - self.followups[side]


@dataclass
class MotifEvent:
    """One error in which a pattern was missed or allowed (for example games and boards)."""

    game: Game
    error: "ProfileError"
    kind: str  # "missed" | "allowed"
    motif: Motif
    after_chance: bool = False  # "missed": the other side's error on the ply before had just left this pattern

    @property
    def side(self) -> str:
        return self.error.side


def _detect(detect: Callable[..., list[Motif]], line: Line, role: str, previous_fen: Optional[str]) -> list[Motif]:
    """``detect(line, role, previous_fen)``; a detector without the optional third argument gets two."""
    if previous_fen is None:
        return list(detect(line, role) or [])
    try:
        return list(detect(line, role, previous_fen) or [])
    except TypeError:
        return list(detect(line, role) or [])


def _previous_positions(errors: Sequence["ProfileError"], games: dict[str, Game]) -> dict[tuple[str, int], str]:
    """(game id, ply) -> the position before the move that led to the error's position, for the errors that don't
    carry it (``ProfileError.previous_fen``): each game replayed once for all its errors."""
    from .deep import previous_fens

    wanted: dict[str, set[int]] = {}
    for e in errors:
        if getattr(e, "previous_fen", None) is None and e.game_id in games and e.ply > 0:
            wanted.setdefault(e.game_id, set()).add(e.ply)
    return {(gid, ply): fen for gid, plies in wanted.items() for ply, fen in previous_fens(games[gid], plies).items()}


def collect(
    records: Sequence[Analysed],
    errors: Iterable["ProfileError"],
    detect: Callable[..., list[Motif]],
    notes: Optional[list[str]] = None,
    gated: Optional[Iterable[str]] = None,
) -> tuple[list[GameCounts], list[MotifEvent]]:
    """Per-game counts and the individual events, from the profiled errors of the analysed games.

    Errors from games outside ``records`` (not analysed, too short, filtered out) are ignored; each error counts
    once per pattern and kind, however often the detector reports it along the line. A line the detectors fail on
    counts as having no pattern (and a note says how many). Best lines are read with the position before the move
    that led to the error's position (``detect(line, "best", previous_fen)``: a capture that ends a trade is no
    hanging piece), from ``ProfileError.previous_fen`` or, when an error lacks it, from one replay of its game;
    a detector that takes only ``(line, role)`` is called with two arguments.

    An error with a refutation line is a chance for the other side, carrying the patterns its refutation shows;
    an error of the other side on the very next ply is that chance missed (``GameCounts.followups`` and the
    ``CHANCE_MISSED`` counts; its "missed" events are marked ``after_chance`` for the patterns the chance had).
    A chance whose refutation carries a pattern in ``gated`` (any pattern when None) is a pattern chance.
    """
    named = None if gated is None else frozenset(gated)
    games: list[GameCounts] = []
    by_id: dict[str, tuple[Game, GameCounts]] = {}
    for r in records:
        mine = sum(1 for p in r.plies if p.is_user)
        gc = GameCounts(r.game.game_id, r.game.time_class, (mine, len(r.plies) - mine), url=r.game.url)
        games.append(gc)
        by_id[r.game.game_id] = (r.game, gc)
    errors = list(errors)
    previous = _previous_positions(errors, {gid: g for gid, (g, _) in by_id.items()})
    seen: set[tuple[str, int]] = set()
    failed = 0
    # first pass: each error once, with the patterns along its lines (allowed None: no refutation line)
    read: list[tuple["ProfileError", Game, GameCounts, int, dict[str, Motif], Optional[dict[str, Motif]]]] = []
    for e in errors:
        entry = by_id.get(e.game_id)
        if entry is None or e.side not in SIDES or (e.game_id, e.ply) in seen:
            continue
        seen.add((e.game_id, e.ply))
        game, gc = entry
        before = getattr(e, "previous_fen", None) or previous.get((e.game_id, e.ply))
        # best line: the patterns the player who went wrong would have carried out ("you" = the side to move);
        # refutation: the ones the other side carries out after the move
        patterns: dict[str, Optional[dict[str, Motif]]] = {}
        for kind, line, role, carrier in (("missed", e.best_line, "best", "you"),
                                          ("allowed", e.refutation, "refutation", "opponent")):
            if line is None:
                patterns[kind] = None
                continue
            try:
                detected = _detect(detect, line, role, before if role == "best" else None)
            except Exception:  # noqa: BLE001 — one odd line must not sink the whole profile
                failed += 1
                detected = []
            found: dict[str, Motif] = {}
            for m in detected:
                if m.theme and (not m.side or m.side == carrier):
                    found.setdefault(m.theme, m)
            patterns[kind] = found
        read.append((e, game, gc, SIDES.index(e.side), patterns["missed"] or {}, patterns["allowed"]))
    # the chance each error with a refutation left the other side: (game, ply) -> (side that erred, its patterns)
    chance_at = {(e.game_id, e.ply): (s, allowed) for e, _, _, s, _, allowed in read if allowed is not None}
    events: list[MotifEvent] = []
    for e, game, gc, s, missed, allowed in read:
        gc.errors[s] += 1
        chance = chance_at.get((e.game_id, e.ply - 1))
        followup = chance is not None and chance[0] != s  # right after a chance the other side left: missed it
        left: dict[str, Motif] = chance[1] if followup else {}
        if followup:
            gc.followups[s] += 1
            gc.pattern_followups[s] += _named_in(left, named)
            for theme in left:
                gc.counts.setdefault((CHANCE_MISSED, theme), [0, 0])[s] += 1
        if allowed is not None:
            gc.chances[1 - s] += 1
            gc.pattern_chances[1 - s] += _named_in(allowed, named)
        for theme, m in missed.items():
            gc.counts.setdefault(("missed", theme), [0, 0])[s] += 1
            events.append(MotifEvent(game, e, "missed", m, after_chance=theme in left))
        for theme, m in (allowed or {}).items():
            gc.counts.setdefault(("allowed", theme), [0, 0])[s] += 1
            if not followup:
                gc.counts.setdefault((OWN_ALLOWED, theme), [0, 0])[s] += 1
            events.append(MotifEvent(game, e, "allowed", m))
    if failed and notes is not None:
        lines = "1 engine line" if failed == 1 else f"{failed} engine lines"
        notes.append(f"Motif profile: the pattern detectors failed on {lines} (counted without a pattern).")
    return games, events


def _named_in(themes: Iterable[str], named: Optional[frozenset[str]]) -> bool:
    """Whether ``themes`` has a named pattern (any, when ``named`` is None)."""
    return any(named is None or t in named for t in themes)


def _sums(games: Sequence[GameCounts], kind: str, theme: str) -> tuple[int, int, int, int]:
    """(your count, your moves, their count, their moves)."""
    x1 = x2 = m1 = m2 = 0
    for g in games:
        a, b = g.count(kind, theme)
        x1, x2, m1, m2 = x1 + a, x2 + b, m1 + g.moves[0], m2 + g.moves[1]
    return x1, m1, x2, m2


def _per100(x: int, moves: int) -> Optional[float]:
    return 100.0 * x / moves if moves else None


def themes_seen(games: Sequence[GameCounts]) -> list[str]:
    return sorted({theme for g in games for (_, theme) in g.counts})


_BASE_COUNTS = ("you_missed", "opp_missed", "you_allowed", "opp_allowed")


def profile_counts(games: Sequence[GameCounts], themes: Iterable[str]) -> dict[str, dict[str, int]]:
    """{theme: {"you_missed", "opp_missed", "you_allowed", "opp_allowed", "you_chances", "you_missed_chances",
    "opp_chances", "opp_missed_chances"}} over every game. Your chances are the patterns your opponents' errors
    left you (their "allowed"), and "missed_chances" how many of them the side missed (its next move was an error
    too)."""
    out = {}
    for theme in themes:
        mx, _, ox, _ = _sums(games, "missed", theme)
        ax, _, bx, _ = _sums(games, "allowed", theme)
        cx, _, dx, _ = _sums(games, CHANCE_MISSED, theme)
        out[theme] = {"you_missed": mx, "opp_missed": ox, "you_allowed": ax, "opp_allowed": bx,
                      "you_chances": bx, "you_missed_chances": cx, "opp_chances": ax, "opp_missed_chances": dx}
    return out


# --------------------------------------------------------------------------- table and chart
def _named(games: Sequence[GameCounts], gated: Iterable[str]) -> list[str]:
    """Named patterns that turned up at all, most frequent first."""
    counts = profile_counts(games, [t for t in themes_seen(games) if t in set(gated)])
    total = {t: sum(c[k] for k in _BASE_COUNTS) for t, c in counts.items()}
    return sorted((t for t in counts if total[t]), key=lambda t: (-total[t], t))


def _games_note(games: Sequence[GameCounts], formats: dict[str, int]) -> str:
    you = sum(g.moves[0] for g in games)
    them = sum(g.moves[1] for g in games)
    mix = ", ".join(f"{n} {tc}" for tc, n in formats.items())
    return (
        f"Per 100 moves of each side ({you:,} of yours, {them:,} of your opponents') in {len(games)} engine-analysed "
        f"games" + (f" ({mix})" if mix else "") + "."
    )


_KIND_NOTE = (
    "Missed: the engine's better move started the pattern for the player who went wrong. Allowed: after a mistake "
    "or blunder, the engine's refutation uses the pattern against the player who made it. Each mistake or blunder "
    "counts once per pattern."
)
# the profile's numbers are not tested as they stand: a gap between the bars is a finding only when listed as one
UNTESTED_NOTE = (
    "These are observations, not findings: a difference here counts only when this section lists it as a finding. "
    "Most patterns are there to be missed because the other side's mistake left them, so the 'missed' numbers also "
    "rise and fall with how often the other side errs."
)


def profile_table(games: Sequence[GameCounts], gated: Iterable[str], formats: dict[str, int]) -> Optional[Table]:
    """Pattern, you missed, they missed, you allowed, they allowed (per 100 moves), and a practice link."""
    rows = []
    for theme in _named(games, gated):
        mx, mm, ox, om = _sums(games, "missed", theme)
        ax, am, bx, bm = _sums(games, "allowed", theme)
        rows.append([motif_name(theme).capitalize(), _per100(mx, mm), _per100(ox, om), _per100(ax, am),
                     _per100(bx, bm), training_url(theme)])
    if not rows:
        return None
    return Table(
        title="Tactical patterns in mistakes and blunders",
        columns=["Pattern", "You missed", "Opponents missed", "You allowed", "Opponents allowed", "Practice"],
        rows=rows,
        formats=["text", "float2", "float2", "float2", "float2", "url"],
        note=f"{_games_note(games, formats)} {_KIND_NOTE} Practice: Lichess puzzles of that theme. {UNTESTED_NOTE}",
        key_columns=[0, 1, 2, 3, 4],
    )


def profile_chart(games: Sequence[GameCounts], gated: Iterable[str], formats: dict[str, int]) -> Optional[Chart]:
    """You vs your opponents per 100 moves, for the most frequent named patterns (missed and allowed)."""
    themes = _named(games, gated)[:CHART_THEMES]
    if not themes:
        return None
    labels, you, them, rows = [], [], [], []
    for theme in themes:
        for kind in KINDS:
            x1, m1, x2, m2 = _sums(games, kind, theme)
            labels.append(f"{motif_name(theme).capitalize()} · {kind}")
            you.append(_per100(x1, m1))
            them.append(_per100(x2, m2))
            rows.append([motif_name(theme).capitalize(), kind, x1, _per100(x1, m1), x2, _per100(x2, m2)])
    return comparison_chart(
        "Tactical patterns you miss and allow, per 100 moves",
        labels,
        [("You", you), ("Opponents", them)],
        value_format="float2",
        note=f"{_games_note(games, formats)} {_KIND_NOTE} {UNTESTED_NOTE}",
        table=Table(
            title="Tactical patterns: counts",
            columns=["Pattern", "Kind", "You", "You /100 moves", "Opponents", "Opponents /100 moves"],
            rows=rows,
            formats=["text", "text", "int", "float2", "int", "float2"],
        ),
    )


# --------------------------------------------------------------------------- claims
@dataclass
class MotifTest:
    """The two tests behind one possible claim, games as the unit.

    ``missed``: ``rate`` is the share of the pattern's chances you missed minus your opponents' share (your chances:
    the patterns their errors left you); ``share`` whether that stands out against the other tactical chances
    (:func:`ratio_contrast_test`). ``allowed``: ``rate`` is your rate per move minus theirs, ``share`` the pattern's
    share of your own errors (not counting the ones right after a chance) minus theirs."""

    theme: str
    kind: str
    rate: MeanTest
    share: MeanTest
    games_with: int  # games in which either side had it (a chance of it, for "missed")
    events: int  # both sides (chances, for "missed")
    n: int = 0  # the sample size the claim rule judges: for "missed", the smaller side's games with a chance of it
    p_rate: float = 1.0  # BH-adjusted
    p_share: float = 1.0


def ratio_contrast_test(rows: Sequence[tuple[int, int, int, int, int, int, int, int]]) -> MeanTest:
    """Whether one share stands out against another more for you than for your opponents: a test of
    log(a1 / b1) - log(a2 / b2), games as the unit.

    ``rows`` holds per game (x_a1, n_a1, x_b1, n_b1, x_a2, n_a2, x_b2, n_b2), with a = x_a / n_a (e.g. the forks
    you were left that you missed) and b = x_b / n_b (the other tactical chances you missed); 1 = you, 2 = your
    opponents. The four shares are ratio estimators with half an event added to each count; the variance is the
    clustered (delta-method) one, never below that of four independent binomial shares. The mean is > 0 when the
    first share stands out more for you."""
    rows = [r for r in rows if any(r)]
    n = len(rows)
    events = [sum(r[2 * k] for r in rows) for k in range(4)]
    totals = [sum(r[2 * k + 1] for r in rows) for k in range(4)]
    if n < 2 or min(totals) <= 0:
        return MeanTest(n, 0.0, float("inf"), 0.0, 1.0)
    shares = [(x + 0.5) / (t + 1.0) for x, t in zip(events, totals)]
    signs = (1.0, -1.0, -1.0, 1.0)
    mean = sum(sign * math.log(p) for sign, p in zip(signs, shares))
    residuals = [
        sum(sign * (r[2 * k] - shares[k] * r[2 * k + 1]) / (shares[k] * (totals[k] + 1.0))
            for k, sign in enumerate(signs))
        for r in rows
    ]
    variance = n / (n - 1) * sum(v * v for v in residuals)
    floor = sum(max(0.0, 1.0 - p) / (p * (t + 1.0)) for p, t in zip(shares, totals))
    variance = max(variance, floor)
    if variance <= 0.0:
        return MeanTest(n, mean, float("inf"), 0.0, 1.0)
    se = math.sqrt(variance)
    z = mean / se
    return MeanTest(n, mean, se, z, 2.0 * (1.0 - normal_cdf(abs(z))))


def _chances(games: Sequence[GameCounts], theme: str) -> tuple[int, int, int, int]:
    """(chances of ``theme`` you missed, chances your opponents' errors left you, theirs missed, theirs)."""
    x1, _, x2, _ = _sums(games, CHANCE_MISSED, theme)
    allowed_you, _, allowed_them, _ = _sums(games, "allowed", theme)
    return x1, allowed_them, x2, allowed_you  # your chances are your opponents' "allowed", and theirs yours


def _missed_rows(g: GameCounts, theme: str) -> tuple[int, int, int, int, int, int, int, int]:
    """One game's (chances of ``theme`` you missed, your chances of it, your other tactical chances missed, your
    other tactical chances), then your opponents' four."""
    out: list[int] = []
    for side in (0, 1):
        missed, chances = g.count(CHANCE_MISSED, theme)[side], g.count("allowed", theme)[1 - side]
        out += [missed, chances, max(0, g.pattern_followups[side] - missed), max(0, g.pattern_chances[side] - chances)]
    return tuple(out)  # type: ignore[return-value]


def motif_tests(
    games: Sequence[GameCounts], gated: Iterable[str], min_games: int = MIN_MOTIF_GAMES,
    min_events: int = MIN_MOTIF_EVENTS,
) -> list[MotifTest]:
    """Every named pattern x {missed, allowed} with enough data, BH-adjusted across all of them (each test family:
    the rates, and the shares)."""
    tests = []
    for theme in sorted(set(gated)):
        # missed, per chance: of the patterns the other side's errors left you (its "allowed"), the ones you missed;
        # judged on the smaller side's games with a chance of it (a two-group comparison)
        rows = [_missed_rows(g, theme) for g in games]
        games_with = sum(1 for r in rows if r[1] or r[5])
        events = sum(r[1] + r[5] for r in rows)
        n = min(sum(1 for r in rows if r[1]), sum(1 for r in rows if r[5]))
        if n >= min_games and events >= min_events:
            rate = clustered_share_test([(r[0], r[1]) for r in rows], [(r[4], r[5]) for r in rows])
            tests.append(MotifTest(theme, "missed", rate, ratio_contrast_test(rows), games_with, events, n))
        # allowed, per move and as a share of the errors that were not a missed chance
        pairs = [g.count("allowed", theme) for g in games]
        own = [g.count(OWN_ALLOWED, theme) for g in games]
        games_with = sum(1 for a, b in pairs if a or b)
        events = sum(a + b for a, b in pairs)
        if games_with >= min_games and events >= min_events:
            rate = clustered_share_test([(a, g.moves[0]) for (a, _), g in zip(pairs, games)],
                                        [(b, g.moves[1]) for (_, b), g in zip(pairs, games)])
            share = clustered_share_test([(a, g.own_errors(0)) for (a, _), g in zip(own, games)],
                                         [(b, g.own_errors(1)) for (_, b), g in zip(own, games)])
            tests.append(MotifTest(theme, "allowed", rate, share, games_with, events, games_with))
    for t, p in zip(tests, bh_adjust([t.rate.p_value for t in tests])):
        t.p_rate = p
    for t, p in zip(tests, bh_adjust([t.share.p_value for t in tests])):
        t.p_share = p
    return tests


def _rate_ratio(x1: int, n1: int, x2: int, n2: int) -> Optional[float]:
    """(x1 / n1) / (x2 / n2) with half an event added to each count (as engine_stats)."""
    if n1 <= 0 or n2 <= 0:
        return None
    return ((x1 + 0.5) / n1) / ((x2 + 0.5) / n2)


def _share(x: int, n: int) -> Optional[float]:
    return x / n if n else None


def _format_chart(games: Sequence[GameCounts], kind: str, theme: str, formats: dict[str, int]) -> Chart:
    """You vs your opponents, all games, then each format with enough games (when there are two+): the share of the
    pattern's chances missed ("missed"), or per 100 moves ("allowed")."""
    groups: list[tuple[str, list[GameCounts]]] = [("All games", list(games))]
    split = [(tc, [g for g in games if g.time_class == tc]) for tc, n in formats.items() if n >= MIN_FORMAT_GAMES]
    if len(split) >= 2:
        groups += [(FORMAT_NAMES.get(tc, tc), gs) for tc, gs in split]
    elif len(split) == 1 and sum(1 for n in formats.values() if n) == 1:  # one format only: name it
        groups = [(FORMAT_NAMES.get(split[0][0], split[0][0]), list(games))]
    mix = ", ".join(f"{n} {tc}" for tc, n in formats.items() if n)
    note = f"Engine-analysed games: {mix}." if mix else f"{len(games)} engine-analysed games."
    if len(groups) > 1:
        note += f" Formats with fewer than {MIN_FORMAT_GAMES} analysed games have no bar of their own."
    labels, you, them, rows = [], [], [], []
    plural = motif_name(theme, True)
    if kind == "missed":
        for label, gs in groups:
            x1, c1, x2, c2 = _chances(gs, theme)
            labels.append(label)
            you.append(_share(x1, c1))
            them.append(_share(x2, c2))
            rows.append([label, len(gs), c1, x1, _share(x1, c1), c2, x2, _share(x2, c2)])
        return comparison_chart(
            f"{plural.capitalize()} missed, share of the chances",
            labels,
            [("You", you), ("Opponents", them)],
            value_format="pct",
            note=f"{note} A chance: the other side's mistake or blunder left the {motif_name(theme)} (the engine's "
            "refutation of it uses one). Missed: the next move was a mistake or blunder too.",
            table=Table(
                title=f"{plural.capitalize()} missed by format",
                columns=["Games", "Analysed games", "Your chances", "You missed", "You missed %", "Their chances",
                         "They missed", "They missed %"],
                rows=rows,
                formats=["text", "int", "int", "int", "pct", "int", "int", "pct"],
                key_columns=[0, 4, 7],
            ),
        )
    for label, gs in groups:
        x1, m1, x2, m2 = _sums(gs, kind, theme)
        labels.append(label)
        you.append(_per100(x1, m1))
        them.append(_per100(x2, m2))
        rows.append([label, len(gs), x1, _per100(x1, m1), x2, _per100(x2, m2)])
    return comparison_chart(
        f"{plural.capitalize()} allowed, per 100 moves",
        labels,
        [("You", you), ("Opponents", them)],
        value_format="float2",
        note=note,
        table=Table(
            title=f"{plural.capitalize()} allowed by format",
            columns=["Games", "Analysed games", "You", "You /100 moves", "Opponents", "Opponents /100 moves"],
            rows=rows,
            formats=["text", "int", "int", "float2", "int", "float2"],
        ),
    )


def _label(fen: str, move: str) -> str:
    try:
        board = chess.Board(fen)
    except ValueError:
        return move
    m = parse_move(board, move)
    return move_label(board, m) if m is not None else move


def _line_text(line: Optional[Line], max_moves: int = 6) -> str:
    """'12.Nd5 exd5 13.Qxe7': the first moves of a line, numbered, up to its first illegal move."""
    if line is None or not line.fen:
        return ""
    try:
        board = chess.Board(line.fen)
    except ValueError:
        return ""
    parts = []
    for i, uci in enumerate(line.moves_uci[:max_moves]):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        label = move_label(board, move)
        parts.append(label if i == 0 or board.turn == chess.WHITE else board.san(move))
        board.push(move)
    return " ".join(parts)


def _marks(motif: Motif) -> list[Mark]:
    return [Mark(sq, "attacker" if i == 0 else "target") for i, sq in enumerate(motif.squares or []) if sq]


def event_diagram(event: MotifEvent) -> Diagram:
    """The position before the error: the move played red, the engine's move green, and the line that follows
    (as far as the pattern, with its squares marked). Your errors for weaknesses, your opponents' for strengths; the
    board always has your side at the bottom."""
    e, g, theme = event.error, event.game, event.motif.theme
    best = e.best_line.moves_uci[0] if e.best_line and e.best_line.moves_uci else None
    played, best_label = _label(e.fen, e.played_uci), _label(e.fen, best) if best else ""
    name = motif_name(theme)
    mine = event.side != "opponent"
    who, whose = ("You", "you") if mine else ("Your opponent", "your opponent")
    if event.kind == "missed":
        line = e.best_line
        title = (f"The {name} you missed" if mine else f"A {name} your opponent missed") + (
            f": {best_label} instead of {played}" if best_label else f" with {played}")
        strip_title = f"What {best_label} starts ({name})" if best_label else f"The engine's line ({name})"
    else:
        line = e.refutation
        title = f"The {name} you allowed after {played}" if mine else f"The {name} your opponent allowed after {played}"
        strip_title = f"What {played} allowed ({name})"
    text = _line_text(line)
    caption = (
        f"{who} played {played} (red)" + (f"; {best_label} (green) was better" if best_label else "")
        + f". It cost {whose} {e.drop:.0f} percentage points of winning chances."
        + (f" The engine's line: {text}." if text else "")
    )
    diagram = position_diagram(title, e.fen, orientation=g.color, played=e.played_uci, best=best, caption=caption,
                               link=g.url, time_class=g.time_class)
    if line is not None and line.moves_uci:
        marks = [[] for _ in line.moves_uci]
        if 0 <= event.motif.ply < len(marks):
            marks[event.motif.ply] = _marks(event.motif)
        frames = max(4, min(event.motif.ply + 1, MAX_STRIP_FRAMES))  # far enough to show the pattern
        strip = line_strip(strip_title, line.fen or e.fen, line.moves_uci, max_frames=frames, marks=marks)
        if strip.frames:
            diagram.strips.append(strip)
    return diagram


def _examples(events: Sequence[MotifEvent], side: str, kind: str, theme: str) -> list[MotifEvent]:
    """The events of one side, pattern and kind, costliest first (then most recent); for "missed", the chances the
    other side had just left (what the claim counts) when there are any."""
    picked = [ev for ev in events if ev.side == side and ev.kind == kind and ev.motif.theme == theme]
    if kind == "missed":
        picked = [ev for ev in picked if ev.after_chance] or picked
    return sorted(picked, key=lambda ev: (-(ev.error.drop or 0.0), -ev.game.end_time.timestamp(), ev.error.ply))


def _urls(events: Sequence[MotifEvent]) -> list[str]:
    out: list[str] = []
    for ev in events:
        if ev.game.url and ev.game.url not in out:
            out.append(ev.game.url)
    return out[:MAX_EXAMPLES]


def _study(theme: str, kind: str) -> list[str]:
    name, plural = motif_name(theme), motif_name(theme, True)
    replay = (
        f"Replay the linked games at the moment of the miss and find the {name} yourself before looking at the "
        "engine's line." if kind == "missed"
        else f"Replay the linked games at the move that allowed the {name} and say what it left open."
    )
    return [
        f"Drill {plural} with themed puzzles at your level: {training_url(theme)}.",
        replay,
        HABITS.get(theme, _DEFAULT_HABIT),
    ]


def _direction(t: MotifTest, r: float, excess: float, ratio: float) -> Optional[str]:
    """"weakness" / "strength" when both tests agree on the direction and both ratios reach ``ratio``."""
    if t.rate.mean > 0 and t.share.mean > 0 and r >= ratio and excess >= ratio:
        return "weakness"
    if t.rate.mean < 0 and t.share.mean < 0 and 1.0 / r >= ratio and 1.0 / excess >= ratio:
        return "strength"
    return None


def motif_claims(
    games: Sequence[GameCounts],
    gated: Iterable[str],
    events: Sequence[MotifEvent] = (),
    *,
    formats: Optional[dict[str, int]] = None,
    min_games: int = MIN_MOTIF_GAMES,
    min_events: int = MIN_MOTIF_EVENTS,
    ratio: float = MOTIF_RATIO,
) -> tuple[list[Insight], dict[str, Any]]:
    """Strengths and weaknesses "you miss / allow <pattern> more (less) often than your opponents", and the stats
    of every test run (for the JSON export).

    "Missed" compares the share of the pattern's chances missed (per chance, so that the other side's error rate
    doesn't count) and must stand out against the share of the other tactical chances missed (the pattern chances
    without it: ``GameCounts.pattern_chances``, counted by ``collect`` with the same ``gated``); "allowed" compares
    the rate per move and the share of your own errors (not the missed chances). Neither comparison with the rest
    is truncated at 1: erring more (or less) overall never becomes a pattern claim in the other direction."""
    formats = dict(formats or {})
    n_games = len(games)
    k1, k2 = sum(g.pattern_chances[0] for g in games), sum(g.pattern_chances[1] for g in games)  # tactical chances
    f1, f2 = sum(g.pattern_followups[0] for g in games), sum(g.pattern_followups[1] for g in games)  # ... missed
    o1, o2 = sum(g.own_errors(0) for g in games), sum(g.own_errors(1) for g in games)
    mix = format_text(formats)
    where = f"In {n_games} engine-analysed games" + (f" ({mix})" if mix else "")
    insights: list[Insight] = []
    stats: dict[str, Any] = {}
    for t in motif_tests(games, gated, min_games, min_events):
        name, plural = motif_name(t.theme), motif_name(t.theme, True)
        ok, confidence = significance(t.rate, min_games, t.p_rate, alpha=STRICT_ALPHA, n=t.n)
        ok_share, conf_share = significance(t.share, min_games, t.p_share, alpha=STRICT_ALPHA, n=t.n)
        ok, confidence = ok and ok_share, min(confidence, conf_share)
        if t.kind == "missed":
            x1, c1, x2, c2 = _chances(games, t.theme)
            m1, _, m2, _ = _sums(games, "missed", t.theme)
            moves1, moves2 = sum(g.moves[0] for g in games), sum(g.moves[1] for g in games)
            r = _rate_ratio(x1, c1, x2, c2)  # share of the chances missed, you vs your opponents
            rest = _rate_ratio(f1 - x1, k1 - c1, f2 - x2, k2 - c2)  # ... of the other tactical chances
            excess = r / rest if r is not None and rest else None
            share1, share2 = _share(x1, c1), _share(x2, c2)
            record = {
                "count": x1, "chances": c1, "rate": share1, "opp_count": x2, "opp_chances": c2, "opp_rate": share2,
                "other_rate": _share(f1 - x1, k1 - c1), "opp_other_rate": _share(f2 - x2, k2 - c2),
                "per100": _per100(x1, moves1), "opp_per100": _per100(x2, moves2),
                "missed_all": m1, "opp_missed_all": m2, "share": share1, "opp_share": share2,
            }
            what = (
                f"{where}, your opponents' mistakes and blunders left you {_article(name)} {name} {c1} times and you "
                f"missed {x1} of them ({pct(share1)}): your next move was a mistake or blunder too. Your opponents "
                f"missed {x2} of the {c2} your mistakes left them ({pct(share2)}). Of the other tactical chances "
                f"each side left, you missed {pct(record['other_rate'])} and your opponents "
                f"{pct(record['opp_other_rate'])}."
            )
        else:
            x1, moves1, x2, moves2 = _sums(games, "allowed", t.theme)
            y1, _, y2, _ = _sums(games, OWN_ALLOWED, t.theme)
            r = _rate_ratio(x1, moves1, x2, moves2)  # per move
            excess = _rate_ratio(y1, o1, y2, o2)  # share of your own errors vs theirs
            share1, share2 = _share(y1, o1), _share(y2, o2)
            record = {
                "count": x1, "opp_count": x2, "per100": _per100(x1, moves1), "opp_per100": _per100(x2, moves2),
                "share": share1, "opp_share": share2,
            }
            what = (
                f"{where}, your mistakes and blunders let your opponent play {_article(name)} {name} {x1} times "
                f"({_per100(x1, moves1):.2f} per 100 moves); your opponents' let you do it "
                f"{_per100(x2, moves2):.2f} times per 100 moves in the same games. Leaving out the mistakes made right "
                f"after one of the other side's (a missed chance), that is {pct(share1)} of your mistakes and "
                f"blunders against {pct(share2)} of theirs."
            )
        record.update(games_with=t.games_with, p_adjusted=t.p_rate, share_p_adjusted=t.p_share)
        stats[f"{t.theme}:{t.kind}"] = dict(record)
        if not (ok and confidence >= MIN_CONFIDENCE) or r is None or excess is None or r <= 0 or excess <= 0:
            continue
        kind_word = _direction(t, r, excess, ratio)
        if kind_word is None:
            continue
        size = min(r, excess) if kind_word == "weakness" else min(1.0 / r, 1.0 / excess)
        verb = "miss" if t.kind == "missed" else "allow"
        more = "more" if kind_word == "weakness" else "less"
        side = "you" if kind_word == "weakness" else "opponent"
        examples = _examples(events, side, t.kind, t.theme)
        insight = Insight(
            id=f"tactics.{kind_word}.motif-{t.kind}.{t.theme}",
            kind=kind_word,  # type: ignore[arg-type]
            category="tactics",
            title=f"You {verb} {plural} {more} often than your opponents",
            detail=what,
            severity=clamp(size - 1.0),
            confidence=confidence,
            evidence={"theme": t.theme, "motif_kind": t.kind, "games": n_games, **record,
                      "drill": training_url(t.theme)},
            study=_study(t.theme, t.kind) if kind_word == "weakness" else [
                f"Keep your eye sharp for {plural}: a few themed puzzles a week ({training_url(t.theme)}).",
            ],
            example_games=_urls(examples),
            formats=formats,
            chart=_format_chart(games, t.kind, t.theme, formats),
        )
        if examples:  # a weakness shows your costliest miss, a strength one of your opponents'
            try:
                insight.diagram = event_diagram(examples[0])
            except Exception:  # noqa: BLE001 — a board is decoration: never lose the finding over it
                insight.diagram = None
        stats[f"{t.theme}:{t.kind}"]["claim"] = kind_word
        insights.append(insight)
    return insights, stats


# --------------------------------------------------------------------------- the coaching step
def _engine_module(modules: Sequence[ModuleResult]) -> Optional[ModuleResult]:
    return next((m for m in modules if m.key in KEYS), None)


def _trim(line: Line, plies: int = PUZZLE_LINE_PLIES) -> Line:
    if len(line.moves_uci) <= plies:
        return line
    return Line(fen=line.fen, moves_uci=list(line.moves_uci[:plies]), moves_san=list(line.moves_san[:plies]),
                cp_end=line.cp_end, mate_end=line.mate_end, fen_end="", depth=line.depth)


def _puzzle_lines(coaching: Coaching, errors: Iterable["ProfileError"], events: Sequence[MotifEvent],
                  gated: frozenset[str], keys: Optional[set[str]] = None) -> None:
    """Your errors' best lines and named patterns for the puzzle export (the deeper coach lines win), for the errors
    in ``keys`` only (``puzzles.puzzle_keys``: the ones the export can use and the explained ones; None = all).

    Themes come only with a line supplied here, and only the patterns that start within the moves kept, so the
    PGN's Themes header describes the solution it prints. A key the puzzle pass (``puzzles.fill_puzzle_lines``)
    has settled, with a line or with a theme list (empty when it found none), is left as it is, and a line that
    starts with the move played is no puzzle."""
    themes: dict[str, set[str]] = {}
    for ev in events:
        if ev.side == "you" and ev.kind == "missed" and ev.motif.theme in gated and ev.motif.ply < PUZZLE_LINE_PLIES:
            themes.setdefault(f"{ev.error.game_id}:{ev.error.ply}", set()).add(ev.motif.theme)
    for e in errors:
        if e.side != "you" or e.best_line is None or not e.best_line.moves_uci:
            continue
        key = f"{e.game_id}:{e.ply}"
        if (keys is not None and key not in keys) or key in coaching.puzzle_lines or key in coaching.puzzle_themes:
            continue
        if e.best_line.moves_uci[0] == e.played_uci:  # the engine's move is the one you played: no puzzle
            continue
        coaching.puzzle_lines[key] = _trim(e.best_line)
        if themes.get(key):
            coaching.puzzle_themes[key] = sorted(themes[key])


def annotate(ctx: "AnalysisContext", coaching: Coaching, modules: list[ModuleResult], cfg: CoachConfig) -> None:
    """Fill ``coaching.motif_profile`` / ``motif_chart``, add them to the Engine review section, and add motif
    claims (claim rule at STRICT_ALPHA, BH across motifs) to that section's insights."""
    from . import deep, motifs, puzzles

    records = join(ctx)
    if not records:
        coaching.notes.append("Motif profile skipped: no engine-analysed games in this selection.")
        return
    errors = list(deep.profile_lines(ctx, cfg, coaching.notes) or [])
    if not errors:
        coaching.notes.append("Motif profile skipped: no engine lines for the mistakes and blunders in your games.")
        return
    gated = frozenset(getattr(motifs, "GATED_THEMES", ()) or ())
    games, events = collect(records, errors, motifs.detect_line, coaching.notes, gated)
    formats = format_counts(r.game for r in records)
    named = [t for t in themes_seen(games) if t in gated]
    coaching.settings["motif_profile"] = {
        "games": len(games),
        "formats": formats,
        "moves": [sum(g.moves[0] for g in games), sum(g.moves[1] for g in games)],
        "errors": [sum(g.errors[0] for g in games), sum(g.errors[1] for g in games)],
        "counts": profile_counts(games, named),
        "unnamed_themes": [t for t in themes_seen(games) if t not in gated],
        "chances": [sum(g.chances[0] for g in games), sum(g.chances[1] for g in games)],
        "chances_missed": [sum(g.followups[0] for g in games), sum(g.followups[1] for g in games)],
    }
    _puzzle_lines(coaching, errors, events, gated, puzzles.puzzle_keys(ctx, coaching))
    coaching.motif_profile = profile_table(games, gated, formats)
    coaching.motif_chart = profile_chart(games, gated, formats)
    if coaching.motif_profile is None:
        coaching.notes.append(
            "Motif profile: no tactical pattern that the detectors can name reliably turned up in your mistakes and "
            "blunders" + ("" if gated else " (no detector has passed its precision check yet)") + "."
        )
    claims, stats = motif_claims(games, gated, events, formats=formats)
    # the counts above are observations; these are the differences that passed the claim rule (for the drills)
    coaching.settings["motif_profile"]["findings"] = [i.id for i in claims]
    engine = _engine_module(modules)
    if engine is None:
        if claims:
            coaching.notes.append("Motif findings left out: the report has no Engine review section to hold them.")
        return
    if coaching.motif_profile is not None:
        engine.tables.append(coaching.motif_profile)
    if coaching.motif_chart is not None:
        engine.charts.append(coaching.motif_chart)
    engine.insights.extend(claims)
    engine.stats["motif_profile"] = {"games": len(games), "formats": formats, "tests": stats}
