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
opponents", and each must pass the project-wide rule: games are the unit (the clustered tests of
``analysis.engine_stats``, pairing you and your opponent in each game), the pattern must be more common per move
*and* a larger share of your errors than of theirs (so making more errors of every kind stays one finding, the
blunder rate), Benjamini-Hochberg adjusted across every pattern and both kinds, ``stats.significance`` at
``stats.STRICT_ALPHA``, minimum samples, and a rate ratio of at least ``MOTIF_RATIO``. Only patterns whose
detector passed the precision gate (``motifs.GATED_THEMES``) are named at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence

import chess

from ..analysis.engine_stats import Analysed, clustered_share_test, join
from ..models import Chart, Coaching, Diagram, Game, Insight, Line, Mark, ModuleResult, Motif, Table
from ..stats import STRICT_ALPHA, MeanTest, bh_adjust, clamp, pct, significance
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
@dataclass
class GameCounts:
    """One engine-analysed game: each side's moves, profiled errors and pattern counts (the unit of every test).

    Pairs are (yours, your opponent's).
    """

    game_id: str
    time_class: str
    moves: tuple[int, int]
    errors: list[int] = field(default_factory=lambda: [0, 0])  # mistakes and blunders with engine lines
    counts: dict[tuple[str, str], list[int]] = field(default_factory=dict)  # (kind, theme) -> errors with it
    url: str = ""

    def count(self, kind: str, theme: str) -> tuple[int, int]:
        pair = self.counts.get((kind, theme))
        return (pair[0], pair[1]) if pair else (0, 0)


@dataclass
class MotifEvent:
    """One error in which a pattern was missed or allowed (for example games and boards)."""

    game: Game
    error: "ProfileError"
    kind: str  # "missed" | "allowed"
    motif: Motif

    @property
    def side(self) -> str:
        return self.error.side


def collect(
    records: Sequence[Analysed],
    errors: Iterable["ProfileError"],
    detect: Callable[[Line, str], list[Motif]],
    notes: Optional[list[str]] = None,
) -> tuple[list[GameCounts], list[MotifEvent]]:
    """Per-game counts and the individual events, from the profiled errors of the analysed games.

    Errors from games outside ``records`` (not analysed, too short, filtered out) are ignored; each error counts
    once per pattern and kind, however often the detector reports it along the line. A line the detectors fail on
    counts as having no pattern (and a note says how many).
    """
    games: list[GameCounts] = []
    by_id: dict[str, tuple[Game, GameCounts]] = {}
    for r in records:
        mine = sum(1 for p in r.plies if p.is_user)
        gc = GameCounts(r.game.game_id, r.game.time_class, (mine, len(r.plies) - mine), url=r.game.url)
        games.append(gc)
        by_id[r.game.game_id] = (r.game, gc)
    events: list[MotifEvent] = []
    seen: set[tuple[str, int]] = set()
    failed = 0
    for e in errors:
        entry = by_id.get(e.game_id)
        if entry is None or e.side not in SIDES or (e.game_id, e.ply) in seen:
            continue
        seen.add((e.game_id, e.ply))
        game, gc = entry
        s = SIDES.index(e.side)
        gc.errors[s] += 1
        # best line: the patterns the player who went wrong would have carried out ("you" = the side to move);
        # refutation: the ones the other side carries out after the move
        for kind, line, role, carrier in (("missed", e.best_line, "best", "you"),
                                          ("allowed", e.refutation, "refutation", "opponent")):
            if line is None:
                continue
            try:
                detected = list(detect(line, role) or [])
            except Exception:  # noqa: BLE001 — one odd line must not sink the whole profile
                failed += 1
                detected = []
            found: dict[str, Motif] = {}
            for m in detected:
                if m.theme and (not m.side or m.side == carrier):
                    found.setdefault(m.theme, m)
            for theme, m in found.items():
                gc.counts.setdefault((kind, theme), [0, 0])[s] += 1
                events.append(MotifEvent(game, e, kind, m))
    if failed and notes is not None:
        notes.append(f"Motif profile: the pattern detectors failed on {failed} engine lines (counted without a "
                     "pattern).")
    return games, events


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


def profile_counts(games: Sequence[GameCounts], themes: Iterable[str]) -> dict[str, dict[str, int]]:
    """{theme: {"you_missed", "opp_missed", "you_allowed", "opp_allowed"}} over every game."""
    out = {}
    for theme in themes:
        mx, _, ox, _ = _sums(games, "missed", theme)
        ax, _, bx, _ = _sums(games, "allowed", theme)
        out[theme] = {"you_missed": mx, "opp_missed": ox, "you_allowed": ax, "opp_allowed": bx}
    return out


# --------------------------------------------------------------------------- table and chart
def _named(games: Sequence[GameCounts], gated: Iterable[str]) -> list[str]:
    """Named patterns that turned up at all, most frequent first."""
    counts = profile_counts(games, [t for t in themes_seen(games) if t in set(gated)])
    return sorted((t for t, c in counts.items() if sum(c.values())), key=lambda t: (-sum(counts[t].values()), t))


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
        note=f"{_games_note(games, formats)} {_KIND_NOTE} Practice: Lichess puzzles of that theme.",
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
        note=f"{_games_note(games, formats)} {_KIND_NOTE}",
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
    theme: str
    kind: str
    rate: MeanTest  # your rate per move minus theirs, games as the unit
    share: MeanTest  # the pattern's share of your errors minus its share of theirs
    games_with: int  # games in which either side had it
    events: int  # both sides
    p_rate: float = 1.0  # BH-adjusted
    p_share: float = 1.0


def motif_tests(
    games: Sequence[GameCounts], gated: Iterable[str], min_games: int = MIN_MOTIF_GAMES,
    min_events: int = MIN_MOTIF_EVENTS,
) -> list[MotifTest]:
    """Every named pattern x {missed, allowed} with enough data, BH-adjusted across all of them (each test family:
    rates and shares)."""
    tests = []
    for theme in sorted(set(gated)):
        for kind in KINDS:
            pairs = [g.count(kind, theme) for g in games]
            games_with = sum(1 for a, b in pairs if a or b)
            events = sum(a + b for a, b in pairs)
            if games_with < min_games or events < min_events:
                continue
            rate = clustered_share_test([(a, g.moves[0]) for (a, _), g in zip(pairs, games)],
                                        [(b, g.moves[1]) for (_, b), g in zip(pairs, games)])
            share = clustered_share_test([(a, g.errors[0]) for (a, _), g in zip(pairs, games)],
                                         [(b, g.errors[1]) for (_, b), g in zip(pairs, games)])
            tests.append(MotifTest(theme, kind, rate, share, games_with, events))
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


def _format_chart(games: Sequence[GameCounts], kind: str, theme: str, formats: dict[str, int]) -> Chart:
    """You vs your opponents per 100 moves: all games, then each format with enough games (when there are two+)."""
    groups: list[tuple[str, list[GameCounts]]] = [("All games", list(games))]
    split = [(tc, [g for g in games if g.time_class == tc]) for tc, n in formats.items() if n >= MIN_FORMAT_GAMES]
    if len(split) >= 2:
        groups += [(FORMAT_NAMES.get(tc, tc), gs) for tc, gs in split]
    elif len(split) == 1 and sum(1 for n in formats.values() if n) == 1:  # one format only: name it
        groups = [(FORMAT_NAMES.get(split[0][0], split[0][0]), list(games))]
    labels, you, them, rows = [], [], [], []
    for label, gs in groups:
        x1, m1, x2, m2 = _sums(gs, kind, theme)
        labels.append(label)
        you.append(_per100(x1, m1))
        them.append(_per100(x2, m2))
        rows.append([label, len(gs), x1, _per100(x1, m1), x2, _per100(x2, m2)])
    verb = "missed" if kind == "missed" else "allowed"
    note = "Engine-analysed games: " + ", ".join(f"{n} {tc}" for tc, n in formats.items()) + "."
    if len(groups) > 1:
        note += f" Formats with fewer than {MIN_FORMAT_GAMES} analysed games have no bar of their own."
    return comparison_chart(
        f"{motif_name(theme, True).capitalize()} {verb}, per 100 moves",
        labels,
        [("You", you), ("Opponents", them)],
        value_format="float2",
        note=note,
        table=Table(
            title=f"{motif_name(theme, True).capitalize()} {verb} by format",
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
    """The position before the error: the move played red, the engine's move green, and the line that follows."""
    e, g, theme = event.error, event.game, event.motif.theme
    best = e.best_line.moves_uci[0] if e.best_line and e.best_line.moves_uci else None
    played, best_label = _label(e.fen, e.played_uci), _label(e.fen, best) if best else ""
    name = motif_name(theme)
    if event.kind == "missed":
        line, title = e.best_line, f"The {name} you missed: {best_label} instead of {played}"
        strip_title = f"What {best_label} starts ({name})"
    else:
        line, title = e.refutation, f"The {name} you allowed after {played}"
        strip_title = f"What {played} allowed ({name})"
    orientation = g.color
    text = _line_text(line)
    caption = (
        f"You played {played} (red)" + (f"; {best_label} (green) was better" if best_label else "")
        + f". It cost {e.drop:.0f} percentage points of winning chances."
        + (f" The engine's line: {text}." if text else "")
    )
    diagram = position_diagram(title, e.fen, orientation=orientation, played=e.played_uci, best=best, caption=caption,
                               link=g.url, time_class=g.time_class)
    if line is not None and line.moves_uci:
        marks = [[] for _ in line.moves_uci]
        if 0 <= event.motif.ply < len(marks):
            marks[event.motif.ply] = _marks(event.motif)
        frames = min(6, max(4, event.motif.ply + 1))  # far enough to show the pattern
        strip = line_strip(strip_title, line.fen or e.fen, line.moves_uci, max_frames=frames, marks=marks)
        if strip.frames:
            diagram.strips.append(strip)
    return diagram


def _examples(events: Sequence[MotifEvent], side: str, kind: str, theme: str) -> list[MotifEvent]:
    """The events of one side, pattern and kind, costliest first (then most recent)."""
    picked = [ev for ev in events if ev.side == side and ev.kind == kind and ev.motif.theme == theme]
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
    of every test run (for the JSON export)."""
    formats = dict(formats or {})
    n_games = len(games)
    e1, e2 = sum(g.errors[0] for g in games), sum(g.errors[1] for g in games)
    insights: list[Insight] = []
    stats: dict[str, Any] = {}
    for t in motif_tests(games, gated, min_games, min_events):
        x1, m1, x2, m2 = _sums(games, t.kind, t.theme)
        ok_rate, conf_rate = significance(t.rate, min_games, t.p_rate, alpha=STRICT_ALPHA, n=t.games_with)
        ok_share, conf_share = significance(t.share, min_games, t.p_share, alpha=STRICT_ALPHA, n=t.games_with)
        confidence = min(conf_rate, conf_share)
        r = _rate_ratio(x1, m1, x2, m2)
        # compared with your other errors: more errors of every kind is the blunder-rate finding, not this one
        rest = _rate_ratio(e1 - x1, m1, e2 - x2, m2)
        stats[f"{t.theme}:{t.kind}"] = {
            "count": x1, "opp_count": x2, "per100": _per100(x1, m1), "opp_per100": _per100(x2, m2),
            "games_with": t.games_with, "p_adjusted": t.p_rate, "share_p_adjusted": t.p_share,
        }
        if not (ok_rate and ok_share and confidence >= MIN_CONFIDENCE) or r is None or rest is None:
            continue
        if t.rate.mean > 0 and t.share.mean > 0 and r >= ratio:
            kind_word, excess = "weakness", r / max(1.0, rest)
        elif t.rate.mean < 0 and t.share.mean < 0 and 1.0 / r >= ratio:
            kind_word, excess = "strength", (1.0 / r) / max(1.0, 1.0 / rest)
        else:
            continue
        if excess < ratio:
            continue
        name, plural = motif_name(t.theme), motif_name(t.theme, True)
        verb = "miss" if t.kind == "missed" else "allow"
        more = "more" if kind_word == "weakness" else "less"
        mix = format_text(formats)
        share1, share2 = (x1 / e1 if e1 else None), (x2 / e2 if e2 else None)
        if t.kind == "missed":
            what = (f"{_article(name)} {name} was there for you {x1} times when you made a mistake or blunder "
                    f"({_per100(x1, m1):.2f} per 100 moves); your opponents missed one {_per100(x2, m2):.2f} times "
                    "per 100 moves in the same games")
        else:
            what = (f"your mistakes and blunders let your opponent play {_article(name)} {name} {x1} times "
                    f"({_per100(x1, m1):.2f} per 100 moves); your opponents' let you do it {_per100(x2, m2):.2f} "
                    "times per 100 moves in the same games")
        detail = (
            f"In {n_games} engine-analysed games" + (f" ({mix})" if mix else "") + f", {what}. That is "
            f"{pct(share1)} of your mistakes and blunders against {pct(share2)} of theirs."
        )
        side = "you" if kind_word == "weakness" else "opponent"
        examples = _examples(events, side, t.kind, t.theme)
        insight = Insight(
            id=f"tactics.{kind_word}.motif-{t.kind}.{t.theme}",
            kind=kind_word,  # type: ignore[arg-type]
            category="tactics",
            title=f"You {verb} {plural} {more} often than your opponents",
            detail=detail,
            severity=clamp(excess - 1.0),
            confidence=confidence,
            evidence={
                "theme": t.theme, "motif_kind": t.kind, "count": x1, "per100": _per100(x1, m1), "opp_count": x2,
                "opp_per100": _per100(x2, m2), "share": share1, "opp_share": share2, "games": n_games,
                "games_with": t.games_with, "p_adjusted": t.p_rate, "share_p_adjusted": t.p_share,
                "drill": training_url(t.theme),
            },
            study=_study(t.theme, t.kind) if kind_word == "weakness" else [
                f"Keep your eye for {plural} sharp: a few themed puzzles a week ({training_url(t.theme)}).",
            ],
            example_games=_urls(examples),
            formats=formats,
            chart=_format_chart(games, t.kind, t.theme, formats),
        )
        if kind_word == "weakness" and examples:
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
                  gated: frozenset[str]) -> None:
    """Your errors' best lines and named patterns for the puzzle export (the deeper coach lines win)."""
    themes: dict[str, set[str]] = {}
    for ev in events:
        if ev.side == "you" and ev.kind == "missed" and ev.motif.theme in gated:
            themes.setdefault(f"{ev.error.game_id}:{ev.error.ply}", set()).add(ev.motif.theme)
    for e in errors:
        if e.side != "you" or e.best_line is None or not e.best_line.moves_uci:
            continue
        key = f"{e.game_id}:{e.ply}"
        coaching.puzzle_lines.setdefault(key, _trim(e.best_line))
        if themes.get(key):
            coaching.puzzle_themes.setdefault(key, sorted(themes[key]))


def annotate(ctx: "AnalysisContext", coaching: Coaching, modules: list[ModuleResult], cfg: CoachConfig) -> None:
    """Fill ``coaching.motif_profile`` / ``motif_chart``, add them to the Engine review section, and add motif
    claims (claim rule at STRICT_ALPHA, BH across motifs) to that section's insights."""
    from . import deep, motifs

    records = join(ctx)
    if not records:
        coaching.notes.append("Motif profile skipped: no engine-analysed games in this selection.")
        return
    errors = list(deep.profile_lines(ctx, cfg, coaching.notes) or [])
    if not errors:
        coaching.notes.append("Motif profile skipped: no engine lines for the mistakes and blunders in your games.")
        return
    gated = frozenset(getattr(motifs, "GATED_THEMES", ()) or ())
    games, events = collect(records, errors, motifs.detect_line, coaching.notes)
    formats = format_counts(r.game for r in records)
    named = [t for t in themes_seen(games) if t in gated]
    coaching.settings["motif_profile"] = {
        "games": len(games),
        "formats": formats,
        "moves": [sum(g.moves[0] for g in games), sum(g.moves[1] for g in games)],
        "errors": [sum(g.errors[0] for g in games), sum(g.errors[1] for g in games)],
        "counts": profile_counts(games, named),
        "unnamed_themes": [t for t in themes_seen(games) if t not in gated],
    }
    _puzzle_lines(coaching, errors, events, gated)
    coaching.motif_profile = profile_table(games, gated, formats)
    coaching.motif_chart = profile_chart(games, gated, formats)
    if coaching.motif_profile is None:
        coaching.notes.append(
            "Motif profile: no tactical pattern that the detectors can name reliably turned up in your mistakes and "
            "blunders" + ("" if gated else " (no detector has passed its precision check yet)") + "."
        )
    claims, stats = motif_claims(games, gated, events, formats=formats)
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
