"""Positions you keep getting wrong, and a personal puzzle set built from your own errors.

Needs engine analysis (``ctx.evals``). Two outputs:

* **Repeated mistakes**: positions (keyed by EPD, so transpositions merge) where you
  made an error in two or more games. These are nearly always opening positions,
  and they are the most concrete study items the tool can produce: "in this exact
  position you keep playing X; Y is better".
* **Puzzles**: every position where your move lost a lot of winning chances,
  exported as PGN (``SetUp``/``FEN`` per puzzle) that Lichess studies and most
  chess GUIs import directly.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import chess
import chess.svg

from ..context import AnalysisContext
from ..models import Diagram, Game, GameEval, Insight, Kpi, ModuleResult, Table
from ..stats import clamp

KEY = "mistakes"
TITLE = "Positions you keep getting wrong"
MIN_DROP = 8.0  # win-% points your move must lose to count as an error (inaccuracy = 5, mistake = 10)
PUZZLE_MIN_DROP = 10.0
MIN_REPEATS = 2
MAX_DIAGRAMS = 4
BOARD_COLORS = {
    "square light": "#E9E4D4",
    "square dark": "#7D8F6E",
    "arrow red": "#C43D3Dcc",  # your move
    "arrow green": "#1F7A3Acc",  # engine's move
}


@dataclass
class MistakeEvent:
    game: Game
    ply: int
    fen: str  # position before the move
    epd: str  # fen without move counters (transposition key)
    san: str  # move played
    best_san: Optional[str]
    drop: float  # win-% points lost, mover's point of view
    judgement: Optional[str]

    @property
    def move_label(self) -> str:
        return move_label(self.fen, self.san)

    @property
    def best_label(self) -> Optional[str]:
        return move_label(self.fen, self.best_san) if self.best_san else None


@dataclass
class RepeatedMistake:
    epd: str
    events: list[MistakeEvent]  # one per game (your costliest error there in that game), most recent first
    reached: int = 0  # games (analysed or not) in which you reached the position with the move
    games_reached: list[str] = field(default_factory=list)

    @property
    def first(self) -> MistakeEvent:
        return self.events[0]

    @property
    def errors(self) -> int:
        """Games in which you went wrong here."""
        return len(self.events)

    @property
    def avg_drop(self) -> float:
        return sum(e.drop for e in self.events) / len(self.events)

    @property
    def played(self) -> Counter:
        return Counter(e.san for e in self.events)

    @property
    def best_san(self) -> Optional[str]:
        best = Counter(e.best_san for e in self.events if e.best_san)
        return best.most_common(1)[0][0] if best else None

    @property
    def weight(self) -> float:
        return self.errors * self.avg_drop


# --------------------------------------------------------------------------- helpers
def move_label(fen: str, san: Optional[str]) -> str:
    """'7...Bg4' / '12.Nf3' from the position before the move."""
    if not san:
        return ""
    parts = fen.split()
    number = parts[5] if len(parts) > 5 else "1"
    return f"{number}.{san}" if parts[1] == "w" else f"{number}...{san}"


def format_line(sans: list[str], start_fen: Optional[str] = None, last_n: Optional[int] = None) -> str:
    """SAN list -> '1.e4 c6 2.d4 d5'; with ``last_n`` only the tail ('… 5.Nf3 e6')."""
    parts = (start_fen or chess.STARTING_FEN).split()
    move_no = int(parts[5]) if len(parts) > 5 else 1
    white = parts[1] == "w"
    labelled: list[tuple[str, str]] = []  # (label with number, label as it appears mid-line)
    for san in sans:
        labelled.append((f"{move_no}.{san}", f"{move_no}.{san}") if white else (f"{move_no}...{san}", san))
        if not white:
            move_no += 1
        white = not white
    if not labelled:
        return ""
    start = 0 if last_n is None or len(labelled) <= last_n else len(labelled) - last_n
    tokens = [labelled[start][0]] + [short for _, short in labelled[start + 1 :]]
    return ("… " if start else "") + " ".join(tokens)


def _board(game: Game) -> chess.Board:
    return chess.Board(game.initial_fen or chess.STARTING_FEN, chess960=game.rules == "chess960")


def _pieces(epd: str) -> int:
    return sum(ch.isalpha() for ch in epd.split(" ", 1)[0])


def collect_errors(games: Iterable[Game], evals: dict[str, GameEval], min_drop: float = MIN_DROP) -> list[MistakeEvent]:
    """Every move of yours that lost at least ``min_drop`` win-% points, with the position before it.

    A move the engine itself prefers is never an error, however much the evaluation drops after it
    (the position was already worse than it looked at the engine's depth).
    """
    by_id = {g.game_id: g for g in games}
    events: list[MistakeEvent] = []
    for gid, ev in evals.items():
        game = by_id.get(gid)
        if game is None:
            continue
        wanted = {
            p.ply: p
            for p in ev.plies
            if p.is_user
            and 0 <= p.ply < game.plies
            and (p.win_before - p.win_after) >= min_drop
            and not (p.best_san and p.best_san == game.moves_san[p.ply])
        }
        if not wanted:
            continue
        last = max(wanted)
        try:
            board = _board(game)
            for ply, san in enumerate(game.moves_san[: last + 1]):
                if ply in wanted:
                    p = wanted[ply]
                    events.append(
                        MistakeEvent(
                            game=game,
                            ply=ply,
                            fen=board.fen(),
                            epd=board.epd(),
                            san=san,
                            best_san=p.best_san if p.best_san != san else None,
                            drop=p.win_before - p.win_after,
                            judgement=p.judgement,
                        )
                    )
                board.push_san(san)
        except ValueError:  # illegal SAN in a corrupt record: keep what we have
            continue
    return events


def find_repeated(
    events: list[MistakeEvent], games: Iterable[Game], min_repeats: int = MIN_REPEATS
) -> list[RepeatedMistake]:
    """Positions (by EPD, so transpositions merge) where you went wrong in at least ``min_repeats`` games.

    A game counts once per position, with your costliest error there, even if the position came up
    (and went wrong) more than once in it.
    """
    groups: dict[str, dict[str, MistakeEvent]] = defaultdict(dict)  # epd -> game id -> costliest error
    for e in events:
        current = groups[e.epd].get(e.game.game_id)
        if current is None or e.drop > current.drop:
            groups[e.epd][e.game.game_id] = e
    repeated = {
        epd: RepeatedMistake(epd=epd, events=sorted(by_game.values(), key=lambda e: e.game.end_time, reverse=True))
        for epd, by_game in groups.items()
        if len(by_game) >= min_repeats
    }
    if not repeated:
        return []
    # How often did you reach each of these positions (analysed or not), at any move number? Captures are
    # irreversible, so a game can stop being searched once fewer pieces are left than in every position.
    fewest = min(_pieces(epd) for epd in repeated)
    for game in games:
        seen: set[str] = set()
        try:
            board = _board(game)
            for ply, san in enumerate(game.moves_san):
                if chess.popcount(board.occupied) < fewest:
                    break
                if game.is_my_ply(ply):
                    epd = board.epd()
                    if epd in repeated and epd not in seen:
                        seen.add(epd)
                        repeated[epd].reached += 1
                        repeated[epd].games_reached.append(game.url)
                board.push_san(san)
        except ValueError:  # unreadable start position or illegal move: keep what was counted
            continue
    out = list(repeated.values())
    for r in out:
        r.reached = max(r.reached, r.errors)
    return sorted(out, key=lambda r: (-r.weight, r.epd))


def _svg(event: MistakeEvent) -> str:
    board = chess.Board(event.fen, chess960=event.game.rules == "chess960")
    arrows = []
    try:
        played = board.parse_san(event.san)
        arrows.append(chess.svg.Arrow(played.from_square, played.to_square, color="red"))
        if event.best_san:
            best = board.parse_san(event.best_san)
            arrows.append(chess.svg.Arrow(best.from_square, best.to_square, color="green"))
    except ValueError:
        pass
    return chess.svg.board(
        board,
        orientation=chess.WHITE if event.game.color == "white" else chess.BLACK,
        arrows=arrows,
        size=280,
        colors=BOARD_COLORS,
    )


def _slug(epd: str) -> str:
    return hashlib.sha1(epd.encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- module
def analyze(ctx: AnalysisContext) -> ModuleResult:
    if not ctx.evals:
        return ModuleResult(
            key=KEY,
            title=TITLE,
            summary="Needs engine analysis. Run with --engine to find the positions you repeatedly get wrong "
            "and to export your own mistakes as puzzles.",
        )
    events = collect_errors(ctx.games, ctx.evals)
    repeated = find_repeated(events, ctx.games)
    puzzles = [e for e in events if e.drop >= PUZZLE_MIN_DROP and e.best_san]
    n_analysed = sum(1 for g in ctx.games if g.game_id in ctx.evals)

    kpis = [
        Kpi("Games analysed", n_analysed, "int"),
        Kpi("Positions with repeated errors", len(repeated), "int"),
        Kpi("Errors in those positions", sum(r.errors for r in repeated), "int"),
        Kpi("Puzzles from your games", len(puzzles), "int", hint="export with --puzzles"),
    ]
    rows = []
    for r in repeated[:15]:
        e = r.first
        played = ", ".join(f"{move_label(e.fen, san)} ×{n}" for san, n in r.played.most_common())
        rows.append(
            [
                format_line(e.game.moves_san[: e.ply], e.game.initial_fen, last_n=8),
                e.game.opening_family or "",
                r.reached,
                r.errors,
                played,
                move_label(e.fen, r.best_san) if r.best_san else "",
                r.avg_drop / 100.0,
                e.game.url,
            ]
        )
    tables = []
    if rows:
        tables.append(
            Table(
                title="Repeated mistakes",
                columns=["Moves to reach it", "Opening", "Reached", "Errors", "You played", "Engine move", "Avg win chance lost", "Game"],
                rows=rows,
                formats=["text", "text", "int", "int", "text", "text", "pct", "url"],
                note=f"An error is a move that lost at least {MIN_DROP:.0f} percentage points of winning chances. "
                "Positions are matched exactly, so transpositions count together.",
            )
        )

    diagrams = []
    for r in repeated[:MAX_DIAGRAMS]:
        e = r.first
        diagrams.append(
            Diagram(
                title=f"{e.game.opening_family or 'Position'}: {move_label(e.fen, r.played.most_common(1)[0][0])}",
                fen=e.fen,
                svg=_svg(e),
                caption=f"After {format_line(e.game.moves_san[: e.ply], e.game.initial_fen, last_n=6)}. "
                f"Red: your move ({r.errors}×). Green: engine's choice.",
                link=e.game.url,
            )
        )

    insights = []
    for r in repeated[:3]:
        e = r.first
        played_san, times = r.played.most_common(1)[0]
        played = move_label(e.fen, played_san)
        best = move_label(e.fen, r.best_san) if r.best_san else None
        line = format_line(e.game.moves_san[: e.ply], e.game.initial_fen, last_n=10)
        title = f"You have played {played} here {times} times" + (f"; {best} is better" if best else "")
        insights.append(
            Insight(
                id=f"{KEY}.weakness.{_slug(r.epd)}",
                kind="weakness",
                category="openings" if e.ply < 40 else "tactics",
                title=title,
                detail=f"After {line} you went wrong in {r.errors} of the {r.reached} games that reached this position, "
                f"losing about {r.avg_drop:.0f} percentage points of winning chances each time."
                + (f" Stockfish prefers {best}." if best else ""),
                severity=clamp(r.weight / 60.0),
                confidence=clamp(0.45 + 0.15 * r.errors, 0.0, 0.95),
                evidence={"fen": e.fen, "errors": r.errors, "reached": r.reached, "avg_drop": round(r.avg_drop, 1),
                          "played": dict(r.played), "best": r.best_san},
                study=[
                    f"Set up the position after {line} and work out why {best or 'the engine move'} beats {played}.",
                    "Add the line to your repertoire notes so you recognise the position next time.",
                    "Replay the line from move 1 a few times, then test yourself on it again in a few days.",
                ],
                example_games=[ev.game.url for ev in r.events][:5],
            )
        )

    if repeated:
        summary = (
            f"In {n_analysed} analysed games you went wrong more than once in {len(repeated)} "
            f"position{'s' if len(repeated) != 1 else ''}. These are the cheapest points to win back: "
            "learn the right move once and it pays off every time the position comes up."
        )
    else:
        summary = (
            f"No position where you erred more than once in {n_analysed} analysed games. "
            f"{len(puzzles)} of your mistakes are available as puzzles."
        )
    return ModuleResult(
        key=KEY,
        title=TITLE,
        summary=summary,
        kpis=kpis,
        tables=tables,
        insights=insights,
        diagrams=diagrams,
        stats={"repeated_positions": len(repeated), "puzzles": len(puzzles), "errors": len(events)},
    )


# --------------------------------------------------------------------------- puzzle export
def build_puzzles(
    games: Iterable[Game], evals: dict[str, GameEval], min_drop: float = PUZZLE_MIN_DROP, limit: int = 300
) -> list[MistakeEvent]:
    """Your costliest mistakes with a known better move, worst first."""
    events = [e for e in collect_errors(games, evals, min_drop=min_drop) if e.best_san]
    events.sort(key=lambda e: (-e.drop, e.game.end_time))
    return events[:limit]


def _pgn_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def puzzles_to_pgn(events: list[MistakeEvent]) -> str:
    """One PGN game per puzzle: the position before your mistake (``SetUp`` / ``FEN``), solution = engine move.

    Each game has result ``*`` and a ``Variant`` header for Chess960, as Lichess's study import expects.
    A Lichess study holds at most 64 chapters, so a longer file is imported in several studies.
    """
    chunks = []
    for i, e in enumerate(events, 1):
        g = e.game
        white, black = (g.username, g.opponent) if g.color == "white" else (g.opponent, g.username)
        headers = [
            ("Event", f"chess-insights puzzle {i}"),
            ("Site", g.url),
            ("Date", g.end_time.strftime("%Y.%m.%d")),
            ("White", white),
            ("Black", black),
            ("Result", "*"),
            *([("Variant", "Chess960")] if g.rules == "chess960" else []),  # castling rules (and moves) differ
            ("SetUp", "1"),
            ("FEN", e.fen),
            ("Annotator", "chess-insights"),
        ]
        comment = (
            f"In the game you played {e.move_label}, losing {e.drop:.0f} percentage points of winning chances. "
            f"Find the better move."
        ).replace("}", ")")
        best = e.best_label or ""
        move_text = best.replace("...", "... ", 1) if "..." in best else best.replace(".", ". ", 1)
        chunks.append(
            "\n".join(f'[{k} "{_pgn_escape(str(v))}"]' for k, v in headers) + f"\n\n{{{comment}}} {move_text} *\n"
        )
    return "\n".join(chunks)
