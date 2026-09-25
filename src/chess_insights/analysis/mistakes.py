"""Positions you keep getting wrong, and a personal puzzle set built from your own errors.

Needs engine analysis (``ctx.evals``). Two outputs:

* **Repeated mistakes**: the same wrong move in the same position (keyed by EPD, so
  transpositions merge) in two or more of your games. These are nearly always opening
  positions, and they are the most concrete study items the tool can produce: "in this
  exact position you keep playing X; Y is better". Stockfish's verdict belongs to the
  position and the move, so every game in which you played that move there counts,
  analysed or not, out of every game that reached the position. It becomes a weakness
  only when the move is a habit rather than a coincidence: played in at least
  ``MIN_CLAIM_REPEATS`` games, and more often than your ordinary error rate explains
  (a one-sided binomial test, Benjamini-Hochberg adjusted over the positions tested,
  through ``stats.significance`` at ``STRICT_ALPHA``). Anything less is an observation.
* **Puzzles**: every position where your move lost a lot of winning chances,
  exported as PGN (``SetUp``/``FEN`` per puzzle) that Lichess studies and most
  chess GUIs import directly.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import chess
import chess.svg

from ..context import AnalysisContext
from ..models import Diagram, Game, GameEval, Insight, Kpi, ModuleResult, PlyEval, Table
from ..stats import STRICT_ALPHA, MeanTest, bh_adjust, clamp, significance

KEY = "mistakes"
TITLE = "Positions you keep getting wrong"
MIN_DROP = 8.0  # win-% points your move must lose to count as an error (inaccuracy = 5, mistake = 10)
PUZZLE_MIN_DROP = 10.0
MIN_REPEATS = 2  # games with the same wrong move in the same position, for the table
MIN_CLAIM_REPEATS = 3  # ... and for a weakness: two games can be a coincidence
PUZZLE_LIMIT = 300  # the costliest puzzles exported by --puzzles
PUZZLE_TABLE_ROWS = 10  # the costliest puzzles listed in the report, each with a Lichess board link
LICHESS_STUDY_CHAPTERS = 64  # a Lichess study holds at most this many puzzles
MAX_INSIGHTS = 3
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
    uci: str = ""  # the move played, in UCI (with ``epd``: the key of a repeated mistake)

    @property
    def move_label(self) -> str:
        return move_label(self.fen, self.san)

    @property
    def best_label(self) -> Optional[str]:
        return move_label(self.fen, self.best_san) if self.best_san else None


@dataclass
class RepeatedMistake:
    """One wrong move (``uci``) in one position (``epd``) that you played in several games."""

    epd: str
    uci: str
    events: list[MistakeEvent]  # analysed games in which the engine flagged it (costliest per game), most recent first
    played_in: list[Game] = field(default_factory=list)  # every game (analysed or not) where you played it here
    drops: list[float] = field(default_factory=list)  # win-% it lost in each analysed game that has it, flagged or not
    reached: int = 0  # games (analysed or not) in which the position came up with you to move
    games_reached: list[str] = field(default_factory=list)
    base_rate: Optional[float] = None  # your error rate per move, the null hypothesis of the test
    p_value: float = 1.0  # one-sided binomial: played it in >= ``errors`` of ``reached`` games at ``base_rate``
    p_adjusted: float = 1.0  # Benjamini-Hochberg over every repeated mistake tested
    significant: bool = False  # a weakness (else an observation)
    confidence: float = 0.0

    @property
    def first(self) -> MistakeEvent:
        return self.events[0]

    @property
    def san(self) -> str:
        return self.first.san

    @property
    def errors(self) -> int:
        """Games (analysed or not) in which you played this move here."""
        return len(self.played_in)

    @property
    def analysed(self) -> int:
        """Engine-analysed games in which you played it here."""
        return len(self.drops)

    @property
    def avg_drop(self) -> float:
        drops = self.drops or [e.drop for e in self.events]
        return sum(drops) / len(drops)

    @property
    def best_san(self) -> Optional[str]:
        best = Counter(e.best_san for e in self.events if e.best_san)
        return best.most_common(1)[0][0] if best else None

    @property
    def weight(self) -> float:
        return self.errors * self.avg_drop

    @property
    def example_games(self) -> list[str]:
        """Analysed games first (they carry the engine's verdict), then the others; most recent first."""
        urls = [e.game.url for e in self.events] + [g.url for g in self.played_in]
        return list(dict.fromkeys(u for u in urls if u))


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


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _is_error(p: PlyEval, san: str, min_drop: float) -> bool:
    """Lost at least ``min_drop`` win-% points, and was not the engine's own choice."""
    return (p.win_before - p.win_after) >= min_drop and not (p.best_san and p.best_san == san)


def _drop(p: PlyEval, san: str) -> float:
    """Win-% points a move lost; none for the engine's own choice (the position was worse than it looked)."""
    return 0.0 if p.best_san and p.best_san == san else max(0.0, p.win_before - p.win_after)


def error_rate(games: Iterable[Game], evals: dict[str, GameEval], min_drop: float = MIN_DROP) -> tuple[int, int]:
    """(your errors, your analysed moves) over the analysed games: the base rate of the repeated-mistake test."""
    errors = moves = 0
    for game in games:
        ev = evals.get(game.game_id)
        if ev is None:
            continue
        for p in ev.plies:
            if p.is_user and 0 <= p.ply < game.plies:
                moves += 1
                errors += _is_error(p, game.moves_san[p.ply], min_drop)
    return errors, moves


def binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    log_p, log_q, log_n = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    total = 0.0
    for j in range(k, n + 1):
        total += math.exp(log_n - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * log_p + (n - j) * log_q)
    return clamp(total)


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
            if p.is_user and 0 <= p.ply < game.plies and _is_error(p, game.moves_san[p.ply], min_drop)
        }
        if not wanted:
            continue
        last = max(wanted)
        try:
            board = _board(game)
            for ply, san in enumerate(game.moves_san[: last + 1]):
                move = board.parse_san(san)
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
                            uci=move.uci(),
                        )
                    )
                board.push(move)
        except ValueError:  # illegal SAN in a corrupt record: keep what we have
            continue
    return events


def find_repeated(
    events: list[MistakeEvent],
    games: Iterable[Game],
    evals: Optional[dict[str, GameEval]] = None,
    min_repeats: int = MIN_REPEATS,
    min_drop: float = MIN_DROP,
) -> list[RepeatedMistake]:
    """The same wrong move in the same position (by EPD, so transpositions merge) in at least ``min_repeats`` games.

    A move is wrong when the engine flagged it in an analysed game (``events``). Stockfish's verdict is about
    the position and the move, so every game in which you played it there counts, analysed or not, and
    ``reached`` counts every game in which the position came up with you to move: both numbers come from
    the same games. A game counts once however often the position recurs in it. Moves whose average loss
    over all analysed games that have them is below ``min_drop`` (the engine does not consistently call
    them errors) are left out. Most costly first.
    """
    flagged: dict[tuple[str, str], dict[str, MistakeEvent]] = defaultdict(dict)  # (epd, uci) -> game -> costliest
    for e in events:
        if not e.uci:
            continue
        current = flagged[(e.epd, e.uci)].get(e.game.game_id)
        if current is None or e.drop > current.drop:
            flagged[(e.epd, e.uci)][e.game.game_id] = e
    if not flagged:
        return []
    epds = {epd for epd, _ in flagged}
    evals = evals or {}
    reached: dict[str, list[Game]] = defaultdict(list)
    played: dict[tuple[str, str], list[Game]] = defaultdict(list)
    drops: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    # Captures are irreversible, so a game can stop being searched once fewer pieces are left than in every position.
    fewest = min(_pieces(epd) for epd in epds)
    for game in games:
        seen: set[object] = set()
        ev = evals.get(game.game_id)
        plies: Optional[dict[int, PlyEval]] = None
        try:
            board = _board(game)
            for ply, san in enumerate(game.moves_san):
                if chess.popcount(board.occupied) < fewest:
                    break
                move = board.parse_san(san)
                if game.is_my_ply(ply):
                    epd = board.epd()
                    if epd in epds:
                        if epd not in seen:
                            seen.add(epd)
                            reached[epd].append(game)
                        key = (epd, move.uci())
                        if key in flagged:
                            if key not in seen:
                                seen.add(key)
                                played[key].append(game)
                            if ev is not None:
                                plies = plies if plies is not None else {p.ply: p for p in ev.plies}
                                p = plies.get(ply)
                                if p is not None:
                                    d = _drop(p, san)
                                    drops[key][game.game_id] = max(d, drops[key].get(game.game_id, d))
                board.push(move)
        except ValueError:  # unreadable start position or illegal move: keep what was counted
            continue
    out = []
    for key, by_game in flagged.items():
        epd, uci = key
        in_games = played.get(key, [])
        for e in by_game.values():  # an analysed error always counts, even if the scan missed it
            if all(g.game_id != e.game.game_id for g in in_games):
                in_games.append(e.game)
        if len(in_games) < min_repeats:
            continue
        rep = RepeatedMistake(
            epd=epd,
            uci=uci,
            events=sorted(by_game.values(), key=lambda e: e.game.end_time, reverse=True),
            played_in=sorted(in_games, key=lambda g: g.end_time, reverse=True),
            drops=list(drops.get(key, {}).values()),
        )
        if rep.avg_drop < min_drop:
            continue
        rep.games_reached = list(dict.fromkeys(g.url for g in reached.get(epd, [])))
        rep.reached = max(len(reached.get(epd, [])), rep.errors)
        out.append(rep)
    return sorted(out, key=lambda r: (-r.weight, r.epd, r.uci))


def assess(repeated: list[RepeatedMistake], errors: int, moves: int) -> list[RepeatedMistake]:
    """Test each repeated mistake against your ordinary error rate (``errors`` per ``moves``) and mark the claims.

    H0: in this position you go wrong no more often than in any other move you play. Playing one specific
    wrong move is rarer than going wrong at all, so the binomial tail of ``errors`` of ``reached`` games at
    that rate is a conservative p-value. The p-values are Benjamini-Hochberg adjusted over all the repeated
    mistakes tested; a claim also needs the same move in at least ``MIN_CLAIM_REPEATS`` games.
    """
    if not repeated:
        return repeated
    rate = (errors + 1.0) / (moves + 2.0)  # never 0 or 1, even with very few analysed moves
    for r in repeated:
        r.base_rate = rate
        r.p_value = binomial_sf(r.errors, r.reached, rate)
    for r, p_adj in zip(repeated, bh_adjust([r.p_value for r in repeated])):
        r.p_adjusted = p_adj
        se = math.sqrt(rate * (1.0 - rate) / r.reached)
        share = r.errors / r.reached
        test = MeanTest(n=r.reached, mean=share - rate, se=se, z=(share - rate) / se, p_value=r.p_value)
        ok, confidence = significance(test, MIN_CLAIM_REPEATS, p_adj, alpha=STRICT_ALPHA)
        r.significant = ok and r.errors >= MIN_CLAIM_REPEATS
        r.confidence = confidence if r.significant else min(confidence, 0.45)
    return repeated


def _arrow(board: chess.Board, move: chess.Move) -> tuple[chess.Square, chess.Square]:
    """(from, to) for a move's arrow; castling points to the king's destination (Chess960 moves encode the rook)."""
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        return move.from_square, chess.square(2 if board.is_queenside_castling(move) else 6, rank)
    return move.from_square, move.to_square


def _svg(event: MistakeEvent) -> str:
    board = chess.Board(event.fen, chess960=event.game.rules == "chess960")
    arrows = []
    try:
        arrows.append(chess.svg.Arrow(*_arrow(board, board.parse_san(event.san)), color="red"))
        if event.best_san:
            arrows.append(chess.svg.Arrow(*_arrow(board, board.parse_san(event.best_san)), color="green"))
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


def fold_follow_ups(repeated: list[RepeatedMistake]) -> tuple[list[RepeatedMistake], dict[int, list[RepeatedMistake]]]:
    """Merge a wrong move that only ever happened in games where you had already gone wrong earlier in the same
    line (9.Nxg5 and then 11.Kh1 in the same three games) into that earlier one: one habit, one finding.

    A follow-up is folded only when its games are a subset of the earlier move's games and it is not a claim while
    the earlier one is only an observation (folding never removes a claim nor promotes one). Returns the kept
    mistakes (in their original order) and, by ``id()`` of each kept one, the follow-ups folded into it.
    """
    kept: list[RepeatedMistake] = []
    folded: dict[int, list[RepeatedMistake]] = {}
    by_ply = sorted(repeated, key=lambda r: (r.first.ply, -r.weight))
    games = {id(r): {g.game_id for g in r.played_in} for r in repeated}
    absorbed: set[int] = set()
    for r in by_ply:
        root = next(
            (
                k
                for k in by_ply
                if id(k) not in absorbed
                and k is not r
                and k.first.ply < r.first.ply
                and games[id(r)] <= games[id(k)]
                and (k.significant or not r.significant)
            ),
            None,
        )
        if root is not None:
            absorbed.add(id(r))
            folded.setdefault(id(root), []).append(r)
    kept = [r for r in repeated if id(r) not in absorbed]
    return kept, folded


# --------------------------------------------------------------------------- module
def _insight(r: RepeatedMistake, follow_ups: Iterable[RepeatedMistake] = ()) -> Insight:
    e = r.first
    played = move_label(e.fen, r.san)
    best = move_label(e.fen, r.best_san) if r.best_san else None
    line = format_line(e.game.moves_san[: e.ply], e.game.initial_fen, last_n=10)
    # named by opening and move number, so the line reads on its own in the lists at the top (no "here")
    fields = e.fen.split()
    move_no = fields[5] if len(fields) > 5 else "?"
    where = f"{e.game.opening_family}, move {move_no}" if e.game.opening_family else f"Move {move_no}"
    title = f"{where}: you played {played} in {r.errors} of {_plural(r.reached, 'game')}" + (
        f"; {best} is better" if best else ""
    )
    analysed = (
        f"Stockfish analysed {r.analysed} of them" if r.analysed > 1 else "Stockfish analysed one of them"
    )
    detail = (
        f"After {line} you played {played} in {r.errors} of the {_plural(r.reached, 'game')} that reached this "
        f"position. {analysed}: the move cost you about {r.avg_drop:.0f} in 100 of your chances to win"
        + (f"; Stockfish prefers {best}." if best else ".")
    )
    if r.significant:
        per = round(1 / r.base_rate) if r.base_rate else 0
        detail += (
            f" You go wrong on about 1 move in {per} overall, so playing this one move in {r.errors} of "
            f"{r.reached} games is a habit, not bad luck."
            if per
            else " That is a habit, not bad luck."
        )
    else:
        detail += " Too few games yet to call it a habit."
    extra = [f for f in follow_ups]
    if extra:
        moves = "; ".join(
            f"{move_label(f.first.fen, f.san)}" + (f" ({move_label(f.first.fen, f.best_san)} is better)" if f.best_san else "")
            for f in sorted(extra, key=lambda f: f.first.ply)
        )
        detail += f" In the same games you went on to play {moves}: fix the first move and the rest goes with it."
    return Insight(
        id=f"{KEY}.{'weakness' if r.significant else 'observation'}.{_slug(r.epd + ' ' + r.uci)}",
        kind="weakness" if r.significant else "observation",
        category="positions",
        title=title,
        detail=detail,
        severity=clamp(r.weight / 60.0),
        confidence=r.confidence,
        evidence={"fen": e.fen, "move": r.san, "errors": r.errors, "reached": r.reached, "analysed": r.analysed,
                  "avg_drop": round(r.avg_drop, 1), "best": r.best_san, "base_rate": r.base_rate,
                  "p_value": r.p_value, "p_adjusted": r.p_adjusted},
        study=[
            f"Set up the position after {line} and work out why {best or 'the engine move'} beats {played}.",
            "Add the line to your repertoire notes so you recognise the position next time.",
            "Replay the line from move 1 a few times, then test yourself on it again in a few days.",
        ],
        example_games=r.example_games[:5],
    )


def analyze(ctx: AnalysisContext) -> ModuleResult:
    if not ctx.evals:
        return ModuleResult(
            key=KEY,
            title=TITLE,
            summary="Needs engine analysis. Run with --engine to find the positions you repeatedly get wrong "
            "and to export your own mistakes as puzzles.",
        )
    events = collect_errors(ctx.games, ctx.evals)
    repeated = assess(find_repeated(events, ctx.games, ctx.evals), *error_rate(ctx.games, ctx.evals))
    distinct, follow_ups = fold_follow_ups(repeated)
    claims = [r for r in distinct if r.significant]
    habits = sum(r.significant for r in repeated)  # counting follow-ups folded into an earlier finding
    puzzles = [e for e in events if e.drop >= PUZZLE_MIN_DROP and e.best_san]
    exported = min(len(puzzles), PUZZLE_LIMIT)
    n_analysed = sum(1 for g in ctx.games if g.game_id in ctx.evals)

    puzzle_file = str(ctx.opt("puzzle_file", "") or "")
    puzzle_hint = f"in {puzzle_file}" if puzzle_file else "export with --puzzles"
    if len(puzzles) > PUZZLE_LIMIT:
        puzzle_hint = f"the {PUZZLE_LIMIT} costliest of {len(puzzles)}; " + puzzle_hint
    if exported > LICHESS_STUDY_CHAPTERS:
        puzzle_hint += f" (a Lichess study holds {LICHESS_STUDY_CHAPTERS}: start with the first ones)"
    kpis = [
        Kpi(
            "Wrong moves you repeated",
            len(repeated),
            "int",
            hint=f"{habits} of them a clear habit",
        ),
        Kpi("Games with those moves", sum(r.errors for r in repeated), "int"),
        Kpi("Puzzles from your games", exported, "int", hint=puzzle_hint),
    ]
    rows = []
    for r in repeated[:15]:
        e = r.first
        rows.append(
            [
                format_line(e.game.moves_san[: e.ply], e.game.initial_fen, last_n=8),
                e.game.opening_family or "",
                r.reached,
                r.errors,
                move_label(e.fen, r.san),
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
                columns=[
                    "Moves to reach it", "Opening", "Reached", "Played it", "Your move", "Engine move",
                    "Avg win chance lost", "Game",
                ],
                rows=rows,
                formats=["text", "text", "int", "int", "text", "text", "pct", "url"],
                note=f"A wrong move lost at least {MIN_DROP:.0f} percentage points of winning chances in Stockfish's "
                "analysis. Reached and Played it count all your games, analysed or not; positions are matched "
                "exactly, so the same position reached by a different move order counts together.",
            )
        )

    costliest = sorted(puzzles, key=lambda e: (-e.drop, e.game.end_time))[:PUZZLE_TABLE_ROWS]
    if costliest:
        tables.append(
            Table(
                title="Your costliest mistakes (puzzles)",
                columns=["Opening", "Your move", "Better", "Winning chances lost", "Position", "Game"],
                rows=[
                    [
                        e.game.opening_family or "",
                        e.move_label,
                        e.best_label or "",
                        e.drop / 100.0,
                        "https://lichess.org/analysis/" + e.fen.replace(" ", "_"),
                        e.game.url,
                    ]
                    for e in costliest
                ],
                formats=["text", "text", "text", "pct", "url", "url"],
                key_columns=[1, 2, 4],
                note="Open the position (a Lichess analysis board) and find the better move before you look at it. "
                + (f"All {exported} are in {puzzle_file}." if puzzle_file else "Export them all with --puzzles."),
            )
        )

    diagrams = []
    for r in distinct[:MAX_DIAGRAMS]:
        e = r.first
        diagrams.append(
            Diagram(
                title=f"{e.game.opening_family or 'Position'}: {move_label(e.fen, r.san)}",
                fen=e.fen,
                svg=_svg(e),
                caption=f"After {format_line(e.game.moves_san[: e.ply], e.game.initial_fen, last_n=6)}. "
                f"Red: your move (in {r.errors} of {_plural(r.reached, 'game')}). Green: engine's choice.",
                link=e.game.url,
            )
        )

    # The claims first (most costly first), then the most costly of the rest as observations.
    shown = (claims + [r for r in distinct if not r.significant])[:MAX_INSIGHTS]
    insights = [_insight(r, follow_ups.get(id(r), ())) for r in shown]

    if repeated:
        summary = (
            f"In {n_analysed} analysed games Stockfish found {_plural(len(repeated), 'wrong move')} you played in "
            f"more than one game in the same position"
            + (f", {habits} of them often enough to be a habit" if habits else "")
            + ". These are the cheapest points to win back: learn the right move once and it pays off every time "
            "the position comes up."
        )
    else:
        summary = (
            f"No wrong move you played in more than one game in the same position, in {n_analysed} analysed games. "
            + (f"{len(puzzles)} of your mistakes are available as puzzles." if len(puzzles) != 1
               else "One of your mistakes is available as a puzzle.")
        )
    return ModuleResult(
        key=KEY,
        title=TITLE,
        summary=summary,
        kpis=kpis,
        tables=tables,
        insights=insights,
        diagrams=diagrams,
        stats={"repeated_positions": len(repeated), "claims": len(claims), "puzzles": len(puzzles),
               "puzzles_exported": exported, "errors": len(events), "puzzle_file": puzzle_file,
               "follow_ups_folded": sum(len(v) for v in follow_ups.values())},
    )


# --------------------------------------------------------------------------- puzzle export
def build_puzzles(
    games: Iterable[Game], evals: dict[str, GameEval], min_drop: float = PUZZLE_MIN_DROP, limit: int = PUZZLE_LIMIT
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
