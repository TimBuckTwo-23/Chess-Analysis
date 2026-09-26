"""Opening facts from public sources for explained positions and your choice points; where you leave theory (C3).

For each explained opening position (before ply ``MAX_OPENING_PLY``) and each of the openings module's choice
points, :func:`gather` collects an ``OpeningFacts``:

* the chess-openings name of the position (or the nearest named one before it), bundled, always available;
* what masters play, and what Lichess players one or two rating groups above you play (opening explorer, needs a
  token), with where your move ranks and one master game;
* up to three engine lines from the Lichess cloud eval;
* two credited sentences from Wikibooks Chess Opening Theory, from the nearest page it has.

Explanations get it as ``Explanation.opening``; the choice points and your repeated mistakes come together in a
table "What stronger players play here" under the openings tables, each with a board. Separately,
:func:`theory_exit` measures where each game leaves named opening theory and where your first inaccuracy comes
(``Coaching.theory_exit``). Everything here is description: no claim is made or changed. Owner: sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence
from urllib.parse import unquote

import chess

from ..models import (
    Chart, Coaching, Color, Diagram, Explanation, Game, ModuleResult, MoveStat, OpeningFacts, Source, Table,
)
from ..stats import MINUS, pct
from ..visuals import comparison_chart, format_counts, line_strip, move_arrow, ordered_formats
from . import rating_map
from .config import CoachConfig
from .sources import cloud_eval, lichess_explorer, openings_db, wikibooks
from .sources.http import TOKEN_NOTE, Fetcher, normalize_fen

if TYPE_CHECKING:
    from ..context import AnalysisContext

log = logging.getLogger(__name__)

MAX_OPENING_PLY = 30  # explained positions before this ply get opening facts
MAX_POSITIONS = 60  # positions per report (choice points first); network requests are capped by MAX_REQUESTS
MIN_THEORY_GAMES = 8  # games of one opening family as one colour for a row in the theory-exit table
MAX_THEORY_ROWS = 12
# network requests per report; cache hits are free, so positions past the cap get their facts in later runs
MAX_REQUESTS = {"explorer": 60, "cloud_eval": 40, "wikibooks": 60}
TABLE_TITLE = "What stronger players play here"
THEORY_TITLE = "Where you leave opening theory"
CHOICE_TABLE_TITLE = "Your choices at key moves"  # analysis/openings.choice_table
CHESS_OPENINGS = Source(name="lichess-org/chess-openings", url=openings_db.SOURCE_URL, license=openings_db.LICENSE)


# --------------------------------------------------------------------------- positions to look up
@dataclass
class Option:
    """One of your moves in a position, with your games and score with it."""

    san: str
    label: str  # "2...Nc6"
    games: int = 0
    score: Optional[float] = None  # your score with it (0..1)


@dataclass
class OpeningPosition:
    fen: str
    moves: list[str]  # SAN from the start ([] when unknown)
    color: Color  # you, the side to move
    options: list[Option]  # your moves here, most played first
    kind: str = "choice"  # "choice" | "repeated" | "error"
    time_class: str = ""  # the game's format ("" for a choice point over several formats)
    formats: dict[str, int] = field(default_factory=dict)  # games per format behind it
    explanation: Optional[Explanation] = None
    facts: Optional[OpeningFacts] = None


def line_text(moves: Sequence[str]) -> str:
    """'1.e4 c5 2.d4'; 'Start' for no moves (as analysis/openings.line_text)."""
    return " ".join(f"{i // 2 + 1}.{san}" if i % 2 == 0 else san for i, san in enumerate(moves)) or "Start"


def _san(label: str) -> str:
    """'5...e5' -> 'e5'; 'Nf3' -> 'Nf3'."""
    return (label or "").split(".")[-1].strip()


def _board(moves: Sequence[str]) -> Optional[chess.Board]:
    board = chess.Board()
    try:
        for san in moves:
            board.push_san(san)
    except ValueError:
        return None
    return board


def _fen_ply(fen: str) -> Optional[int]:
    """Plies from the standard start implied by the FEN's move counter and side to move."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    return (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)


def _moves_to(game: Game, epd: str, hint: Optional[int]) -> Optional[list[str]]:
    """The SAN moves from the start of ``game`` to the position ``epd`` (tries ply ``hint`` first)."""
    if game.initial_fen or game.rules != "chess":
        return None
    target = openings_db.normalize_epd(epd)
    if hint is not None and 0 <= hint <= game.plies:
        board = _board(game.moves_san[:hint])
        if board is not None and board.epd() == target:
            return list(game.moves_san[:hint])
    board = chess.Board()
    for ply, san in enumerate(game.moves_san[:MAX_OPENING_PLY]):
        if board.epd() == target:
            return list(game.moves_san[:ply])
        try:
            board.push_san(san)
        except ValueError:
            return None
    return None


def _explanation_positions(ctx: "AnalysisContext", explanations: Sequence[Explanation]) -> list[OpeningPosition]:
    """Explained positions before ``MAX_OPENING_PLY``: repeated mistakes first (most repeated), then the costliest."""
    by_url = {g.url: g for g in ctx.games}
    out = []
    for e in explanations:
        ply = _fen_ply(e.fen)
        if ply is None or ply >= MAX_OPENING_PLY:
            continue
        moves: Optional[list[str]] = None
        formats: dict[str, int] = {}
        for url in dict.fromkeys([e.game_url, *e.games]):
            game = by_url.get(url)
            if game is None:
                continue
            formats[game.time_class] = formats.get(game.time_class, 0) + 1
            if moves is None:
                moves = _moves_to(game, e.epd or e.fen, ply)
        out.append(
            OpeningPosition(
                fen=e.fen,
                moves=moves or [],
                color=e.color,
                options=[Option(san=_san(e.played), label=e.played, games=max(1, e.repeats))],
                kind=e.kind,
                time_class=e.time_class,
                formats=ordered_formats(formats),
                explanation=e,
            )
        )
    out.sort(key=lambda p: (p.kind != "repeated", -p.options[0].games, -(p.explanation.drop if p.explanation else 0)))
    return out


def _choice_positions(ctx: "AnalysisContext", modules: Sequence[ModuleResult]) -> list[OpeningPosition]:
    """The openings module's choice points, recomputed from the games (so each comes with its games and formats);
    read back from the module's table if that fails."""
    try:
        from ..analysis import openings

        th = openings.Thresholds.from_ctx(ctx)
        by_colour: dict[Color, list[Game]] = {"white": [], "black": []}
        for g in ctx.games:
            if openings.is_eligible(g):
                by_colour[g.color].append(g)
        points = openings.choice_points(by_colour, th)
    except Exception as exc:  # noqa: BLE001 — a changed openings module must not sink the facts
        log.debug("choice points not recomputed (%s); reading the table", exc)
        return _choice_positions_from_table(modules)
    out = []
    for p in points:
        board = _board(p.prefix)
        if board is None:
            continue
        games = [g for o in p.options for g in o.games]
        out.append(
            OpeningPosition(
                fen=board.fen(),
                moves=list(p.prefix),
                color=p.color,
                options=[Option(san=_san(o.label), label=o.label, games=o.n, score=o.summary.score) for o in p.options],
                kind="choice",
                formats=format_counts(games),
            )
        )
    return out


def _count(cell: Any) -> int:
    """A games cell as an int (0 for anything unreadable: a changed table must not sink the facts)."""
    try:
        return int(cell or 0)
    except (TypeError, ValueError):
        return 0


def _choice_positions_from_table(modules: Sequence[ModuleResult]) -> list[OpeningPosition]:
    """Choice points read from the "Your choices at key moves" table (position, your move, games, score ...)."""
    table = next(
        (t for m in modules if m.key == "openings" for t in m.tables if t.title == CHOICE_TABLE_TITLE), None
    )
    if table is None:
        return []
    grouped: dict[str, list[list[Any]]] = {}
    for row in table.rows:
        if len(row) >= 3:
            grouped.setdefault(str(row[0]), []).append(row)
    out = []
    for text, rows in grouped.items():
        moves = [] if text == "Start" else [tok.split(".")[-1] for tok in text.split()]
        board = _board(moves)
        if board is None:
            continue
        options = [
            Option(
                san=_san(str(r[1])), label=str(r[1]), games=_count(r[2]),
                score=r[3] if len(r) > 3 and isinstance(r[3], (int, float)) else None,
            )
            for r in rows
        ]
        color: Color = "white" if board.turn == chess.WHITE else "black"
        out.append(OpeningPosition(fen=board.fen(), moves=moves, color=color, options=options, kind="choice"))
    return out


# --------------------------------------------------------------------------- the facts
def _current_rating(games: Sequence[Game], time_class: str) -> Optional[int]:
    rated = [g for g in games if g.time_class == time_class and g.rated and g.my_rating]
    return max(rated, key=lambda g: g.end_time).my_rating if rated else None


def peer_groups(games: Sequence[Game], time_class: str, cfg: CoachConfig) -> tuple[list[int], str]:
    """Lichess rating groups for "players one or two groups above you", and how they were found (for a note).

    Blitz and rapid positions prefer your current chess.com rating in that format; bullet, daily and mixed
    positions your blitz rating (the explorer is asked for blitz and rapid games), then rapid, then bullet. A
    conversion checked against the rating comparison wins over a rough estimate, whatever the order: with blitz 949
    (checked) and rapid 1132 (estimated), every position uses the blitz conversion.
    """
    order = [time_class] if time_class in ("blitz", "rapid") else []
    order += [tc for tc in ("blitz", "rapid", "bullet") if tc not in order]
    found: list[tuple[str, int, list[int]]] = []
    for tc in order:
        rating = _current_rating(games, tc)
        groups = rating_map.groups_for(rating, tc, cfg.rating_map) if rating else []
        if groups and rating:
            found.append((tc, rating, groups))
    if not found:
        return [], ""
    tc, rating, groups = next(
        (f for f in found if rating_map.is_checked(f[1], f[0], cfg.rating_map)), found[0]
    )
    return groups, rating_map.describe(rating, tc, cfg.rating_map)


def gather(
    fetcher: Fetcher, fen: str, moves: Sequence[str], played: Optional[str], groups: Sequence[int]
) -> OpeningFacts:
    """Everything the public sources say about ``fen`` (reached by ``moves`` from the start, when known).

    ``played`` is your move (SAN), ranked among the masters' and your peers' moves. Sources that are unavailable
    leave their fields empty (and one note in ``fetcher.notes``).
    """
    db = openings_db.load()
    facts = OpeningFacts(peer_groups=list(groups))
    named = db.lookup(fen) or (db.nearest(moves) if moves else None)
    if named is not None:
        facts.eco, facts.name = named.eco, named.name
        facts.sources.append(CHESS_OPENINGS)
    masters = lichess_explorer.masters(fetcher, fen)
    if masters is not None and masters.moves:
        facts.masters = masters.top(3)
        facts.played_rank_masters = masters.rank(played) if played else None
        facts.master_game = masters.top_game
        if masters.source:
            facts.sources.append(masters.source)
    if groups:
        peers = lichess_explorer.lichess(fetcher, fen, groups)
        if peers is not None and peers.moves:
            facts.peers = peers.top(3)
            facts.played_rank_peers = peers.rank(played) if played else None
            if peers.source:
                facts.sources.append(peers.source)
    cloud = cloud_eval.evaluate(fetcher, fen)
    if cloud is not None and cloud.found:
        facts.cloud_lines = cloud.lines
        if cloud.source:
            facts.sources.append(cloud.source)
    if moves:
        snip = wikibooks.snippet(fetcher, moves)
        if snip is not None:
            facts.wiki_text, facts.wiki_url = snip.text, snip.url
            facts.sources.append(snip.source)
    return facts


def _has_external(facts: Optional[OpeningFacts]) -> bool:
    return facts is not None and bool(facts.masters or facts.peers or facts.cloud_lines or facts.wiki_text)


def _pawns_for_you(cp: Optional[int], mate: Optional[int]) -> str:
    """'+0.3' / '−1.5' / 'mate in 3' / 'mated in 2' from the side to move (you)."""
    if mate is not None:
        return f"mate in {mate}" if mate > 0 else f"mated in {-mate}"
    if cp is None:
        return ""
    return f"{cp / 100:+.1f}".replace("-", MINUS)


def _stats_text(stats: Sequence[MoveStat]) -> str:
    """'cxd4 91% (scores 48%) · e6 5% (scores 55%)': how often each move is played, and how it scores for the side
    that plays it."""
    return " · ".join(
        f"{m.san} {pct(m.share)}" + (f" (scores {pct(m.score)})" if m.score is not None else "") for m in stats
    )


def _engine_text(facts: OpeningFacts) -> str:
    """'cxd4 −0.1 · e6 −0.6' (the cloud's top moves, pawns for you)."""
    return " · ".join(
        f"{ln.moves_san[0]} {_pawns_for_you(ln.cp_end, ln.mate_end)}".strip()
        for ln in facts.cloud_lines
        if ln.moves_san
    )


def _reference_move(facts: OpeningFacts) -> Optional[str]:
    """The move stronger players (or the engine) choose most: masters, then your peers, then the cloud's line."""
    if facts.masters:
        return facts.masters[0].san
    if facts.peers:
        return facts.peers[0].san
    if facts.cloud_lines and facts.cloud_lines[0].moves_san:
        return facts.cloud_lines[0].moves_san[0]
    return None


def _you_text(pos: OpeningPosition) -> str:
    """'2...Nc6 (17 games, you score 41%, masters' #4) · 2...cxd4 (56 games, you score 52%, masters' #1)'."""
    facts = pos.facts
    parts = []
    for o in pos.options:
        extra = [f"{o.games} game{'s' if o.games != 1 else ''}"]
        if o.score is not None:
            extra.append(f"you score {pct(o.score)}")
        if facts is not None and facts.masters:
            ranked = [m.san for m in facts.masters]
            r = ranked.index(o.san) + 1 if o.san in ranked else None
            if r is None and o.san == _san(pos.options[0].label) and facts.played_rank_masters:
                r = facts.played_rank_masters
            extra.append(f"masters' #{r}" if r else "not in the masters' top 3")
        parts.append(f"{o.label} ({', '.join(extra)})")
    return " · ".join(parts)


def _wiki_text(facts: OpeningFacts) -> str:
    """The Wikibooks sentences, led by the position of the page they come from ("After 1.e4 c5: ...")."""
    if not facts.wiki_text:
        return ""
    path = facts.wiki_url.split(wikibooks.ROOT + "/", 1)[-1] if wikibooks.ROOT in facts.wiki_url else ""
    moves = [unquote(seg).split(".")[-1].lstrip("_") for seg in path.split("/") if seg]  # "e8%3DQ" -> "e8=Q"
    return f"After {line_text(moves)}: {facts.wiki_text}" if moves else facts.wiki_text


def _diagram(pos: OpeningPosition) -> Optional[Diagram]:
    """The position with the move stronger players choose (green) and yours (red; your other moves grey)."""
    facts = pos.facts
    if facts is None or not _has_external(facts):
        return None
    ref = _reference_move(facts)
    try:
        board = chess.Board(pos.fen)
    except ValueError:
        return None
    wanted = [(ref, "best")] if ref else []
    wanted += [(o.san, "played" if i == 0 else "neutral") for i, o in enumerate(pos.options) if o.san != ref]
    arrows, seen = [], set()
    for move, kind in wanted:
        arrow = move_arrow(board, move, kind) if move else None
        if arrow is not None and (arrow.start, arrow.end) not in seen:
            seen.add((arrow.start, arrow.end))
            arrows.append(arrow)
    who = "masters choose" if facts.masters else "stronger players choose" if facts.peers else "the engine prefers"
    caption = [f"You: {_you_text(pos)}"]
    if facts.masters:
        caption.append(f"Masters: {_stats_text(facts.masters)}")
    if facts.peers:
        caption.append(f"Lichess {rating_map.group_label(facts.peer_groups)}: {_stats_text(facts.peers)}")
    if facts.cloud_lines:
        caption.append(f"Engine (Lichess cloud, pawns for you): {_engine_text(facts)}")
    if facts.name:
        caption.append(f"Opening: {facts.name} ({facts.eco})")
    if pos.formats:
        caption.append("Your games: " + ", ".join(f"{n} {tc}" for tc, n in pos.formats.items()))
    where = f"After {line_text(pos.moves)}" if pos.moves else "Move 1" if pos.kind == "choice" else pos.options[0].label
    strips = []
    top = facts.cloud_lines[0] if facts.cloud_lines else None
    if top is not None and top.moves_uci:  # the engine's line as small boards
        value = _pawns_for_you(top.cp_end, top.mate_end)
        value = value if not value or value.startswith("mate") else f"{value} for you"
        strip = line_strip(f"The engine's line (Lichess cloud{': ' + value if value else ''})", pos.fen, top.moves_uci)
        if strip.frames:
            strips.append(strip)
    return Diagram(
        title=f"{where}: {ref} is what {who}" if ref else where,
        fen=pos.fen,
        caption=". ".join(caption) + ".",
        link=facts.master_game or (pos.explanation.game_url if pos.explanation else ""),
        orientation=pos.color,
        arrows=arrows,
        strips=strips,
        time_class=pos.time_class,
    )


def facts_table(positions: Sequence[OpeningPosition], peer_note: str, notes: Sequence[str]) -> Optional[Table]:
    """'What stronger players play here': one row per position with outside facts; empty columns are dropped."""
    rows = []
    for pos in positions:
        f = pos.facts
        if f is None or not _has_external(f):
            continue
        rows.append([
            line_text(pos.moves) if pos.moves or pos.kind == "choice" else pos.options[0].label,
            _you_text(pos),
            _stats_text(f.masters),
            _stats_text(f.peers),
            _engine_text(f),
            f"{f.name} ({f.eco})" if f.name else "",
            _wiki_text(f),
            f.wiki_url,
            f.master_game,
            _format_list(pos.formats),
        ])
    if not rows:
        return None
    columns = ["Position", "You play", "Masters play", "Players above you play", "Engine (pawns for you)",
               "Opening", "Theory (Wikibooks, CC BY-SA 4.0)", "Wikibooks page", "Master game", "Your games"]
    formats = ["text", "text", "text", "text", "text", "text", "text", "url", "url", "text"]
    keep = [i for i in range(len(columns)) if i < 2 or any(r[i] for r in rows)]
    note = [
        "Your choices at key moves and the positions you keep getting wrong, next to what stronger players "
        "choose. Percentages: how often each move is played there, and how it scores for the side that plays it. "
        "A description, not a verdict on your moves."
    ]
    if 2 in keep:
        note.append("Masters: over-the-board games of 2200+ players (Lichess opening explorer).")
    if 3 in keep and peer_note:
        note.append(peer_note)
    if 4 in keep:
        note.append("Engine: the top moves of the Lichess cloud eval, in pawns from your side.")
    if 6 in keep:
        note.append("Theory: the first sentences of the nearest Wikibooks Chess Opening Theory page (CC BY-SA 4.0).")
    if TOKEN_NOTE in notes:
        note.append("What masters and players above you choose is missing: the Lichess opening explorer needs a "
                    "token (LICHESS_TOKEN or --lichess-token).")
    key = [keep.index(i) for i in (0, 1, 2, 4) if i in keep][:3]
    return Table(
        title=TABLE_TITLE,
        columns=[columns[i] for i in keep],
        rows=[[r[i] for i in keep] for r in rows],
        formats=[formats[i] for i in keep],
        note=" ".join(note),
        key_columns=key,
    )


def _add_sources(e: Explanation, sources: Sequence[Source]) -> None:
    have = {(s.name, s.url) for s in e.sources}
    for s in sources:
        if (s.name, s.url) not in have:
            e.sources.append(s)
            have.add((s.name, s.url))


# --------------------------------------------------------------------------- theory exit
@dataclass
class _Exit:
    last_book_move: int  # the last move (move number) after which the position is a named opening position
    you_left: Optional[bool]  # the first move out of named theory was yours (None: the game ended in theory)
    engine: bool
    first_error_move: Optional[int]  # move number of your first inaccuracy/mistake/blunder (engine games)
    time_class: str


def _first_error_move(plies: Sequence[Any]) -> Optional[int]:
    for pe in plies:
        if getattr(pe, "is_user", False) and getattr(pe, "judgement", None):
            return pe.ply // 2 + 1
    return None


def _format_mix(counts: dict[str, int]) -> str:
    """'650 bullet, 950 blitz and 1,130 rapid'."""
    parts = [f"{n:,} {tc}" for tc, n in ordered_formats(counts).items()]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _format_list(counts: dict[str, int]) -> str:
    """'5 bullet · 12 blitz · 8 rapid' (a table cell)."""
    return " · ".join(f"{n:,} {tc}" for tc, n in ordered_formats(counts).items())


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def theory_exit(games: Sequence[Game], evals: dict[str, Any]) -> tuple[Optional[Table], Optional[Chart]]:
    """Per game, the last ply whose position is a named chess-openings position; per line (opening family and your
    colour), the average last book move, how often you are the one to leave it and, in engine-analysed games,
    your first inaccuracy relative to it. Observation only: (table, chart carrying the table)."""
    from ..analysis import openings

    eligible = [g for g in games if openings.is_eligible(g)]
    named = openings_db.load().last_named_many([g.moves_san for g in eligible])
    by_line: dict[tuple[str, str], list[_Exit]] = {}
    for g, k in zip(eligible, named):
        if k == 0:
            continue
        ev = evals.get(g.game_id)
        by_line.setdefault((g.color, g.opening_family or openings.UNNAMED), []).append(
            _Exit(
                last_book_move=(k - 1) // 2 + 1,
                you_left=g.is_my_ply(k) if k < g.plies else None,
                engine=ev is not None,
                first_error_move=_first_error_move(ev.plies) if ev is not None else None,
                time_class=g.time_class,
            )
        )
    lines = sorted(
        ((key, xs) for key, xs in by_line.items() if len(xs) >= MIN_THEORY_GAMES and key[1] != openings.UNNAMED),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )[:MAX_THEORY_ROWS]
    if not lines:
        return None, None

    rows, labels, exits, errors = [], [], [], []
    all_counts: dict[str, int] = {}
    engine_counts: dict[str, int] = {}
    for (color, family), xs in lines:
        label = f"{openings.display_name(family, color)} ({'White' if color == 'white' else 'Black'})"
        left = [1.0 if x.you_left else 0.0 for x in xs if x.you_left is not None]
        eng = [x for x in xs if x.engine]
        errs = [x for x in eng if x.first_error_move is not None]
        avg_exit = _mean([float(x.last_book_move) for x in xs])
        avg_err = _mean([float(x.first_error_move) for x in errs])  # type: ignore[arg-type]
        after = _mean([float(x.first_error_move - x.last_book_move) for x in errs])  # type: ignore[operator]
        mix: dict[str, int] = {}
        for x in xs:
            mix[x.time_class] = mix.get(x.time_class, 0) + 1
        rows.append([label, len(xs), avg_exit, _mean(left), len(eng), avg_err, after, _format_list(mix)])
        labels.append(label)
        exits.append(avg_exit)
        errors.append(avg_err)
        for x in xs:
            all_counts[x.time_class] = all_counts.get(x.time_class, 0) + 1
            if x.engine:
                engine_counts[x.time_class] = engine_counts.get(x.time_class, 0) + 1
    has_engine = any(r[4] for r in rows)
    note = (
        "Theory: the positions named in lichess-org/chess-openings (CC0). Theory ends: the average last move after "
        "which the position still has a name. You leave first: how often the first move out of named theory was "
        "yours."
    )
    if has_engine:
        note += (
            " First inaccuracy: the average move of your first inaccuracy, mistake or blunder in the "
            "engine-analysed games (games without one are left out); moves after theory: how many moves after the "
            "end of named theory it came (below 0: while still in theory)."
        )
    note += f" From {sum(all_counts.values()):,} games ({_format_mix(all_counts)})"
    if has_engine:
        note += f"; engine columns from {sum(engine_counts.values()):,} ({_format_mix(engine_counts)})."
    else:
        note += "."
    note += " An observation, not a tested claim."
    columns = ["Line", "Games", "Theory ends (move)", "You leave first", "Engine games", "First inaccuracy (move)",
               "Moves after theory", "Formats"]
    formats = ["text", "int", "float1", "pct", "int", "float1", "float1", "text"]
    if not has_engine:
        columns, formats, rows = columns[:4] + columns[7:], formats[:4] + formats[7:], [r[:4] + r[7:] for r in rows]
    table = Table(title=THEORY_TITLE, columns=columns, rows=rows, formats=formats, note=note,
                  key_columns=[0, 2, 5] if has_engine else [0, 2, 3])
    series: list[tuple[str, Sequence[Optional[float]]]] = [("Theory ends (move)", exits)]
    if has_engine:
        series.append(("Your first inaccuracy (move)", errors))
    chart = comparison_chart(
        THEORY_TITLE, labels, series, value_format="float1", kind="hbar", table=table,
        note="Average move number for each of your openings and colours.",
    )
    return table, chart


# --------------------------------------------------------------------------- entry point
def _guarded(notes: list[str], what: str, fn: Callable[[], Any]) -> Any:
    """Run one part of the annotation; a failure becomes one note, so the other parts still run."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — opening facts are optional; the theory exit must survive them
        log.warning("%s failed: %s", what, exc, exc_info=log.isEnabledFor(logging.DEBUG))
        text = f"{what} skipped ({type(exc).__name__}: {exc})."
        if text not in notes:
            notes.append(text)
        return None


def annotate(ctx: "AnalysisContext", coaching: Coaching, modules: list[ModuleResult], cfg: CoachConfig) -> None:
    """Set ``Explanation.opening`` for opening positions, ``coaching.theory_exit``, and add facts under the
    openings tables. The facts and the theory exit fail separately (each leaves a note), and one position that
    fails leaves the others alone."""
    notes = coaching.notes
    openings_mod = next((m for m in modules if m.key == "openings"), None)
    _guarded(notes, "Opening facts", lambda: _annotate_facts(ctx, coaching, modules, openings_mod, cfg))
    exit_table = _guarded(notes, "Theory exit", lambda: theory_exit(ctx.games, ctx.evals))
    table, chart = exit_table if exit_table is not None else (None, None)
    coaching.theory_exit = table
    if chart is not None and openings_mod is not None:
        openings_mod.charts.append(chart)


def _annotate_facts(
    ctx: "AnalysisContext", coaching: Coaching, modules: list[ModuleResult], openings_mod: Optional[ModuleResult],
    cfg: CoachConfig,
) -> None:
    notes = coaching.notes
    fetcher = Fetcher.from_config(cfg, notes, max_requests=MAX_REQUESTS)
    explained = _explanation_positions(ctx, coaching.explanations)
    choices = _choice_positions(ctx, modules) if openings_mod is not None else []
    todo = (choices + explained)[:MAX_POSITIONS]
    groups_by_tc: dict[str, tuple[list[int], str]] = {}
    peer_notes: list[str] = []
    for pos in todo:
        norm = normalize_fen(pos.fen)
        if norm is None:
            continue
        tc = pos.time_class if pos.time_class in ("blitz", "rapid") else ""
        if tc not in groups_by_tc:
            groups_by_tc[tc] = peer_groups(ctx.games, tc, cfg)
        groups, how = groups_by_tc[tc]
        played = pos.options[0].san if pos.options else None
        pos.facts = _guarded(notes, "Opening facts for a position", partial(gather, fetcher, norm, pos.moves, played,
                                                                             groups))
        if pos.facts is None:
            continue
        if pos.facts.peers and how:
            text = (f"Players above you: Lichess blitz and rapid games in rating groups "
                    f"{', '.join(str(g) for g in groups)}; {how}.")
            if text not in peer_notes:
                peer_notes.append(text)
        if pos.explanation is not None:
            pos.explanation.opening = pos.facts
            _add_sources(pos.explanation, pos.facts.sources)
    if openings_mod is not None:
        shown, seen = [], set()
        for pos in todo:  # one row and board per position: a choice point covers an explained move from it
            key = openings_db.normalize_epd(pos.fen)
            if pos.kind in ("choice", "repeated") and key not in seen:
                seen.add(key)
                shown.append(pos)
        table = facts_table(shown, " ".join(peer_notes), notes)
        if table is not None:
            openings_mod.tables.append(table)
        for pos in shown:
            diagram = _diagram(pos)
            if diagram is not None:
                openings_mod.diagrams.append(diagram)
