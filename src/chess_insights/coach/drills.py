"""Puzzle packs for your most-missed motifs and your openings, and the review schedule (C2). Owner: drills.

* **Motif packs**: your three most-missed patterns (the motif profile's counts; failing that, the patterns in the
  positions this report explains, then the engine's own tags: material left hanging, missed mates) each get
  ``cfg.drill_size`` Lichess puzzles of that theme in the rating window, the most popular first (then the most
  played), chosen deterministically. With ``cfg.out_stem`` each pack is written as ``<stem>-drill-<theme>.pgn``.
* **Rating window** (:func:`drill_window`): ``cfg.drill_rating`` when given (``--drill-rating``), kept inside the
  puzzle collection's 800-2200; else (None, the default) your level: the Lichess equivalent
  (``rating_map.lichess_equivalent``) of your latest chess.com rating in your most-played rated format among blitz
  and rapid (else bullet), +-200, inside 800-2200. A note says which window was used and why.
* **Openings pack**: puzzles whose Lichess ``OpeningTags`` (set only for puzzles starting before move 20) match the
  opening families of your three most-played lines with each colour, where you solve for the colour you play
  them with when possible: ``<stem>-drill-openings.pgn``.
* **Review schedule**: your ten costliest mistakes (``analysis.mistakes.build_puzzles``) and the first five puzzles
  of each pack come back 1, 3, 7 and 21 days after the report; items from the previous report's JSON that are due
  move on to their next step, and after the last one they are done (``coaching.settings["review_done"]``) and the
  next mistake or puzzle takes their place.

The packs need the filtered puzzle database (``chess-insights puzzles-db``) at ``cfg.puzzle_db``; without it the
report gets one note instead of packs. Nothing here is a claim: packs and reasons are practice material.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

import chess
import chess.pgn

from ..models import Coaching, Drill, DrillPuzzle, Game, ModuleResult, ReviewItem
from . import puzzles_db, rating_map
from .config import DEFAULT_DRILL_RATING, REVIEW_STEPS_DAYS, CoachConfig
from .profile import motif_name, training_url

if TYPE_CHECKING:
    from ..context import AnalysisContext

NO_DB_NOTE = "Run chess-insights puzzles-db once to get drill packs"
MAX_PACKS = 3  # motif packs
LINES_PER_COLOUR = 3  # most-played lines whose families make the openings pack
OWN_REVIEW = 10  # your costliest mistakes in the review schedule
PACK_REVIEW = 5  # the first puzzles of each pack in the review schedule
OPENINGS = "openings"
OPENINGS_URL = "https://lichess.org/training/openings"
# engine tags (engine.py) -> the Lichess theme that trains them, for when no motif counts exist
TAG_THEMES = (("hung_material", "hangingPiece"), ("missed_mate", "mate"))
TAG_WORDS = {  # (one, several)
    "hung_material": ("left material hanging once", "left material hanging {n} times"),
    "missed_mate": ("missed a forced mate once", "missed {n} forced mates"),
}
MAX_DONE = 2000  # review items remembered as done (the most recent reports' worth)
WINDOW_HALF = 200  # the automatic window: your Lichess-equivalent rating +-200
MIN_WINDOW = 200  # the narrowest automatic window (a rating near or past an end of the puzzle collection)
WINDOW_FORMATS = (("blitz", "rapid"), ("bullet",))  # your most-played rated format in the first group that has one


def _plural(n: int, word: str, plural: Optional[str] = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


# --------------------------------------------------------------------------- which themes
def _profile_themes(coaching: Coaching) -> list[tuple[str, str]]:
    """(theme, reason) from the motif profile's counts: the patterns you missed most, most first."""
    profile = (coaching.settings or {}).get("motif_profile") or {}
    counts = profile.get("counts") or {}
    games = profile.get("games") or 0

    def n(theme: str, key: str) -> int:
        value = (counts.get(theme) or {}).get(key)
        return int(value) if isinstance(value, (int, float)) else 0

    findings = set(profile.get("findings") or [])
    ranked = sorted((t for t in counts if n(t, "you_missed") > 0),
                    key=lambda t: (-n(t, "you_missed"), -n(t, "you_allowed"), t))

    def reason(t: str) -> str:
        text = (f"you missed {_plural(n(t, 'you_missed'), motif_name(t), motif_name(t, True))} in "
                f"{_plural(int(games), 'game')}; your opponents {n(t, 'opp_missed')}")
        if not any(f.endswith(f".{t}") for f in findings):  # counts, not a tested difference: say so
            text += " (counts from the tactical patterns table, not a tested finding)"
        return text

    return [(t, reason(t)) for t in ranked]


def _explanation_themes(coaching: Coaching, gated: Iterable[str]) -> list[tuple[str, str]]:
    """(theme, reason) from the explained positions: their drill themes, else their named patterns."""
    gated = set(gated)
    counter: Counter = Counter()
    for e in coaching.explanations or []:
        themes = list(e.drill_themes or []) or [m.theme for m in e.motifs or [] if m.theme in gated]
        counter.update(set(themes))
    ranked = sorted(counter, key=lambda t: (-counter[t], t))
    return [
        (t, f"suggested for {_plural(counter[t], 'position')} explained in this report")
        for t in ranked
    ]


def _tag_themes(ctx: "AnalysisContext") -> list[tuple[str, str]]:
    """(theme, reason) from the engine's tags on your moves: material left hanging, missed mates."""
    from ..analysis.engine_stats import join

    records = join(ctx)
    found = []
    for tag, theme in TAG_THEMES:
        mine = sum(1 for r in records for p in r.plies if p.is_user and tag in (p.tags or ()))
        theirs = sum(1 for r in records for p in r.plies if not p.is_user and tag in (p.tags or ()))
        if mine:
            one, several = TAG_WORDS[tag]
            words = one if mine == 1 else several.format(n=mine)
            found.append((mine, theme, f"you {words} in {_plural(len(records), 'game')}; your opponents {theirs}"))
    return [(theme, reason) for _, theme, reason in sorted(found, key=lambda x: (-x[0], x[1]))]


def pick_themes(ctx: "AnalysisContext", coaching: Coaching, gated: Iterable[str] = (),
                n: int = MAX_PACKS) -> list[tuple[str, str]]:
    """Up to ``n`` (theme, reason): the motif profile first, then the explanations, then the engine's tags."""
    picked: list[tuple[str, str]] = []
    for source in (lambda: _profile_themes(coaching), lambda: _explanation_themes(coaching, gated),
                   lambda: _tag_themes(ctx)):
        for theme, reason in source():
            if len(picked) >= n:
                return picked
            if theme not in {t for t, _ in picked}:
                picked.append((theme, reason))
    return picked


# --------------------------------------------------------------------------- opening families
_GENERIC = {"opening", "game"}  # chess.com "Ruy Lopez Opening" is Lichess's "Ruy_Lopez"
_TOKEN_ALIASES = {"alekhines": "alekhine", "owens": "owen", "birds": "bird", "defence": "defense",
                  "nimzowitsch-larsen": "nimzo-larsen"}
_FAMILY_ALIASES = {("petrovs", "defense"): [("russian",), ("petrovs", "defense")]}


@lru_cache(maxsize=16384)
def name_tokens(name: str) -> tuple[str, ...]:
    """'Queen's Gambit' / 'Queens_Gambit' -> ('queens', 'gambit'); accents and 'Opening' / 'Game' dropped."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"['’:,.]", "", text).replace("_", " ")
    words = [_TOKEN_ALIASES.get(w, w) for w in text.split()]
    return tuple(w for w in words if w not in _GENERIC)


def family_keys(family: str) -> list[tuple[str, ...]]:
    tokens = name_tokens(family)
    return _FAMILY_ALIASES.get(tokens, [tokens]) if tokens else []


def tags_match(keys: Sequence[tuple[str, ...]], tags: Sequence[str]) -> bool:
    """A puzzle's OpeningTags belong to the family: its family tag starts with the name, or (for a name of two words
    or more, such as 'London System' or 'Evans Gambit') a deeper tag's variation part contains it."""
    if not tags:
        return False
    top = name_tokens(tags[0])
    deeper = [name_tokens(t) for t in tags[1:]]
    for key in keys:
        if not key:
            continue
        if top[: len(key)] == key:
            return True
        if len(key) >= 2:
            for t in deeper:
                rest = t[len(top):] if t[: len(top)] == top else t
                if any(rest[i: i + len(key)] == key for i in range(len(rest) - len(key) + 1)):
                    return True
    return False


def top_families(games: Iterable[Game], lines: int = LINES_PER_COLOUR) -> list[tuple[str, str, int]]:
    """(colour, family, games) for the families of your ``lines`` most-played lines with each colour."""
    games = list(games)
    out: list[tuple[str, str, int]] = []
    for colour in ("white", "black"):
        mine = [g for g in games if g.color == colour and g.opening]
        counts = Counter(g.opening for g in mine)
        families: dict[str, int] = {}
        for line, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:lines]:
            fam = Counter(g.opening_family for g in mine if g.opening == line and g.opening_family).most_common(1)
            family = fam[0][0] if fam else str(line).split(":")[0].strip()
            families[family] = families.get(family, 0) + n
        out += [(colour, f, n) for f, n in families.items() if f]
    return sorted(out, key=lambda x: (-x[2], x[0], x[1]))


# --------------------------------------------------------------------------- choosing puzzles
def _order(row: puzzles_db.PuzzleRow) -> tuple:
    return (-row.popularity, -row.nb_plays, row.puzzle_id)


def collect_candidates(
    path: Path, themes: Sequence[str], families: Sequence[tuple[str, str, int]], rating: tuple[int, int]
) -> tuple[dict[str, list[puzzles_db.PuzzleRow]], list[list[puzzles_db.PuzzleRow]]]:
    """One pass over the subset: in-window rows per theme, and per (colour, family) group, best first.

    Opening rows are ordered by whether you solve for the colour you play that opening with, then popularity.
    """
    lo, hi = rating
    wanted = set(themes)
    by_theme: dict[str, list[puzzles_db.PuzzleRow]] = {t: [] for t in themes}
    keys = [family_keys(f) for _, f, _ in families]
    by_family: list[list[puzzles_db.PuzzleRow]] = [[] for _ in families]
    matches: dict[tuple[str, ...], list[int]] = {}  # opening tags -> the groups they belong to (tags repeat a lot)
    for row in puzzles_db.iter_rows(path, rating=(lo, hi), themes=wanted, tagged=bool(keys)):
        for theme in wanted.intersection(row.themes):
            by_theme[theme].append(row)
        if row.opening_tags:
            if row.opening_tags not in matches:
                matches[row.opening_tags] = [i for i, k in enumerate(keys) if tags_match(k, row.opening_tags)]
            for i in matches[row.opening_tags]:
                by_family[i].append(row)
    for rows in by_theme.values():
        rows.sort(key=_order)
    for (colour, _, _), rows in zip(families, by_family):
        rows.sort(key=lambda r, c=colour: (r.solver != c, *_order(r)))
    return by_theme, by_family


def _take(rows: Iterable[puzzles_db.PuzzleRow], n: int, used: set[str]) -> list[DrillPuzzle]:
    """The first ``n`` rows not used by another pack that are legal chess, as puzzles."""
    out = []
    for row in rows:
        if len(out) >= n:
            break
        if row.puzzle_id in used:
            continue
        puzzle = puzzles_db.to_puzzle(row)
        if puzzle is not None:
            used.add(row.puzzle_id)
            out.append(puzzle)
    return out


def _round_robin(groups: Sequence[Sequence[puzzles_db.PuzzleRow]], n: int, used: set[str]) -> list[DrillPuzzle]:
    """``n`` puzzles taken in turn from each group (best first), skipping repeats."""
    iters = [iter(g) for g in groups]
    out: list[DrillPuzzle] = []
    while iters and len(out) < n:
        alive = []
        for it in iters:
            if len(out) >= n:
                break
            got = _take(it, 1, used)
            if got:
                out += got
                alive.append(it)
        iters = alive
    return out


# --------------------------------------------------------------------------- PGN
def _header_text(text: str) -> str:
    """A PGN header value from the puzzle file: no quotes, brackets, braces or backslashes (python-chess writes
    header values as they are)."""
    return re.sub(r'["\\\[\]{}\s]+', " ", text).strip()


def pack_pgn(drill: Drill) -> str:
    """One PGN game per puzzle: SetUp/FEN = the position after the opponent's first move, the solution moves, and
    a comment with the Lichess link. Readable by Lichess study import and any chess program."""
    chunks = []
    for i, p in enumerate(drill.puzzles, 1):
        board = chess.Board(p.fen)
        game = chess.pgn.Game()
        game.setup(board)
        game.headers["Event"] = f"{drill.title} ({i}/{len(drill.puzzles)})"
        game.headers["Site"] = p.url
        game.headers["Date"] = "????.??.??"
        game.headers["Round"] = str(i)
        game.headers["White"] = "?"
        game.headers["Black"] = "?"
        game.headers["Result"] = "*"
        game.headers["Annotator"] = "chess-insights"
        game.headers["PuzzleId"] = p.puzzle_id
        game.headers["PuzzleRating"] = str(p.rating)
        game.headers["Themes"] = _header_text(" ".join(p.themes))
        if p.opening_tags:
            game.headers["Opening"] = _header_text(p.opening_tags[-1].replace("_", " "))
        side = "White" if board.turn == chess.WHITE else "Black"
        game.comment = f"{side} to move. Lichess puzzle {p.puzzle_id}, rated {p.rating}: {p.url}"
        node: chess.pgn.GameNode = game
        for uci in p.solution_uci:
            node = node.add_variation(chess.Move.from_uci(uci))
        chunks.append(str(game))
    return "\n\n".join(chunks) + ("\n" if chunks else "")


def pack_path(out_stem: Path, theme: str) -> Path:
    """<stem>-drill-<theme>.pgn next to the report (the theme reduced to letters, digits, '-' and '_')."""
    stem = Path(out_stem)
    safe = re.sub(r"[^A-Za-z0-9_-]", "", theme) or "pack"
    return stem.parent / f"{stem.name}-drill-{safe}.pgn"


def _write(drill: Drill, out_stem: Optional[Path], notes: list[str]) -> None:
    """Write the pack next to the report, atomically (a failed write leaves the earlier file whole, never a
    truncated one), as the report files are."""
    from ..report import _clean_text, _write_atomic

    if not out_stem or not drill.puzzles:
        return
    path = pack_path(out_stem, drill.theme)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, _clean_text(pack_pgn(drill)))
    except OSError as exc:
        notes.append(f"The {drill.theme} drill pack could not be written ({exc}).")
        return
    drill.file = path.name


# --------------------------------------------------------------------------- the rating window
def player_rating(games: Iterable[Game]) -> Optional[tuple[str, int, int]]:
    """(format, your latest chess.com rating in it, your rated games in it) for your most-played rated format among
    blitz and rapid (a tie goes to blitz), else bullet; None without a rated game in those formats."""
    games = [g for g in games if g.rated and g.my_rating]
    for group in WINDOW_FORMATS:
        counts = Counter(g.time_class for g in games if g.time_class in group)
        if not counts:
            continue
        tc = max(group, key=lambda t: (counts[t], -group.index(t)))
        latest = max((g for g in games if g.time_class == tc), key=lambda g: g.end_time)
        return tc, int(latest.my_rating), counts[tc]
    return None


def clip_window(lo: int, hi: int) -> Optional[tuple[int, int]]:
    """``lo``-``hi`` inside the puzzle collection's ratings (``puzzles_db.MIN_RATING``-``MAX_RATING``); None when
    they do not overlap."""
    lo, hi = max(int(lo), puzzles_db.MIN_RATING), min(int(hi), puzzles_db.MAX_RATING)
    return (lo, hi) if lo < hi else None


def window_around(rating: int) -> tuple[int, int]:
    """``rating`` +-``WINDOW_HALF``, inside the puzzle collection and at least ``MIN_WINDOW`` wide."""
    lo = min(max(rating - WINDOW_HALF, puzzles_db.MIN_RATING), puzzles_db.MAX_RATING - MIN_WINDOW)
    hi = max(min(rating + WINDOW_HALF, puzzles_db.MAX_RATING), puzzles_db.MIN_RATING + MIN_WINDOW)
    return lo, hi


def drill_window(games: Iterable[Game], cfg: CoachConfig) -> tuple[tuple[int, int], str, dict[str, Any]]:
    """(the puzzle rating window, why it is that window in words, the same as data for the JSON).

    ``cfg.drill_rating`` (``--drill-rating``) when it overlaps the puzzle collection (cut to it); else your level
    (:func:`player_rating` on the Lichess scale, +-``WINDOW_HALF``); else ``DEFAULT_DRILL_RATING``. The words follow
    "Puzzle packs are rated 1190–1590" (:func:`window_note`).
    """
    held = f"{puzzles_db.MIN_RATING}–{puzzles_db.MAX_RATING}"
    chosen = (int(cfg.drill_rating[0]), int(cfg.drill_rating[1])) if cfg.drill_rating else None
    unusable = ""
    if chosen is not None:
        window = clip_window(*chosen)
        if window == chosen:
            return window, ", the range you chose (--drill-rating).", {"source": "chosen", "chosen": list(chosen)}
        if window is not None:
            return window, (f": the part of the range you chose (--drill-rating {chosen[0]}-{chosen[1]}) that the "
                            f"puzzle collection holds ({held})."), {"source": "chosen", "chosen": list(chosen)}
        unusable = (f" The range you chose (--drill-rating {chosen[0]}-{chosen[1]}) has no puzzles: the collection "
                    f"holds {held}.")
    found = player_rating(games)
    if found is None:
        lo, hi = DEFAULT_DRILL_RATING
        return (lo, hi), (": none of these games is a rated blitz, rapid or bullet game to read your level from. "
                          "Choose another range with --drill-rating." + unusable), {"source": "default"}
    tc, rating, n = found
    lichess = rating_map.lichess_equivalent(rating, tc, cfg.rating_map)
    window = window_around(lichess)
    games_word = "game" if n == 1 else "games"
    which = (f"your most-played rated format among blitz and rapid, {n} {games_word}" if tc != "bullet" else
             f"this report has no rated blitz or rapid games; {n} rated bullet {games_word}")
    why = (f", around your level: your chess.com {tc} rating of {rating} ({which}) is about {lichess} on Lichess, "
           f"where the puzzles come from ({rating_map.source_of(rating, tc, cfg.rating_map)})")
    if window != (lichess - WINDOW_HALF, lichess + WINDOW_HALF):
        why += f"; the puzzle collection holds {held}"
    why += ". Choose another range with --drill-rating." + unusable
    return window, why, {"source": "auto", "time_class": tc, "chesscom": rating, "lichess": lichess, "games": n}


def window_note(window: Sequence[int], why: str, packs: bool = True) -> str:
    """"Puzzle packs are rated 1190–1590, around your level: ..." (``why`` from :func:`drill_window`); without
    packs, "No drill packs: the puzzle database has no puzzles for your patterns and openings rated ..."."""
    lo, hi = int(window[0]), int(window[1])
    if packs:
        return f"Puzzle packs are rated {lo}–{hi}{why}"
    return f"No drill packs: the puzzle database has no puzzles for your patterns and openings rated {lo}–{hi}{why}"


# --------------------------------------------------------------------------- packs
def build_packs(
    ctx: "AnalysisContext", coaching: Coaching, cfg: CoachConfig, gated: Iterable[str] = (),
    window: Optional[tuple[int, int]] = None,
) -> list[Drill]:
    """The motif packs and the openings pack (not written yet), with puzzles rated inside ``window`` (default:
    :func:`drill_window`)."""
    themes = pick_themes(ctx, coaching, gated)
    families = top_families(ctx.games)
    lo, hi = window or drill_window(ctx.games, cfg)[0]
    size = max(0, int(cfg.drill_size))
    by_theme, by_family = collect_candidates(Path(cfg.puzzle_db), [t for t, _ in themes], families, (lo, hi))
    used: set[str] = set()
    packs: list[Drill] = []
    for theme, reason in themes:
        puzzles = _take(by_theme.get(theme, []), size, used)
        if puzzles:
            packs.append(Drill(theme=theme, title=f"{len(puzzles)} {motif_name(theme)} puzzles",
                               link=training_url(theme), rating_range=(lo, hi), puzzles=puzzles, reason=reason))
    groups = [rows for rows in by_family if rows]
    puzzles = _round_robin(groups, size, used)
    if puzzles:
        names = []
        for (colour, family, _), rows in zip(families, by_family):
            if rows and family not in names:
                names.append(family)
        listed = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
        packs.append(Drill(
            theme=OPENINGS, title=f"{len(puzzles)} puzzles from your openings", link=OPENINGS_URL,
            rating_range=(lo, hi), puzzles=puzzles,
            reason=f"from the {listed}: the openings of your most-played lines, solving for your side where "
                   "possible",
        ))
    return packs


# --------------------------------------------------------------------------- review schedule
def _days(step: int) -> timedelta:
    return timedelta(days=REVIEW_STEPS_DAYS[min(max(step, 0), len(REVIEW_STEPS_DAYS) - 1)])


def _analysis_url(fen: str, chess960: bool = False) -> str:
    """The position on Lichess's analysis board (Chess960 castling rights need the variant in the path)."""
    return "https://lichess.org/analysis/" + ("chess960/" if chess960 else "") + fen.replace(" ", "_")


def own_items(ctx: "AnalysisContext", limit: int = OWN_REVIEW, skip: Iterable[str] = ()) -> list[ReviewItem]:
    """Your ``limit`` costliest mistakes (with a known better move) that are not done yet (``skip``), as review
    items."""
    from ..analysis.mistakes import build_puzzles

    skip = set(skip)
    items = []
    for e in build_puzzles(ctx.games, ctx.evals, limit=limit + len(skip)):
        item_id = f"own:{e.game.game_id}:{e.ply}"
        if item_id in skip:
            continue
        where = ", ".join(x for x in (e.game.opening_family or "", e.game.time_class or "") if x)
        items.append(ReviewItem(
            item_id=item_id,
            kind="own",
            title=f"Find a better move than {e.move_label}" + (f" ({where})" if where else ""),
            fen=e.fen,
            url=_analysis_url(e.fen, e.game.rules == "chess960"),
            due="",
        ))
        if len(items) >= limit:
            break
    return items


def drill_items(packs: Sequence[Drill], per_pack: int = PACK_REVIEW, skip: Iterable[str] = ()) -> list[ReviewItem]:
    """The first ``per_pack`` puzzles of each pack that are not done yet (``skip``), as review items."""
    skip = set(skip)
    items = []
    for d in packs:
        what = "Puzzle from your openings" if d.theme == OPENINGS else f"{motif_name(d.theme).capitalize()} puzzle"
        fresh = [p for p in d.puzzles if f"lichess:{p.puzzle_id}" not in skip]
        for p in fresh[:per_pack]:
            items.append(ReviewItem(item_id=f"lichess:{p.puzzle_id}", kind="drill",
                                    title=f"{what}, rated {p.rating}", fen=p.fen, url=p.url, due=""))
    return items


def _parse_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def previous_items(previous: Optional[dict[str, Any]], key: str = "review") -> list[ReviewItem]:
    """ReviewItems stored in the previous report's JSON (``previous["coaching"][key]``); unreadable ones skipped."""
    coaching = (previous or {}).get("coaching") if isinstance(previous, dict) else None
    raw = coaching.get(key) if isinstance(coaching, dict) else None
    items: list[ReviewItem] = []
    seen: set[str] = set()
    for d in raw if isinstance(raw, list) else []:
        if not isinstance(d, dict) or not d.get("item_id") or _parse_date(d.get("due")) is None:
            continue
        step = d.get("step")
        item = ReviewItem(
            item_id=str(d["item_id"]),
            kind=str(d.get("kind") or ("own" if str(d["item_id"]).startswith("own:") else "drill")),
            title=str(d.get("title") or ""),
            fen=str(d.get("fen") or ""),
            url=str(d.get("url") or ""),
            due=str(d["due"])[:10],
            step=int(step) if isinstance(step, (int, float)) and not isinstance(step, bool) else 0,
        )
        if item.item_id not in seen:
            seen.add(item.item_id)
            items.append(item)
    return items


def done_items(previous: Optional[dict[str, Any]], today: date) -> list[str]:
    """Ids of the items that have finished their last step (21 days): those already done in earlier reports
    (``previous["coaching"]["settings"]["review_done"]``, and the last report's due items at their last step), and
    those finishing now. They are not scheduled again. Most recent first, at most ``MAX_DONE``."""
    last = len(REVIEW_STEPS_DAYS) - 1
    now = [i.item_id for i in previous_items(previous, "review")
           if i.step >= last and (_parse_date(i.due) or date.max) <= today]
    coaching = (previous or {}).get("coaching") if isinstance(previous, dict) else None
    settings = coaching.get("settings") if isinstance(coaching, dict) else None
    stored = settings.get("review_done") if isinstance(settings, dict) else None
    earlier = [str(x) for x in stored if x] if isinstance(stored, list) else []
    earlier += [i.item_id for i in previous_items(previous, "review_due") if i.step >= last]
    return list(dict.fromkeys(now + earlier))[:MAX_DONE]


def schedule(
    new: Sequence[ReviewItem], previous: Optional[dict[str, Any]], today: date
) -> tuple[list[ReviewItem], list[ReviewItem]]:
    """(this report's schedule, the items due now).

    Items from the previous report that are due (due date <= ``today``) are listed as due and move to their next
    step (1, 3, 7, 21 days), counted from today; after the 21-day step they are done (:func:`done_items`) and leave
    the schedule for good. Items not yet due carry over unchanged. New items start at step 0 (due tomorrow) unless
    they are already scheduled or done. Sorted by due date, your own positions first; deterministic given ``today``.
    """
    last = len(REVIEW_STEPS_DAYS) - 1
    carried: list[ReviewItem] = []
    due_now: list[ReviewItem] = []
    finished = set(done_items(previous, today))
    for item in previous_items(previous, "review"):
        due = _parse_date(item.due)
        if due is not None and due <= today:
            due_now.append(item)
            if item.step < last:
                carried.append(replace(item, step=item.step + 1, due=(today + _days(item.step + 1)).isoformat()))
        else:
            carried.append(item)
    known = {i.item_id for i in carried} | finished
    fresh: list[ReviewItem] = []
    for item in new:
        if item.item_id not in known:
            known.add(item.item_id)
            fresh.append(replace(item, step=0, due=(today + _days(0)).isoformat()))
    review = sorted(carried + fresh, key=lambda i: (i.due, i.kind != "own"))
    return review, due_now


# --------------------------------------------------------------------------- the coaching step
def annotate(ctx: "AnalysisContext", coaching: Coaching, modules: list[ModuleResult], cfg: CoachConfig,
             today: date) -> None:
    """Fill ``coaching.drills`` (and write the PGN packs next to the report), ``coaching.review`` and
    ``coaching.review_due``."""
    from . import motifs

    packs: list[Drill] = []
    path = Path(cfg.puzzle_db) if cfg.puzzle_db else None
    if path is None or not path.is_file():
        coaching.notes.append(NO_DB_NOTE + ".")
    else:
        window, why, basis = drill_window(ctx.games, cfg)
        try:  # a broken subset costs the packs, not the review schedule of your own positions
            packs = build_packs(ctx, coaching, cfg, getattr(motifs, "GATED_THEMES", ()) or (), window)
        except Exception as exc:  # noqa: BLE001
            coaching.notes.append(f"No drill packs: the puzzle database at {path.name} could not be read "
                                  f"({type(exc).__name__}: {exc}). Run chess-insights puzzles-db again.")
            packs = []
        else:
            for drill in packs:
                _write(drill, cfg.out_stem, coaching.notes)
            meta = puzzles_db.read_meta(path)
            coaching.settings["puzzle_db"] = {
                "downloaded": meta.get("downloaded", ""), "rows": meta.get("rows"),
                "license": meta.get("license", "CC0"), "rating": list(window), "size": cfg.drill_size,
                "rating_basis": basis, "rating_note": window_note(window, why),
            }
            coaching.notes.append(window_note(window, why, packs=bool(packs)))
    coaching.drills = packs
    done = done_items(cfg.previous, today)
    new = own_items(ctx, skip=done) + drill_items(packs, skip=done)
    coaching.review, coaching.review_due = schedule(new, cfg.previous, today)
    coaching.settings["review_done"] = done
