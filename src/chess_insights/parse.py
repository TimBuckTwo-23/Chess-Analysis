"""chess.com game JSON (or plain PGN) -> :class:`~chess_insights.models.Game`.

The PGN movetext is tokenised with regular expressions rather than replayed on
a board: a few thousand games parse in a second or two. Move legality is only
checked later, by the engine module, which has to replay positions anyway.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib.parse import unquote, urlparse

from .models import DRAW_CODES, LOSS_CODES, WIN_CODES, Game

log = logging.getLogger(__name__)

STANDARD_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_HEADER_RE = re.compile(r'^\[(\w+)\s+"((?:[^"\\]|\\.)*)"\]\s*$', re.M)
_CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\]")
_TOKEN_RE = re.compile(
    r"\{[^}]*\}"  # comment (may hold [%clk])
    r"|;[^\n]*"  # rest-of-line comment
    r"|\(|\)"  # variation brackets
    r"|\$\d+"  # NAG
    r"|\d+\.(?:\.\.)?"  # move number: "12." or "12..."
    r"|1-0|0-1|1/2-1/2|\*"  # result
    r"|[^\s{}()$;]+"  # anything else: a SAN move (validated below)
)
_SAN_RE = re.compile(
    r"^(?:O-O(?:-O)?|0-0(?:-0)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBN])?|[PNBRQK]@[a-h][1-8]|--)[+#]?$"
)
_MOVE_SEQ_RE = re.compile(r"^\d+\.")  # "3...c5" / "4.Nf3" tokens in ECOUrl slugs

# ECOUrl slugs lose apostrophes and inner hyphens; restore the common ones.
_WORD_FIXES = {
    "Queens": "Queen's",
    "Kings": "King's",
    "Bishops": "Bishop's",
    "Alekhines": "Alekhine's",
    "Petrovs": "Petrov's",
    "Birds": "Bird's",
    "Owens": "Owen's",
    "Larsens": "Larsen's",
    "Defence": "Defense",
}
_PHRASE_FIXES = {
    "Caro Kann": "Caro-Kann",
    "Nimzo Indian": "Nimzo-Indian",
    "Bogo Indian": "Bogo-Indian",
    "Nimzo Larsen": "Nimzo-Larsen",
    "Nimzowitsch Larsen": "Nimzowitsch-Larsen",
    "Semi Slav": "Semi-Slav",
    "Semi Tarrasch": "Semi-Tarrasch",
    "Hyper Accelerated": "Hyper-Accelerated",
    "Richter Rauzer": "Richter-Rauzer",
    "Blackmar Diemer": "Blackmar-Diemer",
    "Smith Morra": "Smith-Morra",
    "Schliemann Jaenisch": "Schliemann-Jaenisch",
    "Four Pawns": "Four Pawns",
}
_FAMILY_ENDINGS = {"Defense", "Game", "Opening", "Gambit", "Attack", "System", "Countergambit"}
# chess.com files some lines of one opening under a separate name; group them for the player.
_FAMILY_ALIASES = {"Giuoco Piano Game": "Italian Game"}
_UNKNOWN_OPENINGS = {"Undefined", "Unknown"}

_TERMINATION_BY_CODE = {
    "checkmated": "checkmate",
    "resigned": "resignation",
    "timeout": "timeout",
    "abandoned": "abandoned",
    "lose": "other",
    "kingofthehill": "variant",
    "threecheck": "variant",
    "bughousepartnerlose": "variant",
    "agreed": "agreement",
    "repetition": "repetition",
    "stalemate": "stalemate",
    "insufficient": "insufficient",
    "50move": "fifty_move",
    "timevsinsufficient": "timeout_vs_insufficient",
}


_KNOWN_CODES = WIN_CODES | DRAW_CODES | LOSS_CODES
_FINAL_RESULTS = ("1-0", "0-1", "1/2-1/2")
# /game/live/123 (current form), /live/game/123 (before ~2021), /game/daily/123, /daily/game/123
_GAME_URL_RE = re.compile(r"/(?:game/(live|daily)|(live|daily)/game)/(\d+)")


# --------------------------------------------------------------------------- small parsers
def parse_pgn_headers(pgn: str) -> dict[str, str]:
    return {k: v.replace('\\"', '"') for k, v in _HEADER_RE.findall((pgn or "").lstrip("\ufeff"))}


def parse_clock(text: str) -> Optional[float]:
    """'0:02:59.9' / '[%clk 1:05:03]' -> seconds."""
    m = _CLK_RE.search(text if text.startswith("[") else f"[%clk {text}]")
    if not m:
        return None
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def _movetext(pgn: str) -> str:
    """Strip the header block."""
    lines = (pgn or "").lstrip("\ufeff").splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("[") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:])


def parse_movetext(pgn: str) -> tuple[list[str], list[Optional[float]]]:
    """SAN plies and the mover's remaining clock after each ply (None when not annotated)."""
    moves: list[str] = []
    clocks: list[Optional[float]] = []
    depth = 0
    for tok in _TOKEN_RE.findall(_movetext(pgn)):
        if tok == "(":
            depth += 1
            continue
        if tok == ")":
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        if tok[0] == "{":
            if moves and clocks[-1] is None:
                m = _CLK_RE.search(tok)
                if m:
                    h, mnt, s = m.groups()
                    clocks[-1] = int(h) * 3600 + int(mnt) * 60 + float(s)
            continue
        if tok in ("1-0", "0-1", "1/2-1/2", "*"):
            break
        if tok[0] in ";$" or tok[0].isdigit() and tok.endswith("."):
            continue
        san = tok.rstrip("!?")
        if _SAN_RE.match(san):
            moves.append(san.replace("0-0-0", "O-O-O").replace("0-0", "O-O"))
            clocks.append(None)
    return moves, clocks


def parse_time_control(tc: Optional[str]) -> tuple[Optional[int], int]:
    """'600' -> (600, 0); '180+2' -> (180, 2); daily '1/86400' -> (86400, 0); '-' -> (None, 0)."""
    tc = (tc or "").strip()
    if not tc or tc == "-":
        return None, 0
    if "/" in tc:
        try:
            return int(tc.split("/", 1)[1]), 0
        except ValueError:
            return None, 0
    base, _, inc = tc.partition("+")
    try:
        return int(float(base)), int(float(inc or 0))
    except ValueError:
        return None, 0


def time_class_for(tc: Optional[str]) -> str:
    """chess.com's classification when only the TimeControl is known (PGN import).

    Estimated duration = base + 40 * increment: < 3 min bullet, < 10 min blitz, else rapid.
    """
    if tc and "/" in tc:
        return "daily"
    base, inc = parse_time_control(tc)
    if base is None:
        return "rapid"
    est = base + 40 * inc
    if est < 180:
        return "bullet"
    if est < 600:
        return "blitz"
    return "rapid"


def opening_from_eco_url(url: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """chess.com ECOUrl -> (opening, family).

    'https://www.chess.com/openings/Caro-Kann-Defense-Advance-Variation-3...c5'
        -> ('Caro-Kann Defense: Advance Variation', 'Caro-Kann Defense')
    """
    if not url:
        return None, None
    path = urlparse(url).path if "://" in url else url
    if "/openings/" in path:
        slug = path.split("/openings/", 1)[1].strip("/")
    else:
        slug = path.rstrip("/").rsplit("/", 1)[-1]
    slug = unquote(slug)
    words: list[str] = []
    for tok in slug.split("-"):
        if not tok or _MOVE_SEQ_RE.match(tok):
            break
        words.append(_WORD_FIXES.get(tok, tok))
    if not words or words[0] in _UNKNOWN_OPENINGS:
        return None, None
    name = " ".join(words)
    for bad, good in _PHRASE_FIXES.items():
        name = name.replace(bad, good)
    parts = name.split(" ")
    cut = next((i for i, w in enumerate(parts[:5]) if w in _FAMILY_ENDINGS), None)
    family_words = parts[: cut + 1] if cut is not None else parts[:2]
    family = " ".join(family_words)
    rest = " ".join(parts[len(family_words) :])
    opening = f"{family}: {rest}" if rest else family
    return opening, _FAMILY_ALIASES.get(family, family)


def normalize_termination(my_code: str, opp_code: str) -> str:
    if my_code in WIN_CODES:
        code = opp_code
    elif opp_code in WIN_CODES:
        code = my_code
    else:
        code = my_code if my_code in DRAW_CODES else opp_code
    return _TERMINATION_BY_CODE.get(code, "other")


def outcome_for(my_code: str, opp_code: str, pgn_result: Optional[str] = None, color: str = "white") -> str:
    if my_code in WIN_CODES:
        return "win"
    if my_code in LOSS_CODES or opp_code in WIN_CODES:
        return "loss"
    if my_code in DRAW_CODES:
        return "draw"
    if opp_code in LOSS_CODES:
        return "win"
    if pgn_result in ("1-0", "0-1"):
        return "win" if (pgn_result == "1-0") == (color == "white") else "loss"
    return "draw"


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _same_position(fen_a: str, fen_b: str) -> bool:
    """Compare placement/side/castling only (chess.com FENs have 4 or 6 fields)."""
    return fen_a.split()[:3] == fen_b.split()[:3]


def fill_pregame_ratings(games: list[Game], max_symmetric_change: int = 40) -> None:
    """Estimate pre-game ratings in place (``games`` sorted by end time).

    chess.com reports each player's rating *after* the game. The player's pre-game
    rating is their post-game rating from the previous rated game in the same pool
    (rules + time class). The opponent's is estimated by assuming the rating change
    was symmetric, which holds for established ratings; for large (provisional)
    swings the opponent's post-game rating is kept.
    """
    last: dict[tuple[str, str], int] = {}
    for g in games:
        if not g.rated:
            g.my_rating_before, g.opp_rating_before = g.my_rating, g.opp_rating
            continue
        pool = (g.rules, g.time_class)
        prev = last.get(pool)
        if prev is not None and g.my_rating is not None and g.opp_rating is not None:
            change = g.my_rating - prev
            g.my_rating_before = prev
            g.opp_rating_before = g.opp_rating + change if abs(change) <= max_symmetric_change else g.opp_rating
        if g.my_rating is not None:
            last[pool] = g.my_rating


def _header_datetime(date: Optional[str], time_: Optional[str]) -> Optional[datetime]:
    if not date or "?" in date:
        return None
    try:
        dt = datetime.strptime(f"{date} {time_ or '00:00:00'}", "%Y.%m.%d %H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- chess.com JSON
def _player(value: Any) -> dict[str, Any]:
    """Archive games have a player object; the current-games endpoint has only the player's API URL."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        return {"username": value.rstrip("/").rsplit("/", 1)[-1]}
    return {}


def game_url_key(url: Optional[str]) -> Optional[str]:
    """'https://www.chess.com/live/game/123' and '.../game/live/123' -> 'live:123' (None if not a game URL).

    chess.com changed the URL form of old games, so this is the era-independent identity of a game.
    Live and daily games have separate id spaces.
    """
    m = _GAME_URL_RE.search(url or "")
    if not m:
        return None
    return f"{m.group(1) or m.group(2)}:{m.group(3)}"


def _fallback_id(raw: dict[str, Any], white: dict[str, Any], black: dict[str, Any], pgn: str) -> str:
    """A stable id for a game without uuid or URL (hand-made JSON, some exports)."""
    basis = "|".join(
        str(x) for x in (raw.get("end_time"), white.get("username"), black.get("username"), raw.get("time_control"), pgn)
    )
    return "game:" + hashlib.sha1(basis.encode("utf-8", "replace")).hexdigest()[:16]


def parse_game(raw: dict[str, Any], username: str) -> Optional[Game]:
    """One raw chess.com game object -> Game.

    None if ``username`` didn't play in it, or if it has no result yet (an unfinished game, e.g.
    from the current-games endpoint).
    """
    white, black = _player(raw.get("white")), _player(raw.get("black"))
    user = username.lower()
    if str(white.get("username", "")).lower() == user:
        color, me, opp = "white", white, black
    elif str(black.get("username", "")).lower() == user:
        color, me, opp = "black", black, white
    else:
        return None

    pgn = raw.get("pgn")
    pgn = pgn if isinstance(pgn, str) else ""
    headers = parse_pgn_headers(pgn)
    my_code, opp_code = str(me.get("result") or ""), str(opp.get("result") or "")
    if my_code not in _KNOWN_CODES and opp_code not in _KNOWN_CODES and headers.get("Result") not in _FINAL_RESULTS:
        return None  # no outcome: never count an unfinished game as a draw
    moves, clocks = parse_movetext(pgn)

    tc = raw.get("time_control") or headers.get("TimeControl")
    base, inc = parse_time_control(tc)
    time_class = raw.get("time_class") or time_class_for(tc)
    if time_class == "daily":
        inc = 0
        # Archived daily games store "time spent / 10" in [%clk], not the remaining
        # time, so the values mean something different; drop them.
        clocks = [None] * len(moves)

    end_ts = raw.get("end_time")
    if end_ts is not None:
        end_time = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
    else:
        end_time = (
            _header_datetime(headers.get("EndDate"), headers.get("EndTime"))
            or _header_datetime(headers.get("UTCDate"), headers.get("UTCTime"))
            or datetime(1970, 1, 1, tzinfo=timezone.utc)
        )
    if raw.get("start_time") is not None:
        start_time: Optional[datetime] = datetime.fromtimestamp(int(raw["start_time"]), tz=timezone.utc)
    else:
        start_time = _header_datetime(headers.get("UTCDate"), headers.get("UTCTime"))

    eco_url = raw.get("eco") or headers.get("ECOUrl")
    opening, family = opening_from_eco_url(eco_url)
    accuracies = raw.get("accuracies")
    accuracies = accuracies if isinstance(accuracies, dict) else {}
    other = "black" if color == "white" else "white"

    # Prefer the PGN FEN: for chess960 it carries the real castling files, while
    # initial_setup uses KQkq. Standard games have initial_setup "" or the start FEN.
    initial = (headers.get("FEN") if headers.get("SetUp") == "1" else None) or raw.get("initial_setup") or None
    if initial and _same_position(initial, STANDARD_START_FEN):
        initial = None

    url = str(raw.get("url") or headers.get("Link") or "")
    return Game(
        game_id=str(raw.get("uuid") or url or _fallback_id(raw, white, black, pgn)),
        url=url,
        username=str(me.get("username", username)),
        color=color,
        opponent=str(opp.get("username", "?")),
        outcome=outcome_for(my_code, opp_code, headers.get("Result"), color),
        my_result_code=my_code,
        opp_result_code=opp_code,
        termination=normalize_termination(my_code, opp_code),
        time_class=str(time_class),
        time_control=str(tc or ""),
        base_seconds=base,
        increment=inc,
        rules=str(raw.get("rules") or "chess"),
        rated=bool(raw.get("rated", True)),
        end_time=end_time,
        start_time=start_time,
        my_rating=_to_int(me.get("rating")),
        opp_rating=_to_int(opp.get("rating")),
        eco=headers.get("ECO"),
        opening=opening,
        opening_family=family,
        my_accuracy=_to_float(accuracies.get(color)),
        opp_accuracy=_to_float(accuracies.get(other)),
        initial_fen=initial,
        moves_san=moves,
        clocks=clocks,
        pgn=pgn,
    )


def _game_order(g: Game) -> tuple[datetime, str]:
    return g.end_time, g.game_id


def merge_games(games: Iterable[Game]) -> list[Game]:
    """Drop repeats (same uuid, or the same game URL in any era's form), sort by (end time, id) and
    reconstruct pre-game ratings. Use it to combine games parsed from several sources."""
    seen: set[str] = set()
    out: list[Game] = []
    for g in games:
        keys = {g.game_id, game_url_key(g.url)} - {None, ""}
        if keys & seen:
            continue
        seen |= keys
        out.append(g)
    out.sort(key=_game_order)
    fill_pregame_ratings(out)
    return out


def parse_games(raws: Iterable[dict[str, Any]], username: str) -> list[Game]:
    """Parse, drop games the player wasn't in (or that are unfinished), de-duplicate, sort by end time."""
    parsed: list[Game] = []
    bad = 0
    for raw in raws:
        try:
            g = parse_game(raw, username)
        except Exception:  # noqa: BLE001 — one malformed record must not sink a 5,000-game history
            bad += 1
            continue
        if g is not None:
            parsed.append(g)
    if bad:
        log.warning("skipped %d malformed game record(s)", bad)
    return merge_games(parsed)


# --------------------------------------------------------------------------- plain PGN import
_TERMINATION_TEXT = [
    (re.compile(r"won by checkmate", re.I), "checkmated"),
    (re.compile(r"won by resignation", re.I), "resigned"),
    (re.compile(r"won on time", re.I), "timeout"),
    (re.compile(r"game abandoned", re.I), "abandoned"),
    (re.compile(r"drawn by agreement", re.I), "agreed"),
    (re.compile(r"drawn by repetition", re.I), "repetition"),
    (re.compile(r"drawn by stalemate", re.I), "stalemate"),
    (re.compile(r"timeout vs insufficient", re.I), "timevsinsufficient"),
    (re.compile(r"drawn by insufficient material", re.I), "insufficient"),
    (re.compile(r"50-move rule", re.I), "50move"),
]


_VARIANT_RULES = {
    "chess960": "chess960",
    "fischerandom": "chess960",
    "crazyhouse": "crazyhouse",
    "bughouse": "bughouse",
    "kingofthehill": "kingofthehill",
    "threecheck": "threecheck",
    "3check": "threecheck",
    "oddschess": "oddschess",
}


def rules_from_headers(headers: dict[str, str]) -> str:
    """chess.com ``rules`` for a PGN without the JSON wrapper: from ``Variant``, else the ``Event`` suffix.

    chess.com writes e.g. ``[Variant "Crazyhouse"]`` and ``[Event "Live Chess - Odds Chess"]`` (odds
    games have no Variant header). Unknown variants keep their own name so they are never taken
    for standard chess.
    """
    variant = re.sub(r"[^a-z0-9]", "", headers.get("Variant", "").lower())
    if variant and variant not in ("standard", "chess", "normal", "fromposition"):
        return _VARIANT_RULES.get(variant, variant)
    event = headers.get("Event", "")
    if " - " in event:
        suffix = re.sub(r"[^a-z0-9]", "", event.rsplit(" - ", 1)[1].lower())
        if suffix in _VARIANT_RULES:
            return _VARIANT_RULES[suffix]
    return "chess"


def split_pgn(text: str) -> list[str]:
    """Split a multi-game PGN file into single-game strings."""
    chunks = re.split(r"(?m)^(?=\[Event\s)", (text or "").lstrip("\ufeff"))
    return [c.strip() + "\n" for c in chunks if c.strip()]


def game_from_pgn(pgn: str, username: str) -> Optional[Game]:
    """Build a Game from PGN alone (e.g. a manual chess.com download) by faking the JSON wrapper."""
    h = parse_pgn_headers(pgn)
    result = h.get("Result", "*")
    term_text = h.get("Termination", "")
    loser_code = next((code for rx, code in _TERMINATION_TEXT if rx.search(term_text)), None)
    if result == "1/2-1/2":
        w_code = b_code = loser_code if loser_code in DRAW_CODES else "agreed"
    elif result in ("1-0", "0-1"):
        loser = loser_code if loser_code in LOSS_CODES else "resigned"
        w_code, b_code = ("win", loser) if result == "1-0" else (loser, "win")
    else:
        return None
    raw = {
        "url": h.get("Link", ""),
        "pgn": pgn,
        "time_control": h.get("TimeControl"),
        "rules": rules_from_headers(h),
        "rated": True,  # not recorded in PGN exports
        "white": {"username": h.get("White", ""), "rating": h.get("WhiteElo"), "result": w_code},
        "black": {"username": h.get("Black", ""), "rating": h.get("BlackElo"), "result": b_code},
    }
    if not raw["url"]:
        raw["uuid"] = f"pgn:{h.get('UTCDate', h.get('Date', ''))}:{h.get('UTCTime', '')}:{h.get('White')}:{h.get('Black')}"
    return parse_game(raw, username)


def games_from_pgn(text: str, username: str) -> list[Game]:
    return parse_games_from_pgns(split_pgn(text), username)


def parse_games_from_pgns(pgns: Iterable[str], username: str) -> list[Game]:
    parsed: list[Game] = []
    for pgn in pgns:
        try:
            g = game_from_pgn(pgn, username)
        except Exception:  # noqa: BLE001 — skip a broken game, keep the file
            log.warning("skipped a PGN game that could not be read")
            continue
        if g is not None:
            parsed.append(g)
    return merge_games(parsed)
