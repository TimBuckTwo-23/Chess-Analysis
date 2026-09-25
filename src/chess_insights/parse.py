"""chess.com game JSON (or plain PGN) -> :class:`~chess_insights.models.Game`.

The PGN movetext is tokenised with regular expressions rather than replayed on
a board: a few thousand games parse in a second or two. Move legality is only
checked later, by the engine module, which has to replay positions anyway.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib.parse import unquote, urlparse

from .models import DRAW_CODES, LOSS_CODES, WIN_CODES, Game

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


# --------------------------------------------------------------------------- small parsers
def parse_pgn_headers(pgn: str) -> dict[str, str]:
    return {k: v.replace('\\"', '"') for k, v in _HEADER_RE.findall(pgn or "")}


def parse_clock(text: str) -> Optional[float]:
    """'0:02:59.9' / '[%clk 1:05:03]' -> seconds."""
    m = _CLK_RE.search(text if text.startswith("[") else f"[%clk {text}]")
    if not m:
        return None
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def _movetext(pgn: str) -> str:
    """Strip the header block."""
    lines = (pgn or "").splitlines()
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
    if not words:
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
    return opening, family


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


def _header_datetime(date: Optional[str], time_: Optional[str]) -> Optional[datetime]:
    if not date or "?" in date:
        return None
    try:
        dt = datetime.strptime(f"{date} {time_ or '00:00:00'}", "%Y.%m.%d %H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- chess.com JSON
def parse_game(raw: dict[str, Any], username: str) -> Optional[Game]:
    """One raw chess.com game object -> Game, or None if ``username`` didn't play in it."""
    white, black = raw.get("white") or {}, raw.get("black") or {}
    user = username.lower()
    if str(white.get("username", "")).lower() == user:
        color, me, opp = "white", white, black
    elif str(black.get("username", "")).lower() == user:
        color, me, opp = "black", black, white
    else:
        return None

    pgn = raw.get("pgn") or ""
    headers = parse_pgn_headers(pgn)
    moves, clocks = parse_movetext(pgn)
    my_code, opp_code = str(me.get("result", "")), str(opp.get("result", ""))

    tc = raw.get("time_control") or headers.get("TimeControl")
    base, inc = parse_time_control(tc)
    time_class = raw.get("time_class") or time_class_for(tc)
    if time_class == "daily":
        inc = 0

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
    accuracies = raw.get("accuracies") or {}
    other = "black" if color == "white" else "white"

    initial = raw.get("initial_setup") or (headers.get("FEN") if headers.get("SetUp") == "1" else None)
    if initial and initial.strip() == STANDARD_START_FEN:
        initial = None

    url = raw.get("url") or headers.get("Link") or ""
    return Game(
        game_id=str(raw.get("uuid") or url),
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


def parse_games(raws: Iterable[dict[str, Any]], username: str) -> list[Game]:
    """Parse, drop games the player wasn't in, de-duplicate, sort by end time."""
    seen: set[str] = set()
    out: list[Game] = []
    for raw in raws:
        try:
            g = parse_game(raw, username)
        except (KeyError, TypeError, ValueError):
            continue  # one malformed record must not sink a 5,000-game history
        if g is None or g.game_id in seen:
            continue
        seen.add(g.game_id)
        out.append(g)
    out.sort(key=lambda g: g.end_time)
    return out


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


def split_pgn(text: str) -> list[str]:
    """Split a multi-game PGN file into single-game strings."""
    chunks = re.split(r"(?m)^(?=\[Event\s)", text or "")
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
        "rules": "chess960" if "960" in h.get("Variant", "") else "chess",
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
    seen: set[str] = set()
    out = []
    for pgn in pgns:
        g = game_from_pgn(pgn, username)
        if g is None or g.game_id in seen:
            continue
        seen.add(g.game_id)
        out.append(g)
    out.sort(key=lambda g: g.end_time)
    return out
