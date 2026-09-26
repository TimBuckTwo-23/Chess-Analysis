"""The coaching packet: every fact the LLM coach may use, as plain JSON (C4). Owner: llm.

The packet is built from the finished report (a ``Report`` or its JSON export, so ``chess-insights ask`` can
build it from the last report on disk) and holds only what the report already established:

* ``claims``: every strength and weakness in the report (id, title, detail, evidence, formats), and the claims
  of each per-format view under ``format_claims``;
* ``study_plan``: each item with its target and baseline;
* ``positions``: up to ``MAX_POSITIONS`` explained positions (FEN, the move played, the engine's best line and
  its refutation as numbered moves, evaluations in pawns or mates from your side, motifs, concept differences,
  opening and tablebase facts with their sources, the format of the game).

``verify`` checks the LLM's text against this same packet, so a fact that is not in it cannot reach the report.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import chess

from ..models import Report
from ..visuals import FORMAT_ORDER

PACKET_VERSION = 1
MAX_POSITIONS = 20
MAX_LIST = 12  # longer evidence lists are cut to this many entries
MAX_DEPTH = 5  # deeper evidence is left out
MAX_TEXT = 1500  # characters of a quoted text (Wikibooks) kept
TRAINING_URL = "https://lichess.org/training/{theme}"


def as_dict(report: Union[Report, dict[str, Any]]) -> dict[str, Any]:
    """The report as its JSON export (a dict passes through unchanged)."""
    if isinstance(report, dict):
        return report
    from ..report.json_export import to_dict

    return to_dict(report)


def _board(fen: str) -> Optional[chess.Board]:
    try:
        return chess.Board(fen)
    except ValueError:
        try:
            return chess.Board(fen, chess960=True)
        except ValueError:
            return None


def _label(board: chess.Board, move: chess.Move) -> str:
    """"6.Ndb5" / "6...a6" for a legal ``move`` in ``board``."""
    san = board.san(move)
    n = board.fullmove_number
    return f"{n}.{san}" if board.turn == chess.WHITE else f"{n}...{san}"


def _parse(board: chess.Board, move: str) -> Optional[chess.Move]:
    """A UCI or (numbered) SAN move that is legal in ``board``; None otherwise."""
    text = str(move or "").strip()
    if not text:
        return None
    try:
        m = chess.Move.from_uci(text)
        if m in board.legal_moves:
            return m
    except ValueError:
        pass
    san = text.replace("…", "...").split(".")[-1].strip()
    try:
        return board.parse_san(san)
    except ValueError:
        return None


def numbered_moves(fen: str, moves_uci: Optional[list[str]] = None, moves_san: Optional[list[str]] = None) -> list[str]:
    """The moves of a line from ``fen`` as numbered SAN ("5...e5", "6.Ndb5"), replayed with python-chess.

    UCI is used when present (the engine's own record), SAN otherwise; the line stops at the first move that is
    not legal, so everything returned can be replayed.
    """
    board = _board(fen)
    if board is None:
        return []
    out: list[str] = []
    for move in list(moves_uci or []) or list(moves_san or []):
        m = _parse(board, move)
        if m is None:
            break
        out.append(_label(board, m))
        board.push(m)
    return out


def line_text(moves: list[str]) -> str:
    """Numbered moves as a line in the usual notation: ["5...e5", "6.Ndb5", "6...a6"] -> "5...e5 6.Ndb5 a6"."""
    out: list[str] = []
    for i, move in enumerate(moves):
        number, dots, san = move.partition("...")
        if dots and i > 0 and moves[i - 1].startswith(f"{number}.") and "..." not in moves[i - 1]:
            out.append(san)
        else:
            out.append(move)
    return " ".join(out)


def _num(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    """``d`` without empty values (None, "", []), so the packet stays short."""
    return {k: v for k, v in d.items() if v is not None and v != "" and v != []}


def _pawns(cp: Any) -> Optional[float]:
    value = _num(cp)
    return None if value is None else round(value / 100.0, 2)


def _line(line: Optional[dict[str, Any]], fen: str) -> Optional[dict[str, Any]]:
    """A Line (JSON) as {"moves", "eval_pawns" | "mate_in"[, "fen", "depth"]}, from the side to move at the line's
    start (you, for the lines of an explained position): eval_pawns below zero and mate_in below zero are bad for
    you."""
    if not isinstance(line, dict):
        return None
    start = str(line.get("fen") or fen)
    same = " ".join(start.split()[:4]) == " ".join(fen.split()[:4])
    if same:  # the same position, maybe found from another game (other move counters): number it as this one
        start = fen
    moves = numbered_moves(start, line.get("moves_uci"), line.get("moves_san"))
    if not moves:
        return None
    mate = line.get("mate_end")
    # a mate is counted as MATE_CP centipawns in the engine's record: give the mate, not a made-up pawn count
    out: dict[str, Any] = {"moves": moves, "eval_pawns": None if mate is not None else _pawns(line.get("cp_end")),
                           "mate_in": mate}
    if not same:
        out["fen"] = start
    if line.get("depth"):
        out["depth"] = line["depth"]
    return _compact(out)


def _move_stats(stats: Any, fen: str) -> list[dict[str, Any]]:
    """Opening-database moves as {"move": "4.Nf3", "games", "share", "score"} (moves that don't parse are left out)."""
    board = _board(fen)
    out = []
    for s in stats or []:
        if not isinstance(s, dict) or board is None:
            continue
        m = _parse(board, s.get("uci") or "") or _parse(board, s.get("san") or "")
        if m is None:
            continue
        out.append(
            _compact(
                {
                    "move": _label(board, m),
                    "games": s.get("games"),
                    "share": s.get("share"),
                    "score": s.get("score"),
                    "avg_rating": s.get("avg_rating"),
                }
            )
        )
    return out


def _cut(text: str, limit: int = MAX_TEXT) -> str:
    """``text`` cut to ``limit`` characters, at the end of a sentence when there is one."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    end = max(head.rfind(". "), head.rfind(".\n"))
    return head[: end + 1] if end > limit // 2 else head.rsplit(" ", 1)[0] + " ..."


def _sources(sources: Any) -> list[dict[str, Any]]:
    return [
        {k: s.get(k) for k in ("name", "url", "retrieved", "license") if s.get(k)}
        for s in sources or []
        if isinstance(s, dict)
    ]


def _opening(facts: Any, fen: str) -> Optional[dict[str, Any]]:
    if not isinstance(facts, dict):
        return None
    out: dict[str, Any] = {
        "eco": facts.get("eco") or "",
        "name": facts.get("name") or "",
        "masters": _move_stats(facts.get("masters"), fen),
        "peers": _move_stats(facts.get("peers"), fen),
        "peer_groups": list(facts.get("peer_groups") or []),
        "your_move_rank_masters": facts.get("played_rank_masters"),
        "your_move_rank_peers": facts.get("played_rank_peers"),
        "master_game": facts.get("master_game") or "",
        "cloud_lines": [x for x in (_line(line, fen) for line in facts.get("cloud_lines") or []) if x],
        "wiki_text": _cut(str(facts.get("wiki_text") or "")),
        "wiki_url": facts.get("wiki_url") or "",
        "sources": _sources(facts.get("sources")),
    }
    return _compact(out) or None


def _trim(value: Any, depth: int = 0) -> Any:
    """Evidence cut to a size an LLM can read: long lists shortened, very deep structures left out."""
    if depth > MAX_DEPTH:
        return None
    if isinstance(value, dict):
        return {str(k): _trim(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_trim(v, depth + 1) for v in value[:MAX_LIST]]
    return value


def _claim(ins: dict[str, Any], view: str = "") -> dict[str, Any]:
    out = {
        "id": ins.get("id"),
        "kind": ins.get("kind"),
        "category": ins.get("category"),
        "title": ins.get("title"),
        "detail": ins.get("detail"),
        "evidence": _trim(ins.get("evidence") or {}),
        "formats": dict(ins.get("formats") or {}),
    }
    if view:
        out["view"] = view
    return out


def claims_of(report: dict[str, Any], view: str = "") -> list[dict[str, Any]]:
    """The report's strengths and weaknesses (the claims), strengths first."""
    return [
        _claim(ins, view)
        for ins in list(report.get("strengths") or []) + list(report.get("weaknesses") or [])
        if isinstance(ins, dict) and ins.get("kind") in ("strength", "weakness")
    ]


def user_ratings(report: Union[Report, dict[str, Any]]) -> dict[str, int]:
    """Your current chess.com rating per time class, from the results section ({"blitz": 949, ...})."""
    data = as_dict(report)
    results = next((m for m in data.get("modules") or [] if isinstance(m, dict) and m.get("key") == "results"), None)
    pools = ((results or {}).get("stats") or {}).get("by_time_control") or {}
    out: dict[str, int] = {}
    for pool, stats in pools.items():
        name = str(pool)
        if "(" in name:  # a variant pool ("Blitz (chess960)")
            continue
        current = (((stats or {}).get("rating") or {}) if isinstance(stats, dict) else {}).get("current")
        if _num(current) is not None:
            out[name.lower()] = int(current)
    order = {tc: i for i, tc in enumerate(FORMAT_ORDER)}
    return dict(sorted(out.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])))


def _position(exp: dict[str, Any], claim_ids: set[str]) -> Optional[dict[str, Any]]:
    fen = str(exp.get("fen") or "")
    board = _board(fen)
    if board is None:
        return None
    best_line = _line(exp.get("best_line"), fen)
    refutation = _line(exp.get("refutation"), fen)
    best_eval = (best_line or {}).get("eval_pawns")
    played_eval = (refutation or {}).get("eval_pawns")
    insight_id = exp.get("insight_id")
    motifs = [
        _compact(
            {
                "theme": m.get("theme"),
                "line": m.get("line"),
                "side": m.get("side"),
                "squares": list(m.get("squares") or []),
                "drill": TRAINING_URL.format(theme=m.get("theme")),
            }
        )
        for m in exp.get("motifs") or []
        if isinstance(m, dict) and m.get("theme")
    ]
    concepts = [
        _compact({"term": c.get("term"), "label": c.get("label") or c.get("term"), "pawns": c.get("value"),
                  "source": c.get("source")})
        for c in exp.get("concepts") or []
        if isinstance(c, dict) and _num(c.get("value")) is not None
    ]
    tablebase = dict(exp.get("tablebase") or {})
    if tablebase.get("best"):  # a move from this position: numbered like every other move in the packet
        m = _parse(board, str(tablebase["best"]))
        if m is not None:
            tablebase["best"] = _label(board, m)
    out: dict[str, Any] = {
        "epd": exp.get("epd") or " ".join(fen.split()[:4]),
        "fen": fen,
        "time_class": exp.get("time_class") or "",
        "you_play": exp.get("color") or "",
        "kind": exp.get("kind") or "error",
        "played": exp.get("played") or "",
        "best": exp.get("best"),
        "best_line": best_line,
        "refutation": refutation,
        "pawns_lost": round(best_eval - played_eval, 2) if best_eval is not None and played_eval is not None else None,
        "win_chance_lost": _num(exp.get("drop")),
        "repeats": exp.get("repeats"),
        "games": list(exp.get("games") or [])[:5],
        "game_url": exp.get("game_url") or "",
        "motifs": motifs,
        "concepts": concepts,
        "facts": list(exp.get("facts") or []),
        "opening": _opening(exp.get("opening"), fen),
        "tablebase": tablebase or None,
        "maia": dict(exp.get("maia") or {}) or None,
        "drills": [TRAINING_URL.format(theme=t) for t in exp.get("drill_themes") or []],
        "insight_id": insight_id,
        "is_claim": bool(insight_id and insight_id in claim_ids),
        "sources": _sources(exp.get("sources")),
        "template_text": exp.get("text") or "",
    }
    return _compact(out)


def build_packet(report: Union[Report, dict[str, Any]], max_positions: int = MAX_POSITIONS) -> dict[str, Any]:
    """Every fact the LLM coach may use, as JSON-able dicts (see the module docstring for the layout)."""
    data = as_dict(report)
    claims = claims_of(data)
    format_claims = {
        str(tc): claims_of(view, str(tc))
        for tc, view in (data.get("format_reports") or {}).items()
        if isinstance(view, dict)
    }
    claim_ids = {c["id"] for c in claims} | {c["id"] for cs in format_claims.values() for c in cs}
    plan = [
        {
            "id": f"plan-{i}",
            "title": item.get("title"),
            "why": item.get("why"),
            "actions": list(item.get("actions") or []),
            "target": item.get("target") or "",
            "baseline": dict(item.get("baseline") or {}),
            "insight_ids": list(item.get("insight_ids") or []),
            "category": item.get("category") or "",
        }
        for i, item in enumerate(data.get("study_plan") or [], start=1)
        if isinstance(item, dict)
    ]
    coaching = data.get("coaching") or {}
    positions: list[dict[str, Any]] = []
    for exp in coaching.get("explanations") or []:  # in the coaching order (costliest, or Maia's ranking)
        if len(positions) >= max_positions:
            break
        pos = _position(exp, claim_ids) if isinstance(exp, dict) else None
        if pos is not None and all(p["epd"] != pos["epd"] for p in positions):  # the LLM names positions by EPD
            positions.append(pos)
    generated = str(data.get("generated_at") or "")
    return {
        "packet_version": PACKET_VERSION,
        "player": {
            "username": data.get("username") or "",
            "ratings": user_ratings(data),
            "games_per_format": dict(data.get("formats") or {}),
            "engine_games_per_format": dict(data.get("engine_formats") or {}),
        },
        "report": {
            "date": generated[:10],
            "filters": data.get("filters") or "",
            "games": data.get("n_games"),
            "view": data.get("time_class") or "all formats",
            "engine_note": data.get("engine_note") or "",
            "summary": list(data.get("summary_lines") or []),
        },
        "claims": claims,
        "format_claims": format_claims,
        "study_plan": plan,
        "drills": [
            {k: d.get(k) for k in ("theme", "title", "link", "file", "reason") if d.get(k)}
            for d in coaching.get("drills") or []
            if isinstance(d, dict)
        ],
        "progress": [
            {k: p.get(k) for k in ("title", "metric", "before", "now", "text", "improved")}
            for p in coaching.get("progress") or []
            if isinstance(p, dict)
        ],
        "positions": positions,
    }


def claim_index(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every claim in the packet (the report's and each format view's) by id; the report's own wins a tie."""
    out: dict[str, dict[str, Any]] = {}
    for claims in (packet.get("format_claims") or {}).values():
        for c in claims or []:
            if c.get("id"):
                out[str(c["id"])] = c
    for c in packet.get("claims") or []:
        if c.get("id"):
            out[str(c["id"])] = c
    return out


def position_index(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Packet positions by EPD (the first one wins when a position appears twice)."""
    out: dict[str, dict[str, Any]] = {}
    for p in packet.get("positions") or []:
        out.setdefault(str(p.get("epd")), p)
    return out
