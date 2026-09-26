"""Fallbacks that give every finding what the report shows next to it: its formats and a picture.

The analysis modules set ``Insight.formats``, ``Insight.chart`` and ``Insight.diagram`` where they know best;
these fallbacks only fill what a module left empty, from facts it already recorded:

* ``fill_formats``: the time classes of the games behind a finding: the format(s) its evidence names
  ("time_class", "time_classes", a rating pool such as "Blitz"); for one position, the games in which it came up;
  for one colour (and opening), those games; else the games the module analysed (the engine-analysed games for
  the engine sections).
* ``fill_visuals``: a small chart built from the finding's evidence (your score against the rating's
  expectation, your rate against your opponents' in the same games ...), or the board for a finding about one
  position. Evidence that fits none of the patterns below leaves the finding without a picture rather than
  with one that says something the numbers don't.

Neither changes which findings exist, their kind, severity, confidence or wording. ``pipeline.build_report``
calls both for every report, the per-format views included.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence

import chess

from .models import TIME_CLASSES, Chart, Diagram, Game, Insight, ModuleResult
from .stats import MINUS, per100_games
from .visuals import (
    FORMAT_NAMES,
    comparison_chart,
    format_counts,
    move_label,
    ordered_formats,
    parse_move,
    position_diagram,
)

if TYPE_CHECKING:
    from .context import AnalysisContext

log = logging.getLogger(__name__)

# Sections whose findings come from the engine-analysed games only.
ENGINE_MODULES = frozenset({"engine_stats", "mistakes"})

YOU, OPPONENTS = "You", "Your opponents"


# --------------------------------------------------------------------------- formats
def module_games(module_key: str, ctx: "AnalysisContext") -> list[Game]:
    """The games a module's findings rest on: the engine-analysed ones for the engine sections, else all."""
    if module_key in ENGINE_MODULES:
        return [g for g in ctx.games if g.game_id in ctx.evals]
    return list(ctx.games)


def _time_class(value: Any) -> Optional[str]:
    """"blitz", "Blitz", "Blitz (chess960)" -> "blitz"; anything else -> None."""
    if not isinstance(value, str):
        return None
    words = value.strip().lower().replace("(", " ").split()
    return words[0] if words and words[0] in TIME_CLASSES else None


def named_formats(evidence: dict[str, Any]) -> list[str]:
    """The time classes a finding's evidence is about ("time_class", "time_classes", a rating "pool"), in order."""
    names: list[str] = []
    candidates: list[Any] = [evidence.get("time_class")]
    if isinstance(evidence.get("time_classes"), (list, tuple)):
        candidates += list(evidence["time_classes"])
    pool = evidence.get("pool")
    candidates.append(pool.get("pool") if isinstance(pool, dict) else pool)
    for value in candidates:
        tc = _time_class(value)
        if tc and tc not in names:
            names.append(tc)
    return names


def _whole(value: Any) -> Optional[int]:
    """A whole number (numpy integers included) as an int; None for anything else, booleans too."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    return int(value)


def _count(value: Any) -> Optional[int]:
    n = _whole(value)
    return n if n is not None and n > 0 else None


def _games_behind(evidence: dict[str, Any], tc: str, single: bool) -> Optional[int]:
    """The number of games the evidence itself gives for format ``tc``, if any."""
    nested = evidence.get(tc)
    if isinstance(nested, dict) and _count(nested.get("n")):
        return _count(nested.get("n"))
    pool = evidence.get("pool")
    if isinstance(pool, dict) and _time_class(pool.get("pool")) == tc:
        return _count(pool.get("n"))
    return _count(evidence.get("n")) if single else None


def _epd(fen: Any) -> Optional[str]:
    """The position part of a FEN (what ``chess.Board.epd()`` gives for the same board); None for anything else."""
    if not isinstance(fen, str):
        return None
    fields = fen.split()
    return " ".join(fields[:4]) if len(fields) >= 4 else None


def games_reaching(epds: Iterable[str], games: Iterable[Game]) -> dict[str, list[Game]]:
    """For each position (EPD), the games in which it came up with you to move; a game counts once.

    The same scan as the repeated-mistakes module's "reached" count (transpositions included), so a finding about
    one position can say which formats its games were. A game stops being searched once fewer pieces are left than
    in every position looked for (captures are irreversible); a game that doesn't replay keeps what was counted.
    """
    wanted = set(epds)
    reached: dict[str, list[Game]] = {}
    if not wanted:
        return reached
    fewest = min(sum(ch.isalpha() for ch in epd.split(" ", 1)[0]) for epd in wanted)
    for game in games:
        seen: set[str] = set()
        try:
            board = chess.Board(game.initial_fen or chess.STARTING_FEN, chess960=game.rules == "chess960")
            for ply, san in enumerate(game.moves_san):
                if chess.popcount(board.occupied) < fewest:
                    break
                if game.is_my_ply(ply):
                    epd = board.epd()
                    if epd in wanted and epd not in seen:
                        seen.add(epd)
                        reached.setdefault(epd, []).append(game)
                board.push_san(san)
        except ValueError:
            continue
    return reached


def _counts_agree(evidence: dict[str, Any], games: Sequence[Game], keys: Sequence[str]) -> bool:
    """Whether ``games`` are the games the evidence counts. The first of ``keys`` present decides: a game count
    must equal ``len(games)``; "wins" and "losses" (checked together) must equal those outcomes in ``games``.
    With none of them present there is nothing to contradict."""
    for key in keys:
        n = _whole(evidence.get(key))
        if n is None:
            continue
        if key in ("wins", "losses"):
            return all(
                sum(g.outcome == outcome for g in games) == _whole(evidence.get(k))
                for k, outcome in (("wins", "win"), ("losses", "loss"))
                if _whole(evidence.get(k)) is not None
            )
        return len(games) == n
    return True


def _opening_games(evidence: dict[str, Any], games: Sequence[Game]) -> Optional[list[Game]]:
    """The games of the colour (and opening family) a finding names ("color" or "colour", "family"), when the
    evidence's own counts confirm they are the games behind it; None otherwise."""
    family, colour = evidence.get("family"), evidence.get("color", evidence.get("colour"))
    if colour not in ("white", "black") or (family is not None and (not isinstance(family, str) or not family)):
        return None
    chosen = [g for g in games if g.color == colour and (family is None or g.opening_family == family)]
    keys = ("n", "games", "wins", "losses") if family is not None else ("n", "games")
    if not chosen or not _counts_agree(evidence, chosen, keys):
        return None
    if family is None and not any(_whole(evidence.get(k)) is not None for k in keys):
        return None  # a colour alone, with no count to confirm it, says too little about which games
    return chosen


def _formats_of(
    ins: Insight, games: Sequence[Game], counts: dict[str, int], reached: dict[str, list[Game]]
) -> dict[str, int]:
    """One finding's formats: see ``fill_formats``."""
    evidence = ins.evidence if isinstance(ins.evidence, dict) else {}
    named = named_formats(evidence)
    formats = {}
    for tc in named:
        n = _games_behind(evidence, tc, single=len(named) == 1) or counts.get(tc)
        if n:
            formats[tc] = n
    if formats:
        return ordered_formats(formats)
    epd = _epd(evidence.get("fen"))
    if epd is not None and reached.get(epd) and _counts_agree(evidence, reached[epd], ("reached",)):
        return format_counts(reached[epd])
    opening = _opening_games(evidence, games)
    if opening:
        return format_counts(opening)
    return dict(counts)


def fill_formats(modules: Iterable[ModuleResult], ctx: "AnalysisContext") -> None:
    """Give every finding without ``formats`` the format mix behind it (see the module docstring).

    Most precise first: the format(s) its evidence names; for a finding about one position (evidence "fen"), the
    games in which the position came up with you to move; for one colour (and opening), those games (each only
    when the evidence's own game counts confirm them); else the games the module analysed.
    """
    modules = list(modules)
    positions: set[str] = set()
    for ins in (i for m in modules for i in m.insights or [] if not i.formats):
        try:
            epd = _epd(ins.evidence.get("fen")) if isinstance(ins.evidence, dict) else None
        except Exception:  # noqa: BLE001 — an odd evidence object: that finding falls back below
            epd = None
        if epd:
            positions.add(epd)
    reached = games_reaching(positions, ctx.games) if positions else {}
    for m in modules:
        if not m.insights:
            continue
        games = module_games(m.key, ctx)
        counts = format_counts(games)
        for ins in m.insights:
            if ins.formats:
                continue
            try:
                ins.formats = _formats_of(ins, games, counts, reached)
            except Exception as exc:  # noqa: BLE001 — one odd finding must not leave the others without formats
                log.warning("no formats for %s: %s", ins.id, exc)


# --------------------------------------------------------------------------- visuals
def _num(value: Any) -> Optional[float]:
    """A finite number (numpy scalars included); None for anything else, text and booleans too."""
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _share(value: Any) -> Optional[float]:
    x = _num(value)
    return x if x is not None and 0.0 <= x <= 1.0 else None


def _ratio(part: Any, whole: Any) -> Optional[float]:
    p, w = _num(part), _num(whole)
    return p / w if p is not None and w and w > 0 and 0 <= p <= w else None


def _games_note(n: Any, extra: str = "") -> str:
    count = _count(n)
    parts = [f"{count} games" if count else "", extra]
    return "; ".join(p for p in parts if p)


def _expected_label(evidence: dict[str, Any]) -> str:
    if evidence.get("expected_kind") == "attenuated":
        return "Expected (allowing for rating noise)"
    return "Expected from ratings"


def _score_of(d: dict[str, Any], prefix: str = "") -> Optional[float]:
    return _share(d.get(f"{prefix}rated_score")) if f"{prefix}rated_score" in d else _share(d.get(f"{prefix}score"))


# evidence key prefix -> the games it is about (for the chart title)
_SCORE_PREFIXES = {"": "", "after_two_losses_": "After two losses in a row: "}


def _score_chart(ins: Insight) -> Optional[Chart]:
    """score vs expected (also ``after_two_losses_score`` ...): "You" vs "Expected from ratings"."""
    ev = ins.evidence
    for prefix, games in _SCORE_PREFIXES.items():
        score, expected = _score_of(ev, prefix), _share(ev.get(f"{prefix}expected"))
        if score is None or expected is None:
            continue
        n = ev.get(f"{prefix}n_rated", ev.get(f"{prefix}n"))
        rest = _num(ev.get("rest_delta"))
        extra = f"your other games: {per100_games(rest)} vs your rating" if rest is not None and not prefix else ""
        return comparison_chart(
            f"{games}your score vs what your ratings predict" if games else "Your score vs what your ratings predict",
            [YOU, _expected_label(ev)],
            [("Score", [score, expected])],
            value_format="pct",
            note=_games_note(n, extra),
        )
    return None


_GROUP_LABELS = {"white": "As White", "black": "As Black", "rest": "Your other games"}


def _grouped_score_chart(ins: Insight) -> Optional[Chart]:
    """Nested score summaries ({"white": {...}, "black": {...}} or {"pool": {...}, "rest": {...}})."""
    labels, mine, expected, counts = [], [], [], []
    for key, d in ins.evidence.items():
        if not isinstance(d, dict):
            continue
        score, exp = _score_of(d), _share(d.get("expected"))
        if score is None or exp is None:
            continue
        if key == "pool":
            label = str(d.get("pool") or "These games")
        elif key == "rest" and isinstance(ins.evidence.get("pool"), dict):
            label = "Your other time controls"
        else:
            label = _GROUP_LABELS.get(key, str(key).replace("_", " ").capitalize())
        labels.append(label)
        mine.append(score)
        expected.append(exp)
        if (n := _count(d.get("n_rated", d.get("n")))) is not None:
            counts.append(f"{label}: {n} games")
    if not labels:
        return None
    return comparison_chart(
        "Your score vs what your ratings predict",
        labels,
        [(YOU, mine), (_expected_label(ins.evidence), expected)],
        value_format="pct",
        note="; ".join(counts),
    )


# (your key, your opponents' key, value format): the same measure for both players in the same games.
_PAIRS = {
    "per100": ("per100", "opp_per100", "float1"),
    "rest_per100": ("rest_per100", "opp_rest_per100", "float1"),
    "other_errors": ("other_errors_per100", "opp_other_errors_per100", "float1"),
    "per100_low": ("per100_low", "opp_per100_low", "float1"),
    "per100_ok": ("per100_ok", "opp_per100_ok", "float1"),
    "rate": ("rate", "opp_rate", "pct"),
    "trouble": ("trouble_rate", "opponent_trouble_rate", "pct"),
    "opening_share": ("your_share", "opponent_share", "pct"),
    "missed_mates": ("missed_mates", "opp_missed_mates", "int"),
    "allowed_mates": ("allowed_mates", "opp_allowed_mates", "int"),
    "abandoned": ("abandoned", "opponent_abandoned", "int"),
}
# (chart title, [(pair row, bar label)], a key the evidence must also have): the first chart whose rows are all in
# the evidence is drawn. "{phase}" is the evidence's game phase, "{kind}" the kind of error (from the finding's id). The key keeps a chart to the finding it describes
# ("per100" is blunders in one finding, mistakes and blunders of one phase in another).
_PAIR_CHARTS: list[tuple[str, list[tuple[str, str]], Optional[str]]] = [
    ("Blunders per 100 moves, you vs your opponents",
     [("per100_low", "Short of time"), ("per100_ok", "With time left")], None),
    ("Mistakes and blunders per 100 moves, you vs your opponents",
     [("per100", "{phase}"), ("rest_per100", "Other phases")], "phase"),
    ("Errors per 100 moves, you vs your opponents",
     [("per100", "{kind}"), ("other_errors", "Other errors")], None),
    ("Blunders per 100 moves, you vs your opponents", [("per100", "Blunders per 100 moves")], "blunders"),
    ("Converting winning positions, you vs your opponents", [("rate", "Winning positions won")], None),
    ("Time trouble, you vs your opponents", [("trouble", "Games in time trouble")], None),
    ("Opening pace, you vs your opponents", [("opening_share", "Clock used on the first moves")], None),
    ("Mates, you vs your opponents", [("missed_mates", "Mates missed"), ("allowed_mates", "Mates allowed")], None),
    ("Abandoned games, you vs your opponents", [("abandoned", "Games abandoned")], None),
]


def _pair_values(ev: dict[str, Any], row: str) -> Optional[tuple[float, float, str]]:
    mine_key, theirs_key, fmt = _PAIRS[row]
    convert = _share if fmt == "pct" else _num
    mine, theirs = convert(ev.get(mine_key)), convert(ev.get(theirs_key))
    if mine is None or theirs is None:
        return None
    return mine, theirs, fmt


# The tactics findings' kinds of error (engine_stats.TACTIC_TEXT slugs), as the Tactics table names them.
_ERROR_KINDS = {"missed-tactics": "Missed tactical shots", "hung-material": "Material left hanging"}


def _row_label(label: str, ins: Insight) -> str:
    phase = str(ins.evidence.get("phase") or "This phase").capitalize()
    kind = next((text for slug, text in _ERROR_KINDS.items() if ins.id.endswith(slug)), "This kind of error")
    return label.format(phase=phase, kind=kind)


# Charts whose rows compare you with yourself (short of time vs with time left, one phase vs the others, one kind
# of error vs the others): without your opponents' numbers they still show what the finding measured.
_OWN_COMPARISONS = frozenset({"per100_low", "per100"})


def _pair_chart(ins: Insight) -> Optional[Chart]:
    """Your rate against your opponents' in the same games (grouped where the evidence has several).

    When your opponents' numbers are missing (they were never that short of time ...), a chart that compares you
    with yourself is drawn with your bars only; any other chart needs both players.
    """
    ev = ins.evidence
    for title, rows, needs in _PAIR_CHARTS:
        values = [_pair_values(ev, row) for row, _ in rows]
        if any(v is None for v in values) or (needs and needs not in ev):
            continue
        return comparison_chart(
            title,
            [_row_label(label, ins) for _, label in rows],
            [(YOU, [v[0] for v in values]), (OPPONENTS, [v[1] for v in values])],  # type: ignore[index]
            value_format=values[0][2],  # type: ignore[index]
            note=_games_note(ev.get("games", ev.get("n"))),
        )
    for title, rows, needs in _PAIR_CHARTS:
        if len(rows) < 2 or rows[0][0] not in _OWN_COMPARISONS or (needs and needs not in ev):
            continue
        mine_keys = [_PAIRS[row][0] for row, _ in rows]
        fmt = _PAIRS[rows[0][0]][2]
        mine = [(_share if fmt == "pct" else _num)(ev.get(key)) for key in mine_keys]
        if any(v is None for v in mine):
            continue
        return comparison_chart(
            title.replace(", you vs your opponents", ""),
            [_row_label(label, ins) for _, label in rows],
            [(YOU, mine)],
            value_format=fmt,
            note=_games_note(ev.get("games", ev.get("n")), "your opponents had too few such moves to compare"),
        )
    return None


def _per_format_pair_chart(ins: Insight) -> Optional[Chart]:
    """One finding over several formats ({"time_classes": [...], "blitz": {...}, "rapid": {...}}): a bar each."""
    ev = ins.evidence
    classes = [tc for tc in ev.get("time_classes") or [] if isinstance(ev.get(tc), dict)]
    if not classes:
        return None
    labels = [FORMAT_NAMES.get(tc, tc) for tc in classes]
    note = ", ".join(f"{FORMAT_NAMES.get(tc, tc)}: {n} games" for tc in classes if (n := _count(ev[tc].get("n"))))
    for _title, rows, needs in _PAIR_CHARTS:
        if len(rows) != 1:
            continue
        row, label = rows[0]
        values = [_pair_values(ev[tc], row) for tc in classes]
        if any(v is None for v in values) or (needs and any(needs not in ev[tc] for tc in classes)):
            continue
        return comparison_chart(
            f"{label} by format, you vs your opponents",
            labels,
            [(YOU, [v[0] for v in values]), (OPPONENTS, [v[1] for v in values])],  # type: ignore[index]
            value_format=values[0][2],  # type: ignore[index]
            note=note,
        )
    scores = [(_score_of(ev[tc]), _share(ev[tc].get("expected"))) for tc in classes]
    if all(s is not None and e is not None for s, e in scores):
        return comparison_chart(
            "Your score vs what your ratings predict, by format",
            labels,
            [(YOU, [s for s, _ in scores]), ("Expected from ratings", [e for _, e in scores])],
            value_format="pct",
            note=note,
        )
    return None


def _delta_chart(ins: Insight) -> Optional[Chart]:
    """Score vs your rating in these games and in your other games (evidence "delta" and "earlier_delta")."""
    ev = ins.evidence
    delta, other = _num(ev.get("delta")), _num(ev.get("earlier_delta"))
    if delta is None or other is None:
        return None
    first = _count(ev.get("from_game"))
    labels = [f"From game {first} of a session on", "Earlier games"] if first else ["These games", "Your other games"]
    return comparison_chart(
        "Score vs your rating",
        labels,
        [("Score vs your rating", [delta, other])],
        value_format="signed_pct",
        reference=0.0,
        note=_games_note(ev.get("n"), "points per 100 games above (+) or below (−) what your ratings predict"),
    )


def _share_chart(ins: Insight) -> Optional[Chart]:
    """Two shares the finding's test compares (quick losses vs quick wins, mated in losses vs mating in wins ...)."""
    ev = ins.evidence
    specs = [
        ("Games ending in checkmate", "Ended in checkmate", ("mated", "losses"), ("mating", "wins"),
         ("Your losses", "Your wins")),
        ("Games over quickly", "Over quickly", ("short_losses", "losses"), ("short_wins", "wins"),
         ("Your losses", "Your wins")),
    ]
    for title, series, (a, a_of), (b, b_of), labels in specs:
        first, second = _ratio(ev.get(a), ev.get(a_of)), _ratio(ev.get(b), ev.get(b_of))
        if first is not None and second is not None:
            note = f"{int(_num(ev[a_of]) or 0)} losses, {int(_num(ev[b_of]) or 0)} wins"
            return comparison_chart(title, list(labels), [(series, [first, second])], value_format="pct", note=note)
    moves, blunders = _ratio(ev.get("moves"), ev.get("total_moves")), _ratio(ev.get("blunders"), ev.get("total_blunders"))
    if ev.get("piece") and moves is not None and blunders is not None:
        return comparison_chart(
            f"{str(ev['piece']).capitalize()} moves",
            ["Share of your moves", "Share of your blunders"],
            [("Share", [moves, blunders])],
            value_format="pct",
            note=_games_note(None, f"{int(_num(ev['total_blunders']) or 0)} blunders in {int(_num(ev['total_moves']) or 0)} moves"),
        )
    return None


def _saved_chart(ins: Insight) -> Optional[Chart]:
    """Lost positions you saved (drew or won) against those you went on to lose (evidence "saved" of
    "lost_positions")."""
    ev = ins.evidence
    lost, saved = _count(ev.get("lost_positions")), _num(ev.get("saved"))
    if lost is None or saved is None or not 0 <= saved <= lost:
        return None
    return comparison_chart(
        "Games where your opponent got a winning position first",
        ["You saved (drew or won)", "You lost"],
        [("Games", [saved, lost - saved])],
        value_format="int",
        note=f"{lost} games",
    )


def _flag_chart(ins: Insight) -> Optional[Chart]:
    """Games lost on time against games won on time (the flag balance the clock findings test)."""
    ev = ins.evidence
    lost, won = _num(ev.get("lost_on_time")), _num(ev.get("won_on_time"))
    if lost is None or won is None:
        return None
    return comparison_chart(
        "Games decided on time",
        ["You lost on time", "You won on time"],
        [("Games", [lost, won])],
        value_format="int",
        note=_games_note(ev.get("n")),
    )


def _accuracy_chart(ins: Insight) -> Optional[Chart]:
    """Engine accuracy in wins / draws / losses (a line at your opponents' average), or chess.com's accuracy."""
    ev = ins.evidence
    by_result = [(label, _num(ev.get(key))) for key, label in (("win", "Wins"), ("draw", "Draws"), ("loss", "Losses"))]
    by_result = [(label, v) for label, v in by_result if v is not None]
    if by_result and "opponents" in ev:
        opp = _num(ev.get("opponents"))
        return comparison_chart(
            "Your engine accuracy by result",
            [label for label, _ in by_result],
            [("Your accuracy", [v for _, v in by_result])],
            value_format="float1",
            reference=opp,
            note=_games_note(ev.get("games"), "the line is your opponents' average" if opp is not None else ""),
        )
    mine, opp = _num(ev.get("mine")), _num(ev.get("opponents"))
    if mine is not None and opp is not None:
        rows = [(YOU, mine)]
        rows += [(label, v) for key, label in (("wins", "In your wins"), ("losses", "In your losses"))
                 if (v := _num(ev.get(key))) is not None]
        rows.append((OPPONENTS, opp))
        return comparison_chart(
            "chess.com accuracy in reviewed games",
            [label for label, _ in rows],
            [("Accuracy", [v for _, v in rows])],
            value_format="float1",
            note=_games_note(ev.get("n")),
        )
    return None


def _by_format_rate_chart(ins: Insight) -> Optional[Chart]:
    """A rate per format ({"by_time_class": {"blitz": 0.08, ...}}), with the overall rate as a line."""
    ev = ins.evidence
    by = ev.get("by_time_class")
    if not isinstance(by, dict):
        return None
    rates = {tc: v for tc, v in ((tc, _share(v)) for tc, v in by.items()) if v is not None}
    if not rates:
        return None
    order = list(ordered_formats({tc: 1 for tc in rates}))
    what = "Draw rate" if ins.category == "endings" and "draws" in ev else "Rate"
    several = len(order) > 1
    return comparison_chart(
        f"{what} by format" if several else what,
        [FORMAT_NAMES.get(tc, tc) for tc in order],
        [(what, [rates[tc] for tc in order])],
        value_format="pct",
        reference=_share(ev.get("rate")) if several else None,
        note=_games_note(ev.get("n"), "the line is your rate over all formats" if several else ""),
    )


def _opening_eval_chart(ins: Insight) -> Optional[Chart]:
    """Stockfish's average evaluation after the opening, in pawns from your side (0 = equal)."""
    ev = ins.evidence
    avg = _num(ev.get("avg_cp"))
    if avg is None or not ev.get("family"):
        return None
    return comparison_chart(
        "Evaluation after move 10",
        [str(ev["family"])],
        [("Pawns, from your side", [avg / 100.0])],
        value_format="signed_float2",
        reference=0.0,
        note=_games_note(ev.get("games"), "0 is an equal position"),
    )


def _rating_chart(ins: Insight) -> Optional[Chart]:
    """A rating trend: your rating at the start and at the end of the period (evidence "first" / "last")."""
    ev = ins.evidence
    first, last, pool = _num(ev.get("first")), _num(ev.get("last")), ev.get("pool")
    if first is None or last is None or not isinstance(pool, str):
        return None
    change = _num(ev.get("change"))
    trend = f"the fitted trend is {change:+.0f} points".replace("-", MINUS) if change is not None else ""
    return comparison_chart(
        f"Your {pool} rating over the period",
        ["First game", "Latest game"],
        [("Rating", [first, last])],
        value_format="rating",
        note=_games_note(ev.get("n"), trend),
    )


def _breadth_chart(ins: Insight) -> Optional[Chart]:
    """Repertoire breadth: how many different moves make up most of your games, per position."""
    ev = ins.evidence
    ks = ev.get("choices_for_coverage")
    if not isinstance(ks, dict) or not ks:
        return None
    rows = [(k, _count(v)) for k, v in ks.items()]
    rows = [(k, v) for k, v in rows if v]
    if not rows:
        return None
    labels = ["Your first move" if k in ("", "first move") else f"Against 1.{k}" for k, _ in rows]
    return comparison_chart(
        "Different moves that make up most of your games",
        labels,
        [("Moves", [float(v) for _, v in rows])],
        value_format="int",
        note=_games_note(ev.get("games"), "fewer moves = a narrower repertoire"),
    )


_CHARTS: Sequence[Callable[[Insight], Optional[Chart]]] = (
    _per_format_pair_chart,
    _grouped_score_chart,
    _score_chart,
    _delta_chart,
    _share_chart,
    _pair_chart,
    _flag_chart,
    _saved_chart,
    _accuracy_chart,
    _by_format_rate_chart,
    _opening_eval_chart,
    _rating_chart,
    _breadth_chart,
)


def chart_from_evidence(ins: Insight) -> Optional[Chart]:
    """The first chart pattern the finding's evidence fits, or None."""
    if not isinstance(ins.evidence, dict):
        return None
    for build in _CHARTS:
        chart = build(ins)
        if chart is not None:
            return chart
    return None


def _board_of(fen: str) -> Optional[chess.Board]:
    for chess960 in (False, True):
        try:
            return chess.Board(fen, chess960=chess960)
        except ValueError:
            continue
    return None


def diagram_from_evidence(ins: Insight) -> Optional[Diagram]:
    """The board for a finding about one position (evidence "fen" and your "move"; the engine's "best" in green).

    The board is seen from the side to move (the move was yours); nothing when your move is not legal there.
    """
    ev = ins.evidence
    fen, move = ev.get("fen"), ev.get("move")
    if not isinstance(fen, str) or not fen or not isinstance(move, str) or not move:
        return None
    board = _board_of(fen)
    played = parse_move(board, move) if board is not None else None
    if board is None or played is None:
        return None
    best = ev.get("best") if isinstance(ev.get("best"), str) else None
    better = parse_move(board, best) if best else None
    played_label = move_label(board, played)
    best_label = move_label(board, better) if better is not None and better != played else None
    return position_diagram(
        f"You played {played_label} here",
        fen,
        orientation="white" if board.turn == chess.WHITE else "black",
        played=played.uci(),
        best=better.uci() if best_label else None,
        caption=f"Red: your move {played_label}."
        + (f" Green: Stockfish's choice, {best_label}." if best_label else ""),
        link=ins.example_games[0] if ins.example_games else "",
    )


def fill_visuals(modules: Iterable[ModuleResult]) -> None:
    """Give every finding with neither a chart nor a diagram one built from its evidence (when one fits)."""
    for m in modules:
        for ins in m.insights or []:
            if ins.chart is not None or ins.diagram is not None:
                continue
            try:
                chart = chart_from_evidence(ins)
                if chart is not None:
                    ins.chart = chart
                else:
                    ins.diagram = diagram_from_evidence(ins)
            except Exception as exc:  # noqa: BLE001 — a picture is never worth a failed report
                log.warning("no fallback picture for %s: %s", ins.id, exc)


def fill(modules: Sequence[ModuleResult], ctx: "AnalysisContext") -> None:
    """Both fallbacks, each isolated: a failure logs a warning and leaves the findings as they were."""
    for name, step in (("formats", lambda: fill_formats(modules, ctx)), ("visuals", lambda: fill_visuals(modules))):
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            log.warning("filling in the findings' %s failed: %s", name, exc)
