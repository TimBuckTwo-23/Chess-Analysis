"""Small builders for the pictures that go with findings: format mixes, comparison charts, boards and strips.

Everything here returns plain model objects (Chart, Diagram, Strip ...); the renderers draw them. Shared by the
analysis modules and the coaching layer so every picture in the report reads the same way.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import chess

from .models import Arrow, Chart, Diagram, Frame, Game, Mark, Series, Strip, Table

FORMAT_ORDER = ("bullet", "blitz", "rapid", "daily")
FORMAT_NAMES = {"bullet": "Bullet", "blitz": "Blitz", "rapid": "Rapid", "daily": "Daily"}
MIN_FORMAT_GAMES = 10  # a format with fewer games gets no bar of its own in a split


def ordered_formats(counts: dict[str, int]) -> dict[str, int]:
    """``counts`` in the report's format order (unknown time classes last), zero counts dropped."""
    known = [(tc, counts[tc]) for tc in FORMAT_ORDER if counts.get(tc)]
    other = sorted((tc, n) for tc, n in counts.items() if tc not in FORMAT_ORDER and n)
    return dict(known + other)


def format_counts(games: Iterable[Game]) -> dict[str, int]:
    """Games per time class, in the report's format order."""
    counts: dict[str, int] = {}
    for g in games:
        counts[g.time_class] = counts.get(g.time_class, 0) + 1
    return ordered_formats(counts)


def format_text(counts: dict[str, int]) -> str:
    """"blitz only", "bullet and blitz", "bullet, blitz and rapid" (in format order)."""
    names = [FORMAT_NAMES.get(tc, tc).lower() for tc in ordered_formats(counts)]
    if not names:
        return ""
    if len(names) == 1:
        return f"{names[0]} only"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def split_by_format(
    games: Sequence[Game],
    metric: Callable[[list[Game]], Optional[float]],
    min_games: int = MIN_FORMAT_GAMES,
) -> tuple[list[str], list[Optional[float]], list[int]]:
    """``metric`` on each format's games: (labels, values, game counts). Formats below ``min_games`` are left out."""
    labels, values, counts = [], [], []
    for tc, n in format_counts(games).items():
        if n < min_games:
            continue
        subset = [g for g in games if g.time_class == tc]
        labels.append(FORMAT_NAMES.get(tc, tc))
        values.append(metric(subset))
        counts.append(n)
    return labels, values, counts


def comparison_chart(
    title: str,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[Optional[float]]]],
    *,
    value_format: str = "pct",
    reference: Optional[float] = None,
    note: str = "",
    kind: str = "hbar",
    table: Optional[Table] = None,
) -> Chart:
    """A small chart for a finding: one bar per label and series ("You" vs "Expected", or one per format)."""
    return Chart(
        kind=kind,  # type: ignore[arg-type]
        title=title,
        labels=list(labels),
        series=[Series(name, [None if v is None else float(v) for v in values]) for name, values in series],
        value_format=value_format,
        reference=reference,
        note=note,
        table=table,
    )


def _squares(board: chess.Board, move: chess.Move) -> tuple[str, str]:
    """(from, to) square names; castling points at the king's destination (Chess960 moves encode the rook)."""
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        to = chess.square(2 if board.is_queenside_castling(move) else 6, rank)
        return chess.square_name(move.from_square), chess.square_name(to)
    return chess.square_name(move.from_square), chess.square_name(move.to_square)


def parse_move(board: chess.Board, move: str) -> Optional[chess.Move]:
    """A SAN or UCI move that is legal in ``board``; None otherwise."""
    if not move:
        return None
    for parse in (board.parse_san, board.parse_uci):
        try:
            m = parse(move.split(".")[-1].lstrip(". ") if parse is board.parse_san else move)
        except ValueError:
            continue
        if m in board.legal_moves:
            return m
    return None


def move_arrow(board: chess.Board, move: str, kind: str) -> Optional[Arrow]:
    """An arrow for a SAN/UCI move in ``board`` (None if it isn't legal there)."""
    m = parse_move(board, move)
    if m is None:
        return None
    start, end = _squares(board, m)
    return Arrow(start, end, kind)


def _board(fen: str) -> Optional[chess.Board]:
    try:
        board = chess.Board(fen)
    except ValueError:
        try:
            board = chess.Board(fen, chess960=True)
        except ValueError:
            return None
    return board


def position_diagram(
    title: str,
    fen: str,
    *,
    orientation: str = "white",
    played: Optional[str] = None,
    best: Optional[str] = None,
    others: Sequence[tuple[str, str]] = (),
    caption: str = "",
    link: str = "",
    time_class: str = "",
    last_move: str = "",
    marks: Sequence[Mark] = (),
) -> Diagram:
    """A board before a decision: your move as a red arrow, the better move green, ``others`` as (move, kind)."""
    board = _board(fen)
    arrows: list[Arrow] = []
    if board is not None:
        for move, kind in [(played, "played"), (best, "best"), *others]:
            if move:
                arrow = move_arrow(board, move, kind)
                if arrow is not None and not any((a.start, a.end) == (arrow.start, arrow.end) for a in arrows):
                    arrows.append(arrow)
    return Diagram(
        title=title,
        fen=fen,
        caption=caption,
        link=link,
        orientation="black" if orientation == "black" else "white",
        arrows=arrows,
        marks=list(marks),
        last_move=last_move,
        time_class=time_class,
    )


def move_label(board: chess.Board, move: chess.Move) -> str:
    """"6.Ndb5" / "6...a6" for ``move`` in ``board``."""
    san = board.san(move)
    n = board.fullmove_number
    return f"{n}.{san}" if board.turn == chess.WHITE else f"{n}...{san}"


def line_strip(
    title: str,
    fen: str,
    moves_uci: Sequence[str],
    *,
    max_frames: int = 4,
    captions: Optional[Sequence[str]] = None,
    marks: Optional[Sequence[Sequence[Mark]]] = None,
) -> Strip:
    """A row of small boards: the position after each of the first ``max_frames`` moves of a line."""
    board = _board(fen)
    frames: list[Frame] = []
    if board is None:
        return Strip(title=title)
    for i, uci in enumerate(moves_uci[:max_frames]):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        label = move_label(board, move)
        board.push(move)
        frames.append(
            Frame(
                fen=board.fen(),
                move=label,
                last_move=uci,
                caption=(captions[i] if captions and i < len(captions) else ""),
                marks=list(marks[i]) if marks and i < len(marks) else [],
            )
        )
    return Strip(title=title, frames=frames)
