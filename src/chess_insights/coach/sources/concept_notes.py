"""Short notes on positional concepts (``data/concepts.json``), each citing a public-domain book chapter.

An explanation links the note for its largest concept difference: ``concept_note("King safety")`` (a Stockfish 16
classical term), ``concept_note("king safety")`` (its plain label) or ``concept_note("isolated_pawns")`` (a
python-chess board fact) all find the note, or None when no note covers the term.
"""

from __future__ import annotations

import functools
import json
import re
from importlib import resources
from typing import Any, Optional

from ...models import Source

# words that point to a concept when no alias matches exactly (checked in this order)
_KEYWORDS = (
    ("bishop pair", "bishop_pair"),
    ("two bishops", "bishop_pair"),
    ("castl", "king_safety"),
    ("king", "king_safety"),
    ("develop", "development"),
    ("outpost", "holes_outposts"),
    ("hole", "holes_outposts"),
    ("weak square", "holes_outposts"),
    ("passed", "passed_pawns"),
    ("open file", "open_files"),
    ("semi open", "open_files"),
    ("isolated", "pawn_structure"),
    ("doubled", "pawn_structure"),
    ("backward", "pawn_structure"),
    ("pawn", "pawn_structure"),
    ("mobility", "piece_activity"),
    ("activity", "piece_activity"),
    ("space", "space"),
    ("centre", "space"),
    ("center", "space"),
)


def _key(text: str) -> str:
    return " ".join(re.sub(r"[_\-]+", " ", str(text or "")).lower().split())


@functools.lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """The bundled notes: {"books": {...}, "concepts": {key: {label, text, aliases, cites}}}."""
    text = resources.files("chess_insights.data").joinpath("concepts.json").read_text(encoding="utf-8")
    return json.loads(text)


@functools.lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, concept in load()["concepts"].items():
        for name in [key, concept.get("label", ""), *concept.get("aliases", [])]:
            out.setdefault(_key(name), key)
    return out


def _resolve(term: str) -> Optional[str]:
    k = _key(term)
    if not k:
        return None
    exact = _aliases().get(k)
    if exact:
        return exact
    for word, concept in _KEYWORDS:
        if word in k:
            return concept
    return None


def concept_note(term_or_label: str) -> Optional[dict[str, Any]]:
    """The note for a concept, as a plain dict, or None when no note covers ``term_or_label``.

    Keys: ``key``, ``label``, ``text`` (two or three sentences in the report's voice), ``cites`` (each with the
    book's ``title``, ``author``, ``ebook`` number, ``chapter``, ``section`` and Gutenberg ``url``), ``credit``
    (one line naming the first citation) and ``url`` (its Gutenberg page).
    """
    key = _resolve(term_or_label)
    if key is None:
        return None
    data = load()
    concept = data["concepts"][key]
    cites = []
    for cite in concept.get("cites", []):
        book = data["books"].get(cite.get("book", ""), {})
        cites.append({
            "title": book.get("title", ""),
            "author": book.get("author", ""),
            "ebook": book.get("ebook"),
            "chapter": cite.get("chapter", ""),
            "section": cite.get("section", ""),
            "url": book.get("url", ""),
            "license": book.get("license", ""),
        })
    first = cites[0] if cites else {}
    credit = (
        f"{first['author'].split()[-1]}, {first['title']} (Project Gutenberg #{first['ebook']}), "
        f"{first['chapter']}: {first['section']}"
        if first else ""
    )
    return {
        "key": key,
        "label": concept.get("label", key),
        "text": concept.get("text", ""),
        "cites": cites,
        "credit": credit,
        "url": first.get("url", ""),
    }


def note_sources(note: Optional[dict[str, Any]]) -> list[Source]:
    """The note's citations as ``Source`` objects (for ``Explanation.sources``)."""
    if not note:
        return []
    return [
        Source(
            name=f"{c['author']}, {c['title']} (Project Gutenberg #{c['ebook']}), {c['section']}",
            url=c["url"],
            license=c.get("license", ""),
        )
        for c in note.get("cites", [])
    ]


def all_notes() -> list[dict[str, Any]]:
    """Every note, in file order (for a glossary)."""
    return [n for n in (concept_note(k) for k in load()["concepts"]) if n]
