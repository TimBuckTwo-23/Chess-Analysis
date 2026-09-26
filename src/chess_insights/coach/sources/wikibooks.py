"""Wikibooks Chess Opening Theory: a short, credited plan text for the nearest position it covers.

The book has one page per position, named by the moves: ``Chess_Opening_Theory/1._e4/1...c5/2._Nf3``. Coverage
is uneven, so :func:`snippet` asks the MediaWiki API which of the pages along the moves exist (one request for
all of them), then fetches a plain-text extract of the deepest one (a second request) and keeps at most two
sentences. The text is CC BY-SA 4.0: it is shown with "Wikibooks, CC BY-SA 4.0" and the page link, and only the
short extract is ever stored (in the sources cache, for 90 days).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from urllib.parse import quote, urlencode

from ...models import Source
from .http import Fetcher

SOURCE = "wikibooks"
API = "https://en.wikibooks.org/w/api.php"
WIKI = "https://en.wikibooks.org/wiki/"
ROOT = "Chess_Opening_Theory"
LICENSE = "CC BY-SA 4.0"
CREDIT = "Wikibooks, CC BY-SA 4.0"
MAX_SENTENCES = 2
# Deeper pages are rare. Every prefix goes into one existence query, so its URL grows with the square of the
# depth: about 5 KB at 30 plies, 9 KB at 40 (past the 8 KB many servers accept). The API takes 50 titles.
MAX_PLIES = 30
EXTRACT_SENTENCES = 4  # asked for (headings count), then cut to MAX_SENTENCES


@dataclass
class WikiSnippet:
    text: str  # at most two sentences
    url: str  # the page
    title: str
    plies: int  # how many of the moves the page covers
    source: Source


def page_title(moves_san: Sequence[str]) -> str:
    """'Chess_Opening_Theory/1. e4/1...c5/2. Nf3' for the position after ``moves_san`` (the API's title form)."""
    parts = [ROOT]
    for i, san in enumerate(moves_san):
        n = i // 2 + 1
        parts.append(f"{n}. {san}" if i % 2 == 0 else f"{n}...{san}")
    return "/".join(parts)


def page_url(title: str) -> str:
    """The reader's link: https://en.wikibooks.org/wiki/Chess_Opening_Theory/1._e4/1...c5"""
    return WIKI + quote(title.replace(" ", "_"), safe="/._-+()!,:")


def extract_url(title: str) -> str:
    """A plain-text extract of the page's first sentences."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsentences": EXTRACT_SENTENCES,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    return f"{API}?{urlencode(params)}"


def exists_url(titles: Sequence[str]) -> str:
    """Which of ``titles`` exist (no text), in one request."""
    params = {"action": "query", "titles": "|".join(titles), "format": "json", "redirects": 1}
    return f"{API}?{urlencode(params)}"


def _norm(title: str) -> str:
    return title.replace("_", " ").strip()


def _pages(body: Any) -> dict[str, dict[str, Any]]:
    """Normalised title -> page dict, following the API's ``normalized`` and ``redirects`` lists."""
    query = body.get("query") if isinstance(body, dict) else None
    if not isinstance(query, dict):
        return {}
    pages = {}
    for page in (query.get("pages") or {}).values():
        if isinstance(page, dict) and page.get("title"):
            pages[_norm(page["title"])] = page
    for key in ("normalized", "redirects"):
        for entry in query.get(key) or []:
            if isinstance(entry, dict) and _norm(entry.get("to", "")) in pages:
                pages.setdefault(_norm(entry.get("from", "")), pages[_norm(entry["to"])])
    return pages


def _exists(page: Optional[dict[str, Any]]) -> bool:
    return bool(page) and "missing" not in page and "invalid" not in page


_HEADING = re.compile(r"^\s*=+.*=+\s*$", re.M)
_CANDIDATE = re.compile(r"[.!?](?=\s+[\"'(A-Z0-9])")
_ABBREVIATIONS = {"mr", "mrs", "dr", "st", "vs", "cf", "e.g", "i.e", "etc", "no", "ch"}


def _ends_sentence(text: str, pos: int) -> bool:
    """Whether the mark at ``text[pos]`` ends a sentence: not after a move number ("3. Qa4+"), an ellipsis
    ("1...c5"), an initial ("E. Lasker") or a common abbreviation."""
    if text[pos] != ".":
        return True
    token = text[:pos].rsplit(None, 1)[-1] if text[:pos].strip() else ""
    token = token.lstrip("(\"'")
    if not token or token.isdigit() or token.endswith("."):
        return False
    if len(token) == 1 and token.isalpha():
        return False
    return token.lower() not in _ABBREVIATIONS


def first_sentences(text: str, n: int = MAX_SENTENCES) -> str:
    """The first ``n`` sentences of an extract, without its section headings; move numbers don't end sentences."""
    body = " ".join(_HEADING.sub(" ", text or "").split())
    if not body:
        return ""
    out = []
    start = 0
    for m in _CANDIDATE.finditer(body):
        if not _ends_sentence(body, m.start()):
            continue
        out.append(body[start : m.end()].strip())
        start = m.end()
        if len(out) == n:
            break
    else:
        rest = body[start:].strip()
        # a trailing fragment ("... is the Open Sicilian: 2.") is where the API cut the extract: drop it
        if rest and len(out) < n and rest[-1] in ".!?" and _ends_sentence(rest, len(rest) - 1):
            out.append(rest)
    return " ".join(out)


def snippet(fetcher: Fetcher, moves_san: Sequence[str]) -> Optional[WikiSnippet]:
    """Up to two sentences from the deepest Wikibooks page along ``moves_san``; None when none is available."""
    moves = list(moves_san[:MAX_PLIES])
    if not moves:
        return None
    titles = [page_title(moves[:k]) for k in range(len(moves), 0, -1)]  # deepest first
    got = fetcher.get(SOURCE, exists_url(titles))
    if got is None or not got.ok:
        return None
    pages = _pages(got.data)
    for k, title in zip(range(len(moves), 0, -1), titles):
        if not _exists(pages.get(_norm(title))):
            continue
        page = fetcher.get(SOURCE, extract_url(title))
        if page is None or not page.ok:
            return None
        found = _pages(page.data).get(_norm(title))
        text = first_sentences(str((found or {}).get("extract") or ""))
        if not text:
            return None
        url = page_url(title)
        return WikiSnippet(
            text=text, url=url, title=_norm(title), plies=k,
            source=Source(name=CREDIT, url=url, retrieved=page.retrieved, license=LICENSE),
        )
    return None


def move_path(moves_san: Sequence[str]) -> str:
    """'1.e4 c5 2.Nf3' for a caption."""
    return " ".join(f"{i // 2 + 1}.{san}" if i % 2 == 0 else san for i, san in enumerate(moves_san))
