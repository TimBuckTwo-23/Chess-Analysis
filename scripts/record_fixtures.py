"""Record public data the test suite and the coaching layer are checked against.

Runs on GitHub's runners (.github/workflows/fixtures.yml), which can reach Lichess, Wikibooks and
Project Gutenberg; the output is committed to the `fixtures` branch and copied into tests/fixtures
and src/chess_insights/data by hand. Every item is independent: one failure is recorded in
manifest.json and the rest still run.

    python scripts/record_fixtures.py OUT_DIR
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = "chess-insights fixture recorder (+https://github.com/TimBuckTwo-23/Chess-Analysis)"
CORE_THEMES = ("fork", "pin", "skewer", "hangingPiece", "discoveredAttack", "backRankMate")
EXTRA_THEMES = (
    "discoveredCheck", "doubleCheck", "trappedPiece", "smotheredMate", "mateIn1", "mateIn2", "mateIn3",
    "deflection", "attraction", "overloading", "advancedPawn",
)
PER_THEME = 80
RANDOM_ROWS = 300
DRILL_ROWS = 6000

# the three habit positions from the coaching plan, and the moves played / preferred there
HABITS = {
    "sicilian_5e5": ("1.e4 c5 2.Nf3 Nc6 3.Nc3 e6 4.d4 cxd4 5.Nxd4", ["e5", "a6", "d6"]),
    "qga_4e4": ("1.d4 d5 2.c4 dxc4 3.Nc3 Nc6", ["e4", "Nf3", "d5"]),
    "sicilian_2nc6": ("1.e4 c5 2.d4", ["Nc6", "cxd4"]),
    "qga_4d5": ("1.d4 d5 2.c4 dxc4 3.e4 e5", ["d5", "Nf3"]),
}


def get(url: str, *, headers: dict | None = None, timeout: float = 60) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers or {})


def record(out: Path, name: str, url: str, **kw) -> dict:
    status, body, headers = get(url, **kw)
    try:
        payload = json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = body.decode("utf-8", "replace")
    doc = {"url": url, "status": status, "content_type": headers.get("Content-Type", ""),
           "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"), "body": payload}
    (out / f"{name}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    time.sleep(1.2)  # one request at a time, politely spaced
    return {"status": status}


def fens() -> dict[str, dict[str, str]]:
    import chess

    result = {}
    for key, (line, moves) in HABITS.items():
        board = chess.Board()
        for tok in line.split():
            if not tok[0].isdigit() or "." not in tok:
                board.push_san(tok)
            else:
                board.push_san(tok.split(".")[-1])
        entry = {"before": board.fen()}
        for san in moves:
            b = board.copy()
            b.push_san(san)
            entry[san] = b.fen()
        result[key] = entry
    return result


def puzzles(out: Path) -> dict:
    import zstandard

    url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    rng = random.Random(20260926)
    by_theme: dict[str, list[dict]] = {t: [] for t in CORE_THEMES + EXTRA_THEMES}
    reservoir: list[dict] = []
    drill: list[dict] = []
    seen = eligible = 0
    header = None
    with urllib.request.urlopen(req, timeout=120) as resp:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(resp), encoding="utf-8")
        for row in csv.reader(reader):
            if header is None:
                header = row
                continue
            seen += 1
            rec = dict(zip(header, row))
            try:
                rating, pop, plays = int(rec["Rating"]), int(rec["Popularity"]), int(rec["NbPlays"])
            except (KeyError, ValueError):
                continue
            if not (800 <= rating <= 2200 and pop >= 80 and plays >= 300):
                continue
            eligible += 1
            themes = rec["Themes"].split()
            for t in themes:
                bucket = by_theme.get(t)
                if bucket is not None:
                    if len(bucket) < PER_THEME:
                        bucket.append(rec)
                    elif rng.random() < 0.002:
                        bucket[rng.randrange(PER_THEME)] = rec
            if len(reservoir) < RANDOM_ROWS:
                reservoir.append(rec)
            elif rng.random() < RANDOM_ROWS / eligible:
                reservoir[rng.randrange(RANDOM_ROWS)] = rec
            if len(drill) < DRILL_ROWS:
                drill.append(rec)
            elif rng.random() < DRILL_ROWS / eligible:
                drill[rng.randrange(DRILL_ROWS)] = rec
    cols = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays", "Themes", "GameUrl",
            "OpeningTags"]
    sample: dict[str, dict] = {}
    for t in CORE_THEMES + EXTRA_THEMES:
        for rec in by_theme[t]:
            sample[rec["PuzzleId"]] = rec
    for rec in reservoir:
        sample[rec["PuzzleId"]] = rec
    for name, rows in (("lichess_puzzles_sample.csv", sorted(sample.values(), key=lambda r: r["PuzzleId"])),
                       ("lichess_puzzles_drill_subset.csv", sorted(drill, key=lambda r: r["PuzzleId"]))):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return {"rows_seen": seen, "eligible": eligible, "sample_rows": len(sample), "drill_rows": len(drill),
            "per_theme": {t: len(v) for t, v in by_theme.items()}}


def openings(out: Path) -> dict:
    rows = 0
    for part in "abcde":
        status, body, _ = get(f"https://raw.githubusercontent.com/lichess-org/chess-openings/master/{part}.tsv")
        if status != 200:
            raise RuntimeError(f"{part}.tsv: HTTP {status}")
        (out / f"chess_openings_{part}.tsv").write_bytes(body)
        rows += body.count(b"\n") - 1
    return {"rows": rows}


def lichess_apis(out: Path) -> dict:
    status = {}
    positions = fens()
    (out / "habit_fens.json").write_text(json.dumps(positions, indent=1), encoding="utf-8")
    for key, entry in positions.items():
        for label, fen in entry.items():
            q = urllib.parse.quote(fen)
            status[f"cloud_eval_{key}_{label}"] = record(
                out, f"cloud_eval_{key}_{label}", f"https://lichess.org/api/cloud-eval?fen={q}&multiPv=3")
    # a position that is surely not in the cloud database (404)
    odd = "8/8/8/3k4/8/2K5/8/7Q w - - 0 1"
    status["cloud_eval_missing"] = record(
        out, "cloud_eval_missing", "https://lichess.org/api/cloud-eval?fen=" + urllib.parse.quote(odd))
    tb = {
        "tablebase_kpk_win": "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1",
        "tablebase_kpk_draw": "4k3/8/4K3/4P3/8/8/8/8 b - - 0 1",
        "tablebase_krk": "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1",
        "tablebase_rook_ending": "8/5k2/8/3R4/5P2/5K2/8/r7 w - - 0 1",
        "tablebase_too_many": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    }
    for name, fen in tb.items():
        status[name] = record(out, name, "https://tablebase.lichess.org/standard?fen=" + urllib.parse.quote(fen))
    for key in ("sicilian_5e5", "qga_4e4"):
        fen = urllib.parse.quote(positions[key]["before"])
        status[f"explorer_masters_{key}"] = record(
            out, f"explorer_masters_{key}", f"https://explorer.lichess.org/masters?fen={fen}&topGames=2")
        status[f"explorer_lichess_{key}"] = record(
            out, f"explorer_lichess_{key}",
            f"https://explorer.lichess.org/lichess?fen={fen}&speeds=blitz,rapid&ratings=1200,1400,1600&topGames=2")
    return status


def wikibooks(out: Path) -> dict:
    status = {}
    pages = {
        "wikibooks_sicilian": "Chess_Opening_Theory/1. e4/1...c5",
        "wikibooks_sicilian_nf3_nc6": "Chess_Opening_Theory/1. e4/1...c5/2. Nf3/2...Nc6",
        "wikibooks_qga": "Chess_Opening_Theory/1. d4/1...d5/2. c4/2...dxc4",
        "wikibooks_missing": "Chess_Opening_Theory/1. d4/1...d5/2. c4/2...dxc4/3. Nc3/3...Nc6/4. e4",
    }
    for name, title in pages.items():
        q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "exsentences": 4,
                                    "titles": title, "format": "json", "redirects": 1})
        status[name] = record(out, name, f"https://en.wikibooks.org/w/api.php?{q}")
    return status


def gutenberg(out: Path) -> dict:
    result = {}
    for num in (33870, 5614, 4542, 4656):
        url = f"https://www.gutenberg.org/cache/epub/{num}/pg{num}.txt"
        status, body, _ = get(url)
        if status != 200:
            result[num] = {"status": status}
            continue
        text = body.decode("utf-8", "replace")
        lines = [ln.strip() for ln in text.splitlines()]
        headings = [ln for ln in lines if ln and len(ln) < 80 and (
            ln.isupper() or ln.upper().startswith("CHAPTER") or ln.upper().startswith("PART ")
            or ln[:4].strip(".").isdigit())]
        (out / f"gutenberg_{num}_headings.json").write_text(json.dumps(headings[:600], indent=1), encoding="utf-8")
        (out / f"gutenberg_{num}.txt").write_text(text, encoding="utf-8")
        result[num] = {"status": status, "headings": len(headings)}
        time.sleep(1.2)
    return result


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures-out")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items": {}}
    for name, fn in (("openings", openings), ("lichess_apis", lichess_apis), ("wikibooks", wikibooks),
                     ("gutenberg", gutenberg), ("puzzles", puzzles)):
        try:
            manifest["items"][name] = {"ok": True, "result": fn(out)}
        except Exception as exc:  # noqa: BLE001 — record and carry on
            manifest["items"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                       "trace": traceback.format_exc()[-2000:]}
        print(name, json.dumps(manifest["items"][name])[:500], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
