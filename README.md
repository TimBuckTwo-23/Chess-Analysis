# chess-insights

Download **every game you have played on chess.com** and turn them into a report
of your real strengths, your weaknesses, and a prioritised study plan.

It uses the free, public [chess.com Published-Data API](https://www.chess.com/news/view/published-data-api)
(no login, no API key), so it only needs your username.

What you get:

* **Results & rating**: how you score against the *Elo expectation* per time control
  and colour, and against weaker or stronger opponents.
* **Openings**: which openings gain or lose you points, as White and as Black,
  with links to the games worth reviewing.
* **Clock**: time-trouble frequency, games lost on time, and whether you burn
  time in the opening compared with your opponents.
* **How games end**: how often you get mated, flag or resign, and whether you
  do better in short or long games.
* **Habits & tilt**: results right after a loss, deep into a session, and by
  time of day.
* **Engine review** (optional, with Stockfish): Lichess-style accuracy, blunder
  rates by game phase, blunders under time pressure, converting winning
  positions, missed tactics, and how you come out of each opening.

Every insight is compared with a benchmark: your Elo expectation, or your
opponents in the same games. Findings from small samples are held back, so the
report doesn't overreact to a handful of games.

## Quick start

Requires Python 3.10+.

```bash
# Windows (use `py`); on macOS/Linux use python3
py -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e .

chess-insights report YOUR_USERNAME --contact you@example.com
```

This downloads your archive into `~/.chess-insights-cache/` (on Windows
`%USERPROFILE%\.chess-insights-cache`; set `CHESS_INSIGHTS_CACHE` or `--cache-dir` to
change it). Later runs only re-check the current month. The report is written to
`reports/<username>.html` (plus `.md` and `.json`). Open the HTML file in a browser.

The username is the name in your profile address, `chess.com/member/NAME`; pasting the
whole profile URL works too. `--contact` puts your email in the User-Agent, which chess.com
asks API users to do (or set `CHESS_INSIGHTS_CONTACT` once).

To only download or update the archive, without a report:

```bash
chess-insights fetch YOUR_USERNAME                 # add --refresh-all to re-check old months
```

### Deeper analysis with Stockfish

Install [Stockfish](https://stockfishchess.org/download/). On Windows, unzip it and note
the path to the `.exe` inside (e.g. `stockfish-windows-x86-64-avx2.exe`). Then run:

```bash
chess-insights report YOUR_USERNAME --engine --stockfish "C:\path\to\stockfish.exe"
```

`--stockfish` also accepts the unzipped folder. Without it, Stockfish is looked for in
`CHESS_INSIGHTS_STOCKFISH`, on `PATH` and in the usual install and Downloads folders.
Add `--puzzles` to also get your own mistakes as a PGN puzzle file.

By default this analyses your 150 most recent games at depth 12, running one Stockfish
per CPU core except one (`--workers N` to change that). Stockfish evaluates every
position once, about 65 positions for a typical game, so 150 games are roughly 10,000
positions.

How long that takes, measured with Stockfish 16 at depth 12 on 6 real chess.com games
(715 positions) on a shared 4-core cloud machine:

| Engines (`--workers`) | Positions per second | 150 games |
|---|---|---|
| 1 | 26 | about 6 minutes |
| 3 | 69 | about 2½ minutes |

A recent desktop CPU is usually faster per core, and more cores help almost linearly.
Results are cached per Stockfish version and depth, so re-running the report only
analyses new games (150 cached games load in about a second). Use `--engine-games` and
`--depth` to trade speed for coverage: each extra level of depth makes the analysis about
1.5 times slower.

### Useful options

| Option | Meaning |
|---|---|
| `--time-class blitz,rapid` | Only these time controls (bullet, blitz, rapid, daily) |
| `--since 2025-01 --until 2025-12` | Only this period; also `--since 2024` or `--since 2025-03-15` |
| `--rated-only` | Skip casual games |
| `--rules all` | Include variants (default: standard chess only; or e.g. `--rules chess,chess960`) |
| `--tz America/New_York` | Your time zone, for time-of-day stats and the `--since`/`--until` day boundaries (default UTC) |
| `--offline` | Use the local cache only |
| `--pgn games.pgn` | Analyse PGN files, a folder of them or a wildcard (`C:\games\*.pgn`) instead of the API; PGNs from other sites or over-the-board games work too |
| `--json archive.json` | Analyse saved chess.com API responses (files, a folder or a wildcard) |
| `--out reports/me --formats html,md` | Output location and formats: writes `reports/me.html` and `reports/me.md`; an existing folder (or a path ending in a slash) gets `<username>.html` … inside it |
| `-v` | Debug logging |

`chess-insights report --help` lists everything. Exit codes: `0` done, `1` no games to
analyse (or the report or the game cache could not be written), `2` a usage mistake (unknown option value,
unknown player, missing file or Stockfish), `3` chess.com could not be reached.

### Try it without an account

```bash
chess-insights demo
```

This generates 400 games for a fictional player with planted strengths and weaknesses
(a leaky Caro-Kann, time trouble in blitz, tilt after losses, weak endgames…). It
then analyses them and reports which planted traits it found.

## Troubleshooting

* **HTTP 403 from chess.com.** chess.com's CDN rejects requests that don't identify
  themselves. Add `--contact you@example.com` (it goes into the User-Agent). If it
  persists, your network is blocking the API (see the next point).
* **"could not reach api.chess.com" / certificate errors.** Some school and company
  networks block `api.chess.com` or inspect HTTPS traffic. Try another network, or point
  Python at your proxy (`set HTTPS_PROXY=http://proxy:port`) and at your company's root
  certificate (`set REQUESTS_CA_BUNDLE=C:\path\to\company-ca.pem`). If the games were
  downloaded before, the report falls back to them; `--offline` skips the network entirely.
  Every retry is announced (`chess.com: …; retrying in 2s`); with no connection at all it
  gives up after one retry instead of waiting minutes.
  Without any access, download your games as PGN from the chess.com website (Archive →
  Download) and run `chess-insights report YOUR_USERNAME --pgn "C:\Downloads\games.pgn"`.
* **"chess.com has no player …".** Use the name from your profile address
  (`chess.com/member/NAME`), not your display name. Closed accounts are reported as such.
* **A month "could not be downloaded right now (404)".** chess.com sometimes fails to serve
  a month for a while; run the same command again later and only the missing months are
  fetched.
* **Stockfish on Windows.** Pass the `.exe` (or its folder) with `--stockfish`, quoting paths
  with spaces, or set `CHESS_INSIGHTS_STOCKFISH` once. With `--engine` but no Stockfish found,
  the report is still written, without the engine sections.
* **Unknown time zone on Windows.** Windows has no time zone database of its own:
  `pip install tzdata`.
* **`chess-insights` is not recognized.** Activate the virtual environment, or run
  `py -m chess_insights report ...` instead.
* **Odd characters in the console.** A Windows console that isn't UTF-8 shows `?` for
  characters it can't display; the report files are always UTF-8.
* **Using it as a library.** `demo` (and `synth.generate_archives`) run workers in separate
  processes, which Windows and macOS start with *spawn*: put the calling code under
  `if __name__ == "__main__":`, or pass `workers=1`.

## How it works

`fetch` → `parse` → `filter` → optional `engine` → analysis modules → ranked insights
→ HTML, Markdown and JSON report. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
module map, [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for how every number is computed, and [PLAN.md](PLAN.md) for the roadmap.

## Development

```bash
pip install -e ".[dev]"
py -m pytest            # engine tests are skipped automatically if Stockfish isn't found
```
