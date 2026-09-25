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

This downloads your archive into `~/.chess-insights-cache/`. Later runs only fetch
the current month. The report is written to `reports/<username>.html` (plus `.md` and
`.json`). Open the HTML file in a browser.

`--contact` puts your email in the User-Agent, which chess.com asks API users to do.

### Deeper analysis with Stockfish

Install [Stockfish](https://stockfishchess.org/download/). On Windows, unzip it and note
the path to `stockfish-windows-x86-64.exe`. Then run:

```bash
chess-insights report YOUR_USERNAME --engine --stockfish "C:\path\to\stockfish.exe"
```

By default this analyses your 150 most recent games at depth 12, using all but one CPU
core. That takes a few minutes, and results are cached. Use `--engine-games` and
`--depth` to trade speed for coverage.

### Useful options

| Option | Meaning |
|---|---|
| `--time-class blitz,rapid` | Only these time controls |
| `--since 2025-01 --until 2025-12` | Only these months |
| `--rated-only` | Skip casual games |
| `--rules all` | Include variants (default: standard chess only) |
| `--tz America/New_York` | Time zone for time-of-day stats |
| `--offline` | Use the local cache only |
| `--pgn games.pgn` | Analyse a PGN export instead of the API |
| `--out reports/me --formats html,md` | Output location and formats |

### Try it without an account

```bash
chess-insights demo
```

This generates 400 games for a fictional player with planted strengths and weaknesses
(a leaky Caro-Kann, time trouble in blitz, tilt after losses, weak endgames…). It
then analyses them and reports which planted traits it found.

## How it works

`fetch` → `parse` → `filter` → optional `engine` → analysis modules → ranked insights
→ HTML, Markdown and JSON report. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
module map and the statistical rules, and [PLAN.md](PLAN.md) for the roadmap.

## Development

```bash
pip install -e ".[dev]"
py -m pytest            # engine tests are skipped automatically if Stockfish isn't found
```
