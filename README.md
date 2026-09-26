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
* **Positions you keep getting wrong** (with Stockfish): the exact positions
  where you have played the same bad move in several games, with the engine's
  move drawn on the board.
* **Personal puzzles** (`--puzzles`): your costliest mistakes exported as a PGN.
  You can import it into a Lichess study.
* **Bullet, blitz and rapid apart**: every finding says which formats it comes from
  ("blitz only", "bullet, blitz and rapid"), and the report has a separate view for each
  format you play often enough (60+ games), with its own findings, sections and study plan.
* **Coaching** (`--coach`, with Stockfish): for your costliest and repeated mistakes, the
  engine's refutation and better line drawn as boards, the tactic behind it (fork, pin,
  back rank ...) and two or three plain sentences on why; puzzle packs for the tactics you
  miss most, at your level; a review schedule; and, with a previous report, your progress
  against its targets. It explains and routes to practice; it adds no claims of its own.

Every insight is compared with a benchmark: what your rating predicts, your other
games, or your opponents in the same games. Findings that could be chance are
held back. On simulated players with no real strengths or weaknesses, a whole report
(its lists and every section) contains on average about 0.2 false strengths or
weaknesses, and fewer than 0.3 in every setting tested (`tests/test_null_calibration.py`).

The report is built to be read on a phone:

1. **Three lines at the top**: your ratings, the first three things to work on, and one
   strength to build on.
2. **Every weakness and strength, one line each**, most important first, each linked to
   its explanation. Nothing is hidden behind a "top 5".
3. **A study plan**: related findings share one item (a slow opening and losing on time
   are one clock problem), easiest changes first (when you play costs nothing; a new
   repertoire takes weeks), at most three actions per item with tick boxes the page
   remembers, a target for your next report ("use at most 30% of your clock on your
   first 15 moves (now 50%)"), and labelled example games ("Loss · 5+0 · vs 1512 · 12 Aug").
4. **One section per topic**: a short summary, key numbers and findings; the charts and
   tables fold away under "Show …", and wide tables show their key columns first.
5. **A picture for every finding**: a small chart of the numbers behind it (you against
   the rating's expectation or against your opponents in the same games), and for a
   position, the board with your move as a red arrow and the better move in green.

## Run it on GitHub (no install, works from a phone)

1. Open this repository on github.com, go to **Actions → report → Run workflow**.
2. Enter your chess.com username. Your time zone (e.g. `America/New_York`) is optional
   but recommended. Tap **Run workflow**.
3. When the run finishes (a few minutes, longer the first time with Stockfish), open it.
   Under **Artifacts**, download the report and open the `.html` file. Every report is
   also saved on the `reports` branch, under `<username>/latest/` and a dated folder, so
   you can compare reports over time.

Or list players in [`players.txt`](players.txt) (one username per line, optionally
followed by a time zone): saving a change to that file on `main` runs a report for everyone
listed. You can edit it in the GitHub app too.

GitHub's machines download your games and run Stockfish, and they remember what they
already downloaded and analysed. A re-run a week later only fetches and analyses the
new games.

What a GitHub run includes by default (the options on the Run workflow form):

* **An engine sample balanced across formats** (`engine_sample: balanced`): 300 games split
  evenly between bullet, blitz and rapid, most recent in each, so the engine sections are not
  all about your bullet. Choose `recent` for your latest games whatever the format.
* **A view per format**: bullet, blitz and rapid each get their own findings and study plan
  when you have enough games in them.
* **Coaching** (`coach`, on by default; `coach_depth` 20, `coach_max` 150 positions): the
  explained mistakes, puzzle drills (`<username>-drill-<theme>.pgn` next to the report, from
  the Lichess puzzle database, downloaded once a month) and the review schedule. The previous
  report on the `reports` branch is used for your progress and for the puzzles due for review.
* **Two optional secrets** (Settings → Secrets and variables → Actions → New repository
  secret). Neither is needed; without them the report says what it skipped:
  * `LICHESS_TOKEN`: a [Lichess personal access token](https://lichess.org/account/oauth/token)
    (no permissions needed) for the opening explorer: what masters and players a little
    stronger than you play in the positions you get wrong.
  * `ANTHROPIC_API_KEY`: lets Claude rewrite the explanations in plainer words and draft a
    weekly plan. Every move and number it writes is checked against the engine's lines and the
    report; anything that doesn't check out keeps the built-in wording.

## Quick start

Requires Python 3.10+.

**Get the code:**

```bash
git clone https://github.com/TimBuckTwo-23/Chess-Analysis
cd Chess-Analysis
```

**Install and run:**

```bash
# Windows (use `py`); on macOS/Linux use python3
py -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
python -m pip install -U pip       # Python 3.10's own pip is too old for `pip install -e .`
pip install -e .

chess-insights report YOUR_USERNAME --tz America/New_York --contact you@example.com
```

Use your own [time zone name](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
for `--tz` (`Europe/London`, `Asia/Kolkata` ...): it puts the time-of-day results in your
local time and turns on the late-night check. If you really live in UTC, use `Etc/UTC`.

In **PowerShell**, activation may fail with "running scripts is disabled on this system".
Either run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, use `cmd.exe`, or skip
activation and call the programs in the environment directly:
`.venv\Scripts\chess-insights report ...` (and `.venv\Scripts\python -m pip ...`).

This downloads your archive into `~/.chess-insights-cache/` (on Windows
`%USERPROFILE%\.chess-insights-cache`; set `CHESS_INSIGHTS_CACHE` or `--cache-dir` to
change it). Later runs only re-check the current month. The report is written to
`reports/<username>.html` (plus `.md` and `.json`); the full paths are printed at the end.
Open the HTML file in a browser (on Windows: `start reports\<username>.html`).

**On your phone:** send yourself the single `.html` file (by email, OneDrive, Google
Drive ...) and open it there. It needs no other files; the fonts load from Google Fonts
when you are online and fall back to the phone's own fonts otherwise.

**What to expect from a first run.** With fewer than about 600 games, expect few or no
strengths and weaknesses for openings and colour: the report only names what the games
clearly show (docs/METHODOLOGY.md, section 2). Habits (after a loss, late at night) and
clock problems show up much sooner. The sections, the positions you keep getting wrong
and your own puzzles (with `--engine`) are useful from the start. Each format you play
60 times or more also gets its own view inside the report; to analyse only some formats,
use e.g. `--time-class blitz,rapid`.

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
`CHESS_INSIGHTS_STOCKFISH`, on `PATH` and in the usual install, Downloads and Desktop folders.
Add `--puzzles` to also get your own mistakes as a PGN puzzle file, written next to the
report (`reports/<username>-puzzles.pgn`); the report lists the ten costliest with a link
that opens each position on a Lichess analysis board, and the study plan uses them.

By default this analyses your 150 most recent games at depth 12, running one Stockfish
per CPU core except one (`--workers N` to change that). Stockfish evaluates every
position once, about 65 positions for a typical game, so 150 games are roughly 10,000
positions. If your most recent games are nearly all bullet, add `--engine-sample balanced`
to split the sample evenly between bullet, blitz and rapid.

Add `--coach` to have the report explain your costliest and repeated mistakes and give you
puzzle drills for the tactics you miss. It re-searches only those positions (at most 150,
depth 20) and caches them, so it adds a few minutes the first time. The drills come from the
Lichess puzzle database: download it once with

```bash
chess-insights puzzles-db
```

(it keeps a filtered subset in the cache folder; without it the drills link to Lichess's
puzzle themes instead). With a report in hand you can ask follow-up questions, answered from
that report and checked against it (needs `ANTHROPIC_API_KEY` and `pip install -e ".[llm]"`):

```bash
chess-insights ask YOUR_USERNAME "why do I lose with the Alapin?"
```

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
1.5 times slower. `--engine-games all` analyses every game (an overnight job for a few
thousand games).

### Useful options

| Option | Meaning |
|---|---|
| `--time-class blitz,rapid` | Only these time controls (bullet, blitz, rapid, daily) |
| `--since 2025-01 --until 2025-12` | Only this period; also `--since 2024` or `--since 2025-03-15` |
| `--rated-only` | Skip casual games |
| `--rules all` | Include variants (default: standard chess only; or e.g. `--rules chess,chess960`) |
| `--tz America/New_York` | Your time zone (recommended): local time-of-day stats, the late-night check and the `--since`/`--until` day boundaries. Without it, times are shown in UTC and the late-night check is off. `--tz UTC` / `GMT` count as not given: use `Etc/UTC` if you really live in UTC, `Europe/London` for the UK |
| `--engine-games 400` / `all` | How many games Stockfish analyses (default 150) |
| `--engine-sample balanced` | Which games Stockfish analyses: `recent` (default: your latest, whatever the format) or `balanced` (the same number of bullet, blitz and rapid games, most recent in each; daily only with `--engine-time-class`) |
| `--engine-time-class blitz,rapid` | Send only these formats to Stockfish; the rest of the report still uses every game |
| `--coach` | With `--engine`: explain your costliest and repeated mistakes, puzzle drills for the tactics you miss, a review schedule |
| `--coach-depth 20` | Stockfish depth for the positions the coaching explains (default 20) |
| `--coach-max 150` | Explain at most this many positions, costliest first (default 150) |
| `--no-motif-profile` | Skip the tactic profile over every error (you vs your opponents; saves a quick engine pass) |
| `--lichess-token TOKEN` | Lichess personal access token for the opening explorer (default: the `LICHESS_TOKEN` environment variable; never printed) |
| `--puzzle-db PATH` | The puzzle file for the drills (default: the subset `chess-insights puzzles-db` keeps in the cache folder) |
| `--drill-rating 1200-1600` | Puzzle rating window for the drills (default 1200-1600) |
| `--practice-minutes 20` | Minutes of practice a day the plan is sized for (default 20) |
| `--coach-llm` | Claude rewrites the explanations and drafts a weekly plan, every move and number checked (needs `ANTHROPIC_API_KEY`) |
| `--llm-model claude-opus-5-5` | The Claude model for `--coach-llm` |
| `--maia` | How findable the better move was at your level (needs `pip install maia2`) |
| `--previous reports/old.json` | The previous report, for progress against its targets and the puzzles due for review (default: the JSON report already at the output path, read before it is replaced) |
| `--offline` | Use the local cache only (and, with `--coach`, no Lichess or Wikibooks requests) |
| `--pgn games.pgn` | Analyse PGN files, a folder of them or a wildcard (`C:\games\*.pgn`) instead of the API; PGNs from other sites or over-the-board games work too |
| `--json archive.json` | Analyse saved chess.com API responses (files, a folder or a wildcard) |
| `--out reports/me --formats html,md` | Output location and formats: writes `reports/me.html` and `reports/me.md`; an existing folder (or a path ending in a slash) gets `<username>.html` … inside it |
| `-v` | Debug logging |

`chess-insights report --help` lists everything. Exit codes: `0` done, `1` no games to
analyse (or the report or the game cache could not be written), `2` a usage mistake (unknown option value,
unknown player, missing file or Stockfish), `3` chess.com (or, for `puzzles-db`, the Lichess database)
could not be reached.

### Try it without an account

```bash
chess-insights demo                                  # about a minute
chess-insights demo --engine --engine-games 400      # the full check: about 6 minutes on 4 cores
```

This generates 400 games for a fictional player with planted strengths and weaknesses
(a leaky Caro-Kann, time trouble in blitz, tilt after losses, weak endgames…), analyses
them, writes `reports/demo.html` (marked as a synthetic player; its game links are not
clickable, since they would point at other people's games) and prints which planted
traits it found: in the report's lists, only in a section, missed, or not tested. Two of
the eight (weak endgames, letting winning positions slip) come from Stockfish's analysis,
so without `--engine` they are "not tested". With `--engine --engine-games 400` it finds 6
of the 8; the other two (the Italian as a strength, rapid better than blitz) are too small
to prove at 400 games. It also lists the claims that match no planted trait.

## Troubleshooting

* **First run against chess.com.** The downloader was built against recorded chess.com
  responses (1,285 real games); it has not yet been run live from a home connection. If
  anything looks wrong, run again with `-v` and keep the output.
* **HTTP 403 from chess.com.** chess.com's CDN rejects requests that don't identify
  themselves. Add `--contact you@example.com` (it goes into the User-Agent). If it
  persists, your network is blocking the API (see the next point).
* **"could not reach api.chess.com" / certificate errors.** Some school and company
  networks block `api.chess.com` or inspect HTTPS traffic. Try another network, or point
  Python at your proxy (`set HTTPS_PROXY=http://proxy:port`; in PowerShell
  `$env:HTTPS_PROXY="http://proxy:port"`) and at your company's root certificate
  (`set REQUESTS_CA_BUNDLE=C:\path\to\company-ca.pem`; PowerShell `$env:REQUESTS_CA_BUNDLE="..."`). If the games were
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
* **Unknown time zone on Windows.** Windows has no time zone database of its own; the
  install already adds the `tzdata` package for that. If it still fails (an old install),
  `pip install tzdata`.
* **`chess-insights` is not recognized.** Activate the virtual environment, or run
  `.venv\Scripts\chess-insights report ...` or `.venv\Scripts\python -m chess_insights report ...`
  (a plain `py -m chess_insights` uses the global Python, which does not have the package).
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
