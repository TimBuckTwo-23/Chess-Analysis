# chess-insights

Download **every game you have played on chess.com** and turn them into a report
of your real strengths, your weaknesses, and a prioritised study plan.

It uses the free, public [chess.com Published-Data API](https://www.chess.com/news/view/published-data-api)
(no login, no API key), so it only needs your username.

What you get:

* **Results & rating**: how you score against the *Elo expectation* per time control
  and colour, and against weaker or stronger opponents.
* **Openings**: which openings gain or lose you points, as White and as Black,
  your choices at the key moves of your main lines (drawn as boards), and links to the
  games worth reviewing.
* **Clock**: time-trouble frequency, games lost on time, and whether you burn
  time in the opening compared with your opponents.
* **How games end**: how often you get mated, flag or resign, and whether you
  do better in short or long games.
* **Habits & tilt**: results right after a loss, deep into a session, and by
  time of day.
* **Development & king safety**: how often you have castled by move 10, how often you bring
  the queen out in the first 8 moves (other than to capture), how many knights and bishops you
  have out by move 10 and how many pawn moves you make in your first 10, each against your
  opponent in the same games. No engine needed.
* **Engine review** (optional, with Stockfish): Lichess-style accuracy, blunder
  rates by game phase, blunders under time pressure, converting winning
  positions, missed tactics, and how you come out of each opening.
* **Positions you keep getting wrong** (with Stockfish): the exact positions
  where you have played the same bad move in several games, with the engine's
  move drawn on the board.
* **Personal puzzles** (`--puzzles`): your costliest mistakes exported as a PGN.
  You can import it into a Lichess study.
* **Bullet, blitz and rapid apart**: every finding says which formats it rests on
  ("blitz only", "bullet, blitz and rapid"), and the report has a separate view for each
  format you play often enough (60+ games), with its own findings, sections and study plan.
* **Coaching** (`--coach`, with Stockfish): for your costliest mistakes, your repeated ones and
  your usual move at the choice points of your main lines (up to 150 positions), the engine's
  better line and the line that refutes your move, drawn as boards; the tactic behind it (fork,
  pin, back-rank mate ...) when a tested detector finds it; what the move changes (king safety,
  piece activity, pawn structure ...); two or three plain sentences on why; and a pointer to
  the chapter of a classic (Capablanca, Lasker) on the main idea. Opening names, Lichess cloud
  evaluations, tablebase results and short credited Wikibooks extracts are added where they
  exist. Puzzle packs from the Lichess puzzle database for the patterns you miss most, a review
  schedule, and, with a previous report, your progress against its targets. The coaching
  explains and routes to practice. The only claims it can add ("you miss forks more often than
  your opponents") go through the same claim rule as every other finding.

Every insight is compared with a benchmark: what your rating predicts, your other
games, or your opponents in the same games. Findings that could be chance are
held back. On simulated players with no real strengths or weaknesses, a whole report
(its lists and every section) contains on average about 0.2 false strengths or
weaknesses (0.23 at 600 games), and fewer than 0.3 in every setting tested
(`tests/test_null_calibration.py`). Each format's view is tested as a report of its own, so
it has its own budget: across all the views of one report, about 0.5 more
([docs/METHODOLOGY.md](docs/METHODOLOGY.md), section 2).

The report is built to be read on a phone:

1. **Three lines at the top**: your ratings, the first three things to work on, and one
   strength to build on.
2. **Every weakness and strength, one line each**, most important first, each linked to
   its explanation and labelled with the formats it rests on. Nothing is hidden behind a
   "top 5".
3. **A study plan**: related findings share one item (a slow opening and losing on time
   are one clock problem), easiest changes first (when you play costs nothing; a new
   repertoire takes weeks), at most three actions per item with tick boxes the page
   remembers, a target for your next report ("use at most 30% of your clock on your
   first 15 moves (now 50%)"), and labelled example games ("Loss · 5+0 · vs 1512 · 12 Aug").
4. **One section per topic**: a short summary, key numbers and findings; the charts and
   tables fold away under "Show …", and wide tables show their key columns first.
5. **A picture for every finding**: a small chart of the numbers behind it (you against
   the rating's expectation or against your opponents in the same games, split by format
   where there are enough games), or the board it is about. When a finding's numbers fit
   no chart, it keeps its text alone rather than get a chart that says something else.
6. **A board for every position**: your side at the bottom, your move as a red arrow, the
   better move green, a threat orange, and a line of play as a strip of small boards.
   Boards link to the game they come from.
7. **A view per format**: when your games cover two or more formats, a switcher at the top
   shows "All formats" and one view for each format with 60 or more games (Bullet, Blitz,
   Rapid, Daily): the same report run on that format's games alone.
8. **Why these moves go wrong** (with `--coach`): one card per explained position (ten open,
   the rest folded away) with the board, both engine lines, the pattern, what the move changes
   and where each fact comes from; the patterns in your mistakes against your opponents'; where
   you leave opening theory; and endgames checked with the tablebase. A finding with an
   explanation links to it.
9. **Practice** (with `--coach`): progress since your last report, positions due for review,
   the puzzle packs, a weekly plan (with the optional AI coach), and the positions coming up
   for review 1, 3, 7 and 21 days after the report.

## Run it on GitHub (no install, works from a phone)

1. Open this repository on github.com, go to **Actions → report → Run workflow**.
2. Enter your chess.com username. Your time zone (e.g. `America/New_York`) is optional
   but recommended. Tap **Run workflow**.
3. When the run finishes, open it. Under **Artifacts**, download the report and open the
   `.html` file. Every report is also saved on the `reports` branch, under
   `<username>/latest/` and a dated folder (`<username>/2026-09-26/`), with the puzzle file
   and the drill packs, so you can compare reports over time.

Or list players in [`players.txt`](players.txt): one username per line, optionally followed
by a time zone (`magnuscarlsen Europe/Oslo`); `#` starts a comment. Saving a change to that
file on `main` runs a report for everyone listed, one after another, with the defaults below.
You can edit it in the GitHub app too.

GitHub's machines download your games and run Stockfish (16, from Ubuntu's package), and they
keep what they already downloaded and analysed: the games, the engine results, the coaching's
lines and the answers from Lichess and Wikibooks. A re-run a week later only fetches and
analyses the new games.

What a GitHub run includes by default (the options on the Run workflow form):

* **An engine sample balanced across formats** (`engine_sample: balanced`, `engine_games` 300,
  `depth` 12): 300 games split evenly between bullet, blitz and rapid, the most recent in each,
  so the engine sections are not all about your bullet. Choose `recent` for your latest games
  whatever the format. The run also writes your puzzle file (`--puzzles`).
* **A view per format**: bullet, blitz and rapid each get their own findings and study plan
  when you have 60 or more games in them.
* **Coaching** (`coach` on; `coach_depth` 20, `coach_max` 150 positions): the explained
  mistakes, the review schedule and the puzzle packs (`<username>-drill-<theme>.pgn` next to
  the report). The previous report on the `reports` branch (`<username>/latest/<username>.json`)
  is used for your progress and for the positions due for review.
* **The Lichess puzzle database, cached by month**: the drills need a filtered copy of it. The
  run restores it from a cache keyed by the month and downloads it only when this month's copy
  is not cached yet (up to 30 minutes are allowed for that). If the download fails, last
  month's copy is used when there is one, else the drills link to Lichess's puzzle themes
  instead. It never stops the report.
* **Two optional secrets** (Settings → Secrets and variables → Actions → New repository
  secret). They reach the program through the environment only, never the command line.
  Neither is needed; without them the report says what it skipped:
  * `LICHESS_TOKEN`: a [Lichess personal access token](https://lichess.org/account/oauth/token)
    (no permissions needed). It unlocks the Lichess opening explorer, which has answered only
    with a token since 2026: which moves masters play, and which moves Lichess players one or
    two rating groups above you play, in your explained opening positions and at your choice
    points, with where your own move ranks. Everything else from Lichess (cloud evaluations,
    the tablebase) and Wikibooks needs no token.
  * `ANTHROPIC_API_KEY`: the run adds `--coach-llm`. Claude rewrites up to 20 explanations in
    plainer words and drafts a weekly plan from the report's facts. Every move, number and
    claim it writes is checked against the engine's lines and the report; a text that fails
    keeps the built-in wording. The workflow does not yet install the Anthropic SDK (the `llm`
    extra), so for now the key has no effect and the report notes that the AI coach was
    skipped (see [PLAN.md](PLAN.md)).

How long a GitHub run takes, measured on the reports of a 3,354-game history on 26 September
2026, before coaching existed: the first run, with the whole archive to download and 300 games
at depth 12, spent 4¼ minutes downloading and analysing (6 minutes 20 seconds for the whole
run); a run with 1,000 games at depth 15 spent 32 minutes. Coaching adds its own searches and
requests (see [How long it takes](#how-long-it-takes)); a run is stopped after 120 minutes.

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
clearly show (docs/METHODOLOGY.md, section 2). Habits (after a loss, late at night), clock
problems and opening habits such as castling late show up much sooner. The sections, the
positions you keep getting wrong and your own puzzles (with `--engine`) are useful from the
start. Each format you play 60 times or more also gets its own view inside the report; to
analyse only some formats, use e.g. `--time-class blitz,rapid`.

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
report (`reports/<username>-puzzles.pgn`, up to 300 puzzles, costliest first); the report
lists the ten costliest with a link that opens each position on a Lichess analysis board,
and the study plan uses them.

By default this analyses your 150 most recent games at depth 12, running one Stockfish
per CPU core except one (`--workers N` to change that). If your most recent games are nearly
all bullet, add `--engine-sample balanced` to split the sample evenly between bullet, blitz
and rapid (the GitHub default).

### Coaching

Add `--coach` (with `--engine`) to have the report explain your costliest and repeated
mistakes and give you puzzle packs for the patterns you miss:

```bash
chess-insights puzzles-db                          # once: the Lichess puzzle database for the drills
chess-insights report YOUR_USERNAME --engine --coach --tz America/New_York
```

`puzzles-db` streams the Lichess puzzle database (CC0, a few hundred MB compressed) once and
keeps a filtered subset in the cache folder (puzzles rated 800–2200, well liked and often
played); without it the drills link to Lichess's puzzle themes instead. The packs come from
a fixed rating window, 1200–1600 by default: set `--drill-rating` to your level.

Stockfish 16 is the version to use for the coaching: it is the last one that reports its
evaluation term by term (king safety, mobility, pawns ...), which the explanations compare.
With a newer Stockfish the explanations use board facts only (the bishop pair, castling, a
king left in the centre, pawn weaknesses) and say so.

With a report in hand you can ask follow-up questions, answered from that report's JSON and
checked against it:

```bash
pip install -e ".[llm]"                            # the Anthropic SDK
chess-insights ask YOUR_USERNAME "why do I lose with the Alapin?"
```

It needs `ANTHROPIC_API_KEY` in the environment; without the key or the SDK, or when the
answer fails the check, it quotes what the report itself says on the subject.

## How long it takes

**Engine analysis.** Stockfish evaluates every position of a game once, about 65 positions
for a typical game, so 150 games are roughly 10,000 positions. Measured with Stockfish 16 at
depth 12 on 6 real chess.com games (715 positions) on a shared 4-core cloud machine:

| Engines (`--workers`) | Positions per second | 150 games |
|---|---|---|
| 1 | 26 | about 6 minutes |
| 3 | 69 | about 2½ minutes |

A recent desktop CPU is usually faster per core, and more cores help almost linearly.
Results are cached per Stockfish version and depth, so re-running the report only
analyses new games (150 cached games load in about a second). Use `--engine-games` and
`--depth` to trade speed for coverage: each extra level of depth makes the analysis about
1.5 times slower. `--engine-games all` analyses every game (an overnight job for a few
thousand games). On GitHub's machines, 1,000 games at depth 15 took 32 minutes.

**Coaching.** These are bounds from the code's limits and from single searches timed with
Stockfish 16 on one thread; a full `--coach` run has not been timed yet:

* The explained positions (at most `--coach-max`, 150) are searched at `--coach-depth` 20
  with three lines, capped at 8 seconds per position plus 4 seconds for the line after your
  move (`--coach-seconds` changes the cap; 0 removes it). One such search takes about 4–7 s
  plus 1.3 s, so 150 positions are about 13–21 minutes of engine time: 3–5 minutes on four
  engines, as on GitHub. Each result is cached, so the next run only searches new positions.
* The pattern profile runs one short search (depth 10, at most 0.5 s) before and after every
  mistake or blunder by either side in the analysed games, about 2,500 positions in a real
  run: about a minute on four engines (`--no-motif-profile` skips it).
* Lichess and Wikibooks are asked one request at a time, at least a second apart per site, and
  at most 60 explorer, 40 cloud-evaluation, 60 Wikibooks and 150 tablebase requests per report:
  a few minutes on the first run. Answers are cached (the explorer and cloud evaluations for
  30 days, Wikibooks for 90, the tablebase for good).
* `chess-insights puzzles-db` downloads a few hundred MB, once.
* The AI coach (`--coach-llm`) is one request to Claude; Maia-2 (`--maia`) runs its network
  once per explained position.

## All options

`chess-insights <command> --help` lists them too. `-v` / `--verbose` (debug logging) works with every
command, and `chess-insights --version` prints the version.

**`report USERNAME`** (`demo` takes the same options from `--time-class` on):

| Option | Meaning |
|---|---|
| `USERNAME` | Your chess.com username, or your profile URL |
| `--cache-dir DIR` | Where downloaded games, engine results, coaching lines and Lichess/Wikibooks answers are kept (default `~/.chess-insights-cache`, or `CHESS_INSIGHTS_CACHE`) |
| `--since 2025-01` | First month to include; also `--since 2024` or `--since 2025-03-15` |
| `--until 2025-12` | Last month to include, in the same forms |
| `--contact you@example.com` | Added to the User-Agent, as chess.com asks API users to do (default: `CHESS_INSIGHTS_CONTACT`) |
| `--offline` | Use the local cache only: no chess.com requests and, with `--coach`, no Lichess or Wikibooks requests and no Maia-2 download (`--coach-llm` still calls Claude) |
| `--pgn games.pgn` | Analyse PGN files, a folder of them or a wildcard (`C:\games\*.pgn`) instead of the API; PGNs from other sites or over-the-board games work too |
| `--json archive.json` | Analyse saved chess.com API responses (files, a folder or a wildcard) |
| `--time-class blitz,rapid` | Only these time controls (bullet, blitz, rapid, daily; repeat the option or separate with commas) |
| `--rules all` | Include variants (default: standard chess only; or e.g. `--rules chess,chess960`) |
| `--rated-only` | Skip casual games |
| `--tz America/New_York` | Your time zone (recommended): local time-of-day results, the late-night check and the `--since`/`--until` day boundaries. Without it, times are in UTC and tested more strictly; `--tz UTC` / `GMT` count as not given (use `Etc/UTC` if you really live in UTC, `Europe/London` for the UK) |
| `--engine` | Run Stockfish: the engine sections, the positions you keep getting wrong, puzzles and coaching |
| `--stockfish PATH` | The Stockfish program or its folder (default: `CHESS_INSIGHTS_STOCKFISH`, `PATH`, the usual install, Downloads and Desktop folders) |
| `--depth 12` | Stockfish depth per position (default 12) |
| `--engine-games 400` / `all` | How many games Stockfish analyses (default 150) |
| `--engine-time-class blitz,rapid` | Send only these formats to Stockfish; the rest of the report still uses every game |
| `--engine-sample balanced` | Which games Stockfish analyses: `recent` (default: your latest, whatever the format) or `balanced` (the same number of bullet, blitz and rapid games, most recent in each; daily only with `--engine-time-class`) |
| `--workers 3` | Stockfish processes in parallel, for the game analysis and the coaching (default: CPUs − 1) |
| `--puzzles` | With `--engine`: write your mistakes as a PGN puzzle file next to the report; with `--coach`, a solution becomes the engine's line, with its pattern and explanation where the coaching has them |
| `--coach` | With `--engine`: explain your costliest and repeated mistakes and your choice points; puzzle packs, a review schedule, progress against the previous report |
| `--coach-depth 20` | Stockfish depth for the positions the coaching explains (default 20) |
| `--coach-max 150` | Explain at most this many positions, costliest first (default 150) |
| `--coach-seconds 8` | Time cap per position for those searches, in seconds; the line after your move gets half (default 8; 0 = no cap) |
| `--no-motif-profile` | Skip the pattern profile over every mistake (you against your opponents) and the pattern findings it can make |
| `--lichess-token TOKEN` | Lichess personal access token for the opening explorer (default: `LICHESS_TOKEN`; never printed) |
| `--puzzle-db PATH` | The puzzle file for the drills, or the folder `puzzles-db --cache-dir` wrote it to (default: the one in the cache folder) |
| `--drill-rating 1200-1600` | Puzzle rating window for the drill packs (default 1200-1600; the downloaded subset holds 800–2200) |
| `--practice-minutes 20` | Minutes of practice a day the weekly plan is sized for (default 20) |
| `--coach-llm` | Claude rewrites the explanations and drafts a weekly plan, every move and number checked (needs `ANTHROPIC_API_KEY` and `pip install -e ".[llm]"`) |
| `--llm-model claude-opus-5-5` | The Claude model for `--coach-llm` (default `claude-opus-5-5`) |
| `--maia` | How often players at your rating find the better move and play yours (Maia-2; needs `pip install maia2`, which brings PyTorch, and downloads its weights on first use) |
| `--previous reports/old.json` | The previous report's JSON, for progress and the positions due for review (default: the JSON report already at the output path, read before it is replaced) |
| `--out reports/me` | Output path stem: writes `reports/me.html` ...; an existing folder (or a path ending in a slash) gets `<username>.html` inside it (default `reports/<username>`) |
| `--formats html,md` | Which files to write: `html`, `md`, `json` (default all three; `ask` and the next report's progress need the JSON) |

**`demo`** (a synthetic player; see below) also takes:

* `--games 400`: number of synthetic games (default 400).
* `--seed 7`: random seed (default 7).
* `--cache-dir DIR`: where to write the synthetic archive (default: a temporary folder, removed afterwards).
* `--no-engine-synth`: generate the games without Stockfish (faster, less realistic).

**`fetch USERNAME`**: `--cache-dir`, `--since`, `--until` and `--contact` as above, and
`--refresh-all` to re-check old months too (picks up new Game Review accuracies).

**`puzzles-db`**: `--cache-dir DIR`, the cache folder to keep the puzzle subset in.

**`ask USERNAME "question"`**:

* `--out reports/me`: the report's path stem, as given to `report --out` (default `reports/<username>`).
* `--cache-dir DIR`: cache folder (default as above).
* `--offline`: no Lichess or Wikibooks requests (the answer still needs Claude).
* `--llm-model MODEL`: the Claude model (default: the coaching's, `claude-opus-5-5`).

Exit codes: `0` done, `1` no games to analyse (or the report or the game cache could not be
written, or `ask` found no report), `2` a usage mistake (unknown option value, unknown player,
missing file or Stockfish), `3` chess.com (or, for `puzzles-db`, the Lichess database) could not
be reached, `130` interrupted with Ctrl+C.

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

* **First run against chess.com.** The downloader has run on GitHub's machines (a
  3,354-game history, 26 September 2026) and was built against recorded chess.com
  responses (1,285 real games); it has not yet been run from a home connection or on
  Windows. If anything looks wrong, run again with `-v` and keep the output.
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
* **The coaching skipped something.** Every coaching step that cannot run (no Stockfish 16
  for the evaluation terms, no puzzle database, no Lichess token, a service that did not
  answer, no API key) leaves one note under "Why these moves go wrong" and in the console
  summary; the rest of the report is unaffected.
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

`fetch` → `parse` → `filter` → optional `engine` → analysis modules → optional coaching
(`--coach`) → formats and a picture for every finding → ranked insights and study plan →
the coaching steps that need the finished report (progress, Maia-2, the AI coach) → the same
analysis per format → HTML, Markdown and JSON report. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map,
[docs/METHODOLOGY.md](docs/METHODOLOGY.md) for how every number is computed,
[PLAN.md](PLAN.md) for the roadmap and what has not been run for real yet, and
[src/chess_insights/data/NOTICE.md](src/chess_insights/data/NOTICE.md) for the data sources
and their licences.

## Development

```bash
pip install -e ".[dev]"
py -m pytest            # engine tests are skipped automatically if Stockfish isn't found
```

`python scripts/motif_precision.py` prints how well the pattern detectors agree with
Lichess's puzzle themes (docs/METHODOLOGY.md, section 6).
