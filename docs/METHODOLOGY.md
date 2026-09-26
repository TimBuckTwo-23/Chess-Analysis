# Methodology

How chess-insights turns a pile of games into claims about your play, and why you
can trust them or not.

## 1. The benchmark: expected score, not win rate

A 55% win rate means nothing on its own. It depends on who you played. Every
result-based metric compares your score with the **Elo expected score** for each
game:

```
E = 1 / (1 + 10 ^ ((opponent_rating − your_rating) / 400))
```

The metric is the average of `score − E` over a group of games (win = 1,
draw = ½, loss = 0). "−0.10 points per game" means you score 10 percentage points
below what your ratings predict in that group. Near 50% that is about 70 Elo.

**Pre-game ratings.** chess.com reports each player's rating *after* the game, which
would slightly reward the winner in hindsight. Your pre-game rating is your
post-game rating from the previous rated game in the same pool (rules + time
control). Your opponent's is estimated by assuming a symmetric rating change.

**Rating noise.** Ratings are noisy estimates of strength, so actual results
regress toward 50%. *Everyone* appears to underperform against lower-rated players
and to overperform against higher-rated ones. How strongly is uncertain: the
attenuated expectation `0.5 + a × (E − 0.5)` fits real results for some `a`
between about 0.75 (a pessimistic guess) and 1.0 (plain Elo); a realistic 75-Elo
noise on the rating difference gives about 0.94. The opponent-strength section
shows the bucket numbers against `a = 0.75`, but only makes a claim that holds for
*every* `a` in that range (see section 2).

**Rating lag.** A rating trails your current strength: an improving player scores
above expectation in almost every game, a declining one below. That is one fact
about the rating, not a strength or weakness of any particular kind of game. So a
claim about a *subset* of your games (an opening, a time of day, opponents of some
strength, a time control) always compares that subset with your *other* games; the
rating's expectation alone is only used for the tables and charts.

**Colour.** At equal ratings White scores about 52% and Black about 48% in club
blitz and rapid, a White-minus-Black gap of about 0.04 points per game. The exact
edge is uncertain (about 0.03 in bullet and at lower ratings, 0.05 or more in
slower games and for strong players), so a colour claim must hold for
every edge from 0.03 to 0.05: "worse with Black" against 0.05, "worse with White"
against 0.03. Openings are compared only with your other openings *of the same
colour*, so neither the edge nor a colour-wide gap turns into opening claims.

## 2. Separating signal from noise

A few hundred games carry much less information than it seems. The standard error
of an average `score − E` is roughly:

| games in the group | ± (1 standard error) | ≈ Elo |
|---|---|---|
| 20 | 0.105 | 73 |
| 50 | 0.066 | 46 |
| 100 | 0.047 | 33 |
| 400 | 0.023 | 16 |

So a 20-game opening needs a ~150 Elo effect before anything can be said, and a
comparison of two groups (an opening against your other openings) is noisier than
either group alone.

### The claim rule

Something becomes a **strength or weakness** only if all three hold; otherwise it
is at most an *observation* (shown in its section, marked "i", never in the lists at
the top or the study plan):

1. **Enough games** in *both* groups compared (module minimums, e.g. 8 rated games
   in an opening family as one colour, 25 in a split such as "right after a
   loss").
2. **A significant test after multiple-testing adjustment.** Difference tests of the
   mean `score − E` between a group and your other games, paired tests against your
   opponents in the same games, (differences of) two-proportion tests for rates.
   When a test is repeated over a family of groups (every opening, every time class,
   every block of the day) the p-values are Benjamini–Hochberg adjusted, and the
   adjusted p must be at or below the family's α (next section).
3. **An effect big enough to matter** (e.g. 0.06 points per game, about 40 Elo).

This rule lives in one place, `stats.significance`, and every module uses it.

The lists at the top of the report show **every** strength and weakness that passed it
(most important first: severity × confidence), one line each; every one is explained in
its section. Findings from two modules about the same opening are merged into one line.

**Confidence** is *not* `1 − p`. A p-value of 0.004 does not make a claim 99.6%
certain, least of all in a report that runs a few dozen tests. The confidence is
the most the evidence can support if a real effect and no effect were equally
likely beforehand (the Sellke–Bayarri–Berger bound, `1 / (1 − e·p·ln p)` for the
adjusted p): p = 0.05 gives 0.71, p = 0.01 gives 0.89, p = 0.001 gives 0.98. It is
scaled down when the smaller group compared has fewer than twice the minimum
number of games, so every claim has a confidence of at least 0.50 and "high
confidence" (≥ 0.7) needs a clear result on a decent sample. **Severity** uses the
effect shrunk toward 0 by its own uncertainty (a normal prior with a standard
deviation of 0.10 points per game): effects that just clear a significance bar are
overestimates on average, most of all on small samples. Small groups are also
**shrunk toward 0** (`stats.shrink`) before ranking or charting, so a 3–0 opening
doesn't top the "strengths" list.

### The error budget: two tiers

A report runs about fifteen families of tests. If a family's claims can go either
way, pure noise makes it claim something with probability about α; if it only ever
claims one direction ("you score *worse* after a loss") about α/2, or α with a
one-sided test. The expected number of false claims per report is roughly the sum,
so the families are split into two tiers:

| α | Claims | Why |
|---|---|---|
| **0.05** | White vs Black; opening families; scoring worse straight after a loss; scoring worse late at night (23:00–03:00 in your time zone) | The claims players most need, and the hardest to detect: effects of 5–15 points per 100 games that only part of the games carry. The last two are fixed in advance with their direction, so they use a one-sided test. |
| **0.01** | Other times of day, time controls, opponent strength, session length, time trouble, losing on time, slow openings, clock handling, quick losses in one opening | Exploratory scans (many ways to get lucky), and clock or quick-loss habits, whose paired or mirror tests have power to spare when the habit is real. |

**Late night needs your time zone.** 23:00–03:00 UTC is early evening in New York.
Without a time zone (no `--tz`, or `--tz UTC` / `GMT`, which count as not given) the same
UTC window is still tested, but at 0.01 like any other block of the day, and it is titled by its
UTC hours ("games started 23:00–03:00 UTC"); the 4-hour blocks are too. Pass your
zone (`--tz America/New_York`, or `--tz Etc/UTC` if you really live in UTC) to get
the late-night test.

### Observations that are never claims

Two patterns follow from the result and from *both* players' resignation habits as
much as from skill, so from results alone they can't be told apart from a habit,
and they are only ever observations (and only when their test passes at 0.01):

* **How losses end.** Being mated in a larger share of your losses than you mate
  in your wins can point at king safety, but it is exactly what never resigning
  looks like: a resignation becomes a mate.
* **Game length.** A player who plays on in lost positions loses more long games; one
  who resigns early loses more short ones; opponents' habits do the same to wins.
  Each length band is compared with your games of other lengths.

The engine analysis (section 3) measures the skills behind them directly
(conversion, endgame accuracy, missed mates).

### One fact, one finding

Several tests can pick up the same thing. The report names it once:

* **A colour gap that one opening accounts for.** If the openings module names an
  opening (a weak Caro-Kann) and the White–Black gap is within the normal range
  without those games, the colour result becomes an observation pointing to the
  opening.
* **An outlier opening.** A weak Caro-Kann lowers "your other Black openings" for
  every other Black family, so an ordinary French can look like a strength. Opening
  claims are accepted strongest first, and each further one must also hold against
  your other openings without the ones already named, and stand out from your
  *typical* opening of that colour (the median), not just from their average.
* **Time trouble and losing on time.** Every flag is time trouble. When losing on
  time is a finding, time trouble is a second one only if it still holds with the
  games you lost on time left out. The same clock habit in several time controls is
  one finding ("… in blitz and rapid").
* **Long sessions and playing on after a loss.** Later games in a session follow a
  loss more often (a session's first game never does). When "after a loss" is a
  finding, "you score worse from the 4th game of a session on" must also hold without
  the games played straight after a loss.
* **Two time controls** are one comparison seen from both ends: at most one
  time-control finding, for the pool further from its own rating. It is worded as
  what it measures ("you outperform your rapid rating more than your other
  ratings"): each time control has its own rating, so a pool whose rating lags your
  recent results looks exactly the same.

### Benchmarks chosen to avoid false claims

* **Self-relative comparisons** everywhere. Clock usage, blunder rates and
  conversion are compared with *your opponents in the same games* (rating-matched
  peers on the same clock and position). Openings are compared with your other
  openings of the same colour; "after a loss", times of day, session length,
  opponent strength and time controls with the rest of your games.
* **Openings**: the family tests are *weighted* Benjamini–Hochberg adjusted with
  weights = games played: a leak in the opening you play in 40% of your games costs
  far more points than one in a sideline, so it gets most of the error budget (the
  family-wise error rate is unchanged). If one opening is almost all of your games
  with a colour, it can't be told apart from that colour and is not claimed.
* **Quick losses in one opening** compare how often your losses in it are over
  within 25 moves with how often your *wins* in it are (a sharp opening is short for
  both sides), and then with the same gap in your other openings (a player who
  resigns early has quick losses in every opening): a difference of differences.
* **Opponent strength** must be significant, in the same direction, against both
  the attenuated (`a = 0.75`) and the plain Elo (`a = 1.0`) expectation. With
  realistic rating noise, `a = 0.75` alone manufactures "you beat lower-rated
  players" and plain Elo alone "you drop points against lower-rated players".
* **Time controls.** Each rating pool is judged against its own rating, so each
  pool's score-vs-expected is near zero by construction; picking the best and the
  worst of several noisy numbers finds a "gap" in pure noise. Each time control is
  instead compared with all your other time controls together, BH-adjusted.
* **Late night** is one pre-specified window, 23:00–03:00 in your time zone (a
  4-hour block starting at midnight would cut it in two). If an overlapping 4-hour
  block is also flagged, the report keeps the one finding with the stronger evidence.
* **The rating trend** is always an observation: a rating is a random walk around
  your strength, and a 90-day swing of 100+ points happens by chance.

### Calibration: how often the report is wrong, and what it finds

`tests/null_world.py` generates a *null world*: games with realistic rating noise
(75 Elo on the true strength difference) and White's first-move edge (0.04), but
where openings, terminations, clocks, schedules and game lengths have nothing to do
with results. Every strength or weakness reported on it is false. The same
generator can *plant* realistic effects, each shifting only the games it concerns,
and habits that are not skill effects (a rating that lags, never resigning).

False claims per report (the null player gives the time zone, `Etc/UTC`, so late night is
tested at 0.05; seeds 0–63 and 1000–1039 for the null world, 24–64 fresh seeds for each
variant). These were measured on the report's top lists when they held at most five
claims; the test now counts every strength and weakness in the lists and in every
section, and re-measuring the 400-game null world that way (seeds 0–63) gives the same
0.19 per report, because no null report had more than two claims:

| World | 400 games | 600 | 1,200 | 3,000 |
|---|---|---|---|---|
| Null world (104 / 64 / 40 / 40 reports) | **0.18** | **0.23** | **0.10** | **0.00** |
| … without a time zone (late night at 0.01) | 0.12 | 0.19 | 0.05 | |
| Every game 5 points per 100 above the rating (a lagging rating) | | 0.25 | 0.04 | 0.00 |
| Every game 5 points per 100 below it | | 0.27 | 0.17 | 0.08 |
| Never resigns: 80% of resignations played on to mate | 0.27 | | 0.10 | |
| … 35% played on | 0.13 | | 0.04 | |
| 80% bullet / 80% rapid | 0.20 / 0.28 | | 0.10 / 0.10 | |
| Rating noise 110 Elo / none | 0.30 / 0.10 | | 0.18 / 0.08 | |
| True strength fixed, rating wanders | 0.15 | | 0.05 | |
| White's edge 0.065 instead of 0.04 (16 reports) | | | 0.38 | 0.19 |

No report had more than 2 false claims. Before this calibration the null world got
4.95 per report at 400 games (up to 10); the first calibration brought that to 0.2,
but a lagging rating still produced 0.7–1.5 per report at 600–1,200 games (3.2 at
3,000, mostly opening "strengths" or "weaknesses"), and never resigning 3.5–5.1 (a
king-safety "weakness" and game-length claims in every report). The last row is
outside the assumed range of White's edge: the extra claims are all "you score worse
with Black", so a player rated 2000+, whose edge is larger, should read that claim
with care. The null world's realised edge is slightly below 0.04 (colour z-scores
average −0.1), so "worse with White" shows up a little more often than "worse with
Black" in it.

Share of 32 worlds (seeds 1000–1031) in which the report's weaknesses name the
planted effect (right kind and category), one effect at a time:

| Planted effect | 400 | 600 | 1,200 | priced in, 400 / 600 |
|---|---|---|---|---|
| Caro-Kann as Black −0.15 (40% of Black games) | 22% | 47% | 97% | 31% / 62% |
| Right after a loss −0.12 | 88% | 88% | 100% | 75% / 84% |
| Late night (23:00–03:00, ~29% of games) −0.10 | 75% | 84% | 94% | 50% / 81% |
| Time trouble in 35% more games, −0.10 in them | 94% | 100% | 100% | 100% / 100% |
| 25% of mate/resignation losses become losses on time | 81% | 88% | 100% | |
| 40% of Italian Game losses (its main White opening) over within 22 moves, score unchanged | 25% | 59% | 100% | |
| Black −0.08 | 28% | 50% | 62% (84% at 2,000) | 41% / 47% |

*Priced in*: the rating has absorbed the effect, so every game is lifted by
`share × effect` (the realistic case for a long-standing habit; the gap to the other
games is the same, and the differences from the plain column are mostly sampling
noise). Several effects at once are found as well: in 1,200-game worlds with tilt,
late night, time trouble and flagging together, each was named in 91–100% of 32
worlds, and with the weak Caro-Kann and the Italian quick losses together in 91% and
100%, with about 0.1 other claims per report.

What this means for you:

* **With 1,200+ games** the report finds effects of these sizes almost always (a
  colour imbalance of 8 points per 100 games needs about 2,000).
* **With 400–600 games** it finds tilt, late night and clock habits most of the time,
  but an opening effect only a quarter to a half of the time: an opening is now
  compared with *your other openings of that colour*, which is what makes the claim
  immune to a lagging rating and to colour effects, and that comparison is noisier
  than a comparison with the rating's expectation (which found the Caro-Kann in 59% /
  81% of worlds, at the price of turning every rating lag into opening claims). The
  opening tables and chart still show every opening's score against expectation.
* The report says less rather than guessing: a missed effect costs a finding, a false
  one sends you to study the wrong thing.
* The null world is idealised (each game's true strength equals the current rating,
  families of tests are roughly independent). The rows above that break those
  assumptions (a lagging or wandering rating, resignation habits, other time-control
  mixes and noise levels) stay at or below 0.3 false claims per report; a White edge
  outside 0.03–0.05 is the one assumption that shows.

Run `CALIBRATION_RUNS=64 CALIBRATION_GAMES=600 pytest tests/test_null_calibration.py`
to re-measure the false-claim rate after changing a threshold, and
`POWER_RUNS=32 pytest tests/test_power.py` to re-measure detection.

These are associations, not causes. "You score worse after 11 pm" might be
tiredness, or it might be the different player pool at that hour. The tool
controls for opponent rating, not for everything.

### Per-format views: each carries its own budget

When your games cover two or more formats, every format with at least 60 games
(`pipeline.MIN_FORMAT_GAMES`) also gets a view of its own: the same analysis run on that
format's games alone, with its own strengths, weaknesses and study plan. Below 60 games most
tests could not say anything (they need 8 games per opening and colour, 25 per split), and the
view would be mostly "not enough data".

A view is a separate report: its claims are tested on that format's games only, with the
same rule, thresholds and adjustments, and they are never added to the main report's lists.
So each view carries its own false-claim budget. On the null world, measured on the views
(24 reports of 1,200 games, 16 of 3,000; seeds 0 upward):

| Null world | Main report | Bullet view | Blitz view | Rapid view | All views together |
|---|---|---|---|---|---|
| 1,200 games (views of about 180 / 730 / 300 games) | 0.17 | 0.08 | 0.17 | 0.29 | 0.54 |
| 3,000 games (views of about 450 / 1,810 / 740 games) | 0.00 | 0.19 | 0.13 | 0.19 | 0.50 |

Each view is about as reliable as a report of its size, but together they are three more
chances to be wrong: across all three views of a report there is about half a false claim on
average, where the main report has about 0.2. Read a finding that appears only in one view
with that in mind. A finding in the main report and in a view is the same fact seen twice,
not two pieces of evidence.

## 3. Engine analysis (optional)

With `--engine`, Stockfish evaluates every position of a sample of your games at a
fixed depth, and the tool applies **Lichess's published formulas** (ported from the
open-source lila/scalachess code) so that numbers are comparable to what you see on
Lichess.

**Which games** (`engine.select_engine_games`; the rest of the report always uses every game):

* `--engine-games N`: how many (default 150; the GitHub workflow uses 300). Only standard
  chess and Chess960 games of at least 10 moves count; a game whose moves don't replay is
  skipped and replaced by the next one.
* `--engine-sample recent` (the default on the command line): the N most recent games. A
  player who has mostly played bullet lately gets engine sections about bullet (295 of the
  300 in the report that prompted this option).
* `--engine-sample balanced` (the default on GitHub): N split evenly between bullet, blitz
  and rapid, the most recent of each; a format with too few games leaves its share to the
  others, and an odd game goes to the slower format. Daily games only when listed with
  `--engine-time-class`.
* `--engine-time-class blitz,rapid`: only these formats.

The report's engine note says what was analysed, e.g. "Stockfish 16 at depth 12 on 300
games: 100 bullet, 100 blitz, 100 rapid (most recent in each)", and each format's view says
how many of its own games were analysed. The engine findings compare you with your opponents
in the same games, so a balanced sample changes what they describe (all your formats instead
of your latest one) but not how they are tested. A balanced sample is less recent in the
formats you play most, so an engine finding describes a longer stretch of your play there.

The formulas:

* **Win %** from centipawns: `50 + 50 · (2 / (1 + e^(−0.00368208 · cp)) − 1)`,
  with cp clamped to ±1000 and mates counted as ±1000.
* **Move accuracy** from the win-% drop Δ:
  `103.1668 · e^(−0.04354 · Δ) − 3.1669 + 1`, bounded 0–100.
* **Game accuracy**: the average of a volatility-weighted mean and a harmonic mean
  of move accuracies.
* **Inaccuracy / mistake / blunder**: the move loses ≥ 5 / 10 / 15 percentage
  points of win chance, computed from the *unclamped* evaluations as Lichess does.
  There are also mate-specific rules (a mate allowed, a mate lost). As on Lichess, a
  move that is the engine's own first choice is never judged, and the mating move is
  left out of game accuracy.
* **Game phases**: Lichess's *Divider*. The middlegame starts when ≤ 10 major/minor
  pieces remain, the back rank empties, or the pieces become mixed. The endgame
  starts when ≤ 6 remain. A move belongs to the phase of the position it creates, so
  the move that trades into an endgame counts as an endgame move.
* **Draws on the board** (threefold repetition, 50-move rule, stalemate,
  insufficient material) are scored as 0.00 without a search, so the move that
  secures a draw is never marked as a mistake.

`tests/test_engine_parity.py` checks these formulas against Lichess's own test
vectors. The few deliberate differences are pinned by `test_deviation_*` tests:
* the start position is evaluated rather than assumed to be +0.15;
* one move's centipawn loss is capped at 1000 (Lichess allows 2000);
* a game that jumps straight to an endgame gets endgame moves;
* accuracy is reported for a side even when the other side never moved;
* a final position drawn by rule (repetition, 50 moves) is scored as a draw without a search.

**How engine findings are tested.** Every engine comparison benchmarks you
against your opponents in the same games. The unit is the *game*: blunders come
in bursts, so counting moves as independent would overstate the evidence. Tests
use cluster-robust (game-level) statistics and the same claim rule and strict alpha
as the rest of the report.
* **A phase or error type** is singled out only when its share of your errors
  differs from your opponents'. A player who errs more everywhere gets one finding,
  not four.
* **Blunders when short of time** are compared within each game phase, because
  time trouble mostly happens in endgames. An endgame weakness is not blamed on the
  clock.
* **Opening outcomes** (the evaluation after move 10) are measured from Stockfish's
  evaluation of the start position, so White's normal first-move edge doesn't count
  as a good opening. Small samples use Student's t.

chess.com's own "accuracy" (CAPS2) is proprietary and uses a different scale. It
appears only for games you ran Game Review on. The report shows it separately and
never mixes it with engine accuracy.

**Repeated mistakes.** The position before each of your engine-flagged errors (a
move losing ≥ 8 points of win chance) is keyed by its EPD: the board, side to move,
castling and en-passant rights. Transpositions are therefore counted together.
* A position where you played the *same* wrong move in two or more games is listed
  in the table with the engine's preferred move. Games you didn't send to the engine
  count too, when they reached the position and repeated the move.
* It becomes a weakness only after at least three such games, and only if that is
  more often than your ordinary error rate explains. That is a one-sided binomial
  test, Benjamini–Hochberg adjusted over the positions tested, at the strict alpha.
* A wrong move you only ever played in games where you had already gone wrong
  earlier in the same line (9.Nxg5, then 11.Kh1 in the same three games) is folded
  into the earlier one: one habit, one finding. Folding never turns an observation
  into a claim or drops a claim.
* `--puzzles` exports your costliest mistakes (up to 300, worst first) as a PGN you
  can import into a Lichess study (64 chapters per study) or any chess GUI. The report
  lists the ten costliest, each with a link to the position on a Lichess analysis board.

## 4. What is excluded

* **Variants** (Chess960, bughouse, odds chess…) are excluded by default. Pass
  `--rules all` to include them.
* **Games that start from a set-up position** (thematic tournaments) are excluded
  from opening statistics.
* **Daily games** are left out of clock and session statistics. Their clocks
  measure days, and archived daily clocks store time spent rather than time
  remaining.
* **Games with fewer than 4 plies** are left out of skill metrics.

## 5. From findings to a study plan

The study plan changes nothing about what is claimed; it only arranges the weaknesses
that passed the rule (section 2):

* **One item per cause.** Weaknesses of the same kind share an item: the clock (a slow
  opening, losing on time, time trouble), when you play (after a loss, late at night,
  long sessions), your own repeated positions, your repertoire (openings and colour), and
  one item each for conversion, game phases, blunders, tactics and accuracy. The item takes
  its title and numbers from its most important finding and lists the others it covers.
* **Easiest first.** Items are ordered by what it takes to act on them, then by
  importance: when you play (costs nothing), the clock (a habit to practise), your own
  positions (ten minutes a day), the repertoire (weeks), technique (longer).
* **At most three actions.** The findings' own actions, taken in turn; general training
  advice only fills an empty place, and never when it repeats an action already there.
* **A target with today's number** ("use at most 30% of your clock on your first 15
  moves (now 50%)"), also stored in the JSON (`study_plan[].baseline`) for comparing
  with the next report.
* **Your own puzzles** (with `--engine`) are always practice material: they join the
  positions item, or make one of their own when no position is a weakness. They are facts
  about your games, not claims.

Some actions quote numbers from your games that were not tested and are not claims:
how each of your choices at a move scored ("after 1.e4 e5 2.Nf3 Nc6: 3.Bc4 66% in 90
games, 3.Bb5 40% in 40"), Stockfish's average evaluation after move 10 in a weak opening
(to tell an opening problem from a middlegame one), your blunder rate late at night
against other times, the phase of your first slip in the winning positions you did not
win, and the share of endgame errors made short of time. They point the practice at the
right place; they do not add findings.

## 6. The coaching layer (`--coach`)

The coaching layer (`chess_insights/coach/`, see docs/ARCHITECTURE.md) explains the report's
findings and routes you to practice. It is study material, not evidence:

* **It never adds, removes or changes a claim.** Every explanation points at a position or a
  finding that is already in the report. The critical positions it explains are your
  engine-flagged errors (a move losing 8 or more points of win chance), your repeated
  mistakes and choice points in your main lines, costliest first, at most `--coach-max`.
* **Explanations come from the engine's lines.** Each position is re-searched deeper
  (`--coach-depth`, default 20) for the best line and the line that refutes your move;
  the tactic (fork, pin, back rank ...) is named from those lines with Lichess's theme names;
  positional terms are compared at the *ends* of the two lines, because right after a move
  they can mislead (the move's immediate threat counts as a plus even when it loses).
* **External facts carry their source** (the Lichess opening explorer, cloud evaluations and
  tablebase, the chess-openings names, Wikibooks), are cached, and are optional: with
  `--offline`, no token or a service down, the report says what it skipped.
* **The one exception, motif claims** ("you miss forks more often than your opponents"), goes
  through the same claim rule as everything else (`stats.significance` at the strict alpha,
  Benjamini–Hochberg across motifs, the game as the unit) and must keep the null world's
  false-claim rate at or below 0.3 per report.
* **The LLM coach** (`--coach-llm`) only rewrites what the templates already say, from a packet
  of the report's facts; any move it names must replay legally from the position along the
  given lines, every number must be in the packet, and every finding it names must be a claim.
  An explanation that fails keeps its template wording, and the number rejected is noted.
* **Format views** get the explanations of their own format's positions; drills, the motif
  profile and the review schedule stay with the main report (they are practice for every
  format).
