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
| **0.01** | Other times of day, time controls, opponent strength, session length, time trouble, losing on time, slow openings, clock handling, quick losses in one opening, opening habits (castling, early queen moves, development, early pawn moves), engine findings, tactical patterns you miss or allow (the coaching's motif claims) | Exploratory scans (many ways to get lucky), and habits measured against your opponents in the same games, whose paired or mirror tests have power to spare when the habit is real. |

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
* **Opening habits** (castling, the queen, development, pawn moves) are compared with your
  opponent in the same game, White and Black weighted equally (next section).

### Opening habits: development and king safety

The "Development & king safety" section (`analysis/structure.py`) measures four habits for
both players in every standard game, and needs no engine:

| Habit | Counted | A finding needs a gap of at least |
|---|---|---|
| Castled by move 10 | yes / no, by each side's own 10th move | 10 percentage points of games |
| Queen out early | a queen move in moves 1–8 that is not a capture (recapturing with the queen is often forced) | 8 percentage points of games |
| Knights and bishops out by move 10 | 0–4 pieces off their starting squares | 0.3 pieces per game |
| Pawn moves in moves 1–10 | 0–10 | 0.5 pawn moves per game (a weakness only: fewer pawn moves is not claimed as a strength) |

* **Paired within the game.** Each game gives one difference, your count minus your
  opponent's, so the opening, the format, the clock, the rating gap and the result are the same
  for both sides and cancel. The game is the unit of the test.
* **Colour-balanced.** White moves first, which changes how soon either side castles or
  develops, so the differences are averaged within each colour first and the two colours count
  equally: a player with more White games is not judged on a colour mix. Within a colour the
  variance never drops below that of two independent binomial shares.
* **One family at the strict alpha.** The four tests are Benjamini–Hochberg adjusted together
  and judged at 0.01 under the claim rule, with at least 30 games in which both players reached
  the window (20 plies for the habits measured by move 10, 16 for the queen), so a game that
  ended on move 7 counts as "not castled" for neither side. Chess960 and games from a set-up
  position are left out. Severity uses the gap shrunk toward 0 by its own noise.
* **Associations, not causes.** "You castle later than your opponents" can come from the
  openings you choose or from opponents who attack early.

On the null world (below), the module adds no false claims: re-measured with it in the report,
64 reports of 600 games (seeds 0–63) had 0.234 false claims per report (15 in all, at most 2 in
one report: colour 5, habits 5, openings 4, clock 1), none of them from this section. Planted
habits in 600-game worlds (the null player castles by move 10 in about half of the games,
brings the queen out early in 14% and has 3.0 of 4 minor pieces out by move 10) are found in
31 of 32 worlds (castling by move 10 in about 35% of games), 32 of 32 (the queen out early in
about 30%) and 32 of 32 (about 0.45 fewer pieces out); `POWER_RUNS=32 pytest tests/test_power.py
-k opening_habit` re-measures them.

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
* **Opening habits** (castling late, the queen out early, slow development, at the sizes
  given in "Opening habits" above) are found in 31–32 of 32 worlds at 600 games: the paired
  test has power to spare.
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

### Format labels and pictures change no claim

Every finding says which formats it rests on (`Insight.formats`, games per time class, shown
as "blitz only" or "bullet, blitz and rapid"). A module sets it where it knows; otherwise
`fill.py` takes the format its evidence names, the games that reached its position, the games
of its opening and colour when the evidence's own counts confirm them, and else every game the
module analysed (the engine-analysed ones for the engine sections). Every finding also gets a
picture: its module's chart or board, or a chart built from its evidence (your score against
the rating's expectation, your rate against your opponents' ...), or the board of its position.
Evidence that fits no pattern gets no picture rather than one that says something the numbers
don't. Neither step changes which findings exist, their kind, severity, confidence or wording.
A chart split by format gives a format its own bar only from 10 games.

### Per-format views: each carries its own budget

When your games cover two or more formats, every format with at least 60 games
(`pipeline.MIN_FORMAT_GAMES`) also gets a view of its own: the same analysis run on that
format's games alone, with its own strengths, weaknesses and study plan. Below 60 games most
tests could not say anything (they need 8 games per opening and colour, 25 per split), and the
view would be mostly "not enough data".

A view is a separate report: its claims are tested on that format's games only, with the
same rule, thresholds and adjustments, and they are never added to the main report's lists.
So each view carries its own false-claim budget. On the null world, measured on the views
(24 reports of 1,200 games, 16 of 3,000; seeds 0 upward; re-measured with the Development &
king safety section in the report, with the same numbers):

| Null world | Main report | Bullet view | Blitz view | Rapid view | All views together |
|---|---|---|---|---|---|
| 1,200 games (views of about 180 / 730 / 300 games) | 0.17 | 0.08 | 0.17 | 0.29 | 0.54 |
| 3,000 games (views of about 450 / 1,810 / 740 games) | 0.00 | 0.19 | 0.13 | 0.19 | 0.50 |

Each view is about as reliable as a report of its size, but together they are three more
chances to be wrong: across all three views of a report there is about half a false claim on
average, where the main report has about 0.2. Read a finding that appears only in one view
with that in mind. A finding in the main report and in a view is the same fact seen twice,
not two pieces of evidence. The views re-run the analysis modules only: the coaching runs
once, on all games, so its motif claims appear in the main report and never in a view.

## 3. Engine analysis (optional)

With `--engine`, Stockfish evaluates every position of a sample of your games at a
fixed depth, and the tool applies **Lichess's published formulas** (ported from the
open-source lila/scalachess code) so that numbers are comparable to what you see on
Lichess.

**Which games** (`engine.select_engine_games`; the rest of the report always uses every game):

* `--engine-games N`: how many (default 150; the GitHub workflow uses 300). Only standard
  chess and Chess960 games of at least 10 plies (five moves each) count; a game whose moves
  don't replay is skipped and replaced by the next one (of the same format, in a balanced
  sample).
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
* **Puzzle packs** (with `--coach`) become actions of the tactics and blunders items, one pack
  per item, preferring the pack for the item's own pattern. They are practice material too.

Some actions quote numbers from your games that were not tested and are not claims:
how each of your choices at a move scored ("after 1.e4 e5 2.Nf3 Nc6: 3.Bc4 66% in 90
games, 3.Bb5 40% in 40"), Stockfish's average evaluation after move 10 in a weak opening
(to tell an opening problem from a middlegame one), your blunder rate late at night
against other times, the phase of your first slip in the winning positions you did not
win, and the share of endgame errors made short of time. They point the practice at the
right place; they do not add findings.

## 6. The coaching layer (`--coach`)

The coaching layer (`chess_insights/coach/`, see docs/ARCHITECTURE.md) explains the report's
findings and routes you to practice. It is study material, not evidence. It needs the engine
analysis (`--engine`); what it cannot do in a run (no Stockfish 16, no puzzle database, no
token, a service down, `--offline`) it skips with one note in the report.

### What it explains

* **The positions** (`coach/critical.py`): your repeated mistakes (the same wrong move in the
  same position, exactly as the "Positions you keep getting wrong" section finds them), then
  every other move of yours that lost at least 8 percentage points of winning chances, costliest
  first, then your usual move at each choice point of your main lines (the positions the
  openings section draws). One entry per position and move, at most `--coach-max` (150); choice
  points keep a fifth of the places left after the repeated mistakes when there are more errors
  than places. Standard chess only. Selecting a position is not a claim about it.
* **Deeper lines** (`coach/deep.py`): each position is searched again at `--coach-depth` (20)
  with three lines before your move (the best line and two alternatives) and one line after it
  (the refutation of your move), both scored from your side. Each search is capped at 8 seconds
  before your move and 4 after it (`--coach-seconds`), and a search that stopped short of the
  depth is counted in the notes.
* **Wording.** When the deeper search finds your move less than 5 percentage points of winning
  chances **and** less than 1 pawn behind its first choice (less than an inaccuracy), the text
  says your move is close to Stockfish's first choice instead of explaining an error; the pawn
  condition matters in lopsided positions, where −10 against −6 is only a few points of winning
  chances. At a choice point, your usual move is explained as a mistake only when it lost at
  least 5 points on average in the analysed games (the openings section's bar): in the opening
  several moves are often about equally good. Evaluations beyond 10 pawns are named in words ("a
  winning position for you"); a number that size means nothing. Material is said to be lost or
  won only when the engine's verdict on the lines backs it. At most three sentences: what the
  refutation does, what it wins (material or a concept), and what to check next time.

### Tactical patterns (motifs)

`coach/motifs.py` names patterns with Lichess's puzzle-theme names (fork, pin, skewer,
hangingPiece, discoveredAttack, backRankMate, discovered and double check, trapped piece,
smothered mate, mate in N, deflection, attraction, advanced pawn), using python-chess attack
maps along the engine's lines: the refutation's patterns carried out by your opponent (what your
move allowed) and the best line's patterns carried out by you (what you missed). Patterns are
looked for in the first 8 plies of a line and a mate in the first 10.

**Forcing only.** An engine line is not a puzzle: past its first quiet move the other side's
replies are the engine's best defence, not forced moves. A pattern is kept only when it comes on
the carrier's first move of the line, or when every earlier move of the carrier was a check or a
capture (the reply to the first quiet move is still in reach). Mates are kept always.

**The precision gate.** A pattern is named only when its detector passed a gate on real Lichess
puzzles (`tests/fixtures/lichess_puzzles_sample.csv`: 1,573 puzzles from the CC0 puzzle
database): at least 20 puzzles tagged and at least 80% of them carrying Lichess's own tag.
Otherwise the text says "a tactic" and the board marks nothing. For the six core themes
(`python scripts/motif_precision.py`; the solver's patterns, from the position after the
opponent's first move):

| Theme | Tagged by us | Precision | Tagged by Lichess | Recall |
|---|---|---|---|---|
| fork | 209 | 0.99 | 207 | 1.00 |
| pin | 141 | 1.00 | 141 | 1.00 |
| skewer | 93 | 1.00 | 93 | 1.00 |
| hangingPiece | 93 | 0.96 | 113 | 0.79 |
| discoveredAttack | 102 | 1.00 | 199 | 0.51 |
| backRankMate | 102 | 1.00 | 102 | 1.00 |

Every other theme listed above passes too (discovered check 0.94, the rest 0.99–1.00), except
mate in 5 (2 puzzles in the sample) and overloading (Lichess no longer tags it, so there is
nothing to measure against); those two are never named. hangingPiece needs the move before the
line to tell a free piece from the end of a trade: when the position before your opponent's last
move is passed, as the coaching does along best lines, it scores 1.00 / 1.00
(`--previous`). The low recall for discovered attacks means about half of them go unnamed: the
profile undercounts that pattern rather than inventing it.

Two limits of this check:

* **Lichess's labels come from its own tagger** (lichess-puzzler), and our definitions follow its
  definitions. Near-perfect agreement shows that we implement those definitions faithfully, not
  that a human coach would describe every position the same way.
* **Engine lines run on after the tactic is over.** Continuing each puzzle with Stockfish 16's
  principal variation (one 30,000-node search from the end of the solution) and counting every
  pattern along the longer line lowers precision on
  the core themes to 0.89–1.00 at 8 plies (the length the coaching reads), 0.84–1.00 at 10 and
  0.81–1.00 at 12, fork the lowest (`scripts/motif_precision.py --extend 8`). These numbers are
  pessimistic: the stress test does not apply the forcing-only rule, and some of the later
  patterns are real ones Lichess had no reason to tag.

### Positional concepts

`coach/concepts.py` compares the two lines where they have played out, not right after your
move: right after 5...e5, Stockfish counts the pawn's attack on the d4 knight as a plus even when
the move loses. Each line is read about 3 moves in (6 plies from the position before your move),
at the first quiet position from there (not in check, nothing left to recapture on the square
of the last move) up to 4 plies later, else the last quiet one before it.

* **Stockfish 16's classical evaluation terms** (material, pawns, mobility, king safety,
  threats, passed pawns, space ...) at the two ends, each blended by its position's material
  phase; the refutation's end minus the best line's end, from your side, in pawns. Differences
  under 0.15 pawns are dropped. When both ends have the same material, a Material difference is
  called piece placement. Stockfish 16 is the last version that prints these terms; with another
  version this step is skipped with a note.
* **Board facts** from python-chess (the bishop pair, castling rights, a king left in the centre
  after move 12 with queens on, isolated, doubled and backward pawns, holes while you have five or
  more pawns and your opponent a knight or bishop): named only when worse for you at the
  refutation's end than at the best line's end and not already true before your move.
* An explanation links a short note on its main concept (`data/concepts.json`, written for this
  project) with the chapter of Capablanca's *Chess Fundamentals* or Lasker's *Chess Strategy*
  that explains it.

### The only claims: motif claims

The motif profile (`coach/profile.py`) counts, for every mistake or blunder by either side in the
engine-analysed games (a short line at depth 10 before and after each move), the patterns missed
(in the best line, for the player who went wrong) and allowed (in the refutation, for the other
side), each error once per pattern. Its table and chart ("Patterns in the mistakes", per 100
moves, you against your opponents in the same games) are observations. The claims "you miss
(allow) forks more (less) often than your opponents" must pass:

* **Two tests, both significant.** Your rate per move against your opponents' in the same games,
  and the pattern's share of your errors against its share of theirs, each a game-clustered test
  (the unit is the game, you and your opponent paired within it, as in the Engine review). The
  share test keeps "you make more errors of every kind" as one finding (the blunder rate), not six.
* **Benjamini–Hochberg** across every named pattern and both kinds, for each of the two tests,
  and `stats.significance` at the strict alpha (0.01), with a confidence of at least 0.5.
* **Big enough:** a rate ratio of at least 1.3 (its inverse for a strength), and still 1.3 after
  dividing by the ratio of your other errors to theirs.
* **Enough data:** at least 30 games in which either side had the pattern and 20 occurrences.
* **Named patterns only:** only gated themes are tested, and only patterns the line forces are
  counted.

The null world has no engine lines, so these claims are calibrated on simulated counts
(`tests/test_profile.py`): in 200 simulated 300-game sets in which both sides share the same
pattern rates, with game-level clustering, no claim was made (the test allows 0.05 per set);
when you make 60% more errors of every kind, none either (30 sets). A fork missed twice as often
per error as your opponents miss it is found in 17 of 20 simulated 300-game sets (the test asks
for 16), never the other way round. `--no-motif-profile` turns the profile and its claims off.

### External facts, drills and progress

* **External facts carry their source** and are optional: the Lichess opening explorer
  (masters, and Lichess blitz and rapid games of players one or two rating groups above you;
  needs a token), Lichess cloud evaluations, the Lichess tablebase (positions with 7 pieces or
  fewer: wins you let slip and draws you lost, by ending type), the bundled chess-openings names
  and Wikibooks. They are cached, limited per report, and never change a claim. The tablebase
  endings are observations: the conversion claim stays with the Engine review's own test.
* **Drills** (`coach/drills.py`): up to three packs of 30 puzzles for the patterns you miss most
  (from the motif profile; failing that, from the explained positions, then from the engine's
  own tags) and one pack from your openings, from the filtered Lichess puzzle database, in the
  `--drill-rating` window (default 1200–1600, not adjusted to your rating), most popular first.
* **Review schedule:** your ten costliest mistakes and the first five puzzles of each pack come
  back 1, 3, 7 and 21 days after the report; items from the previous report's JSON that are due
  move to their next step.
* **Progress** (`coach/progress.py`): each study-plan target next to the previous report's
  number, with the direction that counts as better. It is an observation: two reports usually
  share most of their games, so a change here is not a test of anything.
* **Maia-2** (`--maia`, optional): how often players at your rating find the better move and play
  yours (the blitz model for bullet and blitz, the rapid model for rapid and daily); it orders the
  explanations by cost × chance of finding the better move. It changes no claim.
* **Format views** get the explanations of their own format's positions; drills, the motif
  profile and the review schedule stay with the main report (they are practice for every
  format).

### The rating map

The opening explorer's rating groups and Maia-2 work on the Lichess scale, so a chess.com rating
is converted first (`coach/rating_map.py`): piecewise linear between the rows of a small table per
format, rounded to 5, shown as "about 1390". The table follows the ChessGoals rating comparison
(July 2026), but only two rows were checked against that page (chess.com blitz 900 ≈ Lichess
1360, 1000 ≈ 1425); the other rows are rough estimates, and the gap between the sites moves by
about 30 points a year. Daily games use the rapid rows. So "players one or two groups above you"
and Maia's "at your rating" are approximate. The drill window does not use this map.

### The LLM coach and its verifier

With `--coach-llm` and an API key, Claude gets a packet of what the report already established
(`coach/packet.py`: every claim with its evidence, the claims of each format view, the study plan
with its targets, and up to 20 explained positions with their lines, evaluations, motifs, concept
differences and sources) and returns rewritten explanations and a weekly plan. The packet is
passed as data; the instructions tell the model that text inside it is never an instruction.
`coach/verify.py` checks every text before it replaces a template:

* **Moves:** every move it names is replayed with python-chess from the position along the lines
  the packet gives for it (the best line, the refutation, cloud-evaluation lines, opening-database
  moves, the tablebase move), with the right move number; moves written one after another must
  follow each other in one of those lines. Computer notation ("e6e5") is refused.
* **Material:** "wins your queen" needs a line of that position that captures a queen (yours,
  when it says "your").
* **Numbers:** every number must be in the packet (within 0.1, or the rounding of the digits
  written); percentages are compared with shares and win-% points, evaluations in pawns; a signed
  number needs that sign in the packet unless its sentence names White or Black; "you stand 1.5
  pawns better" must point the way the engine's evaluation does. Links and dates must be in the
  packet too.
* **Claims:** a claim id it cites must be one of the report's; wording that calls something your
  strength, weakness or habit needs a claim of that kind on the same subject.
* **Formats:** a sentence may name its game's format and the formats of the claims it relies on.
* **Style:** at most three sentences per position, pawns rather than centipawns, plain text
  without markup or internal ids.

A text that fails keeps its template wording, and the report counts how many were reworded and
how many failed (the rejected texts and the reasons are kept in the JSON, up to 20). Weekly-plan
entries that name no study-plan item are dropped, and the plan is trimmed to
`--practice-minutes` a day. `chess-insights ask` answers questions through the same checks (at
most eight sentences); when there is no checked answer it quotes what the report says instead.

What the verifier cannot check: evaluations in words without a number ("you were winning"),
whose move a piece move is when it has no move number ("White plays Bxd6"), and chess ideas said
in words only ("the knight beats the bishop"). Those can be wrong in an accepted text. It has not
yet been run against the real API (see PLAN.md).

### Sources and licences

| Source | Used for | Licence and handling |
|---|---|---|
| Lichess puzzle database | Drill packs; the motif precision gate (a 1,573-puzzle sample in the tests) | CC0; downloaded by `chess-insights puzzles-db`, filtered subset kept in the cache |
| lichess-org/chess-openings | Opening names by position | CC0; bundled as `data/openings.tsv` |
| Lichess opening explorer, cloud eval, tablebase | Facts on explained positions | Public API answers, shown with their source, cached locally; never bundled |
| Wikibooks, Chess Opening Theory | Two sentences on the opening of an explained position | CC BY-SA 4.0: at most two sentences, shown with "Wikibooks, CC BY-SA 4.0" and a link to the page; only the extract is cached |
| Capablanca, *Chess Fundamentals* (1921); Lasker, *Chess Strategy* (1915) | Chapter pointers in the concept notes | Public domain in the USA (Project Gutenberg #33870, #5614); only chapter and section titles are cited, the notes are this project's own words |
| Stockfish (16 for the concept terms) | Game analysis and the coaching's searches | GPL-3.0; a separate program you install (Ubuntu's package on GitHub), run over UCI, not bundled |
| Maia-2 (optional) | Move probabilities by rating | MIT; installed with `pip install maia2`, downloads its weights on first use |
| Claude (optional) | Rewording and the weekly plan | Anthropic API with your key; receives the packet above |

Details on the bundled files: `src/chess_insights/data/NOTICE.md`.
