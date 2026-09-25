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
is at most an *observation* (shown in its section, never in the top lists or the
study plan):

1. **Enough games** in *both* groups compared (module minimums, e.g. 8 rated games
   in an opening family as one colour, 20–25 in a split such as "right after a
   loss").
2. **A significant test after multiple-testing adjustment.** Difference tests of the
   mean `score − E` between a group and your other games, paired tests against your
   opponents in the same games, (differences of) two-proportion tests for rates.
   When a test is repeated over a family of groups (every opening, every time class,
   every block of the day) the p-values are Benjamini–Hochberg adjusted, and the
   adjusted p must be at or below the family's α (next section).
3. **An effect big enough to matter** (e.g. 0.06 points per game, about 40 Elo).

This rule lives in one place, `stats.significance`, and every module uses it.

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
Without a time zone (the command line's default, `--tz UTC`) the same UTC window is
still tested, but at 0.01 like any other block of the day, and it is titled by its
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

False claims in the top lists per report (the null player gives the time zone,
`Etc/UTC`, so late night is tested at 0.05; seeds 0–63 and 1000–1039 for the null
world, 24–64 fresh seeds for each variant):

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

Share of 32 worlds (seeds 1000–1031) in which the report's top weaknesses name the
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

## 3. Engine analysis (optional)

With `--engine`, Stockfish evaluates every position of your most recent games at a
fixed depth, and the tool applies **Lichess's published formulas** (ported from the
open-source lila/scalachess code) so that numbers are comparable to what you see on
Lichess:

* **Win %** from centipawns: `50 + 50 · (2 / (1 + e^(−0.00368208 · cp)) − 1)`,
  with cp clamped to ±1000 and mates counted as ±1000.
* **Move accuracy** from the win-% drop Δ:
  `103.1668 · e^(−0.04354 · Δ) − 3.1669 + 1`, bounded 0–100.
* **Game accuracy**: the average of a volatility-weighted mean and a harmonic mean
  of move accuracies.
* **Inaccuracy / mistake / blunder**: the move loses ≥ 5 / 10 / 15 percentage
  points of win chance. There are also mate-specific rules (a mate allowed, a mate
  lost).
* **Game phases**: Lichess's *Divider*. The middlegame starts when ≤ 10 major/minor
  pieces remain, the back rank empties, or the pieces become mixed. The endgame
  starts when ≤ 6 remain.

chess.com's own "accuracy" (CAPS2) is proprietary and uses a different scale. It
appears only for games you ran Game Review on. The report shows it separately and
never mixes it with engine accuracy.

**Repeated mistakes** are found by keying the position before each of your
engine-flagged errors by its EPD (the board, side to move, castling and en-passant
rights). Transpositions are therefore counted together, and positions where you
went wrong in two or more games are listed with the engine's preferred move.

## 4. What is excluded

* **Variants** (Chess960, bughouse, odds chess…) are excluded by default. Pass
  `--rules all` to include them.
* **Games that start from a set-up position** (thematic tournaments) are excluded
  from opening statistics.
* **Daily games** are left out of clock and session statistics. Their clocks
  measure days, and archived daily clocks store time spent rather than time
  remaining.
* **Games with fewer than 4 plies** are left out of skill metrics.
