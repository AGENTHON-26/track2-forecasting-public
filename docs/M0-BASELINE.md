# M0 — the official text-blind baseline, specified

## Executive summary (read this first)

Every Track 2 card is scored as a ratio against one official baseline, called **M0**. Your
composite is divided by M0's composite on that same card, which is why **1.0 means "no better
than a forecast that never read the text"**. This document specifies M0 completely enough that
you can rebuild its forecast yourself, for any card whose panels you hold, and see exactly what
the number you are being compared against is made of.

Two things are deliberately **not** here, and will not be: the generator's source code, and the
per-card scale values it produces. A per-card scale is the baseline's error measured against the
sealed outcome, so a published scale plus a reproducible baseline inverts to the answer. The
method is publishable; the values are not, and no published card carries them.

M0 is **not** the reference CLI in `qfbench2_track_forecasting/cli.py`, and it is not any file in
`baselines/`. Section 7 puts the two side by side, naming the function in the shipped CLI where
each difference lives, so you do not calibrate against the wrong thing.

Read [CONCEPTS.md §13](CONCEPTS.md) first if you have not: it explains where the normalization
sits in the leaderboard. This document is the layer below it.

---

## 1 — What M0 is for

Raw scores carry the units of whatever they forecast. A CPI-index card's CRPS is thousands of
times a bond-yield card's, so a plain average over cards would be an average of the largest
numbers, not of the best forecasts.

So each of the three components of your composite — marginal CRPS, joint variogram, tail — is
divided by **the same component of M0 on the same card** before the weights are applied. After
that division every card is on one scale: 1.0 is M0, below 1.0 beats it, and the leaderboard is
the equal-weight mean over cards.

M0 is a **text-blind joint Gaussian random walk**. It reads the numeric panels the card ships and
nothing else — no corpus, no card prose, no model. That is the point: Track 2 exists to measure
whether reading the documents helps, and the denominator has to be the forecast that did not.

## 2 — What is published, and what is sealed

| | Status | Why |
|---|---|---|
| The procedure (this document) | **Published** | You cannot reason about a ratio whose denominator is undefined. |
| The per-card seed rule | **Published** | §3.9. It is a function of the card id, which you already have. |
| The generator's source code | **Sealed** | It is one step from the values, and reads the sealed answer files to produce them. |
| Every `ref_scale.json` value | **Sealed** | Answer-equivalent — see below. |
| Realized outcomes | **Sealed** | The competition's whole firewall. |

**Why a scale is answer-equivalent.** A card's scale is not a setting; it is M0's *error* against
the realized outcome on that card. Given M0's forecast — which this document makes reproducible —
a published scale can be inverted to recover the outcome it was measured against. That is
sharpest on single-asset cards, which are most of the roster. This is why the values stay sealed
even though the method does not, and why no card released to participants ships the file. The
scorer enforces the same boundary from its own side: it refuses to read a scale from anywhere but
a card's `reference/` directory (`qfbench2_track_forecasting/normalization.py`,
`assert_reference_only`).

Publishing the method settles a real question at a cost we judge small: a reproducible baseline is
one more handle on a scored board, which is why the board reports a single aggregate statistic and
nothing per card. Publishing the values would end the competition outright.

## 3 — The procedure

M0 runs once per card. Its inputs are the card (`card.toml`) and the panels that ship with it.
Its output is a set of **500 joint draws** over the card's grid of `(asset, horizon)` cells — the
same artifact your agent produces, in the same shape.

Throughout: **as-of** is `[provenance] data_cutoff`, the date the card is truncated at.

### 3.1 The history for one asset

Find the asset in the panels: the first `*.parquet` in the unit directory, in sorted filename
order, that contains a row for it. Both `asset` and `asset_id` spellings occur and both are read.

Keep rows dated **at or before the as-of**, sorted by date, and take the **last 300**. That
trailing window is the whole of M0's memory — it does not use the full history. A panel with
fewer than 300 observations at the as-of contributes all of them.

### 3.2 Turning the history into steps, by `target_type`

| `[targets] target_type` | The step series | Why |
|---|---|---|
| `level` | first differences of the values | The target is a level, so the walk moves in level changes. |
| `log_return` | **the row itself** | These panels already ship per-step returns. Differencing them again would be a second difference: it telescopes the drift and inflates the spread by about √2. |

The `log_return` branch carries a tripwire rather than a guess. If the panel's values do not look
like per-step returns — median absolute value at or above **0.2** — the generator **refuses** and
the card is not scaled until a human decides which kind of panel it is. It never falls back to
"assume it is a price series".

One indexing consequence, easy to get wrong: a difference at row *i* spans the interval from row
*i-1* to row *i*, so the differences align to rows 1..n-1. A per-step return at row *i* **is**
row *i*, so the returns align to rows 0..n-1 — and row 0 then falls out anyway under §3.3,
because it has no preceding interval to test. Either way, a step carries the date of the row it
ends on.

### 3.3 The gap rule

Drop any step that spans a hole in the data. A hole is an interval longer than

```
max( 10 x median spacing of the trailing window , 5 days )
```

The median is taken over the intervals between the rows §3.1 selected, not over the asset's full
history. (On a daily panel the two agree; on a transfer card's panel they do not.) The dropped step
becomes missing, not zero.

This matters most on transfer cards, whose target asset ships as an early window plus a single
row at the as-of, with the years between deliberately withheld. Differenced naively that hole
reads as one day in which the asset moved a decade's worth. The card text tells you not to do
that; M0 does not do it either.

### 3.4 Date alignment across assets

Each asset's step series is indexed **by date**. The multi-asset step frame is then the
**intersection of dates present for every asset** — rows where any asset is missing a step,
including one dropped by the gap rule, are dropped for all of them.

This is the only set a covariance can honestly use. Assets with different holiday calendars do
not line up row-for-row, and pairing them by position instead of by date can invert the sign of a
correlation. Where the dates cannot support the alignment — wrong length, unparseable,
duplicated — the generator refuses rather than falling back to positional pairing.

### 3.5 Drift and covariance

From that date-aligned frame:

- **mu** = the mean step per asset (a vector over assets).
- **Sigma** = the covariance of the steps across assets (`numpy.cov`, sample convention, over the
  same intersected rows).

Both are estimated on the trailing window only. There is no shrinkage, no winsorizing, no regime
model. M0 is meant to be the floor.

### 3.6 The anchor

For a `level` target, the walk starts at that asset's **last observation in the panel** — not at
the as-of, and not at any interpolation of it.

The distinction is load-bearing on monthly macro panels, which stop at the as-of minus their
publication lag. Measured 2026-09-18 on the published card `t2-F1-cpi-glidepath-2023`: the as-of
is 2023-07-12 and `CPI_ALL`'s last observation is 2023-05-01, a 72-day lag. Anchoring at the
as-of would start the walk from a value the panel does not contain, and would also mis-count the
steps below.

For a `log_return` target the anchor is **0.0**: the target is a cumulative return over the
horizon, and the last observed return belongs to the history, not to that future total.

### 3.7 Horizon to panel steps

A card's `horizon` is the participant-visible grid key. It is not always stated in the panel's own
units: a few released macro cards state a business-day horizon over a monthly panel. Feeding that
straight into a per-step walk would rescale the baseline by the ratio between the two, so M0
converts.

For each target cell:

1. Take the dates of that asset's trailing window. If there are fewer than three, or the target
   date is missing or malformed, **use the declared horizon** and stop.
2. Estimate the panel's spacing as the mean interval, over intervals that are not holes (§3.3).
3. Count steps from the **last observation** (§3.6) to the target date:
   - spacing **> 20 days** → count **calendar months**: `12·(y1-y0) + (m1-m0)`.
   - otherwise → `round( (target_date - last_observation).days / spacing )`.
4. If that count is ≤ 0, **use the declared horizon**.
5. Otherwise compare the two. Let `ratio = max(steps, horizon) / max(min(steps, horizon), 1)`.
   **The declared horizon wins unless `ratio >= 2`.**

Step 5 is the whole of the override rule, and the 2x threshold is not a tuning knob. A card that
already states its horizon in panel steps lands close to the counted value, and overriding it
there would move a frozen number for no reason. A genuine unit mismatch is never marginal — the
affected cells are all at least 8x apart — so 2x separates the two cases with wide margin on both
sides.

Call the resulting per-cell step count **s**.

**Where the target date comes from, and what that means for you.** M0 reads each cell's target date
from the card's sealed `reference/` directory. Released cards do not publish target dates — the
worked exemplar in §4 is the single exception — so step 1's fallback fires for you on every other
card, and you use the declared horizon.

On daily panels that costs you nothing: the counted and declared values land within 2x of each
other, so step 5 keeps the declared horizon and M0 does the same. **On the four released cards that
sit on the monthly macro panel it is the whole difference**, because that is exactly where the
override was built to fire. Their step counts:

| Card | Declared `horizons` | Panel steps M0 uses |
|---|---|---|
| `t2-F1-cpi-glidepath-2023` | `[140, 160]` | **8, 9** |
| `t2-F1-sahm-watch-2024` | `[145, 165]` | **8, 9** |
| `t2-F4-covid-nfp-2020` | `[21]` | **2** |
| `t2-F4-cpi-vintage-2022` | `[21]` | **2** |

Use those numbers and you reproduce M0 on those four cards; use the declared horizon and you are
forecasting years out with a spread to match. A change is in preparation that restates these four
cards' horizons in panel steps so the table stops being necessary (§8); no scale moves when it
lands, because the conversion is already being applied on the denominator side.

### 3.8 The mean vector and the covariance matrix

Over the card's `d` cells, indexed `i = (asset a, horizon with step count s_i)`:

```
mean[i]   = anchor[a_i] + s_i * mu[a_i]            # anchor is 0 for a log_return target
cov[i, j] = min(s_i, s_j) * Sigma[a_i, a_j]
```

`min(s_i, s_j)` is what makes the draws a **path** rather than a bundle of unrelated marginals: a
random walk observed at two horizons shares the variance accumulated up to the earlier one. That
cross-horizon structure is the part the joint variogram term is there to reward, and it is the
part the shipped reference CLI does not have (§7).

Then `1e-10` is added to the diagonal, and the Cholesky factor is taken of `cov + 1e-9·I`. If that
still fails, the factor is taken of the **diagonal** of `cov + 1e-9·I` — a card whose covariance
cannot be factorized is scaled against independent marginals rather than not at all.

### 3.9 The draws

```
seed    = crc32(unit_id) & 0x7FFFFFFF        # unit_id is card.toml [task] id
rng     = numpy.random.default_rng(seed)
Z       = rng.standard_normal((500, d))
samples = mean + Z @ cholesky_factor.T
```

500 draws, always. The seed is a function of the card id alone, so any party can regenerate any
one card's baseline in isolation, without the rest of the suite and in any order.

**Cell order matters for an exact match**, and it is the one thing here the card does not pin down.
`Z`'s columns are assigned to cells in the order the card's grid is enumerated, so re-ordering the
cells gives different draws from the same seed.

The order is stored with each card's sealed answer rather than derived from the card, and it is not
uniform across the roster: it was fixed by whichever version of the authoring tooling realized that
card. On the released cards it is **sorted by asset id, then by horizon ascending** — measured
2026-09-18, that rule is the stored order on 100 of the 103 answered public cards. On the remaining
three it is the card's own `[targets]` list order instead.

This is not cosmetic. Two orderings of the same cells give the same *distribution* but different
draws, and against a realized outcome the components move: across the affected released cards the
difference reaches about **±30% on a single component**, most often the tail or the joint term.
Sort by asset id and ascending horizon and you will land on M0's own draws on almost every card;
where you do not, expect a few percent to a third on a component, not a factor.

The assets of `mu` and `Sigma` are ordered separately, by sorted asset id; that ordering is internal
to the estimate and changes nothing.

## 4 — Worked example, on a card you already have

`units/t2-EXAMPLE-ust-curve-1m` — 4 UST tenors, `target_type = "level"`, `horizons = [21]`,
as-of `2024-06-28`, target date `2024-07-31`. All figures below are **measured 2026-09-18** from
the published panel in this repository, and none of them touches a sealed artifact. This card
carries no sealed answer, so it has no scale; the example shows the procedure applied to a card you
hold in full, not a record of a scored run.

| Step | On this card |
|---|---|
| Trailing window (§3.1) | `rates_daily.parquet` has 516 rows, 129 per asset at or before the as-of — fewer than 300, so all 129 are used |
| Steps (§3.2) | `level` → first differences, 128 of them per asset |
| Anchor (§3.6) | last observation is `2024-06-28`, the as-of itself: this panel has no publication lag |
| Spacing (§3.7.2) | 1.391 days over non-hole intervals |
| Counted steps (§3.7.3) | `round(33 / 1.391) = 24` |
| Override test (§3.7.5) | `ratio = 24/21 = 1.14 < 2` → **the declared horizon 21 is used** |
| Seed (§3.9) | `crc32("t2-EXAMPLE-ust-curve-1m") & 0x7FFFFFFF = 795546941` |
| Draws | 500 x 4 cells, mean `last + 21·mu`, covariance `21·Sigma` (one horizon, so `min(s,g)` is 21 everywhere) |

Contrast, same repository: `units/t2-F1-cpi-glidepath-2023` is a monthly macro card whose panel
spacing measures 30.4 days, so §3.7 takes the month-counting branch, and whose card states
`horizons = [140, 160]` in business days. Those two are far more than 2x apart, so the counted
month step wins. You cannot finish that arithmetic yourself from the published card: it does not
publish its target dates. That is the expected shape — on most cards you can reproduce M0's
**construction**, and on the sealed set you can reproduce neither the target date nor the outcome.

## 5 — From draws to a scale (what happens on our side)

You do not run this half, and you cannot: it needs the sealed outcome.

M0's 500 draws are scored against the card's realized values with the same composite that scores
you — the card's own `[scoring.params]`: weights, `tail_levels`, and `joint`. The three raw
component values are what `reference/ref_scale.json` stores, and they are stored **raw**: the
baseline is not normalized by itself.

The tail component is computed under the metric the card asks for, `[scoring.params] tail_metric`,
defaulting to `pinball` (`qfbench2_track_forecasting/tail.py`, `DEFAULT_TAIL_METRIC`). A scale and
the scorer that divides by it must be built under the same tail metric — the two metrics are not
in the same units, and mixing them is meaningless rather than merely imprecise. No released card
overrides it — measured 2026-09-18, none of the 104 declares `tail_metric` — so in practice the
default is what every card is scored under.

**A component that comes out zero or negative is stored as 1.0**, i.e. that component is not
normalized at all. The case this exists for: a single-asset card has no pairs, so its variogram is
0 **by construction, not by merit**, and dividing by it would poison the composite. Those cards
are also the ones whose weights are redistributed — see [CONCEPTS.md §13](CONCEPTS.md), step 2 —
so the baseline still anchors at 1.0 there, exactly as it does on a multi-cell card.

## 6 — Fidelity: what you will match, and what you will not

### What you can reproduce, and what you cannot

A scale needs the realized outcome, so no scale is reproducible by anyone outside the organizers.
M0's **500 draws** are a different matter: for a released card whose panels you hold, §3 is enough
to rebuild them, and on most cards you will match them draw for draw. That is enough to answer the
questions that motivated publishing this — what the denominator assumes, where it is weak, and what
beating it requires.

Three things stand between §3 and an exact match, in descending order of size:

- **The four monthly-panel cards.** M0 converts their horizon to panel steps from a target date you
  do not have. Use the table in §3.7 and this disappears; ignore it and you are not close.
- **Cell ordering (§3.9).** Stored with the sealed answer, not derived from the card. The published
  rule matches almost every card; where it does not, a component moves by up to about a third.
- **Sealed cards.** You do not have the panels, the as-of or the target date, so the draws are out
  of reach entirely. This one is by design and is not going away.

Nothing else does. In particular the F2 transfer cards are **not** an exception: each ships its
transfer target's early window in its own panel bundle, and all four single-asset F2 transfer cards
reproduce bit-exactly from published panels (measured 2026-09-18). What is sealed on those cards is
the withheld middle of the series and the outcome, not M0's ability to read what you can read.

### Which revision produced the scales in force

**Measured 2026-09-18**, from the generation metadata each scale file carries: every scale
currently in force was generated on **2026-09-03**, under exactly the procedure specified above.

That generation corrected three things at once, relative to the scales that had been in force
before it. If you have read earlier organizer statements about the baseline, these are the
differences:

| Corrected on 2026-09-03 | Effect |
|---|---|
| Assets in the covariance were aligned by **row position** rather than by date (§3.4) | Reaches only multi-asset cards whose assets have different observation calendars — a handful of them. The largest move in any card's joint component was under 10%, and on one card the affected correlation changed sign. |
| `log_return` panels were **differenced a second time** (§3.2) | Reaches the return-target cards only: drift telescoped, spread inflated by about √2. |
| The tail component used a **coverage** penalty rather than pinball (§5) | Under coverage many tail scales collapsed onto a floor and a ceiling. Under pinball each card's tail scale is distinct and carries the target's units, like the marginal term beside it. |

There is no fourth correction pending against this document. The procedure above is the procedure
that produced the numbers in force.

### One caveat this document cannot resolve for you

A scale is only half of a normalized score; the other half is the scorer that divides by it. The
published package defaults to the pinball tail (`tail.py`, `DEFAULT_TAIL_METRIC`, verified
2026-09-18 in this repository). **Whether the scoring image running on the practice board has been
rebuilt against it is not something this document can promise**, and the same caveat already
attached to the DNF rule in [CONCEPTS.md §13](CONCEPTS.md) applies here. If a Development-board
number does not behave as described here, the deployed bundle predates the refresh. The Final is
scored against the scales and the scorer described above.

## 7 — What M0 is **not**

`qfbench2_track_forecasting/cli.py` is the reference submission CLI: a runnable floor that proves
the interface, passes the gates offline and can be edited into a real agent. **It is not M0**, and
a submission that runs it unchanged does not score 1.0. Every difference below is deliberate.

| | M0 (this document) | Reference CLI |
|---|---|---|
| History used | trailing 300 observations (§3.1) | the full history at or before the as-of (`_series`) |
| Drift on a `level` target | `s · mu` (§3.8) | **none** — `drift = np.zeros(...)` for a level target, a driftless walk (`_draw`) |
| Across horizons | one path: `cov[i,j] = min(s_i,s_j)·Sigma` (§3.8) | a **fresh** innovation per horizon — no cross-horizon covariance at all (`_draw`, the `for hi, h in enumerate(horizons)` loop) |
| Across assets | full covariance of steps (§3.5) | correlation of steps, nearest-PSD clipped, times each asset's own sd |
| Spread with horizon | from `min(s,g)·Sigma` | `sd · sqrt(h)` |
| Horizon units | converted to panel steps (§3.7) | the card's `horizon` used as-is |
| Seed | `crc32(unit_id)` (§3.9) | `--seed`, default **0**, the same for every card |

How far apart that leaves them, measured 2026-09-18 over the 103 released cards that have a
resolved answer: scored against the scales in force, **M0 lands at exactly 1.000000 on every one of
them** — which is what it means for M0 to be the denominator. The shipped reference CLI at its
default seed averages about **1.29** (median 1.05), with five cards at the 4.0 clip. A submission that
runs the CLI unchanged is roughly 29% worse than the baseline it is often mistaken for, and the gap
is not evenly spread: the median card is close, and a handful are pinned at the clip because the
CLI is driftless on level targets while M0 is not.

The five adapter scaffolds in `baselines/` are further away still: whatever model each is named
after, they all return a seeded Gaussian random walk whose draws are, in their own docstring's
words, i.i.d. across assets with no modelled cross-asset dependence
(`baselines/base.py`, `_gaussian_rw_samples`; `baselines/README.md` opens by saying they are
scaffolds). Do not read a gap against any of them as a gap against M0.

## 8 — Open items

Stated so they are not mistaken for settled:

- **Cards whose horizon is restated in panel steps.** A change is in preparation that would state
  the monthly-panel cards' horizons directly in panel steps. If it lands, §3.7's override stops
  firing on those cards and the step count comes from the card itself. No scale moves; the
  conversion becomes visible instead of implicit.
- **The commitment digest.** The evaluation plan carries a `ref_scale_commitment` field. The
  recipe that computes it over the scales is not yet documented, and this document does not
  specify it. When it is, it belongs here.

---

*Questions about this document belong on the public issue tracker. Answer-equivalent material —
scale values, realized outcomes, sealed card identifiers — will not be posted there, here, or
anywhere a participant can read, at any point before results are released.*
