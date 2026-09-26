# Text half — stage 2 (Pun, `feat/text-stage2`)

Builds on Nish's stage 1 (`NISH_TEXT_NOTES.md`). Stage 1 summarizes each document; stage 2 turns
those summaries into the `{shift, widen, skew}` adjustment `build_draws()` consumes.
`read_text_signal()` in `text_signal.py` now runs both stages. `forecast_agent.py` also changed,
in Dew's section: `build_draws()` now has a real skew tilt (see "Skew" below) -- currently
disabled by default pending a clean measurement, so this is otherwise still the thin-delegate
shape described above.

## What stage 2 does

- Takes stage 1's per-document summaries (unmodified) for every card.
- Reads the card's family (`F1`-`F4`, from `card.toml`'s `[metadata].category`) and builds ONE
  more prompt, with a paragraph specific to what that family is actually scored on (weights and
  descriptions straight from `docs/CATEGORIES.md`, not invented here):
  - **F1** — prefer small, justified moves; the numbers alone are usually enough.
  - **F2** — commit to a real `drift_sd` when guidance has shifted; consider widening too, since
    regime shifts create fat tails a narrow distribution misses.
  - **F3** — reason about ONE shared scenario, then derive every asset's numbers from it. Stated
    honestly in the prompt itself: this reply format has no field for a target correlation, so
    consistency has to come from the model's own reasoning, not a number it can state (same
    interface gap Nish's notes already flagged — not solved here, just not hidden).
  - **F4** — move `skew`/`widen` even when the center barely moves; the tail is what's scored.
- One more model call (thinking off — see "Measured" below), reply parsed as JSON with a
  brace-repair fallback, converted to native units via each asset's own historical daily
  volatility (`sigma`), same clamps as before.
- Any failure at any step (no summaries, no endpoint, unparseable reply) returns exact neutral,
  logged, never fatal to the card.

Carried forward from the single-shot pipeline stage 1 removed (git history before
`a5c5922`), since none of it was implicated in that pipeline's measured net-negative result: the
sigma-normalization conversion, the JSON brace-repair parser, the clamp ranges
(`drift_sd` in `[-1.5, 1.5]`, `vol_scale` in `[0.60, 2.00]`, `skew` in `[-1, 1]`).

```bash
python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text --adjust   # full stage-2 output
python3 -m unittest tests.test_text_signal -v                              # 29/29, mocked, no network
```

## Known gap, not fixed here

F3's cross-asset cards still get independent per-asset numbers. The output contract is
`{asset: {shift, widen, skew}}` — there is no field for a covariance or a shared factor, so two
assets that should move together can only do so if the model's own reasoning happens to be
consistent across them. This is the same gap Nish's notes describe for stage 1; stage 2's F3
prompt asks for shared-scenario reasoning explicitly, but the interface itself is unchanged.

## Measured 2026-09-21 (Nemotron 3 Super 120B, real endpoint, `t2-F1-cad-boc-2017`)

- Full pipeline works end to end: 5 real documents (2 BoC speeches, 1 FOMC statement, 1 older BoC
  financial-stability speech, 1 Beige Book) summarized, then synthesized into one adjustment.
- The model's own evidence: *"BoC raised rates 25bps, cites tightening policy, data-dependent
  path; Fed hiked, balance sheet normalization; CAD likely to strengthen modestly."* — reasoning
  about the cross-border rate differential across multiple documents, not just parroting the
  newest one.
- Result: `drift_sd=0.3` -> `shift=+0.0212` on a level of 1.2768 (about 0.3 standard deviations),
  `vol_scale=1.1`, `skew` 0.1 or 0.0 depending on the run (see next point). Non-hedged, modest,
  consistent with the F1 framing's push toward small justified moves rather than zero.
- **Temperature 0 is NOT deterministic on this endpoint** — same finding as Nish's stage-1
  measurement, now confirmed for the stage-2 call too. Two back-to-back calls on the identical
  prompt: `drift_sd` and `vol_scale` identical both times, `skew` came back 0.1 once and 0.0 the
  other. Nothing else varied.
- Thinking OFF for this call, per the removed pipeline's own measurement (git history): thinking
  on made the 120B "halve its committed adjustments" — more hedging, not better reasoning, for a
  "commit to a number" task. Not re-measured against this new prompt specifically; worth
  rechecking if it's ever revisited.

## Request budget

Worst case checked directly: `t2-F2-ecb-qe-telegraph-2014` ships 15 documents, all long enough to
require a real call. 15 summarization calls + 1 adjustment call = 16 requests, under the
confirmed 25-requests-per-unit House allocation (`SUBMISSION_CLI.md`) with room to spare. Stage
1's own retry loop (2 attempts per document on a bad reply) could in theory push a doc-heavy
unit's worst case close to the cap; not something stage 2 introduces, just worth knowing.

**Sharper as of 2026-09-23**: the organizers confirmed directly ([issue #17](https://github.com/Agenthon-2026/track2-forecasting-public/issues/17))
that the 25-request budget is "charged at admission, so a retry can cost a slot" — only a request
refused *before* admission (401/403, e.g. a bad token) is free. That means `call_model()`'s own
retry loop (`_RETRY_WAITS`, up to 3 extra attempts on a 429/5xx) and Stage 1's per-document
retry-on-bad-reply both spend the *same* unit's 25, not a separate allowance. The 16-requests
"room to spare" estimate above assumed zero retries; a congested doc-heavy unit that needs even a
few retries across its calls is meaningfully closer to the cap than that number suggests. Worth
tightening the worst-case math (see Open) rather than trusting the old headroom estimate.

## Measured 2026-09-22 — full practice-set sweep vs. the text-blind baseline

`eval_reports/20260915-195357.json` (before) vs `eval_reports/stage2-live.json` (after, one
real run against the live House endpoint, `feat/text-stage2`). Confirmed the "before" file is the
true text-blind result: `text_signal.py` did not exist on `dev` at all until Nish's stage-1 merge
(2026-09-20), so `read_text_signal()`'s import-failure fallback was returning exact neutral for
every card when that baseline was generated — not a config choice, the module was simply absent.

```
python3 compare_eval.py eval_reports/20260915-195357.json eval_reports/stage2-live.json
```

| family | before | after | delta |
|---|---|---|---|
| T2-F1 | 0.3428 | 0.3360 | -0.0068 (better) |
| T2-F2 | 1.1487 | 1.0603 | **-0.0884 (better, largest)** |
| T2-F3 | 1.0222 | 1.0096 | -0.0125 (better) |
| T2-F4 | 0.4563 | 0.4567 | +0.0004 (worse, negligible) |

Status counts unchanged (90 scored / 13 gates-only both runs) -- stage 2 introduced no new
crashes or inadmissibility. Biggest single-card win: `t2-F2-nirp-paradox-2016`, 4.9827 -> 2.9713
(-2.0114). Biggest regressions are all much smaller in absolute terms (+0.06 to +0.11), on
`t2-F3-cpi-shock-cross-2022`, `t2-F3-inversion-persistence-2019`, `t2-F2-qe2-jackson-hole-2010`,
`t2-F2-cut-sizing-2024`, `t2-F2-time-has-come-2024`.

Two caveats on this result, not swept under the rug:

- **This is one run, and the endpoint is confirmed non-deterministic at temperature 0** (see
  above). The F2 aggregate move (-0.0884, with one card improving by over 2 points) is large
  enough to very likely be real; the smaller family deltas (F1, F3, and especially F4's near-zero
  move) are within a range a second independent run could plausibly shuffle. Not rerun yet.
- **F4 not improving has a specific, testable explanation, not just "it didn't work": the F4
  prompt leans hardest on `skew` and `vol_scale` to move the tail, but `skew` has no downstream
  effect at all** (next section) -- so the one lever F4 pushes hardest on is partly inert.

## Skew — implemented in `build_draws()`, then disabled (2026-09-22)

`skew` used to have no downstream effect at all (item 3, below, before this section existed).
Fixed in `forecast_agent.py` (Dew's section): `build_draws()` now tilts each asset's tail via
Azzalini's skew-normal construction (`_skew_tilt()`), gated by a module flag `_SKEW_ENABLED`.

**Correctness of the tilt itself is solid** — unit-tested directly (`tests/test_build_draws.py`):
identity at `skew=0`, mean/variance preserved for any skew, correct sign, independent per asset.
One real bug caught before that: using `skew` directly as the shape parameter only reaches
~0.08 sample skewness at the full clamp (the skew-normal family's skewness saturates near ±0.995
but climbs very slowly) — scaled `skew` by 5x (`_SKEW_SHAPE_SCALE`) so the full clamp reaches
~0.85 instead, a real, visible tilt rather than a technically-present but imperceptible one.

**Measuring its real effect on scores did not go cleanly.** Ran two full live sweeps
(`eval_reports/stage2-live.json` skew-off, `eval_reports/stage2-skew-live.json` skew-on) and
compared: F1/F2/F3 all got worse, F4 essentially flat. But the single biggest swing,
`t2-F2-nirp-paradox-2016` (+2.0114), is almost the exact same card and magnitude as the biggest
*improvement* in the previous comparison (-2.0114) — a near-exact cancellation that's a signature
of this endpoint's confirmed run-to-run non-determinism, not a real skew effect. `read_text_signal()`
was assumed to re-call the live model fresh on every sweep with nothing cached, so shift/widen
would change between runs too, not just skew. **That assumption turned out to be wrong — see
"Suspected response caching" below, found right after this.**

**Decision: `_SKEW_ENABLED = False` for now**, not a revert. The math stays in place and tested;
it's just kept out of the actual forecast until a real measurement is possible. The right way to
get one: cache one run's `{shift, widen, skew}` per unit, then run `build_draws()` twice against
those *same recorded numbers* — skew forced on vs. off — so only the code path changes between
runs, not the model's answer too. Not built yet.

## RESOLVED: it wasn't caching, it was a silent rate-limit fallback (2026-09-22)

Ran a third full sweep with thinking forced ON for both stages (`eval_reports/stage2-thinking-live.json`,
103 units, `--concurrency 4`, **14.1 minutes total, 8.2 s/unit effective** — dramatically faster
than a single-card timing test predicted; don't extrapolate total sweep time from one card).

Expected this to differ from the skew-on/skew-off sweeps, since thinking changes the request. It
didn't. Checked directly:

```
diff <(... stage2-thinking-live.json ...) <(... stage2-skew-live.json ...)        # 0 lines
diff <(... stage2-thinking-live.json ...) <(... stage2-skewoff-live2.json ...)    # 0 lines
```

**Three full sweeps — skew on, skew off, thinking on — are byte-identical to each other at full
float precision, across all 90 scored units.** Traced one card directly to confirm this isn't
aggregate coincidence, `t2-F1-cad-boc-2017`:

| sweep | composite |
|---|---|
| `stage2-live` (first sweep) | 0.035045 |
| `stage2-skew-live` | 0.031257 |
| `stage2-skewoff-live2` | 0.031257 — identical |
| `stage2-thinking-live` | 0.031257 — identical |

Given a fixed RNG seed (`--seed` defaults to 0) and unchanged panel data, `build_draws()` only
produces different output if `{shift, widen, skew}` differs. For three sweeps with different code
to land on the exact same number, the model's answer itself must have stopped changing —
contradicting direct manual tests on this same card (`--adjust`, called by hand, twice) that
showed real variation: `shift`/`skew` of `+0.021/+0.1`, then `-0.007/-0.1`.

**Confirmed root cause: not caching — silent rate-limit fallback to `NEUTRAL`, invisible in eval
reports.** Verified directly:

1. Called `call_model()` by hand, three times in a row, same/near-identical prompt: three
   genuinely different real replies. Rules out server-side caching at the call level.
2. Ran `forecast_agent.py` standalone twice in immediate succession on `t2-F1-cad-boc-2017` and
   captured stderr directly (not through `run_eval.py`, which swallows it — see below):
   ```
   run1: [text_signal] source=llm family=F1 docs=5 assets=['CAD']            <- real LLM reply
   run2: [text_signal] adjustment call failed: HTTP 429: Too Many Requests; neutral
   ```
   `call_model()` does retry on 429 (2s/5s/10s backoff, `_RETRY_WAITS` in `text_signal.py`), but
   Stage 1's own 8-worker burst per unit plus the Stage 2 call is enough to exhaust the endpoint's
   shared quota for longer than 17s of backoff, so the retries run out and `read_text_signal()`'s
   `except Exception` catches it and returns `NEUTRAL`.
3. `NEUTRAL` + the fixed RNG seed (`--seed` defaults to 0) + unchanged panel data ->
   `build_draws()` produces byte-identical output regardless of what skew/thinking/etc. are set to
   in the code, because the text adjustment never reached it in the first place.
4. **Why this was invisible**: `forecast_agent.py` doesn't crash on a 429 — it degrades to
   neutral and exits 0. `run_eval.py` only echoes subprocess stderr when `returncode != 0`
   (`run_eval.py:67-68`), so the `adjustment call failed` warning never surfaces in a sweep run.
5. **Measured how bad it was**: diffed `stage2-skew-live.json` against the pure text-blind
   baseline (`20260915-195357.json`) unit by unit. **69 of 103 units (67%) are byte-identical to
   the text-blind baseline** — i.e. fell back to neutral. Only 34/103 (33%) ever got a real text
   signal applied. That's why three sweeps with different skew/thinking settings hashed
   identically: two-thirds of each sweep was reading the same fixed neutral value no matter what
   the code did, which swamped whatever real differences existed on the other third.

**Consequence: none of the three live sweeps (`stage2-skew-live`, `stage2-skewoff-live2`,
`stage2-thinking-live`) measure what they were run to measure.** The skew-vs-baseline comparison
in the previous section (sweep 1 vs. sweep 2) is *also* suspect for the same reason, not just
LLM non-determinism as originally guessed — some fraction of the "swing" between those two runs
is almost certainly units randomly flipping between real-signal and rate-limited-neutral, not the
model genuinely changing its mind.

**Needed before any further comparison work**: (a) log each unit's raw model reply / derived
`{shift, widen, skew}` / whether it hit neutral-via-error to a file per sweep, and (b) do
something about the rate limit itself — either throttle Stage 1's 8-worker pool and/or
`run_eval.py`'s `--concurrency` against a shared budget, or treat a 429-exhausted fallback as a
hard failure in `run_eval.py` (nonzero exit) instead of a silent degrade, so sweeps at least fail
loudly instead of quietly measuring the wrong thing. Neither built yet.

## Measured 2026-09-22 (evening) — first sweep with the throttle + issue logging: real, and worse

Full 103-unit sweep, `--concurrency 1`, throttle in place, thinking still forced on
(`eval_reports/stage2-thinking-throttled-live.json`). **7h39m total, ~4.45 min/unit average** —
much slower than the 29 s/unit pre-thinking baseline, consistent with thinking's known ~4-5x
per-call cost stacking across Stage 1's batch + Stage 2's own call.

**The throttle helped but didn't fix it.** 50/103 units (48.5%) still logged a `text_signal_issue`
— down from the earlier ~67% silent-neutral rate, but still roughly half the sweep. New failure
modes showed up that a request-rate throttle can't touch:
- `HTTP 503: Service temporarily overloaded` — a capacity problem, not a quota problem.
- `reply too long (85217 chars): reasoning leaked` — thinking leaking into the reply instead of
  stopping at `</think>`, the exact failure mode Nish's own notes already measured (~1/16 calls)
  with thinking on. Real evidence, not just her prior finding, that thinking's failure mode
  recurs at scale.

Likely explanation for the residual 429s: the House quota may be shared across the whole team (or
even the whole competition, if it's one endpoint per Nemotron deployment), so no amount of
in-process throttling on our side alone can guarantee staying under it. Filed as a GitHub issue
asking the organizers to confirm the actual limit and its scope (per-team vs. per-container).

**The comparison, now with real visibility into which units are contaminated:**

| subset | n | mean(live − baseline) | better | worse | same |
|---|---|---|---|---|---|
| all scored units | 90 | +0.0423 | 28 | 52 | 10 |
| "clean" only (zero logged issues) | 50 | +0.0869 | 18 | 31 | 1 |

Lower composite is better, so a positive mean diff means text is making things worse on average —
in both the full set and the clean subset. Only 10/90 scored units are byte-identical to the
text-blind baseline this time (vs. 67% before), so the throttle+logging did make this a much more
honest measurement than the three earlier sweeps. **Caveat on "clean": it means no logged
failure, not confirmed successful/informative signal** — a unit with no error could still have
gotten a genuinely-neutral reply from the model itself, that's a different thing from a
rate-limited fallback but still not evidence the text signal helped.

**Read on this measurement**: even under the cleanest conditions achieved so far, stage 2's text
adjustment underperforms the text-blind baseline on average. Before concluding this is real
(rather than another confound — thinking's failure modes, or the 50% of units still losing their
signal to 429/503), the two most useful next moves were (a) revert `_THINKING`/`_STAGE2_THINKING`
to `False` and re-measure, since thinking is currently costing both time and a leaked-reasoning
failure mode with no demonstrated benefit, and (b) get the organizers' answer on the real rate
limit so the residual 48.5% contamination can actually be fixed rather than guessed at.

**Update 2026-09-23**: (a) done for stage 1 only, per an explicit decision to isolate the two
stages' thinking settings rather than change both at once -- `_THINKING` reverted to `False`
(landmark docs still get thinking via `_THINKING_TYPES`, unchanged), `_STAGE2_THINKING` left
`True` deliberately. A 3-unit smoke run afterward measured ~40 s/unit vs. the ~4.45 min/unit from
this section's sweep -- consistent with the expected speedup, though 3 units is nowhere near
enough to re-measure the composite-score comparison itself; that still needs a real sweep. (b) is
answered -- see the new section below.

## RESOLVED (organizer-confirmed, 2026-09-23): there is no RPM limit at all — the throttle is a Dev-only defensive measure

Filed as [issue #17](https://github.com/Agenthon-2026/track2-forecasting-public/issues/17); the
organizer's direct reply:

> There is no requests-per-minute limit on the House route. It is the organizer-hosted endpoint
> described in HOUSE-MODEL.md, not the public NIM API, so the free-tier figure you quoted does
> not apply to it. The published limits are the whole model budget... 25 admitted requests per
> unit, at most 4,000 output tokens per request, charged at admission, so a retry can cost a slot
> and a request refused before admission (401 or 403) does not. [...] (1) no rpm limit; nothing on
> the route counts per minute. (2) The budget is per unit, not per team. `MODEL_TOKEN` and the
> proxy login are issued per unit... however many processes or threads you run inside the
> container, they share that unit's 25. (3) The published allowance is written per unit and is not
> tied to a phase, but... Development settings do not certify Final resources.

**What this means, reconciled with what we actually measured:**

- **Our 40 rpm assumption was wrong, but the throttle wasn't pointless.** The 429s we measured
  directly (real, repeated, not imagined) aren't an account-level rate-limit policy — there isn't
  one. They're almost certainly Development-phase shared-infrastructure congestion: every
  competing team hits the same organizer-hosted endpoint during Dev, so a burst from us can still
  get 429/503'd by an overloaded server even with no formal "requests per minute" rule being
  enforced against us specifically. `_throttle()` in `text_signal.py` (`_RATE_LIMIT_RPM = 36`)
  still has real defensive value *locally*, for exactly that shared-congestion reason.
- **It is not required for the real leaderboard submission, and should not be treated as load-
  bearing there.** Two independent reasons: (1) there is no RPM limit to respect on the real
  route at all, confirmed directly; (2) each unit's real run is its own isolated container with
  its own `MODEL_TOKEN` (point 2 above), so there's no possibility of *our own* traffic
  overlapping across units the way local sweeps can. The one thing genuinely left open is
  cross-team congestion during the sealed Final run -- the organizer explicitly declined to
  promise Development-measured behavior carries over ("do not certify Final resources").
- **The real, binding constraint everywhere (Dev and Final alike) is the 25-admitted-requests-
  per-unit count**, not a rate — see the sharpened "Request budget" section above for why our
  retry logic now looks like a bigger risk against that cap than originally estimated.

## Update 2026-09-26 — F3 branch: prompt targeted at the measured failure, skew back on, self-consistency shipped

Context: `feat/f3` fixed the numerical side first (`build_draws()` now walks one shared path across
an F3 card's horizons instead of drawing them independently — see `forecast_models.py`'s own
docstrings for that half; out of scope for this file). This section covers what changed on the
text side on top of that fix, resolving Open items 2 (partially), 3, and 4 below.

**F3's interface gap (Open item 2) — decided, not solved.** Measured the actual leverage of adding
a new schema field (a `beta`/correlation control) against just using the `drift_sd` field that
already exists: an oracle sweep put the schema-field ceiling at ~0.993 normalized composite,
against **0.747 for per-asset `drift_sd` alone** — an order of magnitude stronger, because the
joint variogram is bias-blind to a constant added to every asset but very sensitive to
*differences* between assets, and differential drift is exactly what `drift_sd` already encodes.
**Decision: no new field.** The gap is closed by prompting, not by the contract.

**Why the model wasn't using that lever: it was refusing.** A live sweep before this fix found the
model answering "no view" (`drift_sd=0`) on **81% of assets** across F3's 22 units, with its own
stated reasons giving it away — *"no GBP-specific view," "No CHF-specific policy signal"* — it was
waiting to be told about each asset by name, which is the exact inference F3 exists to test.
Rewrote `_FAMILY_FOCUS["F3"]` to say this outright (most documents won't name most assets — that's
the card design, not missing evidence; answering 0 for that reason is named as the single most
common way to fail this family), and gave Stage 2 cross-asset numbers it never had before: the
date-aligned correlation matrix `build_draws()` actually uses, each asset's trailing move in sigma
units, and a static asset-id glossary (`_ASSET_NOTES`) so a bare ticker carries a direction. Also
fixed a real bug in the same pass: the schema skeleton in `build_adjustment_prompt` only enumerated
`assets[:2]`, silently truncating the "answer for every asset" instruction on any card with more
than 2. Measured effect: non-zero answers went from 19% to 37% of assets on a re-sweep.

Added a family-aware widen clamp (`_WIDEN_CLAMP_BY_FAMILY = {"F3": (0.85, 1.25)}`, default
unchanged) after measuring that F3's joint term is monotone-increasing in widen above 1.0x — the
model could cost itself up to +46% on the term it's actually scored on by hedging wide, which the
old universal `(0.60, 2.00)` clamp allowed.

**Skew (Open item 4) — `_SKEW_ENABLED` flipped back to `True`.** The correctness of the tilt itself
was already solid (unit-tested); what was missing was a clean measurement, since every earlier
attempt at one was confounded by the rate-limit/caching issues above. Got a same-inputs replay
instead: took a recorded sweep's real ledger (3 F3 units with non-zero skew) and ran
`build_draws()` twice against those *same recorded numbers*, skew forced on vs. off — the fix this
file's "Skew" section above said was the right way to measure it and never built. Result: ratios
0.9865 / 1.0000 / 0.9992 (mean 0.9952) — nothing worse, all three unchanged-or-better. Weak
evidence (n=3, one arm), labeled as such, but combined with the structural argument that F4's own
prompt explicitly asks the model to move skew and F4 is scored on exactly the tail it shapes, that
was enough to re-enable it. Lives in `forecast_models.py` now (see the module-split note below),
not `forecast_agent.py` — the flag moved when the models did.

**Self-consistency (Open item 3) — shipped, but the measurement argues against keeping it as-is.**
Stage 2 now samples the identical prompt 3 times (`_STAGE2_SAMPLES = 3`) and combines per-asset,
per-field via median (`_median_reply`), keeping evidence from whichever sample's `drift_sd` lands
closest to the merged value. This was expected to be a clean win — it only reduces sampling
*variance*, and Nish's stage-1 notes already flagged it as the natural next step. A first
side-by-side (single-call vs. median-of-3, same commit, same 20 F3 units) instead measured
**median-of-3 raw composite 0.8937 vs. single-call 0.8889 — about 0.5% worse, not better.**
Per-unit: 12/20 units had all 3 samples independently land on exact neutral (median-of-3 does
nothing there, just triples the Stage-2 call count); of the 8 where it mattered, 5 improved and 3
got worse, but the 3 losses were larger (one alone, `bear-flattener-2022`, +12.7%) than the 5 wins
combined. **This is one replicate, not a settled result** — more replicates were kicked off to see
if it's noise or real; whoever picks this up next should check for that data before assuming
median-of-3 is a net win just because it shipped. If it holds up as a net negative, reverting to a
single Stage-2 call is a one-line change (`_STAGE2_SAMPLES = 1`).

**Also fixed alongside this work, smaller items:** the House endpoint was found to emit spurious,
empty-bodied HTTP 404s under load (verified: 6 rapid probes returned `200,503,404,200,404,200`,
the *same* request succeeding seconds after a 404) — `call_model()`'s retry loop only covered
429/5xx, now also retries 404 (`_RETRYABLE_CODES`). Added a per-asset ledger log
(`[text_signal] adj {asset}: ...`) that `to_adjustments()`'s return value was previously computing
and throwing away — this is what made the median-of-3 measurement above possible to interpret at
all. `run_eval.py` initially misclassified these new healthy ledger lines as failures; fixed with
`_TEXT_SIGNAL_OK_MARKERS`.

**Not touched by this update, still true:** the module split — `build_draws()` and its three
models (M2, the cumulative walk, the plain random walk) moved out of `forecast_agent.py` into
`forecast_models.py` — is Dew's side of the branch; see that module's own docstrings, not this file.

## Open

1. **Highest priority: fix the silent rate-limit fallback and log raw model replies / derived
   adjustments per unit per sweep** (see "RESOLVED" above). Confirmed cause, not just suspected:
   67% of units in a full sweep silently degrade to neutral on a 429 that `run_eval.py` never
   surfaces. Two parts:
   - (a) **Done, 2026-09-22**: `_throttle()` in `text_signal.py` is a shared, thread-safe
     sliding-window limiter (`_RATE_LIMIT_RPM = 36`, originally set as a margin under an assumed
     40 rpm NVIDIA-default limit -- since confirmed by the organizers, 2026-09-23, that no such
     limit actually exists on the House route; see the RPM section below). Still worth keeping as
     a defensive measure against real, observed Dev-phase server congestion, just not because of
     any formal rate policy. Every `call_model()` attempt (Stage 1's worker pool and Stage 2's own
     call alike) passes through it before firing. Tested in `tests/test_text_signal.py`
     (`TestThrottle`) with a mocked clock, no real sleeping. **Not required for the real
     leaderboard submission** — confirmed no RPM limit there either, and each unit runs in its own
     isolated container regardless.
   - (b) **Not done, and now lower priority than it looked**: this does *not* coordinate across
     the separate subprocesses `run_eval.py --concurrency > 1` spawns. Originally framed as "N
     processes could jointly exceed 40 rpm" -- now known (see the RPM section below) there's no
     40 rpm to exceed; the real reason 429s happen is shared Dev-phase server congestion, which
     more concurrent local traffic still makes modestly more likely regardless of the (nonexistent)
     formal limit. Decision unchanged in practice: keep `--concurrency 1` for local sweeps.
   - Still open regardless: making the fallback loud (or at least counted) instead of silent, and
     the raw-reply logging itself. Everything below this item is still blocked on those in
     practice, even where not stated explicitly.
2. **F3's interface gap** — **decided against a schema change, 2026-09-26** (see the Update section
   above): a new correlation field was measured at ~0.993 oracle composite vs. **0.747 for the
   `drift_sd` field that already exists**, so the fix is prompting the model to use the lever it
   already has, not adding one. What's still open: whether the prompt work extracts all of that
   0.747 ceiling — it clearly doesn't yet (37% of assets still get a real answer, not 100%).
3. **Self-consistency — shipped, 2026-09-26, but not a confirmed win.** See the Update section
   above: the first side-by-side measured it ~0.5% worse on raw composite, not better, with 3 units
   losing more than the other 5 gained. More replicates were started to check if that's noise; if
   it holds up, reverting `_STAGE2_SAMPLES` to 1 removes a real 3x Stage-2 cost for no benefit.
4. **`skew` was disabled, now re-enabled (2026-09-26)** — `_SKEW_ENABLED = True`, moved to
   `forecast_models.py` with the rest of the model code. See the Update section above for the
   (weak, n=3) replay evidence this was based on — still worth a real measurement once item 1's
   logging can isolate a text-signal-caused change from this endpoint's own noise.
5. **Thinking: both stages now off.** `_THINKING` back to `False` (2026-09-23) after the real
   full-sweep measurement above reconfirmed Nish's original finding at scale (4-5x slower, a
   leaked-reasoning failure mode, worse composite scores especially in F2). `_STAGE2_THINKING`
   turned `False` on 2026-09-25 — **decided, not measured**, and the distinction matters. What the
   2026-09-22 thinking-on sweep does establish is that stage 2 thinking is not *breaking* anything:
   3 stage-2 failures, all 503/429 transport errors, zero truncated or unparseable replies, so the
   4,000-token cap fits the reasoning and the JSON together. What it cannot establish is whether
   thinking helps or hurts the score, because run-to-run variance on this endpoint is the same
   order as the effect. Deciding that needs replicated sweeps per arm (~3-4 h), and leaving it on
   costs only ~20 s/unit against an 1,800 s budget, so it lost to the prompt A/B on value. It was
   turned off rather than left on because the one measurement that exists on this model (stage 1,
   at scale) says thinking underperforms, and because hedging is the opposite of what F3 rewards.
   Flip it back and re-measure if a text A/B comes out strangely.
6. **No sweep run so far is fully trustworthy.** Even the sweep-1-vs-sweep-2 comparison in
   "Measured 2026-09-22" above (previously the one considered clean) likely has some units
   silently flipped to neutral on one side or the other. Everything needs item 1's logging before
   a comparison actually means anything -- and now that stage 1 thinking is off, a fresh full
   sweep is needed anyway before drawing further conclusions from any of the numbers above.
7. **Request-budget worst case needs re-checking now that retries are confirmed to cost a slot**
   (organizer-confirmed, see "Request budget" and the RPM section above). The old "16 requests,
   room to spare" estimate assumed zero retries; worth directly computing (or measuring) the
   worst-case admitted-request count for the heaviest real unit (`t2-F2-ecb-qe-telegraph-2014`,
   15 documents) assuming every call needs its full retry allowance, to see how close that
   actually comes to the hard 25 cap. Not done yet.
