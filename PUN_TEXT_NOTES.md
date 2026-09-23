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
signal to 429/503), the two most useful next moves are (a) revert `_THINKING`/`_STAGE2_THINKING`
to `False` and re-measure, since thinking is currently costing both time and a leaked-reasoning
failure mode with no demonstrated benefit, and (b) get the organizers' answer on the real rate
limit so the residual 48.5% contamination can actually be fixed rather than guessed at.

## Open

1. **Highest priority: fix the silent rate-limit fallback and log raw model replies / derived
   adjustments per unit per sweep** (see "RESOLVED" above). Confirmed cause, not just suspected:
   67% of units in a full sweep silently degrade to neutral on a 429 that `run_eval.py` never
   surfaces. Two parts:
   - (a) **Done, 2026-09-22**: `_throttle()` in `text_signal.py` is a shared, thread-safe
     sliding-window limiter (`_RATE_LIMIT_RPM = 36`, a margin under the House's confirmed 40
     rpm) that every `call_model()` attempt (Stage 1's worker pool and Stage 2's own call alike)
     must pass through before firing. Fixes the in-process case — a single unit, whether run
     directly or via `run_eval.py --unit`/`--concurrency 1` — for both local eval and the real
     leaderboard (one unit = one process there too). Tested in `tests/test_text_signal.py`
     (`TestThrottle`) with a mocked clock, no real sleeping.
   - (b) **Not done, and intentionally out of scope for now**: this does *not* coordinate across
     the separate subprocesses `run_eval.py --concurrency > 1` spawns — each gets its own 36 rpm
     budget, so N of them running together can still jointly exceed 40. Decision: keep
     `--concurrency 1` for local sweeps until/unless that's needed; a cross-process (file-based)
     shared bucket would be the fix if concurrency comes back.
   - Still open regardless: making the fallback loud (or at least counted) instead of silent, and
     the raw-reply logging itself. Everything below this item is still blocked on those in
     practice, even where not stated explicitly.
2. **F3's interface gap** (above) is real and unaddressed — a structural fix would need the
   output contract itself to change, not just the prompt.
3. **Self-consistency not implemented.** Sampling the stage-2 call a few times and taking the
   median (same idea Nish's own notes list as her #1 priority for stage 1) would likely help,
   and the request budget has room for it — but do item 1 first, or self-consistency just
   averages over a mix of real replies and rate-limited neutrals without knowing which is which.
4. **`skew` is implemented but disabled** (see "Skew" above) — flip `_SKEW_ENABLED` in
   `forecast_agent.py` once its effect can be measured cleanly (item 1) without rate-limit
   fallback confounding it.
5. **Thinking forced ON in both stages as of this session** (`_THINKING`, `_STAGE2_THINKING` in
   `text_signal.py`) — an active, unresolved experiment, not a considered decision. One direct
   manual test showed it changes the model's raw answer (sign flip on `shift` and `skew` for
   `t2-F1-cad-boc-2017`); the sweep-level comparison meant to check its aggregate effect is one of
   the three compromised by the rate-limit issue above, so it hasn't actually been measured yet
   either. Revert to `False` for both, per Nish's and the removed pipeline's prior findings, if
   item 1's logging shows this isn't earning its ~4-5x slowdown.
6. **No sweep run so far is fully trustworthy.** Even the sweep-1-vs-sweep-2 comparison in
   "Measured 2026-09-22" above (previously the one considered clean) likely has some units
   silently flipped to neutral on one side or the other. Everything needs item 1 before it means
   anything.
