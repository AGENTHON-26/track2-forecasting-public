# Changelog

## Executive summary (read this first)

This file separates changes awaiting the next public update from changes already available in
the Track 2 forecasting kit. Each entry says whether scoring is affected. The published
baseline below was checked against public commit `1c6fdb6` on 2026-09-14; the next update has
not yet been published. Release timing follows the organizer announcement in public issue #4.

## Unreleased

**Track scorer code and evaluation cards: unchanged.** These changes update toolkit install
pins, the reference image, documentation and synthetic regression tests. They do not update
the deployed scorer. The local toolkit correction inherited by the new pin is noted below.

- **Toolkit installation:** align the reference Docker image, README and CI at `v2.4.2`,
  including the current submission-packaging command and corrected model-free simulation
  fixture. The image previously installed `v2.3.1`, which refuses an empty `models` array even
  when the evaluation verifier accepts it. The new pin becomes available with the next toolkit
  release; existing toolkit tags remain unchanged. The pin also includes the local CRPS
  correction already released in toolkit `v2.4.1`: a component with zero weight and exactly
  zero reference scale contributes zero instead of poisoning the composite with `NaN`.
  See the toolkit starter-pack changelog; this patch adds no scoring implementation.
- **Diagnostic documentation:** remove remaining claims that the scorer reports information
  uplift, text-ablation results, PIT or interval coverage. The scorer reports the composite
  and its components; participants can run separate diagnostics on labeled data they may use.
- **Log-return regression tests:** extend the existing synthetic fixture to 21- and 127-day
  horizons. Check both the CLI and shared baseline against cumulative `log(1 + r)` returns,
  an exact zero anchor, and the complete history window. This changes the synthetic fixture's
  expected result, not scoring code or the daily/monthly regression expectations.

## Published baseline — checked 2026-09-14

The following changes are already in public commit `1c6fdb6`. Scorer version is `3.1.0`.
All 104 shipped `units/*/card.toml` declare `joint = "variogram"`, so the single-cell guard
below does not change their scoring output.

### Reference forecast producer

- **Cumulative log-return cards** (the 16 practice units with `target_type = "log_return"`):
  the reference `forecast` CLI and shared baseline fallback read the card's `target_type`.
  Panel rows are decimal daily simple returns; the walk starts at 0, centres at
  `horizon × mean(log(1 + r))` and spreads by `sd(log(1 + r)) × sqrt(horizon)`, keeping
  cross-asset correlation. Earlier revisions anchored these cards at the last panel row and
  differenced the returns, inflating the spread by about 1.4×. **Regenerate reference outputs
  made with those earlier revisions for return cards.** `forecast_meta.json` records
  `target_type`; the step transform lives in `qfbench2_track_forecasting/targets.py`.
  `level` cards are unaffected. (Answers issue #2.)
- The synthetic `reg-t2-logreturn` fixture checks both producers. Its published version uses
  a 21-day horizon; the second horizon is part of the unreleased update above.

### Scoring code

- The single-cell `ref_scale` relaxation uses the card's joint statistic: the joint term is
  dropped on a one-cell grid only for the variogram, which is zero by construction. A one-cell
  card declaring another joint statistic is refused as an organizer fault instead of silently
  discarding a defined component.

### Rules and documentation

- **House API allowance:** 25 admitted requests per unit, at most 4,000 output tokens per call,
  both counted by the House route. Participant vendor API keys are not supported. The earlier
  "1,000,000 input + 100,000 output tokens per unit" wording is withdrawn and **nothing replaces
  it**: the model budget is requests per unit, and there is no per-unit token allowance. An
  admitted request is charged before forwarding, so an upstream failure, a lost response or a
  retry can spend a slot.
- **Ties in the ranking score:** if two Final submissions finish this track with the same
  ranking score, the one uploaded earlier is ranked ahead.
- **Artifact policy** (`docs/ARTIFACT-POLICY.md`, revision 2026-09-10.1): fitted non-neural
  models, calibration parameters and static retrieval assets may ship under the information
  cutoff, provenance and disclosure rules stated there; additional pretrained neural
  checkpoints need separate approval. No change to resource limits, the descriptor schema or
  scoring formulas. (Answers issue #5.)
- **Baselines and ablation:** the five files in `baselines/` are Gaussian-random-walk interface
  scaffolds, not implementations of the models they are named after. There is no separate
  ablated-forecast slot. Text ablation is an experiment you run and report yourself.
  (Answers issue #3.)
- **Reasoning baseline:** `baselines/reasoning_agent.py` reaches the model through the
  authenticated House proxy (`MODEL_ENDPOINT`, `MODEL_NAME`, `MODEL_TOKEN`, `http_proxy`),
  one request per call, no retries, redirects or direct fallback; missing configuration yields
  the labelled statistical fallback (`reasoning_applied: false`).

### Documentation still to reconcile

- Explicit horizon units for the four monthly-panel cards (issue #2).
- Participant documentation of operational input limits and failed-request/retry handling
  (issue #2); the code and deployed policy must be checked before updating these instructions.
- A resolvable deployed-scorer identifier and the Final revision announcement (issue #7).
