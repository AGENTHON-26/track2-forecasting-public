# Changelog

Participant-facing changes to the Track 2 forecasting kit. Releases land on Wednesdays by
23:59 AoE (issue #4 in this repository); each entry says what changed and whether scoring
output is affected. Everything here is pinned to a toolkit tag, so re-pin deliberately.

## 2026-09-16

**Scoring output: unchanged** for every shipped unit. Scorer version stays `3.1.0`. All 104
`units/*/card.toml` declare `joint = "variogram"`, so the new single-cell guard below never
fires on a shipped card, and the pinned regression composites for `reg-t2-daily` and
`reg-t2-monthly` are byte-identical before and after this release.

### Reference forecast producer (forecast generation, not scoring)

- **Cumulative log-return cards** (the 16 practice units with `target_type = "log_return"`):
  the reference `forecast` CLI and the shared baseline fallback now read the card's
  `target_type`. For these cards the panel rows are decimal daily simple returns; the walk
  starts at 0, centres at `horizon × mean(log(1 + r))` and spreads by
  `sd(log(1 + r)) × sqrt(horizon)`, keeping cross-asset correlation. Earlier revisions anchored
  these cards at the last panel row and differenced the returns, inflating the spread by about
  1.4×. **Regenerate any reference outputs you produced for return cards.** `forecast_meta.json`
  now records `target_type`; the step transform lives in `qfbench2_track_forecasting/targets.py`.
  `level` cards are unaffected. (Answers issue #2.)
- New regression unit `reg-t2-logreturn` and regression check 13 pin that behaviour for both
  producers at 21- and 127-day horizons.

### Scoring code (no output change on shipped cards)

- The single-cell `ref_scale` relaxation is gated on the card's joint statistic through one
  predicate: the joint term is dropped on a 1-cell grid only for the variogram, which is 0 by
  construction. A 1-cell card declaring another joint statistic is refused as an organizer
  fault instead of silently discarding a defined component.

### Rules and documentation

- **House API allowance:** 25 requests per unit, at most 4,000 output tokens per call. Input
  limits and the accounting for failed or retried requests are not yet finalized and will be
  announced here when they are. Participant vendor API keys are not supported. The earlier
  "1,000,000 input + 100,000 output tokens per unit" text is withdrawn.
- **Artifact policy** (`docs/ARTIFACT-POLICY.md`, revision 2026-09-10.1): which fitted
  non-neural models, calibration parameters and static retrieval assets may ship in an image,
  under the information-cutoff, provenance and disclosure rules stated there; additional
  pretrained neural checkpoints need separate approval. No change to resource limits, the
  descriptor schema or the scoring formula. (Answers issue #5.)
- **Baselines and ablation:** the five files in `baselines/` are Gaussian-random-walk
  interface scaffolds, not implementations of the models they are named after. The scorer
  reports the composite and its components only; it emits no `information_uplift` or
  `text_ablation_delta`, and there is no separate ablated-forecast slot. Text ablation is an
  experiment you run and report yourself. Statements to the contrary that survived in
  `docs/CATEGORIES.md`, and the `pit_50` / `pit_90` rows in the `docs/CONCEPTS.md` glossary,
  are removed. (Answers issue #3.)
- **Reasoning baseline:** `baselines/reasoning_agent.py` reaches the model through the
  authenticated House proxy (`MODEL_ENDPOINT`, `MODEL_NAME`, `MODEL_TOKEN`, `http_proxy`),
  one request per call, no retries, redirects or direct fallback; missing configuration yields
  the labelled statistical fallback (`reasoning_applied: false`).
- **Toolkit pin:** `v2.4.0` everywhere, now including the reference `Dockerfile`, which still
  installed v2.3.1. (v2.3.1 refuses a descriptor with `"models": []` that the evaluation
  verifier accepts.) `pip show qfbench2-common` reports 2.3.1 from the v2.4.0 tag; that is a
  metadata lag, not a wrong install.

### Still open, not in this release

- An explicit horizon unit on the four monthly-panel cards (issue #2).
- The retry and overage rule for model calls (issue #2).
- A resolvable identifier for the deployed scorer revision, and how the Final revision will be
  announced before final submission (issue #7).
