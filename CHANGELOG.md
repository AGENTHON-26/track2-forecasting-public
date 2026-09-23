# Changelog

## Executive summary (read this first)

This file records what changed in the Track 2 forecasting kit, newest first. Each entry says
whether scoring is affected. `## Still to reconcile` lists known open items. `## Unreleased`
lists changes merged for the next public update; each dated section below it is headed by the
UTC date its changes reached public `main` and the commit `main` was at afterwards, down to the
published baseline checked on 2026-09-14. Release timing follows the organizer announcement in
public issue #4.

## Still to reconcile

- A resolvable deployed-scorer identifier and the Final revision announcement (issue #7).

## Unreleased

**Track scorer code and evaluation cards: unchanged.** Documentation, link and CI-comment
corrections; no scoring path, gate, card or published number changes.

- **Last Development runs start by 20:00 UTC on Monday 12 October 2026.** A scheduled
  maintenance window on Tuesday 13 October 2026, 08:00–12:00 UTC stops the evaluation fleet. An
  upload that has not started by 20:00 UTC on 12 October, whenever it was made, is not run, and
  one made during the window shows `Submitting` until 12:00 UTC and is not run. Announced in
  public issue #18; now also in the README's schedule section and in `SUBMISSION_CLI.md`.
  Scoring, limits and the submission contract: unchanged.
- **Final tie-break in `SUBMISSION_CLI.md`.** Its schedule paragraph now carries the sentence the
  README has carried since 2026-09-22: if two Final submissions finish this track with the same
  ranking score, the one uploaded earlier ranks ahead. No rule change.
- **Hub guide links moved to `main`.** The Development runtime guide links in `README.md` and
  `SUBMISSION_CLI.md` pointed at the toolkit's `v2.4.4` copy, which still states the withdrawn
  allowance of 1,000,000 input tokens per unit and lacks the tie-break and `Failed`-upload
  sentences. The House model guide links in `docs/NVIDIA-STACK.md` pointed at `v2.4.3`, whose
  copy still lists an input-token limit among the House limits and lacks the note that
  `low_effort` and `reasoning_budget` pass through unchanged. All of them now point at the
  guides' `main` copies, like the submission-limits link already did. The image-submission,
  descriptor and team-claim links stay on `v2.4.4`; those guides have not changed since.
- **`docs/FORECAST-RESOLUTION-CANDIDATE.md`** said no public toolkit tag carries the module the
  opt-in candidate API needs. Toolkit `v2.4.4`, which this repository pins, carries
  `qfbench2_common.contracts.forecast_protocol`, and public CI installs `v2.4.4` and runs the
  candidate's tests. The same outdated statement in a `.github/workflows/ci.yml` comment is
  corrected. Documentation and comments only.
- **`docs/NVIDIA-STACK.md`, "Where the tooling lives":** an "(In review)" line linked two public
  pull requests that are unrelated documentation fixes. It now links the repo-root `Dockerfile`
  (the reference submission image), the README's end-to-end run section and
  `docs/SOLVER-PLAYBOOK.md`. The playbook's composite formula now names the tail term as the
  pinball loss at the 1/5/95/99% quantiles, as `docs/CONCEPTS.md` states, instead of "PIT
  calibration".
- **House wording.** Three passages in `README.md` and `SUBMISSION_CLI.md` said the House route
  is budgeted "per run"; the budget is per unit (25 admitted requests), as rule 5 of
  `SUBMISSION_CLI.md` states. `MODEL_NAME` is now described as the runtime alias of the House
  model, with the model identity and snapshot for `models[]` in the House model guide, instead of
  "the pinned house-model id". No rule changes.
- **This file.** Sections are now dated by when their changes reached public `main`. The
  2026-09-23 changes it listed as unreleased are dated, and the missing entries for 2026-09-16 to
  2026-09-23 are added. Two rulings of 2026-09-21 that were filed under the 2026-09-14 baseline
  moved to 2026-09-22, and that baseline's House-allowance entry is restored to its published
  wording. Resolved items are removed from "Still to reconcile".

## 2026-09-23 — public `main` at `f84ad29`

**Track scorer code and evaluation cards: unchanged.** These are documentation and build-pin
corrections. They change no scoring path, no gate, no card and no published number.

- **An upload the platform marks `Failed` does not consume a Development attempt** — the
  platform's daily count excludes it. Held and cancelled uploads still count. (`README.md`,
  `SUBMISSION_CLI.md`; the latter's submission-limits link now points at the hub guide's `main`
  copy.)
- **Toolkit pin moved to `v2.4.4` everywhere.** The 2026-09-18 bump reached the README's install
  commands but not `.github/workflows/ci.yml` or the `Dockerfile`, so CI and the reference
  submission image kept validating against `v2.4.2` — whose category enum still contains
  `byo-large` and `byo-small`, the two values the same ruling withdrew. A descriptor naming one
  of them packs cleanly under `v2.4.2`, is then held at organizer intake and never runs, and
  still costs a Development attempt, with no local signal that anything was wrong. The README
  prose also still told you to pin `v2.4.2` while the command beside it installed `v2.4.3`. The
  2026-09-21 bump to `v2.4.4` fixed `ci.yml` and that prose but again missed the `Dockerfile`. All
  three now name `v2.4.4`, and a new stdlib-only CI step fails the build if they disagree again.
- **`README.md` container-environment table corrected.** It described `MODEL_ENDPOINT` as
  already carrying `/v1` and never mentioned `MODEL_TOKEN` at all. `MODEL_ENDPOINT` is the route
  origin with no path, the OpenAI-compatible API is served under `/v1`, and a request without
  `Authorization: Bearer $MODEL_TOKEN` is refused 401 — so an agent built from that table alone
  failed every House call. The 2026-09-17 House-route correction reached `SUBMISSION_CLI.md` and
  `docs/NVIDIA-STACK.md` and missed this table. It now also states `NO_PROXY` and the House
  request allowance, and names `SUBMISSION_CLI.md` as the binding version instead of maintaining
  a second full copy that can drift.
- **`docs/ARTIFACT-POLICY.md` aligned with the 2026-09-18 ruling.** Its executive summary still
  said Track 2 permits artifacts "in both submission categories" and that "the adapter-only rule
  governs language-model serving", and a later paragraph still asserted a one-adapter limit on a
  bring-your-own language-model path — each contradicting the ruling stated in the same file and
  in `README.md`. This was the last file in the tree carrying the withdrawn regime, and it is the
  file `README.md` designates as the authority on what may be packaged. The permitted-artifact
  table is unchanged; what changes is the description of the regime around it. Policy revision
  stamp moved to 2026-09-21.1. The category sentence this file shares with `README.md` and
  `SUBMISSION_CLI.md` also loses its leftover "for this non-adapter path", in all three files.

## 2026-09-22 — public `main` at `8da3c76`

**Scoring formula and scores: unchanged.**

- **House model budget: requests per unit** (ruling of 2026-09-21). 25 admitted requests per
  unit and at most 4,000 output tokens per call, both counted by the House route. The allowance
  of 1,000,000 input tokens per unit that `README.md` and `SUBMISSION_CLI.md` stated from
  2026-09-16 is withdrawn and **nothing replaces it**: there is no per-unit token allowance. An
  admitted request is charged before forwarding, so an upstream failure, a lost response or a
  retry can spend a slot; a request refused before admission costs nothing. (`README.md` "House
  API allocation", `SUBMISSION_CLI.md` rule 5, `docs/NVIDIA-STACK.md`.) (Answers the last open
  item of issue #2.)
- **Ties in the ranking score** (ruling of 2026-09-21): if two Final submissions finish this
  track with the same ranking score, the one uploaded earlier is ranked ahead. The ruling covers
  the Final ranking. (`README.md`.)

## 2026-09-21 — public `main` at `4a14af5`

Reached public `main` in two pushes: `4ee4069` at 01:18 UTC and `4a14af5` at 15:36 UTC.

**Track scorer code changed; scores unchanged for a valid organizer bundle.** Evaluation cards
unchanged.

- **`docs/M0-BASELINE.md` specifies M0**, the text-blind baseline every card's score is
  normalized against: the procedure end to end and the per-card seed, with a worked example on
  `t2-EXAMPLE-ust-curve-1m`. The generator source and the per-card scale values stay sealed. The
  README's "Firewall: what is sealed" now names `reference/ref_scale.json` as answer-equivalent.
  Scoring unaffected. (Answers part of issue #3.)
- **How an upload is made:** an upload is the zip written by `qfbench2 submission pack`,
  uploaded on the track's CodaBench page, not an image reference. New `SUBMISSION_CLI.md` section
  "How an upload is made" and README checklist step 8. Scoring unaffected.
- **Scale-commitment check on the official scoring path.** `score_roster` now reads every card's
  `reference/ref_scale.json` once, before any participant gate, checks those bytes against the
  evaluation plan's `ref_scale_commitment`, and scores from the verified bytes. A missing, linked
  or mismatched scale file aborts the run as an organizer fault. For a valid organizer bundle
  nothing changes: the same bytes are parsed by the same rules and the scoring arithmetic is
  untouched. This was checked by re-scoring synthetic evaluations with the scorer before and
  after the change (identical results), and the regression suite's pinned composites are
  unchanged; that is a check, not a proof for every input. Package version stays `3.1.0`.
- **Opt-in candidate forecast-resolution API** (`qfbench2_track_forecasting.resolution`,
  `docs/FORECAST-RESOLUTION-CANDIDATE.md`), with opt-in precise-cutoff and strict-coverage
  options in `cutoff.py` that only this module uses. Nothing on the live scoring path imports
  it, and every result it returns is non-rankable. No scoring change.
- **Toolkit pin `v2.4.4`** in CI, in the README install commands and in the tag-pinned hub guide
  links of `README.md` and `SUBMISSION_CLI.md`; the `Dockerfile` was missed and fixed on
  2026-09-23. `v2.4.4` adds organizer-side contracts, including
  `qfbench2_common.contracts.forecast_protocol`, which the candidate API above needs. The
  toolkit's `starter-packs/CHANGELOG.md` says when an existing `v2.4.3` install has to move.

## 2026-09-18 — public `main` at `46f40a4`

**Scoring formula: unchanged.**

- **Bring-your-own models and adapters are out of scope** (ruling of 2026-09-18, announced on
  issue #5): `api` is the only submission category on this track. Toolkit `v2.4.3` and later
  refuse to pack a `byo-large` or `byo-small` descriptor, and an upload that still carries one is
  held at the organizer's intake and never run. `README.md`, `SUBMISSION_CLI.md` and
  `docs/NVIDIA-STACK.md` no longer offer a BYO route; `docs/ARTIFACT-POLICY.md` was finished on
  2026-09-23. The README install commands and the tag-pinned hub links moved to `v2.4.3`;
  `ci.yml` and the `Dockerfile` did not, and were fixed on 2026-09-21 and 2026-09-23.

## 2026-09-17 — public `main` at `72ae1f0`

**Scoring: unchanged.** Out-of-cycle correction, announced on issue #4.

- **House route contract.** `MODEL_ENDPOINT` is the route origin with no path; the
  OpenAI-compatible API is served under `/v1` (`POST $MODEL_ENDPOINT/v1/chat/completions`); every
  request needs `Authorization: Bearer $MODEL_TOKEN`. The container-environment table in
  `SUBMISSION_CLI.md` and the House row of `docs/NVIDIA-STACK.md` now say so and link the hub
  guide's "Calling the House route". The README table was missed and fixed on 2026-09-23.

## 2026-09-16 — toolkit `v2.4.2` release, public `main` at `53a414c`

**Track scorer code: unchanged.** These changes updated toolkit install pins, the reference
image, the reference forecast producer, documentation, the four monthly practice specs and
synthetic regression tests. They did not update the deployed scorer. The local toolkit
correction inherited by the pin is noted below. The same release also published the Development
schedule, resource and submission-limit sections of `README.md` and `SUBMISSION_CLI.md`, and
stated a House input allowance of 1,000,000 input tokens per unit there (withdrawn on
2026-09-22, see above).

- **Toolkit installation:** align the reference Docker image, README and CI at `v2.4.2`,
  including the current submission-packaging command and corrected model-free simulation
  fixture. The image previously installed `v2.3.1`, which refuses an empty `models` array even
  when the evaluation verifier accepts it. (Superseded: see the `v2.4.4` entry above — the
  `v2.4.3` bump reached the README and not `ci.yml` or the `Dockerfile`.) The pin also includes
  the local CRPS correction already released in toolkit `v2.4.1`: a component with zero weight
  and exactly zero reference scale contributes zero instead of poisoning the composite with
  `NaN`. See the toolkit starter-pack changelog; this patch adds no scoring implementation.
- **Diagnostic documentation:** remove remaining claims that the scorer reports information
  uplift, text-ablation results, PIT or interval coverage. The scorer reports the composite
  and its components; participants can run separate diagnostics on labeled data they may use.
- **Log-return regression tests:** extend the existing synthetic fixture to 21- and 127-day
  horizons. Check both the CLI and shared baseline against cumulative `log(1 + r)` returns,
  an exact zero anchor, and the complete history window. This changes the synthetic fixture's
  expected result, not scoring code or the daily/monthly regression expectations.
- **Monthly target periods** (answers issue #2, horizon semantics): the four monthly practice
  specs carry explicit `observation_periods`; the horizon integer stays the submission's grid
  key. The reference CLI and reasoning example count monthly steps from the last available panel
  observation, including publication lag (`docs/MONTHLY-HORIZONS.md`). The example card and the
  card template now set `target_frequency = "daily"`, the cadence of target observations rather
  than the forecast lead time. The scorer joins on `(asset, horizon)` and never converts a
  horizon, so scoring is unaffected.

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
- The synthetic `reg-t2-logreturn` fixture checks both producers. It used a 21-day horizon at
  that point; the 127-day horizon was added in the 2026-09-16 release above.

### Scoring code

- The single-cell `ref_scale` relaxation uses the card's joint statistic: the joint term is
  dropped on a one-cell grid only for the variogram, which is zero by construction. A one-cell
  card declaring another joint statistic is refused as an organizer fault instead of silently
  discarding a defined component.

### Rules and documentation

- **House API allowance:** 25 requests per unit, at most 4,000 output tokens per call.
  Participant vendor API keys are not supported. The earlier "1,000,000 input + 100,000
  output tokens per unit" wording is withdrawn. Operational input limits and failed-request
  or retry handling are separate from this kit changelog; this entry makes no change to them.
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
