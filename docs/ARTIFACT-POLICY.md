## Executive summary (read this first)

Track 2 permits ordinary numerical forecasting code and the limited local artifacts below in
both submission categories. The adapter-only rule governs language-model serving; it does not
ban fitted non-neural forecasting or calibration parameters. Every artifact must respect the
task's information cutoff and the no-answer-lookup rule. Extra pretrained neural checkpoints
require separate organizer approval. This clarification changes eligibility documentation,
not the descriptor schema, scoring formula, resource grants or platform availability.

**Policy revision: 2026-09-10.1.** Read alongside the [submission interface](../SUBMISSION_CLI.md)
and [competition data rules](https://www.agenthon.net/rules/).

## What you may package

A Track 2 forecaster may use permitted numerical code without calling the House model. Use `category: "api"` for this non-adapter path; House calls are optional. Use `models: []` only when the submission contains no learned model. Disclose any packaged fitted model with `access: "local"`, its immutable revision and training cutoff; include the House disclosure when used. The existing artifact, data-cutoff and resource rules still apply.

For example, an agent may combine the unchanged House model with a fitted linear forecast and
a covariance estimate. That remains an `api` submission. An agent using the approved LoRA
adapter plus those same numerical artifacts uses a BYO category.

| Artifact | Allowed in `api` and BYO-adapter images? | Required qualification |
|---|---|---|
| Statistical forecasting code, including Theta, AutoARIMA and StatsForecast methods | Yes | Pin package/source versions. A package name does not authorize every checkpoint it can load. |
| Fitted non-neural linear, tree or gradient-boosted-tree models; calibration and covariance parameters | Yes | Disclose actual learned models and the fitting, selection and calibration data. No future information or stored unit answers. |
| Static BM25 retrieval indexes, dictionaries and tables | Yes | Build from permitted data available by the current task cutoff; retrieve only information allowed for that task. No stored answer lookup. |
| Tokenizer-only vocabulary/configuration | Yes | Record the source and immutable version. This allowance includes no additional pretrained neural weights. |
| Pretrained neural time-series, embedding or reranker checkpoints, including Chronos and TimesFM weights | Separate approval required | Being a non-LLM or an auxiliary component does not itself make a checkpoint eligible. |

The one-adapter limit still applies to the language-model path. Full language-model weights,
a second language model or adapter, and a participant-run model server remain outside that
path. All local artifacts share the task's existing resource limits; this policy adds no GPU,
disk, memory or network grant. A permitted category is not a promise that its submission
service is open; use the separately announced access instructions.

## Data cutoff and provenance

Data must be public, lawfully obtained, properly licensed and permitted by the track.
Nonpublic employer/sponsor data and embargoed data require express written authorization.
Features, labels and derived material used for fitting, adaptation, model selection and
calibration must have been available by each relevant task cutoff. A later revised release of
a historical series is not automatically information available at that historical date.
Apply the same cutoff to indexes and caches. Do not fetch new data at evaluation time.

Keep an `ARTIFACT_PROVENANCE.md` record with the source used to build your image, available
for organizer verification. Record each artifact's source/version, license, first-availability
dates, immutable revision or checksum, and which data were used for fitting, selection and
calibration. Explain how the chosen artifact respects the cutoff for each task it handles.
This is review documentation; the descriptor validator does not read or certify its contents.

Use the existing `models[]` entries for actual learned models, with `access: "local"` for a
bundled fitted model and the actual training cutoff. Keep the House/base disclosure too when
it is used. Identify an adapter's own training data in the provenance record as well as the
base's disclosed cutoff. Pure code, dictionaries and tokenizer-only assets belong in the
provenance record, not fictitious model entries. Use `models: []` only if the submission is
genuinely model-free, with the C5 1.1 interface described in the [descriptor guide](../SUBMISSION_CLI.md).
The descriptor is closed: do not add provenance fields, and do not add a sidecar to its upload
ZIP unless the upload instructions explicitly request it. Schema acceptance alone does not
establish artifact or data eligibility.

## Narrow exception for the required base model

Only the exact organizer-approved Nemotron base revisions identified in the model-access
release receive an exception for their general-purpose pretraining on historical tasks.
The exception does not cover task-specific fitting, adaptation, model selection, calibration
or additional data after a task cutoff. Declaring another model's training cutoff does not
qualify it. Use the approved revision list before treating any particular base as covered.
