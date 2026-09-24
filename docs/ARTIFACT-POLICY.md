## Executive summary (read this first)

Track 2 permits ordinary numerical forecasting code and the limited local artifacts below.
`api` is the only submission category on this track, and the House route is the only
language-model serving there is — bring-your-own models and adapters are not part of this
competition (ruling of 2026-09-18). Neither rule bans fitted non-neural forecasting or
calibration parameters. Every artifact must respect the task's information cutoff and the
no-answer-lookup rule. Extra pretrained neural checkpoints require separate organizer approval.
This clarification changes eligibility documentation, not the descriptor schema, scoring
formula, resource grants or platform availability.

**Policy revision: 2026-09-21.1.** Read alongside the [submission interface](../SUBMISSION_CLI.md)
and [competition data rules](https://www.agenthon.net/rules/).

## What you may package

A Track 2 forecaster may use permitted numerical code without calling the House model. Use `category: "api"`; House calls are optional. Use `models: []` only when the submission contains no learned model. Disclose any packaged fitted model with `access: "local"`, its immutable revision and training cutoff; include the House disclosure when used. The existing artifact, data-cutoff and resource rules still apply.

For example, an agent may combine the unchanged House model with a fitted linear forecast and
a covariance estimate. That remains an `api` submission — the only category on this track, since
bring-your-own models and adapters are not part of this competition (ruling of 2026-09-18).

| Artifact | Allowed in an `api` image? | Required qualification |
|---|---|---|
| Statistical forecasting code, including Theta, AutoARIMA and StatsForecast methods | Yes | Pin package/source versions. A package name does not authorize every checkpoint it can load. |
| Fitted non-neural linear, tree or gradient-boosted-tree models; calibration and covariance parameters | Yes | Disclose actual learned models and the fitting, selection and calibration data. No future information or stored unit answers. |
| Static BM25 retrieval indexes, dictionaries and tables | Yes | Build from permitted data available by the current task cutoff; retrieve only information allowed for that task. No stored answer lookup. |
| Tokenizer-only vocabulary/configuration | Yes | Record the source and immutable version. This allowance includes no additional pretrained neural weights. |
| Pretrained neural time-series, embedding or reranker checkpoints, including Chronos and TimesFM weights | Separate approval required | Being a non-LLM or an auxiliary component does not itself make a checkpoint eligible. |

The language-model path is the House route and nothing else. Language-model weights, adapters
of any rank, and a participant-run model server are all outside this competition; a submission
that packages one is not permitted, whether or not it is disclosed. All local artifacts share the task's
existing resource limits; this policy adds no GPU, disk, memory or network grant. A permitted
artifact is not a promise that the submission service is open; use the separately announced
access instructions.

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
bundled fitted model and the actual training cutoff. Keep the House-model disclosure too when it
is used, together with that model's disclosed cutoff. Pure code, dictionaries and tokenizer-only
assets belong in the provenance record, not fictitious model entries. Use `models: []` only if the
submission is genuinely model-free, with the C5 1.1 interface described in the
[descriptor guide](../SUBMISSION_CLI.md).
The descriptor is closed: do not add provenance fields, and do not add a sidecar to its upload
ZIP unless the upload instructions explicitly request it. Schema acceptance alone does not
establish artifact or data eligibility.

## Narrow exception for the House model's pretraining

Only the exact organizer-approved Nemotron revisions served on the House route and identified
in the model-access release receive an exception for their general-purpose pretraining on
historical tasks. The exception does not cover task-specific fitting, adaptation, model
selection, calibration or additional data after a task cutoff. Declaring some other model's
training cutoff does not qualify that model — and since bring-your-own is out of scope, no
other language model may be packaged at all. Use the approved revision list before treating a
particular revision as covered.
