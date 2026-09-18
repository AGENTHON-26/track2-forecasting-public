# The NVIDIA stack and Track 2

How the NVIDIA technology stack maps to Track 2, and how participants should (and should not) use
it. Track 2 asks your agent to turn a frozen numeric panel plus a dated official-document corpus
into a **calibrated probabilistic forecast**, scored offline against realized outcomes
([CONCEPTS.md](CONCEPTS.md)). The NVIDIA stack enters this track through exactly one door: the
**organizer-hosted house-model endpoint** — an open model from the sponsor's Nemotron family,
served behind `$MODEL_ENDPOINT`. Nothing GPU-shaped in the stack affects your rank.

## Framing: leveled by the model, not the metal

Track 2 is leveled the opposite way from Track 3. There, every submission gets identical hardware
and no runtime LLM; here, the LLM is the centerpiece of the loop and the GPU never touches your
rank: no scored component measures hardware. Since the 2026-08 caps change every T2 card grants
the same generous sandbox — 16 vCPU / 128G / `gpu = true`. An `api` submission has no use for
that GPU in its own code. Using or ignoring it moves no score. The network mode is `restricted` — model calls through the audited eval proxy
only, **never data fetching** ([SUBMISSION_CLI.md](../SUBMISSION_CLI.md), "Network modes") — and
official scoring happens organizer-side against sealed realized outcomes, with a scorer that
needs no network at all (local smoke runs use `--network=none`). The house endpoint is free and
identical for everyone (exposed as `$MODEL_ENDPOINT` when available), so what separates
submissions is **elicitation and calibration skill** — how much verifiable predictive signal you
extract from the corpus, and how honestly you spread your uncertainty. Bring-your-own models and
adapters are not part of this competition (ruling of 2026-09-18): the image never carries
language-model weights or an adapter and never runs a model server. Permitted local numerical
artifacts are covered by the [artifact policy](ARTIFACT-POLICY.md); see
[SUBMISSION_CLI.md](../SUBMISSION_CLI.md).

## Per-tool fit

| Tool | Fit for T2 | How to use it | Caveat |
|---|---|---|---|
| **Nemotron behind `$MODEL_ENDPOINT`** | **Core — the one NVIDIA component in your loop** | An OpenAI-compatible chat endpoint — `$MODEL_ENDPOINT` is the route origin, the API is under `/v1` (`POST $MODEL_ENDPOINT/v1/chat/completions`), and the per-unit bearer arrives as `$MODEL_TOKEN`; see [Calling the House route](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/HOUSE-MODEL.md#calling-the-house-route); use the supplied `$MODEL_NAME` runtime alias. The selected model and snapshot are documented in the [House model guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.4.3/docs/HOUSE-MODEL.md). Use it as your **text-reader**: extract stance, dates, revisions and surprises from the corpus, then let them adjust a statistical prior ([CONCEPTS.md](CONCEPTS.md) on text ablation) | Thinking is enabled by default; low-effort reasoning is off by default. Disable thinking per request with `chat_template_kwargs.enable_thinking=false`, as shown below. The published scorer does not run a closed-book recall comparison; a score alone does not establish whether the agent used text or recalled an outcome. Follow the as-of and no-answer-lookup rules in the [README](../README.md#leakage-rules). |
| **NeMo (customization / fine-tuning)** | **No fit** | — | Language-model adaptation has no submission path: bring-your-own models and adapters are not part of this competition. Permitted numerical fitting is distinct from language-model adaptation, and the NeMo customization stack ships in no T2 image |
| **Megatron-LM** | **No fit** | — | A full-weight training stack for a competition in which no participant-trained language model can be submitted. GPU-mandatory with a multi-GB dependency closure and no time-series or forecasting code; it is never part of a T2 image |
| **CUDA / RAPIDS / cuDF** | **No fit** | — | T2's scored pipeline has no GPU surface: solving is file I/O + endpoint calls + sampling, and scoring is CPU CRPS arithmetic, so vendoring GPU libraries only bloats your image |
| **Nsight / DCGM** | **No fit** | — | T2 has no throughput or efficiency component — nothing to profile, nothing to meter (these are Track 3 concerns) |
| **NeMo Guardrails** | **No fit** | — | A participant-side self-check rail relevant only to Track 4's citation surface; T2's admissibility gates are numeric and schema-level, and the rationale review reads your reasoning, not your I/O |

## House model and thinking controls

Track 2 Development uses NVIDIA Nemotron 3 Super 120B-A12B, FP8: model identity
`nvidia/nemotron-3-super-120b-a12b`, reported model/tokenizer snapshot `rl-030326-fp8`.
Use the injected `$MODEL_NAME` for calls; the runtime alias is `house`. The Lightning model
mentioned in a baseline example is not the selected House model. The training cutoff remains
unpublished, as stated in the shared [House model guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.4.3/docs/HOUSE-MODEL.md).

Thinking is on by default and low-effort reasoning is off by default. To disable thinking for
one request with the OpenAI Python client, pass:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

For raw HTTP JSON, put `chat_template_kwargs` at the top level of the request body.
Omitting this option keeps thinking enabled. Use this API option instead of the older system-prompt
toggle. These settings were verified against the selected chat template and synthetic request
rendering on September 16, 2026; this verification does not announce participant access or change
request/token allowances. See the shared guide for the public checkpoint and the limits of local
reproducibility.

## What this means concretely

- **To be admissible:** pass the deterministic gates — g0 integrity, g1 schema (including a
  non-empty `forecast_rationale.md`: required, never scored), g2 cutoff/resource, g3 domain
  semantics. No NVIDIA tool helps or hurts here.
- **To rank well:** lower the composite — 50 % marginal CRPS + 30 % joint variogram + 20 % tail
  penalty, lower is better ([CONCEPTS.md](CONCEPTS.md)). The house endpoint is your only lever
  beyond your own statistics: better elicitation of the corpus, better-calibrated spread.
- **What is actually measured:** the composite and its components. The five named baseline
  adapters are Gaussian-random-walk scaffolds, not measured implementations of the named models.
  The scorer does not emit information-uplift or text-ablation metrics, or a recall-specific
  score. To study the contribution of text, run a separate controlled ablation on labeled data
  you are permitted to use; see [CONCEPTS.md](CONCEPTS.md).

## Where the tooling lives

- Endpoint contract (`MODEL_ENDPOINT`, `MODEL_NAME`), network modes, and the
  submission-categories table: [SUBMISSION_CLI.md](../SUBMISSION_CLI.md); network modes and
  categories are also summarized in the repo [README.md](../README.md).
- Scoring semantics and the composite: [CONCEPTS.md](CONCEPTS.md); task families:
  [CATEGORIES.md](CATEGORIES.md).
- The rationale requirement and how reviewers read it: [RATIONALE-REVIEW.md](RATIONALE-REVIEW.md).
- *(In review)* a reference end-to-end submission — baseline CLI + gate-passing image
  ([#10](https://github.com/Agenthon-2026/track2-forecasting-public/pull/10)) — and a solver
  playbook distilling what measurably works
  ([#11](https://github.com/Agenthon-2026/track2-forecasting-public/pull/11)); both will be
  linked here once merged.
