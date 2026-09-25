# Submitting — Track 2

Everything needed to turn this repo into a CodaBench upload: what goes in the image, how to build
and push it, how to seal the descriptor, and what is still open. The rules themselves live in
[`SUBMISSION_CLI.md`](SUBMISSION_CLI.md) and the organizer guides it links; this file is the
procedure and the reasoning behind it.

**Deadlines.** Development closes **2026-10-12 23:59 AoE**. The joint Final + Verification phase
runs **2026-10-13 – 2026-10-25**, closing 23:59 AoE, and each team makes **one** Final submission
per track.

**Budget.** Track 2 gets **5 uploads per team per day, 20 total** across Development. Held or
cancelled uploads still count; only a platform-marked `Failed` does not. Local validation, builds
and packing cost **nothing** — run them as often as you like. Ties on the Final leaderboard break
in favour of the earlier upload.

---

## 1. What is in the image, and why

Three files: `forecast_agent.py`, `text_signal.py`, and `f1_pipeline/m2_unit.py`. Nothing else —
not `f1_pipeline/data/` (14 MB of notebook/backtest output; `m2_unit.py` reads nothing from it at
run time, only the unit's own panel under `/input`), not the notebooks, not `f1_pipeline/README.md`.

`forecast` is a `/usr/local/bin` shim that execs `python3 /opt/forecast_agent.py "$@"`. There is
deliberately **no `ENTRYPOINT`**: the harness invokes
`docker run <image> forecast --panels … --text … --asof … --out …`, so the verb arrives as the
first argument, and with no ENTRYPOINT the shell resolves it from `PATH` and it never reaches
argparse. Neither `forecast_agent.py` nor the reference CLI accepts a leading `forecast`
positional — both would exit 2 on every unit — so if anyone ever adds an `ENTRYPOINT`, they must
also add a positional argument to absorb the verb.

**The reference CLI (`qfbench2_track_forecasting`) is not in the image.** Its own docstring says it
"ignores `--text` entirely"; shipping it would score the text-blind floor.

**`pandas` is back, for a different reason than it left.** It was first dropped alongside the
reference CLI (its only importer at the time), then re-added when `forecast_agent.py` grew an M2
dispatch: `T2-F1` level cards now fit through `f1_pipeline/m2_unit.py` (ridge location-scale +
empirical residual pool) instead of the shared random walk, and both `forecast_agent.py` and
`m2_unit.py`'s own `read_unit()` (`pd.read_parquet`) need it. Other families still use the random
walk and never touch pandas at runtime, but the dependency is unconditional in the image since
`forecast_agent.py` imports `f1_pipeline.m2_unit` regardless of which unit it happens to run.
`f1_pipeline` needs no `__init__.py` — it resolves as an implicit Python namespace package under
`PYTHONPATH=/opt`, since only the one file is copied in.

**`qfbench2-common` stays in the Dockerfile** even though the agent never imports it.
`.github/workflows/ci.yml` greps this Dockerfile for `refs/tags/vX.Y.Z.tar.gz` and fails the build
with "could not find a pinned toolkit install" if the line disappears.

**The build-time import check covers all three modules, not just two, and asymmetrically.**
`forecast_agent.read_text_signal` catches any exception from the `text_signal` import and falls
back to neutral — a missing `text_signal.py` still builds, runs, exits 0, passes every gate, and
silently scores text-blind. `f1_pipeline.m2_unit` has no such fallback: a missing or broken import
there raises inside `build_draws()` and fails every `T2-F1` level-card unit outright. Both belong
in the same `RUN python3 -c "..."` check for the same underlying reason — fail loud at build time,
not silently (`text_signal`) or expensively (`m2_unit`, unit-by-unit) during scoring.

**`.dockerignore` is an allowlist**, not a denylist. `.env` holds `MODEL_API_KEY` and sits at the
repo root, so a denylist that forgets one line publishes a credential to a public registry. Because
`f1_pipeline` is a directory, allowing just its one file takes three lines, not one: `!f1_pipeline`
lets Docker traverse into the directory at all (the leading `*` stops it there), `f1_pipeline/*`
re-excludes everything inside it, then `!f1_pipeline/m2_unit.py` allows that one file back in. This
cuts the build context from ~470 MB (`.venv` 422 M, `units/` 44 M, `f1_pipeline/data/` 14 M) to
three files — pull time is charged against the unit clock.

## 2. One-time setup

```bash
# The pinned toolkit. 2.3.1 has NO `submission` subcommand at all -- if `qfbench2 submission
# --help` errors with "invalid choice", this is why.
pip install "qfbench2-common[data] @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.4.4#subdirectory=common"
pip install .            # this repo's own deps, for the scorer
pip install pytest       # the test suite needs it; CI runs `python -m pytest tests`
```

### A GitHub token for GHCR

`gh` CLI is not installed here, so make the token in a browser:
**https://github.com/settings/tokens** → *Tokens (classic)* → **Generate new token (classic)**.

- Expiration past **2026-10-25** (Final phase close)
- Scope: **`write:packages`** (auto-selects `read:packages` and `repo`)

Copy the `ghp_…` value — GitHub shows it once. Then:

```bash
export CR_PAT=ghp_xxxxxxxxxxxxxxxxxxxx
echo "$CR_PAT" | docker login ghcr.io -u <your-github-username> --password-stdin
```

Use a **personal** username. `AGENTHON-26` is the org holding the repo, not the image registry.

### Your team credentials

Team Number and Team Key come from your agenthon.net account. The Key is never typed into a
command line, never stored, and never enters the zip — the archive carries an HMAC proof instead.

## 3. Pre-flight — before spending an upload

Build locally for amd64 and load it into Docker (no push):

```bash
docker buildx build --builder qfb2 --platform linux/amd64 -t t2-forecast-agent:preflight --load .
```

If the `qfb2` builder does not exist yet:
`docker buildx create --name qfb2 --driver docker-container --bootstrap`. The default builders use
the `docker` driver, which cannot push and cannot cross-build cleanly.

Then run a unit offline, exactly as the harness will (`-v <unit-dir>:/input:ro` — the unit
directory *itself* is mounted at `/input`, so `panels/` and `text/` are subdirectories, not
separate mounts):

```bash
docker run --rm --network=none --platform linux/amd64 \
  -v "$PWD/units/t2-EXAMPLE-ust-curve-1m:/input:ro" -v /tmp/pf:/output \
  t2-forecast-agent:preflight \
  forecast --panels /input/panels --text /input/text --asof 2024-06-28 --out /output/forecast.parquet

python scoring/scoring.py score --card units/t2-EXAMPLE-ust-curve-1m/card.toml --forecast /tmp/pf/forecast.parquet
qfbench2 smoke units/t2-EXAMPLE-ust-curve-1m /tmp/pf --track forecasting
```

Three units worth covering, because they exercise different shapes:

| Unit | `--asof` | Covers |
|---|---|---|
| `t2-EXAMPLE-ust-curve-1m` | 2024-06-28 | 4 assets × 1 horizon — the joint/co-movement path |
| `t2-F1-cad-boc-2017` | 2017-07-12 | 1 asset × 2 horizons `[126, 189]` — the **M2** path (`f1_pipeline/m2_unit.py`), not the random walk; check the rationale says `Base: M2 --` |
| `t2-F4-covid-mkt-2020` | 2020-02-19 | tail family, random-walk path |

**Pass means:** exit 0; all three of `forecast.parquet`, `forecast_meta.json`,
`forecast_rationale.md` present with the rationale non-empty; `"admissible": true` with
`g0_integrity`, `g1_schema`, `g2_cutoff_resource`, `g3_domain_semantics` all `pass`; and
`qfbench2 smoke` reporting `admissible=True … labels=[]`.

**Read the stderr, it distinguishes two very different things:**

- `MODEL_ENDPOINT is unset` / `no usable summaries; neutral` — **expected** under `--network=none`.
  The agent fell back to neutral because there was no endpoint, which is correct behaviour.
- `[text_signal] unavailable` — **stop**. The module failed to *import*. The run will still exit 0
  and pass every gate while silently forecasting text-blind.

For the live path, drop `--network=none` and add `--env-file .env`. Success looks like
`[text_signal] source=llm family=F4 docs=8 assets=['MKT']`, and the rationale ends `Text used: yes.`

## 4. Build, push, and get the digest

```bash
tools/build_and_push.sh <your-github-username>
```

One script does: amd64 build → push → digest cross-check → anonymous-pull check. Three of its
flags are not optional, and the comments in the script say why:

- `--platform linux/amd64` — the harness pulls amd64. A plain `docker build` on Apple Silicon
  produces arm64, which fails every unit.
- `--builder qfb2` — the default `docker`-driver builders cannot push.
- `--provenance=false --sbom=false` — with attestations on, buildx pushes an OCI *index* (amd64
  plus an `unknown/unknown` entry) and the tag digest is the index digest, not the amd64 manifest
  digest the descriptor wants.

The digest is read from `--metadata-file` and cross-checked against `imagetools inspect`, never
copied off the terminal by eye.

**The first run fails at the anonymous-pull check with HTTP 401.** That is expected: GHCR packages
are private on creation. Open the URL the script prints —
`https://github.com/users/<you>/packages/container/t2-forecast-agent/settings` → *Danger Zone* →
**Change visibility** → **Public** — then re-run. The build is cached, so it is quick, and this
time you get `anonymous pull: OK (HTTP 200)` and the digest.

A logged-in `docker pull` proves nothing about public access, which is why the script fetches an
anonymous `ghcr.io/token` and does a credential-free manifest `HEAD`.

## 5. Descriptor, pack, upload

> **These two commands need a real TTY.** `qfbench2 submission` refuses to run when stdin is not a
> terminal (`no terminal to hide the Team Key prompt; use --team-key-file`). Run them in your own
> terminal — an agent session cannot. The `--team-key-file` escape hatch takes a mode-600 file
> holding the key alone.

```bash
# 1. derive your team_id (prompts "Team Key (hidden):", prints one line and nothing else)
qfbench2 submission alias --team-number <N>

# 2. build and seal the descriptor
python3 tools/make_descriptor.py \
    --team-id team-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
    --repository <your-github-username>/t2-forecast-agent \
    --digest sha256:<from step 4> \
    --out submission.json

# 3. pack (prompts for the Team Key again)
qfbench2 submission pack --descriptor submission.json --team-number <N> --out submission.zip

# 4. confirm what you are about to upload
unzip -p submission.zip submission.json | jq -r '.team_id, .image.digest, .category, .phase'
```

Step 4 must echo your team id, the digest from §4, `api`, `dev`. Then upload `submission.zip` on
the Track 2 CodaBench page **from the team's single designated account**.

For a **Final** submission, pass `--phase final` to `make_descriptor.py`; it sets
`competition_id` to `agenthon2026-forecasting-final` to match.

### Why the descriptor is generated, not hand-written

`tools/make_descriptor.py` fills the fixed Track 2 fields, seals the digest with the toolkit's own
function, then validates twice — through the JSON Schema and through
`SubmissionDescriptor.from_mapping`, which is the authority and catches cross-field problems the
schema cannot see. Hand-writing fails in four ways:

- `descriptor_digest` is a JCS-canonical sha256 over the descriptor **minus that field**, so any
  edit — including a one-character digest fix — invalidates it.
- The schema sets `additionalProperties: false`. One stray key is a rejection, not a warning.
  Legacy keys specifically rejected: `image_digest`, `model_disclosure`, `house_endpoint_only`,
  `open_weights`, `api_domains`.
- `image.digest` must be the pushed **amd64 manifest** digest, not a local image id.
- `team_id` must be the derived alias. `pack` re-derives it and **refuses** a descriptor carrying a
  different one (`descriptor team_id disagrees…`) rather than silently rewriting it.

| Field | Value | Why |
|---|---|---|
| `schema_version` | `1.1.0` | the 2.4.4 contracts module's own `SCHEMA_VERSION`; `1.0.0` also validates |
| `category` | `api` | the only category on this track — 2.4.3 withdrew the `byo-*` pair, and `pack` refuses them |
| `models[]` | the House row, `access: "api"` | declared because the image calls the House route |
| `training_cutoff` | `"unpublished"` | the House guide says declare the literal string; "do not invent a training cutoff" |
| `license` | `MIT` | matches this repo's LICENSE |
| `image_access` | `public` | anything else needs prior written organizer confirmation |

## 6. Failure modes that waste an upload, and what catches each

| Failure | Caught before upload by |
|---|---|
| Image is arm64 | `imagetools inspect` → `linux/amd64`; in-container `platform.machine()` → `x86_64` |
| `text_signal.py` missing → **silent** text-blind scoring | build-time import check; absence of `[text_signal] unavailable` on stderr |
| `f1_pipeline/m2_unit.py` missing → every `T2-F1` level-card unit fails outright | build-time import check (`from f1_pipeline import m2_unit`) |
| Verb unresolvable / shim points at the wrong module | `docker run … forecast --help` exit 0, plus a full offline unit run |
| GHCR package still private | the credential-free `ghcr.io/token` + manifest `HEAD` → 200 |
| Wrong/stale digest, or an attestation-index digest | `--metadata-file` digest == `imagetools inspect` digest; clean-room pull by digest |
| Descriptor rejected at intake | `make_descriptor.py`'s two validation passes |
| Mismatched `team_id` (a held upload still counts) | `team_id` from `submission alias`, re-checked out of the packed zip |
| Missing/blank sidecar → `g1_schema` fails every unit | all three files present, rationale non-empty, `qfbench2 smoke` green |
| Packed with the stale 2.3.1 toolkit | `pip show qfbench2-common` → 2.4.4 before packing |

## 7. Open items

Known and deliberately not addressed in the packaging work. Roughly in order of how much they
cost.

1. ~~The 25-admitted-requests-per-unit budget was unenforced.~~ **Fixed on `dev` (`feac476`),
   picked up here by rebasing.** `text_signal.py`'s `_Budget` now counts every attempt across the
   summarizer's threads, hard-stops at `_REQUEST_BUDGET = 25`, and reserves a slot for the Stage 2
   adjustment call so Stage 1 retries can't starve it. Documents are summarized newest-first, so
   the oldest lose when the budget runs dry; a unit records
   `"request budget exhausted"` on the documents it had to skip. Covered by tests. Worth rechecking
   after any future change to `text_signal.py` — the budget logic and the import self-check in
   §1 verify different things (one counts requests, the other proves the module loads at all).
2. **Text is measured net-negative.** The last clean sweep
   (`eval_reports/stage2-thinking-throttled-live.json`, 103 units, 7h39m) has text-informed
   forecasts behind the text-blind baseline by **+0.0423** mean composite, and **+0.0869** on the
   50 units with zero logged failures. That sweep predates both the request-budget fix (item 1)
   and the M2 model (§1) — the comparison should be re-run before trusting it for a Final decision.
   The image ships text ON regardless — a deliberate choice to exercise the House route in the real
   environment during Development, not an oversight.
3. **No House-call telemetry in the output.** If the proxy or token is wrong in the sealed
   environment, every call fails, `read_text_signal()` returns neutral, and the forecast scores
   text-blind with nothing in `forecast_meta.json` recording that anything broke. One upload
   burned, nothing learned. Recording attempted/succeeded/failed counts would make each Development
   upload a diagnostic.
4. **`forecast_rationale.md` is thin** — a few lines: assets, horizons, base method (now
   distinguishing M2 from the random walk — see §1), and whether text was used. It satisfies
   `g1_schema` (which only checks existence and non-emptiness) and is **never scored**. But it
   feeds a human-review screen over the top of the leaderboard, and it names no anchor, no
   per-adjustment size, and no cited documents — so a reviewer has nothing to check. The reference
   CLI's `_rationale()` (`qfbench2_track_forecasting/cli.py:324-449`) produces the shape the
   contract describes and could be adapted.
5. **`QFBENCH_SEED` is ignored** — the agent uses `--seed`, default 0. The reference CLI does the
   same and T2 verification is statistical (bootstrap-CI overlap), so this is currently fine.
6. **`_series` assumes long-format panels** (`date` / `asset` / `value` columns) and raises
   `SystemExit` otherwise. All 104 local units are long-format, but this is the one input shape
   that would DNF a unit rather than degrade, and it can only be confirmed against a real staged
   unit.
7. **No unit in this repo has a `panels/` subdirectory** — all 104 keep the panel parquet at the
   unit root, and `_read_panels` falls back to `panels_dir.parent`. Staged units really do ship
   `panels/`, so keep passing `/input/panels`; just know local runs exercise the fallback.
8. **Output-dir ownership is unverifiable locally.** If the harness binds `/output` root-owned
   0755, the `USER runner` (uid 1000) process cannot write it. macOS Docker Desktop virtualizes
   bind ownership so it will not reproduce here. Low risk: the shipped reference Dockerfile uses
   the same `USER runner`, so the harness demonstrably accommodates it.
9. **M2's own documented limits carry into the image unchanged** (`f1_pipeline/README.md`): below
   ~1,000 training rows it is worse than a random walk (three short units sit there); the
   calibration constant and residual pool use in-sample residuals, making bands ~9% too narrow;
   and the 1% tails of each residual pool rest on one or two historical episodes. M2 also only
   covers `T2-F1` **level** targets — the two cumulative-log-return F1 units
   (`ai-mom-2024`, `fed-put-2019`) still use the random walk.

## 8. What was verified, and when

**2026-09-24**, against the two-file image (`forecast_agent.py` + `text_signal.py`, before M2):

- Image is `linux/amd64` — `platform.machine()` → `x86_64` from inside the container.
- `forecast --help` resolves to the **agent's** parser (`--panels --text --asof --out --card
  --n-draws --seed`), exit 0.
- All three pre-flight units run offline in the container: exit 0, all three output files written,
  rationale non-empty, `admissible: true`, g0–g3 all `pass`. `qfbench2 smoke` agrees.
- Live House call works from inside the container:
  `[text_signal] source=llm family=F4 docs=8 assets=['MKT']`.
- `/opt` contains exactly the two agent files. No `.env`, no unit data, no `card.toml` anywhere in
  the image.
- Repo suite: 507 passed, 4 skipped. CI pin-consistency check passes.
- `forecast_meta.json` validates against the toolkit's `forecast.schema.json`; `forecast.parquet`
  carries the four columns the `samples` representation mandates
  (`draw:int32, asset:string, horizon:int32, value:double`) and exactly
  `n_draws × assets × horizons` rows.

**2026-09-25**, re-verified after rebasing onto `dev` and adding `f1_pipeline/m2_unit.py` +
`pandas` (§1):

- Build-time check now passes all three imports:
  `import forecast_agent, text_signal; from f1_pipeline import m2_unit`.
- `/opt` contains exactly the three intended files (`find /opt -type f`) — no `f1_pipeline/data/`,
  no notebooks. Image grew 465 MB → 538 MB, entirely pandas and its own dependencies (scipy, pytz,
  etc.); expected, not a regression.
- `t2-F1-cad-boc-2017` (a level card) run offline in the image: `forecast_rationale.md` now reads
  `Base: M2 -- ridge location-scale fitted on panel history, joint bootstrap of standardised
  residuals (f1_pipeline/m2_unit.py)` — confirming the M2 dispatch actually fires inside the
  container, not just on the host. `admissible: true`, g0–g3 all `pass`, `qfbench2 smoke` agrees.
- `t2-F4-covid-mkt-2020` (non-F1) still reports `Base: correlated Gaussian random walk` —
  confirming the family dispatch didn't regress for units outside `T2-F1`.
- `t2-EXAMPLE-ust-curve-1m` re-run for the same reason (non-F1, 4-asset joint path): unaffected,
  `admissible: true`.
- Repo suite: **513 passed, 4 skipped** (up from 507 — new commits merged from `dev` added tests).
  CI pin-consistency check still passes.

Not yet done at either point: the push to GHCR, the descriptor, and the upload — all three need
credentials.
