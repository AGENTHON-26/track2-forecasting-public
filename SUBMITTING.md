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

Two files: `forecast_agent.py` and `text_signal.py`. Nothing else.

`forecast` is a `/usr/local/bin` shim that execs `python3 /opt/forecast_agent.py "$@"`. There is
deliberately **no `ENTRYPOINT`**: the harness invokes
`docker run <image> forecast --panels … --text … --asof … --out …`, so the verb arrives as the
first argument, and with no ENTRYPOINT the shell resolves it from `PATH` and it never reaches
argparse. Neither `forecast_agent.py` nor the reference CLI accepts a leading `forecast`
positional — both would exit 2 on every unit — so if anyone ever adds an `ENTRYPOINT`, they must
also add a positional argument to absorb the verb.

**The reference CLI (`qfbench2_track_forecasting`) is not in the image.** Its own docstring says it
"ignores `--text` entirely"; shipping it would score the text-blind floor. `pandas` was removed at
the same time and for the same reason — it is that package's dependency, and nothing in the agent
path imports it. Those two removals go together: keeping the package while dropping pandas would
leave a verb-adjacent code path that dies on `import pandas`.

**`qfbench2-common` stays in the Dockerfile** even though the agent never imports it.
`.github/workflows/ci.yml` greps this Dockerfile for `refs/tags/vX.Y.Z.tar.gz` and fails the build
with "could not find a pinned toolkit install" if the line disappears.

**`.dockerignore` is an allowlist** (`*`, then two `!` lines). `.env` holds `MODEL_API_KEY` and
sits at the repo root, so a denylist that forgets one line publishes a credential to a public
registry. It also cuts the build context from ~470 MB to two files — pull time is charged against
the unit clock.

**The build-time import check** —
`RUN python3 -c "import forecast_agent, text_signal"` — is the most important line in the
Dockerfile. `forecast_agent.read_text_signal` catches *any* exception from that import and returns
exact-neutral adjustments, so an image missing `text_signal.py` would build, run, exit 0, pass
every gate, and **silently score as the text-blind baseline**. That is the one failure that wastes
an upload with no error visible anywhere. This turns it into a build failure.

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
| `t2-F1-cad-boc-2017` | 2017-07-12 | 1 asset × 2 horizons `[126, 189]` — multi-horizon expansion |
| `t2-F4-covid-mkt-2020` | 2020-02-19 | tail family |

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

1. **The 25-admitted-requests-per-unit budget is not enforced in code.** Stage 1 fires one call per
   document, retries each up to twice (`text_signal.py:476`), and `call_model` retries up to three
   more times (`text_signal.py:546`) — worst case **8 admitted requests per document**, on units
   carrying up to 15 documents. The House route charges at admission and never refunds, and an SDK
   retry can consume another slot with identical content. Blow the budget and the **Stage 2
   adjustment call** — the one that actually produces the forecast signal — is refused, and the
   unit falls back to neutral. It degrades silently; it does not DNF. A counter that hard-stops at
   25 and reserves a slot for Stage 2 is the fix.
2. **Text is measured net-negative.** The last clean sweep
   (`eval_reports/stage2-thinking-throttled-live.json`, 103 units, 7h39m) has text-informed
   forecasts behind the text-blind baseline by **+0.0423** mean composite, and **+0.0869** on the
   50 units with zero logged failures. The image ships text ON anyway — a deliberate choice to
   exercise the House route in the real environment during Development, not an oversight. Revisit
   before the single Final submission.
3. **No House-call telemetry in the output.** If the proxy or token is wrong in the sealed
   environment, every call fails, `read_text_signal()` returns neutral, and the forecast scores
   text-blind with nothing in `forecast_meta.json` recording that anything broke. One upload
   burned, nothing learned. Recording attempted/succeeded/failed counts would make each Development
   upload a diagnostic.
4. **`forecast_rationale.md` is thin** — about 250 bytes: assets, horizons, base method, and
   whether text was used. It satisfies `g1_schema` (which only checks existence and non-emptiness)
   and is **never scored**. But it feeds a human-review screen over the top of the leaderboard, and
   it names no anchor, no per-adjustment size, and no cited documents — so a reviewer has nothing
   to check. The reference CLI's `_rationale()`
   (`qfbench2_track_forecasting/cli.py:324-449`) produces the shape the contract describes and
   could be adapted.
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

## 8. What was verified, and when

Recorded 2026-09-24 against the image built from this Dockerfile:

- Image is `linux/amd64` — `platform.machine()` → `x86_64` from inside the container.
- `forecast --help` resolves to the **agent's** parser (`--panels --text --asof --out --card
  --n-draws --seed`), exit 0.
- All three pre-flight units run offline in the container: exit 0, all three output files written,
  rationale non-empty, `admissible: true`, g0–g3 all `pass`. `qfbench2 smoke` agrees.
- Live House call works from inside the container:
  `[text_signal] source=llm family=F4 docs=8 assets=['MKT']`.
- `/opt` contains exactly the two agent files. No `.env`, no unit data, no `card.toml` anywhere in
  the image.
- Repo suite: **507 passed, 4 skipped**. CI pin-consistency check passes.
- `forecast_meta.json` validates against the toolkit's `forecast.schema.json`; `forecast.parquet`
  carries the four columns the `samples` representation mandates
  (`draw:int32, asset:string, horizon:int32, value:double`) and exactly
  `n_draws × assets × horizons` rows.

Not yet done at that point: the push to GHCR, the descriptor, and the upload — all three need
credentials.
