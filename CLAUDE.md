@AGENTS.md

## Claude Code

### Environment

- System Python is 3.9 — too old for `qfbench2-common` (`requires-python >= 3.13`). A Python
  3.13 venv lives at `.venv/`. Activate with `source .venv/bin/activate` before running anything.
- Install order matters: `pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.3.1#subdirectory=common"`
  first, then `pip install .` from the repo root (brings in pandas/pyarrow). See the main
  README's Quick-start checklist, step 0.
- **No `MODEL_ENDPOINT` is available locally** — the organizer's house-model proxy isn't live
  yet, and this environment has no Anthropic API key either. Every reasoning-related code path
  (`agent/core/prompt.py`, `agent/core/adjust.py`) has only ever been exercised against
  *mocked* model replies (see `agent/tests/`). Real prompting quality against a live model is
  completely untested — flag this rather than assuming the prompt/parsing logic works against
  a real reply just because the mocked tests pass.

### Card facts that aren't obvious from the schema

- The as-of date lives at `card.toml`'s `[provenance] data_cutoff` — there is no field
  literally named `as_of`.
- Card family is `[metadata] category`, one of `T2-F1`, `T2-F2`, `T2-F3`, `T2-F4`.
- A staged unit's panels live under `<unit>/panels/`; the shipped exemplar
  (`t2-EXAMPLE-ust-curve-1m`) keeps its panel at the unit root instead. Code that reads panels
  should check both locations (see `_read_panels` in `qfbench2_track_forecasting/cli.py`).

### The custom submission agent: `agent/`

Not part of the shipped reference code — `qfbench2_track_forecasting/` and `baselines/` are
never edited, only imported from. Full design history, verified results, and plain-English
explanations of the statistics used (block bootstrap, scenario mixtures, why aggregation
matters) are in `agent/README.md` — read that before making changes here, and update it after
any change that affects behavior or adds a phase.

```
agent/core/    the actual submission pipeline: cli.py, engine.py, prompt.py, adjust.py
agent/tools/   dev-only, never imported by core/: selfcheck.py, sweep.py, viz.py
agent/tests/   mocked tests -- none of them need a real model endpoint
```

### Useful commands

```bash
source .venv/bin/activate

# Run the agent on one unit (offline -- no MODEL_ENDPOINT means the labelled fallback runs)
python -m agent.core.cli --panels units/<unit> --text units/<unit>/text \
  --asof <data_cutoff-from-card.toml> --out /tmp/out/forecast.parquet

# Admissibility gates only (no --realized shipped locally, so no score, gates only)
python scoring/scoring.py score --card units/<unit>/card.toml --forecast /tmp/out/forecast.parquet

# Calibration diagnostics that need no realized outcomes -- aggregates across a whole family
python -m agent.tools.sweep --category T2-F4

# Full test suite (all offline/mocked)
python agent/tests/test_cli_mocked.py
python agent/tests/test_adjust.py
python agent/tests/test_cli_scenarios_mocked.py
```

### Gotchas hit while building this

- A fixed random seed shared across units silently breaks any statistic that assumes
  independent samples: it makes "many units" secretly "one random sequence, rescaled." Give
  each unit its own seed (e.g. hash the unit id) when aggregating across units — see
  `agent/tools/sweep.py`'s docstring for the specific bug this caused (a false `z=+18.5`
  signal that dropped to the honest `z=+1.6` once fixed).
- Scaling a distribution (a `vol_scale` multiplier) can mathematically never change its
  kurtosis/shape — only a different sampling mechanism (block bootstrap, Student-t, a scenario
  mixture) can produce fat tails. Don't try to fix "not enough tail risk" with a bigger
  multiplier alone.
- A single card's calibration reading (spread ratio, kurtosis) is too noisy at a few hundred
  draws to trust on its own — aggregate across many cells before concluding anything.
