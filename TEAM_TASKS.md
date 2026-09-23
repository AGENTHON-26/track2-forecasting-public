# Track 2 — Team Task Split

**Goal:** build a Docker agent that forecasts markets by combining a stochastic model (numbers)
with an LLM reading text. Score below **1.0** (the text-blind baseline) on average across cards.

**Team repo:** https://github.com/AGENTHON-26/track2-team (private)

## Setup (everyone, once)

```bash
git clone https://github.com/AGENTHON-26/track2-team
cd track2-team
pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.3.1#subdirectory=common"
pip install .
pip install numpy pyarrow

# make your branch:
git checkout -b feat/timeseries   # Dew   (feat/llm = Nish, feat/eval = Pun)

# confirm the skeleton runs:
python3 forecast_agent.py --panels units/t2-EXAMPLE-ust-curve-1m \
  --text units/t2-EXAMPLE-ust-curve-1m/text --asof 2024-06-28 --out out/forecast.parquet
```

---

## The integration contract (agree on this FIRST)

Everyone builds against one skeleton — `forecast_agent.py` — with two swappable functions plus a
glue step. Commit this stub to `main` **before** branching, so no one blocks anyone.

```python
# Dew owns this:
draws = build_draws(panels, assets, horizons, asof, adjustments)
        # returns a (n_draws x n_assets x n_horizons) numpy array

# Nish owns this:
adjustments = read_text_signal(text_dir, assets)
        # returns per-asset {"shift": +0.15, "widen": 1.4, "skew": 0.0}

# glue (shared): apply adjustments -> draws -> write the 3 output files
```

As long as everyone respects **that dict** (`shift` / `widen` / `skew`) and **that array shape**,
the three parts integrate with no rewrites.

---

## 1 — Dew · Time-series (the numbers)

- Owns `build_draws(...)`. Read the `.parquet` panels, compute **center + spread + correlation**,
  output correlated joint draws.
- Understand the number data structure: columns `[date, asset, value, panel_id]`.
- Multi-asset -> **one shared random roll** (Cholesky), not one per asset. Fat tails for shock cards.
- **Input:** panels + an `adjustments` dict.  **Output:** the draws array.
- **Branch:** `feat/timeseries`

## 2 — Nish · LLM (the text)

- Owns `read_text_signal(...)`. Read the `text/` corpus, call the **NVIDIA Nemotron API**
  (build.nvidia.com, OpenAI-compatible), prompt it to return `{shift, widen, skew}`.
- Understand the text data structure: `text/*.txt` + `corpus_index.json`. Build the prompt / agent.
- Keep the model call **env-driven** (`MODEL_ENDPOINT` / `MODEL_NAME`) so it swaps to the house
  endpoint at scoring with no code change.
- **Input:** text dir.  **Output:** the `adjustments` dict.
- **Branch:** `feat/llm`

## 3 — Pun · Evaluation (the scoreboard)

- We can't push to the leaderboard yet, so build a **local eval harness**: run the agent on practice
  units -> `forecast.parquet` -> score it.
- Build a **realized vector** from public history (FRED for the target dates) so you can compute real
  **CRPS / variogram / tail / composite** and compare version A vs B ("did it improve?").
- Also run the repo's gate check (g0-g3) so nothing DNFs.
- **Input:** a `forecast.parquet`.  **Output:** a score report per unit.
- **Branch:** `feat/eval`

---

## Git flow

```bash
# once, on main: commit the skeleton stub, then everyone branches
git checkout -b feat/timeseries   # Dew
git checkout -b feat/llm          # Nish
git checkout -b feat/eval         # Pun

# work independently -> open a PR per branch -> merge to main when the interface matches
```

Because each person only touches **their own function**, merges won't collide.

---

## Reference

Full visual guide to the project, data, scoring, and how-to-win:
https://claude.ai/code/artifact/a5d41a3d-d1a9-487d-8fc5-7ccbdd6a185c
