# model_baseline — the team's model: the cumulative walk

## Executive summary (read this first)

This folder holds the work on the team's model: the **cumulative walk** in `forecast_models.py`, the one
model production uses for **every card** (since 2026-09-27). Every new model (M2, or anything later) has to
beat it on the same cards before it replaces it.

The model, in one line per piece:

- **Path:** one path per draw from the last value (0 on a log-return card). Each horizon is read off the
  path, so the horizons of a draw are linked. A card with 1 horizon is one leg:
  last value + shock × sd × widen × √steps.
- **Assets:** they move together through the Cholesky factor of their correlation.
- **Steps:** one step is one row of the card's panel. On a daily panel that is a business day, so the
  horizon is used as is. On a monthly panel it is a month, so the horizon is counted in months from the
  last observation (`forecast_models._panel_steps`, the official baseline M0's rule).

Why it is the baseline (backtest in `f1_pipeline/07_backtest_m2_vs_walk.ipynb`): on daily cards M2 does
not beat it in F1, F2 or F4, and on real outcomes it scores better than M2 on most F1 cards. M2 is better
only on the four monthly cards.

Not to be confused with `baselines/` next to this folder: those are the organizers' adapter scaffolds
(Chronos, TimesFM, and others), not this model.

## The main model (since 2026-09-27, in `forecast_models.py`)

**Every card → the cumulative walk**, with its family's settings in `forecast_models.WALK_SETTINGS`
(notebooks 02–05). One model on purpose: the text's `shift` / `widen` / `skew` then mean the same thing on
every card, which keeps the LLM half simple to tune.

| family | window | widen | shocks | sd |
|---|---|---|---|---|
| F1 | 42 | 1.2 | Student-t, ν = 5 | window |
| F2 | 130 | 1.1 | Student-t, ν = 4 | window |
| F3 | 130 | 1.2 | Student-t, ν = 4 | window |
| F4 | 260 | 1.2 | Student-t, ν = 5 | EWMA, half-life 63 |

The text model's `shift` / `widen` / `skew` act on the walk. On F4, when the text is silent, the width is
at least 1.5 × sd.

**M2 is not in production.** Notebook 06 found it better on the 4 visible monthly cards (0.660x the walk on
validation, 0.481x on test). That gain was given up on 2026-09-27 for one model on every card. None of the 4
monthly cards has a realized outcome in the team eval, so the eval does not move.

## What is here

| file | what it is |
|---|---|
| `01_prep_data.ipynb` | notebook 1: **`prepare_unit(unit_dir)`**, one function (a normal cell in §1) that reads any F1–F4 card and returns card, history, steps, horizon steps and the train / validation / test table (by time, purge gaps between blocks); shown on four different cards, checked on all 103 |
| `02_backtest_walk.ipynb` | notebook 2: **`walk_forecast()`** with hyperparameters (`window`, `widen`) and **`card_score()`** (the card's own score), both checked exactly against production and the official scorer; tunes on **train**, chooses the way (one for all / per family / per card) on **validation**, opens **test** once for the final choice |
| `03_report_per_family.ipynb` | notebook 3: the walk with the **one-per-family** settings from notebook 2, backtested on **train, validation and test for every card**; the card's own score (and its three parts) and band coverage, **by category (F1–F4)** and per card |
| `04_student_t.ipynb` | notebook 4: the walk with **Student-t** shocks (same variance, fatter tails; one shared chi-square per draw keeps the Cholesky correlation). ν and `widen` tuned per family on train, kept per family if better on validation, test opened once against the normal model |
| `05_ewma_vol.ipynb` | notebook 5: the walk's sd as an **EWMA** (weights halve every `halflife` steps) instead of a flat window; halflife and `widen` tuned per family on train, kept per family if better on validation (only F4 kept it), test opened once against the Student-t walk |
| `06_monthly_m2_vs_walk.ipynb` | notebook 6: M2 against the latest walk on the 4 monthly cards; decided on validation (M2 0.660x the walk), confirmed on test (0.481x) → M2 better on monthly cards; **not used** (one model for every card since 2026-09-27) |
| `data/cards.csv` | one row per card: shape, frequency, panels, train / test sizes |
| `data/targets.parquet` | every card's train / purged / validation / test / predict rows (written by notebook 1) |
| `data/steps.parquet` | every card's steps, one row per date and asset (written by notebook 1) |
| `data/tune_train.parquet` | every setting's score on every card's train dates (notebook 2, stages 1–2) |
| `data/tune_validation.parquet` | the fixed walk and each way's choice, scored on validation (notebook 2) |
| `data/backtest_walk.parquet`, `data/backtest_walk_cells.parquet` | test scores of the fixed and tuned walks, per test date and per cell with its PIT (notebook 2) |
| `data/walk_settings.csv` | the (window, widen) each way of tuning chose, per card (notebook 2) |
| `data/report_category.csv`, `data/report_per_unit.csv`, `data/report_scores.parquet` | notebook 3: the category summary, the per-card report (train / validation / test) and every scored date |
| `data/t_settings.csv`, `data/t_*.parquet`, `data/t_test_per_unit.csv` | notebook 4: normal and final (t) setting per card, train / validation / test scores, the per-card test comparison |
| `data/ewma_settings.csv`, `data/ewma_*.parquet`, `data/ewma_test_per_unit.csv` | notebook 5: current and final setting per card, train / validation / test scores, per-card test comparison |

Related work elsewhere:

| file | what it is |
|---|---|
| `forecast_models.py` | the code: `_fit_walk`, `_random_walk_model`, `_cumulative_walk_model`, `_panel_steps` |
| `f1_pipeline/08_random_walk_explained.ipynb` | both walks built by hand on fake data, checked against the code |
| `f1_pipeline/07_backtest_m2_vs_walk.ipynb` | M2 against the walk on F1, F2 and F4 |
| `f1_pipeline/backtest_m2.py` | the backtest the notebook runs |
