# F1 pipeline

Four notebooks, run in order, turning Track 2's F1 panels into submission-shaped forecasts.
Everything is text-blind and per-unit; nothing trains across units.

| # | notebook | reads | writes |
|---|----------|-------|--------|
| 01 | `01_prep_xy.ipynb` | `units/t2-F1-*/` panels + cards | `data/f1_Xy.parquet`, `data/f1_columns.json` |
| 02 | `02_split_units.ipynb` | `data/f1_Xy.parquet` | `units/<unit>/xy/<asset>__h<h>.parquet` (43), `data/f1_submodels.csv`, `data/f1_submodel_features.parquet` |
| 03 | `03_model_level.ipynb` | the sub-model files + mapper | `data/f1_backtest.parquet`, `data/f1_predictions.parquet`, `data/f1_residual_pools.parquet`, `data/f1_coefficients.parquet`, `data/f1_model_config.json` |
| 04 | `04_draws.ipynb` | predictions + pools + config | `units/<unit>/forecast/forecast.parquet` + `forecast_meta.json` + `forecast_rationale.md` (21 units) |
| 05 | `05_text_handoff.ipynb` | predictions + pools + one unit's text corpus | `units/<unit>/forecast_text_demo/` — a worked example for the text/LLM collaborator |

Notebook 05 is a handoff document, not part of the build: it explains what the text side receives,
what every date means, and the two-scalar interface (`drift_bp`, `vol_scale`) it hands back.

Run from this directory with the `track2` conda environment:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_prep_xy.ipynb
```

## Scope

Notebooks 03 and 04 cover **level targets only** — 21 units, 41 cells. The two cumulative-log-return
units (`ai-mom-2024`, `fed-put-2019`) are prepared and split by 01–02 but not modelled.

## The model

M2: per cell, a ridge regression gives the centre (mu), a second ridge on log squared residuals gives
the width (sigma), and the standardised residuals are resampled for the shape. A draw picks one
historical date and applies its residual to every cell of the unit, which is what makes the horizons
move together.

`lam_mu = 10`, `lam_sig = 10`, chosen by rolling-origin backtest on origins before 2018 with at least
1,000 training rows, checked on origins from 2018 onward: **0.859 x** the random-walk reference.

## Known limits

- Below ~1,000 training rows M2 is worse than a random walk (notebook 03, section 4.1). The three short
  units (2003, 2004, 2005) sit there.
- The calibration constant `c` and the residual pool use in-sample residuals, which makes bands roughly
  9% too narrow.
- The extreme 1% tails of each residual pool come from one or two historical episodes.
