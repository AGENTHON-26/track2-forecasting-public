# model_experiment_v3 — experiments for the next model version

## Executive summary (read this first)

This folder holds the experiments for **model v3** (branch `feat/model_v3`, created from `dev` at `fbb9213`). Every
idea here is measured against the **current production model**, and only an idea that beats it moves into
`forecast_models.py`.

**Outcome (2026-10-04): M1.5 replaced the walk in production.** On the backtest it scores 0.9444 × M0 against the
walk's 0.9726 (notebooks 06 and 09), and `forecast_models.py` now reproduces notebook 09 draw for draw (notebook 11).
Notebook 06 measured the walk, which is gone from the code: re-run it at commit `fc632cb`.

**Next (notebooks 12–15, not in production yet): M2.0, 0.9155 × M0 on the backtest**, better than M0 on 96 of 103 units, in
four steps from M1.5's 0.9444: the significant recent trend as drift (M1.7, 0.9312); a zero floor on Treasury yields and the
variance ratio on the width (M1.8, 0.9266); 1 draw in 4 carrying the full trend (M1.9, 0.9194); an ensemble of three EWMA
volatility estimates and 2,000 draws (M2.0, 0.9155). Ideas that did not help are listed at the end of this page.

**Leaderboard branches:** `feat/model_v3_m1_5` (production M1.5, notebook 11), `feat/model_v3_m1_9` (production M1.9,
notebook 16) and `feat/model_v3_m2_0` (production M2.0, notebook 17). Each differs from the one before only in
`forecast_models.py`, its tests, and (for M2.0) the default of 2,000 draws in `forecast_agent.py`. All three run with the
**text off** (`forecast_agent.TEXT_SIGNAL = False`): the score measures the model alone and no House model is called, so
pack them with `tools/make_descriptor.py --no-model`.

**The baseline to beat** was the production model when this folder started: the cumulative random walk + Cholesky +
EWMA, with the per-family settings in `forecast_models.WALK_SETTINGS`:

| family | window | widen | shocks | sd |
|---|---|---|---|---|
| F1 | 130 | 0.9 | Student-t, ν = 5 | EWMA, halflife 21 |
| F2 | 130 | 1.1 | Student-t, ν = 4 | plain sd of the window |
| F3 | 130 | 1.2 | Student-t, ν = 4 | plain sd of the window |
| F4 | 260 | 1.2 | Student-t, ν = 5 | EWMA, halflife 63 |

On the first Development submission (01, 2026-09-29) the team agent scored **0.9528** against the organizer reference
(F1 0.978, F2 0.981, F3 0.906, F4 0.947; lower is better).

**How an idea is measured** (notebook 01):

- **Backtest:** in each unit, the latest 30 % of the dates that have at least 100 rows of history and whose answer lies
  inside the history, every 5th one: 23,635 dates over 103 units. The answer is the panel's own value s steps later (on a
  log-return unit, the sum of the next s rows), so no realized outcome is used.
- **Score:** the competition's. Each part (CRPS, joint, tail) is divided by M0's on the same dates and weighted by the card
  (0.5 / 0.3 / 0.2, or 0.714 / 0 / 0.286 on 1-cell units); units count equally. M0 = 1.0; below 1 beats it.
- The drift rule, the window and ν were chosen on this same backtest, so the numbers are optimistic (accepted: the models
  have few settings). The clean check before production is the team eval's real outcomes (`run_eval.py`).

**Model rules (Dew, 2026-10-03):** on a **log-return unit** each panel row is already a log return, summed over the
horizon, exactly as M0 and `build_realized.py` do (no `log1p` conversion). **Drift** is per asset type (in
`transformations.toml`): on for CPI and NFP, whose trends are real; off (centre = the last value) for rates, FX, equity
factors and unemployment, where the backtest showed M0's drift is mostly noise.

**Conventions:** notebooks, numbered `01_…`, `02_…`, each opening with an executive summary. No realized outcome is
written into this folder (AGENTS.md firewall).

| file | what it is |
|---|---|
| `README.md` | this page |
| `01_m0_baseline_backtest.ipynb` | notebook 1: the backtest dates (latest 30 % of each unit, every 5th date), M0 rebuilt exactly at every date and scored with the official scorer; M0's mean score per unit and per family in each category (CRPS, variogram, tail) and their weighted sum |
| `02_m0_explanation.ipynb` | notebook 2: M0 built by hand on one unit (UST_2Y, horizons 126 and 189): history, steps, mean and variance, centre and spread per horizon, the 500 draws; equal to `m0_baseline.draws` exactly |
| `03_m1_log_return.ipynb` | notebook 3: **M1** = M0 on log returns (level units forecast as last × exp(·)); same backtest dates and random numbers; competition score vs M0 per unit, family and asset type |
| `04_m1_1_by_asset_type.ipynb` | notebook 4: **M1.1** = M0 with the step chosen per asset type (rates, NFP, UNRATE: change; FX, CPI: log change; equity factors: the log return itself), the release rule for monthly series; competition score vs M0 per unit, family and asset type |
| `05_m1_2_student_t.ipynb` | notebook 5: **M1.2** = M1.1 (transform per asset type, M0's drift kept) with a Student-t step distribution fitted on a window of 3 months, 6 months, 1 year, M0's 300 rows or 2 years (drift, sd, correlation and ν all from that window); compared with M0 and M1.1 |
| `06_production_backtest.ipynb` | notebook 6: the production model (`forecast_models.build_draws`, family settings, text off) on the same backtest, competition score vs M0 per unit, family and asset type; every model so far side by side; it measured the walk that was production until 2026-10-04 and uses that version's `WALK_SETTINGS`, so re-run it at commit `fc632cb` |
| `07_m1_3_drift_rule_t_window260.ipynb` | notebook 7: **M1.3** = M1.1 + the drift rule + Student-t (ν fitted) on a fixed 260-step window, nothing tuned; built in steps (drift rule, + Student-t, window 260) and compared with every model |
| `08_m1_4_family_nu.ipynb` | notebook 8: **M1.4** = M1.3 with one Student-t ν per family, the best on the backtest (optimistic): F1 4, F2 6, F3 6, F4 5; the ν grid per family and every model side by side |
| `09_m1_5_ewma63.ipynb` | notebook 9: **M1.5** = M1.3 with an EWMA sd (halflife 63, last 504 steps) instead of the plain 260-step sd; correlation, drift and ν still from the 260 steps; nothing tuned; every model side by side |
| `10_m1_6_t_per_leg.ipynb` | notebook 10: **M1.6** = M1.5 drawn like production's joint (F3) design, with a new Student-t χ² for every leg instead of one for the whole path; 0.9448 vs M1.5's 0.9444, so M1.5 is kept |
| `11_production_parity.ipynb` | notebook 11: **production (`forecast_models.py`) = M1.5**: each asset's rule read from the panel that holds it in the unit folder (equal to `transformations.toml` on all 165 assets), and production's draws identical to notebook 09's at all 103 as-ofs and on every backtest date of the 99 daily units; what the text's shift / widen / skew do; it re-runs on `feat/model_v3` and `feat/model_v3_m1_5`, where production is still M1.5 |
| `12_m1_7_recent_trend.ipynb` | notebook 12: **M1.7** = M1.5 with the recent trend as drift: rates, FX, equity factors and unemployment take the mean of the last 126 steps shrunk by its t-statistic (James–Stein, × max(0, 1 − 4 / t²)), CPI and NFP the mean of the last 12 steps; 0.9312 |
| `13_m1_8_floor_variance_ratio.ipynb` | notebook 13: **M1.8** = M1.7 + a zero floor on Treasury yields + the variance ratio on the width (sd × √VR, VR from the asset's whole history, shrunk toward 1 with n0 = 30); 0.9266 |
| `14_m1_9_trend_scenario.ipynb` | notebook 14: **M1.9** = M1.8 + a trend scenario: 1 draw in 4 carries the 260-step window's full trend (forecast combination); with the M0-pool evidence that led to it; 0.9194 |
| `15_m2_0_vol_ensemble.ipynb` | notebook 15: **M2.0** = M1.9 + an ensemble of EWMA volatility estimates (halflives 21, 63 and 252, a third of the draws each) + 2,000 draws; 0.9155; the path from M1.5 |
| `16_production_parity_m1_9.ipynb` | notebook 16 (branch `feat/model_v3_m1_9`): **production = M1.9**: `forecast_models.build_draws` equals notebook 14's draws at all 103 as-ofs and on every backtest date of the 99 daily units; each production switch (`TREND_SHARE`, `VR_SHRINK`, `ZERO_FLOOR`, `RECENT_TREND`) gives back the previous model exactly |
| `17_production_parity_m2_0.ipynb` | notebook 17 (branch `feat/model_v3_m2_0`): **production = M2.0**: `forecast_models.build_draws` with 2,000 draws equals notebook 15's draws at all 103 as-ofs and on every 5th backtest date of the 99 daily units; each production switch (`VOL_POOL`, `TREND_SHARE`, `VR_SHRINK`, `ZERO_FLOOR`, `RECENT_TREND`) gives back the previous model exactly |
| `transformations.toml` | **the transformation rules** (M1.1): asset → type → transform (`diff`, `log_diff`, `as_return`), with the defaults; every notebook reads a unit's `card.toml` and applies it (`unit_rules`) |
| `data/` | the backtest answers; every model's per-date scores (M0's are the denominators for every later notebook); per unit, per family and per asset type tables; `all_models_vs_m0.csv`, the ladder |

## Tried and not kept (scratch tests on the same backtest, competition score × M0)

| idea | tested on | result | why it is out |
|---|---|---|---|
| widen every forecast by 1.1 / 1.2 | M1.5 (0.9444) | 0.9475 / 0.9579 | the misses sit in a few regimes (cycle turns), not everywhere |
| GARCH-style term structure: per-step variance reverts to the long-run one (halflife 63 / 126 / 252) | M1.5 | 0.9510 / 0.9462 / 0.9441 | the long-run variance of the high-rate years is far too wide for the zero-rate years |
| self-calibration: scale by how well EWMA sd × √h sized the asset's own past moves | M1.5 | 0.9448–0.9541 | noisy, and it learns the last regime |
| a significance switch instead of shrinkage: drift only if \|t\| > 2 / 3 / 4 | M1.5 | 0.9653 / 0.9437 / 0.9430 | all or nothing flips too often; James–Stein (M1.7) is smoother |
| EWMA correlation (halflife 63 / 126) instead of the 260-step one | M1.8 + 25 % of M0's draws (0.9218) | 0.9219 / 0.9218 | no change |
| EWMA sd halflife 42 / 126 instead of 63 | same | 0.9216 / 0.9224 | no change |
| faster halflife for horizons up to 42 days (RiskMetrics: 21, 42) | M1.9 (0.9194) | 0.9199 | no gain on daily data |
| Bayesian model averaging of the trend, BIC weight per asset | M1.9 + 2,000 draws (0.9188) | 0.9252 at best | trusts strong trends too much; they reverse |
| the trend averaged over 63 / 126 / 252 steps; ν capped at 8; ν = 5; Student-t for macro too | same | 0.9178–0.9192 | within ±0.001 |
| antithetic draws (z and −z) | M1.9 + 2,000 draws | no gain over plain draws | — |
| a new Student-t χ² per leg (the old production walk's design) | M1.5 | 0.9448 | notebook 10 |

