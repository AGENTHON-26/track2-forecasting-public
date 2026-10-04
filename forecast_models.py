"""
Track 2 — the forecast model: panels (+ the text adjustments) -> joint draws.

## Executive summary (read this first)

ONE MODEL for every card: **M1.9** (`model_experiment_v3/14_m1_9_trend_scenario.ipynb`). It is M1.5
(notebook 09) with three more pieces, each measured in its own notebook:

- **M1.5, the base.** The step depends on the asset type: rates, payrolls and unemployment step by
  their change; FX and price indices by their log change (level = `last x exp(sum of steps)`); on a
  log_return card each row already is the step. Student-t shocks on rates, FX and equity factors,
  nu fitted from the window's kurtosis, one chi-square per draw. EWMA sd (halflife 63); the
  correlation between assets and nu come from the last 260 steps.
- **M1.7, the recent trend as drift** (notebook 12). Rates, FX, equity factors and unemployment:
  the mean m of the last 126 steps x max(0, 1 - 4 / t^2), t = m / (sd / sqrt(126)) (James-Stein:
  no drift unless |t| > 2). CPI and payrolls: the mean of the last 12 monthly steps.
- **M1.8, the width at long horizons** (notebook 13). Treasury yields are floored at 0 %. Each
  asset's sd is multiplied, per horizon, by sqrt(VR): VR = mean(h-step move^2) / (h x mean(1-step
  move^2)) over the asset's whole history, shrunk toward 1 with n / (n + 30), n = history / h.
- **M1.9, a trend scenario** (notebook 14). The last quarter of the draws carry the 260-step
  window's full trend instead of the shrunk one: a mixture of two scenarios in one forecast.

The asset type comes from the unit folder alone: the panel that holds the asset. That is the
parquet's `panel_id` column, which is also its file name and the card's `[panels]` id:

    rates_daily -> rates      g10_fx_daily, em_transfer_early -> fx      factors_daily -> equity factor
    macro_monthly -> by asset id: CPI_* / PCE_* price index, NFP payrolls, UNRATE unemployment

See `PANEL_TYPES`, `MACRO_TYPES` and `ASSET_TYPES`. An asset the tables do not know gets M0's rule:
change, normal shocks, no drift.

Backtest (model_experiment_v3; 23,635 dates over 103 units; competition score vs M0, below 1 beats
M0): **M1.9 0.9194**, M1.5 0.9444. On the 90 cards with real outcomes (text off): M1.9 0.9985,
M1.5 1.032. With the text neutral and `seed = crc32(unit id)`, `build_draws` returns notebook 14's
draws exactly at every card's as-of (`model_experiment_v3/16_production_parity_m1_9.ipynb`). The
switches `RECENT_TREND`, `ZERO_FLOOR`, `VR_SHRINK` and `TREND_SHARE` turn the pieces off; all
four off is M1.5 exactly.

    build_draws(panels, assets, horizons, asof, adjustments, n_draws, seed,
                *, target_type=..., family=...) -> (n_draws, n_assets, n_horizons)

The text's adjustments mean what they meant before. `shift` is added at every horizon in the
asset's own units. `widen` multiplies its sd. `skew` tilts each leg's shock. `family` is
accepted but unused: every family has the same settings.

Research notebooks written against the previous walk (model_baseline/, f1_pipeline/ 06-08 and
backtest_m2.py, hyperparameter_tuning/01, model_experiment_v3/06) call names that are gone here:
`WALK_SETTINGS`, `_fit_walk`, `_cumulative_walk_model`, `_panel_steps`, `_step_frame`. Run them at
commit fc632cb, the last one with the walk.

Runs offline with numpy + pandas + pyarrow.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa

_ASSET_COLS = ("asset", "asset_id")

#: M1.5's settings, the same for every family (model_experiment_v3 notebooks 07-09). None was tuned
#: per family: window 260 and halflife 63 were set in advance, and nu is fitted on each card.
WINDOW = 260                    # steps behind the drift, the correlation between assets and nu
EWMA_HALFLIFE = 63              # the sd: a step's weight halves every 63 steps, over the last 8 x 63
WIDEN = 1.0                     # multiplier on the sd, on top of the text's `widen`
NU_MIN, NU_MAX = 4.2, 30.0      # the fitted nu is kept in this range
NORMAL_BELOW_KURTOSIS = 0.2     # a window with less excess kurtosis than this keeps normal shocks
_MIN_COMMON_STEPS = 30

#: M1.9's pieces on top of M1.5 (model_experiment_v3 notebooks 12-14). Chosen on the backtest: the
#: James-Stein constant and the trend window (scratch tests of constants 2/4/9 and windows
#: 63/126/252 gave 0.931-0.942 for M1.7), the variance ratio's shrinkage (30 over 10) and the trend
#: share (15/25/35 % gave 0.9206/0.9194/0.9208). On the cards' real outcomes only the trend scenario
#: is confirmed (-0.033, 95 % band -0.060 to -0.009); the drift and the width are neutral there.
RECENT_TREND = True             # M1.7: drift = the recent trend (`_recent_trend`), not the type rule
TREND_LOOKBACK = 126            #   steps behind the trend of rates, FX, equity factors, unemployment
TREND_SHRINK = 4.0              #   James-Stein: the trend keeps max(0, 1 - TREND_SHRINK / t^2)
MACRO_TREND_STEPS = 12          #   CPI and payrolls: the mean of the last 12 monthly steps
ZERO_FLOOR = True               # M1.8: Treasury-yield draws below 0 % are set to 0 %
VR_SHRINK = 30.0                # M1.8: variance ratio shrunk toward 1 by n / (n + 30); None = off
TREND_SHARE = 0.25              # M1.9: the last quarter of the draws take the window's full trend

#: How each asset type is modelled (model_experiment_v3/transformations.toml):
#:   transform  diff = x_t - x_{t-1};  log_diff = log(x_t / x_{t-1});  as_return = the row itself
#:   shock      normal or student_t
#:   drift      centre = last + steps x mean step (True) or the last value (False)
#: Drift evidence (notebook 07): without drift, rates 0.939, FX 0.968, equity factors 0.978 and
#: unemployment 0.985 vs M0; with it, CPI 0.873 and payrolls 1.000 (without: 1.189 and 1.538).
ASSET_TYPES = {
    "rates":         {"transform": "diff",      "shock": "student_t", "drift": False},
    "fx":            {"transform": "log_diff",  "shock": "student_t", "drift": False},
    "cpi":           {"transform": "log_diff",  "shock": "normal",    "drift": True},
    "payrolls":      {"transform": "diff",      "shock": "normal",    "drift": True},
    "unemployment":  {"transform": "diff",      "shock": "normal",    "drift": False},
    "equity_factor": {"transform": "as_return", "shock": "student_t", "drift": False},
}
_DEFAULT_RULE = {"transform": "diff", "shock": "normal", "drift": False}     # M0's step, no drift

#: Panel id -> asset type. Every panel the organisers ship is here; a panel id not listed falls
#: back to a keyword (`rates`, `fx`, `factor`) and then to the default rule.
PANEL_TYPES = {
    "rates_daily": "rates",
    "g10_fx_daily": "fx",
    "em_transfer_early": "fx",
    "factors_daily": "equity_factor",
}
#: macro_monthly holds several kinds of series, so its assets are typed by id. CPI_ALL, NFP and
#: UNRATE are backtested; CPI_CORE and the PCE indices are price indices like CPI_ALL.
MACRO_TYPES = {
    "CPI_ALL": "cpi", "CPI_CORE": "cpi", "PCE_ALL": "cpi", "PCE_CORE": "cpi",
    "NFP": "payrolls",
    "UNRATE": "unemployment",
}

#: Per-family FALLBACK width for the walk models, used only when the text half gave this card
#: nothing (every asset exactly neutral: model call failed, timed out, ran out of budget, or no
#: usable summaries). F4 cards are built around a shock the calm history does not show, so a
#: text-less F4 card should still not draw history's too-narrow width. Measured 2026-09-26 on the
#: 29 realized F4 cards (3 draw seeds): random walk 0.4560 -> 0.4085 with 1.5 applied to every
#: card. When stage 2 DOES answer it is not overridden: with the F4 v5 prompt an always-on floor
#: scored 0.3715 vs 0.3781 overall but made the 18 calm cards worse (7 better / 11 worse vs the
#: walk), so it only covers failure. See `text_signal._FAMILY_FOCUS["F4"]` for the prompt.
_FAMILY_WIDEN_FLOOR = {"T2-F4": 1.5}


def _text_was_silent(adjustments: dict[str, dict[str, float]], assets: list[str]) -> bool:
    """True when every asset carries the exact neutral adjustment -- the text half's failure
    shape (`read_text_signal` degrades to neutral rather than raising)."""
    return all(adjustments.get(a, {}).get("shift", 0.0) == 0.0
               and adjustments.get(a, {}).get("widen", 1.0) == 1.0
               and adjustments.get(a, {}).get("skew", 0.0) == 0.0 for a in assets)


#: ON since 2026-09-25, after running the exact comparison the previous comment here demanded:
#: "skew forced on vs. off against ONE recorded set of model adjustments, not two fresh live
#: calls". That was impossible until the stage-2 ledger started being recorded; replaying tonight's
#: F3 sweep with the flag flipped and everything else held (same adjustments, same draws, same
#: seeds, no new model calls) isolates the code path exactly as asked.
#:
#: Result on the 3 F3 units where the model asked for a non-zero skew: composite ratio ON/OFF
#: 0.9865, 1.0000, 0.9992 -- mean 0.9952, nothing worse. Weak evidence, and honestly so: 5 assets,
#: every one at skew=-0.20, on the family where skew matters LEAST (F3 is scored on the joint
#: term, not the tail).
#:
#: The stronger argument is structural. Stage 2 asks for a skew on every card and F4's prompt
#: explicitly instructs the model to "use skew to point the distribution toward the side the shock
#: would move prices" -- F4 is scored primarily on the tail penalty, which is precisely what a
#: tilt shapes. Discarding a value we ask for, on the family whose primary term it exists to move,
#: was the incoherent state. Off was the safe default while nobody could measure it; that is no
#: longer true.
#:
#: STILL UNVALIDATED ON F4. The replay above covers F3 only, because no F4 sweep has been recorded
#: since the ledger landed. Do that before trusting skew to earn anything on the family it is
#: really for.
SKEW_ENABLED = True


def build_draws(
    panels: dict[str, "pa.Table"],
    assets: list[str],
    horizons: list[int],
    asof: str,
    adjustments: dict[str, dict[str, float]],
    n_draws: int,
    seed: int,
    *,
    target_type: str | None = None,
    family: str | None = None,
) -> np.ndarray:
    """Joint draws with Nish's adjustments applied. Returns (n_draws, n_assets, n_horizons).

    Every card takes M1.9. `family` is kept for the contract; the settings do not depend on it.
    """
    return _model(_Request(panels, assets, horizons, asof, adjustments, n_draws, seed,
                           target_type, family))


# ============================================================================
#  The request, the asset rules, and M1.9.
# ============================================================================
@dataclass(frozen=True)
class _Request:
    """One card's forecast request: everything the model needs, in one place."""
    panels: dict[str, "pa.Table"]
    assets: list[str]
    horizons: list[int]
    asof: str
    adjustments: dict[str, dict[str, float]]
    n_draws: int
    seed: int
    target_type: str | None
    family: str | None


def asset_type(asset: str, panel_id: str) -> str | None:
    """The asset's type, from the panel that holds it (None: not recognised, use the default)."""
    if panel_id in PANEL_TYPES:
        return PANEL_TYPES[panel_id]
    if panel_id.startswith("macro") or asset in MACRO_TYPES:
        return MACRO_TYPES.get(asset)
    for key, kind in (("rates", "rates"), ("fx", "fx"), ("factor", "equity_factor")):
        if key in panel_id:
            return kind
    return None


def asset_rule(asset: str, panel_id: str, target_type: str | None) -> dict:
    """{transform, shock, drift} for one asset of a card.

    A log_return card's rows already are log returns, so its transform is `as_return` whatever the
    asset. `as_return` on a level card means nothing, so it falls back to M0's change.
    """
    rule = dict(ASSET_TYPES.get(asset_type(asset, panel_id), _DEFAULT_RULE))
    if target_type == "log_return":
        rule["transform"] = "as_return"
    elif rule["transform"] == "as_return":
        rule["transform"] = "diff"
    return rule


@dataclass(frozen=True)
class _Fit:
    """What the model draws from. Everything is indexed by `assets`, which is SORTED: M0's cell
    order (doc 3.9) is assets by id, then horizons ascending, and the random numbers follow it."""
    assets: list[str]
    rules: dict[str, dict]
    types: dict[str, str | None]   # asset type from the panel ("rates" gets the zero floor)
    log_mode: dict[str, bool]      # stepped in logs: forecast = last x exp(centre + shock)
    last: dict[str, float]         # the last observed value (0.0 on a log_return card)
    final: dict[str, float]        # the last observed row, for the no-new-release rule
    mu: np.ndarray                 # mean step over the window (the trend scenario's drift)
    drift: dict[str, float]        # drift per step: the recent trend (M1.7) or the type rule (M1.5)
    full: dict[str, np.ndarray]    # every step of the history, oldest first (trend, variance ratio)
    sigma: np.ndarray              # one step's covariance: window correlation x EWMA sd x EWMA sd
    nu: float | None               # Student-t degrees of freedom (None: normal shocks)
    steps: dict[str, dict[int, float]]   # per asset, each horizon in the asset's own panel steps


def _fit_model(r: _Request) -> _Fit:
    """Panels -> the drift, the covariance and nu, exactly as notebooks 09 and 12-14 fit them.

    Each asset takes its last WINDOW + 1 rows, steps them in its own transform (M0's gap rule
    drops a step across a hole), then the assets are aligned on the dates they share. Aligning
    by date, not by row, matters on cards that span two panels: on t2-F3-divergence-2014
    UST_10Y/JPY correlates at 0.32 by row against 0.50 by date.
    """
    hist, rules, types = {}, {}, {}
    for a in r.assets:
        panel_id, rows = _history(r.panels, a, r.asof)
        hist[a], rules[a], types[a] = rows, asset_rule(a, panel_id, r.target_type), asset_type(a, panel_id)
    by_asset = sorted(r.assets)
    steps, log_mode, last = {}, {}, {}
    for a in by_asset:
        steps[a], log_mode[a] = _asset_steps(hist[a][-(WINDOW + 1):], rules[a])
        last[a] = 0.0 if rules[a]["transform"] == "as_return" else hist[a][-1][1]
    common = sorted(set.intersection(*(set(steps[a]) for a in by_asset)))
    if len(common) < _MIN_COMMON_STEPS:
        raise SystemExit(
            f"not enough overlapping history to estimate covariance ({len(common)} rows)")
    x = np.array([[steps[a][d] for d in common] for a in by_asset])
    mu, sigma = x.mean(axis=1), np.atleast_2d(np.cov(x))

    # The width: each asset's EWMA sd over its own last 8 x halflife steps. The correlation stays
    # the window's.
    sd_w = np.sqrt(np.diag(sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.nan_to_num(sigma / np.outer(sd_w, sd_w), nan=0.0)
    np.fill_diagonal(corr, 1.0)
    sd_e = np.array([_ewma_sd(_ordered_steps(hist[a][-(int(8 * EWMA_HALFLIFE) + 1):], rules[a]),
                              EWMA_HALFLIFE) for a in by_asset])
    sigma = corr * np.outer(sd_e, sd_e)

    full = {a: _ordered_steps(hist[a], rules[a]) for a in by_asset}
    drift = {a: (mu[i] if rules[a]["drift"] else 0.0) for i, a in enumerate(by_asset)}    # M1.5
    if RECENT_TREND:                                                                     # M1.7
        drift = {a: _recent_trend(full[a], rules[a], mu[i]) for i, a in enumerate(by_asset)}

    t_rows = [i for i, a in enumerate(by_asset) if rules[a]["shock"] == "student_t"]
    return _Fit(
        assets=by_asset, rules=rules, types=types, log_mode=log_mode, last=last,
        final={a: hist[a][-1][1] for a in by_asset}, mu=mu, sigma=sigma, drift=drift, full=full,
        nu=_fit_nu(x[t_rows]) if t_rows else None,
        steps={a: dict(zip(r.horizons, _horizon_steps([d for d, _ in hist[a]], r.horizons, r.asof)))
               for a in by_asset},
    )


def _model(r: _Request) -> np.ndarray:
    """Every card. M0's joint draw with M1.9's centre, width, shocks and scenarios, then the text.

    The cells are (asset, horizon) pairs. cov[(a1, h1), (a2, h2)] = min(steps) x sigma[a1, a2]: the
    covariance of ONE accumulating path per draw, so horizons of one asset correlate at
    sqrt(h1 / h2) and assets keep the window's correlation at every horizon. Each cell is then
    scaled by sqrt(VR) (M1.8). A Student-t cell is scaled by sqrt((nu - 2) / W), with one W per draw
    for the whole path; drawing a new W per leg scored worse (notebook 10). The last TREND_SHARE of
    the draws are re-centred on the window's full trend (M1.9), and Treasury yields are floored at 0.
    """
    f = _fit_model(r)
    ai = {a: i for i, a in enumerate(f.assets)}
    hs = sorted(r.horizons)
    cells = [(a, h) for a in f.assets for h in hs]
    s = f.steps

    def centre(a: str, h: int, drift: dict[str, float]) -> float:
        return (0.0 if f.log_mode[a] else f.last[a]) + s[a][h] * drift[a]

    mean = np.array([centre(a, h, f.drift) for a, h in cells])
    cov = np.array([[min(s[a1][h1], s[a2][h2]) * f.sigma[ai[a1], ai[a2]] for a2, h2 in cells]
                    for a1, h1 in cells])
    cov[np.diag_indices_from(cov)] += 1e-10                     # M0's jitter (doc 3.8)
    jittered = cov + 1e-9 * np.eye(len(cells))
    try:
        chol = np.linalg.cholesky(jittered)
    except np.linalg.LinAlgError:
        chol = np.diag(np.sqrt(np.diag(jittered)))
    dev = np.random.default_rng(r.seed).standard_normal((r.n_draws, len(cells))) @ chol.T

    def adj(a: str, key: str, default: float) -> float:
        return float(r.adjustments.get(a, {}).get(key, default))

    skew = np.array([adj(a, "skew", 0.0) for a in f.assets])
    if SKEW_ENABLED and np.any(skew != 0.0):
        dev = _skew_legs(dev, f, hs, skew, r.seed)
    if VR_SHRINK is not None:                                                          # M1.8
        dev = dev * np.array([np.sqrt(_variance_ratio(f.full[a], s[a][h], f.rules[a]["drift"]))
                              for a, h in cells])
    dev = dev * np.array([WIDEN * adj(a, "widen", 1.0) for a, h in cells])
    t_cells = np.array([f.rules[a]["shock"] == "student_t" for a, h in cells])
    if f.nu is not None and t_cells.any():
        w = np.random.default_rng([r.seed, 7]).chisquare(f.nu, size=(r.n_draws, 1))
        dev = dev * np.where(t_cells, np.sqrt((f.nu - 2.0) / w), 1.0)

    sm = mean + dev
    k = int(round(TREND_SHARE * r.n_draws))
    if k:                           # M1.9: the last k draws: the trend of the window, not shrunk
        full_trend = {a: (f.drift[a] if f.rules[a]["drift"] else f.mu[ai[a]]) for a in f.assets}
        sm[r.n_draws - k:] = np.array([centre(a, h, full_trend) for a, h in cells]) + dev[r.n_draws - k:]
    out = np.empty((r.n_draws, len(r.assets), len(r.horizons)))
    for ci, (a, h) in enumerate(cells):
        if s[a][h] <= 0:            # no new monthly release by the target date: the last one stands
            col = np.full(r.n_draws, f.final[a])
        else:
            col = (f.last[a] * np.exp(sm[:, ci]) if f.log_mode[a] else sm[:, ci]) + adj(a, "shift", 0.0)
            if ZERO_FLOOR and f.types[a] == "rates":                                   # M1.8
                col = np.maximum(col, 0.0)
        out[:, r.assets.index(a), r.horizons.index(h)] = col
    return out


def _recent_trend(v: np.ndarray, rule: dict, mu_window: float) -> float:
    """M1.7's drift per step from an asset's steps `v` (oldest first).

    CPI and payrolls (drift by type): the mean of the last MACRO_TREND_STEPS steps, this year's
    trend rather than the 21-year average. Everything else: the mean m of the last TREND_LOOKBACK
    steps shrunk by its t-statistic t = m / (sd / sqrt(n)): m x max(0, 1 - TREND_SHRINK / t^2).
    With the constant 4, |t| <= 2 gives no drift at all and |t| = 4 keeps 75 %.
    """
    if rule["drift"]:
        return float(v[-MACRO_TREND_STEPS:].mean()) if len(v) >= MACRO_TREND_STEPS else float(mu_window)
    v = v[-TREND_LOOKBACK:]
    sd = v.std(ddof=1)
    t = v.mean() / (sd / np.sqrt(len(v))) if sd > 0 else 0.0
    return float(v.mean() * max(0.0, 1.0 - TREND_SHRINK / max(t * t, 1e-12)))


def _variance_ratio(v: np.ndarray, h_steps: float, demean: bool) -> float:
    """M1.8: VR = mean(h-step move^2) / (h x mean(1-step move^2)) over the steps `v`, shrunk toward 1.

    VR > 1 when moves keep going (rates through policy cycles), < 1 when they cancel. Moves are not
    demeaned unless the asset carries a drift: a trend the model does not forecast is width. The
    estimate is shrunk by the number of independent windows n = len(v) / h: 1 + (VR - 1) x n / (n + 30).
    """
    h = int(h_steps)
    if VR_SHRINK is None or h <= 1 or len(v) < 2 * h + 20:
        return 1.0
    m = v.mean() if demean else 0.0
    c = np.concatenate([[0.0], np.cumsum(v - m)])
    ratio = np.mean((c[h:] - c[:-h]) ** 2) / (h * np.mean((v - m) ** 2))
    n_eff = len(v) / h
    return 1.0 + (ratio - 1.0) * n_eff / (n_eff + VR_SHRINK)


def _skew_legs(dev: np.ndarray, f: _Fit, hs: list[int], skew: np.ndarray, seed: int) -> np.ndarray:
    """The text's skew, on each leg of the path, as the previous walk applied it.

    The path is cut into legs (horizon 1, horizon 1 -> 2, ...). Each leg's move is divided by its sd
    to give a standard shock, tilted per asset by `_skew_tilt` with its own |normal| term, scaled
    back and added up again. At skew = 0 this is never called, so neutral cards draw exactly as
    notebook 14.
    """
    n, k = dev.shape[0], len(hs)
    legs = np.diff(dev.reshape(n, len(f.assets), k), axis=2, prepend=0.0)
    ds = np.diff(np.array([[f.steps[a][h] for h in hs] for a in f.assets]), axis=1, prepend=0.0)
    leg_sd = np.sqrt(np.diag(f.sigma))[:, None] * np.sqrt(np.clip(ds, 0.0, None))
    rng_u = np.random.default_rng([seed, 11])
    for j in range(k):
        ok = leg_sd[:, j] > 0
        z = legs[:, :, j] / np.where(ok, leg_sd[:, j], 1.0)
        u = np.abs(rng_u.standard_normal((n, len(f.assets))))
        legs[:, :, j] = np.where(ok, _skew_tilt(z, u, skew) * leg_sd[:, j], legs[:, :, j])
    return np.cumsum(legs, axis=2).reshape(n, -1)


def _fit_nu(x_t: np.ndarray) -> float | None:
    """nu from the window's excess kurtosis k: a Student-t has k = 6 / (nu - 4), so nu = 4 + 6 / k.

    k is the mean over the card's Student-t assets. Below NORMAL_BELOW_KURTOSIS the shocks stay
    normal.
    """
    z = x_t - x_t.mean(axis=1, keepdims=True)
    k = float(np.mean((z ** 4).mean(axis=1) / (z ** 2).mean(axis=1) ** 2 - 3))
    if not np.isfinite(k) or k < NORMAL_BELOW_KURTOSIS:
        return None
    return float(np.clip(4 + 6 / k, NU_MIN, NU_MAX))


def _ewma_sd(past: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA sd of `past` (rows oldest -> newest; one column per asset, or a single 1-d series): a
    step's weight halves every `halflife` steps back, around the weighted mean. Equals pandas'
    ewm(halflife).std(bias=True)."""
    age = np.arange(len(past))[::-1]
    w = 0.5 ** (age / halflife)
    w = w / w.sum()
    mean = w @ past
    return np.sqrt(w @ (past - mean) ** 2)


# ------------------------------------------------------------------------ history and steps
def _history(panels: dict[str, "pa.Table"], asset: str, asof: str) -> tuple[str, list[tuple[str, float]]]:
    """(panel id, [(date, value), ...]) for one asset, at or before the as-of, sorted by date.

    The first panel holding the asset wins, in the order the panels were read (sorted file names,
    as M0 does). The panel id is the table's own `panel_id` column if it has one, else its key in
    `panels` (the file name).
    """
    seen: set[str] = set()
    for key, t in panels.items():
        acol = next((c for c in _ASSET_COLS if c in t.column_names), None)
        if acol is None:
            continue
        d = t.to_pydict()
        seen.update(str(a) for a in d[acol])
        idx = [i for i, a in enumerate(d[acol]) if str(a) == asset]
        rows = sorted((str(d["date"][i])[:10], float(d["value"][i])) for i in idx
                      if str(d["date"][i])[:10] <= asof and d["value"][i] is not None)
        if rows:
            panel_id = d["panel_id"][idx[0]] if d.get("panel_id") and d["panel_id"][idx[0]] else key
            return str(panel_id), rows
    raise SystemExit(
        f"asset {asset!r} not found in any panel at/before {asof}; "
        f"the panels carry {sorted(seen)}"
    )


def _series(panels: dict[str, "pa.Table"], asset: str, asof: str) -> "pd.Series":
    """One asset's history as a date-indexed series (used by tools/f3_arms.py)."""
    _, rows = _history(panels, asset, asof)
    return pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows]), name=asset)


def _steps(rows: list[tuple[str, float]], as_return: bool) -> dict:
    """M0's steps (doc 3.2-3.3): {date: step}, a step carrying the date of the row it ends on.

    The step is the row itself on a log_return card, the change otherwise. A step across a hole of
    more than max(10 x median spacing, 5 days) is dropped. Transfer cards ship an early window plus
    one row at the as-of: differenced naively, that hole is one "day" worth a decade of moves
    (t2-F2-fragile-five-brl-2013: -0.875 against a typical 0.030, sd inflated 1.4x).
    """
    dates = [dt.date.fromisoformat(d) for d, _ in rows]
    values = np.array([v for _, v in rows], dtype=float)
    gaps = np.array([(dates[i] - dates[i - 1]).days for i in range(1, len(dates))], dtype=float)
    hole = max(10.0 * float(np.median(gaps)), 5.0) if gaps.size else 5.0
    return {dates[i]: (values[i] if as_return else values[i] - values[i - 1])
            for i in range(1, len(dates)) if gaps[i - 1] <= hole}


def _asset_steps(rows: list[tuple[str, float]], rule: dict) -> tuple[dict, bool]:
    """({date: step}, log mode) for one asset in its own transform. log_diff needs every value in
    `rows` above 0; otherwise the asset steps by its change."""
    log_mode = rule["transform"] == "log_diff" and min(v for _, v in rows) > 0
    if log_mode:
        return _steps([(d, np.log(v)) for d, v in rows], False), True
    return _steps(rows, rule["transform"] == "as_return"), False


def _ordered_steps(rows: list[tuple[str, float]], rule: dict) -> np.ndarray:
    """One asset's steps over `rows`, oldest first (the EWMA's input)."""
    st, _ = _asset_steps(rows, rule)
    return np.array([st[d] for d in sorted(st)])


#: A panel whose rows are further apart than this (median, in calendar days) is a monthly panel.
_MONTHLY_SPACING_DAYS = 20


def _horizon_steps(dates: list[str], horizons: list[int], asof: str) -> list[float]:
    """The card's horizons, counted in steps (rows) of this asset's own panel.

    On a daily panel a step is a business day and the horizon already counts them. On a monthly
    panel a step is a month while the card still states business days, so walking h steps would
    walk h MONTHS (3.2x too wide on t2-F4-covid-nfp-2020). There the horizon becomes the number of
    months from the asset's LAST observation (monthly data lags the as-of by its publication delay)
    to the month of `asof + h business days`: the official baseline's rule (docs/M0-BASELINE.md 3.7),
    which reproduces its table for the four monthly cards. Zero months means no new release by the
    target date; the forecast is then the last released value.
    """
    when = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if len(when) < 3 or not float(when.diff().dt.days.median()) > _MONTHLY_SPACING_DAYS:
        return [float(h) for h in horizons]
    last = when.iloc[-1]
    out = []
    for h in horizons:
        target = pd.Timestamp(asof) + pd.offsets.BDay(int(h))
        out.append(float(max(12 * (target.year - last.year) + (target.month - last.month), 0)))
    return out


#: The skew-normal family's sample skewness is bounded (|.| < ~0.995 as shape -> infinity) and
#: rises slowly: shape=1 (the top of `skew`'s own [-1,1] contract if used directly) reaches only
#: ~0.14 sample skewness -- a barely-visible tilt, not the "shock" F4 needs (see NISH_TEXT_NOTES.md
#: and PUN_TEXT_NOTES.md). Scaling `skew` up to a shape parameter of +-5 at the clamp's edge
#: reaches ~0.85 instead -- a strong, visibly asymmetric tail without pinning at the family's
#: near-degenerate ceiling.
_SKEW_SHAPE_SCALE = 5.0


def _skew_tilt(z: np.ndarray, u: np.ndarray, skew: np.ndarray) -> np.ndarray:
    """Azzalini's skew-normal construction, per asset (last axis of `z`/`u`, matching `skew`).

    Mixes the correlated roll `z` with an independent |normal| term `u`, weighted by `delta`,
    then recenters and rescales so the result has mean 0 and unit variance for EVERY value of
    skew -- `shift` and `widen` keep meaning exactly what they already mean upstream, and `skew`
    changes shape only. At skew=0, delta=0 and this returns `z` exactly (see
    `test_skew_zero_is_identity`): every card that never asks for a tilt draws identically to
    before this function existed.

    `u` must be independent PER ASSET, drawn separately from `z` -- it must not touch the
    cross-asset correlation `chol` already encodes elsewhere, which this leaves untouched. A skew
    that reaches across assets (e.g. "both legs of this trade break the same way") is real future
    work, not this.
    """
    alpha = skew * _SKEW_SHAPE_SCALE
    delta = alpha / np.sqrt(1.0 + alpha**2)
    mean_shift = delta * np.sqrt(2.0 / np.pi)
    var_scale = np.sqrt(np.clip(1.0 - delta**2 * (2.0 / np.pi), 1e-8, None))
    return (delta * u + np.sqrt(1.0 - delta**2) * z - mean_shift) / var_scale
