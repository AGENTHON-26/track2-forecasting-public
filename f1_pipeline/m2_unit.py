"""
M2 for one F1 unit: card -> features -> fit -> anchor, mu, sigma and the eps pool.

## Executive summary (read this first)

Give it one unit folder. It returns, for every (asset, horizon) cell the card asks for:

- **anchor** — the last observed value at the as-of date.
- **mu** — the predicted change from the anchor. The forecast centre is `anchor + mu`.
- **sigma** — the predicted width of that change.
- **eps pool** — the standardised historical residuals, one per origin date. They carry the shape.

A draw is then `anchor + mu + sigma * eps(d)`, with the same date `d` used for every cell of the unit.
The CLI stops before drawing and only reports the four ingredients. `draw_joint()` does the drawing
(notebook 04's sampler); `forecast_agent.py`'s `build_draws()` calls it for F1 level cards.

This is the same logic as notebooks `01_prep_xy` -> `02_split_units` -> `03_model_level` (section 5,
the final fit), run on one unit instead of all of them. The penalties are not re-tuned here: the
defaults `lam_mu = lam_sig = 10` are the global choice notebook 03 made by backtest.

Scope matches notebook 03: **level targets only**. Features are still built for cumulative-log-return
units, but fitting one raises an error.

    python f1_pipeline/m2_unit.py units/t2-F1-hawkish-cut-2024
    python f1_pipeline/m2_unit.py units/t2-F1-pause-2006 --out out/m2/pause-2006

The four steps, each a function below:

1. `read_unit(unit_dir)`      — read `card.toml` and the panel.
2. `build_features(unit)`     — one table per cell: features at each origin, and the h-step-ahead target.
3. `fit_m2(cell)`             — ridge for mu, ridge on log squared residuals for sigma, residuals for eps.
4. `report(fits)`             — anchor / mu / sigma table, the eps pools, the coefficients.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAM_MU, LAM_SIG = 10.0, 10.0      # notebook 03, section 4.2: argmin on origins < 2018 with >= 1,000 rows
WINSOR = 4.0                      # standardised features clipped at +-4
SIGMA_CLAMP = (0.25, 4.0)         # sigma kept within 1/4x .. 4x the constant-sigma estimate
MU_CLAMP_SIGMAS = 3.0             # |mu| <= 3 x constant sigma


# ============================================================================
#  1 · Read the card and the panel
# ============================================================================
@dataclass
class Unit:
    unit: str
    panel: str                    # panel file stem, e.g. "rates_daily"
    wide: pd.DataFrame            # date x series, float
    asof: pd.Timestamp            # the card's data_cutoff
    assets: list[str]             # card order
    horizons: list[int]           # card order, business days
    target_type: str              # "level" | "log_return"
    value_unit: str
    freq: str                     # "daily" | "monthly"


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """Long panel [date, asset|asset_id, value] -> wide date x series."""
    acol = "asset" if "asset" in df.columns else "asset_id"
    df = df.assign(date=pd.to_datetime(df["date"].astype(str).str[:10]))
    return df.pivot(index="date", columns=acol, values="value").sort_index().astype(float)


def read_unit(unit_dir: str | pathlib.Path) -> Unit:
    unit_dir = pathlib.Path(unit_dir)
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    panels = sorted(unit_dir.glob("*.parquet"))
    if not panels:
        raise ValueError(f"no panel .parquet in {unit_dir}")
    pq = panels[0]
    t = card["targets"]
    return Unit(unit=card["task"]["id"], panel=pq.stem, wide=_wide(pd.read_parquet(pq)),
                asof=pd.Timestamp(card["provenance"]["data_cutoff"]),
                assets=[str(a) for a in t["asset_ids"]], horizons=[int(h) for h in t["horizons"]],
                target_type=t["target_type"], value_unit=t["value_unit"],
                freq="monthly" if pq.stem == "macro_monthly" else "daily")


def unit_from_panels(panels: dict, assets: list[str], horizons: list[int], asof: str,
                     target_type: str, unit_id: str = "", value_unit: str = "") -> Unit:
    """Same Unit, from panels already in memory (`forecast_agent.py` passes {stem: pyarrow.Table}).

    Picks the first panel, by name, that holds every target asset, and drops rows after `asof`.
    """
    for stem in sorted(panels):
        t = panels[stem]
        df = t.to_pandas() if hasattr(t, "to_pandas") else t
        acol = "asset" if "asset" in df.columns else "asset_id" if "asset_id" in df.columns else None
        if acol is None or not set(assets) <= set(df[acol].astype(str)):
            continue
        wide = _wide(df)
        wide = wide.loc[wide.index <= pd.Timestamp(asof)]
        return Unit(unit=unit_id, panel=stem, wide=wide, asof=pd.Timestamp(asof),
                    assets=[str(a) for a in assets], horizons=[int(h) for h in horizons],
                    target_type=target_type, value_unit=value_unit,
                    freq="monthly" if stem == "macro_monthly" else "daily")
    raise ValueError(f"no single panel holds all of {assets}")


# ============================================================================
#  2 · Build features  (notebook 01, then notebook 02's per-cell feature selection)
# ============================================================================
WINDOWS = {"21": 21, "63": 63, "126": 126, "252": 252}
USD_PER_CCY = {"AUD", "EUR", "GBP", "NZD"}    # quoted USD-per-unit; the other six are units-per-USD


def working_series(wide: pd.DataFrame, asset: str, target_type: str) -> pd.Series:
    s = wide[asset].dropna()                  # the asset's own trading days (panels can be ragged)
    return np.log1p(s).cumsum() if target_type == "log_return" else s


def series_features(x: pd.Series, freq: str, is_level: bool) -> pd.DataFrame:
    """Per-series features from the target's own history. Windows are business days; monthly -> months."""
    scale = 21 if freq == "monthly" else 1
    w = {k: max(1, v // scale) for k, v in WINDOWS.items()}
    d = x.diff(); F = pd.DataFrame(index=x.index)
    F["level"]     = x if is_level else np.nan
    F["log_level"] = np.log(x.clip(lower=1e-3)) if is_level else np.nan
    for k in ("21", "63", "126", "252"): F[f"mom_{k}"] = x - x.shift(w[k])
    F["z_252"]   = (x - x.rolling(w["252"]).mean()) / x.rolling(w["252"]).std()
    F["pos_252"] = (x - x.rolling(w["252"]).min()) / (x.rolling(w["252"]).max() - x.rolling(w["252"]).min())
    for k in ("21", "63", "252"): F[f"rv_{k}"] = d.rolling(w[k]).std()
    F["ewma_vol"]  = d.ewm(alpha=1 - 0.94 ** scale).std()
    F["vol_ratio"] = F["rv_21"] / F["rv_252"]
    F["skew_252"]  = d.rolling(w["252"]).skew()
    F["last_step"] = d
    return F


def context_features(wide: pd.DataFrame, panel: str, freq: str) -> pd.DataFrame:
    """Panel context (`ctx_*`) from the other series in the same panel."""
    C = pd.DataFrame(index=wide.index); m63 = 63 if freq == "daily" else 3
    if panel == "rates_daily":
        C["ctx_slope_10_2"]  = wide["UST_10Y"] - wide["UST_2Y"]
        C["ctx_slope_5_2"]   = wide["UST_5Y"]  - wide["UST_2Y"]
        C["ctx_curv_2_5_10"] = 2 * wide["UST_5Y"] - wide["UST_2Y"] - wide["UST_10Y"]
        C["ctx_ust2y"], C["ctx_ust10y"] = wide["UST_2Y"], wide["UST_10Y"]
        C["ctx_slope_mom_63"] = C["ctx_slope_10_2"].diff(m63)
    elif panel == "g10_fx_daily":
        lg = np.log(wide).apply(lambda col: -col if col.name in USD_PER_CCY else col)
        C["ctx_usd_mom_63"] = lg.diff(m63).mean(axis=1)
        C["ctx_usd_rv_63"]  = lg.diff().mean(axis=1).rolling(m63).std()
    elif panel == "factors_daily":
        mkt, mom = np.log1p(wide["MKT"].dropna()), np.log1p(wide["MOM"].dropna())
        C["ctx_mkt_mom_63"] = mkt.cumsum().diff(m63)
        C["ctx_mkt_rv_63"]  = mkt.rolling(m63).std()
        C["ctx_mom_mom_63"] = mom.cumsum().diff(m63)
    elif panel == "macro_monthly":
        C["ctx_cpi_yoy"]  = wide["CPI_ALL"].pct_change(12) * 100
        C["ctx_core_yoy"] = wide["CPI_CORE"].pct_change(12) * 100
        C["ctx_unrate"]   = wide["UNRATE"]
        C["ctx_unrate_chg_3"] = wide["UNRATE"].diff(3)
        C["ctx_nfp_3m"]   = wide["NFP"].diff(3)
    return C


def steps_ahead_for(unit: Unit, h: int) -> int:
    """Rows ahead the target sits: h on daily panels; on monthly, months from the last row to asof + h BD."""
    if unit.freq == "daily":
        return h
    target_date = unit.asof + pd.offsets.BDay(h); last = unit.wide.index[-1]
    return (target_date.year - last.year) * 12 + (target_date.month - last.month)


@dataclass
class CellData:
    """One (asset, horizon) of the unit: rows = origins, `split` is train / predict."""
    unit: Unit
    asset: str
    horizon: int
    steps: int
    frame: pd.DataFrame           # id + anchor/target/target_change + features, sorted by origin_date
    features: list[str]           # the feature columns that are non-NaN somewhere in the train rows


ID_COLS = ["origin_date", "origin_idx", "horizon_bd", "steps_ahead", "target_date", "split"]
TARGET_COLS = ["anchor", "target", "target_change"]


def build_features(unit: Unit) -> list[CellData]:
    """Notebook 01's `build_unit_rows`, split per cell and feature-filtered the way notebook 02 does."""
    is_level = unit.target_type == "level"
    ctx = context_features(unit.wide, unit.panel, unit.freq)
    cells = []
    for asset in unit.assets:
        x = working_series(unit.wide, asset, unit.target_type)
        F = series_features(x, unit.freq, is_level); n, idx = len(x), x.index
        for h in unit.horizons:
            k = steps_ahead_for(unit, h)
            anchor = x.to_numpy() if is_level else np.zeros(n)
            tgt   = np.full(n, np.nan)
            tdate = np.full(n, np.datetime64("NaT", "ns"))
            tgt[:n - k]   = x.to_numpy()[k:] if is_level else (x.to_numpy()[k:] - x.to_numpy()[:n - k])
            tdate[:n - k] = idx[k:].to_numpy()
            D = pd.DataFrame({"origin_date": idx, "origin_idx": np.arange(n), "horizon_bd": h, "steps_ahead": k,
                              "target_date": tdate, "anchor": anchor, "target": tgt}, index=idx)
            D["target_change"] = D["target"] - D["anchor"]
            is_asof = D["origin_date"] == idx[-1]
            D["split"] = np.where(D["target"].notna(), "train", np.where(is_asof, "predict", "drop"))
            D = pd.concat([D, F, ctx.reindex(idx).ffill()], axis=1).loc[D["split"] != "drop"]
            # monthly as-of rows have no panel date to point at
            m = (D["split"] == "predict") & D["target_date"].isna()
            D.loc[m, "target_date"] = unit.asof + pd.offsets.BDay(h)

            candidates = [c for c in D.columns if c not in ID_COLS + TARGET_COLS]
            train = D[D["split"] == "train"]
            features = [c for c in candidates if train[c].notna().any()]
            frame = D[ID_COLS + TARGET_COLS + features].sort_values("origin_date").reset_index(drop=True)
            if (frame["split"] == "predict").sum() != 1:
                raise ValueError(f"{unit.unit} {asset} h{h}: expected one as-of row, "
                                 f"got {(frame['split'] == 'predict').sum()}")
            cells.append(CellData(unit, asset, h, k, frame, features))
    return cells


# ============================================================================
#  3 · Fit M2  (notebook 03, `m2_fit`, at the as-of origin)
# ============================================================================
def is_scale_feature(c: str) -> bool:
    """Scale features drive sigma; everything else drives mu."""
    return c == "log_level" or "rv" in c or "vol" in c or c == "skew_252"


def ridge_fit(X: np.ndarray, t: np.ndarray, lam: float) -> np.ndarray:
    Xb = np.c_[np.ones(len(X)), X]
    return np.linalg.solve(Xb.T @ Xb + lam * np.diag([0.0] + [1.0] * X.shape[1]), Xb.T @ t)


def ridge_pred(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ b


@dataclass
class Fit:
    asset: str
    horizon: int
    steps: int
    asof_row: pd.Timestamp        # the origin the fit stands at (the asset's last panel date)
    n_train: int
    anchor: float
    mu: float
    sigma: float
    sigma_const: float            # constant-sigma fallback, the reference for both clamps
    calib_c: float                # Jensen / calibration constant
    eps: pd.Series                # standardised residuals, indexed by origin_date
    beta: pd.Series = field(repr=False)    # mu coefficients (standardised features)
    gamma: pd.Series = field(repr=False)   # log-variance coefficients (standardised features)


def fit_m2(cell: CellData, lam_mu: float = LAM_MU, lam_sig: float = LAM_SIG) -> Fit:
    """Location-scale ridge on every row whose outcome was observed by the as-of."""
    if cell.unit.target_type != "level":
        raise ValueError(f"{cell.unit.unit}: M2 covers level targets only (notebook 03); "
                         f"this card's target_type is {cell.unit.target_type!r}")
    df = cell.frame
    mean_cols = [c for c in cell.features if not is_scale_feature(c)]
    vol_cols  = [c for c in cell.features if is_scale_feature(c)]
    odate, tdate = df["origin_date"].to_numpy(), df["target_date"].to_numpy()
    split = df["split"].to_numpy()
    complete = df[cell.features].notna().all(axis=1).to_numpy()
    i = int(np.flatnonzero(split == "predict")[0])
    origin = odate[i]
    m = (split == "train") & complete & (tdate <= origin)     # labelled, full feature row, resolved by the as-of

    Xm_all, Xv_all = df[mean_cols].to_numpy(float), df[vol_cols].to_numpy(float)
    Xm, Xv, t = Xm_all[m], Xv_all[m], df["target_change"].to_numpy(float)[m]
    n = len(t)
    if n < 2:
        raise ValueError(f"{cell.unit.unit} {cell.asset} h{cell.horizon}: only {n} training rows")
    mm, sm = Xm.mean(0), Xm.std(0) + 1e-12
    mv, sv = Xv.mean(0), Xv.std(0) + 1e-12
    Xm = np.clip((Xm - mm) / sm, -WINSOR, WINSOR); Xv = np.clip((Xv - mv) / sv, -WINSOR, WINSOR)
    xm = np.clip((Xm_all[[i]] - mm) / sm, -WINSOR, WINSOR); xv = np.clip((Xv_all[[i]] - mv) / sv, -WINSOR, WINSOR)
    if not (np.isfinite(xm).all() and np.isfinite(xv).all()):
        raise ValueError(f"{cell.unit.unit} {cell.asset} h{cell.horizon}: NaN feature on the as-of row")

    beta = ridge_fit(Xm, t, lam_mu * n)                          # 1. location
    r    = t - ridge_pred(beta, Xm)
    sig0 = float(np.sqrt(np.mean(r ** 2)))                       #    constant-sigma fallback

    gamma   = ridge_fit(Xv, np.log(r ** 2 + 1e-10), lam_sig * n) # 2. scale
    sig_raw = np.exp(ridge_pred(gamma, Xv) / 2)
    c   = float(np.sqrt(np.mean((r / sig_raw) ** 2)))            #    Jensen / calibration constant
    lo, hi = SIGMA_CLAMP[0] * sig0, SIGMA_CLAMP[1] * sig0
    sig = np.clip(sig_raw * c, lo, hi)

    return Fit(asset=cell.asset, horizon=cell.horizon, steps=cell.steps, asof_row=pd.Timestamp(origin),
               n_train=n, anchor=float(df["anchor"].iloc[i]),
               mu=float(np.clip(ridge_pred(beta, xm)[0], -MU_CLAMP_SIGMAS * sig0, MU_CLAMP_SIGMAS * sig0)),
               sigma=float(np.clip(np.exp(ridge_pred(gamma, xv)[0] / 2) * c, lo, hi)),
               sigma_const=sig0, calib_c=c,
               eps=pd.Series(r / sig, index=pd.DatetimeIndex(odate[m], name="origin_date"), name="eps"),  # 3. shape
               beta=pd.Series(beta[1:], mean_cols), gamma=pd.Series(gamma[1:], vol_cols))


# ============================================================================
#  4 · Report
# ============================================================================
def report(unit: Unit, fits: list[Fit], lam_mu: float = LAM_MU,
           lam_sig: float = LAM_SIG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(predictions, eps pools, coefficients) — the same columns notebook 03 saves, for one unit."""
    predictions = pd.DataFrame([{
        "unit": unit.unit, "asset": f.asset, "horizon_bd": f.horizon, "panel": unit.panel, "freq": unit.freq,
        "target_type": unit.target_type, "value_unit": unit.value_unit, "asof": unit.asof,
        "asof_row": f.asof_row, "steps_ahead": f.steps, "n_train": f.n_train,
        "anchor": f.anchor, "mu": f.mu, "centre": f.anchor + f.mu, "sigma": f.sigma,
        "sigma_const": f.sigma_const, "calib_c": f.calib_c,
        "lam_mu": lam_mu, "lam_sig": lam_sig, "n_eps": len(f.eps)} for f in fits])
    pools = pd.concat([pd.DataFrame({"unit": unit.unit, "asset": f.asset, "horizon_bd": f.horizon,
                                     "origin_date": f.eps.index, "eps": f.eps.to_numpy()}) for f in fits],
                      ignore_index=True)
    coefs = pd.DataFrame([{"unit": unit.unit, "asset": f.asset, "horizon_bd": f.horizon, "model": name,
                           "feature": k, "coef": float(v)}
                          for f in fits for name, s in (("mu", f.beta), ("sigma", f.gamma)) for k, v in s.items()])
    return predictions, pools, coefs


def draw_joint(fits: list[Fit], n_draws: int, seed: int,
               shift: np.ndarray | None = None, widen: np.ndarray | None = None) -> np.ndarray:
    """Notebook 04's sampler. Returns [n_draws, n_cells] in the order of `fits`.

    value = anchor + mu + shift + widen * sigma * eps(d), where draw k uses ONE historical date d_k
    for every cell -- that shared date is what makes horizons (and assets) move together.
    `shift` / `widen` are per cell; leave them None for the plain M2 forecast.
    """
    common = fits[0].eps.index
    for f in fits[1:]:
        common = common.intersection(f.eps.index)
    if len(common) == 0:
        raise ValueError("the cells' eps pools share no origin date")
    rng = np.random.default_rng(seed)
    dates = common[rng.integers(0, len(common), size=n_draws)]
    shift = np.zeros(len(fits)) if shift is None else np.asarray(shift, float)
    widen = np.ones(len(fits)) if widen is None else np.asarray(widen, float)
    return np.column_stack([f.anchor + f.mu + shift[j] + widen[j] * f.sigma * f.eps.loc[dates].to_numpy()
                            for j, f in enumerate(fits)])


def run_unit(unit_dir: str | pathlib.Path, lam_mu: float = LAM_MU, lam_sig: float = LAM_SIG):
    """All four steps. Returns (unit, fits, predictions, pools, coefs)."""
    unit = read_unit(unit_dir)
    fits = [fit_m2(c, lam_mu, lam_sig) for c in build_features(unit)]
    return (unit, fits, *report(unit, fits, lam_mu, lam_sig))


def _print_report(unit: Unit, predictions: pd.DataFrame, pools: pd.DataFrame) -> None:
    print(f"{unit.unit}  ·  panel {unit.panel} ({unit.freq})  ·  as-of {unit.asof.date()}  ·  {unit.value_unit}")
    cols = ["asset", "horizon_bd", "steps_ahead", "n_train", "anchor", "mu", "centre", "sigma",
            "sigma_const", "calib_c"]
    print(predictions[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    q = pools.groupby(["asset", "horizon_bd"])["eps"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    q = q.rename(columns={"count": "n_eps"}).drop(columns=["mean", "std"])
    q["first_date"] = pools.groupby(["asset", "horizon_bd"])["origin_date"].min().dt.date
    q["last_date"]  = pools.groupby(["asset", "horizon_bd"])["origin_date"].max().dt.date
    print("\neps pool (standardised residuals):")
    print(q.to_string(float_format=lambda v: f"{v:.3f}"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit M2 on one F1 unit and report anchor, mu, sigma, eps pool.")
    ap.add_argument("unit_dir", type=pathlib.Path, help="unit folder containing card.toml and the panel")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="write predictions / eps_pool / coefficients parquet + config.json here")
    ap.add_argument("--lam-mu", type=float, default=LAM_MU)
    ap.add_argument("--lam-sig", type=float, default=LAM_SIG)
    a = ap.parse_args(argv)

    try:
        unit, _, predictions, pools, coefs = run_unit(a.unit_dir, a.lam_mu, a.lam_sig)
    except ValueError as exc:
        print(f"m2_unit: {exc}", file=sys.stderr)
        return 1
    _print_report(unit, predictions, pools)

    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(a.out / "predictions.parquet", index=False)
        pools.to_parquet(a.out / "eps_pool.parquet", index=False)
        coefs.to_parquet(a.out / "coefficients.parquet", index=False)
        (a.out / "config.json").write_text(json.dumps({
            "model": "M2 ridge location-scale + empirical residual pool", "unit": unit.unit,
            "lam_mu": a.lam_mu, "lam_sig": a.lam_sig, "winsor": WINSOR,
            "sigma_clamp": list(SIGMA_CLAMP), "mu_clamp_sigmas": MU_CLAMP_SIGMAS}, indent=2) + "\n")
        print(f"\nwrote {a.out}/predictions.parquet, eps_pool.parquet, coefficients.parquet, config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
