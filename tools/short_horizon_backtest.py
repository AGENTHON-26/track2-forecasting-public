"""
Short-horizon pseudo-card backtest (horizons 1-4 business days) under the CURRENT leaderboard rule.

Why: the Final's ranked cards have horizons of 1 to 4 business days (organizers, issue #4,
comment 6129776242, 2026-10-12), while every practice card runs 21-128. Nothing in the model was
measured at that range. This cuts pseudo-cards out of the practice panels' own history, at many
as-of dates, and scores each model against M0 exactly as the board now does.

What it never touches: `realized_vectors/`, the Development scores, or any card's text. The
outcome of a pseudo-card is the panel's own next 1-4 rows, and the model only ever sees rows at or
before the as-of (asserted per card). The text half is held neutral; `widen` variants stand in for
what a text-driven width would cost or earn.

The rule (CHANGELOG 2026-10-06; docs/M0-BASELINE.md 3 and 5, upstream commit 60509df):
  - each component (marginal CRPS, joint variogram p=0.5, tail pinball at 1/5/95/99 %) is divided
    by the error M0 EXPECTS to make on that card, from M0's exact mean and covariance:
        marginal  mean_i sd_i / sqrt(pi)
        tail      0.0648939 * mean_i sd_i
        joint     sum_{i != j} Var(|D_ij|^(1/2)),  D_ij ~ N(mean_i - mean_j, C_ii + C_jj - 2 C_ij)
    a component that comes out 0 is stored as 1.0;
  - weights 0.5 / 0.3 / 0.2, a one-cell card redistributes to 0.5/0.7 and 0.2/0.7;
  - the card score is clipped at 8.0; the headline is the mean of the clipped scores.

    python3 tools/short_horizon_backtest.py                 # every variant, default sample
    python3 tools/short_horizon_backtest.py --verify-only   # the self-checks, nothing else
    python3 tools/short_horizon_backtest.py --dates 40 --variants M0,M2.0   # a quick look

Results go to .cache/short_horizon/ (gitignored): one CSV row per (variant, card).
"""

from __future__ import annotations

import argparse
import collections
import glob
import math
import os
import pathlib
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import gamma, hyp1f1
from scipy.stats import norm

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import forecast_models as fm  # noqa: E402
import event_vol  # noqa: E402,F401
from qfbench2_common.scoring.crps import crps_marginal, variogram_score  # noqa: E402
from qfbench2_track_forecasting.tail import tail_pinball  # noqa: E402

OUT_DIR = REPO / ".cache" / "short_horizon"
LEVELS = (0.01, 0.05, 0.95, 0.99)
PHI_MEAN = float(np.mean([norm.pdf(norm.ppf(t)) for t in LEVELS]))   # 0.0648939 (doc 5)
CAP = 8.0
HORIZONS = [1, 2, 3, 4]
M0_WINDOW, M0_DRAWS = 300, 500
MODEL_DRAWS = 2000            # what forecast_agent.py draws on the M2.0 branches
MIN_HISTORY = 800             # rows before the first as-of: room for the 260-step window and VR
FIRST_ASOF = "2003-01-01"

#: Panels used. Daily only: the Final's release cards are a different shape, and the monthly panel
#: carries vintages that disagree across units. em_transfer_early is the transfer cards' gapped
#: window, not a daily history.
PANELS = {"rates_daily": "level", "g10_fx_daily": "level", "factors_daily": "log_return"}

#: Multi-asset (F3-like) cards. One target_type per card, as on real cards.
JOINT_GROUPS = [
    ("curve", ["UST_2Y", "UST_10Y", "UST_30Y"]),
    ("g10-majors", ["EUR", "JPY", "GBP"]),
    ("g10-commodity", ["AUD", "CAD", "NOK", "NZD"]),
    ("rates-x-fx", ["UST_10Y", "EUR", "JPY"]),
    ("factors", ["MKT", "HML", "SMB", "MOM"]),
    ("curve-full", ["UST_2Y", "UST_5Y", "UST_7Y", "UST_10Y", "UST_20Y", "UST_30Y"]),
    ("fx-full", ["AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK"]),
]
JOINT_HORIZONS = [1, 4]

#: Model variants: module-level switches of forecast_models plus a uniform text-style `widen`.
#: All five M1.9/M2.0 switches off is M1.5 exactly (forecast_models docstring).
_M15 = dict(RECENT_TREND=False, ZERO_FLOOR=False, VR_SHRINK=None, TREND_SHARE=0.0, VOL_POOL=None)
VARIANTS: dict[str, dict] = {
    "M0":             {},                                    # handled separately
    "M1.5":           {**_M15},
    "M1.9":           {"VOL_POOL": None},
    "M2.0":           {},                                    # forecast_models as it stands on this branch
    "M2.0 pool21-63-252": {"VOL_POOL": (21, 63, 252)},       # M2.0 as shipped before exp/m2-short-horizon
    "M2.0 pool10-21-63": {"VOL_POOL": (10, 21, 63)},
    "M2.0 pool5-10-21": {"VOL_POOL": (5, 10, 21)},
    "M1.5 hl21":      {**_M15, "EWMA_HALFLIFE": 21},
    "M1.5 hl10":      {**_M15, "EWMA_HALFLIFE": 10},
    "M2.0 widen1.2":  {"_widen": 1.2},
    "M2.0 widen1.5":  {"_widen": 1.5},                       # F4's text floor on m2_0_text
    "M2.0 widen2.0":  {"_widen": 2.0},                       # F4's text floor on exp-f4-width-x2
    # dev's cumulative walk (forecast_models.py at `dev`), with each family's WALK_SETTINGS
    "dev walk F1":    {"_dev_family": "T2-F1"},
    "dev walk F2":    {"_dev_family": "T2-F2"},
    "dev walk F3":    {"_dev_family": "T2-F3"},
    "dev walk F4":    {"_dev_family": "T2-F4"},
    "M2.0 short":     {"VOL_POOL": (10, 21, 63)},            # candidate: same as pool10-21-63
    "zerodrift":      {"ZERO_DRIFT_SHORT": True},
    "sd1.05":         {"SD_MULT": {"rates": 1.05, "fx": 1.05, "equity_factor": 1.05}},
    "sd1.10":         {"SD_MULT": {"rates": 1.10, "fx": 1.10, "equity_factor": 1.10}},
    "sd0.95":         {"SD_MULT": {"rates": 0.95, "fx": 0.95, "equity_factor": 0.95}},
    "winsor4":        {"WINSOR": 4.0},
    "winsor5":        {"WINSOR": 5.0},
    "pca3s.3":        {"CORR_SHRINK": 0.3},
    "pca3s.6":        {"CORR_SHRINK": 0.6},
    "pca2s.5":        {"CORR_SHRINK": 0.5, "CORR_PCA_K": 2},
    "nu6":            {"NU_FIXED": 6.0},
    "nu10":           {"NU_FIXED": 10.0},
    "floor":          {"SD_FLOOR_RATES": 0.005},
    "ev":             {"_ev": "default"},                    # production: event_vol.DEFAULT_TABLE (half strength)
    "ev_full":        {"_ev": {"rates": {"FOMC": 1.2, "FOMC_next": 1.2}, "fx": {"FOMC_next": 1.6, "ECB": 1.25, "BoJ": 1.1}}},
    "ev15":           {"_ev": {"rates": {"FOMC": 1.27, "FOMC_next": 1.12}, "fx": {"FOMC_next": 1.71, "ECB": 1.20, "BoJ": 1.22}}},
    "ev20":           {"_ev": {"rates": {"FOMC": 1.13, "FOMC_next": 1.25}, "fx": {"FOMC_next": 1.62, "ECB": 1.39, "BoJ": 1.06}}},
    "ev_half":        {"_ev": {"rates": {"FOMC": 1.1, "FOMC_next": 1.1}, "fx": {"FOMC_next": 1.3, "ECB": 1.12, "BoJ": 1.05}}},
    "draws8000":      {"_draws": 8000},
    "nustd":          {"NU_STANDARDIZED": True},
}
#: Round-3 candidates: all carry the event scaling ("ev" is the current production model).
_EV = {"_ev": "default"}
for _name, _cfg in {
    "p5-21-63":   {"VOL_POOL": (5, 21, 63)},
    "p7-21-63":   {"VOL_POOL": (7, 21, 63)},
    "p10-21":     {"VOL_POOL": (10, 21)},
    "p10-30":     {"VOL_POOL": (10, 30)},
    "p15-40":     {"VOL_POOL": (15, 40)},
    "p10-21-63-126": {"VOL_POOL": (10, 21, 63, 126)},
    "p10-21-42":  {"VOL_POOL": (10, 21, 42)},
    "p14-28-63":  {"VOL_POOL": (14, 28, 63)},
    "p10x4":      {"VOL_POOL": (10, 10, 21, 63)},
    "p10-21-63x4": {"VOL_POOL": (10, 21, 63, 63)},
    "p21":        {"VOL_POOL": (21,)},
    "corr60":     {"CORR_EWMA_HL": 60},
    "corr126":    {"CORR_EWMA_HL": 126},
    "corr30":     {"CORR_EWMA_HL": 30},
    "lf.5":       {"SD_LONG_FLOOR": 0.5},
    "lf.7":       {"SD_LONG_FLOOR": 0.7},
    "lf.85":      {"SD_LONG_FLOOR": 0.85},
    "vr off":     {"VR_SHRINK": None},
    "vr10":       {"VR_SHRINK": 10.0},
    "vr100":      {"VR_SHRINK": 100.0},
    "ev+zd":      {"ZERO_DRIFT_SHORT": True},
    "dowR":       {"_ev": {"rates": {"FOMC": 1.2, "FOMC_next": 1.2, "dow": {4: 1.08}}, "fx": {"FOMC_next": 1.6, "ECB": 1.25, "BoJ": 1.1}}},
    "dowF":       {"_ev": {"rates": {"FOMC": 1.2, "FOMC_next": 1.2}, "fx": {"FOMC_next": 1.6, "ECB": 1.25, "BoJ": 1.1, "dow": {0: 0.95, 3: 1.05, 4: 1.05}}}},
    "dowRF":      {"_ev": {"rates": {"FOMC": 1.2, "FOMC_next": 1.2, "dow": {4: 1.08}}, "fx": {"FOMC_next": 1.6, "ECB": 1.25, "BoJ": 1.1, "dow": {0: 0.95, 3: 1.05, 4: 1.05}}}},
    "nu5k0.9": {"NU_FIXED": 5.0, "SD_MULT": {"rates": 0.9, "fx": 0.9, "equity_factor": 0.9}},
    "nu5k0.95": {"NU_FIXED": 5.0, "SD_MULT": {"rates": 0.95, "fx": 0.95, "equity_factor": 0.95}},
    "nu5k1.0": {"NU_FIXED": 5.0, "SD_MULT": {"rates": 1.0, "fx": 1.0, "equity_factor": 1.0}},
    "nu6k0.9": {"NU_FIXED": 6.0, "SD_MULT": {"rates": 0.9, "fx": 0.9, "equity_factor": 0.9}},
    "nu6k0.95": {"NU_FIXED": 6.0, "SD_MULT": {"rates": 0.95, "fx": 0.95, "equity_factor": 0.95}},
    "nu6k1.0": {"NU_FIXED": 6.0, "SD_MULT": {"rates": 1.0, "fx": 1.0, "equity_factor": 1.0}},
    "nu8k0.9": {"NU_FIXED": 8.0, "SD_MULT": {"rates": 0.9, "fx": 0.9, "equity_factor": 0.9}},
    "nu8k0.95": {"NU_FIXED": 8.0, "SD_MULT": {"rates": 0.95, "fx": 0.95, "equity_factor": 0.95}},
    "nu8k1.0": {"NU_FIXED": 8.0, "SD_MULT": {"rates": 1.0, "fx": 1.0, "equity_factor": 1.0}},
    "k0.9": {"SD_MULT": {"rates": 0.9, "fx": 0.9, "equity_factor": 0.9}},
    "k0.95": {"SD_MULT": {"rates": 0.95, "fx": 0.95, "equity_factor": 0.95}},
    "vov.10":     {"VOV_SD": 0.10},
    "vov.20":     {"VOV_SD": 0.20},
    "vov.10 nu10": {"VOV_SD": 0.10, "NU_FIXED": 10.0},
    "vov.20 nu15": {"VOV_SD": 0.20, "NU_FIXED": 15.0},
    "vov.20 nomix": {"VOV_SD": 0.20, "VOL_POOL": None},
}.items():
    VARIANTS["e " + _name] = {**_EV, **_cfg}
_DEV_MODULE = REPO / ".cache" / "short_horizon" / "devmod" / "forecast_models_dev.py"
_fm_dev = None


def _dev_model():
    """dev's forecast_models.py, from `git show dev:forecast_models.py` (written by main)."""
    global _fm_dev
    if _fm_dev is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("forecast_models_dev", _DEV_MODULE)
        _fm_dev = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _fm_dev          # dataclasses look the module up while defining
        spec.loader.exec_module(_fm_dev)
    return _fm_dev
_DEFAULTS = {k: getattr(fm, k) for k in
             ("RECENT_TREND", "ZERO_FLOOR", "VR_SHRINK", "TREND_SHARE", "VOL_POOL", "EWMA_HALFLIFE",
              "ZERO_DRIFT_SHORT", "SD_MULT", "CORR_SHRINK", "CORR_PCA_K", "EVENT_SCALE", "NU_FIXED", "VOV_SD", "SD_LONG_FLOOR", "CORR_EWMA_HL", "WINSOR", "NU_STANDARDIZED", "SD_FLOOR_RATES")}


# ----------------------------------------------------------------------------- data
def load_panels() -> dict[str, pd.DataFrame]:
    """Every practice unit's rows, pooled per panel. The worked example is excluded: its rates rows
    disagree with every other unit's on 496 (asset, date) pairs, and no other unit disagrees."""
    frames = collections.defaultdict(list)
    for f in sorted(glob.glob(str(REPO / "units" / "*" / "*.parquet"))):
        if "t2-EXAMPLE" in f:
            continue
        t = pq.read_table(f).to_pandas()
        if "panel_id" not in t or t["panel_id"].iloc[0] not in PANELS:
            continue
        acol = "asset" if "asset" in t else "asset_id"
        frames[t["panel_id"].iloc[0]].append(t.rename(columns={acol: "asset"})[["date", "asset", "value", "panel_id"]])
    out = {}
    for pid, fs in frames.items():
        t = pd.concat(fs)
        t["date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
        dup = t.groupby(["asset", "date"])["value"].nunique()
        assert (dup <= 1).all(), f"{pid}: units disagree on {int((dup > 1).sum())} rows"
        out[pid] = t.drop_duplicates(["asset", "date"]).sort_values(["asset", "date"]).reset_index(drop=True)
    return out


def make_cards(panels: dict[str, pd.DataFrame], n_dates: int, seed: int) -> list[dict]:
    """Pseudo-cards: one per (asset, as-of) with horizons 1-4, plus the joint groups at [1, 4].
    As-of dates are drawn without replacement from rows with MIN_HISTORY rows behind them and four
    ahead, after FIRST_ASOF; the draw is seeded, so a run is reproducible."""
    rng = np.random.default_rng(seed)
    where = {a: pid for pid, t in panels.items() for a in t["asset"].unique()}
    dates = {a: panels[where[a]].loc[panels[where[a]]["asset"] == a, "date"].tolist() for a in where}
    cards = []
    for a in sorted(where):
        ok = [d for i, d in enumerate(dates[a]) if MIN_HISTORY <= i < len(dates[a]) - 4 and d >= FIRST_ASOF]
        for d in sorted(rng.choice(ok, size=min(n_dates, len(ok)), replace=False)):
            cards.append({"kind": "single", "group": where[a], "assets": [a], "horizons": HORIZONS,
                          "asof": str(d), "target_type": PANELS[where[a]]})
    for name, assets in JOINT_GROUPS:
        tt = {PANELS[where[a]] for a in assets}
        assert len(tt) == 1, name
        common = set.intersection(*(set(dates[a][MIN_HISTORY:-4]) for a in assets))
        ok = sorted(d for d in common if d >= FIRST_ASOF)
        for d in sorted(rng.choice(ok, size=min(n_dates, len(ok)), replace=False)):
            cards.append({"kind": "joint", "group": name, "assets": assets, "horizons": JOINT_HORIZONS,
                          "asof": str(d), "target_type": next(iter(tt))})
    for i, c in enumerate(cards):
        c["id"] = f"sh-{c['group']}-{'-'.join(c['assets'])}-{c['asof']}"
        c["n"] = i
    return cards


def card_inputs(panels: dict[str, pd.DataFrame], card: dict) -> tuple[dict[str, pa.Table], dict, np.ndarray]:
    """(the panels the model may see, per-asset history rows, the realized vector in grid order).
    The model's panels hold only this card's assets, only rows at or before the as-of."""
    asof = card["asof"]
    seen, hist, y = {}, {}, []
    for a in card["assets"]:
        pid, t = next((p, t) for p, t in panels.items() if a in set(t["asset"].unique()))
        rows = t[t["asset"] == a]
        past = rows[rows["date"] <= asof]
        future = rows[rows["date"] > asof]["value"].to_numpy()
        hist[a] = list(zip(past["date"], past["value"].astype(float)))
        seen.setdefault(pid, []).append(past)
        for h in card["horizons"]:
            assert len(future) >= h, (card["id"], a, h)
            y.append(float(future[:h].sum()) if card["target_type"] == "log_return" else float(future[h - 1]))
    tables = {pid: pa.Table.from_pandas(pd.concat(fs), preserve_index=False) for pid, fs in seen.items()}
    for t in tables.values():
        assert max(t.column("date").to_pylist()) <= asof, "look-ahead"
    return tables, hist, np.array(y)


# ----------------------------------------------------------------------------- M0 (doc 3)
def m0_distribution(hist: dict, assets: list[str], horizons: list[int], target_type: str):
    """M0's exact mean vector and covariance over the card's cells, in grid order (asset, then
    horizon, in the card's own order). Doc 3.1-3.8; daily panels, so a horizon is its step count."""
    steps, anchor = {}, {}
    for a in sorted(assets):
        rows = hist[a][-M0_WINDOW:]
        dates = [pd.Timestamp(d) for d, _ in rows]
        vals = np.array([v for _, v in rows])
        gaps = np.array([(dates[i] - dates[i - 1]).days for i in range(1, len(dates))], dtype=float)
        hole = max(10.0 * float(np.median(gaps)), 5.0)
        steps[a] = {dates[i]: (vals[i] - vals[i - 1] if target_type == "level" else vals[i])
                    for i in range(1, len(dates)) if gaps[i - 1] <= hole}
        anchor[a] = float(vals[-1]) if target_type == "level" else 0.0
    by_asset = sorted(assets)
    common = sorted(set.intersection(*(set(steps[a]) for a in by_asset)))
    x = np.array([[steps[a][d] for d in common] for a in by_asset])
    mu, sigma = x.mean(axis=1), np.atleast_2d(np.cov(x))
    ix = {a: i for i, a in enumerate(by_asset)}
    cells = [(a, h) for a in assets for h in horizons]
    mean = np.array([anchor[a] + h * mu[ix[a]] for a, h in cells])
    cov = np.array([[min(h1, h2) * sigma[ix[a1], ix[a2]] for a2, h2 in cells] for a1, h1 in cells])
    cov[np.diag_indices_from(cov)] += 1e-10
    return mean, cov


def m0_draws(mean: np.ndarray, cov: np.ndarray, seed: int) -> np.ndarray:
    jit = cov + 1e-9 * np.eye(len(mean))
    try:
        factor = np.linalg.cholesky(jit)
    except np.linalg.LinAlgError:
        factor = np.diag(np.sqrt(np.diag(jit)))
    rng = np.random.default_rng(seed)
    return mean + rng.standard_normal((M0_DRAWS, len(mean))) @ factor.T


def divisors(mean: np.ndarray, cov: np.ndarray) -> tuple[float, float, float]:
    """(marginal, joint, tail) expected-error divisors, doc 5. A zero component is stored as 1.0."""
    c = cov + 1e-9 * np.eye(len(mean))
    sd = np.sqrt(np.diag(c))
    marg = float(np.mean(sd) / math.sqrt(math.pi))
    tail = float(PHI_MEAN * np.mean(sd))
    joint = 0.0
    k = 2 ** 0.25 * gamma(0.75) / math.sqrt(math.pi)
    for i in range(len(mean)):
        for j in range(len(mean)):
            if i == j:
                continue
            delta = mean[i] - mean[j]
            s2 = c[i, i] + c[j, j] - 2 * c[i, j]
            if s2 <= 0:
                continue
            s = math.sqrt(s2)
            e_abs = s * math.sqrt(2 / math.pi) * math.exp(-delta ** 2 / (2 * s2)) + delta * (1 - 2 * norm.cdf(-delta / s))
            e_sqrt = math.sqrt(s) * k * hyp1f1(-0.25, 0.5, -delta ** 2 / (2 * s2))
            joint += e_abs - e_sqrt ** 2
    return (marg or 1.0), (joint or 1.0), (tail or 1.0)


def score(samples: np.ndarray, y: np.ndarray, div: tuple[float, float, float]) -> tuple[float, dict]:
    """(clipped card score, components). `samples` is (n_draws, cells) in grid order."""
    dm, dj, dt = div
    m = crps_marginal(samples, y)
    t = tail_pinball(samples, y, LEVELS)
    if samples.shape[1] == 1:
        raw = (0.5 * m / dm + 0.2 * t / dt) / 0.7
        j = 0.0
    else:
        j = variogram_score(samples, y, p=0.5)
        raw = 0.5 * m / dm + 0.3 * j / dj + 0.2 * t / dt
    return min(raw, CAP), {"raw": raw, "marg": m / dm, "joint": j / dj if samples.shape[1] > 1 else np.nan, "tail": t / dt}


# ----------------------------------------------------------------------------- one card
def event_scale_for(card: dict, table) -> dict:
    """{(asset, h): multiplier} from event_vol for this pseudo-card (dates = the panel's next rows)."""
    import event_vol as ev
    out = {}
    for a in card["assets"]:
        pid, t = next((p, t) for p, t in _PANELS.items() if a in set(t["asset"].unique()))
        cls = {"rates_daily": "rates", "g10_fx_daily": "fx"}.get(pid)
        if cls is None:
            continue
        fut = t[(t["asset"] == a) & (t["date"] > card["asof"])]["date"].tolist()[:max(card["horizons"])]
        sc = ev.horizon_scale(cls, fut, table)
        for h in card["horizons"]:
            out[(a, h)] = sc[h - 1]
    return out


def model_draws(variant: str, tables, card: dict) -> np.ndarray:
    cfg = VARIANTS[variant]
    if "_dev_family" in cfg:
        adj = {a: {"shift": 0.0, "widen": 1.0, "skew": 0.0} for a in card["assets"]}
        seed = zlib.crc32(card["id"].encode()) & 0x7FFFFFFF
        out = _dev_model().build_draws(tables, card["assets"], card["horizons"], card["asof"], adj, 500, seed,
                                       target_type=card["target_type"], family=cfg["_dev_family"])
        return out.reshape(out.shape[0], -1)
    for k, v in _DEFAULTS.items():
        setattr(fm, k, cfg.get(k, v))
    widen = float(cfg.get("_widen", 1.0))
    fm.EVENT_SCALE = None
    if "_ev" in cfg:
        import event_vol as ev
        tbl = None if cfg["_ev"] == "default" else cfg["_ev"]
        fm.EVENT_SCALE = event_scale_for(card, tbl)
    adj = {a: {"shift": 0.0, "widen": widen, "skew": 0.0} for a in card["assets"]}
    seed = zlib.crc32(card["id"].encode()) & 0x7FFFFFFF
    out = fm.build_draws(tables, card["assets"], card["horizons"], card["asof"], adj, int(cfg.get("_draws", MODEL_DRAWS)), seed,
                         target_type=card["target_type"], family=None)
    return out.reshape(out.shape[0], -1)          # (draws, assets x horizons): asset-major, card order


def run_card(args) -> list[dict]:
    card, variants = args
    tables, hist, y = card_inputs(_PANELS, card)
    mean, cov = m0_distribution(hist, card["assets"], card["horizons"], card["target_type"])
    div = divisors(mean, cov)
    seed = zlib.crc32(card["id"].encode()) & 0x7FFFFFFF
    sd0 = np.sqrt(np.diag(cov))
    z0 = float(np.max(np.abs(y - mean) / sd0))    # the outcome's surprise under M0, in M0 sds
    rows = []
    for v in variants:
        if card["kind"] == "single":
            # a single-asset card's four horizons are scored as four one-cell cards (the Final's
            # single-asset cards are one-cell); one model call serves all four
            s_all = m0_draws(mean, cov, seed) if v == "M0" else model_draws(v, tables, card)
            for k, h in enumerate(card["horizons"]):
                cm, cc = mean[k:k + 1], cov[k:k + 1, k:k + 1]
                sc, comp = score(s_all[:, k:k + 1], y[k:k + 1], divisors(cm, cc))
                rows.append({"variant": v, "id": card["id"], "kind": "single", "group": card["group"],
                             "asset": card["assets"][0], "h": h, "asof": card["asof"], "score": sc,
                             "evt": float(event_scale_for(card, None).get((card["assets"][0], h), 1.0)),
                             "z0": float(abs(y[k] - cm[0]) / math.sqrt(cc[0, 0])), **comp})
        else:
            s = m0_draws(mean, cov, seed) if v == "M0" else model_draws(v, tables, card)
            sc, comp = score(s, y, div)
            rows.append({"variant": v, "id": card["id"], "kind": "joint", "group": card["group"],
                         "asset": "+".join(card["assets"]), "h": 0, "asof": card["asof"], "score": sc,
                         "z0": z0, **comp})
    return rows


_PANELS: dict[str, pd.DataFrame] = {}


def _init(panels):
    global _PANELS
    _PANELS = panels


# ----------------------------------------------------------------------------- checks
def verify(panels: dict[str, pd.DataFrame]) -> None:
    """Self-checks. Each prints what it checks and its result; any failure raises."""
    print("== verify")
    rng = np.random.default_rng(7)

    # 1. The divisors are M0's EXPECTED error: score M0's own exact distribution against outcomes
    #    drawn from that same distribution and the average should be 1.0 on each component.
    #    A huge ensemble stands in for the exact distribution, so this tests the formulas, not MC.
    cases = [(np.array([0.0]), np.array([[0.01]])),
             (np.array([1.0, 1.2, 0.9, 1.4]),
              np.array([[1, .5, .7, .5], [.5, 2, .5, 1], [.7, .5, 1.5, .7], [.5, 1, .7, 3.]]) * 0.04)]
    for mean, cov in cases:
        dm, dj, dt = divisors(mean, cov)
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(mean)))
        big = mean + rng.standard_normal((20000, len(mean))) @ L.T
        m, j, t = [], [], []
        for _ in range(300):
            yy = mean + L @ rng.standard_normal(len(mean))
            m.append(crps_marginal(big[:4000], yy))
            t.append(tail_pinball(big, yy, LEVELS))
            if len(mean) > 1:
                j.append(variogram_score(big[:4000], yy, p=0.5))
        print(f"   d={len(mean)}: E[marginal]/div = {np.mean(m) / dm:.3f}   E[tail]/div = {np.mean(t) / dt:.3f}"
              + (f"   E[joint]/div = {np.mean(j) / dj:.3f}" if j else ""))
        for r in [np.mean(m) / dm, np.mean(t) / dt] + ([np.mean(j) / dj] if j else []):
            assert 0.9 < r < 1.1, "a divisor is off by more than 10 %"

    # 2. The E|D|^(1/2) closed form against numerical integration, at delta = 0 and delta != 0.
    from scipy.integrate import quad
    for delta, s in [(0.0, 1.0), (0.7, 0.5), (-2.0, 1.3)]:
        num = quad(lambda z: abs(delta + s * z) ** 0.5 * norm.pdf(z), -12, 12, points=[-delta / s], limit=200)[0]
        cf = math.sqrt(s) * 2 ** 0.25 * gamma(0.75) / math.sqrt(math.pi) * hyp1f1(-0.25, 0.5, -delta ** 2 / (2 * s * s))
        print(f"   E|D|^0.5 delta={delta:+.1f} s={s}: closed form {cf:.6f}  numeric {num:.6f}")
        assert abs(cf - num) < 1e-6

    # 3. M0 here against the repo's m0_baseline.py on a real single-asset card: same mean and
    #    covariance must give the same draws from the same seed.
    import m0_baseline, tomllib
    unit = REPO / "units" / "t2-F1-cad-boc-2017"
    card = tomllib.loads((unit / "card.toml").read_text())
    tg = card["targets"]
    hist = {}
    for a in tg["asset_ids"]:
        hist[a] = m0_baseline._history(unit, a, card["provenance"]["data_cutoff"])
    mean, cov = m0_distribution(hist, list(tg["asset_ids"]), [int(h) for h in tg["horizons"]], tg["target_type"])
    ours = m0_draws(mean, cov, zlib.crc32(card["task"]["id"].encode()) & 0x7FFFFFFF)
    theirs = m0_baseline.draws(unit).reshape(M0_DRAWS, -1)
    print(f"   M0 vs m0_baseline.py on {unit.name}: max |diff| = {np.max(np.abs(ours - theirs)):.2e}")
    assert np.allclose(ours, theirs)

    # 4. Production parity: the "M2.0" variant is forecast_models as shipped (every switch at its
    #    default, widen 1.0), and switching variants back and forth leaves no residue.
    cards = make_cards(panels, 1, seed=11)
    c = next(x for x in cards if x["kind"] == "joint")
    tables, _, _ = card_inputs(panels, c)
    a1 = model_draws("M2.0", tables, c)
    model_draws("M1.5 hl21", tables, c)
    a2 = model_draws("M2.0", tables, c)
    for k, v in _DEFAULTS.items():
        setattr(fm, k, v)
    ref = fm.build_draws(tables, c["assets"], c["horizons"], c["asof"],
                         {a: {"shift": 0.0, "widen": 1.0, "skew": 0.0} for a in c["assets"]},
                         MODEL_DRAWS, zlib.crc32(c["id"].encode()) & 0x7FFFFFFF,
                         target_type=c["target_type"], family=None).reshape(MODEL_DRAWS, -1)
    print(f"   M2.0 variant vs shipped build_draws on {c['id']}: identical={np.array_equal(a1, ref)}, "
          f"after a variant switch identical={np.array_equal(a2, ref)}")
    assert np.array_equal(a1, ref) and np.array_equal(a2, ref)

    # 5. No look-ahead and the right outcome: the realized value is the panel's own next row(s).
    s = next(x for x in cards if x["kind"] == "single" and x["target_type"] == "level")
    tables, hist, y = card_inputs(panels, s)
    t = panels[s["group"]]
    rows = t[(t["asset"] == s["assets"][0]) & (t["date"] > s["asof"])]["value"].to_numpy()
    assert np.allclose(y, rows[:4]) and hist[s["assets"][0]][-1][0] <= s["asof"]
    print(f"   outcome = next 1-4 rows after the as-of, history ends at/before it: ok ({s['id']})")
    print("== verify: all checks passed\n")


# ----------------------------------------------------------------------------- report
def report(df: pd.DataFrame) -> None:
    variants = list(dict.fromkeys(df["variant"]))
    key = ["id", "h"]
    piv = df.pivot_table(index=key, columns="variant", values="score")
    print(f"cards: {piv.shape[0]} scored cells ({(df[df.variant == variants[0]].kind == 'single').sum()} one-cell, "
          f"{(df[df.variant == variants[0]].kind == 'joint').sum()} joint)\n")

    def line(sub: pd.DataFrame, label: str) -> None:
        p = sub.pivot_table(index=key, columns="variant", values="score")
        cells = " ".join(f"{p[v].mean():7.3f}" for v in variants)
        print(f"  {label:26s} n={len(p):5d}  {cells}")

    print(f"  {'mean clipped score':26s} {'':7s}  " + " ".join(f"{v[:7]:>7s}" for v in variants))
    line(df, "ALL")
    for (kind, group), sub in df.groupby(["kind", "group"]):
        line(sub, f"{kind}:{group}")
    for h in HORIZONS:
        line(df[(df.kind == "single") & (df.h == h)], f"one-cell h={h}")
    calm, shock = df[df.z0 < 2], df[df.z0 >= 3]
    line(calm, "outcome within 2 M0-sd")
    line(shock, "outcome beyond 3 M0-sd")

    print("\n  paired vs M2.0 (same cells): mean diff, better/worse, t")
    ref = "M2.0" if "M2.0" in variants else variants[min(1, len(variants) - 1)]
    for v in variants:
        if v == ref:
            continue
        d = piv[v] - piv[ref]
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std() > 0 else 0.0
        print(f"    {v:20s} {d.mean():+.4f}   {int((d < -1e-12).sum()):5d}/{int((d > 1e-12).sum()):<5d}  t={t:+.2f}")
    print("\n  median, share of cells under 1.0, share clipped at 8:")
    for v in variants:
        s = df[df.variant == v]["score"]
        print(f"    {v:20s} median {s.median():.3f}   <1.0 {100 * (s < 1).mean():5.1f}%   =8.0 {100 * (s >= CAP).mean():5.2f}%")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", type=int, default=120, help="as-of dates per asset and per joint group")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args(argv)

    import subprocess
    _DEV_MODULE.parent.mkdir(parents=True, exist_ok=True)
    _DEV_MODULE.write_bytes(subprocess.run(["git", "show", "dev:forecast_models.py"], cwd=REPO,
                                           check=True, capture_output=True).stdout)
    panels = load_panels()
    _init(panels)
    if not a.skip_verify:
        verify(panels)
    if a.verify_only:
        return 0
    variants = a.variants.split(",")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {list(VARIANTS)}")
    cards = make_cards(panels, a.dates, a.seed)
    print(f"{len(cards)} pseudo-cards x {len(variants)} variants, {a.workers} workers")
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(panels,)) as ex:
        for i, r in enumerate(ex.map(run_card, [(c, variants) for c in cards], chunksize=8)):
            rows.extend(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(cards)} cards, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"results_seed{a.seed}_d{a.dates}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} rows, {time.time() - t0:.0f}s)\n")
    report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
