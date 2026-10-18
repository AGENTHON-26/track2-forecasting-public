"""
Robustness check for the Final's new series: synthetic cards the practice set does not have.

The Final's ranked cards (organizers, issue #4, comment 6129776242) add Treasury bills (3M, 6M, 1Y),
five EM currencies from the ECB reference rates (MXN, BRL, ZAR, KRW, INR), a few official-release
cards (GDP, inflation, jobless claims), horizons of 1-4 business days and an as-of of 26 October.
An unscored card counts 8.0, so the first job is that every such card produces admissible output.

Each case writes a synthetic unit (card.toml, forecast_spec.json, panels, a copied text folder),
runs the real agent CLI on it in a subprocess, then runs the organizers' scorer in gates-only mode
(no realized vector). The panel ids of the new series are not published yet, so several plausible
spellings are tried. The House endpoint is unset, so the text half falls back to neutral fast.

    python3 tools/robustness_cards.py            # every case
    python3 tools/robustness_cards.py --keep     # keep the synthetic units for inspection

The data are synthetic random walks at realistic levels and volatilities -- this checks that the
agent runs and its output is admissible and sane, not how well it forecasts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parent.parent
TEXT_SRC = REPO / "units" / "t2-F1-cad-boc-2017" / "text"
ASOF = "2026-10-26"
START = "2012-01-02"

#: (asset, start level, daily sd, kind) -- kind "yield" walks in levels, "fx" in logs
SERIES = {
    "UST_3M": (4.0, 0.02, "yield"), "UST_6M": (3.9, 0.025, "yield"), "UST_1Y": (3.7, 0.03, "yield"),
    "UST_10Y": (4.1, 0.06, "yield"),
    "UST_3M_Z": (0.05, 0.003, "yield"), "UST_6M_Z": (0.08, 0.004, "yield"), "UST_1Y_Z": (0.12, 0.005, "yield"),
    "UST_2Y": (4.0, 0.02, "yield"), "UST_5Y": (4.0, 0.02, "yield"), "UST_30Y": (4.5, 0.02, "yield"),
    "GBP": (0.85, 0.004, "fx"), "JPY": (160.0, 0.006, "fx"), "CHF": (0.95, 0.004, "fx"), "AUD": (1.6, 0.005, "fx"),
    "CAD": (1.45, 0.004, "fx"), "NZD": (1.75, 0.005, "fx"), "SEK": (11.2, 0.004, "fx"), "NOK": (11.6, 0.005, "fx"),
    "MXN": (18.5, 0.008, "fx"), "BRL": (5.4, 0.010, "fx"), "ZAR": (18.0, 0.010, "fx"),
    "KRW": (1380.0, 0.005, "fx"), "INR": (84.0, 0.003, "fx"), "EUR": (1.08, 0.005, "fx"),
}
#: Dates the ECB publishes no reference rate (TARGET holidays), and US bond-market holidays, for the
#: years the synthetic panels cover -- enough to make the two calendars disagree as they will.
def _ecb_holidays(years):
    out = set()
    for y in years:
        out |= {dt.date(y, 1, 1), dt.date(y, 5, 1), dt.date(y, 12, 25), dt.date(y, 12, 26)}
    return out


def _us_holidays(years):
    out = set()
    for y in years:
        out |= {dt.date(y, 1, 1), dt.date(y, 7, 4), dt.date(y, 12, 25), dt.date(y, 11, 11)}
    return out


def _walk(asset: str, dates: list[dt.date], seed: int) -> np.ndarray:
    level, sd, kind = SERIES[asset]
    rng = np.random.default_rng(seed)
    z = rng.standard_t(5, len(dates)) * sd / np.sqrt(5 / 3)
    return level + np.cumsum(z) if kind == "yield" else level * np.exp(np.cumsum(z))


def daily_panel(panel_id: str, assets: list[str], calendar: str, seed: int, n_rows: int | None = None,
                nan_every: int | None = None, constant: str | None = None) -> pd.DataFrame:
    days = pd.bdate_range(START, ASOF).date
    years = range(int(START[:4]), int(ASOF[:4]) + 1)
    hol = _ecb_holidays(years) if calendar == "ecb" else _us_holidays(years)
    days = [d for d in days if d not in hol]
    if n_rows:
        days = days[-n_rows:]
    rows = []
    for i, a in enumerate(assets):
        v = _walk(a, days, seed + i)
        if constant == a:
            v = np.full(len(days), SERIES[a][0])
        if nan_every:
            v = v.astype(float)
            v[::nan_every] = np.nan
        rows += [(d.isoformat(), a, float(x), panel_id) for d, x in zip(days, v)]
    return pd.DataFrame(rows, columns=["date", "asset", "value", "panel_id"])


def release_panel(panel_id: str) -> pd.DataFrame:
    """Official releases at their own frequencies: quarterly GDP growth, monthly CPI inflation,
    weekly initial jobless claims. Each row is dated at its reference period's start."""
    rng = np.random.default_rng(5)
    rows = []
    q = pd.date_range("2012-01-01", "2026-07-01", freq="QS").date
    rows += [(d.isoformat(), "GDP_QOQ_SAAR", float(2.0 + rng.normal(0, 1.5)), panel_id) for d in q]
    m = pd.date_range("2012-01-01", "2026-09-01", freq="MS").date
    cpi = 2.5 + np.cumsum(rng.normal(0, 0.15, len(m)))
    rows += [(d.isoformat(), "CPI_YOY", float(x), panel_id) for d, x in zip(m, cpi)]
    w = pd.date_range("2012-01-07", "2026-10-17", freq="W-SAT").date
    icsa = 220000 + np.cumsum(rng.normal(0, 6000, len(w)))
    rows += [(d.isoformat(), "ICSA", float(x), panel_id) for d, x in zip(w, icsa)]
    return pd.DataFrame(rows, columns=["date", "asset", "value", "panel_id"])


def write_unit(root: pathlib.Path, uid: str, family: str, panels: list[pd.DataFrame], assets: list[str],
               horizons: list[int], target_type: str = "level", frequency: str = "daily") -> pathlib.Path:
    unit = root / uid
    unit.mkdir(parents=True)
    pids = []
    for p in panels:
        pid = p["panel_id"].iloc[0]
        pids.append(pid)
        pq.write_table(pa.Table.from_pandas(p, preserve_index=False), unit / f"{pid}.parquet")
    shutil.copytree(TEXT_SRC, unit / "text")
    card = f'''schema_version = "2.0"

[task]
id    = "{uid}"
track = "forecasting"
title = "synthetic robustness card"
split = "validation"

[metadata]
category         = "T2-{family}"
asset_panel      = "{pids[0]}"
horizons         = {horizons}
n_assets         = {len(assets)}
target_type      = "{target_type}"
target_frequency = "{frequency}"

[provenance]
data_cutoff = "{ASOF}"

[scoring]
verifier            = "t2.crps_composite"
metric              = "crps_composite"
admissibility_gates = ["g0_integrity", "g1_schema", "g2_cutoff_resource", "g3_domain_semantics"]

[scoring.params]
tail_levels    = [0.01, 0.05, 0.95, 0.99]
joint          = "variogram"
representation = "samples"
n_draws_min    = 200

[scoring.params.weights]
marginal = 0.5
joint    = 0.3
tail     = 0.2

[panels]
panel_ids = {json.dumps(pids)}

[text]
path   = "text/"
cutoff = "{ASOF}"

[targets]
asset_ids        = {json.dumps(assets)}
horizons         = {horizons}
target_type      = "{target_type}"
target_frequency = "{frequency}"
value_unit       = "synthetic"
'''
    (unit / "card.toml").write_text(card)
    (unit / "forecast_spec.json").write_text(json.dumps({
        "schema_version": "2", "card_id": uid, "track": "forecasting", "card_family": f"T2-{family}",
        "targets": {"asset_ids": assets, "horizons": horizons, "target_type": target_type,
                    "n_assets": len(assets)},
        "output_format": "samples", "n_draws_min": 200}, indent=2))
    return unit


def cases(root: pathlib.Path) -> list[tuple[str, pathlib.Path, str]]:
    """(label, unit dir, what a pass means)."""
    bills = daily_panel("rates_daily", ["UST_3M", "UST_6M", "UST_1Y", "UST_10Y"], "us", 1)
    em = ["MXN", "BRL", "ZAR", "KRW", "INR"]
    out = []
    out.append(("T-bill, one cell h=1", write_unit(root, "syn-F1-bill-3m", "F1", [bills], ["UST_3M"], [1]), "runs"))
    out.append(("T-bills curve, F3 h=[1,4]",
                write_unit(root, "syn-F3-bills", "F3", [bills], ["UST_3M", "UST_6M", "UST_1Y", "UST_10Y"], [1, 4]), "runs"))
    for pid in ("ecb_fx_daily", "em_fx_daily", "fx_ecb_daily", "ecb_reference_rates"):
        p = daily_panel(pid, em + ["EUR"], "ecb", 2)
        out.append((f"EM FX one cell, panel '{pid}'", write_unit(root, f"syn-F2-mxn-{pid}", "F2", [p], ["MXN"], [2]), "runs"))
    p = daily_panel("ecb_fx_daily", em + ["EUR"], "ecb", 3)
    out.append(("EM FX five, F3 h=[1,2,3,4]", write_unit(root, "syn-F3-em5", "F3", [p], em, [1, 2, 3, 4]), "runs"))
    out.append(("UST (US calendar) x EM FX (ECB calendar)",
                write_unit(root, "syn-F3-ust-x-em", "F3", [bills, p], ["UST_10Y", "MXN", "INR"], [1, 4]), "runs"))
    out.append(("F4 EM FX h=4", write_unit(root, "syn-F4-zar", "F4", [p], ["ZAR"], [4]), "runs"))
    rel = release_panel("official_releases")
    for a, freq in (("GDP_QOQ_SAAR", "quarterly"), ("CPI_YOY", "monthly"), ("ICSA", "weekly")):
        out.append((f"release card {a} ({freq})", write_unit(root, f"syn-F1-{a.lower()}", "F1", [rel], [a], [3], frequency=freq), "runs"))
    rel_m = release_panel("macro_monthly")
    out.append(("release card CPI_YOY on a 'macro_monthly' panel",
                write_unit(root, "syn-F1-cpiyoy-macro", "F1", [rel_m], ["CPI_YOY"], [2], frequency="monthly"), "runs"))
    zirp = daily_panel("rates_daily", ["UST_3M_Z", "UST_6M_Z", "UST_1Y_Z"], "us", 11)
    zirp["value"] = zirp["value"].clip(lower=0.0)          # CMT yields do not go below 0.00; many flat days
    out.append(("ZIRP-style bills near 0 %, F3 h=[1,4]", write_unit(root, "syn-F3-zirp", "F3", [zirp], ["UST_3M_Z", "UST_6M_Z", "UST_1Y_Z"], [1, 4]), "runs"))
    out.append(("ZIRP bill, one cell h=2", write_unit(root, "syn-F1-zirp1", "F1", [zirp], ["UST_3M_Z"], [2]), "runs"))
    flat = daily_panel("rates_daily", ["UST_3M_Z", "UST_6M_Z"], "us", 14, constant="UST_3M_Z")
    out.append(("flat bill (constant 0.05 for 14y), h=[1,4]", write_unit(root, "syn-F3-flatbill", "F3", [flat], ["UST_3M_Z", "UST_6M_Z"], [1, 4]), "runs"))
    out.append(("panel id fx_rates_daily typed as FX", write_unit(root, "syn-F2-fxrates", "F2",
                [daily_panel("fx_rates_daily", ["MXN", "EUR"], "ecb", 15)], ["MXN"], [2]), "runs"))
    fx15 = ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK", "MXN", "BRL", "ZAR", "KRW", "INR"]
    pfx = daily_panel("ecb_fx_daily", fx15, "ecb", 12)
    out.append(("14-asset FX card h=[1,2,3,4] (56 cells)", write_unit(root, "syn-F3-fx14", "F3", [pfx], fx15, [1, 2, 3, 4]), "runs"))
    curve = ["UST_3M", "UST_6M", "UST_1Y", "UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"]
    pc = daily_panel("rates_daily", curve, "us", 13)
    out.append(("7-tenor UST + 14 FX, F3 h=[1,4]", write_unit(root, "syn-F3-mix21", "F3", [pc, pfx], curve + fx15, [1, 4]), "runs"))
    out.append(("weekend as-of? asof Sunday panel ends Fri", write_unit(root, "syn-F2-eur", "F2", [pfx], ["EUR"], [1, 2, 3, 4]), "runs"))
    short = daily_panel("ecb_fx_daily", ["MXN"], "ecb", 4, n_rows=25)
    out.append(("only 25 rows of history", write_unit(root, "syn-F2-short", "F2", [short], ["MXN"], [1]), "runs"))
    peg = daily_panel("ecb_fx_daily", ["MXN", "INR"], "ecb", 6, constant="INR")
    out.append(("a constant (pegged) series", write_unit(root, "syn-F3-peg", "F3", [peg], ["MXN", "INR"], [1, 4]), "runs"))
    holes = daily_panel("ecb_fx_daily", ["KRW"], "ecb", 7, nan_every=17)
    out.append(("NaN values in the panel", write_unit(root, "syn-F2-nan", "F2", [holes], ["KRW"], [2]), "runs"))
    return out


def run(unit: pathlib.Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("MODEL_ENDPOINT", "MODEL_API_KEY")}
    out = unit / "out" / "forecast.parquet"
    t0 = time.time()
    agent = subprocess.run([sys.executable, str(REPO / "forecast_agent.py"), "--panels", str(unit),
                            "--text", str(unit / "text"), "--asof", ASOF, "--out", str(out)],
                           capture_output=True, text=True, env=env, cwd=REPO)
    res = {"exit": agent.returncode, "seconds": round(time.time() - t0, 1),
           "err": "\n".join(l for l in agent.stderr.splitlines() if not l.startswith("[text_signal]"))[-300:]}
    if agent.returncode != 0 or not out.exists():
        return res
    gates = subprocess.run([sys.executable, str(REPO / "scoring" / "scoring.py"), "score",
                            "--card", str(unit / "card.toml"), "--forecast", str(out)],
                           capture_output=True, text=True, cwd=REPO)
    try:
        payload = json.loads(gates.stdout)
        res["admissible"] = bool(payload.get("admissible"))
        res["gates"] = {g: v for g, v in payload.get("gates", {}).items() if v != "pass"}
    except json.JSONDecodeError:
        res["admissible"] = False
        res["gates"] = {"scorer": gates.stderr[-200:]}
    df = pq.read_table(out).to_pandas()
    v = df["value"].to_numpy()
    res["finite"] = bool(np.isfinite(v).all())
    g = df.groupby(["asset", "horizon"])["value"]
    res["min_sd"] = float(g.std().min())
    res["cells"] = int(g.ngroups)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    root = pathlib.Path(tempfile.mkdtemp(prefix="robust-cards-"))
    bad = 0
    for label, unit, _ in cases(root):
        r = run(unit)
        ok = r["exit"] == 0 and r.get("admissible") and r.get("finite") and r.get("min_sd", 0) > 0
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label:46s} exit={r['exit']} {r['seconds']:5.1f}s "
              + (f"cells={r['cells']} min_sd={r['min_sd']:.3g} gates_failed={r['gates'] or '-'}" if r['exit'] == 0 and 'cells' in r
                 else f"\n      {r['err']}"))
    print(f"\n{bad} failing case(s)" + (f"; units kept under {root}" if a.keep else ""))
    if not a.keep:
        shutil.rmtree(root, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
