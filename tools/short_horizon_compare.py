"""
## Executive summary (read this first)

Compare backtest variants from a results CSV (written by short_horizon_backtest.py) against a
reference variant with (a) event vs non-event splits (cells whose horizon path contains a scheduled
central-bank event, `evt` > 1, from event_vol.py; as-of 2015 onward only, where the calendar exists)
and (b) a week-block bootstrap of the mean paired score difference (resampling as-of weeks, since the
real Final shares one as-of week).

    .venv/bin/python tools/short_horizon_compare.py CSV --ref M2.0 --vars ev,ev_half [--since 2015-01-01 --until 2025-12-31]
"""
import argparse
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--ref", default="M2.0"); ap.add_argument("--vars", required=True)
ap.add_argument("--since", default="2015-01-01"); ap.add_argument("--until", default="2025-12-31")
ap.add_argument("--B", type=int, default=2000)
a = ap.parse_args()
df = pd.read_csv(a.csv)
df = df[(df["asof"] >= a.since) & (df["asof"] <= a.until) & (df.kind == "single")]
piv = df.pivot_table(index=["id", "h", "asof", "group"], columns="variant", values="score")
evt = df[df.variant == a.ref].set_index(["id", "h", "asof", "group"])["evt"].reindex(piv.index)
if evt.isna().all():
    evt = df.dropna(subset=["evt"]).drop_duplicates(["id", "h"]).set_index(["id", "h", "asof", "group"])["evt"].reindex(piv.index)
piv = piv.reset_index(); piv["evt"] = evt.values
piv["week"] = pd.to_datetime(piv["asof"]).dt.strftime("%G-%V")
rng = np.random.default_rng(0)
def boot(d):
    g = d.groupby("week").agg(s=("d", "sum"), n=("d", "size"))
    s, n = g.s.to_numpy(), g.n.to_numpy(); k = len(g)
    idx = rng.integers(0, k, (a.B, k))
    m = s[idx].sum(1) / n[idx].sum(1)
    return np.percentile(m, [2.5, 97.5])
for name, sub in [("ALL", piv), ("event", piv[piv.evt > 1.0001]), ("non-event", piv[piv.evt <= 1.0001])]:
    print(f"{name}: n={len(sub)}  ref {a.ref} mean {sub[a.ref].mean():.4f}  M0 {sub['M0'].mean() if 'M0' in sub else float('nan'):.4f}")
    for v in a.vars.split(","):
        s = sub.assign(d=sub[v] - sub[a.ref])
        lo, hi = boot(s)
        t = s.d.mean() / (s.d.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.d.std() > 0 else 0
        print(f"   {v:12s} diff {s.d.mean():+.4f}  t={t:+.2f}  week-block 95% [{lo:+.4f},{hi:+.4f}]  better/worse {(s.d<-1e-12).sum()}/{(s.d>1e-12).sum()}")
