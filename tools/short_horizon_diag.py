"""
## Executive summary (read this first)

Calibration diagnostic for the short-horizon backtest: for one model variant it reports how often
the realized value falls beyond the draws' 1/5/95/99 % quantiles (should be 1/5/5/1 %), the spread
of the PIT (probability integral transform: where the outcome sits in the forecast's CDF), and the
three score components, per asset class and horizon. Uses only the panels' own history (no realized
vectors, no text).

    .venv/bin/python tools/short_horizon_diag.py --variants M2.0 --dates 100
"""
from __future__ import annotations
import argparse, sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import short_horizon_backtest as sb


def one(args):
    card, variant = args
    tables, hist, y = sb.card_inputs(sb._PANELS, card)
    s = sb.model_draws(variant, tables, card)
    rows = []
    for k, h in enumerate(card["horizons"]):
        d = s[:, k]
        grp = card["group"] + ("" if card["asof"] < "2020" else " 20+")
        q = np.quantile(d, [.01, .05, .95, .99])
        rows.append({"group": grp, "h": h, "lo1": y[k] < q[0], "lo5": y[k] < q[1],
                     "hi5": y[k] > q[2], "hi1": y[k] > q[3], "pit": float((d < y[k]).mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="M2.0")
    ap.add_argument("--dates", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261013)
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    panels = sb.load_panels(); sb._init(panels)
    cards = [c for c in sb.make_cards(panels, a.dates, a.seed) if c["kind"] == "single"]
    for v in a.variants.split(","):
        with ProcessPoolExecutor(10, initializer=sb._init, initargs=(panels,)) as ex:
            rows = [r for rr in ex.map(one, [(c, v) for c in cards], chunksize=8) for r in rr]
        df = pd.DataFrame(rows)
        print(f"== {v}: share beyond q1 / q5 / q95 / q99 (target 1/5/5/1 %), PIT sd (uniform .289)")
        for key, sub in [("ALL", df)] + list(df.groupby("group")) + [(f"h={h}", df[df.h == h]) for h in (1, 2, 3, 4)]:
            print(f"  {key:16s} {100*sub.lo1.mean():5.2f} {100*sub.lo5.mean():5.2f} {100*sub.hi5.mean():5.2f} "
                  f"{100*sub.hi1.mean():5.2f}   pit_sd {sub.pit.std():.3f}  n={len(sub)}")



# ---- variogram by pair type (joint cards). A = same asset across horizons, B = rates-rates (different
# assets), C = FX-FX, D = rates-FX, E = other (equity factors). Each type's raw variogram sum is divided by
# M0's EXPECTED joint divisor for the card, so the types add up to the card's normalised joint component.
# Also: bias = mean realised |D|^0.5 / mean forecast m_F (1.00 = no bias), per type.
_CLS = {"rates_daily": "R", "g10_fx_daily": "F", "factors_daily": "Q"}


def _ptype(card, ci, cj, cells, cls):
    (a1, h1), (a2, h2) = cells[ci], cells[cj]
    if a1 == a2:
        return "A"
    k = {cls[a1], cls[a2]}
    return {"R": "B", "F": "C", "Q": "E"}.get(next(iter(k))) if len(k) == 1 else "D"


def pair_terms(args):
    card, variants = args
    tables, hist, y = sb.card_inputs(sb._PANELS, card)
    mean, cov = sb.m0_distribution(hist, card["assets"], card["horizons"], card["target_type"])
    _, dj, _ = sb.divisors(mean, cov)
    seed = sb.zlib.crc32(card["id"].encode()) & 0x7FFFFFFF
    cells = [(a, h) for a in card["assets"] for h in card["horizons"]]
    cls = {a: next(_CLS[p] for p, t in sb._PANELS.items() if a in set(t["asset"].unique()) and p in _CLS)
           for a in card["assets"]}
    out = []
    for v in variants:
        s = sb.m0_draws(mean, cov, seed) if v == "M0" else sb.model_draws(v, tables, card)
        row = {"variant": v, "group": card["group"], "asof": card["asof"]}
        for t in "ABCDE":
            row[t + "_term"] = 0.0; row[t + "_real"] = 0.0; row[t + "_fc"] = 0.0; row[t + "_n"] = 0
        for i in range(len(cells)):
            for j in range(len(cells)):
                if i == j:
                    continue
                t = _ptype(card, i, j, cells, cls)
                m_f = np.mean(np.abs(s[:, i] - s[:, j]) ** 0.5)
                r = abs(y[i] - y[j]) ** 0.5
                row[t + "_term"] += (r - m_f) ** 2 / dj
                row[t + "_real"] += r; row[t + "_fc"] += m_f; row[t + "_n"] += 1
        out.append(row)
    return out


def pairs_main(variants, dates, seed):
    from concurrent.futures import ProcessPoolExecutor
    panels = sb.load_panels(); sb._init(panels)
    cards = [c for c in sb.make_cards(panels, dates, seed) if c["kind"] == "joint"]
    with ProcessPoolExecutor(10, initializer=sb._init, initargs=(panels,)) as ex:
        rows = [r for rr in ex.map(pair_terms, [(c, variants) for c in cards], chunksize=4) for r in rr]
    df = pd.DataFrame(rows)
    for label, sub in (("ALL", df), ("asof<2020", df[df["asof"] < "2020"]), ("asof>=2020", df[df["asof"] >= "2020"])):
        print(f"== {label}: contribution of each pair type to the NORMALISED joint component "
              f"(mean over cards; columns sum to the joint score), and bias = realised/forecast mean |D|^0.5")
        g = sub.groupby("variant")
        tab = g[[t + "_term" for t in "ABCDE"]].mean().loc[variants]
        tab.columns = list("ABCDE"); tab["total"] = tab.sum(axis=1)
        print(tab.round(4))
        bias = pd.DataFrame({t: g[t + "_real"].sum() / g[t + "_fc"].sum() for t in "ABCDE"}).loc[variants]
        print("bias (1.00 = unbiased):"); print(bias.round(3))


def increments(variants, dates, seed):
    """Per-horizon increment variance ratio on one-asset cards (h = 1..4): realised (y_h - y_{h-1})^2 over the
    forecast variance of that increment (from the draws), mean over cards; 1.00 = right size."""
    from concurrent.futures import ProcessPoolExecutor
    panels = sb.load_panels(); sb._init(panels)
    cards = [c for c in sb.make_cards(panels, dates, seed) if c["kind"] == "single"]
    with ProcessPoolExecutor(10, initializer=sb._init, initargs=(panels,)) as ex:
        rows = [r for rr in ex.map(_inc_one, [(c, variants) for c in cards], chunksize=8) for r in rr]
    df = pd.DataFrame(rows)
    for v in variants:
        print(f"== increment variance ratio, {v}")
        d = df[(df.variant == v) & (df["group"] != "factors_daily")]
        print(d.groupby(["group", "event", "k"]).ratio.agg(["mean", "size"]).unstack().round(3).to_string())


def _is_event(card, k):
    """True when step k after the as-of is a scheduled-event step (event_vol, full table) for this asset class."""
    import event_vol as ev
    a = card["assets"][0]
    pid, t = next((p, t) for p, t in sb._PANELS.items() if a in set(t["asset"].unique()))
    cls = {"rates_daily": "rates", "g10_fx_daily": "fx"}.get(pid)
    if cls is None:
        return False
    fut = t[(t["asset"] == a) & (t["date"] > card["asof"])]["date"].tolist()
    return bool(len(fut) >= k and ev.step_mult(cls, fut[k - 1], ev.FULL_TABLE) > 1.0)


def _inc_one(args):
    card, variants = args
    tables, hist, y = sb.card_inputs(sb._PANELS, card)
    out = []
    for v in variants:
        s = sb.model_draws(v, tables, card) if v != "M0" else None
        if s is None:
            mean, cov = sb.m0_distribution(hist, card["assets"], card["horizons"], card["target_type"])
            s = sb.m0_draws(mean, cov, 1)
        inc = np.diff(np.concatenate([np.zeros((s.shape[0], 1)), s], axis=1), axis=1)
        yi = np.diff(np.concatenate([[0.0], y]))
        for k in range(1, 5):
            var = inc[:, k - 1].var()
            if k == 1:
                var = (s[:, 0] - s[:, 0].mean()).var(); yi_k = (y[0] - s[:, 0].mean()) ** 2
            else:
                yi_k = (yi[k - 1] - inc[:, k - 1].mean()) ** 2
            out.append({"variant": v, "group": card["group"], "k": k, "ratio": yi_k / var if var > 0 else np.nan,
                        "event": _is_event(card, k)})
    return out


if __name__ == "__main__":
    import sys as _s
    if "--pairs" in _s.argv:
        i = _s.argv.index("--pairs")
        pairs_main(_s.argv[i + 1].split(","), 300, 20261013)
    elif "--incr" in _s.argv:
        i = _s.argv.index("--incr")
        increments(_s.argv[i + 1].split(","), 150, 20261013)
    else:
        main()
