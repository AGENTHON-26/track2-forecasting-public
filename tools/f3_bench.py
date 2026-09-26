#!/usr/bin/env python3
"""Offline F3 bench: score `build_draws()` against realized vectors, fast and correctly.

`run_eval.py` is the wrong instrument for F3 model work, for two reasons:

  1. It spawns `forecast_agent.py` as a subprocess per unit, so every run makes live House-model
     calls -- ~40 minutes and real API quota to test changes that are pure numpy and never touch
     the text path at all.
  2. It reports the RAW composite. The real scorer divides each component by a reference
     forecast's score on that component first (`qfbench2_track_forecasting/scoring.py:482-484`).
     Raw, the variogram is an unnormalized SUM over cell pairs (~2.2) while the marginal is a MEAN
     over cells (~0.26), so a raw 0.5/0.3/0.2 composite is ~70% variogram by accident of scale --
     and it ranks variants backwards. Measured: a bootstrap sampler has the best raw composite on
     this set and is worse than the parametric path once normalized.

This calls `build_draws()` in-process with text pinned to neutral, so a run is seconds and free,
and reports normalized numbers against a frozen baseline.

    python tools/f3_bench.py --freeze                 # write the baseline (do this once)
    python tools/f3_bench.py                          # score current code against it
    python tools/f3_bench.py --family T2-F2 T2-F4     # the cross-family regression gate
    python tools/f3_bench.py --draws 500              # confirm the ranking holds at what ships

Normalization: with no `ref_scale.json` anywhere in this tree, the baseline frozen by `--freeze`
stands in for the organizer's reference. `normalized = 0.5*(m/m0) + 0.3*(j/j0) + 0.2*(t/t0)`,
averaged across units -- the same operation `scoring.py` performs, against a locally pinned
reference instead of the official one. Absolute values therefore mean nothing; only the ratio
between two runs of this script does.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib

import numpy as np
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import forecast_agent as fa  # noqa: E402
from qfbench2_common.scoring import crps  # noqa: E402
from qfbench2_track_forecasting import tail as tailmod  # noqa: E402

UNITS = REPO / "units"
REALIZED = REPO / "realized_vectors"
BASELINE = REPO / "eval_reports" / "f3_baseline_neutral.json"

#: Card default (`scoring.py:539`); every shipped card uses exactly this.
WEIGHTS = {"marginal": 0.5, "joint": 0.3, "tail": 0.2}
TAIL_LEVELS = (0.01, 0.05, 0.95, 0.99)

#: Monte-Carlo error on the variogram at ~500 draws is the same order as the effects being
#: measured here, so every number is averaged over this many seeds before any ratio is formed.
SEEDS = (0, 1, 2, 3, 4)

NEUTRAL = {"shift": 0.0, "widen": 1.0, "skew": 0.0}


def load_unit(unit_id: str) -> dict | None:
    """Card fields + realized vector, flattened to the scorer's canonical cell order."""
    unit_dir = UNITS / unit_id
    realized_path = REALIZED / f"{unit_id}.parquet"
    if not (unit_dir / "card.toml").is_file() or not realized_path.is_file():
        return None
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    tgt = card["targets"]
    assets = list(tgt["asset_ids"])
    horizons = [int(h) for h in tgt["horizons"]]

    # Canonical flattening is asset-major, horizon-minor in DECLARED order
    # (qfbench2_track_forecasting/grid.py:105-106). Build a lookup rather than trusting row order.
    rv = pq.read_table(realized_path).to_pydict()
    by_cell = {(str(a), int(h)): float(v)
               for a, h, v in zip(rv["asset"], rv["horizon"], rv["value"])}
    try:
        y = np.array([by_cell[(a, h)] for a in assets for h in horizons], dtype=float)
    except KeyError as exc:                      # realized vector doesn't cover the card's grid
        print(f"  {unit_id}: realized vector missing cell {exc}; skipped", file=sys.stderr)
        return None

    return {
        "unit_id": unit_id,
        "unit_dir": unit_dir,
        "assets": assets,
        "horizons": horizons,
        "asof": card["provenance"]["data_cutoff"],
        "target_type": tgt.get("target_type"),
        "family": card.get("metadata", {}).get("category"),
        "y": y,
        "panels": fa._read_panels(unit_dir),
    }


def score_once(unit: dict, n_draws: int, seed: int) -> tuple[float, float, float]:
    """(marginal, joint, tail) for one seed. Text pinned to neutral -- this measures the model."""
    assets, horizons = unit["assets"], unit["horizons"]
    adjustments = {a: dict(NEUTRAL) for a in assets}
    samples = fa.build_draws(
        unit["panels"], assets, horizons, unit["asof"], adjustments, n_draws, seed,
        target_type=unit["target_type"], family=unit["family"],
    )
    # (n_draws, n_assets, n_horizons) -> (n_draws, n_cells), asset-major/horizon-minor.
    flat = samples.reshape(n_draws, len(assets) * len(horizons))
    y = unit["y"]
    return (
        crps.crps_marginal(flat, y),
        crps.variogram_score(flat, y, p=0.5),
        tailmod.tail_pinball(flat, y, TAIL_LEVELS),
    )


def score_unit(unit: dict, n_draws: int) -> dict[str, float]:
    """Seed-averaged components. Average BEFORE forming ratios, never after."""
    runs = np.array([score_once(unit, n_draws, s) for s in SEEDS], dtype=float)
    m, j, t = runs.mean(axis=0)
    return {"marginal": float(m), "joint": float(j), "tail": float(t)}


def normalized(comp: dict[str, float], base: dict[str, float], n_cells: int) -> float:
    """Component ratios against the frozen baseline, then the card's weights.

    Mirrors `scoring.py:453-490` including the single-cell rule (`scoring.py:553-603`): with one
    cell the variogram is structurally zero, so its weight is redistributed over the two terms
    that exist. No F3 unit is single-cell; the F2/F4 regression sweep needs this.
    """
    w_m, w_j, w_t = WEIGHTS["marginal"], WEIGHTS["joint"], WEIGHTS["tail"]
    if n_cells == 1:
        live = w_m + w_t
        w_m, w_j, w_t = w_m / live, 0.0, w_t / live

    total = 0.0
    for key, w in (("marginal", w_m), ("joint", w_j), ("tail", w_t)):
        if w == 0.0:
            continue
        b = base[key]
        total += w * (comp[key] / b if b > 0 else 1.0)
    return total


def collect(families: list[str], n_draws: int) -> dict[str, dict]:
    out = {}
    for card_path in sorted(UNITS.glob("*/card.toml")):
        unit_id = card_path.parent.name
        card = tomllib.loads(card_path.read_text())
        if card.get("metadata", {}).get("category") not in families:
            continue
        unit = load_unit(unit_id)
        if unit is None:
            continue
        try:
            out[unit_id] = {
                "components": score_unit(unit, n_draws),
                "n_cells": len(unit["assets"]) * len(unit["horizons"]),
            }
        except Exception as exc:                 # a unit that errors is a finding, not a crash
            print(f"  {unit_id}: FAILED -- {type(exc).__name__}: {exc}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="f3_bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", nargs="+", default=["T2-F3"],
                   help="card families to score (default: T2-F3)")
    p.add_argument("--draws", type=int, default=4000,
                   help="draws per unit (default 4000; re-check at 500, what ships)")
    p.add_argument("--freeze", action="store_true",
                   help="write the current numbers as the baseline and exit")
    p.add_argument("--baseline", type=pathlib.Path, default=BASELINE)
    a = p.parse_args(argv)

    print(f"scoring families {a.family} at {a.draws} draws, {len(SEEDS)} seeds, text=neutral")
    results = collect(a.family, a.draws)
    if not results:
        print("no units matched", file=sys.stderr)
        return 1

    if a.freeze:
        a.baseline.parent.mkdir(parents=True, exist_ok=True)
        a.baseline.write_text(json.dumps(
            {"draws": a.draws, "seeds": list(SEEDS), "families": a.family,
             "units": {u: r["components"] for u, r in results.items()}}, indent=2) + "\n")
        print(f"\nfroze {len(results)} unit(s) to {a.baseline}")
        print("this is now the denominator; re-freeze only when you deliberately want a new one")
        return 0

    if not a.baseline.is_file():
        print(f"\nno baseline at {a.baseline} -- run with --freeze first", file=sys.stderr)
        return 1
    base_units = json.loads(a.baseline.read_text())["units"]

    print(f"\n{'unit':36s} {'marginal':>10s} {'joint':>10s} {'tail':>9s} {'norm':>7s}  vs base")
    rows, missing = [], []
    for unit_id, r in sorted(results.items()):
        if unit_id not in base_units:
            missing.append(unit_id)
            continue
        comp, base = r["components"], base_units[unit_id]
        norm = normalized(comp, base, r["n_cells"])
        rows.append((unit_id, comp, base, norm))
        mark = "  " if abs(norm - 1.0) < 1e-9 else ("↓ " if norm < 1.0 else "↑ ")
        print(f"{unit_id:36s} {comp['marginal']:10.4f} {comp['joint']:10.4f} "
              f"{comp['tail']:9.4f} {norm:7.4f}  {mark}{(norm - 1) * 100:+.1f}%")

    if missing:
        print(f"\nnot in baseline (ignored): {', '.join(missing)}")
    if not rows:
        print("\nnothing comparable to the baseline", file=sys.stderr)
        return 1

    norms = np.array([r[3] for r in rows])
    wins = int((norms < 1.0).sum())
    print(f"\n{'=' * 78}")
    print(f"mean normalized composite : {norms.mean():.4f}   (1.0 = baseline, lower is better)")
    print(f"wins / units              : {wins}/{len(norms)}")
    for key in ("marginal", "joint", "tail"):
        cur = np.array([r[1][key] for r in rows])
        bas = np.array([r[2][key] for r in rows])
        keep = bas > 0
        ratio = (cur[keep] / bas[keep]).mean() if keep.any() else float("nan")
        # Say how many units the ratio is over. A mean of ratios blows up on a near-zero
        # denominator: F4's joint is structurally 0 on all but 2 units, whose baselines are ~0.01,
        # so "mean joint ratio 1.58" there is two tiny fractions, not a real regression. The
        # normalized composite above is the number that decides anything.
        n_over = int(keep.sum())
        note = "" if n_over == len(rows) else f"   (over {n_over}/{len(rows)} units with a non-zero baseline)"
        print(f"  mean {key:8s} ratio     : {ratio:.4f}{note}")
    # Paired t on the per-unit deltas -- at n=20 an effect under ~3% is not resolvable, so report
    # the statistic rather than letting a mean alone decide.
    d = norms - 1.0
    if len(d) > 1 and d.std(ddof=1) > 0:
        print(f"paired t (norm vs 1.0)    : {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
