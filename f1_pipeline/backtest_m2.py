"""
Rolling-origin backtest: M2 against the team's walks, on any card.

## Executive summary (read this first)

For one card, this stands at many past dates ("origins") inside the card's own panel history and,
at each one, forecasts the card's cells three ways:

- **M2** (`m2_unit.fit_m2(cell, origin=...)`, then `draw_joint`) -- fitted only on rows whose outcome
  was already observed at that origin, exactly like notebook 03's backtest.
- **random walk** (`random_walk` below) -- each horizon drawn on its own from the walk's fit. A
  research baseline: production dropped it on 2026-09-27.
- **cumulative walk** (`forecast_models._cumulative_walk_model`) -- the production model (every card
  since 2026-09-27), given the panel truncated at the origin; identical to the random walk on a
  single-horizon card.

Each forecast is scored with the Track 2 composite (0.5 CRPS + 0.3 variogram + 0.2 tail pinball, the
weights every F1/F2/F4 card declares) against what the panel shows the target did next. Only panel
history before the card's as-of is used -- no sealed outcome, no realized vector.

    from backtest_m2 import backtest_card
    rows = backtest_card("units/t2-F2-whatever-it-takes-2012")      # one row per origin x model

`run(unit_dirs, workers=8)` runs many cards in parallel and returns one DataFrame.

The walks get **no text adjustment** (shift 0, widen 1, skew 0): this compares the numeric models.
"""

from __future__ import annotations

import pathlib
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m2_unit as m2  # noqa: E402
import forecast_agent as fa  # noqa: E402
import forecast_models as fm  # noqa: E402
from qfbench2_common.scoring import crps  # noqa: E402
from qfbench2_track_forecasting.tail import TAIL_METRICS  # noqa: E402

WEIGHTS = (0.5, 0.3, 0.2)                 # marginal, joint, tail -- declared by every F1/F2/F4 card
TAIL_LEVELS = (0.01, 0.05, 0.95, 0.99)
N_DRAWS = 1000
STRIDE = {"daily": 21, "monthly": 2}      # notebook 03's origin spacing
MIN_TRAIN = {"daily": 252, "monthly": 60}  # notebook 03's floor for an origin to be used


def score_composite(samples: np.ndarray, y: np.ndarray) -> dict:
    """Notebook 03's `score_composite`: samples [n_draws, n_cells], y [n_cells]. Lower is better."""
    s, yr = np.asarray(samples, float), np.asarray(y, float)
    w_m, w_j, w_t = WEIGHTS
    marginal = crps.crps_marginal(s, yr)
    tail = TAIL_METRICS["pinball"](s, yr, TAIL_LEVELS)
    if s.shape[1] == 1:                   # variogram is 0 on one cell: redistribute its weight
        joint, composite = 0.0, (w_m * marginal + w_t * tail) / (w_m + w_t)
    else:
        joint = crps.variogram_score(s, yr, p=0.5)
        composite = w_m * marginal + w_j * joint + w_t * tail
    q05, q95 = np.quantile(s, [0.05, 0.95], axis=0)
    return dict(marginal=float(marginal), joint=float(joint), tail=float(tail), composite=float(composite),
                cov90=float(np.mean((yr >= q05) & (yr <= q95))))


def _only_assets(panels: dict, assets: list[str]) -> dict:
    """Panels cut down to the target assets' rows -- the walks read nothing else, and it is much faster."""
    out = {}
    for stem, t in panels.items():
        acol = next((c for c in ("asset", "asset_id") if c in t.column_names), None)
        if acol is None:
            continue
        keep = pc.is_in(pc.cast(t[acol], pa.string()), value_set=pa.array(assets, pa.string()))
        small = t.filter(keep)
        if small.num_rows:
            out[stem] = small
    return out


def _origins(cells: list[m2.CellData], freq: str) -> list[pd.Timestamp]:
    """Notebook 03's `backtest_origins`: every STRIDE-th usable origin of the longest-horizon cell at
    which every cell already has MIN_TRAIN resolved training rows."""
    info = []
    for c in cells:
        df = c.frame
        usable = (df["split"].to_numpy() == "train") & df[c.features].notna().all(axis=1).to_numpy()
        info.append((c, df["origin_date"].to_numpy(), df["target_date"].to_numpy(), usable))
    longest = max(info, key=lambda x: x[0].steps)
    cand = longest[1][longest[3]]
    keep = [o for o in cand
            if all(int((u & (td <= o)).sum()) >= MIN_TRAIN[freq] for _, _, td, u in info)]
    return [pd.Timestamp(o) for o in keep[::STRIDE[freq]]]


def random_walk(req: "fm._Request") -> np.ndarray:
    """Each horizon drawn on its own from the walk's fit (horizons independent). Research only:
    production has used the cumulative walk for every card since 2026-09-27. On a single-horizon
    card the two are the same numbers."""
    fit = fm._fit_walk(req)
    rng, rng_t = np.random.default_rng(req.seed), np.random.default_rng([req.seed, 7])
    out = np.empty((req.n_draws, len(req.assets), len(req.horizons)))
    for hi in range(len(req.horizons)):
        out[:, :, hi] = fit.centre + fit.shock(rng, req.n_draws, rng_t) * fit.scale * np.sqrt(fit.steps[:, hi])
    return out


def backtest_card(unit_dir: str | pathlib.Path, n_draws: int = N_DRAWS) -> list[dict]:
    """One row per (origin, model) with the composite and its parts. A card M2 cannot fit returns a
    single row with `status` saying why."""
    unit_dir = pathlib.Path(unit_dir)
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    t = card["targets"]
    uid, fam = card["task"]["id"], card["metadata"]["category"]
    assets, horizons = [str(a) for a in t["asset_ids"]], [int(h) for h in t["horizons"]]
    asof, ttype = card["provenance"]["data_cutoff"], t["target_type"]
    base = dict(unit=uid, family=fam, target_type=ttype, n_assets=len(assets), horizons=str(horizons))

    panels = fa._read_panels(unit_dir)
    if any("transfer" in stem for stem in panels):
        return [dict(base, status="skipped: transfer card (target history has a deliberate gap)")]
    try:
        unit = m2.unit_from_panels(panels, assets, horizons, asof, ttype, unit_id=uid)
        cells = m2.build_features(unit)
    except Exception as exc:
        return [dict(base, status=f"skipped: M2 cannot load this card ({exc})")]
    base.update(panel=unit.panel, freq=unit.freq)
    small = _only_assets(panels, assets)
    neutral = {a: {"shift": 0.0, "widen": 1.0, "skew": 0.0} for a in assets}
    # On a monthly panel the walk counts its horizon in months from the last observation
    # (forecast_models._panel_steps). At the card's real as-of that is exactly the card's step count
    # -- the same as M2's and the official baseline's (8/9 and 2 on the four monthly cards). A backtest
    # origin has no publication delay, so re-deriving months there would miscount; the walk is given
    # the card's own step count instead, the one production uses and the truth here is measured at.
    card_steps = np.array([float(c.steps) for c in cells[:len(horizons)]])     # cells: asset-major
    steps_for = (lambda s, hz, a: card_steps) if unit.freq == "monthly" else fm._panel_steps
    row_of = [dict(zip(c.frame["origin_date"], range(len(c.frame)))) for c in cells]

    rows = []
    for k, o in enumerate(_origins(cells, unit.freq)):
        y = np.array([c.frame["target"].to_numpy(float)[row_of[j][o]] for j, c in enumerate(cells)])
        fits = [m2.fit_m2(c, origin=o) for c in cells]
        n_train = min(f.n_train for f in fits)
        req = fm._Request(small, assets, horizons, str(o.date()), neutral, n_draws, k, ttype, fam)
        models = {"M2": lambda: m2.draw_joint(fits, n_draws, k),
                  "random walk": lambda: random_walk(req).reshape(n_draws, -1),
                  "cumulative walk": lambda: fm._cumulative_walk_model(req).reshape(n_draws, -1)}
        for name, draw in models.items():
            try:
                saved, fm._panel_steps = fm._panel_steps, steps_for
                try:
                    samples = draw()
                finally:
                    fm._panel_steps = saved
                sc = score_composite(samples, y)
                status = "ok"
            except Exception as exc:                         # a model that cannot forecast this origin
                sc, status = {}, f"error: {type(exc).__name__}: {exc}"[:200]
            rows.append(dict(base, status=status, origin=o, n_train=n_train, model=name, **sc))
    if not rows:
        rows.append(dict(base, status="skipped: no origin with enough training rows"))
    return rows


def _safe(unit_dir: str) -> list[dict]:
    try:
        return backtest_card(unit_dir)
    except Exception as exc:
        return [dict(unit=pathlib.Path(unit_dir).name, status=f"error: {type(exc).__name__}: {exc}"[:200])]


def run(unit_dirs: list[str | pathlib.Path], workers: int = 8) -> pd.DataFrame:
    """Backtest many cards in parallel; one DataFrame, one row per (card, origin, model)."""
    dirs = [str(d) for d in unit_dirs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = [r for rows in ex.map(_safe, dirs) for r in rows]
    return pd.DataFrame(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Backtest M2 vs the walks on F1/F2/F4 cards.")
    ap.add_argument("--families", default="F1,F2,F4")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "data" / "backtest_m2_vs_walk.parquet")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dirs = [d for f in a.families.split(",") for d in sorted((REPO / "units").glob(f"t2-{f}-*"))]
    df = run(dirs, a.workers)
    df.to_parquet(a.out, index=False)
    print(f"{df.unit.nunique()} cards, {len(df):,} rows -> {a.out}")
    print(df.groupby(["family", "status"]).unit.nunique().to_string())
