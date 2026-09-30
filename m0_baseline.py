"""
M0 -- the organizers' text-blind baseline, rebuilt from docs/M0-BASELINE.md, for LOCAL scoring only.

## Executive summary (read this first)

The leaderboard divides each part of a card's score (marginal CRPS, joint variogram, tail) by
the same part of M0's score on that card, applies the card's weights, and ranks by the
equal-weight mean over cards. The per-card M0 values are sealed, but the procedure is published
in full, so M0's forecast can be rebuilt for any card we hold. `run_eval.py` scores it next to
ours and reports the same ratio the leaderboard reports.

Checked against Development submission 01 (`eval_reports/submission-01-dev.csv`): with the
text step off, the agent's forecast is deterministic, so every unit on which the text had no
effect on the platform must reproduce exactly. 22 units did, to four decimals (e.g.
t2-F1-pause-2006 0.4365, t2-F4-hml-covid-2020 1.0064).

The agent never imports this module; it is eval tooling, like `build_realized.py`.

## The procedure (section numbers are docs/M0-BASELINE.md's)

- 3.1  history: the first `*.parquet` in the unit directory, in sorted filename order, holding the
       asset; rows at or before the as-of, sorted by date; the last 300.
- 3.2  steps: first differences for `level`; the rows themselves for `log_return`.
- 3.3  gap rule: drop a step spanning more than max(10 x median spacing, 5 days).
- 3.4  alignment: the intersection of dates present for every asset.
- 3.5  mu = mean step per asset, Sigma = numpy.cov over the same rows.
- 3.6  anchor: the last observation for `level`; 0.0 for `log_return`.
- 3.7  steps per horizon: the declared horizon, except the four monthly-panel cards the doc lists.
- 3.8  mean = anchor + s * mu; cov[i, j] = min(s_i, s_j) * Sigma; Cholesky with the doc's jitter.
- 3.9  seed = crc32(unit_id) & 0x7FFFFFFF; 500 draws; cells sorted by asset id, then horizon.

Where this can differ from the sealed M0: the doc says three released cards use the card's own
`[targets]` order instead of the sorted cell order, and it does not name them. On those, expect a
few percent on one component, not a factor (doc 3.9).
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import tomllib
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_DRAWS = 500
_WINDOW = 300

#: Doc 3.7: on these four monthly-panel cards M0 counts panel steps, not the declared horizon.
_MONTHLY_STEPS: dict[str, dict[int, int]] = {
    "t2-F1-cpi-glidepath-2023": {140: 8, 160: 9},
    "t2-F1-sahm-watch-2024": {145: 8, 165: 9},
    "t2-F4-covid-nfp-2020": {21: 2},
    "t2-F4-cpi-vintage-2022": {21: 2},
}


def _history(unit_dir: pathlib.Path, asset: str, asof: str) -> list[tuple[str, float]]:
    for path in sorted(unit_dir.glob("*.parquet")):
        table = pq.read_table(path).to_pydict()
        acol = "asset" if "asset" in table else ("asset_id" if "asset_id" in table else None)
        if acol is None:
            continue
        rows = sorted(
            (str(d)[:10], float(v))
            for d, a, v in zip(table["date"], table[acol], table["value"])
            if str(a) == asset and str(d)[:10] <= asof and v is not None
        )
        if rows:
            return rows[-_WINDOW:]
    raise KeyError(f"{asset!r} not found in any panel of {unit_dir.name}")


def _steps(rows: list[tuple[str, float]], target_type: str) -> tuple[dict[dt.date, float], float]:
    """(date -> step, last observed value). A step carries the date of the row it ends on."""
    dates = [dt.date.fromisoformat(d) for d, _ in rows]
    values = np.array([v for _, v in rows], dtype=float)
    gaps = np.array([(dates[i] - dates[i - 1]).days for i in range(1, len(dates))], dtype=float)
    hole = max(10.0 * float(np.median(gaps)), 5.0) if gaps.size else 5.0
    steps = {}
    for i in range(1, len(dates)):
        if gaps[i - 1] > hole:
            continue
        steps[dates[i]] = values[i] - values[i - 1] if target_type == "level" else values[i]
    return steps, float(values[-1])


def draws(unit_dir: pathlib.Path) -> np.ndarray:
    """M0's 500 joint draws, shaped (n_draws, n_assets, n_horizons) in the CARD's own order."""
    unit_dir = pathlib.Path(unit_dir)
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    targets = card["targets"]
    unit_id = card["task"]["id"]
    asof = card["provenance"]["data_cutoff"]
    target_type = targets.get("target_type", "level")
    assets = list(targets["asset_ids"])
    horizons = [int(h) for h in targets["horizons"]]

    by_asset = sorted(assets)
    steps, anchor = {}, {}
    for a in by_asset:
        steps[a], last = _steps(_history(unit_dir, a, asof), target_type)
        anchor[a] = last if target_type == "level" else 0.0
    common = sorted(set.intersection(*(set(steps[a]) for a in by_asset)))
    x = np.array([[steps[a][d] for d in common] for a in by_asset])
    mu = x.mean(axis=1)
    sigma = np.atleast_2d(np.cov(x))
    ai = {a: i for i, a in enumerate(by_asset)}

    step_count = _MONTHLY_STEPS.get(unit_id, {})
    cells = [(a, h) for a in by_asset for h in sorted(horizons)]
    s = {h: step_count.get(h, h) for h in horizons}
    mean = np.array([anchor[a] + s[h] * mu[ai[a]] for a, h in cells])
    cov = np.array([[min(s[h1], s[h2]) * sigma[ai[a1], ai[a2]] for a2, h2 in cells]
                    for a1, h1 in cells])
    cov[np.diag_indices_from(cov)] += 1e-10
    jittered = cov + 1e-9 * np.eye(len(cells))
    try:
        factor = np.linalg.cholesky(jittered)
    except np.linalg.LinAlgError:
        factor = np.diag(np.sqrt(np.diag(jittered)))
    rng = np.random.default_rng(zlib.crc32(unit_id.encode()) & 0x7FFFFFFF)
    samples = mean + rng.standard_normal((N_DRAWS, len(cells))) @ factor.T

    out = np.empty((N_DRAWS, len(assets), len(horizons)))
    for ci, (a, h) in enumerate(cells):
        out[:, assets.index(a), horizons.index(h)] = samples[:, ci]
    return out


def write_forecast(unit_dir: pathlib.Path, out: pathlib.Path) -> None:
    """M0's draws as a forecast.parquet plus the sidecar files, in the agent's own format."""
    unit_dir = pathlib.Path(unit_dir)
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    targets = card["targets"]
    assets = list(targets["asset_ids"])
    horizons = [int(h) for h in targets["horizons"]]
    samples = draws(unit_dir)
    d, a, h = np.meshgrid(np.arange(N_DRAWS), np.arange(len(assets)), np.arange(len(horizons)),
                          indexing="ij")
    pq.write_table(pa.table({
        "draw": pa.array(d.ravel(), pa.int32()),
        "asset": pa.array([assets[i] for i in a.ravel()], pa.string()),
        "horizon": pa.array([horizons[i] for i in h.ravel()], pa.int32()),
        "value": pa.array(samples.ravel(), pa.float64()),
    }), out)
    out.parent.joinpath("forecast_meta.json").write_text(json.dumps({
        "unit_id": card["task"]["id"], "asof": card["provenance"]["data_cutoff"],
        "representation": "samples", "asset_ids": assets, "horizons": horizons,
        "n_draws": N_DRAWS,
    }, indent=2) + "\n")
    out.parent.joinpath("forecast_rationale.md").write_text(
        "# M0 baseline\n\nText-blind joint Gaussian random walk (docs/M0-BASELINE.md).\n")


def card_weights(card: dict, cell_count: int) -> tuple[float, float, float]:
    """The scorer's live weights: the card's, renormalized on a one-cell variogram card."""
    w = card.get("scoring", {}).get("params", {}).get(
        "weights", {"marginal": 0.5, "joint": 0.3, "tail": 0.2})
    wm, wj, wt = float(w["marginal"]), float(w["joint"]), float(w["tail"])
    joint = str(card.get("scoring", {}).get("params", {}).get("joint", "variogram"))
    if cell_count == 1 and joint == "variogram":
        return wm / (wm + wt), 0.0, wt / (wm + wt)
    return wm, wj, wt


def normalized(ours: dict, m0: dict, weights: tuple[float, float, float]) -> float:
    """The leaderboard's per-card score: each component divided by M0's, weighted and summed."""
    wm, wj, wt = weights
    score = wm * ours["marginal_crps"] / m0["marginal_crps"]
    if wj:
        score += wj * ours["joint_variogram"] / m0["joint_variogram"]
    return score + wt * ours["tail_penalty"] / m0["tail_penalty"]
