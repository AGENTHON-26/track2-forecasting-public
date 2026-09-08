"""Local self-check diagnostics for a forecast.parquet -- no realized outcomes needed.

Everything here compares our own output against quantities derivable from the as-of panel
alone. There is no `--realized` flag and there never should be one: the point of this tool is
to catch calibration problems (collapsed marginals, independent-marginal joint draws,
Gaussian-shaped tails) before the leaderboard exists to tell us, using only information the
agent itself had access to at forecast time.

Three checks:

1. spread_ratio    -- std(draws) at each (asset, horizon) vs. the panel's own historical
                      horizon-sd (daily_sd * sqrt(horizon)). Ratio near 1.0 means "about as
                      wide as naive historical volatility"; a deliberate text-driven widen or
                      narrow should show up as a deliberate deviation, not an accidental one.
                      Flags only the extremes (near-collapsed or wildly over-dispersed) -- see
                      docs/SOLVER-PLAYBOOK.md step 5.
2. joint_corr_gap  -- for multi-asset cards, how far the draws' cross-asset correlation (at
                      each horizon) sits from the panel's own historical correlation. Large gap
                      with near-zero draw correlation is the "independent marginals" failure
                      mode the joint variogram term punishes (docs/CATEGORIES.md, F3).
3. excess_kurtosis -- sample excess kurtosis of the draws at each (asset, horizon). A Gaussian
                      scores 0 by definition; a linear vol_scale/drift adjustment cannot move
                      this number (see the scaling-preserves-kurtosis discussion). Reported per
                      cell but NOT flagged here: at 500 draws the sampling error on this
                      statistic is roughly +-0.22, comparable to any threshold worth setting, so
                      a single cell's reading is not trustworthy evidence either way. See
                      agent/tools/sweep.py, which aggregates this across many cells (the noise shrinks
                      as sqrt(n)) to get a signal actually worth flagging on.

Usage:
    python -m agent.tools.selfcheck --unit units/t2-F1-cad-boc-2017 --forecast out/.../forecast.parquet
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qfbench2_track_forecasting.cli import _diff_without_gaps, _read_panels, _series  # noqa: E402
from qfbench2_track_forecasting.grid import GridSpec, grid_from_card  # noqa: E402

#: Below this spread_ratio, treat the distribution as suspiciously tight -- the playbook's
#: "single most common point-loser". Above it, suspiciously wide (likely a bug, not a view).
_SPREAD_RATIO_LOW = 0.3
_SPREAD_RATIO_HIGH = 5.0


@dataclass
class CellReport:
    asset: str
    horizon: int
    draw_std: float
    horizon_sd: float
    spread_ratio: float
    excess_kurtosis: float
    flags: list[str] = field(default_factory=list)


@dataclass
class UnitReport:
    unit_id: str
    category: str
    cells: list[CellReport]
    joint_corr_gap: float | None  # None for single-asset cards -- no pair to check
    draws_corr_near_zero: bool | None
    flags: list[str] = field(default_factory=list)

    def all_flags(self) -> list[str]:
        out = list(self.flags)
        for c in self.cells:
            out += [f"{c.asset}@{c.horizon}bd: {f}" for f in c.flags]
        return out


def _load_forecast_matrix(
    forecast_path: pathlib.Path, spec: GridSpec
) -> np.ndarray:
    """[n_draws, n_assets, n_horizons], ordered per `spec`.

    Trusts the file's shape rather than re-deriving `build_sample_matrix`'s adversarial-input
    guarantees (exact occupancy, resource bounds) -- those already ran as gate g3 on this exact
    file before this tool sees it. Re-litigating them here would just be a second copy of that
    check, not a new one.
    """
    df = pd.read_parquet(forecast_path)
    n_draws = int(df["draw"].max()) + 1
    asset_index = {a: i for i, a in enumerate(spec.assets)}
    horizon_index = {h: i for i, h in enumerate(spec.horizons)}
    mat = np.full((n_draws, len(spec.assets), len(spec.horizons)), np.nan)
    for row in df.itertuples(index=False):
        mat[row.draw, asset_index[row.asset], horizon_index[row.horizon]] = row.value
    if np.isnan(mat).any():
        raise ValueError(f"{forecast_path} has a gap in its (draw, asset, horizon) grid")
    return mat


def _historical_stats(
    panels: dict[str, pd.DataFrame], spec: GridSpec, asof: str
) -> tuple[dict[str, float], pd.DataFrame]:
    """Per-asset daily sd, and the assets' pairwise correlation -- both from history <= asof.

    Reuses `_series`/`_diff_without_gaps` from the reference `cli.py` rather than
    reimplementing the asset-column-name and gap-handling logic those already carry.
    """
    hist = {a: _series(panels, a, asof) for a in spec.assets}
    diffs = pd.DataFrame({a: _diff_without_gaps(s) for a, s in hist.items()}).dropna()
    daily_sd = diffs.std().to_dict()
    corr = diffs.corr()
    return daily_sd, corr


def check_unit(unit_dir: pathlib.Path, forecast_path: pathlib.Path, asof: str) -> UnitReport:
    import tomllib

    card = tomllib.loads((unit_dir / "card.toml").read_text())
    unit_id = card["task"]["id"]
    category = card.get("metadata", {}).get("category", "?")
    spec = grid_from_card(card)

    panels_dir = unit_dir / "panels"
    if not panels_dir.is_dir():
        panels_dir = unit_dir
    panels = _read_panels(panels_dir)

    mat = _load_forecast_matrix(forecast_path, spec)
    daily_sd, hist_corr = _historical_stats(panels, spec, asof)

    cells: list[CellReport] = []
    for ai, a in enumerate(spec.assets):
        for hi, h in enumerate(spec.horizons):
            draws = mat[:, ai, hi]
            draw_std = float(draws.std())
            horizon_sd = float(daily_sd[a] * math.sqrt(h))
            ratio = draw_std / horizon_sd if horizon_sd > 0 else float("nan")
            excess_k = float(kurtosis(draws, fisher=True, bias=False))
            flags: list[str] = []
            if ratio < _SPREAD_RATIO_LOW:
                flags.append(
                    f"spread_ratio {ratio:.2f} < {_SPREAD_RATIO_LOW}: distribution reads as "
                    "overconfident relative to historical volatility"
                )
            elif ratio > _SPREAD_RATIO_HIGH:
                flags.append(
                    f"spread_ratio {ratio:.2f} > {_SPREAD_RATIO_HIGH}: distribution reads as "
                    "implausibly wide -- check for a bug before assuming this is a deliberate view"
                )
            # excess_kurtosis is reported, not flagged, per-cell -- see the module docstring
            # and agent/tools/sweep.py for why a single cell's reading isn't trustworthy at n=500.
            cells.append(
                CellReport(
                    asset=a, horizon=h, draw_std=draw_std, horizon_sd=horizon_sd,
                    spread_ratio=ratio, excess_kurtosis=excess_k, flags=flags,
                )
            )

    joint_gap: float | None = None
    near_zero: bool | None = None
    unit_flags: list[str] = []
    if len(spec.assets) > 1:
        # One correlation gap per horizon, then take the worst -- a card can be well-behaved
        # at one horizon and independently-drawn at another.
        gaps = []
        zero_flags = []
        for h in spec.horizons:
            hi = spec.horizons.index(h)
            draws_h = mat[:, :, hi]
            draws_corr = pd.DataFrame(draws_h, columns=spec.assets).corr()
            gap = float(np.abs(draws_corr.to_numpy() - hist_corr.to_numpy()).mean())
            gaps.append(gap)
            off_diag = draws_corr.to_numpy()[~np.eye(len(spec.assets), dtype=bool)]
            hist_off_diag = hist_corr.to_numpy()[~np.eye(len(spec.assets), dtype=bool)]
            zero_flags.append(
                bool(np.abs(off_diag).mean() < 0.1 and np.abs(hist_off_diag).mean() > 0.3)
            )
        joint_gap = max(gaps)
        near_zero = any(zero_flags)
        if near_zero:
            unit_flags.append(
                f"draws show near-zero cross-asset correlation (mean |corr|~0.1) while the "
                f"panel's own history is clearly correlated (mean |corr|~0.3+) -- looks like "
                f"independent-marginal sampling, which the joint variogram term punishes"
            )

    return UnitReport(
        unit_id=unit_id, category=category, cells=cells,
        joint_corr_gap=joint_gap, draws_corr_near_zero=near_zero, flags=unit_flags,
    )


def format_report(report: UnitReport) -> str:
    lines = [f"=== {report.unit_id} ({report.category}) ==="]
    for c in report.cells:
        lines.append(
            f"  {c.asset}@{c.horizon}bd: draw_std={c.draw_std:.6g} horizon_sd={c.horizon_sd:.6g} "
            f"spread_ratio={c.spread_ratio:.2f} excess_kurtosis={c.excess_kurtosis:+.2f}"
        )
    if report.joint_corr_gap is not None:
        lines.append(
            f"  joint_corr_gap={report.joint_corr_gap:.3f} "
            f"(draws vs. historical correlation, mean abs diff)"
        )
    flags = report.all_flags()
    if flags:
        lines.append("  FLAGS:")
        lines += [f"    - {f}" for f in flags]
    else:
        lines.append("  no flags")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", required=True, type=pathlib.Path)
    ap.add_argument("--forecast", required=True, type=pathlib.Path)
    ap.add_argument("--asof", required=True)
    a = ap.parse_args(argv)

    report = check_unit(a.unit, a.forecast, a.asof)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
