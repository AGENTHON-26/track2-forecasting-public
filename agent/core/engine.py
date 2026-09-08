"""Joint block-bootstrap innovations engine -- the fat-tailed, historically-grounded
replacement for `qfbench2_track_forecasting.cli._draw`'s Gaussian random walk.

See docs/SOLVER-PLAYBOOK.md section 4. Two things come from the same mechanism, because each
bootstrapped "block" is a real historical day's joint change across every target asset at
once, not a synthetic single-asset shock:

  * fat tails, because real market history already contains the rare extreme days a Gaussian
    fit to the same variance would almost never produce;
  * cross-asset correlation, because resampling the SAME date for every asset carries along
    whatever they actually did together that day -- no separate correlation-matrix/Cholesky
    step is needed, unlike `cli.py`'s `_draw`.

A regime blend (recent window vs. full history, `_RECENT_WEIGHT` of the time) mixes current
conditions with long-run tail risk, per the playbook's step 2. Cross-horizon correlation for
multi-horizon cards comes for free too: one continuous path is built out to the longest
horizon and every requested horizon reads off a cumulative sum of a shared prefix, so the
63-day and 126-day outcomes in the same draw are never independent of each other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qfbench2_track_forecasting.cli import _diff_without_gaps, _series

#: Below this many overlapping-history rows, there are too few distinct blocks to bootstrap
#: meaningfully. Same floor `qfbench2_track_forecasting.cli._draw` already enforces for its own
#: covariance estimate.
MIN_HISTORY_ROWS = 30
DEFAULT_BLOCK_SIZE = 5
DEFAULT_RECENT_WINDOW = 252
#: Fraction of blocks drawn from the recent window rather than the full history.
DEFAULT_RECENT_WEIGHT = 0.6


def _sample_block(
    diffs: np.ndarray,
    recent_diffs: np.ndarray,
    block_size: int,
    recent_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One block of up to `block_size` consecutive historical joint-change rows.

    Block size is clamped to what the chosen regime actually has, so a short recent window
    degrades to shorter blocks rather than raising -- e.g. a proxy asset with 40 days of
    recent history still bootstraps, just with smaller blocks from that window.
    """
    use_recent = len(recent_diffs) >= 2 and rng.random() < recent_weight
    pool = recent_diffs if use_recent else diffs
    size = min(block_size, len(pool))
    start = int(rng.integers(0, len(pool) - size + 1))
    return pool[start : start + size]


def draw(
    panels: dict[str, pd.DataFrame],
    assets: list[str],
    horizons: list[int],
    asof: str,
    n_draws: int,
    seed: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    recent_weight: float = DEFAULT_RECENT_WEIGHT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Joint block-bootstrap draws.

    Same return shape as `qfbench2_track_forecasting.cli._draw`:
    `(samples[n_draws, n_assets, n_horizons], meta)` with `meta["last"]` / `meta["daily_sd"]`
    per asset -- a drop-in replacement everywhere that function's output is consumed (the
    model prompt and `apply_adjustment`'s clamps in `baselines/reasoning_agent.py` read both
    keys and don't care which engine produced them).
    """
    rng = np.random.default_rng(seed)
    hist = {a: _series(panels, a, asof) for a in assets}
    diffs_df = pd.DataFrame({a: _diff_without_gaps(s) for a, s in hist.items()}).dropna()
    if len(diffs_df) < MIN_HISTORY_ROWS:
        raise SystemExit(
            f"not enough overlapping history to block-bootstrap ({len(diffs_df)} rows, "
            f"need >= {MIN_HISTORY_ROWS})"
        )

    last = np.array([hist[a].iloc[-1] for a in assets], dtype=float)
    daily_sd = diffs_df.std().to_numpy(dtype=float)
    diffs = diffs_df.to_numpy(dtype=float)
    effective_window = min(recent_window, len(diffs))
    recent_diffs = diffs[-effective_window:]

    max_h = max(horizons)
    out = np.empty((n_draws, len(assets), len(horizons)), dtype=float)
    for d in range(n_draws):
        rows: list[np.ndarray] = []
        length = 0
        while length < max_h:
            block = _sample_block(diffs, recent_diffs, block_size, recent_weight, rng)
            rows.append(block)
            length += len(block)
        path = np.concatenate(rows, axis=0)[:max_h]  # [max_h, n_assets]
        cum = np.cumsum(path, axis=0)  # cumulative joint change through day t (0-indexed)
        for hi, h in enumerate(horizons):
            out[d, :, hi] = last + cum[h - 1]

    meta = {
        "last": {a: float(last[i]) for i, a in enumerate(assets)},
        "daily_sd": {a: float(daily_sd[i]) for i, a in enumerate(assets)},
        "n_history_rows": int(len(diffs_df)),
        "block_size": block_size,
        "recent_window": effective_window,
        "recent_weight": recent_weight,
        "engine": "joint_block_bootstrap",
    }
    return out, meta
