"""Generates the comparison plots embedded in agent/README.md.

Not part of the submission pipeline -- this is documentation tooling. Run it after any change
to agent/core/engine.py that should be reflected in the README's pictures:

    python -m agent.tools.viz

Two figures, both comparing the reference Gaussian engine (`qfbench2_track_forecasting.cli
._draw`) against ours (`agent.core.engine.draw`) on the SAME real unit, SAME seed, SAME draw count --
the only thing that differs is the engine, so any visual difference is attributable to that.

1. assets/tail_comparison.png  -- one asset's marginal distribution, overlaid histograms plus
   fitted-Gaussian reference curves, with 1st/99th percentile markers. Shows fat tails directly.
2. assets/joint_comparison.png -- two assets' joint scatter, three panels: real history, the
   Gaussian engine's draws, and ours. Shows the "correlation for free" claim directly: does the
   cloud's SHAPE resemble history, not just its correlation coefficient.
3. assets/scenario_mixture.png -- Phase 4. NOT a real model's output (no MODEL_ENDPOINT is
   available yet) -- a hand-picked, clearly-labeled illustrative scenario pair applied via the
   real agent.core.adjust.apply_scenarios to real Phase-3 bootstrap draws, showing MECHANISM (how a
   weighted scenario mixture reshapes a distribution) rather than any claim about forecast
   quality.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.core.adjust import apply_scenarios
from agent.core.engine import draw as bootstrap_draw
from qfbench2_track_forecasting.cli import _diff_without_gaps, _read_panels, _series
from qfbench2_track_forecasting.cli import _draw as gaussian_draw

# agent/assets/, not agent/tools/assets/ -- this file lives one level deeper than it used to.
ASSETS_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

_SEED = 7
_N_DRAWS = 2000  # more than the submission default, so the pictures are smooth, not noisy


def _load(unit_id: str):
    unit_dir = REPO_ROOT / "units" / unit_id
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    t = card["targets"]
    assets, horizons = list(t["asset_ids"]), [int(h) for h in t["horizons"]]
    asof = card["provenance"]["data_cutoff"]
    panels_dir = unit_dir / "panels"
    if not panels_dir.is_dir():
        panels_dir = unit_dir
    panels = _read_panels(panels_dir)
    return panels, assets, horizons, asof


def _seed_for(unit_id: str) -> int:
    # Distinct per unit -- see agent/tools/sweep.py's docstring on why a shared fixed seed makes
    # "many units" secretly "one sequence copied many times."
    return int(hashlib.sha256(unit_id.encode()).hexdigest()[:8], 16)


def plot_tail_comparison(category: str = "T2-F4") -> pathlib.Path:
    """Pools STANDARDIZED draws across every cell in `category`, rather than showing one
    card's cells -- a single card's kurtosis reading is too noisy at a few thousand draws to
    make a fair picture out of (see agent/README.md's "why we aggregate" section); this is
    the picture that matches what agent/tools/sweep.py actually measures.
    """
    from scipy.stats import kurtosis, norm

    g_pool: list[np.ndarray] = []
    b_pool: list[np.ndarray] = []
    for card_path in sorted((REPO_ROOT / "units").glob("*/card.toml")):
        unit_id = card_path.parent.name
        card = tomllib.loads(card_path.read_text())
        if card.get("metadata", {}).get("category") != category:
            continue
        panels, assets, horizons, asof = _load(unit_id)
        seed = _seed_for(unit_id)
        g_samples, _ = gaussian_draw(panels, assets, horizons, asof, _N_DRAWS, seed)
        b_samples, _ = bootstrap_draw(panels, assets, horizons, asof, _N_DRAWS, seed)
        for ai in range(len(assets)):
            for hi in range(len(horizons)):
                g_vals, b_vals = g_samples[:, ai, hi], b_samples[:, ai, hi]
                g_pool.append((g_vals - g_vals.mean()) / g_vals.std())
                b_pool.append((b_vals - b_vals.mean()) / b_vals.std())

    g_all = np.concatenate(g_pool)
    b_all = np.concatenate(b_pool)

    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi_ = -6, 6
    bins = np.linspace(lo, hi_, 80)
    ax.hist(g_all, bins=bins, density=True, alpha=0.45, color="#4C72B0",
            label=f"Gaussian random walk (reference)  excess kurtosis={kurtosis(g_all):+.2f}")
    ax.hist(b_all, bins=bins, density=True, alpha=0.45, color="#DD8452",
            label=f"Joint block bootstrap (ours)  excess kurtosis={kurtosis(b_all):+.2f}")
    x = np.linspace(lo, hi_, 400)
    ax.plot(x, norm.pdf(x), color="black", lw=1.5, ls="--", label="standard normal (for reference)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_title(
        f"Every {category} cell's draws, standardized to mean 0 / std 1, then pooled "
        f"(n_cells={len(g_pool)})\nlog scale on the y-axis, so the tails aren't invisible"
    )
    ax.set_xlabel("standardized value (in units of that cell's own std)")
    ax.set_ylabel("density (log scale)")
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    out = ASSETS_DIR / "tail_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_joint_comparison(unit_id: str, asset_x: str | None = None, asset_y: str | None = None) -> pathlib.Path:
    """Real history's panel uses OVERLAPPING h-day rolling sums of the daily changes, not raw
    daily changes -- comparing raw 1-day history against h-day-ahead draws would put different
    time-scales side by side and make any shape difference meaningless. All three panels here
    represent the same thing: "the h-day change from some starting point."
    """
    panels, assets, horizons, asof = _load(unit_id)
    ax_name, ay_name = (asset_x or assets[0]), (asset_y or assets[1])
    h = min(horizons)  # the shortest horizon -- least diluted by chaining many blocks together

    hist = {a: _series(panels, a, asof) for a in (ax_name, ay_name)}
    import pandas as pd

    diffs = pd.DataFrame({a: _diff_without_gaps(s) for a, s in hist.items()}).dropna()
    h_day_hist = diffs.rolling(h).sum().dropna()  # overlapping h-day cumulative changes

    g_samples, _ = gaussian_draw(panels, assets, horizons, asof, _N_DRAWS, _SEED)
    b_samples, _ = bootstrap_draw(panels, assets, horizons, asof, _N_DRAWS, _SEED)
    axi, ayi, hi = assets.index(ax_name), assets.index(ay_name), horizons.index(h)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
    panels_to_plot = [
        (h_day_hist[ax_name].to_numpy(), h_day_hist[ay_name].to_numpy(),
         f"Real history\n({h}-day overlapping changes, n={len(h_day_hist)})", "#55A868"),
        (g_samples[:, axi, hi] - g_samples[:, axi, hi].mean(),
         g_samples[:, ayi, hi] - g_samples[:, ayi, hi].mean(),
         f"Gaussian engine's {h}-day draws\n(centered)", "#4C72B0"),
        (b_samples[:, axi, hi] - b_samples[:, axi, hi].mean(),
         b_samples[:, ayi, hi] - b_samples[:, ayi, hi].mean(),
         f"Block-bootstrap {h}-day draws\n(centered)", "#DD8452"),
    ]
    lims = max(
        np.abs(np.concatenate([p[0] for p in panels_to_plot])).max(),
        np.abs(np.concatenate([p[1] for p in panels_to_plot])).max(),
    )
    for ax, (x, y, title, color) in zip(axes, panels_to_plot):
        ax.scatter(x, y, s=6, alpha=0.35, color=color)
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{title}\ncorr={corr:+.2f}")
        ax.set_xlabel(f"{ax_name} change")
        ax.set_ylabel(f"{ay_name} change")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)

    fig.suptitle(
        f"{unit_id}: {ax_name} vs {ay_name} joint behavior, all three panels showing the SAME "
        f"{h}-business-day change -- does the cloud's SHAPE match history, not just its "
        "correlation number?"
    )
    fig.tight_layout()
    out = ASSETS_DIR / "joint_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_scenario_mixture(unit_id: str, asset: str | None = None) -> pathlib.Path:
    """NOT a real model's output -- there is still no MODEL_ENDPOINT available to call. This is
    a hand-picked, clearly-labeled illustrative scenario pair (see SCENARIOS below), applied via
    the real `agent.core.adjust.apply_scenarios` to real Phase-3 bootstrap draws, to show the
    MECHANISM -- how a weighted mixture reshapes a distribution -- not any claim about what a
    real model would say or how good the resulting forecast is.
    """
    panels, assets, horizons, asof = _load(unit_id)
    a = asset or assets[0]
    h = max(horizons)

    b_samples, b_meta = bootstrap_draw(panels, assets, horizons, asof, _N_DRAWS, _SEED)
    last = {x: float(b_meta["last"][x]) for x in assets}
    sd_h = {x: float(b_meta["daily_sd"][x]) * (h ** 0.5) for x in assets}

    # Illustrative only: a calm-continuation scenario (70%) vs. a sharp-shock scenario (30%),
    # both applied to every asset in the card for consistency with the shared-scenario design.
    scenarios = {
        "scenarios": [
            {"weight": 0.7, "basis": "inferred", "because": "illustrative: calm continuation",
             "assets": {x: {"drift_bp": 0.0, "vol_scale": 1.0} for x in assets}},
            {"weight": 0.3, "basis": "inferred", "because": "illustrative: sharp shock",
             "assets": {x: {"drift_bp": -250.0, "vol_scale": 1.8} for x in assets}},
        ]
    }
    mixed, report, _ = apply_scenarios(b_samples, assets, last, sd_h, scenarios, seed=_SEED + 1)

    ai, hi = assets.index(a), horizons.index(h)
    before, after = b_samples[:, ai, hi], mixed[:, ai, hi]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(before.min(), after.min()), max(before.max(), after.max()), 60)
    ax.hist(before, bins=bins, density=True, alpha=0.45, color="#55A868",
            label="Phase 3 alone (block bootstrap, no scenario)")
    ax.hist(after, bins=bins, density=True, alpha=0.45, color="#C44E52",
            label="+ illustrative 70/30 calm/shock scenario mixture")
    r0, r1 = report["scenarios"][0]["weight_realized"], report["scenarios"][1]["weight_realized"]
    ax.set_title(
        f"{unit_id}: {a} at {h} business days -- ILLUSTRATIVE scenario input, not real model output\n"
        f"realized split: {r0:.0%} calm / {r1:.0%} shock (requested 70% / 30%)"
    )
    ax.set_xlabel(f"{a} level")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = ASSETS_DIR / "scenario_mixture.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    p1 = plot_tail_comparison("T2-F4")
    print(f"wrote {p1}")
    p2 = plot_joint_comparison("t2-F3-conundrum-joint-2005", "UST_2Y", "UST_10Y")
    print(f"wrote {p2}")
    p3 = plot_scenario_mixture("t2-F4-aud-gfc-2008")
    print(f"wrote {p3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
