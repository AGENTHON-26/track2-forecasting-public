#!/usr/bin/env python3
"""F3 stage-2 arm comparison: which prompt actually helps, above this endpoint's noise.

`tools/f3_bench.py` pins text to neutral and measures the MODEL. This measures the TEXT, and it
exists because the first attempt to do that could not tell a real effect from noise: five live
sweeps of the shipped pipeline put median-of-3 at 0.9045 against single-call's 0.9038 (+0.07%),
while individual units swung up to 22% between two runs of the SAME arm. A one-sweep-per-arm A/B
on this endpoint measures nothing.

The protocol is Nish's, from the F4 work that did resolve (NISH_TEXT_NOTES.md, "F4"):

  - **Stage-1 summaries are cached** (`TEXT_SIGNAL_CACHE_DIR`), so only the stage-2 prompt differs
    between arms and stage 1's own variance is out of the comparison entirely. The cache key
    covers the document, its prompt and the model, so a prompt change misses cleanly.
  - **N stage-2 model runs x M draw seeds per arm** (default 3 x 3). Averaging draw seeds alone is
    not enough: the model's reply is the noisier of the two.
  - **Odd/even half split reported beside the mean.** An arm that wins on the mean but only on one
    half is fitted to the cards, not to the family -- the same caveat Nish records for F4 v4.

    python tools/f3_arms.py --arms textoff current v2              # the comparison
    python tools/f3_arms.py --arms current --runs 1 --seeds 1      # a quick smoke check
    python tools/f3_arms.py --list                                 # what arms exist

Arms are (stage-2 prompt, thinking) pairs applied by patching `text_signal` in-process; nothing
here edits a file. `textoff` makes no model call at all and is the floor every arm is judged
against -- an arm that does not beat it is worse than deleting the text half for this family.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import statistics
import sys
import tomllib

import numpy as np
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import forecast_agent as fa  # noqa: E402
import forecast_models as fm  # noqa: E402
import text_signal as ts  # noqa: E402
from qfbench2_common.scoring import crps  # noqa: E402
from qfbench2_track_forecasting import tail as tailmod  # noqa: E402

UNITS = REPO / "units"
REALIZED = REPO / "realized_vectors"
WEIGHTS = {"marginal": 0.5, "joint": 0.3, "tail": 0.2}
TAIL_LEVELS = (0.01, 0.05, 0.95, 0.99)
NEUTRAL = {"shift": 0.0, "widen": 1.0, "skew": 0.0}
DEFAULT_CACHE = REPO / ".cache" / "stage1"


# ------------------------------------------------------------------ arms
def _arm_textoff() -> None:
    """No model call. The floor."""


def _arm_current() -> None:
    """What is shipped on dev today: F3 v1 paragraph, thinking off, zeros in the skeleton."""


def _arm_v2() -> None:
    """F3 v2 paragraph + differentiated skeleton + thinking on. All three together."""
    ts._FAMILY_FOCUS["F3"] = ts._F3_FOCUS_V2
    ts._DIFFERENTIATED_EXAMPLE_FAMILIES.add("F3")
    ts._STAGE2_THINKING_BY_FAMILY["F3"] = True


def _arm_v2_nothink() -> None:
    """v2's prompt changes without thinking -- isolates how much of v2 needs the reasoning."""
    ts._FAMILY_FOCUS["F3"] = ts._F3_FOCUS_V2
    ts._DIFFERENTIATED_EXAMPLE_FAMILIES.add("F3")


def _arm_thinkonly() -> None:
    """Thinking on, v1 prompt -- isolates thinking from the prompt rewrite."""
    ts._STAGE2_THINKING_BY_FAMILY["F3"] = True


def _arm_exampleonly() -> None:
    """Only the skeleton placeholders change -- isolates the anchoring effect Nish measured."""
    ts._DIFFERENTIATED_EXAMPLE_FAMILIES.add("F3")


ARMS = {
    "textoff": _arm_textoff,
    "current": _arm_current,
    "v2": _arm_v2,
    "v2-nothink": _arm_v2_nothink,
    "think-only": _arm_thinkonly,
    "example-only": _arm_exampleonly,
}


@contextlib.contextmanager
def _applied(arm: str):
    """Apply an arm's patches, then put `text_signal` back exactly as it was."""
    saved = (
        dict(ts._FAMILY_FOCUS),
        set(ts._DIFFERENTIATED_EXAMPLE_FAMILIES),
        dict(ts._STAGE2_THINKING_BY_FAMILY),
    )
    try:
        ARMS[arm]()
        yield
    finally:
        ts._FAMILY_FOCUS.clear(), ts._FAMILY_FOCUS.update(saved[0])
        ts._DIFFERENTIATED_EXAMPLE_FAMILIES.clear()
        ts._DIFFERENTIATED_EXAMPLE_FAMILIES.update(saved[1])
        ts._STAGE2_THINKING_BY_FAMILY.clear()
        ts._STAGE2_THINKING_BY_FAMILY.update(saved[2])


# ------------------------------------------------------------------ units
def load_unit(unit_id: str) -> dict | None:
    unit_dir = UNITS / unit_id
    realized_path = REALIZED / f"{unit_id}.parquet"
    if not (unit_dir / "card.toml").is_file() or not realized_path.is_file():
        return None
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    tgt = card["targets"]
    assets = list(tgt["asset_ids"])
    horizons = [int(h) for h in tgt["horizons"]]
    rv = pq.read_table(realized_path).to_pydict()
    by_cell = {(str(a), int(h)): float(v)
               for a, h, v in zip(rv["asset"], rv["horizon"], rv["value"])}
    try:
        y = np.array([by_cell[(a, h)] for a in assets for h in horizons], dtype=float)
    except KeyError:
        return None
    # Realized CHANGE per asset at the shortest horizon -- the quantity a drift_sd is a bet on.
    # Kept on the unit so the direction metric below costs nothing per run.
    panels = fa._read_panels(unit_dir)
    asof = card["provenance"]["data_cutoff"]
    h0 = min(horizons)
    change = {}
    for a in assets:
        if (a, h0) not in by_cell:
            continue
        with contextlib.suppress(Exception):
            change[a] = by_cell[(a, h0)] - float(fm._series(panels, a, asof).iloc[-1])
    return {
        "change": change,
        "unit_id": unit_id, "unit_dir": unit_dir, "assets": assets, "horizons": horizons,
        "asof": card["provenance"]["data_cutoff"], "target_type": tgt.get("target_type"),
        "family": card.get("metadata", {}).get("category"), "y": y,
        "panels": panels,
    }


def composite(unit: dict, adjustments: dict, n_draws: int, seed: int) -> float:
    """The card's own weighted composite, single-cell rule included (scoring.py:553-603)."""
    assets, horizons = unit["assets"], unit["horizons"]
    samples = fm.build_draws(
        unit["panels"], assets, horizons, unit["asof"], adjustments, n_draws, seed,
        target_type=unit["target_type"], family=unit["family"],
    )
    flat = samples.reshape(n_draws, len(assets) * len(horizons))
    y = unit["y"]
    m = crps.crps_marginal(flat, y)
    j = crps.variogram_score(flat, y, p=0.5)
    t = tailmod.tail_pinball(flat, y, TAIL_LEVELS)
    w_m, w_j, w_t = WEIGHTS["marginal"], WEIGHTS["joint"], WEIGHTS["tail"]
    if len(y) == 1:
        live = w_m + w_t
        w_m, w_j, w_t = w_m / live, 0.0, w_t / live
    return w_m * m + w_j * j + w_t * t


def adjustments_for(unit: dict, arm: str) -> tuple[dict, dict]:
    """One live stage-2 call under this arm. Returns (adjustments, per-asset drift_sd)."""
    if arm == "textoff":
        return {a: dict(NEUTRAL) for a in unit["assets"]}, {}
    text_dir = unit["unit_dir"] / "text"
    adj = ts.read_text_signal(text_dir, unit["assets"])
    # Recover drift_sd from shift for the zero-rate line: shift = drift_sd * sigma, and sigma is
    # what load_context already computed. One call per unit, not per asset.
    sigma: dict = {}
    with contextlib.suppress(Exception):
        sigma = ts.load_context(text_dir, unit["assets"]).get("sigma", {})
    drifts = {}
    for a in unit["assets"]:
        sig = float(sigma.get(a, 0.0) or 0.0)
        drifts[a] = (adj.get(a, {}).get("shift", 0.0) / sig) if sig else 0.0
    return adj, drifts


def gap_direction(unit: dict, drifts: dict) -> tuple[int, int]:
    """(right, committed) over every asset PAIR, for the gaps this reply actually took a side on.

    This is the metric that decides whether an F3 prompt learned anything, and it is not the
    composite: a prompt can improve the mean by betting bigger on a handful of high-variance
    cards while its directions stay a coin flip -- measured, rev 1 of the v2 paragraph did
    exactly that (52.4%). The variogram scores |asset_i - asset_j|, so the sign of the GAP is
    what a joint answer is really claiming. Ties (an uncommitted gap) are not counted either way.
    """
    change, assets = unit["change"], unit["assets"]
    right = committed = 0
    for i, ai in enumerate(assets):
        for aj in assets[i + 1:]:
            if ai not in change or aj not in change:
                continue
            pred = drifts.get(ai, 0.0) - drifts.get(aj, 0.0)
            if abs(pred) < 1e-9:
                continue
            committed += 1
            right += (pred > 0) == ((change[ai] - change[aj]) > 0)
    return right, committed


# ------------------------------------------------------------------ main
def run_arm(arm: str, units: list[dict], runs: int, seeds: int, n_draws: int) -> dict:
    per_unit: dict[str, list[float]] = {u["unit_id"]: [] for u in units}
    zero_cells = total_cells = 0
    dir_right = dir_committed = 0
    n_runs = 1 if arm == "textoff" else runs        # no model call -> no model variance
    for r in range(n_runs):
        for u in units:
            adj, drifts = adjustments_for(u, arm)
            for a, d in drifts.items():
                total_cells += 1
                zero_cells += (abs(d) < 1e-9)
            r, n = gap_direction(u, drifts)
            dir_right += r
            dir_committed += n
            for s in range(seeds):
                per_unit[u["unit_id"]].append(composite(u, adj, n_draws, s))
        print(f"  [{arm}] model run {r + 1}/{n_runs} done", file=sys.stderr)
    means = {uid: float(np.mean(v)) for uid, v in per_unit.items()}
    ordered = sorted(means)
    return {
        "mean": float(np.mean([means[u] for u in ordered])),
        "even": float(np.mean([means[u] for i, u in enumerate(ordered) if i % 2 == 0])),
        "odd": float(np.mean([means[u] for i, u in enumerate(ordered) if i % 2 == 1])),
        "per_unit": means,
        "zero_rate": (zero_cells / total_cells) if total_cells else float("nan"),
        "gap_dir": (dir_right / dir_committed) if dir_committed else float("nan"),
        "gap_n": dir_committed,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="f3_arms", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=["textoff", "current", "v2"])
    p.add_argument("--runs", type=int, default=3, help="live stage-2 calls per arm (default 3)")
    p.add_argument("--seeds", type=int, default=3, help="draw seeds per model run (default 3)")
    p.add_argument("--draws", type=int, default=500, help="draws per forecast (default 500)")
    p.add_argument("--samples", type=int, default=1,
                   help="stage-2 samples per model run (default 1). The shipped default is 3 "
                        "(median-of-3), but that measured as a no-op (PUN_TEXT_NOTES.md) and "
                        "averaging 3 MODEL RUNS already does the same variance reduction for "
                        "a third of the requests.")
    p.add_argument("--cache", type=pathlib.Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=pathlib.Path, default=None, help="write results JSON here")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)

    if a.list:
        for name, fn in ARMS.items():
            print(f"{name:14s} {(fn.__doc__ or '').splitlines()[0]}")
        return 0

    unknown = [x for x in a.arms if x not in ARMS]
    if unknown:
        print(f"unknown arm(s): {unknown}; --list shows them all", file=sys.stderr)
        return 1

    # Stage-1 summaries cached across arms: this is what makes the comparison about stage 2 only.
    ts._STAGE2_SAMPLES = a.samples
    a.cache.mkdir(parents=True, exist_ok=True)
    os.environ["TEXT_SIGNAL_CACHE_DIR"] = str(a.cache)

    units = [u for u in (load_unit(p.parent.name) for p in sorted(UNITS.glob("t2-F3-*/card.toml")))
             if u is not None]
    if not units:
        print("no F3 units with realized vectors", file=sys.stderr)
        return 1
    print(f"{len(units)} F3 units | {a.runs} model runs x {a.seeds} seeds | {a.draws} draws "
          f"| {a.samples} stage-2 sample(s) | {ts._WORKERS} workers @ {ts._RATE_LIMIT_RPM} rpm "
          f"| cache {a.cache}")

    results = {}
    for arm in a.arms:
        print(f"\n=== arm: {arm} ===", file=sys.stderr)
        with _applied(arm):
            results[arm] = run_arm(arm, units, a.runs, a.seeds, a.draws)

    floor = results.get("textoff", {}).get("per_unit")
    print(f"\n{'arm':14s} {'mean':>9s} {'even':>9s} {'odd':>9s} {'zero%':>7s} {'gapdir':>8s}"
          f"  vs textoff")
    for arm, r in results.items():
        if floor and arm != "textoff":
            wins = sum(r["per_unit"][u] < floor[u] for u in floor)
            cmp = f"  {wins}/{len(floor) - wins} win/loss"
        else:
            cmp = "  (floor)" if arm == "textoff" else ""
        z = "" if r["zero_rate"] != r["zero_rate"] else f"{r['zero_rate'] * 100:6.1f}%"
        g = "" if r["gap_dir"] != r["gap_dir"] else f"{r['gap_dir'] * 100:6.1f}% "
        print(f"{arm:14s} {r['mean']:9.4f} {r['even']:9.4f} {r['odd']:9.4f} {z:>7s} {g:>8s}{cmp}")
    if any(r.get("gap_n") for r in results.values()):
        print("\ngapdir = share of COMMITTED pairwise gaps whose sign matched the realized move. "
              "50% is a coin flip;\nan arm that improves the mean without improving this is "
              "betting bigger, not better.")

    if a.out:
        a.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
