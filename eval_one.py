"""
Track 2 (Pun's eval harness) — walk ONE unit through the whole local pipeline, step by step,
printing what happens at each stage. Uses forecast_agent.py's CURRENT read_text_signal /
build_draws exactly as they stand today (the neutral/baseline stubs) -- once Dew/Nish fill
theirs in, rerunning this needs zero changes here, since it just calls whatever
forecast_agent.py currently is.

The five things that happen, in order, every time:

    unit's card.toml + panels/ + text/
            |
            v
    forecast_agent.py   (subprocess -- same 4 flags the real harness uses)
            |
            v
    forecast.parquet + sidecars
            |
            v                                   realized_vectors/<unit_id>.parquet
    scoring/scoring.py  <----------------------  (built ahead of time by build_realized.py;
            |                                     independent of forecast_agent.py entirely)
            v
    gate pass/fail, and a real composite score wherever a realized vector exists

Usage:
    python eval_one.py t2-F3-conundrum-joint-2005
    python eval_one.py t2-F1-cad-boc-2017          # currently crashes -- single-asset bug
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent
AGENT_SCRIPT = REPO_ROOT / "forecast_agent.py"
REALIZED_DIR = REPO_ROOT / "realized_vectors"


def _hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def ascii_histogram(draws: np.ndarray, realized: float | None, bins: int = 21, width: int = 48) -> str:
    """Same idea as make_distribution_example.py's ascii_histogram, plus a marker for where
    the realized (true) value actually landed -- the whole visual point of CRPS in one picture:
    did the cloud of draws surround the truth, or miss it?
    """
    counts, edges = np.histogram(draws, bins=bins)
    top = counts.max() if counts.max() else 1
    realized_bin = (
        int(np.clip(np.searchsorted(edges, realized, side="right") - 1, 0, bins - 1))
        if realized is not None else None
    )
    lines = []
    for i, (c, lo) in enumerate(zip(counts, edges[:-1])):
        bar = "#" * int(round(width * c / top))
        marker = "  <-- REALIZED" if i == realized_bin else ""
        lines.append(f"  {lo:9.4f} | {bar}{marker}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("unit_id")
    ap.add_argument("--cell", type=int, default=0, help="which (asset, horizon) cell to plot (default: first)")
    a = ap.parse_args(argv)

    unit_dir = REPO_ROOT / "units" / a.unit_id
    if not unit_dir.is_dir():
        print(f"no such unit: {a.unit_id}", file=sys.stderr)
        return 1
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    asof = card["provenance"]["data_cutoff"]
    assets, horizons = card["targets"]["asset_ids"], [int(h) for h in card["targets"]["horizons"]]
    category = card.get("metadata", {}).get("category", "?")

    _hr(f"STEP 1 — the unit: {a.unit_id} ({category})")
    print(f"  as-of date : {asof}")
    print(f"  assets     : {assets}")
    print(f"  horizons   : {horizons} business days")
    print(f"  panels     : {[p.name for p in unit_dir.glob('*.parquet')]}")
    print(f"  text docs  : {len(list((unit_dir / 'text').glob('*.txt')))}")

    _hr("STEP 2 — run forecast_agent.py (today's read_text_signal + build_draws, unchanged)")
    out = REPO_ROOT / "out" / a.unit_id / "forecast.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(AGENT_SCRIPT), "--panels", str(unit_dir), "--text", str(unit_dir / "text"),
         "--asof", asof, "--out", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    print("  $ python forecast_agent.py --panels ... --asof", asof, "--out", out)
    print(" ", result.stdout.strip().replace("\n", "\n  "))
    if result.returncode != 0:
        print("\n  CRASHED. stderr (last 5 lines):")
        for line in result.stderr.strip().splitlines()[-5:]:
            print(f"    {line}")
        print(f"\n  Stopping here -- steps 3-5 need a forecast.parquet that was never written.")
        return 1

    _hr("STEP 3 — what the agent actually produced")
    df = pd.read_parquet(out)
    n_draws = df["draw"].nunique()
    print(f"  {n_draws} draws x {len(assets)} asset(s) x {len(horizons)} horizon(s) = {len(df)} rows")
    for asset in assets:
        for h in horizons:
            vals = df[(df.asset == asset) & (df.horizon == h)]["value"]
            print(f"  {asset:>8} @ {h:>3}bd : mean={vals.mean():8.4f}  std={vals.std():7.4f}  "
                  f"[5%..95%] = [{vals.quantile(.05):.4f}, {vals.quantile(.95):.4f}]")

    _hr("STEP 4 — admissibility gates (scoring/scoring.py, no realized data)")
    gate_result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scoring" / "scoring.py"), "score",
         "--card", str(unit_dir / "card.toml"), "--forecast", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    gate_payload = json.loads(gate_result.stdout)
    print(f"  admissible: {gate_payload['admissible']}")
    for gate, status in gate_payload.get("gates", {}).items():
        print(f"    {gate}: {status}")

    _hr("STEP 5 — real composite score (build_realized.py's mined ground truth)")
    realized_path = REALIZED_DIR / f"{a.unit_id}.parquet"
    if not realized_path.exists():
        print(f"  no realized vector for this unit (build_realized.py couldn't mine full coverage)")
        print(f"  -> gates-only, as reported above. That's the whole pipeline for this unit.")
        return 0

    realized_df = pd.read_parquet(realized_path)
    print("  realized (true) values, mined from sibling panels:")
    for _, row in realized_df.iterrows():
        print(f"    {row['asset']:>8} @ {int(row['horizon']):>3}bd : {row['value']:.4f}")

    score_result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scoring" / "scoring.py"), "score",
         "--card", str(unit_dir / "card.toml"), "--forecast", str(out),
         "--realized", str(realized_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    score_payload = json.loads(score_result.stdout)
    if "composite_score" in score_payload:
        print(f"\n  marginal_crps  : {score_payload['marginal_crps']:.4f}")
        print(f"  joint_variogram: {score_payload['joint_variogram']:.4f}")
        print(f"  tail_penalty   : {score_payload['tail_penalty']:.4f}")
        print(f"  COMPOSITE      : {score_payload['composite_score']:.4f}  (lower is better; "
              f"1.0 = real leaderboard's text-blind baseline, but this raw number isn't "
              f"normalized the same way -- use it for within-run comparison, not as that number)")

    _hr(f"Picture: cell #{a.cell} ({assets[a.cell % len(assets)]} @ "
        f"{horizons[a.cell % len(horizons)]}bd) -- did the draws cover the truth?")
    asset, h = assets[a.cell % len(assets)], horizons[a.cell % len(horizons)]
    cell_draws = df[(df.asset == asset) & (df.horizon == h)]["value"].to_numpy()
    realized_val = realized_df[(realized_df.asset == asset) & (realized_df.horizon == h)]["value"]
    realized_val = float(realized_val.iloc[0]) if not realized_val.empty else None
    print(ascii_histogram(cell_draws, realized_val))
    if realized_val is not None:
        pct_below = float((cell_draws < realized_val).mean() * 100)
        print(f"\n  realized value {realized_val:.4f} sits at the {pct_below:.0f}th percentile of "
              f"the {len(cell_draws)} draws.")
        print("  (near 50th = well-centered; near 0th/100th = the forecast missed the direction "
              "entirely -- exactly what CRPS's marginal term is penalizing.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
