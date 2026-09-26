"""
Track 2 — team agent skeleton.  READ TEAM_TASKS.md FIRST.

This runs end-to-end TODAY as a text-blind baseline (scores ~1.0). Each person fills in
their own function; the shared glue at the bottom never needs to change.

    forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD --out /output/forecast.parquet

THE CONTRACT (do not change the shapes — this is what lets us integrate):
    read_text_signal(text_dir, assets) -> {asset: {"shift": float, "widen": float, "skew": float}}
    build_draws(panels, assets, horizons, asof, adjustments, n_draws, seed) -> np.ndarray
                                                     shape = (n_draws, n_assets, n_horizons)

This file is the contract and the CLI: read the text signal, read the panels, call a model, write
the three output files. The models themselves live in `forecast_models.py` -- M2, the cumulative
walk and the random walk, and the switch that picks between them. `build_draws` is re-exported
here so the contract above stays true at this import path.

Runs offline with numpy + pandas + pyarrow; the F1 M2 path also imports f1_pipeline/m2_unit.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import forecast_models
from forecast_models import build_draws  # re-exported: this is the documented contract path

DEFAULT_DRAWS = 500



# ============================================================================
#  OWNER: NISH  ·  the LLM / text part            branch: feat/llm
#  Read the corpus, ask Nemotron, return per-asset adjustments.
# ============================================================================
def read_text_signal(text_dir: pathlib.Path, assets: list[str]) -> dict[str, dict[str, float]]:
    """Return {asset: {"shift", "widen", "skew"}}.  shift moves the center,
    widen multiplies the spread (>1 = more uncertain), skew tilts the tail.

    Thin delegate: the implementation lives in `text_signal.py` so this branch's diff against
    the shared skeleton stays three lines and cannot collide with feat/timeseries or feat/eval.
    That module reads `text/corpus_index.json` (cutoff-filtered), calls the house endpoint at
    MODEL_ENDPOINT/MODEL_NAME, and degrades to a deterministic offline keyword floor and then to
    exact neutral rather than ever raising -- a card that errors scores 4.0, ignoring the text
    scores 1.0.  Run `python3 text_signal.py --text units/<id>/text` to see what it produced.
    """
    try:
        from text_signal import read_text_signal as _impl
    except Exception as exc:  # module missing/broken -> the text-blind baseline, not a crash
        print(f"[text_signal] unavailable ({exc}); neutral", file=sys.stderr)
        return {a: {"shift": 0.0, "widen": 1.0, "skew": 0.0} for a in assets}
    return _impl(text_dir, assets)



# ============================================================================
#  OWNER: DEW  ·  the time-series / numbers part   branch: feat/model
#  Lives in forecast_models.py: build_draws() and the three models behind it.
# ============================================================================

# ============================================================================
#  SHARED GLUE  ·  do not change the interface.  (Pun's eval runs this whole file.)
# ============================================================================
def _read_panels(panels_dir: pathlib.Path) -> dict[str, "pa.Table"]:
    found = sorted(panels_dir.glob("*.parquet"))
    if not found and panels_dir.parent.is_dir():
        found = sorted(panels_dir.parent.glob("*.parquet"))
    if not found:
        raise SystemExit(f"no .parquet under {panels_dir} (or its parent)")
    return {p.stem: pq.read_table(p) for p in found}


def _find_card(panels: pathlib.Path, explicit: pathlib.Path | None) -> pathlib.Path:
    if explicit:
        return explicit
    for cand in (panels / "card.toml", panels.parent / "card.toml"):
        if cand.exists():
            return cand
    raise SystemExit(f"card.toml not found near {panels}; pass --card")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forecast")
    p.add_argument("--panels", type=pathlib.Path, required=True)
    p.add_argument("--text", type=pathlib.Path, required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--card", type=pathlib.Path, default=None)
    p.add_argument("--n-draws", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    card = tomllib.loads(_find_card(a.panels, a.card).read_text())
    tgt = card["targets"]
    assets = list(tgt["asset_ids"])
    horizons = [int(h) for h in tgt["horizons"]]
    unit_id = card["task"]["id"]
    floor = int(card.get("scoring", {}).get("params", {}).get("n_draws_min", 0) or 0)
    n_draws = max(a.n_draws or DEFAULT_DRAWS, DEFAULT_DRAWS, floor)

    panels = _read_panels(a.panels)
    adjustments = read_text_signal(a.text, assets)          # NISH
    samples = build_draws(panels, assets, horizons, a.asof, adjustments, n_draws, a.seed,  # DEW
                          target_type=tgt.get("target_type"),
                          family=card.get("metadata", {}).get("category"))

    # write the 3 required files
    out_dir = a.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {"draw": [], "asset": [], "horizon": [], "value": []}
    for d in range(n_draws):
        for ai, asset in enumerate(assets):
            for hi, h in enumerate(horizons):
                rows["draw"].append(d)
                rows["asset"].append(asset)
                rows["horizon"].append(h)
                rows["value"].append(float(samples[d, ai, hi]))
    pq.write_table(
        pa.table(
            {
                "draw": pa.array(rows["draw"], pa.int32()),
                "asset": pa.array(rows["asset"], pa.string()),
                "horizon": pa.array(rows["horizon"], pa.int32()),
                "value": pa.array(rows["value"], pa.float64()),
            }
        ),
        a.out,
    )
    (out_dir / "forecast_meta.json").write_text(
        json.dumps(
            {
                "unit_id": unit_id,
                "asof": a.asof,
                "representation": "samples",
                "asset_ids": assets,
                "horizons": horizons,
                "n_draws": n_draws,
            },
            indent=2,
        )
        + "\n"
    )
    # Key-wise, not whole-dict: a whole-dict comparison against a fixed 3-key literal reports
    # "text used: yes" for an all-neutral reply the moment `read_text_signal` grows a fourth key.
    used_text = any(
        v.get("shift", 0.0) != 0.0 or v.get("widen", 1.0) != 1.0 or v.get("skew", 0.0) != 0.0
        for v in adjustments.values()
    )
    (out_dir / "forecast_rationale.md").write_text(
        f"# Forecast rationale — {unit_id}\n\n"
        f"Joint draws for {', '.join(assets)} at horizons {horizons}, as of {a.asof}.\n"
        + (
            "Base: M2 -- ridge location-scale fitted on panel history, joint bootstrap of "
            "standardised residuals (f1_pipeline/m2_unit.py).\n"
            if forecast_models.last_model() == forecast_models.M2 else
            "Base: cumulative correlated Gaussian random walk from panel history -- one "
            "accumulating path per draw, so horizons carry the covariance sqrt(h_j/h_k) rather "
            "than being drawn independently; cross-asset correlation from a date-aligned, "
            "gap-guarded estimate over the trailing 260 rows "
            "(skew implemented but disabled pending measurement -- see _SKEW_ENABLED).\n"
        )
        + f"Text used: {'yes' if used_text else 'no (baseline stub)'}.\n"
    )
    print(f"wrote forecast.parquet + sidecars to {out_dir} "
          f"({len(assets)} assets x {len(horizons)} horizons, {n_draws} draws)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
