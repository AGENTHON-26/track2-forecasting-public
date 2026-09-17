"""
Track 2 — team agent skeleton.  READ TEAM_TASKS.md FIRST.

This runs end-to-end TODAY as a text-blind baseline (scores ~1.0). Each person fills in
their own function; the shared glue at the bottom never needs to change.

    forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD --out /output/forecast.parquet

THE CONTRACT (do not change the shapes — this is what lets us integrate):
    read_text_signal(text_dir, assets) -> {asset: {"shift": float, "widen": float, "skew": float}}
    build_draws(panels, assets, horizons, asof, adjustments, n_draws, seed) -> np.ndarray
                                                     shape = (n_draws, n_assets, n_horizons)

Runs offline with numpy + pyarrow only.
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

DEFAULT_DRAWS = 500
_ASSET_COLS = ("asset", "asset_id")


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
#  OWNER: DEW  ·  the time-series / numbers part   branch: feat/timeseries
#  Turn the panels (+ Nish's adjustments) into correlated joint draws.
# ============================================================================
def build_draws(
    panels: dict[str, "pa.Table"],
    assets: list[str],
    horizons: list[int],
    asof: str,
    adjustments: dict[str, dict[str, float]],
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """Joint Gaussian random walk, correlated ACROSS assets, with Nish's adjustments applied.
    Returns array of shape (n_draws, n_assets, n_horizons).

    BASELINE: single shared correlated roll per draw (Cholesky). Improve me:
    real correlation from history, fat tails (Student-t) for shock cards, skew, etc.
    """
    rng = np.random.default_rng(seed)
    hist = {a: _series(panels, a, asof) for a in assets}
    diffs = np.array(
        [np.diff(hist[a][-260:]) for a in assets], dtype=float
    )  # last ~1y of daily changes, per asset
    m = min(len(d) for d in diffs.tolist()) if diffs.size else 0
    if m < 30:
        raise SystemExit(f"not enough history to estimate covariance ({m} rows)")
    D = np.stack([d[-m:] for d in diffs])  # (n_assets, m)

    last = np.array([hist[a][-1] for a in assets], dtype=float)
    sd = D.std(axis=1)
    # np.corrcoef on a single-row input (single-asset cards) returns a 0-d SCALAR, not a (1,1)
    # matrix -- fill_diagonal then fails with "array must be at least 2-d". atleast_2d fixes the
    # single-asset case (correlation of one variable with itself is trivially [[1.0]]) and is a
    # no-op for multi-asset cards, where corrcoef already returns a proper 2-d matrix.
    corr = np.atleast_2d(np.corrcoef(D))
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    w, v = np.linalg.eigh(corr)  # nearest-PSD nudge
    corr = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    chol = np.linalg.cholesky(corr)

    # apply Nish's adjustments per asset
    shift = np.array([adjustments.get(a, {}).get("shift", 0.0) for a in assets])
    widen = np.array([adjustments.get(a, {}).get("widen", 1.0) for a in assets])
    center = last + shift

    out = np.empty((n_draws, len(assets), len(horizons)), dtype=float)
    for hi, h in enumerate(horizons):
        z = rng.standard_normal((n_draws, len(assets))) @ chol.T  # ONE shared correlated roll
        out[:, :, hi] = center + z * (sd * widen * np.sqrt(h))
    return out


def _series(panels: dict[str, "pa.Table"], asset: str, asof: str) -> np.ndarray:
    """History of one asset up to and including the as-of, from whichever panel holds it."""
    for t in panels.values():
        cols = t.column_names
        acol = next((c for c in _ASSET_COLS if c in cols), None)
        if acol is None:
            continue
        d = t.to_pydict()
        rows = [
            (str(dt)[:10], float(v))
            for dt, a, v in zip(d["date"], d[acol], d["value"])
            if str(a) == asset and str(dt)[:10] <= asof
        ]
        if rows:
            rows.sort()
            return np.array([v for _, v in rows], dtype=float)
    raise SystemExit(f"asset {asset!r} not found in any panel at/before {asof}")


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
    samples = build_draws(panels, assets, horizons, a.asof, adjustments, n_draws, a.seed)  # DEW

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
    used_text = any(
        v != {"shift": 0.0, "widen": 1.0, "skew": 0.0} for v in adjustments.values()
    )
    (out_dir / "forecast_rationale.md").write_text(
        f"# Forecast rationale — {unit_id}\n\n"
        f"Joint draws for {', '.join(assets)} at horizons {horizons}, as of {a.asof}.\n"
        f"Base: correlated Gaussian random walk from panel history.\n"
        f"Text used: {'yes' if used_text else 'no (baseline stub)'}.\n"
    )
    print(f"wrote forecast.parquet + sidecars to {out_dir} "
          f"({len(assets)} assets x {len(horizons)} horizons, {n_draws} draws)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
