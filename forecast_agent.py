"""
Track 2 — team agent skeleton.  READ TEAM_TASKS.md FIRST.

This runs end-to-end TODAY as a text-blind baseline (scores ~1.0). Each person fills in
their own function; the shared glue at the bottom never needs to change.

    forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD --out /output/forecast.parquet

THE CONTRACT (do not change the shapes — this is what lets us integrate):
    read_text_signal(text_dir, assets) -> {asset: {"shift": float, "widen": float, "skew": float}}
    build_draws(panels, assets, horizons, asof, adjustments, n_draws, seed) -> np.ndarray
                                                     shape = (n_draws, n_assets, n_horizons)

Runs offline with numpy + pandas + pyarrow (pandas carries the date-aligned step frame
in build_draws and the F1 M2 path in f1_pipeline/m2_unit.py).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DRAWS = 500
_ASSET_COLS = ("asset", "asset_id")

#: Trailing rows used to estimate sd and the cross-asset correlation. PINNED, and not a free
#: parameter: the window dominates the choice of method. Measured on the 20 F3 units with realized
#: vectors (mean variogram, horizons drawn independently): 260 -> 2.47, 504 -> 2.58, 1260 -> 2.77,
#: full history -> 3.02. Changing it at the same time as anything else makes the comparison
#: uninterpretable, so change it alone or not at all.
_WINDOW = 260

#: Off pending a clean, controlled measurement. Two live full-sweep comparisons
#: (PUN_TEXT_NOTES.md, 2026-09-22) can't isolate skew's real effect from this endpoint's
#: already-confirmed run-to-run non-determinism: `read_text_signal()` re-calls the live model on
#: every run with nothing cached, so shift/widen also change between "before" and "after" sweeps,
#: not just skew. The apparent aggregate regression (F1/F2/F3 worse, F4 flat) is dominated by one
#: card swinging back almost exactly as far as it swung in the opposite direction the previous
#: comparison -- a signature of response variance, not a real skew effect either way. The tilt
#: itself is implemented and unit-tested correctly (`_skew_tilt` below, `tests/test_build_draws.py`)
#: -- this flag does not undo that work, it just keeps it out of the actual forecast until a
#: same-inputs, code-path-only comparison (skew forced on vs. off against ONE recorded set of
#: model adjustments, not two fresh live calls) actually isolates the effect.
_SKEW_ENABLED = False


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
#  Turn the panels (+ Nish's adjustments) into correlated joint draws.
# ============================================================================
#: Families whose LEVEL cards use M2 (f1_pipeline/m2_unit.py) instead of the random walk below.
#: M2 was built and tuned on F1 only (f1_pipeline notebooks 01-04); other families keep the
#: random walk until M2 is measured on them.
_M2_FAMILIES = {"T2-F1"}

#: Which base produced the last build_draws() call -- read by main() for the rationale.
_last_base = "random walk"


def build_draws(
    panels: dict[str, "pa.Table"],
    assets: list[str],
    horizons: list[int],
    asof: str,
    adjustments: dict[str, dict[str, float]],
    n_draws: int,
    seed: int,
    *,
    target_type: str | None = None,
    family: str | None = None,
) -> np.ndarray:
    """Joint draws with Nish's adjustments applied. Returns (n_draws, n_assets, n_horizons).

    F1 LEVEL cards (`family` in _M2_FAMILIES and `target_type == "level"`): M2 -- ridge centre,
    ridge log-variance width, and a joint bootstrap of standardised residuals (one historical
    date per draw, shared by every cell). `shift` moves the centre, `widen` scales sigma; `skew`
    is not applied, since the residual pool already carries the shape. Any M2 failure falls back
    to the random walk rather than crashing the card.

    Everything else, and callers that pass neither keyword: the random walk below -- one
    shared correlated roll per draw (Cholesky), tilted per-asset by `skew` when enabled.
    """
    global _last_base
    if target_type == "level" and family in _M2_FAMILIES:
        try:
            out = _m2_draws(panels, assets, horizons, asof, adjustments, n_draws, seed)
            _last_base = "M2"
            return out
        except Exception as exc:  # a crashed card scores worst-case; the random walk does not
            print(f"[m2] {type(exc).__name__}: {exc}; falling back to the random walk",
                  file=sys.stderr)
    _last_base = "random walk"

    rng = np.random.default_rng(seed)
    n = len(assets)
    hist = {a: _series(panels, a, asof) for a in assets}

    # Steps, aligned BY DATE. The `[assets]` reselect is not cosmetic: every downstream vector
    # (sd, shift, widen, skew) is built in `assets` order, so if the frame's column order ever
    # diverged the Cholesky would be applied to the wrong assets silently.
    # `.dropna()` BEFORE the window, not after -- otherwise a cross-panel card keeps 260 raw rows
    # and then loses a fraction of them to the date intersection, so the effective window differs
    # per card.
    returns_target = target_type == "log_return"
    steps = pd.DataFrame(
        {a: (pd.Series(_log_return_steps(s.to_numpy()), index=s.index) if returns_target
             else _diff_without_gaps(s))
         for a, s in hist.items()}
    )[assets].dropna().iloc[-_WINDOW:]

    if len(steps) < 30:
        raise SystemExit(f"not enough overlapping history to estimate covariance ({len(steps)} rows)")

    D = steps.to_numpy().T                       # (n_assets, window)
    sd = D.std(axis=1)
    # A cumulative log-return target starts at 0; a level target starts at its last observed value.
    #
    # Neither carries a drift term. For levels that is just the random walk. For log returns it is
    # a deliberate departure from the reference CLI (`cli.py:278`), which centres them on
    # `steps.mean() * h` -- a 260-day mean daily return extrapolated linearly over the horizon.
    # Measured on all 15 log_return units with realized vectors, 5 seeds, 4000 draws: zero drift
    # wins 11/15, mean normalized composite ratio 0.8818. The extrapolation is a noisy momentum
    # bet (t2-F4-short-vol-2018: 0.01573 -> 0.00587 without it, -63%); the martingale is both the
    # standard choice for returns and, here, the measured one.
    last = (np.zeros(n) if returns_target
            else np.array([hist[a].iloc[-1] for a in assets], dtype=float))

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
    skew = (
        np.array([adjustments.get(a, {}).get("skew", 0.0) for a in assets])
        if _SKEW_ENABLED else np.zeros(n)
    )

    # ONE accumulating path per draw, not an independent draw per horizon.
    #
    # Each leg adds an increment of sd sqrt(h_k - h_{k-1}), so after the leg ending at h_k the
    # path has variance sd^2 * sum(h_j - h_j-1) = sd^2 * h_k -- IDENTICAL to the old per-horizon
    # formula, which is why the marginal and tail terms are untouched by this change. What it adds
    # is the covariance the old loop threw away: Cov(path_hj, path_hk) = sd^2 * h_j, i.e.
    # rho = sqrt(h_j / h_k).
    #
    # That value is the point, and it is NOT "as much correlation as possible". Measured on the 20
    # F3 units, forcing a flat cross-horizon rho: 0.0 -> 2.471, 0.577 -> 2.231, 0.707 -> 2.225,
    # 1.0 -> 2.487. rho=1 scores as badly as rho=0. The variogram is a proper scoring rule, so the
    # calibrated gap variance (h_k - h_j)*sd^2 is optimal in expectation -- the cumulative path
    # reaches 2.162, beating every flat rho. Do not "improve" this by pushing the correlation up.
    order = np.argsort(horizons)                 # cards ship ascending; don't rely on it
    out = np.empty((n_draws, n, len(horizons)), dtype=float)
    path = np.zeros((n_draws, n))
    prev = 0
    for hi in order:
        h = int(horizons[hi])
        z = rng.standard_normal((n_draws, n)) @ chol.T   # ONE correlated roll per LEG
        u = np.abs(rng.standard_normal((n_draws, n)))    # independent per asset -- skew only
        # max(..., 0) guards a duplicated or unsorted horizon against a silent NaN under sqrt.
        path = path + _skew_tilt(z, u, skew) * (sd * widen * np.sqrt(max(h - prev, 0)))
        prev = h
        out[:, :, hi] = last + shift + path              # write to the ORIGINAL index
    return out


def _m2_draws(
    panels: dict[str, "pa.Table"],
    assets: list[str],
    horizons: list[int],
    asof: str,
    adjustments: dict[str, dict[str, float]],
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """M2 fitted at `asof` on the panel that holds every asset, then notebook 04's joint draw."""
    from f1_pipeline import m2_unit as m2

    unit = m2.unit_from_panels(panels, assets, horizons, asof, target_type="level")
    fits = [m2.fit_m2(c) for c in m2.build_features(unit)]  # asset-major, then horizon: card order
    shift = [adjustments.get(f.asset, {}).get("shift", 0.0) for f in fits]
    widen = [adjustments.get(f.asset, {}).get("widen", 1.0) for f in fits]
    samples = m2.draw_joint(fits, n_draws, seed, shift, widen)
    return samples.reshape(n_draws, len(assets), len(horizons))


#: The skew-normal family's sample skewness is bounded (|.| < ~0.995 as shape -> infinity) and
#: rises slowly: shape=1 (the top of `skew`'s own [-1,1] contract if used directly) reaches only
#: ~0.14 sample skewness -- a barely-visible tilt, not the "shock" F4 needs (see NISH_TEXT_NOTES.md
#: and PUN_TEXT_NOTES.md). Scaling `skew` up to a shape parameter of +-5 at the clamp's edge
#: reaches ~0.85 instead -- a strong, visibly asymmetric tail without pinning at the family's
#: near-degenerate ceiling.
_SKEW_SHAPE_SCALE = 5.0


def _skew_tilt(z: np.ndarray, u: np.ndarray, skew: np.ndarray) -> np.ndarray:
    """Azzalini's skew-normal construction, per asset (last axis of `z`/`u`, matching `skew`).

    Mixes the correlated roll `z` with an independent |normal| term `u`, weighted by `delta`,
    then recenters and rescales so the result has mean 0 and unit variance for EVERY value of
    skew -- `shift` and `widen` keep meaning exactly what they already mean upstream, and `skew`
    changes shape only. At skew=0, delta=0 and this returns `z` exactly (see
    `test_skew_zero_is_identity`): every card that never asks for a tilt draws identically to
    before this function existed.

    `u` must be independent PER ASSET, drawn separately from `z` -- it must not touch the
    cross-asset correlation `chol` already encodes elsewhere, which this leaves untouched. A skew
    that reaches across assets (e.g. "both legs of this trade break the same way") is real future
    work, not this.
    """
    alpha = skew * _SKEW_SHAPE_SCALE
    delta = alpha / np.sqrt(1.0 + alpha**2)
    mean_shift = delta * np.sqrt(2.0 / np.pi)
    var_scale = np.sqrt(np.clip(1.0 - delta**2 * (2.0 / np.pi), 1e-8, None))
    return (delta * u + np.sqrt(1.0 - delta**2) * z - mean_shift) / var_scale


def _diff_without_gaps(s: "pd.Series") -> "pd.Series":
    """First differences, dropping any difference that spans a hole in the data.

    Ported from `qfbench2_track_forecasting/cli.py:_diff_without_gaps` (kept local rather than
    imported: this file deliberately depends on nothing in the toolkit).

    A transfer card ships its target as an early window plus a single row at the as-of date, with
    the years between deliberately withheld. Differenced naively, that hole reads as one day in
    which the asset moved a decade's worth. Measured on t2-F2-fragile-five-brl-2013: one
    gap-spanning "day" of -0.875 against a typical daily move of 0.030, inflating the estimated
    daily sd by 1.4x. t2-F3-fragile-five-joint-2013 has the same 3654-day hole inside its window.
    The threshold adapts to the panel's own spacing, so a gapless daily panel is untouched.
    """
    d = s.diff()
    when = pd.to_datetime(pd.Series(s.index, index=s.index), errors="coerce")
    step = when.diff().dt.days
    if step.notna().sum() == 0:
        return d
    return d.where(step <= max(float(step.median()) * 10.0, 5.0))


def _log_return_steps(values: np.ndarray) -> np.ndarray:
    """Simple returns -> additive log-return steps, for a cumulative log target.

    Mirrors `qfbench2_track_forecasting/targets.py:log_return_steps`, guard included. Without
    this a log_return card is anchored at its last simple return and its already-return data is
    differenced again -- both wrong. Every row is already a step here, so unlike the level path
    nothing is lost to a leading NaN; the reference does the same.
    """
    rows = np.asarray(values, dtype=np.float64)
    if np.any(rows <= -1.0) or np.any(np.isinf(rows)):
        raise ValueError("log_return history requires finite simple returns greater than -1")
    return np.log1p(rows)


def _series(panels: dict[str, "pa.Table"], asset: str, asof: str) -> "pd.Series":
    """History of one asset up to and including the as-of, from whichever panel holds it.

    Returns a DATE-INDEXED series, not a bare array. The index is what lets `build_draws` align
    assets that live in different panel files by date instead of by row position -- six F3 units
    span two panels whose calendars differ, and stacking those positionally silently mis-estimates
    the cross-asset correlation (measured on t2-F3-divergence-2014: UST_10Y/JPY reads 0.32
    positionally against 0.50 date-aligned).
    """
    seen: set[str] = set()
    for t in panels.values():
        cols = t.column_names
        acol = next((c for c in _ASSET_COLS if c in cols), None)
        if acol is None:
            continue
        d = t.to_pydict()
        seen.update(str(a) for a in d[acol])
        rows = [
            (str(dt)[:10], float(v))
            for dt, a, v in zip(d["date"], d[acol], d["value"])
            if str(a) == asset and str(dt)[:10] <= asof
        ]
        if rows:
            rows.sort()
            idx = pd.to_datetime([r[0] for r in rows])
            return pd.Series([r[1] for r in rows], index=idx, name=asset)
    raise SystemExit(
        f"asset {asset!r} not found in any panel at/before {asof}; "
        f"the panels carry {sorted(seen)}"
    )


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
            if _last_base == "M2" else
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
