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
the three output files. The model lives in `forecast_models.py` -- M2.0, one model for every card,
with each asset's step, drift and shocks set by its type. `build_draws` is re-exported here so the
contract above stays true at this import path.

Runs offline with numpy + pandas + pyarrow.
"""

from __future__ import annotations

import argparse
import os
import json
import pathlib
import sys
import tomllib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from forecast_models import build_draws  # re-exported: this is the documented contract path

#: M2.0 draws 2,000: the tail score reads the 1 % and 99 % quantiles off the draws (notebook 15).
DEFAULT_DRAWS = 2000

#: ON on feat/model_v3_m2_0_text: M2.0 with the House-model text half, to compare against the
#: text-off feat/model_v3_m2_0 image. The descriptor must then declare the House model, so pack WITHOUT
#: `tools/make_descriptor.py --no-model`. False = every asset neutral, no House call (then use --no-model).
TEXT_SIGNAL = False  # exp/m2-short-horizon: no evidence text helps at 1-4 days, so OFF (re-enable: True; _safe_text bounds widen to 1.0-1.3)



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
#  Lives in forecast_models.py: build_draws() and M2.0 behind it.
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



#: Wall-clock cap on the whole text half (the Final allows 1,800 s per unit; the model needs seconds).
TEXT_TIME_CAP_S = 600
#: Hard bounds on what the text may ask for. Anything else is clamped; shift is never used.
WIDEN_BOUNDS = (1.0, 1.3)
SKEW_BOUNDS = (-0.3, 0.3)


class _TextTimeout(BaseException):
    pass


#: Wall-clock cap on the model itself (the Final allows 1,800 s per unit; the model takes about a second).
MODEL_TIME_CAP_S = int(os.environ.get("T2_MODEL_CAP_S", "900"))


def _safe_text(text_dir, assets):
    """read_text_signal under a wall-clock cap, clamped; any failure -> exact neutral."""
    import signal
    neutral = {x: {"shift": 0.0, "widen": 1.0, "skew": 0.0} for x in assets}

    def _alarm(*_):
        raise _TextTimeout()
    try:
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(TEXT_TIME_CAP_S)
    except (ValueError, AttributeError):              # not the main thread / no SIGALRM
        old = None
    try:
        raw = read_text_signal(text_dir, assets)
        out = {}
        for x in assets:
            v = raw.get(x, {}) if isinstance(raw, dict) else {}
            w, k = float(v.get("widen", 1.0)), float(v.get("skew", 0.0))
            w = min(max(w, WIDEN_BOUNDS[0]), WIDEN_BOUNDS[1]) if np.isfinite(w) else 1.0
            k = min(max(k, SKEW_BOUNDS[0]), SKEW_BOUNDS[1]) if np.isfinite(k) else 0.0
            out[x] = {"shift": 0.0, "widen": w, "skew": k}
        return out
    except BaseException as exc:
        print(f"[forecast] text half failed ({type(exc).__name__}); model alone", file=sys.stderr)
        return neutral
    finally:
        if old is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


#: Scheduled central-bank event scaling (event_vol.py). Validated on tools/short_horizon_backtest.py
#: (event cells -0.016 vs the unscaled model, out of sample in both 2015-19 and 2020-25).
EVENT_VOL = True


def _set_event_scale(panels, assets, horizons, asof):
    """Fill forecast_models.EVENT_SCALE for daily rates / FX assets; any problem leaves it off (1.0)."""
    import forecast_models as fm
    fm.EVENT_SCALE = None
    if not EVENT_VOL:
        return
    try:
        import event_vol as ev
        steps = ev.future_bdays(asof, max(horizons))
        scale = {}
        for a in assets:
            pid, rows = fm._history(panels, a, asof)
            cls = fm.asset_type(a, pid)
            if cls not in ("rates", "fx") or fm._horizon_steps([d for d, _ in rows], horizons, asof) != [float(h) for h in horizons]:
                continue
            sc = ev.horizon_scale(cls, steps)
            for h in horizons:
                scale[(a, h)] = sc[h - 1]
        fm.EVENT_SCALE = scale or None
    except BaseException as exc:
        print(f"[forecast] event scaling off ({type(exc).__name__}: {exc})", file=sys.stderr)
        fm.EVENT_SCALE = None


def _emergency_draws(panels, assets, horizons, asof, n_draws, seed):
    """Last resort so a card never fails: independent random walks around the last value found for
    each asset, sd 1 % of |last| per sqrt(h) (1.0 if nothing is known). Finite, admissible."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_draws, len(assets), len(horizons)))
    for i, x in enumerate(assets):
        last, sd = 0.0, 1.0
        for t in panels.values():
            try:
                acol = "asset" if "asset" in t.column_names else "asset_id"
                d = t.to_pydict()
                v = [float(val) for dte, ast, val in zip(d["date"], d[acol], d["value"])
                     if str(ast) == x and str(dte)[:10] <= asof and val is not None and np.isfinite(float(val))]
                if len(v) >= 2:
                    last = v[-1]
                    sd = float(np.std(np.diff(v[-300:]))) or 0.01 * abs(last) or 1.0
                    break
            except Exception:
                continue
        z = rng.standard_normal((n_draws, 1)) 
        for j, h in enumerate(horizons):
            out[:, i, j] = last + sd * np.sqrt(max(h, 1)) * z[:, 0]
    return out


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

    neutral = {x: {"shift": 0.0, "widen": 1.0, "skew": 0.0} for x in assets}
    try:
        panels = _read_panels(a.panels)
    except BaseException as exc:                      # unreadable panels: emergency draws below
        print(f"[forecast] panels unreadable ({exc}); emergency draws", file=sys.stderr)
        panels = {}
    # The text half is time-boxed and clamped; any failure leaves the model alone (neutral).
    adjustments = _safe_text(a.text, assets) if TEXT_SIGNAL else neutral              # NISH
    kw = dict(target_type=tgt.get("target_type"), family=card.get("metadata", {}).get("category"))
    samples = None
    _set_event_scale(panels, assets, horizons, a.asof)
    import signal
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_TextTimeout()))
        signal.alarm(MODEL_TIME_CAP_S)
    except (ValueError, AttributeError):
        pass
    for adj in (adjustments, neutral) if adjustments != neutral else (neutral,):
        try:
            samples = build_draws(panels, assets, horizons, a.asof, adj, n_draws, a.seed, **kw)  # DEW
            if np.isfinite(samples).all() and samples.shape == (n_draws, len(assets), len(horizons)):
                break
        except BaseException as exc:                  # incl. SystemExit from the model's refusals
            print(f"[forecast] build_draws failed ({type(exc).__name__}: {exc})", file=sys.stderr)
            if isinstance(exc, _TextTimeout):         # model hit its wall-clock cap: go straight to emergency
                samples = None
                break
        samples = None
    try:
        signal.alarm(0)
    except (ValueError, AttributeError):
        pass
    if samples is None:
        samples = _emergency_draws(panels, assets, horizons, a.asof, n_draws, a.seed)
        adjustments = neutral

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
        + "Base: M2.0 from panel history (forecast_models.py) -- one correlated path per draw, "
          "the step and shocks set by each asset's type, the significant recent trend as drift, "
          "Student-t shocks with nu fitted from the last 260 steps, an ensemble of EWMA volatilities "
          "(halflives 21, 63, 252) times the variance ratio, Treasury yields floored at 0, and 1 draw "
          "in 4 on the full trend.\n"
        + f"Text used: {'yes' if used_text else 'no (baseline stub)'}.\n"
    )
    print(f"wrote forecast.parquet + sidecars to {out_dir} "
          f"({len(assets)} assets x {len(horizons)} horizons, {n_draws} draws)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
