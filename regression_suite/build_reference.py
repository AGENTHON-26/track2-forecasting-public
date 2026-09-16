#!/usr/bin/env python3
"""Build the T2 regression fixtures: synthetic units, a golden forecast, golden expectations.

Everything here is SYNTHETIC. The panels are generated random walks, the corpus documents are
written for the purpose, and the sealed answers are invented — so the fixtures can live in the
public repo without leaking anything, and the golden composite is a property of the scoring code
alone, not of any market.

Two deliberate design points, both learned the hard way:

- The golden forecast is COMMITTED, not regenerated at check time. Scoring a fixed parquet
  exercises only deterministic arithmetic, so the expected composite can be pinned to 1e-9.
  Regenerating through the CLI would couple the pin to numpy's Generator stream, which is not
  guaranteed stable across versions — the CLI is smoke-tested separately, without exact values.
- One unit is MONTHLY with horizons in months. The first build of units-2026H2-B shipped 15 macro
  cards whose business-day targets landed on no observation at all; this fixture keeps the
  monthly path permanently under test.

Run with --regen to rebuild fixtures and rewrite expected.json after an INTENTIONAL scoring
change; the diff of expected.json is then the reviewable artifact.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zlib

try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # py<3.11
    import tomli
    sys.modules["tomllib"] = tomli

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

CARD = """schema_version = "2.0"

[task]
id    = "{uid}"
track = "forecasting"
title = "Regression fixture — {title}"
split = "public-dev"

[metadata]
difficulty  = "medium"
category    = "T2-F1"
asset_panel = "{panel_id}"

[provenance]
license     = "MIT"
data_source = "synthetic (regression fixture)"
data_cutoff = "{asof}"

[contamination]
canary_guid = "{guid}"

[scoring]
verifier            = "t2.crps_composite"
metric              = "crps_composite"
admissibility_gates = ["g0_integrity", "g1_schema", "g2_cutoff_resource", "g3_domain_semantics"]

[scoring.params]
representation  = "samples"
n_draws_min     = 200
require_samples = true
tail_levels     = [0.01, 0.05, 0.95, 0.99]
joint           = "variogram"

[environment]
cpus    = 2
memory  = "4G"
gpu     = false
network = "none"

[text]
source         = "synthetic"
path           = "text/"
cutoff         = "{asof}"
cutoff_checked = true
n_documents    = 2

[targets]
asset_ids        = {assets}
horizons         = {horizons}
target_type      = "{ttype}"
target_frequency = "{freq}"
target_dates     = {tdates}
value_unit       = "{vunit}"
"""

DOC = """# source: synthetic regression fixture | date: {ts} | type: {dt}

Synthetic {dt} document dated {ts}. The committee observed that the synthetic series continued
to evolve as a random walk, and decided to keep the regression suite's expectations unchanged.
"""


def write_unit(uid, title, panel_id, assets, dates, values, asof, horizons, tdates, freq,
               realized_rows, target_type="level", value_unit="synthetic_units"):
    d = HERE / "units" / uid
    (d / "text").mkdir(parents=True, exist_ok=True)

    rows = [{"date": dt, "asset": a, "value": float(v)}
            for a, series in zip(assets, values) for dt, v in zip(dates, series)]
    pd.DataFrame(rows).to_parquet(d / f"{panel_id}.parquet", index=False)

    docs = []
    for i, (ts, dt_) in enumerate([(dates[len(dates) // 2], "fomc_statement"),
                                   (dates[-1], "cb_speech")], 1):
        fn = f"doc_{i:02d}.txt"
        (d / "text" / fn).write_text(DOC.format(ts=ts, dt=dt_))
        docs.append({"doc_id": fn[:-4], "timestamp": ts, "source": "synthetic",
                     "doc_type": dt_, "file": fn})
    (d / "text" / "corpus_index.json").write_text(json.dumps(
        {"card_id": uid, "asof": asof, "documents": docs}, indent=1) + "\n")

    (d / "card.toml").write_text(CARD.format(
        uid=uid, title=title, panel_id=panel_id, asof=asof,
        # crc32, not hash(): str hashes are salted per process, so a --regen would rewrite
        # every canary for no reason.
        guid=f"00000000-0000-4000-8000-{zlib.crc32(uid.encode()) % 10**12:012d}",
        assets=json.dumps(assets), horizons=json.dumps(horizons),
        tdates=json.dumps(tdates), freq=freq, ttype=target_type, vunit=value_unit))

    # The sealed answer lives OUTSIDE the unit tree, under a non-answer-shaped name, so the
    # published repo carries no reference/ dir or realized* file for the pre-flip sweep to trip
    # on. run_regression.py stages it back to reference/realized.parquet in a temp copy at
    # score time, so the real _hydrate_ctx path is still exercised.
    fx = HERE / "fixtures" / uid
    fx.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(realized_rows).to_parquet(fx / "targets.parquet", index=False)
    return d


def build_units(only=None):
    """Build every fixture, or just `only`. Skipped units are neither generated nor rewritten,
    so their committed panels, goldens and pins stay byte-identical."""
    built = []
    rng = np.random.default_rng(42)
    # daily unit: 2 assets, 260 business days ending at the as-of
    ddates = [str(x)[:10] for x in np.busday_offset("2025-06-02", np.arange(260), roll="forward")]
    asof_d = ddates[-1]
    a = 100 + np.cumsum(rng.normal(0, 0.6, 260))
    b = 4.0 + np.cumsum(rng.normal(0, 0.02, 260))
    td_d = str(np.busday_offset(asof_d, 21))[:10]
    if only in (None, "reg-t2-daily"):
      built.append(write_unit(
        "reg-t2-daily", "two-asset daily", "panel_daily", ["SYN_A", "SYN_B"],
        ddates, [a, b], asof_d, [21], [td_d], "daily",
        [{"draw": 0, "asset": "SYN_A", "horizon": 21, "value": round(float(a[-1]) + 1.1, 4),
          "target_date": td_d},
         {"draw": 0, "asset": "SYN_B", "horizon": 21, "value": round(float(b[-1]) - 0.05, 4),
          "target_date": td_d}]))

    # monthly unit: observations on the 1st, horizon counted in MONTHS
    # 41 observations: the CLI needs >=30 diffs for its covariance estimate
    mdates = [f"{y}-{m:02d}-01" for y in (2023, 2024, 2025, 2026) for m in range(1, 13)][:41]
    asof_m = "2026-06-09"
    c = 300 + np.cumsum(rng.normal(0.4, 0.5, len(mdates)))
    if only in (None, "reg-t2-monthly"):
      built.append(write_unit(
        "reg-t2-monthly", "monthly index", "panel_monthly", ["SYN_CPI"],
        mdates, [c], asof_m, [3], ["2031-02-13"], "monthly",
        [{"draw": 0, "asset": "SYN_CPI", "horizon": 3, "value": round(float(c[-1]) + 1.2, 4),
          "target_date": "2031-02-13"}]))

    # returns unit: two factor-style assets whose panel rows are decimal SIMPLE returns and whose
    # target is the cumulative log return over 21 BD (the shape of the 16 `log_return` cards).
    # Own RNG stream so it can be regenerated alone. The LAST row of SYN_MOM is forced to +0.05:
    # a producer that anchors at the last panel row instead of at 0 (the reference CLI before
    # staging PR #14, public #2) centres its 21-day draws near 0.05 where the correct centre is
    # 21 x mean(log1p(r)) ~ 0.008, and one that differences the rows inflates the spread by
    # ~sqrt(2). Check 13 in run_regression.py measures both, on the CLI and on the baseline.
    if only in (None, "reg-t2-logreturn"):
        rr = np.random.default_rng(4242)
        corr = np.array([[1.0, -0.4], [-0.4, 1.0]])
        z = rr.standard_normal((260, 2)) @ np.linalg.cholesky(corr).T
        mom = 0.0004 + 0.011 * z[:, 0]
        hml = 0.0003 + 0.007 * z[:, 1]
        mom[-1] = 0.05
        built.append(write_unit(
            "reg-t2-logreturn", "two-factor cumulative log return", "panel_factors",
            ["SYN_MOM", "SYN_HML"], ddates, [mom, hml], asof_d, [21], [td_d], "daily",
            [{"draw": 0, "asset": "SYN_MOM", "horizon": 21, "value": 0.021, "target_date": td_d},
             {"draw": 0, "asset": "SYN_HML", "horizon": 21, "value": -0.013, "target_date": td_d}],
            target_type="log_return", value_unit="cumulative_log_return"))
    return built


def golden_submission(unit_dir, out_dir):
    from qfbench2_track_forecasting import cli
    card = _load_card(unit_dir)
    cli.main(["--panels", str(unit_dir), "--text", str(unit_dir / "text"),
              "--asof", card["provenance"]["data_cutoff"],
              "--out", str(out_dir / "forecast.parquet"), "--seed", "7"])


def _load_card(unit_dir):
    import tomllib
    return tomllib.loads((unit_dir / "card.toml").read_text())


def score_minimal(unit_dir, out_dir):
    """Exactly the CodaBench driver's ctx shape — {unit_dir, output_dir} and nothing else.

    The committed unit carries no reference/ (the answer lives in fixtures/<uid>/targets.parquet,
    outside the unit tree, per review of #13) — so, like run_regression's check 1, the answer is
    staged back into a TEMP COPY before scoring. Without this the verifier scores the unit as an
    unranked smoke mount, `v.score` is None, and --regen crashes trying to format it — which is
    exactly what happened the first time --regen ran after the fixtures moved.
    """
    import shutil
    import tempfile

    from qfbench2_track_forecasting import scoring
    staged = pathlib.Path(tempfile.mkdtemp()) / unit_dir.name
    shutil.copytree(unit_dir, staged)
    (staged / "reference").mkdir(exist_ok=True)
    shutil.copy(HERE / "fixtures" / unit_dir.name / "targets.parquet",
                staged / "reference" / "realized.parquet")
    ctx = {"unit_dir": staged, "output_dir": out_dir}
    v = scoring.build_verifier(ctx).run(ctx)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true",
                    help="rebuild fixtures and REWRITE expected.json")
    ap.add_argument("--only", default=None, metavar="UID",
                    help="with --regen: rebuild ONE fixture and merge its pin into expected.json, "
                         "leaving every other fixture and pin byte-identical")
    args = ap.parse_args()

    exp_path = HERE / "expected.json"
    if not args.regen and exp_path.exists():
        print("fixtures exist; use --regen to rebuild")
        return 0

    units = build_units(only=args.only)
    if not units:
        print(f"no fixture named {args.only!r}")
        return 2
    expected = {"numpy_at_regen": np.__version__, "units": {}}
    if args.only and exp_path.exists():
        expected["units"] = json.loads(exp_path.read_text())["units"]
    for d in units:
        gd = d / "golden_submission"
        gd.mkdir(exist_ok=True)
        golden_submission(d, gd)
        v = score_minimal(d, gd)
        assert v.admissible, f"golden submission inadmissible on {d.name}: {v.labels}"
        expected["units"][d.name] = {
            "admissible": True,
            "gates": {k: r.passed for k, r in v.gate_results.items()},
            "composite": v.score,
            "detail": {k: val for k, val in v.detail.items()
                       if isinstance(val, (int, float))},
        }
        print(f"{d.name}: composite {v.score:.9f}")
    exp_path.write_text(json.dumps(expected, indent=1) + "\n")
    print(f"wrote {exp_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
