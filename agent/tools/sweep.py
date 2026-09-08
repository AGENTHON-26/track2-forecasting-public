"""Run the agent across many practice units and aggregate selfcheck diagnostics per family.

No MODEL_ENDPOINT is required or used here -- this measures the current engine (Phase 1's
harness-wrapped baselines.reasoning_agent, offline fallback path only) so later phases have a
concrete before/after. In particular, this is where the "still exactly Gaussian" claim gets
checked against real numbers instead of left as theory: per-cell kurtosis noise at 500 draws is
too large to trust on its own (see agent/tools/selfcheck.py's docstring), but averaging across dozens
of cells shrinks that noise by roughly sqrt(n), which is enough to read a real signal.

Nothing here reads a realized value -- there is no --realized flag, and this only ever compares
our own draws against quantities derived from the as-of panel.

Usage:
    python -m agent.tools.sweep --category T2-F4
    python -m agent.tools.sweep                      # every family
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tempfile
import tomllib
from collections import defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.core.cli import main as agent_main  # noqa: E402
from agent.tools.selfcheck import check_unit  # noqa: E402

#: |z| below this on a family's mean excess kurtosis means "not statistically distinguishable
#: from a flat Gaussian" -- worth flagging only for F4, which requires fat tails outright.
_FLAT_Z_THRESHOLD = 2.0


def discover_units(units_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p.parent for p in units_dir.glob("*/card.toml"))


def run_sweep(units_dir: pathlib.Path, category_filter: str | None = None) -> dict:
    per_category_kurt: dict[str, list[float]] = defaultdict(list)
    per_category_spread: dict[str, list[float]] = defaultdict(list)
    failures: list[tuple[str, str]] = []
    skipped = 0

    for unit_dir in discover_units(units_dir):
        card = tomllib.loads((unit_dir / "card.toml").read_text())
        category = card.get("metadata", {}).get("category")
        if category is None or (category_filter and category != category_filter):
            skipped += 1
            continue
        asof = card.get("provenance", {}).get("data_cutoff")
        if not asof:
            skipped += 1
            continue

        panels_dir = unit_dir / "panels"
        if not panels_dir.is_dir():
            panels_dir = unit_dir
        text_dir = unit_dir / "text"

        # `_draw`'s RNG is seeded straight from `--seed`, which defaults to a fixed 0. Every
        # single-asset unit would then generate the IDENTICAL underlying standard-normal
        # sequence (kurtosis is scale/shift invariant, so rescaling per unit doesn't change
        # that) -- 34 correlated copies of one sample, not 34 independent ones. A per-unit
        # seed is what makes the aggregate's independence assumption (and its SEM) valid.
        seed = int(hashlib.sha256(unit_dir.name.encode()).hexdigest()[:8], 16)
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "forecast.parquet"
            try:
                rc = agent_main(
                    [
                        "--panels", str(panels_dir),
                        "--text", str(text_dir),
                        "--asof", asof,
                        "--out", str(out),
                        "--card", str(unit_dir / "card.toml"),
                        "--seed", str(seed),
                    ]
                )
                if rc != 0:
                    failures.append((unit_dir.name, f"agent exited {rc}"))
                    continue
                report = check_unit(unit_dir, out, asof)
            except Exception as exc:  # noqa: BLE001 -- one bad unit must not kill the sweep
                failures.append((unit_dir.name, f"{type(exc).__name__}: {exc}"))
                continue

        for c in report.cells:
            per_category_kurt[category].append(c.excess_kurtosis)
            per_category_spread[category].append(c.spread_ratio)

    return {
        "kurtosis": per_category_kurt,
        "spread": per_category_spread,
        "failures": failures,
        "skipped": skipped,
    }


def summarize(results: dict) -> str:
    lines = []
    for category in sorted(results["kurtosis"]):
        vals = np.array(results["kurtosis"][category])
        n = len(vals)
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        z = mean / sem if sem > 0 else float("nan")
        spread_vals = np.array(results["spread"][category])
        lines.append(
            f"{category}: n_cells={n} mean_excess_kurtosis={mean:+.3f} "
            f"(SEM={sem:.3f}, z={z:+.1f}) mean_spread_ratio={spread_vals.mean():.2f}"
        )
        if category == "T2-F4" and abs(z) < _FLAT_Z_THRESHOLD:
            lines.append(
                f"  FLAG: F4's mean excess kurtosis is not statistically distinguishable from "
                f"0 (|z|={abs(z):.1f} < {_FLAT_Z_THRESHOLD}) -- the engine is not producing fat "
                f"tails in aggregate, exactly as predicted from the linear vol_scale/drift math"
            )
    if results["skipped"]:
        lines.append(f"skipped (no category or no data_cutoff): {results['skipped']}")
    if results["failures"]:
        lines.append(f"failures: {len(results['failures'])}")
        for name, err in results["failures"][:10]:
            lines.append(f"  {name}: {err}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--units-dir", type=pathlib.Path, default=REPO_ROOT / "units")
    ap.add_argument("--category", default=None, help="e.g. T2-F4; omit to sweep all families")
    a = ap.parse_args(argv)

    results = run_sweep(a.units_dir, a.category)
    print(summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
