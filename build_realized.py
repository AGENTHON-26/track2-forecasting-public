"""
Track 2 (Pun's eval harness) — mine realized target values from sibling practice panels.

Why this exists: to score our own forecasts for real (CRPS/variogram/tail composite, not just
gates), we need to know what actually happened at each unit's target date. All our as-of dates
are in the past, and the practice cards are NOT mutually independent -- they're drawn from the
same underlying series, each truncated only at its own as-of date (repo README, "Leakage rules":
measured 75 of 103 units have every realized value appear exactly in a sibling unit's panel).

This is legitimate for OUR OWN local eval tooling, not the submitted agent: forecast_agent.py
never reads this file and never sees another unit's panel. We're building the "was it actually
right" side of scoring, after the agent has already forecast independently -- exactly what the
organizers do with their own sealed realized.parquet. The "no cross-unit lookup" rule forbids the
SUBMITTED AGENT from doing this to cheat at inference time; it says nothing about how we score
our own practice runs afterward.

More accurate than a third-party source (e.g. FRED) would be: same data pipeline, same vintage,
same tickers, zero vendor/revision mismatch. Zero setup: no API key, no network.

## Method

Rather than computing a calendar target date from a generic business-day calendar (which drifts
from real market holidays over 100+ business days), we use the data's own trading-day sequence:

1. Pool every (asset, date, value) row from every unit's panel files into one global,
   deduplicated, date-sorted series PER ASSET. This reconstructs the full available history for
   each asset across all practice units combined (mirrors the "146,692 rows" pool the README's own
   leakage measurement describes).
2. For a given unit's target (asset, horizon), find the unit's own as-of row inside that asset's
   global series, then step forward exactly `horizon` ROWS (not days) -- since the global series
   is built entirely from real trading-day rows, stepping N rows forward IS "N business days
   later", with no calendar-guessing involved.
3. If every (asset, horizon) cell for a unit resolves this way, write
   `realized_vectors/<unit_id>.parquet` in the exact [asset, horizon, value] shape
   `scoring/scoring.py --realized` expects. If any cell doesn't resolve (no sibling extends far
   enough), the unit is skipped entirely -- partial/wrong data is worse than no data.

**Excluded on purpose: macro-panel targets.** `macro_monthly.parquet` is monthly-spaced (one row
per month), not daily -- confirmed by inspection. Horizons are stated in business days, so
"step forward N rows" would silently mean "N months" instead of "N business days" for a
macro-only target. Rather than get this quietly wrong, any unit whose target asset is found ONLY
in a macro_monthly.parquet is skipped and reported separately.

Usage:
    python build_realized.py                  # mine all units, write realized_vectors/
    python build_realized.py --report-only     # just print coverage, write nothing
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tomllib
from collections import defaultdict

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent
UNITS_DIR = REPO_ROOT / "units"
OUT_DIR = REPO_ROOT / "realized_vectors"
_ASSET_COLS = ("asset", "asset_id")


#: Not one of the 103/104 counted practice units -- it ships no forecast_spec.json (confirmed:
#: repo README says so explicitly). Its rates_daily.parquet does NOT track the same canonical
#: series the real units share: checked directly against a real 2024 unit over 124 common dates,
#: mean abs diff 0.16, std 0.20, range -0.46..+0.42 -- noisy, not a constant offset, consistent
#: with separate/illustrative teaching data rather than the shared historical series. Must be
#: excluded from the global pool or it corrupts majority votes on every date it overlaps.
_EXCLUDED_UNITS = {"t2-EXAMPLE-ust-curve-1m"}


def _iter_unit_dirs() -> list[pathlib.Path]:
    return sorted(
        p.parent for p in UNITS_DIR.glob("*/card.toml") if p.parent.name not in _EXCLUDED_UNITS
    )


def _read_panel_rows(unit_dir: pathlib.Path) -> list[tuple[str, str, str, float]]:
    """(panel_filename, asset, date, value) for every row in every parquet under this unit."""
    rows = []
    for pq_path in sorted(unit_dir.glob("*.parquet")):
        df = pd.read_parquet(pq_path)
        acol = next((c for c in _ASSET_COLS if c in df.columns), None)
        if acol is None or "date" not in df.columns or "value" not in df.columns:
            continue
        for asset, date, value in zip(df[acol], df["date"], df["value"]):
            rows.append((pq_path.name, str(asset), str(date)[:10], float(value)))
    return rows


def build_global_series() -> tuple[dict[str, dict[str, float]], dict[str, set[str]]]:
    """Returns (asset -> {date: value}, asset -> {panel filenames it was ever seen in}).

    Uses MAJORITY VOTE per (asset, date), not "first value seen" -- an earlier version kept
    whichever source was scanned first, which happened to only be safe by alphabetical luck.
    Real disagreements exist in this data: e.g. UST_10Y on 2024-01-02 reads 3.95 in 6 of 7 units
    that carry that date, and 4.3104 in exactly one -- `t2-EXAMPLE-ust-curve-1m`, the worked
    exemplar, which is excluded from this pool entirely (see `_EXCLUDED_UNITS`) precisely because
    it isn't one of the counted practice units and doesn't track the same canonical series.
    (First guess when this surfaced was that `t2-F1-conflicting-texts-2024` -- named for a
    same-day "data conflict" premise -- was the deliberately-salted outlier. Checked directly and
    that guess was wrong: it's one of the 6 agreeing on 3.95. The exemplar was the actual outlier.
    Left this note because the wrong guess was reasonable-sounding and worth not repeating.)
    A real market yield has one true historical value; when N-1 sources agree and 1 doesn't, the
    1 is the anomaly, regardless of scan order. Ties (no strict majority) are logged and the
    first-seen value is kept, since a genuine tie means no source is more trustworthy than another
    by this method alone.

    The second return value is what lets us detect "this asset only ever appears in a monthly
    panel" without re-scanning.
    """
    votes: dict[str, dict[str, dict[float, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    first_seen_order: dict[str, dict[str, float]] = defaultdict(dict)
    asset_panels: dict[str, set[str]] = defaultdict(set)
    for unit_dir in _iter_unit_dirs():
        for panel_name, asset, date, value in _read_panel_rows(unit_dir):
            asset_panels[asset].add(panel_name)
            # Bucket floats to a tolerance so 4.3104 and 4.31040000001 don't split the vote.
            bucket = round(value, 6)
            votes[asset][date][bucket] += 1
            first_seen_order[asset].setdefault(date, bucket)

    by_asset: dict[str, dict[str, float]] = defaultdict(dict)
    disputed = 0
    ties = 0
    for asset, by_date in votes.items():
        for date, counts in by_date.items():
            if len(counts) == 1:
                by_asset[asset][date] = next(iter(counts))
                continue
            disputed += 1
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])
            top_count = ranked[0][1]
            tied = [v for v, c in ranked if c == top_count]
            if len(tied) > 1:
                ties += 1
                by_asset[asset][date] = first_seen_order[asset][date]
            else:
                by_asset[asset][date] = ranked[0][0]
    if disputed:
        print(f"note: {disputed} (asset, date) pair(s) had disagreeing sources -- resolved by "
              f"majority vote ({ties} of those were exact ties, kept first-seen)", file=sys.stderr)
    return by_asset, asset_panels


def mine_unit(
    unit_dir: pathlib.Path,
    global_series: dict[str, dict[str, list]],  # asset -> sorted [(date, value), ...]
    asset_panels: dict[str, set[str]],
) -> tuple[str, list[tuple[str, int, float]] | None, str]:
    """Returns (unit_id, rows_or_None, reason). rows is None iff coverage was incomplete."""
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    unit_id = card["task"]["id"]
    asof = card["provenance"]["data_cutoff"]
    targets = card["targets"]
    assets, horizons = list(targets["asset_ids"]), [int(h) for h in targets["horizons"]]

    rows: list[tuple[str, int, float]] = []
    for asset in assets:
        if asset not in global_series:
            return unit_id, None, f"asset {asset!r} not found in any panel at all"
        if asset_panels[asset] == {"macro_monthly.parquet"}:
            return unit_id, None, f"asset {asset!r} only in macro_monthly.parquet (monthly, not daily -- skipped on purpose)"

        series = global_series[asset]  # sorted list of (date, value)
        dates = [d for d, _ in series]
        try:
            asof_idx = dates.index(asof)
        except ValueError:
            return unit_id, None, f"as-of date {asof} not found in asset {asset!r}'s own series"

        for h in horizons:
            target_idx = asof_idx + h
            if target_idx >= len(series):
                return unit_id, None, (
                    f"asset {asset!r} horizon {h}bd: no sibling panel extends "
                    f"{target_idx - len(series) + 1} row(s) further than we have"
                )
            rows.append((asset, h, series[target_idx][1]))
    return unit_id, rows, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-only", action="store_true", help="print coverage, write nothing")
    a = ap.parse_args(argv)

    print("Pooling every panel row across all units...")
    by_asset, asset_panels = build_global_series()
    global_series = {asset: sorted(d.items()) for asset, d in by_asset.items()}
    total_rows = sum(len(v) for v in global_series.values())
    print(f"  {len(global_series)} distinct assets, {total_rows} distinct (asset, date) rows\n")

    covered, skipped = [], []
    for unit_dir in _iter_unit_dirs():
        unit_id, rows, reason = mine_unit(unit_dir, global_series, asset_panels)
        if rows is None:
            skipped.append((unit_id, reason))
            continue
        covered.append((unit_id, rows))

    print(f"Full coverage: {len(covered)} / {len(covered) + len(skipped)} units")
    if not a.report_only:
        OUT_DIR.mkdir(exist_ok=True)
        for unit_id, rows in covered:
            df = pd.DataFrame(rows, columns=["asset", "horizon", "value"])
            df.to_parquet(OUT_DIR / f"{unit_id}.parquet", index=False)
        print(f"Wrote {len(covered)} file(s) to {OUT_DIR}/")

    print(f"\nSkipped: {len(skipped)}")
    for unit_id, reason in skipped:
        print(f"  {unit_id}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
