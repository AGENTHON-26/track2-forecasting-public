"""Run the text half over every shipped unit and print one line per unit.

This is the sanity check that matters before trusting any change to the lexicon or the prompt:
a signal that is hawkish on all 104 cards is a constant, not a signal, and a constant that
happens to point the right way on the practice set will not survive the sealed one.

    python3 tools/sweep_text_signal.py                 # offline keyword floor, ~20s
    python3 tools/sweep_text_signal.py --csv out/sweep.csv
    MODEL_ENDPOINT=... MODEL_NAME=... python3 tools/sweep_text_signal.py --limit 5

Prints, per unit: the drift in sigma (comparable across yields/FX/factors), the widen, and how
much of the corpus survived the cutoff filter and the prompt budget.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import text_signal as ts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sweep_text_signal")
    p.add_argument("--units", type=pathlib.Path, default=pathlib.Path("units"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--csv", type=pathlib.Path, default=None)
    p.add_argument("--quiet", action="store_true", help="suppress the per-call stderr line")
    a = p.parse_args(argv)

    units = sorted(d for d in a.units.iterdir() if d.is_dir() and (d / "card.toml").is_file())
    if a.limit:
        units = units[: a.limit]

    rows: list[dict] = []
    err = sys.stderr
    if a.quiet:
        sys.stderr = open("/dev/null", "w")  # noqa: SIM115
    try:
        for u in units:
            card = tomllib.loads((u / "card.toml").read_text())
            assets = list(card["targets"]["asset_ids"])
            adj = ts.read_text_signal(u / "text", assets)
            led = ts.LAST_LEDGER
            for asset in assets:
                per = led.get("assets", {}).get(asset, {})
                rows.append(
                    {
                        "unit": u.name,
                        "family": card["metadata"].get("category", ""),
                        "asset": asset,
                        "unit_of_value": (led.get("value_unit") or "")[:24],
                        "source": led.get("source", ""),
                        "drift_sd": round(float(per.get("drift_sd", 0.0)), 3),
                        "shift": round(float(adj[asset]["shift"]), 5),
                        "widen": round(float(adj[asset]["widen"]), 3),
                        "sigma": round(float(per.get("sigma", 0.0)), 5),
                        "docs": led.get("n_docs_used", 0),
                        "cut": led.get("n_docs_excluded_by_cutoff", 0),
                        "dropped": led.get("n_docs_dropped_by_budget", 0),
                        "note": per.get("note", "") or led.get("reason", ""),
                    }
                )
    finally:
        if a.quiet:
            sys.stderr.close()
            sys.stderr = err

    hdr = f"{'unit':<34}{'asset':<10}{'src':<10}{'driftSD':>8}{'shift':>11}{'widen':>7}{'docs':>5}{'drop':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['unit']:<34}{r['asset']:<10}{r['source']:<10}{r['drift_sd']:>8.3f}"
            f"{r['shift']:>11.5f}{r['widen']:>7.3f}{r['docs']:>5}{r['dropped']:>5}"
        )

    drifts = [r["drift_sd"] for r in rows]
    widens = [r["widen"] for r in rows]
    pos = sum(1 for d in drifts if d > 0.01)
    neg = sum(1 for d in drifts if d < -0.01)
    flat = len(drifts) - pos - neg
    print(f"\n{len(rows)} asset-rows over {len(units)} units")
    print(f"  drift_sd : {pos} up / {neg} down / {flat} flat   "
          f"mean {statistics.fmean(drifts):+.3f}  sd {statistics.pstdev(drifts):.3f}  "
          f"min {min(drifts):+.3f}  max {max(drifts):+.3f}")
    print(f"  widen    : mean {statistics.fmean(widens):.3f}  "
          f"min {min(widens):.3f}  max {max(widens):.3f}")
    print(f"  corpus   : {sum(r['docs'] for r in rows)//max(1,len(rows))} docs/row avg, "
          f"{sum(r['dropped'] for r in rows)} dropped by prompt budget")
    by_src: dict[str, int] = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"  source   : {by_src}")
    bad = [r for r in rows if r["note"]]
    if bad:
        print(f"\n  {len(bad)} rows carry a note:")
        for r in bad[:15]:
            print(f"    {r['unit']:<34}{r['asset']:<10}{r['note'][:70]}")

    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with a.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
