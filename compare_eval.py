"""
Track 2 (Pun's eval harness) — compare two eval_reports/*.json runs.

This is the "did it improve?" tool TEAM_TASKS.md asks for: run run_eval.py before a change,
run it again after, then point this at the two report files.

Usage:
    python run_eval.py --label "before Dew's fix" --out eval_reports/before.json
    # ... make the change to forecast_agent.py ...
    python run_eval.py --label "after Dew's fix" --out eval_reports/after.json
    python compare_eval.py eval_reports/before.json eval_reports/after.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=pathlib.Path)
    ap.add_argument("after", type=pathlib.Path)
    a = ap.parse_args(argv)

    before, after = load(a.before), load(a.after)
    print(f"BEFORE: {a.before.name}  ({before.get('label') or 'no label'}, {before['generated_at']})")
    print(f"AFTER:  {a.after.name}  ({after.get('label') or 'no label'}, {after['generated_at']})")

    print("\nStatus counts:")
    statuses = sorted(set(before["status_counts"]) | set(after["status_counts"]))
    for s in statuses:
        b, af = before["status_counts"].get(s, 0), after["status_counts"].get(s, 0)
        arrow = f" ({af - b:+d})" if af != b else ""
        print(f"  {s}: {b} -> {af}{arrow}")

    print("\nComposite score by family (lower is better):")
    families = sorted(set(before["family_summary"]) | set(after["family_summary"]))
    for fam in families:
        b = before["family_summary"].get(fam)
        af = after["family_summary"].get(fam)
        if b and af:
            delta = af["mean"] - b["mean"]
            direction = "better" if delta < 0 else "worse" if delta > 0 else "unchanged"
            print(f"  {fam}: n={b['n']}->{af['n']}  mean {b['mean']:.4f} -> {af['mean']:.4f} "
                  f"({delta:+.4f}, {direction})")
        elif af:
            print(f"  {fam}: new this run, n={af['n']} mean={af['mean']:.4f}")
        elif b:
            print(f"  {fam}: no longer scored (was n={b['n']} mean={b['mean']:.4f})")

    before_units, after_units = before["units"], after["units"]
    all_ids = sorted(set(before_units) | set(after_units))

    newly_crashed = [u for u in all_ids if before_units.get(u, {}).get("status") != "agent_crashed"
                     and after_units.get(u, {}).get("status") == "agent_crashed"]
    newly_fixed = [u for u in all_ids if before_units.get(u, {}).get("status") == "agent_crashed"
                   and after_units.get(u, {}).get("status") != "agent_crashed"]
    if newly_fixed:
        print(f"\nNewly working ({len(newly_fixed)}), previously agent_crashed:")
        for u in newly_fixed[:15]:
            print(f"  {u}: -> {after_units[u]['status']}")
    if newly_crashed:
        print(f"\nREGRESSION -- newly crashing ({len(newly_crashed)}), previously OK:")
        for u in newly_crashed[:15]:
            print(f"  {u}: was {before_units[u]['status']}")

    both_scored = [u for u in all_ids
                   if before_units.get(u, {}).get("status") == "scored"
                   and after_units.get(u, {}).get("status") == "scored"]
    if both_scored:
        deltas = sorted(
            ((u, after_units[u]["composite"] - before_units[u]["composite"]) for u in both_scored),
            key=lambda x: x[1],
        )
        print(f"\nBiggest composite improvements (scored in both runs, {len(both_scored)} total):")
        for u, d in deltas[:5]:
            print(f"  {u}: {before_units[u]['composite']:.4f} -> {after_units[u]['composite']:.4f} ({d:+.4f})")
        print("Biggest composite regressions:")
        for u, d in deltas[-5:]:
            print(f"  {u}: {before_units[u]['composite']:.4f} -> {after_units[u]['composite']:.4f} ({d:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
