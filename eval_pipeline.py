"""
Track 2 (Pun's eval harness) — the ONE command that runs the whole local eval pipeline.

For anyone new to this (including another Claude session): this is the entry point. It runs,
in order:

    1. build_realized.py  -- mine real target values from sibling practice panels
    2. run_eval.py         -- run forecast_agent.py on every practice unit, score each one,
                               write a timestamped JSON report to eval_reports/

That's the whole thing. See EVAL_PIPELINE.md for the full picture, including how to use
eval_one.py (look closely at one unit) and compare_eval.py (did a change help or hurt).

Step 1 is SKIPPED BY DEFAULT: realized_vectors/ is committed to the repo (units/ rarely
changes), so most runs don't need to re-mine it. Pass --remine-realized to force a rebuild
(e.g. after new practice units are added). If realized_vectors/ doesn't exist at all yet --
a fresh checkout that predates it, or someone deleted it -- step 1 runs automatically
regardless of the flag, since running the sweep with zero coverage would be silently useless.

Usage:
    python eval_pipeline.py                          # everything, default settings
    python eval_pipeline.py --label "after Nish's LLM call"
    python eval_pipeline.py --remine-realized         # force-rebuild realized_vectors/ first
                                                       # (only needed if units/ changed)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], step: str) -> int:
    print(f"\n{'#' * 70}\n# {step}\n{'#' * 70}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=None, help="note stored in the eval report (e.g. what changed)")
    ap.add_argument("--out", default=None, help="report path (default: eval_reports/<timestamp>.json)")
    ap.add_argument("--remine-realized", action="store_true",
                     help="force-rebuild realized_vectors/ even though it's already committed "
                          "(only needed after units/ changes)")
    ap.add_argument("--concurrency", type=int, default=None,
                     help="passed straight through to run_eval.py -- run this many units at "
                          "once instead of one at a time (default: 1, sequential)")
    a = ap.parse_args(argv)

    realized_dir = REPO_ROOT / "realized_vectors"
    must_mine = a.remine_realized or not any(realized_dir.glob("*.parquet"))
    if must_mine:
        reason = "--remine-realized passed" if a.remine_realized else "realized_vectors/ is missing or empty"
        rc = _run([sys.executable, "build_realized.py"],
                  f"STEP 1/2 — mining realized values from sibling panels ({reason})")
        if rc != 0:
            print("build_realized.py failed -- stopping before running the agent sweep.", file=sys.stderr)
            return rc
    else:
        print(f"Skipping build_realized.py -- reusing the {len(list(realized_dir.glob('*.parquet')))} "
              f"file(s) already in realized_vectors/ (pass --remine-realized to force a rebuild).")

    run_eval_cmd = [sys.executable, "run_eval.py"]
    if a.label:
        run_eval_cmd += ["--label", a.label]
    if a.out:
        run_eval_cmd += ["--out", a.out]
    if a.concurrency:
        run_eval_cmd += ["--concurrency", str(a.concurrency)]
    rc = _run(run_eval_cmd, "STEP 2/2 — running forecast_agent.py + scoring on every practice unit")
    if rc != 0:
        return rc

    print(f"\n{'#' * 70}")
    print("# Done. See EVAL_PIPELINE.md for what to do next:")
    print("#   python eval_one.py <unit_id>              -- look closely at one unit")
    print("#   python compare_eval.py <before> <after>   -- did a change help or hurt?")
    print(f"{'#' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
