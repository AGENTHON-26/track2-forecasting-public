"""
Track 2 (Pun's eval harness) — run forecast_agent.py across practice units and score it.

Per TEAM_TASKS.md: run the agent on practice units -> forecast.parquet -> score it (gates
always; real composite CRPS/variogram/tail wherever build_realized.py mined a realized vector),
so we can tell whether a change to Dew's or Nish's code actually helped.

A per-unit failure (the agent crashes, exits non-zero, or fails a gate) is recorded and reported,
never allowed to kill the whole sweep -- catching exactly this kind of failure, cheaply and in
bulk, is the point of this harness.

Every run writes a JSON report to eval_reports/ (default: timestamped; override with --out) --
that's what makes this a pipeline the whole team can use, not just something run ad hoc in one
session: check the report into a PR, or point compare_eval.py at two of them to see whether a
change to forecast_agent.py helped or hurt.

Usage:
    python run_eval.py                       # every unit with a realized vector + gates-only for the rest
    python run_eval.py --gates-only           # skip composite scoring even where realized data exists
    python run_eval.py --unit t2-F1-cad-boc-2017   # a single unit, verbose (no report file written)
    python run_eval.py --out my_report.json   # explicit report path instead of the timestamped default
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parent
UNITS_DIR = REPO_ROOT / "units"
REPORTS_DIR = REPO_ROOT / "eval_reports"
REALIZED_DIR = REPO_ROOT / "realized_vectors"
AGENT_SCRIPT = REPO_ROOT / "forecast_agent.py"

#: Same exclusion as build_realized.py -- not one of the counted practice units.
_EXCLUDED_UNITS = {"t2-EXAMPLE-ust-curve-1m"}


#: `[text_signal]` lines that are NOT failures. Anything else on that prefix is a call that
#: failed or fell back to neutral.
#:
#: `source=llm` is the success line. `adj ` is the per-asset ledger -- it reports what the model
#: said and what the clamps did, on a unit that worked, so counting it as an issue would mark
#: every healthy unit as broken and destroy the one signal that tells a real sweep from a
#: silently-neutral one.
_TEXT_SIGNAL_OK_MARKERS = ("source=llm", "[text_signal] adj ")


def _extract_text_signal_issues(stderr: str) -> list[str]:
    """`[text_signal]` lines that mean a call failed or fell back to neutral.

    forecast_agent.py doesn't crash on a rate-limited or malformed model reply -- it degrades to
    NEUTRAL and exits 0 -- so without this, a sweep's report can't tell a unit that silently lost
    its text signal from one that never had a chance to be wrong. See PUN_TEXT_NOTES.md,
    "silent rate-limit fallback": 67% of three earlier sweeps had fallen back this way, which is
    why three genuinely different code versions scored identically.
    """
    return [
        line for line in stderr.splitlines()
        if line.startswith("[text_signal]")
        and not any(ok in line for ok in _TEXT_SIGNAL_OK_MARKERS)
    ]


def _iter_unit_dirs() -> list[pathlib.Path]:
    return sorted(
        p.parent for p in UNITS_DIR.glob("*/card.toml") if p.parent.name not in _EXCLUDED_UNITS
    )


def run_one(unit_dir: pathlib.Path, gates_only: bool) -> dict:
    """Times the whole unit (agent + scoring) and stamps the result with elapsed_seconds,
    regardless of which branch below it returns from -- see _run_one for the actual work."""
    start = time.perf_counter()
    result = _run_one(unit_dir, gates_only)
    result["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    return result


def _run_one(unit_dir: pathlib.Path, gates_only: bool) -> dict:
    card = tomllib.loads((unit_dir / "card.toml").read_text())
    unit_id = card["task"]["id"]
    asof = card["provenance"]["data_cutoff"]
    realized_path = REALIZED_DIR / f"{unit_id}.parquet"

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "forecast.parquet"
        agent_result = subprocess.run(
            [sys.executable, str(AGENT_SCRIPT),
             "--panels", str(unit_dir), "--text", str(unit_dir / "text"),
             "--asof", asof, "--out", str(out)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        text_signal_issues = _extract_text_signal_issues(agent_result.stderr)
        if agent_result.returncode != 0:
            tail = "\n".join(agent_result.stderr.strip().splitlines()[-3:])
            return {"unit_id": unit_id, "status": "agent_crashed", "detail": tail,
                    "text_signal_issues": text_signal_issues}

        score_args = [sys.executable, str(REPO_ROOT / "scoring" / "scoring.py"), "score",
                      "--card", str(unit_dir / "card.toml"), "--forecast", str(out)]
        if not gates_only and realized_path.exists():
            score_args += ["--realized", str(realized_path)]
        score_result = subprocess.run(score_args, capture_output=True, text=True, cwd=REPO_ROOT)
        try:
            payload = json.loads(score_result.stdout)
        except json.JSONDecodeError:
            return {"unit_id": unit_id, "status": "scorer_crashed",
                    "detail": score_result.stderr.strip()[-300:],
                    "text_signal_issues": text_signal_issues}

        if not payload.get("admissible", False):
            failing_gate = next(
                (g for g, v in payload.get("gates", {}).items() if v != "pass"), "?"
            )
            return {"unit_id": unit_id, "status": "inadmissible", "detail": failing_gate,
                    "gates": payload.get("gates", {}), "text_signal_issues": text_signal_issues}

        # The --realized branch of scoring.py has no "scored" key at all -- it just adds
        # composite_score directly to an admissible payload. Check for that key's presence,
        # not a "scored" boolean that only exists in the no-realized (gates-only) branch.
        if "composite_score" in payload:
            return {"unit_id": unit_id, "status": "scored",
                    "composite": payload["composite_score"],
                    "marginal_crps": payload.get("marginal_crps"),
                    "joint_variogram": payload.get("joint_variogram"),
                    "tail_penalty": payload.get("tail_penalty"),
                    "tail_metric": payload.get("tail_metric"),
                    "category": card.get("metadata", {}).get("category"),
                    "text_signal_issues": text_signal_issues}
        return {"unit_id": unit_id, "status": "gates_only",
                "category": card.get("metadata", {}).get("category"),
                "text_signal_issues": text_signal_issues}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--unit", default=None, help="run just this one unit id (no report file written)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                     help="report file path (default: eval_reports/<timestamp>.json)")
    ap.add_argument("--label", default=None, help="a short note stored in the report (e.g. a git ref/commit)")
    ap.add_argument("--limit", type=int, default=None,
                     help="run only the first N units (sorted order), report file still written -- "
                          "for a quick look at a real (but small, and non-representative) report "
                          "without paying for a full sweep")
    ap.add_argument("--require-clean-text", action="store_true",
                     help="refuse to print the family score table if ANY unit's text signal "
                          "failed or fell back to neutral. Use this for every text A/B: a sweep "
                          "with silent fallbacks measures the endpoint's mood, not your change. "
                          "PUN_TEXT_NOTES.md records three sweeps of genuinely different code "
                          "that scored identically because 67%% of units had fallen back.")
    ap.add_argument("--concurrency", type=int, default=1,
                     help="run this many units' forecast_agent.py + scoring at once (default: "
                          "1, sequential -- each unit is a separate subprocess, so raising this "
                          "is safe from a Python-GIL perspective; the real constraint is how hard "
                          "several units' worth of concurrent model calls hit the shared "
                          "endpoint. Keep this modest -- e.g. 4 -- since Stage 1's own "
                          "per-document summarization already runs 8-way in parallel WITHIN one "
                          "unit, so --concurrency 4 means up to 32 simultaneous calls, not 4.")
    a = ap.parse_args(argv)

    unit_dirs = [d for d in _iter_unit_dirs() if a.unit is None or d.name == a.unit]
    if not unit_dirs:
        print(f"no unit matched {a.unit!r}", file=sys.stderr)
        return 1
    if a.limit is not None:
        unit_dirs = unit_dirs[:a.limit]

    by_status = defaultdict(list)
    composites_by_category = defaultdict(list)
    all_results = []
    concurrency = max(1, a.concurrency)
    wall_start = time.perf_counter()
    if concurrency == 1:
        results: list[dict] = [run_one(unit_dir, a.gates_only) for unit_dir in unit_dirs]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_one, unit_dir, a.gates_only) for unit_dir in unit_dirs]
            results = [f.result() for f in as_completed(futures)]  # completion order, not input order
    wall_seconds = round(time.perf_counter() - wall_start, 2)

    for result in results:
        by_status[result["status"]].append(result)
        all_results.append(result)
        if result["status"] == "scored":
            composites_by_category[result.get("category", "?")].append(result["composite"])
        if a.unit:
            print(json.dumps(result, indent=2))

    units_with_issues = [r for r in all_results if r.get("text_signal_issues")]
    per_unit_seconds = [r["elapsed_seconds"] for r in all_results]

    if not a.unit:
        total = len(unit_dirs)
        avg_seconds = sum(per_unit_seconds) / len(per_unit_seconds) if per_unit_seconds else 0.0
        print(f"Ran {total} unit(s) in {wall_seconds:.1f}s wall-clock "
              f"(avg {avg_seconds:.1f}s/unit, concurrency={concurrency}):")
        for status in ("scored", "gates_only", "inadmissible", "agent_crashed", "scorer_crashed"):
            items = by_status.get(status, [])
            if items:
                print(f"  {status}: {len(items)}")
        if by_status.get("agent_crashed"):
            print("\nagent_crashed detail (first 5):")
            for r in by_status["agent_crashed"][:5]:
                print(f"  {r['unit_id']}: {r['detail']}")
        if by_status.get("inadmissible"):
            print("\ninadmissible detail (first 5):")
            for r in by_status["inadmissible"][:5]:
                print(f"  {r['unit_id']}: failed {r['detail']}")
        if units_with_issues:
            print(f"\ntext_signal issues in {len(units_with_issues)}/{total} unit(s) -- forecast_agent.py "
                  f"doesn't crash on these, so they'd otherwise be invisible (full list also in the "
                  f"report's \"units\" for units not shown here):")
            for r in units_with_issues[:15]:
                for line in r["text_signal_issues"]:
                    print(f"  {r['unit_id']}: {line}")
            if len(units_with_issues) > 15:
                print(f"  ... and {len(units_with_issues) - 15} more unit(s); see the report file")
        slowest = sorted(all_results, key=lambda r: r["elapsed_seconds"], reverse=True)[:5]
        if slowest:
            print("\nslowest 5 unit(s):")
            for r in slowest:
                print(f"  {r['unit_id']}: {r['elapsed_seconds']:.1f}s")
        if a.require_clean_text and units_with_issues:
            print(f"\n{'=' * 78}")
            print(f"REFUSING to report scores: {len(units_with_issues)}/{total} unit(s) lost or "
                  f"degraded their text signal.")
            print("Those units forecast text-blind, so a comparison against another sweep would "
                  "be measuring\nhow the endpoint behaved today, not the change under test. "
                  "Re-run when the endpoint is\nhealthy, or drop --require-clean-text to see the "
                  "numbers anyway and treat them as untrusted.")
            print(f"{'=' * 78}")
            return 2

        if composites_by_category:
            print("\nComposite score by family (lower is better; 1.0 = text-blind baseline on the "
                  "REAL leaderboard -- this raw composite is NOT normalized the same way, so treat "
                  "it as a within-run comparison tool, not a leaderboard-equivalent number):")
            for cat, vals in sorted(composites_by_category.items()):
                print(f"  {cat}: n={len(vals)} mean={sum(vals)/len(vals):.4f} "
                      f"min={min(vals):.4f} max={max(vals):.4f}")

    if a.unit:
        return 0  # single-unit runs are for looking, not for the team-shared report

    family_summary = {
        cat: {"n": len(vals), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
        for cat, vals in composites_by_category.items()
    }
    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": a.label,
        "gates_only_mode": a.gates_only,
        "total_units": len(unit_dirs),
        "status_counts": {status: len(items) for status, items in by_status.items()},
        "units_with_text_signal_issues": len(units_with_issues),
        "wall_seconds": wall_seconds,
        "avg_unit_seconds": round(sum(per_unit_seconds) / len(per_unit_seconds), 2) if per_unit_seconds else 0.0,
        "concurrency": concurrency,
        "family_summary": family_summary,
        "units": {r["unit_id"]: r for r in all_results},
    }
    out_path = a.out or (REPORTS_DIR / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
