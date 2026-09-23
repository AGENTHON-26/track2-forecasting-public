"""Score repeated summary runs against the answer key in tools/summary_eval/key/ (no model calls).

For every document in both the runs and the key: which key points each run's summary covers
(regex, see tools/summary_key.py), a check that every number in the summary exists in the source
(or is a correct difference of two source numbers), bullet count and time — per run, so run-to-run
variance is visible.

    python3 tools/eval_summaries.py out/base_r1.json out/base_r2.json out/base_r3.json
    python3 tools/eval_summaries.py out/v8r1 out/v8r2          # prefix: <p>_median.json + <p>_largest.json
      --misses        print every must point missed in any run, with its source quote
      --json PATH     write the per-doc / per-point results for diffing two prompt versions
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from statistics import mean

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import text_signal as ts  # noqa: E402
from summary_key import covers, load_keys, scorable  # noqa: E402

_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _source(r: dict) -> str:
    text_dir = ROOT / "units" / r["unit"] / "text"
    index = json.loads((text_dir / "corpus_index.json").read_text())
    f = next(d["file"] for d in index["documents"] if (d.get("doc_id") or "") == r["doc_id"])
    body = (text_dir / f).read_text(errors="replace")
    if r["doc_type"] == "positioning_report":
        body = ts.main_contract_rows(body)
    return body.replace(",", "").replace("‑", "-")


def _unsupported_numbers(r: dict) -> list[str]:
    """Numbers in the summary that are neither in the source nor a difference of two source numbers."""
    src = _source(r)
    src_nums = {float(n) for n in _NUM.findall(src.replace(",", "")) if n.replace(".", "").isdigit()}
    bad = []
    for n in {m.rstrip(".").replace(",", "") for m in _NUM.findall(r["summary"] or "")}:
        if len(n.replace(".", "")) < 2 or n in src:
            continue
        v = float(n)
        if not any(abs(abs(a - b) - v) < 1e-9 for a in src_nums for b in src_nums if a > v):
            bad.append(n)
    return bad


def load_run(arg: str) -> dict[str, dict]:
    """A run is a JSON list of summarize_doc outputs, or a legacy <prefix> of _median/_largest pairs."""
    paths = [pathlib.Path(arg)] if arg.endswith(".json") else [pathlib.Path(f"{arg}_{p}.json") for p in ("median", "largest")]
    return {r["doc_id"]: r for p in paths if p.is_file() for r in json.loads(p.read_text())}


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--split", choices=["dev", "test"],
                    help="score only this half of tools/summary_eval/split.json (tune on dev, report test once)")
    a = ap.parse_args()

    keys = load_keys()
    if a.split:
        split = json.loads((ROOT / "tools" / "summary_eval" / "split.json").read_text())
        keys = {d: k for d, k in keys.items() if split.get(d) == a.split}
    runs = [load_run(r) for r in a.runs]
    doc_ids = [d for d in keys if any(d in run for run in runs)]
    absent = sorted(set(keys) - set(doc_ids))
    if not doc_ids:
        print("no document in these runs has a key file")
        return 1

    results = []  # one row per (doc, point)
    for doc_id in doc_ids:
        for p in keys[doc_id]["points"]:
            if not scorable(p):
                results.append({"doc_id": doc_id, **p, "hits": None})
                continue
            hits = []
            for run in runs:
                r = run.get(doc_id)
                hits.append(None if r is None or r.get("summary") is None else covers(p, r["summary"]))
            results.append({"doc_id": doc_id, **p, "hits": hits})
    doc_type = {d: next(iter(r[d] for r in runs if d in r))["doc_type"] for d in doc_ids}

    # --- per doc: must covered / must, per run -------------------------------------------------
    n = len(runs)
    print(f"{'doc_type':19} {'doc':34} " + " ".join(f"{'r' + str(i + 1):>5}" for i in range(n))
          + "  all-pts  unstable")
    for doc_id in doc_ids:
        rows = [x for x in results if x["doc_id"] == doc_id and x["hits"] is not None]
        must = [x for x in rows if x["tier"] == "must"]
        cells = []
        for i in range(n):
            got = [x["hits"][i] for x in must if x["hits"][i] is not None]
            cells.append(f"{sum(got)}/{len(must)}" if got or not must else "  -")
        all_hits = [h for x in rows for h in x["hits"] if h is not None]
        unstable = sum(1 for x in rows if len({h for h in x["hits"] if h is not None}) > 1)
        print(f"{doc_type[doc_id]:19} {doc_id[:34]:34} " + " ".join(f"{c:>5}" for c in cells)
              + f"  {100 * sum(all_hits) / max(len(all_hits), 1):5.0f}%  {unstable or '':>8}")

    # --- per doc_type and overall ----------------------------------------------------------------
    def recall(rows: list[dict], i: int | None = None) -> tuple[int, int]:
        hs = [h for x in rows for j, h in enumerate(x["hits"]) if h is not None and (i is None or j == i)]
        return sum(hs), len(hs)

    scored = [x for x in results if x["hits"] is not None]
    print(f"\n{'doc_type':19} {'must recall':>16} {'all-point recall':>18}  per-run must")
    for t in sorted(set(doc_type.values())):
        rows = [x for x in scored if doc_type[x["doc_id"]] == t]
        must = [x for x in rows if x["tier"] == "must"]
        per_run = " ".join(f"{100 * h / c:.0f}%" if c else "-" for h, c in (recall(must, i) for i in range(n)))
        print(f"{t:19} {_pct(*recall(must)):>16} {_pct(*recall(rows)):>18}  {per_run}")
    must = [x for x in scored if x["tier"] == "must"]
    print(f"{'ALL':19} {_pct(*recall(must)):>16} {_pct(*recall(scored)):>18}  "
          + " ".join(f"{100 * h / c:.0f}%" if c else "-" for h, c in (recall(must, i) for i in range(n))))
    kinds: dict[str, list[dict]] = {}
    for x in scored:
        kinds.setdefault(x["kind"], []).append(x)
    print("by kind: " + ", ".join(f"{k} {_pct(*recall(v))}" for k, v in sorted(kinds.items())))
    n_judge = sum(1 for x in results if x["hits"] is None)
    print(f"{len(doc_ids)} docs, {len(scored)} scored points, {n_judge} judge_only points not scored"
          + (f"; {len(absent)} key docs not in these runs" if absent else ""))

    # --- size, time, unsupported numbers -------------------------------------------------------
    print(f"\n{'doc_type':19} {'doc':34} {'bullets':>12} {'chars':>17} {'seconds':>14}  unsupported numbers")
    numbers = {}
    for doc_id in doc_ids:
        rs = [run[doc_id] for run in runs if doc_id in run]
        b = [(x["summary"] or "").count("\n-") + 1 if x["summary"] else 0 for x in rs]
        c = [x.get("summary_chars", len(x["summary"] or "")) for x in rs]
        t = [x.get("seconds", 0) for x in rs]
        bad = sorted({v for x in rs if x.get("summary") for v in _unsupported_numbers(x)})
        numbers[doc_id] = bad
        err = [x["error"] for x in rs if x.get("error")]
        print(f"{doc_type[doc_id]:19} {doc_id[:34]:34} {'/'.join(map(str, b)):>12} "
              f"{'/'.join(map(str, c)):>17} {'/'.join(f'{v:.0f}' for v in t):>14}  "
              f"{', '.join(bad) or '-'}{'  ERR ' + err[0][:40] if err else ''}")

    if a.misses:
        print("\nmust points missed in at least one run:")
        for x in must:
            if not all(h for h in x["hits"] if h is not None):
                marks = "".join("-" if h is None else ("✅" if h else "❌") for h in x["hits"])
                print(f"  {marks} {x['doc_id']} {x['id']} [{x['kind']}] {x['point']}\n"
                      f"      quote: {x['quote'][:160]}")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "runs": a.runs,
            "points": [{k: x[k] for k in ("doc_id", "id", "tier", "kind", "point", "hits")} for x in results],
            "unsupported_numbers": numbers,
        }, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
