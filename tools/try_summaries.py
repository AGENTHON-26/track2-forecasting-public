"""Run the stage-1 summary agent on one real document of each doc_type.

Picks the median-size admissible document of every type across all units, summarizes them in
parallel, and prints each summary with its size reduction and wall time.

    python3 tools/try_summaries.py [--out out/summaries.json]

Reads MODEL_ENDPOINT / MODEL_API_KEY from the shell or from the repo's .env (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import text_signal as ts  # noqa: E402


def load_dotenv(path: pathlib.Path) -> None:
    """KEY=VALUE lines from the repo's .env, without overriding what the shell already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def pick_one_per_type(pick: str = "median") -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for text_dir in sorted(ROOT.glob("units/*/text")):
        _, docs = ts.load_corpus(text_dir)
        for d in docs:
            by_type.setdefault(d["doc_type"], []).append({**d, "unit": text_dir.parent.name})
    picked = []
    for doc_type, docs in sorted(by_type.items()):
        docs.sort(key=lambda d: len(d["text"]))
        i = {"median": len(docs) // 2, "largest": -1, "p25": len(docs) // 4}[pick]
        picked.append(docs[i])
    return picked


def from_manifest(path: pathlib.Path) -> list[dict]:
    """The documents listed in a summary-eval manifest (tools/build_summary_evalset.py)."""
    docs = []
    for m in json.loads(path.read_text()):
        text_dir = ROOT / "units" / m["unit"] / "text"
        _, admissible = ts.load_corpus(text_dir)
        d = next(d for d in admissible if d["doc_id"] == m["doc_id"])
        docs.append({**d, "unit": m["unit"]})
    return docs


def timed(doc: dict) -> dict:
    t0 = time.monotonic()
    out = ts.summarize_doc(doc)
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--pick", choices=["median", "largest", "p25"], default="median",
                   help="which document of each type: median size, largest, or 25th percentile")
    p.add_argument("--patience", action="store_true",
                   help="local dev: wait minutes, not seconds, on 429 (build.nvidia.com throttling)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel model calls; lower this when the endpoint answers 429")
    p.add_argument("--manifest", type=pathlib.Path, default=None,
                   help="summarize exactly these docs, e.g. tools/summary_eval/manifest.json (overrides --pick)")
    a = p.parse_args()
    load_dotenv(ROOT / ".env")
    if a.patience:
        ts._RETRY_WAITS = (10.0, 30.0, 60.0, 120.0, 240.0)

    # Exercise every prompt, including fomc_statement, whose docs are normally passed through.
    ts._PASSTHROUGH_CHARS = 0
    docs = from_manifest(a.manifest) if a.manifest else pick_one_per_type(a.pick)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(timed, docs))
    wall = time.monotonic() - t0

    for r in results:
        tag = "summary" if r["summarized"] else ("ERROR" if r["error"] else "verbatim")
        print(f"=== {r['doc_type']}  {r['unit']} / {r['doc_id']} ({r['timestamp']})")
        print(f"    {r['full_chars']} -> {r['summary_chars']} chars  {r['seconds']}s  [{tag}]")
        print(r["summary"] if r["summary"] is not None else f"    {r['error']}")
        print()
    ok = sum(r["summarized"] for r in results)
    print(f"{ok}/{len(results)} summarized, wall {wall:.0f}s")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
