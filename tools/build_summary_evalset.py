"""Pick the documents the summary-coverage eval runs on (deterministic, no model calls).

About 8 unique documents per doc_type, spread across the size distribution, plus every document
that the legacy FACTS table in tools/eval_summaries.py covers. Writes tools/summary_eval/manifest.json.

    python3 tools/build_summary_evalset.py [--per-type 8] [--all]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import text_signal as ts  # noqa: E402

OUT = ROOT / "tools" / "summary_eval" / "manifest.json"

#: The 16 documents the legacy FACTS table checks; always included so old and new numbers compare.
LEGACY = [
    "beige_book-2022-07-31", "beige_book-2024-11-30", "bis_bernanke_2008-07-15",
    "bis_shirakawa_2012-11-12", "moderna_8k_2020-10-29_d134701dex991",
    "pfizer_8k_2020-10-27_pfe-09272020xex99", "fomc-minutes-20220504", "fomc-minutes-20150128",
    "fomc-statement-2024-07-31", "fomc-statement-2014-09-17", "powell_jackson_hole_2022",
    "boe_mpc_statement_20220922", "macro_release-2022-05-11", "macro_release-2015-02-26",
    "cftc_cot_japanese_2007", "cftc_cot_crude_2014",
]


def unique_docs() -> dict[str, dict]:
    """doc_id -> first admissible occurrence (units in sorted order), with its unit and file."""
    docs: dict[str, dict] = {}
    for text_dir in sorted(ROOT.glob("units/*/text")):
        _, admissible = ts.load_corpus(text_dir)
        ids = {d["doc_id"] for d in admissible}
        index = json.loads((text_dir / "corpus_index.json").read_text())
        for e in index.get("documents", []):
            doc_id = e.get("doc_id") or pathlib.Path(e["file"]).stem
            if doc_id not in ids or doc_id in docs:
                continue
            text = (text_dir / e["file"]).read_text(errors="replace")
            docs[doc_id] = {
                "doc_id": doc_id,
                "unit": text_dir.parent.name,
                "file": e["file"],
                "doc_type": str(e.get("doc_type") or "unknown"),
                "timestamp": str(e.get("timestamp", ""))[:10],
                "chars": len(text),
            }
    return docs


def spread(docs: list[dict], k: int) -> list[dict]:
    """k documents at evenly spaced size quantiles, smallest and largest included."""
    docs = sorted(docs, key=lambda d: (d["chars"], d["doc_id"]))
    if len(docs) <= k:
        return docs
    idx = sorted({round(i * (len(docs) - 1) / (k - 1)) for i in range(k)})
    return [docs[i] for i in idx]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-type", type=int, default=8)
    p.add_argument("--all", action="store_true", help="every unique document (full-corpus eval)")
    a = p.parse_args()

    docs = unique_docs()
    missing = [d for d in LEGACY if d not in docs]
    if missing:
        print(f"legacy docs not found: {missing}", file=sys.stderr)

    by_type: dict[str, list[dict]] = {}
    for d in docs.values():
        by_type.setdefault(d["doc_type"], []).append(d)

    chosen: dict[str, dict] = {}
    for doc_type, group in sorted(by_type.items()):
        legacy = [d for d in group if d["doc_id"] in LEGACY]
        rest = [d for d in group if d["doc_id"] not in LEGACY]
        extra = rest if a.all else spread(rest, max(a.per_type - len(legacy), 0))
        for d in legacy + extra:
            chosen[d["doc_id"]] = d

    rows = sorted(chosen.values(), key=lambda d: (d["doc_type"], d["chars"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n")

    print(f"{'doc_type':19} {'n':>3} {'min':>8} {'median':>8} {'max':>8}")
    for doc_type in sorted(by_type):
        c = sorted(d["chars"] for d in rows if d["doc_type"] == doc_type)
        print(f"{doc_type:19} {len(c):3} {c[0]:8} {c[len(c) // 2]:8} {c[-1]:8}")
    print(f"{len(rows)} docs, {sum(d['chars'] for d in rows):,} chars -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
