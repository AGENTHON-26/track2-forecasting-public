"""Does the free regex check agree with a careful reader? (the honesty check on the answer key)

`prepare` writes one task file per batch: each document's summary plus its key points, WITHOUT the
patterns, so the reader judges coverage on meaning alone. A Sonnet subagent labels each point
covered/missed into tools/summary_eval/audit/<doc_id>.json:

    {"doc_id": "...", "labels": {"p1": true, "p2": false, ...}, "notes": {"p2": "why"}}

`compare` then scores regex vs reader on the same run: agreement, false passes (regex says covered,
reader says missed -> pattern too loose) and false misses (pattern too strict), per kind.

    python3 tools/audit_key.py prepare out/key_r1.json --batches 4
    python3 tools/audit_key.py compare out/key_r1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from summary_key import EVAL_DIR, covers, load_keys, scorable  # noqa: E402

AUDIT_DIR = EVAL_DIR / "audit"
TASK_DIR = EVAL_DIR / "audit_tasks"


def _run(path: str) -> dict[str, dict]:
    return {r["doc_id"]: r for r in json.loads(pathlib.Path(path).read_text())}


def prepare(run_path: str, batches: int, only: list[str] | None = None) -> int:
    run, keys = _run(run_path), load_keys()
    docs = [d for d in keys if run.get(d, {}).get("summary") and (not only or d in only)]
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    for old in TASK_DIR.glob("batch_*.md"):
        old.unlink()
    per = -(-len(docs) // batches)
    for b in range(batches):
        chunk = docs[b * per:(b + 1) * per]
        if not chunk:
            continue
        out = ["# Coverage audit — judge by meaning, not wording\n",
               "For each point below, decide whether THE SUMMARY states it. Paraphrase counts;",
               "a number with different formatting counts; the wrong direction or wrong number does NOT.",
               "Write your answers to tools/summary_eval/audit/<doc_id>.json as",
               '`{"doc_id": ..., "labels": {"p1": true, ...}, "notes": {"p1": "short reason if unsure"}}`.\n']
        for doc_id in chunk:
            r = run[doc_id]
            out.append(f"\n## {doc_id}  ({r['doc_type']})\n\n### SUMMARY\n{r['summary']}\n\n### POINTS")
            for p in keys[doc_id]["points"]:
                if scorable(p):
                    out.append(f"- {p['id']} [{p['tier']}] {p['point']}")
        (TASK_DIR / f"batch_{b + 1}.md").write_text("\n".join(out) + "\n")
    print(f"{len(docs)} docs -> {TASK_DIR}/batch_*.md")
    return 0


def compare(run_path: str) -> int:
    run, keys = _run(run_path), load_keys()
    audits = {p.stem: json.loads(p.read_text()) for p in AUDIT_DIR.glob("*.json")} if AUDIT_DIR.is_dir() else {}
    if not audits:
        print(f"no audit files in {AUDIT_DIR} — run `prepare` and have a reader label them first")
        return 1
    per_kind: dict[str, Counter] = {}
    rows = []
    for doc_id, audit in sorted(audits.items()):
        summary = (run.get(doc_id) or {}).get("summary")
        if summary is None:
            continue
        for p in keys[doc_id]["points"]:
            if not scorable(p) or p["id"] not in audit.get("labels", {}):
                continue
            rx, human = covers(p, summary), bool(audit["labels"][p["id"]])
            c = per_kind.setdefault(p["kind"], Counter())
            c["n"] += 1
            c["agree"] += rx == human
            if rx and not human:
                c["false_pass"] += 1
                rows.append(("FALSE PASS ", doc_id, p, audit.get("notes", {}).get(p["id"], "")))
            if human and not rx:
                c["false_miss"] += 1
                rows.append(("FALSE MISS ", doc_id, p, audit.get("notes", {}).get(p["id"], "")))
    print(f"{'kind':12} {'points':>7} {'agreement':>11} {'too loose':>10} {'too strict':>11}")
    tot = Counter()
    for kind, c in sorted(per_kind.items()):
        tot.update(c)
        print(f"{kind:12} {c['n']:7} {100 * c['agree'] / c['n']:10.0f}% {c['false_pass']:10} {c['false_miss']:11}")
    if tot["n"]:
        print(f"{'ALL':12} {tot['n']:7} {100 * tot['agree'] / tot['n']:10.0f}% {tot['false_pass']:10} {tot['false_miss']:11}")
    for tag, doc_id, p, note in rows:
        print(f"\n{tag} {doc_id} {p['id']} [{p['kind']}] {p['point'][:90]}")
        if note:
            print(f"    reader: {note[:150]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "compare"])
    ap.add_argument("run")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="audit just these doc_ids (e.g. to repeat an earlier audit)")
    a = ap.parse_args()
    return prepare(a.run, a.batches, a.only) if a.action == "prepare" else compare(a.run)


if __name__ == "__main__":
    sys.exit(main())
