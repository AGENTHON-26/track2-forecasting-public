"""Check the answer key itself before trusting it to grade summaries (free, no model calls).

Per point: patterns compile; the quote is in the source; the patterns match the source; all
paraphrases are covered (not too strict); no decoy is covered (not too loose). Warnings for
patterns that fire everywhere in the source and for points another doc's paraphrases also cover.

    python3 tools/validate_summary_key.py                 # every key file
    python3 tools/validate_summary_key.py DOC_ID [...]    # just these
Exit code 1 if any point has an error.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from summary_key import covers, load_keys, manifest, normalize, scorable, source_text

TIERS = {"must", "should"}
KINDS = {"number", "name", "phrase", "concept", "judge_only"}
GENERIC = 25  # a pattern matching the source more often than this is probably not specific to the point


def check_doc(key: dict, doc: dict, others: list[dict]) -> tuple[list[str], list[str]]:
    errors, warns = [], []
    src = normalize(source_text(doc))
    src_low = src.lower()
    points = key.get("points", [])
    tiers = Counter(p.get("tier") for p in points)
    if not 5 <= len(points) <= 10:
        warns.append(f"{len(points)} points (want 5-10)")
    if not 3 <= tiers["must"] <= 6:
        warns.append(f"{tiers['must']} must points (want 3-6)")

    for p in points:
        pid = f"{p.get('id', '?')}"
        e = lambda msg: errors.append(f"{pid}: {msg}")  # noqa: E731
        if p.get("tier") not in TIERS:
            e(f"tier {p.get('tier')!r}")
        if p.get("kind") not in KINDS:
            e(f"kind {p.get('kind')!r}")
        if not p.get("point"):
            e("empty point")
        quote = normalize(p.get("quote", ""))
        if not quote or quote.lower() not in src_low:
            e(f"quote not in source: {quote[:70]!r}")
        if not scorable(p):
            continue
        bad = False
        for rx in p.get("patterns", []) + p.get("not_patterns", []):
            try:
                re.compile(rx)
            except re.error as exc:
                e(f"bad regex {rx!r}: {exc}")
                bad = True
        if bad:
            continue
        src_pats = p.get("source_pattern") or p["patterns"]
        if not covers({"patterns": src_pats, "match": p.get("match") if not p.get("source_pattern") else "any"}, src):
            e("patterns do not match the source")
        for rx in src_pats:
            n = len(re.findall(rx, src, re.I | re.S))
            if n > GENERIC:
                warns.append(f"{pid}: pattern {rx!r} matches the source {n}x (too generic?)")
        # A number pattern must not match a LONGER number: "5 billion" inside "15 billion",
        # "3.6 percent" inside "13.6 percent". Tested by growing each number in a paraphrase.
        for s0 in p.get("paraphrases", [])[:1]:
            for m in set(re.findall(r"\d[\d,]*\.?\d*", s0)):
                if covers(p, s0.replace(m, "1" + m)) and covers(p, s0):
                    warns.append(f"{pid}: still matches when {m} becomes 1{m} (add a (?<!\\d) guard)")
        paras, decoys = p.get("paraphrases", []), p.get("decoys", [])
        if len(paras) < 3:
            e(f"{len(paras)} paraphrases (need 3)")
        if len(decoys) < 2:
            e(f"{len(decoys)} decoys (need 2)")
        for s in paras:
            if not covers(p, s):
                e(f"too strict, misses paraphrase: {s[:70]!r}")
        for s in decoys:
            if covers(p, s):
                e(f"too loose, covers decoy: {s[:70]!r}")
        leaks = [o["doc_id"] for o in others
                 if any(covers(p, s) for q in o.get("points", []) for s in q.get("paraphrases", []))]
        if leaks:
            warns.append(f"{pid}: also covered by paraphrases of {', '.join(leaks[:3])}")
    return errors, warns


def main(only: list[str]) -> int:
    docs, keys = manifest(), load_keys()
    todo = only or sorted(keys)
    kinds, n_err = Counter(), 0
    for doc_id in todo:
        if doc_id not in keys:
            print(f"✗ {doc_id}: no key file")
            n_err += 1
            continue
        key, doc = keys[doc_id], docs.get(doc_id)
        if doc is None:
            print(f"✗ {doc_id}: not in manifest")
            n_err += 1
            continue
        others = [k for d, k in keys.items() if d != doc_id and docs.get(d, {}).get("doc_type") == doc["doc_type"]]
        errors, warns = check_doc(key, doc, others)
        kinds.update(p.get("kind") for p in key.get("points", []))
        n_err += len(errors)
        print(f"{'✗' if errors else '✓'} {doc_id}  ({len(key.get('points', []))} points)")
        for m in errors:
            print(f"    ERROR {m}")
        for m in warns:
            print(f"    warn  {m}")
    missing = sorted(set(docs) - set(keys))
    print(f"\n{len(todo)} key files, {n_err} errors; kinds: {dict(kinds)}")
    if not only and missing:
        print(f"{len(missing)} manifest docs have no key yet")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
