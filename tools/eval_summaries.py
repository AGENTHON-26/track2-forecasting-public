"""Evaluate repeated runs of tools/try_summaries.py on the 16 test documents.

For every document: two hand-picked key facts that must appear, a check that every number in the
summary exists in the source (or is a correct difference of two source numbers), bullet count and
time — reported per run so run-to-run variance is visible.

    python3 tools/eval_summaries.py out/v8r1 out/v8r2 out/v8r3
    (each prefix has <prefix>_median.json and <prefix>_largest.json)
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import text_signal as ts  # noqa: E402

_DISTRICTS = (r"Atlanta|Philadelphia|Chicago|Boston|New York|Cleveland|Richmond|Dallas|"
              r"Kansas City|Minneapolis|St\. Louis|San Francisco")

#: doc_id -> [(label, regex that must match | callable returning True when OK)]
FACTS: dict[str, list[tuple[str, object]]] = {
    "beige_book-2022-07-31": [("names districts", _DISTRICTS), ("recession risk", r"recession")],
    "beige_book-2024-11-30": [("tariffs", r"tariff"), ("names districts", _DISTRICTS)],
    "bis_bernanke_2008-07-15": [("payrolls -94,000", r"94,000"), ("unemployment 5-1/2", r"5-1/2")],
    "bis_shirakawa_2012-11-12": [("1 percent goal", r"1 percent"), ("+11 trillion yen", r"11 trillion")],
    "moderna_8k_2020-10-29_d134701dex991": [("revenue 157.9m", r"157\.9"), ("1.1bn deposits", r"1\.1 billion")],
    "pfizer_8k_2020-10-27_pfe-09272020xex99": [
        ("revenue guidance 48.8", r"48\.8"),
        ("EPS not in 'billion'", lambda s: not re.search(r"EPS[^.;]*\$\d\.\d\d[^.;]*billion", s)),
    ],
    "fomc-minutes-20220504": [("50bp hike", r"50 basis point"), ("runoff June 1", r"June 1")],
    "fomc-minutes-20150128": [("'patient' guidance", r"patient"), ("Lacker dissent", r"Lacker")],
    "fomc-statement-2024-07-31": [("5-1/4 to 5-1/2", r"5-1/4 to 5-1/2"), ("'greater confidence'", r"greater confidence")],
    "fomc-statement-2014-09-17": [
        ("Fisher+Plosser dissent", lambda s: "Fisher" in s and "Plosser" in s),
        ("'considerable time'", r"considerable time"),
    ],
    "powell_jackson_hole_2022": [("'sufficiently restrictive'", r"sufficiently restrictive"), ("2.25 to 2.5", r"2\.25 to 2\.5")],
    "boe_mpc_statement_20220922": [("5-3-1 split", r"three members|five members|5[–-]3[–-]1"), ("Bank Rate 2.25%", r"2\.25\s?%")],
    "macro_release-2022-05-11": [("headline 8.3 y/y", r"8\.3 percent"), ("core 6.2 y/y", r"6\.2 percent")],
    "macro_release-2015-02-26": [("headline -0.7 m/m", r"0\.7 percent"), ("core 1.6 y/y", r"1\.6 percent")],
    "cftc_cot_japanese_2007": [
        ("latest -126,773", r"-126,?773"),
        ("883 direction right", lambda s: not re.search(r"(decreas|reduc|shrank|narrow|fell)[^.\n]*883", s, re.I)),
    ],
    "cftc_cot_crude_2014": [
        ("latest 253,001", r"253,?001"),
        ("shorts 165,287->147,620 not 'rose'", lambda s: not re.search(r"(rose|increas)[^.\n]*165,?287[^.\n]*147,?620", s, re.I)),
    ],
}

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


def main(prefixes: list[str]) -> int:
    runs = []
    for p in prefixes:
        docs = {}
        for part in ("median", "largest"):
            for r in json.loads(pathlib.Path(f"{p}_{part}.json").read_text()):
                docs[r["doc_id"]] = r
        runs.append(docs)

    n_ok = n_all = 0
    print(f"{'doc_type':19} {'doc':26} {'fact':34}" + "".join(f" r{i + 1}" for i in range(len(runs))))
    for doc_id, facts in FACTS.items():
        r0 = runs[0].get(doc_id)
        if r0 is None:
            continue
        for label, rule in facts:
            cells = []
            for run in runs:
                s = (run.get(doc_id) or {}).get("summary") or ""
                ok = rule(s) if callable(rule) else bool(re.search(rule, s, re.I))
                n_ok += ok
                n_all += 1
                cells.append(" ✅" if ok else " ❌")
            print(f"{r0['doc_type']:19} {doc_id[:26]:26} {label:34}" + "".join(cells))
    print(f"\nkey facts present: {n_ok}/{n_all}")

    print(f"\n{'doc_type':19} {'doc':26} {'bullets':>12} {'chars':>17} {'seconds':>17}  unsupported numbers")
    for doc_id in FACTS:
        rs = [run[doc_id] for run in runs if doc_id in run]
        if not rs:
            continue
        b = [(x["summary"] or "").count("\n-") + 1 if x["summary"] else 0 for x in rs]
        c = [x["summary_chars"] for x in rs]
        t = [x["seconds"] for x in rs]
        bad = sorted({n for x in rs for n in _unsupported_numbers(x)})
        err = [x["error"] for x in rs if x["error"]]
        print(f"{rs[0]['doc_type']:19} {doc_id[:26]:26} {'/'.join(map(str, b)):>12} "
              f"{'/'.join(map(str, c)):>17} {'/'.join(f'{v:.0f}' for v in t):>17}  "
              f"{', '.join(bad) or '-'}{'  ERR ' + err[0][:40] if err else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
