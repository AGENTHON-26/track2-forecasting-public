"""Key-free structural probes for FOMC minutes, for testing a prompt on documents the key never saw.

The answer key is written per document, so it cannot say anything about held-out documents. These
probes instead derive what to look for FROM EACH SOURCE automatically — the rate level the minutes
state, the surnames in the vote paragraph, the figures the minutes attach to PCE and to the
unemployment rate — and then ask whether the summary carries them. Nothing here is hand-written per
document, so a prompt cannot be tuned to it the way it could be tuned to a key.

Weaker than the key (it checks presence, not meaning) but unbiased between two prompts, which is
what a generalization test needs.

    python3 tools/probe_minutes.py out/hold_v1_r1.json out/hold_v3_r1.json
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import text_signal as ts  # noqa: E402
from summary_key import normalize  # noqa: E402

_NUM = re.compile(r"\d[\d,]*\.?\d*")
_RANGE = re.compile(r"target range for the federal funds rate (?:at|to) ([^.,;]{3,40})", re.I)
_VOTERS = re.compile(r"Voting against(?: this action)?:?\s*([^.]{0,200})", re.I)
_NAME = re.compile(r"\b(?:Ms\.|Mr\.|Messrs\.|Mses\.)?\s*([A-Z][a-z]{3,15})\b")
_STOP = {"Voting", "Against", "This", "Action", "Committee", "Chair", "Vice", "President", "Governor"}


def _near(text: str, anchor: str, window: int = 220) -> set[str]:
    """Figures the document states within `window` characters of an anchor phrase."""
    out: set[str] = set()
    for m in re.finditer(anchor, text, re.I):
        seg = text[m.end(): m.end() + window]
        out |= {n for n in _NUM.findall(seg) if "." in n and len(n) <= 5}
    return out


def expectations(doc: dict) -> dict:
    """What THIS document says, found automatically — the yardstick for its summary."""
    body = normalize(doc["text"])
    rng = _RANGE.search(body)
    dissent = _VOTERS.search(body)
    names = set()
    if dissent:
        names = {n for n in _NAME.findall(dissent.group(1)) if n not in _STOP}
    return {
        "range": rng.group(1).strip() if rng else "",
        "dissenters": names,
        "pce": _near(body, r"PCE price index|personal consumption expenditures price index"),
        "urate": _near(body, r"unemployment rate"),
        "body": body,
    }


def probe(summary: str, exp: dict) -> dict:
    s = normalize(summary or "")
    got = {}
    got["rate level"] = bool(exp["range"]) and exp["range"].lower()[:18] in s.lower()
    got["dissenter named"] = bool(exp["dissenters"]) and any(n in s for n in exp["dissenters"])
    got["PCE figure"] = bool(exp["pce"]) and any(n in s for n in exp["pce"])
    got["unemployment figure"] = bool(exp["urate"]) and any(n in s for n in exp["urate"])
    applicable = {
        "rate level": bool(exp["range"]), "dissenter named": bool(exp["dissenters"]),
        "PCE figure": bool(exp["pce"]), "unemployment figure": bool(exp["urate"]),
    }
    src_nums = {n.replace(",", "") for n in _NUM.findall(exp["body"])}
    bad = [n for n in {m.rstrip(".").replace(",", "") for m in _NUM.findall(s)}
           if len(n.replace(".", "")) > 2 and n not in src_nums]
    return {"got": got, "applicable": applicable, "unsupported": bad}


def main(paths: list[str]) -> int:
    corpus: dict[str, dict] = {}
    for text_dir in sorted(ROOT.glob("units/*/text")):
        for d in ts.load_corpus(text_dir)[1]:
            corpus.setdefault(d["doc_id"], d)

    print(f"{'run':28}{'docs':>6}{'failed':>8}{'rate':>10}{'dissent':>10}{'PCE':>10}{'u-rate':>10}{'bad nums':>10}")
    for path in paths:
        rows = json.loads(pathlib.Path(path).read_text())
        tally = {k: [0, 0] for k in ("rate level", "dissenter named", "PCE figure", "unemployment figure")}
        failed = bad_total = 0
        for r in rows:
            if not r.get("summary"):
                failed += 1
                continue
            res = probe(r["summary"], expectations(corpus[r["doc_id"]]))
            for k, ok in res["got"].items():
                if res["applicable"][k]:
                    tally[k][1] += 1
                    tally[k][0] += ok
            bad_total += len(res["unsupported"])
        def pct(k: str) -> str:
            got, n = tally[k]
            return f"{100 * got / n:.0f}% ({n})" if n else "-"
        print(f"{pathlib.Path(path).stem:28}{len(rows):>6}{failed:>8}"
              f"{pct('rate level'):>10}{pct('dissenter named'):>10}{pct('PCE figure'):>10}"
              f"{pct('unemployment figure'):>10}{bad_total:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
