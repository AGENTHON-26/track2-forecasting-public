"""Shared by the summary-coverage tools: load the answer key and decide whether a text covers a point.

A key file (tools/summary_eval/key/<doc_id>.json) holds the important points of one document.
Each point:

    id, point, tier ("must" | "should"),
    kind ("number" | "name" | "phrase" | "concept" | "judge_only"),
    patterns     regexes, case-insensitive, `.` spans newlines
    match        "any" (default) | "all"   -- how `patterns` combine
    not_patterns a hit on any of these fails the point (wrong direction, wrong number)
    quote        exact sentence from the source the point comes from
    source_pattern  optional: what to look for in the SOURCE instead of `patterns`, for table
                 documents (CFTC) where the column label sits in a header row far from the number
    paraphrases  3 ways a summarizer might write it -- each must be covered
    decoys       near-misses (opposite direction, wrong number) -- none may be covered

`judge_only` points have no patterns and are reported, never scored by regex.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import text_signal as ts  # noqa: E402

EVAL_DIR = ROOT / "tools" / "summary_eval"
KEY_DIR = EVAL_DIR / "key"
MANIFEST = EVAL_DIR / "manifest.json"

_FLAGS = re.I | re.S
# Soft hyphens and zero-width spaces are PDF-extraction artifacts ("Al\xadthough") and are dropped.
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"} | {"’": "'", "‘": "'", "“": '"', "”": '"', "\xa0": " "}
                        | {c: None for c in "\xad\u200b\ufeff"})


def normalize(text: str) -> str:
    """Same normalization for sources, summaries and paraphrases: unicode dashes/quotes, whitespace."""
    return re.sub(r"\s+", " ", (text or "").translate(_DASHES)).strip()


def manifest() -> dict[str, dict]:
    return {d["doc_id"]: d for d in json.loads(MANIFEST.read_text())}


def source_text(doc: dict) -> str:
    """The document as the summarizer sees it (CFTC tables reduced to the main contract)."""
    body = (ROOT / "units" / doc["unit"] / "text" / doc["file"]).read_text(errors="replace")
    if doc["doc_type"] == "positioning_report":
        body = ts.main_contract_rows(body)
    return body


def load_keys() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(KEY_DIR.glob("*.json"))}


def scorable(point: dict) -> bool:
    return point.get("kind") != "judge_only" and bool(point.get("patterns"))


def bullets(text: str) -> list[str]:
    """A summary as its bullets. One bullet is one claim, and a claim must be judged on its own.

    Windows like `.{0,80}` otherwise run past the end of a bullet into the next one, which made
    `not_patterns` fire on an unrelated neighbouring claim ("downside risks to growth. - Inflation
    has remained elevated" tripped a `downside.{0,80}inflation` guard).
    """
    parts = [p for p in re.split(r"\n(?=\s*(?:[-*•]|\d+[.)]))", text or "") if p.strip()]
    return parts or [text or ""]


def covers(point: dict, text: str) -> bool:
    """Does `text` state this point? Some one bullet must satisfy it, with no `not_patterns` hit."""
    pats = point.get("patterns", [])
    if not pats:
        return False
    nots = point.get("not_patterns", [])
    need_all = point.get("match") == "all"
    for part in bullets(text):
        t = normalize(part)
        hits = [re.search(p, t, _FLAGS) is not None for p in pats]
        if (all(hits) if need_all else any(hits)) and not any(re.search(p, t, _FLAGS) for p in nots):
            return True
    return False


def dump_sources(out_dir: pathlib.Path) -> None:
    """Write each manifest doc, as the summarizer sees it, wrapped at spaces so it can be read in pages.

    Wrapping only turns spaces into newlines, so `normalize` gives the same text as the original.
    """
    import textwrap

    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, doc in manifest().items():
        lines = []
        for para in source_text(doc).splitlines():
            lines += textwrap.wrap(para, 400, break_long_words=False, break_on_hyphens=False) or [""]
        (out_dir / f"{doc_id}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    dump_sources(EVAL_DIR / "src")
    print(f"wrote {EVAL_DIR / 'src'}")
