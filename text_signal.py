"""
Track 2 — the TEXT half.  Owner: Nish.  Branch: feat/llm.

Rebuilt as a staged pipeline. This file is STAGE 1 only: one summary agent per document.

    summarize_corpus(text_dir) -> [{doc_id, timestamp, doc_type, ..., summary, error}, ...]

Each admissible document (from `corpus_index.json`, timestamp <= as-of) is sent WHOLE to the
model with a prompt written for its doc type — an FOMC statement, a Beige Book and a CFTC table
are structured differently, so they are asked different questions. The reply is the document's
main points, much shorter than the source. Short documents are passed through verbatim.

`read_text_signal()` returns exact neutral (the text-blind 1.0 baseline) until stage 2 turns the
summaries into adjustments. It does not run the summaries: they would spend the per-unit time
budget for no effect on the forecast.

Nothing here raises. A failed call records `error` on that document and the rest carry on.

Dev run:
    python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text
    python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text --prompt fomc_minutes
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

NEUTRAL = {"shift": 0.0, "widen": 1.0, "skew": 0.0}

#: Documents shorter than this are already small; summarizing them only loses detail.
_PASSTHROUGH_CHARS = 3_000
_WORKERS = 8
#: A ceiling, not a target: a normal reply is ~400 tokens. Set well above that so a reply that runs
#: long comes back complete rather than cut off mid-sentence.
_MAX_TOKENS = 4_000
#: Off: measured 4-5x slower on (23-30 s vs 6 s per doc) and it leaked its reasoning into the
#: reply once in 16 calls, looping until the cap. The small quality gain was not worth either.
_THINKING = False
#: Doc types that get thinking ON anyway. Landmarks mix two formats (policy decisions and speeches)
#: under one label, so the model has to work out which one it is reading. The reasoning is billed
#: against the same token cap, so these calls get a larger one.
_THINKING_TYPES = {"landmark"}
_MAX_TOKENS_THINKING = 16_000
#: A real 12-bullet minutes summary is ~3.5k chars; far past that is not a summary.
_MAX_SUMMARY_CHARS = 8_000
_TIMEOUT_SEC = 300.0
#: Seconds to wait before each retry of an overloaded (429 / 5xx) reply.
_RETRY_WAITS = (2.0, 5.0, 10.0)


# ----------------------------------------------------------------------------- the entry point
def read_text_signal(
    text_dir: pathlib.Path, assets: list[str]
) -> dict[str, dict[str, float]]:
    """Exact neutral until stage 2 exists."""
    return {a: dict(NEUTRAL) for a in assets}


# ----------------------------------------------------------------------------- the corpus
def load_corpus(text_dir: pathlib.Path) -> tuple[str, list[dict[str, Any]]]:
    """(as-of, admissible documents newest first, full text).

    Driven by `corpus_index.json`, never by a glob: a file with no index entry has no timestamp
    and cannot be shown to predate the as-of.
    """
    text_dir = pathlib.Path(text_dir)
    asof = _asof(text_dir)
    try:
        index = json.loads((text_dir / "corpus_index.json").read_text(encoding="utf-8"))
    except Exception:
        return asof, []
    docs: list[dict[str, Any]] = []
    for entry in index.get("documents", []):
        ts = str(entry.get("timestamp", ""))[:10]
        path = text_dir / str(entry.get("file", ""))
        if not ts or ts > asof or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        docs.append(
            {
                "doc_id": entry.get("doc_id") or path.stem,
                "timestamp": ts,
                "doc_type": str(entry.get("doc_type") or "unknown"),
                "source": entry.get("source", ""),
                "text": text,
            }
        )
    docs.sort(key=lambda d: d["timestamp"], reverse=True)
    return asof, docs


def _asof(text_dir: pathlib.Path) -> str:
    """The card's text cutoff, else the index's as-of. Unknown -> a date that admits nothing."""
    for card_path in (text_dir.parent / "card.toml", text_dir / "card.toml"):
        if card_path.is_file():
            with contextlib.suppress(Exception):
                card = tomllib.loads(card_path.read_text(encoding="utf-8"))
                cut = card.get("text", {}).get("cutoff") or card.get("provenance", {}).get(
                    "data_cutoff"
                )
                if cut:
                    return str(cut)[:10]
    with contextlib.suppress(Exception):
        idx = json.loads((text_dir / "corpus_index.json").read_text(encoding="utf-8"))
        if idx.get("asof"):
            return str(idx["asof"])[:10]
    return "0000-00-00"


# ----------------------------------------------------------------------------- the prompts
_FRAME = """You summarize one financial document for a macro forecaster.
The document is dated {date}. Use only what the document says; do not use any knowledge of
events after {date}.
Ignore website navigation, cookie notices, menus, footers, footnote markers and other boilerplate.
Keep numbers, dates, rates and quoted phrases exactly as written, and keep every number's unit
exactly as written (per share, million, billion, percent, basis points) -- never change a unit.
Only state what the document says. Never add topics, names, causes or numbers it does not contain,
and do not write bullets about what the document does NOT say.
Output bullet points only (lines starting with "- "), one sentence each, at most {bullets} bullets.
First cover EVERY item in the checklist below, in order, each in its own bullet; skip an item only
if the document has nothing on it. Then use any remaining bullets for other important points.
Condense, do not copy paragraphs.
If the document has no monetary-policy or market-relevant content, reply with the single line:
- no monetary-policy or market-relevant content

This document is {kind}. Checklist:
{focus}"""

_FOCUS: dict[str, tuple[str, str]] = {
    "fomc_statement": (
        "an FOMC policy statement",
        "- the decision and the federal funds target range\n"
        "- the vote with every voter named, and any dissents with the dissenter's preferred action\n"
        "- forward-guidance wording about future policy, quoted exactly\n"
        "- balance sheet / asset purchase decisions\n"
        "- the Committee's assessment of inflation, employment and the balance of risks",
    ),
    "fomc_minutes": (
        "FOMC meeting minutes",
        "- participants' views on the future policy path, KEEPING the quantifiers exactly\n"
        "  (all / most / many / several / some / a few / a couple)\n"
        "- the assessment of inflation and of the labor market\n"
        "- risks participants flagged and in which direction\n"
        "- balance sheet discussion\n"
        "- the staff economic outlook\n"
        "- any visible disagreement or dissent",
    ),
    "cb_speech": (
        "a central banker's speech",
        "- speaker, institution and role (first bullet)\n"
        "- the speaker's stance on the policy path (tighter / easier / on hold) and why\n"
        "- views on inflation, labor market and growth\n"
        "- any explicit hint about the next policy moves, quoted exactly\n"
        "- if the speech is mainly not about monetary policy, say so in one bullet and keep only\n"
        "  the policy-relevant points",
    ),
    "landmark": (
        "a landmark policy communication (testimony, key speech or announcement)",
        "- who, where and when (first bullet)\n"
        "- the single most important policy signal or commitment, quoted exactly\n"
        "- the conditions attached to it\n"
        "- the economic assessment behind it",
    ),
    "beige_book": (
        "a Federal Reserve Beige Book (a web page scrape, much of it boilerplate)",
        "- overall national economic activity and its direction\n"
        "- employment and wages\n"
        "- prices and input costs\n"
        "- the outlook and the sources of uncertainty contacts reported\n"
        "- notable divergences between districts or sectors",
    ),
    "macro_release": (
        "an official macroeconomic data release (a web page scrape with tables)",
        "- the release name and the reference period\n"
        "- the headline figure, month-over-month and year-over-year\n"
        "- core / ex-food-and-energy or equivalent figures\n"
        "- the change versus the prior period and any revisions\n"
        "- the components that drove the change",
    ),
    "positioning_report": (
        "a CFTC Commitments of Traders positioning table",
        "- the table has one row per date: the main contract (the largest by open interest)\n"
        "- the market and the latest report date\n"
        "- the latest non-commercial net position and open interest\n"
        "- the change versus the prior few weeks\n"
        "- the trend and the extremes (largest net long / short) over the window",
    ),
    "corporate_8k": (
        "a corporate 8-K filing / earnings release",
        "- the company and the reporting period\n"
        "- headline results (revenue, EPS) versus the prior period\n"
        "- guidance changes (raised / lowered / reaffirmed) with the numbers\n"
        "- forward-looking remarks and the one or two most material events",
    ),
    "default": (
        "a financial or economic document",
        "- who published it and what it is about\n"
        "- the main facts and figures\n"
        "- anything that bears on interest rates, inflation, growth or markets",
    ),
}


#: Room for the whole checklist plus the document's other main points. At 8 the model filled the
#: slots with detail and dropped checklist items (tariffs, core CPI y/y, Powell's key quote).
_BULLETS = 15


#: Repeated AFTER the document. On a 50k+ char input the system prompt is far behind the model and
#: the length rule got ignored (34 bullets on a 95k-char minutes); restated last, it is followed.
_REMINDER = (
    "\n\n=== END OF DOCUMENT ===\n"
    "Now write the summary: every checklist item first, in order, then other important points; "
    "at most {bullets} bullets, one sentence each. "
    "Only what the document says; no bullets about what it does not say."
)


def prompt_for(doc_type: str, date: str) -> str:
    kind, focus = _FOCUS.get(doc_type, _FOCUS["default"])
    bullets = _BULLETS
    return _FRAME.format(date=date, kind=kind, focus=focus, bullets=bullets)


# ----------------------------------------------------------------------------- per-type cleanup
_COT_ROW = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s+(\d+)\s")


def main_contract_rows(text: str) -> str:
    """CFTC tables: keep only the largest-open-interest row per date.

    Four of the six shipped COT files list 2-5 unlabeled contracts per date (a crude file mixes
    a 1.4M-open-interest row with 150k ones). Told to pick the right row, the model still took
    extremes from the small contracts and reported a net-short swing that never happened in the
    main one. Selecting rows is arithmetic, so the code does it. Unparseable text is returned as is.
    """
    best: dict[str, tuple[int, str]] = {}
    other: list[str] = []
    for line in text.splitlines():
        m = _COT_ROW.match(line)
        if m is None:
            other.append(line)
        elif int(m.group(2)) > best.get(m.group(1), (-1, ""))[0]:
            best[m.group(1)] = (int(m.group(2)), line)
    if not best:
        return text
    return "\n".join(other + [best[d][1] for d in sorted(best)])


_CLEANUP = {"positioning_report": main_contract_rows}


# ----------------------------------------------------------------------------- the agents
def summarize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """One document -> its main points. Never raises."""
    meta = {k: v for k, v in doc.items() if k != "text"}
    text = doc.get("text", "")
    out = {**meta, "full_chars": len(text), "summary": None, "summarized": False, "error": ""}
    try:
        text = _CLEANUP.get(doc["doc_type"], lambda t: t)(text)
    except Exception:
        pass
    if len(text) < _PASSTHROUGH_CHARS:
        out["summary"] = text.strip()
    else:
        summary, err = None, ""
        # Two attempts: once in ~50 calls the model returned a bare "-" and nothing else.
        for _ in range(2):
            try:
                user = text + _REMINDER.format(bullets=_BULLETS)
                summary, err = call_model(
                    prompt_for(doc["doc_type"], doc["timestamp"]),
                    user,
                    thinking=_THINKING or doc["doc_type"] in _THINKING_TYPES,
                )
            except Exception as exc:
                summary, err = None, f"{type(exc).__name__}: {exc}"
            if summary is not None and len(re.sub(r"[\W_]", "", summary)) < 5:
                summary, err = None, f"reply had no content: {summary!r}"
                continue
            if summary is not None and len(summary) > _MAX_SUMMARY_CHARS:
                # With thinking on, the reasoning can leak into `content` untagged and loop until the
                # token cap (seen: 29,755 chars of "Also 2000-2008. Also 1971-1979. ...").
                summary, err = None, f"reply too long ({len(summary)} chars): reasoning leaked"
                continue
            break
        out["summary"], out["error"], out["summarized"] = summary, err, summary is not None
    out["summary_chars"] = len(out["summary"] or "")
    return out


def summarize_corpus(text_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Every admissible document summarized, newest first. Runs the calls in parallel."""
    try:
        _, docs = load_corpus(text_dir)
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            return list(pool.map(summarize_doc, docs))
    except Exception as exc:
        print(f"[text_signal] summarize_corpus failed: {exc}", file=sys.stderr)
        return []


def call_model(system: str, user: str, thinking: bool = False) -> tuple[str | None, str]:
    """(reply text, error). POST $MODEL_ENDPOINT/v1/chat/completions with stdlib urllib only.

    The scoring image has no `openai` package. MODEL_TOKEN is the House grant, MODEL_API_KEY
    the local-dev key. The thinking switch is sent explicitly on every request.
    """
    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip()
    if not endpoint:
        return None, "MODEL_ENDPOINT is unset"
    model = os.environ.get("MODEL_NAME", "").strip() or "nvidia/nemotron-3-super-120b-a12b"
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": _MAX_TOKENS_THINKING if thinking else _MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
    ).encode("utf-8")
    base = endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"  # the House route admits only /v1/chat/completions
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    token = os.environ.get("MODEL_TOKEN", "").strip() or os.environ.get("MODEL_API_KEY", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    payload = None
    for wait in (*_RETRY_WAITS, None):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = ""
            with contextlib.suppress(Exception):
                detail = exc.read().decode("utf-8", "replace")[:300]
            if wait is None or not (exc.code == 429 or exc.code >= 500):
                return None, f"HTTP {exc.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        time.sleep(wait)
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, "reply had no choices[0].message.content"
    if not isinstance(content, str) or not content.strip():
        return None, "empty reply"
    # The reasoning normally arrives in `reasoning_content`; some servers inline it instead.
    content = re.sub(r"(?s)^.*</think>", "", content).strip()
    if not content:
        return None, "reply had only reasoning, no summary"
    return content, ""


# ----------------------------------------------------------------------------- dev CLI
def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="text_signal", description="Stage 1: per-document summaries.")
    p.add_argument("--text", type=pathlib.Path, help="units/<id>/text")
    p.add_argument("--prompt", metavar="DOC_TYPE", help="print that doc type's prompt and exit")
    p.add_argument("--json", action="store_true", help="print the summaries as JSON")
    a = p.parse_args(argv)

    if a.prompt:
        print(prompt_for(a.prompt, "YYYY-MM-DD"))
        return 0
    if a.text is None:
        p.error("--text is required")

    results = summarize_corpus(a.text)
    if a.json:
        print(json.dumps(results, indent=2))
        return 0
    total_in = sum(r["full_chars"] for r in results)
    total_out = sum(r["summary_chars"] for r in results)
    for r in results:
        tag = "summary" if r["summarized"] else ("ERROR" if r["error"] else "verbatim")
        print(f"=== {r['timestamp']} {r['doc_type']} {r['doc_id']}  "
              f"{r['full_chars']} -> {r['summary_chars']} chars ({tag})")
        print(r["summary"] if r["summary"] is not None else f"  {r['error']}")
        print()
    print(f"{len(results)} docs, {total_in} -> {total_out} chars")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
