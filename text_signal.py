"""
Track 2 — the TEXT half.  Owner: Nish (stage 1) / Pun (stage 2).  Branch: feat/llm.

Staged pipeline. Stage 1 (Nish) summarizes each document; stage 2 (this addition) turns those
summaries into the `{shift, widen, skew}` adjustment `build_draws()` consumes:

    summarize_corpus(text_dir) -> [{doc_id, timestamp, doc_type, ..., summary, error}, ...]
    read_text_signal(text_dir, assets) -> {asset: {"shift": float, "widen": float, "skew": float}}

Stage 1, unchanged: each admissible document (from `corpus_index.json`, timestamp <= as-of) is
sent WHOLE to the model with a prompt written for its doc type. Short documents pass through
verbatim. See the stage-1 section below for detail.

Stage 2: the summaries (not the raw documents -- that budget was already spent) go into ONE more
model call, with a prompt that also depends on the card's family (F1-F4), since what a good
adjustment looks like differs by family (see docs/CATEGORIES.md): F1 wants small, justified
moves; F2 wants real commitment when guidance has shifted; F3 wants one coherent scenario applied
consistently across assets (the interface still can't express a target correlation directly --
see the F3 note below); F4 wants the tails to move even when the center barely does. The reply is
converted from model-friendly units (a move in standard deviations of the asset's own forecast)
to the contract's native-unit `shift`, using each asset's own historical daily volatility from its
panel -- so "+0.3" means the same thing on a yield, on JPY and on a factor return.

Nothing here raises. Any failure at any step -- no documents, no endpoint, a reply that won't
parse -- degrades to NEUTRAL (the text-blind 1.0 baseline), logged to stderr, never fatal to the
card (an unhandled exception here would cost 4.0; ignoring the text costs 1.0).

Dev run:
    python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text
    python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text --prompt fomc_minutes
    python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text --adjust
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import os
import pathlib
import re
import sys
import threading
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
#: Reverted 2026-09-23 (Pun) back to Nish's original off, after a real full-sweep measurement with
#: thinking forced on (see PUN_TEXT_NOTES.md, "throttle helps but doesn't fix it, thinking
#: underperforms") reconfirmed her 2026-09-20 finding at scale: 4-5x slower (23-30 s vs 6 s per
#: doc), it leaked its reasoning into the reply once in 16 calls (looping until the cap), and the
#: sweep's own composite scores came out worse on average with it on, worst of all in the family
#: (F2) that most needs a committed answer -- consistent with thinking making the model hedge.
_THINKING = False
#: Doc types that get thinking ON anyway. Landmark used to be here: its reasoning needed a 16,000
#: token budget, and the house rule is at most 4,000 output tokens per request, reasoning included.
#: At 4,000 with thinking on the model spent the whole budget reasoning and 4 of 8 landmark
#: documents came back with no summary at all (17-19k chars of cut-off reasoning in `content`).
_THINKING_TYPES: set[str] = set()
_MAX_TOKENS_THINKING = 4_000
#: A real summary is 2-7k chars; the decision-first minutes checklist reaches 9.7k on one doc. A
#: runaway reply (reasoning looping until the cap) was 29k. 12k chars is ~3k tokens, inside the
#: 4,000-token cap, so nothing legitimate is discarded and a loop still is.
_MAX_SUMMARY_CHARS = 12_000
_TIMEOUT_SEC = 300.0
#: Seconds to wait before each retry of an overloaded (429 / 5xx) reply.
_RETRY_WAITS = (2.0, 5.0, 10.0)

#: HTTP codes worth another attempt, on top of every 5xx.
#:
#: 429 is the obvious one. **404 is not, and is here because this endpoint emits it spuriously
#: under load.** Measured 2026-09-25, six identical probes three seconds apart against
#: $MODEL_ENDPOINT/chat/completions: 200, 503, 404, 200, 404, 200 -- the 404s carry an EMPTY body
#: and are followed by a 200 on the identical request, so they are infrastructure, not a missing
#: route. Because a 404 is normally permanent, the retry loop returned on the first one, and an
#: F3 sweep "completed" units in 1.6-4.0 s having made no successful call at all: every document
#: and the stage-2 adjustment silently fell back to neutral, which the scores would have reported
#: as a real text-blind result.
#:
#: If the House route ever 404s for a real reason (a wrong path, a retired model id), this costs
#: three extra attempts before the same failure is reported -- cheap next to silently forecasting
#: text-blind. Revisit if the endpoint stops doing this.
_RETRYABLE_CODES = frozenset({404, 429})

#: The House endpoint's shared quota, measured 2026-09-22: 40 requests/minute. Kept a margin
#: under it rather than 40 itself -- Stage 1's own worker pool can burst several requests within
#: the same second, and a 429 that exhausts _RETRY_WAITS degrades the whole card to NEUTRAL with
#: no visible failure (see PUN_TEXT_NOTES.md, "silent rate-limit fallback"). Local eval sweeps
#: should keep --concurrency at 1 for now -- this only coordinates calls within one process, not
#: across the separate subprocesses run_eval.py spawns per unit.
_RATE_LIMIT_RPM = 36
_rate_lock = threading.Lock()
_request_times: collections.deque[float] = collections.deque()


def _throttle() -> None:
    """Block until issuing another request keeps this process under _RATE_LIMIT_RPM in any
    trailing 60s window. One shared budget for every call_model() call in this process -- Stage
    1's worker pool and Stage 2's own call all draw from it, so they can't jointly overrun it."""
    with _rate_lock:
        while True:
            now = time.monotonic()
            while _request_times and now - _request_times[0] >= 60.0:
                _request_times.popleft()
            if len(_request_times) < _RATE_LIMIT_RPM:
                _request_times.append(now)
                return
            time.sleep(_request_times[0] + 60.0 - now)
#: House rule: 25 admitted requests per unit, charged on admission -- a retry, a failed call or a
#: lost response spends a slot too. Stage 1 spends up to one per document (plus retries); stage 2
#: needs exactly one, so one is held back for it. The busiest unit needs 16 with no retries.
_REQUEST_BUDGET = 25
_STAGE2_RESERVE = 1


class _Budget:
    """Admitted-request counter for one unit, shared by the summarizer's worker threads."""

    def __init__(self, total: int = _REQUEST_BUDGET, reserve: int = _STAGE2_RESERVE):
        self.left, self.reserve, self.spent = total, reserve, 0
        self._lock = threading.Lock()

    def take(self, reserved: bool = False) -> bool:
        """Claim one request. Stage 1 may not eat into the reserve; stage 2 (`reserved`) may."""
        with self._lock:
            if self.left - (0 if reserved else self.reserve) <= 0:
                return False
            self.left -= 1
            self.spent += 1
            return True

#: Stage-2 clamps, carried forward from the removed single-shot pipeline (git history before
#: "feat(text): stage 1" -- the clamps were never the reason it measured net negative; the
#: budget-truncated single-shot excerpts were the likely cause, and stage 1 already replaces
#: those with whole-document summaries).
_DRIFT_SD_CLAMP = 1.5
_WIDEN_CLAMP = (0.60, 2.00)
_SKEW_CLAMP = (-1.0, 1.0)

#: F3 pays 0.3 on the joint variogram, and that term is monotone increasing in the width of the
#: draws above 1.0: measured on the 20 F3 units, a global vol multiplier of 1.0 / 1.25 / 1.50 /
#: 2.00 gives joint 2.162 / 2.252 / 2.465 / 3.166, and the normalized composite goes 0.978 at
#: 1.00 to 1.056 at 1.30. The default ceiling of 2.00 therefore lets the model cost the card ~46%
#: on its primary term by widening, which on F3 is never the right answer to uncertainty -- a
#: wider joint distribution has wider gaps between every pair of cells, and gaps are what is
#: scored. Other families keep the wide clamp: F4 is single-cell (the joint weight is
#: redistributed away entirely) and is scored on exactly the tail that widening helps.
_WIDEN_CLAMP_BY_FAMILY = {"F3": (0.85, 1.25)}


def _widen_clamp(family: str | None) -> tuple[float, float]:
    """The vol_scale bounds for this family. Used for BOTH the prompt text and the enforcement,
    so the range the model is told and the range it is held to cannot drift apart."""
    return _WIDEN_CLAMP_BY_FAMILY.get(str(family), _WIDEN_CLAMP)

#: OFF since 2026-09-25. Decided on inherited evidence plus cost, NOT measured here -- recorded
#: that way on purpose so nobody later mistakes it for a settled result.
#:
#: What is actually known:
#:   - Stage 1's thinking WAS measured at scale on this same model and endpoint and came out
#:     worse: 4-5x slower, reasoning leaking into replies, and worse composites -- worst in the
#:     family that most needs a committed answer (`_THINKING` above, reverted 2026-09-23).
#:   - The older claim this flag was set to test: thinking made the 120B halve its committed
#:     adjustments. Hedging is the opposite of what the scoring rewards here -- on F3 the measured
#:     strongest text lever is the SIZE and spread of per-asset drift_sd (oracle 0.747 normalized
#:     composite), and both the F2 and F3 prompts now explicitly ask the model not to hedge.
#:   - It does NOT break stage 2: the 2026-09-22 thinking-on sweep had 3 stage-2 failures, all
#:     503/429 transport errors, and zero truncated or unparseable replies. So the 4,000-token cap
#:     accommodates the reasoning and the JSON together, unlike the landmark case in `_THINKING`.
#:
#: What is NOT known: whether it helps or hurts the score at stage 2 specifically. Deciding that
#: needs replicated sweeps per arm (run-to-run variance on this endpoint is the same order as the
#: effect -- see PUN_TEXT_NOTES.md), roughly 3-4 hours, and the downside of simply leaving it on
#: is only ~20 s/unit against a 1,800 s budget. That measurement lost to the prompt A/B on value.
#: Flip back to True and re-measure if a text A/B ever comes out strangely.
_STAGE2_THINKING = False


# ----------------------------------------------------------------------------- the entry point
def read_text_signal(
    text_dir: pathlib.Path, assets: list[str]
) -> dict[str, dict[str, float]]:
    """Stage 1 (summarize) + stage 2 (adjust). Never raises -- see the module docstring."""
    text_dir = pathlib.Path(text_dir)
    neutral = {a: dict(NEUTRAL) for a in assets}
    try:
        budget = _Budget()
        raw_docs = summarize_corpus(text_dir, budget)
        for d in raw_docs:
            if not d.get("summary") and d.get("error"):
                print(f"[text_signal] doc {d.get('doc_id', '?')} ({d.get('doc_type', '?')}) "
                      f"failed: {d['error']}", file=sys.stderr)
        summaries = [s for s in raw_docs if s.get("summary")]
        if not summaries:
            print("[text_signal] no usable summaries; neutral", file=sys.stderr)
            return neutral

        ctx = load_context(text_dir, assets)
        system, user = build_adjustment_prompt(summaries, assets, ctx)
        content, err = call_model(system, user, thinking=_STAGE2_THINKING, budget=budget, reserved=True)
        if content is None:
            print(f"[text_signal] adjustment call failed: {err}; neutral", file=sys.stderr)
            return neutral

        raw = _extract_json_object(content)
        if raw is None:
            print(f"[text_signal] adjustment reply did not parse: {content[:200]!r}; neutral",
                  file=sys.stderr)
            return neutral

        adjustments, ledger = to_adjustments(raw, assets, ctx)
        print(f"[text_signal] source=llm family={ctx['family']} docs={len(summaries)} "
              f"requests={budget.spent}/{_REQUEST_BUDGET} assets={list(adjustments)}", file=sys.stderr)

        # The ledger used to be computed here and dropped on the floor. Without it a sweep's
        # numbers cannot be explained after the fact -- you can see that a card scored worse but
        # not whether the model said anything, whether a clamp ate it, or whether an asset came
        # back missing and was silently neutralized. Half of why the three sweeps in
        # PUN_TEXT_NOTES.md were unreadable. One line per asset, stderr only, so run_eval.py
        # captures it per unit without changing any output contract.
        _log_ledger(ledger, assets, ctx)
        return adjustments
    except Exception as exc:  # never let the text half fail the card
        print(f"[text_signal] {type(exc).__name__}: {exc}; neutral", file=sys.stderr)
        return neutral


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
When you quote, quote only the words that carry the signal -- a phrase, never a whole paragraph.
Only state what the document says. Never add topics, names, causes or numbers it does not contain,
and do not write bullets about what the document does NOT say.
Output bullet points only (lines starting with "- "), one sentence each, at most {bullets} bullets.
First cover EVERY item in the checklist below, in order, each in its own bullet; skip an item only
if the document has nothing on it. Then use any remaining bullets for other important points --
use the full allowance when the document has that much to say; do not stop early.
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
        # The decision, the level, the names and the guidance wording come first: with the old
        # discussion-first checklist the model wrote 15 bullets of staff outlook and never said
        # what the Committee did (41% of must-facts vs 97% for statements, which ask decision-first).
        "- the policy decision taken at this meeting and the resulting federal funds target range\n"
        "- every dissent AND abstention, with the person NAMED and what they preferred, including\n"
        "  votes on directives, authorizations and resolutions, not only the rate vote\n"
        "- wording kept, changed or dropped versus the previous statement, quoted exactly\n"
        "- participants' views on the future policy path, KEEPING the quantifiers exactly\n"
        "  (all / most / many / several / some / a few / a couple)\n"
        "- the inflation figures cited: headline and core PCE 12-month rates\n"
        "- the labor figures cited: unemployment rate and the pace of payroll gains\n"
        "- risks participants flagged and in which direction, naming whatever specific country,\n"
        "  institution, event or policy the minutes give as the cause, never only a general phrase\n"
        "- balance sheet and money-market operations with their amounts, caps and rates\n"
        "- what market pricing implied about the policy path\n"
        "- the staff economic outlook",
    ),
    "cb_speech": (
        "a central banker's speech",
        # Tried 2026-09-24: asking for cited figures, dates and institutional commitments moved the
        # unseen test half 44 -> 58% but the dev half 76 -> 52% (net 59 -> 56 over all eight docs).
        # No measurable gain, so the original stays. Speeches remain the open problem.
        # Measured 2026-09-24: the body of a speech (the third quarter of the text) had 14% of its
        # facts covered against 60%+ for the opening and the close, and the old "say so and keep only
        # the policy-relevant points" exit collapsed non-policy speeches to ~900 chars (Dudley on
        # trade: 1 of 24 facts). Every section gets covered; non-policy speeches keep their economics.
        "- speaker, institution and role (first bullet)\n"
        "- the speaker's stance on the policy path (tighter / easier / on hold) and why\n"
        "- views on inflation, labor market and growth\n"
        "- any explicit hint about the next policy moves, quoted exactly\n"
        "- the figures and arguments from the BODY of the speech, section by section, not only\n"
        "  its opening and its conclusion\n"
        "- if the speech is mainly not about monetary policy, say so in one bullet, then still\n"
        "  cover its economic content: the mechanisms, figures and conclusions the speaker gives",
    ),
    "landmark": (
        "a landmark policy communication (testimony, key speech or announcement)",
        # Same spine as the statement checklist: with thinking off, the vote and the names went
        # missing (BoE 5-3-1 split 3/3 -> 1/3) because nothing asked for them.
        "- who, where and when (first bullet)\n"
        "- the decision or announcement and the resulting rate, programme or purchase level\n"
        "- the vote, with every dissent NAMED and the dissenter's preferred action\n"
        "- the single most important policy signal or commitment, quoted exactly\n"
        "- the conditions attached to it\n"
        "- the economic assessment behind it, with the figures cited",
    ),
    "beige_book": (
        "a Federal Reserve Beige Book (a web page scrape, much of it boilerplate)",
        "- overall national economic activity and its direction, with the count of Districts\n"
        "  reporting growth, no change or decline when the book gives one\n"
        "- employment and wages\n"
        "- prices and input costs\n"
        "- every figure the book cites (survey readings, inflation expectations, percentages of\n"
        "  contacts), each with its source District or survey\n"
        "- the outlook and the sources of uncertainty contacts reported\n"
        "- the Districts NAMED as diverging from the national picture, and how",
    ),
    "macro_release": (
        "an official macroeconomic data release (a web page scrape with tables)",
        "- the release name and the reference period\n"
        "- the headline figure, month-over-month and year-over-year\n"
        "- core / ex-food-and-energy or equivalent figures\n"
        # Tried 2026-09-24: a line for the comparisons the release makes ("smallest since March
        # 2021") cost 91 -> 86% on the test half and 100 -> 94% on dev. Reverted.
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
        # Tried 2026-09-24: adding management's reasons and capital actions cost 80 -> 72% must-recall
        # on both halves (more items competing for the same bullets). Reverted.
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

_COT_FULL = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s*$")


def cot_summary(text: str) -> str | None:
    """A CFTC positioning table as bullets, computed rather than asked for.

    The model got the LEVELS right and the DIRECTION wrong: a net short that grew from -125,890 to
    -126,773 came back as "a decrease of 883" in 3 of 3 runs, and a forecaster acting on that
    would lean the wrong way. Every fact here is arithmetic on the rows, so the code does it and
    the sign is always right. None when the table does not parse (the model then gets the text).
    """
    rows = [m.groups() for m in map(_COT_FULL.match, main_contract_rows(text).splitlines()) if m]
    if len(rows) < 2:
        return None
    rows = sorted((d, int(oi), int(lg), int(sh), int(net)) for d, oi, lg, sh, net in rows)
    market = next((ln.strip() for ln in text.splitlines() if ln.lower().startswith("market:")), "")
    d, oi, lg, sh, net = rows[-1]
    pd_, poi, _, _, pnet = rows[-2]

    def side(n: int) -> str:
        return "net short" if n < 0 else "net long"

    def move(a: int, b: int) -> str:
        if (a < 0) != (b < 0):
            return f"flipped from {side(a)} to {side(b)}"
        grew = abs(b) > abs(a)
        verb = ("deepened" if grew else "narrowed") if b < 0 else ("grew" if grew else "shrank")
        return f"{side(b)} {verb} by {abs(b - a):,} contracts"

    def gmove(a: int, b: int, what: str) -> str:
        return f"{what} {'rose' if b > a else 'fell' if b < a else 'was unchanged'} from {a:,} to {b:,}"

    _, _, plg, psh, _ = rows[-2]
    lo = min(rows, key=lambda r: r[4]); hi = max(rows, key=lambda r: r[4])
    deepest = f"the deepest net short of the window was {lo[4]:+,} ({lo[0]})" if lo[4] < 0 else ""
    largest = f"the largest net long of the window was {hi[4]:+,} ({hi[0]})" if hi[4] > 0 else ""
    extremes = "; ".join(x for x in (deepest, largest) if x) or \
        f"the net position ranged from {lo[4]:+,} ({lo[0]}) to {hi[4]:+,} ({hi[0]})"
    jumps = [(abs(b[4] - a[4]), a, b) for a, b in zip(rows, rows[1:])]
    jsz, ja, jb = max(jumps)
    ref = lo if net < 0 else hi  # the extreme on the current side, to describe the move since it
    out = [f"- CFTC Commitments of Traders, {market or 'positioning table'}",
           f"- Latest report {d}: non-commercial net position {net:+,} contracts ({side(net)}), gross long "
           f"{lg:,}, gross short {sh:,}, open interest {oi:,}.",
           f"- Week over week the {move(pnet, net)}, from {pnet:+,} on {pd_} to {net:+,} on {d}.",
           f"- Week over week {gmove(plg, lg, 'gross long')} and {gmove(psh, sh, 'gross short')}.",
           f"- Over the {len(rows)}-week window {extremes}.",
           f"- The largest one-week move was {jsz:,} contracts, from {ja[4]:+,} ({ja[0]}) to {jb[4]:+,} ({jb[0]}), "
           f"when the {move(ja[4], jb[4])}."]
    if ref[0] != d:
        out.append(f"- Since that {'deepest net short' if net < 0 else 'largest net long'} on {ref[0]} the "
                   f"{move(ref[4], net)} to the latest reading.")
    else:
        out.append(f"- The latest reading is the {'deepest net short' if net < 0 else 'largest net long'} of the window.")
    if net == (min(rows, key=lambda r: abs(r[4]))[4]) and lo[4] * hi[4] > 0:
        out.append(f"- The latest {side(net)} is the smallest of the window.")
    if len(rows) >= 5:
        d4, _, _, _, n4 = rows[-5]
        out.append(f"- Versus four weeks earlier ({d4}, {n4:+,}) the {move(n4, net)}.")
    shorts = sum(r[4] < 0 for r in rows)
    out.append(f"- Non-commercial traders were net short in {shorts} of {len(rows)} weeks and net "
               f"long in {len(rows) - shorts}." if 0 < shorts < len(rows) else
               f"- Non-commercial traders were {side(net)} in every one of the {len(rows)} weeks.")
    first = rows[0]
    out.append(f"- Since the start of the window ({first[0]}, {first[4]:+,}) the {move(first[4], net)}.")
    glo = min(rows, key=lambda r: r[2]); ghi = max(rows, key=lambda r: r[2])
    slo = min(rows, key=lambda r: r[3]); shi = max(rows, key=lambda r: r[3])
    out.append(f"- Gross long peaked at {ghi[2]:,} ({ghi[0]}) and bottomed at {glo[2]:,} ({glo[0]}); gross short "
               f"peaked at {shi[3]:,} ({shi[0]}) and bottomed at {slo[3]:,} ({slo[0]}).")
    olo = min(rows, key=lambda r: r[1]); ohi = max(rows, key=lambda r: r[1])
    out.append(f"- Open interest peaked at {ohi[1]:,} ({ohi[0]}) and bottomed at {olo[1]:,} ({olo[0]}); the latest "
               f"{oi:,} is {'up' if oi > poi else 'down'} {abs(oi - poi):,} from {poi:,} the week before.")
    return "\n".join(out)


# ----------------------------------------------------------------------------- the agents
def summarize_doc(doc: dict[str, Any], budget: _Budget | None = None) -> dict[str, Any]:
    """One document -> its main points. Never raises. `budget`: the unit's request counter."""
    meta = {k: v for k, v in doc.items() if k != "text"}
    text = doc.get("text", "")
    out = {**meta, "full_chars": len(text), "summary": None, "summarized": False, "error": ""}
    try:
        text = _CLEANUP.get(doc["doc_type"], lambda t: t)(text)
    except Exception:
        pass
    computed = None
    if doc.get("doc_type") == "positioning_report":
        with contextlib.suppress(Exception):
            computed = cot_summary(text)
    if computed is not None:
        out["summary"], out["summarized"], out["computed"] = computed, True, True
    elif len(text) < _PASSTHROUGH_CHARS:
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
                    budget=budget,
                )
            except Exception as exc:
                summary, err = None, f"{type(exc).__name__}: {exc}"
            if summary is not None and len(re.sub(r"[\W_]", "", summary)) < 5:
                summary, err = None, f"reply had no content: {summary!r}"
                continue
            if summary is not None and len(summary) > _MAX_SUMMARY_CHARS:
                # With thinking on, the reasoning can leak into `content` untagged and loop until the
                # token cap (seen: 29,755 chars of "Also 2000-2008. Also 1971-1979. ...").
                summary, err = None, f"reply too long ({len(summary)} chars): runaway reply"
                continue
            break
        out["summary"], out["error"], out["summarized"] = summary, err, summary is not None
    out["summary_chars"] = len(out["summary"] or "")
    return out


def summarize_corpus(text_dir: pathlib.Path, budget: _Budget | None = None) -> list[dict[str, Any]]:
    """Every admissible document summarized, newest first. Runs the calls in parallel.

    Newest first also decides who loses when the request budget runs out: the oldest documents.
    """
    budget = budget or _Budget()
    try:
        _, docs = load_corpus(text_dir)
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            return list(pool.map(lambda d: summarize_doc(d, budget), docs))
    except Exception as exc:
        print(f"[text_signal] summarize_corpus failed: {exc}", file=sys.stderr)
        return []


def call_model(
    system: str, user: str, thinking: bool = False,
    budget: _Budget | None = None, reserved: bool = False,
) -> tuple[str | None, str]:
    """(reply text, error). POST $MODEL_ENDPOINT/v1/chat/completions with stdlib urllib only.

    The scoring image has no `openai` package. MODEL_TOKEN is the House grant, MODEL_API_KEY
    the local-dev key. The thinking switch is sent explicitly on every request. Every attempt,
    retries included, claims one slot from `budget` first; none left means no request is sent.
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
        if budget is not None and not budget.take(reserved):
            return None, "request budget exhausted"
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = ""
            with contextlib.suppress(Exception):
                detail = exc.read().decode("utf-8", "replace")[:300]
            worth_retrying = exc.code in _RETRYABLE_CODES or exc.code >= 500
            if wait is None or not worth_retrying:
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


# ----------------------------------------------------------------------------- stage 2: context
#: What each target id IS, and -- for FX -- which way its quote runs. The model otherwise sees a
#: bare string like `NOK` or `HML` and has to infer both.
#:
#: The direction half is not cosmetic. The prompt already warns that "usd_per_eur RISES when the
#: dollar WEAKENS", but the card's own `value_unit` is a single field shared by every asset, and
#: on a multi-asset card it degenerates to prose like "H.10 native quote (GBP,EUR: USD-per-ccy;
#: CHF,JPY: ccy-per-USD)" -- so per-asset direction is genuinely not machine-readable from the
#: card. The H.10 convention is consistent across all 104 shipped cards and is stated once here.
_ASSET_NOTES: dict[str, str] = {
    # US Treasury constant-maturity yields, percent per annum. Higher = yields up = prices down.
    "UST_2Y": "US 2-year Treasury yield, % p.a. (policy-path sensitive)",
    "UST_5Y": "US 5-year Treasury yield, % p.a.",
    "UST_7Y": "US 7-year Treasury yield, % p.a.",
    "UST_10Y": "US 10-year Treasury yield, % p.a. (term-premium sensitive)",
    "UST_30Y": "US 30-year Treasury yield, % p.a. (term-premium sensitive)",
    # FX, H.10 native quotes. USD-per-ccy: RISES when the dollar WEAKENS.
    "EUR": "EUR/USD as USD per EUR -- RISES when the dollar weakens",
    "GBP": "GBP/USD as USD per GBP -- RISES when the dollar weakens",
    "AUD": "AUD/USD as USD per AUD -- RISES when the dollar weakens",
    "NZD": "NZD/USD as USD per NZD -- RISES when the dollar weakens",
    # ccy-per-USD: RISES when the dollar STRENGTHENS.
    "JPY": "USD/JPY as JPY per USD -- RISES when the dollar strengthens",
    "CHF": "USD/CHF as CHF per USD -- RISES when the dollar strengthens (funding/haven currency)",
    "CAD": "USD/CAD as CAD per USD -- RISES when the dollar strengthens (oil-sensitive)",
    "SEK": "USD/SEK as SEK per USD -- RISES when the dollar strengthens",
    "NOK": "USD/NOK as NOK per USD -- RISES when the dollar strengthens (oil-sensitive)",
    "DKK": "USD/DKK as DKK per USD -- RISES when the dollar strengthens (pegged to EUR)",
    "CNY": "USD/CNY as CNY per USD -- RISES when the dollar strengthens (managed)",
    "INR": "USD/INR as INR per USD -- RISES when the dollar strengthens",
    "BRL": "USD/BRL as BRL per USD -- RISES when the dollar strengthens",
    # Equity factor returns (cumulative log return over the horizon).
    "MKT": "US equity market excess return factor",
    "HML": "value-minus-growth equity factor return",
    "SMB": "small-minus-big equity factor return",
    "MOM": "momentum equity factor return",
    "QMJ": "quality-minus-junk equity factor return",
    # Macro releases.
    "CPI_ALL": "US CPI all-items index (1982-84=100)",
    "UNRATE": "US unemployment rate, percent (U-3)",
    "NFP": "US nonfarm payrolls, change in thousands of jobs",
}


def load_context(text_dir: pathlib.Path, assets: list[str]) -> dict[str, Any]:
    """as-of date, horizons, value unit, family (F1-F4) and a per-asset (level, sigma).

    Everything here degrades rather than fails: no card means the family and horizons fall back
    to a generic default; no panel means sigma falls back to a fraction of the level, and to 0.0
    if there is no level either (which forces `shift` to 0 downstream -- see `to_adjustments`).
    """
    unit = text_dir.parent
    ctx: dict[str, Any] = {
        "asof": _asof(text_dir),
        "horizons": [21],
        "value_unit": "unknown",
        "target_type": "level",
        "family": "default",
        "level": {},
        "sigma": {},
    }
    card_path = next((p for p in (unit / "card.toml", text_dir / "card.toml") if p.is_file()), None)
    if card_path is not None:
        with contextlib.suppress(Exception):
            card = tomllib.loads(card_path.read_text(encoding="utf-8"))
            tgt = card.get("targets", {})
            ctx["horizons"] = [int(h) for h in tgt.get("horizons", ctx["horizons"])]
            ctx["value_unit"] = str(tgt.get("value_unit", ctx["value_unit"]))
            ctx["target_type"] = str(tgt.get("target_type", ctx["target_type"]))
            category = str(card.get("metadata", {}).get("category", ""))  # e.g. "T2-F1"
            fam = category.rsplit("-", 1)[-1]
            if fam in _FAMILY_FOCUS:
                ctx["family"] = fam

    level, sd_daily, hist = _panel_stats(unit, assets, ctx["asof"])
    ctx["level"] = level
    ctx["cross"] = _cross_asset_stats(hist, assets, ctx["horizons"])
    h0 = max(1, min(ctx["horizons"]))
    for a in assets:
        if a in sd_daily:
            ctx["sigma"][a] = sd_daily[a] * math.sqrt(h0)
        elif a in level and level[a]:
            ctx["sigma"][a] = abs(level[a]) * 0.05  # no panel history: 5% of level, deliberately small
        else:
            ctx["sigma"][a] = 0.0  # no scale is knowable -> shift stays 0, widen still works
    ctx["sigma_horizon"] = h0
    return ctx


def _panel_stats(
    unit: pathlib.Path, assets: list[str], asof: str
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    """(last level, daily sd, date->value history) per asset from the unit's own panel parquet(s).

    The third return exists so `_cross_asset_stats` can align assets by date without re-reading
    every parquet; on a 10-asset card that matters.

    Empty on any error.

    Carried forward from the removed single-shot pipeline (git history before "feat(text): stage
    1") -- this idea is orthogonal to why that pipeline measured net negative. Converting to a
    per-asset standard-deviation scale is what lets `drift_sd` mean the same thing on a yield, on
    JPY and on a factor return, instead of asking the model to reason in each asset's own units.
    """
    level: dict[str, float] = {}
    sd: dict[str, float] = {}
    hist: dict[str, dict[str, float]] = {}
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except Exception:
        return level, sd, hist
    files = sorted(unit.glob("*.parquet")) + sorted(unit.glob("panels/*.parquet"))
    for path in files:
        try:
            d = pq.read_table(path).to_pydict()
        except Exception:
            continue
        acol = next((c for c in ("asset", "asset_id") if c in d), None)
        if acol is None or "date" not in d or "value" not in d:
            continue
        for a in assets:
            if a in level:
                continue
            rows = sorted(
                (str(dt)[:10], float(v))
                for dt, aa, v in zip(d["date"], d[acol], d["value"], strict=False)
                if str(aa) == a and str(dt)[:10] <= asof and v is not None
            )
            if len(rows) < 31:
                continue
            series = np.array([v for _, v in rows], dtype=float)
            level[a] = float(series[-1])
            hist[a] = {d: v for d, v in rows}
            diffs = np.diff(series[-260:])
            if diffs.size >= 30 and math.isfinite(float(diffs.std())):
                sd[a] = float(diffs.std())
    return level, sd, hist


def _cross_asset_stats(
    hist: dict[str, dict[str, float]], assets: list[str], horizons: list[int]
) -> dict[str, Any]:
    """Date-aligned correlations and trailing moves -- the cross-asset numbers Stage 2 never had.

    Until this existed the model was shown each asset's level and sigma and nothing else, then
    asked on F3 cards to make every asset move "consistently with each other". It had no way to
    know what consistent looks like for this particular set: no correlations, no relative moves,
    nothing pairwise. It was being asked for a joint view on marginal information.

    Aligned on the date intersection, matching how `forecast_agent.build_draws` estimates the
    covariance it will actually draw from -- so the correlations quoted here are the ones the
    sampler uses, not a different estimate the model would then be arguing against.

    Returns {} for a single-asset card (nothing to be cross about) or on any failure.
    """
    out: dict[str, Any] = {}
    if len(assets) < 2:
        return out
    try:
        import numpy as np

        common = sorted(set.intersection(*(set(hist[a]) for a in assets if a in hist)))
        if len(common) < 31:
            return out
        wide = np.array([[hist[a][d] for d in common] for a in assets], dtype=float)

        steps = np.diff(wide, axis=1)[:, -260:]
        if steps.shape[1] < 30:
            return out
        corr = np.corrcoef(steps)
        if np.all(np.isfinite(corr)):
            out["corr"] = {assets[i]: {assets[j]: float(corr[i, j]) for j in range(len(assets))}
                           for i in range(len(assets))}

        # Trailing realized move over each of the card's own horizons, in sigma units -- the
        # "what is already priced" the prompt asks the model to reason about and never supplied.
        moves: dict[str, dict[int, float]] = {}
        for i, a in enumerate(assets):
            sd_a = float(np.diff(wide[i])[-260:].std())
            if not (math.isfinite(sd_a) and sd_a > 0):
                continue
            per_h = {}
            for h in horizons:
                if wide.shape[1] > h:
                    per_h[int(h)] = float((wide[i, -1] - wide[i, -1 - h]) / (sd_a * math.sqrt(h)))
            if per_h:
                moves[a] = per_h
        if moves:
            out["trailing_sigma"] = moves
    except Exception:
        return {}
    return out


# ----------------------------------------------------------------------------- stage 2: prompts
#: One paragraph per family, spliced into the shared frame below. Weights and "what's primary"
#: are from docs/CATEGORIES.md, not invented here.
_FAMILY_FOCUS: dict[str, str] = {
    "F1": (
        "This is an F1 (continuation-with-context) card: marginal CRPS is the primary score "
        "(50% weight). The numeric history alone is usually sufficient here -- the question is "
        "whether these summaries add INCREMENTAL signal on top of it. Prefer small, well-"
        "justified moves; a large drift_sd needs a real change in guidance to back it up, not "
        "routine language repeated from the last release."
    ),
    "F2": (
        "This is an F2 (text-cued regime shift) card: marginal CRPS is primary (50%), tail "
        "penalty is the secondary differentiator (20%). The whole point of this family is that "
        "a regime change is signalled in text BEFORE the panel moves -- a model that only reads "
        "the numbers will miss it entirely. If the summaries show a real change in tone or "
        "guidance versus what the recent numbers imply, do not hedge toward zero: commit to a "
        "real drift_sd, and consider raising vol_scale too, since regime shifts create fat-"
        "tailed outcomes that a narrow distribution will miss."
    ),
    "F3": (
        "This is an F3 (cross-asset reasoning) card: the joint variogram is the primary score "
        "(30% weight). It compares the DIFFERENCES between your assets -- |asset_i - asset_j| for "
        "every pair -- against what actually happened. Two consequences, and they are the whole "
        "game here. First, a constant added to every asset is free: it changes no difference, so "
        "it changes nothing. Second, the numbers you give ARE a joint statement, because what is "
        "scored is the pattern of gaps between them -- which assets move more than which others, "
        "and in which direction relative to each other. Work out ONE coherent macro scenario from "
        "the summaries, then derive every asset's numbers from that same scenario. Before "
        "answering, check your own work: for each pair of assets, does drift_sd[i] - drift_sd[j] "
        "say what your scenario says about that pair? A set of numbers that is individually "
        "plausible but pairwise incoherent scores worse here than a smaller, consistent set."
    ),
    "F4": (
        "This is an F4 (tail/shock-from-text) card: the tail penalty is the primary score (20% "
        "weight). The recent numeric history may look calm -- that is exactly what this family "
        "tests. If the summaries foreshadow a shock (a surprise reading, an urgent tone, a "
        "warning of exceptional measures), widen vol_scale and use skew to point the "
        "distribution toward the side the shock would move prices, even if drift_sd itself stays "
        "modest -- the tail, not the center, is what this card is scored on."
    ),
    "default": (
        "Treat this like a general macro forecasting card: weigh the summaries for anything "
        "directional or that changes how confident you should be, and answer 0 / 1.0 / 0 for any "
        "asset they say nothing about."
    ),
}


def build_adjustment_prompt(
    summaries: list[dict[str, Any]], assets: list[str], ctx: dict[str, Any]
) -> tuple[str, str]:
    """(system, user). system carries the persistent rules and schema; user carries the case."""
    system = (
        "You are a macro forecaster adjusting a statistical forecast using document summaries. "
        f"As-of date: {ctx['asof']}. Nothing after this date is known to you; reason only from "
        "the summaries given, never from outside knowledge of what happened later.\n\n"
        f"{_FAMILY_FOCUS.get(ctx['family'], _FAMILY_FOCUS['default'])}\n\n"
        "For EACH asset give three numbers:\n"
        f"  drift_sd  : where the centre of the distribution should move, in standard deviations "
        f"of the forecast at the shortest horizon. Range [-{_DRIFT_SD_CLAMP}, {_DRIFT_SD_CLAMP}]; "
        "0 if the summaries say nothing directional for that asset.\n"
        f"  vol_scale : multiplier on that same standard deviation, range "
        f"[{_widen_clamp(ctx.get('family'))[0]}, {_widen_clamp(ctx.get('family'))[1]}]. "
        ">1 when the summaries show disagreement, "
        "two-sided risk or an unresolved decision; <1 only when they REMOVE uncertainty (an "
        "explicit commitment or a peg).\n"
        "  skew      : tail tilt in [-1, 1]. Positive = the upside tail is the fatter one.\n\n"
        "Quote conventions matter for direction: a unit like usd_per_eur RISES when the dollar "
        "WEAKENS, while jpy_per_usd RISES when the dollar STRENGTHENS. Check the value unit given "
        "below before choosing a sign, and answer 0 if the convention makes you unsure.\n\n"
        "The statistical forecast already assumes the summaries say nothing. Answering all zeros "
        "for every asset is identical to not reading them -- if a summary gives you something "
        "concrete to react to, react to it; the ranges above already bound how far. Remember "
        "that a decision already delivered is priced into the level given below -- what usually "
        "is NOT priced is what the summaries say about the path from here.\n\n"
        "All three numbers must be finite (no NaN or Infinity). Every asset listed below must "
        "appear as a key -- an asset you omit is scored as 'no view', which on a joint card is "
        "itself a claim about that asset relative to the others. Reply with JSON only, no prose, "
        "no markdown fence:\n"
        # Every asset, not assets[:2]: a 10-asset card used to be shown a 2-key skeleton, which
        # is exactly the shape of reply that then came back.
        '{"assets": {'
        + ", ".join(
            f'"{a}": {{"drift_sd": 0.0, "vol_scale": 1.0, "skew": 0.0, '
            '"evidence": "<=15 words"}'
            for a in assets
        )
        + "}}"
    )

    lines = [
        f"Target: {ctx['target_type']} of each asset, {ctx['horizons']} business days after the as-of.",
        f"Value unit: {ctx['value_unit']}",
        "",
        f"Assets. `level` is the value on the as-of date; `sigma` is the forecast's own standard "
        f"deviation at the shortest horizon ({ctx['sigma_horizon']} BD) -- the unit your answer "
        "is measured in.",
    ]
    for a in assets:
        lvl = ctx["level"].get(a)
        sig = ctx["sigma"].get(a, 0.0)
        head = (f"  {a}: level {lvl:.6f}, sigma {sig:.6f}" if lvl is not None
                else f"  {a}: level unknown, sigma {sig:.6f}")
        note = _ASSET_NOTES.get(a)
        lines.append(f"{head}  -- {note}" if note else head)

    # Cross-asset numbers, on multi-asset cards. Absent before: the model was asked for a joint
    # view while being shown only per-asset marginals.
    cross = ctx.get("cross") or {}
    if cross.get("trailing_sigma"):
        lines += ["", "Trailing move already realized, in each asset's own sigma at that horizon "
                      "(what is arguably already priced):"]
        for a in assets:
            per_h = cross["trailing_sigma"].get(a)
            if per_h:
                moves = ", ".join(f"{h} BD {v:+.2f}s" for h, v in sorted(per_h.items()))
                lines.append(f"  {a}: {moves}")
    if cross.get("corr"):
        lines += ["", "How these assets have actually moved together (correlation of daily "
                      "changes, last 260 business days). This is the correlation the forecast "
                      "itself uses, so a pair listed near +1 will move together in every draw "
                      "whatever you answer -- your numbers say how far each one moves, and a "
                      "scenario that BREAKS a usual relationship has to say so through the gap "
                      "between their drift_sd values:"]
        for i, a in enumerate(assets):
            row = cross["corr"].get(a, {})
            cells = ", ".join(f"{b} {row.get(b, float('nan')):+.2f}" for b in assets[:i])
            if cells:
                lines.append(f"  {a} vs {cells}")

    lines += ["", f"Document summaries ({len(summaries)}), newest first:"]
    for s in summaries:
        lines += [f"--- {s['doc_id']} | {s['timestamp']} | {s['doc_type']} ---", s["summary"], ""]
    return system, "\n".join(lines)


def _log_ledger(ledger: dict[str, Any], assets: list[str], ctx: dict[str, Any]) -> None:
    """One `[text_signal] adj` line per asset: what the model said and what survived.

    Never raises -- a logging failure must not cost a card its text signal.
    """
    try:
        # An omitted asset still gets a ledger row -- neutral values plus a note saying so --
        # so detect it by the note, not by a missing key.
        missing = [a for a in assets
                   if "no entry for this asset" in str(
                       (ledger.get(a) or {}).get("note", "") if isinstance(ledger.get(a), dict) else "")]
        for a in assets:
            row = ledger.get(a)
            if not isinstance(row, dict):
                continue
            drift = row.get("drift_sd")
            drift_s = f"{drift:+.3f}" if isinstance(drift, (int, float)) else "n/a"
            note = row.get("note") or ""
            ev = str(row.get("because") or "")[:80]
            print(f"[text_signal] adj {a}: drift_sd={drift_s} "
                  f"shift={row.get('shift', 0.0):+.6f} widen={row.get('widen', 1.0):.3f} "
                  f"skew={row.get('skew', 0.0):+.3f}"
                  + (f" | {note}" if note else "")
                  + (f" | {ev}" if ev else ""), file=sys.stderr)
        if missing:
            # On a joint card this is not a small thing: an asset left at exact neutral while the
            # others move is itself a claim about that asset relative to them.
            # Deliberately NOT the "adj " prefix: that marks the healthy per-asset rows, which
            # run_eval.py filters out. A partial reply is a real degradation and should surface
            # in the sweep's issue list alongside a failed call.
            print(f"[text_signal] partial reply: {len(missing)} of {len(assets)} assets missing, "
                  f"left neutral: {missing}", file=sys.stderr)
    except Exception:
        pass


# ----------------------------------------------------------------------------- stage 2: parse + clamp
def _extract_json_object(content: str) -> dict[str, Any] | None:
    """The first JSON object in `content`, repairing an unbalanced tail if needed.

    Carried forward from the removed single-shot pipeline (git history): that pipeline measured a
    model dropping a trailing brace on an otherwise-good reply -- three opened, two closed, with
    `finish_reason: "stop"`, not a truncation. A strict parse throws away a perfectly good reading
    over a typo; this tries the longest well-formed prefix first, then closes whatever brackets
    and strings are still open. It only ever ADDS closers, never invents a value.
    """
    start = content.find("{")
    if start < 0:
        return None
    blob = content[start:].strip()
    for end in range(len(blob), 0, -1):
        if blob[end - 1] != "}":
            continue
        try:
            out = json.loads(blob[:end])
        except ValueError:
            continue
        return out if isinstance(out, dict) else None

    depth: list[str] = []
    in_str = False
    escaped = False
    for ch in blob:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth.append("}" if ch == "{" else "]")
        elif ch in "}]" and depth:
            depth.pop()
    if not depth:
        return None
    repaired = blob + ('"' if in_str else "") + "".join(reversed(depth))
    try:
        out = json.loads(repaired)
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def to_adjustments(
    raw: dict[str, Any], assets: list[str], ctx: dict[str, Any]
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """{drift_sd, vol_scale, skew} per asset -> the contract's {shift, widen, skew}, clamped.

    `shift = drift_sd * sigma`: `build_draws()` adds `shift` at every horizon without scaling it
    by sqrt(h), so sizing against the shortest horizon's sigma bounds the move everywhere instead
    of putting the near-horizon centre several sigma off its anchor at the long end.

    Anything missing, unreadable or non-finite drops to neutral FOR THAT ASSET, never for the
    whole card -- one bad asset should not zero out the ones the model answered fine.
    """
    top = raw if isinstance(raw, dict) else {}
    nested = top.get("assets")
    per = nested if isinstance(nested, dict) and any(a in nested for a in assets) else top

    adjustments: dict[str, dict[str, float]] = {}
    ledger: dict[str, Any] = {}
    for a in assets:
        spec = per.get(a)
        note = ""
        if not isinstance(spec, dict):
            adjustments[a] = dict(NEUTRAL)
            ledger[a] = {**NEUTRAL, "note": "no entry for this asset; neutral"}
            continue
        try:
            drift_sd = float(spec.get("drift_sd", 0.0))
            vol = float(spec.get("vol_scale", 1.0))
            skew = float(spec.get("skew", 0.0))
        except (TypeError, ValueError):
            drift_sd, vol, skew = 0.0, 1.0, 0.0
            note = "unreadable numbers; adjustment dropped"
        if not all(math.isfinite(x) for x in (drift_sd, vol, skew)):
            # json.loads accepts the bare literals NaN/Infinity; one of those reaching the output
            # parquet fails gate g3 (all-NaN column), so it is refused here, not there.
            drift_sd, vol, skew = 0.0, 1.0, 0.0
            note = "non-finite numbers; adjustment dropped"

        clamped_drift = min(max(drift_sd, -_DRIFT_SD_CLAMP), _DRIFT_SD_CLAMP)
        if clamped_drift != drift_sd:
            note = (note + "; " if note else "") + f"drift_sd {drift_sd:+.2f} clamped"
        w_lo, w_hi = _widen_clamp(ctx.get("family"))
        vol_c = min(max(vol, w_lo), w_hi)
        if vol_c != vol:
            note = (note + "; " if note else "") + f"vol_scale {vol:.2f} clamped"
        skew_c = min(max(skew, _SKEW_CLAMP[0]), _SKEW_CLAMP[1])

        sigma = float(ctx["sigma"].get(a, 0.0))
        if not math.isfinite(sigma) or sigma <= 0:
            shift = 0.0
            note = (note + "; " if note else "") + "no sigma for this asset; shift forced to 0"
        else:
            shift = clamped_drift * sigma

        adjustments[a] = {"shift": float(shift), "widen": float(vol_c), "skew": float(skew_c)}
        ledger[a] = {
            **adjustments[a],
            "drift_sd": clamped_drift,
            "sigma": sigma,
            "because": str(spec.get("evidence") or "")[:300],
            "note": note,
        }
    return adjustments, ledger


# ----------------------------------------------------------------------------- dev CLI
def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="text_signal", description="Stage 1 summaries + stage 2 adjustments.")
    p.add_argument("--text", type=pathlib.Path, help="units/<id>/text")
    p.add_argument("--prompt", metavar="DOC_TYPE", help="print that doc type's prompt and exit")
    p.add_argument("--json", action="store_true", help="print the summaries as JSON")
    p.add_argument("--adjust", action="store_true",
                   help="run the full stage-2 pipeline and print {shift, widen, skew} per asset")
    a = p.parse_args(argv)

    if a.prompt:
        print(prompt_for(a.prompt, "YYYY-MM-DD"))
        return 0
    if a.text is None:
        p.error("--text is required")

    if a.adjust:
        assets: list[str] = []
        card = a.text.parent / "card.toml"
        if card.is_file():
            assets = list(tomllib.loads(card.read_text())["targets"]["asset_ids"])
        if not assets:
            print("no assets: is --text pointing at a real unit's text/ dir?", file=sys.stderr)
            return 2
        adjustments = read_text_signal(a.text, assets)
        print(json.dumps(adjustments, indent=2))
        return 0

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
