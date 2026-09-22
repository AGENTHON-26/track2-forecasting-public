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
import contextlib
import json
import math
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

#: Stage-2 clamps, carried forward from the removed single-shot pipeline (git history before
#: "feat(text): stage 1" -- the clamps were never the reason it measured net negative; the
#: budget-truncated single-shot excerpts were the likely cause, and stage 1 already replaces
#: those with whole-document summaries).
_DRIFT_SD_CLAMP = 1.5
_WIDEN_CLAMP = (0.60, 2.00)
_SKEW_CLAMP = (-1.0, 1.0)


# ----------------------------------------------------------------------------- the entry point
def read_text_signal(
    text_dir: pathlib.Path, assets: list[str]
) -> dict[str, dict[str, float]]:
    """Stage 1 (summarize) + stage 2 (adjust). Never raises -- see the module docstring."""
    text_dir = pathlib.Path(text_dir)
    neutral = {a: dict(NEUTRAL) for a in assets}
    try:
        summaries = [s for s in summarize_corpus(text_dir) if s.get("summary")]
        if not summaries:
            print("[text_signal] no usable summaries; neutral", file=sys.stderr)
            return neutral

        ctx = load_context(text_dir, assets)
        system, user = build_adjustment_prompt(summaries, assets, ctx)
        content, err = call_model(system, user, thinking=False)
        # Thinking OFF here too, for the same reason as the removed pipeline's equivalent call
        # (git history): measured there that thinking ON made the 120B halve its committed
        # adjustments -- more hedging, not better reasoning, for exactly the kind of "commit to a
        # number" task this call is. Not re-measured against this new prompt; worth rechecking.
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
              f"assets={list(adjustments)}", file=sys.stderr)
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


# ----------------------------------------------------------------------------- stage 2: context
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

    level, sd_daily = _panel_stats(unit, assets, ctx["asof"])
    ctx["level"] = level
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
) -> tuple[dict[str, float], dict[str, float]]:
    """(last level, daily sd) per asset from the unit's own panel parquet(s). Empty on any error.

    Carried forward from the removed single-shot pipeline (git history before "feat(text): stage
    1") -- this idea is orthogonal to why that pipeline measured net negative. Converting to a
    per-asset standard-deviation scale is what lets `drift_sd` mean the same thing on a yield, on
    JPY and on a factor return, instead of asking the model to reason in each asset's own units.
    """
    level: dict[str, float] = {}
    sd: dict[str, float] = {}
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except Exception:
        return level, sd
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
            diffs = np.diff(series[-260:])
            if diffs.size >= 30 and math.isfinite(float(diffs.std())):
                sd[a] = float(diffs.std())
    return level, sd


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
        "(30% weight), which checks whether your assets move together the way they actually do. "
        "Work out ONE coherent macro scenario from the summaries first, then derive every asset's "
        "numbers FROM that same scenario, so they move consistently with each other rather than "
        "being judged independently. (This reply format has no field for a target correlation --"
        " consistency has to come from your reasoning about one shared scenario, not from a "
        "number you can state directly.)"
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
        f"[{_WIDEN_CLAMP[0]}, {_WIDEN_CLAMP[1]}]. >1 when the summaries show disagreement, "
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
        "All three numbers must be finite (no NaN or Infinity). Reply with JSON only, no prose, "
        "no markdown fence:\n"
        '{"assets": {'
        + ", ".join(
            f'"{a}": {{"drift_sd": 0.0, "vol_scale": 1.0, "skew": 0.0, '
            '"evidence": "<=15 words"}'
            for a in assets[:2]
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
        lines.append(
            f"  {a}: level {lvl:.6f}, sigma {sig:.6f}" if lvl is not None
            else f"  {a}: level unknown, sigma {sig:.6f}"
        )
    lines += ["", f"Document summaries ({len(summaries)}), newest first:"]
    for s in summaries:
        lines += [f"--- {s['doc_id']} | {s['timestamp']} | {s['doc_type']} ---", s["summary"], ""]
    return system, "\n".join(lines)


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
        vol_c = min(max(vol, _WIDEN_CLAMP[0]), _WIDEN_CLAMP[1])
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
