"""
Track 2 — the TEXT half.  Owner: Nish.  Branch: feat/llm.

`forecast_agent.read_text_signal()` is a three-line delegate into this file. Everything that
reads the corpus, talks to the model, and turns words into numbers lives here, so the diff in
the shared skeleton stays tiny and merges with Dew's and Pun's branches cannot collide.

    read_text_signal(text_dir, assets) -> {asset: {"shift": float, "widen": float, "skew": float}}

THE CONTRACT (fixed — see TEAM_TASKS.md):
    shift  additive, in the ASSET'S OWN UNITS (percent for UST_2Y, JPY-per-USD for JPY, ...)
    widen  multiplier on the spread, 1.0 = unchanged, >1 = more uncertain
    skew   tail tilt in [-1, 1]; Dew's baseline ignores it today, we still emit it

Two things make that contract awkward and this module exists to absorb both:

1. `shift` is in native units, but the 104 shipped units span yields (~4.3), FX levels (~150
   for JPY) and factor returns (~0.006). "+0.15" is 15bp on a yield and noise on JPY. So the
   model is never asked for a native number. It is asked for `drift_sd` — a move in units of
   the forecast's own standard deviation — and this module multiplies by the per-asset sigma
   it estimates from the panel. One scale, every card.

2. The signature gets `text_dir` and `assets` only: no card, no asof, no panel. But the text
   dir always sits next to them (`units/<id>/text`), so the card, the as-of date and the panel
   parquet are read from `text_dir.parent` — best-effort, never fatal. Nothing outside
   `/input` is touched and the signature is unchanged.

FAILURE IS NEVER FATAL. A card that errors scores 4.0; ignoring the text scores 1.0. Every
path here is wrapped, and a broken model call degrades to the offline heuristic and then to
exact neutral, saying which in the ledger.

MODES (env `TEXT_SIGNAL_MODE`, default `auto`):
    auto        call the model if MODEL_ENDPOINT is set, else heuristic
    llm         model only; if it fails, fall back to heuristic
    heuristic   offline keyword baseline, no network, deterministic   <-- works with no API key
    off         exact neutral (reproduces the text-blind 1.0 baseline)

Dev run:
    python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text
    python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text --show-prompt
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import tomllib
import urllib.error
import urllib.request
from typing import Any

# ----------------------------------------------------------------------------- tunables
NEUTRAL = {"shift": 0.0, "widen": 1.0, "skew": 0.0}

_TIMEOUT_SEC = float(os.environ.get("MODEL_TIMEOUT", "90"))
#: Prompt budget. The biggest unit (t2-F1-sahm-watch-2024) ships ~87k words ~= 570k chars;
#: nothing useful survives sending that, and the house endpoint has a context limit.
_TOTAL_CHAR_BUDGET = 60_000
_MAX_DOCS = 10
#: Per doc-type character cap and selection priority (lower = kept first). An FOMC statement is
#: ~2.5k chars and is the single most informative document on a rates card; a Beige Book is 200k
#: chars of anecdote. They must not get the same budget.
_DOC_RULES: dict[str, tuple[int, int]] = {
    "fomc_statement": (6_000, 0),
    "landmark": (8_000, 1),
    "fomc_minutes": (14_000, 2),
    "macro_release": (5_000, 3),
    "cb_speech": (7_000, 4),
    "positioning_report": (4_000, 5),
    "corporate_8k": (4_000, 6),
    "beige_book": (5_000, 7),
}
_DEFAULT_RULE = (5_000, 8)

#: Clamps. `drift_sd` is in standard deviations of the forecast at the SHORTEST horizon, because
#: Dew's `build_draws` adds `shift` at every horizon without scaling it — sized against the
#: longest horizon it would be several sigma wide at the short one.
_DRIFT_SD_CLAMP = 1.5
_WIDEN_CLAMP = (0.60, 2.00)
_SKEW_CLAMP = (-1.0, 1.0)
#: The offline heuristic is a floor, not a forecast. It is allowed a fraction of the model's room.
_HEURISTIC_DRIFT_SD_CLAMP = 0.50
_HEURISTIC_WIDEN_CLAMP = (0.90, 1.45)
#: Centre of the raw uncertainty score, measured across all 104 shipped units
#: (`tools/sweep_text_signal.py`): min 0.0, p25 1.8, median 3.8, p75 4.9, max 13.1.
#:
#: Without this the floor returned widen > 1 on every single card -- 169 of 169 asset-rows, mean
#: 1.176 -- which is not a text signal at all, it is a global bet that the statistical baseline is
#: under-dispersed. Centred on the median, a card only widens when it is uncertain RELATIVE TO A
#: TYPICAL CARD, and a well-telegraphed one (t2-F1-measured-pace-2004 scores 0.0) narrows. The
#: ranking the score produces is the check that it means something: the widest are
#: t2-F3-election-2024-joint (13.1), t2-F4-hml-covid-2020 (9.5) and t2-F4-covid-mkt-2020 (8.9).
_UNC_CENTER = 3.8
_UNC_SCALE = 4.0

#: Set by the last `read_text_signal` call; the CLI and (optionally) the rationale read it.
LAST_LEDGER: dict[str, Any] = {}


# ----------------------------------------------------------------------------- the entry point
def read_text_signal(
    text_dir: pathlib.Path, assets: list[str]
) -> dict[str, dict[str, float]]:
    """{asset: {shift, widen, skew}}. Never raises — see the module docstring."""
    global LAST_LEDGER
    text_dir = pathlib.Path(text_dir)
    adjustments = {a: dict(NEUTRAL) for a in assets}
    ledger: dict[str, Any] = {
        "mode_requested": os.environ.get("TEXT_SIGNAL_MODE", "auto").strip().lower(),
        "source": "neutral",
        "reason": "",
        "n_docs_used": 0,
        "n_docs_excluded_by_cutoff": 0,
        "n_docs_dropped_by_budget": 0,
        "assets": {},
    }
    try:
        mode = ledger["mode_requested"]
        if mode == "off":
            ledger["reason"] = "TEXT_SIGNAL_MODE=off"
            return adjustments

        ctx = load_context(text_dir, assets)
        ledger["asof"] = ctx["asof"]
        ledger["horizons"] = ctx["horizons"]
        ledger["value_unit"] = ctx["value_unit"]

        docs, n_excluded, n_dropped = read_corpus(text_dir, ctx["asof"])
        ledger["n_docs_used"] = len(docs)
        ledger["n_docs_excluded_by_cutoff"] = n_excluded
        ledger["n_docs_dropped_by_budget"] = n_dropped
        if not docs:
            ledger["reason"] = "no admissible documents at/before the as-of date"
            return adjustments

        raw: dict[str, dict[str, float]] | None = None
        if mode in ("auto", "llm"):
            raw, why, trace = llm_signal(docs, assets, ctx)
            if raw is None:
                ledger["llm_skipped_reason"] = why
                if trace:
                    ledger["llm_trace_chars"] = len(trace)
            else:
                ledger["source"] = "llm"
        if raw is None and mode in ("auto", "llm", "heuristic"):
            raw = heuristic_signal(docs, assets, ctx)
            ledger["source"] = "heuristic"
            ledger.setdefault("reason", "")

        adjustments, per_asset = to_adjustments(raw or {}, assets, ctx, ledger["source"])
        ledger["assets"] = per_asset
    except Exception as exc:  # never let the text half fail the card
        ledger["source"] = "neutral"
        ledger["reason"] = f"{type(exc).__name__}: {exc}"
        adjustments = {a: dict(NEUTRAL) for a in assets}
    LAST_LEDGER = ledger
    print(
        f"[text_signal] source={ledger['source']} docs={ledger['n_docs_used']} "
        f"{ledger.get('reason') or ledger.get('llm_skipped_reason') or ''}".rstrip(),
        file=sys.stderr,
    )
    return adjustments


# ----------------------------------------------------------------------------- unit context
def load_context(text_dir: pathlib.Path, assets: list[str]) -> dict[str, Any]:
    """as-of date, horizons, value unit, and a per-asset (level, daily sd) from the panel.

    Read from `text_dir.parent` — the unit directory — because the fixed signature does not
    hand us the card. Every field degrades: no card means the as-of comes from
    `corpus_index.json`, and no panel means the drift clamp falls back to a fraction of the
    level (and to 0 if there is no level either).
    """
    unit = text_dir.parent
    ctx: dict[str, Any] = {
        "asof": "9999-12-31",
        "horizons": [21],
        "value_unit": "unknown",
        "target_type": "level",
        "title": "",
        "note": "",
        "level": {},
        "sd_daily": {},
        "sigma": {},
        "assets": list(assets),
    }
    card_path = next((p for p in (unit / "card.toml", text_dir / "card.toml") if p.is_file()), None)
    if card_path is not None:
        with contextlib.suppress(Exception):
            card = tomllib.loads(card_path.read_text(encoding="utf-8"))
            tgt = card.get("targets", {})
            ctx["horizons"] = [int(h) for h in tgt.get("horizons", ctx["horizons"])]
            ctx["value_unit"] = str(tgt.get("value_unit", ctx["value_unit"]))
            ctx["target_type"] = str(tgt.get("target_type", ctx["target_type"]))
            ctx["asof"] = str(
                card.get("text", {}).get("cutoff")
                or card.get("provenance", {}).get("data_cutoff")
                or ctx["asof"]
            )[:10]
            ctx["title"] = str(card.get("task", {}).get("title", ""))
            ctx["note"] = str(card.get("text", {}).get("notes", ""))
    idx = text_dir / "corpus_index.json"
    if ctx["asof"] == "9999-12-31" and idx.is_file():
        with contextlib.suppress(Exception):
            ctx["asof"] = str(json.loads(idx.read_text(encoding="utf-8")).get("asof", ""))[:10]

    level, sd_daily = _panel_stats(unit, assets, ctx["asof"])
    ctx["level"], ctx["sd_daily"] = level, sd_daily
    h0 = max(1, min(ctx["horizons"]))
    for a in assets:
        if a in sd_daily:
            ctx["sigma"][a] = sd_daily[a] * math.sqrt(h0)
        elif a in level and level[a]:
            ctx["sigma"][a] = abs(level[a]) * 0.05  # no panel: 5% of the level, deliberately small
        else:
            ctx["sigma"][a] = 0.0  # no scale is knowable -> shift stays 0, widen still works
    ctx["sigma_horizon"] = h0
    return ctx


def _panel_stats(
    unit: pathlib.Path, assets: list[str], asof: str
) -> tuple[dict[str, float], dict[str, float]]:
    """(last level, daily sd) per asset from the unit's parquet panels. Empty dicts on any error."""
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


# ----------------------------------------------------------------------------- the corpus
def read_corpus(
    text_dir: pathlib.Path, asof: str
) -> tuple[list[dict[str, Any]], int, int]:
    """Admissible documents, newest first, trimmed to the prompt budget.

    Returns `(docs, excluded_by_cutoff, dropped_by_budget)`.

    Driven by `corpus_index.json`, never by `glob("*.txt")`. A file with no index entry has no
    timestamp, and a document whose date cannot be established cannot be shown to predate the
    as-of — reading one is a leak before any gate runs. The index `note` on every shipped unit
    is explicit that minutes and COT timestamps are PUBLIC RELEASE dates; that is what makes
    them usable at all.
    """
    index_path = text_dir / "corpus_index.json"
    if not index_path.is_file():
        return [], 0, 0
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return [], 0, 0

    kept: list[dict[str, Any]] = []
    excluded = 0
    for doc in index.get("documents", []):
        ts = str(doc.get("timestamp", ""))[:10]
        path = text_dir / str(doc.get("file", ""))
        if not ts or ts > asof or not path.is_file():
            excluded += 1
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            excluded += 1
            continue
        doc_type = str(doc.get("doc_type", "") or "unknown")
        cap, prio = _DOC_RULES.get(doc_type, _DEFAULT_RULE)
        kept.append(
            {
                "doc_id": doc.get("doc_id") or path.stem,
                "timestamp": ts,
                "doc_type": doc_type,
                "source": doc.get("source", ""),
                "note": doc.get("note", ""),
                "full_chars": len(body),
                "text": salient_excerpt(body, cap),
                "_prio": prio,
            }
        )

    # Newest first within priority: a December statement outranks a November one, and both
    # outrank a Beige Book of any date.
    kept.sort(key=lambda d: (d["_prio"], _neg_date(d["timestamp"])))
    selected: list[dict[str, Any]] = []
    spent = 0
    for d in kept:
        if len(selected) >= _MAX_DOCS or spent + len(d["text"]) > _TOTAL_CHAR_BUDGET:
            continue
        selected.append(d)
        spent += len(d["text"])
    dropped = len(kept) - len(selected)
    selected.sort(key=lambda d: d["timestamp"], reverse=True)  # present newest first
    return selected, excluded, dropped


def _neg_date(ts: str) -> str:
    """Sort key that puts later dates first inside an ascending sort."""
    return "".join(chr(ord("9") - (ord(c) - ord("0"))) if c.isdigit() else c for c in ts)


#: Sentences worth keeping when a document is too long to send whole. Policy language, not prose.
_SALIENT = re.compile(
    r"target range|federal funds rate|policy rate|bank rate|deposit facility|"
    r"inflation|disinflation|price stability|unemployment|labor market|employment|"
    r"participants (?:noted|judged|agreed|expressed)|the Committee|Governing Council|"
    r"risks?|uncertain|outlook|guidance|projections|dot plot|balance sheet|"
    r"asset purchases|quantitative|tighten|accommodat|restrictive|"
    r"basis points|percentage point|forecast|expect",
    re.I,
)


def salient_excerpt(body: str, cap: int) -> str:
    """Head of the document plus the policy-bearing sentences, trimmed to `cap` characters.

    Head-truncation alone throws away the operative paragraph of a 200k-character Beige Book and
    keeps its table of contents. The head is kept because releases front-load the decision; the
    rest is filled with sentences that actually mention policy, in document order.
    """
    body = re.sub(r"[ \t]+", " ", body).strip()
    if len(body) <= cap:
        return body
    head_cap = max(600, cap // 3)
    head = body[:head_cap]
    remainder = body[head_cap:]
    picked: list[str] = []
    room = cap - len(head) - 40
    for sentence in re.split(r"(?<=[.!?])\s+", remainder):
        if room <= 0:
            break
        if len(sentence) < 40 or not _SALIENT.search(sentence):
            continue
        picked.append(sentence.strip())
        room -= len(sentence) + 1
    tail = " ".join(picked)
    out = f"{head}\n[...]\n{tail}" if tail else head
    # Hard trim: the sentence loop checks the budget BEFORE appending, so the last sentence can
    # overshoot it. The cap is a budget, not a suggestion -- 104 units x one overshoot each is a
    # context-limit failure on the widest cards.
    return out[:cap]


# ----------------------------------------------------------------------------- the offline floor
#: Hawkish (+) / dovish (-) FORWARD GUIDANCE. Weights are deliberately blunt: this is the floor
#: the model has to beat, not a tuned lexicon.
#:
#: The decision itself is weighted near zero on purpose. On these cards the as-of date IS the
#: meeting date, so the cut or hike is already in the last panel level and in the market — what
#: moves a 2Y over the next 126 business days is the GUIDANCE about the path, not the action just
#: taken. Measured on t2-F1-hawkish-cut-2024 (a 25bp cut delivered with a dot plot cut from four
#: 2025 cuts to two: hawkish on net), weighting "lowered the target range" at 3.0 made the floor
#: read the card DOVISH, 8.0 to 4.6, and pushed the 2Y down when the card's whole premise is that
#: it should go up. Action verbs at 0.4, path language at full weight, and it reads it correctly.
#:
#: Phrases beat single words: "cut" appears in "rate cut" and in "cut the pace of purchases",
#: which point opposite ways.
_HAWKISH = {
    # --- the action taken: already priced, near-zero weight (see above) ---
    r"rais(?:e|ed|ing) the target range|increase the target range": 0.4,
    # --- the path from here: this is the signal ---
    r"further (?:policy )?(?:firming|tightening)": 2.5,
    r"additional (?:policy )?firming|more restrictive": 2.5,
    r"fewer (?:rate )?cuts|shallower(?: cut| easing)? path": 3.0,
    r"extent and timing": 2.5,
    r"carefully assess|proceed carefully": 1.5,
    r"sufficiently restrictive": 2.0,
    r"restrictive (?:stance|policy|territory)": 1.5,
    r"higher for longer": 2.5,
    r"upside risks? to inflation": 2.2,
    r"inflation pressures? (?:have )?(?:increased|intensified|persist)": 1.8,
    r"remains? (?:somewhat )?elevated": 1.2,
    r"not (?:yet )?(?:appropriate|confident)|more confidence(?: is)? (?:needed|required)": 1.5,
    r"tight(?:en|ening)(?: monetary policy| financial conditions)?": 1.0,
    r"vigilan(?:t|ce)": 0.8,
    r"strong(?:er)? than expected|solid pace|robust": 0.6,
    r"\bpatient\b": 0.8,
    r"dot plot|median projection": 0.5,
}
_DOVISH = {
    # --- the action taken: already priced, near-zero weight ---
    r"lower(?:ed|ing)? the target range|reduce the target range": 0.4,
    r"cut the (?:target range|policy rate)": 0.4,
    # --- the path from here ---
    r"additional (?:rate )?cuts|further easing|further reductions": 2.5,
    r"more cuts than|deeper(?: cut| easing)? path": 3.0,
    r"accommodat(?:ive|ion)": 1.5,
    r"downside risks? to (?:employment|growth|activity)": 2.2,
    r"risks? (?:have )?shifted to the downside": 2.2,
    r"labor market (?:has )?(?:eased|weakened|softened|cooled)": 1.5,
    r"greater confidence": 1.2,
    r"inflation has (?:declined|eased|moderated|made progress)": 1.0,
    r"disinflation": 1.0,
    r"recession|contraction|deteriorat": 1.5,
    r"asset purchase|quantitative easing|expand(?:ing)? (?:its )?balance sheet": 2.0,
    r"unemployment rate (?:has )?(?:risen|moved up|increased)": 1.0,
    r"act as appropriate|prepared to (?:adjust|act)": 0.8,
}
#: Uncertainty language -> `widen`. Disagreement inside a committee is the single most reliable
#: text signal for spread: when participants split, the realised path fans out.
_UNCERTAIN = {
    r"uncertain(?:ty|ties)?": 0.9,
    r"the economic outlook is uncertain": 1.5,
    r"risks? to both sides|two-sided risks?": 1.0,
    r"some participants.{0,120}(?:while |other|however)": 1.4,
    r"participants (?:differed|disagreed|expressed a range)": 1.6,
    r"a (?:wide )?range of views": 1.2,
    r"volatil(?:e|ity)": 0.8,
    r"tariff|trade (?:war|tension)|geopolitic": 1.0,
    r"data[- ]dependent|meeting by meeting|meeting-by-meeting": 0.8,
    r"unusually (?:elevated|wide)|difficult to (?:predict|assess)": 1.2,
    r"dissent|voting against": 1.0,
}
#: Assets whose direction the offline floor is allowed to move. A hawkish surprise raises yields —
#: that mapping is stable. On FX it depends on the quote convention (usd_per_eur rises when the
#: DOLLAR weakens, jpy_per_usd rises when it strengthens) and on factor returns it is not defined
#: at all, so the floor leaves those centred and only widens. The model is told the convention
#: and may move them; the keyword baseline may not.
_RATE_ASSET = re.compile(r"^(UST|DGS|GILT|JGB|BUND|OAT|BTP|SW|AU|CA|NZ)[_A-Z0-9]*\d+Y?$|_\d+Y$")


def heuristic_signal(
    docs: list[dict[str, Any]], assets: list[str], ctx: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Deterministic keyword baseline: no network, no key, same answer every run.

    Returns the same normalised shape the model is asked for:
        {asset: {"drift_sd": float, "vol_scale": float, "skew": float, "because": str}}

    Recency-weighted: the newest document carries full weight, each older one 0.75x. A December
    statement that changed the guidance should not be outvoted by three speeches from October.
    """
    hawk = dove = unc = 0.0
    weight = 1.0
    cites: list[tuple[float, str]] = []
    for doc in docs:
        text = doc["text"]
        h = _score(text, _HAWKISH)
        d = _score(text, _DOVISH)
        u = _score(text, _UNCERTAIN)
        hawk += weight * h
        dove += weight * d
        unc += weight * u
        if abs(h - d) > 0.5:
            cites.append((weight * abs(h - d), f"{doc['doc_id']} ({doc['timestamp']})"))
        weight *= 0.75

    net = hawk - dove
    # tanh keeps a document dump from saturating the signal: 6 net hits ~= 0.8 of the clamp.
    drift_sd = _HEURISTIC_DRIFT_SD_CLAMP * math.tanh(net / 6.0)
    lo, hi = _HEURISTIC_WIDEN_CLAMP
    # Relative to a typical card, not to zero -- see _UNC_CENTER.
    t = math.tanh((unc - _UNC_CENTER) / _UNC_SCALE)
    vol_scale = 1.0 + (hi - 1.0) * t if t > 0 else 1.0 + (1.0 - lo) * t
    vol_scale = min(max(vol_scale, lo), hi)
    cites.sort(reverse=True)
    because = (
        f"keyword floor: hawkish {hawk:.1f} vs dovish {dove:.1f}, "
        f"uncertainty {unc:.1f} vs a {_UNC_CENTER} median"
        + (f"; strongest: {cites[0][1]}" if cites else "")
    )

    out: dict[str, dict[str, float]] = {}
    for a in assets:
        directional = bool(_RATE_ASSET.match(a)) or "percent_per_annum" in ctx["value_unit"]
        out[a] = {
            "drift_sd": drift_sd if directional else 0.0,
            "vol_scale": vol_scale,
            "skew": 0.0,
            "because": because if directional else f"{because}; non-rate asset: spread only",
        }
    return out


def _score(text: str, lexicon: dict[str, float]) -> float:
    """Summed weight of the lexicon's matches, capped per phrase so one repeated term cannot win."""
    total = 0.0
    for pattern, w in lexicon.items():
        n = len(re.findall(pattern, text, re.I))
        if n:
            total += w * min(n, 3) / 1.0 if w < 0 else w * min(n, 3)
    return total


# ----------------------------------------------------------------------------- the model
def build_prompt(
    docs: list[dict[str, Any]], assets: list[str], ctx: dict[str, Any]
) -> str:
    """The prompt states the level AND the sigma the answer is measured in.

    `drift_sd` rather than a native-unit move or basis points of the level: the shipped panels
    include negative anchors (`t2-F1-ai-mom-2024` MOM is -0.0065) where a bp-of-level drift moves
    the distribution the wrong way, and anchors small enough that 250bp of them is a rounding
    error against the forecast's own width. One standard deviation means the same thing on a
    yield, on JPY and on a factor return.
    """
    h = ctx["horizons"]
    lines = [
        "You are a macro forecaster adjusting a statistical forecast using dated documents.",
        f"As-of date: {ctx['asof']}. Nothing after this date is known to you.",
        "Do not use knowledge of what happened later; reason only from the documents below.",
        "",
        f"Target: {ctx['target_type']} of each asset, {h} business days after the as-of.",
        f"Value unit: {ctx['value_unit']}.",
    ]
    if ctx.get("title"):
        lines.append(f"Card: {ctx['title']}")
    lines += [
        "",
        "Assets. `level` is the value on the as-of date; `sigma` is the statistical standard",
        f"deviation of the forecast at the shortest horizon ({ctx['sigma_horizon']} BD) — the unit",
        "your answer is measured in.",
    ]
    for a in assets:
        lvl = ctx["level"].get(a)
        sig = ctx["sigma"].get(a, 0.0)
        lines.append(
            f"  {a}: level {lvl:.6f}, sigma {sig:.6f}" if lvl is not None
            else f"  {a}: level unknown, sigma {sig:.6f}"
        )
    lines += [
        "",
        "Quote conventions matter for direction: a unit like usd_per_eur RISES when the dollar",
        "WEAKENS, while jpy_per_usd RISES when the dollar STRENGTHENS. Check the value unit above",
        "before choosing a sign, and answer 0 if the convention makes you unsure.",
        "",
        f"Documents ({len(docs)}), newest first. Long documents are excerpted.",
    ]
    for d in docs:
        trunc = " (excerpt)" if d["full_chars"] > len(d["text"]) else ""
        lines += [
            f"--- {d['doc_id']} | {d['timestamp']} | {d['doc_type']}{trunc} ---",
            d["text"],
            "",
        ]
    lines += [
        "For EACH asset give three numbers:",
        f"  drift_sd  : where the centre of the distribution should move, in sigma. Range"
        f" [-{_DRIFT_SD_CLAMP}, {_DRIFT_SD_CLAMP}]; 0 if the documents say nothing directional.",
        "              Sign is in the asset's own unit as printed above.",
        f"  vol_scale : multiplier on sigma, range [{_WIDEN_CLAMP[0]}, {_WIDEN_CLAMP[1]}]. >1 when",
        "              the documents show disagreement, two-sided risk or an unresolved decision;",
        "              <1 only when they REMOVE uncertainty (an explicit commitment or a peg).",
        "  skew      : tail tilt in [-1, 1]. Positive = the upside tail is the fatter one.",
        "",
        "First decide `tone`, then map it to the numbers. `tone` is one of:",
        "  hawkish  the documents point to tighter policy / higher rates than the path implied",
        "  dovish   they point to easier policy / lower rates than the path implied",
        "  neutral  ONLY when the documents contain no forward guidance at all. Central bank",
        "           language is always hedged -- hedged is not the same as neutral, and a",
        "           document that hedges in one direction is not neutral.",
        "",
        "How big is big, in sigma. Use this scale, it is not symmetric around 0 by accident:",
        "  0.0 - 0.1   a routine meeting, guidance unchanged from the previous one",
        "  0.2 - 0.5   the guidance language CHANGED -- a phrase added, dropped or qualified",
        "  0.6 - 1.0   a clear shift: new conditions on the next move, a changed projection,",
        "              a dissent, or a committee visibly split",
        "  1.0 - 1.5   a genuine surprise the market cannot have priced",
        "",
        "The statistical forecast ALREADY assumes the documents say nothing. Answering all",
        "zeros is identical to not reading them. Do not hedge toward 0 for safety -- the",
        "ranges above already bound you, and a small committed number beats a confident zero.",
        "",
        "Remember the decision itself is already in the level printed above. A cut that was",
        "delivered is priced; what is NOT priced is what the documents say about the path from",
        "here -- the vote split, the projections, the conditions attached to the next move.",
        "",
        "All three must be finite numbers. NaN and Infinity are rejected and that asset's",
        "adjustment is dropped.",
        "",
        "Reply with JSON only, no prose, no markdown fence. The three numbers come FIRST and",
        "`evidence` LAST -- a doc_id and at most 15 words. Keep evidence short; if the reply is",
        "cut off the numbers must already be complete.",
        '{"assets": {'
        + ", ".join(
            f'"{a}": {{"tone": "hawkish|dovish|neutral", "drift_sd": 0.0, "vol_scale": 1.0, '
            '"skew": 0.0, "evidence": "<doc_id>: <=15 words"}'
            for a in assets[:2]
        )
        + "}}",
    ]
    return "\n".join(lines)


def llm_signal(
    docs: list[dict[str, Any]], assets: list[str], ctx: dict[str, Any]
) -> tuple[dict[str, dict[str, float]] | None, str, str]:
    """(normalised per-asset dict, skip reason, reasoning trace). The ONLY network call we make."""
    prompt = build_prompt(docs, assets, ctx)
    parsed, why, trace = call_model(prompt)
    if parsed is None:
        return None, why, trace
    top = parsed if isinstance(parsed, dict) else {}
    nested = top.get("assets")
    # Models comply with the nested shape about half the time and otherwise key assets at the top
    # level. Both are honest readings of the instruction; accept both, but require a real match so
    # an unrecognised shape is reported as a skip and never applied as a silent zero.
    per = nested if isinstance(nested, dict) and any(a in nested for a in assets) else top
    if not any(isinstance(per.get(a), dict) for a in assets):
        return None, "reply named none of the requested assets", trace
    return {a: per[a] for a in assets if isinstance(per.get(a), dict)}, "", trace


def call_model(prompt: str) -> tuple[dict[str, Any] | None, str, str]:
    """POST to the OpenAI-compatible endpoint. stdlib urllib only.

    Not the `openai` package: the scoring image installs `numpy/pandas/pyarrow/jsonschema` and
    nothing else, and an ImportError inside the agent costs the card 4.0. `urllib` is in the
    standard library and the wire format is the same three fields.

    Local dev  : MODEL_ENDPOINT=https://integrate.api.nvidia.com/v1 MODEL_API_KEY=nvapi-...
    At scoring : MODEL_ENDPOINT/MODEL_NAME are injected, there is no participant key, and egress
                 is the audited proxy only — HTTP(S)_PROXY is honoured by urllib automatically.
                 House access arrives as MODEL_TOKEN plus an authenticated http_proxy;
                 MODEL_API_KEY is the local-dev route only (baselines/README.md:150).
    """
    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip()
    # Only a local-dev fallback: at scoring MODEL_NAME is injected and names the House pin,
    # NVIDIA Nemotron 3 Super 120B-A12B FP8 behind the `house` alias (organizers, public #14).
    # This is the build.nvidia.com id for that same checkpoint family, so local runs exercise the
    # capacity we will actually be served. The old llama-3.3 stub id is dead (HTTP 410).
    model = os.environ.get("MODEL_NAME", "").strip() or "nvidia/nemotron-3-super-120b-a12b"
    if not endpoint:
        return None, "MODEL_ENDPOINT is unset (offline: using the keyword floor)", ""

    cache_dir = os.environ.get("TEXT_SIGNAL_CACHE", "").strip()
    key = hashlib.sha256(f"{endpoint}|{model}|{prompt}".encode()).hexdigest()[:16]
    cache_file = pathlib.Path(cache_dir) / f"{key}.json" if cache_dir else None
    if cache_file is not None and cache_file.is_file():
        try:
            return json.loads(cache_file.read_text()), "", "(cached)"
        except Exception:
            pass

    # The House renderer has thinking ON by default -- a request that omits the option renders
    # identically to enable_thinking=True (organizers, public #14). So send the flag on every
    # request rather than relying on a server default. Off is measured strictly better here: the
    # 30B burned 83 s/card and never reached the JSON, the 120B halved its committed adjustments.
    thinking = os.environ.get("MODEL_THINKING", "off").strip().lower() in ("1", "on", "true")
    try:
        max_tokens = max(1, int(os.environ.get("MODEL_MAX_TOKENS", "3000")))
    except ValueError:
        max_tokens = 3000

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
            # Top level, not extra_body: that is the OpenAI-client wrapper for the same field,
            # and the House request normalizer preserves it on the raw JSON path.
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    # MODEL_TOKEN is the House grant; MODEL_API_KEY is the local key. Never the other way round.
    token = (
        os.environ.get("MODEL_TOKEN", "").strip() or os.environ.get("MODEL_API_KEY", "").strip()
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")[:300]
        return None, f"HTTP {exc.code}: {detail}", ""
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}", ""

    try:
        choice = payload["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError):
        return None, "reply had no choices[0].message.content", ""
    trace = message.get("reasoning_content") or ""
    if not isinstance(content, str):
        return None, "choices[0].message.content was not a string", trace

    start = content.find("{")
    if start < 0:
        if choice.get("finish_reason") == "length":
            # Not "the model ignored the schema" — we did not give it room to answer.
            why = f"hit max_tokens={max_tokens} before any JSON; raise MODEL_MAX_TOKENS"
            return None, why, trace
        return None, "reply contained no JSON object", trace
    out = _parse_json_object(content[start:])
    if out is None:
        return None, f"reply JSON did not parse: {content[start:][:200]!r}", trace
    if cache_file is not None:
        with contextlib.suppress(Exception):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(out, indent=2))
    return out, "", trace


def _parse_json_object(blob: str) -> dict[str, Any] | None:
    """Parse the first JSON object in `blob`, repairing an unbalanced tail if needed.

    Measured against nvidia/nemotron-3.5-lightning-30b-a3b on t2-F1-ai-mom-2024: the model
    returned

        {"assets": {"MOM": {"drift_sd": 0.0, "vol_scale": 1.0, "skew": 0.0, "because": "..."}}

    -- three braces open, two closed, and `finish_reason` was "stop", not "length". It is not a
    truncation we caused by under-budgeting tokens; the model simply dropped the last brace. A
    strict `rfind("}")` parse fails, the adjustment is discarded, and a perfectly good reading of
    the documents is thrown away on a typo.

    Tries longest-prefix first so a well-formed object is never altered, then closes unterminated
    strings and appends the missing brackets. Only ever ADDS closers -- it cannot invent a value.

    This repair is why `build_prompt` puts the three numbers BEFORE `evidence` in the response
    schema. An evidence-first schema had the model quoting a long passage and running out of
    completion budget inside that string, so the truncated reply carried no numbers at all and
    there was nothing to recover. Numbers first means a cut-off reply still yields a usable
    adjustment and loses only the citation.
    """
    blob = blob.strip()
    for end in range(len(blob), 0, -1):
        if blob[end - 1] != "}":
            continue
        try:
            out = json.loads(blob[:end])
        except ValueError:
            continue
        return out if isinstance(out, dict) else None

    # Nothing parsed as-is. Walk the text tracking string state and bracket depth, then close it.
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


# ----------------------------------------------------------------------------- normalise + clamp
def to_adjustments(
    raw: dict[str, Any], assets: list[str], ctx: dict[str, Any], source: str
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Normalised {drift_sd, vol_scale, skew} -> the contract's {shift, widen, skew}, clamped.

    `shift = drift_sd * sigma`, where sigma is the forecast's own standard deviation at the
    SHORTEST horizon. Dew's `build_draws` adds `shift` at every horizon without scaling it by
    sqrt(h), so sizing against the longest horizon would put the near-horizon centre several
    sigma off its anchor. Sized against the short horizon the move is bounded everywhere.

    Anything non-finite, unreadable or unmatched drops to neutral FOR THAT ASSET and says so.
    """
    drift_clamp = _HEURISTIC_DRIFT_SD_CLAMP if source == "heuristic" else _DRIFT_SD_CLAMP
    widen_clamp = _HEURISTIC_WIDEN_CLAMP if source == "heuristic" else _WIDEN_CLAMP
    adjustments: dict[str, dict[str, float]] = {}
    ledger: dict[str, Any] = {}
    for a in assets:
        spec = raw.get(a)
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
            # json.loads accepts the bare literals NaN and Infinity, and one of them produces an
            # all-NaN parquet that g3_domain_semantics refuses. Never let one reach the output.
            drift_sd, vol, skew = 0.0, 1.0, 0.0
            note = "non-finite numbers; adjustment dropped"

        clamped_drift = min(max(drift_sd, -drift_clamp), drift_clamp)
        if clamped_drift != drift_sd:
            note = (note + "; " if note else "") + f"drift_sd {drift_sd:+.2f} clamped"
        vol_c = min(max(vol, widen_clamp[0]), widen_clamp[1])
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
            "level": ctx["level"].get(a),
            "because": str(spec.get("evidence") or spec.get("because") or "")[:300],
            "note": note,
        }
    return adjustments, ledger


# ----------------------------------------------------------------------------- dev CLI
def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="text_signal",
        description="Inspect what the text half would hand build_draws for one unit.",
    )
    p.add_argument("--text", type=pathlib.Path, required=True, help="units/<id>/text")
    p.add_argument("--assets", nargs="*", default=None, help="default: the card's asset_ids")
    p.add_argument("--show-prompt", action="store_true", help="print the prompt and exit")
    p.add_argument("--show-docs", action="store_true", help="list the selected corpus")
    a = p.parse_args(argv)

    assets = a.assets
    if not assets:
        card = a.text.parent / "card.toml"
        if card.is_file():
            assets = list(tomllib.loads(card.read_text())["targets"]["asset_ids"])
    if not assets:
        print("no assets: pass --assets", file=sys.stderr)
        return 2

    ctx = load_context(a.text, assets)
    docs, excluded, dropped = read_corpus(a.text, ctx["asof"])
    if a.show_docs:
        print(f"as-of {ctx['asof']}  horizons {ctx['horizons']}  unit {ctx['value_unit']}")
        print(f"{len(docs)} selected, {excluded} excluded by cutoff, {dropped} dropped by budget\n")
        for d in docs:
            print(f"  {d['timestamp']}  {d['doc_type']:<18} {d['doc_id']:<42} "
                  f"{len(d['text']):>6}/{d['full_chars']:<7} chars")
        print(f"\n  total prompt chars: {sum(len(d['text']) for d in docs)}")
        return 0
    if a.show_prompt:
        print(build_prompt(docs, assets, ctx))
        return 0

    adjustments = read_text_signal(a.text, assets)
    print(json.dumps({"adjustments": adjustments, "ledger": LAST_LEDGER}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
