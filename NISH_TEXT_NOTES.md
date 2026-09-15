# The text half — what the corpus is, and what `read_text_signal` does with it

Branch `feat/llm`. Owner: Nish. Everything here is measured from the 104 units in `units/`,
not estimated — `tools/sweep_text_signal.py` reproduces every number.

---

## 1. What the text data actually is

Each unit is a frozen, dated news-and-policy corpus as it stood on the card's as-of date:

```
units/<card-id>/
  card.toml                     <- as-of, target assets, horizons, value unit, scoring weights
  <panel>.parquet               <- the numbers   [date, asset, value, panel_id]
  text/
    corpus_index.json           <- THE MANIFEST. doc_id, timestamp, doc_type, source, file
    fomc_statement_20241218.txt
    fomc_minutes_..._released_2024-11-26.txt
    beigebook_202411.txt
    cpi_2024-12-11.txt
```

**627 documents over 104 units** — median 5 per unit, range 2 to 15.

| doc_type | count | typical size | what it is |
|---|---:|---:|---|
| `cb_speech` | 227 | 10–40k chars | BIS/ECB central-banker speeches. The most numerous and the most varied. |
| `fomc_statement` | 155 | ~2.4k chars | The decision itself. Tiny, dense, and the highest signal per character. |
| `beige_book` | 97 | **up to 235k chars** | Regional anecdote. Enormous, mostly filler. |
| `fomc_minutes` | 61 | ~50k chars | Where committee *disagreement* is visible. Best source for the spread. |
| `macro_release` | 53 | 20–140k chars | BLS/BEA CPI and payrolls. Mostly tables. |
| `landmark` | 21 | ~8k chars | Set-piece speeches (Jackson Hole, "whatever it takes"). |
| `corporate_8k` | 7 | small | SEC EDGAR exhibits. |
| `positioning_report` | 6 | small | CFTC commitments-of-traders. |

Four card families, and the text plays a different role in each
(`docs/CATEGORIES.md` is authoritative):

| family | units | what the text is for | score component it lives on |
|---|---:|---|---|
| F1 continuation-with-context | 23 | incremental signal on a series you can already see | marginal CRPS |
| F2 text-cued regime shift | 27 | the early warning, before the numbers turn | CRPS + tail |
| F3 cross-asset reasoning | 23 | text about one market, forecast for correlated ones | **joint variogram** |
| F4 tail/shock-from-text | 31 | does the text foreshadow the shock | **tail penalty** |

### Three properties that drive every design decision below

**a. The corpus is far too large to send.** Raw total: **28,082,152 characters** across the
104 units. Median unit 277,856 chars; the widest (`t2-F1-sahm-watch-2024`) 628,286 — roughly
157k tokens for one card. The prompt budget in `text_signal.py` cuts that **8.2x**, to a median
of ~33k chars (~8k tokens) and a hard ceiling of 60k chars.

**b. `corpus_index.json` is the only legitimate way in.** Never `glob("*.txt")`. A file with no
index entry has no timestamp, and a document whose date cannot be established cannot be shown to
predate the as-of — reading one is a leak before any gate runs. The index `note` on every shipped
unit says minutes and COT timestamps are **public release dates, not meeting dates**; that is
precisely what makes them admissible.

**c. The decision is already priced; the guidance is not.** On these cards the as-of date *is*
the meeting date, so the cut or hike that the statement describes is already in the last panel
level. What is left to forecast over the next 126 business days is the *path*. This is not a
theory — see §4.

---

## 2. What `read_text_signal` returns, and why it is shaped that way

The contract is fixed: `{asset: {"shift", "widen", "skew"}}`, with `shift` **additive in the
asset's own units**. That last part is the whole difficulty. Across the 104 units the 169
asset-series span:

- yields around **4.3** (`percent_per_annum`, 44 units)
- FX levels around **150** (`jpy_per_usd`) or **1.1** (`usd_per_eur`)
- factor returns around **0.006** (`cumulative_log_return`, 16 units) — some **negative**

"+0.15" is 15bp on a 2Y and noise on JPY. So **the model is never asked for a native number.**
It is asked for `drift_sd` — a move in units of the forecast's own standard deviation — and
`text_signal.py` multiplies by the per-asset sigma it estimates from the panel:

```
shift = drift_sd  x  sigma        sigma = sd(daily changes, last 260d) x sqrt(shortest horizon)
```

One scale, every card. Sigma is measured at the **shortest** horizon because `build_draws` adds
`shift` at every horizon without scaling it by `sqrt(h)` — sized against the longest, the
near-horizon centre would land several sigma off its anchor.

The signature only receives `text_dir` and `assets`, so the card, the as-of and the panel are
read from `text_dir.parent` (the unit directory). Nothing outside `/input` is touched and the
signature is unchanged, per the integration contract.

---

## 3. Modes — this runs today with no API key

`TEXT_SIGNAL_MODE`:

| mode | behaviour |
|---|---|
| `auto` *(default)* | call the model if `MODEL_ENDPOINT` is set, else the keyword floor |
| `llm` | model; falls back to the floor if the call fails |
| `heuristic` | **offline keyword floor. No network, no key, deterministic.** |
| `off` | exact neutral — reproduces the text-blind 1.0 baseline for A/B |

```bash
python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text            # see the adjustments
python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text --show-docs   # see the selection
python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text --show-prompt # see the prompt
python3 tools/sweep_text_signal.py --quiet --csv out/sweep.csv             # all 104 units, ~20s
python3 -m unittest tests.test_text_signal -v                              # 19 tests
```

When the key arrives, nothing changes but the environment:

```bash
export MODEL_ENDPOINT=https://integrate.api.nvidia.com/v1
export MODEL_NAME=nvidia/llama-3.3-nemotron-super-49b-v1
export MODEL_API_KEY=nvapi-...          # local dev only; there is NO participant key at scoring
export TEXT_SIGNAL_CACHE=out/model-cache  # replies cached by prompt hash; out/ is gitignored
```

The call is **stdlib `urllib`, not the `openai` package**, on purpose: the scoring image installs
`numpy/pandas/pyarrow/jsonschema` and nothing else, and an `ImportError` inside the agent costs
that card 4.0. `HTTP(S)_PROXY` — how egress reaches the organizer's audited proxy — is honoured
by `urllib` automatically.

---

## 4. The one result worth reading

The first version of the keyword floor got `t2-F1-hawkish-cut-2024` **backwards**. That card is a
25bp cut delivered alongside a dot plot cut from four 2025 cuts to two — hawkish on net, and the
2Y should rise. The floor read it dovish, 8.0 to 4.6, and pushed the 2Y down. The reason, from
the term-level diagnostic:

```
DOVE   +3.95  lower(ed|ing)? the target range        <- the action. Already in the level.
HAWK   +1.58  remains? (somewhat )?elevated
HAWK   +1.20  extent and timing                      <- the actual signal, outvoted 3:1
```

Weighting the decision verb at full strength means the floor just reports what the central bank
*did*, which the panel already knows. Action verbs are now at **0.4** and path language at full
weight, and the card reads correctly: **+13bp, widen 1.33**. `tests/test_text_signal.py`
pins this as a regression test.

The same correction applies to whatever the LLM is prompted to do. The prompt says the as-of level
explicitly so the model can see the decision is already in it.

### And one calibration fix

Before centring, the floor returned `widen > 1` on **169 of 169** asset-rows, mean 1.176. That is
not a text signal — it is a blanket bet that the statistical baseline is under-dispersed. The raw
uncertainty score is now centred on its cross-corpus median (3.8; p25 1.8, p75 4.9, max 13.1), so
a card widens only when it is uncertain *relative to a typical card*. Mean is now 1.044, range
0.926–1.442, and the ranking is the evidence that it means something:

- widest: `t2-F3-election-2024-joint` 13.1, `t2-F4-hml-covid-2020` 9.5, `t2-F4-covid-mkt-2020` 8.9
- narrowest: `t2-F1-measured-pace-2004` 0.0, `t2-F2-considerable-removal-2004` 0.0 — both cards
  about *well-telegraphed* policy, which is the right answer

Current floor over all 169 rows: 28 up / 56 down / 85 flat, mean drift −0.067 sd, sd 0.236.
The 85 flat rows are FX and factor assets, which the floor deliberately refuses to sign — on
`usd_per_eur` a rising number means a *weaker dollar* and on `jpy_per_usd` a *stronger* one, and a
keyword count cannot tell those apart. The prompt states the convention and lets the model try.

---

## 5. Failure handling — why there is so much of it

Per the README: a card that is inadmissible, errors, or is never attempted takes a pre-committed
worst case of **4.0**. Ignoring the text entirely scores **1.0**. An exception in the text half is
therefore four times worse than not having a text half.

So nothing here raises. Every stage degrades: model → keyword floor → exact neutral, and the
ledger says which (`[text_signal] source=... docs=...` on stderr, full JSON in `LAST_LEDGER`).
Specifically guarded, each with a test:

- `NaN` / `Infinity` — legal JSON literals to `json.loads`; one unguarded produces an all-NaN
  parquet that `g3_domain_semantics` refuses. Dropped to neutral for that asset.
- unbounded `drift_sd` — clamped to ±1.5 sigma (the floor gets ±0.5).
- `vol_scale` — clamped to [0.60, 2.00] (the floor to [0.90, 1.45]).
- an asset with **no panel** (F2 transfer cards, where the target may be absent by design) —
  no sigma means no scale, so `shift` is forced to 0 while `widen` still applies.
- reply keyed at the top level instead of under `"assets"` — both accepted; a reply naming *no*
  requested asset is reported as a skip, never applied as a silent zero.
- markdown fences, prose before the JSON, a reply truncated by `max_tokens` — all handled, the
  last with a message naming the fix.

---

## 6. Open items

- **Blocker, not mine:** `build_draws` crashes on every single-asset card
  (`np.corrcoef` of a `(1, m)` array is 0-d → `np.fill_diagonal` raises). That is **77 of the 104
  units**. Verified one-line fix for `feat/timeseries`: `corr = np.atleast_2d(np.corrcoef(D))`.
- Related: `_series` raises `SystemExit` when an asset is in no panel. On a sealed F2 transfer
  card where the target is absent by design, that is a 4.0. Worth a fallback on Dew's branch.
- `skew` is returned but `build_draws` ignores it. It is the natural lever for F4 (31 units, tail
  penalty is the primary component) — worth wiring up.
- The floor is **untested against realized outcomes**. Whether any of this beats 1.0 is Pun's
  harness to answer; `TEXT_SIGNAL_MODE=off` is the A/B control, and `out/sweep_heuristic.csv` is
  the per-card record to diff against.
