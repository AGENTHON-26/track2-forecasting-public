# Text half — staged rebuild (Nish, `feat/llm`)

The single-prompt pipeline (keyword floor, excerpt budgets, drift clamps, `TEXT_SIGNAL_MODE`) was
measured net negative (23 wins / 39 losses over 90 units) and has been removed. Its notes and the
sweep tool are in git history before this commit.

The rebuild goes in stages. `read_text_signal()` returns exact neutral (the text-blind 1.0
baseline) until a later stage produces adjustments.

## Stage 1 — per-document summary agents (`text_signal.py`)

- Documents come from `corpus_index.json`, filtered to `timestamp <= as-of` (card `[text].cutoff`).
- Each document is sent **whole** to the model with a checklist prompt written for its `doc_type`
  (`fomc_statement`, `fomc_minutes`, `cb_speech`, `landmark`, `beige_book`, `macro_release`,
  `positioning_report`, `corporate_8k`, plus a `default`): checklist items first, then other main
  points, at most 15 one-sentence bullets. The length rule is repeated after the document.
- Documents under 3,000 chars are passed through verbatim.
- CFTC tables: the code keeps only the largest-open-interest row per date before anything else
  (4 of 6 files mix 2-5 unlabeled contracts per date).
- Thinking is OFF, except for `landmark` (which mixes policy decisions and speeches under one label).
- 429/5xx retried with backoff; empty or runaway (>8k char) replies retried once. Nothing raises.

```bash
python3 text_signal.py --text units/t2-F1-hawkish-cut-2024/text          # summaries + size table
python3 text_signal.py --prompt beige_book                                # show one prompt
python3 tools/try_summaries.py --pick median --out out/run_median.json   # 1 real doc per type
python3 tools/eval_summaries.py out/v8r1 out/v8r2 out/v8r3               # facts / numbers / time
python3 -m unittest tests.test_text_signal -v
```

Env: `MODEL_ENDPOINT`, `MODEL_NAME` (injected at scoring), `MODEL_TOKEN` (House) or
`MODEL_API_KEY` (local dev only, in the gitignored `.env`).

## Measured 2026-09-20 (Nemotron 3 Super 120B, 16 docs x 3 runs)

- 79/96 key-fact checks pass. Statements, speeches, CPI, 8-Ks, Beige Books and the BoE decision are
  reliable. The largest doc (239k-char Beige Book) fits in one call.
- Thinking on everywhere: 4-5x slower and leaked its reasoning into the reply once in 16 calls.
  On for landmark only: BoE 5-3-1 split went 0/3 -> 3/3; landmark calls take 35-140 s.
- Temperature 0 is NOT deterministic on this endpoint; length swings 5-15 bullets on one doc, and
  the short runs are the ones that drop facts.

Open:
1. CFTC direction is wrong in 3/3 runs ("decrease" in net short when it grew): compute CFTC
   summaries in code instead of with the model.
2. Minutes checklist lacks "forward-guidance language agreed for the statement" (misses
   "can be patient" on t2-F1-patient-dropped-2015, 3/3).
3. Landmark allows only one quote (misses Powell's "sufficiently restrictive", 3/3).
4. Random misses: a per-field form with a code check for empty fields, instead of free bullets.

## Summary-coverage eval (2026-09-22/23)

A fixed answer key says what each document's important points are, so a summary run can be scored
for **zero tokens** and the same way every time. Built once by Sonnet subagents, then hardened.
Everything lives in `tools/summary_eval/` — see its README for the format and the pattern rules.

- **58 documents** (~8 per doc_type across the size range, including the 16 old FACTS docs),
  **594 points**, 0 validator errors. One point = one fact.
- Each point carries 3 paraphrases and 2 decoys. The validator proves the patterns catch the
  paraphrases, reject the decoys and match the source, so a key change cannot quietly rot.
- `tools/eval_summaries.py` scores against the key (must-recall, per-kind, per-run stability,
  `--misses`, `--json`). The old 2-facts-x-16-docs FACTS table is gone (it lives in the key now).

**Is a keyword check trustworthy?** Measured, not assumed: readers graded 221 points on real
summaries by meaning, blind to the patterns. Regex vs reader agreement **98%** (number 98, concept
98, name 100, phrase 100), 1 too loose / 3 too strict. It was 84% before two fixes that mattered:
1. **Compound points** — "4.0% y/y, the smallest since March 2021" scored full credit for half the
   fact. Split into atomic points (441 -> 594); every "too loose" case came from these.
2. **`not_patterns` leaked across bullets** — "downside risks to growth. - Inflation has remained
   elevated" tripped an inflation guard. `covers()` now judges one bullet at a time.
   Also: `[^.]{0,60}` windows die at the first decimal (262 patterns), and bare numbers need
   `(?<!\d)` guards or "5 billion" matches inside "15 billion".

**Baseline of the stage-1 prompts as of 4b591e6** (58 docs x 3 runs): must-recall **74%**, all-point
66%. fomc_statement 97, macro_release 93, landmark 81, corporate_8k 80, beige_book 73,
positioning 60, cb_speech 59, **fomc_minutes 41**. By kind: phrase 70, number 68, concept 65,
**name 51** — dissenters and named districts are dropped far more often than numbers.

### Prompt changes made against it

**fomc_minutes 41% -> 78% must-recall, 33% -> 78% all-point** (8 docs x 3 runs). Not a length
problem: the 94k-char minutes produced 14 bullets and the 49k one produced 6. The old checklist
asked only for discussion (participants' views, staff outlook, risks) and the model obeyed —
the May 2022 summary never stated the 50bp hike. Rebuilt decision-first like the statement
checklist, then added the figures (PCE, unemployment, payrolls), operations with amounts, named
causes, non-rate dissents and market pricing. Numbers went 59% -> 91%.
**Generalization check** (`tools/probe_minutes.py`, key-free probes derived from each source): on
10 minutes the key never saw, v1 -> new: rate level 38 -> 57%, dissenter named 11 -> 50%, PCE figure
0 -> 67%, unemployment figure 10 -> 100%, no invented numbers.

**Quote length rule** (shared frame): "quote exactly" made the model paste whole statement
paragraphs; replies of 9k chars tripped the 8k runaway guard and whole summaries were discarded.
"A phrase, never a whole paragraph" fixed most of it, not all: with the final prompt
`fomc-minutes-20150917` still comes back at 8.2-9.7k chars in 3 of 3 runs and is thrown away
(the guard's error says "reasoning leaked"; it is long bullets). Final minutes numbers, 8 docs x
3 runs, `out/min_final_r*.json`: **82% must, 80% all-point** (85/81/81), 7/8 docs per run.

**Thinking off everywhere** (was on for landmark). House rule: at most 4,000 output tokens per
request, reasoning included; landmark thinking needed 16,000. At 4,000 with thinking on, 4 of 8
landmark docs came back empty. Thinking off: landmark **81% -> 76%**, the loss all in names (BoE
5-3-1 split 3/3 -> 1/3) — the landmark checklist has no line asking for the vote.

### Second pass, 2026-09-24 — all types, with a dev/test split

`tools/summary_eval/split.json` halves the key by doc_type (hash of doc_id). Tuning looked only at
dev-half misses; the test half was scored once per version. Final = 58 docs x 3 runs
(`out/v6_r*.json`, scores in `out/eval_v6.json`; CFTC rows are the computed output).

| | before (committed) | final | test half before -> after |
|---|---|---|---|
| **overall must-recall** | 77% | **83%** (84/80/83) | 77 -> 83% |
| beige_book | 73 | **84** | 72 -> 89 |
| landmark | 76 | **86** | 76 -> 78 (dev 77 -> 95) |
| positioning_report | 60 | **88** | 72 -> 83 |
| fomc_minutes | 82 | 83 | 94 -> 93 |
| macro_release | 93 | 95 | 93 -> 91 |
| fomc_statement | 97 | 94 | 100 -> 100 |
| cb_speech | 59 | 56 (reverted) | 44 -> 58, dev 76 -> 52 |
| corporate_8k | 80 | 72 (reverted) | 67 -> 57, dev 88 -> 82 |

What changed and stayed:
- **CFTC is computed, not summarized** (`cot_summary()`): latest net/long/short/OI, week-over-week
  move with its direction, gross moves, window extremes, largest one-week swing. Deterministic,
  zero tokens, sign always right — the model wrote a deepening net short as "a decrease of 883".
  88% must vs the model's 60%. The checklist is only the fallback if a table fails to parse.
- **landmark**: the vote line (every dissent named + preferred action) and the decision level.
  76 -> 86% with thinking OFF — above the 81% thinking-on figure.
- **beige_book**: District counts, survey figures with their source, Districts named as diverging.
  Numbers had 14% recall; 73 -> 84%.
- **Runaway guard 8k -> 12k chars** (~3k tokens, inside the 4,000 cap). Zero failures in 174
  calls; largest legitimate reply 9,842 chars would have been discarded before.
- **Tried and dropped**: a 40-words-per-bullet bound (fixed the guard problem but cost minutes
  94 -> 80 and macro 93 -> 82 on the test half — number-dense bullets need the room); the cb_speech
  and corporate_8k checklist additions (no gain / a loss, see table). Both reverted to the text
  that was measured.

### Third pass, 2026-09-24 — speeches, with 12 more keyed docs

**Diagnosis first, by position.** Coverage of the 8 original speeches by where the fact sits in
the text: opening 64%, second quarter 42%, **third quarter 14%**, close 60% — the model summarizes
intro and conclusion and skips the body. And the old exit clause ("mainly not about policy: say
so and keep only the policy-relevant points") collapsed non-policy speeches: Dudley on trade got
750-1,170-char summaries and 1 of 24 facts. Two generic lines: cover the body section by section;
for a non-policy speech, say so and then still cover its economic content.

**Keyed set 8 -> 20 speeches** (Opus subagents, `manifest.json` / `split.json` updated, 70 keys,
0 errors). Old prompt on the 12 new ones: 72% — so the original 8 were a hard draw (Dudley,
Gieve, Wilkins), and "speeches at 56%" overstated it.

| speeches | before | after |
|---|---|---|
| all 20, must-recall | 66% | **73%** |
| all 20, all-point | 55% | **67%** |
| numbers | 60% | **82%** |
| the 12 unseen by any prompt | 72% | 76% |
| test half (10) / dev half (10) | 78 / 26 | 90 / 38 |

Also in this pass: **"use the full allowance; do not stop early"** in the shared frame (the model
wrote 8-12 of 15 bullets and dropped late items). Overall up on both halves (test 86 -> 87, dev
80 -> 84); one runaway reply in ~200 calls (a long speech hit the 4k-token cap; the retry got
it). **Tried and reverted:** a macro line for the comparisons a release makes ("smallest since
March 2021") — 91 -> 86% test, 100 -> 94% dev; kept as a comment. Macro re-measured with it
reverted but the frame line kept: **87%** (v6: 95%) — the "full allowance" line costs macro ~8
points while lifting minutes 83 -> 86, beige 84 -> 89, corporate_8k 72 -> 80, landmark 86 -> 88
and speeches; kept on the overall rule (test 86 -> 88, dev 80 -> 80).

**Final, 70 docs x 3 runs (`out/eval_final.json`): must-recall 83% (82/85/83), all-point 74%;
test half 88%, dev half 80%.** By type: statement 97, beige 89, landmark 88, CFTC 88, macro 87,
minutes 86, corporate_8k 80, cb_speech 73.

**Request budget (2026-09-24).** House rule: **25 admitted requests per unit**, ≤4,000 output
tokens each, and a failed call or a retry spends a slot too (README "House API allocation";
CHANGELOG). The busiest unit needs 16 calls with no retries; the old retry logic could spend up
to 8 per document. `text_signal._Budget` now counts every attempt per unit, stage 1 may not eat
the one slot reserved for stage 2, and the oldest documents are the ones dropped if it runs out.
No behaviour change while nothing fails. CFTC (computed) and pass-through docs cost nothing.

Open (prompt work):
1. **cb_speech 56%** is the one unsolved type. Speeches are the least uniform class (some carry
   no policy content); a checklist tweak did not move it. Next idea: a two-line classifier pass
   (policy speech vs not) before the checklist, or a `default`-style short list for non-policy ones.
2. **corporate_8k** has 5 docs / 21 test points — one miss is 5 points. Needs more keyed docs
   before any conclusion.
3. Guard label was "reasoning leaked"; it is now "runaway reply". The one real leak seen (29k
   chars) was with thinking ON, which is now off everywhere.
4. 3 runs, always. Per-pass overall was 84/80/83 on identical inputs.
5. Higher recall has not yet been shown to move the composite score — run the scorer old vs new
   over the 90 realized units next.
