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

## F4 — tail/shock cards (2026-09-26)

Scored in-process with the real composite (`qfbench2_track_forecasting.scoring._composite`, card
weights, single-cell renormalisation, seed 0, 500 draws) over the 29 realized F4 units. The numbers
reproduce `eval_reports/*.json` exactly for the text-off arm.

**What F4 is scored on.** All but two F4 cards are single-cell, so the composite is
0.714 x CRPS + 0.286 x tail, and the tail term is `pinball` — a distance, not coverage. Text-off,
10 of 32 F4 cells fall outside the random walk's own 1%/99% (hike-cycle-2021Q4b +8.1 sigma,
nok-covid-2020 +6.3, chf-floor-strain-2015 -6.1). A too-narrow tail is charged in proportion.

**What the LLM had been doing.** With replies that actually arrive, the deployed stage 2 gives
vol_scale 0.8-1.3 (cap 2.0) and |skew| <= 0.3 on every F4 card, and on chf-floor it narrowed and
pointed the wrong way. Direction is right 19 / wrong 9 overall but **6 right / 5 wrong on the
|z|>2 cards** — a coin flip where it matters. In the 2026-09-22 report the six worst F4 units
scored identical to text-off: those replies had failed or truncated (thinking on can hit
max_tokens mid-JSON; seen on factor-stress-2008).

| arm (29 units) | mean | wins/losses vs walk |
|---|---|---|
| random walk, text off | 0.4563 | — |
| walk, widen x1.5 (text off) | 0.4126 | 17/12 |
| walk, widen x2 (text off) | 0.3908 | 16/13 |
| LLM answers as deployed | 0.4111 | 15/14 |
| LLM answers + F4 floor 1.5 | **0.3878** | **18/11** |
| LLM answers + F4 floor 2.0 | 0.3714 | 16/13 (7/8 on one half) |
| Student-t innovations alone | 0.4782 | 9/20 |
| M2 on F4 level cards | 0.4498 | 10/8 |
| directional jump keyed on the LLM's sign | 0.3642 | 11/18 |

Kept: **F4 widen floor 1.5** in `build_draws` (`_FAMILY_WIDEN_FLOOR`) — helps on both halves of an
odd/even split, costs the 15 calmest cards 0.0234 -> 0.0247, and the LLM's own widen still applies
above it. Also kept: one no-thinking retry when the stage-2 reply does not parse
(`_STAGE2_RESERVE` 1 -> 2). Not kept: skew on (+0.01 worse), directional jumps (mean improves,
most units lose), M2 for F4, and a v2 "shock block" prompt (explicit shock_prob / size, vol cap 3):
without thinking the 120B mostly returns the template zeros ("no shock signals; routine") and
overshoots when it fires.

Endpoint note: 36 rpm with 8 workers hit 429s and a 40-minute read timeout partway through a 31-unit
sweep; 12 rpm / 3 workers was clean. 429 retries spend the 25-request unit budget and end in neutral.

### F4 prompt v3 and thinking (2026-09-26, later)

After dev's #14 merge (thinking off, median-of-3), stage 2 answered exact neutral on 24 of 29 F4
cards, so text added nothing on F4. Thinking is now ON for F4 only (`_STAGE2_THINKING_BY_FAMILY`),
same 4,000-token cap; a reply cut mid-JSON is re-asked once without thinking (seen 1 in ~130).

All rows: 29 realized F4 cards, random walk base, 3 model runs x 3 draw seeds, lower is better.
Summaries cached, so only the stage-2 prompt differs between rows.

| arm | mean | even half | odd half | wins / losses vs floor |
|---|---|---|---|---|
| random walk alone | 0.4560 | 0.3360 | 0.5846 | 12 / 17 |
| + floor 1.5 (text off) | 0.4085 | 0.3011 | 0.5236 | — |
| + previous F4 paragraph | 0.4200 | 0.2815 | 0.5684 | 9 / 20 |
| + width-only (routine/warning/shock) | 0.3898 | 0.2889 | 0.4978 | 13 / 15 |
| + v3, all three values (shipped) | 0.3562 | 0.2492 | 0.4708 | 15 / 11 |
| v3, width + drift only | 0.3525 | 0.2425 | 0.4704 | 15 / 11 |
| v3, drift only (width left to floor) | 0.3539 | 0.2429 | 0.4728 | 14 / 12 |
| v3, width + skew only | 0.4077 | 0.3054 | 0.5172 | 9 / 17 |

Why v3 works and its caveat: the comment above `_FAMILY_FOCUS["F4"]` in `text_signal.py`.
Short version: meanings for the width numbers, market-structure rules for direction instead of
central-bank tone, and the floor making the direction bet affordable. The gain sits on JPY carry
2007, JPY crowding 2024 and SVB 2023, and the rules were written after seeing those cards.

Also measured, not kept: M2 as the F4 base (18 level cards; tied with the walk once averaged over
seeds: 0.6523 vs 0.6489 text off). More draws (1,000-5,000) do not change the expected score, but
at 500 draws identical forecasts score 0.398-0.426 across seeds, so a single-seed F4 difference
under ~0.03 is noise.

### F4 prompt v4 (shipped) and the floor as a fallback (2026-09-26, latest)

v3 still called four shock cards "routine" although the warning was in their summaries, as one
bullet among many routine ones: NOK covid 2020 and covid rates 2020 (coronavirus hitting
markets), taper warning 2013 (reducing the pace of purchases), US downgrade watch 2011 (debt
limit). v4 = v3 + one paragraph: judge width by the single most alarming bullet, one real warning
is enough for 2.0+, "on hold" does not cancel it. Same protocol as above.

| arm | mean | even half | odd half |
|---|---|---|---|
| random walk | 0.4560 | 0.3360 | 0.5846 |
| v3, no floor | 0.3603 | 0.2469 | 0.4818 |
| v4, no floor | **0.3296** | 0.2351 | 0.4309 |
| v4 + always-on floor 1.5 | 0.3299 | 0.2352 | 0.4313 |
| v4 + agreement gate on drift | 0.3291 | 0.2292 | 0.4363 |
| shipped code path (median-of-3, fallback floor) | 0.3277 | | |

Width answers of 2.0+: v3 6%, v4 69%. The four missed cards now get 1.8-2.5. Direction on the cells
that moved more than 2 sigma: previous prompt 3 right / 7 wrong, v3 6 / 4, v4 10 / 1.

Decisions: the 1.5 floor is now a FALLBACK (`forecast_models._text_was_silent`): it applies only
when every asset on the card came back exactly neutral, i.e. the text half failed. That keeps
0.4085 instead of 0.4560 on a text-less F4 card and never overrides an answer (fires on 1 of 29
cards with v4 answering, a calm one). The agreement gate was not built: it moved the mean by
0.0005. Caveat: v4's examples are the cards that failed, on the same cards measured; expect less
on unseen cards. The F1 M2-centre context (`_MODEL_CONTEXT_ON`) ships OFF until an F1 sweep
measures it.

### F4 prompt v5 (shipped 2026-09-28) -- chosen over v4 for overfitting

v4's example list named the very shocks it had missed on these cards. Splitting the 29 cards by
whether they were used to write the rules exposes it (3 model runs x 3 draw seeds):

| setup | all 29 | even | odd | 11 shock cards used for the rules | 18 other cards |
|---|---|---|---|---|---|
| random walk | 0.4560 | 0.3360 | 0.5846 | 1.1256 | 0.0468 |
| v4 | 0.3297 | 0.2352 | 0.4310 | 0.7713 (10 better / 1 worse) | 0.0598 (7 / 11) |
| **v5 (shipped)** | **0.3781** | 0.2621 | 0.5025 | 0.9175 (8 / 3) | **0.0485 (9 / 9)** |
| v5 + always-on floor 1.5 | 0.3715 | 0.2601 | 0.4908 | 0.8986 (9 / 2) | 0.0493 (7 / 11) |
| v5 + drift x2 in code | 0.3578 | 0.2381 | 0.4862 | 0.8420 (7 / 4) | 0.0620 (7 / 11) |
| v5 + WARNING item in stage-1 checklists | 0.3677 | 0.2704 | 0.4719 | 0.8813 (9 / 2) | 0.0538 (8 / 10) |

Shipped code path (median-of-3, fallback floor): 0.3713.

v5 describes shocks by kind (no event, country or date from the practice cards), adds a
RELEVANCE rule (the warning must reach this asset through a stated channel; factor portfolios are
hedged against broad macro warnings), and keeps the width meanings and the market-structure
direction rules. It is even with the random walk on the 18 cards not used to write any rule and
keeps about 60% of v4's gain on the shock cards. None of the add-ons passed: the floor and the
doubled drift bring the calm-card cost back, and the WARNING checklist item helps shock cards but
flags generic risks (cyber attacks, bank NPLs) and is worse on one half. Where v5 still misses:
NOK Feb 2020 and the 2013 taper warning (warning seen, not connected to the asset), and 2008
10-year funding stress (the "stress sends yields down" rule does not hold there).

Re-measured on the corpus after #41 (59 documents added, 86 F4 files touched), same protocol:

| setup | all 29 | even | odd | 11 shock cards | 18 other cards |
|---|---|---|---|---|---|
| random walk | 0.4560 | 0.3360 | 0.5846 | 1.1256 | 0.0468 |
| v5 | 0.3589 | 0.2390 | 0.4873 | 0.8718 (10 better / 1 worse) | 0.0454 (8 / 10) |
| v5, shipped code path (median-of-3) | 0.3431 | 0.2310 | 0.4633 | 0.8343 | 0.0429 |

v5 improves on the new corpus (0.3781 -> 0.3589) and stays even with the random walk on the
18 cards no rule was written from. v4 was not re-run on the new corpus.
