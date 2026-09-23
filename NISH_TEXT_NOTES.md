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

Open (prompt work):
0. **The 8k-char guard vs long bullets**: one minutes doc loses its whole summary 3/3. Either cap
   bullet length in the prompt or lift `_MAX_SUMMARY_CHARS` (9.7k chars is ~2.4k tokens, inside
   the 4k rule) — and rename the "reasoning leaked" error, which is now usually wrong.
1. **landmark**: add the vote / named-dissent line the statement checklist has; expect the 5 points back.
2. **cb_speech 59%**: same disease as minutes — no line for the figures a speech cites, nothing
   forcing the commitment to be quoted.
3. **positioning 60%**: direction wrong ("decrease of 883" for a net short that grew). Compute CFTC
   changes in code, not with the model.
4. 157 of 585 points are never covered in any run (`--misses` lists them with the quote);
   149 flip between runs. Always compare 3 runs.
5. Local runs need `--patience` or build.nvidia.com 429s the whole run away.
