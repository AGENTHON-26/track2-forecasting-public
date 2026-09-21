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
