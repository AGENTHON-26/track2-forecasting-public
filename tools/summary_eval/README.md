# Summary-coverage eval

Does a stage-1 summary contain the important points of its document? An answer key written once by
Sonnet says what those points are; checking a summary against it costs **zero tokens** and gives the
same answer every time, so prompt changes can be compared without paying for a judge.

## Layout

| path | what it is |
|---|---|
| `manifest.json` | the 58 documents the eval runs on (≈8 per doc_type across the size range, plus the 16 legacy test docs). Built by `tools/build_summary_evalset.py`. |
| `key/<doc_id>.json` | the answer key: the important points of that document, with regex patterns, and the tests that prove the patterns work. |
| `src/<doc_id>.txt` | derived, gitignored: each document as the summarizer sees it (CFTC reduced to the main contract), wrapped so it can be read in pages. Rebuild with `python3 tools/summary_key.py`. |
| `audit/<doc_id>.json` | a reader's covered/missed labels for one run, used to measure whether the regex agrees with a human judgement. |

## A point

```json
{"id": "p1", "point": "Headline CPI rose 4.0% over the 12 months to May",
 "tier": "must", "kind": "number",
 "patterns": ["(headline|all items|\\bcpi\\b).{0,80}(?<!\\d)4\\.0(?!\\d)\\s*(percent|%)"],
 "match": "any", "not_patterns": ["(fell|declin\\w*).{0,40}(?<!\\d)4\\.0(?!\\d)"],
 "quote": "the all items index increased 4.0 percent for the 12 months ending May",
 "paraphrases": ["CPI +4.0% y/y in May", "...", "..."],
 "decoys": ["Headline CPI fell 4.0% y/y in May", "Headline CPI rose 4.9% y/y in May"]}
```

- `tier` — `must` (the summary is wrong without it) or `should`.
- `kind` — `number`, `name`, `phrase`, `concept`, or `judge_only` (kept and reported, never scored).
- `patterns` / `match` / `not_patterns` — how a summary is checked. Case-insensitive, `.` spans
  newlines, and the text is normalized first (unicode dashes and quotes, soft hyphens, whitespace).
- `paraphrases` (3) and `decoys` (2) — the tests. Every paraphrase must be covered and no decoy may
  be, which is what keeps the patterns honest.
- `source_pattern` — optional, for table documents (CFTC) where the column label is nowhere near the
  number: the validator checks the source with this instead of `patterns`.

**One point states one fact.** A point asserting a level *and* its change, or a figure *and* a
superlative, lets a half-right summary score full credit — this was measured, and it was the single
biggest source of disagreement with human readers.

## Writing patterns (learned the hard way)

- Windows are `.{0,80}`. Never `[^.]{0,60}` — financial text is full of decimals ("0.2 percentage
  point to 63.4 percent") and the window dies at the first period. Never `[^;]` either; paraphrases
  contain semicolons.
- Guard bare numbers with `(?<!\d)` / `(?!\d)`, or "5 billion" matches inside "15 billion".
- Anchor a number to its subject; never match a bare short number.
- `not_patterns` must cover both word orders ("lumber … accelerated" and "accelerated … lumber").
- Cover the synonyms a terse summarizer actually writes: jobless rate, factory payrolls, voted
  against, short side, +261k, ex-food-and-energy.

## Commands

```bash
python3 tools/build_summary_evalset.py                 # rebuild manifest.json (--all for every doc)
python3 tools/summary_key.py                           # rebuild src/ from the units
python3 tools/validate_summary_key.py [DOC_ID ...]     # check the key: quotes, paraphrases, decoys

python3 tools/try_summaries.py --manifest tools/summary_eval/manifest.json \
        --workers 2 --patience --out out/run_r1.json   # summarize the eval set (repeat 3x)
python3 tools/eval_summaries.py out/run_r1.json out/run_r2.json out/run_r3.json \
        --misses --json out/eval_run.json              # score: must-recall, stability, missed points

python3 tools/audit_key.py prepare out/run_r1.json --batches 4   # then a reader labels audit/*.json
python3 tools/audit_key.py compare out/run_r1.json               # regex vs reader agreement

python3 tools/probe_minutes.py out/hold_v1.json out/hold_v2.json  # key-free check on UNSEEN minutes
```

`probe_minutes.py` is the overfitting guard: it derives what to look for from each source (the
rate level it states, the names in its vote paragraph, the PCE and unemployment figures) so a
prompt change can be tested on documents that have no key.

`--patience` stretches 429 backoff from 17s to minutes: build.nvidia.com throttles hard, and without
it a whole run silently comes back empty. It is local-dev only and does not touch the scored path.

Run **three** passes and compare. Temperature 0 is not deterministic on this endpoint, and the
short runs are the ones that drop facts.
