# T2 scoring regression suite

Modeled on `track3-simulation-public/regression_suite`, which Track 2 did not have. The
motivating incident is recorded rather than implied: the CodaBench driver's **first real T2
invocation (2026-08-16) crashed with `KeyError: 'card'`** and, once patched, produced a **NaN
leaderboard** — defects that survived for weeks because nothing continuously scored a known
submission against a known unit.

## What is pinned

| fixture | what it locks |
|---|---|
| `units/reg-t2-daily` | 2-asset daily unit, 21-BD horizon — composite **0.225323241** to 1e-9 |
| `units/reg-t2-monthly` | monthly unit, horizon in **months** — composite **0.470657943**. Keeps the monthly path under test forever: the first 2026H2-B build shipped 15 macro cards whose business-day targets landed on no observation |
| `units/reg-t2-logreturn` | 2-factor daily unit whose panel rows are decimal **simple returns** and whose target is the 21-BD **cumulative log return** — composite **0.007077530**. Check 13 also runs the CLI and the shared baseline fallback on it and measures both: centre `h × mean(log1p r)` anchored at 0, spread `sd(log1p r) × sqrt(h)`, and the two agree. The last panel row is forced to +0.05 so a producer that anchors at the last row (the reference CLI before staging PR #14, public #2) lands visibly off-centre, and one that differences the rows inflates the spread by ~sqrt(2) — both measured on the unfixed CLI: centre 0.050 vs 0.016, spread ratio 1.38–1.41 |
| driver seam | both are scored through the **minimal ctx `{unit_dir, output_dir}`** — exactly what `score.py` passes — so the hydration path cannot silently regress |
| failure taxonomy | missing/empty rationale → g1 naming the file · missing meta → g0 · post-as-of doc → g2 · thin draws → g3 |
| public smoke | no `reference/` → admissible, score `None` |
| the CLI | fresh run on all three units passes the gates (values deliberately not pinned) |

Everything is **synthetic** — generated random walks, invented answers — so the fixtures carry no
market data and no leakage surface, and the pinned composite is a property of the scoring code
alone.

## Where the answer lives

The sealed answer is committed under `fixtures/<uid>/targets.parquet`, **not** in the unit tree:
the published repo must carry no `reference/` directory or `realized*` file for the pre-flip
sweep to trip on (`AGENTS.md` names both as forbidden patterns). Check 1 stages the fixture back
into a temp copy as `reference/realized.parquet` before scoring, so the real `_hydrate_ctx` path
is exactly what runs — the relocation is a publication-hygiene move, not a change to what is
tested. Check 0 fails CI if any answer-shaped path is ever committed under `regression_suite/`,
and `.github/validate_units.py` sweeps the same tree, since its per-unit firewall only reaches
`units/`.

## Why the golden forecast is committed, not regenerated

Scoring a fixed parquet exercises only deterministic arithmetic, so the composite pins to 1e-9.
Regenerating through the CLI at check time would couple the pin to numpy's `Generator` stream,
which is not guaranteed stable across versions — the same class of pinned-tool-vs-floating-dep
mismatch that kept this repo's CI red for a week in August. The CLI is smoke-tested separately,
without exact values.

## Use

```bash
python regression_suite/run_regression.py          # exit 0 iff everything holds
python regression_suite/build_reference.py --regen # after an INTENTIONAL scoring change
```

After a `--regen`, the diff of `expected.json` is the reviewable artifact: a scoring change that
cannot explain its composite drift in review should not merge.

CI wiring note: adding this to `.github/workflows/ci.yml` needs a push with `workflow` scope,
which the automation token here does not carry — one line for whoever holds it:
`python regression_suite/run_regression.py` after the toolkit install step.
