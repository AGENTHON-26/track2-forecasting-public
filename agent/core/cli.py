"""Track-2 submission agent.

Three pieces, none of them the shipped reference code (that's imported unchanged, never
edited):

- Statistical half: `agent.core.engine`'s joint block-bootstrap (Phase 3) instead of the reference
  `cli.py`'s Gaussian random walk.
- Reasoning half: `agent.core.prompt`'s card-level scenario prompt and `agent.core.adjust`'s scenario
  application (Phase 4) instead of `baselines.reasoning_agent`'s flat single-adjustment
  contract. Corpus reading (`read_corpus`) and the model HTTP call (`call_model`) ARE reused
  unchanged from `baselines.reasoning_agent` -- those don't depend on the reply's shape.

Harness contract: `forecast --panels ... --text ... --asof ... --out ...`, no `--card` flag
(see SUBMISSION_CLI.md). `card.toml` is auto-detected under `--panels`, then its parent,
matching `qfbench2_track_forecasting/cli.py`'s own fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.core.adjust import apply_scenarios  # noqa: E402
from agent.core.engine import draw as _draw  # noqa: E402
from agent.core.prompt import build_scenario_prompt  # noqa: E402
from baselines.reasoning_agent import _MIN_DRAWS  # noqa: E402
from baselines.reasoning_agent import call_model  # noqa: E402
from baselines.reasoning_agent import read_corpus  # noqa: E402
from qfbench2_track_forecasting.cli import _read_panels  # noqa: E402


def _find_card(panels_dir: pathlib.Path) -> pathlib.Path:
    for cand in (panels_dir / "card.toml", panels_dir.parent / "card.toml"):
        if cand.exists():
            return cand
    raise SystemExit(
        f"card.toml not found in {panels_dir} or {panels_dir.parent}; pass --card explicitly"
    )


def main(argv: list[str] | None = None) -> int:
    import tomllib

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--panels", type=pathlib.Path, required=True)
    ap.add_argument("--text", type=pathlib.Path, required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--card", type=pathlib.Path, default=None)
    ap.add_argument("--n-draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    card_path = a.card or _find_card(a.panels)
    card = tomllib.loads(card_path.read_text(encoding="utf-8"))
    t = card["targets"]
    assets, horizons = list(t["asset_ids"]), [int(h) for h in t["horizons"]]

    panels = _read_panels(a.panels)
    n_draws = max(a.n_draws, _MIN_DRAWS)
    if n_draws != a.n_draws:
        print(f"note: --n-draws {a.n_draws} raised to the contract floor {_MIN_DRAWS}")
    samples, draw_meta = _draw(panels, assets, horizons, a.asof, n_draws, a.seed)
    last = {x: float(draw_meta["last"][x]) for x in assets}
    sd_h = {x: float(draw_meta["daily_sd"][x]) * math.sqrt(max(horizons)) for x in assets}

    docs, excluded, truncated = read_corpus(a.text, a.asof)
    if not docs:
        parsed, reason, _trace = None, "no corpus document is dated at or before the as-of date", ""
    else:
        parsed, reason, _trace = call_model(
            build_scenario_prompt(assets, horizons, a.asof, t["target_type"], last, sd_h, docs)
        )

    scenario_report: dict = {"scenarios": []}
    matched = 0
    if parsed is None:
        reasoning_applied = False
    else:
        # A seed distinct from the engine's own (`a.seed`) -- the scenario assignment is an
        # independent random choice from which historical blocks got bootstrapped, and reusing
        # the same seed would correlate the two for no reason.
        adjusted, scenario_report, matched = apply_scenarios(
            samples, assets, last, sd_h, parsed, seed=a.seed + 1_000_003
        )
        if matched == 0:
            keys = sorted(parsed)[:8] if isinstance(parsed, dict) else []
            reasoning_applied = False
            reason = (
                f"reply named none of the requested assets {assets} in any scenario; "
                f"its top-level keys were {keys}"
            )
        else:
            samples, reasoning_applied, reason = adjusted, True, ""

    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"draw": d, "asset": x, "horizon": h, "value": float(samples[d, i, j])}
        for d in range(samples.shape[0])
        for i, x in enumerate(assets)
        for j, h in enumerate(horizons)
    ]
    pd.DataFrame(rows).to_parquet(out, index=False)

    engine_name = draw_meta.get("engine", "unknown")
    n_scenarios = len(scenario_report.get("scenarios", []))
    (out.parent / "forecast_meta.json").write_text(
        json.dumps(
            {
                "unit_id": card["task"]["id"],
                "asof": a.asof,
                "representation": "samples",
                "asset_ids": assets,
                "horizons": horizons,
                "n_draws": n_draws,
                "target": t["target_type"],
                "reasoning_applied": reasoning_applied,
                "reasoning_skipped_reason": reason if not reasoning_applied else "",
                "engine": engine_name,
                "n_scenarios": n_scenarios,
                "rationale": {
                    "file": "forecast_rationale.md",
                    "method": (
                        f"{engine_name} + {n_scenarios}-scenario mixture"
                        if reasoning_applied
                        else f"{engine_name}, unadjusted (reasoning skipped)"
                    ),
                    "documents_read": len(docs),
                    "documents_excluded_by_cutoff": excluded,
                    "documents_over_prompt_budget": truncated,
                },
            },
            indent=2,
        )
    )

    scenario_lines: list[str] = []
    for si, sc in enumerate(scenario_report.get("scenarios", [])):
        scenario_lines += [
            f"### Scenario {si + 1}: weight {sc['weight_requested']:.2f} requested / "
            f"{sc['weight_realized']:.2f} realized -- basis: **{sc['basis']}**",
            "",
            f"{sc['because']}" if sc["because"] else "_(no citation given)_",
            "",
            "| asset | drift_bp | vol_scale | shift | note |",
            "|---|---|---|---|---|",
        ]
        scenario_lines += [
            f"| {asset} | {v['drift_bp']:+.1f} | {v['vol_scale']:.2f} | {v['shift']:+.6g} | {v['note']} |"
            for asset, v in sc["assets"].items()
        ]
        scenario_lines.append("")

    (out.parent / "forecast_rationale.md").write_text(
        "\n".join(
            [
                f"# Forecast rationale -- {card['task']['id']}",
                "",
                f"**As of {a.asof}. Assets: {', '.join(assets)}. Horizons: {horizons}.**",
                "",
                "## Statistical half",
                "",
                f"Joint block bootstrap: {draw_meta.get('block_size')}-day blocks, "
                f"{int(draw_meta.get('recent_weight', 0) * 100)}% drawn from the most recent "
                f"{draw_meta.get('recent_window')} days and the rest from the full "
                f"{draw_meta.get('n_history_rows')}-row history. Each block resamples a real "
                "historical day's joint change across every target asset at once, so both fat "
                "tails and cross-asset correlation come from actual market history rather than "
                "a fitted Gaussian shape (agent/core/engine.py).",
                "",
                "## Reasoning half",
                "",
                f"- documents read: **{len(docs)}** (dated <= {a.asof})",
                f"- documents excluded by the cutoff or a missing index entry: **{excluded}**",
                f"- documents dropped by the prompt budget: **{truncated}**",
                f"- assets named by the reply: **{matched} of {len(assets)}**",
                f"- adjustment applied: **{reasoning_applied}**",
                *([f"- skipped because: {reason}"] if not reasoning_applied else []),
                "",
                "Each scenario below describes ONE possible future and is applied to its own "
                "share of the draws (per the realized weight); together they form the mixture. "
                'A scenario\'s `basis` is "established" (a stated fact -- may narrow the '
                'distribution) or "inferred" (read from tone -- may only shift it, never '
                "narrow it below the statistical floor).",
                "",
                *(scenario_lines if scenario_lines else [
                    "No scenario was applied; the numbers above are the statistical floor.",
                ]),
                "## What would change this",
                "",
                "A document dated at or before the as-of date that contradicts the cited ones. "
                "Anything after that date is not knowable here and was not read.",
            ]
        )
        + "\n"
    )

    print(f"wrote {out.name} + sidecars to {out.parent} (engine={engine_name})")
    print(f"  {len(assets)} asset(s) x {len(horizons)} horizon(s), {n_draws} draws")
    print(
        f"  corpus: {len(docs)} read, {excluded} excluded, {truncated} over budget"
        f" · reasoning_applied={reasoning_applied} ({n_scenarios} scenario(s))"
        + (f" ({reason})" if not reasoning_applied else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
