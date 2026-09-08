"""Card-level scenario prompt.

Asks the model for a SHARED set of scenarios that apply consistently across every target asset
in the card, rather than one independent adjustment per asset. See
docs/SOLVER-PLAYBOOK.md section 3 ("Scenarios (shape)") and docs/CATEGORIES.md's F3 guidance:
"the text should inform the joint scenario ... ALL tenors are elevated" -- a scenario describes
ONE POSSIBLE WORLD, and every asset's own drift/vol is specified together within it, so a
single per-draw scenario choice (agent/core/adjust.py) moves every asset consistently rather than
each tenor picking its own story independently.

Reply shape asked for:

    {"scenarios": [
      {"weight": 0.6, "basis": "inferred", "because": "...",
       "assets": {"<asset>": {"drift_bp": .., "vol_scale": ..}, ...}},
      ...
    ]}

`basis` is "established" (the text states a fact already known -- may narrow the distribution)
or "inferred" (read from tone -- may shift the center, must never narrow). That rule is
enforced in agent/core/adjust.py as a hard clamp, not just requested here -- a model can ignore a
prompt instruction, so the enforcement has to live in code that runs regardless of what the
model does.
"""

from __future__ import annotations

from typing import Any


def build_scenario_prompt(
    assets: list[str],
    horizons: list[int],
    asof: str,
    target_type: str,
    last: dict[str, float],
    sd_h: dict[str, float],
    docs: list[dict[str, Any]],
) -> str:
    lines = [
        "You are adjusting a statistical forecast using dated documents.",
        f"As-of date: {asof}. Nothing after this date is known to you.",
        f"Target type: {target_type}. Horizons (business days): {horizons}.",
        "",
        "Per asset: the level at the as-of date, and the statistical standard deviation of the",
        f"forecast at the longest horizon ({max(horizons)} business days):",
    ]
    lines += [f"  {a}: level {last[a]:.6f}, horizon sd {sd_h[a]:.6f}" for a in assets]
    lines += ["", f"Documents ({len(docs)}), newest first:"]
    for d in docs:
        lines += [f"--- {d['doc_id']} ({d['timestamp']}, {d['doc_type']}) ---", d["text"], ""]
    lines += [
        "Describe your view as 1 to 4 SCENARIOS -- possible futures, not a single averaged",
        "guess. Use more than one scenario whenever the documents describe a branching outcome",
        "(e.g. hike vs hold, crisis resolves vs escalates). Each scenario applies to ALL assets",
        "together: within one scenario, every asset's numbers should be consistent with the",
        "SAME story (e.g. if the scenario is 'Fed stays restrictive', every rate tenor should",
        "reflect that, not just one).",
        "",
        "For each scenario, give:",
        "  weight    : probability of this scenario; weights across all scenarios should sum",
        "              to approximately 1.0.",
        '  basis     : "established" if a document STATES this as fact (a decision taken, a',
        "              number released, a level announced) -- may justify a NARROWER",
        "              distribution than the statistical baseline.",
        '              "inferred" if you are reading TONE or implication, not a stated fact --',
        "              may shift the center but must NOT be used to narrow the distribution.",
        "  because   : one sentence citing a doc_id.",
        "  assets    : for EACH asset, two numbers:",
        "    drift_bp  : shift in this scenario, in basis points of the MAGNITUDE of the",
        "                current level. Positive means up regardless of the sign of that level.",
        "                The resulting shift is clamped to +-3 horizon standard deviations.",
        "    vol_scale : multiplier on the statistical standard deviation. >1 widens, <1",
        '                narrows -- narrowing only takes effect when basis is "established".',
        "",
        "Both numbers must be finite. NaN and Infinity are rejected and the scenario is dropped.",
        "",
        "Reply with JSON only, no prose:",
        '{"scenarios": [',
        '  {"weight": <float>, "basis": "established"|"inferred", "because": "<sentence>",',
        '   "assets": {"<asset>": {"drift_bp": <float>, "vol_scale": <float>}, ...}}',
        "]}",
    ]
    return "\n".join(lines)
