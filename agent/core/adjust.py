"""Applies a card-level scenario mixture to joint draws.

Two things this file enforces in CODE rather than trusting the prompt to be followed:

1. **Shared scenario per draw.** Each draw index is assigned to exactly one scenario (weighted
   by the model's stated probabilities), and that SAME assignment is used for every asset in
   that draw -- never resampled per asset. This is what makes "in this draw the Fed stays
   hawkish" apply consistently across the whole yield curve rather than independently per
   tenor. See docs/CATEGORIES.md's F3 guidance and agent/core/prompt.py.

2. **The established-vs-inferred clamp.** A scenario marked "inferred" is floored at
   vol_scale=1.0 regardless of what the model asked for -- tone-based reasoning is never
   allowed to buy extra confidence. Only "established" (a stated fact) may narrow. See
   docs/SOLVER-PLAYBOOK.md section 3 and docs/RATIONALE-REVIEW.md: CRPS punishes overconfidence
   far harder than being appropriately wide, and an over-narrow, oddly-precise distribution is
   exactly the signature that review process looks for in a memorized/backward-built forecast.

Backward compatible with the flat, single-adjustment shape from `baselines.reasoning_agent`
(either `{"assets": {...}}` or a bare `{asset: {...}}` dict) -- treated as one scenario, weight
1.0, basis "inferred". "Inferred" is the conservative default here on purpose: an old-style
reply that never mentioned scenarios or a basis should never be able to accidentally unlock
narrowing it never explicitly earned.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_VOL_FLOOR_INFERRED = 1.0
_VOL_FLOOR_ESTABLISHED = 0.5
#: Loosened from Phase 1-3's 2.0 ceiling -- F4 cards need room for genuinely extreme scenarios
#: (docs/CATEGORIES.md: "does your model's 99th percentile allow for a 5-10x move").
_VOL_CEILING = 4.0
#: Basis points are stated against the MAGNITUDE of the level, then clamped on the one scale
#: comparable across yields/FX/factors: horizon standard deviations. Same ceiling Phase 1-3 used.
_DRIFT_SD_CLAMP = 3.0
_MAX_SCENARIOS = 6


def _finite_float(x: Any, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _normalize_scenarios(parsed: Any, assets: list[str]) -> list[dict[str, Any]] | None:
    """New `{"scenarios": [...]}` shape, or the old flat per-asset shape treated as one
    scenario. Returns None if neither shape is recognizable."""
    if not isinstance(parsed, dict):
        return None

    if isinstance(parsed.get("scenarios"), list) and parsed["scenarios"]:
        out = []
        for s in parsed["scenarios"][:_MAX_SCENARIOS]:
            if not isinstance(s, dict):
                continue
            a = s.get("assets")
            out.append(
                {
                    "weight": s.get("weight", 0.0),
                    "basis": s.get("basis", "inferred"),
                    "because": s.get("because", ""),
                    "assets": a if isinstance(a, dict) else {},
                }
            )
        return out or None

    nested = parsed.get("assets")
    per_asset = nested if isinstance(nested, dict) and any(a in nested for a in assets) else parsed
    if isinstance(per_asset, dict) and any(a in per_asset for a in assets):
        return [{"weight": 1.0, "basis": "inferred", "because": "", "assets": per_asset}]
    return None


def apply_scenarios(
    samples: np.ndarray,
    assets: list[str],
    last: dict[str, float],
    sd_h: dict[str, float],
    parsed: Any,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], int]:
    """Returns `(adjusted_samples, report, matched_asset_count)`.

    `report["scenarios"]` carries, per scenario: the requested vs. empirically realized draw
    fraction (a sanity check that the weighted assignment behaved as asked), basis, citation,
    and the per-asset clamped drift/vol/shift actually applied.
    """
    scenarios = _normalize_scenarios(parsed, assets)
    if not scenarios:
        return samples, {"scenarios": []}, 0

    weights = np.array([max(_finite_float(s.get("weight"), 0.0), 0.0) for s in scenarios])
    if weights.sum() <= 0:
        weights = np.ones(len(scenarios))
    weights = weights / weights.sum()

    n_draws = samples.shape[0]
    rng = np.random.default_rng(seed)
    assignment = rng.choice(len(scenarios), size=n_draws, p=weights)

    # Centre computed ONCE from the original, unmodified samples -- reused for every scenario's
    # subset. Reading it back from `out` mid-loop would mix in earlier scenarios' edits, since
    # each scenario only touches its own (disjoint) subset of draws but the running mean of the
    # whole column would already reflect them.
    centre_by_asset = {
        a: samples[:, ai, :].mean(axis=0, keepdims=True) for ai, a in enumerate(assets)
    }

    out = samples.copy()
    matched_assets: set[str] = set()
    scenario_reports: list[dict[str, Any]] = []

    for si, scenario in enumerate(scenarios):
        mask = assignment == si
        basis = str(scenario.get("basis", "inferred")).strip().lower()
        if basis not in ("established", "inferred"):
            basis = "inferred"
        vol_floor = _VOL_FLOOR_ESTABLISHED if basis == "established" else _VOL_FLOOR_INFERRED

        per_asset_report: dict[str, Any] = {}
        for ai, a in enumerate(assets):
            spec = scenario["assets"].get(a) if isinstance(scenario.get("assets"), dict) else None
            note = ""
            if isinstance(spec, dict):
                matched_assets.add(a)
                drift_bp = _finite_float(spec.get("drift_bp"), 0.0)
                vol = _finite_float(spec.get("vol_scale"), 1.0)
            else:
                drift_bp, vol = 0.0, 1.0

            vol_clamped = min(max(vol, vol_floor), _VOL_CEILING)
            if vol_clamped != vol:
                note = f"vol_scale {vol:.2f} clamped to {vol_clamped:.2f} (floor {vol_floor:.1f} for basis={basis})"

            shift = abs(last[a]) * drift_bp / 10_000.0
            ceiling = _DRIFT_SD_CLAMP * sd_h[a]
            shift_clamped = shift
            if abs(shift) > ceiling:
                shift_clamped = math.copysign(ceiling, shift)
                clamp_note = f"drift clamped from {shift:+.6g} to {shift_clamped:+.6g} ({_DRIFT_SD_CLAMP} sd)"
                note = f"{note}; {clamp_note}" if note else clamp_note

            if mask.any():
                centre = centre_by_asset[a]
                out[mask, ai, :] = centre + (samples[mask, ai, :] - centre) * vol_clamped + shift_clamped

            per_asset_report[a] = {
                "drift_bp": drift_bp,
                "vol_scale": vol_clamped,
                "shift": shift_clamped,
                "note": note,
            }

        scenario_reports.append(
            {
                "weight_requested": float(weights[si]),
                "weight_realized": float(mask.mean()),
                "basis": basis,
                "because": str(scenario.get("because", ""))[:300],
                "assets": per_asset_report,
            }
        )

    return out, {"scenarios": scenario_reports}, len(matched_assets)
