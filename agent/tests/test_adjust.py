"""Unit tests for agent/adjust.py -- the shared-scenario-assignment and established/inferred
clamp logic that agent/prompt.py's contract depends on. Pure numpy, no model call, no I/O.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.core.adjust import apply_scenarios  # noqa: E402

ASSETS = ["UST_2Y", "UST_10Y"]
N_DRAWS = 20_000
LAST = {"UST_2Y": 4.0, "UST_10Y": 4.5}
SD_H = {"UST_2Y": 0.5, "UST_10Y": 0.4}  # horizon sd, so 3-sd ceiling is 1.5 / 1.2


def _flat_samples() -> np.ndarray:
    # Zero-variance base so any spread in the output is attributable to the scenario mixture's
    # OWN mean-shift-by-scenario, not to noise already present in the draws.
    return np.zeros((N_DRAWS, len(ASSETS), 1))


def test_backward_compat_flat_shape_is_one_inferred_scenario():
    parsed = {"UST_2Y": {"drift_bp": 100.0, "vol_scale": 1.5, "because": "old shape"}}
    out, report, matched = apply_scenarios(_flat_samples(), ASSETS, LAST, SD_H, parsed, seed=1)
    assert matched == 1
    assert len(report["scenarios"]) == 1
    assert report["scenarios"][0]["basis"] == "inferred"
    assert report["scenarios"][0]["weight_requested"] == 1.0
    # UST_10Y wasn't named -> no-op for it (shift 0, vol 1 -> value stays 0 since base is flat)
    assert np.allclose(out[:, 1, 0], 0.0)
    # UST_2Y: drift_bp=100 -> shift = 4.0 * 100/10000 = 0.04, well under the 1.5 ceiling
    assert np.allclose(out[:, 0, 0], 0.04)


def test_shared_scenario_assignment_across_assets():
    """The whole point of Phase 4's design: for a given draw, EVERY asset must reflect the SAME
    scenario, not independently-chosen ones. Scenario A shifts both assets sharply positive;
    scenario B shifts both sharply negative. If assignment were independent per asset, some
    draws would show asset 1 positive and asset 2 negative -- this test asserts that never
    happens.
    """
    parsed = {
        "scenarios": [
            {
                "weight": 0.5, "basis": "inferred", "because": "hawkish",
                "assets": {"UST_2Y": {"drift_bp": 1000.0, "vol_scale": 1.0},
                           "UST_10Y": {"drift_bp": 1000.0, "vol_scale": 1.0}},
            },
            {
                "weight": 0.5, "basis": "inferred", "because": "dovish",
                "assets": {"UST_2Y": {"drift_bp": -1000.0, "vol_scale": 1.0},
                           "UST_10Y": {"drift_bp": -1000.0, "vol_scale": 1.0}},
            },
        ]
    }
    out, report, matched = apply_scenarios(_flat_samples(), ASSETS, LAST, SD_H, parsed, seed=2)
    assert matched == 2
    a2y, a10y = out[:, 0, 0], out[:, 1, 0]
    # Every draw: both assets positive together, or both negative together -- never mixed.
    same_sign = (np.sign(a2y) == np.sign(a10y))
    assert same_sign.all(), "assets disagreed on sign within a single draw -- assignment leaked"
    # Both scenarios should actually have fired (not e.g. one dominating due to a bug).
    assert (a2y > 0).mean() > 0.3
    assert (a2y < 0).mean() > 0.3
    # Realized weights should be close to the requested 0.5/0.5 at 20k draws.
    realized = sorted(sc["weight_realized"] for sc in report["scenarios"])
    assert abs(realized[0] - 0.5) < 0.02
    assert abs(realized[1] - 0.5) < 0.02


def test_inferred_cannot_narrow_but_established_can():
    parsed_inferred = {
        "scenarios": [
            {"weight": 1.0, "basis": "inferred", "because": "tone reading",
             "assets": {"UST_2Y": {"drift_bp": 0.0, "vol_scale": 0.3}}}
        ]
    }
    parsed_established = {
        "scenarios": [
            {"weight": 1.0, "basis": "established", "because": "stated fact",
             "assets": {"UST_2Y": {"drift_bp": 0.0, "vol_scale": 0.3}}}
        ]
    }
    _, report_i, _ = apply_scenarios(_flat_samples(), ASSETS, LAST, SD_H, parsed_inferred, seed=3)
    _, report_e, _ = apply_scenarios(_flat_samples(), ASSETS, LAST, SD_H, parsed_established, seed=3)

    vol_i = report_i["scenarios"][0]["assets"]["UST_2Y"]["vol_scale"]
    vol_e = report_e["scenarios"][0]["assets"]["UST_2Y"]["vol_scale"]
    assert vol_i == 1.0, f"inferred basis must floor vol_scale at 1.0, got {vol_i}"
    assert vol_e == 0.5, f"established basis floors only at 0.5 (the absolute floor), got {vol_e}"
    assert "clamped" in report_i["scenarios"][0]["assets"]["UST_2Y"]["note"]
    assert "clamped" in report_e["scenarios"][0]["assets"]["UST_2Y"]["note"]


def test_drift_still_clamped_to_horizon_sd_ceiling():
    parsed = {
        "scenarios": [
            {"weight": 1.0, "basis": "established", "because": "huge number",
             "assets": {"UST_2Y": {"drift_bp": 999999.0, "vol_scale": 1.0}}}
        ]
    }
    _, report, _ = apply_scenarios(_flat_samples(), ASSETS, LAST, SD_H, parsed, seed=4)
    shift = report["scenarios"][0]["assets"]["UST_2Y"]["shift"]
    ceiling = 3.0 * SD_H["UST_2Y"]
    assert abs(shift - ceiling) < 1e-9, f"expected clamp to {ceiling}, got {shift}"


def test_invalid_weights_fall_back_to_equal():
    parsed = {
        "scenarios": [
            {"weight": "not-a-number", "basis": "inferred", "assets": {"UST_2Y": {"drift_bp": 0, "vol_scale": 1}}},
            {"weight": float("nan"), "basis": "inferred", "assets": {"UST_2Y": {"drift_bp": 0, "vol_scale": 1}}},
        ]
    }
    _, report, matched = apply_scenarios(_flat_samples(), ASSETS, LAST, SD_H, parsed, seed=5)
    assert matched == 1
    weights = [sc["weight_requested"] for sc in report["scenarios"]]
    assert all(abs(w - 0.5) < 1e-9 for w in weights)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
