"""End-to-end test of the NEW multi-scenario reply shape through the real CLI, on a multi-asset
F3 card -- the case Phase 4's shared-per-draw-scenario design exists for. Complements
test_adjust.py (which checks the mechanism in isolation) by checking it survives the full
pipeline: prompt -> (mocked) reply -> write -> gates.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.core.cli import main as agent_main  # noqa: E402


def _fake_two_scenario_reply(prompt: str):
    assets = re.findall(r"^  (\S+): level", prompt, flags=re.MULTILINE)
    assert assets, f"mock could not find asset lines in prompt:\n{prompt[:400]}"
    return (
        {
            "scenarios": [
                {
                    "weight": 0.6, "basis": "inferred", "because": "test: hawkish hold continues",
                    "assets": {a: {"drift_bp": 80.0, "vol_scale": 1.1} for a in assets},
                },
                {
                    "weight": 0.4, "basis": "established", "because": "test: stated dovish pivot",
                    "assets": {a: {"drift_bp": -150.0, "vol_scale": 0.8} for a in assets},
                },
            ]
        },
        "",
        "",
    )


def run(unit_id: str, asof: str, out_dir: pathlib.Path) -> dict:
    unit_dir = REPO_ROOT / "units" / unit_id
    out = out_dir / "forecast.parquet"
    with mock.patch("agent.core.cli.call_model", side_effect=_fake_two_scenario_reply):
        rc = agent_main(
            ["--panels", str(unit_dir), "--text", str(unit_dir / "text"),
             "--asof", asof, "--out", str(out)]
        )
    assert rc == 0

    meta = json.loads((out_dir / "forecast_meta.json").read_text())
    assert meta["reasoning_applied"] is True, meta
    assert meta["n_scenarios"] == 2, meta

    rationale = (out_dir / "forecast_rationale.md").read_text()
    assert "Scenario 1" in rationale and "Scenario 2" in rationale
    assert "established" in rationale and "inferred" in rationale

    result = subprocess.run(
        [sys.executable, "scoring/scoring.py", "score",
         "--card", str(unit_dir / "card.toml"), "--forecast", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["admissible"] is True, payload
    return {"meta": meta, "gate_payload": payload}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        # A multi-asset F3 card -- the case where shared-per-draw scenario assignment matters.
        result = run("t2-F3-conundrum-joint-2005", "2005-02-18", pathlib.Path(tmp))
        print(f"n_scenarios={result['meta']['n_scenarios']} "
              f"admissible={result['gate_payload']['admissible']}")
    print("OK: multi-scenario reply applies across a multi-asset card and stays admissible.")
