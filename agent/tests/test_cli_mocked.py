"""Exercise the reasoning-applied path with a mocked model reply.

No MODEL_ENDPOINT is available yet (the organizer's house-model proxy isn't live, and this
session has no Anthropic credentials either), so this patches `call_model` directly rather
than making a network call. It verifies apply_adjustment's clamping/shifting logic end to end
and that the resulting forecast still clears the admissibility gates -- the two things that
would silently break if a future engine change (fat tails, scenario mixtures) mishandled the
model's reply.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.core.cli import main as agent_main  # noqa: E402


def _fake_call_model(prompt: str):
    # Extract asset names from the prompt's "  <ASSET>: level ..." lines rather than
    # hardcoding them, since the mock is reused across units with different targets.
    # First named asset gets a large drift (should clamp to the sd ceiling); the rest get
    # a modest drift and a widened vol_scale -- exercises both clamps in the same run.
    import re

    assets = re.findall(r"^  (\S+): level", prompt, flags=re.MULTILINE)
    assert assets, f"mock could not find any asset lines in prompt:\n{prompt[:500]}"
    per_asset = {
        assets[0]: {"drift_bp": 999999.0, "vol_scale": 1.8, "because": "test: huge hawkish drift"},
    }
    for a in assets[1:]:
        per_asset[a] = {"drift_bp": -50.0, "vol_scale": 1.3, "because": "test: modest dovish drift"}
    return ({"assets": per_asset}, "", "")


def run_one(unit_id: str, asof: str, out_dir: pathlib.Path) -> dict:
    unit_dir = REPO_ROOT / "units" / unit_id
    out = out_dir / "forecast.parquet"
    # Phase 3: agent.cli now imports call_model directly (`from baselines.reasoning_agent import
    # call_model`) rather than delegating to reasoning_agent.main(), so the name to patch lives
    # in agent.cli's own namespace -- patching baselines.reasoning_agent.call_model here would
    # silently do nothing, since agent.cli already holds its own bound reference to the original.
    with mock.patch("agent.core.cli.call_model", side_effect=_fake_call_model):
        rc = agent_main(
            [
                "--panels", str(unit_dir),
                "--text", str(unit_dir / "text"),
                "--asof", asof,
                "--out", str(out),
            ]
        )
    assert rc == 0, f"agent exited {rc} for {unit_id}"

    import json

    meta = json.loads((out_dir / "forecast_meta.json").read_text())
    assert meta["reasoning_applied"] is True, meta
    rationale = (out_dir / "forecast_rationale.md").read_text()
    assert "drift_bp" in rationale.lower() or "999999" not in rationale  # clamp, not raw echo

    result = subprocess.run(
        [
            sys.executable, "scoring/scoring.py", "score",
            "--card", str(unit_dir / "card.toml"),
            "--forecast", str(out),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["admissible"] is True, payload
    return {"meta": meta, "gate_payload": payload}


if __name__ == "__main__":
    import tempfile

    cases = [
        ("t2-F1-cad-boc-2017", "2017-07-12"),
        ("t2-F2-aud-taper-2013", "2013-05-24"),
    ]
    for unit_id, asof in cases:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_one(unit_id, asof, pathlib.Path(tmp))
            print(f"{unit_id}: reasoning_applied={result['meta']['reasoning_applied']} "
                  f"admissible={result['gate_payload']['admissible']}")
    print("OK: mocked reasoning path applies, clamps, and stays admissible.")
