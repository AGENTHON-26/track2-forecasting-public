"""The leaderboard branches submit the model alone: forecast_agent makes no text call.

Plain unittest, no network -- consistent with tests/test_build_draws.py.

    python3 -m unittest tests.test_text_off -v
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import forecast_agent as fa  # noqa: E402

UNIT = REPO / "units" / "t2-F3-divergence-2014"
NEUTRAL = {"shift": 0.0, "widen": 1.0, "skew": 0.0}


class TestTextOff(unittest.TestCase):
    def test_no_text_call_and_every_asset_neutral(self):
        if not UNIT.is_dir():
            self.skipTest("units/ not checked out")
        self.assertFalse(fa.TEXT_SIGNAL, "these branches submit the model alone")
        asof = tomllib.loads((UNIT / "card.toml").read_text())["provenance"]["data_cutoff"]
        seen, real = {}, fa.build_draws

        def spy(panels, assets, horizons, when, adjustments, *args, **kw):
            seen["adjustments"] = adjustments
            return real(panels, assets, horizons, when, adjustments, *args, **kw)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(fa, "read_text_signal", side_effect=AssertionError("the text half was called")), \
                mock.patch.object(fa, "build_draws", side_effect=spy):
            out = pathlib.Path(tmp) / "forecast.parquet"
            argv = ["--panels", str(UNIT), "--text", str(UNIT / "text"), "--asof", asof, "--out", str(out)]
            self.assertEqual(fa.main(argv), 0)
            self.assertTrue(out.exists())
            self.assertIn("Text used: no", (pathlib.Path(tmp) / "forecast_rationale.md").read_text())
        self.assertEqual(set(seen["adjustments"]), {"UST_10Y", "EUR", "JPY"})
        self.assertTrue(all(v == NEUTRAL for v in seen["adjustments"].values()))


if __name__ == "__main__":
    unittest.main()
