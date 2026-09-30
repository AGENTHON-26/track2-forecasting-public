"""m0_baseline.py against the worked example and the rules in docs/M0-BASELINE.md."""
from __future__ import annotations

import pathlib
import sys
import unittest
import zlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import m0_baseline as m0  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parent.parent
_EXAMPLE = _REPO / "units" / "t2-EXAMPLE-ust-curve-1m"


@unittest.skipUnless(_EXAMPLE.is_dir(), "needs the shipped example unit")
class TestWorkedExample(unittest.TestCase):
    def test_seed_matches_the_doc(self):
        self.assertEqual(zlib.crc32(b"t2-EXAMPLE-ust-curve-1m") & 0x7FFFFFFF, 795546941)

    def test_window_uses_all_129_rows_and_128_steps(self):
        rows = m0._history(_EXAMPLE, "UST_2Y", "2024-06-28")
        self.assertEqual(len(rows), 129)
        steps, last = m0._steps(rows, "level")
        self.assertEqual(len(steps), 128)
        self.assertEqual(last, rows[-1][1])

    def test_draws_are_deterministic_and_card_shaped(self):
        a, b = m0.draws(_EXAMPLE), m0.draws(_EXAMPLE)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (500, 4, 1))


class TestWeights(unittest.TestCase):
    def test_one_cell_variogram_card_renormalizes(self):
        wm, wj, wt = m0.card_weights({}, 1)
        self.assertAlmostEqual(wm, 0.5 / 0.7)
        self.assertEqual(wj, 0.0)
        self.assertAlmostEqual(wt, 0.2 / 0.7)

    def test_multi_cell_card_keeps_card_weights(self):
        self.assertEqual(m0.card_weights({}, 4), (0.5, 0.3, 0.2))

    def test_identical_forecast_scores_one(self):
        comps = {"marginal_crps": 0.3, "joint_variogram": 0.2, "tail_penalty": 0.05}
        self.assertAlmostEqual(m0.normalized(comps, comps, (0.5, 0.3, 0.2)), 1.0)


if __name__ == "__main__":
    unittest.main()
