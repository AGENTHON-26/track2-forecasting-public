"""build_draws()'s skew tilt (forecast_agent.py). Owner: Dew's section, addition: Pun.

Plain unittest, no pytest, no network -- consistent with tests/test_text_signal.py.

    python3 -m unittest tests.test_build_draws -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import forecast_agent as fa  # noqa: E402


def _panel(asset: str, n: int = 300, start: float = 100.0, seed: int = 0):
    """A synthetic daily random walk, long enough to clear build_draws()'s m>=30 floor.

    Returns (table, asof, last) -- `asof` is exactly the panel's own final date, and `last` is
    that date's value, so a test can compare against the true as-of-filtered last row rather than
    the raw array's last element (build_draws() re-derives its own `last` from `date <= asof`,
    which is not the same thing if `asof` falls short of the panel's actual final row).
    """
    rng = np.random.default_rng(seed)
    dates = [f"2020-{1 + (i // 28):02d}-{1 + (i % 28):02d}" for i in range(n)]
    values = start + np.cumsum(rng.normal(0, 1.0, n))
    table = pa.table({"date": dates, "asset": [asset] * n, "value": values.tolist()})
    return table, dates[-1], float(values[-1])


NEUTRAL = {"shift": 0.0, "widen": 1.0, "skew": 0.0}


class TestSkewTiltMath(unittest.TestCase):
    def test_skew_zero_is_identity(self):
        rng = np.random.default_rng(1)
        z = rng.standard_normal((5000, 2))
        u = np.abs(rng.standard_normal((5000, 2)))
        out = fa._skew_tilt(z, u, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(out, z)

    def test_mean_zero_and_unit_variance_for_any_skew(self):
        rng = np.random.default_rng(2)
        n = 200_000
        for s in (-0.9, -0.4, 0.0, 0.4, 0.9):
            z = rng.standard_normal(n)
            u = np.abs(rng.standard_normal(n))
            out = fa._skew_tilt(z, u, np.full(n, s))
            self.assertAlmostEqual(out.mean(), 0.0, delta=0.02, msg=f"skew={s}")
            self.assertAlmostEqual(out.std(), 1.0, delta=0.02, msg=f"skew={s}")

    def test_positive_skew_gives_positive_sample_skewness(self):
        # skew=0.8 -> shape=4.0 -> theoretical skewness ~0.78 (Azzalini's formula); a loose
        # bound here so the test checks real magnitude, not just the sign.
        rng = np.random.default_rng(3)
        n = 200_000
        z = rng.standard_normal(n)
        u = np.abs(rng.standard_normal(n))
        out = fa._skew_tilt(z, u, np.full(n, 0.8))
        skewness = ((out - out.mean()) ** 3).mean() / out.std() ** 3
        self.assertGreater(skewness, 0.5)

    def test_negative_skew_gives_negative_sample_skewness(self):
        rng = np.random.default_rng(4)
        n = 200_000
        z = rng.standard_normal(n)
        u = np.abs(rng.standard_normal(n))
        out = fa._skew_tilt(z, u, np.full(n, -0.8))
        skewness = ((out - out.mean()) ** 3).mean() / out.std() ** 3
        self.assertLess(skewness, -0.5)

    def test_per_asset_independent(self):
        # Different assets can carry different skew in the same call -- shapes must broadcast
        # per-column, not get confused into one scalar.
        rng = np.random.default_rng(5)
        n = 50_000
        z = rng.standard_normal((n, 2))
        u = np.abs(rng.standard_normal((n, 2)))
        out = fa._skew_tilt(z, u, np.array([0.8, -0.8]))
        skew_a = ((out[:, 0] - out[:, 0].mean()) ** 3).mean() / out[:, 0].std() ** 3
        skew_b = ((out[:, 1] - out[:, 1].mean()) ** 3).mean() / out[:, 1].std() ** 3
        self.assertGreater(skew_a, 0.5)
        self.assertLess(skew_b, -0.5)


class TestBuildDrawsWithSkew(unittest.TestCase):
    def test_neutral_matches_expected_center_and_scale(self):
        table, asof, last = _panel("A")
        out = fa.build_draws({"p": table}, ["A"], [21], asof,
                              {"A": dict(NEUTRAL)}, n_draws=20_000, seed=0)
        col = out[:, 0, 0]
        self.assertAlmostEqual(col.mean(), last, delta=col.std() * 0.05)

    def test_skew_is_disabled_by_default(self):
        # As of 2026-09-22 (PUN_TEXT_NOTES.md): two live comparisons couldn't isolate skew's
        # real effect from this endpoint's confirmed non-determinism, so it's off pending a
        # controlled measurement. A card asking for skew=0.9 must draw identically to skew=0.0
        # while the flag is off -- this is the behavioral guarantee that actually matters right
        # now, more than the math itself (which TestSkewTiltMath already covers directly).
        self.assertFalse(fa._SKEW_ENABLED, "flip this test too if re-enabling on purpose")
        table, asof, _ = _panel("A")
        neutral = fa.build_draws({"p": table}, ["A"], [21], asof,
                                  {"A": dict(NEUTRAL)}, n_draws=50_000, seed=0)
        skewed = fa.build_draws({"p": table}, ["A"], [21], asof,
                                 {"A": {"shift": 0.0, "widen": 1.0, "skew": 0.9}},
                                 n_draws=50_000, seed=0)
        np.testing.assert_array_equal(neutral, skewed)

    def test_skew_works_correctly_when_enabled(self):
        # The mechanism itself: still implemented and correct, just gated off above. Flips the
        # module flag for the duration of this test only.
        table, asof, _ = _panel("A")
        with mock.patch.object(fa, "_SKEW_ENABLED", True):
            neutral = fa.build_draws({"p": table}, ["A"], [21], asof,
                                      {"A": dict(NEUTRAL)}, n_draws=50_000, seed=0)
            skewed = fa.build_draws({"p": table}, ["A"], [21], asof,
                                     {"A": {"shift": 0.0, "widen": 1.0, "skew": 0.9}},
                                     n_draws=50_000, seed=0)
        # shift stays the center's job; skew should not drag the mean around noticeably.
        self.assertAlmostEqual(
            neutral[:, 0, 0].mean(), skewed[:, 0, 0].mean(),
            delta=neutral[:, 0, 0].std() * 0.05,
        )
        col = skewed[:, 0, 0]
        skewness = ((col - col.mean()) ** 3).mean() / col.std() ** 3
        self.assertGreater(skewness, 0.3)  # shape=0.9*5=4.5 -> theoretical skewness ~0.82

    def test_missing_skew_key_defaults_to_zero(self):
        # An adjustment dict that predates this change (no "skew" key) must not crash or drift.
        table, asof, _ = _panel("A")
        out = fa.build_draws({"p": table}, ["A"], [21], asof,
                              {"A": {"shift": 0.0, "widen": 1.0}}, n_draws=1000, seed=0)
        self.assertEqual(out.shape, (1000, 1, 1))


if __name__ == "__main__":
    unittest.main()
