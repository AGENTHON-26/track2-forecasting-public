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

import forecast_agent as fa  # noqa: E402  -- the contract entry point (build_draws, _read_panels)
import forecast_models as fm  # noqa: E402  -- the models themselves, and the switch


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
        out = fm._skew_tilt(z, u, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(out, z)

    def test_mean_zero_and_unit_variance_for_any_skew(self):
        rng = np.random.default_rng(2)
        n = 200_000
        for s in (-0.9, -0.4, 0.0, 0.4, 0.9):
            z = rng.standard_normal(n)
            u = np.abs(rng.standard_normal(n))
            out = fm._skew_tilt(z, u, np.full(n, s))
            self.assertAlmostEqual(out.mean(), 0.0, delta=0.02, msg=f"skew={s}")
            self.assertAlmostEqual(out.std(), 1.0, delta=0.02, msg=f"skew={s}")

    def test_positive_skew_gives_positive_sample_skewness(self):
        # skew=0.8 -> shape=4.0 -> theoretical skewness ~0.78 (Azzalini's formula); a loose
        # bound here so the test checks real magnitude, not just the sign.
        rng = np.random.default_rng(3)
        n = 200_000
        z = rng.standard_normal(n)
        u = np.abs(rng.standard_normal(n))
        out = fm._skew_tilt(z, u, np.full(n, 0.8))
        skewness = ((out - out.mean()) ** 3).mean() / out.std() ** 3
        self.assertGreater(skewness, 0.5)

    def test_negative_skew_gives_negative_sample_skewness(self):
        rng = np.random.default_rng(4)
        n = 200_000
        z = rng.standard_normal(n)
        u = np.abs(rng.standard_normal(n))
        out = fm._skew_tilt(z, u, np.full(n, -0.8))
        skewness = ((out - out.mean()) ** 3).mean() / out.std() ** 3
        self.assertLess(skewness, -0.5)

    def test_per_asset_independent(self):
        # Different assets can carry different skew in the same call -- shapes must broadcast
        # per-column, not get confused into one scalar.
        rng = np.random.default_rng(5)
        n = 50_000
        z = rng.standard_normal((n, 2))
        u = np.abs(rng.standard_normal((n, 2)))
        out = fm._skew_tilt(z, u, np.array([0.8, -0.8]))
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

    def test_skew_reaches_the_draws(self):
        # Enabled 2026-09-25. It was off from 2026-09-22 pending "a same-inputs, code-path-only
        # comparison" that nobody could run until the stage-2 ledger was recorded; replaying that
        # ledger with the flag flipped gave a mean composite ratio of 0.9952 on the F3 units that
        # asked for a skew, nothing worse. The structural argument carried more weight than the
        # measurement: stage 2 asks for a skew on every card and F4's prompt explicitly tells the
        # model to use it, on the family scored primarily on the tail that a tilt shapes.
        #
        # The inverse of the test this replaces: a card asking for skew must now draw DIFFERENTLY
        # from a neutral one.
        self.assertTrue(fm.SKEW_ENABLED, "flip this test too if disabling on purpose")
        table, asof, _ = _panel("A")
        neutral = fa.build_draws({"p": table}, ["A"], [21], asof,
                                  {"A": dict(NEUTRAL)}, n_draws=50_000, seed=0)
        skewed = fa.build_draws({"p": table}, ["A"], [21], asof,
                                 {"A": {"shift": 0.0, "widen": 1.0, "skew": 0.9}},
                                 n_draws=50_000, seed=0)
        self.assertFalse(np.array_equal(neutral, skewed), "skew=0.9 drew identically to skew=0")
        # and it tilts the right way: positive skew fattens the UPPER tail
        col = skewed[:, 0, 0]
        centred = col - col.mean()
        sample_skewness = (centred ** 3).mean() / (centred.std() ** 3)
        self.assertGreater(sample_skewness, 0.3, f"upper tail not fattened: {sample_skewness:.3f}")


    def test_skew_works_correctly_when_enabled(self):
        # The mechanism itself: still implemented and correct, just gated off above. Flips the
        # module flag for the duration of this test only.
        table, asof, _ = _panel("A")
        with mock.patch.object(fm, "SKEW_ENABLED", True):
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


def _bdays(n: int, start: str = "2018-01-01") -> list[str]:
    """n real consecutive business days as YYYY-MM-DD.

    Not the f"2020-{i//28}" shortcut used above: past n=392 that rolls into month 15 and pandas
    rejects it outright, which is the right behaviour now that _series parses dates rather than
    sorting them as strings.
    """
    import pandas as _pd
    return [d.strftime("%Y-%m-%d") for d in _pd.bdate_range(start=start, periods=n)]


def _panel_truth(table, assets, window=260):
    """The sd and correlation build_draws will estimate from this panel's last `window` steps.

    Tests assert against THIS, not against the generator's theoretical parameters: a 260-sample
    sd carries ~4% estimation error, so comparing to the theoretical 1.0 tests the fixture's luck
    rather than the code.
    """
    import pandas as _pd
    d = table.to_pydict()
    frame = _pd.DataFrame({"date": d["date"], "asset": d["asset"], "value": d["value"]})
    wide = frame.pivot(index="date", columns="asset", values="value").sort_index()
    steps = wide[list(assets)].diff().dropna().iloc[-window:]
    return steps.to_numpy().std(axis=0), steps.corr().to_numpy()


def _multi_panel(assets, n=400, seed=0, corr=0.0, start=100.0):
    """One panel holding several assets on a shared date grid, with a known step correlation."""
    rng = np.random.default_rng(seed)
    k = len(assets)
    cov = np.full((k, k), corr, dtype=float)
    np.fill_diagonal(cov, 1.0)
    steps = rng.multivariate_normal(np.zeros(k), cov, size=n)
    dates = _bdays(n)
    cols = {"date": [], "asset": [], "value": []}
    for j, a in enumerate(assets):
        vals = start + np.cumsum(steps[:, j])
        cols["date"] += dates
        cols["asset"] += [a] * n
        cols["value"] += vals.tolist()
    return pa.table(cols), dates[-1]


class TestJointStructure(unittest.TestCase):
    """The cross-horizon and cross-asset structure the variogram actually scores.

    Nothing in this file before these tests called build_draws() with more than one asset or more
    than one horizon, so the joint behaviour was entirely unpinned.
    """

    def test_cross_horizon_correlation_is_sqrt_ratio(self):
        """Same asset, two horizons -> rho = sqrt(h1/h2), the random-walk truth.

        Before the cumulative path this was ~0: each horizon drew its own independent shock. On
        F3 that is the majority of the variogram's cell pairs (every F3 card has 2 horizons, so
        cross-horizon pairs are 53-67% of all pairs).
        """
        assets = ["A", "B", "C", "D"]
        table, asof = _multi_panel(assets, seed=3, corr=0.3)
        # family="T2-F3" selects the cumulative walk -- routing is explicit now, so a caller that
        # does not name a joint-scored family gets independent horizons by design.
        out = fa.build_draws({"p": table}, assets, [63, 126], asof,
                             {a: dict(NEUTRAL) for a in assets}, 20000, 0, family="T2-F3")
        expected = np.sqrt(63 / 126)
        for i, a in enumerate(assets):
            rho = np.corrcoef(out[:, i, 0], out[:, i, 1])[0, 1]
            self.assertAlmostEqual(rho, expected, delta=0.02, msg=f"{a}: rho={rho:.4f}")

    def test_marginal_variance_is_unchanged_by_the_path(self):
        """std at horizon h is still sd*sqrt(h).

        This is the guard that the joint fix never leaks into the marginal or tail terms -- the
        cumulative path adds covariance ACROSS horizons without touching any single horizon's
        marginal distribution.
        """
        assets = ["A", "B"]
        table, asof = _multi_panel(assets, seed=4, corr=0.0)
        out = fa.build_draws({"p": table}, assets, [21, 84], asof,
                             {a: dict(NEUTRAL) for a in assets}, 20000, 0, family="T2-F3")
        sd, _ = _panel_truth(table, assets)
        for i in range(len(assets)):
            for hi, h in enumerate([21, 84]):
                self.assertAlmostEqual(out[:, i, hi].std() / (sd[i] * np.sqrt(h)), 1.0, delta=0.03)

    def test_single_horizon_is_distributionally_unchanged(self):
        """A one-horizon card must behave exactly as it did before the cumulative path.

        This is the whole cross-family regression guard: all 27 F2 units and all 31 F4 units are
        single-horizon, and with one leg the path reduces to z*sd*sqrt(h) -- the old formula. If
        this test fails, those 58 units moved and the change is not safe to ship.
        """
        assets = ["A", "B", "C"]
        table, asof = _multi_panel(assets, seed=5, corr=0.5)
        out = fa.build_draws({"p": table}, assets, [21], asof,
                             {a: dict(NEUTRAL) for a in assets}, 20000, 0)
        sd, corr_true = _panel_truth(table, assets)
        for i in range(len(assets)):
            self.assertAlmostEqual(out[:, i, 0].std() / (sd[i] * np.sqrt(21)), 1.0, delta=0.03)
        # and the cross-asset correlation is still the panel's own
        rho = np.corrcoef(out[:, 0, 0], out[:, 1, 0])[0, 1]
        self.assertAlmostEqual(rho, corr_true[0, 1], delta=0.03)

    def test_cross_asset_correlation_survives(self):
        """The Cholesky structure is preserved by the accumulation, at every horizon."""
        assets = ["A", "B"]
        table, asof = _multi_panel(assets, seed=6, corr=0.7)
        out = fa.build_draws({"p": table}, assets, [21, 63], asof,
                             {a: dict(NEUTRAL) for a in assets}, 20000, 0, family="T2-F3")
        _, corr_true = _panel_truth(table, assets)
        for hi in range(2):
            rho = np.corrcoef(out[:, 0, hi], out[:, 1, hi])[0, 1]
            self.assertAlmostEqual(rho, corr_true[0, 1], delta=0.03)


class TestAlignmentAndGaps(unittest.TestCase):
    def test_assets_in_different_panels_align_by_date(self):
        """Two panels with different calendars must correlate on the date intersection.

        Positional stacking (the old behaviour) reads a shifted, wrong correlation. Six F3 units
        span two panels: measured on t2-F3-divergence-2014, UST_10Y/JPY reads 0.32 positionally
        against 0.50 date-aligned.
        """
        rng = np.random.default_rng(7)
        n = 400
        dates = _bdays(n)
        steps = rng.multivariate_normal([0, 0], [[1.0, 0.8], [0.8, 1.0]], size=n)
        a_vals = 100 + np.cumsum(steps[:, 0])
        b_vals = 100 + np.cumsum(steps[:, 1])
        # B's panel is missing a scattered handful of dates -- a different calendar, as a real
        # rates-vs-FX pair has.
        drop = {13, 51, 97, 150, 201, 260, 301}
        pa_tbl = pa.table({"date": dates, "asset": ["A"] * n, "value": a_vals.tolist()})
        keep = [i for i in range(n) if i not in drop]
        pb_tbl = pa.table({"date": [dates[i] for i in keep], "asset": ["B"] * len(keep),
                           "value": [b_vals[i] for i in keep]})
        out = fa.build_draws({"pa": pa_tbl, "pb": pb_tbl}, ["A", "B"], [21], dates[-1],
                             {a: dict(NEUTRAL) for a in ["A", "B"]}, 20000, 0)
        # The truth is the correlation on the DATE INTERSECTION, which is what date-aligned
        # stacking recovers and positional stacking does not.
        import pandas as _pd
        sa = _pd.Series(a_vals, index=_pd.to_datetime(dates))
        sb = _pd.Series([b_vals[i] for i in keep], index=_pd.to_datetime([dates[i] for i in keep]))
        joined = _pd.DataFrame({"A": sa, "B": sb}).diff().dropna().iloc[-260:]
        expected = joined.corr().to_numpy()[0, 1]
        rho = np.corrcoef(out[:, 0, 0], out[:, 1, 0])[0, 1]
        self.assertAlmostEqual(rho, expected, delta=0.04,
                               msg=f"date-aligned corr {expected:.3f} not recovered: {rho:.3f}")

    def test_a_decade_hole_does_not_inflate_sd(self):
        """A deliberate history gap must not be differenced across.

        Transfer cards ship an early window plus a single as-of anchor row. Differenced naively
        that hole is one 'day' worth a decade of movement: measured on
        t2-F2-fragile-five-brl-2013, a -0.875 step against a typical 0.030, inflating sd by 1.4x.
        """
        rng = np.random.default_rng(8)
        early = _bdays(300, start="2003-01-01")
        vals = list(100 + np.cumsum(rng.normal(0, 1.0, 300)))
        gapped = pa.table({"date": early + ["2013-05-24"], "asset": ["A"] * 301,
                           "value": vals + [40.0]})            # a huge jump across a 10-year hole
        clean = pa.table({"date": early, "asset": ["A"] * 300, "value": vals})
        adj = {"A": dict(NEUTRAL)}
        g = fa.build_draws({"p": gapped}, ["A"], [21], "2013-05-24", adj, 8000, 0)
        c = fa.build_draws({"p": clean}, ["A"], [21], early[-1], adj, 8000, 0)
        ratio = g[:, 0, 0].std() / c[:, 0, 0].std()
        self.assertLess(ratio, 2.0, f"the gap inflated sd by {ratio:.2f}x")

    def test_asset_order_follows_the_argument_not_the_panel(self):
        """sd/shift/widen are built in `assets` order; the step frame must match it.

        If the frame's column order ever diverged, the Cholesky would be applied to the wrong
        assets silently -- no exception, just a wrong correlation structure.
        """
        # B is far more volatile than A; ask for them in the non-panel order.
        rng = np.random.default_rng(9)
        n = 400
        dates = _bdays(n)
        cols = {"date": [], "asset": [], "value": []}
        for a, vol in (("A", 0.1), ("B", 5.0)):
            cols["date"] += dates
            cols["asset"] += [a] * n
            cols["value"] += (100 + np.cumsum(rng.normal(0, vol, n))).tolist()
        table = pa.table(cols)
        out = fa.build_draws({"p": table}, ["B", "A"], [21], dates[-1],
                             {a: dict(NEUTRAL) for a in ["A", "B"]}, 8000, 0)
        sd_b, sd_a = out[:, 0, 0].std(), out[:, 1, 0].std()
        self.assertGreater(sd_b, 10 * sd_a, "B (vol 5.0) must be the first output column")


class TestLogReturnTarget(unittest.TestCase):
    def test_log_return_centres_on_zero_not_the_last_return(self):
        """A cumulative log-return target starts at 0, with no drift extrapolation.

        The panel holds decimal simple returns, so the old behaviour -- anchor at the last row and
        difference the rows -- was wrong twice over. Zero drift rather than `steps.mean()*h` is a
        measured choice: 11/15 log_return units improve, mean ratio 0.8818.
        """
        rng = np.random.default_rng(10)
        n = 400
        dates = _bdays(n)
        rets = rng.normal(0.002, 0.01, n)          # a clear positive mean, to catch drift
        rets[-1] = 0.05                            # a large final return: the old anchor
        table = pa.table({"date": dates, "asset": ["A"] * n, "value": rets.tolist()})
        out = fa.build_draws({"p": table}, ["A"], [21], dates[-1], {"A": dict(NEUTRAL)},
                             20000, 0, target_type="log_return", family="T2-F4")
        centre = out[:, 0, 0].mean()
        self.assertAlmostEqual(centre, 0.0, delta=0.01,
                               msg=f"log_return centre should be ~0, got {centre:.4f}")


class TestModelDispatch(unittest.TestCase):
    """The switch in `_select_model` -- which family gets which model.

    Routing is explicit and family-based, so it is worth pinning: a change here silently moves a
    whole family onto a different model, and every one of these choices is a measured one.
    """

    def test_the_routing_table(self):
        cases = {
            # F1 level cards are M2's home: measured 0.832x the walk on F1's own units. This is
            # the only family-keyed branch, because M2 is a fitted model.
            ("T2-F1", "level", (126, 189)): fm.M2,
            # ...but M2 covers level targets only, so F1's 2 log_return units take a walk.
            ("T2-F1", "log_return", (127,)): fm.RANDOM_WALK,
            # Multi-horizon -> the path is accumulated. Today this is exactly F3's 22 units.
            ("T2-F3", "level", (63, 126)): fm.CUMULATIVE_WALK,
            ("T2-F3", "log_return", (21, 63)): fm.CUMULATIVE_WALK,
            # Single-horizon -> one draw per horizon. All 27 F2 and all 31 F4 units today.
            ("T2-F2", "level", (21,)): fm.RANDOM_WALK,
            ("T2-F4", "log_return", (21,)): fm.RANDOM_WALK,
            # The reason the middle branch tests shape and not family: the organizers' own
            # example cards in docs/CATEGORIES.md are multi-horizon for BOTH F2 ("GBP/USD at
            # horizons 21 BD and 63 BD") and F4 ("UST_2Y, UST_10Y at 63 BD and 126 BD"), even
            # though no shipped dev unit in those families is. A sealed card shaped like either
            # must still get its cross-horizon structure -- on the F2 shape that pair is the
            # entire off-diagonal of the variogram.
            ("T2-F2", "level", (21, 63)): fm.CUMULATIVE_WALK,
            ("T2-F4", "level", (63, 126)): fm.CUMULATIVE_WALK,
            # An unknown or absent family must still produce a forecast, and must not lose the
            # path just because its metadata is unfamiliar.
            ("T2-F9", "level", (21, 63)): fm.CUMULATIVE_WALK,
            (None, None, (21,)): fm.RANDOM_WALK,
        }
        for (family, target_type, horizons), expected in cases.items():
            name, model = fm._select_model(family, target_type, list(horizons))
            self.assertEqual(name, expected,
                             f"{family}/{target_type}/h={list(horizons)} routed to {name}")
            self.assertTrue(callable(model))

    def test_m2_failure_falls_back_instead_of_losing_the_card(self):
        """M2 refuses cards whose assets span two panel files. A raise would score the card at the
        pre-committed worst case (4.0); the walk scores ~1.0, so the fallback is worth a lot."""
        assets = ["A", "B"]
        table, asof = _multi_panel(assets, seed=11, corr=0.2)
        with mock.patch.object(fm, "_m2_model", side_effect=ValueError("no single panel")):
            out = fa.build_draws({"p": table}, assets, [21], asof,
                                 {a: dict(NEUTRAL) for a in assets}, 500, 0,
                                 target_type="level", family="T2-F1")
        self.assertEqual(out.shape, (500, 2, 1))
        self.assertEqual(fm.last_model(), fm.RANDOM_WALK)

    def test_single_horizon_walks_agree(self):
        """On a single horizon the cumulative and independent walks are the same model.

        This is what makes the routing safe for F2 (27 units) and F4 (31), all single-horizon:
        sending them to the plain walk is a naming choice, not a behaviour change.
        """
        assets = ["A", "B"]
        table, asof = _multi_panel(assets, seed=12, corr=0.4)
        args = ({"p": table}, assets, [21], asof, {a: dict(NEUTRAL) for a in assets}, 4000, 0)
        cumulative = fa.build_draws(*args, family="T2-F3")
        independent = fa.build_draws(*args, family="T2-F2")
        np.testing.assert_allclose(cumulative, independent, rtol=0, atol=0)
