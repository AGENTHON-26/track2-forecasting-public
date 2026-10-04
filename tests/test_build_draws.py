"""build_draws() -- M1.9, the production model (forecast_models.py) -- and its skew tilt. Owner: Dew's section,
addition: Pun.

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

#: M1.9's pieces switched off, which is M1.5 exactly. The tests of the core walk (text adjustments,
#: joint structure, gaps, monthly steps, asset rules, Student-t) pin those mechanics on their own;
#: `TestM19Pieces` tests each piece.
M15_CORE = dict(RECENT_TREND=False, ZERO_FLOOR=False, VR_SHRINK=None, TREND_SHARE=0.0)


class CoreTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.multiple(fm, **M15_CORE)
        patcher.start()
        self.addCleanup(patcher.stop)


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


class TestBuildDrawsWithSkew(CoreTest):
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


class TestTextAdjustments(CoreTest):
    """widen multiplies the sd and shift moves every horizon in the asset's own units, on every card.
    (M1.5's own widen is 1.0; the silent-text F4 floor of 1.5 was removed on 2026-09-27.)"""

    def test_widen_scales_every_deviation_from_the_centre(self):
        table, asof, last = _panel("A")
        args = ({"p": table}, ["A"], [21, 63], asof)
        base = fa.build_draws(*args, {"A": dict(NEUTRAL)}, 2000, 0)
        wide = fa.build_draws(*args, {"A": {"shift": 0.0, "widen": 1.5, "skew": 0.0}}, 2000, 0)
        np.testing.assert_allclose(wide - last, 1.5 * (base - last), rtol=1e-12, atol=1e-12)

    def test_shift_is_added_in_native_units_also_on_log_stepped_fx(self):
        for panel_id in ("p", "g10_fx_daily"):              # M0's change, and FX's log change
            table, asof, _ = _panel("A")
            table = table.append_column("panel_id", pa.array([panel_id] * table.num_rows))
            args = ({"p": table}, ["A"], [21, 63], asof)
            base = fa.build_draws(*args, {"A": dict(NEUTRAL)}, 2000, 0)
            moved = fa.build_draws(*args, {"A": {"shift": 0.25, "widen": 1.0, "skew": 0.0}}, 2000, 0)
            np.testing.assert_allclose(moved - base, 0.25, rtol=0, atol=1e-9, err_msg=panel_id)


def _bdays(n: int, start: str = "2018-01-01") -> list[str]:
    """n real consecutive business days as YYYY-MM-DD.

    Not the f"2020-{i//28}" shortcut used above: past n=392 that rolls into month 15 and pandas
    rejects it outright, which is the right behaviour now that _series parses dates rather than
    sorting them as strings.
    """
    import pandas as _pd
    return [d.strftime("%Y-%m-%d") for d in _pd.bdate_range(start=start, periods=n)]


def _panel_truth(table, assets, window=260):
    """The sd and correlation build_draws will estimate from this panel: each asset's EWMA sd
    (halflife 63, over its last 8 x 63 steps) and the correlation of the last `window` steps.

    Tests assert against THIS, not against the generator's theoretical parameters: a 260-sample
    sd carries ~4% estimation error, so comparing to the theoretical 1.0 tests the fixture's luck
    rather than the code.
    """
    import pandas as _pd
    d = table.to_pydict()
    frame = _pd.DataFrame({"date": d["date"], "asset": d["asset"], "value": d["value"]})
    wide = frame.pivot(index="date", columns="asset", values="value").sort_index()
    steps = wide[list(assets)].diff().dropna()
    sd = np.array([fm._ewma_sd(steps[a].to_numpy()[-int(8 * fm.EWMA_HALFLIFE):], fm.EWMA_HALFLIFE) for a in assets])
    return sd, steps.iloc[-window:].corr().to_numpy()


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


class TestJointStructure(CoreTest):
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
        # Two horizons select the cumulative walk; family=None keeps the plain settings, so this
        # pins the walk's structure rather than one family's tuning.
        out = fa.build_draws({"p": table}, assets, [63, 126], asof,
                             {a: dict(NEUTRAL) for a in assets}, 20000, 0, family=None)
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
                             {a: dict(NEUTRAL) for a in assets}, 20000, 0, family=None)
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
                             {a: dict(NEUTRAL) for a in assets}, 20000, 0, family=None)
        _, corr_true = _panel_truth(table, assets)
        for hi in range(2):
            rho = np.corrcoef(out[:, 0, hi], out[:, 1, hi])[0, 1]
            self.assertAlmostEqual(rho, corr_true[0, 1], delta=0.03)


class TestAlignmentAndGaps(CoreTest):
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


class TestMonthlyHorizonSteps(CoreTest):
    """On a monthly panel one step is a month, but cards state horizons in business days. The model
    must count months from the asset's last observation to asof + h business days (the official
    baseline's rule, docs/M0-BASELINE.md section 3.7) -- not walk h months."""

    def _monthly(self, n: int = 120):
        rng = np.random.default_rng(3)
        dates = [f"{2010 + i // 12}-{1 + i % 12:02d}-01" for i in range(n)]      # 2010-01 .. 2019-12
        values = 100 + np.cumsum(rng.normal(0, 1, n))
        return pa.table({"date": dates, "asset": ["X"] * n, "value": values.tolist()})

    def test_monthly_horizon_is_counted_in_months_from_the_last_observation(self):
        table = self._monthly()
        # last observation 2019-12-01; the as-of lags it by two months, like published macro data
        req = fm._Request({"m": table}, ["X"], [21, 63], "2020-01-31", {"X": dict(NEUTRAL)}, 4000, 0,
                          "level", None)
        fit = fm._fit_model(req)
        # 2020-01-31 + 21 BD = 2020-03-02 -> March, 3 months after Dec; + 63 BD = 2020-04-29 -> 4 months
        self.assertEqual(fit.steps["X"], {21: 3.0, 63: 4.0})
        out = fm._model(req)
        sd = np.sqrt(fit.sigma[0, 0])
        for hi, k in enumerate((3, 4)):
            self.assertAlmostEqual(out[:, 0, hi].std() / (sd * np.sqrt(k)), 1.0, delta=0.05)

    def test_m0s_four_monthly_cards_are_reproduced(self):
        """The doc's table (m0_baseline._MONTHLY_STEPS) follows from the dates on all four cards."""
        units = pathlib.Path(__file__).resolve().parent.parent / "units"
        table = {"t2-F1-cpi-glidepath-2023": {140: 8, 160: 9}, "t2-F1-sahm-watch-2024": {145: 8, 165: 9},
                 "t2-F4-covid-nfp-2020": {21: 2}, "t2-F4-cpi-vintage-2022": {21: 2}}
        import tomllib
        for unit, want in table.items():
            if not (units / unit).is_dir():
                self.skipTest("units/ not checked out")
            card = tomllib.loads((units / unit / "card.toml").read_text())
            asof, asset = card["provenance"]["data_cutoff"], card["targets"]["asset_ids"][0]
            _, rows = fm._history(fa._read_panels(units / unit), asset, asof)
            got = fm._horizon_steps([d for d, _ in rows], list(want), asof)
            self.assertEqual(got, [float(v) for v in want.values()], unit)

    def test_no_new_release_keeps_the_last_value(self):
        table = self._monthly()            # last observation 2019-12-01
        out = fa.build_draws({"m": table}, ["X"], [1], "2019-12-02", {"X": dict(NEUTRAL)}, 100, 0)
        self.assertTrue(np.all(out[:, 0, 0] == table.column("value").to_pylist()[-1]))

    def test_daily_horizon_is_unchanged(self):
        table, asof = _multi_panel(["A"], n=400, seed=2)
        req = fm._Request({"p": table}, ["A"], [21, 63], asof, {"A": dict(NEUTRAL)}, 10, 0, "level", None)
        self.assertEqual(fm._fit_model(req).steps["A"], {21: 21.0, 63: 63.0})


class TestLogReturnTarget(CoreTest):
    def test_log_return_centres_on_zero_not_the_last_return(self):
        """A cumulative log-return target starts at 0, with no drift extrapolation.

        Each row is the step (M0's rule, docs/M0-BASELINE.md 3.2): the cumulative target is the sum
        of the next h rows. No drift: equity factors are centred on 0 (notebook 07: 0.978 vs M0).
        """
        rng = np.random.default_rng(10)
        n = 400
        dates = _bdays(n)
        rets = rng.normal(0.002, 0.01, n)          # a clear positive mean, to catch drift
        rets[-1] = 0.05                            # a large final return: the old anchor
        table = pa.table({"date": dates, "asset": ["A"] * n, "value": rets.tolist(),
                          "panel_id": ["factors_daily"] * n})
        out = fa.build_draws({"p": table}, ["A"], [21], dates[-1], {"A": dict(NEUTRAL)},
                             20000, 0, target_type="log_return", family="T2-F4")
        centre = out[:, 0, 0].mean()
        self.assertAlmostEqual(centre, 0.0, delta=0.01,
                               msg=f"log_return centre should be ~0, got {centre:.4f}")


def _typed_panel(asset: str, panel_id: str, n: int = 600, seed: int = 0, start: float = 100.0,
                 trend: float = 0.0, df: float | None = None):
    """A daily panel tagged with a real panel id; Student-t steps when `df` is given."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_t(df, n) if df else rng.normal(0, 1.0, n)
    values = start + trend * np.arange(n) + np.cumsum(steps)
    dates = _bdays(n)
    return pa.table({"date": dates, "asset": [asset] * n, "value": values.tolist(),
                     "panel_id": [panel_id] * n}), dates[-1], values


class TestAssetRules(CoreTest):
    """The asset type comes from the unit folder: the panel that holds the asset (its `panel_id`,
    which is also its file name and the card's [panels] id); macro assets by id."""

    def test_every_panel_the_organisers_ship(self):
        cases = {("UST_20Y", "rates_daily", "level"): ("diff", "student_t", False),
                 ("NZD", "g10_fx_daily", "level"): ("log_diff", "student_t", False),
                 ("INR", "em_transfer_early", "level"): ("log_diff", "student_t", False),
                 ("BAB", "factors_daily", "log_return"): ("as_return", "student_t", False),
                 ("CPI_ALL", "macro_monthly", "level"): ("log_diff", "normal", True),
                 ("PCE_CORE", "macro_monthly", "level"): ("log_diff", "normal", True),
                 ("NFP", "macro_monthly", "level"): ("diff", "normal", True),
                 ("UNRATE", "macro_monthly", "level"): ("diff", "normal", False),
                 ("UST_10Y", "rates_daily", "log_return"): ("as_return", "student_t", False),
                 ("XYZ", "some_new_panel", "level"): ("diff", "normal", False)}
        for (asset, panel, target), want in cases.items():
            r = fm.asset_rule(asset, panel, target)
            self.assertEqual((r["transform"], r["shock"], r["drift"]), want, f"{asset} in {panel}")

    def test_the_tables_panel_id_wins_over_the_file_name(self):
        table, asof, _ = _typed_panel("A", "g10_fx_daily")
        self.assertEqual(fm._history({"whatever": table}, "A", asof)[0], "g10_fx_daily")
        plain = table.drop_columns(["panel_id"])
        self.assertEqual(fm._history({"rates_daily": plain}, "A", asof)[0], "rates_daily")

    def test_drift_reaches_the_centre_only_where_the_rule_says(self):
        table, asof, values = _typed_panel("NFP", "macro_monthly", trend=0.5, seed=4)
        steps = np.diff(values)[-fm.WINDOW:]
        out = fa.build_draws({"p": table}, ["NFP"], [21], asof, {"NFP": dict(NEUTRAL)}, 20000, 0)
        self.assertAlmostEqual(out[:, 0, 0].mean(), values[-1] + 21 * steps.mean(), delta=0.15)
        table, asof, values = _typed_panel("UNRATE", "macro_monthly", trend=0.5, seed=4)
        out = fa.build_draws({"p": table}, ["UNRATE"], [21], asof, {"UNRATE": dict(NEUTRAL)}, 20000, 0)
        self.assertAlmostEqual(out[:, 0, 0].mean(), values[-1], delta=0.15)

    def test_fx_is_stepped_in_logs(self):
        """A log-stepped asset stays positive and its median is the last value."""
        table, asof, values = _typed_panel("EUR", "g10_fx_daily", start=60.0, seed=6)
        out = fa.build_draws({"p": table}, ["EUR"], [252], asof, {"EUR": dict(NEUTRAL)}, 20000, 0)[:, 0, 0]
        self.assertTrue(np.all(out > 0))
        self.assertAlmostEqual(np.median(out) / values[-1], 1.0, delta=0.01)


class TestOneModel(CoreTest):
    """Every card takes the one model -- no per-family settings (2026-10-04), so the text's
    shift / widen / skew mean the same thing on every card."""

    def test_every_card_takes_the_model(self):
        daily, asof = _multi_panel(["A", "B"], n=400, seed=5, corr=0.3)
        n = 120
        monthly = pa.table({"date": [f"{2010 + i // 12}-{1 + i % 12:02d}-01" for i in range(n)],
                            "asset": ["X"] * n, "value": (100 + np.arange(n) * 0.1).tolist()})
        cases = [({"p": daily}, ["A", "B"], [21, 63], asof, "level", "T2-F3"),
                 ({"p": daily}, ["A"], [21], asof, "level", "T2-F2"),
                 ({"p": daily}, ["B"], [21], asof, "level", "T2-F4"),
                 ({"m": monthly}, ["X"], [140, 160], "2020-01-31", "level", "T2-F1"),
                 ({"p": daily}, ["A"], [21, 63], asof, None, None)]
        for panels, assets, horizons, when, target, family in cases:
            adj = {a: dict(NEUTRAL) for a in assets}
            got = fa.build_draws(panels, assets, horizons, when, adj, 500, 3,
                                 target_type=target, family=family)
            want = fm._model(fm._Request(panels, assets, horizons, when, adj, 500, 3, target, None))
            np.testing.assert_array_equal(got, want, err_msg=f"{family} {assets} {horizons}")

    def test_single_horizon_by_hand(self):
        """By hand: value = last + z * sqrt(h * sd^2 + jitter) * sqrt((nu - 2) / W) -- one rates asset,
        one horizon, Student-t steps so that nu is fitted. No drift on rates."""
        table, asof, values = _typed_panel("UST_10Y", "rates_daily", df=5, seed=8)
        n, h, seed = 1000, 63, 4
        out = fa.build_draws({"p": table}, ["UST_10Y"], [h], asof, {"UST_10Y": dict(NEUTRAL)}, n, seed,
                             target_type="level")[:, 0, 0]
        steps = np.diff(values)
        nu = fm._fit_nu(steps[-fm.WINDOW:][None, :])
        self.assertIsNotNone(nu)
        sd = fm._ewma_sd(steps[-int(8 * fm.EWMA_HALFLIFE):], fm.EWMA_HALFLIFE)
        z = np.random.default_rng(seed).standard_normal((n, 1))[:, 0]
        w = np.random.default_rng([seed, 7]).chisquare(nu, size=(n, 1))[:, 0]
        by_hand = values[-1] + z * np.sqrt(h * sd ** 2 + 1e-10 + 1e-9) * np.sqrt((nu - 2) / w)
        np.testing.assert_allclose(out, by_hand, rtol=0, atol=1e-10)


class TestM15Settings(CoreTest):
    """The EWMA sd, and the Student-t shocks with nu fitted from the window."""

    def test_ewma_sd_matches_pandas(self):
        x = np.random.default_rng(3).normal(0, 1, (400, 2))
        import pandas as pd
        ref = pd.DataFrame(x).ewm(halflife=63, adjust=True).std(bias=True).iloc[-1].to_numpy()
        np.testing.assert_allclose(fm._ewma_sd(x, 63), ref, rtol=1e-9)

    def test_nu_is_fitted_from_kurtosis(self):
        rng = np.random.default_rng(5)
        self.assertIsNone(fm._fit_nu(rng.normal(0, 1, (1, 5000))))           # normal steps stay normal
        nu = fm._fit_nu(rng.standard_t(6, (1, 20000)))                         # t(6): k = 3, nu = 6
        self.assertTrue(5.0 < nu < 7.5, nu)

    def test_student_t_keeps_variance_and_correlation_and_fattens_tails(self):
        """Same steps, typed as rates (Student-t) vs an unknown panel (normal): same sd per cell, same
        correlation between assets, more mass beyond 3 sd."""
        rng = np.random.default_rng(22)
        n = 900
        steps = rng.standard_t(5, (n, 2)) @ np.linalg.cholesky([[1.0, 0.6], [0.6, 1.0]]).T
        dates = _bdays(n)
        def panel(pid):
            return pa.table({"date": dates * 2, "asset": ["A"] * n + ["B"] * n,
                             "value": (100 + np.cumsum(steps, axis=0)).T.ravel().tolist(),
                             "panel_id": [pid] * (2 * n)})
        args = (["A", "B"], [21], dates[-1], {a: dict(NEUTRAL) for a in "AB"}, 200_000, 0)
        normal = fa.build_draws({"p": panel("p")}, *args)[:, :, 0]
        student = fa.build_draws({"p": panel("rates_daily")}, *args)[:, :, 0]
        zn, zt = normal - normal.mean(0), student - student.mean(0)
        np.testing.assert_allclose(zt.std(0) / zn.std(0), 1.0, atol=0.02)
        self.assertAlmostEqual(np.corrcoef(zt.T)[0, 1], np.corrcoef(zn.T)[0, 1], delta=0.02)
        beyond = lambda z: np.mean(np.abs(z / z.std(0)) > 3)
        self.assertGreater(beyond(zt), 2 * beyond(zn))


class TestM19Pieces(unittest.TestCase):
    """M1.9's pieces on top of M1.5, each on its own (model_experiment_v3 notebooks 12-14)."""

    def test_recent_trend_is_shrunk_by_its_t_statistic(self):
        rates, cpi = fm.ASSET_TYPES["rates"], fm.ASSET_TYPES["cpi"]
        rng = np.random.default_rng(1)
        strong, noise = rng.normal(0.5, 1.0, 400), rng.normal(0.0, 1.0, 400)
        for v in (strong, noise):
            w = v[-126:]
            t = w.mean() / (w.std(ddof=1) / np.sqrt(126))
            self.assertAlmostEqual(fm._recent_trend(v, rates, 9.9), w.mean() * max(0.0, 1.0 - 4.0 / t ** 2), places=12)
        self.assertGreater(fm._recent_trend(strong, rates, 0.0), 0.3)          # t ~ 5.6: most of it kept
        # CPI and payrolls: the plain mean of the last 12 monthly steps, or the window's drift if too short
        self.assertAlmostEqual(fm._recent_trend(strong, cpi, 9.9), strong[-12:].mean(), places=12)
        self.assertEqual(fm._recent_trend(strong[:5], cpi, 9.9), 9.9)

    def test_zero_floor_on_treasury_yields(self):
        n, rng = 600, np.random.default_rng(11)
        dates = _bdays(n)
        vals = np.abs(rng.normal(0.25, 0.02, n))                                 # a yield just above 0 %
        table = pa.table({"date": dates, "asset": ["UST_2Y"] * n, "value": vals.tolist(), "panel_id": ["rates_daily"] * n})
        args = ({"p": table}, ["UST_2Y"], [126], dates[-1], {"UST_2Y": dict(NEUTRAL)}, 4000, 0)
        on = fa.build_draws(*args)
        with mock.patch.object(fm, "ZERO_FLOOR", False):
            off = fa.build_draws(*args)
        self.assertGreater((off < 0).mean(), 0.05)                              # the walk alone goes below 0
        self.assertTrue(np.all(on >= 0.0))
        np.testing.assert_array_equal(on, np.maximum(off, 0.0))

    def test_variance_ratio(self):
        rng, h = np.random.default_rng(5), 126
        walk = rng.normal(0, 1, 5000)
        trend = np.convolve(rng.normal(0, 1, 5100), np.ones(20) / 20, mode="valid")[:5000]   # moves keep going
        revert = np.diff(rng.normal(0, 1, 5001))                                             # moves cancel
        self.assertAlmostEqual(fm._variance_ratio(walk, h, False), 1.0, delta=0.15)
        self.assertGreater(fm._variance_ratio(trend, h, False), 1.5)
        self.assertLess(fm._variance_ratio(revert, h, False), 0.6)
        c = np.concatenate([[0.0], np.cumsum(trend)])
        raw = np.mean((c[h:] - c[:-h]) ** 2) / (h * np.mean(trend ** 2))
        n = len(trend) / h
        self.assertAlmostEqual(fm._variance_ratio(trend, h, False), 1 + (raw - 1) * n / (n + 30), places=12)
        self.assertEqual(fm._variance_ratio(walk, 1, False), 1.0)                # one step: nothing to correct

    def test_trend_scenario_moves_the_last_quarter(self):
        table, asof, _ = _typed_panel("UST_10Y", "rates_daily", start=100.0, trend=0.05, seed=3)
        n, h = 1000, 126
        args = ({"p": table}, ["UST_10Y"], [h], asof, {"UST_10Y": dict(NEUTRAL)}, n, 0)
        mix = fa.build_draws(*args)[:, 0, 0]
        with mock.patch.object(fm, "TREND_SHARE", 0.0):
            one = fa.build_draws(*args)[:, 0, 0]
        k = int(round(0.25 * n))
        np.testing.assert_array_equal(mix[: n - k], one[: n - k])
        f = fm._fit_model(fm._Request({"p": table}, ["UST_10Y"], [h], asof, {}, n, 0, "level", None))
        np.testing.assert_allclose(mix[n - k:] - one[n - k:], h * (f.mu[0] - f.drift["UST_10Y"]), rtol=0, atol=1e-9)

    def test_single_horizon_by_hand(self):
        """All of M1.9 on one rates asset, by hand: the first 3/4 of the draws centred on
        last + h x shrunk trend, the last 1/4 on last + h x window trend; each shock is
        z x sqrt(h sd^2 + jitter) x sqrt(VR) x sqrt((nu - 2) / W); floored at 0."""
        table, asof, values = _typed_panel("UST_10Y", "rates_daily", df=5, seed=8, start=100.0, trend=0.4)
        n, h, seed = 1000, 63, 4
        out = fa.build_draws({"p": table}, ["UST_10Y"], [h], asof, {"UST_10Y": dict(NEUTRAL)}, n, seed,
                             target_type="level")[:, 0, 0]
        steps = np.diff(values)
        win, last126 = steps[-fm.WINDOW:], steps[-126:]
        nu = fm._fit_nu(win[None, :])
        self.assertIsNotNone(nu)
        sd = fm._ewma_sd(steps[-int(8 * fm.EWMA_HALFLIFE):], fm.EWMA_HALFLIFE)
        t = last126.mean() / (last126.std(ddof=1) / np.sqrt(126))
        drift = last126.mean() * max(0.0, 1.0 - 4.0 / t ** 2)
        self.assertGreater(drift, 0.0)
        c = np.concatenate([[0.0], np.cumsum(steps)])
        raw = np.mean((c[h:] - c[:-h]) ** 2) / (h * np.mean(steps ** 2))
        m = len(steps) / h
        vr = 1 + (raw - 1) * m / (m + 30)
        z = np.random.default_rng(seed).standard_normal((n, 1))[:, 0]
        w = np.random.default_rng([seed, 7]).chisquare(nu, size=(n, 1))[:, 0]
        shock = z * np.sqrt(h * sd ** 2 + 1e-10 + 1e-9) * np.sqrt(vr) * np.sqrt((nu - 2) / w)
        centre = np.where(np.arange(n) < n - 250, values[-1] + h * drift, values[-1] + h * win.mean())
        np.testing.assert_allclose(out, np.maximum(centre + shock, 0.0), rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
