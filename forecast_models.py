"""
Track 2 — the forecast models: panels (+ the text adjustments) -> joint draws.

Split out of `forecast_agent.py`, which is now the contract and the CLI. Everything here is about
turning numbers into a distribution; nothing here reads the text corpus or touches argv.

    build_draws(panels, assets, horizons, asof, adjustments, n_draws, seed,
                *, target_type=..., family=...) -> (n_draws, n_assets, n_horizons)

ONE MODEL for every card: the CUMULATIVE WALK. One correlated path per draw, from the last
value, with Student-t shocks; each horizon is read off the path. Only its settings (window, widen,
Student-t shocks, EWMA volatility) change by family, in `WALK_SETTINGS`. On a single-horizon card
the path has one leg: centre + shock * scale * sqrt(steps).

One model on purpose (2026-09-27): the text's `shift`, `widen` and `skew` then mean the same thing
on every card, which is what the LLM half is tuned against. M2 (f1_pipeline) stays a research
model; it scored better on the 4 visible monthly cards (model_baseline notebook 06) but is not in
production.

Runs offline with numpy + pandas + pyarrow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa

_ASSET_COLS = ("asset", "asset_id")

#: Trailing rows used to estimate sd and the cross-asset correlation. PINNED, and not a free
#: parameter: the window dominates the choice of method. Measured on the 20 F3 units with realized
#: vectors (mean variogram, horizons drawn independently): 260 -> 2.47, 504 -> 2.58, 1260 -> 2.77,
#: full history -> 3.02. Changing it at the same time as anything else makes the comparison
#: uninterpretable, so change it alone or not at all.
_WINDOW = 260

#: The walk's settings per family -- the main model since 2026-09-27, chosen in model_baseline
#: notebooks 02-05: tuned on each card's train dates, kept only if better than the previous model
#: on validation, then judged once on test.
#:   window    steps the sd and the cross-asset correlation are measured on
#:   widen     multiplier on the sd (on top of the text's `widen`)
#:   nu        Student-t degrees of freedom for the shocks (unit variance; None = normal)
#:   halflife  EWMA sd, weights halving every `halflife` steps (None = the plain sd of `window`)
#: Test results against the previous step: Student-t 0.997x (78/103 cards better, 98% band 93.5% ->
#: 96.1%); EWMA kept on F4 only, 0.994x there. A family not listed gets the plain production walk.
#: F1 changed on 2026-09-28 to the family-global setting of the grid search in
#: hyperparameter_tuning/01 (480 settings, 60/20/20 split, picked on validation): EWMA sd, halflife
#: 21, widen 0.9. Against the old F1 setting (42 / 1.2 / nu 5 / window sd) it scored 0.974x on the
#: grid's test block and 0.891x on the team eval's real outcomes (12 of 18 cards better); F2-F4 were
#: kept, since their family globals did not beat production on either check.
WALK_SETTINGS = {
    "T2-F1": {"window": 130, "widen": 0.9, "nu": 5, "halflife": 21},
    "T2-F2": {"window": 130, "widen": 1.1, "nu": 4, "halflife": None},
    "T2-F3": {"window": 130, "widen": 1.2, "nu": 4, "halflife": None},
    "T2-F4": {"window": 260, "widen": 1.2, "nu": 5, "halflife": 63},
}
_PLAIN_WALK = {"window": _WINDOW, "widen": 1.0, "nu": None, "halflife": None}

#: Trend (drift) per family, as a multiple of the mean step over the trailing `trend_window`
#: rows -- the official baseline's own centre rule (docs/M0-BASELINE.md 3.5 and 3.8: mean = anchor
#: + steps * mean step, over its last 300 observations). A family absent here, or `trend` 0,
#: centres on the last value as before. Settings are read through `_trend_cfg` so a family can
#: opt in without touching WALK_SETTINGS.
#:
#: The trend fills in ONLY where the text gave that asset no centre view (shift == 0): a stated
#: view from the documents replaces the historical drift, it does not add to it. Measured
#: 2026-09-29 on F4 (29 cards, leaderboard metric vs rebuilt M0, recorded v5 answers, 3 seeds):
#: trend added on top of every text view 0.840 vs 0.842 (helps one half, hurts the other);
#: trend only where the text has no view 0.809 vs 0.816 (21 better / 8 worse); text-off F4
#: 1.004 -> 0.956. LEVEL targets only: on the 11 F4 log_return cards the trend hurts (0.788 ->
#: 0.832, 4 better / 7 worse; all 15 log_return cards 0.839 -> 0.877), which is Pun's earlier
#: finding for the log-return centre, re-measured on the corrected realized values. Text-off on
#: the other families with the same rule: F1 1.031 -> 0.960, F2 1.078 -> 1.059, F3 0.967 -> 0.994
#: (worse) -- so it is on for F4 only; the other families are Pun's call.
TREND_SETTINGS: dict[str, dict[str, float]] = {"T2-F4": {"trend": 1.0, "trend_window": 300}}
_TREND_DEFAULT = {"trend": 0.0, "trend_window": 300}


def _trend_cfg(family: str | None) -> dict[str, float]:
    return {**_TREND_DEFAULT, **TREND_SETTINGS.get(family or "", {})}
#: Per-family FALLBACK width, used only when the text half gave the card nothing (every asset
#: exactly neutral: the model call failed, timed out, ran out of budget, or no usable summaries).
#: F4 cards are built around a shock the calm history does not show, so a text-less F4 card
#: should not draw history's width. Measured 2026-09-29 on the 29 realized F4 cards, leaderboard
#: metric vs the rebuilt M0 (docs/M0-BASELINE.md), 3 draw seeds, text silent: 1.004 with nothing,
#: 0.956 with the trend below alone, 0.918 with trend + 1.25 (1.5 was measured earlier as too wide
#: once the text answers). It never overrides an answer.
_FAMILY_WIDEN_FLOOR = {"T2-F4": 1.25}


def _text_was_silent(adjustments: dict[str, dict[str, float]], assets: list[str]) -> bool:
    """True when every asset carries the exact neutral adjustment -- the text half's failure
    shape (`read_text_signal` degrades to neutral rather than raising)."""
    return all(adjustments.get(a, {}).get("shift", 0.0) == 0.0
               and adjustments.get(a, {}).get("widen", 1.0) == 1.0
               and adjustments.get(a, {}).get("skew", 0.0) == 0.0 for a in assets)


#: ON since 2026-09-25, after running the exact comparison the previous comment here demanded:
#: "skew forced on vs. off against ONE recorded set of model adjustments, not two fresh live
#: calls". That was impossible until the stage-2 ledger started being recorded; replaying tonight's
#: F3 sweep with the flag flipped and everything else held (same adjustments, same draws, same
#: seeds, no new model calls) isolates the code path exactly as asked.
#:
#: Result on the 3 F3 units where the model asked for a non-zero skew: composite ratio ON/OFF
#: 0.9865, 1.0000, 0.9992 -- mean 0.9952, nothing worse. Weak evidence, and honestly so: 5 assets,
#: every one at skew=-0.20, on the family where skew matters LEAST (F3 is scored on the joint
#: term, not the tail).
#:
#: The stronger argument is structural. Stage 2 asks for a skew on every card and F4's prompt
#: explicitly instructs the model to "use skew to point the distribution toward the side the shock
#: would move prices" -- F4 is scored primarily on the tail penalty, which is precisely what a
#: tilt shapes. Discarding a value we ask for, on the family whose primary term it exists to move,
#: was the incoherent state. Off was the safe default while nobody could measure it; that is no
#: longer true.
#:
#: STILL UNVALIDATED ON F4. The replay above covers F3 only, because no F4 sweep has been recorded
#: since the ledger landed. Do that before trusting skew to earn anything on the family it is
#: really for.
SKEW_ENABLED = True

def build_draws(
    panels: dict[str, "pa.Table"],
    assets: list[str],
    horizons: list[int],
    asof: str,
    adjustments: dict[str, dict[str, float]],
    n_draws: int,
    seed: int,
    *,
    target_type: str | None = None,
    family: str | None = None,
) -> np.ndarray:
    """Joint draws with Nish's adjustments applied. Returns (n_draws, n_assets, n_horizons).

    Every card takes the cumulative walk; `family` picks its settings (`WALK_SETTINGS`).
    """
    return _cumulative_walk_model(_Request(panels, assets, horizons, asof, adjustments, n_draws,
                                           seed, target_type, family))


# ============================================================================
#  The request, and the walk.
# ============================================================================
@dataclass(frozen=True)
class _Request:
    """One card's forecast request: everything the walk needs, in one place."""
    panels: dict[str, "pa.Table"]
    assets: list[str]
    horizons: list[int]
    asof: str
    adjustments: dict[str, dict[str, float]]
    n_draws: int
    seed: int
    target_type: str | None
    family: str | None

    @property
    def returns_target(self) -> bool:
        """A cumulative log-return card: the panel holds per-period returns, not levels."""
        return self.target_type == "log_return"


# ------------------------------------------------------------------------ the walk
@dataclass(frozen=True)
class _WalkFit:
    """What the walk is fitted from -- everything except how the path is drawn.

    Fitting is where all the care is (date alignment, gap guarding, log-return handling, the
    PSD-repaired correlation), so it lives apart from the five lines that draw the path.
    """
    centre: np.ndarray          # last observed value (0 for a log-return card) + the text shift
    scale: np.ndarray           # per-asset daily sd, times the text's widen
    drift: np.ndarray           # per-asset trend per panel step (zeros unless TREND_SETTINGS says)
    chol: np.ndarray            # lower-triangular factor of the cross-asset correlation
    skew: np.ndarray            # per-asset tail tilt, all zeros while SKEW_ENABLED is False
    steps: np.ndarray           # (n_assets, n_horizons): each horizon in the asset's own panel steps
    nu: float | None = None     # Student-t degrees of freedom of the shocks (None = normal)

    def shock(self, rng: np.random.Generator, n_draws: int,
              rng_t: np.random.Generator | None = None) -> np.ndarray:
        """One correlated, optionally skew-tilted, optionally Student-t standard innovation per draw.

        Every leg of the path is drawn here, so the cross-asset structure, the skew tilt and the
        tail shape are defined once. The Student-t is multivariate: every asset of a draw shares one
        chi-square, so assets have their extreme moves together and the correlation is unchanged;
        (nu - 2) keeps the variance at 1, so `scale` means the same with or without it. The
        chi-square comes from its own generator `rng_t`, so the normal part of each draw is the
        same number as in the normal walk.
        """
        n = len(self.scale)
        z = rng.standard_normal((n_draws, n)) @ self.chol.T   # correlated across assets
        u = np.abs(rng.standard_normal((n_draws, n)))         # independent per asset -- skew only
        z = _skew_tilt(z, u, self.skew)
        if self.nu is not None:
            z = z * np.sqrt((self.nu - 2.0) / rng_t.chisquare(self.nu, size=(n_draws, 1)))
        return z


def _fit_walk(r: _Request) -> _WalkFit:
    """Panels -> the scale, correlation and centre the walk draws from.

    Every correctness fix lives here rather than in either model, because none of them is about
    F3: the gap guard's worst case is an F2 transfer card (measured 1.40x inflated sd on
    t2-F2-fragile-five-brl-2013) and the log-return centring carries 11 F4 units.
    """
    cfg = WALK_SETTINGS.get(r.family, _PLAIN_WALK)
    tcfg = _trend_cfg(r.family)
    hist = {a: _series(r.panels, a, r.asof) for a in r.assets}
    hl = cfg["halflife"]
    steps = _step_frame(hist, r.assets, r.returns_target,
                        rows=max(cfg["window"], int(8 * hl) if hl else 0,
                                 int(tcfg["trend_window"]) if tcfg["trend"] else 0))
    drift = (tcfg["trend"] * steps.iloc[-int(tcfg["trend_window"]):].to_numpy().mean(axis=0)
             if tcfg["trend"] and not r.returns_target else np.zeros(len(r.assets)))
    # a text view on the centre replaces the historical drift for that asset
    drift = np.where(np.array([r.adjustments.get(a, {}).get("shift", 0.0) for a in r.assets]) == 0.0,
                     drift, 0.0)
    # The last `window` steps always give the correlation between assets. They give the sd only when
    # the family has no halflife; with one (F1, F4) the sd is the EWMA of the last 8 x halflife steps
    # and `window` matters only on cards with more than one asset.
    D = steps.iloc[-cfg["window"]:].to_numpy().T  # (n_assets, window)
    n = len(r.assets)
    sd = D.std(axis=1) if hl is None else _ewma_sd(steps.to_numpy()[-int(8 * hl):], hl)

    # A cumulative log-return target starts at 0; a level target starts at its last observed
    # value. Neither carries a drift term. For levels that is just the random walk. For log
    # returns it is a deliberate departure from the reference CLI (`cli.py:278`), which centres
    # them on `steps.mean() * h` -- a 260-day mean daily return extrapolated over the horizon.
    # Measured on all 15 log_return units with realized vectors, 5 seeds, 4000 draws: zero drift
    # wins 11/15, mean normalized composite ratio 0.8818. The extrapolation is a noisy momentum
    # bet (t2-F4-short-vol-2018: 0.01573 -> 0.00587 without it); the martingale is both the
    # standard choice for returns and, here, the measured one.
    last = (np.zeros(n) if r.returns_target
            else np.array([hist[a].iloc[-1] for a in r.assets], dtype=float))

    # np.corrcoef on a single-row input (single-asset cards) returns a 0-d SCALAR, not a (1,1)
    # matrix -- fill_diagonal then fails with "array must be at least 2-d". atleast_2d fixes the
    # single-asset case (correlation of one variable with itself is trivially [[1.0]]) and is a
    # no-op for multi-asset cards, where corrcoef already returns a proper 2-d matrix.
    corr = np.atleast_2d(np.corrcoef(D))
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    w, v = np.linalg.eigh(corr)                  # nearest-PSD nudge
    corr = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T

    def per_asset(key: str, default: float) -> np.ndarray:
        return np.array([r.adjustments.get(a, {}).get(key, default) for a in r.assets])

    # The family's widen times the text's, on every card the text answered.
    widen = cfg["widen"] * per_asset("widen", 1.0)
    if _text_was_silent(r.adjustments, r.assets):   # the text half failed: see _FAMILY_WIDEN_FLOOR
        widen = np.maximum(widen, cfg["widen"] * _FAMILY_WIDEN_FLOOR.get(r.family or "", 1.0))

    return _WalkFit(
        centre=last + per_asset("shift", 0.0),
        scale=sd * widen,
        drift=np.asarray(drift, dtype=float),
        chol=np.linalg.cholesky(corr),
        skew=per_asset("skew", 0.0) if SKEW_ENABLED else np.zeros(n),
        steps=np.array([_panel_steps(hist[a], r.horizons, r.asof) for a in r.assets]),
        nu=cfg["nu"],
    )


def _ewma_sd(past: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA sd per column of `past` (rows oldest -> newest): a step's weight halves every
    `halflife` steps back, around the weighted mean. Equals pandas' ewm(halflife).std(bias=True)."""
    age = np.arange(len(past))[::-1]
    w = 0.5 ** (age / halflife)
    w = w / w.sum()
    mean = w @ past
    return np.sqrt(w @ (past - mean) ** 2)


#: A panel whose rows are further apart than this (median, in calendar days) is a monthly panel.
_MONTHLY_SPACING_DAYS = 20


def _panel_steps(s: "pd.Series", horizons: list[int], asof: str) -> np.ndarray:
    """The card's horizons, counted in steps (rows) of this asset's own panel.

    `scale` is the sd of ONE STEP of the panel, so the walk's width is scale * sqrt(steps). On a
    daily panel a step is a business day and the card's horizon already counts them: returned
    unchanged. On a monthly panel a step is a month, but the card still states its horizon in
    business days, so sqrt(21) would walk 21 MONTHS for a 21-business-day card (3.2x too wide on
    t2-F4-covid-nfp-2020, 4.2x on t2-F1-cpi-glidepath-2023). There the horizon becomes the number
    of months from the asset's LAST observation -- not the as-of, which monthly data lags by its
    publication delay -- to the month of `asof + h business days`. That is the official baseline's
    rule (docs/M0-BASELINE.md, section 3.7, whose table lists these four cards) and M2's
    (f1_pipeline/m2_unit.steps_ahead_for). Backtest on the 4 monthly F1/F4 cards: walk 0.64x.

    Assumes monthly cards keep stating business days, as every shipped one does; M0-BASELINE.md
    section 8 says a restatement in panel steps is in preparation, and would need this revisited.
    """
    h = np.array([float(x) for x in horizons])
    when = pd.to_datetime(pd.Series(s.index), errors="coerce").dropna()
    if len(when) < 3 or not float(when.diff().dt.days.median()) > _MONTHLY_SPACING_DAYS:
        return h
    last = when.iloc[-1]
    months = []
    for x in horizons:
        target = pd.Timestamp(asof) + pd.offsets.BDay(int(x))
        k = 12 * (target.year - last.year) + (target.month - last.month)
        months.append(float(k) if k > 0 else float(x))
    return np.array(months)


def _cumulative_walk_model(r: _Request) -> np.ndarray:
    """Every card. ONE accumulating path per draw: horizon h_k is reached by walking there through h_1.

    A single-horizon card is one leg: centre + shock * scale * sqrt(steps).

    Each leg adds an increment of sd*sqrt(h_k - h_k-1), so after the leg ending at h_k the path
    has variance sd^2 * sum(h_j - h_j-1) = sd^2 * h_k -- identical to drawing that horizon on its
    own, which is why this moves the joint term without touching the marginal or tail (measured:
    marginal ratio 1.0006 on F3). What it adds is the covariance an independent draw throws away:
    Cov(path_hj, path_hk) = sd^2 * h_j, i.e. rho = sqrt(h_j / h_k).

    That value is the point, and it is NOT "as much correlation as possible". Measured on the 20
    F3 units, forcing a flat cross-horizon rho: 0.0 -> 2.471, 0.577 -> 2.231, 0.707 -> 2.225,
    1.0 -> 2.487. rho=1 scores as badly as rho=0. The variogram is a proper scoring rule, so the
    calibrated gap variance is optimal in expectation -- this reaches 2.162, beating every flat
    rho. Do not "improve" it by pushing the correlation up.
    """
    fit = _fit_walk(r)
    rng, rng_t = np.random.default_rng(r.seed), np.random.default_rng([r.seed, 7])
    out = np.empty((r.n_draws, len(r.assets), len(r.horizons)), dtype=float)
    path = np.zeros((r.n_draws, len(r.assets)))
    prev = np.zeros(len(r.assets))
    for hi in np.argsort(r.horizons):            # cards ship ascending; don't rely on it
        steps = fit.steps[:, hi]                 # per asset, in its own panel's steps
        # maximum(..., 0) guards a duplicated or unsorted horizon against a silent NaN under sqrt.
        path = path + fit.shock(rng, r.n_draws, rng_t) * fit.scale * np.sqrt(np.maximum(steps - prev, 0))
        prev = steps
        out[:, :, hi] = fit.centre + fit.drift * steps + path   # write to the ORIGINAL index
    return out


#: The skew-normal family's sample skewness is bounded (|.| < ~0.995 as shape -> infinity) and
#: rises slowly: shape=1 (the top of `skew`'s own [-1,1] contract if used directly) reaches only
#: ~0.14 sample skewness -- a barely-visible tilt, not the "shock" F4 needs (see NISH_TEXT_NOTES.md
#: and PUN_TEXT_NOTES.md). Scaling `skew` up to a shape parameter of +-5 at the clamp's edge
#: reaches ~0.85 instead -- a strong, visibly asymmetric tail without pinning at the family's
#: near-degenerate ceiling.
_SKEW_SHAPE_SCALE = 5.0


def _skew_tilt(z: np.ndarray, u: np.ndarray, skew: np.ndarray) -> np.ndarray:
    """Azzalini's skew-normal construction, per asset (last axis of `z`/`u`, matching `skew`).

    Mixes the correlated roll `z` with an independent |normal| term `u`, weighted by `delta`,
    then recenters and rescales so the result has mean 0 and unit variance for EVERY value of
    skew -- `shift` and `widen` keep meaning exactly what they already mean upstream, and `skew`
    changes shape only. At skew=0, delta=0 and this returns `z` exactly (see
    `test_skew_zero_is_identity`): every card that never asks for a tilt draws identically to
    before this function existed.

    `u` must be independent PER ASSET, drawn separately from `z` -- it must not touch the
    cross-asset correlation `chol` already encodes elsewhere, which this leaves untouched. A skew
    that reaches across assets (e.g. "both legs of this trade break the same way") is real future
    work, not this.
    """
    alpha = skew * _SKEW_SHAPE_SCALE
    delta = alpha / np.sqrt(1.0 + alpha**2)
    mean_shift = delta * np.sqrt(2.0 / np.pi)
    var_scale = np.sqrt(np.clip(1.0 - delta**2 * (2.0 / np.pi), 1e-8, None))
    return (delta * u + np.sqrt(1.0 - delta**2) * z - mean_shift) / var_scale


def _step_frame(hist: dict[str, "pd.Series"], assets: list[str],
                returns_target: bool, rows: int = _WINDOW) -> "pd.DataFrame":
    """Per-asset histories -> one date-aligned matrix of steps, trimmed to the trailing window.

    Three things here are load-bearing and easy to undo by accident:

    `[assets]` reselects explicitly. Every downstream vector (sd, shift, widen, skew) is built in
    `assets` order, so if the frame's column order ever diverged the Cholesky would be applied to
    the wrong assets -- silently, with no exception, just a wrong correlation structure.

    `.dropna()` comes BEFORE the window, not after. The other way round a cross-panel card keeps
    260 raw rows and then loses a fraction of them to the date intersection, so the effective
    window would differ per card.

    A log_return panel already holds per-period returns, so its steps are log1p(row) rather than
    a difference of rows.
    """
    frame = pd.DataFrame(
        {a: (pd.Series(_log_return_steps(s.to_numpy()), index=s.index) if returns_target
             else _diff_without_gaps(s))
         for a, s in hist.items()}
    )[assets].dropna().iloc[-rows:]
    if len(frame) < 30:
        raise SystemExit(
            f"not enough overlapping history to estimate covariance ({len(frame)} rows)")
    return frame


def _diff_without_gaps(s: "pd.Series") -> "pd.Series":
    """First differences, dropping any difference that spans a hole in the data.

    Ported from `qfbench2_track_forecasting/cli.py:_diff_without_gaps` (kept local rather than
    imported: this file deliberately depends on nothing in the toolkit).

    A transfer card ships its target as an early window plus a single row at the as-of date, with
    the years between deliberately withheld. Differenced naively, that hole reads as one day in
    which the asset moved a decade's worth. Measured on t2-F2-fragile-five-brl-2013: one
    gap-spanning "day" of -0.875 against a typical daily move of 0.030, inflating the estimated
    daily sd by 1.4x. t2-F3-fragile-five-joint-2013 has the same 3654-day hole inside its window.
    The threshold adapts to the panel's own spacing, so a gapless daily panel is untouched.
    """
    d = s.diff()
    when = pd.to_datetime(pd.Series(s.index, index=s.index), errors="coerce")
    step = when.diff().dt.days
    if step.notna().sum() == 0:
        return d
    return d.where(step <= max(float(step.median()) * 10.0, 5.0))


def _log_return_steps(values: np.ndarray) -> np.ndarray:
    """Simple returns -> additive log-return steps, for a cumulative log target.

    Mirrors `qfbench2_track_forecasting/targets.py:log_return_steps`, guard included. Without
    this a log_return card is anchored at its last simple return and its already-return data is
    differenced again -- both wrong. Every row is already a step here, so unlike the level path
    nothing is lost to a leading NaN; the reference does the same.
    """
    rows = np.asarray(values, dtype=np.float64)
    if np.any(rows <= -1.0) or np.any(np.isinf(rows)):
        raise ValueError("log_return history requires finite simple returns greater than -1")
    return np.log1p(rows)


def _series(panels: dict[str, "pa.Table"], asset: str, asof: str) -> "pd.Series":
    """History of one asset up to and including the as-of, from whichever panel holds it.

    Returns a DATE-INDEXED series, not a bare array. The index is what lets `build_draws` align
    assets that live in different panel files by date instead of by row position -- six F3 units
    span two panels whose calendars differ, and stacking those positionally silently mis-estimates
    the cross-asset correlation (measured on t2-F3-divergence-2014: UST_10Y/JPY reads 0.32
    positionally against 0.50 date-aligned).
    """
    seen: set[str] = set()
    for t in panels.values():
        cols = t.column_names
        acol = next((c for c in _ASSET_COLS if c in cols), None)
        if acol is None:
            continue
        d = t.to_pydict()
        seen.update(str(a) for a in d[acol])
        rows = [
            (str(dt)[:10], float(v))
            for dt, a, v in zip(d["date"], d[acol], d["value"])
            if str(a) == asset and str(dt)[:10] <= asof
        ]
        if rows:
            rows.sort()
            idx = pd.to_datetime([r[0] for r in rows])
            return pd.Series([r[1] for r in rows], index=idx, name=asset)
    raise SystemExit(
        f"asset {asset!r} not found in any panel at/before {asof}; "
        f"the panels carry {sorted(seen)}"
    )
