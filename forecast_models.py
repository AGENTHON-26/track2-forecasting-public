"""
Track 2 — the forecast models: panels (+ the text adjustments) -> joint draws.

Split out of `forecast_agent.py`, which is now the contract and the CLI. Everything here is about
turning numbers into a distribution; nothing here reads the text corpus or touches argv.

    build_draws(panels, assets, horizons, asof, adjustments, n_draws, seed,
                *, target_type=..., family=...) -> (n_draws, n_assets, n_horizons)

THREE MODELS, chosen by `_select_model`:

    M2                ridge location-scale + a joint bootstrap of standardised residuals, from
                      f1_pipeline. F1 level cards only -- that is where it was fitted.
    cumulative walk   one accumulating correlated path per draw. Multi-horizon cards.
    random walk       one correlated draw per horizon. Everything else.

The two walks share their whole fitting step (`_fit_walk`) and differ only in how the legs are
combined, so neither is a copy of the other.

Runs offline with numpy + pandas + pyarrow; the M2 path additionally imports f1_pipeline.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
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
#: really for. M2 (F1 level cards) ignores skew by design -- its residual pool already carries the
#: empirical shape.
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

    This function is only the dispatch. Pick the model in `_select_model`, read the model in its
    own function below; nothing about one family's model is tangled into another's.
    """
    global _last_model
    request = _Request(panels, assets, horizons, asof, adjustments, n_draws, seed,
                       target_type, family)
    name, model = _select_model(family, target_type, horizons)
    try:
        out = model(request)
    except Exception as exc:
        # A card that raises scores the pre-committed worst case (4.0); the walk scores ~1.0. So
        # any model failure falls back rather than propagating -- M2 in particular refuses cards
        # whose assets span two panel files, which is 6 of the 22 F3 units and the reason M2 is
        # not used there.
        if name == RANDOM_WALK:
            raise
        print(f"[{name}] {type(exc).__name__}: {exc}; falling back to the random walk",
              file=sys.stderr)
        name, model = RANDOM_WALK, _random_walk_model
        out = model(request)
    _last_model = name
    return out


# ============================================================================
#  The three models, and the switch that chooses between them.
# ============================================================================
@dataclass(frozen=True)
class _Request:
    """One card's forecast request, as every model receives it.

    A single shape for all three models is what lets `_select_model` be a plain lookup instead of
    three different call signatures at the call site.
    """
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


#: Model names. These reach the reader twice -- in the stderr fallback line and in
#: forecast_rationale.md -- so they are constants rather than repeated literals.
M2 = "M2"
CUMULATIVE_WALK = "cumulative walk"
RANDOM_WALK = "random walk"


def _select_model(family: str | None, target_type: str | None,
                  horizons: list[int]) -> tuple[str, "_Model"]:
    """THE SWITCH. Which model runs this card, and why.

        F1, level target   -> M2                 ridge location-scale + residual bootstrap.
                                                 21 of 23 F1 units. Measured 0.832x the walk on
                                                 F1's own units; measured WORSE than the walk on
                                                 F3 (1.045x, and it cannot even load 4 of 17 of
                                                 them), so it stays where it was tuned. This is
                                                 the one branch that is about a family, because
                                                 M2 is a fitted model and F1 is what it was fitted
                                                 on.
        more than 1 horizon -> cumulative walk   one accumulating path per draw, so the horizons
                                                 of an asset are correlated at sqrt(h_j/h_k)
                                                 instead of independent. Today that is exactly
                                                 F3's 22 units.
        everything else    -> random walk        one draw per horizon. Today: all 27 F2 units, all
                                                 31 F4 units, and F1's 2 log_return units.

    ON THE MIDDLE BRANCH BEING SHAPE, NOT FAMILY. It reads as "F3's model", and today it selects
    exactly F3, because F3 is currently the only family shipping a multi-horizon walk card. But
    the rule it encodes is not about F3: a value at horizon h_k IS reached by walking through
    h_j, whatever family the card belongs to. Keying on `T2-F3` would make a correctness property
    conditional on a metadata string.

    That distinction is not hypothetical. Every shipped F2 and F4 dev unit is single-horizon, but
    the organizers' own example cards in docs/CATEGORIES.md are not: F2's is "GBP/USD at horizons
    21 BD and 63 BD" and F4's is "UST_2Y, UST_10Y at 63 BD and 126 BD". On a sealed card shaped
    like either, a family switch would quietly hand back the cross-horizon structure -- and on the
    F2 one it would cost the entire joint term, since a 1-asset 2-horizon card has exactly one
    off-diagonal pair and that pair IS the cross-horizon pair.

    Both walks share their entire fitting step (`_fit_walk`) and differ only in how the legs are
    put together -- see each function.
    """
    if target_type == "level" and family in _M2_FAMILIES:
        return M2, _m2_model
    if len(horizons) > 1:
        return CUMULATIVE_WALK, _cumulative_walk_model
    return RANDOM_WALK, _random_walk_model


#: Families whose LEVEL cards use M2 (f1_pipeline/m2_unit.py). Tuned on F1 only (f1_pipeline
#: notebooks 01-04), and measured worse than the walk everywhere else, so it stays here. This is
#: the only family-keyed routing decision: M2 is a fitted model, and a family is the right scope
#: for "where was this fitted". The walk split below is about card shape instead.
_M2_FAMILIES = {"T2-F1"}


# ---------------------------------------------------------------- model 1: M2 (F1 level cards)
def _m2_model(r: _Request) -> np.ndarray:
    """Ridge centre, ridge log-variance width, and a joint bootstrap of standardised residuals
    sharing one historical date per draw (f1_pipeline/m2_unit.py; notebooks 01-04).

    `skew` is deliberately not applied: the residual pool already carries the empirical shape.
    """
    from f1_pipeline import m2_unit as m2

    unit = m2.unit_from_panels(r.panels, r.assets, r.horizons, r.asof, target_type="level")
    fits = [m2.fit_m2(c) for c in m2.build_features(unit)]  # asset-major, then horizon: card order
    shift = [r.adjustments.get(f.asset, {}).get("shift", 0.0) for f in fits]
    widen = [r.adjustments.get(f.asset, {}).get("widen", 1.0) for f in fits]
    samples = m2.draw_joint(fits, r.n_draws, r.seed, shift, widen)
    return samples.reshape(r.n_draws, len(r.assets), len(r.horizons))


# ------------------------------------------------- models 2 & 3: the two correlated walks
@dataclass(frozen=True)
class _WalkFit:
    """What both walks are fitted from -- everything that does not depend on how legs combine.

    Fitting is where all the care is (date alignment, gap guarding, log-return handling, the
    PSD-repaired correlation), and it is identical for both walks. Keeping it here is what stops
    the two models from being near-copies of each other: each is then five lines that say only
    what makes it different.
    """
    centre: np.ndarray          # last observed value (0 for a log-return card) + the text shift
    scale: np.ndarray           # per-asset daily sd, times the text's widen
    chol: np.ndarray            # lower-triangular factor of the cross-asset correlation
    skew: np.ndarray            # per-asset tail tilt, all zeros while SKEW_ENABLED is False

    def shock(self, rng: np.random.Generator, n_draws: int) -> np.ndarray:
        """One correlated, optionally skew-tilted standard innovation per draw.

        Both walks draw their legs through here, so the cross-asset structure and the skew tilt
        are defined once.
        """
        n = len(self.scale)
        z = rng.standard_normal((n_draws, n)) @ self.chol.T   # correlated across assets
        u = np.abs(rng.standard_normal((n_draws, n)))         # independent per asset -- skew only
        return _skew_tilt(z, u, self.skew)


def _fit_walk(r: _Request) -> _WalkFit:
    """Panels -> the scale, correlation and centre both walks draw from.

    Every correctness fix lives here rather than in either model, because none of them is about
    F3: the gap guard's worst case is an F2 transfer card (measured 1.40x inflated sd on
    t2-F2-fragile-five-brl-2013) and the log-return centring carries 11 F4 units.
    """
    hist = {a: _series(r.panels, a, r.asof) for a in r.assets}
    steps = _step_frame(hist, r.assets, r.returns_target)
    D = steps.to_numpy().T                       # (n_assets, window)
    n = len(r.assets)

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

    return _WalkFit(
        centre=last + per_asset("shift", 0.0),
        scale=D.std(axis=1) * per_asset("widen", 1.0),
        chol=np.linalg.cholesky(corr),
        skew=per_asset("skew", 0.0) if SKEW_ENABLED else np.zeros(n),
    )


def _cumulative_walk_model(r: _Request) -> np.ndarray:
    """F3. ONE accumulating path per draw: horizon h_k is reached by walking there through h_1.

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
    rng = np.random.default_rng(r.seed)
    out = np.empty((r.n_draws, len(r.assets), len(r.horizons)), dtype=float)
    path = np.zeros((r.n_draws, len(r.assets)))
    prev = 0
    for hi in np.argsort(r.horizons):            # cards ship ascending; don't rely on it
        h = int(r.horizons[hi])
        # max(..., 0) guards a duplicated or unsorted horizon against a silent NaN under sqrt.
        path = path + fit.shock(rng, r.n_draws) * fit.scale * np.sqrt(max(h - prev, 0))
        prev = h
        out[:, :, hi] = fit.centre + path        # write to the ORIGINAL index
    return out


def _random_walk_model(r: _Request) -> np.ndarray:
    """F2, F4, and F1's log_return units. Each horizon drawn independently from the anchor.

    Assets are still correlated within a horizon (that is `fit.shock`); horizons are not
    correlated with each other. Every card routed here is single-horizon, where that distinction
    does not exist -- this and the cumulative walk produce the same distribution, from the same
    number of draws off the same generator. It is the honest model for a single-horizon card and
    keeps the reader from having to reason about an accumulation that never happens.
    """
    fit = _fit_walk(r)
    rng = np.random.default_rng(r.seed)
    out = np.empty((r.n_draws, len(r.assets), len(r.horizons)), dtype=float)
    for hi, h in enumerate(r.horizons):
        out[:, :, hi] = fit.centre + fit.shock(rng, r.n_draws) * fit.scale * np.sqrt(int(h))
    return out


#: A model turns one request into (n_draws, n_assets, n_horizons). Declared after the models so
#: the names above are in scope.
_Model = Callable[[_Request], np.ndarray]


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
                returns_target: bool) -> "pd.DataFrame":
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
    )[assets].dropna().iloc[-_WINDOW:]
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



#: Which model produced the most recent `build_draws()` call. `forecast_agent.main()` reads it
#: through `last_model()` for the rationale file; the models themselves never read it.
_last_model = RANDOM_WALK


def last_model() -> str:
    """The model name the most recent `build_draws()` call actually used.

    Not necessarily the one `_select_model` picked: an M2 failure falls back to the walk, and the
    rationale should say what ran.
    """
    return _last_model

