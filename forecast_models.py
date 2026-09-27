"""
Track 2 — the forecast models: panels (+ the text adjustments) -> joint draws.

Split out of `forecast_agent.py`, which is now the contract and the CLI. Everything here is about
turning numbers into a distribution; nothing here reads the text corpus or touches argv.

    build_draws(panels, assets, horizons, asof, adjustments, n_draws, seed,
                *, target_type=..., family=...) -> (n_draws, n_assets, n_horizons)

THREE MODELS, chosen by `_select_model`:

    M2                ridge location-scale + a joint bootstrap of standardised residuals, from
                      f1_pipeline. MONTHLY cards (any family): it beat the tuned walk there on
                      validation (0.660x) and on test (0.481x) -- model_baseline notebook 06.
    cumulative walk   one accumulating correlated path per draw. Multi-horizon daily cards.
    random walk       one correlated draw per horizon. Every other daily card.

The two walks share their whole fitting step (`_fit_walk`) and differ only in how the legs are
combined, so neither is a copy of the other. Their settings (window, widen, Student-t shocks,
EWMA volatility) are per family, in `WALK_SETTINGS`.

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

#: The walk's settings per family -- the main model since 2026-09-27, chosen in model_baseline
#: notebooks 02-05: tuned on each card's train dates, kept only if better than the previous model
#: on validation, then judged once on test.
#:   window    steps the sd and the cross-asset correlation are measured on
#:   widen     multiplier on the sd (on top of the text's `widen`)
#:   nu        Student-t degrees of freedom for the shocks (unit variance; None = normal)
#:   halflife  EWMA sd, weights halving every `halflife` steps (None = the plain sd of `window`)
#: Test results against the previous step: Student-t 0.997x (78/103 cards better, 98% band 93.5% ->
#: 96.1%); EWMA kept on F4 only, 0.994x there. A family not listed gets the plain production walk.
WALK_SETTINGS = {
    "T2-F1": {"window": 42, "widen": 1.2, "nu": 5, "halflife": None},
    "T2-F2": {"window": 130, "widen": 1.1, "nu": 4, "halflife": None},
    "T2-F3": {"window": 130, "widen": 1.2, "nu": 4, "halflife": None},
    "T2-F4": {"window": 260, "widen": 1.2, "nu": 5, "halflife": 63},
}
_PLAIN_WALK = {"window": _WINDOW, "widen": 1.0, "nu": None, "halflife": None}

#: Per-family FALLBACK width for the walk models, used only when the text half gave this card
#: nothing (every asset exactly neutral: model call failed, timed out, ran out of budget, or no
#: usable summaries). F4 cards are built around a shock the calm history does not show, so a
#: text-less F4 card should still not draw history's too-narrow width. Measured 2026-09-26 on the
#: 29 realized F4 cards (3 draw seeds): random walk 0.4560 -> 0.4085 with 1.5 applied to every
#: card. When stage 2 DOES answer, the F4 v4 prompt already asks for enough width (69% of its
#: answers are 2.0+), and an always-on floor added nothing (0.3296 without vs 0.3299 with), so it
#: no longer overrides the model. See `text_signal._FAMILY_FOCUS["F4"]` for the prompt.
_FAMILY_WIDEN_FLOOR = {"T2-F4": 1.5}


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
#: really for. M2 (monthly cards) ignores skew by design -- its residual pool already carries the
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
    name, model = _select_model(family, target_type, horizons, monthly=_is_monthly(request))
    try:
        out = model(request)
    except Exception as exc:
        # A card that raises scores the pre-committed worst case (4.0); the walk scores ~1.0. So
        # an M2 failure falls back to the walk for the card's shape rather than propagating -- M2
        # refuses, for instance, cards whose assets span two panel files.
        if name != M2:
            raise
        name, model = _walk_for(horizons)
        print(f"[{M2}] {type(exc).__name__}: {exc}; falling back to the {name}", file=sys.stderr)
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


def _select_model(family: str | None, target_type: str | None, horizons: list[int],
                  *, monthly: bool = False) -> tuple[str, "_Model"]:
    """THE SWITCH. Which model runs this card, and why.

        monthly panel       -> M2                ridge location-scale + residual bootstrap. Any
        (level/log_return)                       family: the rule is about the DATA, since hidden
                                                 cards may carry monthly panels in any family.
                                                 Measured on the 4 visible monthly cards against the
                                                 tuned walk: 0.660x on validation, 0.481x on test
                                                 (model_baseline notebook 06), better on all four.
        more than 1 horizon -> cumulative walk   one accumulating path per draw, so the horizons
                                                 of an asset are correlated at sqrt(h_j/h_k)
                                                 instead of independent.
        everything else    -> random walk        one draw per horizon.

    Daily F1 cards moved from M2 to the walk on 2026-09-27: on real outcomes the walk scored better
    (team eval, F1 raw mean 0.2354 vs 0.2790; M2 better on 5 of 15 daily cards), and in the
    model_baseline backtest M2 only tied the walk on daily data.

    ON THE WALK BRANCHES BEING SHAPE, NOT FAMILY. A value at horizon h_k IS reached by walking
    through h_j, whatever family the card belongs to, so the cumulative walk is chosen by the number
    of horizons. The organizers' own example cards in docs/CATEGORIES.md are multi-horizon for F2
    and F4 too, even though no shipped F2/F4 unit is. The FAMILY only picks the walk's settings
    (`WALK_SETTINGS`).
    """
    if monthly and target_type in _M2_TARGETS:
        return M2, _m2_model
    return _walk_for(horizons)


def _walk_for(horizons: list[int]) -> tuple[str, "_Model"]:
    """The walk a card gets on its shape alone: cumulative for multi-horizon, else random."""
    if len(horizons) > 1:
        return CUMULATIVE_WALK, _cumulative_walk_model
    return RANDOM_WALK, _random_walk_model


#: Target types M2 fits. A log_return card is the same model with anchor 0: M2 predicts the
#: cumulative log return sum(log(1+r)) over the horizon directly (f1_pipeline notebook 03, section 4.4).
_M2_TARGETS = {"level", "log_return"}


def _is_monthly(r: _Request) -> bool:
    """True when the card's first asset sits on a monthly panel (rows more than 20 days apart)."""
    s = _series(r.panels, r.assets[0], r.asof)
    when = pd.to_datetime(pd.Series(s.index), errors="coerce").dropna()
    return len(when) >= 3 and float(when.diff().dt.days.median()) > _MONTHLY_SPACING_DAYS


# ---------------------------------------------------------------- model 1: M2 (monthly cards)
def _m2_model(r: _Request) -> np.ndarray:
    """Ridge centre, ridge log-variance width, and a joint bootstrap of standardised residuals
    sharing one historical date per draw (f1_pipeline/m2_unit.py; notebooks 01-04).

    `skew` is deliberately not applied: the residual pool already carries the empirical shape.
    """
    from f1_pipeline import m2_unit as m2

    unit = m2.unit_from_panels(r.panels, r.assets, r.horizons, r.asof, target_type=r.target_type)
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
    steps: np.ndarray           # (n_assets, n_horizons): each horizon in the asset's own panel steps
    nu: float | None = None     # Student-t degrees of freedom of the shocks (None = normal)

    def shock(self, rng: np.random.Generator, n_draws: int,
              rng_t: np.random.Generator | None = None) -> np.ndarray:
        """One correlated, optionally skew-tilted, optionally Student-t standard innovation per draw.

        Both walks draw their legs through here, so the cross-asset structure, the skew tilt and the
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
    """Panels -> the scale, correlation and centre both walks draw from.

    Every correctness fix lives here rather than in either model, because none of them is about
    F3: the gap guard's worst case is an F2 transfer card (measured 1.40x inflated sd on
    t2-F2-fragile-five-brl-2013) and the log-return centring carries 11 F4 units.
    """
    cfg = WALK_SETTINGS.get(r.family, _PLAIN_WALK)
    hist = {a: _series(r.panels, a, r.asof) for a in r.assets}
    hl = cfg["halflife"]
    steps = _step_frame(hist, r.assets, r.returns_target,
                        rows=max(cfg["window"], int(8 * hl) if hl else 0))
    D = steps.iloc[-cfg["window"]:].to_numpy().T  # (n_assets, window): sd and correlation
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

    # The family's widen times the text's. If the text half said nothing, the F4 fallback floor
    # applies to that total -- it is a floor on the width, not a second multiplier.
    widen = cfg["widen"] * per_asset("widen", 1.0)
    if _text_was_silent(r.adjustments, r.assets):
        widen = np.maximum(widen, _FAMILY_WIDEN_FLOOR.get(r.family or "", 1.0))

    return _WalkFit(
        centre=last + per_asset("shift", 0.0),
        scale=sd * widen,
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
    """Multi-horizon daily cards. ONE accumulating path per draw: horizon h_k is reached by walking there through h_1.

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
        out[:, :, hi] = fit.centre + path        # write to the ORIGINAL index
    return out


def _random_walk_model(r: _Request) -> np.ndarray:
    """Single-horizon daily cards. Each horizon drawn independently from the anchor.

    Assets are still correlated within a horizon (that is `fit.shock`); horizons are not
    correlated with each other. Every card routed here is single-horizon, where that distinction
    does not exist -- this and the cumulative walk produce the same distribution, from the same
    number of draws off the same generator. It is the honest model for a single-horizon card and
    keeps the reader from having to reason about an accumulation that never happens.
    """
    fit = _fit_walk(r)
    rng, rng_t = np.random.default_rng(r.seed), np.random.default_rng([r.seed, 7])
    out = np.empty((r.n_draws, len(r.assets), len(r.horizons)), dtype=float)
    for hi in range(len(r.horizons)):
        out[:, :, hi] = fit.centre + fit.shock(rng, r.n_draws, rng_t) * fit.scale * np.sqrt(fit.steps[:, hi])
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



#: Which model produced the most recent `build_draws()` call. `forecast_agent.main()` reads it
#: through `last_model()` for the rationale file; the models themselves never read it.
_last_model = RANDOM_WALK


def last_model() -> str:
    """The model name the most recent `build_draws()` call actually used.

    Not necessarily the one `_select_model` picked: an M2 failure falls back to the walk, and the
    rationale should say what ran.
    """
    return _last_model

