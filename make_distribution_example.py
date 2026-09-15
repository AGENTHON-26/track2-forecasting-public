"""
Track 2 — a tiny worked example: how to turn history into a distribution of draws.

Single asset, one horizon. Reads a real panel from the exemplar unit, then builds a
forecast the same way the reference agent does:

    draw = last_value + (random bell-curve number) * (daily_vol * sqrt(horizon))

Run it:
    python make_distribution_example.py

Only needs numpy + pyarrow (already used by the repo). No network, no model.
"""

import numpy as np
import pyarrow.parquet as pq

# ---- settings you can change ------------------------------------------------
PANEL   = "units/t2-EXAMPLE-ust-curve-1m/rates_daily.parquet"
ASSET   = "UST_10Y"      # which series to forecast
ASOF    = "2024-06-28"   # the cutoff: we may only look at data on/before this
HORIZON = 126            # forecast this many business days ahead (~6 months)
N_DRAWS = 500            # how many possible futures to generate
SEED    = 0              # fixes the randomness so results are reproducible

# ---- an optional "LLM" adjustment (pretend the text told us something) ------
# Set these to simulate the reasoning step. 0 / 1.0 = no change (pure baseline).
SHIFT      = 0.0    # move the center up (+) or down (-), in % points
WIDEN      = 1.0    # multiply the spread: >1 = more uncertain, <1 = more confident
# -----------------------------------------------------------------------------


def load_series(path, asset, asof):
    """Return the asset's values (as a numpy array), in date order, up to as-of."""
    t = pq.read_table(path).to_pydict()
    rows = [
        (str(d)[:10], float(v))
        for d, a, v in zip(t["date"], t["asset"], t["value"])
        if a == asset and str(d)[:10] <= asof
    ]
    rows.sort()                      # sort by date
    return np.array([v for _, v in rows], dtype=float)


def make_draws(history, horizon, n_draws, seed, shift=0.0, widen=1.0):
    """The core recipe. Returns n_draws possible future values."""
    # Step 1 — anchor: the last observed value
    last = history[-1]

    # Step 2 — how much it wiggles day to day (volatility)
    daily_changes = np.diff(history)          # today - yesterday, for every day
    daily_vol = daily_changes.std()

    # Step 3 — scale the wiggle to the forecast horizon (grows with sqrt of time)
    spread = daily_vol * np.sqrt(horizon)

    # ---- the LLM's optional reshaping of the cloud ----
    center = last + shift          # text can move the center
    spread = spread * widen        # text can widen / tighten the cloud

    # Step 4 — draw n random numbers from a bell curve around the center
    rng = np.random.default_rng(seed)
    draws = center + rng.standard_normal(n_draws) * spread

    return draws, last, daily_vol, spread, center


def ascii_histogram(draws, bins=21, width=48):
    """A rough text picture of the distribution."""
    counts, edges = np.histogram(draws, bins=bins)
    top = counts.max()
    lines = []
    for c, lo in zip(counts, edges[:-1]):
        bar = "#" * int(round(width * c / top)) if top else ""
        lines.append(f"  {lo:6.2f} | {bar}")
    return "\n".join(lines)


def main():
    history = load_series(PANEL, ASSET, ASOF)
    draws, last, vol, spread, center = make_draws(
        history, HORIZON, N_DRAWS, SEED, shift=SHIFT, widen=WIDEN
    )

    p = np.percentile(draws, [1, 5, 50, 95, 99])

    print(f"Asset {ASSET}  as-of {ASOF}  horizon {HORIZON} business days\n")
    print(f"  history rows used     : {len(history)}")
    print(f"  anchor (last value)   : {last:.3f} %")
    print(f"  daily volatility      : {vol:.4f}  (std of daily changes)")
    print(f"  spread at horizon     : {spread:.3f}  (= vol x sqrt({HORIZON}))")
    if SHIFT or WIDEN != 1.0:
        print(f"  LLM adjustment        : shift {SHIFT:+.2f}, widen x{WIDEN}")
        print(f"  adjusted center       : {center:.3f} %")
    print(f"\n  {N_DRAWS} draws — the distribution:")
    print(f"    1st  pct : {p[0]:.2f}   (a bad-case low)")
    print(f"    5th  pct : {p[1]:.2f}")
    print(f"    median   : {p[2]:.2f}   (middle guess)")
    print(f"    95th pct : {p[3]:.2f}")
    print(f"    99th pct : {p[4]:.2f}   (a bad-case high)")
    print(f"    range    : {draws.min():.2f} to {draws.max():.2f}\n")
    print("  shape (each row is a value bucket; bars = how many draws landed there):")
    print(ascii_histogram(draws))
    print("\n  -> These 500 numbers ARE your forecast for this asset.")
    print("     Reality will be ONE value; CRPS scores how well this cloud covered it.")


if __name__ == "__main__":
    main()
