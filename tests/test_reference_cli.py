"""Synthetic checks of the reference producer's target interpretation and public CLI."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_track_forecasting import cli


def _panels(values: np.ndarray) -> tuple[dict[str, pd.DataFrame], str]:
    dates = pd.bdate_range("2025-01-01", periods=len(values))
    frame = pd.DataFrame({"date": dates, "asset": "A", "value": values})
    return {"synthetic": frame}, dates[-1].strftime("%Y-%m-%d")


def test_level_forecasts_preserve_the_previous_fixed_seed_output() -> None:
    panels, asof = _panels(100.0 + np.cumsum(np.tile([1.0, 0.0, -1.0, 0.0], 20)))
    samples, stats = cli._draw(panels, ["A"], [1, 21], asof, 500, 17)
    # Recorded by executing the released producer before the return-target correction.
    previous = [
        [[100.77858376552923, 103.060180948387]],
        [[100.23926821159226, 103.49856301908079]],
        [[99.61824444819105, 95.15970482426789]],
        [[99.10901906510834, 93.45386754186264]],
    ]
    np.testing.assert_allclose(samples[:4], previous, rtol=0, atol=1e-12)
    assert stats["last"] == {"A": 100.0}
    assert stats["n_history_rows"] == 79


def test_cumulative_returns_use_the_return_distribution_not_its_order() -> None:
    values = np.tile([0.001, 0.005, -0.002, 0.004], 20)
    panels, asof = _panels(values)
    # The same return observations in reverse order have the same step distribution, while
    # the most recent return and the first-difference history both change.
    reversed_panels, _ = _panels(values[::-1])
    samples, stats = cli._draw(panels, ["A"], [1, 21], asof, 500, 17, target_type="log_return")
    reversed_samples, _ = cli._draw(
        reversed_panels, ["A"], [1, 21], asof, 500, 17, target_type="log_return"
    )
    np.testing.assert_allclose(samples, reversed_samples, rtol=0, atol=1e-15)
    assert stats["last"] == {"A": 0.0}
    assert stats["n_history_rows"] == len(values)


def test_return_drift_and_spread_scale_with_the_horizon() -> None:
    values = np.tile([0.001, 0.005, -0.002, 0.004], 20)
    panels, asof = _panels(values)
    samples, _ = cli._draw(panels, ["A"], [1, 21], asof, 50000, 17, target_type="log_return")
    np.testing.assert_allclose(samples.mean(axis=0)[0], [0.002, 0.042], rtol=0, atol=0.0003)
    # For independent daily steps the standard deviation grows with sqrt(h), not h.
    sd = values.std(ddof=1)
    np.testing.assert_allclose(samples.std(axis=0)[0], sd * np.sqrt([1, 21]), rtol=0.02)


def test_return_forecast_keeps_cross_asset_dependence_and_asof_cutoff() -> None:
    values = np.tile([0.001, 0.005, -0.002, 0.004], 20)
    panels, asof = _panels(values)
    first = panels["synthetic"]
    second = first.assign(asset="B", value=-first["value"])
    clean = {"synthetic": pd.concat([first, second], ignore_index=True)}
    future_date = (dt.date.fromisoformat(asof) + dt.timedelta(days=1)).isoformat()
    future = pd.DataFrame({"date": [future_date], "asset": ["A"], "value": [999.0]})
    contaminated = {"synthetic": pd.concat([clean["synthetic"], future], ignore_index=True)}
    samples, _ = cli._draw(clean, ["A", "B"], [21], asof, 500, 17, target_type="log_return")
    cutoff_samples, _ = cli._draw(
        contaminated, ["A", "B"], [21], asof, 500, 17, target_type="log_return"
    )
    np.testing.assert_array_equal(samples, cutoff_samples)
    assert np.corrcoef(samples[:, :, 0].T)[0, 1] < -0.99


def test_cli_uses_the_card_target_type_for_the_written_forecast(tmp_path: Path) -> None:
    panels, asof = _panels(np.tile([0.001, 0.005, -0.002, 0.004], 20))
    unit = tmp_path / "unit"
    unit.mkdir()
    panels["synthetic"].to_parquet(unit / "synthetic.parquet", index=False)
    (unit / "card.toml").write_text(
        '[task]\nid = "t2-synthetic-return"\n'
        '[targets]\nasset_ids = ["A"]\nhorizons = [21]\ntarget_type = "log_return"\n'
    )
    output = tmp_path / "output" / "forecast.parquet"
    assert (
        cli.main(
            [
                "--panels",
                str(unit),
                "--text",
                str(unit / "text"),
                "--asof",
                asof,
                "--out",
                str(output),
                "--n-draws",
                "10000",
                "--seed",
                "17",
            ]
        )
        == 0
    )
    draws = pd.read_parquet(output)
    assert abs(draws["value"].mean() - 0.042) < 0.0005
    metadata = json.loads((output.parent / "forecast_meta.json").read_text())
    assert metadata["target"] == "log_return"
    assert metadata["n_draws"] == len(draws)
