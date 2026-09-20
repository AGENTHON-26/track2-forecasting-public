"""## Executive summary (read this first)
Synthetic end-to-end checks for monthly CLI sampling and reasoning fallback.
They test lag-aware variance, shared path increments and unchanged submission keys.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from baselines import reasoning_agent
from qfbench2_track_forecasting import cli


@pytest.fixture
def monthly_unit(tmp_path):
    unit = tmp_path / "unit"
    unit.mkdir()
    values = 100 + np.cumsum(np.tile([1.0, 0.0, -1.0, 0.0], 12))
    panel = pd.DataFrame(
        {
            "asset": "INDEX",
            "date": pd.date_range("2027-01-01", periods=48, freq="MS"),
            "value": values,
        }
    )
    panel.to_parquet(unit / "monthly.parquet", index=False)
    (unit / "card.toml").write_text(
        '[task]\nid = "t2-synthetic-monthly"\n[targets]\nasset_ids = ["INDEX"]\n'
        'horizons = [21, 42]\ntarget_type = "level"\ntarget_frequency = "monthly"\n'
    )
    (unit / "forecast_spec.json").write_text(
        json.dumps(
            {"targets": {"horizons": [21, 42], "observation_periods": ["2031-03", "2031-04"]}}
        )
    )
    text = unit / "text"
    text.mkdir()
    (text / "synthetic.txt").write_text("Synthetic evidence supplied before the forecast cutoff.")
    (text / "corpus_index.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "synthetic",
                        "timestamp": "2031-02-01",
                        "file": "synthetic.txt",
                        "doc_type": "test",
                    }
                ]
            }
        )
    )
    return unit, panel


def arguments(unit, out, draws=10000):
    return [
        "--panels",
        str(unit),
        "--text",
        str(unit / "text"),
        "--asof",
        "2031-02-14",
        "--card",
        str(unit / "card.toml"),
        "--out",
        str(out),
        "--n-draws",
        str(draws),
        "--seed",
        "17",
    ]


def test_cli_retains_keys_and_uses_monthly_spread_from_lagged_anchor(monthly_unit, tmp_path):
    unit, panel = monthly_unit
    out = tmp_path / "out/forecast.parquet"
    assert cli.main(arguments(unit, out)) == 0
    forecast = pd.read_parquet(out)
    assert list(forecast["horizon"].unique()) == [21, 42]
    sd = panel["value"].diff().std()
    np.testing.assert_allclose(
        forecast.groupby("horizon")["value"].std(), sd * np.sqrt([3, 4]), rtol=0.025
    )
    # Counting 21 monthly innovations would overstate even the nearer target's sd by >2x.
    assert forecast.loc[forecast.horizon == 21, "value"].std() < sd * np.sqrt(21) / 2
    rationale = (out.parent / "forecast_rationale.md").read_text().lower()
    assert "monthly steps" in rationale and "publication lag" in rationale
    assert "daily" not in rationale and "business days" not in rationale


def test_same_month_start_end_labels_produce_identical_parquet_bytes(monthly_unit, tmp_path):
    unit, _ = monthly_unit
    results = []
    for name, dates in [
        ("start", ["2031-03-01", "2031-04-01"]),
        ("end", ["2031-03-31", "2031-04-30"]),
    ]:
        (unit / "forecast_spec.json").write_text(
            json.dumps({"targets": {"horizons": [21, 42], "target_dates": dates}})
        )
        out = tmp_path / name / "forecast.parquet"
        cli.main(arguments(unit, out, 500))
        results.append(out.read_bytes())
    assert results[0] == results[1]


@pytest.mark.parametrize("entrypoint", [cli.main, reasoning_agent.main])
def test_missing_mapping_refuses_before_output(monthly_unit, tmp_path, entrypoint):
    unit, _ = monthly_unit
    (unit / "forecast_spec.json").write_text("{}")
    out = tmp_path / "out/forecast.parquet"
    with pytest.raises(SystemExit, match="explicit observation-period mapping"):
        entrypoint(arguments(unit, out))
    assert not out.exists()


def test_cumulative_paths_have_the_expected_covariance_and_ignore_key_order(monthly_unit):
    _, panel = monthly_unit
    panels = {"monthly": panel}
    samples, meta = cli._draw(
        panels, ["INDEX"], [42, 21], "2031-02-14", 40000, 17, panel_steps=np.array([[4, 3]])
    )
    reverse, _ = cli._draw(
        panels, ["INDEX"], [21, 42], "2031-02-14", 40000, 17, panel_steps=np.array([[3, 4]])
    )
    np.testing.assert_array_equal(samples[:, :, ::-1], reverse)
    variance = panel["value"].diff().var()
    cov = np.cov(samples[:, 0, :].T)
    np.testing.assert_allclose(cov, variance * np.array([[4, 3], [3, 3]]), rtol=0.025)
    assert "daily_sd" not in meta


def test_two_keys_for_same_period_reuse_the_exact_same_path_value(monthly_unit):
    _, panel = monthly_unit
    samples, _ = cli._draw(
        {"monthly": panel},
        ["INDEX"],
        [21, 42],
        "2031-02-14",
        500,
        17,
        panel_steps=np.array([[3, 3]]),
    )
    np.testing.assert_array_equal(samples[:, :, 0], samples[:, :, 1])


def test_joint_paths_align_calendar_months_when_assets_have_different_anchors(monthly_unit):
    _, first = monthly_unit
    second = pd.concat(
        [
            first.assign(asset="OTHER"),
            pd.DataFrame(
                {"asset": ["OTHER"], "date": [pd.Timestamp("2031-01-01")], "value": [101.0]}
            ),
        ]
    )
    samples, _ = cli._draw(
        {"monthly": pd.concat([first, second])},
        ["INDEX", "OTHER"],
        [21],
        "2031-02-14",
        40000,
        17,
        panel_steps=np.array([[3], [2]]),
    )
    variance = first["value"].diff().var()
    np.testing.assert_allclose(
        np.cov(samples[:, :, 0].T), variance * np.array([[3, 2], [2, 2]]), rtol=0.025
    )


def test_monthly_differences_exclude_missing_months_and_cutoff_still_applies(monthly_unit):
    _, panel = monthly_unit
    hole = panel.drop(index=20)
    future = pd.concat(
        [
            hole,
            pd.DataFrame(
                {"asset": ["INDEX"], "date": [pd.Timestamp("2031-03-01")], "value": [999.0]}
            ),
        ]
    )
    a, stats = cli._draw(
        {"monthly": hole}, ["INDEX"], [21], "2031-02-14", 500, 17, panel_steps=np.array([[3]])
    )
    b, _ = cli._draw(
        {"monthly": future}, ["INDEX"], [21], "2031-02-14", 500, 17, panel_steps=np.array([[3]])
    )
    np.testing.assert_array_equal(a, b)
    assert stats["n_history_rows"] == 45


@pytest.mark.parametrize("model_succeeds", [False, True])
def test_reasoning_uses_monthly_sd_in_prompt_clamp_and_fallback(
    monthly_unit, tmp_path, monkeypatch, model_succeeds
):
    unit, panel = monthly_unit
    prompts = []

    def reply(prompt):
        prompts.append(prompt)
        return (
            ({"assets": {"INDEX": {"drift_bp": 1e9, "vol_scale": 1}}}, "", "")
            if model_succeeds
            else (None, "synthetic unavailable model", "")
        )

    monkeypatch.setattr(reasoning_agent, "call_model", reply)
    out = tmp_path / "reasoning/forecast.parquet"
    assert reasoning_agent.main(arguments(unit, out, 500)) == 0
    expected, meta = cli._draw(
        {"monthly": panel},
        ["INDEX"],
        [21, 42],
        "2031-02-14",
        500,
        17,
        panel_steps=np.array([[3, 4]]),
    )
    sd_h = panel["value"].diff().std() * 2
    actual = pd.read_parquet(out)["value"].to_numpy().reshape(500, 1, 2)
    np.testing.assert_allclose(
        actual, expected + (3 * sd_h if model_succeeds else 0), rtol=0, atol=1e-12
    )
    assert f"horizon sd {sd_h:.6f}" in prompts[0]
    assert "monthly steps" in prompts[0]
    assert "business days" not in prompts[0] and "daily" not in prompts[0]
    rationale = (out.parent / "forecast_rationale.md").read_text().lower()
    assert "monthly sd" in rationale and "daily" not in rationale
    sidecar = json.loads((out.parent / "forecast_meta.json").read_text())
    assert sidecar["reasoning_applied"] is model_succeeds


@pytest.mark.parametrize("entrypoint", [cli.main, reasoning_agent.main])
@pytest.mark.parametrize("target_type", ["level", "log_return"])
@pytest.mark.parametrize("declaration", ["targets", "metadata"])
def test_stale_monthly_declaration_on_daily_series_keeps_daily_artifact_bytes(
    tmp_path, entrypoint, target_type, declaration
):
    unit = tmp_path / "daily-unit"
    unit.mkdir()
    dates = pd.bdate_range("2030-01-01", periods=80)
    asof = str(dates[-1].date())
    values = (
        100 + np.cumsum(np.tile([1.0, 0.0, -1.0, 0.0], 20))
        if target_type == "level"
        else np.tile([0.001, 0.005, -0.002, 0.004], 20)
    )
    pd.DataFrame({"asset": "INDEX", "date": dates, "value": values}).to_parquet(
        unit / "daily.parquet", index=False
    )
    artifacts = []
    for frequency in ["daily", "monthly"]:
        card_text = (
            '[task]\nid = "t2-synthetic-daily"\n[metadata]\n'
            f'target_frequency = "{frequency}"\n[targets]\nasset_ids = ["INDEX"]\n'
            f'horizons = [1, 21]\ntarget_type = "{target_type}"\n'
        )
        if declaration == "targets":
            card_text += f'target_frequency = "{frequency}"\n'
        (unit / "card.toml").write_text(card_text)
        out = tmp_path / frequency / "forecast.parquet"
        args = arguments(unit, out, 500)
        args[args.index("--asof") + 1] = asof
        if frequency == "monthly":
            with pytest.warns(UserWarning, match="using daily sampling"):
                entrypoint(args)
        else:
            entrypoint(args)
        artifacts.append({p.name: p.read_bytes() for p in out.parent.iterdir()})
    assert artifacts[0] == artifacts[1]


def test_explicit_monthly_periods_conflicting_with_daily_observations_refuse(
    monthly_unit, tmp_path
):
    unit, panel = monthly_unit
    panel["date"] = pd.bdate_range("2030-01-01", periods=len(panel))
    panel.to_parquet(unit / "monthly.parquet", index=False)
    out = tmp_path / "out/forecast.parquet"
    with pytest.raises(SystemExit, match="conflicts with daily target observations"):
        cli.main(arguments(unit, out, 500))
    assert not out.exists()


def test_duplicate_monthly_observations_are_not_reinterpreted_as_daily(monthly_unit, tmp_path):
    unit, panel = monthly_unit
    duplicate = panel.copy()
    duplicate["date"] += pd.Timedelta(days=1)
    pd.concat([panel, duplicate]).to_parquet(unit / "monthly.parquet", index=False)
    out = tmp_path / "out/forecast.parquet"
    with pytest.raises(SystemExit, match="unique monthly observations"):
        cli.main(arguments(unit, out, 500))
    assert not out.exists()
