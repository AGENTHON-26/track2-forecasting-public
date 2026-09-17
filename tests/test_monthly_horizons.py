"""## Executive summary (read this first)
Synthetic checks for monthly target-period resolution and publication-lag handling.
The examples contain no task data. The output horizon keys never change.
"""

import copy

import numpy as np
import pytest

from qfbench2_track_forecasting.horizons import HorizonMetadataError, monthly_horizon_steps


def resolve(spec, *, card=None, assets=None, horizons=None, anchors=None, asof="2031-02-14"):
    return monthly_horizon_steps(
        assets or ["INDEX"],
        horizons or [42, 65],
        anchors or {"INDEX": "2030-12-01"},
        asof=asof,
        card=card,
        forecast_spec=spec,
    )


def test_existing_dates_preserve_keys_and_count_from_lagged_panel():
    spec = {
        "targets": {
            "asset_ids": ["INDEX"],
            "horizons": [42, 65],
            "target_dates": ["2031-03-31", "2031-04-01"],
        }
    }
    before = copy.deepcopy(spec)
    np.testing.assert_array_equal(resolve(spec), [[3, 4]])
    assert spec == before
    assert spec["targets"]["horizons"] == [42, 65]


def test_periods_and_start_or_end_labels_describe_same_observation_month():
    spec = {"targets": {"horizons": [42, 65], "observation_periods": ["2031-03", "2031-04"]}}
    for label in ("2031-03-01", "2031-03-31"):
        card = {"targets": {"horizons": [42, 65], "target_dates": [label, "2031-04-30"]}}
        np.testing.assert_array_equal(resolve(spec, card=card), [[3, 4]])


def test_per_asset_anchors_and_unsorted_output_order():
    spec = {
        "questions": [
            {"asset": a, "horizon": h, "observation_period": p}
            for a in ["A", "B"]
            for h, p in [(42, "2031-03"), (65, "2031-04")]
        ]
    }
    np.testing.assert_array_equal(
        resolve(
            spec,
            assets=["B", "A"],
            horizons=[65, 42],
            anchors={"A": "2030-12-01", "B": "2031-01-01"},
        ),
        [[3, 2], [4, 3]],
    )


def test_existing_question_dates_need_no_parallel_metadata():
    spec = {
        "questions": [
            {"asset": "INDEX", "horizon": 42, "target_date": "2031-03-01"},
            {"asset": "INDEX", "horizon": 65, "target_date": "2031-04-30"},
        ]
    }
    np.testing.assert_array_equal(resolve(spec), [[3, 4]])


def test_nowcast_period_can_precede_asof_but_must_follow_panel_anchor():
    spec = {"targets": {"horizons": [42], "observation_periods": ["2031-01"]}}
    np.testing.assert_array_equal(resolve(spec, horizons=[42]), [[1]])


@pytest.mark.parametrize("bad", ["2031-13", "2031-00", "31-01", "2031-1", None, 203101])
def test_invalid_observation_period_refuses(bad):
    with pytest.raises(HorizonMetadataError):
        resolve({"targets": {"horizons": [42, 65], "observation_periods": [bad, "2031-04"]}})


@pytest.mark.parametrize("bad", ["2031-02-30", "2031-03-01T00:00:00", "UNKNOWN", None])
def test_invalid_existing_date_refuses(bad):
    with pytest.raises(HorizonMetadataError):
        resolve({"targets": {"horizons": [42, 65], "target_dates": [bad, "2031-04-01"]}})


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"targets": {"horizons": [42, 65], "observation_periods": ["2031-03"]}},
        {"targets": {"horizons": [42, 42], "observation_periods": ["2031-03", "2031-04"]}},
        {"targets": {"horizons": [42, 66], "observation_periods": ["2031-03", "2031-04"]}},
        {"targets": {"horizons": [42, 65], "observation_periods": ["2030-12", "2031-04"]}},
        {"questions": [{"asset": "INDEX", "horizon": 42, "observation_period": "2031-03"}]},
        {"questions": [{"asset": "INDEX", "horizon": True, "observation_period": "2031-03"}]},
        {
            "questions": [
                {"asset": "INDEX", "horizon": 42, "observation_period": "2031-03"},
                {"asset": "INDEX", "horizon": 42, "observation_period": "2031-03"},
            ]
        },
    ],
)
def test_incomplete_malformed_or_ambiguous_grid_refuses(spec):
    with pytest.raises(HorizonMetadataError):
        resolve(spec)


def test_conflicting_date_and_period_refuse():
    spec = {"targets": {"horizons": [42, 65], "observation_periods": ["2031-03", "2031-04"]}}
    card = {"targets": {"horizons": [42, 65], "target_dates": ["2031-02-28", "2031-04-30"]}}
    with pytest.raises(HorizonMetadataError, match="Conflicting"):
        resolve(spec, card=card)


def test_post_cutoff_anchor_refuses():
    spec = {"targets": {"horizons": [42, 65], "observation_periods": ["2031-03", "2031-04"]}}
    with pytest.raises(HorizonMetadataError, match="cutoff"):
        resolve(spec, anchors={"INDEX": "2031-02-15"})
