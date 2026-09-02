"""T2-7: a rankable unit without a complete positive finite scale is an ORGANIZER failure.

Three faults, all armed and all latent because today every unit that has a scale is exactly a
unit that has an answer:

* a partial scale (`{"tail": 1.0}`) raised an **uncaught `KeyError: 'marginal'`** out of the
  scorer, because `crps_composite` indexes `ref_scale["marginal"]` unconditionally whenever the
  dict is truthy;
* `ctx.setdefault("ref_scale", None)` made a missing scale file a silent fall back to raw
  components, and the driver then averaged raw and normalized composites together.
* the joint component does not exist on a 1-cell variogram grid, so the correct scale
  (`joint: 0.0`) was refused as non-positive -- 60 of 104 public cards. Latent only because the
  generator writes a placeholder `1.0`.

They go live the moment the backfill runs, because `build_ref_scales.py` is a separate manual step
after `backfill_realized.py` with nothing enforcing the pairing.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from conftest import build_submission, build_unit
from qfbench2_common.contracts import OrganizerFault

from qfbench2_track_forecasting.normalization import (
    REF_SCALE_COMPONENTS,
    NormalizationMode,
    RefScale,
    load_ref_scale,
)


def _write_scale(reference: pathlib.Path, payload: object) -> None:
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "ref_scale.json").write_text(json.dumps(payload), encoding="utf-8")


def test_positive_control_complete_scale_loads(tmp_path: pathlib.Path) -> None:
    reference = tmp_path / "reference"
    _write_scale(reference, {"marginal": 0.5, "joint": 2.0, "tail": 0.25})
    scale = load_ref_scale(reference)
    assert scale.as_mapping(joint_weight=0.3) == {"marginal": 0.5, "joint": 2.0, "tail": 0.25}


def test_missing_scale_is_an_organizer_fault_not_a_raw_fallback(tmp_path: pathlib.Path) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    with pytest.raises(OrganizerFault) as exc:
        load_ref_scale(reference)
    assert "fallback" in str(exc.value)


@pytest.mark.parametrize("present", ["marginal", "joint", "tail"])
def test_partial_scale_is_refused_rather_than_raising_a_keyerror(
    tmp_path: pathlib.Path, present: str
) -> None:
    reference = tmp_path / "reference"
    _write_scale(reference, {present: 1.0})
    with pytest.raises(OrganizerFault) as exc:
        load_ref_scale(reference)
    assert "missing" in str(exc.value)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_scale_is_refused(tmp_path: pathlib.Path, bad: float) -> None:
    reference = tmp_path / "reference"
    _write_scale(reference, {"marginal": bad, "joint": 1.0, "tail": 1.0})
    with pytest.raises(OrganizerFault):
        load_ref_scale(reference)


def test_unknown_key_in_the_scale_is_refused(tmp_path: pathlib.Path) -> None:
    reference = tmp_path / "reference"
    _write_scale(reference, {"marginal": 1.0, "joint": 1.0, "tail": 1.0, "energy": 1.0})
    with pytest.raises(OrganizerFault):
        load_ref_scale(reference)


def test_positive_control_the_generator_shape_loads(tmp_path: pathlib.Path) -> None:
    """`build_ref_scales.py` writes `method`, `seed` and `generated` beside the three components.

    Every scale file the generator has written carries them. A loader that refused them would reject
    every legitimate unit, and a gate that rejects the legitimate case makes every rejection
    beside it uninterpretable.
    """
    reference = tmp_path / "reference"
    _write_scale(
        reference,
        {
            "marginal": 0.0533,
            "joint": 1.0,
            "tail": 0.12,
            "method": "m0_text_blind_grw",
            "seed": 2140827255,
            "generated": "2026-08-01T00:00:00Z",
        },
    )
    scale = load_ref_scale(reference)
    assert scale.as_mapping(joint_weight=0.3) == {"marginal": 0.0533, "joint": 1.0, "tail": 0.12}


def test_non_numeric_scale_value_is_refused(tmp_path: pathlib.Path) -> None:
    reference = tmp_path / "reference"
    _write_scale(reference, {"marginal": "1.0", "joint": 1.0, "tail": 1.0})
    with pytest.raises(OrganizerFault):
        load_ref_scale(reference)


def test_unparseable_scale_is_an_organizer_fault_not_a_participant_failure(
    tmp_path: pathlib.Path,
) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "ref_scale.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(OrganizerFault):
        load_ref_scale(reference)


def test_the_scale_is_never_loaded_from_the_participant_tree(tmp_path: pathlib.Path) -> None:
    """`ref_scale.json` is answer-equivalent (C6). A participant-reachable copy is refused."""
    from qfbench2_track_forecasting.normalization import assert_reference_only

    reference = tmp_path / "reference"
    reference.mkdir()
    outside = tmp_path / "res" / "u-abcd1234" / "ref_scale.json"
    outside.parent.mkdir(parents=True)
    with pytest.raises(OrganizerFault):
        assert_reference_only(outside, reference)


def test_scoring_refuses_ref_scale_mode_with_no_reference_root(tmp_path: pathlib.Path) -> None:
    from qfbench2_track_forecasting.scoring import hydrate_ctx

    unit = build_unit(tmp_path / "unit")
    out = build_submission(tmp_path / "out")
    import tomllib

    ctx = {
        "card": tomllib.loads((unit / "card.toml").read_text(encoding="utf-8")),
        "output_dir": out,
        "reference_root": None,
        "normalization_mode": NormalizationMode.REF_SCALE,
    }
    with pytest.raises(OrganizerFault):
        hydrate_ctx(ctx)


def test_hydrate_does_not_default_the_scale_to_none_under_ref_scale_mode(
    tmp_path: pathlib.Path,
) -> None:
    """The exact line removed: `ctx.setdefault("ref_scale", None)` after the file lookup."""
    from qfbench2_track_forecasting.scoring import hydrate_ctx

    unit = build_unit(tmp_path / "unit")
    (unit / "reference" / "ref_scale.json").unlink()
    out = build_submission(tmp_path / "out")
    import tomllib

    ctx = {
        "card": tomllib.loads((unit / "card.toml").read_text(encoding="utf-8")),
        "unit_dir": unit,
        "output_dir": out,
        "normalization_mode": NormalizationMode.REF_SCALE,
    }
    with pytest.raises(OrganizerFault):
        hydrate_ctx(ctx)


def test_smoke_path_is_named_unrankable_rather_than_silently_raw(tmp_path: pathlib.Path) -> None:
    from qfbench2_track_forecasting.scoring import hydrate_ctx

    unit = build_unit(tmp_path / "unit", with_reference=False)
    out = build_submission(tmp_path / "out")
    ctx = {"unit_dir": unit, "output_dir": out}
    hydrate_ctx(ctx)
    assert ctx["normalization_mode"] is NormalizationMode.RAW_UNRANKABLE
    assert ctx["grid_source"] == "card"
    assert ctx["ref_scale"] is None


def test_refscale_construction_is_the_validation() -> None:
    assert set(REF_SCALE_COMPONENTS) == {"marginal", "joint", "tail"}
    with pytest.raises(OrganizerFault):
        RefScale(marginal=1.0, joint=0.0, tail=1.0)


# --------------------------------------------------------------------------- #
# The joint component does not exist on a 1-cell variogram grid                #
# --------------------------------------------------------------------------- #
# See normalization._OPTIONAL_COMPONENT for why, and for the blast radius.


def test_single_cell_zero_joint_is_the_correct_value_not_a_fault(tmp_path: pathlib.Path) -> None:
    """The regression. Before `cell_count`, this raised `ref_scale.joint=0.0 is not positive`."""
    reference = tmp_path / "reference"
    _write_scale(reference, {"marginal": 0.5, "joint": 0.0, "tail": 0.12})
    scale = load_ref_scale(reference, cell_count=1)
    assert scale.joint is None
    assert scale.as_mapping(joint_weight=0.0) == {"marginal": 0.5, "joint": 1.0, "tail": 0.12}


@pytest.mark.parametrize(
    "payload",
    [
        {"marginal": 0.5, "tail": 0.12},  # key absent
        {"marginal": 0.5, "joint": None, "tail": 0.12},
        {"marginal": 0.5, "joint": 0.0, "tail": 0.12},
        {"marginal": 0.5, "joint": -0.0, "tail": 0.12},
        {"marginal": 0.5, "joint": 0, "tail": 0.12},  # JSON int
        {"marginal": 0.5, "joint": 1.0, "tail": 0.12},  # what the generator writes today
        {"marginal": 0.5, "joint": 2.5, "tail": 0.12},
    ],
)
def test_every_numeric_joint_is_dropped_on_a_single_cell_grid(
    tmp_path: pathlib.Path, payload: dict[str, object]
) -> None:
    """Dropped whatever the file says. Carrying a positive one through would make `joint is None`
    mean "the file was honest" rather than "the grid has no joint", and `as_mapping`'s guard would
    then cover only the honest minority."""
    reference = tmp_path / "reference"
    _write_scale(reference, payload)
    assert load_ref_scale(reference, cell_count=1).joint is None


@pytest.mark.parametrize("joint", [-1.0, float("inf"), float("nan"), "1.0", True, [1.0]])
@pytest.mark.parametrize("cell_count", [None, 1, 2])
def test_a_joint_that_is_not_a_scale_is_refused_on_every_grid_shape(
    tmp_path: pathlib.Path, cell_count: int | None, joint: object
) -> None:
    """The relaxation widens which VALUES are acceptable, never the type. `true` matters most:
    `float(False) == 0.0`, so a JSON bool would read as "component absent" without the guard."""
    reference = tmp_path / "reference"
    _write_scale(reference, {"marginal": 0.5, "joint": joint, "tail": 0.12})
    with pytest.raises(OrganizerFault):
        load_ref_scale(reference, cell_count=cell_count)


@pytest.mark.parametrize("dropped", ["marginal", "tail"])
@pytest.mark.parametrize("how", ["absent", "null"])
def test_the_single_cell_relaxation_applies_to_joint_and_nothing_else(
    tmp_path: pathlib.Path, dropped: str, how: str
) -> None:
    """A 1-cell grid still HAS a marginal and a tail. Dropping one is the partial-scale KeyError,
    not a grid property -- and `_number` would raise a bare KeyError rather than OrganizerFault."""
    reference = tmp_path / "reference"
    payload: dict[str, object] = {"marginal": 0.5, "joint": 0.0, "tail": 0.12}
    if how == "null":
        payload[dropped] = None
    else:
        del payload[dropped]
    _write_scale(reference, payload)
    with pytest.raises(OrganizerFault, match=dropped):
        load_ref_scale(reference, cell_count=1)


def test_a_single_cell_energy_card_keeps_the_strict_rule(tmp_path: pathlib.Path) -> None:
    """The relaxation is a property of the STATISTIC, not the grid: `energy_score` on one cell
    equals the marginal CRPS, so its scale genuinely exists and `0.0` is still a defect."""
    reference = tmp_path / "reference"
    _write_scale(reference, {"marginal": 0.5, "joint": 0.0, "tail": 0.12})
    with pytest.raises(OrganizerFault, match="not positive"):
        load_ref_scale(reference, cell_count=1, joint_statistic="energy")


@pytest.mark.parametrize("joint", [0.0, -1.0])
def test_multi_cell_still_requires_a_positive_joint(tmp_path: pathlib.Path, joint: float) -> None:
    reference = tmp_path / "reference"
    _write_scale(reference, {"marginal": 0.5, "joint": joint, "tail": 0.12})
    with pytest.raises(OrganizerFault):
        load_ref_scale(reference, cell_count=2)


@pytest.mark.parametrize("cell_count", [None, 2])
def test_an_absent_joint_is_refused_off_the_single_cell_path(
    tmp_path: pathlib.Path, cell_count: int | None
) -> None:
    """`None` means the caller did not say, and a caller that did not say gets the strict rule."""
    reference = tmp_path / "reference"
    _write_scale(reference, {"marginal": 0.5, "tail": 0.12})
    with pytest.raises(OrganizerFault):
        load_ref_scale(reference, cell_count=cell_count)


def test_an_absent_joint_cannot_be_handed_to_a_live_joint_weight() -> None:
    """The placeholder is reachable only where the composite multiplies it by zero."""
    scale = RefScale(marginal=0.5, joint=None, tail=0.12)
    with pytest.raises(OrganizerFault) as exc:
        scale.as_mapping(joint_weight=0.3)
    assert "0.3" in str(exc.value)


def test_a_none_joint_cannot_reach_a_live_weight_through_the_scorer(
    tmp_path: pathlib.Path,
) -> None:
    """The direct test above proves the guard exists; this proves it is WIRED. `official.py`
    preloads `ctx["ref_scale"]`, so `hydrate_ctx` never gets a second chance to fix a bad shape."""
    import tomllib

    from qfbench2_track_forecasting.scoring import build_verifier

    unit = build_unit(tmp_path / "unit")  # 2 assets x 2 horizons = 4 cells
    out = build_submission(tmp_path / "out")
    ctx = {
        "card": tomllib.loads((unit / "card.toml").read_text(encoding="utf-8")),
        "unit_dir": unit,
        "output_dir": out,
        "normalization_mode": NormalizationMode.REF_SCALE,
        "ref_scale": RefScale(marginal=1.0, joint=None, tail=1.0),
    }
    with pytest.raises(OrganizerFault, match="does not exist for this unit"):
        build_verifier(ctx).run(ctx)


_SINGLE_CELL_SCALES = [
    {"marginal": 0.5, "joint": 0.0, "tail": 0.12},  # the reported bug
    {"marginal": 0.5, "joint": None, "tail": 0.12},
    {"marginal": 0.5, "tail": 0.12},
    {"marginal": 0.5, "joint": 1.0, "tail": 0.12},  # what the generator writes today
]


def _score_single_cell(tmp_path: pathlib.Path, scale: dict[str, object]) -> object:
    import tomllib

    from qfbench2_track_forecasting.scoring import build_verifier

    unit = build_unit(tmp_path / "unit", assets=["SYN-A"], horizons=[1], ref_scale=scale)
    out = build_submission(tmp_path / "out", assets=["SYN-A"], horizons=[1])
    ctx = {
        "card": tomllib.loads((unit / "card.toml").read_text(encoding="utf-8")),
        "unit_dir": unit,
        "output_dir": out,
        "normalization_mode": NormalizationMode.REF_SCALE,
    }
    return build_verifier(ctx).run(ctx), ctx


@pytest.mark.parametrize("scale", _SINGLE_CELL_SCALES)
def test_a_single_cell_unit_scores_under_ref_scale_mode(
    tmp_path: pathlib.Path, scale: dict[str, object]
) -> None:
    """End to end: the shape that used to abort the evaluation now produces a composite."""
    verdict, ctx = _score_single_cell(tmp_path, scale)
    assert verdict.admissible, verdict.labels
    assert ctx["ref_scale"].joint is None
    assert verdict.detail["cell_count"] == 1
    assert verdict.detail["joint"] == 0.0
    weights = verdict.detail["weights_effective"]
    assert weights[1] == 0.0
    assert sum(weights) == pytest.approx(1.0)
    assert verdict.score == pytest.approx(
        weights[0] * verdict.detail["marginal"] / 0.5 + weights[2] * verdict.detail["tail"] / 0.12
    )


def test_the_on_disk_joint_encoding_cannot_move_a_single_cell_score(
    tmp_path: pathlib.Path,
) -> None:
    """The safety property, and the reason the generator can switch from `1.0` to `0.0` mid
    competition without moving anybody's score: on a 1-cell grid the joint slot is not a scale,
    so what the file put there must be immaterial."""
    scores = {
        i: _score_single_cell(tmp_path / str(i), scale)[0].score
        for i, scale in enumerate(_SINGLE_CELL_SCALES)
    }
    assert len(set(scores.values())) == 1, scores
