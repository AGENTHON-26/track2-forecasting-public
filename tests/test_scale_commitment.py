"""## Executive summary (read this first)

The signed normalization commitment must name the exact scale bytes used for scoring.
These synthetic tests exercise byte tampering, roster/path substitution and the two scoring
entrypoints. A verified snapshot stays immutable when files change after verification.
No test fixture contains competition references or outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from conftest import make_plan
from qfbench2_common.contracts import EvaluationPlan, OrganizerFault, digest_json
from test_official import HANDLES, build_evaluation

from qfbench2_track_forecasting import normalization, official
from qfbench2_track_forecasting.normalization import (
    RefScale,
    RefScaleBundle,
    VerifiedRefScales,
    load_verified_ref_scales,
    read_ref_scale_bundle,
)
from qfbench2_track_forecasting.scoring import hydrate_ctx


def _fixture(root: pathlib.Path):
    ref = root / "ref"
    payloads = {}
    for index, handle in enumerate(HANDLES):
        directory = ref / handle / "reference"
        directory.mkdir(parents=True)
        (directory.parent / "card.toml").write_text('[scoring.params]\njoint = "variogram"\n')
        payloads[handle] = json.dumps(
            {
                "marginal": 1.0 + index,
                "joint": 1.0,
                "tail": 1.0,
                "method": "synthetic-m0-v1",
            }
        ).encode()
        (directory / "ref_scale.json").write_bytes(payloads[handle])
    # Independent reproduction of the shipping staging algorithm, not the helper under test.
    commitment = digest_json(
        {handle: "sha256:" + hashlib.sha256(raw).hexdigest() for handle, raw in payloads.items()}
    )
    plan = EvaluationPlan.from_mapping(make_plan(HANDLES, scale_commitment=commitment))
    return ref, plan, payloads


def test_staging_encoding_matches_existing_exact_file_algorithm(tmp_path):
    ref, plan, payloads = _fixture(tmp_path)
    staged = read_ref_scale_bundle(ref, HANDLES)
    verified = load_verified_ref_scales(plan, ref)
    assert staged.commitment == plan.normalization["ref_scale_commitment"]
    assert dict(verified.bundle.scales) == payloads
    assert read_ref_scale_bundle(ref, list(reversed(HANDLES))).commitment == staged.commitment
    for index, entry in enumerate(plan.expected_units):
        scale = verified.load_scale(
            plan, entry, ref / entry.unit_handle / "reference", cell_count=4
        )
        assert scale.marginal == 1.0 + index


@pytest.mark.parametrize("mutation", ["one_byte", "swapped", "wrong_m0", "missing", "unexpected"])
def test_changed_inputs_abort_without_numeric_or_path_leaks(tmp_path, mutation):
    ref, plan, payloads = _fixture(tmp_path)
    first = ref / HANDLES[0] / "reference" / "ref_scale.json"
    if mutation == "one_byte":
        first.write_bytes(payloads[HANDLES[0]] + b" ")
    elif mutation == "swapped":
        first.write_bytes(payloads[HANDLES[1]])
        (ref / HANDLES[1] / "reference" / "ref_scale.json").write_bytes(payloads[HANDLES[0]])
    elif mutation == "wrong_m0":
        first.write_bytes(payloads[HANDLES[0]].replace(b"synthetic-m0-v1", b"synthetic-m0-v2"))
    elif mutation == "missing":
        first.unlink()
    else:
        (ref / "u-99998888").mkdir()
    with pytest.raises(OrganizerFault) as exc:
        load_verified_ref_scales(plan, ref)
    reason = str(exc.value)
    assert str(tmp_path) not in reason
    assert all(handle not in reason for handle in HANDLES)
    assert "sha256:" not in reason
    assert "1.0" not in reason


@pytest.mark.parametrize("part", ["file", "reference", "unit", "hardlink", "fifo"])
def test_links_or_special_nodes_cannot_supply_scales(tmp_path, part):
    ref, plan, _ = _fixture(tmp_path)
    unit = ref / HANDLES[0]
    reference = unit / "reference"
    scale = reference / "ref_scale.json"
    outside = tmp_path / "outside"
    if part in {"file", "hardlink", "fifo"}:
        scale.rename(outside)
        if part == "file":
            scale.symlink_to(outside)
        elif part == "hardlink":
            os.link(outside, scale)
        else:
            os.mkfifo(scale)
    else:
        target = reference if part == "reference" else unit
        target.rename(outside)
        target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OrganizerFault):
        load_verified_ref_scales(plan, ref)


def test_snapshot_is_immutable_and_never_rereads_mutated_files(tmp_path):
    ref, plan, _ = _fixture(tmp_path)
    verified = load_verified_ref_scales(plan, ref)
    with pytest.raises(FrozenInstanceError):
        verified.plan_digest = "changed"
    with pytest.raises(TypeError):
        verified.bundle.scales[HANDLES[0]] = b"changed"
    shutil.rmtree(ref)
    for index, entry in enumerate(plan.expected_units):
        scale = verified.load_scale(
            plan, entry, ref / entry.unit_handle / "reference", cell_count=4
        )
        assert scale.marginal == 1.0 + index


def test_manually_constructed_snapshot_cannot_claim_a_different_commitment(tmp_path):
    ref, plan, payloads = _fixture(tmp_path)
    payloads[HANDLES[0]] = b'{"marginal": 987654.321, "joint": 1, "tail": 1}'
    bundle = RefScaleBundle(ref, payloads)
    payloads.clear()
    assert len(bundle.scales) == len(HANDLES)
    assert "987654.321" not in repr(bundle)
    forged = VerifiedRefScales(plan.plan_digest, bundle)
    with pytest.raises(OrganizerFault):
        forged.load_scale(
            plan, plan.expected_units[0], ref / HANDLES[0] / "reference", cell_count=4
        )


@pytest.mark.parametrize(
    "payload",
    [b"[", b"1", b"[" * 2000, b'{"marginal": ' + b"9" * 4000 + b', "joint": 1, "tail": 1}'],
)
def test_committed_malformed_or_unrepresentable_scale_is_an_organizer_fault(tmp_path, payload):
    ref, _, _ = _fixture(tmp_path)
    (ref / HANDLES[0] / "reference" / "ref_scale.json").write_bytes(payload)
    plan = EvaluationPlan.from_mapping(
        make_plan(HANDLES, scale_commitment=read_ref_scale_bundle(ref, HANDLES).commitment)
    )
    with pytest.raises(OrganizerFault):
        load_verified_ref_scales(plan, ref)


@pytest.mark.parametrize("kind", ["mapping", "namespace"])
def test_verified_constructor_refuses_mutable_lookalike_bundle(tmp_path, kind):
    ref, plan, payloads = _fixture(tmp_path)
    payloads[HANDLES[0]] = b'{"marginal": 987654.321, "joint": 1, "tail": 1}'
    fields = {
        "reference_root": ref.absolute(),
        "scales": payloads,
        "commitment": plan.normalization["ref_scale_commitment"],
    }
    fake = fields if kind == "mapping" else SimpleNamespace(**fields)
    with pytest.raises(OrganizerFault, match="canonical immutable bundle"):
        VerifiedRefScales(plan.plan_digest, fake)


def test_verified_constructor_rederives_nested_commitment(tmp_path):
    ref, plan, payloads = _fixture(tmp_path)
    payloads[HANDLES[0]] = b'{"marginal": 987654.321, "joint": 1, "tail": 1}'
    bundle = RefScaleBundle(ref, payloads)
    object.__setattr__(bundle, "commitment", plan.normalization["ref_scale_commitment"])
    forged = VerifiedRefScales(plan.plan_digest, bundle, dict.fromkeys(HANDLES, "variogram"))
    with pytest.raises(OrganizerFault):
        forged.load_scale(
            plan, plan.expected_units[0], ref / HANDLES[0] / "reference", cell_count=4
        )


@pytest.mark.parametrize("change", ["cell_count", "joint_statistic"])
def test_verified_snapshot_refuses_grid_or_joint_substitution(tmp_path, change):
    ref, plan, _ = _fixture(tmp_path)
    verified = load_verified_ref_scales(plan, ref)
    kwargs = {"cell_count": 4, "joint_statistic": "variogram"}
    kwargs[change] = 1 if change == "cell_count" else "energy"
    with pytest.raises(OrganizerFault):
        verified.load_scale(plan, plan.expected_units[0], ref / HANDLES[0] / "reference", **kwargs)
    with pytest.raises(TypeError):
        verified.joint_statistics[HANDLES[0]] = "energy"


@pytest.mark.parametrize("card", [b"[", b"scoring = 1", b'[scoring.params]\njoint = "unknown"'])
def test_preflight_rejects_invalid_joint_card_metadata(tmp_path, card):
    ref, plan, _ = _fixture(tmp_path)
    (ref / HANDLES[-1] / "card.toml").write_bytes(card)
    with pytest.raises(OrganizerFault):
        load_verified_ref_scales(plan, ref)


@pytest.mark.parametrize("mutation", ["absent", "symlink", "hardlink", "fifo", "oversize"])
def test_preflight_rejects_unsafe_card_metadata(tmp_path, mutation):
    ref, plan, _ = _fixture(tmp_path)
    card = ref / HANDLES[-1] / "card.toml"
    outside = tmp_path / "outside.toml"
    card.rename(outside)
    if mutation == "symlink":
        card.symlink_to(outside)
    elif mutation == "hardlink":
        os.link(outside, card)
    elif mutation == "fifo":
        os.mkfifo(card)
    elif mutation == "oversize":
        card.write_bytes(b" " * (normalization.DEFAULT_LIMITS.max_meta_bytes + 1))
    with pytest.raises(OrganizerFault):
        load_verified_ref_scales(plan, ref)


def test_preflight_preserves_single_cell_zero_joint_but_refuses_energy(tmp_path):
    ref, _, plan = build_evaluation(
        tmp_path,
        handles=[HANDLES[0]],
        grids={HANDLES[0]: (["SYN-A"], [1])},
        ref_scale={"marginal": 1.0, "joint": 0.0, "tail": 1.0},
    )
    verified = load_verified_ref_scales(plan, ref)
    assert (
        verified.load_scale(
            plan, plan.expected_units[0], ref / HANDLES[0] / "reference", cell_count=1
        ).joint
        is None
    )
    card = ref / HANDLES[0] / "card.toml"
    card.write_text(card.read_text().replace('joint       = "variogram"', 'joint       = "energy"'))
    with pytest.raises(OrganizerFault, match="not positive"):
        load_verified_ref_scales(plan, ref)


@pytest.mark.parametrize("mismatch", ["plan", "root", "entry"])
def test_verified_bundle_cannot_be_reused_with_another_context(tmp_path, mismatch):
    ref, plan, _ = _fixture(tmp_path)
    verified = load_verified_ref_scales(plan, ref)
    entry = plan.expected_units[0]
    reference = ref / entry.unit_handle / "reference"
    if mismatch == "plan":
        plan = EvaluationPlan.from_mapping(make_plan(HANDLES))
    elif mismatch == "root":
        reference = tmp_path / "unrelated" / "reference"
    else:
        entry = EvaluationPlan.from_mapping(make_plan(["u-99998888"])).expected_units[0]
    with pytest.raises(OrganizerFault):
        verified.load_scale(plan, entry, reference, cell_count=4)


def test_official_checks_all_scales_before_any_participant_verifier(tmp_path, monkeypatch):
    ref, res, plan = build_evaluation(tmp_path, broken={HANDLES[0]: "no_output"})
    scale = ref / HANDLES[-1] / "reference" / "ref_scale.json"
    scale.write_bytes(scale.read_bytes() + b" ")
    monkeypatch.setattr(official, "build_verifier", lambda ctx: pytest.fail("participant gate ran"))
    with pytest.raises(OrganizerFault, match="commitment mismatch"):
        official.score_roster(plan, ref, res)


def test_official_reads_each_scale_only_once_and_uses_verified_bytes(tmp_path, monkeypatch):
    ref, res, plan = build_evaluation(tmp_path)
    baseline = official.score_roster(plan, ref, res)
    reads = []
    reader = normalization._read_scale_bytes
    verifier = official.load_verified_ref_scales

    def record_read(*args, **kwargs):
        reads.append(args[1])
        return reader(*args, **kwargs)

    def verify_then_mutate(*args, **kwargs):
        snapshot = verifier(*args, **kwargs)
        for handle in HANDLES:
            (ref / handle / "reference" / "ref_scale.json").write_text("{}")
        return snapshot

    monkeypatch.setattr(normalization, "_read_scale_bytes", record_read)
    monkeypatch.setattr(official, "load_verified_ref_scales", verify_then_mutate)
    result = official.score_roster(plan, ref, res)
    assert result.aggregate.value == baseline.aggregate.value
    assert reads == [(handle, "reference") for handle in HANDLES]


def test_production_hydration_refuses_missing_or_fake_verification_and_ignores_preloaded_scale(
    tmp_path,
):
    ref, res, plan = build_evaluation(tmp_path)
    entry = plan.expected_units[0]
    snapshot = load_verified_ref_scales(plan, ref)
    base = {
        "plan": plan,
        "plan_entry": entry,
        "unit_handle": entry.unit_handle,
        "unit_dir": ref / entry.unit_handle,
        "output_dir": res / entry.unit_handle,
        "ref_scale": RefScale(123.0, 456.0, 789.0),
    }
    for fake in (None, {}, object(), snapshot.bundle):
        with pytest.raises(OrganizerFault):
            hydrate_ctx({**base, "verified_ref_scales": fake})
    ctx = {**base, "verified_ref_scales": snapshot}
    hydrate_ctx(ctx)
    assert ctx["ref_scale"] == RefScale(1.0, 1.0, 1.0)


def test_untrusted_unknown_scale_key_does_not_leak_its_text(tmp_path):
    ref, plan, _ = _fixture(tmp_path)
    marker = "private-numeric-denominator-983762.654321"
    (ref / HANDLES[0] / "reference" / "ref_scale.json").write_text(json.dumps({marker: 1.0}))
    plan = EvaluationPlan.from_mapping(
        make_plan(HANDLES, scale_commitment=read_ref_scale_bundle(ref, HANDLES).commitment)
    )
    with pytest.raises(OrganizerFault) as exc:
        load_verified_ref_scales(plan, ref)
    assert marker not in str(exc.value)
