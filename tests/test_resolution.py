"""Development-only end-to-end rehearsal of candidate-2; no real unit, outcome or signer.

## Executive summary (read this first)

The signed examples use only the toolkit's published development seed and fixture documents.
All data and source snapshots are synthetic. Package commitments are computed from the actual
imported Common and Track 2 source trees. These tests exercise the existing gates and scoring
math, and demonstrate that ordinary C2 binding still refuses the later resolution C1.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from qfbench2_common.contracts import (
    ContractError,
    EvaluationPlan,
    OrganizerFault,
    RunRecord,
    attestation_payload,
    digest_json,
    digest_tree,
    sign_payload,
)
from qfbench2_common.contracts.digest import sha256_bytes
from qfbench2_common.contracts.fixtures import DEV_KEY_ID, DEV_SEED, dev_trust_store, load_fixture

from qfbench2_track_forecasting import resolution as subject


def _encoded(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False).encode()


def _signed(value: dict, at: str) -> dict:
    value = copy.deepcopy(value)
    value.pop("signature", None)
    value["signature"] = sign_payload(
        value, seed=DEV_SEED, key_id=DEV_KEY_ID, signed_at=at
    ).to_mapping()
    return value


def _commit(values: dict[str, bytes]) -> str:
    return digest_json({k: sha256_bytes(v) for k, v in values.items()})


def _parquet(columns: dict[str, list]) -> bytes:
    stream = pa.BufferOutputStream()
    pq.write_table(pa.table(columns), stream)
    return stream.getvalue().to_pybytes()


def _card(index: int) -> bytes:
    return f"""schema_version = "2.0"
[task]
id = "t2-SYN-adapter-{index}"
track = "forecasting"
title = "Synthetic forecast protocol rehearsal"
split = "private-test"
[metadata]
author_name = "Synthetic fixture"
difficulty = "easy"
category = "synthetic"
tags = ["synthetic"]
[provenance]
license = "BSD-3-Clause"
data_cutoff = "2026-10-01"
public_release_date = "2026-01-01"
redistributable = false
manifest = "manifest.json"
[contamination]
canary_guid = "00000000-0000-4000-8000-{index + 1:012d}"
[scoring]
verifier = "t2.crps_composite"
metric = "crps_composite"
admissibility_gates = ["g0_integrity", "g1_schema", "g2_cutoff_resource", "g3_domain_semantics"]
[scoring.params]
tail_levels = [0.01, 0.05, 0.95, 0.99]
tail_metric = "pinball"
joint = "variogram"
[scoring.params.weights]
marginal = 0.5
joint = 0.3
tail = 0.2
[environment]
cpus = 2
memory = "4G"
gpu = false
network = "restricted"
[text]
path = "text/"
cutoff = "2026-10-01"
[targets]
asset_ids = ["SYN-A", "SYN-B"]
horizons = [1, 5]
target_type = "level"
target_dates = ["2026-10-05", "2026-10-09"]
""".encode()


def _refresh(c: dict) -> None:
    """Publish a coherent synthetic chain after an intentional semantic fixture change."""
    protocol = json.loads(c["protocol"])
    protocol["scoring_inputs"]["cards_commitment"] = _commit(
        {h: members["card.toml"] for h, members in c["input_snapshots"].items()}
    )
    protocol["scoring_inputs"]["snapshots_commitment"] = digest_json(
        {h: _commit(members) for h, members in c["input_snapshots"].items()}
    )
    protocol = _signed(protocol, "2026-09-30T00:00:00Z")
    c["protocol"] = _encoded(protocol)
    descriptor = json.loads(c["descriptor"])
    for handle, members in c["forecasts"].items():
        tree = load_fixture("c3_artifact_tree.json")
        tree["rejections"] = []
        tree["entries"] = [
            {
                "path": p,
                "size_bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "mode_bits": 0o644,
                "num_rows": None,
            }
            for p, raw in sorted(members.items())
        ]
        tree["root_digest"] = digest_tree(tree["entries"])
        c["trees"][handle] = _encoded(tree)
        record = (
            json.loads(c["records"][handle])
            if handle in c["records"]
            else load_fixture("c2_run_record.json")
        )
        record.update(unit_handle=handle, run_id="synthetic-" + handle)
        record["bindings"].update(
            plan_digest=protocol["signature"]["payload_digest"],
            descriptor_digest=descriptor["descriptor_digest"],
            sanitized_tree_digest=tree["root_digest"],
        )
        record["image"]["resolved_digest"] = descriptor["image"]["digest"]
        record["timing"].update(started_at="2026-10-01T01:00:00Z", ended_at="2026-10-01T01:04:12Z")
        record["attestation"]["signature"] = sign_payload(
            attestation_payload(record),
            seed=DEV_SEED,
            key_id=DEV_KEY_ID,
            signed_at="2026-10-01T01:05:00Z",
        ).to_mapping()
        c["records"][handle] = _encoded(record)
    receipt = _signed(
        dict(
            schema_version="candidate-2",
            kind="forecast_receipt",
            protocol_digest=protocol["signature"]["payload_digest"],
            descriptor_digest=descriptor["descriptor_digest"],
            image_digest=descriptor["image"]["digest"],
            config_digest=sha256_bytes(c["config"]),
            model_dependencies_commitment=_commit(c["model_dependencies"]),
            records_commitment=_commit(c["records"]),
            trees_commitment=_commit(c["trees"]),
        ),
        "2026-10-01T02:00:00Z",
    )
    c["receipt"] = _encoded(receipt)
    plan = copy.deepcopy(protocol["resolution_template"])
    plan["normalization"]["ref_scale_commitment"] = _commit(c["scales"])
    plan = _signed(plan, "2026-10-06T00:00:00Z")
    c["resolution"] = _encoded(
        _signed(
            dict(
                schema_version="candidate-2",
                kind="forecast_resolution",
                supersedes=protocol["signature"]["payload_digest"],
                receipt_digest=receipt["signature"]["payload_digest"],
                plan=plan,
                outcomes_commitment=_commit(c["outcomes"]),
                scales_commitment=_commit(c["scales"]),
            ),
            "2026-10-06T00:01:00Z",
        )
    )


def _case() -> dict:
    plan = load_fixture("c1/forecasting_final.expanded.json")
    plan.pop("signature")
    plan["normalization"] = {"mode": "ref_scale"}
    runtime = subject.runtime_source_digests()
    plan["scorer"].update(
        package="qfbench2_track_forecasting", digest=runtime["track_source_tree_digest"]
    )
    handles = [e["unit_handle"] for e in plan["roster"]["expected_units"]]
    for entry in plan["roster"]["expected_units"]:
        grid = dict(assets=["SYN-A", "SYN-B"], horizons=[1, 5], cell_count=4)
        grid["digest"] = digest_json(grid)
        entry["grid"] = grid
    policy = {
        h: [
            dict(
                asset=a,
                horizon=n,
                source="synthetic-source",
                first_public_not_before="2026-10-05T12:00:00Z",
            )
            for a in ("SYN-A", "SYN-B")
            for n in (1, 5)
        ]
        for h in handles
    }
    protocol = dict(
        schema_version="candidate-2",
        kind="forecast_protocol",
        resolution_template=plan,
        schedule=dict(
            information_cutoff="2026-10-01T00:00:00Z",
            forecast_deadline="2026-10-02T00:00:00Z",
            resolution_deadline="2026-11-30T23:59:59Z",
        ),
        scale_recipe=dict(
            implementation_digest=digest_json("synthetic unreproduced scale recipe"),
            version="synthetic-1",
            seed_policy=dict(algorithm="synthetic-fixed", seed=42),
            weights=dict(marginal=0.5, joint=0.3, tail=0.2),
            joint_statistics={h: "variogram" for h in handles},
        ),
        outcome_policy=dict(missing="abort_whole_evaluation", vintage="first_public", cells=policy),
        scoring_inputs=dict(runtime=runtime),
    )
    descriptor = load_fixture("c5/forecasting_final.json")
    c = dict(
        protocol=_encoded(protocol),
        descriptor=_encoded(descriptor),
        config=b'{"seed":17}',
        model_dependencies={
            m["name"]: b"synthetic model bytes; no model is loaded" for m in descriptor["models"]
        },
        records={},
        trees={},
        forecasts={},
        outcomes={},
        scales={},
        input_snapshots={},
        source_snapshots={sha256_bytes(b"synthetic public release"): b"synthetic public release"},
        organizer_trust=dev_trust_store(),
        runner_trust=dev_trust_store(),
        receipt_trust=dev_trust_store(),
        now=datetime(2026, 10, 7, tzinfo=UTC),
        require_production_trust=False,
    )
    for index, handle in enumerate(handles):
        c["input_snapshots"][handle] = {
            "card.toml": _card(index),
            "panels/synthetic.parquet": _parquet({"date": ["2026-09-30"], "value": [1.0]}),
            "text/corpus_index.json": _encoded(
                {
                    "documents": [
                        {"doc_id": "synthetic", "path": "synthetic.md", "timestamp": "2026-09-30"}
                    ]
                }
            ),
            "text/synthetic.md": b"Synthetic dated input.",
        }
        cells = policy[handle]
        c["forecasts"][handle] = {
            "forecast_meta.json": _encoded(
                dict(
                    schema_version="2.0",
                    unit_id=f"t2-SYN-adapter-{index}",
                    asof="2026-10-01",
                    target="level",
                    representation="samples",
                    n_draws=200,
                    asset_ids=["SYN-A", "SYN-B"],
                    horizons=[1, 5],
                )
            ),
            "forecast_rationale.md": b"Synthetic rationale; presence only.",
            "forecast.parquet": _parquet(
                {
                    "draw": [draw for draw in range(200) for _ in cells],
                    "asset": [cell["asset"] for _ in range(200) for cell in cells],
                    "horizon": [cell["horizon"] for _ in range(200) for cell in cells],
                    "value": [0.5 + draw / 200 for draw in range(200) for _ in cells],
                }
            ),
        }
        c["outcomes"][handle] = _encoded(
            {
                "cells": [
                    dict(
                        asset=cell["asset"],
                        horizon=cell["horizon"],
                        value=0.75,
                        source=cell["source"],
                        first_public_at="2026-10-05T12:00:00Z",
                        source_content_digest=next(iter(c["source_snapshots"])),
                    )
                    for cell in cells
                ]
            }
        )
        c["scales"][handle] = _encoded(dict(marginal=1.0, joint=2.0, tail=3.0))
    _refresh(c)
    return c


def test_actual_canonical_end_to_end_is_explicitly_not_adopted() -> None:
    c = _case()
    result = subject.score_forecast_resolution(**c)
    assert not result.rankable and not result.production_image_certified
    assert result.chain.unit_count == len(result.diagnostic.rows) == 3
    assert not result.diagnostic.operator_reasons
    assert result.diagnostic.aggregate.value > 0
    assert "scale_recipe_generation_not_reproduced" in result.acceptance_blockers
    assert result.runtime_source_digests == subject.runtime_source_digests()
    record = RunRecord.from_mapping(json.loads(next(iter(c["records"].values()))))
    plan = EvaluationPlan.from_mapping(json.loads(c["resolution"])["plan"])
    with pytest.raises(ContractError, match="plan_digest"):
        record.verify_bindings(plan_digest=plan.plan_digest)


def test_default_production_trust_refuses_published_development_signatures() -> None:
    c = _case()
    del c["require_production_trust"]
    with pytest.raises(OrganizerFault):
        subject.score_forecast_resolution(**c)


@pytest.mark.parametrize(
    "key",
    [
        "config",
        "model_dependencies",
        "records",
        "trees",
        "forecasts",
        "outcomes",
        "scales",
        "input_snapshots",
        "source_snapshots",
    ],
)
def test_mutated_retained_bytes_are_organizer_faults(key: str) -> None:
    c = _case()
    if key == "config":
        c[key] += b" "
    elif key in ("forecasts", "input_snapshots"):
        members = c[key][next(iter(c[key]))]
        members[next(iter(members))] += b" "
    else:
        c[key][next(iter(c[key]))] += b" "
    with pytest.raises(OrganizerFault):
        subject.score_forecast_resolution(**c)


@pytest.mark.parametrize(
    "key",
    [
        "model_dependencies",
        "records",
        "trees",
        "forecasts",
        "outcomes",
        "scales",
        "input_snapshots",
        "source_snapshots",
    ],
)
def test_missing_retained_bytes_never_become_participant_failures(key: str) -> None:
    c = _case()
    c[key].pop(next(iter(c[key])))
    with pytest.raises(OrganizerFault):
        subject.score_forecast_resolution(**c)


@pytest.mark.parametrize("key", ["common_source_tree_digest", "track_source_tree_digest"])
def test_correctly_signed_wrong_runtime_is_refused(key: str) -> None:
    c = _case()
    protocol = json.loads(c["protocol"])
    protocol["scoring_inputs"]["runtime"][key] = digest_json("different synthetic runtime")
    if key == "track_source_tree_digest":
        protocol["resolution_template"]["scorer"]["digest"] = protocol["scoring_inputs"]["runtime"][
            key
        ]
    c["protocol"] = _encoded(protocol)
    _refresh(c)
    with pytest.raises(OrganizerFault, match="source identity"):
        subject.score_forecast_resolution(**c)


@pytest.mark.parametrize(
    "old,new",
    [
        (b"marginal = 0.5", b"marginal = 0.6"),
        (b'data_cutoff = "2026-10-01"', b'data_cutoff = "2026-10-02"'),
    ],
)
def test_authentic_but_inconsistent_card_is_an_organizer_fault(old: bytes, new: bytes) -> None:
    c = _case()
    members = c["input_snapshots"][next(iter(c["input_snapshots"]))]
    members["card.toml"] = members["card.toml"].replace(old, new)
    _refresh(c)
    with pytest.raises(OrganizerFault):
        subject.score_forecast_resolution(**c)


def test_late_committed_panel_is_refused_by_canonical_input_gate() -> None:
    c = _case()
    members = c["input_snapshots"][next(iter(c["input_snapshots"]))]
    members["panels/synthetic.parquet"] = _parquet({"date": ["2026-10-02"], "value": [1.0]})
    _refresh(c)
    with pytest.raises(OrganizerFault):
        subject.score_forecast_resolution(**c)


def test_late_committed_text_is_refused_by_canonical_input_gate() -> None:
    c = _case()
    members = c["input_snapshots"][next(iter(c["input_snapshots"]))]
    members["text/corpus_index.json"] = members["text/corpus_index.json"].replace(
        b"2026-09-30", b"2026-10-02"
    )
    _refresh(c)
    with pytest.raises(OrganizerFault):
        subject.score_forecast_resolution(**c)


@pytest.mark.parametrize("directory", ["panels/", "text/"])
def test_signed_incomplete_input_tree_is_an_organizer_fault(directory: str) -> None:
    c = _case()
    handle = next(iter(c["input_snapshots"]))
    c["input_snapshots"][handle] = {
        name: raw
        for name, raw in c["input_snapshots"][handle].items()
        if not name.startswith(directory)
    }
    _refresh(c)
    with pytest.raises(OrganizerFault):
        subject.score_forecast_resolution(**c)


def test_trust_mode_does_not_coerce_false_like_values() -> None:
    c = _case()
    c["require_production_trust"] = 0
    with pytest.raises(OrganizerFault, match="must be boolean"):
        subject.score_forecast_resolution(**c)


def test_participant_schema_defect_remains_a_canonical_participant_refusal() -> None:
    c = _case()
    members = c["forecasts"][next(iter(c["forecasts"]))]
    meta = json.loads(members["forecast_meta.json"])
    meta["n_draws"] = 3
    members["forecast_meta.json"] = _encoded(meta)
    _refresh(c)
    result = subject.score_forecast_resolution(**c)
    assert len(result.diagnostic.operator_reasons) == 1
    assert len(result.diagnostic.rows) == 3


def test_scratch_mutation_is_refused_and_original_bytes_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _case()
    retained = copy.deepcopy(c["forecasts"])
    score = subject.score_roster

    def mutated(plan: EvaluationPlan, ref: Path, res: Path, **kwargs: Any) -> Any:
        result = score(plan, ref, res, **kwargs)
        path = res / plan.expected_handles[0] / "forecast_rationale.md"
        path.chmod(0o600)
        path.write_bytes(b"changed")
        return result

    monkeypatch.setattr(subject, "score_roster", mutated)
    with pytest.raises(OrganizerFault, match="inputs changed"):
        subject.score_forecast_resolution(**c)
    assert c["forecasts"] == retained


def _edit_record(c: dict, change: Any) -> None:
    handle = next(iter(c["records"]))
    record = json.loads(c["records"][handle])
    change(record)
    record["attestation"]["signature"] = sign_payload(
        attestation_payload(record),
        seed=DEV_SEED,
        key_id=DEV_KEY_ID,
        signed_at=record["attestation"]["signature"]["signed_at"],
    ).to_mapping()
    c["records"][handle] = _encoded(record)
    receipt = json.loads(c["receipt"])
    receipt["records_commitment"] = _commit(c["records"])
    receipt = _signed(receipt, receipt["signature"]["signed_at"])
    c["receipt"] = _encoded(receipt)
    resolution = json.loads(c["resolution"])
    resolution["receipt_digest"] = receipt["signature"]["payload_digest"]
    c["resolution"] = _encoded(_signed(resolution, resolution["signature"]["signed_at"]))


@pytest.mark.parametrize(
    "change",
    [
        lambda record: record["image"].update(
            resolved_digest=sha256_bytes(b"wrong synthetic image")
        ),
        lambda record: record["timing"].update(ended_at="2026-10-03T00:00:00Z"),
        lambda record: record["bindings"].update(plan_digest=sha256_bytes(b"unrelated plan")),
    ],
)
def test_authentic_wrong_image_late_execution_or_other_binding_refuses(change: Any) -> None:
    c = _case()
    _edit_record(c, change)
    with pytest.raises(OrganizerFault):
        subject.score_forecast_resolution(**c)


def test_authentic_resolution_cannot_substitute_another_source() -> None:
    c = _case()
    handle = next(iter(c["outcomes"]))
    outcome = json.loads(c["outcomes"][handle])
    outcome["cells"][0]["source"] = "different-synthetic-source"
    c["outcomes"][handle] = _encoded(outcome)
    _refresh(c)
    with pytest.raises(OrganizerFault):
        subject.score_forecast_resolution(**c)


def test_committed_input_mutation_during_gate_scan_is_not_an_accepted_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _case()
    scan = subject.scan_panel_cutoff

    def mutated(root: Path, asof: str) -> Any:
        result = scan(root, asof)
        path = root / "synthetic.parquet"
        path.chmod(0o600)
        path.write_bytes(_parquet({"date": ["2026-09-30"], "value": [99.0]}))
        return result

    monkeypatch.setattr(subject, "scan_panel_cutoff", mutated)
    with pytest.raises(OrganizerFault, match="staging differs"):
        subject.score_forecast_resolution(**c)


def test_uncommitted_imported_namespace_member_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _case()
    module = ModuleType("qfbench2_track_forecasting.injected")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(OrganizerFault, match="outside its complete source commitment"):
        subject.score_forecast_resolution(**c)
