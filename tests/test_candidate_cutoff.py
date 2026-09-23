"""Candidate input cutoff and exact corpus coverage controls.

## Executive summary (read this first)

These synthetic files test the optional candidate policy through the canonical scanners.
Default callers keep their date-level behavior. No production input or signer is used.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from qfbench2_common.contracts import OrganizerFault

from qfbench2_track_forecasting.cutoff import scan_panel_cutoff, scan_text_corpus_cutoff
from qfbench2_track_forecasting.failures import T2Refusal
from qfbench2_track_forecasting.limits import ParseLimits

ASOF = "2026-10-01"
CUTOFF = "2026-10-01T00:00:00Z"


def _corpus(root: Path, documents: list[dict], files: tuple[str, ...] = ()) -> Path:
    root.mkdir()
    (root / "corpus_index.json").write_text(json.dumps({"documents": documents}))
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic document.\n")
    return root


def _document(**extra: Any) -> dict:
    return {"doc_id": "synthetic", "timestamp": "2026-09-30", **extra}


def _scan(root: Path, kind: str, value: Any, cutoff: str = CUTOFF) -> None:
    if kind == "text":
        _corpus(root, [_document(timestamp=value)])
        scan_text_corpus_cutoff(root, ASOF, information_cutoff=cutoff, strict_coverage=True)
    else:
        root.mkdir()
        pq.write_table(pa.table({"date": [value]}), root / "synthetic.parquet")
        scan_panel_cutoff(root, ASOF, information_cutoff=cutoff)


@pytest.mark.parametrize("kind", ["text", "panel"])
@pytest.mark.parametrize(
    "value",
    [
        "2026-09-30",
        "2026-10-01T00:00:00Z",
        "2026-09-30T23:59:59.999999999Z",
        "2026-10-01T02:00:00+02:00",
        "2026-09-30T20:00:00-04:00",
    ],
)
def test_strict_cutoff_accepts_proven_in_range_instants(tmp_path, kind, value):
    _scan(tmp_path / kind, kind, value)


@pytest.mark.parametrize("kind", ["text", "panel"])
@pytest.mark.parametrize(
    "value",
    [
        "2026-10-01T03:00:00Z",
        "2026-10-01T00:00:00.000000001Z",
        "2026-09-30T23:30:00-01:00",
        "2026-10-01",  # A date alone does not prove any time before the midnight cutoff.
        "2026-10-02",
        "2026-09-31",
        "2026-09-31T00:00:00Z",
        "2026-09-30T25:00:00Z",
        "2026-09-30T23:59:60Z",
        "2026-09-30T00:00:00",
        "2026-09-30T00:00:00-00:00",
        "2026-09-30T00:00:00+24:00",
        "2026-09-30T00:00:00+00:60",
        "2026-09-30garbage",
        "2026-09-30T00:00:00.0000000001Z",
        None,
    ],
)
def test_strict_cutoff_refuses_late_ambiguous_or_invalid_input(tmp_path, kind, value):
    with pytest.raises(OrganizerFault):
        _scan(tmp_path / kind, kind, value)


@pytest.mark.parametrize("kind", ["text", "panel"])
def test_strict_cutoff_keeps_fractional_protocol_boundary(tmp_path, kind):
    _scan(
        tmp_path / "equal", kind, "2026-10-01T00:00:00.123456Z", CUTOFF.replace("00Z", "00.123456Z")
    )
    with pytest.raises(OrganizerFault):
        _scan(
            tmp_path / "late",
            kind,
            "2026-10-01T00:00:00.123456001Z",
            CUTOFF.replace("00Z", "00.123456Z"),
        )


@pytest.mark.parametrize("kind", ["text", "panel"])
def test_whole_day_bound_accepts_next_midnight_but_keeps_card_asof(tmp_path, kind):
    _scan(tmp_path / "whole-day", kind, ASOF, "2026-10-02T00:00:00Z")
    with pytest.raises(OrganizerFault):
        _scan(tmp_path / "past-card", kind, "2026-10-02T00:00:00Z", "2026-10-03T00:00:00Z")


@pytest.mark.parametrize(
    ("values", "accepted"),
    [
        (pa.array([date(2026, 9, 30)], type=pa.date32()), True),
        (pa.array([date(2026, 9, 30)], type=pa.date64()), True),
        (pa.array([date(2026, 10, 1)], type=pa.date32()), False),
        (pa.array([date(2026, 10, 1)], type=pa.date64()), False),
        (pa.array([datetime(2026, 10, 1, tzinfo=UTC)], type=pa.timestamp("us", "UTC")), True),
        (pa.array([datetime(2026, 10, 1, tzinfo=UTC)], type=pa.timestamp("ns", "+02:00")), True),
        (pa.array([datetime(2026, 9, 30)], type=pa.timestamp("us")), False),
        (
            pa.array(
                [
                    pa.scalar(
                        datetime(2026, 10, 1, tzinfo=UTC), type=pa.timestamp("ns", "UTC")
                    ).value
                    + 1
                ],
                type=pa.timestamp("ns", "UTC"),
            ),
            False,
        ),
    ],
)
def test_strict_cutoff_uses_real_arrow_date_and_timestamp_cells(tmp_path, values, accepted):
    pq.write_table(pa.table({"date": values}), tmp_path / "synthetic.parquet")
    if accepted:
        assert scan_panel_cutoff(tmp_path, ASOF, information_cutoff=CUTOFF).scanned_rows == 1
    else:
        with pytest.raises(OrganizerFault):
            scan_panel_cutoff(tmp_path, ASOF, information_cutoff=CUTOFF)


@pytest.mark.parametrize("bad_asof", ["2026-09-31", "2026-13-01"])
def test_strict_cutoff_rejects_impossible_card_calendar_date(tmp_path, bad_asof):
    root = _corpus(tmp_path / "corpus", [_document()])
    with pytest.raises(OrganizerFault):
        scan_text_corpus_cutoff(root, bad_asof, information_cutoff=CUTOFF, strict_coverage=True)


@pytest.mark.parametrize(
    "entry",
    [
        _document(path="./synthetic.md"),
        _document(path="../synthetic.md"),
        _document(path="/synthetic.md"),
        _document(path="synthetic.md", file="other.md"),
        _document(path="synthetic.md", file=""),
        _document(path="synthetic.md", file=None),
        _document(path="a//synthetic.md"),
        _document(path="corpus_index.json"),
        _document(path="cafe\u0301.md"),
    ],
)
def test_strict_corpus_rejects_ambiguous_or_noncanonical_paths(tmp_path, entry):
    root = _corpus(tmp_path / "corpus", [entry], ("synthetic.md",))
    with pytest.raises(OrganizerFault):
        scan_text_corpus_cutoff(root, ASOF, strict_coverage=True)


@pytest.mark.parametrize("files", [("synthetic.md",), ("nested/corpus_index.json",)])
def test_pathless_index_never_covers_any_external_file(tmp_path, files):
    root = _corpus(tmp_path / "corpus", [_document(text="Synthetic inline text.")], files)
    with pytest.raises(OrganizerFault, match="do not cover each other exactly"):
        scan_text_corpus_cutoff(root, ASOF, strict_coverage=True)


def test_strict_corpus_accepts_inline_documents_without_external_files(tmp_path):
    root = _corpus(tmp_path / "corpus", [_document(text="Synthetic inline text.")])
    verdict = scan_text_corpus_cutoff(root, ASOF, information_cutoff=CUTOFF, strict_coverage=True)
    assert verdict.indexed_documents == 1 and verdict.files_on_disk == 0


def test_strict_corpus_accepts_nested_index_name_when_explicitly_dated(tmp_path):
    path = "nested/corpus_index.json"
    root = _corpus(tmp_path / "corpus", [_document(file=path)], (path,))
    verdict = scan_text_corpus_cutoff(root, ASOF, information_cutoff=CUTOFF, strict_coverage=True)
    assert verdict.files_on_disk == 1


def test_strict_corpus_accepts_matching_path_aliases_and_inline_record(tmp_path):
    root = _corpus(
        tmp_path / "corpus",
        [
            _document(path="synthetic.md", file="synthetic.md"),
            _document(doc_id="inline", text="Synthetic inline text."),
        ],
        ("synthetic.md",),
    )
    assert scan_text_corpus_cutoff(root, ASOF, strict_coverage=True).indexed_documents == 2


def test_strict_corpus_rejects_repeated_file_mapping(tmp_path):
    root = _corpus(tmp_path / "corpus", [_document(path="synthetic.md")] * 2, ("synthetic.md",))
    with pytest.raises(OrganizerFault, match="repeats a file path"):
        scan_text_corpus_cutoff(root, ASOF, strict_coverage=True)


def test_strict_scans_preserve_parser_limits(tmp_path):
    root = _corpus(tmp_path / "corpus", [_document()])
    with pytest.raises(T2Refusal):
        scan_text_corpus_cutoff(
            root,
            ASOF,
            information_cutoff=CUTOFF,
            strict_coverage=True,
            limits=ParseLimits(max_meta_bytes=1),
        )
    panels = tmp_path / "panels"
    panels.mkdir()
    pq.write_table(pa.table({"date": ["2026-09-30"]}), panels / "synthetic.parquet")
    with pytest.raises(T2Refusal):
        scan_panel_cutoff(panels, ASOF, information_cutoff=CUTOFF, limits=ParseLimits(max_rows=0))


def test_default_callers_keep_date_level_and_index_only_compatibility(tmp_path):
    root = _corpus(tmp_path / "corpus", [_document(timestamp="2026-10-01T03:00:00Z")])
    assert scan_text_corpus_cutoff(root, ASOF).indexed_documents == 1
    panels = tmp_path / "panels"
    panels.mkdir()
    pq.write_table(pa.table({"date": ["2026-10-01T03:00:00Z"]}), panels / "synthetic.parquet")
    assert scan_panel_cutoff(panels, ASOF).scanned_rows == 1
