"""Stage 1 of the text half: per-document summary agents.

Plain `unittest`, no pytest and no `qfbench2_common`, so it runs in a bare checkout:

    python3 -m unittest tests.test_text_signal -v

The model call is monkeypatched; nothing here touches the network.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import text_signal as ts  # noqa: E402

SHIPPED_TYPES = [
    "fomc_statement", "fomc_minutes", "cb_speech", "landmark",
    "beige_book", "macro_release", "positioning_report", "corporate_8k",
]


def _unit(tmp: pathlib.Path, docs, asof: str = "2024-12-18", index: bool = True) -> pathlib.Path:
    """Minimal unit dir. docs = [(doc_id, timestamp, doc_type, body)]."""
    text = tmp / "text"
    text.mkdir(parents=True)
    (tmp / "card.toml").write_text(f'[text]\ncutoff = "{asof}"\n')
    entries = []
    for doc_id, ts_, doc_type, body in docs:
        (text / f"{doc_id}.txt").write_text(body)
        entries.append({"doc_id": doc_id, "file": f"{doc_id}.txt", "timestamp": ts_,
                        "doc_type": doc_type, "source": "test"})
    if index:
        (text / "corpus_index.json").write_text(json.dumps({"asof": asof, "documents": entries}))
    return text


class TestCorpus(unittest.TestCase):
    def test_cutoff_excludes_future_docs(self):
        with tempfile.TemporaryDirectory() as d:
            text = _unit(pathlib.Path(d), [
                ("old", "2024-11-07", "fomc_statement", "a"),
                ("same_day", "2024-12-18", "fomc_statement", "b"),
                ("future", "2024-12-19", "fomc_statement", "c"),
            ])
            asof, docs = ts.load_corpus(text)
            self.assertEqual(asof, "2024-12-18")
            self.assertEqual([x["doc_id"] for x in docs], ["same_day", "old"])  # newest first

    def test_unindexed_files_are_never_read(self):
        with tempfile.TemporaryDirectory() as d:
            text = _unit(pathlib.Path(d), [("x", "2024-01-01", "cb_speech", "body")], index=False)
            self.assertEqual(ts.load_corpus(text)[1], [])

    def test_full_text_is_kept(self):
        body = "word " * 50_000
        with tempfile.TemporaryDirectory() as d:
            text = _unit(pathlib.Path(d), [("big", "2024-01-01", "beige_book", body)])
            self.assertEqual(ts.load_corpus(text)[1][0]["text"], body)


class TestPrompts(unittest.TestCase):
    def test_every_shipped_type_has_its_own_prompt(self):
        prompts = {t: ts.prompt_for(t, "2024-01-01") for t in SHIPPED_TYPES}
        self.assertEqual(len(set(prompts.values())), len(SHIPPED_TYPES))
        default = ts.prompt_for("default", "2024-01-01")
        for t, p in prompts.items():
            self.assertNotEqual(p, default, t)
            self.assertIn("2024-01-01", p)

    def test_unknown_type_gets_default(self):
        self.assertEqual(ts.prompt_for("mystery", "2024-01-01"),
                         ts.prompt_for("default", "2024-01-01"))


class TestAgents(unittest.TestCase):
    def test_short_doc_passes_through_without_a_call(self):
        doc = {"doc_id": "s", "timestamp": "2024-01-01", "doc_type": "fomc_statement",
               "text": "  The Committee decided to maintain the target range.  "}
        with mock.patch.object(ts, "call_model") as call:
            out = ts.summarize_doc(doc)
        call.assert_not_called()
        self.assertFalse(out["summarized"])
        self.assertEqual(out["summary"], doc["text"].strip())

    def test_long_doc_sends_whole_text_with_type_prompt(self):
        body = "x" * 200_000
        doc = {"doc_id": "b", "timestamp": "2024-04-17", "doc_type": "beige_book", "text": body}
        with mock.patch.object(ts, "call_model", return_value=("- activity flat", "")) as call:
            out = ts.summarize_doc(doc)
        system, user = call.call_args.args
        self.assertFalse(call.call_args.kwargs["thinking"])
        self.assertTrue(user.startswith(body))  # whole doc, then the format reminder
        self.assertIn("at most 15 bullets", user[len(body):])
        self.assertEqual(system, ts.prompt_for("beige_book", "2024-04-17"))
        self.assertTrue(out["summarized"])
        self.assertEqual(out["summary"], "- activity flat")
        self.assertEqual(out["full_chars"], 200_000)
        self.assertNotIn("text", out)

    def test_no_endpoint_records_error_and_never_raises(self):
        with tempfile.TemporaryDirectory() as d:
            text = _unit(pathlib.Path(d), [("m", "2024-01-01", "fomc_minutes", "y" * 10_000)])
            with mock.patch.dict(os.environ, {"MODEL_ENDPOINT": ""}):
                out = ts.summarize_corpus(text)
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0]["summary"])
        self.assertIn("MODEL_ENDPOINT", out[0]["error"])

    def test_exception_in_call_is_contained(self):
        doc = {"doc_id": "e", "timestamp": "2024-01-01", "doc_type": "cb_speech", "text": "z" * 5000}
        with mock.patch.object(ts, "call_model", side_effect=RuntimeError("boom")):
            out = ts.summarize_doc(doc)
        self.assertIsNone(out["summary"])
        self.assertIn("boom", out["error"])

    def test_missing_dir_returns_empty(self):
        self.assertEqual(ts.summarize_corpus(pathlib.Path("/nonexistent/text")), [])


class TestCleanup(unittest.TestCase):
    def test_cot_keeps_largest_open_interest_row_per_date(self):
        text = ("CFTC header\nreport_date  open_interest  noncomm_long  noncomm_short  noncomm_net\n"
                " 2014-11-11         155473          9516          69101       -59585\n"
                " 2014-11-11        1509371        442119         165287       276832\n"
                " 2014-11-04        1400000        400000         150000       250000\n")
        out = ts.main_contract_rows(text)
        self.assertNotIn("-59585", out)
        self.assertIn("276832", out)
        self.assertLess(out.index("2014-11-04"), out.index("2014-11-11"))  # chronological
        self.assertTrue(out.startswith("CFTC header"))

    def test_empty_reply_is_retried_then_reported(self):
        doc = {"doc_id": "e", "timestamp": "2024-01-01", "doc_type": "macro_release", "text": "z" * 5000}
        with mock.patch.object(ts, "call_model", side_effect=[("-", ""), ("- CPI rose 0.3 percent in May", "")]):
            self.assertEqual(ts.summarize_doc(doc)["summary"], "- CPI rose 0.3 percent in May")
        with mock.patch.object(ts, "call_model", return_value=("-", "")):
            out = ts.summarize_doc(doc)
        self.assertIsNone(out["summary"])
        self.assertIn("no content", out["error"])


class TestContract(unittest.TestCase):
    def test_read_text_signal_is_neutral_with_no_corpus(self):
        # Stage 2 no longer returns exact neutral unconditionally -- but a directory with no
        # admissible documents at all is still one of the paths that does.
        out = ts.read_text_signal(pathlib.Path("/nonexistent/text"), ["UST_2Y", "JPY"])
        self.assertEqual(out, {a: {"shift": 0.0, "widen": 1.0, "skew": 0.0}
                               for a in ["UST_2Y", "JPY"]})


class TestFamilyPrompts(unittest.TestCase):
    _CTX = {"asof": "2024-01-01", "horizons": [21], "value_unit": "percent_per_annum",
            "target_type": "level", "family": "F1", "level": {"UST_2Y": 4.5},
            "sigma": {"UST_2Y": 0.1}, "sigma_horizon": 21}

    def _system(self, family: str) -> str:
        ctx = {**self._CTX, "family": family}
        system, _ = ts.build_adjustment_prompt(
            [{"doc_id": "d", "timestamp": "2024-01-01", "doc_type": "fomc_statement",
              "summary": "- the Committee held rates steady"}],
            ["UST_2Y"], ctx,
        )
        return system

    def test_every_family_gets_distinct_focus(self):
        systems = {f: self._system(f) for f in ("F1", "F2", "F3", "F4")}
        self.assertEqual(len(set(systems.values())), 4)
        for f, s in systems.items():
            self.assertIn(f, s)

    def test_unknown_family_gets_default_focus(self):
        self.assertEqual(self._system("F9"), self._system("default"))

    def test_user_message_carries_summaries_and_asset_context(self):
        ctx = {**self._CTX}
        _, user = ts.build_adjustment_prompt(
            [{"doc_id": "fomc-2024-01-01", "timestamp": "2024-01-01", "doc_type": "fomc_statement",
              "summary": "- held rates steady"}],
            ["UST_2Y"], ctx,
        )
        self.assertIn("fomc-2024-01-01", user)
        self.assertIn("held rates steady", user)
        self.assertIn("level 4.500000", user)
        self.assertIn("sigma 0.100000", user)


class TestExtractJsonObject(unittest.TestCase):
    def test_well_formed_object(self):
        out = ts._extract_json_object('noise before {"assets": {"UST_2Y": {"drift_sd": 0.3}}}')
        self.assertEqual(out, {"assets": {"UST_2Y": {"drift_sd": 0.3}}})

    def test_repairs_missing_closing_brace(self):
        # The exact defect the removed pipeline measured: three braces opened, two closed,
        # finish_reason "stop" -- not a truncation, the model just dropped the last brace.
        broken = '{"assets": {"MOM": {"drift_sd": 0.0, "vol_scale": 1.0, "skew": 0.0}}'
        out = ts._extract_json_object(broken)
        self.assertEqual(out, {"assets": {"MOM": {"drift_sd": 0.0, "vol_scale": 1.0, "skew": 0.0}}})

    def test_no_json_returns_none(self):
        self.assertIsNone(ts._extract_json_object("sorry, I cannot help with that"))

    def test_never_invents_a_value(self):
        # Only closers are appended; an unterminated string is closed, not completed with data.
        out = ts._extract_json_object('{"assets": {"JPY": {"evidence": "unterminated')
        self.assertEqual(out, {"assets": {"JPY": {"evidence": "unterminated"}}})


class TestToAdjustments(unittest.TestCase):
    _CTX = {"sigma": {"UST_2Y": 0.20, "JPY": 0.0}}

    def test_converts_drift_sd_to_shift_via_sigma(self):
        raw = {"assets": {"UST_2Y": {"drift_sd": 0.5, "vol_scale": 1.2, "skew": 0.1}}}
        out, ledger = ts.to_adjustments(raw, ["UST_2Y"], self._CTX)
        self.assertAlmostEqual(out["UST_2Y"]["shift"], 0.5 * 0.20)
        self.assertAlmostEqual(out["UST_2Y"]["widen"], 1.2)
        self.assertAlmostEqual(out["UST_2Y"]["skew"], 0.1)

    def test_clamps_out_of_range_values(self):
        raw = {"assets": {"UST_2Y": {"drift_sd": 99.0, "vol_scale": 99.0, "skew": 99.0}}}
        out, ledger = ts.to_adjustments(raw, ["UST_2Y"], self._CTX)
        self.assertAlmostEqual(out["UST_2Y"]["shift"], ts._DRIFT_SD_CLAMP * 0.20)
        self.assertAlmostEqual(out["UST_2Y"]["widen"], ts._WIDEN_CLAMP[1])
        self.assertAlmostEqual(out["UST_2Y"]["skew"], ts._SKEW_CLAMP[1])
        self.assertIn("clamped", ledger["UST_2Y"]["note"])

    def test_missing_asset_gets_neutral_others_unaffected(self):
        raw = {"assets": {"UST_2Y": {"drift_sd": 0.4, "vol_scale": 1.0, "skew": 0.0}}}
        out, ledger = ts.to_adjustments(raw, ["UST_2Y", "JPY"], {"sigma": {"UST_2Y": 0.2}})
        self.assertNotEqual(out["UST_2Y"], ts.NEUTRAL)
        self.assertEqual(out["JPY"], ts.NEUTRAL)

    def test_non_finite_numbers_become_neutral_for_that_asset(self):
        raw = {"assets": {"UST_2Y": {"drift_sd": float("nan"), "vol_scale": 1.0, "skew": 0.0}}}
        out, ledger = ts.to_adjustments(raw, ["UST_2Y"], self._CTX)
        self.assertEqual(out["UST_2Y"], ts.NEUTRAL)
        self.assertIn("non-finite", ledger["UST_2Y"]["note"])

    def test_no_sigma_forces_shift_to_zero_but_keeps_widen(self):
        raw = {"assets": {"JPY": {"drift_sd": 1.0, "vol_scale": 1.3, "skew": 0.0}}}
        out, ledger = ts.to_adjustments(raw, ["JPY"], self._CTX)  # JPY sigma is 0.0
        self.assertEqual(out["JPY"]["shift"], 0.0)
        self.assertAlmostEqual(out["JPY"]["widen"], 1.3)

    def test_accepts_top_level_assets_without_nesting(self):
        # Some replies key assets at the top level instead of under "assets" -- same tolerance
        # the removed pipeline had for this exact ambiguity.
        raw = {"UST_2Y": {"drift_sd": 0.3, "vol_scale": 1.0, "skew": 0.0}}
        out, _ = ts.to_adjustments(raw, ["UST_2Y"], self._CTX)
        self.assertAlmostEqual(out["UST_2Y"]["shift"], 0.3 * 0.20)


class TestReadTextSignalStage2(unittest.TestCase):
    def test_successful_adjustment_end_to_end(self):
        fake_summaries = [
            {"doc_id": "fomc-2024-01-01", "timestamp": "2024-01-01", "doc_type": "fomc_statement",
             "summary": "- the Committee signalled further tightening ahead", "error": ""},
        ]
        fake_ctx = {"asof": "2024-01-01", "horizons": [21], "value_unit": "percent_per_annum",
                    "target_type": "level", "family": "F2", "level": {"UST_2Y": 4.5},
                    "sigma": {"UST_2Y": 0.2}, "sigma_horizon": 21}
        reply = json.dumps({"assets": {"UST_2Y": {"drift_sd": 0.6, "vol_scale": 1.4, "skew": 0.2,
                                                    "evidence": "further tightening ahead"}}})
        with tempfile.TemporaryDirectory() as d:
            text = _unit(pathlib.Path(d), [("x", "2024-01-01", "fomc_statement", "irrelevant")])
            with mock.patch.object(ts, "summarize_corpus", return_value=fake_summaries), \
                 mock.patch.object(ts, "load_context", return_value=fake_ctx), \
                 mock.patch.object(ts, "call_model", return_value=(reply, "")) as call:
                out = ts.read_text_signal(text, ["UST_2Y"])
        self.assertAlmostEqual(out["UST_2Y"]["shift"], 0.6 * 0.2)
        self.assertAlmostEqual(out["UST_2Y"]["widen"], 1.4)
        self.assertAlmostEqual(out["UST_2Y"]["skew"], 0.2)
        self.assertFalse(call.call_args.kwargs["thinking"])  # off, per the removed pipeline's finding

    def test_unparseable_reply_falls_back_to_neutral(self):
        with tempfile.TemporaryDirectory() as d:
            text = _unit(pathlib.Path(d), [("x", "2024-01-01", "fomc_statement", "irrelevant")])
            with mock.patch.object(ts, "summarize_corpus",
                                    return_value=[{"doc_id": "x", "timestamp": "2024-01-01",
                                                    "doc_type": "fomc_statement", "summary": "- ok"}]), \
                 mock.patch.object(ts, "call_model", return_value=("not json at all", "")):
                out = ts.read_text_signal(text, ["UST_2Y"])
        self.assertEqual(out, {"UST_2Y": dict(ts.NEUTRAL)})

    def test_adjustment_call_failure_falls_back_to_neutral(self):
        with tempfile.TemporaryDirectory() as d:
            text = _unit(pathlib.Path(d), [("x", "2024-01-01", "fomc_statement", "irrelevant")])
            with mock.patch.object(ts, "summarize_corpus",
                                    return_value=[{"doc_id": "x", "timestamp": "2024-01-01",
                                                    "doc_type": "fomc_statement", "summary": "- ok"}]), \
                 mock.patch.object(ts, "call_model", return_value=(None, "HTTP 500")):
                out = ts.read_text_signal(text, ["UST_2Y"])
        self.assertEqual(out, {"UST_2Y": dict(ts.NEUTRAL)})


if __name__ == "__main__":
    unittest.main()
