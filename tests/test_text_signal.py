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
    def test_read_text_signal_is_exact_neutral(self):
        out = ts.read_text_signal(pathlib.Path("/nonexistent/text"), ["UST_2Y", "JPY"])
        self.assertEqual(out, {a: {"shift": 0.0, "widen": 1.0, "skew": 0.0}
                               for a in ["UST_2Y", "JPY"]})


if __name__ == "__main__":
    unittest.main()
