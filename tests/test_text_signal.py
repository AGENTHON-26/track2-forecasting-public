"""The text half: the contract it must honour and the failures it must survive.

Plain `unittest`, no pytest and no `qfbench2_common`, so it runs in a bare checkout:

    python3 -m unittest tests.test_text_signal -v

Every test here is about the same property. `read_text_signal` sits on the path to a scored
submission, a card that raises takes the pre-committed worst case of 4.0, and ignoring the text
entirely scores 1.0 -- so an exception in the text half is four times worse than not having one.
It must always return the contract's shape, whatever the corpus, the card or the model does.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import text_signal as ts  # noqa: E402

KEYS = {"shift", "widen", "skew"}


def _unit(tmp: pathlib.Path, docs: list[tuple[str, str, str, str]], asof: str = "2024-12-18"):
    """Build a minimal unit dir. docs = [(doc_id, timestamp, doc_type, body)]."""
    text = tmp / "text"
    text.mkdir(parents=True, exist_ok=True)
    index = {"card_id": "t2-TEST", "asof": asof, "documents": []}
    for doc_id, ts_, doc_type, body in docs:
        fn = f"{doc_id}.txt"
        (text / fn).write_text(body, encoding="utf-8")
        index["documents"].append(
            {"doc_id": doc_id, "timestamp": ts_, "doc_type": doc_type, "file": fn}
        )
    (text / "corpus_index.json").write_text(json.dumps(index))
    (tmp / "card.toml").write_text(
        f"[task]\nid='t2-TEST'\n\n[text]\ncutoff='{asof}'\n\n"
        "[targets]\nasset_ids=['UST_2Y']\nhorizons=[21]\nvalue_unit='percent_per_annum'\n"
        "target_type='level'\n"
    )
    return text


class ContractShape(unittest.TestCase):
    """Whatever happens, the caller gets {asset: {shift, widen, skew}} of floats."""

    def _check(self, adj, assets):
        self.assertEqual(set(adj), set(assets))
        for a in assets:
            self.assertEqual(set(adj[a]), KEYS, f"{a} has the wrong keys")
            for k, v in adj[a].items():
                self.assertIsInstance(v, float, f"{a}.{k} is not a float")
                self.assertTrue(v == v and abs(v) != float("inf"), f"{a}.{k} is not finite")

    def test_missing_directory(self):
        self._check(ts.read_text_signal(pathlib.Path("/nonexistent/text"), ["UST_2Y"]), ["UST_2Y"])

    def test_no_corpus_index(self):
        with tempfile.TemporaryDirectory() as d:
            text = pathlib.Path(d) / "text"
            text.mkdir()
            (text / "stray.txt").write_text("the Committee raised the target range")
            adj = ts.read_text_signal(text, ["UST_2Y"])
        self._check(adj, ["UST_2Y"])
        # An unindexed file has no timestamp, so it cannot be shown to predate the as-of. Not read.
        self.assertEqual(adj["UST_2Y"], ts.NEUTRAL)

    def test_corrupt_corpus_index(self):
        with tempfile.TemporaryDirectory() as d:
            text = pathlib.Path(d) / "text"
            text.mkdir()
            (text / "corpus_index.json").write_text("{ this is not json")
            self._check(ts.read_text_signal(text, ["UST_2Y", "JPY"]), ["UST_2Y", "JPY"])

    def test_empty_asset_list(self):
        self.assertEqual(ts.read_text_signal(pathlib.Path("/nonexistent"), []), {})


class Cutoff(unittest.TestCase):
    """A document the index dates after the as-of has leaked before any gate runs."""

    def test_post_asof_document_is_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            text = _unit(
                pathlib.Path(d),
                [
                    ("before", "2024-12-18", "fomc_statement", "the Committee is patient. " * 20),
                    ("after", "2024-12-19", "fomc_statement", "LEAKED " * 200),
                ],
            )
            docs, excluded, _ = ts.read_corpus(text, "2024-12-18")
        self.assertEqual([d["doc_id"] for d in docs], ["before"])
        self.assertEqual(excluded, 1)
        self.assertNotIn("LEAKED", " ".join(d["text"] for d in docs))

    def test_indexed_but_missing_file_is_excluded_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            text = _unit(pathlib.Path(d), [("gone", "2024-01-01", "fomc_statement", "x")])
            (text / "gone.txt").unlink()
            docs, excluded, _ = ts.read_corpus(text, "2024-12-18")
        self.assertEqual(docs, [])
        self.assertEqual(excluded, 1)


class PromptBudget(unittest.TestCase):
    def test_excerpt_never_exceeds_its_cap(self):
        body = "The Committee judges that the risks to inflation are elevated. " * 4000
        for cap in (600, 5_000, 14_000):
            self.assertLessEqual(len(ts.salient_excerpt(body, cap)), cap)

    def test_excerpt_keeps_policy_sentences_over_filler(self):
        body = (
            "Cover page. " * 50
            + "Table of contents. " * 400
            + "The Committee decided to raise the target range for the federal funds rate. "
            + "Filler about the weather. " * 400
        )
        out = ts.salient_excerpt(body, 2_000)
        self.assertIn("target range", out)

    def test_total_budget_is_respected_on_the_widest_shipped_unit(self):
        unit = pathlib.Path(__file__).resolve().parent.parent / "units" / "t2-F1-sahm-watch-2024"
        if not unit.is_dir():
            self.skipTest("units/ not present")
        docs, _, _ = ts.read_corpus(unit / "text", "2024-12-31")
        self.assertLessEqual(sum(len(d["text"]) for d in docs), ts._TOTAL_CHAR_BUDGET)
        self.assertLessEqual(len(docs), ts._MAX_DOCS)


class ModelReplies(unittest.TestCase):
    """Everything a model can send that must not reach the parquet."""

    CTX = {
        "sigma": {"UST_2Y": 0.5},
        "level": {"UST_2Y": 4.35},
        "horizons": [21],
        "value_unit": "percent_per_annum",
        "sigma_horizon": 21,
        "target_type": "level",
        "asof": "2024-12-18",
        "title": "",
    }

    def test_non_finite_is_dropped(self):
        # json.loads accepts the bare literals NaN and Infinity. One of them unguarded produces an
        # all-NaN parquet that g3_domain_semantics refuses -- an inadmissible submission, not a
        # bad one.
        raw = json.loads('{"UST_2Y": {"drift_sd": NaN, "vol_scale": Infinity, "skew": 0.0}}')
        adj, led = ts.to_adjustments(raw, ["UST_2Y"], self.CTX, "llm")
        self.assertEqual(adj["UST_2Y"], ts.NEUTRAL)
        self.assertIn("non-finite", led["UST_2Y"]["note"])

    def test_absurd_drift_is_clamped(self):
        raw = {"UST_2Y": {"drift_sd": 1e12, "vol_scale": 50.0, "skew": 99.0}}
        adj, led = ts.to_adjustments(raw, ["UST_2Y"], self.CTX, "llm")
        self.assertEqual(adj["UST_2Y"]["shift"], ts._DRIFT_SD_CLAMP * 0.5)
        self.assertEqual(adj["UST_2Y"]["widen"], ts._WIDEN_CLAMP[1])
        self.assertEqual(adj["UST_2Y"]["skew"], ts._SKEW_CLAMP[1])
        self.assertIn("clamped", led["UST_2Y"]["note"])

    def test_strings_and_nulls_are_dropped(self):
        raw = {"UST_2Y": {"drift_sd": "up a lot", "vol_scale": None, "skew": 0.0}}
        adj, _ = ts.to_adjustments(raw, ["UST_2Y"], self.CTX, "llm")
        self.assertEqual(adj["UST_2Y"], ts.NEUTRAL)

    def test_unnamed_asset_is_neutral_not_an_error(self):
        adj, led = ts.to_adjustments({"EUR": {"drift_sd": 1.0}}, ["UST_2Y"], self.CTX, "llm")
        self.assertEqual(adj["UST_2Y"], ts.NEUTRAL)
        self.assertIn("no entry", led["UST_2Y"]["note"])

    def test_no_sigma_forces_shift_to_zero(self):
        # `shift` is additive in the asset's own units. With no panel there is no scale, and a
        # number in unknown units is worse than no number.
        ctx = {**self.CTX, "sigma": {}, "level": {}}
        adj, led = ts.to_adjustments({"UST_2Y": {"drift_sd": 1.0}}, ["UST_2Y"], ctx, "llm")
        self.assertEqual(adj["UST_2Y"]["shift"], 0.0)
        self.assertIn("no sigma", led["UST_2Y"]["note"])

    def test_widen_survives_when_shift_cannot(self):
        ctx = {**self.CTX, "sigma": {}, "level": {}}
        adj, _ = ts.to_adjustments({"UST_2Y": {"vol_scale": 1.6}}, ["UST_2Y"], ctx, "llm")
        self.assertEqual(adj["UST_2Y"]["widen"], 1.6)

    def test_unreachable_endpoint_degrades_not_raises(self):
        env = dict(os.environ)
        os.environ["MODEL_ENDPOINT"] = "http://127.0.0.1:1/v1"
        os.environ["MODEL_NAME"] = "nobody"
        os.environ["MODEL_TIMEOUT"] = "2"
        try:
            with tempfile.TemporaryDirectory() as d:
                text = _unit(
                    pathlib.Path(d),
                    [("s", "2024-12-18", "fomc_statement", "The Committee is patient. " * 30)],
                )
                adj = ts.read_text_signal(text, ["UST_2Y"])
            self.assertEqual(set(adj["UST_2Y"]), KEYS)
            self.assertEqual(ts.LAST_LEDGER["source"], "heuristic")
            self.assertTrue(ts.LAST_LEDGER["llm_skipped_reason"])
        finally:
            os.environ.clear()
            os.environ.update(env)


class Heuristic(unittest.TestCase):
    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            text = _unit(
                pathlib.Path(d),
                [("s", "2024-12-18", "fomc_statement", "extent and timing of additional " * 20)],
            )
            a = ts.read_text_signal(text, ["UST_2Y"])
            b = ts.read_text_signal(text, ["UST_2Y"])
        self.assertEqual(a, b)

    def test_guidance_outweighs_the_action_already_taken(self):
        """A cut delivered with a shallower path must not read as dovish.

        This is the t2-F1-hawkish-cut-2024 failure in miniature: weighting the decision verb at
        full strength made the floor read that card 8.0 dovish to 4.6 hawkish and push the 2Y the
        wrong way. The action is in the last panel level already; the path is what is left to
        forecast.
        """
        cut_with_hawkish_path = (
            "The Committee decided to lower the target range by 1/4 percentage point. "
            "In considering the extent and timing of additional adjustments, the Committee will "
            "carefully assess incoming data. Inflation remains somewhat elevated and the median "
            "projection shows fewer cuts than previously indicated."
        )
        with tempfile.TemporaryDirectory() as d:
            text = _unit(
                pathlib.Path(d), [("s", "2024-12-18", "fomc_statement", cut_with_hawkish_path)]
            )
            ts.read_text_signal(text, ["UST_2Y"])
        # Asserted on drift_sd, not shift: this synthetic unit ships no panel, so there is no
        # sigma to scale by and the guard above correctly forces `shift` to 0. The lexicon's
        # verdict is the thing under test.
        drift = ts.LAST_LEDGER["assets"]["UST_2Y"]["drift_sd"]
        self.assertGreater(drift, 0.0, f"a hawkish cut must raise the 2Y centre, got {drift}")

    def test_off_mode_is_exactly_neutral(self):
        env = dict(os.environ)
        os.environ["TEXT_SIGNAL_MODE"] = "off"
        try:
            with tempfile.TemporaryDirectory() as d:
                text = _unit(
                    pathlib.Path(d), [("s", "2024-12-18", "fomc_statement", "hawkish " * 50)]
                )
                adj = ts.read_text_signal(text, ["UST_2Y"])
            # The glue's `used_text` check compares against this dict by equality.
            self.assertEqual(adj["UST_2Y"], {"shift": 0.0, "widen": 1.0, "skew": 0.0})
        finally:
            os.environ.clear()
            os.environ.update(env)


if __name__ == "__main__":
    unittest.main(verbosity=2)
