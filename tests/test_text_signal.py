"""Stage 1 of the text half: per-document summary agents.

Plain `unittest`, no pytest and no `qfbench2_common`, so it runs in a bare checkout:

    python3 -m unittest tests.test_text_signal -v

The model call is monkeypatched; nothing here touches the network.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import tempfile
import threading
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
        # Checks the OR-combination itself (_THINKING flag or a doc-type carve-out), not a
        # hardcoded literal -- _THINKING is a live experiment (see its own comment) and this
        # test should stay meaningful whichever way it's currently set.
        self.assertEqual(call.call_args.kwargs["thinking"], ts._THINKING or "beige_book" in ts._THINKING_TYPES)
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


class TestComputedCot(unittest.TestCase):
    TABLE = ("CFTC Commitments of Traders\nMarket: JAPANESE YEN (CME). last report 2007-07-17\n"
             "report_date  open_interest  noncomm_long  noncomm_short  noncomm_net\n"
             " 2007-07-03        326811         43373         198511      -155138\n"
             " 2007-07-10        320333         51549         177439      -125890\n"
             " 2007-07-17        303595         50431         177204      -126773\n")

    def test_direction_of_a_deepening_net_short_is_right(self):
        out = ts.cot_summary(self.TABLE)
        self.assertIn("net short deepened by 883", out)          # the model wrote "a decrease of 883"
        self.assertIn("-126,773", out)
        self.assertIn("down 16,738 from 320,333", out)            # open interest, week over week

    def test_positioning_report_never_calls_the_model(self):
        doc = {"doc_id": "c", "timestamp": "2007-07-17", "doc_type": "positioning_report", "text": self.TABLE}
        with mock.patch.object(ts, "call_model") as call:
            out = ts.summarize_doc(doc)
        call.assert_not_called()
        self.assertTrue(out["summarized"] and out["computed"])

    def test_unparseable_table_falls_back_to_the_model(self):
        doc = {"doc_id": "c", "timestamp": "2007-07-17", "doc_type": "positioning_report", "text": "no rows here " * 300}
        with mock.patch.object(ts, "call_model", return_value=("- nothing", "")) as call:
            out = ts.summarize_doc(doc)
        call.assert_called_once()
        self.assertEqual(out["summary"], "- nothing")


class TestRequestBudget(unittest.TestCase):
    """House rule: 25 admitted requests per unit, retries and failures included."""

    def _http_error(self, code):
        import urllib.error
        return urllib.error.HTTPError("u", code, "err", {}, None)

    def test_reserve_is_kept_for_stage_two(self):
        b = ts._Budget(total=3, reserve=1)
        self.assertTrue(b.take()); self.assertTrue(b.take())
        self.assertFalse(b.take())              # stage 1 may not touch the reserve
        self.assertTrue(b.take(reserved=True))  # stage 2 may
        self.assertFalse(b.take(reserved=True))
        self.assertEqual(b.spent, 3)

    def test_every_retry_spends_a_slot_and_stops_when_out(self):
        budget = ts._Budget(total=2, reserve=0)
        env = {"MODEL_ENDPOINT": "https://h/v1", "MODEL_TOKEN": "t"}
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(ts.urllib.request, "urlopen", side_effect=self._http_error(503)) as open_, \
             mock.patch.object(ts.time, "sleep"):
            out, err = ts.call_model("s", "u", budget=budget)
        self.assertIsNone(out)
        self.assertEqual(open_.call_count, 2)   # not the 4 attempts the retry table allows
        self.assertEqual(budget.left, 0)
        self.assertIn("budget", err)

    def test_corpus_stops_sending_when_the_budget_is_gone(self):
        with tempfile.TemporaryDirectory() as d:
            text = _unit(pathlib.Path(d), [(f"m{i}", f"2024-01-0{i}", "fomc_minutes", "y" * 10_000) for i in range(1, 5)])
            budget = ts._Budget(total=3, reserve=1)  # room for 2 documents
            env = {"MODEL_ENDPOINT": "https://h/v1", "MODEL_TOKEN": "t"}
            reply = mock.MagicMock()
            reply.__enter__.return_value.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "- held rates"}}]}).encode()
            with mock.patch.dict(os.environ, env), \
                 mock.patch.object(ts.urllib.request, "urlopen", return_value=reply) as open_, \
                 mock.patch.object(ts, "_WORKERS", 1):
                out = ts.summarize_corpus(text, budget)
        self.assertEqual(open_.call_count, 2)
        self.assertEqual(sum(1 for o in out if o["summary"]), 2)
        self.assertEqual([o["doc_id"] for o in out if o["summary"]], ["m4", "m3"])  # newest first win
        self.assertTrue(all("budget" in o["error"] for o in out if not o["summary"]))
        self.assertEqual(budget.left, 1)  # the stage-2 slot is untouched


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
        # Checks it's wired to the named flag, not a hardcoded literal -- _STAGE2_THINKING is a
        # live experiment (see its own comment) and this test should stay meaningful either way.
        self.assertEqual(call.call_args.kwargs["thinking"], ts._STAGE2_THINKING)

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


class TestThrottle(unittest.TestCase):
    """_throttle() is the shared gate in front of every call_model() attempt -- Stage 1's worker
    pool and Stage 2's own call all draw from the same budget. See PUN_TEXT_NOTES.md, "silent
    rate-limit fallback": an unthrottled burst can exhaust the House's 40 rpm quota, and a 429
    that survives all retries silently degrades the whole card to NEUTRAL with no visible error.
    """

    def setUp(self):
        ts._request_times.clear()
        self._orig_limit = ts._RATE_LIMIT_RPM

    def tearDown(self):
        ts._RATE_LIMIT_RPM = self._orig_limit
        ts._request_times.clear()

    def test_admits_up_to_the_limit_without_sleeping(self):
        ts._RATE_LIMIT_RPM = 3
        with mock.patch.object(ts.time, "monotonic", return_value=0.0), \
             mock.patch.object(ts.time, "sleep") as sleep:
            for _ in range(3):
                ts._throttle()
        sleep.assert_not_called()
        self.assertEqual(len(ts._request_times), 3)

    def test_blocks_until_the_window_clears_once_the_limit_is_hit(self):
        ts._RATE_LIMIT_RPM = 2
        clock = {"t": 0.0}

        with mock.patch.object(ts.time, "monotonic", side_effect=lambda: clock["t"]), \
             mock.patch.object(ts.time, "sleep",
                                side_effect=lambda s: clock.__setitem__("t", clock["t"] + s)) as sleep:
            ts._throttle()  # t=0, 1st of 2 admitted
            ts._throttle()  # t=0, 2nd of 2 admitted -- limit now reached
            ts._throttle()  # over the limit: must wait for the window to clear
        sleep.assert_called_once()
        self.assertEqual(clock["t"], 60.0)
        self.assertEqual(len(ts._request_times), 1)  # only the 3rd call's timestamp remains

    def test_concurrent_callers_never_jointly_exceed_the_limit_in_one_window(self):
        ts._RATE_LIMIT_RPM = 4
        errors: list[BaseException] = []

        def run():
            try:
                ts._throttle()
            except BaseException as exc:  # noqa: BLE001 -- surfacing it is the point
                errors.append(exc)

        with mock.patch.object(ts.time, "monotonic", return_value=0.0), \
             mock.patch.object(ts.time, "sleep",
                                side_effect=AssertionError("exactly at the limit; must not block")):
            threads = [threading.Thread(target=run) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(ts._request_times), 4)


if __name__ == "__main__":
    unittest.main()


class TestF3CrossAssetPrompt(unittest.TestCase):
    """Stage 2's F3 handling: the joint framing, the cross-asset numbers, and the tighter clamp.

    Nothing in this file before these tests called build_adjustment_prompt with more than one
    asset, so the multi-asset rendering -- which is the entire F3 case -- was unpinned.
    """

    ASSETS = ["UST_2Y", "EUR", "GBP", "JPY"]

    def _ctx(self, family="F3"):
        return {
            "asof": "2022-06-15", "horizons": [21, 63], "value_unit": "mixed",
            "target_type": "level", "family": family, "sigma_horizon": 21,
            "level": {a: 1.0 for a in self.ASSETS},
            "sigma": {a: 0.1 for a in self.ASSETS},
            "cross": {
                "corr": {a: {b: (1.0 if a == b else 0.5) for b in self.ASSETS}
                         for a in self.ASSETS},
                "trailing_sigma": {a: {21: 1.2, 63: -0.4} for a in self.ASSETS},
            },
        }

    def _prompt(self, family="F3"):
        return ts.build_adjustment_prompt(
            [{"doc_id": "d1", "timestamp": "2022-06-15", "doc_type": "fomc_statement",
              "summary": "- hiked 75bp"}], self.ASSETS, self._ctx(family))

    def test_every_asset_appears_in_the_reply_schema(self):
        """The skeleton used to show assets[:2] beside a 10-asset list -- and got 2 keys back."""
        system, _ = self._prompt()
        for a in self.ASSETS:
            self.assertIn(f'"{a}"', system, f"{a} missing from the reply schema")

    def test_f3_focus_does_not_claim_correlation_is_unexpressible(self):
        """The old F3 paragraph ended by apologising that the format had no correlation field.

        That is false in the way that matters: the variogram scores |asset_i - asset_j|, so the
        per-asset drift_sd values ARE a joint statement. Measured, per-asset drift is worth ~0.75
        normalized composite against ~0.99 for an explicit correlation control, so the apology was
        steering the model away from its strongest lever.
        """
        system, _ = self._prompt()
        self.assertNotIn("no field for a target correlation", system)
        self.assertIn("DIFFERENCES between your assets", system)

    def test_cross_asset_numbers_reach_the_user_message(self):
        """Correlations and trailing moves -- the model had neither before."""
        _, user = self._prompt()
        self.assertIn("moved together", user)
        self.assertIn("Trailing move already realized", user)
        self.assertIn("EUR +0.50", user)          # a rendered pairwise correlation

    def test_asset_glossary_states_the_fx_direction(self):
        """A bare `JPY` does not say which way the quote runs; the card's value_unit often can't
        either, because it is one field shared by every asset on the card."""
        _, user = self._prompt()
        self.assertIn("RISES when the dollar strengthens", user)   # JPY, ccy-per-USD
        self.assertIn("RISES when the dollar weakens", user)       # EUR/GBP, USD-per-ccy

    def test_f3_widen_clamp_is_tighter_and_is_enforced(self):
        """F3 pays 0.3 on a term that gets monotonically worse as the draws widen.

        The default ceiling of 2.00 lets the model cost itself ~46% on its primary term; measured,
        the normalized composite goes 0.978 at widen 1.00 to 1.056 at 1.30.
        """
        self.assertEqual(ts._widen_clamp("F3"), (0.85, 1.25))
        self.assertEqual(ts._widen_clamp("F1"), ts._WIDEN_CLAMP)
        self.assertEqual(ts._widen_clamp(None), ts._WIDEN_CLAMP)

        ctx = {"sigma": {"EUR": 0.02}, "family": "F3"}
        adj, _ = ts.to_adjustments({"assets": {"EUR": {"drift_sd": 0.0, "vol_scale": 1.9}}},
                                   ["EUR"], ctx)
        self.assertEqual(adj["EUR"]["widen"], 1.25)
        # the same reply on a non-F3 card keeps the wide ceiling
        ctx_f4 = {"sigma": {"EUR": 0.02}, "family": "F4"}
        adj4, _ = ts.to_adjustments({"assets": {"EUR": {"drift_sd": 0.0, "vol_scale": 1.9}}},
                                    ["EUR"], ctx_f4)
        self.assertEqual(adj4["EUR"]["widen"], 1.9)

    def test_prompt_states_the_clamp_it_actually_enforces(self):
        """The stated range and the enforced range come from one resolver, so they cannot drift."""
        system_f3, _ = self._prompt("F3")
        self.assertIn("[0.85, 1.25]", system_f3)
        system_f1, _ = self._prompt("F1")
        self.assertIn(f"[{ts._WIDEN_CLAMP[0]}, {ts._WIDEN_CLAMP[1]}]", system_f1)


class TestLedgerLogging(unittest.TestCase):
    def test_missing_asset_is_reported_not_silently_neutralized(self):
        """An omitted asset is left at exact neutral, which on a joint card is itself a claim.

        It still gets a ledger row, so it has to be detected by its note rather than by absence.
        """
        ctx = {"sigma": {"A": 1.0, "B": 1.0}, "family": "F3"}
        _, ledger = ts.to_adjustments({"assets": {"A": {"drift_sd": 0.5}}}, ["A", "B"], ctx)
        with mock.patch("sys.stderr", new=io.StringIO()) as err:
            ts._log_ledger(ledger, ["A", "B"], ctx)
            out = err.getvalue()
        self.assertIn("adj A:", out)
        self.assertIn("MISSING from the reply", out)
        self.assertIn("'B'", out)

    def test_logging_never_raises(self):
        """A logging failure must not cost a card its text signal."""
        with mock.patch("sys.stderr", new=io.StringIO()):
            ts._log_ledger({"A": "not-a-dict"}, ["A"], {})
            ts._log_ledger({}, ["A"], {})
