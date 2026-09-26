"""Pun's eval harness: the text-signal-issue extraction used to surface silent fallbacks.

Plain `unittest`, no pytest and no `qfbench2_common`:

    python3 -m unittest tests.test_run_eval -v
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import run_eval as re  # noqa: E402


class TestExtractTextSignalIssues(unittest.TestCase):
    def test_success_line_alone_is_not_an_issue(self):
        stderr = "[text_signal] source=llm family=F1 docs=5 assets=['CAD']\n"
        self.assertEqual(re._extract_text_signal_issues(stderr), [])

    def test_neutral_fallback_is_an_issue(self):
        stderr = "[text_signal] adjustment call failed: HTTP 429: Too Many Requests; neutral\n"
        self.assertEqual(
            re._extract_text_signal_issues(stderr),
            ["[text_signal] adjustment call failed: HTTP 429: Too Many Requests; neutral"],
        )

    def test_per_doc_failure_is_an_issue(self):
        stderr = "[text_signal] doc fomc-2024-01 (fomc_statement) failed: HTTP 429: ...\n"
        self.assertEqual(len(re._extract_text_signal_issues(stderr)), 1)

    def test_no_usable_summaries_is_an_issue(self):
        stderr = "[text_signal] no usable summaries; neutral\n"
        self.assertEqual(len(re._extract_text_signal_issues(stderr)), 1)

    def test_unrelated_stderr_lines_are_ignored(self):
        stderr = "some other warning\ntraceback line\n"
        self.assertEqual(re._extract_text_signal_issues(stderr), [])

    def test_mixed_stderr_keeps_only_the_problem_lines_in_order(self):
        stderr = (
            "[text_signal] doc a failed: timeout\n"
            "[text_signal] source=llm family=F2 docs=1 assets=['UST_2Y']\n"
            "[text_signal] doc b failed: HTTP 500\n"
        )
        self.assertEqual(
            re._extract_text_signal_issues(stderr),
            ["[text_signal] doc a failed: timeout", "[text_signal] doc b failed: HTTP 500"],
        )

    def test_empty_stderr_gives_no_issues(self):
        self.assertEqual(re._extract_text_signal_issues(""), [])


class TestRunOneTiming(unittest.TestCase):
    """run_one() wraps _run_one() purely to stamp elapsed_seconds -- check that stamp lands on
    every status branch (_run_one's own logic is exercised end-to-end elsewhere; a full sweep
    depends on every unit's timing being present regardless of how it turned out, since a slow
    agent_crashed or scorer_crashed unit is exactly the kind of thing worth noticing)."""

    def test_elapsed_seconds_is_added_on_success(self):
        with mock.patch.object(re, "_run_one", return_value={"unit_id": "x", "status": "scored"}):
            result = re.run_one(pathlib.Path("/fake"), False)
        self.assertIn("elapsed_seconds", result)
        self.assertIsInstance(result["elapsed_seconds"], float)
        self.assertGreaterEqual(result["elapsed_seconds"], 0.0)

    def test_elapsed_seconds_is_added_even_when_the_unit_crashed(self):
        with mock.patch.object(re, "_run_one",
                                return_value={"unit_id": "x", "status": "agent_crashed",
                                              "detail": "boom", "text_signal_issues": []}):
            result = re.run_one(pathlib.Path("/fake"), False)
        self.assertIn("elapsed_seconds", result)

    def test_elapsed_seconds_reflects_a_slow_call(self):
        clock = {"t": 0.0}

        def fake_run_one(unit_dir, gates_only):
            clock["t"] += 5.0
            return {"unit_id": "x", "status": "gates_only"}

        with mock.patch.object(re, "_run_one", side_effect=fake_run_one), \
             mock.patch.object(re.time, "perf_counter", side_effect=lambda: clock["t"]):
            result = re.run_one(pathlib.Path("/fake"), False)
        self.assertEqual(result["elapsed_seconds"], 5.0)


class TestLimit(unittest.TestCase):
    """--limit N: a quick, real (but small) report without paying for a full sweep."""

    def test_limit_runs_only_the_first_n_units_in_sorted_order(self):
        fake_dirs = [pathlib.Path(f"/fake/unit-{i}") for i in range(5)]
        seen = []

        def fake_run_one(unit_dir, gates_only):
            seen.append(unit_dir.name)
            return {"unit_id": unit_dir.name, "status": "gates_only", "text_signal_issues": [],
                    "elapsed_seconds": 0.0}

        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            with mock.patch.object(re, "_iter_unit_dirs", return_value=fake_dirs), \
                 mock.patch.object(re, "run_one", side_effect=fake_run_one):
                rc = re.main(["--limit", "2", "--out", str(out)])
            self.assertEqual(rc, 0)
            report = json.loads(out.read_text())
        self.assertEqual(seen, ["unit-0", "unit-1"])
        self.assertEqual(report["total_units"], 2)

    def test_no_limit_runs_every_unit(self):
        fake_dirs = [pathlib.Path(f"/fake/unit-{i}") for i in range(3)]

        def fake_run_one(unit_dir, gates_only):
            return {"unit_id": unit_dir.name, "status": "gates_only", "text_signal_issues": [],
                    "elapsed_seconds": 0.0}

        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "report.json"
            with mock.patch.object(re, "_iter_unit_dirs", return_value=fake_dirs), \
                 mock.patch.object(re, "run_one", side_effect=fake_run_one):
                re.main(["--out", str(out)])
            report = json.loads(out.read_text())
        self.assertEqual(report["total_units"], 3)


if __name__ == "__main__":
    unittest.main()


class TestLedgerLinesAreNotIssues(unittest.TestCase):
    """The per-asset ledger shares the `[text_signal]` prefix but reports success, not failure.

    Without this distinction every healthy unit would be counted as broken, which would destroy
    the one signal that separates a real sweep from a silently-neutral one -- the exact failure
    PUN_TEXT_NOTES.md documents.
    """

    def test_per_asset_ledger_rows_are_not_issues(self):
        stderr = (
            "[text_signal] source=llm family=F3 docs=8 requests=9/25 assets=['UST_2Y', 'EUR']\n"
            "[text_signal] adj UST_2Y: drift_sd=+0.800 shift=+0.240000 widen=1.250 skew=+0.000"
            " | vol_scale 1.90 clamped | 75bp hike, more to come\n"
            "[text_signal] adj EUR: drift_sd=-0.500 shift=-0.010000 widen=1.000 skew=+0.000\n"
        )
        self.assertEqual(re._extract_text_signal_issues(stderr), [])

    def test_a_partial_reply_IS_an_issue(self):
        """Assets the model omitted are left at exact neutral. On a joint card that is itself a
        claim about them, so it belongs in the issue list -- hence its own prefix, not `adj `."""
        stderr = (
            "[text_signal] source=llm family=F3 docs=8 requests=9/25 assets=['A', 'B']\n"
            "[text_signal] adj A: drift_sd=+0.500 shift=+0.100000 widen=1.000 skew=+0.000\n"
            "[text_signal] partial reply: 1 of 2 assets missing, left neutral: ['B']\n"
        )
        issues = re._extract_text_signal_issues(stderr)
        self.assertEqual(len(issues), 1)
        self.assertIn("partial reply", issues[0])

    def test_a_real_failure_still_surfaces_beside_ledger_rows(self):
        stderr = (
            "[text_signal] adj A: drift_sd=+0.500 shift=+0.100000 widen=1.000 skew=+0.000\n"
            "[text_signal] doc beige_book-2024-04-30 (beige_book) failed: HTTP 503\n"
        )
        issues = re._extract_text_signal_issues(stderr)
        self.assertEqual(len(issues), 1)
        self.assertIn("HTTP 503", issues[0])


class TestHelpRenders(unittest.TestCase):
    def test_help_does_not_crash_on_a_literal_percent(self):
        """argparse %-formats help strings, so a literal `67%` in help text raises TypeError.

        Cheap to write, and it caught exactly that.
        """
        parser_ok = True
        try:
            with mock.patch("sys.stdout", new=io.StringIO()):
                try:
                    re.main(["--help"])
                except SystemExit:
                    pass
        except TypeError:
            parser_ok = False
        self.assertTrue(parser_ok, "--help raised; check for an unescaped % in a help string")
