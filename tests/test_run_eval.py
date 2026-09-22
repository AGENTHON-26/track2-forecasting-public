"""Pun's eval harness: the text-signal-issue extraction used to surface silent fallbacks.

Plain `unittest`, no pytest and no `qfbench2_common`:

    python3 -m unittest tests.test_run_eval -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
