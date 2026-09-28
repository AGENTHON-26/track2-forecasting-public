#!/usr/bin/env python3
"""Remove website furniture from scraped plain-text corpus documents.

One module, one algorithm: the same file is used to clean archived source
documents and the copies of them that ship inside a card's text corpus.
Standard library only.

What it removes, and when
-------------------------
Every rule is written out below; nothing is done by hand.

1. Line endings are normalised to LF (CRLF first, then any lone CR).
2. A text holding two or more U+FEFF marks is a concatenation of web pages
   (a fetcher joined several pages into one file).  It is split on U+FEFF,
   every page is cleaned on its own, and the non-empty pages are rejoined
   with one blank line.  A page footer can therefore never cut off the pages
   that follow it.
3. Navigation cut, only on a page whose first 8,000 characters carry a
   site-navigation marker (NAV):
   - head: the page starts at the earliest HEAD marker found in its first
     40,000 characters; a page with no HEAD marker is left alone;
   - Beige Book front matter: on a single page, or on the first page of a
     multi-page scrape, the start moves back to the last standalone
     "Beige Book[ - <date>]" title line in the 3,000 characters before that
     marker (or to the bare release-date line directly above the title).
     Between there and the marker only whole lines on the closed
     BB_FRONT_DROP lists are deleted,
     so the title, release date, "National Summary" and the
     "This report was prepared at the Federal Reserve Bank of ..." preamble
     are kept.  Later pages of a multi-page scrape start at their District
     heading;
   - tail: the page ends at the first TAIL marker more than 2,000
     characters after the head marker.
4. Footers, on every page:
   - FOOTER_TAIL: the earliest marker in the last 40 % of the page that has
     a FOOTER_WORDS match within the next 4,000 characters.  If the first
     line of the cut text, after an optional "Return to top" / "Back to top"
     line, is a bare date ("February 18, 2009"), that line is the page's
     dateline and is kept: the older Federal Reserve layout prints it
     between the last "Return to top" and the site links;
   - older Federal Reserve layout: the final "Return to top" line when a
     "Last update:" line follows it;
   - FOMC breadcrumb: the final "Return to top" line when nothing but
     "FOMC", "|" and "Monetary policy" lines follows it.
5. Inline separators: each remaining "Return to top" / "Back to top" line
   is deleted together with one blank line after it.
6. Related-content blocks appended after the document: an ECB
   "SEE ALSO / Find out more about related content" block, and a Bank of
   England related-news block ("Monetary Policy Committee voting history"
   followed by "Other Monetary Policy Committee news", or an "Other news"
   line followed by "News // " items).  These lists are filled at retrieval
   time and can name events after the document's own date.
7. Trailing BLS page label: when the document ends with a bare
   "Last Modified Date:" line (and its date, on the same or the next line)
   and nothing else, those lines are removed.  Only at the very end of the
   document; with site links after it the footer rule (4) applies instead.
8. BLS reissue notes: in a file named cpi_YYYY-MM-DD.txt or
   empsit_YYYY-MM-DD.txt (the release date), two forms of reissue text
   whose date is later than the release date are removed:
   - a parenthesised "(NOTE: This release was reissued on <date> ...)"
     paragraph;
   - a numbered table-footnote line "<n> ... reissued on <date>." (the
     footnote marker <n> then has no footnote text).
   Each removal is reported (see --log).  So are the reissue mentions the
   rule keeps: a post-dated mention in any other form (a post-dated line)
   and a NOTE that carries no date (an undated note).  Same-day notes are
   kept without a report.  The rest of the text is not changed: a reissued
   release keeps the reissued tables.

Guards
------
- Tail guard: a tail cut (rules 3, 4, 6, 7) that would remove text containing
  a body or page-start marker (TAIL_FORBIDDEN) is refused.
- Shrink guard: prose(t) counts the characters on lines that are at least
  60 characters long once stripped (navigation lines are short).  A file
  whose prose would fall below half of what it was is refused, unless its
  name is on ALLOWLIST.  On a multi-page scrape every page must also keep at
  least 500 prose characters.
- A refused file raises ShrinkGuardError from clean(); the driver leaves it
  untouched and exits 2.

Output
------
If no rule changes the text, clean() returns its input unchanged (CRLF and
missing final newlines included).  Otherwise it returns the result stripped,
with LF line endings and one final newline.  Cleaning cleaned output changes
nothing.

Command line
------------
    declutter.py [--check] [--log PATH] [--report PATH] PATH [PATH ...]

PATH is a file or a directory (searched recursively for *.txt, in sorted
order).  Without --check changed files are rewritten in place (UTF-8, LF).
--check writes nothing at all (no document, no log, no report) and exits 1
if any file would change; combining it with --log or --report is a usage
error.  --log writes the reissue-note report as a sorted TSV; --report writes
one TSV row per file (chars and prose before and after, rules fired).  File
names in both are relative to the directory argument they were found under
(the file name itself for a file argument).

Exit status: 0 nothing (left) to change, 1 --check found changes, 2 at
least one file refused, 3 usage or read error.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

__all__ = ["clean", "prose", "ShrinkGuardError", "ALLOWLIST", "main"]

# File names exempt from the shrink guard.  Expected to stay empty: a file
# that needs an entry here needs a better rule instead.
ALLOWLIST: frozenset[str] = frozenset()

SHRINK_MIN_KEEP = 0.5       # keep at least this share of prose characters
PAGE_MIN_PROSE = 500        # per page of a multi-page scrape
PROSE_MIN_LINE = 60         # a line this long (stripped) counts as prose
NAV_WINDOW = 8000
HEAD_WINDOW = 40000
TAIL_MIN_GAP = 2000
BB_FRONT_WINDOW = 3000      # a Beige Book title this close before the body marker
FOOTER_ZONE = 0.6           # footer markers are searched in the last 40 %
FOOTER_LOOKAHEAD = 4000


class ShrinkGuardError(Exception):
    """Cleaning would remove real document text; the file must stay as is."""


# --- rule 3: navigation --------------------------------------------------

NAV = re.compile(r"(Skip to Content|skip to main navigation|An official website|"
                 r"What's New · What's Next|Monetary Policy Principles and Practice)", re.I)

DISTRICT_HEADING = re.compile(
    r"(?m)^[ \t]*Federal Reserve Bank of [A-Z][A-Za-z. ]+[ \t]*\n\s*Summary of Economic Activity")

# Head markers; the earliest match wins.
HEAD = [
    ("bls-embargo", re.compile(r"Transmission of material in this (news )?release\s+is\s+embargoed")),
    ("bls-technical-information", re.compile(r"(?im)^[ \t]*(FOR )?TECHNICAL INFORMATION:")),
    ("bls-title", re.compile(r"\bTHE EMPLOYMENT SITUATION\b")),
    ("bls-title", re.compile(r"\bCONSUMER PRICE INDEX\b")),
    ("bea-title", re.compile(r"\bGROSS DOMESTIC PRODUCT\b")),
    ("bea-title", re.compile(r"\bREAL GROSS DOMESTIC PRODUCT\b")),
    ("bea-title", re.compile(r"\bPERSONAL INCOME AND OUTLAYS\b")),
    ("beige-book", re.compile(r"\bOverall Economic Activity\b")),
    ("beige-book", re.compile(r"\bSummary of Commentary on Current Economic Conditions\b")),
    ("beige-book", re.compile(r"Beige Book:? National Summary")),
    ("beige-book", DISTRICT_HEADING),
    ("fomc-minutes", re.compile(r"Minutes of the Federal Open Market Committee")),
]

TAIL = [
    re.compile(r"\nLast Modified Date"),
    re.compile(r"\bBack to Top\b(?!.*\bBack to Top\b)", re.S),
    re.compile(r"Board of Governors of the Federal Reserve System\s*\n\s*20th Street"),
    re.compile(r"\bConnect With BLS\b"),
    re.compile(r"\bU\.S\. Bureau of Labor Statistics\s*\|\s*"),
]

# Text that must never be inside a tail cut: body headings and the first
# line of a scraped page.
TAIL_FORBIDDEN = re.compile(
    r"Summary of Economic Activity|Overall Economic Activity|Skip to main content")

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
_MONTH_RX = "|".join(MONTHS)

BB_TITLE = re.compile(
    r"(?m)^[ \t]*Beige Book(?: -{1,2} [A-Z][a-z]+(?: \d{1,2},)? \d{4})?[ \t]*$")
DATE_LINE = re.compile(rf"^[ \t]*(?:{_MONTH_RX}) \d{{1,2}}, \d{{4}}[ \t]*$")
DISTRICTS = frozenset({
    "Boston", "New York", "Philadelphia", "Cleveland", "Richmond", "Atlanta",
    "Chicago", "St. Louis", "Minneapolis", "Kansas City", "Dallas", "San Francisco"})
BB_FRONT_DROP_EXACT = frozenset({
    "PDF", "Print", "Full Report", "Summary", "Full Report National Summary",
    "Full Report Summary"})
BB_FRONT_DROP_BANK = re.compile(r"^Federal Reserve Bank of [A-Z][A-Za-z. ]+$")

# --- rule 4: footers -----------------------------------------------------

FOOTER_TAIL = [
    re.compile(r"\n\s*Last Modified Date", re.I),                                # BLS
    re.compile(r"\n\s*Economic Releases\s*\n", re.I),                            # BLS secondary footer
    re.compile(r"\n\s*(Back|Return) to top\s*\n", re.I),                         # Federal Reserve
    re.compile(r"\n\s*Last Update:\s*\n", re.I),
    re.compile(r"\n\s*Board of Governors of the Federal Reserve System\s*\n\s*About the Fed", re.I),
    re.compile(r"\n\s*U\.S\. Bureau of Labor Statistics\s*\n\s*(Current Population Survey|Postal Square)", re.I),
    re.compile(r"\n\s*Accessibility\s*\n\s*Contact Us\s*\n\s*Disclaimer", re.I),   # older Fed pages
    re.compile(r"\n\s*Home \| ", re.I),
]
FOOTER_WORDS = re.compile(
    r"(Freedom of Information|FOIA|Accessibility Statement|Subscribe to RSS|No Fear Act|"
    r"USA\.gov|Privacy and Security Statement|Linking and Copyright Info|PDF Reader|"
    r"Federal Reserve Facebook Page|Website Policies|Important Website Notices)", re.I)

RETURN_TO_TOP = re.compile(r"(?im)^[ \t]*Return to top[ \t]*$")
LAST_UPDATE_LINE = re.compile(r"(?im)^[ \t]*Last update:")
FOMC_BREADCRUMB = frozenset({"FOMC", "|", "Monetary policy"})

# --- rule 5: inline separators -------------------------------------------

SEPARATOR_LINE = re.compile(r"(?i)^[ \t]*(?:Return|Back) to top[ \t]*$")

# --- rule 6: related-content blocks --------------------------------------

ECB_SEE_ALSO = re.compile(r"[ \t]*\bSEE ALSO\b\s+Find out more about related content")
BOE_MPC_NEWS = re.compile(r"\n[ \t]*Monetary Policy Committee voting history\s*\n")
BOE_OTHER_NEWS = re.compile(r"(?m)^Other news[ \t]*$")

# --- rule 7: trailing BLS page label -------------------------------------

LAST_MODIFIED_END = re.compile(
    rf"\n[ \t]*Last Modified Date:[ \t]*(?:\n[ \t]*)?(?:(?:{_MONTH_RX}) \d{{1,2}}, \d{{4}})?\s*\Z")

# --- rule 8: BLS reissue notes -------------------------------------------

BLS_NAME = re.compile(r"^(?:cpi|empsit)_(\d{4})-(\d{2})-(\d{2})\.txt$")
REISSUE_NOTE = re.compile(
    r"\(NOTE:\s+(?:This (?:news )?release was reissued(?:\s+on)?|BLS reissued this news release on)"
    r"\s+(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+)?"
    rf"({_MONTH_RX})\s+(\d{{1,2}}),\s+(\d{{4}})")
REISSUE_WORD = re.compile(r"(?i)\breissued\b")
REISSUE_FOOTNOTE = re.compile(
    rf"(?m)^[ \t]*\d{{1,2}}[ \t]+[^\n]*?\breissued on[ \t]+({_MONTH_RX})[ \t]+(\d{{1,2}}),[ \t]+(\d{{4}})"
    r"\.?[ \t]*$")
ANY_DATE = re.compile(rf"({_MONTH_RX})\s+(\d{{1,2}}),\s+(\d{{4}})")
NOTE_LINE = re.compile(r"^\(?NOTE:")


# --- helpers -------------------------------------------------------------

def prose(text: str) -> int:
    """Characters on lines that are at least PROSE_MIN_LINE long once stripped."""
    return sum(len(s) for s in (ln.strip() for ln in text.split("\n")) if len(s) >= PROSE_MIN_LINE)


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _check_tail(removed: str, rule: str) -> None:
    m = TAIL_FORBIDDEN.search(removed)
    if m:
        raise ShrinkGuardError(f"{rule}: the text it would cut contains {m.group(0)!r}")


def _date(month: str, day: str, year: str) -> _dt.date:
    return _dt.date(int(year), MONTHS.index(month) + 1, int(day))


def _drop_lines(text: str, is_drop) -> tuple[str, int]:
    """Delete whole lines for which is_drop(stripped_line) is true, each with
    one blank line directly after it (never the last, partial line)."""
    lines = text.split("\n")
    out: list[str] = []
    dropped = 0
    i = 0
    while i < len(lines):
        if is_drop(lines[i].strip()):
            dropped += 1
            i += 1
            if i < len(lines) - 1 and not lines[i].strip():
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), dropped


# --- rule 3 --------------------------------------------------------------

def _bb_front_start(page: str, body: int) -> int:
    """Start of the Beige Book title block before the body marker at `body`."""
    titles = list(BB_TITLE.finditer(page, max(0, body - BB_FRONT_WINDOW), body))
    if not titles:
        return body
    start = titles[-1].start()
    before = page[:start].rstrip("\n").split("\n")
    # nearest preceding non-blank line
    j = len(before) - 1
    while j >= 0 and not before[j].strip():
        j -= 1
    if j >= 0 and DATE_LINE.match(before[j]):
        start = sum(len(ln) + 1 for ln in before[:j])
    return start


def _bb_front_drop(stripped: str) -> bool:
    return (stripped in BB_FRONT_DROP_EXACT
            or stripped.startswith("About This Publication")
            or stripped in DISTRICTS
            or bool(BB_FRONT_DROP_BANK.match(stripped)))


def _nav_cut(page: str, first_page: bool, fired: list[str]) -> str:
    body, kind = None, None
    for name, rx in HEAD:
        m = rx.search(page)
        if m and m.start() < HEAD_WINDOW and (body is None or m.start() < body):
            body, kind = m.start(), name
    if body is None:
        return page
    start = body
    if kind == "beige-book" and first_page:
        start = _bb_front_start(page, body)
    end = len(page)
    for rx in TAIL:
        for m in rx.finditer(page):
            if m.start() > body + TAIL_MIN_GAP:
                end = min(end, m.start())
                break
    _check_tail(page[end:], "navigation tail cut")
    head, n = _drop_lines(page[start:body], _bb_front_drop)
    out = (head + page[body:end]).strip() + "\n"
    if out == page:
        return page
    fired.append(f"nav-head({kind})")
    if n:
        fired.append(f"beige-book-front-matter(-{n} lines)")
    if end < len(page):
        fired.append("nav-tail")
    return out


# --- rule 4 --------------------------------------------------------------

def _footer_cut(page: str, fired: list[str]) -> str:
    n = len(page)
    best = None
    for rx in FOOTER_TAIL:
        for m in rx.finditer(page):
            i = m.start()
            if i < n * FOOTER_ZONE:
                continue
            if not FOOTER_WORDS.search(page, i, i + FOOTER_LOOKAHEAD):
                continue
            best = i if best is None else min(best, i)
    if best is None:
        return page
    _check_tail(page[best:], "footer cut")
    out = page[:best].rstrip() + "\n"
    dateline = _footer_dateline(page[best:])
    if dateline is None:
        fired.append("footer")
        return out
    fired.append("footer(dateline kept)")
    return out + dateline + "\n"


def _footer_dateline(removed: str) -> str | None:
    """The page's dateline at the top of a cut footer: the first line, after
    an optional "Return to top" / "Back to top" line, if it is a bare date."""
    lines = [ln.strip() for ln in removed.split("\n") if ln.strip()]
    if lines and SEPARATOR_LINE.match(lines[0]):
        lines = lines[1:]
    if lines and DATE_LINE.match(lines[0]):
        return lines[0]
    return None


def _older_fed_footer(page: str, fired: list[str]) -> str:
    ms = list(RETURN_TO_TOP.finditer(page))
    if not ms or not LAST_UPDATE_LINE.search(page, ms[-1].end()):
        return page
    _check_tail(page[ms[-1].start():], "older Fed footer cut")
    fired.append("older-fed-footer")
    return page[:ms[-1].start()].rstrip() + "\n"


def _fomc_breadcrumb_footer(page: str, fired: list[str]) -> str:
    ms = list(RETURN_TO_TOP.finditer(page))
    if not ms:
        return page
    rest = [ln.strip() for ln in page[ms[-1].end():].split("\n") if ln.strip()]
    if not rest or not all(ln in FOMC_BREADCRUMB for ln in rest):
        return page
    fired.append("fomc-breadcrumb-footer")
    return page[:ms[-1].start()].rstrip() + "\n"


# --- rule 5 --------------------------------------------------------------

def _drop_separators(page: str, fired: list[str]) -> str:
    out, n = _drop_lines(page, lambda s: bool(SEPARATOR_LINE.match(s)))
    if n:
        fired.append(f"separators(-{n})")
    return out


# --- rule 6 --------------------------------------------------------------

def _related_content_cut(page: str, fired: list[str]) -> str:
    cuts = []
    m = ECB_SEE_ALSO.search(page)
    if m:
        cuts.append((m.start(), "ecb-see-also"))
    for m in BOE_MPC_NEWS.finditer(page):
        if "Other Monetary Policy Committee news" in page[m.end():m.end() + 400]:
            cuts.append((m.start(), "boe-mpc-news"))
            break
    for m in BOE_OTHER_NEWS.finditer(page):
        if "News // " in page[m.end():m.end() + 200]:
            cuts.append((m.start(), "boe-other-news"))
            break
    if not cuts:
        return page
    pos, rule = min(cuts)
    _check_tail(page[pos:], rule)
    fired.append(rule)
    return page[:pos].rstrip() + "\n"


# --- rule 7 --------------------------------------------------------------

def _last_modified_end(text: str, fired: list[str]) -> str:
    m = LAST_MODIFIED_END.search(text)
    if not m:
        return text
    _check_tail(text[m.start():], "trailing Last Modified Date cut")
    fired.append("last-modified-footer")
    return text[:m.start()].rstrip() + "\n"


# --- rule 8 --------------------------------------------------------------

def _matching_paren(text: str, i: int) -> int | None:
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j
    return None


def _reissue_notes(text: str, name: str, fired: list[str], notes: list[tuple]) -> str:
    m = BLS_NAME.match(name)
    if not m:
        return text
    release = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    removed_spans = []
    for nm in REISSUE_NOTE.finditer(text):
        reissue = _date(nm.group(1), nm.group(2), nm.group(3))
        if reissue <= release:
            continue
        close = _matching_paren(text, nm.start())
        line_start = text.rfind("\n", 0, nm.start()) + 1
        line_end = text.find("\n", close) if close is not None else -1
        if line_end == -1:
            line_end = len(text)
        if (close is None or text[line_start:nm.start()].strip()
                or text[close + 1:line_end].strip()):
            notes.append((release.isoformat(), reissue.isoformat(), "kept-unparsed-note",
                          " ".join(text[nm.start():nm.start() + 300].split())))
            continue
        end = line_end + 1 if line_end < len(text) else line_end
        # one immediately following whitespace-only line goes with it
        nxt = text.find("\n", end)
        if end < len(text) and nxt != -1 and not text[end:nxt].strip():
            end = nxt + 1
        removed_spans.append((line_start, end))
        notes.append((release.isoformat(), reissue.isoformat(), "removed",
                      " ".join(text[nm.start():close + 1].split())))
    n_notes = len(removed_spans)
    for fm in REISSUE_FOOTNOTE.finditer(text):
        reissue = _date(fm.group(1), fm.group(2), fm.group(3))
        if reissue <= release or any(a <= fm.start() < b for a, b in removed_spans):
            continue
        end = fm.end() + 1 if fm.end() < len(text) else fm.end()
        # one immediately following whitespace-only line goes with it
        nxt = text.find("\n", end)
        if end < len(text) and nxt != -1 and not text[end:nxt].strip():
            end = nxt + 1
        removed_spans.append((fm.start(), end))
        notes.append((release.isoformat(), reissue.isoformat(), "removed",
                      " ".join(fm.group(0).split())))
    for a, b in sorted(removed_spans, reverse=True):
        text = text[:a] + text[b:]
    if n_notes:
        fired.append(f"reissue-notes(-{n_notes})")
    if len(removed_spans) > n_notes:
        fired.append(f"reissue-footnotes(-{len(removed_spans) - n_notes})")
    # report what the rule keeps: post-dated lines and undated NOTE paragraphs
    for wm in REISSUE_WORD.finditer(text):
        line_start = text.rfind("\n", 0, wm.start()) + 1
        line_end = text.find("\n", wm.end())
        line = text[line_start:line_end if line_end != -1 else len(text)]
        dm = ANY_DATE.search(text, wm.end(), wm.end() + 80)
        if dm:
            when = _date(dm.group(1), dm.group(2), dm.group(3))
            if when > release:
                notes.append((release.isoformat(), when.isoformat(), "kept-postdated-line",
                              " ".join(line.split())))
        elif NOTE_LINE.match(line.strip()):
            para_end = text.find("\n\n", line_start)
            para = text[line_start:para_end if para_end != -1 else len(text)]
            notes.append((release.isoformat(), "", "kept-undated-note", " ".join(para.split())))
    return text


# --- driver-independent entry point --------------------------------------

def clean(text: str, name: str) -> tuple[str, dict]:
    """Clean one document.

    `name` is the file's base name; it selects the BLS reissue rule and the
    shrink-guard allowlist.  Returns (cleaned_text, report).  Raises
    ShrinkGuardError when a guard refuses the file.
    """
    fired: list[str] = []
    notes: list[tuple] = []
    t = _normalise(text)
    if t.count("\ufeff") >= 2:
        raw_pages = [p for p in t.split("\ufeff") if p.strip()]
        multi = True
    else:
        raw_pages = [t]
        multi = False
    pages = []
    for idx, page in enumerate(raw_pages):
        if NAV.search(page[:NAV_WINDOW]):
            page = _nav_cut(page, idx == 0, fired)
        page = _footer_cut(page, fired)
        page = _older_fed_footer(page, fired)
        page = _fomc_breadcrumb_footer(page, fired)
        page = _drop_separators(page, fired)
        page = _related_content_cut(page, fired)
        if multi and prose(page) < PAGE_MIN_PROSE and name not in ALLOWLIST:
            raise ShrinkGuardError(
                f"page {idx + 1} of {len(raw_pages)} would keep {prose(page)} prose characters "
                f"(< {PAGE_MIN_PROSE})")
        pages.append(page)
    if multi:
        fired.insert(0, f"page-split({len(raw_pages)})")
        out = "\n\n".join(p.strip() for p in pages if p.strip())
    else:
        out = pages[0]
    out = _last_modified_end(out, fired)
    out = _reissue_notes(out, name, fired, notes)
    report = {"rules": fired, "notes": notes,
              "chars_before": len(text), "prose_before": prose(t)}
    if not fired:
        report.update(chars_after=len(text), prose_after=report["prose_before"])
        return text, report
    out = out.strip() + "\n"
    report.update(chars_after=len(out), prose_after=prose(out))
    if (report["prose_after"] < SHRINK_MIN_KEEP * report["prose_before"]
            and name not in ALLOWLIST):
        raise ShrinkGuardError(
            f"prose would fall from {report['prose_before']} to {report['prose_after']} "
            f"characters (< {SHRINK_MIN_KEEP:.0%}); rules: {', '.join(fired)}")
    return out, report


# --- command line ---------------------------------------------------------

def _targets(paths: list[str]) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for arg in paths:
        p = Path(arg)
        if p.is_dir():
            for f in sorted(p.rglob("*.txt")):
                if f.is_file():
                    found.append((f, f.relative_to(p).as_posix()))
        elif p.is_file():
            found.append((p, p.name))
        else:
            raise FileNotFoundError(arg)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Remove website navigation, footers and related-content blocks "
                    "from scraped text documents.")
    ap.add_argument("paths", nargs="+", metavar="PATH", help="file or directory (*.txt, recursive)")
    ap.add_argument("--check", action="store_true",
                    help="write nothing at all; exit 1 if any file would change "
                         "(not combinable with --log or --report)")
    ap.add_argument("--log", metavar="PATH", help="write the reissue-note report (TSV)")
    ap.add_argument("--report", metavar="PATH", help="write the per-file report (TSV)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 3
    if args.check and (args.log or args.report):
        print("declutter: --check writes nothing, so it cannot be combined with --log or "
              "--report; run without --check (on a copy) to get the log or the report",
              file=sys.stderr)
        return 3
    try:
        targets = _targets(args.paths)
    except FileNotFoundError as exc:
        print(f"declutter: no such file or directory: {exc}", file=sys.stderr)
        return 3

    rows, note_rows = [], []
    n_changed = n_refused = 0
    for path, label in targets:
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"declutter: cannot read {path}: {exc}", file=sys.stderr)
            return 3
        try:
            out, rep = clean(text, path.name)
        except ShrinkGuardError as exc:
            n_refused += 1
            print(f"REFUSED {label}: {exc}", file=sys.stderr)
            rows.append((label, len(text), len(text), prose(_normalise(text)),
                         prose(_normalise(text)), "refused", str(exc)))
            continue
        changed = out != text
        if changed:
            n_changed += 1
            if args.check:
                print(f"would change {label}: {', '.join(rep['rules'])}")
            else:
                path.write_bytes(out.encode("utf-8"))
        rows.append((label, rep["chars_before"], rep["chars_after"], rep["prose_before"],
                     rep["prose_after"], "changed" if changed else "unchanged",
                     ", ".join(rep["rules"])))
        for release, reissue, action, note in rep["notes"]:
            note_rows.append((label, release, reissue, action, note))

    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("path\tchars_before\tchars_after\tprose_before\tprose_after\tstatus\trules\n")
            for r in rows:
                fh.write("\t".join(str(x) for x in r) + "\n")
    if args.log:
        with open(args.log, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("file\trelease_date\treissue_date\taction\tnote\n")
            for r in sorted(note_rows):
                fh.write("\t".join(r) + "\n")

    verb = "would change" if args.check else "changed"
    print(f"declutter: {len(targets)} files, {n_changed} {verb}, {n_refused} refused, "
          f"{sum(1 for r in note_rows if r[3] == 'removed')} reissue notes removed",
          file=sys.stderr)
    if n_refused:
        return 2
    if args.check and n_changed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
