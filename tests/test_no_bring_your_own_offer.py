#!/usr/bin/env python3
"""No published document offers a bring-your-own model. Stdlib-only; runs as a script or pytest.

Ruling of 2026-09-18: bring-your-own models and adapters are not part of this competition. Every
submission is an `api` submission, and the only language model it may call is the House model
served at `$MODEL_ENDPOINT`.

This file refuses the copy that was published rather than looking for the copy that replaced it.
A document carrying both would satisfy "is the new sentence there?" and still tell a team to
train a rank-64 LoRA adapter and ship it -- which is the whole failure this test exists to
prevent. WITHDRAWN is the wording this repository's documents published before the removal
(commit 72ae1f0), plus a few variant spellings of it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SKIP_PARTS = {".git", "node_modules", "__pycache__", "units"}

#: The changelog records the withdrawal, so it quotes the withdrawn wording and names the withdrawn
#: categories on purpose. It is a record of what changed, not a description of what is offered.
SKIP_FILES = {"CHANGELOG.md"}

#: Withdrawn categories. They may still be NAMED -- a document saying they are invalid has to say
#: which values it means -- but only to be closed off, which is what CLOSURE requires below.
WITHDRAWN_CATEGORIES = ("byo-small", "byo-large")
CLOSURE = ("not part of this competition",)

#: The bring-your-own copy, as this repository published it.
WITHDRAWN = (
    "Bring your own adapter",
    "bring your own adapter",
    "Bringing your own model",
    "bring your own **adapter**",
    "adapter-only BYO",
    "adapter-only",
    "lora adapter",
    "adapter_model.safetensors",
    "adapter_config.json",
    "max_lora_rank",
    "max-lora-rank",
    "enable-lora",
    "lora-modules",
    "BYO entry",
    "BYO entries",
    "BYO run",
    "BYO submission",
    "BYO model",
    "a BYO ",
    "the `byo` path",
    "byo-adapter",
    "BYO-adapter",
    "There is no small-weights tier",
    "there is no small-weights tier",
    "no small-weights tier",
    "your adapter loaded",
    "with your adapter",
    "rank ≤ 64",
    "rank <= 64",
)


def _skipped(relative: pathlib.Path) -> bool:
    # Any dot-directory is skipped too: CI may check another repository out inside this one
    # (for example a toolkit under `.ci-toolkit/`), and its documents are not this repository's.
    parts = relative.parts
    return (
        bool(SKIP_PARTS & set(parts))
        or any(part.startswith(".") for part in parts)
        or relative.as_posix() in SKIP_FILES
    )


def _candidates() -> list[pathlib.Path]:
    """Tracked Markdown files when this is a git checkout; every Markdown file otherwise."""
    if (REPO / ".git").exists():
        try:
            listed = subprocess.run(
                ["git", "ls-files", "-z", "--", "*.md"],
                cwd=REPO,
                capture_output=True,
                check=True,
            ).stdout.decode("utf-8")
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            return [pathlib.Path(name) for name in listed.split("\0") if name]
    return [path.relative_to(REPO) for path in REPO.rglob("*.md")]


def _documents() -> list[pathlib.Path]:
    found = sorted(
        REPO / relative
        for relative in _candidates()
        if not _skipped(relative) and (REPO / relative).is_file()
    )
    if len(found) < 3:
        raise AssertionError(
            f"the document glob matched {len(found)} files; it has stopped reading the repository"
        )
    return found


def _flat(path: pathlib.Path) -> str:
    """The document as one line. Every assertion about a SENTENCE has to survive a rewrap."""
    return " ".join(path.read_text(encoding="utf-8").split())


def _offers() -> list[str]:
    return [
        f"{path.relative_to(REPO)} still carries {phrase!r}"
        for path in _documents()
        for phrase in WITHDRAWN
        if phrase in _flat(path)
    ]


def _unclosed() -> list[str]:
    found = []
    for path in _documents():
        flat = _flat(path)
        if not any(value in flat for value in WITHDRAWN_CATEGORIES):
            continue
        if not all(phrase in flat for phrase in CLOSURE):
            found.append(
                f"{path.relative_to(REPO)} names a withdrawn category without closing it off"
            )
    return found


def test_no_document_offers_or_describes_a_bring_your_own_model() -> None:
    assert not _offers(), "\n".join(_offers())


def test_a_document_naming_a_withdrawn_category_closes_it_off() -> None:
    assert not _unclosed(), "\n".join(_unclosed())


def main() -> int:
    """Script entry point, so a repository whose CI runs stdlib-only checks can run this too."""
    failures = _offers() + _unclosed()
    print(f"{len(_documents())} documents checked against {len(WITHDRAWN)} withdrawn phrases")
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("OK: no published document offers a bring-your-own model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
