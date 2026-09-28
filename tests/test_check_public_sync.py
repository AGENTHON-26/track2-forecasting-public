"""scripts/check_public_sync.sh against small synthetic histories. Needs only git and bash.

The script has two halves: every public commit must be in this history, and public main's files
must be the files of some commit on this repository's main line (its first-parent chain). The
second half exists because the first passes by construction once a public commit is merged here,
whatever the merge kept. Each case builds a throwaway repository with git plumbing and runs the
script with STAGING_REF / PUBLIC_REF, so nothing here reads this repository's history, adds a
remote or touches the network.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "check_public_sync.sh"

BASE = {"README.md": "v1\n", "a.txt": "a\n"}


class Repo:
    """A bare-bones repository whose commits are built from {path: text} snapshots."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_AUTHOR_NAME="synthetic",
            GIT_AUTHOR_EMAIL="synthetic@example.invalid",
            GIT_COMMITTER_NAME="synthetic",
            GIT_COMMITTER_EMAIL="synthetic@example.invalid",
            GIT_AUTHOR_DATE="2020-01-01T00:00:00Z",
            GIT_COMMITTER_DATE="2020-01-01T00:00:00Z",
        )
        self.git("init", "-q")

    def git(self, *args: str, stdin: str | None = None, **extra: str) -> str:
        done = subprocess.run(
            ["git", *args],
            cwd=self.root,
            env={**self.env, **extra},
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    def commit(self, files: dict[str, str], *parents: str, message: str = "c") -> str:
        index = str(self.root / ".git" / "synthetic-index")
        self.git("read-tree", "--empty", GIT_INDEX_FILE=index)
        for path, text in files.items():
            blob = self.git("hash-object", "-w", "--stdin", stdin=text)
            self.git(
                "update-index",
                "--add",
                "--cacheinfo",
                f"100644,{blob},{path}",
                GIT_INDEX_FILE=index,
            )
        tree = self.git("write-tree", GIT_INDEX_FILE=index)
        flags = [arg for parent in parents for arg in ("-p", parent)]
        return self.git("commit-tree", tree, *flags, "-m", message)

    def check(self, staging: str, public: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=self.root,
            env={**self.env, "STAGING_REF": staging, "PUBLIC_REF": public},
            capture_output=True,
            text=True,
            check=False,
        )


def test_fast_forward_publish_passes_and_reports_the_pending_files(tmp_path: pathlib.Path) -> None:
    repo = Repo(tmp_path)
    s0 = repo.commit(BASE)
    published = repo.commit({**BASE, "a.txt": "a2\n"}, s0)
    pending = repo.commit({**BASE, "a.txt": "a2\n", "b.txt": "b\n"}, published)
    result = repo.check(pending, published)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: OK" in result.stdout
    assert "b.txt" in result.stdout


def test_publish_commit_merged_back_without_content_change_passes(tmp_path: pathlib.Path) -> None:
    repo = Repo(tmp_path)
    s0 = repo.commit(BASE)
    s1 = repo.commit({**BASE, "a.txt": "a2\n"}, s0)
    publish = repo.commit({**BASE, "a.txt": "a2\n"}, s0, message="publish")
    merge_back = repo.commit({**BASE, "a.txt": "a2\n"}, s1, publish, message="merge back")
    result = repo.check(merge_back, publish)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Nothing pending" in result.stdout


def test_staging_moving_between_publish_and_merge_back_passes(tmp_path: pathlib.Path) -> None:
    repo = Repo(tmp_path)
    s0 = repo.commit(BASE)
    s1 = repo.commit({**BASE, "a.txt": "a2\n"}, s0)
    publish = repo.commit({**BASE, "a.txt": "a2\n"}, s0, message="publish")
    s2 = repo.commit({**BASE, "a.txt": "a2\n", "c.txt": "c\n"}, s1)
    merge_back = repo.commit({**BASE, "a.txt": "a2\n", "c.txt": "c\n"}, s2, publish)
    result = repo.check(merge_back, publish)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "c.txt" in result.stdout


def test_public_edit_merged_back_without_its_content_fails(tmp_path: pathlib.Path) -> None:
    repo = Repo(tmp_path)
    s0 = repo.commit(BASE)
    s1 = repo.commit({**BASE, "a.txt": "a2\n"}, s0)
    publish = repo.commit({**BASE, "a.txt": "a2\n"}, s0, message="publish")
    merge_back = repo.commit({**BASE, "a.txt": "a2\n"}, s1, publish)
    edited = {**BASE, "a.txt": "a2\n", "README.md": "v1\npublic-only line\n"}
    public_edit = repo.commit(edited, publish, message="edit on public")
    # What `git merge -s ours` records: both parents, staging's files only.
    ours = repo.commit({**BASE, "a.txt": "a2\n"}, merge_back, public_edit)
    result = repo.check(ours, public_edit)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "public has commits missing here: 0" in result.stdout
    assert "RESULT: FAIL" in result.stdout
    assert "M README.md  [differs in staging]" in result.stdout
    assert "a.txt" not in result.stdout.split("RESULT: FAIL", 1)[1]


def test_public_edit_merged_back_with_content_fails_until_the_next_publish(
    tmp_path: pathlib.Path,
) -> None:
    # Strict rule: public main must BE a snapshot of this main line, so a public edit that came
    # back with its content still fails while staging carries unpublished work on top of it.
    repo = Repo(tmp_path)
    s0 = repo.commit(BASE)
    publish = repo.commit(BASE, s0, message="publish")
    s1 = repo.commit(BASE, s0, publish)
    s2 = repo.commit({**BASE, "c.txt": "c\n"}, s1)
    edited = {**BASE, "README.md": "v2\n"}
    public_edit = repo.commit(edited, publish, message="edit on public")
    real_merge = repo.commit({**edited, "c.txt": "c\n"}, s2, public_edit)
    result = repo.check(real_merge, public_edit)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "M README.md  [same in staging]" in result.stdout
    republish = repo.commit({**edited, "c.txt": "c\n"}, public_edit, message="publish")
    merge_back = repo.commit({**edited, "c.txt": "c\n"}, real_merge, republish)
    result = repo.check(merge_back, republish)
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_commit_not_merged_here_fails_the_history_half(tmp_path: pathlib.Path) -> None:
    repo = Repo(tmp_path)
    s0 = repo.commit(BASE)
    s1 = repo.commit({**BASE, "a.txt": "a2\n"}, s0)
    public_edit = repo.commit({**BASE, "README.md": "v2\n"}, s0, message="edit on public")
    result = repo.check(s1, public_edit)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "public was edited directly" in result.stdout


def test_a_public_ref_that_does_not_resolve_is_undetermined_not_a_pass(
    tmp_path: pathlib.Path,
) -> None:
    repo = Repo(tmp_path)
    s0 = repo.commit(BASE)
    result = repo.check(s0, "refs/remotes/nowhere/main")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "UNDETERMINED" in result.stdout
