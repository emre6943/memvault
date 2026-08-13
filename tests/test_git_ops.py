"""Git operations for the ingestion pass.

These run against real repositories in `tmp_path` rather than a mocked `subprocess`. The whole
value of this module is that it agrees with git about pathspecs, porcelain output, and partial
commits, and a mock would only ever agree with the assumptions that wrote it.

Two behaviours are tested hardest because the pass breaks loudly without them: a pathspec that
matches nothing must be dropped rather than passed to `git add`, which exits 128 on it; and a
commit must carry exactly the paths it was given, so a human's unrelated staged work is not
swept into an automated commit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from memvault.git_ops import (
    DirtyVaultError,
    GitError,
    StatusEntry,
    commit,
    dirty_paths,
    ensure_clean,
    has_staged_changes,
    head_sha,
    ignored,
    parse_status,
    stage,
    stageable,
    status,
    template_root,
    tracked,
)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout


def make_repo(tmp_path: Path, name: str = "vault", *, initial: bool = True) -> Path:
    """A real git repo with an identity of its own, so the ambient config cannot decide a test."""
    root = tmp_path / name
    (root / "inbox").mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.name", "MemVault Test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    if initial:
        (root / "README.md").write_text("# vault\n", encoding="utf-8")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "initial")
    return root


def subjects(root: Path) -> list[str]:
    payload = git(root, "log", "--format=%s")
    return [line for line in payload.splitlines() if line]


class TestParseStatus:
    """Porcelain parsing, including the two shapes that break a naive split."""

    def test_an_untracked_file_is_read_as_changed(self) -> None:
        entries = parse_status("?? notes/new.md\0")

        assert entries == (StatusEntry("?", "?", "notes/new.md"),)
        assert entries[0].changed

    def test_a_staged_addition_and_a_worktree_edit_are_distinguished(self) -> None:
        entries = parse_status("A  a.md\0 M b.md\0")

        assert [entry.path for entry in entries] == ["a.md", "b.md"]
        assert entries[0].index_status == "A"
        assert entries[1].worktree_status == "M"

    def test_a_rename_consumes_its_origin_path(self) -> None:
        entries = parse_status("R  new.md\0old.md\0?? other.md\0")

        assert [entry.path for entry in entries] == ["new.md", "other.md"]

    def test_a_path_with_spaces_survives_nul_separation(self) -> None:
        entries = parse_status("?? notes/a note with spaces.md\0")

        assert entries[0].path == "notes/a note with spaces.md"

    def test_empty_output_is_no_entries(self) -> None:
        assert parse_status("") == ()

    def test_an_entry_describes_itself_for_a_message(self) -> None:
        assert StatusEntry("?", "?", "a.md").describe() == "?? a.md"


class TestStatusAndCleanliness:
    def test_a_fresh_repository_is_clean(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)

        assert status(root) == ()
        ensure_clean(root, label="vault")

    def test_an_untracked_file_makes_it_dirty(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "stray.md").write_text("x\n", encoding="utf-8")

        assert dirty_paths(root) == ("stray.md",)

    def test_a_modified_tracked_file_makes_it_dirty(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "README.md").write_text("# edited\n", encoding="utf-8")

        assert dirty_paths(root) == ("README.md",)

    def test_an_ignored_prefix_is_not_dirt(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "inbox" / "note.txt").write_text("hello\n", encoding="utf-8")

        assert dirty_paths(root, ignore=("inbox",)) == ()
        assert dirty_paths(root) == ("inbox/note.txt",)

    def test_an_ignored_prefix_does_not_match_a_sibling_by_name(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "inbox-archive").mkdir()
        (root / "inbox-archive" / "old.md").write_text("x\n", encoding="utf-8")

        assert dirty_paths(root, ignore=("inbox",)) == ("inbox-archive/old.md",)

    def test_ensure_clean_names_the_offending_path_and_the_remedy(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "stray.md").write_text("x\n", encoding="utf-8")

        with pytest.raises(DirtyVaultError, match="stray.md"):
            ensure_clean(root, label="personal")

    def test_ensure_clean_truncates_a_long_list(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        for index in range(8):
            (root / f"stray{index}.md").write_text("x\n", encoding="utf-8")

        with pytest.raises(DirtyVaultError, match="and 3 more"):
            ensure_clean(root, label="personal")

    def test_ensure_clean_states_the_scope_it_was_guarding(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "transcripts").mkdir()
        (root / "transcripts" / "stray.md").write_text("x\n", encoding="utf-8")

        with pytest.raises(DirtyVaultError) as caught:
            ensure_clean(root, label="personal", limit_to=("transcripts",))

        assert "guarding transcripts" in str(caught.value)
        assert "transcripts/stray.md" in str(caught.value)


class TestTemplateRoot:
    """A path template's guarded prefix, derived so it cannot drift from where writes land."""

    def test_a_nested_template_yields_its_top_directory(self) -> None:
        assert template_root("transcripts/{yyyy}/{mm}/{date}-{slug}.md") == "transcripts"

    def test_a_single_segment_root_is_returned_whole(self) -> None:
        assert template_root("repos/{project}") == "repos"

    def test_a_deep_static_prefix_still_yields_only_the_top_directory(self) -> None:
        """Coarse on purpose: a pass writing through both templates needs `repos` guarded."""
        assert template_root("repos/Vault/digest/{yyyy}-W{ww}.md") == "repos"

    def test_a_template_with_no_placeholder_is_still_a_root(self) -> None:
        assert template_root("notes.md") == "notes.md"

    def test_a_leading_placeholder_has_no_static_root(self) -> None:
        assert template_root("{project}/learnings.md") == ""


class TestStaging:
    def test_a_path_that_git_cannot_see_is_dropped_rather_than_passed(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)

        assert stageable(root, ["inbox/never-existed.txt"]) == ()

    def test_a_deleted_tracked_path_is_still_stageable(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "inbox" / "note.txt").write_text("hello\n", encoding="utf-8")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "drop arrives")
        (root / "inbox" / "note.txt").unlink()

        assert stageable(root, ["inbox/note.txt"]) == ("inbox/note.txt",)

    def test_staging_an_unmatched_path_alongside_a_real_one_does_not_abort(
        self, tmp_path: Path
    ) -> None:
        root = make_repo(tmp_path)
        (root / "notes.md").write_text("x\n", encoding="utf-8")

        staged = stage(root, ["notes.md", "inbox/never-existed.txt"])

        assert staged == ("notes.md",)
        assert has_staged_changes(root, list(staged))

    def test_staging_records_a_deletion(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "README.md").unlink()

        stage(root, ["README.md"])

        assert has_staged_changes(root, ["README.md"])

    def test_an_ignored_path_is_never_staged(self, tmp_path: Path) -> None:
        """`local_only` is enforced by .gitignore, so `git add` must never be asked to try."""
        root = make_repo(tmp_path)
        (root / ".gitignore").write_text("private/\n", encoding="utf-8")
        (root / "private").mkdir()
        (root / "private" / "secret.md").write_text("x\n", encoding="utf-8")
        (root / "open.md").write_text("x\n", encoding="utf-8")

        staged = stage(root, ["private/secret.md", "open.md"])

        assert staged == ("open.md",)
        assert ignored(root, ["private/secret.md", "open.md"]) == {"private/secret.md"}

    def test_tracked_reports_only_what_git_knows(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "stray.md").write_text("x\n", encoding="utf-8")

        assert tracked(root, ["README.md", "stray.md"]) == {"README.md"}

    def test_nothing_staged_is_not_a_change(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)

        assert has_staged_changes(root, ["README.md"]) is False
        assert has_staged_changes(root, []) is False


class TestCommit:
    def test_a_commit_returns_its_sha_and_vault(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "notes.md").write_text("x\n", encoding="utf-8")
        stage(root, ["notes.md"])

        reference = commit(root, "mem: ingest 1 item(s)", ["notes.md"], vault="personal")

        assert reference is not None
        assert reference.vault == "personal"
        assert reference.sha == head_sha(root)
        assert reference.short_sha == reference.sha[:7]
        assert subjects(root)[0] == "mem: ingest 1 item(s)"

    def test_nothing_to_commit_returns_none_rather_than_failing(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)

        assert commit(root, "mem: nothing", ["README.md"], vault="personal") is None
        assert len(subjects(root)) == 1

    def test_an_unmatched_pathspec_alone_returns_none(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)

        assert commit(root, "mem: nothing", ["ghost.md"], vault="personal") is None

    def test_a_commit_carries_only_the_paths_it_was_given(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)
        (root / "ours.md").write_text("ours\n", encoding="utf-8")
        (root / "theirs.md").write_text("theirs\n", encoding="utf-8")
        git(root, "add", "theirs.md")
        stage(root, ["ours.md"])

        commit(root, "mem: ours only", ["ours.md"], vault="personal")

        listed = git(root, "show", "--name-only", "--format=", "HEAD").split()
        assert listed == ["ours.md"]
        assert dirty_paths(root) == ("theirs.md",)

    def test_a_commit_works_on_an_unborn_branch(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path, initial=False)
        (root / "first.md").write_text("x\n", encoding="utf-8")
        stage(root, ["first.md"])

        reference = commit(root, "mem: first", ["first.md"], vault="personal")

        assert reference is not None
        assert subjects(root) == ["mem: first"]

    def test_head_is_none_before_the_first_commit(self, tmp_path: Path) -> None:
        assert head_sha(make_repo(tmp_path, initial=False)) is None


class TestFailures:
    def test_a_failing_git_command_raises_with_its_own_output(self, tmp_path: Path) -> None:
        root = make_repo(tmp_path)

        with pytest.raises(GitError, match="rev-parse"):
            from memvault.git_ops import run_git

            run_git(root, "rev-parse", "--verify", "refs/heads/nope")

    def test_a_missing_repository_raises_rather_than_reporting_clean(self, tmp_path: Path) -> None:
        with pytest.raises(GitError):
            status(tmp_path / "not-a-repo")

    def test_a_dirty_vault_error_is_a_git_error(self) -> None:
        assert issubclass(DirtyVaultError, GitError)
