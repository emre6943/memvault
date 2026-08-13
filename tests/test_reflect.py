"""The weekly reflection pass: a window of raw material in, durable memory and a digest out.

Written in the order the plan asks for. **The append-only tests come first**, because a
regression that rewrites `learnings.md` destroys history that has no other copy, and it is the
one failure here that a green suite and a plausible-looking diff would both hide.

Everything runs against real git repositories in `tmp_path` and fixture SQLite databases built
with the claude-mem schema. The curator is injected through its protocol, so no test spawns a
subprocess, reads the real claude-mem database, or reaches the network; the notifier is injected
for the same reason.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sqlite3
import subprocess
from collections.abc import Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pytest
import yaml

from memvault.classify import CommandResult
from memvault.config import ClassifierConfig, Config, IndexConfig, VaultConfig, load_config
from memvault.index import build_index
from memvault.recall import recall
from memvault.reflect import (
    MAX_REFLECTION_QUESTIONS,
    ClaudeCliCurator,
    Curation,
    CurationFailed,
    CurationOutcome,
    FiledFile,
    LearningEntry,
    ProjectDigest,
    QuirkEntry,
    ReflectionMaterial,
    ReflectionQuestion,
    ReflectStatus,
    append_block,
    build_curation_prompt,
    cited_questions,
    default_window,
    filed_in_window,
    insert_block_at_top,
    learnings_skeleton,
    quirks_skeleton,
    reflect_log_skeleton,
    run_reflect,
)

SINCE = date(2026, 7, 26)
UNTIL = date(2026, 8, 1)
TODAY = date(2026, 8, 1)

LEARNING = LearningEntry(
    project="Sapik",
    claim="APNs payloads need a gcm.message_id or firebase_messaging drops the tap",
    takeaway=(
        "The plugin's native code ignores any notification without that key, so a raw APNs "
        "push looks delivered and does nothing on tap."
    ),
    context="transcripts/2026/07/2026-07-28-push-taps.md",
)

QUIRK = QuirkEntry(
    project="Sapik",
    behavior="Push banners appear but taps go nowhere",
    symptom="The banner shows and plays a sound; tapping it opens the app wherever it was.",
    cause="firebase_messaging only handles notifications carrying gcm.message_id.",
    fix="Stamp a UUID into gcm.message_id on every push the backend sends.",
    context="transcripts/2026/07/2026-07-28-push-taps.md",
)

CURATION = Curation(
    summary="A week spent chasing one push-notification bug to its real cause.",
    learnings=(LEARNING,),
    quirks=(QUIRK,),
    projects=(ProjectDigest(project="Sapik", bullets=("Found the push-tap root cause.",)),),
    themes=("Verify the layer boundary before fixing the logic above it.",),
    open_threads=("The Android side of the same payload is untested.",),
    model="claude-stub",
)

TRANSCRIPT = "transcripts/2026/07/2026-07-28-push-taps.md"

QUESTION = ReflectionQuestion(
    question="Why did the push banners arrive while the taps did nothing?",
    answer=(
        "firebase_messaging drops any notification without gcm.message_id, so delivery and "
        "tap handling looked like one feature when they are two."
    ),
    citations=(TRANSCRIPT,),
)

#: The same week, from a curator that also answered the questions it posed. Kept apart from
#: `CURATION` so every pre-existing test still exercises the shape of a curator that returns no
#: reflection at all — which is what a stub written before this field existed does.
REFLECTED = dataclasses.replace(CURATION, reflection=(QUESTION,))

SCHEMA = """
CREATE TABLE sdk_sessions (memory_session_id TEXT PRIMARY KEY, project TEXT);
CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_session_id TEXT NOT NULL,
    project TEXT NOT NULL,
    text TEXT, type TEXT NOT NULL, title TEXT, subtitle TEXT, facts TEXT, narrative TEXT,
    concepts TEXT, files_read TEXT, files_modified TEXT, prompt_number INTEGER,
    created_at TEXT NOT NULL, created_at_epoch INTEGER NOT NULL, content_hash TEXT,
    generated_by_model TEXT, merged_into_project TEXT, agent_type TEXT, agent_id TEXT,
    metadata TEXT
);
CREATE TABLE session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_session_id TEXT NOT NULL,
    project TEXT NOT NULL,
    request TEXT, investigated TEXT, learned TEXT, completed TEXT, next_steps TEXT,
    files_read TEXT, files_edited TEXT, notes TEXT, prompt_number INTEGER,
    created_at TEXT NOT NULL, created_at_epoch INTEGER NOT NULL, merged_into_project TEXT
);
"""


# --------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout


def make_repo(tmp_path: Path, name: str) -> Path:
    """A vault-shaped git repo with its own identity, so ambient git config decides nothing."""
    root = tmp_path / name
    (root / "inbox").mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.name", "MemVault Test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    (root / "README.md").write_text("# vault\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    return root


def commit_count(root: Path) -> int:
    return len([line for line in git(root, "log", "--format=%s").splitlines() if line])


def write_config(tmp_path: Path, personal: Path, **overrides: Any) -> Path:
    vault: dict[str, Any] = {
        "root": str(personal),
        "digest_template": "repos/Vault/digest/{yyyy}-W{ww}.md",
        "memory_template": "repos/{project}",
        "claude_mem_projects": {
            "Sapik": ["Sapik"],
            "Homeworld": ["Homeworld"],
            "tcg-vendor": ["tcg-vendor", "TCGVendor"],
        },
    }
    vault.update(overrides)
    data = {"default_vault": "personal", "vaults": {"personal": vault}}
    path = tmp_path / "memvault.config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def scratch(tmp_path: Path) -> tuple[Config, VaultConfig]:
    config = load_config(write_config(tmp_path, make_repo(tmp_path, "personal")))
    return config, config.vault("personal")


def file_transcript(
    vault: VaultConfig,
    relative: str,
    *,
    title: str = "Push taps",
    ingested: date = date(2026, 7, 28),
    project: str | None = "Sapik",
    body: str = "Ada: the push taps still do nothing.",
    content_id: str = "abc123",
    commit: bool = True,
) -> Path:
    """A vault file shaped the way U4 files one, without running the ingestion pass.

    Committed by default, because that is the state a real vault is in when reflection runs:
    ingestion files and commits, and reflection refuses to start on a dirty vault.
    """
    path = vault.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter: dict[str, Any] = {
        "title": title,
        "date": ingested.isoformat(),
        "classification": "personal",
        "summary": "A debugging session about iOS push taps.",
        "ingested": ingested.isoformat(),
        "content_id": content_id,
    }
    if project:
        frontmatter["project"] = project
    path.write_text(
        "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n\n" + body + "\n",
        encoding="utf-8",
    )
    if commit:
        git(vault.root, "add", "-A", "--", relative)
        git(vault.root, "commit", "-q", "-m", f"mem: ingest {relative}")
    return path


def make_db(
    tmp_path: Path,
    rows: tuple[tuple[str, date, str], ...] = (("Sapik", date(2026, 7, 29), "Found the guard"),),
    *,
    name: str = "claude-mem.db",
) -> Path:
    """A claude-mem fixture with the live schema and millisecond timestamps."""
    path = tmp_path / name
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for project, day, title in rows:
        stamp = datetime.combine(day, time(12, 0))
        conn.execute(
            "INSERT INTO observations (memory_session_id, project, type, title, subtitle, facts, "
            "narrative, concepts, files_modified, created_at, created_at_epoch) "
            "VALUES ('orphan', ?, 'discovery', ?, '', '[]', 'a narrative', '[]', '[]', ?, ?)",
            (project, title, stamp.isoformat(), int(stamp.timestamp() * 1000)),
        )
    conn.commit()
    conn.close()
    return path


class StubCurator:
    """A curator that returns what it was given. Records the material it saw."""

    def __init__(self, outcome: CurationOutcome = CURATION) -> None:
        self.outcome = outcome
        self.seen: list[ReflectionMaterial] = []

    def curate(self, material: ReflectionMaterial) -> CurationOutcome:
        self.seen.append(material)
        return self.outcome


def reflect(
    scratch: tuple[Config, VaultConfig],
    *,
    curator: StubCurator | None = None,
    db_path: Path | str | None = None,
    since: date = SINCE,
    until: date = UNTIL,
) -> Any:
    config, vault = scratch
    return run_reflect(
        config,
        vault,
        curator=curator or StubCurator(),
        since=since,
        until=until,
        today=TODAY,
        db_path=db_path if db_path is not None else "/nonexistent/claude-mem.db",
        notifier=None,
    )


def memory(vault: VaultConfig, project: str, name: str) -> Path:
    return vault.root / "repos" / project / name


def seed_project(vault: VaultConfig, project: str, *, entries: str = "") -> None:
    """A project directory as the vault already holds it, skeleton plus any prior entries."""
    directory = vault.root / "repos" / project
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "learnings.md").write_text(learnings_skeleton(project) + entries, encoding="utf-8")
    (directory / "quirks.md").write_text(quirks_skeleton(project), encoding="utf-8")
    (directory / "reflect-log.md").write_text(reflect_log_skeleton(project), encoding="utf-8")
    git(vault.root, "add", "-A")
    git(vault.root, "commit", "-q", "-m", "seed memory")


PRIOR_LEARNING = (
    "\n## 2026-05-10 — newest-first lists need newest-first iteration\n"
    "- Context: adhoc:build-37-fixes\n"
    "- Takeaway: Server returns messages DESC, so iterating reversed gives oldest-first.\n"
)


# --------------------------------------------------------------------------------------
# Append-only. First, and deliberately so.
# --------------------------------------------------------------------------------------


class TestAppendOnly:
    """History with no other copy. Every write here is an addition or it is a bug."""

    def test_existing_learnings_survive_byte_for_byte(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik", entries=PRIOR_LEARNING)
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        before = memory(vault, "Sapik", "learnings.md").read_bytes()

        reflect(scratch, db_path=make_db(tmp_path))

        after = memory(vault, "Sapik", "learnings.md").read_bytes()
        assert after.startswith(before)
        assert len(after) > len(before)

    def test_only_additions_appear_and_they_appear_at_the_end(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik", entries=PRIOR_LEARNING)
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        reflect(scratch, db_path=make_db(tmp_path))

        text = memory(vault, "Sapik", "learnings.md").read_text(encoding="utf-8")
        # `## YYYY-MM-DD — ...` inside the skeleton's fenced format block is documentation, not
        # an entry, so only dated headings count.
        headings = [line for line in text.splitlines() if line.startswith("## 2026-")]
        assert headings[0].startswith("## 2026-05-10")
        assert headings[-1] == f"## {TODAY.isoformat()} — {LEARNING.claim}"
        assert "- Takeaway: " + LEARNING.takeaway in text

    def test_the_commit_diff_contains_no_deletions(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """The property R15's review surface rests on: a reviewer only ever reads additions."""
        _, vault = scratch
        seed_project(vault, "Sapik", entries=PRIOR_LEARNING)
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        reflect(scratch, db_path=make_db(tmp_path))

        numstat = git(vault.root, "show", "--format=", "--numstat", "HEAD")
        deletions = [line for line in numstat.splitlines() if line and line.split("\t")[1] != "0"]
        assert deletions == []

    def test_a_quirk_is_appended_in_the_vaults_format_without_a_date_in_its_heading(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        reflect(scratch, db_path=make_db(tmp_path))

        text = memory(vault, "Sapik", "quirks.md").read_text(encoding="utf-8")
        assert f"## {QUIRK.behavior}" in text
        assert f"- Symptom: {QUIRK.symptom}" in text
        assert f"- Cause: {QUIRK.cause}" in text
        assert f"- Fix: {QUIRK.fix}" in text
        assert f"- First seen: {TODAY.isoformat()} ({QUIRK.context})" in text

    def test_an_optional_fix_line_is_omitted_rather_than_written_empty(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        curation = Curation(summary="s", quirks=(QuirkEntry(**{**vars(QUIRK), "fix": None}),))

        reflect(scratch, curator=StubCurator(curation), db_path=make_db(tmp_path))

        assert "- Fix:" not in memory(vault, "Sapik", "quirks.md").read_text(encoding="utf-8")


class TestAppendPrimitives:
    """The two writers, tested directly — they are where an append-only bug would live."""

    def test_append_block_preserves_a_file_without_a_trailing_newline(self, tmp_path: Path) -> None:
        path = tmp_path / "learnings.md"
        path.write_text("# X — Learnings\n\n## 2026-01-01 — a claim\n- Takeaway: x", "utf-8")
        before = path.read_text(encoding="utf-8")

        assert append_block(path, "## 2026-02-02 — new", "## 2026-02-02 — new\n- Takeaway: y")

        after = path.read_text(encoding="utf-8")
        assert after.startswith(before)
        assert after.endswith("## 2026-02-02 — new\n- Takeaway: y\n")

    def test_a_heading_already_present_is_not_appended_again(self, tmp_path: Path) -> None:
        path = tmp_path / "learnings.md"
        path.write_text("# X\n\n## 2026-01-01 — a claim\n- Takeaway: x\n", encoding="utf-8")
        before = path.read_text(encoding="utf-8")

        assert not append_block(path, "## 2026-01-01 — a claim", "## 2026-01-01 — a claim\n- y")
        assert path.read_text(encoding="utf-8") == before

    def test_a_heading_inside_the_format_fence_is_not_mistaken_for_an_entry(
        self, tmp_path: Path
    ) -> None:
        """The skeleton documents the entry format in a fenced block. It is not an entry."""
        path = tmp_path / "learnings.md"
        path.write_text(learnings_skeleton("X"), encoding="utf-8")

        assert append_block(path, "## YYYY-MM-DD — <one-line summary>", "## YYYY-MM-DD — x")

    def test_insert_at_top_places_the_entry_above_the_newest_and_below_the_format_block(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "reflect-log.md"
        path.write_text(
            reflect_log_skeleton("X") + "\n## 2026-07-22 — an older pass\n- Window: a..b\n",
            encoding="utf-8",
        )
        before = path.read_text(encoding="utf-8")

        assert insert_block_at_top(path, "## 2026-08-01 — new", "## 2026-08-01 — new\n- Window: c")

        after = path.read_text(encoding="utf-8")
        headings = [line for line in after.splitlines() if line.startswith("## ")]
        assert headings == [
            "## YYYY-MM-DD — <one-line summary>",  # inside the fence, documentation
            "## 2026-08-01 — new",
            "## 2026-07-22 — an older pass",
        ]
        # Nothing existing was rewritten: the header is still the prefix and the old entry is
        # still present verbatim.
        assert after.startswith("# X — Reflect Log\n")
        assert "## 2026-07-22 — an older pass\n- Window: a..b\n" in after
        assert len(after) > len(before)

    def test_insert_at_top_into_a_file_with_no_entries_yet_appends(self, tmp_path: Path) -> None:
        path = tmp_path / "reflect-log.md"
        path.write_text(reflect_log_skeleton("X"), encoding="utf-8")
        before = path.read_text(encoding="utf-8")

        assert insert_block_at_top(path, "## 2026-08-01 — first", "## 2026-08-01 — first\n- a")

        after = path.read_text(encoding="utf-8")
        assert after.startswith(before)


# --------------------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------------------


class TestReflectionPass:
    """AE9: a window of material becomes memory, a digest, and one commit."""

    def test_a_window_appends_entries_and_writes_a_digest(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        before = commit_count(vault.root)

        report = reflect(scratch, db_path=make_db(tmp_path))

        assert report.status is ReflectStatus.OK
        assert report.exit_code == 0
        assert report.counts("learning") == 1
        assert report.counts("quirk") == 1
        assert report.digest_path == "repos/Vault/digest/2026-W31.md"
        assert (vault.root / report.digest_path).exists()
        assert commit_count(vault.root) == before + 1

    def test_the_digest_carries_the_curators_judgement_and_its_own_provenance(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        report = reflect(scratch, db_path=make_db(tmp_path))

        assert report.digest_path is not None
        text = (vault.root / report.digest_path).read_text(encoding="utf-8")
        assert CURATION.summary in text
        assert "## Sapik" in text
        assert "- Found the push-tap root cause." in text
        assert "## Cross-cutting themes" in text
        assert "## Open threads" in text
        assert "1 filed transcript(s), 1 observation(s), 0 session summary(s)" in text

    def test_the_reflect_log_entry_lands_at_the_top(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        (vault.root / "repos" / "Sapik" / "reflect-log.md").write_text(
            reflect_log_skeleton("Sapik") + "\n## 2026-07-22 — an older pass\n- Window: a..b\n",
            encoding="utf-8",
        )
        git(vault.root, "add", "-A")
        git(vault.root, "commit", "-q", "-m", "older pass")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        reflect(scratch, db_path=make_db(tmp_path))

        text = memory(vault, "Sapik", "reflect-log.md").read_text(encoding="utf-8")
        entries = [line for line in text.splitlines() if line.startswith("## 2026-")]
        assert entries[0].startswith(f"## {TODAY.isoformat()} — memvault reflect")
        assert entries[1] == "## 2026-07-22 — an older pass"
        assert f"- Window: {SINCE.isoformat()}..{UNTIL.isoformat()}" in text
        assert "- Digest: repos/Vault/digest/2026-W31.md" in text

    def test_the_commit_message_names_what_was_appended(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        reflect(scratch, db_path=make_db(tmp_path))

        message = git(vault.root, "log", "-1", "--format=%B")
        assert "mem: reflect 2026-07-26..2026-08-01" in message
        assert "1 learning(s), 1 quirk(s)" in message
        assert "repos/Vault/digest/2026-W31.md" in message

    def test_the_curator_sees_both_sources(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        curator = StubCurator()

        reflect(scratch, curator=curator, db_path=make_db(tmp_path))

        material = curator.seen[0]
        assert [item.relative_path for item in material.filed] == [
            "transcripts/2026/07/2026-07-28-push-taps.md"
        ]
        assert [item.title for item in material.observations] == ["Found the guard"]
        assert material.projects == ("Sapik",)


class TestEmptyWindow:
    """Nothing happened is a legitimate week, and it must cost nothing."""

    def test_an_empty_window_writes_nothing_and_creates_no_commit(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        before = commit_count(vault.root)
        before_learnings = memory(vault, "Sapik", "learnings.md").read_bytes()
        curator = StubCurator()

        report = reflect(scratch, curator=curator, db_path=make_db(tmp_path, rows=()))

        assert report.appended == ()
        assert report.digest_path is None
        assert report.commits == ()
        assert commit_count(vault.root) == before
        assert memory(vault, "Sapik", "learnings.md").read_bytes() == before_learnings
        assert curator.seen == []  # not even asked

    def test_material_outside_the_window_does_not_count_as_material(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/05/2026-05-01-old.md", ingested=date(2026, 5, 1))

        report = reflect(scratch, db_path=make_db(tmp_path, rows=()))

        assert report.digest_path is None
        assert report.commits == ()

    def test_a_curation_with_nothing_in_it_writes_nothing(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """A quiet week the curator read and found nothing durable in. Still no digest."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        before = commit_count(vault.root)

        report = reflect(
            scratch,
            curator=StubCurator(Curation(summary="A quiet week.")),
            db_path=make_db(tmp_path),
        )

        assert report.appended == ()
        assert report.digest_path is None
        assert commit_count(vault.root) == before


class TestIdempotence:
    """Two passes over the same window is the ordinary case for an overlapping schedule."""

    def test_a_second_pass_appends_nothing_new(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        db = make_db(tmp_path)

        reflect(scratch, db_path=db)
        after_first = memory(vault, "Sapik", "learnings.md").read_bytes()
        commits_after_first = commit_count(vault.root)

        second = reflect(scratch, db_path=db)

        assert memory(vault, "Sapik", "learnings.md").read_bytes() == after_first
        assert second.appended == ()
        # The learning and the quirk were recognized as already present. No reflect-log entry is
        # even attempted: nothing was appended, so there is nothing for a log entry to record.
        assert [item.kind for item in second.duplicates] == ["learning", "quirk"]
        assert commit_count(vault.root) == commits_after_first

    def test_a_second_pass_leaves_one_of_each_entry(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        db = make_db(tmp_path)

        reflect(scratch, db_path=db)
        reflect(scratch, db_path=db)

        learnings = memory(vault, "Sapik", "learnings.md").read_text(encoding="utf-8")
        quirks = memory(vault, "Sapik", "quirks.md").read_text(encoding="utf-8")
        log = memory(vault, "Sapik", "reflect-log.md").read_text(encoding="utf-8")
        assert learnings.count(LEARNING.claim) == 1
        assert quirks.count(f"## {QUIRK.behavior}") == 1
        assert log.count(f"## {TODAY.isoformat()} — memvault reflect") == 1


class TestProjectDirectories:
    """A project with no memory directory yet is the first week of a new piece of work."""

    def test_a_missing_project_directory_is_created_from_the_skeleton(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        assert not (vault.root / "repos" / "Sapik").exists()

        report = reflect(scratch, db_path=make_db(tmp_path))

        assert report.created_projects == ("Sapik",)
        directory = vault.root / "repos" / "Sapik"
        assert {path.name for path in directory.iterdir()} == {
            "README.md",
            "learnings.md",
            "quirks.md",
            "reflect-log.md",
        }
        assert (
            (directory / "learnings.md")
            .read_text(encoding="utf-8")
            .startswith("# Sapik — Learnings")
        )
        assert LEARNING.claim in (directory / "learnings.md").read_text(encoding="utf-8")

    def test_a_created_directory_is_part_of_the_commit(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        reflect(scratch, db_path=make_db(tmp_path))

        names = git(vault.root, "show", "--format=", "--name-only", "HEAD")
        assert "repos/Sapik/learnings.md" in names
        assert "repos/Sapik/README.md" in names

    def test_an_entry_for_a_project_the_vault_does_not_claim_is_dropped(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """The boundary again, one layer up: a curated entry cannot invent a project."""
        _, vault = scratch
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        curation = Curation(
            summary="s",
            learnings=(LearningEntry(project="oryx", claim="c", takeaway="t"),),
            projects=(ProjectDigest(project="oryx", bullets=("x",)),),
        )

        report = reflect(scratch, curator=StubCurator(curation), db_path=make_db(tmp_path))

        assert report.unknown_projects == ("oryx",)
        assert report.appended == ()
        assert not (vault.root / "repos" / "oryx").exists()

    def test_an_existing_directory_makes_a_project_known_even_without_an_alias(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """A project someone created by hand is a project this vault claims."""
        _, vault = scratch
        seed_project(vault, "Poems")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        curation = Curation(
            summary="s", learnings=(LearningEntry(project="Poems", claim="c", takeaway="t"),)
        )

        report = reflect(scratch, curator=StubCurator(curation), db_path=make_db(tmp_path))

        assert report.unknown_projects == ()
        assert report.counts("learning") == 1


class TestClaudeMemDegradation:
    """A third-party database is allowed to be missing. It is not allowed to stop the pass."""

    def test_a_missing_database_degrades_to_vault_only_reflection_with_a_warning(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        report = reflect(scratch, db_path=tmp_path / "absent.db")

        assert report.status is ReflectStatus.PARTIAL
        assert report.exit_code == 3
        assert any("claude-mem unavailable" in warning for warning in report.warnings)
        # It still reflected: the vault's own material was enough.
        assert report.counts("learning") == 1
        assert report.digest_path is not None

    def test_an_unreadable_database_is_a_warning_not_a_crash(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"not a database")

        report = reflect(scratch, db_path=corrupt)

        assert report.status is ReflectStatus.PARTIAL
        assert report.commits  # the pass still committed what it learned

    def test_work_observations_are_reported_as_skipped_without_degrading_the_run(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        db = make_db(
            tmp_path,
            rows=(
                ("Sapik", date(2026, 7, 29), "mine"),
                ("oryx", date(2026, 7, 29), "work"),
                ("olyx", date(2026, 7, 29), "also work"),
            ),
        )
        curator = StubCurator()

        report = reflect(scratch, curator=curator, db_path=db)

        assert [item.title for item in curator.seen[0].observations] == ["mine"]
        assert sorted(report.skipped_projects) == ["olyx (1)", "oryx (1)"]
        assert report.status is ReflectStatus.OK

    def test_two_spellings_of_one_project_reach_the_curator_as_one(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md", project=None)
        db = make_db(
            tmp_path,
            rows=(
                ("tcg-vendor", date(2026, 7, 28), "one"),
                ("TCGVendor", date(2026, 7, 29), "two"),
            ),
        )
        curator = StubCurator()

        reflect(scratch, curator=curator, db_path=db)

        assert curator.seen[0].projects == ("tcg-vendor",)


class TestClaudeMemDisabled:
    """A source the vault never enabled is a clean skip, not a degraded pass (R11).

    This is the state every adopter but Ada starts in: no plugin installed, no project map.
    Reporting that as a warning would make the first weekly pass anybody runs open with a
    problem about a third-party database they have never heard of — and a status that is always
    PARTIAL is a status nobody reads.
    """

    def unmapped(self, tmp_path: Path, **overrides: Any) -> tuple[Config, VaultConfig]:
        config = load_config(write_config(tmp_path, make_repo(tmp_path, "personal"), **overrides))
        return config, config.vault("personal")

    def test_an_explicitly_disabled_source_leaves_the_pass_clean(self, tmp_path: Path) -> None:
        scratch = self.unmapped(tmp_path, claude_mem=False)
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        report = reflect(scratch, db_path=tmp_path / "absent.db")

        assert report.warnings == ()
        assert report.status is ReflectStatus.OK
        assert report.exit_code == 0
        # It still reflected: the vault's own material was always the authoritative half.
        assert report.counts("learning") == 1

    def test_a_vault_with_no_project_map_leaves_the_pass_clean(self, tmp_path: Path) -> None:
        scratch = self.unmapped(tmp_path, claude_mem_projects={})
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        report = reflect(scratch, db_path=tmp_path / "absent.db")

        assert report.warnings == ()
        assert report.exit_code == 0

    def test_a_present_database_is_not_read_when_the_source_is_off(self, tmp_path: Path) -> None:
        # The toggle is the gate, not the file's absence: a machine that happens to run the
        # plugin still contributes nothing to a vault that did not ask for it.
        scratch = self.unmapped(tmp_path, claude_mem=False)
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        curator = StubCurator()

        reflect(scratch, curator=curator, db_path=make_db(tmp_path))

        assert curator.seen[0].observations == ()

    def test_the_digest_does_not_mention_a_source_the_vault_turned_off(
        self, tmp_path: Path
    ) -> None:
        scratch = self.unmapped(tmp_path, claude_mem=False)
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")

        report = reflect(scratch, db_path=tmp_path / "absent.db")

        assert report.digest_path is not None
        text = (vault.root / report.digest_path).read_text(encoding="utf-8")
        assert "claude-mem" not in text


class TestFailure:
    """A failed pass writes nothing at all — a half-curated week reads as a quiet one."""

    def test_a_failed_curation_writes_nothing_and_exits_one(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        before = memory(vault, "Sapik", "learnings.md").read_bytes()
        commits = commit_count(vault.root)

        report = reflect(
            scratch,
            curator=StubCurator(CurationFailed("the curator exited 1: no error output")),
            db_path=make_db(tmp_path),
        )

        assert report.status is ReflectStatus.FAILED
        assert report.exit_code == 1
        assert memory(vault, "Sapik", "learnings.md").read_bytes() == before
        assert commit_count(vault.root) == commits

    def test_dirt_outside_the_memory_tree_does_not_stop_the_pass(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """Reflect appends under repos/. A dirty reminders.md is not its business."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        (vault.root / "reminders.md").write_text("someone was working\n", encoding="utf-8")

        report = reflect(scratch, db_path=make_db(tmp_path))

        assert report.failure is None

    def test_a_dirty_memory_tree_stops_the_pass_before_it_writes(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        (vault.root / "repos" / "Sapik" / "notes.md").write_text(
            "someone was working\n", encoding="utf-8"
        )
        before = memory(vault, "Sapik", "learnings.md").read_bytes()

        report = reflect(scratch, db_path=make_db(tmp_path))

        assert report.status is ReflectStatus.FAILED
        assert report.failure is not None
        assert "uncommitted changes" in report.failure
        assert memory(vault, "Sapik", "learnings.md").read_bytes() == before

    def test_a_notifier_is_called_once_on_failure_and_never_on_success(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        config, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        calls: list[tuple[str, str, str]] = []

        def notifier(job: str, reason: str, log_path: str) -> bool:
            calls.append((job, reason, log_path))
            return True

        run_reflect(
            config,
            vault,
            curator=StubCurator(),
            since=SINCE,
            until=UNTIL,
            today=TODAY,
            db_path=make_db(tmp_path),
            notifier=notifier,
        )
        assert calls == []

        run_reflect(
            config,
            vault,
            curator=StubCurator(CurationFailed("nope")),
            since=SINCE,
            until=UNTIL,
            today=TODAY,
            db_path=make_db(tmp_path),
            notifier=notifier,
            log_path="/tmp/memvault-reflect.log",
        )
        assert [call[0] for call in calls] == ["reflect"]
        assert calls[0][2] == "/tmp/memvault-reflect.log"


class TestWindowAndMaterial:
    """What counts as this window's material, and what the curator is shown."""

    def test_the_default_window_is_seven_inclusive_days_ending_today(self) -> None:
        assert default_window(date(2026, 8, 1)) == (date(2026, 7, 26), date(2026, 8, 1))
        assert default_window(date(2026, 8, 1), days=1) == (date(2026, 8, 1), date(2026, 8, 1))

    def test_a_backwards_window_is_refused(self, scratch: tuple[Config, VaultConfig]) -> None:
        config, vault = scratch

        with pytest.raises(ValueError, match="starts after it ends"):
            run_reflect(
                config,
                vault,
                curator=StubCurator(),
                since=date(2026, 8, 2),
                until=date(2026, 8, 1),
                notifier=None,
            )

    def test_only_files_carrying_a_content_id_count_as_filed_material(
        self, scratch: tuple[Config, VaultConfig]
    ) -> None:
        """Otherwise last week's digest becomes next week's input, and the vault would
        slowly start reflecting on itself."""
        _, vault = scratch
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        hand_written = vault.root / "repos" / "Vault" / "digest" / "2026-W30.md"
        hand_written.parent.mkdir(parents=True)
        hand_written.write_text(
            "---\ntitle: Digest\ndate: 2026-07-28\n---\n\nlast week\n", encoding="utf-8"
        )

        filed = filed_in_window(vault, SINCE, UNTIL)

        assert [item.relative_path for item in filed] == [
            "transcripts/2026/07/2026-07-28-push-taps.md"
        ]

    def test_filed_material_is_bounded_by_the_window(
        self, scratch: tuple[Config, VaultConfig]
    ) -> None:
        _, vault = scratch
        file_transcript(vault, "transcripts/a.md", ingested=SINCE, content_id="a")
        file_transcript(vault, "transcripts/b.md", ingested=UNTIL, content_id="b")
        file_transcript(vault, "transcripts/c.md", ingested=date(2026, 7, 25), content_id="c")
        file_transcript(vault, "transcripts/d.md", ingested=date(2026, 8, 2), content_id="d")

        filed = filed_in_window(vault, SINCE, UNTIL)

        assert [item.relative_path for item in filed] == ["transcripts/a.md", "transcripts/b.md"]

    def test_the_inbox_is_not_material(self, scratch: tuple[Config, VaultConfig]) -> None:
        _, vault = scratch
        vault.inbox.mkdir(parents=True, exist_ok=True)
        (vault.inbox / "drop.md").write_text(
            "---\ncontent_id: pending\ningested: 2026-07-28\n---\n\nnot filed yet\n",
            encoding="utf-8",
        )

        assert filed_in_window(vault, SINCE, UNTIL) == ()

    def test_the_prompt_names_the_window_the_projects_and_the_material(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        file_transcript(vault, "transcripts/2026/07/2026-07-28-push-taps.md")
        curator = StubCurator()
        reflect(scratch, curator=curator, db_path=make_db(tmp_path))

        prompt = build_curation_prompt(curator.seen[0])

        assert "Window: 2026-07-26 to 2026-08-01" in prompt
        assert "Projects in scope: Sapik" in prompt
        assert "transcripts/2026/07/2026-07-28-push-taps.md" in prompt
        assert "Found the guard" in prompt

    def test_the_prompt_asks_for_questions_answered_with_paths_it_showed(
        self, scratch: tuple[Config, VaultConfig]
    ) -> None:
        """A citation is only checkable against the filed list, so the prompt must offer it."""
        _, vault = scratch
        file_transcript(vault, TRANSCRIPT)

        prompt = build_curation_prompt(
            ReflectionMaterial(
                vault=vault.name,
                since=SINCE,
                until=UNTIL,
                filed=filed_in_window(vault, SINCE, UNTIL),
            )
        )

        assert "reflection" in prompt
        assert "citations" in prompt
        assert "Filed into the vault" in prompt


# --------------------------------------------------------------------------------------
# The reflection note (R6)
# --------------------------------------------------------------------------------------


CITATION_RE = re.compile(r"^- cites \[\[(?P<target>.+)\]\]$", re.MULTILINE)


def note_citations(text: str) -> list[str]:
    """The paths a rendered reflection note points at, read the way the indexer reads them."""
    return CITATION_RE.findall(text)


class WordEmbedder:
    """A deterministic embedder over three declared terms, so similarity is a fixture fact.

    Real embeddings are not the subject here — whether the written note reaches recall at all
    is — but recall runs both modes, so it needs a geometry rather than a hash.
    """

    TERMS = ("push", "tap", "apns")
    model_id = "word-v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(text.casefold().count(term)) for term in self.TERMS] for text in texts]


class TestReflectionNote:
    """R6: a note that answers its own questions, cites real files, and never feeds itself."""

    def test_the_note_carries_each_answer_with_the_files_it_rests_on(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)

        report = reflect(scratch, curator=StubCurator(REFLECTED), db_path=make_db(tmp_path))

        assert report.reflection_path == "reflections/2026/2026-08-01-reflection.md"
        text = (vault.root / report.reflection_path).read_text(encoding="utf-8")
        assert f"## {QUESTION.question}" in text
        assert QUESTION.answer in text
        assert note_citations(text) == [TRANSCRIPT]

    def test_every_citation_in_the_note_opens(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """The verification the plan asks for, as a test: a cited path is a path that exists."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)

        report = reflect(scratch, curator=StubCurator(REFLECTED), db_path=make_db(tmp_path))

        assert report.reflection_path is not None
        text = (vault.root / report.reflection_path).read_text(encoding="utf-8")
        cited = note_citations(text)
        assert cited
        assert all((vault.root / path).is_file() for path in cited)

    def test_a_citation_the_window_did_not_file_is_dropped_and_the_rest_kept(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """An invented path is shaped exactly like a real one; only the filed set can tell."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)
        invented = "transcripts/2026/07/2026-07-30-a-session-that-never-happened.md"
        curation = dataclasses.replace(
            REFLECTED,
            reflection=(dataclasses.replace(QUESTION, citations=(invented, TRANSCRIPT)),),
        )

        report = reflect(scratch, curator=StubCurator(curation), db_path=make_db(tmp_path))

        assert report.reflection_path is not None
        text = (vault.root / report.reflection_path).read_text(encoding="utf-8")
        assert note_citations(text) == [TRANSCRIPT]
        assert invented not in text

    def test_an_answer_whose_citations_all_vanish_takes_the_note_with_it(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """A note of uncited claims is worse than no note: it reads exactly like a cited one."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)
        curation = dataclasses.replace(
            REFLECTED,
            reflection=(dataclasses.replace(QUESTION, citations=("notes/invented.md",)),),
        )

        report = reflect(scratch, curator=StubCurator(curation), db_path=make_db(tmp_path))

        assert report.status is ReflectStatus.OK
        assert report.reflection_path is None
        assert not (vault.root / "reflections").exists()

    def test_the_note_declares_its_kind_and_carries_no_content_id(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """`content_id` is what marks filed material — stamping one would enrol the note."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)

        report = reflect(scratch, curator=StubCurator(REFLECTED), db_path=make_db(tmp_path))

        assert report.reflection_path is not None
        frontmatter = yaml.safe_load(
            (vault.root / report.reflection_path).read_text(encoding="utf-8").split("---\n")[1]
        )
        assert frontmatter["kind"] == "reflection"
        assert "content_id" not in frontmatter
        assert frontmatter["date"] == UNTIL.isoformat()

    def test_last_windows_note_is_not_this_windows_material(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """The loop R6 exists to prevent: a vault that starts reflecting on its reflections."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)
        first = reflect(scratch, curator=StubCurator(REFLECTED), db_path=make_db(tmp_path))
        assert first.reflection_path is not None

        curator = StubCurator(REFLECTED)
        reflect(
            scratch,
            curator=curator,
            db_path=make_db(tmp_path),
            since=date(2026, 7, 28),
            until=date(2026, 8, 1),
        )

        seen = [item.relative_path for item in curator.seen[0].filed]
        assert seen == [TRANSCRIPT]
        assert first.reflection_path not in seen

    def test_a_reflection_kind_file_is_refused_even_when_it_carries_a_content_id(
        self, scratch: tuple[Config, VaultConfig]
    ) -> None:
        """The belt-and-braces half: the kind alone disqualifies a file as material."""
        _, vault = scratch
        path = vault.root / "reflections" / "2026" / "2026-08-01-reflection.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\ntitle: Reflection\nkind: reflection\ningested: 2026-07-28\n"
            "content_id: somehow\n---\n\nbody\n",
            encoding="utf-8",
        )

        assert filed_in_window(vault, SINCE, UNTIL) == ()

    def test_dirt_under_the_note_root_stops_the_pass_before_it_writes(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """The note root joins the guard, or the pass folds somebody's draft into its commit."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)
        stale = vault.root / "reflections" / "half-written.md"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("someone was writing here\n", encoding="utf-8")

        report = reflect(scratch, curator=StubCurator(REFLECTED), db_path=make_db(tmp_path))

        assert report.status is ReflectStatus.FAILED
        assert report.failure is not None
        assert "uncommitted changes" in report.failure
        assert report.reflection_path is None

    def test_the_note_is_committed_and_named_where_the_pass_is_reviewed(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)

        report = reflect(scratch, curator=StubCurator(REFLECTED), db_path=make_db(tmp_path))

        assert report.reflection_path is not None
        assert git(vault.root, "status", "--porcelain") == ""
        assert f"Reflection: {report.reflection_path}" in git(
            vault.root, "log", "-1", "--format=%B"
        )
        log = memory(vault, "Sapik", "reflect-log.md").read_text(encoding="utf-8")
        assert f"- Reflection: {report.reflection_path}" in log

    def test_a_second_pass_over_the_same_window_replaces_the_note(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """One window, one answer: a re-run is a re-render, not a second near-identical note."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)

        first = reflect(scratch, curator=StubCurator(REFLECTED), db_path=make_db(tmp_path))
        second = reflect(scratch, curator=StubCurator(REFLECTED), db_path=make_db(tmp_path))

        assert first.reflection_path == second.reflection_path
        notes = sorted(p.name for p in (vault.root / "reflections" / "2026").iterdir())
        assert notes == ["2026-08-01-reflection.md"]

    def test_a_degraded_claude_mem_still_produces_a_cited_note(
        self, scratch: tuple[Config, VaultConfig]
    ) -> None:
        """Vault material alone is enough — the note cites files, not observations."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)

        report = reflect(scratch, curator=StubCurator(REFLECTED))

        assert report.status is ReflectStatus.PARTIAL
        assert report.reflection_path is not None
        text = (vault.root / report.reflection_path).read_text(encoding="utf-8")
        assert note_citations(text) == [TRANSCRIPT]

    def test_the_note_is_indexed_and_recallable(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """The point of writing it: a later question finds the reflection, not just the source."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)
        report = reflect(scratch, curator=StubCurator(REFLECTED), db_path=make_db(tmp_path))
        assert report.reflection_path is not None

        index_config = IndexConfig(path=str(tmp_path / "index" / "{vault}.db"))
        build_index(vault, index_config, WordEmbedder())
        response = recall(vault, index_config, WordEmbedder(), "push banners tap gcm.message_id")

        assert report.reflection_path in [result.path for result in response.results]

    def test_a_curation_that_is_only_a_reflection_still_runs(
        self, scratch: tuple[Config, VaultConfig], tmp_path: Path
    ) -> None:
        """Answered questions are content; a week that produced only them is not a quiet week."""
        _, vault = scratch
        seed_project(vault, "Sapik")
        file_transcript(vault, TRANSCRIPT)
        curation = Curation(summary="One thing happened.", reflection=(QUESTION,))

        report = reflect(scratch, curator=StubCurator(curation), db_path=make_db(tmp_path))

        assert not curation.empty
        assert report.reflection_path is not None
        assert report.counts("learning") == 0


class TestCitationFiltering:
    """`cited_questions` in isolation: the rule that makes the note trustworthy."""

    @staticmethod
    def material(*paths: str) -> ReflectionMaterial:
        return ReflectionMaterial(
            vault="personal",
            since=SINCE,
            until=UNTIL,
            filed=tuple(
                FiledFile(relative_path=path, title=path, day=UNTIL.isoformat()) for path in paths
            ),
        )

    def test_only_paths_the_window_filed_survive(self) -> None:
        kept = cited_questions(
            (dataclasses.replace(QUESTION, citations=("a.md", "b.md", "c.md")),),
            self.material("a.md", "c.md"),
        )
        assert [entry.citations for entry in kept] == [("a.md", "c.md")]

    def test_an_entry_with_nothing_left_is_dropped_rather_than_written_bare(self) -> None:
        assert (
            cited_questions(
                (dataclasses.replace(QUESTION, citations=("gone.md",)),), self.material("a.md")
            )
            == ()
        )

    def test_an_empty_window_vouches_for_nothing(self) -> None:
        assert cited_questions((QUESTION,), self.material()) == ()


class TestCuratorReply:
    """What a real reply is read into, through the seam a real run uses."""

    @staticmethod
    def curation(payload: object) -> Curation:
        def runner(argv: Sequence[str], *, prompt: str, timeout: int) -> CommandResult:
            return CommandResult(exit_code=0, stdout=json.dumps({"result": json.dumps(payload)}))

        outcome = ClaudeCliCurator(ClassifierConfig(), runner=runner).curate(
            ReflectionMaterial(vault="personal", since=SINCE, until=UNTIL)
        )
        assert isinstance(outcome, Curation)
        return outcome

    def test_a_reply_with_questions_becomes_reflection_entries(self) -> None:
        curation = self.curation(
            {
                "summary": "one thing",
                "reflection": [
                    {"question": "What broke?", "answer": "The guard.", "citations": ["a.md"]}
                ],
            }
        )
        assert curation.reflection == (
            ReflectionQuestion(question="What broke?", answer="The guard.", citations=("a.md",)),
        )

    def test_an_uncited_answer_never_becomes_an_entry(self) -> None:
        """Dropped at the reply boundary, so nothing downstream has to carry a maybe-cited case."""
        assert (
            self.curation(
                {
                    "reflection": [
                        {"question": "What broke?", "answer": "The guard.", "citations": []}
                    ]
                }
            ).reflection
            == ()
        )

    def test_a_reply_with_no_reflection_field_is_still_a_curation(self) -> None:
        """A curator that answers nothing is a quiet week, not a failed call."""
        assert self.curation({"summary": "quiet"}).reflection == ()

    def test_a_flood_of_questions_is_truncated_rather_than_written_whole(self) -> None:
        payload = {
            "reflection": [
                {"question": f"Q{n}", "answer": "a", "citations": ["a.md"]} for n in range(20)
            ]
        }
        entries = self.curation(payload).reflection
        assert len(entries) == MAX_REFLECTION_QUESTIONS
        assert entries[0].question == "Q0"
