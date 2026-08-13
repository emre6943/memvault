"""The ingestion pass: inbox in, vault content and one commit out.

Written in the order the plan asks for. Idempotency and crash recovery come first, because they
are the properties that make KTD5's two-phase design safe rather than merely ordered, and they
are the ones a happy-path suite would never notice were missing. The happy path follows them.

Everything runs against real git repositories in `tmp_path`. The classifier and the distiller
are injected through their protocols, so no test spawns a subprocess or reaches the network, and
the notifier is injected too so no test posts a macOS notification.

The three scratch-vault facts these tests keep asserting, because each is a requirement:
the inbox drains to empty (R6), a re-run changes nothing (R3), and every change the pass made is
visible in one commit's diff (R7).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from memvault import distill as distill_module
from memvault.classify import Classification, NeedsReview, Outcome
from memvault.config import Config, VaultConfig, load_config
from memvault.distill import Distillation, DistillationFailed, DistillOutcome
from memvault.git_ops import GitError
from memvault.inbox import DeclaredMetadata, InboxRecord, discover
from memvault.ingest import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_PARTIAL,
    MARKER_SUFFIX,
    DrainedItem,
    HeldItem,
    IngestReport,
    IngestStatus,
    NoteFailure,
    build_commit_message,
    content_id_of,
    filed_ids,
    marker_for,
    run_ingest,
)
from memvault.route import route
from memvault.segment import segments_from_ranges
from memvault.writer import WriteError, file_record

INGESTED = date(2026, 8, 1)

PERSONAL_BODY = "Coffee with Deniz about the erfpacht deadline on the Amsterdam flat."

WORK_BODY = (
    "Ada: Sam called about the price list tool this morning.\n"
    "Deniz: We agreed to freeze the template editor until the Excel export is stable, because "
    "shipping both surfaces at once burned us on the last release.\n"
    "Ada: I will send the revised quote on Monday."
)

#: A sentence lifted verbatim out of `WORK_BODY`, for the guard to catch.
LEAKED_SENTENCE = (
    "We agreed to freeze the template editor until the Excel export is stable, because "
    "shipping both surfaces at once burned us on the last release."
)

PARAPHRASE = Distillation(
    title="Price list tool priorities",
    decisions=("The editor work stays on hold until the spreadsheet output is dependable.",),
    action_items=("Send the client an updated quote at the start of next week.",),
    model="claude-stub",
)


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


def subjects(root: Path) -> list[str]:
    return [line for line in git(root, "log", "--format=%s").splitlines() if line]


def commit_count(root: Path) -> int:
    return len(subjects(root))


def show(root: Path, ref: str = "HEAD") -> str:
    return git(root, "show", "--name-status", "--format=%B", ref)


@dataclass(frozen=True)
class Scratch:
    """A configured pair of scratch vaults, loaded through the real config loader."""

    config: Config
    personal: VaultConfig
    work: VaultConfig
    path: Path


def write_config(tmp_path: Path, personal: Path, work: Path, **sections: Any) -> Path:
    data: dict[str, Any] = {
        "default_vault": "personal",
        "vaults": {
            "personal": {"root": str(personal), "work_route": "work"},
            "work": {"root": str(work)},
        },
    }
    data.update(sections)
    path = tmp_path / "memvault.config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def scratch(tmp_path: Path) -> Scratch:
    path = write_config(tmp_path, make_repo(tmp_path, "personal"), make_repo(tmp_path, "work"))
    config = load_config(path)
    return Scratch(config, config.vault("personal"), config.vault("work"), path)


def drop(vault: VaultConfig, name: str, body: str) -> Path:
    """Write one drop into a vault's inbox, creating any subdirectory it names."""
    path = vault.inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def dirty_transcript(vault: VaultConfig, name: str, *, body: str = "x\n") -> Path:
    """Write a file into the tree ingest guards, which is what makes a vault dirty *to ingest*.

    Dirt elsewhere in the vault is deliberately tolerated, so a test that wants a refusal has to
    put it where the pass actually writes.
    """
    path = vault.root / "transcripts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def record_named(filename: str) -> InboxRecord:
    """A throwaway record, used only to shape a verdict's title and slug from a filename."""
    return InboxRecord(id="", body="", declared=DeclaredMetadata(), filename=filename, size_bytes=0)


def personal_verdict(record: InboxRecord) -> Classification:
    """A confident verdict whose title tracks the filename, so drops file to distinct paths."""
    stem = Path(record.filename).stem
    return Classification(
        classification="personal",
        confidence=0.95,
        title=stem.replace("-", " ").capitalize(),
        slug=stem,
        summary=f"Notes from {stem}.",
        model="claude-stub",
    )


def work_verdict(record: InboxRecord) -> Classification:
    stem = Path(record.filename).stem
    return Classification(
        classification="work",
        confidence=0.93,
        title=stem.replace("-", " ").capitalize(),
        slug=stem,
        summary=f"Work notes from {stem}.",
        project="PdfToExcel",
        model="claude-stub",
    )


class StubClassifier:
    """A `Classifier` that answers per filename and remembers what it was asked."""

    def __init__(self, outcomes: dict[str, Outcome] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[str] = []

    def classify(self, record: InboxRecord) -> Outcome:
        self.calls.append(record.filename)
        if record.filename in self.outcomes:
            return self.outcomes[record.filename]
        if record.unparseable:
            # Mirrors `ClaudeCliClassifier`, which holds a drop whose own metadata is broken
            # rather than spending a model call on it.
            return NeedsReview(f"the drop's own metadata could not be read: {record.unparseable}")
        return personal_verdict(record)


class StubDistiller:
    """A `Distiller` that answers from a script and remembers whether it was asked."""

    def __init__(self, outcome: DistillOutcome | None = None) -> None:
        self.outcome: DistillOutcome = PARAPHRASE if outcome is None else outcome
        self.calls: list[str] = []

    def distill(self, record: InboxRecord, verdict: Classification) -> DistillOutcome:
        self.calls.append(record.filename)
        return self.outcome


class ExplodingClassifier(StubClassifier):
    """A `Classifier` with a bug in it, for the containment the pass promises regardless."""

    def __init__(self, filename: str) -> None:
        super().__init__()
        self.filename = filename

    def classify(self, record: InboxRecord) -> Outcome:
        if record.filename == self.filename:
            raise RuntimeError("something nobody anticipated")
        return super().classify(record)


class Recorder:
    """A `Notifier` that records instead of notifying."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, job: str, reason: str, log_path: str) -> bool:
        self.calls.append((job, reason, log_path))
        return True


def ingest(scratch: Scratch, **kwargs: Any) -> IngestReport:
    """Run one pass with stubs, unless the test supplied its own."""
    kwargs.setdefault("classifier", StubClassifier())
    kwargs.setdefault("distiller", StubDistiller())
    kwargs.setdefault("notifier", None)
    kwargs.setdefault("ingested", INGESTED)
    return run_ingest(scratch.config, scratch.personal, **kwargs)


def vault_files(root: Path) -> list[str]:
    """Every filed Markdown file — the inbox, the README, and git internals aside."""
    found = []
    for path in root.rglob("*.md"):
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or ".git" in path.parts or path.name == "README.md":
            continue
        if relative.startswith("inbox/"):
            continue
        found.append(relative)
    return sorted(found)


def inbox_files(vault: VaultConfig) -> list[str]:
    return sorted(
        path.relative_to(vault.inbox).as_posix()
        for path in vault.inbox.rglob("*")
        if path.is_file()
    )


# ---------------------------------------------------------------------------------------------
# Idempotency and crash recovery come first: they are what make the two-phase order safe.
# ---------------------------------------------------------------------------------------------


class TestIdempotency:
    """AE5: running the pass twice over the same content changes nothing the second time."""

    def test_a_second_pass_produces_no_second_commit(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        ingest(scratch)
        after_first = commit_count(scratch.personal.root)

        second = ingest(scratch)

        assert commit_count(scratch.personal.root) == after_first
        assert second.commits == ()

    def test_a_second_pass_writes_no_duplicate_vault_file(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        ingest(scratch)
        filed = vault_files(scratch.personal.root)

        ingest(scratch)

        assert vault_files(scratch.personal.root) == filed
        assert len(filed) == 1

    def test_a_second_pass_asks_the_classifier_nothing(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        ingest(scratch)

        classifier = StubClassifier()
        ingest(scratch, classifier=classifier)

        assert classifier.calls == []

    def test_the_same_body_re_dropped_with_new_metadata_is_recognized(
        self, scratch: Scratch
    ) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        ingest(scratch)

        drop(
            scratch.personal,
            "coffee-again.md",
            f"---\nsource: whatsapp\ntags: [erfpacht]\n---\n\n{PERSONAL_BODY}",
        )
        report = ingest(scratch)

        assert len(report.recovered) == 1
        assert report.filed == ()
        assert len(vault_files(scratch.personal.root)) == 1
        assert inbox_files(scratch.personal) == []

    def test_two_identical_bodies_in_one_pass_file_once(self, scratch: Scratch) -> None:
        drop(scratch.personal, "a.md", PERSONAL_BODY)
        drop(scratch.personal, "b.md", PERSONAL_BODY)

        report = ingest(scratch)

        assert len(report.filed) == 1
        assert len(report.recovered) == 1
        assert inbox_files(scratch.personal) == []

    def test_a_hand_written_vault_file_counts_as_filed(self, scratch: Scratch) -> None:
        """The id index is derived from files, so a file placed by any means is authoritative."""
        record = drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        identity = discover(scratch.personal)[0].id
        filed = scratch.personal.root / "transcripts" / "2026" / "08" / "by-hand.md"
        filed.parent.mkdir(parents=True)
        filed.write_text(f"---\ntitle: By hand\ncontent_id: {identity}\n---\n\nx\n", "utf-8")
        git(scratch.personal.root, "add", "-A")
        git(scratch.personal.root, "commit", "-q", "-m", "filed by hand")

        report = ingest(scratch)

        assert report.filed == ()
        assert len(report.recovered) == 1
        assert not record.exists()


class TestCrashRecovery:
    """A crash between the vault write and the inbox removal must cost nothing."""

    def crash(self, scratch: Scratch) -> InboxRecord:
        """Leave the vault exactly as an interrupted pass would: filed, but not drained."""
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        record = discover(scratch.personal)[0]
        verdict = personal_verdict(record)
        routing = route(record, verdict, scratch.config, scratch.personal)
        file_record(record, verdict, routing, scratch.personal.root, ingested=INGESTED)
        git(scratch.personal.root, "add", "-A")
        git(scratch.personal.root, "commit", "-q", "-m", "interrupted pass")
        return record

    def test_the_next_pass_recognizes_the_id_as_filed(self, scratch: Scratch) -> None:
        record = self.crash(scratch)

        report = ingest(scratch)

        assert report.filed == ()
        assert [item.record_id for item in report.recovered] == [record.id]

    def test_the_next_pass_removes_the_inbox_entry(self, scratch: Scratch) -> None:
        self.crash(scratch)

        ingest(scratch)

        assert inbox_files(scratch.personal) == []

    def test_the_next_pass_does_not_duplicate_the_vault_file(self, scratch: Scratch) -> None:
        self.crash(scratch)
        before = vault_files(scratch.personal.root)

        ingest(scratch)

        assert vault_files(scratch.personal.root) == before

    def test_the_recovery_is_itself_a_commit(self, scratch: Scratch) -> None:
        self.crash(scratch)
        before = commit_count(scratch.personal.root)

        report = ingest(scratch)

        assert commit_count(scratch.personal.root) == before + 1
        assert len(report.commits) == 1
        assert "D\tinbox/coffee.md" in show(scratch.personal.root)


# ---------------------------------------------------------------------------------------------
# The happy path.
# ---------------------------------------------------------------------------------------------


class TestFullPass:
    """AE1: drops go in, filed Markdown and an empty inbox come out, in one commit."""

    @pytest.fixture
    def three(self, scratch: Scratch) -> IngestReport:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        drop(scratch.personal, "reading.txt", "Finished the Ishiguro novel on the train.")
        drop(scratch.personal, "2026-08/whatsapp/plans", "Deniz suggested Sunday for the move.")
        return ingest(scratch)

    def test_every_drop_is_filed(self, scratch: Scratch, three: IngestReport) -> None:
        assert len(three.filed) == 3
        assert len(vault_files(scratch.personal.root)) == 3
        assert all(item.path.is_file() for item in three.filed)

    def test_the_inbox_drains_to_empty(self, scratch: Scratch, three: IngestReport) -> None:
        assert inbox_files(scratch.personal) == []

    def test_emptied_subdirectories_are_pruned(self, scratch: Scratch, three: IngestReport) -> None:
        assert list(scratch.personal.inbox.iterdir()) == []

    def test_the_pass_makes_exactly_one_commit(self, scratch: Scratch, three: IngestReport) -> None:
        assert len(three.commits) == 1
        assert commit_count(scratch.personal.root) == 2

    def test_the_commit_shows_every_vault_change_the_pass_made(
        self, scratch: Scratch, three: IngestReport
    ) -> None:
        diff = show(scratch.personal.root)

        for item in three.filed:
            assert f"A\t{item.relative_path}" in diff

    def test_a_committed_drop_leaves_its_deletion_in_the_diff(self, scratch: Scratch) -> None:
        """A feeder commits what it drops, so the drain is visible as a deletion (R7)."""
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        git(scratch.personal.root, "add", "-A")
        git(scratch.personal.root, "commit", "-q", "-m", "a feeder drops a note")

        ingest(scratch)
        diff = show(scratch.personal.root)

        assert "D\tinbox/coffee.md" in diff
        assert "A\ttranscripts/2026/08/2026-08-01-coffee.md" in diff

    def test_the_filed_file_carries_its_provenance(
        self, scratch: Scratch, three: IngestReport
    ) -> None:
        filed = next(item for item in three.filed if item.title == "Coffee")
        text = filed.path.read_text(encoding="utf-8")

        assert "original_filename: coffee.md" in text
        assert "classifier_model: claude-stub" in text
        assert PERSONAL_BODY in text

    def test_the_run_reports_success(self, scratch: Scratch, three: IngestReport) -> None:
        assert three.status is IngestStatus.OK
        assert three.exit_code == EXIT_OK
        assert three.failure is None

    def test_the_vault_is_clean_afterwards(self, scratch: Scratch, three: IngestReport) -> None:
        assert git(scratch.personal.root, "status", "--porcelain") == ""

    def test_an_uncommitted_drop_files_and_drains_without_a_pathspec_error(
        self, scratch: Scratch
    ) -> None:
        """A hand-pasted drop is never committed by anyone; the pass must survive it."""
        drop(scratch.personal, "pasted.txt", PERSONAL_BODY)

        report = ingest(scratch)

        assert report.failure is None
        assert len(report.filed) == 1
        assert inbox_files(scratch.personal) == []


class TestCommitMessage:
    """The message is the review surface for anyone reading `git log` a year from now."""

    def test_it_names_the_count_and_the_vault(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        drop(scratch.personal, "reading.txt", "Finished the Ishiguro novel.")

        ingest(scratch)

        assert subjects(scratch.personal.root)[0] == "mem: ingest 2 item(s) into personal"

    def test_it_lists_every_filed_title_and_path(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)

        report = ingest(scratch)
        body = git(scratch.personal.root, "log", "-1", "--format=%b")

        assert "Filed 1 into personal:" in body
        assert f"- {report.filed[0].summary_line()}" in body

    def test_it_lists_held_items_with_their_reasons(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        drop(scratch.personal, "murky.txt", "???")
        classifier = StubClassifier({"murky.txt": NeedsReview("confidence 0.30 is too low")})

        ingest(scratch, classifier=classifier)
        body = git(scratch.personal.root, "log", "-1", "--format=%b")

        assert "Held in the inbox for review: 1" in body
        assert "- murky.txt — confidence 0.30 is too low" in body

    def test_a_recovery_only_pass_says_so_in_its_subject(self) -> None:
        report = IngestReport(
            vault="personal",
            recovered=(DrainedItem("a.md", "abc", "its content is already filed in the vault"),),
        )

        assert build_commit_message(report).startswith(
            "mem: ingest — 1 inbox entry(s) already filed in personal"
        )

    def test_a_hold_only_pass_says_so_in_its_subject(self) -> None:
        report = IngestReport(vault="personal", held=(HeldItem("a.md", "abc", "unreadable"),))

        assert build_commit_message(report).startswith(
            "mem: ingest — 1 item(s) held in the personal inbox"
        )

    def test_blocked_notes_are_named(self) -> None:
        report = IngestReport(
            vault="personal",
            note_failures=(NoteFailure("a.md", "abc", "the note reproduced 92% of a sentence"),),
        )

        assert "Work notes blocked: 1" in build_commit_message(report)
        assert "- a.md — the note reproduced 92% of a sentence" in build_commit_message(report)


class TestNeedsReview:
    """AE4: an item the classifier cannot place stays put, and says why."""

    @pytest.fixture
    def partial(self, scratch: Scratch) -> IngestReport:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        drop(scratch.personal, "reading.txt", "Finished the Ishiguro novel.")
        drop(scratch.personal, "murky.md", "???")
        classifier = StubClassifier(
            {"murky.md": NeedsReview("confidence 0.30 is below the threshold 0.70", 0.3)}
        )
        return ingest(scratch, classifier=classifier)

    def test_the_other_two_file_and_drain(self, scratch: Scratch, partial: IngestReport) -> None:
        assert len(partial.filed) == 2
        assert inbox_files(scratch.personal) == ["murky.md", f"murky.md{MARKER_SUFFIX}"]

    def test_the_held_item_gets_no_vault_file(
        self, scratch: Scratch, partial: IngestReport
    ) -> None:
        assert len(vault_files(scratch.personal.root)) == 2
        assert "murky" not in " ".join(vault_files(scratch.personal.root))

    def test_the_marker_names_the_reason_on_one_line(
        self, scratch: Scratch, partial: IngestReport
    ) -> None:
        marker = marker_for(scratch.personal.inbox / "murky.md")
        text = marker.read_text(encoding="utf-8")

        assert text == "confidence 0.30 is below the threshold 0.70\n"
        assert len(text.splitlines()) == 1

    def test_the_exit_status_signals_partial_completion(self, partial: IngestReport) -> None:
        assert partial.status is IngestStatus.PARTIAL
        assert partial.exit_code == EXIT_PARTIAL
        assert partial.failure is None

    def test_the_pass_still_commits_the_work_it_did(
        self, scratch: Scratch, partial: IngestReport
    ) -> None:
        assert len(partial.commits) == 1
        assert f"A\tinbox/murky.md{MARKER_SUFFIX}" in show(scratch.personal.root)

    def test_the_marker_is_not_mistaken_for_a_drop_next_pass(
        self, scratch: Scratch, partial: IngestReport
    ) -> None:
        classifier = StubClassifier(
            {"murky.md": NeedsReview("confidence 0.30 is below the threshold 0.70", 0.3)}
        )

        second = ingest(scratch, classifier=classifier)

        assert classifier.calls == ["murky.md"]
        assert second.commits == ()

    def test_fixing_the_item_files_it_and_removes_its_marker(
        self, scratch: Scratch, partial: IngestReport
    ) -> None:
        second = ingest(scratch)

        assert len(second.filed) == 1
        assert inbox_files(scratch.personal) == []
        assert not marker_for(scratch.personal.inbox / "murky.md").exists()

    def test_an_unparseable_drop_is_held_rather_than_filed(self, scratch: Scratch) -> None:
        drop(scratch.personal, "broken.md", "---\ndate: soon\n---\n\nSomething happened.")

        report = ingest(scratch)

        assert len(report.held) == 1
        assert "date" in report.held[0].reason
        assert vault_files(scratch.personal.root) == []

    def test_a_write_failure_holds_only_its_own_drop(
        self, scratch: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Containment (R5): one drop that cannot be written must not stop the other nine."""
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        drop(scratch.personal, "odd.md", "Another note entirely.")
        real = file_record

        def refuse(record: InboxRecord, *args: Any, **kwargs: Any) -> Any:
            if record.filename == "odd.md":
                raise WriteError("the destination directory is read-only")
            return real(record, *args, **kwargs)

        monkeypatch.setattr("memvault.ingest.file_record", refuse)

        report = ingest(scratch)

        assert len(report.filed) == 1
        assert len(report.held) == 1
        assert "the vault write failed" in report.held[0].reason
        assert inbox_files(scratch.personal) == ["odd.md", f"odd.md{MARKER_SUFFIX}"]

    def test_an_unexpected_error_holds_only_its_own_drop(self, scratch: Scratch) -> None:
        """Containment again, for the failure nobody anticipated: a classifier that raises."""
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        drop(scratch.personal, "odd.md", "Another note entirely.")

        report = ingest(scratch, classifier=ExplodingClassifier("odd.md"))

        assert len(report.filed) == 1
        assert len(report.held) == 1
        assert "RuntimeError" in report.held[0].reason
        assert report.status is IngestStatus.PARTIAL


class TestEmptyInbox:
    def test_it_is_a_no_op_with_no_commit(self, scratch: Scratch) -> None:
        before = commit_count(scratch.personal.root)

        report = ingest(scratch)

        assert report.commits == ()
        assert commit_count(scratch.personal.root) == before
        assert report.status is IngestStatus.OK
        assert report.exit_code == EXIT_OK

    def test_the_classifier_is_never_reached(self, scratch: Scratch) -> None:
        classifier = StubClassifier()

        ingest(scratch, classifier=classifier)

        assert classifier.calls == []

    def test_a_dirty_vault_with_an_empty_inbox_is_still_a_no_op(self, scratch: Scratch) -> None:
        """Nothing is about to be committed, so there is nothing for cleanliness to protect."""
        (scratch.personal.root / "stray.md").write_text("x\n", encoding="utf-8")

        report = ingest(scratch)

        assert report.failure is None
        assert report.commits == ()


class TestDirtyVault:
    """The commit must contain the pass's work and nothing else, so the pass refuses to start."""

    def test_dirt_outside_the_transcript_directory_does_not_stop_the_pass(
        self, scratch: Scratch
    ) -> None:
        """Ingest writes transcripts. A half-written note under repos/ is not its business."""
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        memory_dir = scratch.personal.root / "repos" / "Chad"
        memory_dir.mkdir(parents=True)
        (memory_dir / "learnings.md").write_text("half a thought\n", encoding="utf-8")

        report = ingest(scratch)

        assert report.failure is None
        assert report.filed != ()

    def test_an_untracked_file_in_the_transcript_tree_stops_the_pass(
        self, scratch: Scratch
    ) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        dirty_transcript(scratch.personal, "stray.md")

        report = ingest(scratch)

        assert report.failure is not None
        assert "transcripts/stray.md" in report.failure
        assert report.exit_code == EXIT_FAILED

    def test_a_refusal_says_it_is_a_refusal(self, scratch: Scratch) -> None:
        """Retry-later versus broken (R7): the debounce runner branches on exactly this."""
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        dirty_transcript(scratch.personal, "stray.md")

        report = ingest(scratch)

        assert report.refused is True
        assert report.dirty is not None

    def test_an_ordinary_failure_is_not_a_refusal(self, scratch: Scratch) -> None:
        """A failed commit must not be retried quietly forever; only a dirty tree earns that."""
        report = IngestReport(vault="personal", failure="git commit failed")

        assert report.refused is False

    def test_an_empty_inbox_still_reports_the_dirt_it_found(self, scratch: Scratch) -> None:
        """Under idle-triggering most passes file nothing, and the old order asked nothing.

        A vault that would refuse the moment a drop arrived logged the same "inbox is empty"
        line as a healthy one, every time, for as long as the dirt sat there.
        """
        dirty_transcript(scratch.personal, "stray.md")

        report = ingest(scratch)

        assert report.dirty is not None
        assert "transcripts/stray.md" in report.dirty

    def test_an_empty_inbox_over_a_dirty_tree_is_still_not_a_failure(
        self, scratch: Scratch
    ) -> None:
        """Nothing is about to be committed, so there is nothing for cleanliness to protect."""
        dirty_transcript(scratch.personal, "stray.md")

        report = ingest(scratch)

        assert report.failure is None
        assert report.refused is False
        assert report.exit_code == EXIT_OK

    def test_a_clean_empty_pass_claims_no_dirt(self, scratch: Scratch) -> None:
        assert ingest(scratch).dirty is None

    def test_nothing_is_written_before_it_refuses(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        dirty_transcript(scratch.personal, "stray.md")
        classifier = StubClassifier()

        ingest(scratch, classifier=classifier)

        assert classifier.calls == []
        assert vault_files(scratch.personal.root) == ["transcripts/stray.md"]
        assert inbox_files(scratch.personal) == ["coffee.md"]

    def test_a_modified_tracked_transcript_stops_it_too(self, scratch: Scratch) -> None:
        dirty_transcript(scratch.personal, "earlier.md", body="# earlier\n")
        git(scratch.personal.root, "add", "-A")
        git(scratch.personal.root, "commit", "-q", "-m", "seed a transcript")
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        dirty_transcript(scratch.personal, "earlier.md", body="# edited\n")

        assert ingest(scratch).failure is not None

    def test_a_dirty_note_directory_in_the_work_vault_stops_it(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        notes = scratch.work.root / "notes"
        notes.mkdir(parents=True, exist_ok=True)
        (notes / "stray.md").write_text("x\n", encoding="utf-8")

        report = ingest(scratch)

        assert report.failure is not None
        assert "'work'" in report.failure

    def test_work_vault_dirt_outside_the_note_directory_does_not_stop_it(
        self, scratch: Scratch
    ) -> None:
        """A route target is somebody's active working tree; its unrelated work is not ours.

        The pass's commit there is one note staged by exact path, so uncommitted work elsewhere
        in that repo cannot ride along — and demanding it be clean would refuse the pass for
        reasons that have nothing to do with it.
        """
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        (scratch.work.root / "tickets").mkdir(parents=True, exist_ok=True)
        (scratch.work.root / "tickets" / "TICKET-1.md").write_text("wip\n", encoding="utf-8")

        report = ingest(scratch)

        assert report.failure is None
        assert len(report.filed) == 1

    def test_unrelated_work_vault_dirt_stays_out_of_the_note_commit(self, scratch: Scratch) -> None:
        drop(scratch.personal, "sprint.md", WORK_BODY)
        (scratch.work.root / "tickets").mkdir(parents=True, exist_ok=True)
        (scratch.work.root / "tickets" / "TICKET-1.md").write_text("wip\n", encoding="utf-8")

        ingest(scratch)

        assert "TICKET-1" not in show(scratch.work.root)

    def test_an_uncommitted_inbox_drop_is_not_dirt(self, scratch: Scratch) -> None:
        """The documented minimum drop is a pasted file nobody committed."""
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)

        report = ingest(scratch)

        assert report.failure is None
        assert len(report.filed) == 1


class TestGitFailure:
    """A failed commit must leave the inbox exactly as it found it, so the run is retryable."""

    @pytest.fixture
    def broken(self, scratch: Scratch, monkeypatch: pytest.MonkeyPatch) -> IngestReport:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)

        def explode(*args: Any, **kwargs: Any) -> None:
            raise GitError("index.lock exists")

        monkeypatch.setattr("memvault.ingest.commit", explode)
        return ingest(scratch)

    def test_the_inbox_entries_are_put_back(self, scratch: Scratch, broken: IngestReport) -> None:
        assert inbox_files(scratch.personal) == ["coffee.md"]
        assert (scratch.personal.inbox / "coffee.md").read_text(encoding="utf-8") == PERSONAL_BODY

    def test_the_failure_is_reported_rather_than_raised(self, broken: IngestReport) -> None:
        assert broken.failure is not None
        assert "index.lock" in broken.failure
        assert broken.status is IngestStatus.FAILED
        assert broken.exit_code == EXIT_FAILED

    def test_no_commit_was_recorded(self, scratch: Scratch, broken: IngestReport) -> None:
        assert broken.commits == ()
        assert commit_count(scratch.personal.root) == 1

    def test_the_retry_refuses_because_the_vault_is_now_dirty(
        self, scratch: Scratch, broken: IngestReport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The vault file it wrote is still there; a human decides whether to keep it."""
        monkeypatch.undo()
        retry = ingest(scratch)

        assert retry.failure is not None
        assert "uncommitted changes" in retry.failure

    def test_committing_the_leftovers_makes_the_retry_a_clean_recovery(
        self, scratch: Scratch, broken: IngestReport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.undo()
        git(scratch.personal.root, "add", "-A")
        git(scratch.personal.root, "commit", "-q", "-m", "salvage")

        retry = ingest(scratch)

        assert retry.filed == ()
        assert len(retry.recovered) == 1
        assert inbox_files(scratch.personal) == []


class TestWorkRouting:
    """R9 through the pass: raw to the personal vault, distilled notes to the work vault."""

    def test_a_work_drop_files_raw_and_distils_a_note(self, scratch: Scratch) -> None:
        drop(scratch.personal, "standup.md", WORK_BODY)
        classifier = StubClassifier({"standup.md": work_verdict(record_named("standup.md"))})

        report = ingest(scratch, classifier=classifier)

        assert len(report.filed) == 1
        assert report.filed[0].vault == "personal"
        assert len(report.notes) == 1
        assert report.notes[0].vault == "work"

    def test_both_vaults_get_a_commit(self, scratch: Scratch) -> None:
        drop(scratch.personal, "standup.md", WORK_BODY)
        classifier = StubClassifier({"standup.md": work_verdict(record_named("standup.md"))})

        report = ingest(scratch, classifier=classifier)

        assert {reference.vault for reference in report.commits} == {"personal", "work"}
        assert commit_count(scratch.personal.root) == 2
        assert commit_count(scratch.work.root) == 2

    def test_the_work_commit_message_names_nothing_from_the_personal_vault(
        self, scratch: Scratch
    ) -> None:
        """A transcript's path is its title slugged, so the full ledger must not cross either."""
        drop(scratch.personal, "acme-layoffs.md", WORK_BODY)
        classifier = StubClassifier({"acme-layoffs.md": work_verdict(record_named("acme-layoffs"))})

        ingest(scratch, classifier=classifier)
        message = git(scratch.work.root, "log", "-1", "--format=%B")

        assert message.startswith("mem: ingest 1 distilled note(s) into work")
        assert "acme" not in message
        assert "transcripts/" not in message
        assert "notes/2026/2026-08-01-price-list-tool-priorities.md" in message

    def test_no_raw_sentence_reaches_the_work_vault(self, scratch: Scratch) -> None:
        drop(scratch.personal, "standup.md", WORK_BODY)
        classifier = StubClassifier({"standup.md": work_verdict(record_named("standup.md"))})

        ingest(scratch, classifier=classifier)

        corpus = "\n".join(
            path.read_text(encoding="utf-8")
            for path in scratch.work.root.rglob("*.md")
            if ".git" not in path.parts
        )
        assert LEAKED_SENTENCE not in corpus

    def test_a_leaking_note_blocks_only_the_work_write(self, scratch: Scratch) -> None:
        drop(scratch.personal, "standup.md", WORK_BODY)
        classifier = StubClassifier({"standup.md": work_verdict(record_named("standup.md"))})
        leaky = StubDistiller(
            Distillation(title="Standup", decisions=(LEAKED_SENTENCE,), model="claude-stub")
        )

        report = ingest(scratch, classifier=classifier, distiller=leaky)

        assert len(report.filed) == 1
        assert report.notes == ()
        assert len(report.note_failures) == 1
        assert "verbatim" in report.note_failures[0].reason
        assert commit_count(scratch.work.root) == 1

    def test_a_blocked_note_still_drains_the_inbox_entry(self, scratch: Scratch) -> None:
        """The raw filing succeeded, which is what the inbox contract's promise is about."""
        drop(scratch.personal, "standup.md", WORK_BODY)
        classifier = StubClassifier({"standup.md": work_verdict(record_named("standup.md"))})
        leaky = StubDistiller(
            Distillation(title="Standup", decisions=(LEAKED_SENTENCE,), model="claude-stub")
        )

        report = ingest(scratch, classifier=classifier, distiller=leaky)

        assert inbox_files(scratch.personal) == []
        assert report.status is IngestStatus.PARTIAL

    def test_a_failed_distiller_costs_a_note_and_not_a_memory(self, scratch: Scratch) -> None:
        drop(scratch.personal, "standup.md", WORK_BODY)
        classifier = StubClassifier({"standup.md": work_verdict(record_named("standup.md"))})
        broken = StubDistiller(DistillationFailed("the distiller exited 1: no error output"))

        report = ingest(scratch, classifier=classifier, distiller=broken)

        assert len(report.filed) == 1
        assert len(report.note_failures) == 1
        assert vault_files(scratch.work.root) == []

    def test_an_empty_distillation_is_an_ordinary_day(self, scratch: Scratch) -> None:
        drop(scratch.personal, "standup.md", WORK_BODY)
        classifier = StubClassifier({"standup.md": work_verdict(record_named("standup.md"))})
        quiet = StubDistiller(Distillation(title="Standup", model="claude-stub"))

        report = ingest(scratch, classifier=classifier, distiller=quiet)

        assert report.notes == ()
        assert report.note_failures == ()
        assert report.skipped_notes and "no decisions" in report.skipped_notes[0]
        assert report.status is IngestStatus.OK

    def test_a_personal_drop_never_reaches_the_distiller(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        distiller = StubDistiller()

        ingest(scratch, distiller=distiller)

        assert distiller.calls == []


class TestLocalOnly:
    """AE2: a private drop is filed where git will never push it, and the pass survives that."""

    @pytest.fixture
    def private(self, scratch: Scratch) -> IngestReport:
        (scratch.personal.root / ".gitignore").write_text("private/\n", encoding="utf-8")
        git(scratch.personal.root, "add", "-A")
        git(scratch.personal.root, "commit", "-q", "-m", "ignore the private prefix")
        drop(scratch.personal, "diary.md", "---\nlocal_only: true\n---\n\nA private thought.")
        return ingest(scratch)

    def test_it_lands_under_the_gitignored_prefix(
        self, scratch: Scratch, private: IngestReport
    ) -> None:
        assert len(private.filed) == 1
        assert private.filed[0].relative_path.startswith("private/")
        assert private.filed[0].local_only

    def test_the_pass_does_not_fail_on_the_ignored_path(self, private: IngestReport) -> None:
        assert private.failure is None
        assert private.status is IngestStatus.OK

    def test_the_private_file_is_not_in_the_commit(
        self, scratch: Scratch, private: IngestReport
    ) -> None:
        assert "private/" not in show(scratch.personal.root)

    def test_the_inbox_still_drains(self, scratch: Scratch, private: IngestReport) -> None:
        assert inbox_files(scratch.personal) == []

    def test_a_second_pass_still_recognizes_it_as_filed(
        self, scratch: Scratch, private: IngestReport
    ) -> None:
        drop(scratch.personal, "diary-again.md", "A private thought.")

        second = ingest(scratch)

        assert second.filed == ()
        assert len(second.recovered) == 1


class TestGuardConfiguration:
    """U5's guard dials are config, and the pass is what threads them through."""

    def test_the_config_defaults_match_the_guards_own(self, scratch: Scratch) -> None:
        assert scratch.config.distill.overlap_threshold == distill_module.DEFAULT_OVERLAP_THRESHOLD
        assert scratch.config.distill.shingle_size == distill_module.DEFAULT_SHINGLE_SIZE

    def test_a_relaxed_threshold_lets_a_copied_sentence_through(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path,
            make_repo(tmp_path, "personal"),
            make_repo(tmp_path, "work"),
            distill={"overlap_threshold": 1.0},
        )
        config = load_config(path)
        loose = Scratch(config, config.vault("personal"), config.vault("work"), path)
        drop(loose.personal, "standup.md", WORK_BODY)
        classifier = StubClassifier({"standup.md": work_verdict(record_named("standup.md"))})
        leaky = StubDistiller(
            Distillation(title="Standup", decisions=(LEAKED_SENTENCE,), model="claude-stub")
        )

        report = ingest(loose, classifier=classifier, distiller=leaky)

        assert len(report.notes) == 1
        assert report.note_failures == ()

    def test_both_dials_reach_the_distiller(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write_config(
            tmp_path,
            make_repo(tmp_path, "personal"),
            make_repo(tmp_path, "work"),
            distill={"overlap_threshold": 0.15, "shingle_size": 9},
        )
        config = load_config(path)
        tuned = Scratch(config, config.vault("personal"), config.vault("work"), path)
        drop(tuned.personal, "standup.md", WORK_BODY)
        seen: dict[str, Any] = {}

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return distill_module.distill_record(*args, **kwargs)

        monkeypatch.setattr("memvault.ingest.distill_record", spy)
        classifier = StubClassifier({"standup.md": work_verdict(record_named("standup.md"))})

        ingest(tuned, classifier=classifier)

        assert seen["overlap_threshold"] == 0.15
        assert seen["shingle_size"] == 9


class TestNotification:
    """Failures notify; success is silent."""

    def test_a_held_item_notifies(self, scratch: Scratch) -> None:
        drop(scratch.personal, "murky.md", "???")
        recorder = Recorder()
        classifier = StubClassifier({"murky.md": NeedsReview("nothing legible in it")})

        ingest(scratch, classifier=classifier, notifier=recorder, log_path="/tmp/ingest.log")

        assert len(recorder.calls) == 1
        job, reason, log_path = recorder.calls[0]
        assert job == "ingest"
        assert "nothing legible in it" in reason
        assert log_path == "/tmp/ingest.log"

    def test_a_clean_pass_is_silent(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        recorder = Recorder()

        ingest(scratch, notifier=recorder)

        assert recorder.calls == []

    def test_a_failed_pass_notifies_with_the_failure(self, scratch: Scratch) -> None:
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        dirty_transcript(scratch.personal, "stray.md")
        recorder = Recorder()

        ingest(scratch, notifier=recorder)

        assert len(recorder.calls) == 1
        assert "uncommitted changes" in recorder.calls[0][1]


class TestFiledIdScan:
    """The id index is read off the files, which is what makes it impossible to go stale."""

    def test_a_file_without_a_content_id_contributes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.md"
        path.write_text("# Just a note\n", encoding="utf-8")

        assert content_id_of(path) is None

    def test_broken_frontmatter_reads_as_no_id_rather_than_raising(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.md"
        path.write_text("---\ncontent_id: [unclosed\n---\n\nbody\n", encoding="utf-8")

        assert content_id_of(path) is None

    def test_an_id_is_read_off_the_frontmatter(self, tmp_path: Path) -> None:
        path = tmp_path / "filed.md"
        path.write_text("---\ntitle: X\ncontent_id: abc123\n---\n\nbody\n", encoding="utf-8")

        assert content_id_of(path) == "abc123"

    def test_the_inbox_is_excluded_from_the_scan(self, scratch: Scratch) -> None:
        """A drop cannot declare itself already filed by carrying a `content_id`."""
        drop(scratch.personal, "sneaky.md", "---\ncontent_id: abc123\n---\n\nA real memory.")

        assert filed_ids(scratch.personal) == set()

    def test_filed_files_are_found_at_any_depth(self, scratch: Scratch) -> None:
        filed = scratch.personal.root / "transcripts" / "2026" / "08" / "x.md"
        filed.parent.mkdir(parents=True)
        filed.write_text("---\ncontent_id: deadbeef\n---\n\nbody\n", encoding="utf-8")

        assert filed_ids(scratch.personal) == {"deadbeef"}


class TestIgnoreDirty:
    """A vault that is also a memory store has other automation writing under it constantly."""

    def _scratch_ignoring(self, tmp_path: Path, *prefixes: str) -> Scratch:
        personal = make_repo(tmp_path, "personal")
        work = make_repo(tmp_path, "work")
        data = {
            "default_vault": "personal",
            "vaults": {
                "personal": {
                    "root": str(personal),
                    "work_route": "work",
                    "ignore_dirty": list(prefixes),
                },
                "work": {"root": str(work)},
            },
        }
        path = tmp_path / "memvault.config.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        config = load_config(path)
        return Scratch(config, config.vault("personal"), config.vault("work"), path)

    def test_dirt_under_an_ignored_prefix_does_not_stop_the_pass(self, tmp_path: Path) -> None:
        scratch = self._scratch_ignoring(tmp_path, "memory/auto")
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        churn = scratch.personal.root / "memory" / "auto" / "Thing"
        churn.mkdir(parents=True)
        (churn / "fact.md").write_text("rewritten by another job\n", encoding="utf-8")

        report = ingest(scratch)

        assert report.failure is None
        assert len(report.filed) == 1

    def test_the_ignored_path_stays_out_of_the_commit(self, tmp_path: Path) -> None:
        scratch = self._scratch_ignoring(tmp_path, "memory/auto")
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        churn = scratch.personal.root / "memory" / "auto" / "Thing"
        churn.mkdir(parents=True)
        (churn / "fact.md").write_text("rewritten by another job\n", encoding="utf-8")

        ingest(scratch)

        assert "memory/auto" not in show(scratch.personal.root)

    def test_dirt_outside_the_ignored_prefix_still_stops_the_pass(self, tmp_path: Path) -> None:
        """`ignore_dirty` forgives its own prefixes, not the tree the pass writes into."""
        scratch = self._scratch_ignoring(tmp_path, "memory/auto")
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)
        dirty_transcript(scratch.personal, "stray.md")

        report = ingest(scratch)

        assert report.failure is not None


# --- Segmentation -----------------------------------------------------------------------
#
# The theme: splitting changes how many things get classified, and nothing else. Raw bodies
# still reach only the personal vault, the work vault still receives distilled notes only, and
# the drop still leaves the inbox exactly once everything it contained has landed.


class BlockSegmenter:
    """Splits on blank lines, so a test drop's paragraphs become its segments."""

    def __init__(self, topics: tuple[str, ...] = ()) -> None:
        self.topics = topics
        self.calls: list[str] = []

    def segment(self, record: InboxRecord, max_segments: int) -> Any:
        self.calls.append(record.filename)
        lines = record.body.split("\n")
        triples: list[tuple[int, int, str]] = []
        start = 1
        for index, line in enumerate(lines, start=1):
            if line.strip() == "" and start <= index - 1:
                topic = self.topics[len(triples)] if len(triples) < len(self.topics) else ""
                triples.append((start, index, topic))
                start = index + 1
        if start <= len(lines):
            topic = self.topics[len(triples)] if len(triples) < len(self.topics) else ""
            triples.append((start, len(lines), topic))
        return segments_from_ranges(record, lines, triples)


class BodyClassifier:
    """A `Classifier` answering on what the body says, since segments share a filename."""

    def __init__(self, confidences: dict[str, float] | None = None) -> None:
        self.confidences = confidences or {}
        self.bodies: list[str] = []

    def classify(self, record: InboxRecord) -> Outcome:
        self.bodies.append(record.body)
        head = record.body.split("\n")[0].strip().rstrip(".")
        slug = head.lower().replace(" ", "-")[:40] or "empty"
        work = "work" in record.body.lower() or "jordi" in record.body.lower()
        confidence = self.confidences.get(head, 0.95)
        return Classification(
            classification="work" if work else "personal",
            confidence=confidence,
            title=head[:60] or "Untitled",
            slug=slug,
            summary=f"About {head[:40]}.",
            model="claude-stub",
        )


THREE_TOPICS = (
    "Allotment shed\nWe repainted it on Saturday and the soil was still wet.\n"
    "\n"
    "Moving in\nWe decided to wait until the spring before deciding anything.\n"
    "\n"
    "Sam and the work pipeline\nThe importer rewrite slipped again this week."
)


def segmented(tmp_path: Path, **sections: Any) -> Scratch:
    personal, work = make_repo(tmp_path, "personal"), make_repo(tmp_path, "work")
    base: dict[str, Any] = {"segment": {"enabled": True, "min_chars": 0}}
    base.update(sections)
    path = write_config(tmp_path, personal, work, **base)
    config = load_config(path)
    return Scratch(config, config.vault("personal"), config.vault("work"), path)


def run(scratch: Scratch, **kwargs: Any) -> IngestReport:
    kwargs.setdefault("classifier", BodyClassifier())
    kwargs.setdefault("segmenter", BlockSegmenter())
    kwargs.setdefault("distiller", StubDistiller(PARAPHRASE))
    return run_ingest(
        scratch.config,
        scratch.personal,
        ingested=INGESTED,
        notifier=None,
        **kwargs,
    )


class TestSegmentedIngest:
    def test_a_three_topic_drop_files_three_raw_items_and_one_work_note(
        self, tmp_path: Path
    ) -> None:
        scratch = segmented(tmp_path)
        drop(scratch.personal, "conversation.md", THREE_TOPICS)

        report = run(scratch)

        assert len(report.filed) == 3
        assert all(f.vault == "personal" for f in report.filed)
        assert len(report.notes) == 1
        assert report.notes[0].vault == "work"

    def test_no_verbatim_sentence_of_any_segment_reaches_the_work_vault(
        self, tmp_path: Path
    ) -> None:
        scratch = segmented(tmp_path)
        drop(scratch.personal, "conversation.md", THREE_TOPICS)

        run(scratch)

        work_text = "\n".join(
            p.read_text(encoding="utf-8") for p in scratch.work.root.rglob("*.md")
        )
        for sentence in (
            "We repainted it on Saturday and the soil was still wet",
            "We decided to wait until the spring before deciding anything",
            "The importer rewrite slipped again this week",
        ):
            assert sentence not in work_text

    def test_filed_segments_carry_their_parent_and_position(self, tmp_path: Path) -> None:
        scratch = segmented(tmp_path)
        drop(scratch.personal, "conversation.md", THREE_TOPICS)

        report = run(scratch)

        parents = set()
        for written in report.filed:
            front = yaml.safe_load(written.path.read_text(encoding="utf-8").split("---")[1])
            assert front["segment_count"] == 3
            assert front["segment_index"] in (1, 2, 3)
            parents.add(front["parent_id"])
        assert len(parents) == 1

    def test_an_unsegmented_drop_gains_no_segment_frontmatter(self, tmp_path: Path) -> None:
        scratch = segmented(tmp_path, segment={"enabled": False})
        drop(scratch.personal, "single.md", PERSONAL_BODY)

        report = run(scratch)

        front = yaml.safe_load(report.filed[0].path.read_text(encoding="utf-8").split("---")[1])
        assert "parent_id" not in front
        assert "segment_index" not in front
        assert "segment_count" not in front

    def test_a_declared_classification_skips_segmentation(self, tmp_path: Path) -> None:
        scratch = segmented(tmp_path)
        drop(
            scratch.personal, "declared.md", "---\nclassification: personal\n---\n\n" + THREE_TOPICS
        )
        segmenter = BlockSegmenter()

        report = run(scratch, segmenter=segmenter)

        assert len(report.filed) == 1
        assert segmenter.calls == []

    def test_everything_lands_in_one_commit(self, tmp_path: Path) -> None:
        scratch = segmented(tmp_path)
        before = commit_count(scratch.personal.root)
        drop(scratch.personal, "conversation.md", THREE_TOPICS)

        run(scratch)

        assert commit_count(scratch.personal.root) == before + 1

    def test_the_inbox_drains_once_every_segment_is_filed(self, tmp_path: Path) -> None:
        scratch = segmented(tmp_path)
        drop(scratch.personal, "conversation.md", THREE_TOPICS)

        run(scratch)

        assert discover(scratch.personal) == []

    def test_a_rerun_files_nothing_and_makes_no_commit(self, tmp_path: Path) -> None:
        scratch = segmented(tmp_path)
        drop(scratch.personal, "conversation.md", THREE_TOPICS)
        run(scratch)
        after_first = commit_count(scratch.personal.root)

        drop(scratch.personal, "conversation.md", THREE_TOPICS)
        second = run(scratch)

        # The re-dropped file is created and then drained inside the same pass, so git sees a
        # net-zero diff and there is nothing to commit. Re-syncing an unchanged conversation is
        # therefore invisible in the log rather than a run of empty commits — which is what
        # makes a repeated full phone backup safe to feed straight back in.
        assert second.filed == ()
        assert commit_count(scratch.personal.root) == after_first
        assert discover(scratch.personal) == []

    def test_a_low_confidence_segment_holds_the_whole_drop(self, tmp_path: Path) -> None:
        scratch = segmented(tmp_path, classifier={"confidence_threshold": 0.9})
        drop(scratch.personal, "conversation.md", THREE_TOPICS)
        classifier = BodyClassifier(confidences={"Moving in": 0.2})

        report = run(scratch, classifier=classifier)

        assert report.status is IngestStatus.PARTIAL
        assert len(report.held) == 1
        assert discover(scratch.personal) != []

    def test_the_marker_names_which_segment_failed(self, tmp_path: Path) -> None:
        scratch = segmented(tmp_path, classifier={"confidence_threshold": 0.9})
        drop(scratch.personal, "conversation.md", THREE_TOPICS)

        run(scratch, classifier=BodyClassifier(confidences={"Moving in": 0.2}))

        marker = marker_for(scratch.personal.inbox / "conversation.md")
        assert "segment 2 of 3" in marker.read_text(encoding="utf-8")

    def test_a_held_segment_leaves_the_earlier_ones_filed_and_refiles_nothing_on_retry(
        self, tmp_path: Path
    ) -> None:
        scratch = segmented(tmp_path, classifier={"confidence_threshold": 0.9})
        drop(scratch.personal, "conversation.md", THREE_TOPICS)
        run(scratch, classifier=BodyClassifier(confidences={"Moving in": 0.2}))

        second = run(scratch)

        assert [f.record_id for f in second.filed] != []
        assert discover(scratch.personal) == []

    def test_work_route_never_suppresses_the_note(self, tmp_path: Path) -> None:
        scratch = segmented(tmp_path, sources={"conversate": {"work_route": "never"}})
        drop(
            scratch.personal,
            "conversation.md",
            "---\nsource: conversate\n---\n\n" + THREE_TOPICS,
        )

        report = run(scratch)

        assert len(report.filed) == 3
        assert report.notes == ()
        assert list(scratch.work.root.rglob("notes/**/*.md")) == []

    def test_a_local_only_drop_files_every_segment_under_the_private_prefix(
        self, tmp_path: Path
    ) -> None:
        scratch = segmented(tmp_path)
        drop(scratch.personal, "private.md", "---\nlocal_only: true\n---\n\n" + THREE_TOPICS)

        report = run(scratch)

        assert len(report.filed) == 3
        assert all(f.relative_path.startswith("private/") for f in report.filed)
        assert report.notes == ()

    def test_a_segmenter_failure_holds_the_drop_without_classifying(self, tmp_path: Path) -> None:
        class Refusing:
            def segment(self, record: InboxRecord, max_segments: int) -> Any:
                return NeedsReview("the model returned overlapping ranges")

        scratch = segmented(tmp_path)
        drop(scratch.personal, "conversation.md", THREE_TOPICS)
        classifier = BodyClassifier()

        report = run(scratch, segmenter=Refusing(), classifier=classifier)

        assert report.status is IngestStatus.PARTIAL
        assert classifier.bodies == []
        assert "could not be split" in report.held[0].reason


# --- Topic-area routing -----------------------------------------------------------------
#
# A drop's distilled note lands in the topic area it belongs to, beside the raw transcript it
# links back to. The note is additive: nothing about it may cost the raw filing or the drain.

AREAS = [
    {
        "name": "finance",
        "note_template": "finance/adhoc/{date}-{slug}.md",
        "when": "money",
        "workspaces": ["Finance"],
    }
]


def scratch_with_areas(tmp_path: Path, **sections: Any) -> Scratch:
    personal = make_repo(tmp_path, "personal")
    work = make_repo(tmp_path, "work")
    data: dict[str, Any] = {
        "default_vault": "personal",
        "vaults": {
            "personal": {"root": str(personal), "work_route": "work", "areas": AREAS},
            "work": {"root": str(work)},
        },
    }
    data.update(sections)
    path = tmp_path / "memvault.config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_config(path)
    return Scratch(config, config.vault("personal"), config.vault("work"), path)


def area_notes(root: Path) -> list[str]:
    return [p for p in vault_files(root) if p.startswith("finance/adhoc/")]


class TestAreaRouting:
    def test_a_mapped_workspace_files_a_topic_note(self, tmp_path: Path) -> None:
        scratch = scratch_with_areas(tmp_path)
        drop(scratch.personal, "money.md", "---\nworkspace: Finance\n---\n\nA note about money.\n")

        report = ingest(scratch)

        assert report.failure is None
        assert len(area_notes(scratch.personal.root)) == 1
        assert report.unrouted == ()

    def test_the_note_and_the_transcript_land_in_one_commit(self, tmp_path: Path) -> None:
        scratch = scratch_with_areas(tmp_path)
        drop(scratch.personal, "money.md", "---\nworkspace: Finance\n---\n\nA note about money.\n")

        ingest(scratch)

        shown = show(scratch.personal.root)
        assert "transcripts/" in shown
        assert "finance/adhoc/" in shown

    def test_the_note_links_back_to_the_transcript_it_summarizes(self, tmp_path: Path) -> None:
        scratch = scratch_with_areas(tmp_path)
        drop(scratch.personal, "money.md", "---\nworkspace: Finance\n---\n\nA note about money.\n")

        ingest(scratch)

        note = scratch.personal.root / area_notes(scratch.personal.root)[0]
        transcripts = [
            p for p in vault_files(scratch.personal.root) if p.startswith("transcripts/")
        ]
        assert f"source_path: {transcripts[0]}" in note.read_text(encoding="utf-8")

    def test_an_unroutable_drop_still_files_and_still_drains(self, tmp_path: Path) -> None:
        scratch = scratch_with_areas(tmp_path)
        drop(scratch.personal, "vague.md", PERSONAL_BODY)

        report = ingest(scratch)

        assert report.failure is None
        assert inbox_files(scratch.personal) == []
        assert area_notes(scratch.personal.root) == []
        assert report.unrouted == ("vague.md",)

    def test_the_commit_message_names_what_was_not_routed(self, tmp_path: Path) -> None:
        scratch = scratch_with_areas(tmp_path)
        drop(scratch.personal, "vague.md", PERSONAL_BODY)

        ingest(scratch)

        assert "No area chosen" in show(scratch.personal.root)
        assert "vague.md" in show(scratch.personal.root)

    def test_a_vault_with_no_areas_reports_nothing_as_unrouted(self, scratch: Scratch) -> None:
        """Not routing is different from failing to route."""
        drop(scratch.personal, "coffee.md", PERSONAL_BODY)

        report = ingest(scratch)

        assert report.unrouted == ()

    def test_a_failing_topic_note_does_not_cost_the_transcript(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The note is additive; a broken one must not lose the raw filing or the drain."""
        scratch = scratch_with_areas(tmp_path)

        def explode(**_: Any) -> None:
            raise WriteError("disk full")

        monkeypatch.setattr("memvault.ingest.write_topic_note", explode)
        drop(scratch.personal, "money.md", "---\nworkspace: Finance\n---\n\nA note about money.\n")

        report = ingest(scratch)

        assert report.failure is None
        assert len(report.filed) == 1
        assert inbox_files(scratch.personal) == []
        assert report.unrouted == ("money.md",)


class TestTopicNoteAlongsideTheWorkNote:
    """A work drop can owe two notes. `_note_destination` refuses to guess between them."""

    def test_a_work_drop_with_an_area_writes_both_notes(self, tmp_path: Path) -> None:
        scratch = scratch_with_areas(tmp_path)
        drop(scratch.personal, "money.md", "---\nworkspace: Finance\n---\n\nWork money talk.\n")
        classifier = StubClassifier({"money.md": work_verdict(record_named("money.md"))})

        report = ingest(scratch, classifier=classifier)

        assert report.failure is None
        assert len(area_notes(scratch.personal.root)) == 1, "topic note missing"
        assert len(report.notes) == 1, "cross-vault note missing"
        assert report.notes[0].vault == "work"


class TestReingestWithAreas:
    """One drop, two files, one content id — the id index must still see a single filed item."""

    def test_a_second_pass_over_the_same_drop_files_nothing_new(self, tmp_path: Path) -> None:
        scratch = scratch_with_areas(tmp_path)
        body = "---\nworkspace: Finance\n---\n\nA note about money.\n"
        drop(scratch.personal, "money.md", body)
        ingest(scratch)
        before = sorted(vault_files(scratch.personal.root))

        drop(scratch.personal, "money-again.md", body)
        second = ingest(scratch)

        assert sorted(vault_files(scratch.personal.root)) == before
        assert second.filed == ()
        assert inbox_files(scratch.personal) == []


class TestUnroutedIsPerDropNotPerSegment:
    """A split drop reaches `_ingest_segment` once per segment; the report counts drops."""

    def test_a_segmented_drop_is_named_once(self, tmp_path: Path) -> None:
        scratch = scratch_with_areas(tmp_path, segment={"enabled": True, "min_chars": 0})
        drop(
            scratch.personal,
            "many.md",
            "First paragraph about nothing in particular.\n"
            "\n"
            "Second paragraph, also unremarkable.\n"
            "\n"
            "Third paragraph to be sure.\n",
        )

        report = ingest(
            scratch,
            classifier=BodyClassifier(),
            segmenter=BlockSegmenter(),
        )

        assert report.unrouted == ("many.md",), "one drop, named once"
