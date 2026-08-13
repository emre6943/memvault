"""The distilled note filed into a topic area, and its link back to the raw transcript."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from memvault.area import AreaChoice
from memvault.classify import Classification, Relation
from memvault.config import VaultConfig
from memvault.inbox import DeclaredMetadata, InboxRecord
from memvault.route import Destination, DestinationKind
from memvault.topic_note import write_topic_note
from memvault.writer import RELATIONS_HEADING, WriteError

DAY = date(2026, 8, 10)
RAW_PATH = "transcripts/2026/08/2026-08-10-a-drop.md"
CHOICE = AreaChoice("finance", "finance/adhoc/{date}-{slug}.md", "workspace")
NOTE_DEST = Destination(
    vault="personal",
    kind=DestinationKind.NOTE,
    path_template="finance/adhoc/{date}-{slug}.md",
)
SUMMARY = "the summary the classifier wrote"


@pytest.fixture
def vault(tmp_path: Path) -> VaultConfig:
    root = tmp_path / "personal"
    (root / "inbox").mkdir(parents=True)
    return VaultConfig(name="personal", root=root, inbox=root / "inbox")


def a_record(**over: Any) -> InboxRecord:
    fields: dict[str, Any] = {
        "id": "abc123",
        "body": "body\n",
        "declared": DeclaredMetadata(source="chad-web", declared_keys=frozenset({"source"})),
        "filename": "a-drop.md",
        "size_bytes": 6,
    }
    return InboxRecord(**{**fields, **over})


def a_verdict(**over: Any) -> Classification:
    fields: dict[str, Any] = {
        "classification": "personal",
        "confidence": 0.9,
        "title": "A drop",
        "slug": "a-drop",
        "summary": SUMMARY,
    }
    return Classification(**{**fields, **over})


def write(vault: VaultConfig, **over: Any) -> Any:
    kwargs: dict[str, Any] = {
        "record": a_record(),
        "verdict": a_verdict(),
        "destination": NOTE_DEST,
        "vault": vault,
        "raw_path": RAW_PATH,
        "area": CHOICE,
        "ingested": DAY,
    }
    return write_topic_note(**{**kwargs, **over})


def read_frontmatter(path: Path) -> dict[str, Any]:
    _, block, _ = path.read_text(encoding="utf-8").split("---\n", 2)
    return dict(yaml.safe_load(block))


def body_of(path: Path) -> str:
    return path.read_text(encoding="utf-8").split("---\n", 2)[2]


class TestTopicNote:
    def test_it_files_at_the_areas_template(self, vault: VaultConfig) -> None:
        written = write(vault)

        assert written.relative_path == "finance/adhoc/2026-08-10-a-drop.md"

    def test_it_links_back_to_the_raw_transcript(self, vault: VaultConfig) -> None:
        assert read_frontmatter(write(vault).path)["source_path"] == RAW_PATH

    def test_the_body_carries_the_classifiers_summary(self, vault: VaultConfig) -> None:
        assert SUMMARY in body_of(write(vault).path)

    def test_the_body_links_to_the_transcript_relatively(self, vault: VaultConfig) -> None:
        """From finance/adhoc/ the transcript is two levels up, so a viewer can follow it."""
        assert "../../transcripts/2026/08/2026-08-10-a-drop.md" in body_of(write(vault).path)

    def test_it_records_why_this_area_was_chosen(self, vault: VaultConfig) -> None:
        front = read_frontmatter(write(vault).path)

        assert front["area"] == "finance"
        assert front["area_via"] == "workspace"

    def test_it_shares_the_content_id_of_its_transcript(self, vault: VaultConfig) -> None:
        """Same drop, same id — so the id index treats the pair as one filed item."""
        assert read_frontmatter(write(vault).path)["content_id"] == "abc123"

    def test_it_carries_the_drops_source(self, vault: VaultConfig) -> None:
        assert read_frontmatter(write(vault).path)["source"] == "chad-web"

    def test_a_second_note_for_the_same_day_and_slug_does_not_overwrite(
        self, vault: VaultConfig
    ) -> None:
        first = write(vault)
        second = write(vault)

        assert first.path != second.path
        assert first.path.exists() and second.path.exists()
        assert second.collided

    def test_an_empty_summary_still_produces_a_readable_note(self, vault: VaultConfig) -> None:
        """A thin classification must not yield a note that is only frontmatter."""
        written = write(vault, verdict=a_verdict(summary=""))

        assert body_of(written.path).strip() != ""

    def test_a_title_with_no_slug_is_slugified(self, vault: VaultConfig) -> None:
        written = write(vault, verdict=a_verdict(slug="", title="Bir şey oldu"))

        assert written.relative_path.startswith("finance/adhoc/2026-08-10-")

    def test_a_declared_date_dates_the_note_not_the_ingest_day(self, vault: VaultConfig) -> None:
        """A note must carry the same date as the transcript it describes."""
        record = a_record(
            declared=DeclaredMetadata(date="2026-07-01", declared_keys=frozenset({"date"}))
        )

        assert write(vault, record=record).relative_path == "finance/adhoc/2026-07-01-a-drop.md"

    def test_the_written_file_describes_itself_for_the_commit(self, vault: VaultConfig) -> None:
        written = write(vault)

        assert written.vault == "personal"
        assert written.kind is DestinationKind.NOTE
        assert written.record_id == "abc123"
        assert written.title == "A drop"
        assert written.classification == "personal"

    def test_a_local_only_destination_marks_the_note(self, vault: VaultConfig) -> None:
        destination = Destination(
            vault="personal",
            kind=DestinationKind.NOTE,
            path_template="private/finance/adhoc/{date}-{slug}.md",
            local_only=True,
        )

        written = write(vault, destination=destination)

        assert written.local_only is True
        assert written.relative_path.startswith("private/")

    def test_the_classifiers_relations_are_rendered_as_wikilink_bullets(
        self, vault: VaultConfig
    ) -> None:
        """The graph grows from ingestion: a note carries the links the classifier saw."""
        verdict = a_verdict(relations=(Relation("part_of", "repos/MemVault"),))

        body = body_of(write(vault, verdict=verdict).path)

        assert "- part_of [[repos/MemVault]]" in body

    def test_the_transcript_link_survives_alongside_the_relations(self, vault: VaultConfig) -> None:
        verdict = a_verdict(relations=(Relation("about", "ranking"),))

        body = body_of(write(vault, verdict=verdict).path)

        assert RAW_PATH in body
        assert "[[ranking]]" in body

    def test_a_note_with_no_relations_carries_no_empty_relations_block(
        self, vault: VaultConfig
    ) -> None:
        assert RELATIONS_HEADING not in body_of(write(vault).path)

    def test_the_classifiers_importance_lands_in_the_notes_frontmatter(
        self, vault: VaultConfig
    ) -> None:
        front = read_frontmatter(write(vault, verdict=a_verdict(importance=8)).path)

        assert front["importance"] == 8

    def test_a_declared_importance_wins_here_too(self, vault: VaultConfig) -> None:
        record = a_record(
            declared=DeclaredMetadata(importance=3, declared_keys=frozenset({"importance"}))
        )

        front = read_frontmatter(write(vault, record=record, verdict=a_verdict(importance=8)).path)

        assert front["importance"] == 3

    def test_a_note_without_importance_omits_the_key(self, vault: VaultConfig) -> None:
        assert "importance" not in read_frontmatter(write(vault).path)

    def test_a_raw_destination_is_refused(self, vault: VaultConfig) -> None:
        """Raw bodies are the transcript writer's job; mixing them here would leak verbatim."""
        destination = Destination(
            vault="personal",
            kind=DestinationKind.RAW,
            path_template="finance/adhoc/{date}-{slug}.md",
        )

        with pytest.raises(WriteError, match="raw"):
            write(vault, destination=destination)
