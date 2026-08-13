"""Inbox discovery, frontmatter parsing, and normalization.

Two themes run through these tests.

Identity must depend on content and nothing else — not line endings, not a BOM, not the
metadata wrapped around it — because that is the whole basis of idempotent ingestion.

And a bad drop must survive. A file with broken frontmatter comes back as a record carrying the
reason, and it must not stop the walk that found it; only files with no usable content at all
are skipped, and always with a logged reason.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import pytest

from memvault.config import VaultConfig, load_config
from memvault.inbox import DeclaredMetadata, InboxRecord, discover, normalize
from tests.test_config import make_vault, write_config


@pytest.fixture
def vault(tmp_path: Path) -> VaultConfig:
    root = make_vault(tmp_path, "personal")
    config = write_config(tmp_path, {"vaults": {"personal": {"root": str(root)}}})
    return load_config(config).vault()


def drop(vault: VaultConfig, name: str, content: str | bytes) -> Path:
    """Place a file in the inbox the way a feeder or a person would."""
    path = vault.inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def by_name(records: list[InboxRecord]) -> dict[str, InboxRecord]:
    return {record.filename: record for record in records}


class TestNormalization:
    """`normalize` is pure, so the parsing rules can be pinned down without a filesystem."""

    def test_bare_text_leaves_every_inferred_field_empty(self) -> None:
        record = normalize("Coffee with Deniz about the erfpacht deadline.", "note.txt")

        assert record.body == "Coffee with Deniz about the erfpacht deadline."
        assert record.declared == DeclaredMetadata()
        assert record.declared.declared_keys == frozenset()
        assert record.unparseable is None

    def test_declared_classification_survives_normalization_verbatim(self) -> None:
        text = "---\nclassification: work\n---\nStandup notes."

        record = normalize(text, "standup.md")

        assert record.declared.classification == "work"
        assert record.body == "Standup notes."
        assert record.unparseable is None

    def test_declared_keys_name_exactly_the_fields_that_suppress_inference(self) -> None:
        text = "---\nsource: whatsapp\ntags: [house]\n---\nBody."

        record = normalize(text, "note.md")

        assert record.declared.declared_keys == {"source", "tags"}
        assert record.declared.declares("source")
        assert not record.declared.declares("classification")

    def test_full_frontmatter_round_trips_into_declared_metadata(self) -> None:
        text = (
            "---\n"
            "source: conversate\n"
            'date: "2026-07-14"\n'
            "tags: [meeting, acmefarm]\n"
            "classification: work\n"
            "local_only: true\n"
            "participants: [Ada, Sam]\n"
            "---\n"
            "We agreed to cut over after the Resend work."
        )

        record = normalize(text, "call.md")

        assert record.declared == DeclaredMetadata(
            source="conversate",
            date="2026-07-14",
            tags=("meeting", "acmefarm"),
            classification="work",
            local_only=True,
            participants=("Ada", "Sam"),
            declared_keys=frozenset(
                {"source", "date", "tags", "classification", "local_only", "participants"}
            ),
        )

    def test_a_declared_importance_is_read_as_a_number(self) -> None:
        record = normalize("---\nimportance: 9\n---\nBody.", "note.md")

        assert record.declared.importance == 9
        assert record.declared.declares("importance")
        assert record.unparseable is None

    def test_an_undeclared_importance_leaves_the_field_empty(self) -> None:
        """Absent means nobody judged, so the classifier's own score is free to stand."""
        record = normalize("Body.", "note.txt")

        assert record.declared.importance is None
        assert not record.declared.declares("importance")

    def test_an_unquoted_yaml_date_renders_as_an_iso_string(self) -> None:
        record = normalize("---\ndate: 2026-07-14\n---\nBody.", "note.md")

        assert record.declared.date == "2026-07-14"

    def test_a_single_string_tag_becomes_a_one_item_list(self) -> None:
        record = normalize("---\ntags: house\n---\nBody.", "note.md")

        assert record.declared.tags == ("house",)

    def test_unknown_frontmatter_keys_are_ignored_rather_than_held_for_review(self) -> None:
        text = "---\nsource: g2\ndevice_battery: 42\n---\nBody."

        record = normalize(text, "note.md")

        assert record.unparseable is None
        assert record.declared.source == "g2"

    def test_an_empty_frontmatter_block_is_not_a_problem(self) -> None:
        record = normalize("---\n---\nBody.", "note.md")

        assert record.unparseable is None
        assert record.body == "Body."

    def test_interior_blank_lines_and_leading_indentation_are_content(self) -> None:
        text = "    indented first line\n\nsecond paragraph"

        record = normalize(text, "note.txt")

        assert record.body == text

    def test_byte_size_counts_the_original_text(self) -> None:
        record = normalize("İstanbul", "note.txt")

        assert record.size_bytes == len("İstanbul".encode())


class TestUnparseableFrontmatter:
    """A drop the parser cannot fully read is held with a reason, never dropped."""

    def test_malformed_yaml_is_retained_with_the_reason(self) -> None:
        text = "---\ntags: [unclosed\nsource: phone\n---\nThe transcript body."

        record = normalize(text, "broken.md")

        assert record.unparseable is not None
        assert "not valid YAML" in record.unparseable
        assert record.body == "The transcript body."

    def test_a_fenced_block_that_is_not_a_mapping_is_reported_as_not_frontmatter(self) -> None:
        text = "---\n- one\n- two\n---\nBody."

        record = normalize(text, "note.md")

        assert record.unparseable is not None
        assert "not frontmatter" in record.unparseable
        assert record.body == text

    def test_a_markdown_rule_at_the_top_of_a_file_does_not_swallow_the_first_section(
        self,
    ) -> None:
        text = "---\n\nThe opening section, under a horizontal rule.\n\n---\n\nThe rest."

        record = normalize(text, "note.md")

        assert "The opening section" in record.body
        assert "The rest." in record.body

    def test_an_unclosed_fence_keeps_the_whole_file_as_body(self) -> None:
        text = "---\nsource: phone\nSomething that was never fenced off."

        record = normalize(text, "note.md")

        assert record.unparseable is not None
        assert "never closed" in record.unparseable
        assert record.body == text

    def test_a_non_iso_date_is_rejected_by_name(self) -> None:
        record = normalize("---\ndate: 14/07/2026\n---\nBody.", "note.md")

        assert record.unparseable is not None
        assert "'date' must be YYYY-MM-DD" in record.unparseable

    def test_a_non_boolean_local_only_is_refused_rather_than_coerced(self) -> None:
        record = normalize('---\nlocal_only: "no"\n---\nBody.', "note.md")

        assert record.unparseable is not None
        assert "'local_only' must be true or false" in record.unparseable
        assert record.declared.local_only is None

    def test_a_word_where_importance_should_be_is_rejected_by_name(self) -> None:
        """A declared field overrides the model, so an unreadable one is fixed, not guessed at."""
        record = normalize('---\nimportance: "high"\n---\nBody.', "note.md")

        assert record.unparseable is not None
        assert "'importance' must be a whole number from 1 to 10" in record.unparseable
        assert record.declared.importance is None

    def test_an_out_of_range_declared_importance_is_rejected(self) -> None:
        record = normalize("---\nimportance: 42\n---\nBody.", "note.md")

        assert record.unparseable is not None
        assert "'importance'" in record.unparseable

    def test_an_unrecognized_classification_is_rejected_by_name(self) -> None:
        record = normalize("---\nclassification: wrok\n---\nBody.", "note.md")

        assert record.unparseable is not None
        assert "'classification' must be one of" in record.unparseable

    def test_several_bad_fields_are_reported_together(self) -> None:
        record = normalize("---\ndate: soon\nclassification: wrok\n---\nBody.", "note.md")

        assert record.unparseable is not None
        assert "'date'" in record.unparseable
        assert "'classification'" in record.unparseable

    def test_the_reason_stays_on_one_line_so_it_fits_a_review_marker(self) -> None:
        record = normalize("---\ntags: [unclosed\n---\nBody.", "note.md")

        assert record.unparseable is not None
        assert "\n" not in record.unparseable


class TestContentIdentity:
    """Identity tracks the body, so metadata edits never look like new memories."""

    def test_crlf_and_lf_versions_share_an_id(self) -> None:
        lf = normalize("first line\nsecond line", "unix.txt")
        crlf = normalize("first line\r\nsecond line", "windows.txt")

        assert lf.id == crlf.id

    def test_a_lone_cr_is_normalized_too(self) -> None:
        lf = normalize("first line\nsecond line", "unix.txt")
        cr = normalize("first line\rsecond line", "classic-mac.txt")

        assert lf.id == cr.id

    def test_the_same_body_under_different_frontmatter_shares_an_id(self) -> None:
        plain = normalize("The transcript body.", "a.txt")
        wrapped = normalize(
            "---\nsource: phone\nclassification: personal\n---\nThe transcript body.", "b.md"
        )

        assert plain.id == wrapped.id

    def test_a_different_body_produces_a_different_id(self) -> None:
        first = normalize("The transcript body.", "a.txt")
        second = normalize("A different transcript body.", "b.txt")

        assert first.id != second.id

    def test_trailing_whitespace_does_not_change_identity(self) -> None:
        bare = normalize("Body.", "a.txt")
        padded = normalize("\nBody.   \n\n", "b.txt")

        assert bare.id == padded.id

    def test_turkish_text_hashes_the_same_however_it_was_composed(self) -> None:
        text = "\u0130stanbul'da g\u00fcne\u015fli bir g\u00fcn"

        composed = normalize(unicodedata.normalize("NFC", text), "nfc.txt")
        decomposed = normalize(unicodedata.normalize("NFD", text), "nfd.txt")

        assert unicodedata.normalize("NFD", text) != unicodedata.normalize("NFC", text)
        assert composed.id == decomposed.id
        assert composed.body == decomposed.body

    def test_turkish_content_survives_normalization_intact(self) -> None:
        text = "Şişli'deki toplantıda ığdır çöreği yedik."

        record = normalize(text, "note.txt")

        assert record.body == text


class TestDiscovery:
    """Walking a real inbox: what becomes a record, what is skipped, and how loudly."""

    def test_a_bare_text_drop_is_discovered(self, vault: VaultConfig) -> None:
        drop(vault, "note.txt", "Phone note about the mortgage advisor.")

        records = discover(vault)

        assert len(records) == 1
        assert records[0].body == "Phone note about the mortgage advisor."
        assert records[0].filename == "note.txt"

    def test_a_nested_subdirectory_of_drops_is_discovered(self, vault: VaultConfig) -> None:
        drop(vault, "top.txt", "top level")
        drop(vault, "2026-07/deep/inner.md", "nested")

        records = discover(vault)

        assert [record.filename for record in records] == ["2026-07/deep/inner.md", "top.txt"]

    def test_an_extensionless_file_is_a_valid_drop(self, vault: VaultConfig) -> None:
        drop(vault, "transcript", "Pasted from a phone with no extension.")

        records = discover(vault)

        assert [record.filename for record in records] == ["transcript"]

    def test_records_carry_their_path_and_byte_size(self, vault: VaultConfig) -> None:
        path = drop(vault, "note.txt", "İstanbul")

        record = discover(vault)[0]

        assert record.path == path
        assert record.size_bytes == path.stat().st_size

    def test_an_empty_file_is_skipped_with_a_logged_reason(
        self, vault: VaultConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        drop(vault, "empty.txt", "")

        with caplog.at_level(logging.INFO, logger="memvault.inbox"):
            records = discover(vault)

        assert records == []
        assert "empty.txt: skipped, no content" in caplog.text

    def test_a_whitespace_only_file_is_skipped_with_a_logged_reason(
        self, vault: VaultConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        drop(vault, "blank.txt", "   \n\n\t\n")

        with caplog.at_level(logging.INFO, logger="memvault.inbox"):
            records = discover(vault)

        assert records == []
        assert "blank.txt: skipped, no content" in caplog.text

    def test_frontmatter_with_no_body_is_skipped(self, vault: VaultConfig) -> None:
        drop(vault, "meta-only.md", "---\nsource: phone\n---\n")

        assert discover(vault) == []

    def test_a_binary_file_is_skipped_without_raising(
        self, vault: VaultConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        drop(vault, "photo.txt", b"\x89PNG\r\n\x1a\n\xff\xfe\x00binary")

        with caplog.at_level(logging.INFO, logger="memvault.inbox"):
            records = discover(vault)

        assert records == []
        assert "photo.txt: skipped, not UTF-8 text" in caplog.text

    def test_an_unsupported_extension_is_skipped_with_a_logged_reason(
        self, vault: VaultConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        drop(vault, "scan.pdf", "not really a pdf, but not a drop either")

        with caplog.at_level(logging.INFO, logger="memvault.inbox"):
            records = discover(vault)

        assert records == []
        assert "scan.pdf: skipped" in caplog.text

    def test_hidden_files_and_directories_are_not_drops(self, vault: VaultConfig) -> None:
        drop(vault, ".gitkeep", "")
        drop(vault, ".DS_Store", "junk")
        drop(vault, ".archive/old.md", "archived")
        drop(vault, "real.md", "real content")

        records = discover(vault)

        assert [record.filename for record in records] == ["real.md"]

    def test_a_utf8_bom_does_not_change_identity(self, vault: VaultConfig) -> None:
        drop(vault, "with-bom.txt", b"\xef\xbb\xbfSame content")
        drop(vault, "without-bom.txt", "Same content")

        records = by_name(discover(vault))

        assert records["with-bom.txt"].id == records["without-bom.txt"].id

    def test_a_malformed_drop_does_not_stop_the_walk(self, vault: VaultConfig) -> None:
        drop(vault, "a-broken.md", "---\ntags: [unclosed\n---\nStill worth keeping.")
        drop(vault, "b-good.md", "---\nclassification: personal\n---\nFine.")

        records = by_name(discover(vault))

        assert set(records) == {"a-broken.md", "b-good.md"}
        assert records["a-broken.md"].unparseable is not None
        assert records["b-good.md"].unparseable is None

    def test_a_missing_inbox_yields_no_records_and_a_warning(
        self, vault: VaultConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        vault.inbox.rmdir()

        with caplog.at_level(logging.WARNING, logger="memvault.inbox"):
            records = discover(vault)

        assert records == []
        assert str(vault.inbox) in caplog.text

    def test_an_empty_inbox_yields_no_records(self, vault: VaultConfig) -> None:
        assert discover(vault) == []

    def test_the_same_content_dropped_twice_yields_two_records_sharing_an_id(
        self, vault: VaultConfig
    ) -> None:
        drop(vault, "one.txt", "Duplicated content.")
        drop(vault, "two.txt", "Duplicated content.")

        records = discover(vault)

        assert len(records) == 2
        assert records[0].id == records[1].id


class TestMixedFolderVerification:
    """The unit's stated verification: a mixed folder, twice, with stable ids."""

    @pytest.fixture
    def mixed_inbox(self, vault: VaultConfig) -> VaultConfig:
        drop(vault, "bare.txt", "A bare text note from a phone.")
        drop(
            vault,
            "meeting/2026-07-14.md",
            "---\nsource: conversate\nclassification: work\nparticipants: [Ada, Sam]\n---\n"
            "Sam wants a tab per supplier.",
        )
        drop(vault, "empty.txt", "")
        drop(vault, "photo.txt", b"\xff\xd8\xff\xe0\x00\x10JFIF")
        return vault

    def test_only_the_real_drops_become_records(self, mixed_inbox: VaultConfig) -> None:
        records = discover(mixed_inbox)

        assert [record.filename for record in records] == ["bare.txt", "meeting/2026-07-14.md"]

    def test_declared_metadata_is_carried_and_inference_left_to_the_classifier(
        self, mixed_inbox: VaultConfig
    ) -> None:
        records = by_name(discover(mixed_inbox))

        assert records["meeting/2026-07-14.md"].declared.classification == "work"
        assert records["meeting/2026-07-14.md"].declared.participants == ("Ada", "Sam")
        assert records["bare.txt"].declared.classification is None

    def test_ids_are_identical_across_two_runs(self, mixed_inbox: VaultConfig) -> None:
        first = [record.id for record in discover(mixed_inbox)]
        second = [record.id for record in discover(mixed_inbox)]

        assert first == second


# --- Declared workspace -----------------------------------------------------------------
#
# Only some feeders know a workspace. Chad sets it; Conversate, Apple Notes and manual drops
# do not, so the field is optional everywhere and purely a routing hint.


def test_a_declared_workspace_is_read() -> None:
    record = normalize("---\nsource: chad-web\nworkspace: Finance\n---\n\nbody\n", "drop.md")

    assert record.declared.workspace == "Finance"
    assert record.declared.declares("workspace")


def test_a_drop_without_a_workspace_declares_none() -> None:
    record = normalize("---\nsource: conversate\n---\n\nbody\n", "drop.md")

    assert record.declared.workspace is None
    assert not record.declared.declares("workspace")


def test_a_workspace_is_stripped() -> None:
    record = normalize("---\nworkspace: '  Finance  '\n---\n\nbody\n", "drop.md")

    assert record.declared.workspace == "Finance"


def test_a_blank_workspace_is_a_frontmatter_error() -> None:
    """Consistent with every other text field: a blank value is a feeder bug, said out loud."""
    record = normalize("---\nworkspace: '   '\n---\n\nbody\n", "drop.md")

    assert record.unparseable is not None
    assert "workspace" in record.unparseable


def test_an_unknown_workspace_value_is_not_rejected() -> None:
    """Routing ignores a workspace it does not recognize; parsing has no opinion on the value."""
    record = normalize("---\nworkspace: Something Nobody Mapped\n---\n\nbody\n", "drop.md")

    assert record.unparseable is None
    assert record.declared.workspace == "Something Nobody Mapped"
