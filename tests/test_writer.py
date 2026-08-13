"""Filing a routed record into a vault.

Three properties carry most of the weight here.

A filed file must explain itself: every provenance field the pipeline knows is asserted on the
frontmatter, and the content id in it must match the id of the drop it came from, because that
equality is what makes a filed memory traceable back to what arrived.

A filed file must never be half a file. The write is driven through a failing `os.replace` to
prove that an interruption leaves the vault untouched rather than holding a truncated document
that a later pass would read as real.

And two memories must never become one. Collisions suffix, and slugs keep Turkish letters
distinct from their ASCII lookalikes — `açık` and `acik` are different words and must be
different files.

Routing runs for real rather than being faked: the writer is fed the `RoutingResult` the router
actually produces, so a change to the routing table that broke filing would fail here too.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from memvault.classify import Classification, Relation
from memvault.config import ClassifierConfig, Config, IndexConfig, VaultConfig
from memvault.inbox import DeclaredMetadata, InboxRecord, content_id
from memvault.route import Destination, DestinationKind, route
from memvault.writer import (
    FRONTMATTER_ORDER,
    RELATIONS_HEADING,
    WriteError,
    WrittenFile,
    file_record,
    render_path,
    render_relations,
    slugify,
    write_document,
)

INGESTED = date(2026, 8, 1)
BODY = "Coffee with Deniz about the erfpacht deadline.\n\n  Indented follow-up line."


def vault_config(name: str, root: Path, **overrides: Any) -> VaultConfig:
    settings: dict[str, Any] = {"name": name, "root": root, "inbox": root / "inbox"}
    settings.update(overrides)
    return VaultConfig(**settings)


def config_of(*vaults: VaultConfig) -> Config:
    return Config(
        source=Path("memvault.config.yaml"),
        default_vault=vaults[0].name,
        vaults={vault.name: vault for vault in vaults},
        classifier=ClassifierConfig(),
        index=IndexConfig(),
    )


def drop(body: str = BODY, filename: str = "note.md", **declared: Any) -> InboxRecord:
    """A record whose id is a real hash of its body, as `inbox.normalize` would produce."""
    return InboxRecord(
        id=content_id(body),
        body=body,
        declared=DeclaredMetadata(**declared, declared_keys=frozenset(declared)),
        filename=filename,
        size_bytes=len(body.encode("utf-8")),
    )


def verdict(**overrides: Any) -> Classification:
    settings: dict[str, Any] = {
        "classification": "personal",
        "confidence": 0.92,
        "title": "Coffee with Deniz",
        "slug": "coffee-with-deniz",
        "summary": "An erfpacht deadline came up.",
        "tags": ("house", "amsterdam"),
        "model": "claude-sonnet-4-6",
    }
    settings.update(overrides)
    return Classification(**settings)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "personal"
    path.mkdir()
    return path


@pytest.fixture
def work_root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def file_one(
    root: Path,
    *,
    record: InboxRecord | None = None,
    outcome: Classification | None = None,
    vault: VaultConfig | None = None,
    ingested: date = INGESTED,
) -> WrittenFile:
    """Route one record for real, then file it. The seam under test starts after routing."""
    record = record or drop()
    outcome = outcome or verdict()
    vault = vault or vault_config("personal", root)
    routing = route(record, outcome, config_of(vault), vault)
    return file_record(record, outcome, routing, root, ingested=ingested)


def read_document(path: Path) -> tuple[dict[str, Any], str]:
    """Split a filed file back into frontmatter and body, the way any reader would."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, frontmatter, body = text.split("---\n", 2)
    parsed = yaml.safe_load(frontmatter)
    assert isinstance(parsed, dict)
    return parsed, body


class TestFilingAPersonalRecord:
    """The ordinary path: one drop, one file, every provenance field on it."""

    def test_the_file_lands_at_the_templated_path(self, root: Path) -> None:
        written = file_one(root)

        assert written.relative_path == "transcripts/2026/08/2026-08-01-coffee-with-deniz.md"
        assert written.path == root / written.relative_path
        assert written.path.is_file()

    def test_every_provenance_field_is_populated(self, root: Path) -> None:
        record = drop(source="conversate", date="2026-07-28")

        written = file_one(root, record=record)
        frontmatter, _ = read_document(written.path)

        assert frontmatter == {
            "title": "Coffee with Deniz",
            "date": "2026-07-28",
            "source": "conversate",
            "classification": "personal",
            "tags": ["house", "amsterdam"],
            "summary": "An erfpacht deadline came up.",
            "ingested": "2026-08-01",
            "original_filename": "note.md",
            "content_id": record.id,
            "classifier_model": "claude-sonnet-4-6",
            "confidence": 0.92,
        }

    def test_the_content_id_matches_the_drop_it_came_from(self, root: Path) -> None:
        record = drop()

        written = file_one(root, record=record)
        frontmatter, _ = read_document(written.path)

        assert frontmatter["content_id"] == content_id(record.body)
        assert written.record_id == record.id

    def test_frontmatter_keys_keep_a_fixed_human_readable_order(self, root: Path) -> None:
        written = file_one(root, record=drop(source="whatsapp"))
        frontmatter, _ = read_document(written.path)

        order = [key for key in FRONTMATTER_ORDER if key in frontmatter]

        assert list(frontmatter) == order
        assert list(frontmatter)[0] == "title"

    def test_the_declared_date_dates_the_file_and_its_path(self, root: Path) -> None:
        written = file_one(root, record=drop(date="2026-03-09"))
        frontmatter, _ = read_document(written.path)

        assert written.relative_path.startswith("transcripts/2026/03/2026-03-09-")
        assert frontmatter["date"] == "2026-03-09"
        assert frontmatter["ingested"] == "2026-08-01"

    def test_a_drop_that_declared_no_date_is_dated_by_its_ingest_day(self, root: Path) -> None:
        written = file_one(root)
        frontmatter, _ = read_document(written.path)

        assert frontmatter["date"] == "2026-08-01"

    def test_an_absent_source_omits_the_key_rather_than_writing_it_empty(self, root: Path) -> None:
        written = file_one(root)
        frontmatter, _ = read_document(written.path)

        assert "source" not in frontmatter

    def test_the_result_carries_what_a_commit_message_needs(self, root: Path) -> None:
        written = file_one(root)

        assert written.vault == "personal"
        assert written.kind is DestinationKind.RAW
        assert written.classification == "personal"
        assert not written.collided
        assert written.summary_line() == (
            "Coffee with Deniz — transcripts/2026/08/2026-08-01-coffee-with-deniz.md"
        )

    def test_the_document_is_valid_markdown_with_a_readable_frontmatter_block(
        self, root: Path
    ) -> None:
        written = file_one(root)

        text = written.path.read_text(encoding="utf-8")

        assert text.startswith("---\ntitle: Coffee with Deniz\n")
        assert "\n---\n\n" in text
        assert text.endswith("\n")


class TestBodyFidelity:
    """The filed body must be the normalized body, unchanged."""

    def test_the_body_is_byte_identical_to_the_normalized_input(self, root: Path) -> None:
        written = file_one(root)

        _, body = read_document(written.path)

        assert body == f"\n{BODY}\n"
        assert body.strip("\n") == BODY

    def test_turkish_text_survives_the_round_trip(self, root: Path) -> None:
        body = "Deniz'le İstanbul'da kısa bir sohbet. Açık ve net."

        written = file_one(root, record=drop(body=body))
        _, filed = read_document(written.path)

        assert filed.strip("\n") == body

    def test_interior_indentation_and_blank_lines_are_content(self, root: Path) -> None:
        body = "Plan:\n\n    code_block_line()\n\nDone."

        written = file_one(root, record=drop(body=body))
        _, filed = read_document(written.path)

        assert filed.strip("\n") == body

    def test_a_body_that_looks_like_frontmatter_is_not_re_parsed(self, root: Path) -> None:
        body = "---\nnot: metadata\n---\nStill body."

        written = file_one(root, record=drop(body=body))

        assert body in written.path.read_text(encoding="utf-8")


class TestCollisions:
    """Two memories with one name are still two memories."""

    def test_a_second_file_with_the_same_slug_and_date_is_suffixed(self, root: Path) -> None:
        first = file_one(root, record=drop(body="First conversation."))
        second = file_one(root, record=drop(body="Second conversation."))

        assert first.relative_path == "transcripts/2026/08/2026-08-01-coffee-with-deniz.md"
        assert second.relative_path == "transcripts/2026/08/2026-08-01-coffee-with-deniz-2.md"

    def test_both_files_persist_with_their_own_content(self, root: Path) -> None:
        first = file_one(root, record=drop(body="First conversation."))
        second = file_one(root, record=drop(body="Second conversation."))

        assert read_document(first.path)[1].strip("\n") == "First conversation."
        assert read_document(second.path)[1].strip("\n") == "Second conversation."
        assert first.record_id != second.record_id

    def test_the_suffix_is_reported_on_the_result(self, root: Path) -> None:
        file_one(root, record=drop(body="First."))
        second = file_one(root, record=drop(body="Second."))

        assert second.suffix == 2
        assert second.collided

    def test_suffixes_keep_climbing(self, root: Path) -> None:
        paths = [file_one(root, record=drop(body=f"Body {n}.")).relative_path for n in range(4)]

        assert [Path(path).stem.rsplit("-", 1)[-1] for path in paths[1:]] == ["2", "3", "4"]

    def test_nothing_is_ever_overwritten(self, root: Path) -> None:
        file_one(root, record=drop(body="First."))
        file_one(root, record=drop(body="Second."))
        file_one(root, record=drop(body="Third."))

        assert len(list((root / "transcripts/2026/08").iterdir())) == 3

    def test_the_same_slug_on_a_different_date_does_not_collide(self, root: Path) -> None:
        first = file_one(root, record=drop(body="A.", date="2026-08-01"))
        second = file_one(root, record=drop(body="B.", date="2026-08-02"))

        assert not second.collided
        assert first.relative_path != second.relative_path


class TestLocalOnly:
    """R10 / KTD7: the gitignored prefix, and nowhere else."""

    def test_the_file_lands_under_the_gitignored_prefix(self, root: Path) -> None:
        written = file_one(root, record=drop(local_only=True))

        assert written.relative_path.startswith("private/transcripts/")
        assert (root / "private").is_dir()

    def test_it_lands_nowhere_else(self, root: Path) -> None:
        file_one(root, record=drop(local_only=True))

        outside = [
            path
            for path in root.rglob("*")
            if path.is_file() and not path.relative_to(root).as_posix().startswith("private/")
        ]

        assert outside == []

    def test_the_flag_is_recorded_in_the_frontmatter(self, root: Path) -> None:
        written = file_one(root, record=drop(local_only=True))
        frontmatter, _ = read_document(written.path)

        assert frontmatter["local_only"] is True
        assert written.local_only

    def test_an_ordinary_record_omits_the_flag_rather_than_writing_false(self, root: Path) -> None:
        written = file_one(root)
        frontmatter, _ = read_document(written.path)

        assert "local_only" not in frontmatter
        assert not written.local_only

    def test_a_configured_prefix_is_honored(self, root: Path) -> None:
        vault = vault_config("personal", root, local_only_prefix="not-pushed")

        written = file_one(root, record=drop(local_only=True), vault=vault)

        assert written.relative_path.startswith("not-pushed/transcripts/")


class TestParticipants:
    """R11: named people must be findable later, and never invented."""

    def test_participants_round_trip_into_the_frontmatter(self, root: Path) -> None:
        written = file_one(root, outcome=verdict(participants=("Ada", "Deniz")))
        frontmatter, _ = read_document(written.path)

        assert frontmatter["participants"] == ["Ada", "Deniz"]

    def test_declared_participants_win_over_the_classifiers(self, root: Path) -> None:
        written = file_one(
            root,
            record=drop(participants=("Deniz",)),
            outcome=verdict(participants=("Somebody", "Else")),
        )
        frontmatter, _ = read_document(written.path)

        assert frontmatter["participants"] == ["Deniz"]

    def test_absent_participants_omit_the_key_rather_than_writing_an_empty_list(
        self, root: Path
    ) -> None:
        written = file_one(root)
        frontmatter, _ = read_document(written.path)

        assert "participants" not in frontmatter

    def test_a_participant_is_greppable_in_the_filed_file(self, root: Path) -> None:
        written = file_one(root, outcome=verdict(participants=("Deniz Yılmaz",)))

        assert "Deniz Yılmaz" in written.path.read_text(encoding="utf-8")


class TestDeclaredTags:
    """R2 again, at the layer that writes the field."""

    def test_declared_tags_win_over_the_classifiers(self, root: Path) -> None:
        written = file_one(root, record=drop(tags=("erfpacht",)))
        frontmatter, _ = read_document(written.path)

        assert frontmatter["tags"] == ["erfpacht"]

    def test_the_classifiers_tags_are_used_when_none_were_declared(self, root: Path) -> None:
        written = file_one(root)
        frontmatter, _ = read_document(written.path)

        assert frontmatter["tags"] == ["house", "amsterdam"]

    def test_no_tags_at_all_omits_the_key(self, root: Path) -> None:
        written = file_one(root, outcome=verdict(tags=()))
        frontmatter, _ = read_document(written.path)

        assert "tags" not in frontmatter


class TestAtomicity:
    """KTD5's first half: the vault write lands whole, or not at all."""

    def test_an_interrupted_write_leaves_no_partial_file(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(src: Any, dst: Any) -> None:
            raise OSError("interrupted between temp write and rename")

        monkeypatch.setattr(os, "replace", refuse)

        with pytest.raises(OSError, match="interrupted"):
            file_one(root)

        assert [path for path in root.rglob("*") if path.is_file()] == []

    def test_an_interrupted_write_leaves_no_temp_file_behind(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("boom")))

        with pytest.raises(OSError):
            file_one(root)

        assert list(root.rglob("*.tmp")) == []

    def test_an_earlier_file_is_untouched_by_a_later_failure(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        survivor = file_one(root, record=drop(body="First."))
        before = survivor.path.read_text(encoding="utf-8")

        monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("boom")))
        with pytest.raises(OSError):
            file_one(root, record=drop(body="Second."))

        assert survivor.path.read_text(encoding="utf-8") == before

    def test_a_keyboard_interrupt_also_cleans_up(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupt(src: Any, dst: Any) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(os, "replace", interrupt)

        with pytest.raises(KeyboardInterrupt):
            file_one(root)

        assert [path for path in root.rglob("*") if path.is_file()] == []


class TestSlugs:
    """Filesystem-safe without being lossy."""

    def test_a_plain_title_becomes_a_hyphenated_slug(self) -> None:
        assert slugify("Coffee with Deniz") == "coffee-with-deniz"

    def test_path_separators_are_stripped(self) -> None:
        assert slugify("notes/2026/august") == "notes-2026-august"
        assert slugify("windows\\path\\thing") == "windows-path-thing"

    def test_a_traversal_attempt_cannot_survive_slugging(self) -> None:
        assert slugify("../../etc/passwd") == "etc-passwd"

    def test_filesystem_unsafe_characters_are_replaced(self) -> None:
        assert slugify('a:b*c?d"e<f>g|h') == "a-b-c-d-e-f-g-h"

    def test_control_characters_and_emoji_do_not_survive(self) -> None:
        assert slugify("release \x00party 🎉 time") == "release-party-time"

    def test_runs_of_separators_collapse_and_edges_are_trimmed(self) -> None:
        assert slugify("  --- hello ---  world --- ") == "hello-world"

    def test_turkish_letters_are_preserved_rather_than_folded(self) -> None:
        assert slugify("Açık ve kısa bir görüşme") == "açık-ve-kısa-bir-görüşme"

    @pytest.mark.parametrize(
        ("turkish", "ascii_lookalike"),
        [
            ("kısa", "kisa"),
            ("açık", "acik"),
            ("şu", "su"),
            ("göz", "goz"),
            ("üç", "uc"),
            ("değil", "degil"),
        ],
    )
    def test_distinct_titles_do_not_collapse_to_the_same_slug(
        self, turkish: str, ascii_lookalike: str
    ) -> None:
        assert slugify(turkish) != slugify(ascii_lookalike)

    def test_a_dotted_capital_i_lowercases_without_leaving_a_combining_mark(self) -> None:
        slug = slugify("İstanbul buluşması")

        assert slug == "istanbul-buluşması"
        assert "̇" not in slug

    def test_a_very_long_title_is_truncated_on_a_word_boundary(self) -> None:
        slug = slugify(" ".join(["sürdürülebilir"] * 20))

        assert len(slug) <= 80
        assert not slug.endswith("-")
        assert slug.split("-")[-1] == "sürdürülebilir"

    def test_a_single_unbroken_long_word_is_still_truncated(self) -> None:
        slug = slugify("a" * 300)

        assert len(slug) == 80

    def test_a_title_with_nothing_slugworthy_yields_an_empty_slug(self) -> None:
        assert slugify("🎉 --- ///") == ""

    def test_digits_and_other_scripts_survive(self) -> None:
        assert slugify("Q3 2026 планы") == "q3-2026-планы"


class TestSlugFallbacks:
    """The classifier's slug, its title, then the content id."""

    def test_the_classifiers_slug_is_used_when_it_is_usable(self, root: Path) -> None:
        written = file_one(root, outcome=verdict(slug="a-good-slug", title="Something Else"))

        assert written.slug == "a-good-slug"

    def test_the_title_is_slugged_when_no_slug_was_returned(self, root: Path) -> None:
        written = file_one(root, outcome=verdict(slug="", title="Erfpacht deadline"))

        assert written.slug == "erfpacht-deadline"

    def test_a_slug_that_is_itself_unsafe_is_cleaned_rather_than_trusted(self, root: Path) -> None:
        written = file_one(root, outcome=verdict(slug="../../escape/attempt"))

        assert written.slug == "escape-attempt"
        assert written.path.is_relative_to(root)

    def test_a_title_that_slugs_to_nothing_falls_back_to_the_content_id(self, root: Path) -> None:
        record = drop()

        written = file_one(root, record=record, outcome=verdict(slug="", title="🎉🎉🎉"))

        assert written.slug == f"untitled-{record.id[:8]}"
        assert written.path.is_file()

    def test_the_fallback_is_stable_across_runs(self, root: Path, tmp_path: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        record = drop()

        first = file_one(root, record=record, outcome=verdict(slug="", title="🎉"))
        second = file_one(other, record=record, outcome=verdict(slug="", title="🎉"))

        assert first.relative_path == second.relative_path


class TestPathTemplates:
    """Everything about layout comes from config; nothing is hardcoded."""

    def test_the_configured_transcript_template_is_used(self, root: Path) -> None:
        vault = vault_config("personal", root, transcript_template="raw/{date}-{slug}.md")

        written = file_one(root, vault=vault)

        assert written.relative_path == "raw/2026-08-01-coffee-with-deniz.md"

    def test_every_placeholder_renders(self) -> None:
        rendered = render_path(
            "{yyyy}/{mm}/{dd}/W{ww}/{date}-{slug}.md", day=date(2026, 8, 1), slug="s"
        )

        assert rendered == "2026/08/01/W31/2026-08-01-s.md"

    def test_a_week_numbered_template_files_by_iso_week(self, root: Path) -> None:
        vault = vault_config("personal", root, transcript_template="{yyyy}/W{ww}/{slug}.md")

        written = file_one(root, vault=vault)

        assert written.relative_path == "2026/W31/coffee-with-deniz.md"

    def test_an_unknown_placeholder_names_itself_and_the_supported_set(self) -> None:
        with pytest.raises(WriteError, match="nope"):
            render_path("{nope}/{slug}.md", day=INGESTED, slug="s")

    def test_the_error_lists_what_is_supported(self) -> None:
        with pytest.raises(WriteError, match=r"\{yyyy\}"):
            render_path("{nope}.md", day=INGESTED, slug="s")

    def test_a_template_that_escapes_the_vault_is_refused(self) -> None:
        with pytest.raises(WriteError, match="leaves the vault"):
            render_path("../outside/{slug}.md", day=INGESTED, slug="s")

    def test_an_absolute_template_is_refused(self) -> None:
        with pytest.raises(WriteError, match="leaves the vault"):
            render_path("/etc/{slug}.md", day=INGESTED, slug="s")

    def test_a_template_naming_no_file_is_refused(self) -> None:
        with pytest.raises(WriteError, match="names no file"):
            render_path("transcripts/{yyyy}/", day=INGESTED, slug="s")


class TestDestinationHandling:
    """The writer files raw bodies. It refuses to be the one that leaks them."""

    def test_a_note_destination_is_refused(self, root: Path, work_root: Path) -> None:
        personal = vault_config("personal", root, work_route="work")
        work = vault_config("work", work_root)
        record = drop()
        outcome = verdict(classification="work")
        routing = route(record, outcome, config_of(personal, work), personal)
        note = routing.of_kind(DestinationKind.NOTE)[0]

        with pytest.raises(WriteError, match="refusing to write the raw body"):
            file_record(record, outcome, routing, work_root, destination=note, ingested=INGESTED)

        assert list(work_root.rglob("*.md")) == []

    def test_work_material_still_files_raw_into_the_personal_vault(
        self, root: Path, work_root: Path
    ) -> None:
        personal = vault_config("personal", root, work_route="work")
        work = vault_config("work", work_root)
        record = drop()
        outcome = verdict(classification="work")
        routing = route(record, outcome, config_of(personal, work), personal)

        written = file_record(record, outcome, routing, root, ingested=INGESTED)

        assert written.vault == "personal"
        assert written.classification == "work"
        assert list(work_root.rglob("*.md")) == []

    def test_a_held_record_has_no_destination_to_file(self, root: Path) -> None:
        from memvault.classify import NeedsReview

        record = drop()
        vault = vault_config("personal", root)
        routing = route(record, NeedsReview("held"), config_of(vault), vault)

        with pytest.raises(WriteError, match="exactly one raw destination"):
            file_record(record, verdict(), routing, root, ingested=INGESTED)

        assert [path for path in root.rglob("*") if path.is_file()] == []

    def test_a_named_destination_overrides_the_routing_results_own(self, root: Path) -> None:
        record = drop()
        vault = vault_config("personal", root)
        routing = route(record, verdict(), config_of(vault), vault)
        elsewhere = Destination(
            vault="personal", kind=DestinationKind.RAW, path_template="archive/{slug}.md"
        )

        written = file_record(
            record, verdict(), routing, root, destination=elsewhere, ingested=INGESTED
        )

        assert written.relative_path == "archive/coffee-with-deniz.md"


class TestWriteDocument:
    """The lower-level entry point U5 files distilled notes through."""

    def test_it_writes_frontmatter_and_body(self, root: Path) -> None:
        placement = write_document(
            root,
            "notes/2026/note.md",
            frontmatter={"title": "A note", "date": "2026-08-01"},
            body="Distilled.",
        )

        assert placement.path.read_text(encoding="utf-8") == (
            "---\ntitle: A note\ndate: '2026-08-01'\n---\n\nDistilled.\n"
        )
        assert not placement.collided

    def test_it_suffixes_on_collision_too(self, root: Path) -> None:
        write_document(root, "notes/note.md", frontmatter={"title": "One"}, body="a")
        second = write_document(root, "notes/note.md", frontmatter={"title": "Two"}, body="b")

        assert second.relative_path == "notes/note-2.md"
        assert second.suffix == 2

    def test_a_key_outside_the_fixed_order_is_refused(self, root: Path) -> None:
        with pytest.raises(WriteError, match="fixed order"):
            write_document(root, "notes/note.md", frontmatter={"invented": "yes"}, body="a")

    def test_list_values_are_indented_under_their_key(self, root: Path) -> None:
        placement = write_document(
            root,
            "notes/note.md",
            frontmatter={"title": "A note", "participants": ["Ada", "Deniz"]},
            body="a",
        )

        assert "participants:\n  - Ada\n  - Deniz\n" in placement.path.read_text(encoding="utf-8")

    def test_an_empty_body_writes_frontmatter_alone(self, root: Path) -> None:
        placement = write_document(root, "notes/note.md", frontmatter={"title": "Bare"}, body="")

        assert placement.path.read_text(encoding="utf-8") == "---\ntitle: Bare\n---\n"


class TestImportance:
    """How much a filed item deserves to resurface — stamped at filing, ranked on much later."""

    def test_the_classifiers_importance_lands_in_the_frontmatter(self, root: Path) -> None:
        written = file_one(root, outcome=verdict(importance=7))
        frontmatter, _ = read_document(written.path)

        assert frontmatter["importance"] == 7

    def test_a_verdict_without_importance_files_cleanly_with_the_key_absent(
        self, root: Path
    ) -> None:
        """Absent means nobody judged, which recall reads as neutral — not as unimportant."""
        written = file_one(root)
        frontmatter, _ = read_document(written.path)

        assert "importance" not in frontmatter

    def test_a_declared_importance_wins_over_the_classifiers(self, root: Path) -> None:
        record = drop(importance=9)

        written = file_one(root, record=record, outcome=verdict(importance=2))
        frontmatter, _ = read_document(written.path)

        assert frontmatter["importance"] == 9

    def test_importance_keeps_its_place_in_the_fixed_order(self, root: Path) -> None:
        written = file_one(root, outcome=verdict(importance=7))
        frontmatter, _ = read_document(written.path)

        order = [key for key in FRONTMATTER_ORDER if key in frontmatter]

        assert list(frontmatter) == order


class TestSupersession:
    """`superseded_by` is never written by the filing path, but must be writable at all."""

    def test_render_document_accepts_it(self, root: Path) -> None:
        placement = write_document(
            root,
            "notes/note.md",
            frontmatter={"title": "Old", "superseded_by": "notes/newer.md"},
            body="a",
        )

        assert "superseded_by: notes/newer.md" in placement.path.read_text(encoding="utf-8")

    def test_a_genuinely_unknown_key_is_still_refused(self, root: Path) -> None:
        """The order stays a closed list: two new keys must not open it to any key."""
        with pytest.raises(WriteError, match="fixed order"):
            write_document(
                root,
                "notes/note.md",
                frontmatter={"title": "Old", "supersedes": "notes/older.md"},
                body="a",
            )


class TestRenderRelations:
    """Relation lines are what grows the graph, so their shape is pinned down here."""

    def test_a_relation_renders_as_a_typed_wikilink_bullet(self) -> None:
        rendered = render_relations((Relation("part_of", "repos/MemVault"),))

        assert "- part_of [[repos/MemVault]]" in rendered

    def test_the_block_is_headed_so_a_reader_knows_what_the_bullets_are(self) -> None:
        rendered = render_relations((Relation("about", "PdfToExcel"),))

        assert rendered.startswith(RELATIONS_HEADING)

    def test_no_relations_renders_nothing_rather_than_an_empty_heading(self) -> None:
        assert render_relations(()) == ""

    def test_relations_keep_the_order_they_were_given(self) -> None:
        rendered = render_relations(
            (Relation("part_of", "repos/MemVault"), Relation("about", "ranking"))
        )

        assert rendered.index("part_of") < rendered.index("about")
