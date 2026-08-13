"""Work-vault distillation and the verbatim-leakage guard.

This file is where R9 is either true or it is not. Two things are therefore tested harder than
anything else here.

The guard is tested as a mechanism rather than as a prompt: a stubbed distiller is made to copy
a sentence, change one word of it, recase it, re-punctuate it, and write it in Turkish, and each
of those must fail the work write. The tests that matter most assert what is *absent* — no file
in the work vault, no sentence of the transcript anywhere under its root.

And the asymmetry is tested end to end. AE3 is executed literally: a pair of scratch vaults, a
raw drop filed into the personal one by U4's writer, a note distilled into the work one, and
then a grep of the whole work vault for a verbatim sentence of the transcript, which must find
nothing.

No test spawns a subprocess. The distiller is injected through its protocol, and the two tests
that exercise the real CLI distiller drive it through a fake `CommandRunner`.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from memvault.classify import Classification, CommandResult, Relation
from memvault.config import ClassifierConfig, Config, IndexConfig, VaultConfig
from memvault.distill import (
    DEFAULT_OVERLAP_THRESHOLD,
    DEFAULT_SHINGLE_SIZE,
    ClaudeCliDistiller,
    Distillation,
    DistillationFailed,
    Distiller,
    DistillOutcome,
    DistillResult,
    build_distill_prompt,
    distill_record,
    note_slug,
    verbatim_overlap,
)
from memvault.inbox import DeclaredMetadata, InboxRecord, content_id
from memvault.route import Destination, DestinationKind, route
from memvault.writer import RELATIONS_HEADING, WriteError, file_record

INGESTED = date(2026, 8, 1)

#: One work conversation, written so its sentences are unmistakable when they turn up
#: somewhere they should not.
BODY = (
    "Ada: Sam called this morning about the price list tool. He is fed up with the manual "
    "copy and paste, and he wants one Excel tab per supplier before the end of the month.\n"
    "Deniz: We agreed to freeze the template editor until the Excel export is stable, because "
    "shipping both surfaces at once burned us on the last release.\n"
    "Ada: I will send Sam the revised quote on Monday, and then we should ship it."
)

#: The sentence AE3 greps the work vault for. Verbatim from `BODY`, minus the speaker label.
VERBATIM_SENTENCE = (
    "We agreed to freeze the template editor until the Excel export is stable, because "
    "shipping both surfaces at once burned us on the last release."
)

PARAPHRASE = Distillation(
    title="Price list tool priorities",
    decisions=("The editor work stays on hold until the spreadsheet output is dependable.",),
    action_items=("Send the client an updated quote at the start of next week.",),
    learnings=("Releasing two unfinished surfaces together has cost this team before.",),
    model="claude-sonnet-4-6",
)


def leaky(sentence: str = VERBATIM_SENTENCE) -> Distillation:
    """A distillation whose first decision is lifted straight out of the raw body."""
    return Distillation(
        title="Price list tool priorities",
        decisions=(sentence,),
        action_items=("Send the client an updated quote at the start of next week.",),
        model="claude-sonnet-4-6",
    )


class StubDistiller:
    """A `Distiller` that answers from a script and remembers whether it was asked."""

    def __init__(self, outcome: DistillOutcome | None = None) -> None:
        self.outcome: DistillOutcome = PARAPHRASE if outcome is None else outcome
        self.calls: list[tuple[InboxRecord, Classification]] = []

    def distill(self, record: InboxRecord, verdict: Classification) -> DistillOutcome:
        self.calls.append((record, verdict))
        return self.outcome


class FakeRunner:
    """A `CommandRunner` for the two tests that exercise the real CLI distiller."""

    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.calls: list[tuple[list[str], str, int]] = []

    def __call__(self, argv: Any, *, prompt: str, timeout: int) -> CommandResult:
        self.calls.append((list(argv), prompt, timeout))
        return self.result


def envelope(reply: Any) -> str:
    """What `claude -p --output-format json` prints: the reply inside a result object."""
    text = reply if isinstance(reply, str) else json.dumps(reply)
    return json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": text})


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


def drop(body: str = BODY, filename: str = "standup.md", **declared: Any) -> InboxRecord:
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
        "classification": "work",
        "confidence": 0.9,
        "title": "Standup with Sam",
        "slug": "standup-with-jordi",
        "summary": "The price list tool came up.",
        "participants": ("Ada", "Deniz"),
        "tags": ("pdftoexcel",),
        "project": "PdfToExcel",
        "model": "claude-sonnet-4-6",
    }
    settings.update(overrides)
    return Classification(**settings)


@pytest.fixture
def personal_root(tmp_path: Path) -> Path:
    path = tmp_path / "personal"
    path.mkdir()
    return path


@pytest.fixture
def work_root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def distil(
    work_root: Path,
    *,
    record: InboxRecord | None = None,
    outcome: Classification | None = None,
    distiller: Distiller | None = None,
    personal_root: Path | None = None,
    **kwargs: Any,
) -> DistillResult:
    """Route one record for real, then distil it. The seam under test starts after routing."""
    record = record or drop()
    outcome = outcome or verdict()
    personal = vault_config("personal", personal_root or work_root.parent, work_route="work")
    work = vault_config("work", work_root)
    routing = route(record, outcome, config_of(personal, work), personal)
    return distill_record(
        record,
        outcome,
        routing,
        work_root,
        distiller=distiller or StubDistiller(),
        ingested=kwargs.pop("ingested", INGESTED),
        **kwargs,
    )


def work_files(work_root: Path) -> list[Path]:
    return [path for path in work_root.rglob("*") if path.is_file()]


def read_document(path: Path) -> tuple[dict[str, Any], str]:
    """Split a written note back into frontmatter and body, the way any reader would."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, frontmatter, body = text.split("---\n", 2)
    parsed = yaml.safe_load(frontmatter)
    assert isinstance(parsed, dict)
    return parsed, body


class TestParaphrasePasses:
    """The ordinary path: a note that says what happened without saying how it was said."""

    def test_a_paraphrased_note_is_written(self, work_root: Path) -> None:
        result = distil(work_root)

        assert result.wrote
        assert result.written is not None
        assert result.written.path.is_file()

    def test_it_lands_at_the_work_vaults_configured_note_path(self, work_root: Path) -> None:
        result = distil(work_root)

        assert result.written is not None
        assert result.written.relative_path == "notes/2026/2026-08-01-price-list-tool-priorities.md"
        assert result.written.vault == "work"
        assert result.written.kind is DestinationKind.NOTE

    def test_the_body_carries_every_non_empty_section(self, work_root: Path) -> None:
        result = distil(work_root)

        assert result.written is not None
        _, body = read_document(result.written.path)

        assert "## Decisions" in body
        assert "## Action items" in body
        assert "## Learnings" in body
        assert "- Send the client an updated quote at the start of next week." in body

    def test_an_empty_section_is_left_out_rather_than_written_bare(self, work_root: Path) -> None:
        result = distil(work_root, distiller=StubDistiller(leaky("Nothing copied here at all.")))

        assert result.written is not None
        _, body = read_document(result.written.path)

        assert "## Learnings" not in body
        assert "## Decisions" in body

    def test_the_guard_ran_and_found_nothing(self, work_root: Path) -> None:
        result = distil(work_root)

        assert result.leakage is not None
        assert result.leakage.overlap == 0.0
        assert not result.leakage.tripped
        assert result.leakage.sentences_checked == 4

    def test_the_result_carries_what_a_commit_message_needs(self, work_root: Path) -> None:
        result = distil(work_root)

        assert result.written is not None
        assert result.written.title == "Price list tool priorities"
        assert result.written.classification == "work"
        assert result.written.summary_line().endswith(
            "notes/2026/2026-08-01-price-list-tool-priorities.md"
        )

    def test_writing_is_logged_with_both_ends_of_the_move(
        self, work_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="memvault.distill"):
            distil(work_root)

        assert "distilled standup.md into work/notes/" in caplog.text


class TestNoteProvenance:
    """The note points back at the personal file by id, and by nothing else."""

    def test_the_personal_file_is_referenced_by_content_id(self, work_root: Path) -> None:
        record = drop()

        result = distil(work_root, record=record)

        assert result.written is not None
        frontmatter, body = read_document(result.written.path)
        assert frontmatter["content_id"] == record.id
        assert record.id in body

    def test_the_footer_names_the_distiller_and_where_the_material_stays(
        self, work_root: Path
    ) -> None:
        result = distil(work_root)

        assert result.written is not None
        _, body = read_document(result.written.path)

        assert "*Distilled by claude-sonnet-4-6 from personal-vault item" in body
        assert "stays in that vault" in body

    def test_the_drops_own_filename_never_crosses(self, work_root: Path) -> None:
        record = drop(filename="acme-layoffs-call.md")

        result = distil(work_root, record=record)

        assert result.written is not None
        assert "acme-layoffs-call" not in result.written.path.read_text(encoding="utf-8")

    def test_the_classifiers_summary_of_the_raw_material_never_crosses(
        self, work_root: Path
    ) -> None:
        outcome = verdict(summary="Sam is furious about the manual copy and paste.")

        result = distil(work_root, outcome=outcome)

        assert result.written is not None
        assert "furious" not in result.written.path.read_text(encoding="utf-8")

    def test_routing_metadata_survives_into_the_note(self, work_root: Path) -> None:
        result = distil(work_root)

        assert result.written is not None
        frontmatter, _ = read_document(result.written.path)

        assert frontmatter["classification"] == "work"
        assert frontmatter["project"] == "PdfToExcel"
        assert frontmatter["participants"] == ["Ada", "Deniz"]
        assert frontmatter["tags"] == ["pdftoexcel"]
        assert frontmatter["ingested"] == "2026-08-01"

    def test_the_note_is_dated_the_same_day_as_the_transcript(self, work_root: Path) -> None:
        result = distil(work_root, record=drop(date="2026-07-28"))

        assert result.written is not None
        frontmatter, _ = read_document(result.written.path)

        assert frontmatter["date"] == "2026-07-28"
        assert result.written.relative_path == "notes/2026/2026-07-28-price-list-tool-priorities.md"

    def test_declared_tags_win_here_too(self, work_root: Path) -> None:
        result = distil(work_root, record=drop(tags=("pricing",)))

        assert result.written is not None
        frontmatter, _ = read_document(result.written.path)

        assert frontmatter["tags"] == ["pricing"]

    def test_a_second_note_with_the_same_name_is_suffixed_rather_than_overwriting(
        self, work_root: Path
    ) -> None:
        first = distil(work_root, record=drop(body=BODY + "\nAda: One more thing."))
        second = distil(work_root)

        assert first.written is not None
        assert second.written is not None
        assert second.written.relative_path.endswith("-2.md")
        assert len(work_files(work_root)) == 2


class TestCrossVaultRelations:
    """A relation may only name something the work vault already holds.

    A wikilink pointing at a personal-vault path would describe the private vault's shape inside
    a shared file — the same disclosure the shingle guard exists to prevent, arriving through a
    field the shingle guard has no opinion about. The restriction is enforced here, in the writer
    path, rather than asked for in the prompt: a model that ignores the request must still fail
    to leak.
    """

    def existing_note(self, work_root: Path, relative: str = "notes/2026/roadmap.md") -> str:
        path = work_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\ntitle: Roadmap\n---\n\nA note already in the work vault.\n")
        return relative

    def test_a_relation_resolving_inside_the_work_vault_is_written(self, work_root: Path) -> None:
        target = self.existing_note(work_root)
        outcome = verdict(relations=(Relation("part_of", target),))

        result = distil(work_root, outcome=outcome)

        assert result.written is not None
        _, body = read_document(result.written.path)
        assert f"- part_of [[{target}]]" in body

    def test_a_target_named_without_its_extension_still_resolves(self, work_root: Path) -> None:
        """`[[notes/2026/roadmap]]` is how a wikilink is normally written."""
        self.existing_note(work_root)
        outcome = verdict(relations=(Relation("about", "notes/2026/roadmap"),))

        result = distil(work_root, outcome=outcome)

        assert result.written is not None
        assert "[[notes/2026/roadmap]]" in read_document(result.written.path)[1]

    def test_a_personal_vault_path_is_dropped_and_the_note_still_files(
        self, work_root: Path
    ) -> None:
        """Fail toward filing with fewer relations: the note's body is unaffected by the drop."""
        outcome = verdict(
            relations=(Relation("about", "transcripts/2026/08/2026-08-01-acme-layoffs.md"),)
        )

        result = distil(work_root, outcome=outcome)

        assert result.wrote
        assert result.written is not None
        assert "acme-layoffs" not in result.written.path.read_text(encoding="utf-8")

    def test_the_dropped_relation_is_logged(
        self, work_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        outcome = verdict(relations=(Relation("about", "repos/Homeworld/learnings.md"),))

        with caplog.at_level(logging.INFO, logger="memvault.distill"):
            distil(work_root, outcome=outcome)

        assert "relation" in caplog.text
        assert "repos/Homeworld/learnings.md" in caplog.text

    def test_a_note_whose_relations_were_all_dropped_carries_no_relations_block(
        self, work_root: Path
    ) -> None:
        outcome = verdict(relations=(Relation("about", "notes/2026/never-written.md"),))

        result = distil(work_root, outcome=outcome)

        assert result.written is not None
        assert RELATIONS_HEADING not in result.written.path.read_text(encoding="utf-8")

    def test_a_surviving_relation_is_still_read_by_the_leakage_guard(self, work_root: Path) -> None:
        """A leaked sentence promoted into a relation target must not slip past the guard."""
        target = self.existing_note(work_root, f"notes/2026/{VERBATIM_SENTENCE[:80]}.md")
        outcome = verdict(relations=(Relation("about", target),))

        result = distil(work_root, outcome=outcome)

        assert not result.wrote
        assert result.needs_review


class TestNoteImportance:
    """The work note carries the same 1-10 judgement the personal file does."""

    def test_the_classifiers_importance_reaches_the_note(self, work_root: Path) -> None:
        result = distil(work_root, outcome=verdict(importance=6))

        assert result.written is not None
        assert read_document(result.written.path)[0]["importance"] == 6

    def test_a_note_without_importance_omits_the_key(self, work_root: Path) -> None:
        result = distil(work_root)

        assert result.written is not None
        assert "importance" not in read_document(result.written.path)[0]


class TestLeakageBlocksTheWrite:
    """A copied sentence fails the write. This is the whole unit."""

    def test_a_copied_sentence_trips_the_guard(self, work_root: Path) -> None:
        result = distil(work_root, distiller=StubDistiller(leaky()))

        assert result.leakage is not None
        assert result.leakage.tripped
        assert result.leakage.overlap > 0.9

    def test_nothing_is_written_to_the_work_vault(self, work_root: Path) -> None:
        distil(work_root, distiller=StubDistiller(leaky()))

        assert work_files(work_root) == []

    def test_the_item_is_marked_for_review(self, work_root: Path) -> None:
        result = distil(work_root, distiller=StubDistiller(leaky()))

        assert result.needs_review
        assert not result.wrote
        assert result.review is not None
        assert "verbatim" in result.review

    def test_the_reason_names_the_numbers_and_where_to_look(self, work_root: Path) -> None:
        result = distil(work_root, distiller=StubDistiller(leaky()))

        assert result.review is not None
        assert "sentence 3 of the raw body" in result.review
        assert "threshold of 40%" in result.review

    def test_the_reason_does_not_reproduce_the_leaked_text(self, work_root: Path) -> None:
        result = distil(work_root, distiller=StubDistiller(leaky()))

        assert result.review is not None
        assert "template editor" not in result.review
        assert "\n" not in result.review

    def test_blocking_is_logged_as_a_warning(
        self, work_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="memvault.distill"):
            distil(work_root, distiller=StubDistiller(leaky()))

        assert "standup.md: work note blocked" in caplog.text

    def test_the_personal_vault_file_is_unaffected(
        self, personal_root: Path, work_root: Path
    ) -> None:
        record = drop()
        outcome = verdict()
        personal = vault_config("personal", personal_root, work_route="work")
        work = vault_config("work", work_root)
        routing = route(record, outcome, config_of(personal, work), personal)
        filed = file_record(record, outcome, routing, personal_root, ingested=INGESTED)
        before = filed.path.read_text(encoding="utf-8")

        distill_record(
            record,
            outcome,
            routing,
            work_root,
            distiller=StubDistiller(leaky()),
            ingested=INGESTED,
        )

        assert filed.path.read_text(encoding="utf-8") == before
        assert VERBATIM_SENTENCE in before
        assert work_files(work_root) == []

    def test_a_copy_hidden_among_paraphrase_still_trips(self, work_root: Path) -> None:
        smuggled = Distillation(
            title="Price list tool priorities",
            decisions=("The editor work stays on hold until the spreadsheet output is ready.",),
            action_items=("Send the client an updated quote at the start of next week.",),
            learnings=(
                "Nothing surprising came up.",
                VERBATIM_SENTENCE,
                "The team is comfortable with the plan.",
            ),
        )

        result = distil(work_root, distiller=StubDistiller(smuggled))

        assert result.leakage is not None and result.leakage.tripped
        assert work_files(work_root) == []

    def test_a_copy_promoted_into_the_title_cannot_slip_through_the_filename(
        self, work_root: Path
    ) -> None:
        titled = Distillation(
            title=VERBATIM_SENTENCE,
            decisions=("The editor work stays on hold until the spreadsheet output is ready.",),
        )

        result = distil(work_root, distiller=StubDistiller(titled))

        assert result.needs_review
        assert work_files(work_root) == []


class TestNearMisses:
    """Cosmetic edits to a copied sentence do not make it a paraphrase."""

    def test_one_changed_word_still_trips(self, work_root: Path) -> None:
        edited = VERBATIM_SENTENCE.replace("stable", "solid")

        result = distil(work_root, distiller=StubDistiller(leaky(edited)))

        assert result.leakage is not None
        assert result.leakage.overlap > 0.8
        assert result.leakage.tripped
        assert work_files(work_root) == []

    def test_recasing_and_re_punctuating_change_nothing(self, work_root: Path) -> None:
        disguised = (
            "we agreed to FREEZE the template editor — until the Excel export is stable; "
            "because shipping both surfaces at once burned us on the last release!!"
        )

        result = distil(work_root, distiller=StubDistiller(leaky(disguised)))

        assert result.leakage is not None and result.leakage.tripped
        assert work_files(work_root) == []

    def test_reflowing_across_line_breaks_changes_nothing(self, work_root: Path) -> None:
        reflowed = VERBATIM_SENTENCE.replace(", because", ",\nbecause")

        result = distil(work_root, distiller=StubDistiller(leaky(reflowed)))

        assert result.leakage is not None and result.leakage.tripped
        assert work_files(work_root) == []

    def test_swapping_a_word_at_the_very_start_still_trips(self, work_root: Path) -> None:
        edited = "They agreed to freeze the template editor until the Excel export is stable, "
        edited += "because shipping both surfaces at once burned us on the last release."

        result = distil(work_root, distiller=StubDistiller(leaky(edited)))

        assert result.leakage is not None and result.leakage.tripped


class TestShortCommonPhrases:
    """The guard must not fire on language nobody owns."""

    def test_we_should_ship_it_does_not_trip(self, work_root: Path) -> None:
        harmless = Distillation(
            title="Price list tool priorities",
            decisions=("We should ship it.",),
            action_items=("Send the client an updated quote at the start of next week.",),
        )

        result = distil(work_root, distiller=StubDistiller(harmless))

        assert result.wrote
        assert result.leakage is not None
        assert result.leakage.overlap == 0.0

    @pytest.mark.parametrize(
        "phrase",
        [
            "We should ship it.",
            "Let us ship it.",
            "He is fed up.",
            "Before the end of the month.",
        ],
    )
    def test_short_phrases_lifted_from_the_body_still_pass(
        self, work_root: Path, phrase: str
    ) -> None:
        result = distil(
            work_root,
            distiller=StubDistiller(
                Distillation(title="Priorities", decisions=(phrase,), action_items=("Move on.",))
            ),
        )

        assert result.wrote, f"{phrase!r} should be too short to be evidence of copying"

    def test_a_short_raw_sentence_cannot_be_leaked_at_all(self, work_root: Path) -> None:
        record = drop(body="Ship it on Monday.")

        result = distil(
            work_root,
            record=record,
            distiller=StubDistiller(
                Distillation(title="Priorities", decisions=("Ship it on Monday.",))
            ),
        )

        assert result.wrote
        assert result.leakage is not None
        assert result.leakage.sentences_checked == 0


class TestTurkish:
    """The guard is not an English-language mechanism."""

    TURKISH_BODY = (
        "Ada: Sam bu sabah fiyat listesi aracı hakkında aradı.\n"
        "Deniz: Ekip, şablon düzenleyicisini Excel dışa aktarımı kararlı hale gelene kadar "
        "dondurmaya karar verdi.\n"
        "Ada: Pazartesi günü revize teklifi göndereceğim."
    )

    def test_a_copied_turkish_sentence_trips(self, work_root: Path) -> None:
        copied = (
            "Ekip, şablon düzenleyicisini Excel dışa aktarımı kararlı hale gelene kadar "
            "dondurmaya karar verdi."
        )

        result = distil(
            work_root,
            record=drop(body=self.TURKISH_BODY),
            distiller=StubDistiller(Distillation(title="Öncelikler", decisions=(copied,))),
        )

        assert result.leakage is not None
        assert result.leakage.overlap > 0.9
        assert result.leakage.tripped
        assert work_files(work_root) == []

    def test_a_turkish_paraphrase_passes(self, work_root: Path) -> None:
        paraphrase = Distillation(
            title="Fiyat listesi önceliği",
            decisions=("Dışa aktarım işi bitene kadar düzenleyici beklemede kalacak.",),
            action_items=("Yeni teklif hafta başında müşteriye gidecek.",),
        )

        result = distil(
            work_root, record=drop(body=self.TURKISH_BODY), distiller=StubDistiller(paraphrase)
        )

        assert result.wrote
        assert result.leakage is not None
        assert result.leakage.overlap == 0.0

    def test_a_recased_turkish_copy_still_trips(self, work_root: Path) -> None:
        copied = (
            "EKİP, ŞABLON DÜZENLEYİCİSİNİ Excel dışa aktarımı kararlı hale gelene kadar "
            "dondurmaya karar verdi!"
        )

        result = distil(
            work_root,
            record=drop(body=self.TURKISH_BODY),
            distiller=StubDistiller(Distillation(title="Öncelikler", decisions=(copied,))),
        )

        assert result.leakage is not None and result.leakage.tripped


class TestEmptyDistillation:
    """An unremarkable standup should not litter the work vault."""

    def test_nothing_is_written(self, work_root: Path) -> None:
        result = distil(work_root, distiller=StubDistiller(Distillation(title="Standup")))

        assert work_files(work_root) == []
        assert not result.wrote

    def test_it_is_a_skip_rather_than_a_review(self, work_root: Path) -> None:
        result = distil(work_root, distiller=StubDistiller(Distillation(title="Standup")))

        assert not result.needs_review
        assert result.skipped is not None
        assert "no decisions, action items, or learnings" in result.skipped

    def test_the_reason_is_logged(self, work_root: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="memvault.distill"):
            distil(work_root, distiller=StubDistiller(Distillation(title="Standup")))

        assert "standup.md: the distiller found no decisions" in caplog.text

    def test_a_title_alone_is_not_content(self, work_root: Path) -> None:
        assert Distillation(title="Standup").empty
        assert not Distillation(learnings=("Something durable.",)).empty


class TestDistillerFailure:
    """A failed distiller costs a note, never the memory."""

    def test_a_failure_writes_nothing_and_asks_for_review(self, work_root: Path) -> None:
        failed = StubDistiller(DistillationFailed("the distiller exited 1: credit balance too low"))

        result = distil(work_root, distiller=failed)

        assert work_files(work_root) == []
        assert result.needs_review
        assert result.review is not None
        assert "exited 1" in result.review

    def test_the_personal_filing_stands(self, personal_root: Path, work_root: Path) -> None:
        record = drop()
        outcome = verdict()
        personal = vault_config("personal", personal_root, work_route="work")
        work = vault_config("work", work_root)
        routing = route(record, outcome, config_of(personal, work), personal)
        filed = file_record(record, outcome, routing, personal_root, ingested=INGESTED)

        result = distill_record(
            record,
            outcome,
            routing,
            work_root,
            distiller=StubDistiller(DistillationFailed("invalid output")),
            ingested=INGESTED,
        )

        assert filed.path.is_file()
        assert VERBATIM_SENTENCE in filed.path.read_text(encoding="utf-8")
        assert result.written is None

    def test_no_leakage_report_is_invented_for_a_failure(self, work_root: Path) -> None:
        result = distil(work_root, distiller=StubDistiller(DistillationFailed("boom")))

        assert result.leakage is None
        assert result.distillation is None


class TestConfigurableThreshold:
    """The threshold is a parameter, and so is the shingle size."""

    def test_the_defaults_are_the_documented_ones(self) -> None:
        assert DEFAULT_SHINGLE_SIZE == 6
        assert DEFAULT_OVERLAP_THRESHOLD == 0.4

    def test_raising_the_threshold_lets_a_copy_through(self, work_root: Path) -> None:
        result = distil(work_root, distiller=StubDistiller(leaky()), overlap_threshold=1.0)

        assert result.wrote
        assert result.leakage is not None
        assert result.leakage.overlap > 0.9

    def test_lowering_the_threshold_catches_a_partial_borrowing(self, work_root: Path) -> None:
        borrowed = Distillation(
            title="Priorities",
            decisions=("The team will freeze the template editor until the Excel export settles.",),
        )

        lenient = distil(work_root, distiller=StubDistiller(borrowed))
        strict = distil(work_root, distiller=StubDistiller(borrowed), overlap_threshold=0.2)

        assert lenient.wrote
        assert strict.needs_review
        assert strict.leakage is not None
        assert 0.2 < strict.leakage.overlap <= DEFAULT_OVERLAP_THRESHOLD

    def test_a_score_exactly_at_the_threshold_passes(self) -> None:
        report = verbatim_overlap("a b c d e f g h", "a b c d e f", shingle_size=6, threshold=0.75)

        assert report.overlap == 0.75
        assert not report.tripped

    def test_a_zero_threshold_blocks_any_matching_run_without_blocking_clean_notes(
        self, work_root: Path
    ) -> None:
        clean = distil(work_root, overlap_threshold=0.0)
        copied = distil(work_root, distiller=StubDistiller(leaky()), overlap_threshold=0.0)

        assert clean.wrote
        assert copied.needs_review

    def test_a_larger_shingle_size_is_more_permissive(self, work_root: Path) -> None:
        borrowed = Distillation(
            title="Priorities",
            decisions=("The team will freeze the template editor until the Excel export settles.",),
        )

        coarse = distil(work_root, distiller=StubDistiller(borrowed), shingle_size=12)
        fine = distil(
            work_root, distiller=StubDistiller(borrowed), shingle_size=4, overlap_threshold=0.2
        )

        assert coarse.leakage is not None and coarse.leakage.overlap == 0.0
        assert fine.leakage is not None and fine.leakage.overlap > coarse.leakage.overlap


class TestOverlapMeasurement:
    """The guard on its own, away from files."""

    def test_an_exact_copy_scores_one(self) -> None:
        sentence = "one two three four five six seven eight"

        assert verbatim_overlap(sentence, sentence).overlap == 1.0

    def test_unrelated_text_scores_zero(self) -> None:
        report = verbatim_overlap(
            "The template editor is frozen until the export stabilises.",
            "Nothing here resembles that at all, in any respect whatsoever.",
        )

        assert report.overlap == 0.0

    def test_a_sentence_shorter_than_a_shingle_is_not_scored(self) -> None:
        report = verbatim_overlap("we should ship it", "we should ship it")

        assert report.overlap == 0.0
        assert report.sentences_checked == 0

    def test_the_worst_sentence_is_the_score(self) -> None:
        raw = "One two three four five six seven. Alpha beta gamma delta epsilon zeta eta."

        report = verbatim_overlap(raw, "Alpha beta gamma delta epsilon zeta eta.")

        assert report.overlap == 1.0
        assert report.sentence_index == 1
        assert report.sentence_words == 7

    def test_a_long_note_does_not_dilute_a_copied_sentence(self) -> None:
        raw = "Alpha beta gamma delta epsilon zeta eta theta."
        padding = " ".join(f"filler{n}" for n in range(400))

        report = verbatim_overlap(raw, f"{padding} Alpha beta gamma delta epsilon zeta eta theta.")

        assert report.overlap == 1.0

    def test_line_breaks_in_the_raw_body_split_sentences(self) -> None:
        report = verbatim_overlap("first line here\nsecond line there", "nothing")

        assert report.sentences_checked == 0

    def test_a_shingle_size_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            verbatim_overlap("a b c", "a b c", shingle_size=0)

    def test_an_empty_body_scores_zero_rather_than_raising(self) -> None:
        report = verbatim_overlap("", "anything at all goes here for a while")

        assert report.overlap == 0.0
        assert report.sentence_index is None

    def test_the_reason_reads_without_a_sentence_to_point_at(self) -> None:
        report = verbatim_overlap("", "")

        assert "the raw body" in report.reason()


class TestNothingOwed:
    """Personal material never reaches an LLM for a note nobody asked for."""

    def test_a_personal_record_is_a_no_op(self, work_root: Path) -> None:
        distiller = StubDistiller()

        result = distil(work_root, outcome=verdict(classification="personal"), distiller=distiller)

        assert distiller.calls == []
        assert result.skipped == "no work-vault note is owed"
        assert work_files(work_root) == []

    def test_a_local_only_record_is_a_no_op(self, work_root: Path) -> None:
        distiller = StubDistiller()

        result = distil(work_root, record=drop(local_only=True), distiller=distiller)

        assert distiller.calls == []
        assert result.skipped is not None
        assert work_files(work_root) == []

    def test_a_vault_with_no_work_route_is_a_no_op(self, work_root: Path) -> None:
        record, outcome = drop(), verdict()
        personal = vault_config("personal", work_root.parent)
        routing = route(record, outcome, config_of(personal), personal)
        distiller = StubDistiller()

        result = distill_record(
            record, outcome, routing, work_root, distiller=distiller, ingested=INGESTED
        )

        assert distiller.calls == []
        assert result.skipped is not None

    def test_a_held_record_is_a_no_op(self, work_root: Path) -> None:
        from memvault.classify import NeedsReview

        record = drop()
        personal = vault_config("personal", work_root.parent, work_route="work")
        work = vault_config("work", work_root)
        routing = route(record, NeedsReview("held"), config_of(personal, work), personal)
        distiller = StubDistiller()

        result = distill_record(
            record, verdict(), routing, work_root, distiller=distiller, ingested=INGESTED
        )

        assert distiller.calls == []
        assert work_files(work_root) == []
        assert not result.needs_review

    def test_a_raw_destination_is_refused(self, work_root: Path) -> None:
        record, outcome = drop(), verdict()
        personal = vault_config("personal", work_root.parent, work_route="work")
        work = vault_config("work", work_root)
        routing = route(record, outcome, config_of(personal, work), personal)
        raw = routing.of_kind(DestinationKind.RAW)[0]

        with pytest.raises(WriteError, match="refusing to distil"):
            distill_record(
                record,
                outcome,
                routing,
                work_root,
                distiller=StubDistiller(),
                destination=raw,
                ingested=INGESTED,
            )

        assert work_files(work_root) == []

    def test_a_named_destination_overrides_the_routing_results_own(self, work_root: Path) -> None:
        elsewhere = Destination(
            vault="work", kind=DestinationKind.NOTE, path_template="digest/{slug}.md"
        )
        record, outcome = drop(), verdict()
        personal = vault_config("personal", work_root.parent, work_route="work")
        work = vault_config("work", work_root)
        routing = route(record, outcome, config_of(personal, work), personal)

        result = distill_record(
            record,
            outcome,
            routing,
            work_root,
            distiller=StubDistiller(),
            destination=elsewhere,
            ingested=INGESTED,
        )

        assert result.written is not None
        assert result.written.relative_path == "digest/price-list-tool-priorities.md"


class TestAE3EndToEnd:
    """AE3, executed literally against a scratch pair of vaults.

    A `work` drop is filed raw into the personal vault and distilled into the work vault, and
    then every file under the work vault's root is searched for a verbatim sentence of the
    transcript. Finding one would mean R9 is false.
    """

    @staticmethod
    def _run(personal_root: Path, work_root: Path) -> tuple[Path, Path]:
        record, outcome = drop(), verdict()
        personal = vault_config("personal", personal_root, work_route="work")
        work = vault_config("work", work_root)
        config = config_of(personal, work)
        routing = route(record, outcome, config, personal)

        filed = file_record(record, outcome, routing, personal_root, ingested=INGESTED)
        result = distill_record(
            record, outcome, routing, work_root, distiller=StubDistiller(), ingested=INGESTED
        )

        assert result.written is not None
        return filed.path, result.written.path

    def test_the_raw_transcript_is_in_the_personal_vault(
        self, personal_root: Path, work_root: Path
    ) -> None:
        raw_path, _ = self._run(personal_root, work_root)

        assert VERBATIM_SENTENCE in raw_path.read_text(encoding="utf-8")

    def test_grepping_the_work_vault_for_a_verbatim_sentence_finds_nothing(
        self, personal_root: Path, work_root: Path
    ) -> None:
        self._run(personal_root, work_root)

        hits = [
            path
            for path in work_files(work_root)
            if VERBATIM_SENTENCE in path.read_text(encoding="utf-8")
        ]

        assert hits == []

    @pytest.mark.parametrize(
        "fragment",
        [
            "Sam called this morning about the price list tool",
            "fed up with the manual copy and paste",
            "one Excel tab per supplier before the end of the month",
            "freeze the template editor until the Excel export is stable",
            "burned us on the last release",
            "send Sam the revised quote on Monday",
        ],
    )
    def test_no_fragment_of_the_transcript_reaches_the_work_vault(
        self, personal_root: Path, work_root: Path, fragment: str
    ) -> None:
        self._run(personal_root, work_root)

        text = "\n".join(path.read_text(encoding="utf-8") for path in work_files(work_root))

        assert fragment not in text

    def test_the_work_vault_holds_exactly_one_file_and_it_is_the_note(
        self, personal_root: Path, work_root: Path
    ) -> None:
        _, note_path = self._run(personal_root, work_root)

        assert work_files(work_root) == [note_path]
        assert "## Decisions" in note_path.read_text(encoding="utf-8")

    def test_the_note_and_the_transcript_agree_on_which_item_they_describe(
        self, personal_root: Path, work_root: Path
    ) -> None:
        raw_path, note_path = self._run(personal_root, work_root)

        raw_frontmatter, _ = read_document(raw_path)
        note_frontmatter, _ = read_document(note_path)

        assert note_frontmatter["content_id"] == raw_frontmatter["content_id"]


class TestSlugChoice:
    """The distiller's title names the note; the classifier's is the fallback."""

    def test_the_distillers_title_wins(self) -> None:
        slug = note_slug(drop(), verdict(), PARAPHRASE)

        assert slug == "price-list-tool-priorities"

    def test_the_classifiers_slug_is_the_fallback(self) -> None:
        slug = note_slug(drop(), verdict(), Distillation(learnings=("Something.",)))

        assert slug == "standup-with-jordi"

    def test_the_content_id_is_the_last_resort(self) -> None:
        record = drop()

        slug = note_slug(record, verdict(slug="", title="🎉"), Distillation())

        assert slug == f"untitled-{record.id[:8]}"

    def test_a_turkish_title_keeps_its_letters(self) -> None:
        slug = note_slug(drop(), verdict(), Distillation(title="Açık kararlar"))

        assert slug == "açık-kararlar"


class TestClaudeCliDistiller:
    """The real distiller, driven through a fake runner. No subprocess, no network."""

    def test_the_configured_command_is_asked_for_json_output(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope({"title": "t", "decisions": ["d"]})))

        ClaudeCliDistiller(ClassifierConfig(), runner=runner).distill(drop(), verdict())

        assert runner.calls[-1][0] == ["claude", "-p", "--output-format", "json"]

    def test_a_configured_model_is_passed_through_and_recorded(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope({"title": "t", "learnings": ["l"]})))

        outcome = ClaudeCliDistiller(
            ClassifierConfig(model="claude-haiku-4-5"), runner=runner
        ).distill(drop(), verdict())

        assert runner.calls[-1][0][-2:] == ["--model", "claude-haiku-4-5"]
        assert isinstance(outcome, Distillation)
        assert outcome.model == "claude-haiku-4-5"

    def test_a_well_formed_reply_becomes_a_distillation(self) -> None:
        reply = {
            "title": "Price list tool priorities",
            "decisions": ["Editor work is on hold."],
            "action_items": ["- Send the quote."],
            "learnings": ["Two surfaces at once is a mistake."],
        }
        runner = FakeRunner(CommandResult(0, envelope(reply)))

        outcome = ClaudeCliDistiller(ClassifierConfig(), runner=runner).distill(drop(), verdict())

        assert outcome == Distillation(
            title="Price list tool priorities",
            decisions=("Editor work is on hold.",),
            action_items=("Send the quote.",),
            learnings=("Two surfaces at once is a mistake.",),
        )

    def test_a_bare_string_list_field_is_accepted(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope({"decisions": "Only one decision."})))

        outcome = ClaudeCliDistiller(ClassifierConfig(), runner=runner).distill(drop(), verdict())

        assert isinstance(outcome, Distillation)
        assert outcome.decisions == ("Only one decision.",)

    def test_an_empty_reply_object_is_an_empty_distillation_not_a_failure(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope({})))

        outcome = ClaudeCliDistiller(ClassifierConfig(), runner=runner).distill(drop(), verdict())

        assert isinstance(outcome, Distillation)
        assert outcome.empty

    def test_a_non_zero_exit_fails_with_its_stderr(self) -> None:
        runner = FakeRunner(CommandResult(1, stderr="Credit balance is too low"))

        outcome = ClaudeCliDistiller(ClassifierConfig(), runner=runner).distill(drop(), verdict())

        assert isinstance(outcome, DistillationFailed)
        assert "exited 1" in outcome.reason
        assert "Credit balance is too low" in outcome.reason

    def test_a_timeout_fails_and_names_the_limit(self) -> None:
        runner = FakeRunner(CommandResult(exit_code=-1, timed_out=True))

        outcome = ClaudeCliDistiller(ClassifierConfig(timeout_seconds=90), runner=runner).distill(
            drop(), verdict()
        )

        assert isinstance(outcome, DistillationFailed)
        assert "90s" in outcome.reason

    def test_invalid_json_fails_rather_than_raising(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope('{"decisions": [')))

        outcome = ClaudeCliDistiller(ClassifierConfig(), runner=runner).distill(drop(), verdict())

        assert isinstance(outcome, DistillationFailed)
        assert "bare JSON object" in outcome.reason

    def test_a_json_array_instead_of_an_object_fails(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope([{"decisions": []}])))

        outcome = ClaudeCliDistiller(ClassifierConfig(), runner=runner).distill(drop(), verdict())

        assert isinstance(outcome, DistillationFailed)
        assert "not an object" in outcome.reason

    def test_failure_reasons_name_the_distiller_rather_than_the_classifier(self) -> None:
        runner = FakeRunner(CommandResult(0, stdout="I cannot do that."))

        outcome = ClaudeCliDistiller(ClassifierConfig(), runner=runner).distill(drop(), verdict())

        assert isinstance(outcome, DistillationFailed)
        assert "the distiller's output was not JSON" in outcome.reason
        assert "classifier" not in outcome.reason

    def test_a_failure_stays_on_one_line_so_it_fits_a_review_marker(self) -> None:
        runner = FakeRunner(CommandResult(2, stderr="first line\nsecond line\n"))

        outcome = ClaudeCliDistiller(ClassifierConfig(), runner=runner).distill(drop(), verdict())

        assert isinstance(outcome, DistillationFailed)
        assert "\n" not in outcome.reason

    def test_the_default_runner_is_wired_in_without_being_called(self) -> None:
        assert ClaudeCliDistiller(ClassifierConfig()).argv() == [
            "claude",
            "-p",
            "--output-format",
            "json",
        ]

    def test_the_real_distiller_satisfies_the_protocol(self) -> None:
        assert isinstance(ClaudeCliDistiller(ClassifierConfig()), Distiller)

    def test_a_stub_satisfies_the_protocol(self) -> None:
        assert isinstance(StubDistiller(), Distiller)


class TestPrompt:
    """The prompt asks for paraphrase; the guard is what enforces it."""

    def test_the_body_and_the_classification_reach_the_prompt(self) -> None:
        prompt = build_distill_prompt(drop(), verdict())

        assert BODY in prompt
        assert "Classification: work" in prompt

    def test_the_project_is_named_when_there_is_one(self) -> None:
        assert "Project: PdfToExcel" in build_distill_prompt(drop(), verdict())

    def test_no_project_line_appears_when_there_is_none(self) -> None:
        assert "Project:" not in build_distill_prompt(drop(), verdict(project=None))

    def test_the_prompt_forbids_quoting_and_says_why(self) -> None:
        prompt = build_distill_prompt(drop(), verdict())

        assert "Do not quote." in prompt
        assert "automated check" in prompt

    def test_the_prompt_permits_an_empty_note(self) -> None:
        assert "Leave a list empty" in build_distill_prompt(drop(), verdict())

    def test_braces_in_the_body_do_not_break_prompt_construction(self) -> None:
        prompt = build_distill_prompt(drop(body='Config was {"vault": "work"}.'), verdict())

        assert '{"vault": "work"}' in prompt
