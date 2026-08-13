"""Classification to destinations.

The property under test is an asymmetry: raw content reaches the personal vault on every branch
and the work vault on none. It is asserted from both sides throughout — that the note is there,
and that no raw destination ever names the work vault — because the second half is the one that
matters and the one a plausible refactor would quietly break.

Nothing here touches the filesystem. `VaultConfig` and `Config` are built in memory rather than
loaded, so no vault has to exist for the routing table to be exercised, and a routing test can
never accidentally pass by reading something off disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from memvault.area import AreaChoice
from memvault.classify import Classification, Classifier, NeedsReview, Outcome
from memvault.config import ClassifierConfig, Config, IndexConfig, VaultConfig
from memvault.inbox import DeclaredMetadata, InboxRecord
from memvault.route import Destination, DestinationKind, RoutingResult, route


def vault_config(name: str, **overrides: Any) -> VaultConfig:
    """A vault, described without one existing. No `load_config`, so no filesystem."""
    settings: dict[str, Any] = {
        "name": name,
        "root": Path("/vaults") / name,
        "inbox": Path("/vaults") / name / "inbox",
    }
    settings.update(overrides)
    return VaultConfig(**settings)


def config_of(*vaults: VaultConfig, default: str | None = None) -> Config:
    return Config(
        source=Path("memvault.config.yaml"),
        default_vault=default or vaults[0].name,
        vaults={vault.name: vault for vault in vaults},
        classifier=ClassifierConfig(),
        index=IndexConfig(),
    )


def drop(**declared: Any) -> InboxRecord:
    """A record whose declared metadata is exactly the keys named here."""
    return InboxRecord(
        id="c0ffee",
        body="The transcript body.",
        declared=DeclaredMetadata(**declared, declared_keys=frozenset(declared)),
        filename="note.md",
        size_bytes=20,
    )


def verdict(classification: str = "personal", **overrides: Any) -> Classification:
    settings: dict[str, Any] = {
        "classification": classification,
        "confidence": 0.9,
        "title": "A conversation",
        "slug": "a-conversation",
        "summary": "What was said.",
    }
    settings.update(overrides)
    return Classification(**settings)


class StubClassifier:
    """A `Classifier` that returns what the test told it to. No subprocess, no network."""

    def __init__(self, outcome: Outcome) -> None:
        self._outcome = outcome

    def classify(self, record: InboxRecord) -> Outcome:
        return self._outcome


@pytest.fixture
def personal() -> VaultConfig:
    return vault_config("personal", work_route="work")


@pytest.fixture
def work() -> VaultConfig:
    return vault_config("work", note_template="learnings/{yyyy}/{date}-{slug}.md")


@pytest.fixture
def config(personal: VaultConfig, work: VaultConfig) -> Config:
    return config_of(personal, work, default="personal")


def routed(
    config: Config, vault: VaultConfig, outcome: Outcome, record: InboxRecord | None = None
) -> RoutingResult:
    """Route one record through a stubbed classifier, exercising the injection seam."""
    record = record or drop()
    classifier: Classifier = StubClassifier(outcome)
    return route(record, classifier.classify(record), config, vault)


class TestPersonal:
    """The simple branch: nothing crosses."""

    def test_a_personal_verdict_routes_to_the_personal_vault_only(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("personal"))

        assert result.destinations == (
            Destination(
                vault="personal",
                kind=DestinationKind.RAW,
                path_template="transcripts/{yyyy}/{mm}/{date}-{slug}.md",
                local_only=False,
            ),
        )

    def test_no_note_is_owed_for_personal_material(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("personal"))

        assert result.of_kind(DestinationKind.NOTE) == ()
        assert result.notes == ()

    def test_the_classification_is_carried_on_the_result(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("personal"))

        assert result.classification == "personal"
        assert result.record_id == "c0ffee"
        assert not result.held


class TestWorkAndMixed:
    """R9's asymmetry: raw stays personal, only a distilled note crosses."""

    @pytest.mark.parametrize("classification", ["work", "mixed"])
    def test_raw_content_is_filed_to_the_personal_vault(
        self, config: Config, personal: VaultConfig, classification: str
    ) -> None:
        result = routed(config, personal, verdict(classification))

        raw = result.of_kind(DestinationKind.RAW)

        assert len(raw) == 1
        assert raw[0].vault == "personal"

    @pytest.mark.parametrize("classification", ["work", "mixed"])
    def test_the_work_vault_receives_a_distilled_note_and_nothing_else(
        self, config: Config, personal: VaultConfig, classification: str
    ) -> None:
        result = routed(config, personal, verdict(classification))

        crossing = [d for d in result.destinations if d.vault == "work"]

        assert crossing == [
            Destination(
                vault="work",
                kind=DestinationKind.NOTE,
                path_template="learnings/{yyyy}/{date}-{slug}.md",
                local_only=False,
            )
        ]

    @pytest.mark.parametrize("classification", ["personal", "work", "mixed"])
    def test_no_branch_ever_sends_raw_content_to_the_work_vault(
        self, config: Config, personal: VaultConfig, classification: str
    ) -> None:
        result = routed(config, personal, verdict(classification))

        assert not [
            d for d in result.destinations if d.vault == "work" and d.kind is DestinationKind.RAW
        ]

    def test_mixed_routes_exactly_as_work_does(self, config: Config, personal: VaultConfig) -> None:
        as_work = routed(config, personal, verdict("work"))
        as_mixed = routed(config, personal, verdict("mixed"))

        assert as_work.destinations == as_mixed.destinations

    def test_an_unrecognized_label_falls_through_to_raw_only(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("confidential"))

        assert [d.kind for d in result.destinations] == [DestinationKind.RAW]


class TestReviewPath:
    """KTD6, seen from the routing side: a held item has nowhere to go."""

    def test_a_held_item_produces_an_empty_destination_list(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, NeedsReview("confidence 0.30 is below 0.70"))

        assert result.destinations == ()
        assert result.held

    def test_the_reason_is_recorded_for_the_review_marker(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, NeedsReview("confidence 0.30 is below 0.70"))

        assert result.review == "confidence 0.30 is below 0.70"
        assert result.classification is None

    def test_a_held_local_only_item_writes_nowhere_at_all(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, NeedsReview("unreadable"), drop(local_only=True))

        assert result.destinations == ()
        assert result.local_only


class TestLocalOnly:
    """KTD7: the gitignored prefix, and the boundary it also implies."""

    def test_raw_filing_lands_under_the_gitignored_prefix(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("personal"), drop(local_only=True))

        assert result.destinations[0].path_template.startswith("private/")
        assert result.destinations[0].local_only

    def test_the_prefix_is_configurable(self, work: VaultConfig) -> None:
        vault = vault_config("personal", local_only_prefix="not-pushed")
        config = config_of(vault, work, default="personal")

        result = routed(config, vault, verdict("personal"), drop(local_only=True))

        assert result.destinations[0].path_template.startswith("not-pushed/")

    def test_an_empty_prefix_leaves_the_template_alone(self, work: VaultConfig) -> None:
        vault = vault_config("personal", local_only_prefix="")
        config = config_of(vault, work, default="personal")

        result = routed(config, vault, verdict("personal"), drop(local_only=True))

        assert result.destinations[0].path_template == "transcripts/{yyyy}/{mm}/{date}-{slug}.md"

    def test_a_local_only_drop_sends_no_note_across_to_work(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("work"), drop(local_only=True))

        assert [d.vault for d in result.destinations] == ["personal"]

    def test_the_suppressed_work_note_is_explained(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("work"), drop(local_only=True))

        assert len(result.notes) == 1
        assert "local_only" in result.notes[0]

    def test_a_drop_that_declared_nothing_is_not_local_only(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("personal"))

        assert not result.local_only
        assert not result.destinations[0].path_template.startswith("private/")


class TestNoWorkRoute:
    """A single-vault install is a supported configuration, so work material must still file."""

    def test_work_material_files_raw_when_no_work_route_is_configured(self) -> None:
        solo = vault_config("personal")
        config = config_of(solo)

        result = routed(config, solo, verdict("work"))

        assert [d.vault for d in result.destinations] == ["personal"]
        assert result.of_kind(DestinationKind.RAW)[0].kind is DestinationKind.RAW

    def test_the_missing_route_is_explained_rather_than_silently_dropped(self) -> None:
        solo = vault_config("personal")
        config = config_of(solo)

        result = routed(config, solo, verdict("work"))

        assert len(result.notes) == 1
        assert "work_route" in result.notes[0]

    def test_the_item_is_not_held_for_review_over_a_configuration_gap(self) -> None:
        solo = vault_config("personal")
        config = config_of(solo)

        result = routed(config, solo, verdict("work"))

        assert not result.held

    def test_the_note_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        solo = vault_config("personal")
        config = config_of(solo)

        with caplog.at_level(logging.INFO, logger="memvault.route"):
            routed(config, solo, verdict("work"))

        assert "work_route" in caplog.text


class TestParticipants:
    """R11: participant metadata rides along so a deletion request can be honored."""

    def test_participants_from_the_classifier_reach_the_routing_result(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("work", participants=("Ada", "Sam")))

        assert result.participants == ("Ada", "Sam")

    def test_declared_participants_win_over_the_classifiers(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(
            config,
            personal,
            verdict("personal", participants=("Somebody", "Else")),
            drop(participants=("Deniz",)),
        )

        assert result.participants == ("Deniz",)

    def test_participants_survive_the_review_path_too(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, NeedsReview("held"), drop(participants=("Deniz",)))

        assert result.participants == ("Deniz",)

    def test_absent_participants_stay_absent_rather_than_becoming_a_placeholder(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("personal"))

        assert result.participants == ()


class TestTemplates:
    """Destinations name the configured template, never a hardcoded path."""

    def test_the_raw_destination_uses_the_vaults_transcript_template(
        self, work: VaultConfig
    ) -> None:
        vault = vault_config(
            "personal", work_route="work", transcript_template="raw/{date}-{slug}.md"
        )
        config = config_of(vault, work, default="personal")

        result = routed(config, vault, verdict("personal"))

        assert result.destinations[0].path_template == "raw/{date}-{slug}.md"

    def test_the_note_destination_uses_the_target_vaults_note_template(
        self, personal: VaultConfig
    ) -> None:
        target = vault_config("work", note_template="distilled/{date}.md")
        config = config_of(personal, target, default="personal")

        result = routed(config, personal, verdict("work"))

        assert result.of_kind(DestinationKind.NOTE)[0].path_template == "distilled/{date}.md"

    def test_no_destination_is_an_absolute_path(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed(config, personal, verdict("work"), drop())

        assert all(not d.path_template.startswith("/") for d in result.destinations)


# --- Topic notes inside the personal vault ----------------------------------------------
#
# The topic note and the cross-vault note are independent decisions: one is about *where in
# this vault* a summary belongs, the other about *whether anything crosses a vault boundary*.

CHOICE = AreaChoice("finance", "finance/adhoc/{date}-{slug}.md", "workspace")


def routed_with_area(
    config: Config,
    vault: VaultConfig,
    outcome: Outcome,
    *,
    area: AreaChoice | None,
    record: InboxRecord | None = None,
) -> RoutingResult:
    record = record or drop()
    classifier: Classifier = StubClassifier(outcome)
    return route(record, classifier.classify(record), config, vault, area=area)


class TestPersonalTopicNote:
    def test_a_personal_drop_with_an_area_gains_a_note_in_its_own_vault(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed_with_area(config, personal, verdict("personal"), area=CHOICE)

        notes = [d for d in result.destinations if d.kind is DestinationKind.NOTE]
        assert [d.path_template for d in notes] == ["finance/adhoc/{date}-{slug}.md"]
        assert notes[0].vault == "personal"

    def test_a_personal_drop_without_an_area_is_unchanged(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed_with_area(config, personal, verdict("personal"), area=None)

        assert [d.kind for d in result.destinations] == [DestinationKind.RAW]

    def test_a_work_drop_with_an_area_gets_both_notes(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed_with_area(config, personal, verdict("work"), area=CHOICE)

        assert {(d.vault, d.kind) for d in result.destinations} == {
            ("personal", DestinationKind.RAW),
            ("personal", DestinationKind.NOTE),
            ("work", DestinationKind.NOTE),
        }

    def test_a_work_drop_without_an_area_is_unchanged(
        self, config: Config, personal: VaultConfig
    ) -> None:
        result = routed_with_area(config, personal, verdict("work"), area=None)

        assert {(d.vault, d.kind) for d in result.destinations} == {
            ("personal", DestinationKind.RAW),
            ("work", DestinationKind.NOTE),
        }

    def test_local_only_keeps_the_topic_note_but_drops_the_crossing_one(
        self, config: Config, personal: VaultConfig
    ) -> None:
        """The topic note follows the raw into the gitignored prefix; nothing crosses vaults."""
        result = routed_with_area(
            config, personal, verdict("work"), area=CHOICE, record=drop(local_only=True)
        )

        assert {d.vault for d in result.destinations} == {"personal"}
        note = next(d for d in result.destinations if d.kind is DestinationKind.NOTE)
        assert note.path_template == "private/finance/adhoc/{date}-{slug}.md"
        assert note.local_only is True

    def test_a_held_record_gets_no_topic_note(self, config: Config, personal: VaultConfig) -> None:
        result = routed_with_area(config, personal, NeedsReview("too vague"), area=CHOICE)

        assert result.destinations == ()
