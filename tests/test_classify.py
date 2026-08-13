"""Classification of inbox drops.

One theme runs through all of it: every way this can go wrong ends in `NeedsReview`, and none
of them ends in a guess. Malformed output, a failed subprocess, and low confidence are all
tested as held items rather than as exceptions, because an exception mid-pass would take the
nine good drops down with the bad one.

No test here spawns a subprocess. The classifier's runner is injected, and the two tests that
exercise the default runner monkeypatch `subprocess.run` itself — the `claude` binary is never
invoked and nothing reaches the network.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

import pytest

from memvault.classify import (
    MAX_RELATIONS,
    Classification,
    Classifier,
    ClaudeCliClassifier,
    CommandResult,
    NeedsReview,
    Relation,
    build_prompt,
    run_command,
)
from memvault.config import AreaConfig, ClassifierConfig
from memvault.inbox import InboxRecord, normalize


def record(
    text: str = "Standup: Sam wants a tab per supplier.", name: str = "note.md"
) -> InboxRecord:
    """A record the way `inbox.normalize` would produce one."""
    return normalize(text, name)


def payload(**overrides: Any) -> dict[str, Any]:
    """A well-formed reply, before whatever this test wants to break about it."""
    base: dict[str, Any] = {
        "classification": "work",
        "confidence": 0.92,
        "title": "Standup with Sam",
        "slug": "standup-with-jordi",
        "summary": "Sam wants one Excel tab per supplier.",
        "participants": ["Ada", "Sam"],
        "tags": ["standup", "pdftoexcel"],
        "project": "PdfToExcel",
    }
    base.update(overrides)
    return base


def envelope(reply: Any) -> str:
    """What `claude -p --output-format json` actually prints: the reply inside a result object."""
    text = reply if isinstance(reply, str) else json.dumps(reply)
    return json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": text})


class FakeRunner:
    """A `CommandRunner` that answers from a script and remembers how it was called."""

    def __init__(self, result: CommandResult | None = None) -> None:
        self.result = result or CommandResult(exit_code=0, stdout=envelope(payload()))
        self.calls: list[tuple[list[str], str, int]] = []

    def __call__(self, argv: Any, *, prompt: str, timeout: int) -> CommandResult:
        self.calls.append((list(argv), prompt, timeout))
        return self.result

    @property
    def argv(self) -> list[str]:
        return self.calls[-1][0]

    @property
    def prompt(self) -> str:
        return self.calls[-1][1]


def classifier(runner: FakeRunner, **settings: Any) -> ClaudeCliClassifier:
    return ClaudeCliClassifier(ClassifierConfig(**settings), runner=runner)


def hold(stdout: str = "", **result_kwargs: Any) -> NeedsReview:
    """Classify a plain record against a canned command result, expecting it to be held."""
    result = CommandResult(
        exit_code=result_kwargs.pop("exit_code", 0), stdout=stdout, **result_kwargs
    )
    outcome = classifier(FakeRunner(result)).classify(record())

    assert isinstance(outcome, NeedsReview), f"expected a hold, got {outcome!r}"
    return outcome


class TestCommandConstruction:
    """KTD4's invocation, pinned down: the CLI, JSON output, and the prompt on stdin."""

    def test_the_configured_command_is_asked_for_json_output(self) -> None:
        runner = FakeRunner()

        classifier(runner).classify(record())

        assert runner.argv == ["claude", "-p", "--output-format", "json"]

    def test_an_absolute_command_path_is_used_verbatim(self) -> None:
        runner = FakeRunner()

        classifier(runner, command="/opt/homebrew/bin/claude").classify(record())

        assert runner.argv[0] == "/opt/homebrew/bin/claude"

    def test_a_configured_model_is_passed_through(self) -> None:
        runner = FakeRunner()

        classifier(runner, model="claude-haiku-4-5").classify(record())

        assert runner.argv[-2:] == ["--model", "claude-haiku-4-5"]

    def test_no_model_flag_is_sent_when_none_is_configured(self) -> None:
        runner = FakeRunner()

        classifier(runner).classify(record())

        assert "--model" not in runner.argv

    def test_the_prompt_carries_the_body_and_the_filename(self) -> None:
        runner = FakeRunner()
        drop = record("Coffee with Deniz about the erfpacht deadline.", "2026-07/coffee.md")

        classifier(runner).classify(drop)

        assert "Coffee with Deniz about the erfpacht deadline." in runner.prompt
        assert "2026-07/coffee.md" in runner.prompt

    def test_the_prompt_names_the_fields_the_drop_declared(self) -> None:
        runner = FakeRunner()
        drop = record("---\nclassification: work\nsource: conversate\n---\nBody.", "call.md")

        classifier(runner).classify(drop)

        assert "do not contradict" in runner.prompt
        assert "classification: work" in runner.prompt
        assert "source: conversate" in runner.prompt

    def test_a_drop_that_declared_nothing_gets_no_declared_block(self) -> None:
        runner = FakeRunner()

        classifier(runner).classify(record("Just a note.", "note.txt"))

        assert "do not contradict" not in runner.prompt

    def test_the_configured_timeout_reaches_the_runner(self) -> None:
        runner = FakeRunner()

        classifier(runner, timeout_seconds=45).classify(record())

        assert runner.calls[-1][2] == 45

    def test_braces_in_the_body_do_not_break_prompt_construction(self) -> None:
        drop = record('Config was {"vault": "personal"} and it broke.', "note.md")

        prompt = build_prompt(drop)

        assert '{"vault": "personal"}' in prompt


class TestSuccessfulVerdict:
    """A well-formed reply becomes a verdict the pipeline can act on."""

    def test_a_complete_reply_becomes_a_classification(self) -> None:
        outcome = classifier(FakeRunner()).classify(record())

        assert outcome == Classification(
            classification="work",
            confidence=0.92,
            title="Standup with Sam",
            slug="standup-with-jordi",
            summary="Sam wants one Excel tab per supplier.",
            participants=("Ada", "Sam"),
            tags=("standup", "pdftoexcel"),
            project="PdfToExcel",
        )

    def test_a_bare_reply_object_without_the_cli_envelope_is_accepted(self) -> None:
        runner = FakeRunner(CommandResult(exit_code=0, stdout=json.dumps(payload())))

        outcome = classifier(runner).classify(record())

        assert isinstance(outcome, Classification)
        assert outcome.classification == "work"

    def test_a_single_string_participant_becomes_a_one_item_tuple(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(participants="Deniz"))))

        outcome = classifier(runner).classify(record())

        assert isinstance(outcome, Classification)
        assert outcome.participants == ("Deniz",)

    def test_absent_optional_fields_come_back_empty_rather_than_invented(self) -> None:
        reply = {"classification": "personal", "confidence": 0.8, "title": "A note"}
        runner = FakeRunner(CommandResult(0, envelope(reply)))

        outcome = classifier(runner).classify(record())

        assert outcome == Classification(classification="personal", confidence=0.8, title="A note")

    def test_a_null_project_is_recorded_as_none(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(project=None))))

        outcome = classifier(runner).classify(record())

        assert isinstance(outcome, Classification)
        assert outcome.project is None

    def test_the_model_is_recorded_for_provenance(self) -> None:
        runner = FakeRunner()

        outcome = classifier(runner, model="claude-haiku-4-5").classify(record())

        assert isinstance(outcome, Classification)
        assert outcome.model == "claude-haiku-4-5"

    def test_a_classification_is_accepted_case_insensitively(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(classification="Work"))))

        outcome = classifier(runner).classify(record())

        assert isinstance(outcome, Classification)
        assert outcome.classification == "work"


class TestConfidenceGate:
    """KTD6: below the threshold is a hard stop, never a default to personal."""

    def test_low_confidence_is_held_with_both_numbers_named(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(confidence=0.41))))

        outcome = classifier(runner, confidence_threshold=0.7).classify(record())

        assert isinstance(outcome, NeedsReview)
        assert "0.41" in outcome.reason
        assert "0.70" in outcome.reason

    def test_a_held_item_records_the_confidence_it_was_held_for(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(confidence=0.41))))

        outcome = classifier(runner, confidence_threshold=0.7).classify(record())

        assert isinstance(outcome, NeedsReview)
        assert outcome.confidence == 0.41

    def test_no_classification_leaks_out_of_a_low_confidence_reply(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(confidence=0.1))))

        outcome = classifier(runner, confidence_threshold=0.7).classify(record())

        assert not isinstance(outcome, Classification)

    def test_confidence_exactly_at_the_threshold_is_accepted(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(confidence=0.7))))

        outcome = classifier(runner, confidence_threshold=0.7).classify(record())

        assert isinstance(outcome, Classification)

    def test_the_threshold_is_configurable(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(confidence=0.41))))

        outcome = classifier(runner, confidence_threshold=0.3).classify(record())

        assert isinstance(outcome, Classification)

    def test_the_gate_applies_even_when_the_classification_was_declared(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(confidence=0.2))))
        drop = record("---\nclassification: personal\n---\nBody.", "note.md")

        outcome = classifier(runner, confidence_threshold=0.7).classify(drop)

        assert isinstance(outcome, NeedsReview)

    def test_holding_is_logged_with_the_filename(self, caplog: pytest.LogCaptureFixture) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(confidence=0.1))))

        with caplog.at_level(logging.INFO, logger="memvault.classify"):
            classifier(runner).classify(record(name="standup.md"))

        assert "standup.md: held for review" in caplog.text


class TestMalformedOutput:
    """A reply that is not the JSON object it was asked for is held, never salvaged."""

    def test_prose_around_the_json_is_held_rather_than_salvaged(self) -> None:
        chatty = f"Sure! Here is the classification:\n\n{json.dumps(payload())}\n\nHope that helps."

        outcome = hold(stdout=envelope(chatty))

        assert "bare JSON object" in outcome.reason

    def test_a_markdown_fence_around_the_json_is_held(self) -> None:
        fenced = f"```json\n{json.dumps(payload())}\n```"

        outcome = hold(stdout=envelope(fenced))

        assert "bare JSON object" in outcome.reason

    def test_prose_instead_of_the_cli_envelope_is_held(self) -> None:
        outcome = hold(stdout="I could not classify that, sorry.")

        assert "was not JSON" in outcome.reason

    def test_invalid_json_is_held(self) -> None:
        outcome = hold(stdout=envelope('{"classification": "work", '))

        assert "bare JSON object" in outcome.reason

    def test_a_json_array_instead_of_an_object_is_held(self) -> None:
        outcome = hold(stdout=envelope([payload()]))

        assert "not an object" in outcome.reason

    def test_empty_output_is_held(self) -> None:
        outcome = hold(stdout="   \n")

        assert "no output" in outcome.reason

    def test_an_error_envelope_is_held(self) -> None:
        error = json.dumps({"type": "result", "is_error": True, "subtype": "error_max_turns"})

        outcome = hold(stdout=error)

        assert "error_max_turns" in outcome.reason

    def test_an_unrecognized_classification_is_held_by_name(self) -> None:
        outcome = hold(stdout=envelope(payload(classification="confidential")))

        assert "'confidential'" in outcome.reason

    def test_a_missing_required_field_is_named(self) -> None:
        reply = payload()
        del reply["title"]

        outcome = hold(stdout=envelope(reply))

        assert "missing required field(s): title" in outcome.reason

    def test_several_missing_fields_are_named_together(self) -> None:
        outcome = hold(stdout=envelope({"summary": "only a summary"}))

        assert "classification" in outcome.reason
        assert "confidence" in outcome.reason
        assert "title" in outcome.reason

    def test_a_non_numeric_confidence_is_held(self) -> None:
        outcome = hold(stdout=envelope(payload(confidence="very")))

        assert "non-numeric confidence" in outcome.reason

    def test_a_confidence_outside_zero_to_one_is_held(self) -> None:
        outcome = hold(stdout=envelope(payload(confidence=42)))

        assert "outside 0-1" in outcome.reason

    def test_a_held_reason_stays_on_one_line_so_it_fits_a_review_marker(self) -> None:
        outcome = hold(stdout=envelope("line one\nline two\n{not json"))

        assert "\n" not in outcome.reason


class TestSubprocessFailure:
    """A classifier that did not answer is a held item, not an exception."""

    def test_a_non_zero_exit_is_held_with_its_stderr(self) -> None:
        outcome = hold(exit_code=1, stderr="Credit balance is too low")

        assert "exited 1" in outcome.reason
        assert "Credit balance is too low" in outcome.reason

    def test_a_silent_failure_still_names_the_exit_code(self) -> None:
        outcome = hold(exit_code=127)

        assert "exited 127" in outcome.reason
        assert "no error output" in outcome.reason

    def test_a_timeout_is_held_and_names_the_limit(self) -> None:
        runner = FakeRunner(CommandResult(exit_code=-1, timed_out=True))

        outcome = classifier(runner, timeout_seconds=90).classify(record())

        assert isinstance(outcome, NeedsReview)
        assert "90s" in outcome.reason

    def test_multiline_stderr_is_flattened_into_the_reason(self) -> None:
        outcome = hold(exit_code=2, stderr="first line\nsecond line\n")

        assert "\n" not in outcome.reason
        assert "first line second line" in outcome.reason


class TestDeclaredOverrides:
    """R2: a declared field suppresses inference for itself, and for nothing else."""

    def test_a_declared_classification_overrides_the_classifier(self) -> None:
        runner = FakeRunner(CommandResult(0, envelope(payload(classification="work"))))
        drop = record("---\nclassification: personal\n---\nBody.", "note.md")

        outcome = classifier(runner).classify(drop)

        assert isinstance(outcome, Classification)
        assert outcome.classification == "personal"

    def test_the_rest_of_the_verdict_survives_a_declared_classification(self) -> None:
        runner = FakeRunner()
        drop = record("---\nclassification: personal\n---\nBody.", "note.md")

        outcome = classifier(runner).classify(drop)

        assert isinstance(outcome, Classification)
        assert outcome.title == "Standup with Sam"
        assert outcome.confidence == 0.92

    def test_declared_participants_override_the_classifiers(self) -> None:
        runner = FakeRunner()
        drop = record("---\nparticipants: [Deniz]\n---\nBody.", "note.md")

        outcome = classifier(runner).classify(drop)

        assert isinstance(outcome, Classification)
        assert outcome.participants == ("Deniz",)

    def test_declared_tags_override_the_classifiers(self) -> None:
        runner = FakeRunner()
        drop = record("---\ntags: [house]\n---\nBody.", "note.md")

        outcome = classifier(runner).classify(drop)

        assert isinstance(outcome, Classification)
        assert outcome.tags == ("house",)

    def test_a_declared_empty_participant_list_still_suppresses_the_classifiers(self) -> None:
        runner = FakeRunner()
        drop = record("---\nparticipants: []\n---\nBody.", "note.md")

        outcome = classifier(runner).classify(drop)

        assert isinstance(outcome, Classification)
        assert outcome.participants == ()

    def test_an_undeclared_field_still_comes_from_the_classifier(self) -> None:
        runner = FakeRunner()
        drop = record("---\ntags: [house]\n---\nBody.", "note.md")

        outcome = classifier(runner).classify(drop)

        assert isinstance(outcome, Classification)
        assert outcome.participants == ("Ada", "Sam")

    def test_a_drop_whose_own_metadata_is_unparseable_is_held_without_an_llm_call(self) -> None:
        runner = FakeRunner()
        drop = record("---\ntags: [unclosed\n---\nBody.", "broken.md")

        outcome = classifier(runner).classify(drop)

        assert isinstance(outcome, NeedsReview)
        assert "metadata could not be read" in outcome.reason
        assert runner.calls == []


class TestDefaultRunner:
    """The one place a subprocess would be spawned, with `subprocess.run` monkeypatched out."""

    @staticmethod
    def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        seen: dict[str, Any] = {}

        def fake_run(argv: Any, **kwargs: Any) -> Any:
            seen["argv"] = argv
            seen["kwargs"] = kwargs
            return subprocess.CompletedProcess(argv, 0, stdout="out", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        return seen

    def test_the_child_inherits_this_processes_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._capture(monkeypatch)

        run_command(["claude", "-p"], prompt="hello", timeout=10)

        assert "env" not in seen["kwargs"], "passing env= would drop CLAUDE_CONFIG_DIR (KTD4)"

    def test_no_shell_is_involved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._capture(monkeypatch)

        run_command(["claude", "-p"], prompt="hello", timeout=10)

        assert seen["kwargs"].get("shell") is not True
        assert seen["argv"] == ["claude", "-p"]

    def test_the_prompt_goes_on_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._capture(monkeypatch)

        run_command(["claude", "-p"], prompt="the transcript", timeout=10)

        assert seen["kwargs"]["input"] == "the transcript"

    def test_a_timeout_comes_back_as_a_result_rather_than_an_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def timeout(*_: Any, **__: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=10)

        monkeypatch.setattr(subprocess, "run", timeout)

        result = run_command(["claude"], prompt="hello", timeout=10)

        assert result.timed_out
        assert result.exit_code != 0

    def test_a_missing_binary_comes_back_as_a_result_rather_than_an_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(*_: Any, **__: Any) -> Any:
            raise FileNotFoundError(2, "No such file or directory")

        monkeypatch.setattr(subprocess, "run", missing)

        result = run_command(["/nowhere/claude"], prompt="hello", timeout=10)

        assert result.exit_code != 0
        assert "/nowhere/claude" in result.stderr

    def test_the_default_runner_is_wired_in_without_being_called(self) -> None:
        instance = ClaudeCliClassifier(ClassifierConfig())

        assert instance.argv() == ["claude", "-p", "--output-format", "json"]


class TestProtocolConformance:
    """The seam that keeps the real binary out of every other test."""

    def test_the_real_classifier_satisfies_the_protocol(self) -> None:
        assert isinstance(ClaudeCliClassifier(ClassifierConfig()), Classifier)

    def test_a_stub_satisfies_the_protocol(self) -> None:
        class Stub:
            def classify(self, record: InboxRecord) -> Classification:
                return Classification(classification="personal", confidence=1.0, title="t")

        assert isinstance(Stub(), Classifier)


# --- Area selection ---------------------------------------------------------------------
#
# The area is decided on the call that already runs. A second round trip per drop to answer
# "which folder" would double the pass's cost to say something the same context already knows.

FINANCE_AREA = AreaConfig(
    name="finance", note_template="finance/adhoc/{date}-{slug}.md", when="markets, macro"
)
PROJECT_AREA = AreaConfig(name="Sapik", note_template="repos/Sapik/adhoc/{date}-{slug}.md")


def verdict_from(**overrides: Any) -> Classification:
    """Classify a plain record against a canned reply, expecting a verdict rather than a hold."""
    runner = FakeRunner(CommandResult(exit_code=0, stdout=envelope(payload(**overrides))))
    outcome = classifier(runner).classify(record())

    assert isinstance(outcome, Classification), f"expected a verdict, got {outcome!r}"
    return outcome


def test_the_prompt_lists_the_areas_and_their_hints() -> None:
    prompt = build_prompt(record(), areas=(FINANCE_AREA, PROJECT_AREA))

    assert "finance" in prompt
    assert "markets, macro" in prompt
    assert "Sapik" in prompt
    assert "null" in prompt


def test_a_prompt_with_no_areas_asks_for_no_area() -> None:
    assert "area" not in build_prompt(record(), areas=()).lower()


def test_the_classifier_renders_the_areas_it_was_built_with() -> None:
    runner = FakeRunner()
    ClaudeCliClassifier(ClassifierConfig(), runner=runner, areas=(FINANCE_AREA,)).classify(record())

    assert "markets, macro" in runner.prompt


def test_a_returned_area_is_parsed() -> None:
    assert verdict_from(area="finance").area == "finance"


def test_a_null_area_is_none() -> None:
    assert verdict_from(area=None).area is None


def test_a_missing_area_is_none() -> None:
    verdict = verdict_from()

    assert verdict.area is None


def test_a_blank_area_is_none() -> None:
    assert verdict_from(area="   ").area is None


def test_the_area_list_does_not_swallow_the_blank_line_before_the_metadata() -> None:
    """Without a trailing newline the areas run straight into `Filename:`."""
    prompt = build_prompt(record(), areas=(FINANCE_AREA,))

    assert "\n\nFilename:" in prompt


def test_a_prompt_without_areas_keeps_the_same_blank_line() -> None:
    assert "\n\nFilename:" in build_prompt(record(), areas=())


# --- Importance and relations -----------------------------------------------------------
#
# Both ride the call that already runs, and both are optional: the pipeline would rather file an
# item with no importance and no relations than hold it over a field that only affects ranking.


def test_the_prompt_asks_for_an_importance_score() -> None:
    assert "importance" in build_prompt(record())


def test_the_prompt_asks_for_relations() -> None:
    assert "relations" in build_prompt(record())


def test_a_returned_importance_is_parsed() -> None:
    assert verdict_from(importance=7).importance == 7


def test_a_missing_importance_is_none() -> None:
    """Absent is a real answer: recall reads it as neutral, not as unimportant."""
    assert verdict_from().importance is None


def test_an_out_of_range_importance_is_dropped_rather_than_held() -> None:
    """Importance only tilts ranking, so a bad one costs the field, never the filing."""
    outcome = verdict_from(importance=42)

    assert outcome.importance is None
    assert outcome.title == "Standup with Sam"


def test_a_non_numeric_importance_is_dropped_rather_than_held() -> None:
    assert verdict_from(importance="very high").importance is None


def test_a_numeric_string_importance_is_read() -> None:
    """Models answer a numeric field with a string often enough to accommodate it."""
    assert verdict_from(importance="8").importance == 8


def test_a_boolean_importance_is_not_read_as_one() -> None:
    """`True` is an int in Python and would silently become importance 1."""
    assert verdict_from(importance=True).importance is None


def test_relations_are_parsed_into_predicate_and_target() -> None:
    outcome = verdict_from(
        relations=[{"predicate": "part_of", "target": "repos/MemVault"}],
    )

    assert outcome.relations == (Relation("part_of", "repos/MemVault"),)


def test_no_relations_is_an_empty_tuple() -> None:
    assert verdict_from().relations == ()


def test_a_predicate_written_as_a_phrase_is_normalized() -> None:
    outcome = verdict_from(relations=[{"predicate": "Part Of", "target": "repos/MemVault"}])

    assert outcome.relations == (Relation("part_of", "repos/MemVault"),)


def test_a_malformed_predicate_drops_that_relation_only() -> None:
    outcome = verdict_from(
        relations=[
            {"predicate": "!!", "target": "repos/MemVault"},
            {"predicate": "about", "target": "ranking"},
        ]
    )

    assert outcome.relations == (Relation("about", "ranking"),)


def test_a_target_the_model_already_wrapped_in_brackets_is_unwrapped() -> None:
    """Rendering would otherwise produce `[[[[x]]]]`, which no wikilink parser reads."""
    outcome = verdict_from(relations=[{"predicate": "about", "target": "[[ranking]]"}])

    assert outcome.relations == (Relation("about", "ranking"),)


def test_a_relation_with_no_target_is_dropped() -> None:
    assert verdict_from(relations=[{"predicate": "about", "target": "  "}]).relations == ()


def test_a_target_that_climbs_out_of_the_vault_is_dropped() -> None:
    """A relation is a vault-relative pointer; `../` is not one, whatever the model meant."""
    outcome = verdict_from(relations=[{"predicate": "about", "target": "../secrets.md"}])

    assert outcome.relations == ()


def test_duplicate_relations_are_emitted_once() -> None:
    outcome = verdict_from(
        relations=[
            {"predicate": "about", "target": "ranking"},
            {"predicate": "about", "target": "ranking"},
        ]
    )

    assert outcome.relations == (Relation("about", "ranking"),)


def test_a_flood_of_relations_is_capped() -> None:
    """One runaway reply must not bury a note's own body under wikilinks."""
    outcome = verdict_from(
        relations=[{"predicate": "about", "target": f"note-{n}"} for n in range(50)]
    )

    assert len(outcome.relations) == MAX_RELATIONS


def test_relations_that_are_not_a_list_are_ignored() -> None:
    assert verdict_from(relations="part_of repos/MemVault").relations == ()
