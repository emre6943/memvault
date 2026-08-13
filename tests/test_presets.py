"""Classifier presets: one table, four call sites, four envelopes.

Two things are being defended here. The first is the argv of each preset, which is this repo's
whole contract with a program it does not own — an argv that drifts does not crash, it holds an
entire ingest pass for review, and a held pass looks like a bad week rather than a bug. So the
documented command line of every preset is written out literally.

The second is that all four model-facing call sites — classifier, distiller, segmenter, curator
— go through the table rather than each carrying its own copy. That is the failure this unit
exists to prevent: before it, four copies of the same command line meant three chances for one
of them to be forgotten, and the forgotten one would keep classifying with the wrong CLI's flags
until somebody read the logs.

No test here spawns a subprocess. Every CLI is a `FakeRunner` returning what that CLI's
documented non-interactive output looks like.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import pytest

from memvault.classify import (
    Classification,
    ClaudeCliClassifier,
    CommandResult,
    NeedsReview,
    _reply_text,
)
from memvault.config import CLASSIFIER_PRESETS, ClassifierConfig, SegmentConfig
from memvault.distill import ClaudeCliDistiller, DistillationFailed
from memvault.inbox import InboxRecord, normalize
from memvault.presets import (
    PRESETS,
    Envelope,
    build_argv,
    envelope_for,
    preset_for,
    unwrap,
)
from memvault.reflect import ClaudeCliCurator, CurationFailed, ReflectionMaterial
from memvault.segment import ClaudeCliSegmenter

# --------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------


def record(text: str = "Standup: Sam wants a tab per supplier.") -> InboxRecord:
    return normalize(text, "note.md")


def payload(**overrides: Any) -> dict[str, Any]:
    """A well-formed classifier reply — the bare JSON object every preset must deliver."""
    base: dict[str, Any] = {
        "classification": "work",
        "confidence": 0.92,
        "title": "Standup with Sam",
        "summary": "Sam wants one Excel tab per supplier.",
    }
    base.update(overrides)
    return base


class FakeRunner:
    """A `CommandRunner` that answers from a script and remembers how it was called."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.calls: list[list[str]] = []

    def __call__(self, argv: Any, *, prompt: str, timeout: int) -> CommandResult:
        self.calls.append(list(argv))
        return CommandResult(exit_code=0, stdout=self.stdout)

    @property
    def argv(self) -> list[str]:
        return self.calls[-1]


def claude_stdout(reply: Any) -> str:
    """What `claude -p --output-format json` prints: the reply inside a result object."""
    text = reply if isinstance(reply, str) else json.dumps(reply)
    return json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": text})


def codex_stdout(*events: dict[str, Any]) -> str:
    """What `codex exec --json` prints: one JSON event per line."""
    return "\n".join(json.dumps(event) for event in events) + "\n"


def codex_message(text: str) -> dict[str, Any]:
    """The current shape of codex's assistant message event."""
    return {
        "type": "item.completed",
        "item": {"id": "item_1", "type": "agent_message", "text": text},
    }


def codex_reply(reply: Any) -> str:
    """A plausible codex stream: some narration, then the answer."""
    return codex_stdout(
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "item.started", "item": {"id": "item_0", "type": "reasoning"}},
        codex_message(json.dumps(reply)),
        {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 3}},
    )


def opencode_stdout(reply: Any) -> str:
    """What a CLI that streams prose prints: the object inside a fenced block."""
    return (
        "I read the drop and classified it.\n\n"
        "```json\n" + json.dumps(reply) + "\n```\n\nLet me know if that looks wrong.\n"
    )


def classify_with(preset: str, stdout: str) -> tuple[Any, FakeRunner]:
    """Run the real classifier against one CLI's output under one preset."""
    runner = FakeRunner(stdout)
    config = ClassifierConfig(preset=preset)
    return ClaudeCliClassifier(config, runner=runner).classify(record()), runner


def distill_verdict() -> Classification:
    return Classification(classification="work", confidence=0.9, title="Standup with Sam")


def curation_material() -> ReflectionMaterial:
    return ReflectionMaterial(vault="personal", since=date(2026, 8, 1), until=date(2026, 8, 7))


# --------------------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------------------


class TestTheTable:
    """A preset accepted by config with no row here would fail at the first drop, not at load."""

    def test_the_table_and_the_config_agree_on_the_preset_names(self) -> None:
        assert tuple(PRESETS) == CLASSIFIER_PRESETS

    def test_every_preset_knows_its_own_name(self) -> None:
        # The name is what a log line and the model-flag warning quote, so a mismatch would
        # send a reader looking at the wrong row.
        assert all(key == preset.name for key, preset in PRESETS.items())

    def test_an_unknown_preset_raises_rather_than_falling_back(self) -> None:
        # Config rejects it at load; reaching here means code constructed the config directly.
        # Falling back to claude would run somebody's codex binary with claude's flags and
        # report the resulting failure as a bad reply.
        with pytest.raises(ValueError, match="unknown classifier preset"):
            preset_for("gemini")

    def test_the_default_preset_is_claude(self) -> None:
        assert envelope_for(ClassifierConfig()) is Envelope.CLAUDE_JSON


# --------------------------------------------------------------------------------------
# The documented command lines
# --------------------------------------------------------------------------------------


class TestArgv:
    """Each preset's documented command line, written out rather than derived."""

    def test_claude_is_byte_identical_to_the_command_line_that_preceded_presets(self) -> None:
        assert build_argv(ClassifierConfig()) == ["claude", "-p", "--output-format", "json"]

    def test_claude_appends_the_model(self) -> None:
        config = ClassifierConfig(model="claude-sonnet-4-5")

        assert build_argv(config) == [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            "claude-sonnet-4-5",
        ]

    def test_codex_reads_the_prompt_from_stdin_and_streams_json(self) -> None:
        config = ClassifierConfig(command="codex", preset="codex")

        assert build_argv(config) == ["codex", "exec", "--json", "--skip-git-repo-check", "-"]

    def test_codex_keeps_the_stdin_sentinel_last_when_a_model_is_named(self) -> None:
        # `-` is positional. A model flag inserted after it would be read as the prompt.
        config = ClassifierConfig(command="codex", preset="codex", model="gpt-5.1-codex")

        argv = build_argv(config)

        assert argv[-1] == "-"
        assert argv[-3:-1] == ["--model", "gpt-5.1-codex"]

    def test_opencode_runs_non_interactively(self) -> None:
        config = ClassifierConfig(command="opencode", preset="opencode")

        assert build_argv(config) == ["opencode", "run"]

    def test_opencode_appends_the_model(self) -> None:
        config = ClassifierConfig(
            command="opencode", preset="opencode", model="anthropic/claude-sonnet-4-5"
        )

        assert build_argv(config) == [
            "opencode",
            "run",
            "--model",
            "anthropic/claude-sonnet-4-5",
        ]

    def test_custom_runs_the_command_exactly_as_written(self) -> None:
        # The escape hatch: no flags can be guessed for a wrapper script, so none are added.
        config = ClassifierConfig(command="/opt/bin/my-classifier", preset="custom")

        assert build_argv(config) == ["/opt/bin/my-classifier"]

    def test_custom_leaves_a_configured_model_to_the_command_and_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Dropping a configured model silently would be a trap: the pass would keep running,
        # on a model nobody chose.
        config = ClassifierConfig(command="wrapper.sh", preset="custom", model="some-model")

        with caplog.at_level(logging.INFO, logger="memvault.presets"):
            argv = build_argv(config)

        assert argv == ["wrapper.sh"]
        assert "some-model" in caplog.text


# --------------------------------------------------------------------------------------
# The envelopes, round-tripped through the real classifier
# --------------------------------------------------------------------------------------


class TestEnvelopeRoundTrips:
    """Each preset, fed what its CLI documents printing, yields the same classification."""

    def test_claude_unwraps_its_result_object(self) -> None:
        outcome, _ = classify_with("claude", claude_stdout(payload()))

        assert isinstance(outcome, Classification)
        assert outcome.title == "Standup with Sam"

    def test_codex_takes_the_reply_out_of_its_event_stream(self) -> None:
        # Under the claude envelope this same stdout is not JSON at all, so a preset that was
        # not consulted would hold the drop here.
        outcome, _ = classify_with("codex", codex_reply(payload()))

        assert isinstance(outcome, Classification)
        assert outcome.title == "Standup with Sam"

    def test_opencode_takes_the_reply_out_of_the_prose_around_it(self) -> None:
        outcome, _ = classify_with("opencode", opencode_stdout(payload()))

        assert isinstance(outcome, Classification)
        assert outcome.title == "Standup with Sam"

    def test_custom_reads_the_output_as_the_reply(self) -> None:
        outcome, _ = classify_with("custom", json.dumps(payload()) + "\n")

        assert isinstance(outcome, Classification)
        assert outcome.title == "Standup with Sam"

    def test_every_preset_still_accepts_a_bare_reply(self) -> None:
        # The fallback that lets a stub, or a naive CLI that prints only what the model said,
        # work under any preset — and is why adding a preset cannot break the shipped ones.
        for name in PRESETS:
            outcome, _ = classify_with(name, json.dumps(payload()))

            assert isinstance(outcome, Classification), f"{name} rejected a bare reply"


# --------------------------------------------------------------------------------------
# Envelope details worth their own tests
# --------------------------------------------------------------------------------------


class TestClaudeEnvelope:
    def test_an_error_envelope_is_reported_rather_than_parsed(self) -> None:
        stdout = json.dumps({"is_error": True, "subtype": "usage_limit"})

        reply, problem = unwrap(stdout, Envelope.CLAUDE_JSON)

        assert reply is None
        assert problem is not None and "usage_limit" in problem

    def test_empty_output_is_named_as_such(self) -> None:
        assert unwrap("   \n", Envelope.CLAUDE_JSON)[1] == "the classifier produced no output"

    def test_the_subject_is_the_callers_own_name(self) -> None:
        # One wording serves four call sites; the noun is the only difference between them.
        assert unwrap("", Envelope.CLAUDE_JSON, subject="the curator")[1] == (
            "the curator produced no output"
        )


class TestCodexStream:
    def test_the_last_assistant_message_wins(self) -> None:
        stdout = codex_stdout(codex_message("first"), codex_message("second"))

        assert unwrap(stdout, Envelope.CODEX_JSONL)[0] == "second"

    def test_the_older_flat_event_shape_is_read_too(self) -> None:
        # codex has shipped both shapes; a preset that recognized only the current one would
        # hold every drop on an older install.
        stdout = codex_stdout({"id": "0", "msg": {"type": "agent_message", "message": "hello"}})

        assert unwrap(stdout, Envelope.CODEX_JSONL)[0] == "hello"

    def test_an_error_event_is_reported(self) -> None:
        stdout = codex_stdout({"type": "error", "message": "stream disconnected"})

        reply, problem = unwrap(stdout, Envelope.CODEX_JSONL)

        assert reply is None
        assert problem is not None and "stream disconnected" in problem

    def test_a_run_that_errored_and_then_answered_has_answered(self) -> None:
        stdout = codex_stdout(
            {"type": "error", "message": "retrying"}, codex_message('{"ok": true}')
        )

        assert unwrap(stdout, Envelope.CODEX_JSONL) == ('{"ok": true}', None)

    def test_a_stream_with_no_assistant_message_is_a_problem_not_a_guess(self) -> None:
        stdout = codex_stdout({"type": "turn.completed", "usage": {}})

        reply, problem = unwrap(stdout, Envelope.CODEX_JSONL)

        assert reply is None
        assert problem is not None and "no assistant message" in problem


class TestTextOutput:
    def test_a_fenced_block_wins_over_braces_in_the_prose(self) -> None:
        stdout = 'Something like {"guess": 1} came to mind.\n\n```json\n{"real": 1}\n```\n'

        assert unwrap(stdout, Envelope.TEXT)[0] == '{"real": 1}'

    def test_an_unfenced_object_is_found_between_the_prose(self) -> None:
        stdout = 'Here is the answer:\n{"classification": "work"}\nHope that helps.\n'

        assert unwrap(stdout, Envelope.TEXT)[0] == '{"classification": "work"}'

    def test_prose_with_no_object_is_a_problem(self) -> None:
        reply, problem = unwrap("I could not read that file, sorry.", Envelope.TEXT)

        assert reply is None
        assert problem is not None and "no JSON object" in problem


# --------------------------------------------------------------------------------------
# The four call sites
# --------------------------------------------------------------------------------------


class TestEveryCallSiteDelegates:
    """The invariant this unit exists for: one table, four callers, no private copies."""

    def codex(self) -> ClassifierConfig:
        return ClassifierConfig(command="codex", preset="codex", model="gpt-5.1-codex")

    def test_the_classifier_builds_its_command_line_from_the_table(self) -> None:
        config = self.codex()

        assert ClaudeCliClassifier(config).argv() == build_argv(config)

    def test_the_distiller_builds_its_command_line_from_the_table(self) -> None:
        config = self.codex()

        assert ClaudeCliDistiller(config).argv() == build_argv(config)

    def test_the_segmenter_builds_its_command_line_from_the_table(self) -> None:
        config = self.codex()

        segmenter = ClaudeCliSegmenter(config, SegmentConfig(enabled=True, min_chars=0))

        assert segmenter.argv() == build_argv(config)

    def test_the_curator_builds_its_command_line_from_the_table(self) -> None:
        config = self.codex()

        assert ClaudeCliCurator(config).argv() == build_argv(config)

    def test_the_distiller_reads_the_presets_envelope(self) -> None:
        # A `codex` stream is not valid JSON as a whole, so a site still reading the claude
        # envelope would fail with "was not JSON" instead of naming what codex reported.
        runner = FakeRunner(codex_stdout({"type": "error", "message": "boom"}))

        outcome = ClaudeCliDistiller(self.codex(), runner=runner).distill(
            record(), distill_verdict()
        )

        assert isinstance(outcome, DistillationFailed)
        assert "boom" in outcome.reason

    def test_the_segmenter_reads_the_presets_envelope(self) -> None:
        runner = FakeRunner(codex_stdout({"type": "error", "message": "boom"}))
        segmenter = ClaudeCliSegmenter(
            self.codex(), SegmentConfig(enabled=True, min_chars=0), runner=runner
        )

        outcome = segmenter.segment(record(), 12)

        assert isinstance(outcome, NeedsReview)
        assert "boom" in outcome.reason

    def test_the_curator_reads_the_presets_envelope(self) -> None:
        runner = FakeRunner(codex_stdout({"type": "error", "message": "boom"}))

        outcome = ClaudeCliCurator(self.codex(), runner=runner).curate(curation_material())

        assert isinstance(outcome, CurationFailed)
        assert "boom" in outcome.reason

    def test_the_classifier_reads_the_presets_envelope(self) -> None:
        outcome, _ = classify_with("codex", codex_stdout({"type": "error", "message": "boom"}))

        assert isinstance(outcome, NeedsReview)
        assert "boom" in outcome.reason

    def test_the_command_the_runner_receives_is_the_one_the_table_describes(self) -> None:
        # Belt and braces on `argv()`: what actually reaches the runner is what matters.
        _, runner = classify_with("codex", codex_reply(payload()))

        assert runner.argv == ["claude", "exec", "--json", "--skip-git-repo-check", "-"]


class TestReplyTextDefault:
    """`_reply_text` keeps its old signature so a caller that names no envelope is unchanged."""

    def test_the_default_envelope_is_still_claudes(self) -> None:
        assert _reply_text(claude_stdout('{"ok": true}')) == ('{"ok": true}', None)
