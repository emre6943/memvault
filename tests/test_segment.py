"""Segmentation.

The theme of these tests: a segmenter may only ever produce spans of the original text, and any
doubt about that holds the drop. Content is never allowed to go missing quietly — most of the
cases below are a model behaving plausibly while losing a passage, and every one of them must
come back as `NeedsReview` rather than as a smaller set of segments.
"""

from __future__ import annotations

import json
from typing import Any

from memvault.classify import CommandResult, NeedsReview
from memvault.config import ClassifierConfig, SegmentConfig
from memvault.inbox import InboxRecord, content_id, normalize
from memvault.segment import (
    ClaudeCliSegmenter,
    FakeSegmenter,
    Segment,
    build_prompt,
    number_lines,
    segment_record,
    whole_drop,
)

BODY = "\n".join(
    [
        "We repainted the shed at the allotment.",
        "The soil was still wet from Tuesday.",
        "Then we talked about whether to move in together.",
        "She wants to wait until spring.",
        "At work the deploy pipeline broke again.",
        "Sam is rewriting the importer.",
    ]
)


def record(text: str = BODY, name: str = "conversation.md") -> InboxRecord:
    """A record the way `inbox.normalize` would produce one."""
    return normalize(text, name)


def envelope(reply: Any) -> str:
    """What `claude -p --output-format json` actually prints."""
    text = reply if isinstance(reply, str) else json.dumps(reply)
    return json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": text})


def reply(*spans: tuple[int, int, str]) -> str:
    return envelope(
        {"segments": [{"start_line": s, "end_line": e, "topic": t} for s, e, t in spans]}
    )


class FakeRunner:
    """A `CommandRunner` answering from a script, remembering how it was called."""

    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.calls: list[tuple[list[str], str, int]] = []

    def __call__(self, argv: Any, *, prompt: str, timeout: int) -> CommandResult:
        self.calls.append((list(argv), prompt, timeout))
        return self.result


def cli(result: CommandResult, config: SegmentConfig | None = None) -> ClaudeCliSegmenter:
    return ClaudeCliSegmenter(
        ClassifierConfig(),
        config or SegmentConfig(enabled=True, min_chars=0),
        runner=FakeRunner(result),
    )


# --- Splitting ---------------------------------------------------------------------------


def test_a_multi_topic_body_returns_segments_that_reconstruct_it() -> None:
    drop = record()
    outcome = cli(
        CommandResult(0, reply((1, 2, "allotment"), (3, 4, "us"), (5, 6, "work")))
    ).segment(drop, 12)

    assert not isinstance(outcome, NeedsReview)
    assert len(outcome) == 3
    assert "\n".join(s.body for s in outcome) == drop.body
    assert [s.topic for s in outcome] == ["allotment", "us", "work"]
    assert [s.index for s in outcome] == [1, 2, 3]
    assert {s.count for s in outcome} == {3}


def test_a_single_topic_body_returns_one_segment() -> None:
    drop = record()
    outcome = cli(CommandResult(0, reply((1, 6, "allotment")))).segment(drop, 12)

    assert not isinstance(outcome, NeedsReview)
    assert len(outcome) == 1
    assert outcome[0].body == drop.body


def test_every_segment_carries_the_parent_id() -> None:
    drop = record()
    outcome = cli(CommandResult(0, reply((1, 3, "a"), (4, 6, "b")))).segment(drop, 12)

    assert not isinstance(outcome, NeedsReview)
    assert {s.parent_id for s in outcome} == {drop.id}


# --- Identity ----------------------------------------------------------------------------


def test_segment_ids_are_the_hash_of_the_segment_body() -> None:
    drop = record()
    outcome = cli(CommandResult(0, reply((1, 3, "a"), (4, 6, "b")))).segment(drop, 12)

    assert not isinstance(outcome, NeedsReview)
    for segment in outcome:
        assert segment.id == content_id(segment.body)


def test_segment_ids_are_stable_across_identical_runs() -> None:
    spans = ((1, 3, "a"), (4, 6, "b"))
    first = cli(CommandResult(0, reply(*spans))).segment(record(), 12)
    second = cli(CommandResult(0, reply(*spans))).segment(record(), 12)

    assert not isinstance(first, NeedsReview)
    assert not isinstance(second, NeedsReview)
    assert [s.id for s in first] == [s.id for s in second]


def test_a_one_character_edit_changes_only_the_affected_segment_id() -> None:
    spans = ((1, 3, "a"), (4, 6, "b"))
    original = cli(CommandResult(0, reply(*spans))).segment(record(), 12)
    edited = cli(CommandResult(0, reply(*spans))).segment(record(BODY.replace("wet", "damp")), 12)

    assert not isinstance(original, NeedsReview)
    assert not isinstance(edited, NeedsReview)
    assert original[0].id != edited[0].id
    assert original[1].id == edited[1].id


# --- Refusing to lose content ------------------------------------------------------------


def test_segments_that_do_not_reach_the_last_line_are_refused() -> None:
    outcome = cli(CommandResult(0, reply((1, 2, "a"), (3, 4, "b")))).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)
    assert "every line must be covered" in outcome.reason


def test_a_gap_between_segments_is_refused() -> None:
    outcome = cli(CommandResult(0, reply((1, 2, "a"), (4, 6, "b")))).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)
    assert "contiguous" in outcome.reason


def test_overlapping_segments_are_refused() -> None:
    outcome = cli(CommandResult(0, reply((1, 4, "a"), (3, 6, "b")))).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)
    assert "contiguous" in outcome.reason


def test_a_segment_running_past_the_end_is_refused() -> None:
    outcome = cli(CommandResult(0, reply((1, 99, "a")))).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)
    assert "past the last line" in outcome.reason


def test_an_inverted_segment_is_refused() -> None:
    outcome = cli(CommandResult(0, reply((1, 2, "a"), (3, 2, "b")))).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)


def test_more_segments_than_the_limit_is_refused_rather_than_truncated() -> None:
    spans = tuple((i, i, f"t{i}") for i in range(1, 7))
    outcome = cli(CommandResult(0, reply(*spans))).segment(record(), 3)

    assert isinstance(outcome, NeedsReview)
    assert "above the limit" in outcome.reason


# --- Malformed replies -------------------------------------------------------------------


def test_non_json_reply_is_refused() -> None:
    outcome = cli(CommandResult(0, envelope("not json at all"))).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)
    assert "not JSON" in outcome.reason


def test_reply_without_a_segments_list_is_refused() -> None:
    outcome = cli(CommandResult(0, envelope({"parts": []}))).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)
    assert "segments" in outcome.reason


def test_empty_segments_list_is_refused() -> None:
    outcome = cli(CommandResult(0, envelope({"segments": []}))).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)


def test_a_segment_missing_line_numbers_is_refused() -> None:
    outcome = cli(CommandResult(0, envelope({"segments": [{"topic": "a"}]}))).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)
    assert "start_line" in outcome.reason


def test_timeout_is_refused_not_raised() -> None:
    outcome = cli(CommandResult(exit_code=0, timed_out=True)).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)
    assert "did not answer" in outcome.reason


def test_nonzero_exit_is_refused_not_raised() -> None:
    outcome = cli(CommandResult(exit_code=2, stderr="boom")).segment(record(), 12)

    assert isinstance(outcome, NeedsReview)
    assert "exited 2" in outcome.reason


def test_an_unparseable_drop_is_refused_without_calling_the_model() -> None:
    runner = FakeRunner(CommandResult(0, reply((1, 6, "a"))))
    segmenter = ClaudeCliSegmenter(
        ClassifierConfig(), SegmentConfig(enabled=True, min_chars=0), runner=runner
    )

    outcome = segmenter.segment(normalize("---\ndate: soon\n---\n\nbody", "bad.md"), 12)

    assert isinstance(outcome, NeedsReview)
    assert runner.calls == []


# --- Turkish -----------------------------------------------------------------------------


def test_turkish_text_segments_without_mangling() -> None:
    body = (
        "İstanbul'da spor rutini.\nDün akşam koştuk.\n"
        "İşte yeni bir proje var.\nJasper'ı gözden geçirdik."
    )
    outcome = cli(CommandResult(0, reply((1, 2, "spor"), (3, 4, "iş")))).segment(record(body), 12)

    assert not isinstance(outcome, NeedsReview)
    assert "\n".join(s.body for s in outcome) == normalize(body, "x.md").body
    assert "İstanbul" in outcome[0].body


# --- The skip paths ----------------------------------------------------------------------


def test_a_declared_classification_skips_segmentation_entirely() -> None:
    drop = normalize("---\nclassification: personal\n---\n\n" + BODY, "declared.md")
    segmenter = FakeSegmenter()

    outcome = segment_record(
        drop,
        should_segment=True,
        config=SegmentConfig(enabled=True, min_chars=0),
        segmenter=segmenter,
    )

    assert not isinstance(outcome, NeedsReview)
    assert len(outcome) == 1
    assert outcome[0].id == drop.id
    assert outcome[0].is_whole_drop


def test_segmentation_off_returns_the_whole_drop() -> None:
    drop = record()

    outcome = segment_record(
        drop,
        should_segment=False,
        config=SegmentConfig(enabled=False, min_chars=0),
        segmenter=FakeSegmenter(),
    )

    assert not isinstance(outcome, NeedsReview)
    assert outcome == whole_drop(drop)


def test_a_body_under_min_chars_is_not_segmented() -> None:
    drop = record("Short note.")

    outcome = segment_record(
        drop,
        should_segment=True,
        config=SegmentConfig(enabled=True, min_chars=2000),
        segmenter=FakeSegmenter(),
    )

    assert not isinstance(outcome, NeedsReview)
    assert len(outcome) == 1
    assert outcome[0].id == drop.id


def test_the_whole_drop_segment_keeps_the_records_own_id() -> None:
    drop = record()
    only = whole_drop(drop)[0]

    assert only.id == drop.id
    assert only.parent_id == drop.id
    assert only.is_whole_drop


def test_a_segmented_drop_is_not_marked_whole() -> None:
    outcome = cli(CommandResult(0, reply((1, 3, "a"), (4, 6, "b")))).segment(record(), 12)

    assert not isinstance(outcome, NeedsReview)
    assert all(not s.is_whole_drop for s in outcome)


# --- Prompt ------------------------------------------------------------------------------


def test_the_prompt_numbers_every_line_and_states_the_last() -> None:
    drop = record()
    prompt = build_prompt(drop, max_segments=5)

    assert "1\tWe repainted the shed at the allotment." in prompt
    assert "6\tSam is rewriting the importer." in prompt
    assert "from 1 to 6" in prompt
    assert "at most 5 segments" in prompt


def test_number_lines_round_trips() -> None:
    lines, _ = number_lines(BODY)

    assert "\n".join(lines) == BODY


def test_the_prompt_carries_declared_metadata() -> None:
    drop = normalize("---\nsource: conversate\n---\n\n" + BODY, "c.md")

    assert "source: conversate" in build_prompt(drop, max_segments=5)


# --- The fake ----------------------------------------------------------------------------


def test_the_fake_segmenter_splits_on_its_marker() -> None:
    drop = record("First topic.\n---SPLIT---\nSecond topic.")
    outcome = FakeSegmenter(topics=("one", "two")).segment(drop, 12)

    assert not isinstance(outcome, NeedsReview)
    assert len(outcome) == 2
    assert isinstance(outcome[0], Segment)
