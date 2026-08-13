"""Splitting one drop into topical segments, so each part is routed on its own merits.

The engine was built on an assumption that ambient capture breaks: that one drop is about one
thing. A recorded conversation wanders — a real 350-message conversation in this vault covers
community gardening, relationship decisions, workplace social dynamics, fitness, and an argument
about food, in that order. Every whole-drop verdict for such a body is wrong. `personal` throws
away the work material; `work` and `mixed` both send a distilled note *about the relationship
material* into the work vault, because the note is written from the whole body.

Splitting first and asking per part restores the original assumption at a finer grain.

**This module cannot widen what crosses into the work vault.** It produces bodies; `route.py`
still decides destinations and `writer.py` still refuses to send a raw body to a note
destination. What changes is only how many bodies there are.

Two rules earn their own explanation, because the obvious implementation of each is wrong.

**Segments are spans of the original, never model-written text.** The segmenter returns line
ranges and this module slices them; the model never supplies prose. A segmenter that returned
rewritten text would put a paraphrase into the personal vault's raw transcript, which is
supposed to hold what was actually said. Every returned range is checked for contiguity and
total coverage before any slice is taken, so a model that quietly drops the boring middle of a
conversation fails the drop instead of silently losing it.

**A segment's identity is the hash of its own body**, computed through the same normalization
`inbox` uses. Re-running a sync over an unchanged conversation therefore re-derives identical
ids and the pass files nothing (KTD2). If a later model splits the same conversation
differently, the changed segments file as new items rather than overwriting the old ones — which
shows up in a diff, where it can be reviewed, instead of vanishing.

Failure is always `NeedsReview`, never a fallback split. There is no "well, one segment then"
branch anywhere in this file: a drop the segmenter could not handle is held in the inbox, which
is discovered today, rather than misfiled, which is discovered in six months (R9, KTD1).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, runtime_checkable

from memvault.classify import (
    CommandRunner,
    NeedsReview,
    _reply_text,
    run_command,
)
from memvault.config import ClassifierConfig, SegmentConfig
from memvault.inbox import InboxRecord, _normalize_text, content_id
from memvault.presets import build_argv, envelope_for
from memvault.presets import flatten as _flatten

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    """One topical part of a drop, carrying its own identity and its parent's.

    `topic` is a short label the model supplied. It is used for the filename slug and nothing
    else — no routing decision reads it — so a poor label costs readability, not correctness.
    """

    index: int
    count: int
    body: str
    id: str
    parent_id: str
    topic: str = ""

    @property
    def is_whole_drop(self) -> bool:
        """Whether this segment is the entire drop rather than a part of it.

        The unsegmented path produces exactly one of these, and `writer` uses it to decide
        whether to emit segment provenance at all — a drop that was never split should not gain
        `segment_index: 1 of 1` frontmatter it did not have before this feature existed.
        """
        return self.count == 1


#: What a segmenter returns. Callers branch on the type; there is no partial success.
SegmentOutcome: TypeAlias = "tuple[Segment, ...] | NeedsReview"


@runtime_checkable
class Segmenter(Protocol):
    """The seam. Tests inject stubs through it; the model tier is swapped behind it."""

    def segment(self, record: InboxRecord, max_segments: int) -> SegmentOutcome: ...


PROMPT_TEMPLATE = """\
You are splitting one recorded item into topical segments so each part can be filed separately.

The text below is numbered by line. Divide it into consecutive runs of lines, where each run is
about one subject. A segment boundary belongs where the subject genuinely changes, not every
time the speaker pauses.

Rules you must follow exactly:
- Segments must be contiguous and must together cover every line, from 1 to {last_line}.
- The first segment starts at line 1. The last segment ends at line {last_line}.
- Produce at most {max_segments} segments. Prefer few, substantial segments over many small ones.
- If the whole text is about one subject, return exactly one segment covering everything.
- Never rewrite, summarize, translate, or quote the text. You return line numbers only.

Reply with a single JSON object and nothing else: no preamble, no explanation, no Markdown code
fence. A reply that is not parseable JSON is discarded and the item is held for a human.

{{"segments": [{{"start_line": 1, "end_line": 42, "topic": "short lowercase label"}}]}}

{declared}Filename: {filename}

Numbered text:
{numbered}
"""


def number_lines(body: str) -> tuple[list[str], str]:
    """Return the body's lines and a line-numbered rendering for the prompt."""
    lines = body.split("\n")
    numbered = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))
    return lines, numbered


def build_prompt(record: InboxRecord, max_segments: int) -> str:
    """Render the segmentation prompt for one record. Pure."""
    from memvault.classify import _declared_block

    lines, numbered = number_lines(record.body)
    return PROMPT_TEMPLATE.format(
        declared=_declared_block(record.declared),
        filename=record.filename,
        numbered=numbered,
        last_line=len(lines),
        max_segments=max_segments,
    )


def _ranges(
    data: Any, line_count: int, max_segments: int
) -> tuple[list[tuple[int, int, str]], str | None]:
    """Validate a reply's segment list into `(start, end, topic)` triples, or explain why not.

    Every check here exists because its absence loses content silently. Contiguity and full
    coverage together mean the segments reconstruct the drop exactly; without them a model that
    skips a passage would file a conversation with a hole in it that nobody would notice.
    """
    if not isinstance(data, dict):
        return [], f"the segmenter's reply was {type(data).__name__}, not an object"

    raw = data.get("segments")
    if not isinstance(raw, list) or not raw:
        return [], "the segmenter's reply carried no 'segments' list"
    if len(raw) > max_segments:
        return [], f"the segmenter returned {len(raw)} segments, above the limit of {max_segments}"

    triples: list[tuple[int, int, str]] = []
    expected_start = 1
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            return [], f"segment {position} was {type(item).__name__}, not an object"
        try:
            start = int(item["start_line"])
            end = int(item["end_line"])
        except (KeyError, TypeError, ValueError):
            return [], f"segment {position} did not carry usable start_line and end_line"
        if start != expected_start:
            return [], (
                f"segment {position} starts at line {start}, but the previous segment ended at "
                f"{expected_start - 1} — segments must be contiguous"
            )
        if end < start:
            return [], f"segment {position} ends at line {end}, before its start at {start}"
        if end > line_count:
            return [], f"segment {position} ends at line {end}, past the last line {line_count}"
        topic = item.get("topic")
        triples.append((start, end, str(topic).strip() if topic else ""))
        expected_start = end + 1

    if expected_start != line_count + 1:
        return [], (
            f"the segments stop at line {expected_start - 1} but the text runs to {line_count} — "
            "every line must be covered"
        )
    return triples, None


def segments_from_ranges(
    record: InboxRecord,
    lines: list[str],
    triples: list[tuple[int, int, str]],
) -> SegmentOutcome:
    """Slice validated ranges into segments, dropping ones that hold no text.

    A run of blank lines between two subjects is legitimately part of neither. It is normalized
    away here rather than filed as an empty memory — but only after coverage has already been
    proven, so this can never be the reason content goes missing.
    """
    built: list[tuple[str, str]] = []
    for start, end, topic in triples:
        body = _normalize_text("\n".join(lines[start - 1 : end]))
        if body:
            built.append((body, topic))

    if not built:
        return NeedsReview("every segment the segmenter returned was empty after normalization")

    count = len(built)
    return tuple(
        Segment(
            index=position,
            count=count,
            body=body,
            id=content_id(body),
            parent_id=record.id,
            topic=topic,
        )
        for position, (body, topic) in enumerate(built, start=1)
    )


def whole_drop(record: InboxRecord) -> tuple[Segment]:
    """The single-segment form of a drop that was not split.

    Used by every path that skips segmentation — a declared classification, a body under
    `min_chars`, a source policy with segmentation off — so the ingest pass has one shape to
    handle rather than two. `id` is the record's own, which is what keeps an unsegmented drop
    filing under exactly the identity it had before this module existed.
    """
    return (
        Segment(
            index=1,
            count=1,
            body=record.body,
            id=record.id,
            parent_id=record.id,
        ),
    )


class FakeSegmenter:
    """Splits on a literal marker, so tests exercise the pipeline without a subprocess."""

    def __init__(self, marker: str = "---SPLIT---", topics: tuple[str, ...] = ()) -> None:
        self._marker = marker
        self._topics = topics

    def segment(self, record: InboxRecord, max_segments: int) -> SegmentOutcome:
        parts = [p for p in record.body.split(self._marker)]
        lines, _ = number_lines(record.body)
        if len(parts) == 1:
            return segments_from_ranges(record, lines, [(1, len(lines), "")])

        triples: list[tuple[int, int, str]] = []
        cursor = 1
        for position, part in enumerate(parts):
            span = len(part.split("\n"))
            end = min(cursor + span - 1, len(lines))
            topic = self._topics[position] if position < len(self._topics) else ""
            triples.append((cursor, end, topic))
            cursor = end + 1
        if triples:
            triples[-1] = (triples[-1][0], len(lines), triples[-1][2])
        return segments_from_ranges(record, lines, triples)


class ClaudeCliSegmenter:
    """The real segmenter: one `claude` CLI invocation per drop, mirroring the classifier.

    Construction takes no I/O and the runner is injectable, so instantiating this in a test is
    harmless — only calling `segment` with the default runner reaches a subprocess.
    """

    def __init__(
        self,
        classifier: ClassifierConfig,
        config: SegmentConfig,
        *,
        runner: CommandRunner = run_command,
    ) -> None:
        self._classifier = classifier
        self._config = config
        self._runner = runner

    def argv(self) -> list[str]:
        """The command line, exposed so a caller can log or assert on it.

        Built from the classifier config's preset — the segmenter shares the classifier's CLI
        because it is the same command answering a different question (see `presets`).
        """
        return build_argv(self._classifier)

    def segment(self, record: InboxRecord, max_segments: int) -> SegmentOutcome:
        outcome = self._segment(record, max_segments)
        if isinstance(outcome, NeedsReview):
            logger.info("%s: held for review — %s", record.filename, outcome.reason)
        return outcome

    def _segment(self, record: InboxRecord, max_segments: int) -> SegmentOutcome:
        if record.unparseable:
            return NeedsReview(f"the drop's own metadata could not be read: {record.unparseable}")

        lines, _ = number_lines(record.body)
        result = self._runner(
            self.argv(),
            prompt=build_prompt(record, max_segments),
            timeout=self._config.timeout_seconds,
        )

        if result.timed_out:
            return NeedsReview(
                f"the segmenter did not answer within {self._config.timeout_seconds}s"
            )
        if result.exit_code != 0:
            detail = _flatten(result.stderr) or "no error output"
            return NeedsReview(f"the segmenter exited {result.exit_code}: {detail}")

        reply, error = _reply_text(result.stdout, envelope_for(self._classifier))
        if error is not None or reply is None:
            return NeedsReview(error or "the segmenter produced no usable reply")

        try:
            data = json.loads(reply)
        except json.JSONDecodeError as exc:
            return NeedsReview(f"the segmenter's reply was not JSON: {_flatten(exc)}")

        triples, problem = _ranges(data, len(lines), max_segments)
        if problem is not None:
            return NeedsReview(problem)

        return segments_from_ranges(record, lines, triples)


def segment_record(
    record: InboxRecord,
    *,
    should_segment: bool,
    config: SegmentConfig,
    segmenter: Segmenter,
) -> SegmentOutcome:
    """Decide whether to split this drop, and split it if so.

    Three ways a drop skips segmentation, each returning the whole drop as one segment:

    - **A declared classification.** The contract promises that declaring it skips inference
      (R2/R7). Segmenting anyway and then stamping the declared label onto each part would
      honour the letter of that and break its spirit — the author said what this is, and the
      engine would still be making per-part judgments about it.
    - **Segmentation off** for this source, or globally.
    - **A body under `min_chars`**, which never pays for a model call.
    """
    if record.declared.declares("classification"):
        return whole_drop(record)
    if not should_segment:
        return whole_drop(record)
    if len(record.body) < config.min_chars:
        return whole_drop(record)
    return segmenter.segment(record, config.max_segments)
