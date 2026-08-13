"""Distilling work material into the only thing the work vault is allowed to hold.

R9 is the single strongest privacy claim this system makes: raw transcripts never reach the
work vault, which receives learnings, decisions, and action items and nothing else. `route.py`
expresses that as an asymmetry in a table, and `writer.py` refuses to write a raw body to a
`NOTE` destination. This module is the third and last mechanical guard, and the only one that
inspects text rather than types.

**The guard, in one sentence:** before anything is written to the work vault, every sentence of
the raw body is checked for how much of itself survives verbatim into the rendered note, and a
sentence reproduced above a configured fraction fails the write and marks the item for review.
It is not a request to the model. The prompt asks for paraphrase; this checks.

Four choices make that check hard to slip past, and each one is here because the obvious
alternative is not.

**Word n-grams, not sentences.** Comparing whole sentences for equality is defeated by changing
one word. Shingles of `DEFAULT_SHINGLE_SIZE` consecutive words are not: an edited word breaks
only the shingles that span it, and the untouched runs on either side still match. Six is the
default because it is long enough that two independently written sentences about the same
subject essentially never share one, and short enough that a single substitution in the middle
of an ordinary sentence still leaves matching runs to both sides of it.

**The raw sentence is the denominator, not the note.** Scoring "how much of the note is copied"
sounds right and is weak: one lifted sentence inside a long note is a small fraction of it and
would slide under any useful threshold. Scoring "how much of this transcript sentence came
through" does not dilute — a copied sentence scores near 1.0 whether the note around it is
three lines or thirty. The note's score is the worst sentence's, because one leaked sentence is
a leak.

**Coverage is measured in words, not in matching shingles.** A one-word edit in the middle of a
fifteen-word sentence destroys six of its ten shingles — a shingle-count ratio would read 0.4
and could be argued under a threshold. The words those surviving shingles cover are fourteen of
fifteen, which reads 0.93 and is the truth: the sentence came through.

**A sentence shorter than the shingle size scores zero, by construction.** This is what keeps
`"we should ship it"` from tripping the guard, and it is a rule rather than a stopword list: a
phrase too short to be distinctive is not evidence of copying. It is also the one deliberate
hole in the guard, and it is bounded at five words.

Normalization runs both sides through the same reduction — NFC, lowercase, and every character
that is not a letter or digit treated as a separator — so recasing, re-punctuating, or
reflowing a copied sentence changes nothing about its score. Turkish letters are *not* folded to
ASCII, matching `writer.slugify`: folding would make `açık` and `acik` the same word and invent
matches, and a model retyping Turkish into ASCII is not a threat model. The guard therefore
works on Turkish exactly as it works on English.

The text handed to the guard is the *whole* rendered document plus the path it would be written
to — frontmatter values and the filename slug included. A leaked sentence promoted into a title
would otherwise reach the work vault through the one field nobody thought to check.

Two smaller rules, both from the plan. An empty extraction writes nothing and logs why, because
an unremarkable standup should not litter the work vault. And a distiller that fails blocks only
the work write: the personal filing already happened and stands, so the failure costs a note,
not a memory.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TypeAlias, runtime_checkable

# `_reply_text`, `flatten`, and `_material_date` are borrowed rather than reimplemented. Each
# encodes an agreement this module has to hold to exactly: both LLM calls unwrap the same CLI
# envelope, and a note must carry the same date as the transcript it describes. A second copy
# of either would be free to drift, and the drift would be silent.
from memvault.classify import (
    Classification,
    CommandRunner,
    Relation,
    _reply_text,
    run_command,
)
from memvault.config import ClassifierConfig
from memvault.inbox import InboxRecord
from memvault.presets import build_argv, envelope_for
from memvault.presets import flatten as _flatten
from memvault.route import Destination, DestinationKind, RoutingResult
from memvault.writer import (
    WriteError,
    WrittenFile,
    _material_date,
    importance_of,
    render_path,
    render_relations,
    slugify,
    write_document,
)

logger = logging.getLogger(__name__)

#: How many consecutive words make a shingle. See the module docstring for why six: long
#: enough that an independent paraphrase never shares one, short enough that a single edited
#: word does not hide the rest of the sentence it sits in.
DEFAULT_SHINGLE_SIZE = 6

#: The fraction of one raw sentence that may survive verbatim into the note before the write
#: fails. A sentence more than this reproduced is copied, not paraphrased.
#:
#: Comparison is strictly greater-than, matching the classifier's confidence gate: a score
#: exactly at the threshold passes. That also gives `0.0` a useful meaning — the strictest
#: setting, where any matching run at all blocks the write — which `>=` would turn into a
#: configuration that blocks every note including clean ones.
DEFAULT_OVERLAP_THRESHOLD = 0.4

#: Sentence boundaries for the raw body: a line break, or terminal punctuation followed by
#: space. Over-splitting is safe in one direction only — a fragment below the shingle size
#: scores zero — so the pattern stays conservative rather than clever about abbreviations.
_SENTENCE_BREAK = re.compile(r"(?:[\n\r]+|(?<=[.!?…])\s+)")

#: Leading bullet and list markers, stripped so a model that answers with `"- Ship it"`
#: produces the same item as one that answers with `"Ship it"`.
_BULLET_PREFIX = re.compile(r"^[-*•–—\s]+")

PROMPT_TEMPLATE = """\
You are writing one short note for a shared work memory vault. It is distilled from an item
that was captured into a private personal vault, and that item stays private: it is never
copied, quoted, or reproduced here.

Extract only what is worth knowing weeks from now:
  decisions      what was decided
  action_items   what someone committed to do next
  learnings      durable insight — something still true on a different project

Paraphrase everything. Do not quote. Do not reuse the source's wording, its sentence shapes, or
its order. Write each item as one short sentence in your own words. An automated check compares
this note against the source and rejects the whole note when it reproduces the source's
phrasing, so copying one line costs every line.

Leave a list empty when there is genuinely nothing of that kind. An unremarkable conversation
should produce an empty note; padding it is worse than writing nothing at all.

Reply with a single JSON object and nothing else: no preamble, no explanation, no Markdown code
fence. A reply that is not parseable JSON is discarded and no note is written.

Fields:
  title          a short neutral title for the note, in sentence case
  decisions      a list of strings
  action_items   a list of strings
  learnings      a list of strings

Classification: {classification}
{context}
Content:
{body}
"""


@dataclass(frozen=True)
class Distillation:
    """What the second LLM call extracted, before anything has been written.

    `model` is carried for provenance the same way `Classification.model` is — a reader of the
    note should be able to tell which model wrote the words in front of them.
    """

    title: str = ""
    decisions: tuple[str, ...] = ()
    action_items: tuple[str, ...] = ()
    learnings: tuple[str, ...] = ()
    model: str | None = None

    @property
    def empty(self) -> bool:
        """Whether there is nothing here worth a file.

        A title alone is not content: a model asked for a note about an unremarkable standup
        will happily name it and leave every list empty, and naming a thing is not learning it.
        """
        return not (self.decisions or self.action_items or self.learnings)

    def render(self) -> str:
        """The note body as Markdown, with empty sections left out rather than written bare."""
        sections = (
            ("Decisions", self.decisions),
            ("Action items", self.action_items),
            ("Learnings", self.learnings),
        )
        blocks = [
            "## " + heading + "\n\n" + "\n".join(f"- {item}" for item in items)
            for heading, items in sections
            if items
        ]
        return "\n\n".join(blocks)


@dataclass(frozen=True)
class DistillationFailed:
    """Not an extraction: the reason no note can be written from this drop.

    Phrased for the human who reads the review marker, and kept to one line for the same
    reason `NeedsReview.reason` is.
    """

    reason: str


#: What a distiller returns. As in `classify`, there is no third state and in particular no
#: "extracted, but badly" that a caller could mistake for content.
DistillOutcome: TypeAlias = Distillation | DistillationFailed


@runtime_checkable
class Distiller(Protocol):
    """The seam. Tests inject stubs through it; no test spawns a subprocess."""

    def distill(self, record: InboxRecord, verdict: Classification) -> DistillOutcome: ...


@dataclass(frozen=True)
class LeakageReport:
    """How much of the raw body survived verbatim into the note, and how that was measured.

    `sentence_index` names the worst sentence by position rather than by text. The report ends
    up in logs and notifications, which are not places raw material should be reproduced — the
    whole point of this module is that transcript text stays in one vault. A reviewer counting
    sentences in the drop finds it; a log file scraped by something else does not.
    """

    overlap: float
    threshold: float
    shingle_size: int
    sentence_index: int | None = None
    sentence_words: int = 0
    sentences_checked: int = 0

    @property
    def tripped(self) -> bool:
        """Whether this note must not be written."""
        return self.overlap > self.threshold

    def reason(self) -> str:
        """One line naming the numbers and where to look, and never the text itself."""
        where = (
            f"sentence {self.sentence_index + 1} of the raw body ({self.sentence_words} words)"
            if self.sentence_index is not None
            else "the raw body"
        )
        return (
            f"the distilled note reproduces {self.overlap:.0%} of {where} verbatim, above the "
            f"configured leakage threshold of {self.threshold:.0%} "
            f"({self.shingle_size}-word shingles); nothing was written to the work vault"
        )


@dataclass(frozen=True)
class DistillResult:
    """The outcome of one work-vault distillation, with every branch expressed as data.

    Exactly one of `written`, `review`, and `skipped` is set. `review` means a human should
    look — the distiller failed, or the guard caught the note leaking — and `skipped` means
    nothing was owed or nothing was found, which is an ordinary day. The raw filing done by U4
    is untouched by all three.
    """

    record_id: str
    written: WrittenFile | None = None
    review: str | None = None
    skipped: str | None = None
    leakage: LeakageReport | None = None
    distillation: Distillation | None = None

    @property
    def needs_review(self) -> bool:
        return self.review is not None

    @property
    def wrote(self) -> bool:
        return self.written is not None


def _tokens(text: str) -> list[str]:
    """Reduce text to comparable words.

    Letters and digits survive in whatever script they are written in; everything else — case,
    punctuation, quote style, line breaks, emoji — is a separator. Two renderings of the same
    sentence therefore tokenize identically, which is what makes recasing and re-punctuating
    useless as an evasion.

    Turkish is not folded to ASCII, deliberately, for the reason `writer.slugify` gives: `açık`
    and `acik` are different words, and folding them together would invent matches rather than
    catch them. `İ` is mapped before lowercasing so it does not decompose into a bare `i` plus
    a combining mark that the next step would drop anyway.
    """
    lowered = unicodedata.normalize("NFC", text).replace("İ", "i").lower()
    normalized = unicodedata.normalize("NFC", lowered)

    words: list[str] = []
    current: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category[0] in ("L", "N"):
            current.append(char)
        elif category[0] == "M":
            continue
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def _sentences(text: str) -> list[str]:
    """Split the raw body into the units the guard scores, keeping empty pieces out."""
    return [piece for piece in _SENTENCE_BREAK.split(text) if piece.strip()]


def _shingles(words: Sequence[str], size: int) -> set[tuple[str, ...]]:
    """Every run of `size` consecutive words, as a set for O(1) lookup."""
    return {tuple(words[start : start + size]) for start in range(len(words) - size + 1)}


def _covered_fraction(words: Sequence[str], haystack: set[tuple[str, ...]], size: int) -> float:
    """What fraction of `words` sits inside a run of `size` words found in `haystack`.

    Measured in words rather than in matching shingles: a single edited word breaks every
    shingle spanning it, but the runs either side of it still cover almost the whole sentence,
    and that is the honest description of a near-copy.

    A sentence shorter than one shingle scores zero. That is the rule that lets short common
    phrases through, and it is bounded — nothing under `size` words can ever trip the guard.
    """
    if len(words) < size:
        return 0.0

    covered = [False] * len(words)
    for start in range(len(words) - size + 1):
        if tuple(words[start : start + size]) in haystack:
            for index in range(start, start + size):
                covered[index] = True
    return sum(covered) / len(words)


def verbatim_overlap(
    raw_body: str,
    note_text: str,
    *,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> LeakageReport:
    """Score how much of `raw_body` survives verbatim into `note_text`. Pure — no I/O.

    The note is treated as one stream of words, so a copied sentence is found however the note
    reflowed or re-bulleted it. The raw body is scored sentence by sentence and the note takes
    the worst sentence's score, because one leaked sentence is a leak whatever surrounds it.
    """
    if shingle_size < 1:
        raise ValueError(f"shingle_size must be at least 1, got {shingle_size}")

    note_shingles = _shingles(_tokens(note_text), shingle_size)

    worst = 0.0
    worst_index: int | None = None
    worst_words = 0
    checked = 0

    for index, sentence in enumerate(_sentences(raw_body)):
        words = _tokens(sentence)
        if len(words) < shingle_size:
            continue
        checked += 1
        fraction = _covered_fraction(words, note_shingles, shingle_size)
        if fraction > worst:
            worst, worst_index, worst_words = fraction, index, len(words)

    return LeakageReport(
        overlap=worst,
        threshold=threshold,
        shingle_size=shingle_size,
        sentence_index=worst_index,
        sentence_words=worst_words,
        sentences_checked=checked,
    )


def _bullets(value: Any) -> tuple[str, ...]:
    """Read a list-ish JSON value as note bullets.

    A bare string is accepted as a one-item list for the reason `classify._string_tuple` gives:
    a model answering a list field with a string is a formatting slip in descriptive content,
    not a decision worth holding an item over. Leading list markers are stripped so the
    rendered note does not end up with `- - Ship it`.
    """
    if isinstance(value, str):
        raw_items: list[Any] = [value]
    elif isinstance(value, list):
        raw_items = list(value)
    else:
        return ()

    items = []
    for item in raw_items:
        text = _BULLET_PREFIX.sub("", " ".join(str(item).split())).strip()
        if text:
            items.append(text)
    return tuple(items)


def build_distill_prompt(record: InboxRecord, verdict: Classification) -> str:
    """Render the distillation prompt for one record. Pure."""
    context = ""
    if verdict.project:
        context = f"Project: {verdict.project}\n"
    return PROMPT_TEMPLATE.format(
        classification=verdict.classification,
        context=context,
        body=record.body,
    )


def _extraction(reply: str, config: ClassifierConfig) -> DistillOutcome:
    """Read one reply into an extraction, or into the reason it is not one."""
    try:
        data: Any = json.loads(reply)
    except json.JSONDecodeError as exc:
        return DistillationFailed(
            f"the distiller's reply was not the bare JSON object it was asked for: {_flatten(exc)}"
        )

    if not isinstance(data, dict):
        return DistillationFailed(
            f"the distiller replied with a JSON {type(data).__name__}, not an object"
        )

    return Distillation(
        title=str(data.get("title") or "").strip(),
        decisions=_bullets(data.get("decisions")),
        action_items=_bullets(data.get("action_items")),
        learnings=_bullets(data.get("learnings")),
        model=config.model,
    )


class ClaudeCliDistiller:
    """The real distiller: a second `claude` CLI invocation, after classification.

    It reuses `ClassifierConfig` rather than inventing a config section of its own. The two
    calls want the same three things — a command, a model, and a timeout — and U1's schema is
    closed to unknown keys, so a `distiller:` section is a config change rather than a default.
    U6 or U10 can split them when there is a reason to run the two at different model tiers.

    Construction takes no I/O and the runner is injectable, exactly as in `classify`, so only
    calling `distill` with the default runner reaches a subprocess.
    """

    def __init__(self, config: ClassifierConfig, *, runner: CommandRunner = run_command) -> None:
        self._config = config
        self._runner = runner

    def argv(self) -> list[str]:
        """The command line, exposed so a caller can log or assert on it.

        Built from `classifier.preset` — see `presets` for why all four model-facing call
        sites share one table rather than one copy each.
        """
        return build_argv(self._config)

    def distill(self, record: InboxRecord, verdict: Classification) -> DistillOutcome:
        outcome = self._distill(record, verdict)
        if isinstance(outcome, DistillationFailed):
            logger.info("%s: no work note — %s", record.filename, outcome.reason)
        return outcome

    def _distill(self, record: InboxRecord, verdict: Classification) -> DistillOutcome:
        result = self._runner(
            self.argv(),
            prompt=build_distill_prompt(record, verdict),
            timeout=self._config.timeout_seconds,
        )

        if result.timed_out:
            return DistillationFailed(
                f"the distiller did not answer within {self._config.timeout_seconds}s"
            )
        if result.exit_code != 0:
            detail = _flatten(result.stderr) or "no error output"
            return DistillationFailed(f"the distiller exited {result.exit_code}: {detail}")

        reply, error = _reply_text(result.stdout, envelope_for(self._config))
        if error is not None or reply is None:
            # `_reply_text` words its reasons for the classifier, which is the only other
            # caller. Every one of them opens with the same three words, so re-aiming them is
            # exact — and cheaper than a second copy of the envelope's shape to keep in step.
            message = error or "the distiller produced no usable reply"
            return DistillationFailed(message.replace("the classifier", "the distiller"))

        return _extraction(reply, self._config)


def note_slug(record: InboxRecord, verdict: Classification, distillation: Distillation) -> str:
    """The distiller's title, then the classifier's, then the content id.

    The distiller's title comes first because it describes the note, and the note is what this
    file is. The id fallback is the same deliberately ugly one `writer` uses: unique, stable,
    and visibly a signal that no title survived slugging.
    """
    for candidate in (distillation.title, verdict.slug, verdict.title):
        slug = slugify(candidate)
        if slug:
            return slug
    return f"untitled-{record.id[:8]}"


def build_note_frontmatter(
    record: InboxRecord,
    verdict: Classification,
    routing: RoutingResult,
    distillation: Distillation,
    *,
    day: date,
    ingested: date,
) -> dict[str, Any]:
    """Assemble the provenance block for a distilled note.

    `content_id` is the whole of the cross-vault reference (R9): it names the personal-vault
    file this note came from without naming its path, its filename, or a word of its content.
    Anyone holding both vaults can find the transcript by grepping the personal one for the id;
    anyone holding only this vault learns nothing from it.

    Two fields the raw writer records are deliberately absent. `original_filename` names a file
    in the other vault, and a drop called `acme-layoffs.md` would leak through a metadata field
    while the guard watched the body. `summary` is the classifier's prose about the raw
    material, and the note's own body is already the summary this vault is entitled to.
    """
    tags = record.declared.tags if record.declared.declares("tags") else verdict.tags
    return {
        "title": distillation.title or verdict.title,
        "date": day.isoformat(),
        "source": record.declared.source,
        "classification": routing.classification,
        "project": verdict.project,
        "participants": list(routing.participants),
        "tags": list(tags),
        "ingested": ingested.isoformat(),
        "content_id": record.id,
        "importance": importance_of(record, verdict),
        "classifier_model": verdict.model,
    }


def _resolves_in_vault(root: Path, target: str) -> bool:
    """Whether this relation target names something the destination vault actually holds.

    Existence is the test, not plausibility. A target that resolves to nothing here is either a
    path in the *other* vault or a title only the other vault knows, and both describe the
    private vault's shape to a reader who is not entitled to it. In the personal vault an
    unresolved link is an ordinary pending edge; across this boundary it is a disclosure, so the
    two are treated differently on purpose.

    `x` and `x.md` are both tried, because a wikilink is normally written without its extension.
    """
    relative = PurePosixPath(target)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    candidate = root.joinpath(*relative.parts)
    return candidate.exists() or candidate.with_name(f"{candidate.name}.md").exists()


def vault_local_relations(
    relations: Sequence[Relation], root: Path, *, filename: str, vault: str
) -> tuple[Relation, ...]:
    """Keep only the relations whose targets live inside the destination vault. Logs the rest.

    A dropped relation costs an edge and nothing else — the note's body says the same thing
    either way — so this fails toward filing rather than toward the inbox. That is the opposite
    of the leakage guard's choice, and deliberately: the guard catches the *content* being
    copied, where the only safe answer is to write nothing, while here the offending field is
    simply left out and what remains is still a complete note.
    """
    kept: list[Relation] = []
    for relation in relations:
        if _resolves_in_vault(root, relation.target):
            kept.append(relation)
        else:
            logger.info(
                "%s: dropped relation %s [[%s]] — it resolves to nothing in vault %r, so it "
                "would describe another vault's shape inside this one",
                filename,
                relation.predicate,
                relation.target,
                vault,
            )
    return tuple(kept)


def _provenance_footer(record: InboxRecord, distillation: Distillation) -> str:
    """One line saying what this note is and where the thing it describes lives.

    The distiller's model goes here rather than in frontmatter because U4 fixed the frontmatter
    key order and `classifier_model` already means something else: the model whose judgement
    routed this material. Both facts are worth having and only one has a key.
    """
    by_model = f" by {distillation.model}" if distillation.model else ""
    return (
        f"*Distilled{by_model} from personal-vault item `{record.id}`. "
        "The material it came from stays in that vault.*"
    )


def _guardable_text(frontmatter: dict[str, Any]) -> str:
    """Flatten frontmatter values into plain text for the guard to read.

    Only the *values* matter: a key name is this engine's word, not the drop's. Rendering them
    through YAML would work too, but this cannot raise on a value the dumper dislikes, and the
    guard must never be the thing that fails a write for an unrelated reason.
    """
    parts: list[str] = []
    for value in frontmatter.values():
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return "\n".join(parts)


def _note_destination(
    routing: RoutingResult, destination: Destination | None
) -> Destination | None:
    """The note destination to write, or None when this record is owed no note.

    None is an ordinary answer, not an error: `personal` material, a `local_only` drop, and a
    single-vault install all reach here legitimately and all mean the same thing — nothing
    crosses.
    """
    if destination is not None:
        return destination

    notes = routing.of_kind(DestinationKind.NOTE)
    if not notes:
        return None
    if len(notes) > 1:
        raise WriteError(
            f"routing produced {len(notes)} note destinations for {routing.record_id}. "
            "Name one explicitly with `destination=`."
        )
    return notes[0]


def distill_record(
    record: InboxRecord,
    verdict: Classification,
    routing: RoutingResult,
    work_root: Path,
    *,
    distiller: Distiller,
    destination: Destination | None = None,
    ingested: date | None = None,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> DistillResult:
    """Distil one routed record into the work vault, or explain why nothing was written.

    The entry point U6 calls, after `writer.file_record` has filed the raw body into the
    personal vault. Nothing this function does can affect that filing: it either writes one new
    file into `work_root` or writes nothing at all.

    The order of the checks matters. A record owed no note never reaches the distiller, so
    personal material is not sent to an LLM to produce something that would be discarded. The
    guard runs after rendering and before writing, on the rendered document *and* the path it
    would occupy, so a leak has no field left to travel through.
    """
    target = _note_destination(routing, destination)
    if target is None:
        return DistillResult(record_id=record.id, skipped="no work-vault note is owed")

    if target.kind is not DestinationKind.NOTE:
        raise WriteError(
            f"{record.filename}: refusing to distil into a {target.kind.value!r} destination "
            f"in vault {target.vault!r}. Distilled notes go to note destinations; raw bodies "
            "are filed by the transcript writer."
        )

    outcome = distiller.distill(record, verdict)
    if isinstance(outcome, DistillationFailed):
        logger.info("%s: work note blocked — %s", record.filename, outcome.reason)
        return DistillResult(record_id=record.id, review=outcome.reason)

    if outcome.empty:
        reason = (
            "the distiller found no decisions, action items, or learnings worth keeping, so "
            f"no note was written to {target.vault!r}"
        )
        logger.info("%s: %s", record.filename, reason)
        return DistillResult(record_id=record.id, skipped=reason, distillation=outcome)

    ingested = ingested or date.today()
    day = _material_date(record, ingested)
    slug = note_slug(record, verdict, outcome)

    relative_path = render_path(target.path_template, day=day, slug=slug)
    frontmatter = build_note_frontmatter(
        record, verdict, routing, outcome, day=day, ingested=ingested
    )
    relations = render_relations(
        vault_local_relations(
            verdict.relations, work_root, filename=record.filename, vault=target.vault
        )
    )
    body = (
        f"{outcome.render()}\n\n"
        + (f"{relations}\n\n" if relations else "")
        + _provenance_footer(record, outcome)
    )

    # Everything that would land in the work vault is checked, including the frontmatter values
    # and the filename: a leaked sentence promoted into a title must not travel through the one
    # field the body check would never see.
    candidate = f"{relative_path}\n{_guardable_text(frontmatter)}\n{body}"
    leakage = verbatim_overlap(
        record.body,
        candidate,
        shingle_size=shingle_size,
        threshold=overlap_threshold,
    )
    if leakage.tripped:
        reason = leakage.reason()
        logger.warning("%s: work note blocked — %s", record.filename, reason)
        return DistillResult(
            record_id=record.id, review=reason, leakage=leakage, distillation=outcome
        )

    placement = write_document(work_root, relative_path, frontmatter=frontmatter, body=body)
    logger.info("distilled %s into %s/%s", record.filename, target.vault, placement.relative_path)

    written = WrittenFile(
        vault=target.vault,
        kind=DestinationKind.NOTE,
        path=placement.path,
        relative_path=placement.relative_path,
        record_id=record.id,
        title=frontmatter["title"],
        slug=slug,
        classification=routing.classification,
        suffix=placement.suffix,
    )
    return DistillResult(
        record_id=record.id, written=written, leakage=leakage, distillation=outcome
    )
