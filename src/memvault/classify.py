"""Classification of inbox drops, behind an interface.

Classification decides where a memory lands, and a memory that lands in the wrong vault is
discovered months later while a stuck inbox is discovered today. So this module has exactly one
opinion, and it runs through every branch: it would rather return nothing than return a guess.
Malformed output, a failed subprocess, and a confidence below the configured threshold all come
back as `NeedsReview` carrying the reason (KTD6). There is no fallback classification anywhere
in this file, deliberately.

The real classifier shells out to a CLI (KTD4) — no SDK, no API key. Which CLI, and what its
output is wrapped in, comes from `classifier.preset` and lives in `presets`; the default is
`claude -p --output-format json`, which is what this vault runs. The child process inherits
this one's environment, `CLAUDE_CONFIG_DIR` included, which is what runs classification on the
existing subscription. It also runs without a shell, so a machine whose interactive shell
shadows `claude` with a guard function is unaffected; where that is not enough,
`classifier.command` takes an absolute path.

Everything reaches the pipeline through the `Classifier` protocol, so a test injects a stub and
no test ever spawns a subprocess. The seam is also where a model tier gets swapped.

Declared frontmatter short-circuits the inference for its own field and nothing else (R2): a
drop that declared `classification: personal` is filed personal whatever the model concluded,
but the model still supplies the title and summary nobody wrote down.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, runtime_checkable

from memvault.config import AreaConfig, ClassifierConfig
from memvault.inbox import CLASSIFICATIONS, DeclaredMetadata, InboxRecord
from memvault.presets import Envelope, build_argv, envelope_for, unwrap
from memvault.presets import flatten as _flatten

logger = logging.getLogger(__name__)

#: Fields a reply must carry to be usable. A reply missing any of them is malformed rather than
#: merely sparse: without a classification there is no route, and without a title there is no
#: filename for U4 to render.
REQUIRED_FIELDS = ("classification", "confidence", "title")

#: The band `importance` lives in. Ten points is enough resolution to separate "worth
#: resurfacing" from "worth keeping" without inviting a model to agonize over 63 versus 67.
IMPORTANCE_MIN, IMPORTANCE_MAX = 1, 10

#: How many relations one reply may contribute. A note is a memory, not a link farm: a runaway
#: reply must not bury a two-line summary under fifty wikilinks, and the graph gains nothing
#: from a node that points at everything. Excess is dropped with a log, not held.
MAX_RELATIONS = 12

#: A predicate is an identifier, not a sentence. Normalization already lowercases and joins
#: words with underscores, so anything still failing this is not a relation type.
_PREDICATE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Wrapping a target in wikilink brackets is the model doing the rendering's job. Stripped
#: rather than rejected, because `[[ranking]]` plainly means `ranking`.
_WIKILINK_WRAPPER = re.compile(r"^\[\[(.*)\]\]$")

PROMPT_TEMPLATE = """\
You are filing one item that was dropped into a personal memory vault's inbox.

Decide whether its content is personal, work, or a mix of both, and describe it well enough to
file and to find again later.

Reply with a single JSON object and nothing else: no preamble, no explanation, no Markdown code
fence. A reply that is not parseable JSON is discarded and the item is held for a human.

Fields:
  classification  exactly one of: personal, work, mixed
  confidence      0.0 to 1.0 — how sure you are of the classification, honestly
  title           a short human title in sentence case
  slug            a lowercase filename-safe form of the title, words joined by hyphens
  summary         one or two sentences saying what this item is
  participants    people involved, as a list of names; [] when none are identifiable
  tags            a few lowercase topic tags, as a list
  project         the project this belongs to, or null when it belongs to none
  importance      1 to 10 — how much this deserves to resurface months from now. 5 is ordinary;
                  8 and above is a decision, a commitment, or something that changed direction
  relations       what this item connects to in the vault, as a list of objects:
                    predicate  a short lowercase verb phrase, such as part_of, about, follows_up
                    target     a vault-relative path or note title, without brackets
                  [] when nothing obvious connects. Do not invent paths; name only what you were
                  told about above
{area_field}
{declared}Filename: {filename}

Content:
{body}
"""


@dataclass(frozen=True)
class Relation:
    """One typed edge from the item being filed to something else in the vault.

    Rendered into note bodies as `- <predicate> [[<target>]]` and parsed back out of them by the
    indexer, which is what keeps the graph a property of the files rather than of the database.
    `target` is text, not a resolved path: whether it names anything is a whole-corpus question
    the indexer answers later, and a link to a file that does not exist yet is an ordinary state
    rather than an error.
    """

    predicate: str
    target: str

    def render(self) -> str:
        """The line as it appears in a note body."""
        return f"- {self.predicate} [[{self.target}]]"


@dataclass(frozen=True)
class Classification:
    """A verdict the pipeline can act on.

    `model` is carried for provenance — U4 writes it into the filed file's frontmatter, so a
    later reader can tell which model's judgement put the file where it is.
    """

    classification: str
    confidence: float
    title: str
    slug: str = ""
    summary: str = ""
    participants: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    project: str | None = None
    #: Which topic area the distilled note belongs in, chosen from the list the prompt
    #: enumerated. None means the model declined, which is a permitted answer and not a failure:
    #: the note is additive, so no note beats a note in the wrong place.
    area: str | None = None
    #: How much this deserves to resurface, 1-10, or None when the model gave no usable answer.
    #: None is not zero and not low: ranking reads an absent score as neutral, because a corpus
    #: filed before this field existed must not sink under one filed after it.
    importance: int | None = None
    #: What this item connects to. Emitted into note bodies, never into frontmatter — the graph
    #: belongs in the text a person reads, where an edit to it is an ordinary edit.
    relations: tuple[Relation, ...] = ()
    model: str | None = None


@dataclass(frozen=True)
class NeedsReview:
    """Not a verdict: the reason this drop stays in the inbox.

    `reason` is written into the review marker beside the drop, so it is phrased for the human
    who will read it and kept to a single line.
    """

    reason: str
    confidence: float | None = None


#: What a classifier returns. Callers branch on the type; there is no third state, and in
#: particular no "classified, but badly" that could be mistaken for a verdict.
Outcome: TypeAlias = Classification | NeedsReview


@runtime_checkable
class Classifier(Protocol):
    """The seam. Tests inject stubs through it; the model tier is swapped behind it."""

    def classify(self, record: InboxRecord) -> Outcome: ...


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one classifier invocation, with failure expressed as data.

    A timeout and a missing binary are ordinary results here rather than exceptions, because
    both are ordinary at this boundary and both end the same way: the item is held.
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class CommandRunner(Protocol):
    """How the classifier reaches a subprocess. Injectable so tests never spawn one."""

    def __call__(self, argv: Sequence[str], *, prompt: str, timeout: int) -> CommandResult: ...


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Read a list-ish JSON value as a tuple of non-empty strings.

    A model asked for a list occasionally answers with a bare string. That is a formatting slip
    in a descriptive field, not a routing decision, so it is accommodated rather than held.
    """
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _importance(value: Any) -> int | None:
    """Read the reply's `importance` tolerantly, or answer None. Never a reason to hold.

    Importance only tilts ranking, months later and by a hair. A drop held in the inbox over an
    unreadable one would trade a real memory for a sorting preference, so every failure — a
    word, a null, a 42 — comes back as "nobody judged" and the item files exactly as it would
    have before this field existed.

    Booleans are refused before anything else because `True` is an `int` in Python and would
    otherwise arrive as a perfectly valid importance of 1.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = round(float(value))
    except (TypeError, ValueError):
        return None
    if not IMPORTANCE_MIN <= number <= IMPORTANCE_MAX:
        return None
    return number


def _relation(item: Any) -> Relation | None:
    """Read one relation object, or None when it is not one.

    Both halves are normalized rather than trusted: a predicate written as `"Part Of"` is the
    same edge as `part_of` and becomes it, while a target is stripped of the wikilink brackets a
    model sometimes adds. What survives normalization still has to look like a vault-relative
    pointer — an absolute path or one climbing out with `..` is dropped, since a relation is a
    link inside the vault by definition and a link outside it is a mistake in any vault.
    """
    if not isinstance(item, dict):
        return None

    predicate = "_".join(str(item.get("predicate") or "").lower().replace("-", " ").split())
    if not _PREDICATE.match(predicate):
        return None

    target = " ".join(str(item.get("target") or "").split())
    unwrapped = _WIKILINK_WRAPPER.match(target)
    if unwrapped:
        target = unwrapped.group(1).strip()
    if not target or "[[" in target or "]]" in target:
        return None
    if target.startswith("/") or ".." in target.split("/"):
        return None

    return Relation(predicate=predicate, target=target)


def _relations(value: Any) -> tuple[Relation, ...]:
    """Read the reply's `relations` list, keeping the ones that are usable and in order."""
    if not isinstance(value, list):
        return ()

    kept: list[Relation] = []
    for item in value:
        relation = _relation(item)
        if relation is None or relation in kept:
            continue
        kept.append(relation)

    if len(kept) > MAX_RELATIONS:
        logger.info(
            "the classifier returned %d relations; keeping the first %d",
            len(kept),
            MAX_RELATIONS,
        )
    return tuple(kept[:MAX_RELATIONS])


def run_command(argv: Sequence[str], *, prompt: str, timeout: int) -> CommandResult:
    """Run the classifier command with the prompt on stdin.

    No `env=` is passed, so the child inherits this process's environment — `CLAUDE_CONFIG_DIR`
    included, which is what lets classification run on the existing subscription (KTD4). No
    shell is involved either, so a shell function shadowing `claude` never applies; where the
    author still wants a specific binary, `classifier.command` takes an absolute path.

    The prompt goes on stdin rather than in an argument because a transcript can be far longer
    than the platform's argument limit.
    """
    try:
        completed = subprocess.run(
            list(argv),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(exit_code=-1, timed_out=True)
    except OSError as exc:
        return CommandResult(exit_code=-1, stderr=f"could not run {argv[0]!r}: {_flatten(exc)}")

    return CommandResult(
        exit_code=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _declared_block(declared: DeclaredMetadata) -> str:
    """Tell the model what the drop already said about itself.

    The engine overrides these fields regardless of what comes back, but a model that can see
    them writes a summary consistent with them instead of one that argues with the frontmatter.
    """
    lines = []
    if declared.declares("classification") and declared.classification:
        lines.append(f"  classification: {declared.classification}")
    if declared.declares("source") and declared.source:
        lines.append(f"  source: {declared.source}")
    if declared.declares("date") and declared.date:
        lines.append(f"  date: {declared.date}")
    if declared.declares("participants") and declared.participants:
        lines.append(f"  participants: {', '.join(declared.participants)}")
    if declared.declares("tags") and declared.tags:
        lines.append(f"  tags: {', '.join(declared.tags)}")

    if not lines:
        return ""

    return (
        "The author already stated the following; do not contradict it:\n"
        + "\n".join(lines)
        + "\n\n"
    )


def _area_field(areas: Sequence[AreaConfig]) -> str:
    """Offer the model a closed list of destinations, and permission to decline.

    Names alone are ambiguous — "finance" could mean anything — so each area's `when` hint is what
    actually carries this vault owner's meaning. Projects have no hint because their names already
    are one. A vault that declares no areas gets no field at all rather than an empty list, which
    would invite the model to invent one.
    """
    if not areas:
        return ""

    lines = [
        f"                    - {area.name}" + (f": {area.when}" if area.when else "")
        for area in areas
    ]
    # The trailing newline keeps the blank line the template puts between the field list and
    # the metadata block. Without it the areas run straight into `Filename:`.
    return (
        "  area            exactly one name from the list below, or null when none clearly "
        "fits.\n"
        "                  null is a good answer — a note filed in the wrong place is worse "
        "than no note.\n" + "\n".join(lines) + "\n"
    )


def build_prompt(record: InboxRecord, areas: Sequence[AreaConfig] = ()) -> str:
    """Render the classification prompt for one record. Pure."""
    return PROMPT_TEMPLATE.format(
        area_field=_area_field(areas),
        declared=_declared_block(record.declared),
        filename=record.filename,
        body=record.body,
    )


def _reply_text(
    stdout: str, envelope: Envelope = Envelope.CLAUDE_JSON
) -> tuple[str | None, str | None]:
    """Unwrap one CLI's output to get at the model's reply.

    Which envelope that is comes from `classifier.preset` — `presets.unwrap` holds the four
    shapes and the reasons for each. The default is claude's, so a caller that names none is
    reading the CLI this vault has always run, and every one of them keeps the bare-reply
    fallback that lets a stub answer without faking an envelope.
    """
    return unwrap(stdout, envelope)


def _verdict(reply: str, config: ClassifierConfig, declared: DeclaredMetadata) -> Outcome:
    """Read one reply into a verdict, or into the reason it is not one.

    The confidence gate runs before the declared-classification override, and that ordering is
    deliberate: confidence describes how well the model read the drop as a whole, not only how
    sure it is of the label. A drop the model could barely make sense of is worth a human's
    glance even when the author already named its vault, and releasing it costs one edit.
    """
    try:
        data: Any = json.loads(reply)
    except json.JSONDecodeError as exc:
        return NeedsReview(
            f"the classifier's reply was not the bare JSON object it was asked for: {_flatten(exc)}"
        )

    if not isinstance(data, dict):
        return NeedsReview(
            f"the classifier replied with a JSON {type(data).__name__}, not an object"
        )

    missing = [key for key in REQUIRED_FIELDS if data.get(key) in (None, "")]
    if missing:
        return NeedsReview(
            f"the classifier's reply is missing required field(s): {', '.join(missing)}"
        )

    label = str(data["classification"]).strip().lower()
    if label not in CLASSIFICATIONS:
        return NeedsReview(
            f"the classifier returned classification {data['classification']!r}, which is not "
            f"one of {', '.join(sorted(CLASSIFICATIONS))}"
        )

    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        return NeedsReview(
            f"the classifier returned a non-numeric confidence: {data['confidence']!r}"
        )
    if not 0.0 <= confidence <= 1.0:
        return NeedsReview(f"the classifier returned a confidence outside 0-1: {confidence}")

    if confidence < config.confidence_threshold:
        return NeedsReview(
            f"confidence {confidence:.2f} is below the configured threshold "
            f"{config.confidence_threshold:.2f}, so the item stays in the inbox rather than "
            "being filed on a guess",
            confidence=confidence,
        )

    if declared.declares("classification") and declared.classification:
        label = declared.classification

    return Classification(
        classification=label,
        confidence=confidence,
        title=str(data["title"]).strip(),
        slug=str(data.get("slug") or "").strip(),
        summary=str(data.get("summary") or "").strip(),
        participants=(
            declared.participants
            if declared.declares("participants")
            else _string_tuple(data.get("participants"))
        ),
        tags=(declared.tags if declared.declares("tags") else _string_tuple(data.get("tags"))),
        project=(str(data["project"]).strip() or None) if data.get("project") else None,
        area=(str(data["area"]).strip() or None) if data.get("area") else None,
        importance=_importance(data.get("importance")),
        relations=_relations(data.get("relations")),
        model=config.model,
    )


class ClaudeCliClassifier:
    """The real classifier: one `claude` CLI invocation per drop (KTD4).

    Construction takes no I/O and the runner is injectable, so instantiating this in a test is
    harmless — only calling `classify` with the default runner reaches a subprocess.
    """

    def __init__(
        self,
        config: ClassifierConfig,
        *,
        runner: CommandRunner = run_command,
        areas: Sequence[AreaConfig] = (),
    ) -> None:
        self._config = config
        self._runner = runner
        # Held here rather than passed to `classify` so the `Classifier` protocol stays
        # `classify(record)`: the areas are a property of the vault being ingested, fixed for a
        # whole pass, and widening the seam would make every stub carry an argument it ignores.
        self._areas = tuple(areas)

    def argv(self) -> list[str]:
        """The command line, exposed so a caller can log or assert on it.

        Built from `classifier.preset` rather than here, so the four model-facing call sites
        cannot drift apart — see `presets`.
        """
        return build_argv(self._config)

    def classify(self, record: InboxRecord) -> Outcome:
        outcome = self._classify(record)
        if isinstance(outcome, NeedsReview):
            logger.info("%s: held for review — %s", record.filename, outcome.reason)
        return outcome

    def _classify(self, record: InboxRecord) -> Outcome:
        if record.unparseable:
            # Spending an LLM call on a drop that must be held either way is pure waste, and
            # the model would be reading metadata the parser already rejected.
            return NeedsReview(f"the drop's own metadata could not be read: {record.unparseable}")

        result = self._runner(
            self.argv(),
            prompt=build_prompt(record, self._areas),
            timeout=self._config.timeout_seconds,
        )

        if result.timed_out:
            return NeedsReview(
                f"the classifier did not answer within {self._config.timeout_seconds}s"
            )
        if result.exit_code != 0:
            detail = _flatten(result.stderr) or "no error output"
            return NeedsReview(f"the classifier exited {result.exit_code}: {detail}")

        reply, error = _reply_text(result.stdout, envelope_for(self._config))
        if error is not None or reply is None:
            return NeedsReview(error or "the classifier produced no usable reply")

        return _verdict(reply, self._config, record.declared)
