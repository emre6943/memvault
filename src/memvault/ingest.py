"""The ingestion pass: a full inbox in, vault content and one reviewable commit out.

This module composes U2-U5 and owns the two things none of them could own alone — the order
operations happen in, and what happens when one of them fails.

**The order is the safety property (KTD5).** For each drop: the vault write lands first, the
inbox entry is removed second, and both are staged into the same commit. A crash between the
two leaves the inbox entry present, so the next pass sees the drop again — and the id index
below is what makes that a no-op instead of a duplicate. The reverse order has no such
recovery: an inbox entry removed before the vault write is a memory that never existed.

**The id index is derived, never stored.** Before filing anything, the pass reads the
`content_id` out of the frontmatter of every filed file in the vault. That is R17's rule
applied to this pass: files are the truth, and a cache of which ids are filed would be a second
source of truth for exactly the fact whose wrongness duplicates memories. Deleting a cache is
free; discovering months later that one was stale is not. The scan reads the first few kilobytes
of each Markdown file once per pass, which is cheap enough that the cache buys nothing worth its
failure mode.

**Failure is contained per drop (R5).** One drop that cannot be classified, written, or read
holds only itself: it keeps its inbox file, gains a `<name>.needs-review` marker naming the
reason on one line, and is left out of the drain. The other nine file and commit normally, and
the run reports partial completion rather than failing. One bad drop must not block ten good
ones, and a stuck inbox is discovered today while a misfiled memory is discovered in six months.

**A dirty transcript tree stops the pass before it writes.** The commit is the review surface, so
it must contain the pass's work and nothing else. The guard is scoped to the tree this pass
actually writes into: a pathspec commit takes the working-tree content of the paths it names, so
uncommitted lines in a file the pass is about to write are the hazard worth refusing. Dirt
elsewhere in the vault — somebody's half-written note, an edited spec — is deliberately tolerated,
because the pass stages by exact path and cannot sweep it in. Uncommitted changes *inside the
inbox* are tolerated for a second reason: a hand-pasted `note.txt` that nobody committed is the
documented minimum drop.

The guard is consulted before the inbox is read, and what it found is reported either way. An
empty inbox is still a no-op — dirt cannot endanger a commit that will not happen — but under
idle-triggering (R7) most passes are empty passes, and the old order let a vault sit dirty for a
week while every log line said "nothing to ingest". `IngestReport.dirty` carries the finding and
`IngestReport.refused` distinguishes "somebody has uncommitted work open, come back later" from
"the pass is broken", which is what lets the debounce runner retry instead of alerting.

**Nothing is pushed.** That stays with the vault's own backup job, which keeps `local_only`
handling in one place (KTD7).

Two decisions the plan left to this unit, recorded here because both are load-bearing:

*A blocked work note does not hold the drop in the inbox.* When the raw filing succeeded and
only the distilled note failed — the leakage guard tripped, or the distiller errored — the
inbox entry is still drained. The contract's promise is that draining follows a successful
vault write, and one happened. Holding the drop instead would achieve nothing: the id index
would recognize it as filed on the next pass and drain it silently, quietly discarding the very
signal the marker was meant to raise. The failure surfaces in the report, the commit message,
and the exit status instead.

*A failed git command restores the inbox.* The removed drops are read into memory before they
are unlinked and written back if the commit fails, so the pass is retryable exactly as the plan
requires. The vault files it wrote stay where they are: they are already protected by the id
index, so the retry re-files nothing, and leaving them means the next pass finds a dirty vault
and refuses — which is the correct amount of noise for "a commit failed and a human should
look".
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from enum import StrEnum
from pathlib import Path

import yaml

from memvault.area import AreaChoice, choose_area, project_areas
from memvault.classify import Classification, Classifier, ClaudeCliClassifier, NeedsReview
from memvault.config import AreaConfig, Config, VaultConfig
from memvault.distill import ClaudeCliDistiller, Distiller, distill_record
from memvault.git_ops import CommitRef, GitError, commit, ensure_clean, stage, template_root
from memvault.inbox import InboxRecord, discover
from memvault.index import iter_vault_files
from memvault.notify import notify_failure
from memvault.route import Destination, DestinationKind, RoutingResult, route
from memvault.segment import (
    ClaudeCliSegmenter,
    Segment,
    Segmenter,
    segment_record,
)
from memvault.topic_note import write_topic_note
from memvault.writer import WriteError, WrittenFile, file_record

logger = logging.getLogger(__name__)

#: A held drop keeps its file and gains a sibling named for it. Every marker therefore ends in
#: an extension `inbox.read_drop` does not accept, so discovery skips markers on the next pass
#: without needing a rule of its own — the convention is self-enforcing.
MARKER_SUFFIX = ".needs-review"

#: What the id index scans. Only Markdown, because only Markdown is filed; the inbox is
#: excluded by `filed_ids` so a drop that happens to carry a `content_id` in its own
#: frontmatter cannot declare itself already filed.
FILED_SCAN_INCLUDE = ("**/*.md",)
FILED_SCAN_EXCLUDE = (".git/**", "node_modules/**")

#: Enough of a file to hold any plausible frontmatter block, and small enough that scanning a
#: whole vault stays a few milliseconds of reads.
FRONTMATTER_HEAD_CHARS = 4096

#: Where an unattended run is told to look. U10's launchd job owns the actual file; this is the
#: default the notification names when nobody said otherwise.
DEFAULT_LOG_PATH = "~/Library/Logs/memvault/ingest.log"

EXIT_OK = 0
EXIT_FAILED = 1
#: Distinct from 1 so a scheduler can tell "some drops need a human" from "the pass broke", and
#: distinct from 2, which argparse already spends on usage errors.
EXIT_PARTIAL = 3

#: How a failure reaches a human on an unattended run. Injectable so no test posts a
#: notification, and so a caller running interactively can pass None.
Notifier = Callable[[str, str, str], bool]


class IngestStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class HeldItem:
    """A drop that stayed in the inbox, and the marker that says why."""

    filename: str
    record_id: str
    reason: str
    marker: str | None = None


@dataclass(frozen=True)
class DrainedItem:
    """An inbox entry removed without being filed, because its content already was."""

    filename: str
    record_id: str
    reason: str


@dataclass(frozen=True)
class NoteFailure:
    """A work-vault note that was not written. The raw filing it came from stands."""

    filename: str
    record_id: str
    reason: str


@dataclass(frozen=True)
class IngestReport:
    """Everything one pass did, in the shape a commit message and an exit status need."""

    vault: str
    filed: tuple[WrittenFile, ...] = ()
    notes: tuple[WrittenFile, ...] = ()
    held: tuple[HeldItem, ...] = ()
    recovered: tuple[DrainedItem, ...] = ()
    note_failures: tuple[NoteFailure, ...] = ()
    skipped_notes: tuple[str, ...] = ()
    #: Drops that filed raw but chose no area, or whose topic note failed. Reported rather than
    #: warned about: choosing none is the designed answer when nothing fits.
    unrouted: tuple[str, ...] = ()
    commits: tuple[CommitRef, ...] = ()
    #: What the cleanliness guard found, whether or not it stopped the pass. Set on an empty
    #: inbox too: under idle-triggering most passes have nothing to file, and a vault that has
    #: been quietly dirty for a week would otherwise look identical to a healthy one in the log
    #: right up until the day a drop arrives and the pass suddenly refuses.
    dirty: str | None = None
    failure: str | None = None

    @property
    def refused(self) -> bool:
        """The pass declined to write because the tree was dirty.

        Different in kind from every other failure, and the difference is what R7's retry-later
        semantics rest on: nothing is broken, somebody simply has uncommitted work open, so the
        right response is to come back later rather than to report a failed pass.
        """
        return self.dirty is not None and self.status is IngestStatus.FAILED

    @property
    def status(self) -> IngestStatus:
        if self.failure is not None:
            return IngestStatus.FAILED
        if self.held or self.note_failures:
            return IngestStatus.PARTIAL
        return IngestStatus.OK

    @property
    def exit_code(self) -> int:
        return {
            IngestStatus.OK: EXIT_OK,
            IngestStatus.PARTIAL: EXIT_PARTIAL,
            IngestStatus.FAILED: EXIT_FAILED,
        }[self.status]

    def summary_line(self) -> str:
        """One line for a log, a notification, or the end of a terminal run."""
        parts = [
            f"{len(self.filed)} filed",
            f"{len(self.notes)} distilled",
            f"{len(self.held)} held",
        ]
        if self.recovered:
            parts.append(f"{len(self.recovered)} already filed")
        if self.note_failures:
            parts.append(f"{len(self.note_failures)} note(s) blocked")
        return ", ".join(parts)


def _reason_for(segment: Segment, reason: str) -> str:
    """Name the segment a review marker is about, when there is more than one.

    An unsplit drop is described exactly as it was before segmentation existed — no prefix at
    all. Saying "segment 1 of 1" would invite the reader to go looking for a part 2 that does
    not exist, and it would change the wording of every marker the vault has already collected.
    """
    if segment.is_whole_drop:
        return reason
    label = f"segment {segment.index} of {segment.count}"
    if segment.topic:
        label = f"{label} ({segment.topic})"
    return f"{label}: {reason}"


def _without_notes(routing: RoutingResult, source: str | None) -> RoutingResult:
    """Strip note destinations because this source's policy forbids them.

    Narrowing only, and applied after `route` rather than inside it: the routing table stays
    the single description of what the classification earns, and this is visibly a policy
    override on top of it. The reason is recorded in `notes` so a suppressed work note is
    explainable from the pass log rather than only from config.
    """
    kept: tuple[Destination, ...] = tuple(
        d for d in routing.destinations if d.kind is not DestinationKind.NOTE
    )
    if len(kept) == len(routing.destinations):
        return routing

    reason = (
        f"classified {routing.classification!r}, but source {source or 'unset'!r} is configured "
        "work_route: never; no distilled note crosses"
    )
    logger.info("%s", reason)
    return replace(routing, destinations=kept, notes=routing.notes + (reason,))


def marker_for(drop: Path) -> Path:
    """The review marker that belongs beside this drop."""
    return drop.with_name(drop.name + MARKER_SUFFIX)


def one_line(reason: str) -> str:
    """Reduce a reason to the single line a marker holds."""
    return " ".join(reason.split())


def _relative(path: Path, root: Path) -> str | None:
    """`path` as a posix path under `root`, or None when it is not under it at all."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def content_id_of(path: Path) -> str | None:
    """Read a filed file's `content_id`, or None when it carries none.

    Malformed frontmatter reads as "no id" rather than raising: a file this pass did not write
    is not the pass's problem, and refusing to run because one vault file has a broken header
    would make an unrelated edit block every future ingestion.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(FRONTMATTER_HEAD_CHARS)
    except OSError as exc:
        logger.debug("%s: could not be read while scanning for filed ids: %s", path, exc)
        return None

    lines = head.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    for index in range(1, len(lines)):
        if lines[index].strip() not in {"---", "..."}:
            continue
        try:
            parsed = yaml.safe_load("\n".join(lines[1:index]))
        except yaml.YAMLError:
            return None
        if not isinstance(parsed, dict):
            return None
        value = parsed.get("content_id")
        return value.strip() if isinstance(value, str) and value.strip() else None

    return None


def filed_ids(vault: VaultConfig) -> set[str]:
    """Every content id already filed in this vault, read from the files themselves.

    Rebuilt on every pass rather than cached. See the module docstring: this set is what makes
    a re-run a no-op, and a stale answer here duplicates memories.
    """
    excludes = list(FILED_SCAN_EXCLUDE)
    inbox = _relative(vault.inbox, vault.root)
    if inbox:
        excludes.append(f"{inbox}/**")

    found = {
        content_id
        for path, _ in iter_vault_files(vault.root, FILED_SCAN_INCLUDE, excludes)
        if (content_id := content_id_of(path)) is not None
    }
    logger.debug("vault %r: %d id(s) already filed", vault.name, len(found))
    return found


def _bullets(lines: Iterable[str]) -> list[str]:
    return [f"- {line}" for line in lines]


def build_note_message(report: IngestReport, scope: str) -> str:
    """The commit message for a vault that only received distilled notes.

    Deliberately narrow. The full ledger names filed transcripts by title and by path, and a
    transcript's path is its title slugged — so writing the ingest vault's message into the work
    vault's history would push `2026-08-01-acme-layoffs-call.md` across the boundary through the
    one surface the note guard never inspects. This message names only what already lives in the
    vault that will hold it.
    """
    notes = tuple(item for item in report.notes if item.vault == scope)
    subject = f"mem: ingest {len(notes)} distilled note(s) into {scope}"
    if not notes:
        return subject + "\n"
    body = f"Distilled {len(notes)} into {scope}:\n" + "\n".join(
        _bullets(item.summary_line() for item in notes)
    )
    return f"{subject}\n\n{body}\n"


def build_commit_message(report: IngestReport, scope: str | None = None) -> str:
    """Summarize a pass as a commit message: counts in the subject, titles in the body.

    The subject names what a reader of `git log` needs; the body names every file so the pass
    is reviewable without opening the diff. Held items are listed with their reasons, because
    the commit that drained nine drops is exactly where someone will look to find out about the
    tenth.

    `scope` is the vault the message is being written into. Any vault other than the one being
    ingested gets the narrow note-only message instead.
    """
    if scope is not None and scope != report.vault:
        return build_note_message(report, scope)

    filed, held, recovered = report.filed, report.held, report.recovered

    if filed:
        subject = f"mem: ingest {len(filed)} item(s) into {report.vault}"
    elif recovered:
        subject = f"mem: ingest — {len(recovered)} inbox entry(s) already filed in {report.vault}"
    elif held:
        subject = f"mem: ingest — {len(held)} item(s) held in the {report.vault} inbox"
    else:
        subject = f"mem: ingest {report.vault}"

    blocks: list[str] = []
    if filed:
        blocks.append(
            f"Filed {len(filed)} into {report.vault}:\n"
            + "\n".join(_bullets(item.summary_line() for item in filed))
        )
    if report.notes:
        vaults = ", ".join(sorted({item.vault for item in report.notes}))
        blocks.append(
            f"Distilled {len(report.notes)} into {vaults}:\n"
            + "\n".join(_bullets(item.summary_line() for item in report.notes))
        )
    if recovered:
        blocks.append(
            f"Already filed, inbox entry removed: {len(recovered)}\n"
            + "\n".join(_bullets(item.filename for item in recovered))
        )
    if held:
        blocks.append(
            f"Held in the inbox for review: {len(held)}\n"
            + "\n".join(_bullets(f"{item.filename} — {item.reason}" for item in held))
        )
    if report.note_failures:
        blocks.append(
            f"Work notes blocked: {len(report.note_failures)}\n"
            + "\n".join(
                _bullets(f"{item.filename} — {item.reason}" for item in report.note_failures)
            )
        )
    if report.unrouted:
        # Named, not counted silently: a run where nothing found an area is the signal that the
        # `when` hints need work, and the commit is where someone will notice.
        blocks.append(
            f"No area chosen, filed raw only: {len(report.unrouted)}\n"
            + "\n".join(_bullets(report.unrouted))
        )

    return "\n\n".join([subject, *blocks]) + "\n"


@dataclass
class _Removal:
    """One inbox file removed by this pass, kept in memory until the commit lands."""

    path: Path
    payload: bytes


@dataclass
class _Pass:
    """One ingestion run over one vault. Mutable by design — it is a ledger being filled."""

    config: Config
    vault: VaultConfig
    classifier: Classifier
    distiller: Distiller
    segmenter: Segmenter
    ingested: date

    filed: list[WrittenFile] = field(default_factory=list)
    notes: list[WrittenFile] = field(default_factory=list)
    held: list[HeldItem] = field(default_factory=list)
    recovered: list[DrainedItem] = field(default_factory=list)
    note_failures: list[NoteFailure] = field(default_factory=list)
    skipped_notes: list[str] = field(default_factory=list)
    unrouted: list[str] = field(default_factory=list)
    commits: list[CommitRef] = field(default_factory=list)
    failure: str | None = None

    dirty: str | None = None

    _areas: tuple[AreaConfig, ...] = ()
    _seen: set[str] = field(default_factory=set)
    _removals: list[_Removal] = field(default_factory=list)
    _vault_paths: list[str] = field(default_factory=list)
    _work_paths: list[str] = field(default_factory=list)

    @property
    def work(self) -> VaultConfig | None:
        return self.config.work_target(self.vault)

    def run(self) -> IngestReport:
        # The guard runs before the inbox is read, not after. Under idle-triggering the common
        # pass has an empty inbox, and the old order returned "nothing to ingest" without ever
        # asking whether the tree was clean — so a vault that would refuse the moment a drop
        # arrived logged exactly the same line as a healthy one, every time, for as long as the
        # dirt sat there. Learning that on the day it matters is a week too late.
        self.dirty = self._dirty_reason()

        records = discover(self.vault)
        if not records:
            if self.dirty is not None:
                logger.warning(
                    "vault %r: inbox is empty — nothing to ingest, but the next pass will refuse: "
                    "%s",
                    self.vault.name,
                    one_line(self.dirty),
                )
            else:
                logger.info("vault %r: inbox is empty — nothing to ingest", self.vault.name)
            return self.report()

        if self.dirty is not None:
            logger.error("vault %r: refusing to ingest — %s", self.vault.name, self.dirty)
            self.failure = self.dirty
            return self.report()

        self._seen = filed_ids(self.vault)
        # Once per pass, not per drop: deriving the project areas stats the memory tree, and a
        # pass with fifty drops must not walk it fifty times. Declared areas come first so a
        # vault that names an area after one of its own projects keeps what it wrote.
        self._areas = tuple(self.vault.areas) + project_areas(self.vault)

        for record in records:
            try:
                self._ingest(record)
            except Exception as exc:  # containment (R5): one bad drop holds only itself
                logger.exception("%s: ingestion failed", record.filename)
                self._hold(record, f"ingesting it raised {type(exc).__name__}: {exc}")

        self._commit()
        return self.report()

    def report(self) -> IngestReport:
        return IngestReport(
            vault=self.vault.name,
            filed=tuple(self.filed),
            notes=tuple(self.notes),
            held=tuple(self.held),
            recovered=tuple(self.recovered),
            note_failures=tuple(self.note_failures),
            skipped_notes=tuple(self.skipped_notes),
            # Deduplicated, order preserved: a segmented drop reaches `_ingest_segment` once per
            # segment, and a reader counting "no area chosen: 3" should learn that three drops
            # found no home, not that one drop was split three ways.
            unrouted=tuple(dict.fromkeys(self.unrouted)),
            commits=tuple(self.commits),
            dirty=self.dirty,
            failure=self.failure,
        )

    def _ignored(self, vault: VaultConfig) -> tuple[str, ...]:
        """Prefixes whose uncommitted changes do not block the pass.

        The inbox always qualifies. `ignore_dirty` adds paths owned by *other* automation in the
        same repo — a vault that is also a Claude Code memory store has files rewritten under it
        continuously and committed by its own nightly job, and a pass that refused to run
        whenever one of those was pending would never run at all. The pass still stages only
        the paths it wrote, so an ignored path cannot ride along into its commit.
        """
        inbox = _relative(vault.inbox, vault.root)
        prefixes = (inbox,) if inbox else ()
        return prefixes + vault.ignore_dirty

    def _dirty_reason(self) -> str | None:
        """What the cleanliness guard objects to, or None when it objects to nothing."""
        try:
            self._ensure_clean()
        except GitError as exc:
            return str(exc)
        return None

    def _ensure_clean(self) -> None:
        # Only the directory transcripts land in. The pass stages by exact path, so unrelated
        # work elsewhere in the vault cannot ride along — guarding the whole repo would refuse
        # the pass for reasons that have nothing to do with it.
        ensure_clean(
            self.vault.root,
            label=self.vault.name,
            ignore=self._ignored(self.vault),
            limit_to=(template_root(self.vault.transcript_template),),
        )
        work = self.work
        if work is not None and work.root != self.vault.root:
            # Only the directory notes land in. A route target is somebody's active working
            # tree — the personal pass has no standing to demand it be clean, and its own
            # commit there is one note staged by exact path.
            ensure_clean(
                work.root,
                label=work.name,
                ignore=self._ignored(work),
                limit_to=(template_root(work.note_template),),
            )

    def _ingest(self, record: InboxRecord) -> None:
        if record.path is None:
            logger.debug("%s: no inbox file to drain; skipped", record.filename)
            return

        if record.id in self._seen:
            reason = "its content is already filed in the vault"
            logger.info("%s: %s — removing the inbox entry", record.filename, reason)
            self._drain(record)
            self.recovered.append(DrainedItem(record.filename, record.id, reason))
            return

        source = record.declared.source
        segments = segment_record(
            record,
            should_segment=self.config.should_segment(source),
            config=self.config.segment,
            segmenter=self.segmenter,
        )
        if isinstance(segments, NeedsReview):
            self._hold(record, f"the drop could not be split: {segments.reason}")
            return

        # Every segment must land before the drop leaves the inbox (KTD3). A partial pass is
        # safe to re-run — content-hash identity means the already-filed segments are skipped
        # on the next attempt — but it must not look finished, so a single held segment leaves
        # the whole conversation visible in the inbox with a marker beside it.
        for segment in segments:
            if not self._ingest_segment(record, segment, source):
                return

        self._drain(record)

    def _ingest_segment(self, record: InboxRecord, segment: Segment, source: str | None) -> bool:
        """File one segment. Returns whether the drop may still be drained."""
        if segment.id in self._seen:
            logger.info(
                "%s: already filed; leaving it alone (%s)",
                record.filename,
                _reason_for(segment, "content id matches a file in the vault"),
            )
            return True

        # A record standing in for this segment, so `route`, `file_record`, and `distill_record`
        # need to know nothing about segmentation. Path is cleared because only the parent drop
        # is ever drained or marked, and a segment carrying one could remove the file its
        # siblings still need.
        piece = replace(record, body=segment.body, id=segment.id, path=None)

        outcome = self.classifier.classify(piece)
        if isinstance(outcome, NeedsReview):
            self._hold(record, _reason_for(segment, outcome.reason))
            return False

        threshold = self.config.confidence_threshold_for(source)
        if outcome.confidence < threshold:
            self._hold(
                record,
                _reason_for(
                    segment,
                    f"confidence {outcome.confidence:.2f} is below the threshold "
                    f"{threshold:.2f} configured for source {source or 'unset'!r}",
                ),
            )
            return False

        area = choose_area(declared=piece.declared, verdict=outcome, areas=self._areas)
        if area is None and self._areas:
            # Only worth reporting when there was something to choose from. A vault that
            # declares no areas is not "unrouted", it simply does not route.
            self.unrouted.append(record.filename)

        routing = route(piece, outcome, self.config, self.vault, area=area)
        if routing.held:
            self._hold(record, routing.review or "routing declined to place this item")
            return False

        if not self.config.allows_work_note(source):
            routing = _without_notes(routing, source)

        try:
            written = file_record(
                piece,
                outcome,
                routing,
                self.vault.root,
                ingested=self.ingested,
                segment=segment,
            )
        except WriteError as exc:
            self._hold(record, _reason_for(segment, f"the vault write failed: {exc}"))
            return False

        self.filed.append(written)
        self._track(self._vault_paths, written.path, self.vault.root)
        self._seen.add(segment.id)

        self._write_topic_note(piece, outcome, routing, area, written)
        self._distil(piece, outcome, routing)
        return True

    def _write_topic_note(
        self,
        record: InboxRecord,
        verdict: Classification,
        routing: RoutingResult,
        area: AreaChoice | None,
        raw: WrittenFile,
    ) -> None:
        """File the topic note beside the raw transcript, if one was owed.

        Additive by design: the transcript is already filed and the drop already drains, so a
        failure here costs the note and nothing else. It is reported as unrouted rather than
        held — the material is in the vault either way.
        """
        if area is None:
            return
        destinations = [
            d for d in routing.of_kind(DestinationKind.NOTE) if d.vault == self.vault.name
        ]
        if not destinations:
            return

        try:
            note = write_topic_note(
                record=record,
                verdict=verdict,
                destination=destinations[0],
                vault=self.vault,
                raw_path=raw.relative_path,
                area=area,
                ingested=self.ingested,
            )
        except (WriteError, OSError) as exc:
            logger.warning("%s: topic note failed — %s", record.filename, one_line(str(exc)))
            self.unrouted.append(record.filename)
            return

        self.filed.append(note)
        self._track(self._vault_paths, note.path, self.vault.root)

    def _distil(self, record: InboxRecord, verdict: Classification, routing: RoutingResult) -> None:
        work = self.work
        if work is None:
            return
        # Scoped by vault, not by kind: a drop can now owe two notes — one to a topic area in
        # this vault, one crossing into the work vault — and only the second is the distiller's.
        # `_note_destination` refuses to guess between them, so naming it here is what keeps a
        # `work` drop with an area from failing outright.
        crossing = [d for d in routing.of_kind(DestinationKind.NOTE) if d.vault == work.name]
        if not crossing:
            return

        try:
            result = distill_record(
                record,
                verdict,
                routing,
                work.root,
                distiller=self.distiller,
                destination=crossing[0],
                ingested=self.ingested,
                shingle_size=self.config.distill.shingle_size,
                overlap_threshold=self.config.distill.overlap_threshold,
            )
        except (WriteError, OSError) as exc:
            self.note_failures.append(
                NoteFailure(
                    record.filename, record.id, f"the note write failed: {one_line(str(exc))}"
                )
            )
            return

        if result.written is not None:
            self.notes.append(result.written)
            self._track(self._work_paths, result.written.path, work.root)
        elif result.review is not None:
            self.note_failures.append(
                NoteFailure(record.filename, record.id, one_line(result.review))
            )
        elif result.skipped is not None:
            self.skipped_notes.append(f"{record.filename}: {one_line(result.skipped)}")

    def _hold(self, record: InboxRecord, reason: str) -> None:
        """Leave the drop where it is and say why, on one line, in a file beside it."""
        single = one_line(reason)
        marker: str | None = None

        if record.path is not None:
            path = marker_for(record.path)
            try:
                path.write_text(single + "\n", encoding="utf-8")
            except OSError as exc:
                logger.warning("%s: could not write a review marker: %s", record.filename, exc)
            else:
                marker = _relative(path, self.vault.inbox) or path.name
                self._track(self._vault_paths, path, self.vault.root)

        logger.info("%s: held in the inbox — %s", record.filename, single)
        self.held.append(HeldItem(record.filename, record.id, single, marker))

    def _drain(self, record: InboxRecord) -> None:
        """Remove the inbox entry and any stale marker, keeping both restorable."""
        drop = record.path
        if drop is None:
            return

        for path in (marker_for(drop), drop):
            if not path.exists():
                continue
            self._removals.append(_Removal(path, path.read_bytes()))
            path.unlink()
            self._track(self._vault_paths, path, self.vault.root)

        _prune_empty(drop.parent, self.vault.inbox)

    def _track(self, bucket: list[str], path: Path, root: Path) -> None:
        relative = _relative(path, root)
        if relative is None:
            # An inbox configured outside its vault cannot be committed with it. Filing still
            # works; the drain simply is not part of the audit trail, which is the operator's
            # choice to have made.
            logger.debug("%s is outside %s and will not be committed", path, root)
            return
        bucket.append(relative)

    def _repos(self) -> list[tuple[Path, str, list[str]]]:
        """The repositories this pass touched, in the order they were written to."""
        work = self.work
        groups: dict[Path, tuple[str, list[str]]] = {}
        candidates = [(self.vault.root, self.vault.name, self._vault_paths)]
        if work is not None:
            candidates.append((work.root, work.name, self._work_paths))

        for root, name, paths in candidates:
            if not paths:
                continue
            _, collected = groups.setdefault(root, (name, []))
            collected.extend(paths)
        return [(root, name, paths) for root, (name, paths) in groups.items()]

    def _commit(self) -> None:
        report = self.report()
        try:
            for root, name, paths in self._repos():
                staged = stage(root, paths)
                reference = commit(root, build_commit_message(report, name), staged, vault=name)
                if reference is not None:
                    self.commits.append(reference)
        except GitError as exc:
            logger.error("vault %r: commit failed — %s", self.vault.name, exc)
            self.failure = one_line(str(exc))
            self._restore()

    def _restore(self) -> None:
        """Put back every inbox entry this pass removed, so the run can simply be re-run."""
        for removal in reversed(self._removals):
            try:
                removal.path.parent.mkdir(parents=True, exist_ok=True)
                removal.path.write_bytes(removal.payload)
            except OSError as exc:
                logger.error("could not restore inbox entry %s: %s", removal.path, exc)
        if self._removals:
            logger.info("restored %d inbox entry(s) after the failed commit", len(self._removals))
        self._removals.clear()


def _prune_empty(directory: Path, stop: Path) -> None:
    """Remove directories left empty by a drain, up to but never including the inbox itself."""
    current = directory
    while current != stop and stop in current.parents:
        try:
            if any(current.iterdir()):
                return
            current.rmdir()
        except OSError:
            return
        current = current.parent


def run_ingest(
    config: Config,
    vault: VaultConfig,
    *,
    classifier: Classifier | None = None,
    distiller: Distiller | None = None,
    segmenter: Segmenter | None = None,
    ingested: date | None = None,
    notifier: Notifier | None = notify_failure,
    log_path: str = DEFAULT_LOG_PATH,
) -> IngestReport:
    """Run one ingestion pass and report what it did.

    The classifier and the distiller are injected rather than constructed unconditionally, so a
    test never spawns a subprocess and a caller can swap the model tier. Both defaults are
    I/O-free to construct — only calling them reaches the CLI.
    """
    # The classifier is told the areas at construction so the `Classifier` protocol stays
    # `classify(record)`. Derived once here, and again inside the pass for selection — both
    # cheap, and threading one list through two layers would couple them for no gain.
    areas = tuple(vault.areas) + project_areas(vault)
    classifier = classifier or ClaudeCliClassifier(config.classifier, areas=areas)
    distiller = distiller or ClaudeCliDistiller(config.classifier)
    segmenter = segmenter or ClaudeCliSegmenter(config.classifier, config.segment)

    if isinstance(classifier, ClaudeCliClassifier):
        logger.debug("vault %r: classifying with `%s`", vault.name, " ".join(classifier.argv()))

    report = _Pass(
        config=config,
        vault=vault,
        classifier=classifier,
        distiller=distiller,
        segmenter=segmenter,
        ingested=ingested or date.today(),
    ).run()

    logger.info("vault %r: %s", vault.name, report.summary_line())

    if notifier is not None and report.status is not IngestStatus.OK:
        notifier("ingest", _failure_message(report), log_path)

    return report


def _failure_message(report: IngestReport) -> str:
    """One line for the notification: what went wrong, not the whole ledger."""
    if report.failure is not None:
        return report.failure
    reasons: Sequence[str] = [item.reason for item in report.held] + [
        item.reason for item in report.note_failures
    ]
    count = len(reasons)
    lead = f"{count} item(s) need review"
    return f"{lead}: {reasons[0]}" if count == 1 else lead
