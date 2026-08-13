"""The weekly reflection pass: a window of raw material in, durable memory and a digest out.

Ingestion decides *where* a memory lives. This pass decides *what was worth remembering*, and
it is the only part of the system that writes into files a human also writes into by hand. That
one fact sets every rule below.

**Appending is the whole contract (R13).** `learnings.md`, `quirks.md`, and `reflect-log.md`
hold history that exists nowhere else — no export, no backup that predates the mistake, no way
to tell from a diff that a rewrite lost something rather than reformatted it. So this module
never rewrites a memory file. It reads the existing bytes, adds to them, and writes the result;
every test that matters here asserts the old content survives byte-for-byte as a prefix. The one
exception is `reflect-log.md`, which is newest-first by the vault's own convention, so its entry
is inserted directly below the format block at the top — still an insertion, still nothing
overwritten.

**Duplicates are prevented by reading, not by remembering.** Before appending, the file's own
headings are read and a heading already present is skipped. That is what makes two passes over
the same window a no-op, and it needs no state file to be right — the same reasoning that keeps
U6's id index derived rather than cached.

**The digest is not memory.** It is a rendered view of one window, so it is written whole and
overwritten on a re-run. Memory appends and dedupes; a derived artifact of a fixed window has
exactly one correct content and rewriting it loses nothing.

**The reflection note is memory, and it is the one thing here written to be read back by
search (R6).** The digest reports what happened; the note answers questions the window raised,
and every answer carries `- cites [[path]]` lines pointing at the files it was drawn from — so
a recall hit on the note is a lead to the material, not a claim to be taken on faith. Two rules
keep that from turning into a loop. Every citation must name a file in *this* window's
`ReflectionMaterial.filed`, so a model cannot cite a path it invented; and `filed_in_window`
refuses `kind: reflection` outright, so week N's note can never become week N+1's input. The
second rule is belt and braces — the note carries no `content_id` and would be skipped anyway —
but a vault reflecting on its own reflections degrades silently rather than loudly, and one
misplaced key would be enough to start it.

**Material comes from two places and one of them is optional.** The vault's own filed
transcripts are authoritative and always available. claude-mem is a third-party database that
may be missing, locked, or schema-shifted, so `claude_mem.fetch_window` degrades to a warning
and the pass runs on vault material alone (R12). It reads only projects the vault's alias map
names — see `claude_mem` for why that is a boundary control and not a filter. A vault that
never enabled the source at all is a different case and costs nothing: no warning, no PARTIAL
(R11), because a plugin the adopter does not run is not a degradation of anything.

**The curator is injected, exactly as the classifier and distiller are.** No test spawns a
subprocess, and the model tier stays swappable. A curator that fails writes nothing at all: a
half-curated week is worse than an un-curated one, because the missing half looks like an
absence of news.

**Every pass commits (R15).** The diff is the review surface for automated memory changes, and
an append-only diff is the one shape a reviewer can check at a glance.

Three decisions the plan left open, resolved here:

*The window is a date range, defaulted rather than remembered.* Seven days ending today, with
`--since` / `--until` overrides. A "last run" marker would be a second source of truth about
what has been reflected on, and the heading-dedupe above already makes an overlapping window
harmless — which is the property a marker would have been protecting.

*The reflect-log entry lands in every project the pass appended to*, not in one central log.
That matches the vault, where a project's `reflect-log.md` records the reflections that touched
it, and it keeps the log useful when reading one project's directory in isolation.

*A skipped project is reported but does not make the run partial.* The claude-mem database
always holds work projects for a personal vault, so treating every skip as a problem would make
every run look degraded and train the eye to ignore the status. Only a claude-mem read that
failed outright does that, because it means the pass reflected on less than it should have.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

import yaml

# Borrowed rather than reimplemented, for the same reason `distill` borrows from `classify`:
# both LLM calls unwrap the same CLI envelope, and a second copy would be free to drift.
from memvault.classify import CommandRunner, Relation, _reply_text, run_command
from memvault.claude_mem import (
    Observation,
    ObservationWindow,
    SessionSummary,
    fetch_window,
)
from memvault.config import ClassifierConfig, Config, VaultConfig
from memvault.git_ops import CommitRef, GitError, commit, ensure_clean, stage, template_root
from memvault.index import iter_vault_files

# Exit codes are shared with the ingestion pass deliberately: one scheduler reads both, and 3
# must mean "a human should look" in each of them.
from memvault.ingest import EXIT_FAILED, EXIT_OK, EXIT_PARTIAL, one_line
from memvault.notify import notify_failure
from memvault.presets import build_argv, envelope_for
from memvault.presets import flatten as _flatten

# The kind the note declares, imported from the module that reads it rather than re-spelled
# here: recall filters `kind: reflection` out of its graph sources precisely because these notes
# cite everything, and two string literals a directory apart would be free to drift.
from memvault.recall import REFLECTION_KIND
from memvault.writer import _atomic_write, render_document, render_path

logger = logging.getLogger(__name__)

#: Default window length. Weekly, matching the launchd job U10 installs and the cadence the
#: vault's own `/reflect` was run at by hand.
DEFAULT_WINDOW_DAYS = 7

#: Where an unattended run is told to look, named in the failure notification.
DEFAULT_LOG_PATH = "~/Library/Logs/memvault/reflect.log"

#: What the filed-material scan covers. Only Markdown is filed, and only files carrying a
#: `content_id` are treated as filed material — which is what keeps memory files, digests, and
#: hand-written vault notes out of the pass's own input.
FILED_SCAN_INCLUDE = ("**/*.md",)
FILED_SCAN_EXCLUDE = (".git/**", "node_modules/**")

#: Enough of a file to hold any plausible frontmatter block.
FRONTMATTER_HEAD_CHARS = 4096

#: Caps on how much material is rendered into one prompt. A quiet week is far under them; a
#: month-long catch-up window is not, and an unbounded prompt would fail the call outright
#: instead of reflecting on the most recent material.
MAX_PROMPT_FILED = 60
MAX_PROMPT_OBSERVATIONS = 200
MAX_PROMPT_SUMMARIES = 60

#: How a reflection note points back at what it was drawn from: a typed relation line the
#: indexer parses into a graph edge, not a bare wikilink it would ignore.
CITATION_PREDICATE = "cites"

#: How many questions one note may carry. A reflection that poses twenty questions has posed
#: none — nobody reads to the bottom of it — and the prompt asks for a handful. Excess is
#: dropped with a log rather than held, matching how the classifier treats an over-long reply.
MAX_REFLECTION_QUESTIONS = 8

#: The three files a project's memory directory holds, plus its README.
LEARNINGS_FILE = "learnings.md"
QUIRKS_FILE = "quirks.md"
REFLECT_LOG_FILE = "reflect-log.md"
README_FILE = "README.md"

Notifier = Callable[[str, str, str], bool]

PROMPT_TEMPLATE = """\
You are the weekly curator for a personal memory vault. You are given everything that happened
in one window: material filed into the vault, and observations recorded by the assistant during
work sessions. Your job is to decide what is worth remembering months from now, and to write a
digest of the window.

Be ruthless. Most of a week is not worth a permanent entry. A learning is durable only if it
would still be useful on a different project; a quirk is worth recording only if it surprised
someone and would surprise them again. Ten entries a week means none of them will be read.
Zero entries is a legitimate answer for a quiet week.

Write in the first person, past tense, plainly. Do not quote the source material verbatim; say
what it means.

Reply with a single JSON object and nothing else: no preamble, no explanation, no Markdown code
fence. A reply that is not parseable JSON is discarded and nothing is written.

Fields:
  summary       one or two sentences describing the window as a whole
  learnings     list of objects, each:
                  project    which project it belongs to, from the projects listed below
                  claim      a one-line summary stated as a claim, not a topic
                  context    where this came from, e.g. a filed path or a session subject
                  takeaway   what is now known, in one or two sentences
  quirks        list of objects, each:
                  project    which project it belongs to
                  behavior   the surprising behavior, as a short heading
                  symptom    what looks wrong
                  cause      what is actually happening
                  fix        how to get past it, or null when there is none yet
                  context    where this came from
  projects      list of objects, each:
                  project    the project name
                  bullets    2-6 short lines on what shipped, broke, or was decided
  themes        list of short lines: patterns across projects this window
  open_threads  list of short lines: things that look genuinely unresolved
  reflection    list of objects — the questions this window raises, answered. Each:
                  question   a question you pose yourself about this window, in one line
                  answer     the answer, in one to three sentences
                  citations  list of the exact vault paths listed under "Filed into the vault
                             this window" that support the answer

Ask the questions someone re-reading this window in six months would want answered: what
changed, what is still open, what pattern is forming. Three to five is plenty; fewer is fine.

Every answer must cite at least one path, copied exactly from the filed list — an answer you
cannot cite is dropped, and a path that is not on that list is dropped with it. Do not cite a
path you were not shown, and do not invent one that looks plausible.

Window: {since} to {until}
Projects in scope: {projects}

{material}"""


class ReflectStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class FiledFile:
    """One vault file filed inside the window, described from its own frontmatter."""

    relative_path: str
    title: str
    day: str
    project: str | None = None
    classification: str | None = None
    summary: str = ""
    tags: tuple[str, ...] = ()

    def prompt_line(self) -> str:
        head = f"- {self.day} {self.relative_path} — {self.title}"
        if self.project:
            head += f" [{self.project}]"
        return head + (f"\n  {self.summary}" if self.summary else "")


@dataclass(frozen=True)
class ReflectionMaterial:
    """Everything one window offered, in the shape the curator is handed.

    `projects` is the union of the projects named by filed material and by claude-mem, and it
    is what the prompt offers the curator to choose from. A curator that invents a project
    outside it is writing into a directory nobody asked for, which `_Pass` refuses.
    """

    vault: str
    since: date
    until: date
    filed: tuple[FiledFile, ...] = ()
    observations: tuple[Observation, ...] = ()
    summaries: tuple[SessionSummary, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.filed or self.observations or self.summaries)

    @property
    def projects(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for item in self.filed:
            if item.project:
                seen.setdefault(item.project, None)
        for observation in self.observations:
            seen.setdefault(observation.project, None)
        for summary in self.summaries:
            seen.setdefault(summary.project, None)
        return tuple(seen)

    def counts_line(self) -> str:
        return (
            f"{len(self.filed)} filed transcript(s), {len(self.observations)} observation(s), "
            f"{len(self.summaries)} session summary(s)"
        )


@dataclass(frozen=True)
class LearningEntry:
    """A proposed append to a project's `learnings.md`."""

    project: str
    claim: str
    takeaway: str
    context: str = ""

    def heading(self, day: date) -> str:
        return f"## {day.isoformat()} — {self.claim}"

    def render(self, day: date, fallback_context: str) -> str:
        return "\n".join(
            [
                self.heading(day),
                f"- Context: {self.context or fallback_context}",
                f"- Takeaway: {self.takeaway}",
            ]
        )


@dataclass(frozen=True)
class QuirkEntry:
    """A proposed append to a project's `quirks.md`.

    No date in the heading, matching the vault: a quirk is a standing property of the thing,
    and when it was first met belongs in `First seen` where it can be read without being
    mistaken for the date it last bit someone.
    """

    project: str
    behavior: str
    symptom: str
    cause: str
    fix: str | None = None
    context: str = ""

    def heading(self) -> str:
        return f"## {self.behavior}"

    def render(self, day: date, fallback_context: str) -> str:
        lines = [self.heading(), f"- Symptom: {self.symptom}", f"- Cause: {self.cause}"]
        if self.fix:
            lines.append(f"- Fix: {self.fix}")
        lines.append(f"- First seen: {day.isoformat()} ({self.context or fallback_context})")
        return "\n".join(lines)


@dataclass(frozen=True)
class ReflectionQuestion:
    """One self-posed question, its answer, and the files the answer rests on.

    `citations` are vault-relative paths. They are not trusted as given: `cited_questions`
    keeps only the ones the window actually filed, because a citation is the whole value of the
    note and an unverifiable one is worse than none — it reads exactly like a real one.
    """

    question: str
    answer: str
    citations: tuple[str, ...] = ()

    def heading(self) -> str:
        return f"## {self.question}"

    def render(self) -> str:
        lines = [self.heading(), "", self.answer, ""]
        lines += [
            Relation(predicate=CITATION_PREDICATE, target=path).render() for path in self.citations
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class ProjectDigest:
    """One project's section of the digest."""

    project: str
    bullets: tuple[str, ...] = ()


@dataclass(frozen=True)
class Curation:
    """What the curator proposed, before anything has been written."""

    summary: str = ""
    learnings: tuple[LearningEntry, ...] = ()
    quirks: tuple[QuirkEntry, ...] = ()
    projects: tuple[ProjectDigest, ...] = ()
    themes: tuple[str, ...] = ()
    open_threads: tuple[str, ...] = ()
    reflection: tuple[ReflectionQuestion, ...] = ()
    model: str | None = None

    @property
    def empty(self) -> bool:
        """Whether there is nothing here worth writing down.

        A summary alone is not content, for the same reason a distillation's title alone is
        not: a model asked to describe a quiet week will describe it rather than admit there
        was nothing in it. Answered questions *are* content, though nothing else survived the
        curator's own filter — a week whose only durable output is "here is what changed and
        here is where to read it" is a week worth writing the note for.
        """
        return not (
            self.learnings or self.quirks or self.projects or self.themes or self.reflection
        )


@dataclass(frozen=True)
class CurationFailed:
    """Not a curation: the reason nothing can be written from this window."""

    reason: str


#: What a curator returns. As in `classify` and `distill`, there is no third state.
CurationOutcome: TypeAlias = Curation | CurationFailed


@runtime_checkable
class Curator(Protocol):
    """The seam. Tests inject stubs through it; no test spawns a subprocess."""

    def curate(self, material: ReflectionMaterial) -> CurationOutcome: ...


@dataclass(frozen=True)
class AppendedEntry:
    """One block this pass added to a memory file."""

    project: str
    kind: str
    relative_path: str
    heading: str

    def summary_line(self) -> str:
        return f"{self.project} — {self.heading.lstrip('# ').strip()}"


@dataclass(frozen=True)
class ReflectReport:
    """Everything one pass did, in the shape a commit message and an exit status need."""

    vault: str
    since: date
    until: date
    material: ReflectionMaterial | None = None
    appended: tuple[AppendedEntry, ...] = ()
    duplicates: tuple[AppendedEntry, ...] = ()
    created_projects: tuple[str, ...] = ()
    unknown_projects: tuple[str, ...] = ()
    skipped_projects: tuple[str, ...] = ()
    digest_path: str | None = None
    reflection_path: str | None = None
    warnings: tuple[str, ...] = ()
    commits: tuple[CommitRef, ...] = ()
    failure: str | None = None

    @property
    def status(self) -> ReflectStatus:
        if self.failure is not None:
            return ReflectStatus.FAILED
        if self.warnings:
            return ReflectStatus.PARTIAL
        return ReflectStatus.OK

    @property
    def exit_code(self) -> int:
        return {
            ReflectStatus.OK: EXIT_OK,
            ReflectStatus.PARTIAL: EXIT_PARTIAL,
            ReflectStatus.FAILED: EXIT_FAILED,
        }[self.status]

    def counts(self, kind: str) -> int:
        return sum(1 for item in self.appended if item.kind == kind)

    def summary_line(self) -> str:
        parts = [
            f"{self.counts('learning')} learning(s)",
            f"{self.counts('quirk')} quirk(s)",
        ]
        if self.duplicates:
            parts.append(f"{len(self.duplicates)} already present")
        if self.created_projects:
            parts.append(f"{len(self.created_projects)} project dir(s) created")
        parts.append("digest written" if self.digest_path else "no digest")
        parts.append("reflection written" if self.reflection_path else "no reflection")
        return f"{self.since.isoformat()}..{self.until.isoformat()}: " + ", ".join(parts)


# --------------------------------------------------------------------------------------
# Reading the window's material out of the vault
# --------------------------------------------------------------------------------------


def read_frontmatter(path: Path) -> dict[str, Any] | None:
    """Parse a vault file's frontmatter, or None when it has none that parses.

    Reads the head of the file rather than all of it: a vault of transcripts is scanned once
    per pass and a transcript can be long. Malformed frontmatter reads as absent rather than
    raising, for the reason `ingest.content_id_of` gives — a file this engine did not write
    must not be able to stop a pass.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(FRONTMATTER_HEAD_CHARS)
    except OSError as exc:
        logger.debug("%s: could not be read while gathering material: %s", path, exc)
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
        return parsed if isinstance(parsed, dict) else None

    return None


def _as_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def filed_in_window(vault: VaultConfig, since: date, until: date) -> tuple[FiledFile, ...]:
    """Every file this engine filed into the vault within the window.

    Selected by the presence of a `content_id`, which only U4 and U5 write. That is what keeps
    memory files, digests, and hand-written vault notes out of the material a pass reflects on
    — otherwise last week's digest would become next week's input and the vault would slowly
    start reflecting on itself.

    `kind: reflection` is refused on top of that, and deliberately redundantly: this pass's own
    note carries no `content_id` and is already excluded, but it is the one file in the vault
    written *by* the pass in the shape of the material it reads, so the loop it would close is
    the expensive one. Two independent reasons to skip it cost one line.
    """
    excludes = list(FILED_SCAN_EXCLUDE)
    try:
        inbox = vault.inbox.relative_to(vault.root).as_posix()
    except ValueError:
        inbox = ""
    if inbox:
        excludes.append(f"{inbox}/**")

    found: list[FiledFile] = []
    for path, relative in iter_vault_files(vault.root, FILED_SCAN_INCLUDE, excludes):
        frontmatter = read_frontmatter(path)
        if not frontmatter or not str(frontmatter.get("content_id") or "").strip():
            continue
        if str(frontmatter.get("kind") or "").strip() == REFLECTION_KIND:
            continue

        day = _as_date(frontmatter.get("ingested")) or _as_date(frontmatter.get("date"))
        if day is None or not since <= day <= until:
            continue

        project = frontmatter.get("project")
        found.append(
            FiledFile(
                relative_path=relative,
                title=str(frontmatter.get("title") or path.stem),
                day=day.isoformat(),
                project=str(project).strip() if project else None,
                classification=(
                    str(frontmatter["classification"]).strip()
                    if frontmatter.get("classification")
                    else None
                ),
                summary=str(frontmatter.get("summary") or "").strip(),
                tags=_string_tuple(frontmatter.get("tags")),
            )
        )

    return tuple(sorted(found, key=lambda item: (item.day, item.relative_path)))


def gather(
    vault: VaultConfig,
    since: date,
    until: date,
    *,
    db_path: str | Path | None = None,
) -> tuple[ReflectionMaterial, ObservationWindow]:
    """Collect the window's material from both sources, tolerating the loss of one."""
    window = fetch_window(vault, since, until, db_path=db_path)
    material = ReflectionMaterial(
        vault=vault.name,
        since=since,
        until=until,
        filed=filed_in_window(vault, since, until),
        observations=window.observations,
        summaries=window.summaries,
    )
    return material, window


# --------------------------------------------------------------------------------------
# The curator
# --------------------------------------------------------------------------------------


def _truncated(items: Sequence[str], cap: int, label: str) -> list[str]:
    if len(items) <= cap:
        return list(items)
    return [*items[:cap], f"  (+{len(items) - cap} more {label} not shown)"]


def render_material(material: ReflectionMaterial) -> str:
    """The material block of the prompt: what happened, as lines a model can read."""
    blocks: list[str] = []

    if material.filed:
        lines = _truncated(
            [item.prompt_line() for item in material.filed], MAX_PROMPT_FILED, "filed item(s)"
        )
        blocks.append("Filed into the vault this window:\n" + "\n".join(lines))

    if material.observations:
        rendered = []
        for observation in material.observations:
            line = f"- {observation.summary_line()}"
            if observation.facts:
                line += "\n  " + "\n  ".join(observation.facts)
            rendered.append(line)
        lines = _truncated(rendered, MAX_PROMPT_OBSERVATIONS, "observation(s)")
        blocks.append("Session observations:\n" + "\n".join(lines))

    if material.summaries:
        rendered = []
        for summary in material.summaries:
            parts = [f"- {summary.project}: {summary.request}"]
            for label, value in (
                ("learned", summary.learned),
                ("completed", summary.completed),
                ("next", summary.next_steps),
            ):
                if value:
                    parts.append(f"  {label}: {value}")
            rendered.append("\n".join(parts))
        lines = _truncated(rendered, MAX_PROMPT_SUMMARIES, "session summary(s)")
        blocks.append("Session summaries:\n" + "\n".join(lines))

    return "\n\n".join(blocks) if blocks else "Nothing happened in this window."


def build_curation_prompt(material: ReflectionMaterial) -> str:
    """Render the curation prompt for one window. Pure."""
    return PROMPT_TEMPLATE.format(
        since=material.since.isoformat(),
        until=material.until.isoformat(),
        projects=", ".join(material.projects) or "(none named)",
        material=render_material(material),
    )


def _entry_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    return one_line(str(value)) if value not in (None, "") else ""


def _learnings(raw: object) -> tuple[LearningEntry, ...]:
    if not isinstance(raw, list):
        return ()
    entries = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        project, claim = _entry_text(item, "project"), _entry_text(item, "claim")
        takeaway = _entry_text(item, "takeaway")
        if not (project and claim and takeaway):
            logger.info("curator: dropped a learning missing project, claim, or takeaway")
            continue
        entries.append(
            LearningEntry(
                project=project,
                claim=claim,
                takeaway=takeaway,
                context=_entry_text(item, "context"),
            )
        )
    return tuple(entries)


def _quirks(raw: object) -> tuple[QuirkEntry, ...]:
    if not isinstance(raw, list):
        return ()
    entries = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        project, behavior = _entry_text(item, "project"), _entry_text(item, "behavior")
        symptom, cause = _entry_text(item, "symptom"), _entry_text(item, "cause")
        if not (project and behavior and symptom and cause):
            logger.info("curator: dropped a quirk missing project, behavior, symptom, or cause")
            continue
        entries.append(
            QuirkEntry(
                project=project,
                behavior=behavior,
                symptom=symptom,
                cause=cause,
                fix=_entry_text(item, "fix") or None,
                context=_entry_text(item, "context"),
            )
        )
    return tuple(entries)


def _project_digests(raw: object) -> tuple[ProjectDigest, ...]:
    if not isinstance(raw, list):
        return ()
    digests = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        project = _entry_text(item, "project")
        bullets = tuple(one_line(line) for line in _string_tuple(item.get("bullets")))
        if project and bullets:
            digests.append(ProjectDigest(project=project, bullets=bullets))
    return tuple(digests)


def _reflection(raw: object) -> tuple[ReflectionQuestion, ...]:
    """Read the reply's self-posed questions, dropping any that cannot stand as one.

    A question with no answer, or an answer with no citation at all, is discarded here rather
    than written and marked incomplete: the note exists to be trusted at a glance, and an
    uncited claim inside it is indistinguishable from a cited one once it is on the page.
    Whether the cited paths are *real* is a separate question, answered by `cited_questions`
    against the window that was actually read.
    """
    if not isinstance(raw, list):
        return ()

    entries: list[ReflectionQuestion] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        question, answer = _entry_text(item, "question"), _entry_text(item, "answer")
        citations = tuple(dict.fromkeys(_string_tuple(item.get("citations"))))
        if not (question and answer and citations):
            logger.info("curator: dropped a reflection entry missing question, answer, or citation")
            continue
        entries.append(ReflectionQuestion(question=question, answer=answer, citations=citations))

    if len(entries) > MAX_REFLECTION_QUESTIONS:
        logger.info(
            "curator: keeping %d of %d reflection question(s)",
            MAX_REFLECTION_QUESTIONS,
            len(entries),
        )
    return tuple(entries[:MAX_REFLECTION_QUESTIONS])


def cited_questions(
    questions: Sequence[ReflectionQuestion], material: ReflectionMaterial
) -> tuple[ReflectionQuestion, ...]:
    """Keep only what this window can vouch for: real citations, and answers that kept one.

    The filed set is the whole authority. A model handed sixty paths and asked for citations
    will occasionally return a sixty-first that looks exactly like the others — same directory,
    same date shape — and a reader has no way to tell it apart once it is written down. Pure,
    so the rule is testable without running a pass.
    """
    known = {item.relative_path for item in material.filed}
    kept: list[ReflectionQuestion] = []
    for entry in questions:
        citations = tuple(path for path in entry.citations if path in known)
        for missing in [path for path in entry.citations if path not in known]:
            logger.info(
                "curator: dropped a citation to %r, which this window did not file", missing
            )
        if not citations:
            logger.info("curator: dropped the answer to %r — no citation survived", entry.question)
            continue
        kept.append(
            ReflectionQuestion(question=entry.question, answer=entry.answer, citations=citations)
        )
    return tuple(kept)


def _curation(reply: str, config: ClassifierConfig) -> CurationOutcome:
    """Read one reply into a curation, or into the reason it is not one."""
    try:
        data: Any = json.loads(reply)
    except json.JSONDecodeError as exc:
        return CurationFailed(
            f"the curator's reply was not the bare JSON object it was asked for: {_flatten(exc)}"
        )

    if not isinstance(data, dict):
        return CurationFailed(
            f"the curator replied with a JSON {type(data).__name__}, not an object"
        )

    return Curation(
        summary=one_line(str(data.get("summary") or "")),
        learnings=_learnings(data.get("learnings")),
        quirks=_quirks(data.get("quirks")),
        projects=_project_digests(data.get("projects")),
        themes=tuple(one_line(line) for line in _string_tuple(data.get("themes"))),
        open_threads=tuple(one_line(line) for line in _string_tuple(data.get("open_threads"))),
        reflection=_reflection(data.get("reflection")),
        model=config.model,
    )


class ClaudeCliCurator:
    """The real curator: one `claude` CLI invocation per pass (KTD4).

    Reuses `ClassifierConfig` for the same reason `ClaudeCliDistiller` does — the three calls
    want a command, a model, and a timeout, and U1's schema is closed to unknown keys.

    Construction takes no I/O and the runner is injectable, so only calling `curate` with the
    default runner reaches a subprocess.
    """

    def __init__(self, config: ClassifierConfig, *, runner: CommandRunner = run_command) -> None:
        self._config = config
        self._runner = runner

    def argv(self) -> list[str]:
        """The command line, exposed so a caller can log or assert on it.

        Built from `classifier.preset`, like the other three model-facing calls (see
        `presets`).
        """
        return build_argv(self._config)

    def curate(self, material: ReflectionMaterial) -> CurationOutcome:
        outcome = self._curate(material)
        if isinstance(outcome, CurationFailed):
            logger.warning("curation failed — %s", outcome.reason)
        return outcome

    def _curate(self, material: ReflectionMaterial) -> CurationOutcome:
        result = self._runner(
            self.argv(),
            prompt=build_curation_prompt(material),
            timeout=self._config.timeout_seconds,
        )

        if result.timed_out:
            return CurationFailed(
                f"the curator did not answer within {self._config.timeout_seconds}s"
            )
        if result.exit_code != 0:
            detail = _flatten(result.stderr) or "no error output"
            return CurationFailed(f"the curator exited {result.exit_code}: {detail}")

        reply, error = _reply_text(result.stdout, envelope_for(self._config))
        if error is not None or reply is None:
            # `_reply_text` words its reasons for the classifier, its first caller. Every one
            # opens with the same three words, so re-aiming them is exact.
            message = error or "the curator produced no usable reply"
            return CurationFailed(message.replace("the classifier", "the curator"))

        return _curation(reply, self._config)


# --------------------------------------------------------------------------------------
# Writing into memory files, append-only
# --------------------------------------------------------------------------------------


def learnings_skeleton(project: str) -> str:
    return (
        f"# {project} — Learnings\n\n"
        "Append-only takeaways. Each entry: date, context ref, the takeaway in 1-2 lines.\n\n"
        "Format:\n"
        "```\n"
        "## YYYY-MM-DD — <one-line summary>\n"
        "- Context: <where this came from>\n"
        "- Takeaway: <what I now know>\n"
        "```\n"
    )


def quirks_skeleton(project: str) -> str:
    return (
        f"# {project} — Quirks\n\n"
        "Append-only personal gotchas. Things that bit me and shouldn't bite me again.\n\n"
        "Format:\n"
        "```\n"
        "## <short title>\n"
        "- Symptom: <what looks wrong>\n"
        "- Cause: <what's actually happening>\n"
        "- First seen: YYYY-MM-DD (<where this came from>)\n"
        "```\n\n"
        "If a quirk is repo-truth (any future contributor would benefit), promote it to the "
        "repo's CLAUDE.md.\n"
    )


def reflect_log_skeleton(project: str) -> str:
    return (
        f"# {project} — Reflect Log\n\n"
        "Append-only chronological log of reflection passes. Newest entries on top.\n\n"
        "Format:\n"
        "```\n"
        "## YYYY-MM-DD — <one-line summary>\n"
        "- Window: <since>..<until>\n"
        "- Sources: <what the pass read>\n"
        "- Appended: <what it wrote>\n"
        "- Digest: <path>\n"
        "```\n"
    )


def readme_skeleton(project: str, day: date) -> str:
    return (
        f"# {project}\n\n"
        f"Personal memory for {project}. Created by `memvault reflect` on {day.isoformat()}, "
        "because the window carried material for this project and the vault had no directory "
        "for it yet. Replace this paragraph with a one-line orientation: what the project is "
        "and where it is deployed.\n\n"
        "- `learnings.md` — append-only takeaways\n"
        "- `quirks.md` — surprises worth not repeating\n"
        "- `reflect-log.md` — what each pass did, newest first\n"
    )


def _headings(text: str) -> set[str]:
    """Every `## ` heading in this document, ignoring anything inside a fenced code block.

    The fence rule is load-bearing rather than tidy: each of these files opens with a fenced
    *format* block containing a literal `## YYYY-MM-DD — <one-line summary>`. Reading that as a
    heading would make the skeleton itself look like an existing entry, and inserting above it
    would put a real entry inside the documentation of the format.
    """
    found: set[str] = set()
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced and line.startswith("## "):
            found.add(line.strip())
    return found


def _first_entry_index(lines: Sequence[str]) -> int | None:
    """Where the newest entry starts, or None when the file holds no entries yet."""
    fenced = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced and line.startswith("## "):
            return index
    return None


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def append_block(path: Path, heading: str, block: str) -> bool:
    """Append a block to the end of a memory file. Returns whether anything was written.

    The existing text is preserved byte for byte and the new block is added after it — the one
    operation this module is allowed to perform on a file that holds history. A heading already
    present is a no-op, which is what makes a re-run over the same window harmless.
    """
    existing = _read(path)
    if heading.strip() in _headings(existing):
        logger.info("%s: %r is already present; not appending it again", path.name, heading)
        return False

    prefix = existing if not existing or existing.endswith("\n") else existing + "\n"
    separator = "\n" if prefix else ""
    _atomic_write(path, f"{prefix}{separator}{block.rstrip()}\n")
    return True


def insert_block_at_top(path: Path, heading: str, block: str) -> bool:
    """Insert a block above the newest existing entry. Returns whether anything was written.

    `reflect-log.md` is newest-first by the vault's convention, so its entries go in at the top
    — directly below the header and its fenced format block, never above them. Still an
    insertion: no existing line is rewritten, only moved down.
    """
    existing = _read(path)
    if heading.strip() in _headings(existing):
        logger.info("%s: %r is already present; not inserting it again", path.name, heading)
        return False

    body = block.rstrip() + "\n"
    if not existing:
        _atomic_write(path, body)
        return True

    # Split with line endings kept, so head and tail are spliced back byte for byte. The only
    # bytes this function may add are the new entry and, at most, one separating newline —
    # which is what makes the resulting diff a pure insertion.
    lines = existing.splitlines(keepends=True)
    index = _first_entry_index(lines)
    if index is None:
        prefix = existing if existing.endswith("\n") else existing + "\n"
        _atomic_write(path, f"{prefix}\n{body}")
        return True

    head = "".join(lines[:index])
    if not head.endswith("\n\n"):
        head = (head if head.endswith("\n") else head + "\n") + "\n"
    _atomic_write(path, head + body + "\n" + "".join(lines[index:]))
    return True


# --------------------------------------------------------------------------------------
# The digest
# --------------------------------------------------------------------------------------


def render_digest(
    material: ReflectionMaterial,
    curation: Curation,
    window: ObservationWindow,
    *,
    day: date,
) -> str:
    """The digest body: what happened, by project, then across projects.

    Rendered here rather than by the model. The curator supplies judgement — which bullets are
    worth writing — and this supplies the shape, so every digest in the vault reads the same
    way and a reader can skim a year of them.
    """
    blocks: list[str] = []
    if curation.summary:
        blocks.append(curation.summary)

    for digest in curation.projects:
        blocks.append(
            f"## {digest.project}\n\n" + "\n".join(f"- {bullet}" for bullet in digest.bullets)
        )

    if curation.themes:
        blocks.append(
            "## Cross-cutting themes\n\n" + "\n".join(f"- {line}" for line in curation.themes)
        )
    if curation.open_threads:
        blocks.append(
            "## Open threads\n\n" + "\n".join(f"- {line}" for line in curation.open_threads)
        )

    provenance = [f"- Read: {material.counts_line()}"]
    if window.degraded:
        provenance.append(f"- claude-mem: unavailable — {window.degraded}")
    else:
        skipped = window.skipped_line()
        if skipped:
            provenance.append(f"- Outside this vault, skipped: {skipped}")
    provenance.append(f"- Generated by `memvault reflect` on {day.isoformat()}")

    blocks.append("## Sources\n\n" + "\n".join(provenance))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------------------
# The reflection note
# --------------------------------------------------------------------------------------


def render_reflection_note(
    material: ReflectionMaterial,
    questions: Sequence[ReflectionQuestion],
    *,
    day: date,
) -> str:
    """The reflection note's body: each question, its answer, and where the answer came from.

    The citations are rendered as typed relation lines rather than as prose links, so the
    indexer turns them into real edges and a later query on a cited transcript can surface the
    note that discusses it. They sit under their own answer instead of in one block at the
    bottom: the point of the note is that a specific claim is traceable, and a shared footer
    would only tell a reader that the note as a whole read some files.
    """
    blocks = [entry.render() for entry in questions]
    blocks.append(
        "## Sources\n\n"
        + "\n".join(
            [
                f"- Window: {material.since.isoformat()}..{material.until.isoformat()}",
                f"- Read: {material.counts_line()}",
                f"- Generated by `memvault reflect` on {day.isoformat()}",
            ]
        )
    )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------------------


@dataclass
class _Pass:
    """One reflection run over one vault. Mutable by design — it is a ledger being filled."""

    config: Config
    vault: VaultConfig
    curator: Curator
    since: date
    until: date
    today: date
    db_path: str | Path | None = None

    material: ReflectionMaterial | None = None
    appended: list[AppendedEntry] = field(default_factory=list)
    duplicates: list[AppendedEntry] = field(default_factory=list)
    created_projects: list[str] = field(default_factory=list)
    unknown_projects: list[str] = field(default_factory=list)
    skipped_projects: list[str] = field(default_factory=list)
    digest_path: str | None = None
    reflection_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    commits: list[CommitRef] = field(default_factory=list)
    failure: str | None = None

    _paths: list[str] = field(default_factory=list)

    def run(self) -> ReflectReport:
        material, window = gather(self.vault, self.since, self.until, db_path=self.db_path)
        self.material = material
        self.skipped_projects = [f"{item.project} ({item.rows})" for item in window.skipped]
        if window.degraded:
            self.warnings.append(f"claude-mem unavailable: {one_line(window.degraded)}")

        if material.empty:
            logger.info(
                "vault %r: nothing filed or observed between %s and %s — nothing to reflect on",
                self.vault.name,
                self.since.isoformat(),
                self.until.isoformat(),
            )
            return self.report()

        try:
            # Only the trees this pass appends to. A pathspec commit takes the working-tree
            # content of the paths it names, so uncommitted lines in a file reflect is about to
            # append to would be swept into a machine-authored commit — that is the hazard worth
            # refusing. Dirt anywhere else is somebody else's work in progress.
            ensure_clean(
                self.vault.root,
                label=self.vault.name,
                ignore=self._ignored(),
                limit_to=(
                    template_root(self.vault.memory_template),
                    template_root(self.vault.digest_template),
                    template_root(self.vault.reflection_template),
                ),
            )
        except GitError as exc:
            logger.error("vault %r: refusing to reflect — %s", self.vault.name, exc)
            self.failure = one_line(str(exc))
            return self.report()

        outcome = self.curator.curate(material)
        if isinstance(outcome, CurationFailed):
            self.failure = one_line(outcome.reason)
            return self.report()

        if outcome.empty:
            logger.info(
                "vault %r: the window held material but nothing worth remembering", self.vault.name
            )
            return self.report()

        self._write_memory(outcome)
        self._write_digest(material, outcome, window)
        self._write_reflection_note(material, outcome)
        self._write_reflect_logs(material, outcome)
        self._commit()
        return self.report()

    def report(self) -> ReflectReport:
        return ReflectReport(
            vault=self.vault.name,
            since=self.since,
            until=self.until,
            material=self.material,
            appended=tuple(self.appended),
            duplicates=tuple(self.duplicates),
            created_projects=tuple(self.created_projects),
            unknown_projects=tuple(self.unknown_projects),
            skipped_projects=tuple(self.skipped_projects),
            digest_path=self.digest_path,
            reflection_path=self.reflection_path,
            warnings=tuple(self.warnings),
            commits=tuple(self.commits),
            failure=self.failure,
        )

    def _ignored(self) -> tuple[str, ...]:
        try:
            return (self.vault.inbox.relative_to(self.vault.root).as_posix(),)
        except ValueError:
            return ()

    def _fallback_context(self) -> str:
        return f"memvault reflect {self.since.isoformat()}..{self.until.isoformat()}"

    def _memory_dir(self, project: str) -> Path:
        return self.vault.root / self.vault.memory_template.format(project=project)

    def _ensure_project(self, project: str) -> Path:
        """The project's memory directory, created from the skeleton when it does not exist.

        A project with no directory is the ordinary case for a new project, not an error: the
        alternative is a pass that fails on the first week of every new piece of work, exactly
        when its early surprises are worth recording.
        """
        directory = self._memory_dir(project)
        if directory.is_dir():
            return directory

        directory.mkdir(parents=True, exist_ok=True)
        for name, content in (
            (README_FILE, readme_skeleton(project, self.today)),
            (LEARNINGS_FILE, learnings_skeleton(project)),
            (QUIRKS_FILE, quirks_skeleton(project)),
            (REFLECT_LOG_FILE, reflect_log_skeleton(project)),
        ):
            path = directory / name
            if not path.exists():
                _atomic_write(path, content)
                self._track(path)

        logger.info("vault %r: created a memory directory for %s", self.vault.name, project)
        self.created_projects.append(project)
        return directory

    def _known(self, project: str) -> bool:
        """Whether this vault claims the project the curator named.

        The alias map is the vault's statement of what belongs to it. A curated entry for a
        project outside it is refused rather than filed under a new directory, because that is
        the shape a leaked work project would take: plausible content, a directory nobody
        created on purpose, and a diff that looks like ordinary growth.
        """
        if self.vault.canonical_project(project) is not None:
            return True
        if self._memory_dir(project).is_dir():
            return True
        logger.warning(
            "vault %r: dropping an entry for project %r, which is not in claude_mem_projects "
            "and has no memory directory",
            self.vault.name,
            project,
        )
        if project not in self.unknown_projects:
            self.unknown_projects.append(project)
        return False

    def _write_memory(self, curation: Curation) -> None:
        fallback = self._fallback_context()

        for learning in curation.learnings:
            if not self._known(learning.project):
                continue
            directory = self._ensure_project(learning.project)
            path = directory / LEARNINGS_FILE
            heading = learning.heading(self.today)
            entry = AppendedEntry(
                project=learning.project,
                kind="learning",
                relative_path=self._relative(path),
                heading=heading,
            )
            if append_block(path, heading, learning.render(self.today, fallback)):
                self.appended.append(entry)
                self._track(path)
            else:
                self.duplicates.append(entry)

        for quirk in curation.quirks:
            if not self._known(quirk.project):
                continue
            directory = self._ensure_project(quirk.project)
            path = directory / QUIRKS_FILE
            heading = quirk.heading()
            entry = AppendedEntry(
                project=quirk.project,
                kind="quirk",
                relative_path=self._relative(path),
                heading=heading,
            )
            if append_block(path, heading, quirk.render(self.today, fallback)):
                self.appended.append(entry)
                self._track(path)
            else:
                self.duplicates.append(entry)

    def _write_digest(
        self, material: ReflectionMaterial, curation: Curation, window: ObservationWindow
    ) -> None:
        """Write the window's digest, overwriting any digest already there for it (R14).

        Overwriting rather than suffixing, and it is the only file this pass treats that way: a
        digest is a rendered view of a fixed window, so a second render of the same window is
        the same artifact. Memory files hold history and are appended to; this holds a summary
        and is replaced.
        """
        relative = render_path(
            self.vault.digest_template, day=self.until, slug=f"reflect-{self.until.isoformat()}"
        )
        path = self.vault.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)

        frontmatter = {
            "title": f"Digest {material.since.isoformat()} — {material.until.isoformat()}",
            "date": self.until.isoformat(),
            "source": "memvault reflect",
            "summary": curation.summary,
        }
        body = render_digest(material, curation, window, day=self.today)
        _atomic_write(path, render_document(frontmatter, body))

        self.digest_path = relative
        self._track(path)
        logger.info("vault %r: digest written to %s", self.vault.name, relative)

    def _write_reflection_note(self, material: ReflectionMaterial, curation: Curation) -> None:
        """Write the window's reflection note, or nothing when nothing survives citation (R6).

        Frontmatter declares `kind: reflection` and deliberately carries no `content_id`. The id
        is what marks a file as *material this engine filed from a drop*, and this note was
        written from the vault rather than into it — stamping one would enrol the note as input
        to the next pass, which is the exact loop `filed_in_window` is guarding against.

        Overwritten on a re-run over the same window, like the digest: the note is this pass's
        answer to one window, and a second answer to the same window replaces the first rather
        than sitting beside it. Memory files append because they hold history; this holds a
        reading of a fixed set of files.
        """
        questions = cited_questions(curation.reflection, material)
        if not questions:
            if curation.reflection:
                logger.info(
                    "vault %r: no reflection note — none of the curator's answers cited a file "
                    "this window filed",
                    self.vault.name,
                )
            return

        relative = render_path(
            self.vault.reflection_template,
            day=self.until,
            slug=f"reflection-{self.until.isoformat()}",
        )
        path = self.vault.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)

        frontmatter = {
            "title": (f"Reflection {material.since.isoformat()} — {material.until.isoformat()}"),
            "date": self.until.isoformat(),
            "kind": REFLECTION_KIND,
            "source": "memvault reflect",
            "summary": curation.summary,
        }
        body = render_reflection_note(material, questions, day=self.today)
        _atomic_write(path, render_document(frontmatter, body))

        self.reflection_path = relative
        self._track(path)
        logger.info(
            "vault %r: reflection note written to %s (%d question(s))",
            self.vault.name,
            relative,
            len(questions),
        )

    def _write_reflect_logs(self, material: ReflectionMaterial, curation: Curation) -> None:
        """Log the pass in every project it touched, newest-first.

        Only projects that received an append are logged. A reflect-log entry saying "nothing
        was appended" in fifteen directories every week is noise that would bury the entries
        that mean something.
        """
        touched = sorted({item.project for item in self.appended})
        heading = (
            f"## {self.today.isoformat()} — memvault reflect "
            f"{self.since.isoformat()}..{self.until.isoformat()}"
        )

        for project in touched:
            learnings = sum(
                1 for item in self.appended if item.project == project and item.kind == "learning"
            )
            quirks = sum(
                1 for item in self.appended if item.project == project and item.kind == "quirk"
            )
            lines = [
                heading,
                f"- Window: {self.since.isoformat()}..{self.until.isoformat()}",
                f"- Sources: {material.counts_line()}",
                f"- Appended: {learnings} learning(s), {quirks} quirk(s)",
            ]
            if self.digest_path:
                lines.append(f"- Digest: {self.digest_path}")
            if self.reflection_path:
                lines.append(f"- Reflection: {self.reflection_path}")
            if curation.summary:
                lines.append(f"- Summary: {curation.summary}")

            path = self._memory_dir(project) / REFLECT_LOG_FILE
            entry = AppendedEntry(
                project=project,
                kind="reflect-log",
                relative_path=self._relative(path),
                heading=heading,
            )
            if insert_block_at_top(path, heading, "\n".join(lines)):
                self.appended.append(entry)
                self._track(path)
            else:
                self.duplicates.append(entry)

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.vault.root).as_posix()
        except ValueError:
            return path.as_posix()

    def _track(self, path: Path) -> None:
        relative = self._relative(path)
        if relative not in self._paths:
            self._paths.append(relative)

    def _commit(self) -> None:
        if not self._paths:
            return
        try:
            staged = stage(self.vault.root, self._paths)
            reference = commit(
                self.vault.root,
                build_commit_message(self.report()),
                staged,
                vault=self.vault.name,
            )
        except GitError as exc:
            logger.error("vault %r: commit failed — %s", self.vault.name, exc)
            self.failure = one_line(str(exc))
            return

        if reference is not None:
            self.commits.append(reference)


def _bullets(lines: Iterable[str]) -> list[str]:
    return [f"- {line}" for line in lines]


def build_commit_message(report: ReflectReport) -> str:
    """Summarize a pass as a commit message: counts in the subject, entries in the body.

    The body names every append so the pass is reviewable from `git log` alone, and the diff —
    which is the review surface R15 asks for — then only has to confirm that every change is an
    addition.
    """
    learnings, quirks = report.counts("learning"), report.counts("quirk")
    subject = (
        f"mem: reflect {report.since.isoformat()}..{report.until.isoformat()} — "
        f"{learnings} learning(s), {quirks} quirk(s) in {report.vault}"
    )

    blocks: list[str] = []
    if report.material is not None:
        blocks.append(f"Read: {report.material.counts_line()}")
    if report.appended:
        blocks.append(
            f"Appended {len(report.appended)}:\n"
            + "\n".join(_bullets(item.summary_line() for item in report.appended))
        )
    if report.created_projects:
        blocks.append(
            "Created memory directories:\n" + "\n".join(_bullets(report.created_projects))
        )
    if report.digest_path:
        blocks.append(f"Digest: {report.digest_path}")
    if report.reflection_path:
        blocks.append(f"Reflection: {report.reflection_path}")
    if report.skipped_projects:
        blocks.append(
            "Outside this vault, skipped:\n" + "\n".join(_bullets(report.skipped_projects))
        )
    if report.warnings:
        blocks.append("Warnings:\n" + "\n".join(_bullets(report.warnings)))

    return "\n\n".join([subject, *blocks]) + "\n"


def default_window(today: date, days: int = DEFAULT_WINDOW_DAYS) -> tuple[date, date]:
    """The window a pass runs over when nobody named one: `days` back, ending today."""
    if days < 1:
        raise ValueError(f"a reflection window must be at least one day, got {days}")
    return today - timedelta(days=days - 1), today


def run_reflect(
    config: Config,
    vault: VaultConfig,
    *,
    curator: Curator | None = None,
    since: date | None = None,
    until: date | None = None,
    days: int = DEFAULT_WINDOW_DAYS,
    today: date | None = None,
    db_path: str | Path | None = None,
    notifier: Notifier | None = notify_failure,
    log_path: str = DEFAULT_LOG_PATH,
) -> ReflectReport:
    """Run one reflection pass and report what it did.

    The curator is injected rather than constructed unconditionally, so a test never spawns a
    subprocess and a caller can swap the model tier. The default is I/O-free to construct —
    only calling it reaches the CLI.
    """
    today = today or date.today()
    end = until or today
    start = since or default_window(end, days)[0]

    if start > end:
        raise ValueError(f"the window starts after it ends: {start.isoformat()}..{end.isoformat()}")

    report = _Pass(
        config=config,
        vault=vault,
        curator=curator or ClaudeCliCurator(config.classifier),
        since=start,
        until=end,
        today=today,
        db_path=db_path,
    ).run()

    logger.info("vault %r: %s", vault.name, report.summary_line())

    if notifier is not None and report.status is not ReflectStatus.OK:
        notifier("reflect", _failure_message(report), log_path)

    return report


def _failure_message(report: ReflectReport) -> str:
    """One line for the notification: what went wrong, not the whole ledger."""
    if report.failure is not None:
        return report.failure
    return report.warnings[0] if report.warnings else "the reflection pass reported a problem"
