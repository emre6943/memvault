"""`memvault mcp` — the vault served over MCP, for every harness that is not Claude Code.

Claude Code gets skills and hooks. Codex, opencode, Cursor and Claude Desktop get this: one
stdio server exposing four tools, so an adopter on any of them reaches the same vault with the
same guarantees rather than a thinner copy of them (R10).

**Four tools, two of which write, and the write surfaces are narrow by construction.**
`recall` and `vault_status` read and carry read-only annotations. `write_inbox` can only create
a file directly inside the configured inbox — the drop then travels the ordinary ingest path,
through classification, routing, and the distill guard, so a tool call cannot file anything
anywhere by itself. `write_note` can only write into an area the vault's own config names. The
engine builds no approval layer of its own: the harness's permission prompt is the gate, and
these two bounds are what remains true when a harness auto-approves.

**Startup is cheap on purpose.** A client bounds how long a server may take to come up, so the
lifespan opens the SQLite index and nothing else; the embedding model — a ~130MB download on
first use — loads on the first query that needs it and is memoized behind a lock, because sync
tool handlers run on worker threads and two concurrent queries would otherwise build two.

**Nothing this process prints may reach stdout.** stdout is the protocol wire, and one stray
line corrupts the session. Three things keep it clean, in descending order of who is
responsible: the SDK's stdio transport diverts fd 1 to stderr for the duration and serves the
wire from a private duplicate; logging is configured onto stderr here; and the embedder is
wrapped so that anything it prints — fastembed's download bar is the known offender, suppressed
by environment variable as well — is redirected while it runs. The last is what protects the
non-stdio transports and the in-process client, where the SDK's diversion does not apply.

**The vault directory is reconciled against the config, and a mismatch is fatal at startup.**
The harness is configured with a path and the engine is configured with a config file; when
those two disagree, tools would answer about one vault while the hooks and the scheduler worked
on another, and nothing about the output would say so.

Tools are plain `def`. The SDK runs a sync handler on a worker thread, so blocking on SQLite,
the filesystem, or an ONNX model is correct here rather than something to wrap in `async`.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from memvault.area import project_areas
from memvault.cli import HealthReport, check_health
from memvault.config import AreaConfig, Config, IndexConfig, VaultConfig
from memvault.inbox import DECLARED_KEYS, IMPORTANCE_RANGE, TEXT_SUFFIXES, normalize
from memvault.index import Embedder, IndexBuildError, count_chunks
from memvault.recall import DEFAULT_LIMIT, RecallError, query_embedder
from memvault.recall import recall as run_recall
from memvault.writer import (
    WriteError,
    _atomic_write,
    render_path,
    slugify,
    write_document,
)

logger = logging.getLogger(__name__)

#: The server's name on the wire. Stable, because a harness stores it in its own config.
SERVER_NAME = "memvault"

#: What a note written through `write_note` declares itself to be, and where it says it came
#: from. Both are provenance: a note that appeared through a tool call is a different kind of
#: fact from one the ingest pass distilled, and six months later nobody remembers which.
NOTE_KIND = "note"
NOTE_SOURCE = "mcp"

#: Environment variables that turn progress bars and telemetry off in the embedding stack.
#: Set before the model is ever constructed, because fastembed's first-run download prints a
#: bar by default and a bar on stdout is a corrupted MCP session.
PROGRESS_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TQDM_DISABLE": "1",
}

#: How long a generated inbox filename's slug may be. Long enough to recognise in a directory
#: listing, short enough to leave room for the date and the content-id suffix.
GENERATED_SLUG_CHARS = 40


class MCPStartupError(Exception):
    """Raised before the server serves anything: the vault could not be agreed on.

    Deliberately not a tool error. A client that reaches a running server and gets a refusal
    will retry; a mismatch between the directory it was pointed at and the config the engine
    read is not something a retry fixes, so the process refuses to start at all.
    """


@contextmanager
def quiet_stdout() -> Iterator[None]:
    """Send anything written to `sys.stdout` inside this block to stderr instead.

    Rebinding the name is enough: the stdio transport captured its own stream object at startup
    and writes through that, while a library calling `print()` looks `sys.stdout` up at call
    time and lands on stderr. Nothing is swallowed — the words still reach a terminal and a log,
    just not the wire.
    """
    original = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original


class _QuietEmbedder:
    """An embedder whose output cannot reach stdout, whatever it prints.

    Wrapping rather than trusting the environment variables: they cover the two bars this stack
    is known to print today, and this covers the third one somebody adds later.
    """

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.model_id = inner.model_id

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with quiet_stdout():
            return self.inner.embed(texts)


#: How an embedder is obtained from index config. A seam so tests never load a real model, and
#: so U9's keyword-only degrade has one place to plug into.
EmbedderFactory = Callable[[IndexConfig], Embedder]


@dataclass(frozen=True)
class RecallHit:
    """One cited passage, as a tool result. The path is what the caller acts on."""

    path: str
    heading: str
    snippet: str
    score: float
    #: Which retrieval modes found this — `keyword`, `semantic`, `graph`, or a `+` join of them.
    #: `graph` alone means the file matched nothing itself and is here because a stronger hit
    #: links to it, which is a lead rather than an answer.
    modes: str
    date: str | None = None
    source: str | None = None
    #: Set when the file's frontmatter says something replaced it. A reader who acts on this
    #: passage needs to know that before they act, not after.
    superseded_by: str | None = None


@dataclass(frozen=True)
class RecallOutput:
    """Results plus the sentence explaining an empty list, which is a real answer."""

    query: str
    message: str
    chunks_searched: int
    results: list[RecallHit] = field(default_factory=list)


@dataclass(frozen=True)
class DropWritten:
    """Where a queued drop landed, and what happens to it next."""

    path: str
    filename: str
    content_id: str
    message: str


@dataclass(frozen=True)
class NoteWritten:
    """Where a note landed. The path is vault-relative and may carry a collision suffix."""

    path: str
    area: str
    title: str
    message: str


@dataclass(frozen=True)
class StatusCheck:
    """One health finding, flattened for a caller that cannot read a terminal."""

    status: str
    message: str
    problem: bool


@dataclass(frozen=True)
class VaultStatus:
    """What `memvault doctor` would say, as a record rather than as status lines."""

    vault: str
    root: str
    config_path: str
    healthy: bool
    inbox_pending: int
    held_for_review: int
    index_path: str
    #: `current`, `stale`, or `missing` — see `cli.INDEX_CURRENT` and its neighbours.
    index_state: str
    #: None when there is no readable index to count, which is not the same as an empty one.
    indexed_chunks: int | None = None
    checks: list[StatusCheck] = field(default_factory=list)


def reconcile_vault(
    config: Config, directory: str | Path | None, *, requested: str | None = None
) -> VaultConfig:
    """Decide which configured vault a `memvault mcp <directory>` invocation means.

    The directory is the authority when there is one: a harness is configured with a path, and
    that path is the user's statement of what they want served. It must match a configured
    vault's root exactly — matching nothing is refused rather than defaulted, because a silent
    default here means the tools answer about one vault while the scheduler files into another
    and no output ever says so.

    `--vault` may still be passed; it then has to agree, and a disagreement is named rather
    than resolved in either direction's favour.
    """
    if directory is None:
        return config.vault(requested)

    wanted = Path(directory).expanduser().resolve()
    known = ", ".join(f"{v.name} at {v.root}" for v in sorted(config.vaults.values(), key=_by_name))
    matches = [vault for vault in config.vaults.values() if vault.root == wanted]

    if not matches:
        raise MCPStartupError(
            f"{wanted} is not the root of any vault in {config.source}. Configured: "
            f"{known or '(none)'}. Point the server at one of those directories, or add this "
            f"one to the config with `memvault init --adopt {wanted}`."
        )

    if requested is not None:
        chosen = config.vault(requested)
        if chosen.root != wanted:
            raise MCPStartupError(
                f"--vault {requested!r} is rooted at {chosen.root}, but the server was pointed "
                f"at {wanted}. Those are two different vaults; drop one of the two arguments."
            )
        return chosen

    if len(matches) > 1:
        names = ", ".join(sorted(vault.name for vault in matches))
        raise MCPStartupError(
            f"{wanted} is the root of more than one configured vault ({names}). "
            "Name the one you mean with --vault."
        )

    return matches[0]


def _by_name(vault: VaultConfig) -> str:
    return vault.name


def configure_logging(level: int = logging.INFO) -> None:
    """Put every log line on stderr, because stdout is the protocol.

    `force=True` because a library that logged during import may already have installed a
    handler on stdout, and one such line ends the session.
    """
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def suppress_progress_bars() -> None:
    """Turn the embedding stack's progress bars off, without overriding a deliberate setting."""
    for key, value in PROGRESS_ENV.items():
        os.environ.setdefault(key, value)


class _Service:
    """What the four tools share: the config, one vault, one index handle, one embedder.

    The index connection is opened once by the lifespan and used from worker threads, so it is
    opened with `check_same_thread=False` and every read takes the lock. It serves the cheap
    whole-index questions `vault_status` asks. `recall` is deliberately not routed through it:
    that function owns its own connection for the length of a query, which is what keeps a long
    scan off this handle and out of the lock.
    """

    def __init__(
        self,
        config: Config,
        vault: VaultConfig,
        *,
        embedder_factory: EmbedderFactory = query_embedder,
    ) -> None:
        self.config = config
        self.vault = vault
        self.db_path = config.index.db_path(vault.name)
        self._embedder_factory = embedder_factory
        self._embedder: Embedder | None = None
        self._embedder_lock = threading.Lock()
        self._index: sqlite3.Connection | None = None
        self._index_lock = threading.Lock()

    def open(self) -> None:
        """Open the index if one exists. Never creates it: a server is not an indexer."""
        if not self.db_path.exists():
            logger.info(
                "no index at %s — recall will report it until `memvault index` runs", self.db_path
            )
            return
        try:
            self._index = sqlite3.connect(str(self.db_path), check_same_thread=False)
        except sqlite3.Error as exc:
            logger.warning("could not open the index at %s: %s", self.db_path, exc)

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None

    def indexed_chunks(self) -> int | None:
        """How many chunks the index holds, or None when there is no readable index.

        None rather than 0: an index that cannot be read and an index with nothing in it call
        for different actions, and a zero would send someone looking for missing files.
        """
        with self._index_lock:
            if self._index is None:
                return None
            try:
                return count_chunks(self._index)
            except sqlite3.Error:
                return None

    def embedder(self) -> Embedder:
        """The query embedder, built once on first use.

        Under a lock because sync tool handlers run on worker threads: two queries arriving
        together would otherwise each pay the model load, and the second would throw its copy
        away.
        """
        with self._embedder_lock:
            if self._embedder is None:
                with quiet_stdout():
                    self._embedder = _QuietEmbedder(self._embedder_factory(self.config.index))
            return self._embedder

    def areas(self) -> tuple[AreaConfig, ...]:
        """Every area a note may be written into: the configured ones, then the projects.

        The same list the ingestion pass builds, in the same order, so "a named area" means one
        thing in this vault rather than two.
        """
        return tuple(self.vault.areas) + project_areas(self.vault)


def _day(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ToolError(f"{name} must be a YYYY-MM-DD date, got {value!r}") from exc


def _checked_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only what the inbox contract reads, and name anything else.

    The file contract ignores unknown frontmatter keys, because a feeder carries its own
    bookkeeping and should not need an engine change to do so. A tool call is the other case: a
    caller who misspells `classification` has made a mistake it can fix in the next call, and
    telling it beats filing a drop whose declared classification was silently dropped.
    """
    if not metadata:
        return {}
    unknown = sorted(key for key in metadata if key not in DECLARED_KEYS)
    if unknown:
        raise ToolError(
            f"metadata key(s) the inbox does not read: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(DECLARED_KEYS))}"
        )
    return {key: value for key, value in metadata.items() if value is not None}


def _drop_document(text: str, metadata: dict[str, Any]) -> str:
    """A drop as it will sit in the inbox: declared frontmatter, then the text unchanged."""
    if not metadata:
        return text
    rendered = yaml.safe_dump(
        metadata, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000
    )
    return f"---\n{rendered}---\n\n{text}"


def _checked_filename(name: str, inbox: Path) -> str:
    """Refuse anything that is not a plain file name directly inside the inbox.

    A drop names a file, never a place. Separators, `..`, absolute paths and leading dots are
    all refused by shape, and the resolved path is then checked against the inbox as well, so a
    symlink cannot achieve what the shape check forbids.
    """
    cleaned = name.strip()
    if not cleaned:
        raise ToolError("filename must not be empty")
    if "/" in cleaned or "\\" in cleaned or cleaned in (".", ".."):
        raise ToolError(
            f"filename {name!r} must be a plain file name — drops are written directly into the "
            "inbox, never into a subdirectory or anywhere outside it"
        )
    if cleaned.startswith("."):
        raise ToolError(
            f"filename {name!r} starts with a dot, and the inbox skips hidden entries — the drop "
            "would sit there forever without being read"
        )
    suffix = Path(cleaned).suffix.lower()
    if suffix and suffix not in TEXT_SUFFIXES:
        raise ToolError(
            f"filename {name!r} has extension {suffix!r}; the inbox reads "
            f"{' or '.join(sorted(TEXT_SUFFIXES))} or no extension at all"
        )
    if not (inbox / cleaned).resolve().is_relative_to(inbox.resolve()):
        raise ToolError(f"filename {name!r} resolves outside the inbox at {inbox}")
    return cleaned


def _generated_filename(body: str, content_id: str, day: str | None) -> str:
    """A readable, unique name for a drop that did not bring one.

    The content id goes in the name rather than only in the file, so the same text queued twice
    lands on the same name and the second call is refused instead of quietly doubling a memory.
    """
    first = next((line for line in body.splitlines() if line.strip()), "")
    slug = slugify(first, max_length=GENERATED_SLUG_CHARS) or "drop"
    return f"{day or date.today().isoformat()}-{slug}-{content_id[:8]}.md"


def _resolve_area(name: str, areas: Sequence[AreaConfig]) -> AreaConfig:
    """Find the named area, or refuse listing the ones that exist.

    First match wins on a case-folded name, which is the rule `area.choose_area` already
    applies: a configured area beats a project directory that happens to share its name.
    """
    wanted = name.strip().casefold()
    for area in areas:
        if area.name.casefold() == wanted:
            return area
    known = ", ".join(area.name for area in areas) or "(none configured)"
    raise ToolError(
        f"{name!r} is not an area of this vault. Known areas: {known}. "
        "Notes are written into a configured area, never to an arbitrary path."
    )


def _status_of(report: HealthReport, chunks: int | None) -> VaultStatus:
    return VaultStatus(
        vault=report.vault,
        root=str(report.root),
        config_path=str(report.config_path),
        healthy=report.healthy,
        inbox_pending=report.inbox_pending,
        held_for_review=report.held_for_review,
        index_path=str(report.index_path),
        index_state=report.index_state,
        indexed_chunks=chunks,
        checks=[
            StatusCheck(status=check.keyword.lower(), message=check.message, problem=check.problem)
            for check in report.checks
        ],
    )


def build_server(
    config: Config,
    vault: VaultConfig,
    *,
    embedder_factory: EmbedderFactory = query_embedder,
) -> MCPServer[_Service]:
    """Assemble the server for one vault. Constructs nothing expensive.

    The tools close over one `_Service` rather than reading it back out of the request context:
    there is exactly one vault per process, so threading it through the protocol would be
    ceremony around a value that is already in scope.
    """
    service = _Service(config, vault, embedder_factory=embedder_factory)

    @asynccontextmanager
    async def lifespan(_server: MCPServer[_Service]) -> AsyncIterator[_Service]:
        """Open the index, and nothing else. The embedder waits until something asks it to."""
        service.open()
        try:
            yield service
        finally:
            service.close()

    server: MCPServer[_Service] = MCPServer(
        SERVER_NAME,
        instructions=(
            f"Memory for the vault {vault.name!r} at {vault.root}: plain Markdown files in one "
            "git repository. Search it with `recall` before answering from memory — results "
            "cite vault-relative paths, so open what looks relevant instead of loading memory "
            "in bulk. Queue new material with `write_inbox`; it is filed, with provenance, by "
            "the next ingest pass rather than immediately."
        ),
        lifespan=lifespan,
    )

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=True, idempotent_hint=True, open_world_hint=False
        )
    )
    def recall(
        query: str,
        limit: int = DEFAULT_LIMIT,
        sources: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> RecallOutput:
        """Search the vault by keyword and by meaning, and return cited passages.

        Reads only. Results are pointers: each carries a vault-relative path and a snippet for
        deciding what to open, not the whole file. An empty result list is an answer — the
        message says whether the vault holds nothing on the subject or a filter excluded it.

        `sources` keeps only files whose declared source or top-level vault directory matches
        (`transcripts`, `whatsapp`, `notes`, …). `since` and `until` are `YYYY-MM-DD` and filter
        on the file's own date.
        """
        try:
            response = run_recall(
                service.vault,
                service.config.index,
                service.embedder(),
                query,
                limit=limit,
                sources=tuple(sources or ()),
                since=_day(since, "since"),
                until=_day(until, "until"),
                min_similarity=service.config.index.min_similarity,
                ranking=service.config.ranking,
            )
        except (RecallError, IndexBuildError) as exc:
            raise ToolError(str(exc)) from exc

        return RecallOutput(
            query=response.query,
            message=response.message,
            chunks_searched=response.chunks_searched,
            results=[
                RecallHit(
                    path=result.path,
                    heading=result.heading,
                    snippet=result.snippet,
                    score=result.score,
                    modes=result.modes,
                    date=None if result.date is None else result.date.isoformat(),
                    source=result.source,
                    superseded_by=result.superseded_by,
                )
                for result in response.results
            ],
        )

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        )
    )
    def write_inbox(
        text: str,
        filename: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DropWritten:
        """Queue material for the vault by writing one file into its inbox.

        Writes exactly one new file, directly inside the inbox, and nothing else anywhere. It
        does not file, classify, or commit: the next `memvault ingest` run reads the drop,
        classifies it, files it with provenance, and removes it from the inbox.

        `metadata` may declare any of the inbox contract's fields — `source`, `date`, `tags`,
        `classification` (personal/work/mixed), `local_only`, `participants`, `workspace`,
        `importance` (1-10) — and each one overrides what the classifier would have inferred.
        An unreadable value is refused here rather than held for review later.

        Refuses an existing filename rather than overwriting: two drops are two memories.
        """
        declared = _checked_metadata(metadata)
        document = _drop_document(text, declared)

        probe = normalize(document, filename or "drop.md")
        if probe.unparseable:
            raise ToolError(f"the drop's metadata could not be used: {probe.unparseable}")
        if not probe.body:
            raise ToolError("the drop has no content once metadata and blank lines are removed")

        inbox = service.vault.inbox
        name = (
            _checked_filename(filename, inbox)
            if filename is not None
            else _generated_filename(probe.body, probe.id, probe.declared.date)
        )
        path = inbox / name
        if path.exists():
            raise ToolError(
                f"{name} is already waiting in the inbox. Pick another filename, or run "
                "`memvault ingest` to drain what is there first."
            )

        inbox.mkdir(parents=True, exist_ok=True)
        # Atomic, borrowing writer's helper the way `reflect` already does: a half-written drop
        # is worse than none, because the next ingest would file the truncated half as a memory.
        _atomic_write(path, document if document.endswith("\n") else document + "\n")
        logger.info("queued %s in %s", name, inbox)

        return DropWritten(
            path=str(path.relative_to(service.vault.root))
            if path.is_relative_to(service.vault.root)
            else str(path),
            filename=name,
            content_id=probe.id,
            message=(
                "queued in the inbox — the next `memvault ingest` files it with provenance and "
                "commits the move"
            ),
        )

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        )
    )
    def write_note(
        area: str,
        title: str,
        body: str,
        summary: str | None = None,
        tags: list[str] | None = None,
        importance: int | None = None,
    ) -> NoteWritten:
        """Write a note straight into one of the vault's configured areas.

        For material that is already a conclusion. Anything that is raw — a conversation, a
        pasted thread, something to be classified — belongs in `write_inbox` instead, so it is
        filed with provenance and routed by the pipeline rather than placed by hand.

        `area` must be one of the vault's configured areas or an existing project directory;
        anything else is refused with the list of what exists. The note lands at the area's own
        path template and never overwrites: a name already taken gets a numeric suffix.

        `importance` is 1-10 and tells recall how hard to work to resurface this later.
        """
        chosen = _resolve_area(area, service.areas())

        if importance is not None and not IMPORTANCE_RANGE[0] <= importance <= IMPORTANCE_RANGE[1]:
            low, high = IMPORTANCE_RANGE
            raise ToolError(
                f"importance must be a whole number from {low} to {high}, got {importance}"
            )

        slug = slugify(title)
        if not slug:
            raise ToolError(
                f"title {title!r} leaves nothing usable as a filename — it needs at least one "
                "letter or digit"
            )

        day = date.today()
        try:
            relative = render_path(chosen.note_template, day=day, slug=slug)
        except WriteError as exc:
            raise ToolError(str(exc)) from exc

        frontmatter: dict[str, Any] = {
            "title": title.strip(),
            "date": day.isoformat(),
            "kind": NOTE_KIND,
            "source": NOTE_SOURCE,
            "area": chosen.name,
            # Provenance, in the vocabulary the ingestion path already uses for this key: a
            # note that looks misfiled should be explicable rather than a mystery.
            "area_via": NOTE_SOURCE,
            "tags": list(tags or ()),
            "summary": summary,
            "importance": importance,
        }

        try:
            placement = write_document(
                service.vault.root, relative, frontmatter=frontmatter, body=body.strip()
            )
        except WriteError as exc:
            raise ToolError(str(exc)) from exc

        logger.info("wrote %s into area %s", placement.relative_path, chosen.name)
        return NoteWritten(
            path=placement.relative_path,
            area=chosen.name,
            title=title.strip(),
            message=(
                "written and uncommitted — commit it with the vault's own history, and re-run "
                "`memvault index` before expecting recall to return it"
            ),
        )

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=True, idempotent_hint=True, open_world_hint=False
        )
    )
    def vault_status() -> VaultStatus:
        """Report whether this vault is healthy, and what is waiting in it.

        The same checks `memvault doctor` runs: is the config readable, does the inbox exist
        and how much is queued in it, is anything held for review, is the search index present
        and newer than the files, does the work route resolve. Reads only.
        """
        return _status_of(check_health(service.config, service.vault), service.indexed_chunks())

    return server


def serve(
    config: Config,
    *,
    directory: str | Path | None = None,
    requested_vault: str | None = None,
    embedder_factory: EmbedderFactory = query_embedder,
) -> None:
    """Reconcile the vault, then serve it on stdin/stdout until the client disconnects.

    Everything that can refuse happens before the transport is claimed, so a misconfigured
    server fails with a readable line on stderr rather than with a client-side timeout.
    """
    vault = reconcile_vault(config, directory, requested=requested_vault)
    configure_logging()
    suppress_progress_bars()
    logger.info("serving vault %r at %s over stdio", vault.name, vault.root)
    build_server(config, vault, embedder_factory=embedder_factory).run()


__all__ = [
    "NOTE_KIND",
    "NOTE_SOURCE",
    "PROGRESS_ENV",
    "SERVER_NAME",
    "DropWritten",
    "EmbedderFactory",
    "MCPStartupError",
    "NoteWritten",
    "RecallHit",
    "RecallOutput",
    "StatusCheck",
    "VaultStatus",
    "build_server",
    "configure_logging",
    "quiet_stdout",
    "reconcile_vault",
    "serve",
    "suppress_progress_bars",
]
