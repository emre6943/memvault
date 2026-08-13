"""Read-only queries against the claude-mem observation database.

claude-mem is a third-party Claude Code plugin whose database records what happened in past
sessions. The weekly pass reads it as *additional* raw material alongside the vault's own filed
transcripts, and it reads it strictly read-only: nothing here writes, and a database that cannot
be opened degrades the pass to vault-only reflection rather than failing it.

Its schema is at version 32, carries several dead columns, and has already survived one
destructive cleanup, so it is treated as a moving target rather than a contract. Four properties
of the live file drive every decision in this module, and each was verified against it:

**`*_epoch` columns are milliseconds.** Filtering a millisecond column with a seconds-scale
bound matches every row ever written, and the result looks like a very productive week rather
than like a bug. So the scale is *sampled* at connect time from the largest epoch in the table
and the window is scaled to match, instead of being assumed.

**The connection is opened `file:...?mode=ro`, never `?immutable=1`.** `immutable` promises
SQLite the file cannot change, which makes it skip the WAL — and claude-mem runs in WAL mode, so
the most recent sessions, the ones a weekly pass exists to read, are exactly the rows that would
go missing.

**Nothing joins through `sdk_sessions`.** Only about 21% of observations still have a session
row: a prior plugin cleanup pruned sessions while their observations survived, and foreign keys
are off in this database so nothing stopped it. An `INNER JOIN` would silently discard
four-fifths of the corpus and return a plausible-looking remainder. Both tables carry their own
`project` and their own timestamps, which is all this module needs from them.

**One database holds work and personal observations.** Work projects like `acme-api` sit
beside personal ones. Every row is therefore passed through the vault's own
`claude_mem_projects` alias map, and a project the map does not name is **skipped and counted**,
never defaulted into the vault. That is a boundary control rather than a filter: a leaked work
observation would be appended to personal memory as a legitimate-looking learning, and the diff
that is supposed to be the review surface would show nothing unusual.

**The whole source is optional.** claude-mem is a plugin most adopters do not run, so a vault
that says `claude_mem: false` — or that names no `claude_mem_projects`, which is where every new
vault starts — is skipped here before any file is touched, and comes back as an empty *available*
window (R11). Only an enabled source that could not be read degrades the pass, because only then
did the pass reflect on less than it should have.

Two smaller rules, both from the plan. Rows whose `merged_into_project` is set are excluded even
though the column is currently unset everywhere, because the plugin may begin populating it. And
`facts`, `concepts`, and `files_modified` are JSON arrays whose malformed values degrade the one
row that carries them — a single bad value must not cost a whole week's reflection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import quote

from memvault.config import VaultConfig

logger = logging.getLogger(__name__)

#: Where the plugin keeps its database. Overridable per call so no test reads the real one.
DEFAULT_DB_PATH = "~/.claude-mem/claude-mem.db"

#: Long enough to ride out another process's write transaction, short enough that an unattended
#: pass does not sit on a lock for minutes before degrading.
CONNECT_TIMEOUT_SECONDS = 5.0

#: Above this, an epoch value cannot be seconds: 1e11 seconds is the year 5138. Below it, it
#: cannot be milliseconds after 1973. Any real timestamp from either scale lands unambiguously
#: on one side, which is what makes sampling a single row a sufficient test.
MILLISECOND_FLOOR = 100_000_000_000

#: The columns each query reads. Named explicitly rather than `SELECT *` so a plugin migration
#: that adds or reorders columns cannot change what this module thinks it is reading.
_OBSERVATION_COLUMNS = (
    "project",
    "type",
    "title",
    "subtitle",
    "facts",
    "narrative",
    "concepts",
    "files_modified",
    "created_at",
    "created_at_epoch",
)

_SUMMARY_COLUMNS = (
    "project",
    "request",
    "investigated",
    "learned",
    "completed",
    "next_steps",
    "notes",
    "created_at",
    "created_at_epoch",
)

#: Applied to both tables. The column is unset throughout the live database today; the filter is
#: here so that the day the plugin starts merging projects, merged rows do not arrive twice.
_UNMERGED = "(merged_into_project IS NULL OR merged_into_project = '')"


class ClaudeMemError(Exception):
    """Raised when the claude-mem database cannot be read.

    Callers that want the pass to survive it use `fetch_window`, which turns this into a
    degraded window. It is an exception at the lower level so that a caller cannot read a
    failed query as an empty week.
    """


class EpochScale(StrEnum):
    """Which unit this database's `*_epoch` columns are in."""

    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    #: No rows to sample. The window is still computed — in milliseconds, the observed scale —
    #: but an empty table returns nothing either way, so the choice costs nothing.
    UNKNOWN = "unknown"

    @property
    def divisor(self) -> int:
        """What to divide a millisecond bound by to reach this scale."""
        return 1000 if self is EpochScale.SECONDS else 1


def _json_list(raw: object, column: str) -> tuple[tuple[str, ...], bool]:
    """Read a JSON-array column into a tuple of strings, reporting whether it was malformed.

    A value that is not parseable JSON, or that parses to something other than a list, comes
    back empty with the flag set: the row is still worth reading, and the plan is explicit that
    one bad value degrades its own row rather than the pass.
    """
    if raw in (None, ""):
        return (), False
    if not isinstance(raw, str):
        return (), True

    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("claude-mem: %s column is not valid JSON; reading it as empty", column)
        return (), True

    if not isinstance(parsed, list):
        logger.debug(
            "claude-mem: %s column is a JSON %s, not a list", column, type(parsed).__name__
        )
        return (), True

    return tuple(str(item).strip() for item in parsed if str(item).strip()), False


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


@dataclass(frozen=True)
class Observation:
    """One claude-mem observation, mapped onto a vault project.

    `project` is the vault's canonical name and `raw_project` is what the database said. Both
    are kept because the alias map is the boundary control: a reader of a digest should be able
    to see which raw key was folded into which canonical project without re-querying.
    """

    project: str
    raw_project: str
    type: str
    title: str
    subtitle: str
    narrative: str
    facts: tuple[str, ...]
    concepts: tuple[str, ...]
    files_modified: tuple[str, ...]
    created_at: str
    created_at_epoch: int
    malformed: tuple[str, ...] = ()

    def summary_line(self) -> str:
        """One line for a prompt or a log: what it is, and what it was about."""
        head = self.title or self.narrative or self.type
        tail = f" — {self.subtitle}" if self.subtitle else ""
        return f"{self.project} [{self.type}] {head}{tail}"


@dataclass(frozen=True)
class SessionSummary:
    """One claude-mem session summary. Every field is plain prose, not JSON."""

    project: str
    raw_project: str
    request: str
    investigated: str
    learned: str
    completed: str
    next_steps: str
    notes: str
    created_at: str
    created_at_epoch: int

    def summary_line(self) -> str:
        head = self.request or self.completed or self.learned
        return f"{self.project}: {head}"


@dataclass(frozen=True)
class SkippedProject:
    """A project key the vault's alias map does not name, and how much of it was left behind.

    Counted rather than merely dropped. The counts are what tell an author that a real project
    is missing an alias entry, as opposed to the work projects that are *meant* to be skipped.
    """

    project: str
    observations: int = 0
    summaries: int = 0

    @property
    def rows(self) -> int:
        return self.observations + self.summaries


@dataclass(frozen=True)
class ObservationWindow:
    """Everything claude-mem had to say about one window, for one vault.

    `degraded` is set when the database could not be read at all. It is a string rather than a
    flag because the reason is what a human needs: a missing file and a locked one call for
    different responses, and the pass still runs either way.
    """

    since: date
    until: date
    scale: EpochScale = EpochScale.UNKNOWN
    observations: tuple[Observation, ...] = ()
    summaries: tuple[SessionSummary, ...] = ()
    skipped: tuple[SkippedProject, ...] = ()
    degraded: str | None = None

    @property
    def available(self) -> bool:
        return self.degraded is None

    @property
    def empty(self) -> bool:
        return not self.observations and not self.summaries

    @property
    def projects(self) -> tuple[str, ...]:
        """Canonical projects with material in this window, in first-seen order."""
        seen: dict[str, None] = {}
        for item in self.observations:
            seen.setdefault(item.project, None)
        for summary in self.summaries:
            seen.setdefault(summary.project, None)
        return tuple(seen)

    def skipped_line(self) -> str:
        """One line naming what was left outside the vault's boundary, or an empty string."""
        if not self.skipped:
            return ""
        return ", ".join(f"{item.project} ({item.rows})" for item in self.skipped)


def resolve_db_path(path: str | Path | None = None) -> Path:
    """Expand a configured or default path. No I/O, so a missing file is not an error yet."""
    return Path(path if path is not None else DEFAULT_DB_PATH).expanduser()


def read_only_uri(path: Path) -> str:
    """The connection URI for a read-only open.

    `mode=ro` and nothing else. `immutable=1` would be faster and would silently skip the WAL,
    which on a WAL-mode database means dropping the newest sessions — precisely the ones a
    weekly pass is about. The path is percent-encoded so a `?` or `#` in it cannot terminate
    the path component and turn part of the filename into a query parameter.
    """
    return f"file:{quote(str(path))}?mode=ro"


@contextmanager
def open_database(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the database read-only, or raise `ClaudeMemError` saying why it could not be.

    A missing file is caught before connecting: SQLite's URI `mode=ro` reports it as a generic
    "unable to open database file", which reads like a permissions problem and sends a reader
    looking in the wrong place.
    """
    if not path.exists():
        raise ClaudeMemError(
            f"the claude-mem database was not found at {path}. Reflection continues on the "
            "vault's own filed material only."
        )

    try:
        connection = sqlite3.connect(read_only_uri(path), uri=True, timeout=CONNECT_TIMEOUT_SECONDS)
    except sqlite3.Error as exc:
        raise ClaudeMemError(f"{path}: could not be opened read-only: {exc}") from exc

    connection.row_factory = sqlite3.Row
    try:
        yield connection
    except sqlite3.Error as exc:
        raise ClaudeMemError(f"{path}: query failed: {exc}") from exc
    finally:
        connection.close()


def detect_epoch_scale(conn: sqlite3.Connection) -> EpochScale:
    """Sample the newest timestamp and decide what unit this database counts in.

    Sampled rather than assumed. The columns are milliseconds today; if a plugin migration ever
    normalizes them to seconds, an assumed scale would turn every query into "match everything"
    and the pass would cheerfully reflect on the entire history as though it were one week.
    """
    newest: int | None = None
    for table in ("observations", "session_summaries"):
        try:
            row = conn.execute(f"SELECT MAX(created_at_epoch) AS newest FROM {table}").fetchone()
        except sqlite3.Error as exc:
            logger.debug("claude-mem: could not sample %s for its epoch scale: %s", table, exc)
            continue
        value = row["newest"] if row is not None else None
        if isinstance(value, int) and (newest is None or value > newest):
            newest = value

    if newest is None:
        logger.debug("claude-mem: no rows to sample; assuming milliseconds")
        return EpochScale.UNKNOWN

    scale = EpochScale.MILLISECONDS if newest >= MILLISECOND_FLOOR else EpochScale.SECONDS
    if scale is EpochScale.SECONDS:
        logger.warning(
            "claude-mem: timestamps sampled at %d look like seconds, not the milliseconds this "
            "database has always used. Scaling the window to match rather than matching every row.",
            newest,
        )
    return scale


def _day_start_ms(day: date) -> int:
    """Local midnight as a millisecond epoch. Local, because a window is a human's week."""
    return int(datetime.combine(day, time.min).timestamp() * 1000)


def window_bounds(since: date, until: date, scale: EpochScale) -> tuple[int, int]:
    """Half-open epoch bounds for an inclusive day range, in this database's own unit."""
    divisor = scale.divisor
    return _day_start_ms(since) // divisor, _day_start_ms(until + timedelta(days=1)) // divisor


def _select(table: str, columns: Sequence[str]) -> str:
    return (
        f"SELECT {', '.join(columns)} FROM {table} "  # noqa: S608 - column names are literals
        f"WHERE created_at_epoch >= ? AND created_at_epoch < ? AND {_UNMERGED} "
        "ORDER BY created_at_epoch ASC"
    )


def _observation(row: sqlite3.Row, project: str) -> Observation:
    facts, facts_bad = _json_list(row["facts"], "facts")
    concepts, concepts_bad = _json_list(row["concepts"], "concepts")
    files, files_bad = _json_list(row["files_modified"], "files_modified")
    malformed = tuple(
        name
        for name, bad in (
            ("facts", facts_bad),
            ("concepts", concepts_bad),
            ("files_modified", files_bad),
        )
        if bad
    )
    return Observation(
        project=project,
        raw_project=_text(row["project"]),
        type=_text(row["type"]),
        title=_text(row["title"]),
        subtitle=_text(row["subtitle"]),
        narrative=_text(row["narrative"]),
        facts=facts,
        concepts=concepts,
        files_modified=files,
        created_at=_text(row["created_at"]),
        created_at_epoch=int(row["created_at_epoch"]),
        malformed=malformed,
    )


def _summary(row: sqlite3.Row, project: str) -> SessionSummary:
    return SessionSummary(
        project=project,
        raw_project=_text(row["project"]),
        request=_text(row["request"]),
        investigated=_text(row["investigated"]),
        learned=_text(row["learned"]),
        completed=_text(row["completed"]),
        next_steps=_text(row["next_steps"]),
        notes=_text(row["notes"]),
        created_at=_text(row["created_at"]),
        created_at_epoch=int(row["created_at_epoch"]),
    )


class _Skips:
    """Tally of rows left outside the vault, by the raw project key that named them."""

    def __init__(self) -> None:
        self._observations: dict[str, int] = {}
        self._summaries: dict[str, int] = {}

    def observation(self, project: str) -> None:
        self._observations[project] = self._observations.get(project, 0) + 1

    def summary(self, project: str) -> None:
        self._summaries[project] = self._summaries.get(project, 0) + 1

    def collect(self) -> tuple[SkippedProject, ...]:
        names = sorted(set(self._observations) | set(self._summaries))
        return tuple(
            SkippedProject(
                project=name,
                observations=self._observations.get(name, 0),
                summaries=self._summaries.get(name, 0),
            )
            for name in names
        )


def fetch_window(
    vault: VaultConfig,
    since: date,
    until: date,
    *,
    db_path: str | Path | None = None,
) -> ObservationWindow:
    """Read one window of observations and session summaries for this vault's projects.

    Never raises for an unreadable database: a missing or locked file comes back as a window
    carrying `degraded`, so the weekly pass falls through to vault-only reflection with a
    warning instead of dying on a dependency it does not control.

    Both tables are queried standalone. Neither joins `sdk_sessions`, and neither should ever
    be made to — see the module docstring for the four-fifths of the corpus that would vanish.

    A vault that has not enabled this source gets an empty window that is *available* rather
    than degraded (R11), and nothing here opens a file. That distinction is the whole toggle:
    `degraded` turns a reflect run PARTIAL, and a third-party plugin the adopter never
    installed must not make every weekly pass report a problem.
    """
    if not vault.claude_mem_enabled:
        logger.debug(
            "vault %r: claude-mem is off; reflecting on the vault's own material only",
            vault.name,
        )
        return ObservationWindow(since=since, until=until)

    path = resolve_db_path(db_path)

    try:
        with open_database(path) as conn:
            scale = detect_epoch_scale(conn)
            start, end = window_bounds(since, until, scale)

            observation_rows = conn.execute(
                _select("observations", _OBSERVATION_COLUMNS), (start, end)
            ).fetchall()
            summary_rows = conn.execute(
                _select("session_summaries", _SUMMARY_COLUMNS), (start, end)
            ).fetchall()
    except ClaudeMemError as exc:
        logger.warning("claude-mem unavailable: %s", exc)
        return ObservationWindow(since=since, until=until, degraded=str(exc))

    skips = _Skips()
    observations: list[Observation] = []
    summaries: list[SessionSummary] = []

    for row in observation_rows:
        raw = _text(row["project"])
        canonical = vault.canonical_project(raw)
        if canonical is None:
            skips.observation(raw)
            continue
        observations.append(_observation(row, canonical))

    for row in summary_rows:
        raw = _text(row["project"])
        canonical = vault.canonical_project(raw)
        if canonical is None:
            skips.summary(raw)
            continue
        summaries.append(_summary(row, canonical))

    skipped = skips.collect()
    if skipped:
        logger.info(
            "claude-mem: skipped %d row(s) from project(s) outside vault %r: %s",
            sum(item.rows for item in skipped),
            vault.name,
            ", ".join(f"{item.project} ({item.rows})" for item in skipped),
        )

    return ObservationWindow(
        since=since,
        until=until,
        scale=scale,
        observations=tuple(observations),
        summaries=tuple(summaries),
        skipped=skipped,
    )
