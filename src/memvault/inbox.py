"""Inbox discovery, frontmatter parsing, and normalization.

The inbox is the engine's only entrance, and it is deliberately undemanding: a bare `.txt`
pasted from a phone must ingest as readily as a feeder-generated Markdown file carrying full
metadata (R1). Everything downstream — classification, routing, filing — reads the record this
module produces and never the file on disk.

Two properties earn the complexity here.

Identity is a SHA-256 over the *body*, taken after frontmatter is stripped and the text is
normalized. Metadata is therefore free to change without changing identity: the same transcript
re-dropped with a corrected date is still recognized as already filed, which is what makes a
re-run a no-op instead of a duplicate (R3).

Nothing usable is discarded quietly. A drop whose frontmatter cannot be used keeps its body and
carries the reason in `unparseable`, so the pass can hold it for review rather than guessing at
it or dropping it (R5). Only genuinely contentless or unreadable files are skipped, and every
skip is logged with its reason.

This module infers nothing. It records what a drop declares and leaves the rest empty for the
classifier (U3) to fill, which is what makes R2's "explicit metadata always wins" a property of
the data rather than a rule someone has to remember.

The file format accepted here is a published interface for future feeders: see
`docs/inbox-contract.md`.
"""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeAlias

import yaml

from memvault.config import VaultConfig

logger = logging.getLogger(__name__)

#: A frontmatter field reader: raw YAML value in, cleaned value plus an optional reason out.
Coercer: TypeAlias = Callable[[Any], tuple[Any, str | None]]

FRONTMATTER_FENCE = "---"
FRONTMATTER_TERMINATORS = frozenset({"---", "..."})

#: Extensions a drop may carry. An extensionless file is accepted too — a transcript saved
#: from a phone often has no extension at all.
TEXT_SUFFIXES = frozenset({".md", ".txt"})

#: The three classifications the router understands (R8). Declared elsewhere is a mistake worth
#: naming at the door rather than discovering in `route.py`.
CLASSIFICATIONS = frozenset({"personal", "work", "mixed"})

#: Frontmatter keys the engine reads (R2). Anything else is ignored, not rejected — a feeder
#: may carry its own bookkeeping and should not need engine changes to do so.
DECLARED_KEYS = frozenset(
    {
        "source",
        "date",
        "tags",
        "classification",
        "local_only",
        "participants",
        "workspace",
        "importance",
    }
)

#: The band a declared `importance` must fall in, matching the classifier's own scale. Named
#: here rather than imported from `classify` because this module is the door and knows nothing
#: about what happens behind it.
IMPORTANCE_RANGE = (1, 10)


@dataclass(frozen=True)
class DeclaredMetadata:
    """What a drop said about itself, and nothing more.

    `declared_keys` is the load-bearing field: it names which keys were actually present, so a
    consumer can suppress inference for exactly those fields (R2). A `classification` of None
    with `"classification"` absent from `declared_keys` means "nobody said" — distinct from a
    value that was declared and rejected, which shows up as an `unparseable` record instead.
    """

    source: str | None = None
    #: The feeder's own workspace name, when it has one. Chad sets it; Conversate, Apple Notes
    #: and manual drops do not. Purely a routing hint — an unrecognized value is ignored by
    #: routing rather than rejected here, so a feeder can rename its workspaces freely.
    workspace: str | None = None
    date: str | None = None
    tags: tuple[str, ...] = ()
    classification: str | None = None
    local_only: bool | None = None
    participants: tuple[str, ...] = ()
    #: How much this deserves to resurface, 1-10, when the author said so. An override like
    #: every other declared field: it suppresses the classifier's own score and nothing else.
    importance: int | None = None
    declared_keys: frozenset[str] = field(default_factory=frozenset)

    def declares(self, key: str) -> bool:
        """Whether the drop declared `key`, and inference for it should therefore be skipped."""
        return key in self.declared_keys


@dataclass(frozen=True)
class InboxRecord:
    """One drop, normalized.

    `id` identifies the content, not the file: two drops of the same body under different names
    and different metadata share an id. `unparseable` carries the reason when frontmatter could
    not be used; such a record is still returned, because the pipeline never silently discards a
    drop (R5) — holding it for review is the ingestion pass's job, not this module's.
    """

    id: str
    body: str
    declared: DeclaredMetadata
    filename: str
    size_bytes: int
    path: Path | None = None
    unparseable: str | None = None


def content_id(body: str) -> str:
    """Stable identity for a normalized body."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    """Collapse the differences that are not content.

    Line endings become LF so the same transcript saved on Windows and on macOS hashes alike.
    Unicode is composed to NFC so `İstanbul` typed on a Mac and on a phone agree — a real
    concern for a vault that holds Turkish. Surrounding blank lines go, because an editor
    adding a trailing newline is not a new memory. Interior whitespace and leading indentation
    survive untouched: a code block at the top of a drop is content.
    """
    lf = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", lf).strip("\n").rstrip()


def _flatten(value: object) -> str:
    """One-line rendering, so a multi-line YAML error still fits in a review marker."""
    return " ".join(str(value).split())


def _split_frontmatter(text: str) -> tuple[str | None, str, str | None]:
    """Separate a leading YAML frontmatter block from the body.

    Returns `(frontmatter_source, body, error)`. An unclosed fence yields the whole text as the
    body: without a terminator there is no way to tell metadata from content, and keeping
    everything is the only option that cannot lose a memory.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return None, text, None

    for index in range(1, len(lines)):
        if lines[index].strip() in FRONTMATTER_TERMINATORS:
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :]), None

    return (
        None,
        text,
        ("frontmatter opened with '---' but was never closed; the whole file was kept as body"),
    )


def _coerce_text(value: Any, key: str) -> tuple[str | None, str | None]:
    if isinstance(value, str) and value.strip():
        return value.strip(), None
    return None, f"frontmatter '{key}' must be a non-empty string, got {value!r}"


def _coerce_text_list(value: Any, key: str) -> tuple[tuple[str, ...], str | None]:
    if isinstance(value, str):
        stripped = value.strip()
        return ((stripped,) if stripped else ()), None
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip()), None
    return (), f"frontmatter '{key}' must be a string or a list of strings, got {value!r}"


def _coerce_date(value: Any) -> tuple[str | None, str | None]:
    """Accept what YAML hands back for a date, and insist on `YYYY-MM-DD` for anything else.

    An unquoted `date: 2026-08-01` arrives as a `datetime.date`, a quoted one as a string, and
    both must land on the same rendering — the vault dates everything `YYYY-MM-DD`.
    """
    if isinstance(value, datetime):
        return value.date().isoformat(), None
    if isinstance(value, date):
        return value.isoformat(), None
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date().isoformat(), None
        except ValueError:
            pass
    return None, f"frontmatter 'date' must be YYYY-MM-DD, got {value!r}"


def _coerce_local_only(value: Any) -> tuple[bool | None, str | None]:
    """Only a real boolean counts.

    Truthiness would happily read `local_only: "no"` as true and `local_only: 0` as false. This
    flag decides whether an item is filed somewhere git will never push it, so a coercion bug
    here is a disclosure, and refusing the value is cheap by comparison.
    """
    if isinstance(value, bool):
        return value, None
    return None, (
        f"frontmatter 'local_only' must be true or false, got {value!r}. It decides whether the "
        "item is filed where git can push it, so it is not guessed."
    )


def _coerce_importance(value: Any) -> tuple[int | None, str | None]:
    """Only a whole number in range counts.

    The classifier is tolerant with its own `importance` — a bad one there costs a ranking hint
    and nothing else. A declared one is different: it overrides the model, so a value the parser
    cannot read is a statement the author will never learn was ignored. Rejecting it by name
    costs one edit and keeps the override honest.

    Booleans are refused first: `True` is an `int` in Python and would file as importance 1.
    """
    low, high = IMPORTANCE_RANGE
    if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
        return value, None
    return None, (
        f"frontmatter 'importance' must be a whole number from {low} to {high}, got {value!r}"
    )


def _coerce_classification(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, str) and value.strip().lower() in CLASSIFICATIONS:
        return value.strip().lower(), None
    return None, (
        f"frontmatter 'classification' must be one of {', '.join(sorted(CLASSIFICATIONS))}, "
        f"got {value!r}"
    )


@dataclass(frozen=True)
class _Parsed:
    """Outcome of reading a fenced block.

    `was_frontmatter` is False when the block parsed cleanly into something that is not a
    mapping — a sign the fences were never metadata at all, most often a Markdown file opening
    with a horizontal rule. The caller then keeps the whole file as body rather than swallowing
    its first section.
    """

    declared: DeclaredMetadata
    error: str | None = None
    was_frontmatter: bool = True


def _parse_declared(source: str, filename: str) -> _Parsed:
    """Read a frontmatter block into declared metadata, collecting every problem it has.

    Problems accumulate rather than short-circuiting: a reviewer fixing a held drop should see
    all of what is wrong with it in one pass, not one error per re-run.
    """
    try:
        data: Any = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        return _Parsed(DeclaredMetadata(), f"frontmatter is not valid YAML: {_flatten(exc)}")

    if data is None:
        return _Parsed(DeclaredMetadata())
    if not isinstance(data, dict):
        return _Parsed(
            DeclaredMetadata(),
            (
                f"a '---' fenced block at the top of the file is not frontmatter: it holds "
                f"{type(data).__name__}, not a mapping. The whole file was kept as body."
            ),
            was_frontmatter=False,
        )

    unknown = sorted(str(key) for key in data if str(key) not in DECLARED_KEYS)
    if unknown:
        logger.info(
            "%s: ignoring frontmatter key(s) the engine does not read: %s",
            filename,
            ", ".join(unknown),
        )

    errors: list[str] = []

    def read(key: str, coerce: Coercer, absent: Any) -> Any:
        if key not in data:
            return absent
        value, error = coerce(data[key])
        if error:
            errors.append(error)
        return value

    declared = DeclaredMetadata(
        source=read("source", lambda v: _coerce_text(v, "source"), None),
        workspace=read("workspace", lambda v: _coerce_text(v, "workspace"), None),
        date=read("date", _coerce_date, None),
        tags=read("tags", lambda v: _coerce_text_list(v, "tags"), ()),
        classification=read("classification", _coerce_classification, None),
        local_only=read("local_only", _coerce_local_only, None),
        participants=read("participants", lambda v: _coerce_text_list(v, "participants"), ()),
        importance=read("importance", _coerce_importance, None),
        declared_keys=frozenset(str(key) for key in data if str(key) in DECLARED_KEYS),
    )
    return _Parsed(declared, "; ".join(errors) or None)


def normalize(
    text: str,
    filename: str,
    *,
    path: Path | None = None,
    size_bytes: int | None = None,
) -> InboxRecord:
    """Turn raw drop text into a record. Pure — no filesystem access.

    A record comes back for any input, including an empty one; deciding that a body is too
    empty to file belongs to `discover`, which is also where the skip gets logged.
    """
    normalized = _normalize_text(text)
    frontmatter, raw_body, error = _split_frontmatter(normalized)

    declared = DeclaredMetadata()
    if frontmatter is not None:
        parsed = _parse_declared(frontmatter, filename)
        declared, error = parsed.declared, parsed.error
        if not parsed.was_frontmatter:
            raw_body = normalized

    body = _normalize_text(raw_body)
    return InboxRecord(
        id=content_id(body),
        body=body,
        declared=declared,
        filename=filename,
        size_bytes=len(text.encode("utf-8")) if size_bytes is None else size_bytes,
        path=path,
        unparseable=error,
    )


def read_drop(path: Path, inbox: Path) -> InboxRecord | None:
    """Read one inbox file into a record, or return None and log why it was skipped.

    Skips are for files that are not drops at all — a hidden `.DS_Store`, a photo, an empty
    placeholder. Anything with UTF-8 text in it becomes a record, however broken its metadata.
    """
    if not path.is_file():
        return None

    relative = path.relative_to(inbox)
    name = relative.as_posix()

    if any(part.startswith(".") for part in relative.parts):
        logger.debug("%s: skipped, hidden entries are not drops", name)
        return None

    suffix = path.suffix.lower()
    if suffix and suffix not in TEXT_SUFFIXES:
        logger.info(
            "%s: skipped, %r is not a text drop (expected %s or no extension)",
            name,
            path.suffix,
            " or ".join(sorted(TEXT_SUFFIXES)),
        )
        return None

    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.warning("%s: skipped, could not be read: %s", name, _flatten(exc))
        return None

    try:
        # utf-8-sig also strips the BOM some editors prepend, which would otherwise make the
        # same content hash differently depending on where it was typed.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.info("%s: skipped, not UTF-8 text", name)
        return None

    record = normalize(text, name, path=path, size_bytes=len(raw))
    if not record.body:
        logger.info("%s: skipped, no content once frontmatter and blank lines were removed", name)
        return None

    return record


def discover(vault: VaultConfig) -> list[InboxRecord]:
    """Walk a vault's inbox, including subdirectories, and return every drop it holds.

    Ordering is by inbox-relative path so two runs over the same inbox produce the same list in
    the same order, which is what lets a pass's commit be compared against its predecessor.
    Records may share an id when the same content was dropped twice; de-duplication is the
    ingestion pass's call, not discovery's.
    """
    inbox = vault.inbox
    if not inbox.is_dir():
        logger.warning("vault %r: inbox not found at %s — nothing to discover", vault.name, inbox)
        return []

    candidates = sorted(inbox.rglob("*"), key=lambda p: p.relative_to(inbox).as_posix())
    records = [record for path in candidates if (record := read_drop(path, inbox)) is not None]

    logger.debug("vault %r: discovered %d drop(s) in %s", vault.name, len(records), inbox)
    return records
