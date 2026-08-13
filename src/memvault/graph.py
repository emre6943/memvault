"""The graph, read back out of the Markdown that holds it.

Why parse rather than record: the pipeline that files a note could write its edges straight into
the database as it goes, and then deleting the database would lose them. Every edge, every typed
observation, and every ranking fact therefore has a home in the file — a body line a person can
edit, or a frontmatter key — and this module is the one place that turns those bytes back into
rows (R1). Nothing here reads the index, so the parse is a pure function of one file's text.

Two shapes are recognized, both deliberately typed:

    - [decision] Went with the erfpacht buyout #house
    - part_of [[repos/MemVault/learnings.md]]

A bare `[[wikilink]]` in a sentence is prose, not an edge. Requiring a predicate is what keeps
the graph a set of statements somebody meant to make, rather than a by-product of writing in a
vault where linking is cheap. A single-letter category is likewise not a type: `- [x] done` is a
Markdown checkbox, and reading it as an observation would fill the table with task list noise.

Fences are respected for the same reason `chunk_markdown` respects them — a documentation file
showing what a relation line looks like must not thereby claim the relation.

Resolution is not parsing. Whether `[[ranking]]` names a file is a question about the whole
corpus, and the corpus changes under an incremental index: the target may be filed a week after
the note that points at it. So `resolve_link` is a separate pure function over a snapshot of the
corpus, which the indexer recomputes globally after every pass. A link that resolves to nothing
is pending, never an error — a note may legitimately point at something not written yet.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime
from typing import Any

import yaml

from memvault.inbox import FRONTMATTER_FENCE, FRONTMATTER_TERMINATORS

#: The band `importance` lives in, matching the classifier's. Repeated as literals rather than
#: imported so this parser stays independent of the writing path: a hand-edited `importance: 8`
#: must read the same whether or not a model ever saw the file.
IMPORTANCE_MIN, IMPORTANCE_MAX = 1, 10

#: `- [category] text`. The category is an identifier of at least two characters: one character
#: is a checkbox (`- [x]`, `- [ ]`), and a category with a space in it is a Markdown link label.
_OBSERVATION_RE = re.compile(r"^\s*[-*]\s+\[(?P<category>[a-z][a-z0-9_-]+)\]\s+(?P<text>\S.*?)\s*$")

#: `- predicate [[target]]`, with anything after the link ignored so a hand-written explanation
#: does not cost the edge. The predicate is lowercase snake_case, which is what the classifier
#: normalizes to and what distinguishes a relation from a sentence that happens to hold a link.
_RELATION_RE = re.compile(
    r"^\s*[-*]\s+(?P<predicate>[a-z][a-z0-9_]*)\s+\[\[(?P<target>[^\[\]]*)\]\]"
)

#: `#tag`: a letter or underscore first, so `#2026-08` in a sentence about a month is a date
#: rather than a tag, and not preceded by a word character or another `#`, so neither a URL
#: fragment nor a `##` heading marker becomes one.
_TAG_RE = re.compile(r"(?<![\w#])#([^\W\d][\w/-]*)", re.UNICODE)

_ISO_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


@dataclass(frozen=True)
class Observation:
    """One typed statement lifted out of a note body.

    `text` is the line as its author wrote it, tags included. Stripping the tags out would render
    a sentence nobody typed, and the file is what a result ultimately points at.
    """

    category: str
    text: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Link:
    """One typed edge out of a file, with its target still as text.

    Unresolved on purpose: `target` is what the file says, and what it points at is decided later
    against the whole corpus. Storing only the resolved path would make a link that resolves
    later indistinguishable from one that never resolved.
    """

    predicate: str
    target: str


@dataclass(frozen=True)
class FileMetadata:
    """The frontmatter facts ranking reads, coerced but never invented.

    Every field is optional and `None` means nobody said — which is not the same as a low score
    or an old date. A corpus filed before these keys existed must rank as unjudged rather than as
    unimportant.
    """

    title: str | None = None
    kind: str | None = None
    importance: int | None = None
    date: str | None = None
    superseded_by: str | None = None


@dataclass(frozen=True)
class ParsedFile:
    """Everything one file contributes to the graph."""

    metadata: FileMetadata = FileMetadata()
    links: tuple[Link, ...] = ()
    observations: tuple[Observation, ...] = ()


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Separate a leading YAML frontmatter block from the body it heads.

    Malformed or unterminated frontmatter degrades to "no metadata, all body" rather than
    raising: a broken header is a reason to know less about a file, never a reason to drop it out
    of the index. The body is returned unchanged so line-based parsing sees exactly what a reader
    of the file sees below the fence.
    """
    if not text.startswith(FRONTMATTER_FENCE):
        return {}, text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return {}, text
    for position in range(1, len(lines)):
        if lines[position].strip() in FRONTMATTER_TERMINATORS:
            block = "".join(lines[1:position])
            body = "".join(lines[position + 1 :])
            try:
                parsed = yaml.safe_load(block)
            except yaml.YAMLError:
                return {}, body
            return (parsed if isinstance(parsed, dict) else {}), body
    return {}, text


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _importance(value: Any) -> int | None:
    """Read an importance score, or decide nobody gave one.

    Tolerant in the same direction as the classifier's coercer: a string of digits is a number,
    and anything out of band is no answer rather than a clamped one. Booleans are excluded before
    the int check because `True` is an `int` in Python and `importance: true` says nothing.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            return None
    if not isinstance(value, int):
        return None
    return value if IMPORTANCE_MIN <= value <= IMPORTANCE_MAX else None


def _day(value: Any) -> str | None:
    """Normalize a frontmatter date to an ISO day string.

    YAML parses an unquoted `2026-08-12` into a `date` and a timestamp into a `datetime`, so a
    file written by hand and a file written by the pipeline arrive here as different types for
    the same fact.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date_type):
        return value.isoformat()
    if isinstance(value, str):
        match = _ISO_DAY_RE.match(value.strip())
        if match is not None:
            try:
                return date_type(int(match[1]), int(match[2]), int(match[3])).isoformat()
            except ValueError:
                return None
    return None


def parse_metadata(frontmatter: Mapping[str, Any]) -> FileMetadata:
    """Coerce the frontmatter keys ranking reads, leaving every unusable value absent.

    `ingested` stands in for `date` because a filed transcript that declared no material date is
    still dated by the day it entered the vault — the same fallback `writer._material_date` uses
    when it writes the file.
    """
    return FileMetadata(
        title=_text(frontmatter.get("title")),
        kind=_text(frontmatter.get("kind")),
        importance=_importance(frontmatter.get("importance")),
        date=_day(frontmatter.get("date")) or _day(frontmatter.get("ingested")),
        superseded_by=_text(frontmatter.get("superseded_by")),
    )


def iter_content_lines(body: str) -> Iterator[str]:
    """Yield the body's lines with fenced blocks left out.

    A file documenting the syntax — this repo's own docs do — would otherwise contribute the
    example edges it is describing, and a vault whose graph is half examples is worse than one
    with no graph at all.
    """
    fence: str | None = None
    for line in body.splitlines():
        stripped = line.strip()
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            continue
        yield line


def clean_target(target: str) -> str:
    """Reduce a wikilink's inside to the name it points at.

    Obsidian-style display aliases (`[[path|shown as]]`) and heading anchors (`[[note#section]]`)
    both address a file with decoration attached; the decoration is for a reader and the file is
    what the index needs. Returns `""` when nothing addressable is left, which the caller reads
    as "not a link" rather than as a link to the vault root.
    """
    cleaned = target.split("|", 1)[0].split("#", 1)[0].strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:].lstrip()
    return cleaned


def parse_body(body: str) -> tuple[tuple[Link, ...], tuple[Observation, ...]]:
    """Read a note body into its typed edges and typed statements, in order of appearance.

    Order is the file's own, which is what makes a rebuilt index compare equal to an incremental
    one row for row rather than merely set for set.
    """
    links: list[Link] = []
    observations: list[Observation] = []

    for line in iter_content_lines(body):
        relation = _RELATION_RE.match(line)
        if relation is not None:
            target = clean_target(relation.group("target"))
            if target:
                links.append(Link(predicate=relation.group("predicate"), target=target))
            continue

        observation = _OBSERVATION_RE.match(line)
        if observation is not None:
            text = observation.group("text").strip()
            observations.append(
                Observation(
                    category=observation.group("category"),
                    text=text,
                    tags=tuple(dict.fromkeys(_TAG_RE.findall(text))),
                )
            )

    return tuple(links), tuple(observations)


def parse_file(text: str) -> ParsedFile:
    """Everything one vault file contributes to the graph: its facts, its edges, its statements."""
    frontmatter, body = split_frontmatter(text)
    links, observations = parse_body(body)
    return ParsedFile(metadata=parse_metadata(frontmatter), links=links, observations=observations)


def normalize_alias(text: str) -> str:
    """Fold a name to the form two spellings of it share: case and inner whitespace."""
    return " ".join(text.casefold().split())


def _aliases_for(path: str, title: str | None) -> Iterator[str]:
    without_extension = path[:-3] if path.endswith(".md") else path
    yield without_extension
    yield without_extension.rsplit("/", 1)[-1]
    if title:
        yield title


def build_alias_index(
    entries: Iterable[tuple[str, str | None]],
) -> dict[str, tuple[str, ...]]:
    """Map every name a file can be linked by to the files answering to it.

    A name claimed by two files stays claimed by both: the ambiguity is the answer, and picking
    one would make the resolved graph depend on which file was indexed first.
    """
    claims: dict[str, list[str]] = {}
    for path, title in entries:
        for alias in _aliases_for(path, title):
            normalized = normalize_alias(alias)
            if not normalized:
                continue
            holders = claims.setdefault(normalized, [])
            if path not in holders:
                holders.append(path)
    return {alias: tuple(sorted(holders)) for alias, holders in claims.items()}


def resolve_link(
    target: str,
    *,
    paths: Set[str],
    aliases: Mapping[str, Sequence[str]],
) -> str | None:
    """Which file this link names, or `None` when the corpus cannot say.

    A vault-relative path wins outright, with or without the `.md`, because a path is an
    unambiguous statement and a title is a hopeful one. Only then does a name match: unique, or
    pending. Pending is a normal state — the target may not be written yet, or two files may
    share a title — and it is recomputed on every pass, so it is never permanent.
    """
    cleaned = clean_target(target)
    if not cleaned:
        return None
    for candidate in (cleaned, f"{cleaned}.md"):
        if candidate in paths:
            return candidate
    holders = aliases.get(normalize_alias(cleaned), ())
    return holders[0] if len(holders) == 1 else None
