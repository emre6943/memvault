"""Filing a routed record into a vault, with the provenance that explains it.

A vault file has to answer, months later and without the pipeline that produced it, one
question: where did this come from? So every file this module writes opens with frontmatter
naming its source, the drop it was read from, the content id that identifies it, and the model
whose judgement filed it there (R4). Nothing about that is reconstructible afterwards, which is
why it is written at filing time rather than derived later.

Three properties are worth stating up front, because each one exists to survive a failure.

**Writes are atomic.** Content goes to a temp file in the destination directory and is moved
into place with `os.replace`. A crash therefore leaves either the whole file or no file, never
half of one — the half that matters for KTD5, whose other half is U6 removing the inbox entry
only after this returns. A vault file that exists is a vault file that is complete.

**Collisions suffix, never overwrite.** Two drops with the same title on the same day are two
memories. The second gets `-2` and both survive; a vault that silently overwrote one of them
would lose material with no way to notice.

**`local_only` needs nothing from this module.** The destination arrives with the gitignored
prefix already applied by `route.py` (KTD7), so there is no second place for the rule to be
remembered and no way for this module to forget it. It only renders the template it was handed.

Slugs keep Turkish as Turkish. Folding `ş` to `s` and `ı` to `i` would collapse `açık` and
`acik` onto one filename, and a vault whose content is half Turkish would quietly stack
unrelated memories on the same name. Letters and digits survive in whatever script they are
written in; everything else — separators, punctuation, emoji, control characters — becomes a
hyphen, which is what makes the result filesystem-safe without making it lossy.
"""

from __future__ import annotations

import logging
import os
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from memvault.classify import Classification, Relation
from memvault.inbox import InboxRecord
from memvault.route import Destination, DestinationKind, RoutingResult
from memvault.segment import Segment

logger = logging.getLogger(__name__)

#: The placeholders a path template may use. Anything else is a config error, reported with
#: this list rather than as a `KeyError` from inside `str.format`.
TEMPLATE_PLACEHOLDERS = ("yyyy", "mm", "dd", "date", "slug", "ww")

#: Frontmatter key order: what the file *is* first, how it *got here* second. Fixed rather
#: than sorted, so two files filed a year apart read the same way and a diff of one stays
#: legible. Keys whose value is absent are dropped, never written empty.
FRONTMATTER_ORDER = (
    "title",
    "date",
    # What sort of file this is. Only a topic note declares it; a filed transcript's kind is
    # evident from where it lives, and adding it retroactively would rewrite every existing file.
    "kind",
    "source",
    "classification",
    "project",
    # Topic-note routing: which area received it, and what decided that — a workspace the feeder
    # declared, a project the classifier named, or the model's own choice. Recorded so a note
    # that looks misfiled can be explained rather than guessed at.
    "area",
    "area_via",
    "participants",
    "tags",
    "summary",
    # How much this deserves to resurface, 1-10. The classifier stamps it and a declared value
    # overrides it; absent is a real answer and means nobody judged, which recall reads as
    # neutral rather than as unimportant.
    "importance",
    # The file that replaced this one, vault-relative. Never written by the filing path — a drop
    # being filed cannot know what it obsoletes — but recognized here so a hand- or
    # reflection-written supersession is a first-class key rather than a stowaway.
    "superseded_by",
    "local_only",
    "ingested",
    "original_filename",
    "content_id",
    # A topic note's link back to the raw transcript it summarizes, vault-relative. The note is
    # additive and the transcript is the truth, so this is the field that makes the note
    # followable rather than a second, thinner copy.
    "source_path",
    # Segment provenance. Absent entirely on a drop that was never split, so filing an
    # unsegmented drop produces exactly the frontmatter it produced before segmentation
    # existed — and `parent_id` present is therefore a reliable signal that this file is one
    # part of something larger, whose siblings share its `parent_id`.
    "parent_id",
    "segment_index",
    "segment_count",
    "classifier_model",
    "confidence",
)

#: Long enough to stay readable, short enough that the rendered filename — date prefix, slug,
#: collision suffix, extension — clears the 255-byte limit even when every character is a
#: multi-byte one.
SLUG_MAX_LENGTH = 80

#: A slug is only ever this many collisions deep before something is clearly wrong; probing
#: forever would turn a bug into a hang.
MAX_COLLISION_SUFFIX = 999

#: What a block of relation lines is called in a note body. A heading rather than bare bullets,
#: because the lines are read by people as often as by the indexer and `- part_of [[x]]` under
#: no heading reads as a typo.
RELATIONS_HEADING = "## Relations"


class WriteError(Exception):
    """Raised when a file cannot be filed: a bad template, an escaping path, a failed write."""


@dataclass(frozen=True)
class Placement:
    """Where a document actually landed.

    `suffix` is the number appended to resolve a collision, and 0 when the first candidate
    name was free. A caller that wants to notice repeated titles reads it; a caller that only
    wants the path ignores it.
    """

    path: Path
    relative_path: str
    suffix: int = 0

    @property
    def collided(self) -> bool:
        return self.suffix > 0


@dataclass(frozen=True)
class WrittenFile:
    """One filed vault file, described for the pass that has to commit it.

    Carries enough to write a commit message without re-reading the file: which vault, which
    path, what it is called, and what it was classified as.
    """

    vault: str
    kind: DestinationKind
    path: Path
    relative_path: str
    record_id: str
    title: str
    slug: str
    classification: str | None = None
    local_only: bool = False
    suffix: int = 0

    @property
    def collided(self) -> bool:
        """Whether the first candidate name was taken and this file was suffixed."""
        return self.suffix > 0

    def summary_line(self) -> str:
        """One line for a commit message body."""
        return f"{self.title} — {self.relative_path}"


class _FrontmatterDumper(yaml.SafeDumper):
    """A dumper that indents sequence items under their key.

    PyYAML's default puts the dash at the parent's indentation, which is valid YAML that no
    one writes by hand. Vault frontmatter is read by people far more often than by this
    program, so it is written the way the vault's existing files are.
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


def slugify(text: str, *, max_length: int = SLUG_MAX_LENGTH) -> str:
    """Reduce a title to a filesystem-safe slug, preserving distinctions between titles.

    Letters and digits survive in any script — `ş`, `ğ`, `ı`, `ö`, `ü`, `ç` included, because
    folding them to ASCII is what collapses `açık` and `acik` onto one name. Combining marks
    are dropped; everything else becomes a hyphen, so path separators, quotes, colons, emoji,
    and control characters all leave without taking their neighbours with them.

    `İ` is mapped to `i` before lowercasing. Python's own `"İ".lower()` yields `i` followed by
    a combining dot, which would otherwise be dropped as a mark and leave a bare `i` anyway —
    doing it deliberately keeps the intent visible.

    Returns `""` when nothing survives. The caller decides on a fallback; inventing one here
    would hide the case from a reader of this function.
    """
    lowered = unicodedata.normalize("NFC", text).replace("İ", "i").lower()
    normalized = unicodedata.normalize("NFC", lowered)

    pieces: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category[0] in ("L", "N"):
            pieces.append(char)
        elif category[0] == "M":
            continue
        else:
            pieces.append("-")

    slug = "-".join(part for part in "".join(pieces).split("-") if part)
    if len(slug) <= max_length:
        return slug

    clipped = slug[:max_length]
    # Prefer a whole-word cut, but only when one is available: a single very long word must
    # still be truncated rather than slugging to nothing.
    head, _, _ = clipped.rpartition("-")
    return (head or clipped).strip("-")


def render_path(template: str, *, day: date, slug: str) -> str:
    """Render a vault-relative path from a configured template.

    The template comes from config, so a mistake in it is an authoring error and is reported
    as one: the offending placeholder and the supported set, not a bare `KeyError` from
    somewhere inside `str.format`.

    `{ww}` is the ISO week number, which in the first and last days of a year belongs to a
    different ISO year than `{yyyy}`. A template using both across a year boundary should
    expect that; the alternative — an ISO year that disagrees with the file's own `date` — is
    more surprising, not less.
    """
    values = {
        "yyyy": f"{day.year:04d}",
        "mm": f"{day.month:02d}",
        "dd": f"{day.day:02d}",
        "date": day.isoformat(),
        "ww": f"{day.isocalendar().week:02d}",
        "slug": slug,
    }

    try:
        rendered = template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        raise WriteError(
            f"path template {template!r} could not be rendered ({exc}). "
            f"Supported placeholders: {', '.join('{' + p + '}' for p in TEMPLATE_PLACEHOLDERS)}"
        ) from exc

    relative = PurePosixPath(rendered)
    if relative.is_absolute() or ".." in relative.parts:
        raise WriteError(
            f"path template {template!r} rendered to {rendered!r}, which leaves the vault. "
            "Templates are vault-relative and may not contain '..' or a leading '/'."
        )
    if rendered.endswith("/") or not relative.name:
        raise WriteError(
            f"path template {template!r} rendered to {rendered!r}, which names no file"
        )

    return relative.as_posix()


def render_document(frontmatter: Mapping[str, Any], body: str) -> str:
    """Assemble a Markdown document: generated frontmatter, then the body unchanged.

    The body is written verbatim — U2 already normalized it, and re-normalizing here would
    make the filed file disagree with the content id in its own frontmatter.
    """
    fields = {
        key: frontmatter[key]
        for key in FRONTMATTER_ORDER
        if frontmatter.get(key) not in (None, "", (), [])
    }
    unknown = sorted(set(frontmatter) - set(FRONTMATTER_ORDER))
    if unknown:
        raise WriteError(
            f"frontmatter key(s) with no place in the fixed order: {', '.join(unknown)}. "
            f"Add them to FRONTMATTER_ORDER rather than relying on insertion order."
        )

    rendered = yaml.dump(
        fields,
        Dumper=_FrontmatterDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1000,
    )

    document = f"---\n{rendered}---\n"
    return f"{document}\n{body}\n" if body else document


def render_relations(relations: Sequence[Relation]) -> str:
    """Render relation lines as a Markdown block, or nothing at all when there are none.

    The block is what turns a note into a node: the indexer parses `- <predicate> [[target]]`
    out of the body, so the graph has a home in the file rather than only in a table that a
    rebuild would have to invent. An empty tuple renders to `""` rather than to a bare heading —
    a section promising links and listing none is worse than no section.
    """
    if not relations:
        return ""
    return RELATIONS_HEADING + "\n\n" + "\n".join(relation.render() for relation in relations)


def importance_of(record: InboxRecord, verdict: Classification) -> int | None:
    """The importance to file this item with: what the drop declared, else what the model said.

    Declared metadata overrides inference for its own field and nothing else (R2), and this is
    the field's third writer — the raw transcript, the topic note, and the distilled note all
    have to agree, so they all ask here.
    """
    if record.declared.declares("importance"):
        return record.declared.importance
    return verdict.importance


def _free_path(root: Path, relative_path: str) -> tuple[Path, str, int]:
    """Find the first unused name for this path, suffixing numerically on collision.

    Single-writer by assumption: one ingestion pass runs at a time, so probing and then
    writing is enough. Two concurrent passes over one vault would be a problem well before
    this line.
    """
    relative = PurePosixPath(relative_path)
    stem, suffix_ext = relative.stem, relative.suffix

    for attempt in range(MAX_COLLISION_SUFFIX):
        # The first candidate carries no suffix; the second is `-2`, so the number in the name
        # is the number of files that share the title, which is what a reader would expect.
        suffix = 0 if attempt == 0 else attempt + 1
        name = f"{stem}{suffix_ext}" if suffix == 0 else f"{stem}-{suffix}{suffix_ext}"
        candidate = relative.with_name(name)
        path = root / candidate
        if not path.exists():
            return path, candidate.as_posix(), suffix

    raise WriteError(
        f"{relative_path}: more than {MAX_COLLISION_SUFFIX} files already share this name in "
        f"{root}. Something is filing the same item repeatedly."
    )


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` via a temp file in the same directory, then rename.

    `os.replace` is atomic within a filesystem, and the temp file is created beside its target
    so the two never straddle one. On any failure — including an interrupt — the temp file is
    removed, so a crash leaves the vault exactly as it was rather than holding a partial file
    that later looks like a filed memory.
    """
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def write_document(
    root: Path,
    relative_path: str,
    *,
    frontmatter: Mapping[str, Any],
    body: str,
) -> Placement:
    """Write one document into a vault, suffixing rather than overwriting on collision.

    The lower-level entry point: U5 files a distilled note through it with a body this module
    never sees. `file_record` is the raw-transcript path.
    """
    path, candidate, suffix = _free_path(root, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, render_document(frontmatter, body))

    if suffix:
        logger.info("%s: name was taken, filed as %s instead", relative_path, candidate)

    return Placement(path=path, relative_path=candidate, suffix=suffix)


def build_frontmatter(
    record: InboxRecord,
    verdict: Classification,
    routing: RoutingResult,
    *,
    day: date,
    ingested: date,
    local_only: bool,
    segment: Segment | None = None,
) -> dict[str, Any]:
    """Assemble the provenance block for a filed transcript.

    Absent fields are left out rather than written empty, so a reader can tell "nobody said"
    from "said nothing" — an empty `participants: []` would read as a positive claim that the
    material involves no one, which is exactly the claim a deletion request must not trust
    (R11). `local_only` appears only when true for the same reason: its absence is its default.

    Declared tags are re-applied here, as `route.py` re-applies declared participants: R2 holds
    at every layer that could write the field, not only at the one that usually does.
    """
    tags = record.declared.tags if record.declared.declares("tags") else verdict.tags
    return {
        "title": verdict.title,
        "date": day.isoformat(),
        "source": record.declared.source,
        "classification": routing.classification,
        "project": verdict.project,
        "participants": list(routing.participants),
        "tags": list(tags),
        "summary": verdict.summary,
        "importance": importance_of(record, verdict),
        "local_only": True if local_only else None,
        "ingested": ingested.isoformat(),
        "original_filename": record.filename,
        "content_id": record.id,
        "parent_id": None if segment is None or segment.is_whole_drop else segment.parent_id,
        "segment_index": None if segment is None or segment.is_whole_drop else segment.index,
        "segment_count": None if segment is None or segment.is_whole_drop else segment.count,
        "classifier_model": verdict.model,
        "confidence": round(verdict.confidence, 2),
    }


def _material_date(record: InboxRecord, ingested: date) -> date:
    """When the material happened, which is not when it was ingested.

    A declared `date` wins (R2); a drop that declared none is dated by the day it was filed,
    since guessing a date out of a transcript's contents is the classifier's business and it
    is not asked for one.
    """
    declared = record.declared.date
    if not declared:
        return ingested
    try:
        return date.fromisoformat(declared)
    except ValueError:
        # U2 validates the format, so reaching here means a record was assembled by hand.
        logger.warning(
            "%s: declared date %r is not YYYY-MM-DD; dating the file by its ingest day instead",
            record.filename,
            declared,
        )
        return ingested


def _slug_for(record: InboxRecord, verdict: Classification) -> str:
    """The classifier's slug, its title, or the content id — the first that survives.

    The id fallback is deliberately ugly. A file called `untitled-9f2c1a0b` is a visible
    signal that the classifier gave a title nothing could be made of, and it is still unique,
    stable across re-runs, and traceable back to the drop it came from.
    """
    for candidate in (verdict.slug, verdict.title):
        slug = slugify(candidate)
        if slug:
            return slug
    return f"untitled-{record.id[:8]}"


def file_record(
    record: InboxRecord,
    verdict: Classification,
    routing: RoutingResult,
    root: Path,
    *,
    destination: Destination | None = None,
    ingested: date | None = None,
    segment: Segment | None = None,
) -> WrittenFile:
    """File one drop's raw body into a vault, with provenance frontmatter.

    `destination` defaults to the routing result's single raw destination, which is what the
    ingestion pass wants; passing one explicitly is for a caller that routed several.

    A `NOTE` destination is refused rather than filed. Notes carry distilled text and this
    function writes the raw body — handing it a note destination would put raw material in the
    work vault, which is the one thing the routing table exists to prevent (R9).
    """
    if destination is None:
        raw = routing.of_kind(DestinationKind.RAW)
        if len(raw) != 1:
            raise WriteError(
                f"{record.filename}: expected exactly one raw destination to file into, "
                f"got {len(raw)}. Name one explicitly with `destination=`."
            )
        destination = raw[0]

    if destination.kind is not DestinationKind.RAW:
        raise WriteError(
            f"{record.filename}: refusing to write the raw body to a "
            f"{destination.kind.value!r} destination in vault {destination.vault!r}. "
            "Distilled notes are written by the distiller, never by the transcript writer."
        )

    ingested = ingested or date.today()
    day = _material_date(record, ingested)
    slug = _slug_for(record, verdict)

    relative_path = render_path(destination.path_template, day=day, slug=slug)
    frontmatter = build_frontmatter(
        record,
        verdict,
        routing,
        day=day,
        ingested=ingested,
        local_only=destination.local_only,
        segment=segment,
    )

    placement = write_document(root, relative_path, frontmatter=frontmatter, body=record.body)

    logger.info("filed %s as %s/%s", record.filename, destination.vault, placement.relative_path)

    return WrittenFile(
        vault=destination.vault,
        kind=destination.kind,
        path=placement.path,
        relative_path=placement.relative_path,
        record_id=record.id,
        title=verdict.title,
        slug=slug,
        classification=routing.classification,
        local_only=destination.local_only,
        suffix=placement.suffix,
    )
