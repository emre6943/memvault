"""The distilled note filed into a topic area of the vault that already holds the raw drop.

**This is not `distill.py`, and must not become it.** That module exists to enforce R9 — raw
transcripts never reach the *work* vault — and pays for it with a six-word shingle leakage guard
and a second LLM call. A topic note lives in the same vault as its own raw body, one directory
away and linked on purpose, so there is nothing to guard against. Applying that guard here would
buy no privacy and cost fidelity: it would reject a good summary for quoting a phrase.

The two are told apart by the destination's vault, and a future refactor that "unifies" them
would silently delete the privacy guarantee. They are separate modules so that is hard to do by
accident.

**No model is called.** The classifier already returned a `summary` for this drop on the call that
classified it. Generating a second, differently-worded summary would cost a round trip per drop to
say the same thing.

**The note is additive.** It carries no information the vault does not already hold — it exists so
a topic directory shows what happened in it, and so `/recall` has something short to rank. That is
why its failure is survivable and why the caller treats a raised `WriteError` as "no note" rather
than "the drop failed".
"""

from __future__ import annotations

import logging
import posixpath
from datetime import date

from memvault.area import AreaChoice
from memvault.classify import Classification
from memvault.config import VaultConfig
from memvault.inbox import InboxRecord
from memvault.route import Destination, DestinationKind
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


def _relative_link(from_relative: str, to_relative: str) -> str:
    """A link from one vault-relative path to another, so it resolves in a Markdown viewer.

    The frontmatter's `source_path` is the vault-relative truth and the one a tool should read;
    this is the same fact spelled for a human who clicked into the note.
    """
    return posixpath.relpath(to_relative, start=posixpath.dirname(from_relative))


def write_topic_note(
    *,
    record: InboxRecord,
    verdict: Classification,
    destination: Destination,
    vault: VaultConfig,
    raw_path: str,
    area: AreaChoice,
    ingested: date,
) -> WrittenFile:
    """File one topic note. `raw_path` is vault-relative and becomes the note's back-link.

    `ingested` is passed rather than defaulted: `_material_date` dates the note by what the drop
    declared and falls back to the day it was filed, and a note must carry the same date as the
    transcript it describes — which means both have to be told the same day.
    """
    if destination.kind is not DestinationKind.NOTE:
        raise WriteError(
            f"{record.filename}: refusing to write a topic note into a "
            f"{destination.kind.value!r} destination. Raw bodies are filed by the transcript "
            "writer; a note destination is the only one that may receive a summary."
        )

    day = _material_date(record, ingested)
    slug = verdict.slug or slugify(verdict.title)
    relative = render_path(destination.path_template, day=day, slug=slug)

    frontmatter: dict[str, object] = {
        "title": verdict.title,
        "date": day.isoformat(),
        "kind": "topic-note",
        "area": area.name,
        # Why this area, so a note that looks misfiled can be explained rather than guessed at.
        "area_via": area.via,
        # The same id as the transcript: one drop, two files, one filed item to the id index.
        "content_id": record.id,
        "source_path": raw_path,
        "importance": importance_of(record, verdict),
    }
    if record.declared.source:
        frontmatter["source"] = record.declared.source

    summary = verdict.summary.strip()
    link = _relative_link(relative, raw_path)
    # The relations go last, under the transcript link rather than above it: the link is what a
    # person opens the note for, and the relation block is for the indexer and for whoever is
    # following the graph. Every target the classifier named is kept — this note lives in the
    # same vault as its own transcript, so there is no boundary here for a wikilink to cross.
    relations = render_relations(verdict.relations)
    body = (
        (f"{summary}\n\n" if summary else "")
        + f"Full transcript: [`{raw_path}`]({link})\n"
        + (f"\n{relations}\n" if relations else "")
    )

    placement = write_document(vault.root, relative, frontmatter=frontmatter, body=body)
    if placement.suffix:
        logger.info(
            "%s: topic note collided; filed as %s", record.filename, placement.relative_path
        )
    logger.info("filed %s into %s/%s", record.filename, area.name, placement.relative_path)

    return WrittenFile(
        vault=vault.name,
        kind=DestinationKind.NOTE,
        path=placement.path,
        relative_path=placement.relative_path,
        record_id=record.id,
        title=verdict.title,
        slug=slug,
        classification=verdict.classification,
        local_only=destination.local_only,
        suffix=placement.suffix,
    )
