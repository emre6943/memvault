"""Turning a classification into a set of destinations.

Pure by construction: a record, a verdict, and config go in; a list of destinations comes out.
Nothing here touches the filesystem, which is the point — the rule that protects the boundary
between the two vaults is then a property of a table that can be read and tested on its own,
rather than something inferred from what a writer happened to do afterwards.

The rule (R9), stated as the asymmetry it is: **raw content goes to the personal vault on every
branch, and to the work vault on none.** A `work` or `mixed` drop is filed raw exactly where a
`personal` one is, and the work vault receives only a distilled-note destination. U5 writes that
note; this module only decides that it is owed one.

A `NOTE` destination therefore comes in two shapes, told apart by its vault. One crosses into the
work vault under the rule above and is written by the leakage-guarded distiller. The other stays
in the vault being ingested, filing a summary into a topic area beside the raw body it links to —
no guard, because there is no boundary to guard. The two are decided independently: a `work` drop
can owe both, a `personal` drop can owe the second alone.

Two branches earn their own explanation.

`local_only` suppresses the work note entirely rather than routing it somewhere gitignored. The
flag means this drop does not leave the personal vault, and a distilled note in a work repo has
left it — even one git will not push, since the work vault is a shared context and its checkout
is on someone's disk. Raw filing still happens, under the gitignored prefix (KTD7).

A vault with no `work_route` configured files `work` material raw and notes why. A single-vault
install is a supported configuration (R21), so encountering work material there is a fact about
the setup, not an error: refusing would make the engine unusable for anyone with one vault, and
holding the item would fill the inbox with drops no configuration change is coming for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from memvault.area import AreaChoice
from memvault.classify import Classification, NeedsReview, Outcome
from memvault.config import Config, VaultConfig
from memvault.inbox import InboxRecord

logger = logging.getLogger(__name__)

#: The classifications that owe the work vault a distilled note. Membership is opt-in on
#: purpose: an unrecognized label falls through to raw-only, which is the safe branch.
DISTILLED_CLASSIFICATIONS = frozenset({"work", "mixed"})


class DestinationKind(StrEnum):
    """What kind of file a destination expects.

    `RAW` is the drop's own body. `NOTE` is a distilled note, and the two are never
    interchangeable: a `NOTE` destination must never be handed raw content.
    """

    RAW = "raw"
    NOTE = "note"


@dataclass(frozen=True)
class Destination:
    """One file this record owes, named by vault and path template.

    The template is already resolved for `local_only`, so a writer renders it without needing
    to remember the gitignore rule — the prefix is visible in the value it was handed.
    """

    vault: str
    kind: DestinationKind
    path_template: str
    local_only: bool = False


@dataclass(frozen=True)
class RoutingResult:
    """Where one record goes, or why it goes nowhere.

    `participants` rides along because a deletion request months later needs to find every file
    a named person appears in, and the filed frontmatter is the only place that will record it
    (R11). `notes` carries the decisions that produced no destination, so a skipped work write
    is explainable from the log rather than only from this module's source.
    """

    record_id: str
    classification: str | None
    destinations: tuple[Destination, ...] = ()
    participants: tuple[str, ...] = ()
    local_only: bool = False
    notes: tuple[str, ...] = ()
    review: str | None = None

    @property
    def held(self) -> bool:
        """Whether this record stays in the inbox instead of being filed."""
        return self.review is not None

    def of_kind(self, kind: DestinationKind) -> tuple[Destination, ...]:
        return tuple(d for d in self.destinations if d.kind is kind)


def _under(prefix: str, template: str) -> str:
    """Place a path template beneath a prefix, tolerating slashes on either side."""
    cleaned = prefix.strip("/")
    return f"{cleaned}/{template}" if cleaned else template


def route(
    record: InboxRecord,
    outcome: Outcome,
    config: Config,
    vault: VaultConfig,
    *,
    area: AreaChoice | None = None,
) -> RoutingResult:
    """Decide the destinations for one classified record. Pure — no I/O.

    `vault` is the vault being ingested into, always the one that receives raw content.

    `area` is the topic destination chosen for this drop, or None when nothing fit. It is
    independent of the classification: whether a summary belongs in `finance/` is a different
    question from whether anything crosses into the work vault, and both answers can be yes.
    """
    local_only = bool(record.declared.local_only)

    if isinstance(outcome, NeedsReview):
        # Nothing is written for a held item, not even under the gitignored prefix: an empty
        # destination list is what makes "fail toward the inbox" observable to the caller.
        return RoutingResult(
            record_id=record.id,
            classification=None,
            participants=record.declared.participants,
            local_only=local_only,
            review=outcome.reason,
        )

    verdict: Classification = outcome
    # Declared wins here as it does in `classify`, so the rule holds even for a caller that
    # assembled the verdict some other way.
    participants = (
        record.declared.participants
        if record.declared.declares("participants")
        else verdict.participants
    )

    raw_template = vault.transcript_template
    if local_only:
        raw_template = _under(vault.local_only_prefix, raw_template)

    destinations = [
        Destination(
            vault=vault.name,
            kind=DestinationKind.RAW,
            path_template=raw_template,
            local_only=local_only,
        )
    ]
    notes: list[str] = []

    if area is not None:
        # This vault already holds the drop's raw body, so a note pointing at it discloses
        # nothing new — which is why the leakage guard governing the cross-vault note has no
        # role here. `local_only` still applies: a private drop's note stays private with it.
        area_template = area.note_template
        if local_only:
            area_template = _under(vault.local_only_prefix, area_template)
        destinations.append(
            Destination(
                vault=vault.name,
                kind=DestinationKind.NOTE,
                path_template=area_template,
                local_only=local_only,
            )
        )

    if verdict.classification in DISTILLED_CLASSIFICATIONS:
        target = config.work_target(vault)
        if target is None:
            notes.append(
                f"classified {verdict.classification!r}, but vault {vault.name!r} has no "
                "work_route configured; filed raw to it alone"
            )
        elif local_only:
            notes.append(
                f"classified {verdict.classification!r}, but the drop is local_only; no "
                f"distilled note crosses into {target.name!r}"
            )
        else:
            destinations.append(
                Destination(
                    vault=target.name,
                    kind=DestinationKind.NOTE,
                    path_template=target.note_template,
                )
            )

    for note in notes:
        logger.info("%s: %s", record.filename, note)

    return RoutingResult(
        record_id=record.id,
        classification=verdict.classification,
        destinations=tuple(destinations),
        participants=participants,
        local_only=local_only,
        notes=tuple(notes),
    )
