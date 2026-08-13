"""Which topic area a drop's distilled note belongs in.

Selection is deliberately ordered cheapest-first, and every step is allowed to decline.

**A declared workspace wins.** It is a fact the feeder knows rather than a judgement, so when a
drop carries one that maps, no model opinion overrides it. Chad is the only feeder that has one
today; everything arriving from the phone has none.

**A recognized project comes next.** The classifier already returns `project`, and a project
directory that exists on disk is as concrete as a workspace alias. A `project` naming nothing real
is ignored rather than trusted — models invent plausible names.

**The model's `area` is the fallback**, for material that has neither.

**Declining is a real answer.** Returning None means the drop files raw and gains no note, which
is the designed behaviour rather than a failure: the note is additive, so a wrong one is worse
than none.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from memvault.classify import Classification
from memvault.config import AreaConfig, VaultConfig
from memvault.inbox import DeclaredMetadata

#: Appended to a project's memory directory to reach its dated notes. `/reflect` already writes
#: `adhoc/` by hand, so an ingested note lands beside the ones written deliberately.
PROJECT_NOTE_SUFFIX = "adhoc/{date}-{slug}.md"


@dataclass(frozen=True)
class AreaChoice:
    """The area a note will be filed into, and what decided it."""

    name: str
    note_template: str
    #: "workspace", "project", or "classifier" — recorded so a surprising note is explicable
    #: afterwards rather than a mystery.
    via: str


def project_areas(vault: VaultConfig) -> tuple[AreaConfig, ...]:
    """One area per existing project directory. Does I/O; call once per pass, not per drop.

    Projects are not enumerated in config because their names are self-describing and the set
    changes whenever a project is added. A directory that exists is the authority.
    """
    root = vault.memory_template.split("{", 1)[0].strip("/")
    if not root:
        return ()
    base = vault.root / root
    if not base.is_dir():
        return ()
    return tuple(
        AreaConfig(
            name=child.name,
            note_template=(
                f"{vault.memory_template.format(project=child.name)}/{PROJECT_NOTE_SUFFIX}"
            ),
        )
        for child in sorted(base.iterdir())
        if child.is_dir() and not child.name.startswith(".")
    )


def choose_area(
    *,
    declared: DeclaredMetadata,
    verdict: Classification,
    areas: Sequence[AreaConfig],
) -> AreaChoice | None:
    """Pick one area for this drop, or None when nothing fits. Pure."""
    if not areas:
        return None

    # First occurrence wins. Callers build this list as declared areas followed by ones derived
    # from the project directories, and the two can collide: a vault that declares an area named
    # after one of its own projects would otherwise be silently overruled by the directory. What
    # someone wrote in config beats what was inferred from the filesystem.
    by_name: dict[str, AreaConfig] = {}
    for area in areas:
        by_name.setdefault(area.name.casefold(), area)

    if declared.workspace:
        wanted = declared.workspace.casefold()
        for area in areas:
            if wanted == area.name.casefold() or any(
                wanted == alias.casefold() for alias in area.workspaces
            ):
                return AreaChoice(area.name, area.note_template, "workspace")

    if verdict.project:
        found = by_name.get(verdict.project.casefold())
        if found is not None:
            return AreaChoice(found.name, found.note_template, "project")

    if verdict.area:
        found = by_name.get(verdict.area.casefold())
        if found is not None:
            return AreaChoice(found.name, found.note_template, "classifier")

    return None
