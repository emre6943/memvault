"""Choosing which topic area a drop's distilled note belongs in.

The three branches are tested in precedence order — a declared workspace, then a project the
classifier recognized, then the model's own `area` — along with the two ways selection declines:
nothing matched, and nothing was declared to match against.
"""

from __future__ import annotations

from pathlib import Path

from memvault.area import AreaChoice, choose_area, project_areas
from memvault.classify import Classification
from memvault.config import AreaConfig, VaultConfig
from memvault.inbox import DeclaredMetadata

FINANCE = AreaConfig(
    name="finance",
    note_template="finance/adhoc/{date}-{slug}.md",
    when="markets, macro, portfolio",
    workspaces=("Finance",),
)
REALTOR = AreaConfig(
    name="realtor", note_template="realtor/adhoc/{date}-{slug}.md", when="house hunting"
)
SAPIK = AreaConfig(name="Sapik", note_template="repos/Sapik/adhoc/{date}-{slug}.md")


def declared(**kwargs: object) -> DeclaredMetadata:
    keys = frozenset(key for key, value in kwargs.items() if value is not None)
    return DeclaredMetadata(declared_keys=keys, **kwargs)  # type: ignore[arg-type]


def verdict(**kwargs: object) -> Classification:
    base: dict[str, object] = {"classification": "personal", "confidence": 0.9, "title": "T"}
    return Classification(**{**base, **kwargs})  # type: ignore[arg-type]


def a_vault(tmp_path: Path) -> VaultConfig:
    root = tmp_path / "vault"
    root.mkdir(exist_ok=True)
    return VaultConfig(name="personal", root=root, inbox=root / "inbox")


class TestChooseArea:
    def test_a_matching_workspace_wins_over_the_verdict(self) -> None:
        choice = choose_area(
            declared=declared(workspace="Finance"),
            verdict=verdict(project="realtor"),
            areas=(FINANCE, REALTOR),
        )

        assert choice == AreaChoice("finance", "finance/adhoc/{date}-{slug}.md", "workspace")

    def test_workspace_matching_is_case_insensitive(self) -> None:
        choice = choose_area(
            declared=declared(workspace="finance"), verdict=verdict(), areas=(FINANCE,)
        )

        assert choice is not None and choice.via == "workspace"

    def test_an_area_name_itself_works_as_a_workspace(self) -> None:
        """A workspace called `realtor` maps even though the area declares no aliases."""
        choice = choose_area(
            declared=declared(workspace="realtor"), verdict=verdict(), areas=(FINANCE, REALTOR)
        )

        assert choice is not None and choice.name == "realtor"

    def test_an_unknown_workspace_falls_through_rather_than_erroring(self) -> None:
        choice = choose_area(
            declared=declared(workspace="General"),
            verdict=verdict(project="realtor"),
            areas=(FINANCE, REALTOR),
        )

        assert choice == AreaChoice("realtor", "realtor/adhoc/{date}-{slug}.md", "project")

    def test_a_recognized_project_is_chosen(self) -> None:
        choice = choose_area(
            declared=declared(), verdict=verdict(project="Sapik"), areas=(FINANCE, SAPIK)
        )

        assert choice == AreaChoice("Sapik", "repos/Sapik/adhoc/{date}-{slug}.md", "project")

    def test_a_project_that_is_not_a_known_area_is_ignored(self) -> None:
        """Models invent plausible project names; only a real destination counts."""
        choice = choose_area(
            declared=declared(), verdict=verdict(project="Nonexistent"), areas=(FINANCE,)
        )

        assert choice is None

    def test_nothing_matching_yields_none(self) -> None:
        assert choose_area(declared=declared(), verdict=verdict(), areas=(FINANCE,)) is None

    def test_an_empty_area_list_yields_none(self) -> None:
        choice = choose_area(declared=declared(workspace="Finance"), verdict=verdict(), areas=())

        assert choice is None

    def test_the_models_area_is_used_when_nothing_else_matches(self) -> None:
        choice = choose_area(
            declared=declared(), verdict=verdict(area="realtor"), areas=(FINANCE, REALTOR)
        )

        assert choice == AreaChoice("realtor", "realtor/adhoc/{date}-{slug}.md", "classifier")

    def test_a_matching_workspace_beats_the_models_area(self) -> None:
        choice = choose_area(
            declared=declared(workspace="Finance"),
            verdict=verdict(area="realtor"),
            areas=(FINANCE, REALTOR),
        )

        assert choice is not None and choice.via == "workspace"

    def test_a_recognized_project_beats_the_models_area(self) -> None:
        choice = choose_area(
            declared=declared(),
            verdict=verdict(project="Sapik", area="finance"),
            areas=(FINANCE, SAPIK),
        )

        assert choice is not None and choice.via == "project"

    def test_an_invented_area_name_is_declined(self) -> None:
        """Models invent plausible names; only a declared area counts."""
        choice = choose_area(declared=declared(), verdict=verdict(area="taxes"), areas=(FINANCE,))

        assert choice is None


class TestProjectAreas:
    def test_each_project_directory_becomes_an_area(self, tmp_path: Path) -> None:
        vault = a_vault(tmp_path)
        for name in ("Sapik", "Chad"):
            (vault.root / "repos" / name).mkdir(parents=True)

        found = {area.name: area.note_template for area in project_areas(vault)}

        assert found == {
            "Sapik": "repos/Sapik/adhoc/{date}-{slug}.md",
            "Chad": "repos/Chad/adhoc/{date}-{slug}.md",
        }

    def test_a_vault_with_no_memory_root_yields_nothing(self, tmp_path: Path) -> None:
        assert project_areas(a_vault(tmp_path)) == ()

    def test_files_beside_the_project_directories_are_skipped(self, tmp_path: Path) -> None:
        vault = a_vault(tmp_path)
        (vault.root / "repos").mkdir(parents=True)
        (vault.root / "repos" / "README.md").write_text("x\n", encoding="utf-8")
        (vault.root / "repos" / "Sapik").mkdir()

        assert [area.name for area in project_areas(vault)] == ["Sapik"]

    def test_dot_directories_are_skipped(self, tmp_path: Path) -> None:
        vault = a_vault(tmp_path)
        (vault.root / "repos" / ".cache").mkdir(parents=True)
        (vault.root / "repos" / "Sapik").mkdir()

        assert [area.name for area in project_areas(vault)] == ["Sapik"]


class TestNameCollisions:
    """Declared areas and areas derived from project directories can share a name."""

    CONFIG_AREA = AreaConfig(
        name="MemVault", note_template="notes/{date}-{slug}.md", when="declared in config"
    )
    DERIVED_AREA = AreaConfig(
        name="MemVault", note_template="repos/MemVault/adhoc/{date}-{slug}.md"
    )

    def test_a_declared_area_beats_a_project_directory_of_the_same_name(self) -> None:
        """Explicit config beats what was inferred from the filesystem."""
        choice = choose_area(
            declared=declared(),
            verdict=verdict(project="MemVault"),
            areas=(self.CONFIG_AREA, self.DERIVED_AREA),
        )

        assert choice is not None
        assert choice.note_template == "notes/{date}-{slug}.md"

    def test_the_same_precedence_holds_for_a_workspace_match(self) -> None:
        choice = choose_area(
            declared=declared(workspace="MemVault"),
            verdict=verdict(),
            areas=(self.CONFIG_AREA, self.DERIVED_AREA),
        )

        assert choice is not None
        assert choice.note_template == "notes/{date}-{slug}.md"

    def test_the_same_precedence_holds_for_the_models_area(self) -> None:
        choice = choose_area(
            declared=declared(),
            verdict=verdict(area="MemVault"),
            areas=(self.CONFIG_AREA, self.DERIVED_AREA),
        )

        assert choice is not None
        assert choice.note_template == "notes/{date}-{slug}.md"
