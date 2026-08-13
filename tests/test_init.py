"""`memvault init`: the one command that runs before there is anything to run against.

Two themes run through these tests.

**A refusal must cost nothing.** Every way init can say no — a config that already says something
else, a git repository the user declined — has to leave the disk exactly as it was, because the
person meeting a refusal is a first-time adopter who cannot tell a half-written vault from a
whole one.

**An adopted tree belongs to somebody else.** The directory arrives lived-in and dirty, and the
only acceptable footprint is the directories the engine writes into. Not one existing byte moves.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from memvault.config import CONFIG_ENV_VAR, XDG_CONFIG_HOME_VAR, load_config
from memvault.init_cmd import (
    SKELETON_DIRS,
    STARTER_FILES,
    InitError,
    default_config_path,
    default_vault_name,
    init_vault,
    looks_lived_in,
    normalize_vault_name,
    render_config,
    wiring,
    wiring_lines,
)
from tests.conftest import ABSENT_HOME


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def make_repo(root: Path) -> Path:
    """A git repository with an identity, so commits work under any developer's global config."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    return root


def approve(_question: str) -> bool:
    """Answer yes, standing in for `--git` or a human at a terminal."""
    return True


def snapshot(root: Path) -> dict[str, tuple[str, float]]:
    """Every file under `root`, with its content and mtime — the before/after of "untouched"."""
    return {
        str(path.relative_to(root)): (path.read_text(encoding="utf-8"), path.stat().st_mtime)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git/" not in f"{path.relative_to(root)}/"
    }


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Where these tests let init write its config: never a real home."""
    return tmp_path / "conf" / "config.yaml"


class TestScaffold:
    """A fresh directory has to come out the other side usable by every other subcommand."""

    def test_a_fresh_directory_becomes_a_vault_whose_config_loads(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        report = init_vault(tmp_path / "notes", config_path=config_path, confirm=approve)

        config = load_config(report.config_path)
        vault = config.vault()
        assert vault.name == "notes"
        assert vault.root == tmp_path / "notes"
        assert vault.inbox.is_dir()

    def test_the_skeleton_survives_a_clone(self, tmp_path: Path, config_path: Path) -> None:
        """Git does not store empty directories, so the shape needs placeholders to travel."""
        root = tmp_path / "notes"
        init_vault(root, config_path=config_path, confirm=approve)

        for relative in ("inbox", "transcripts", "notes", "repos", "reflections", "memory/auto"):
            assert (root / relative / ".gitkeep").is_file(), relative

    def test_starter_context_is_written_for_both_harness_conventions(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        root = tmp_path / "notes"
        init_vault(root, config_path=config_path, confirm=approve)

        assert (root / "CLAUDE.md").read_text(encoding="utf-8").startswith("# This vault")
        assert (root / "AGENTS.md").read_text(encoding="utf-8") == (root / "CLAUDE.md").read_text(
            encoding="utf-8"
        )

    def test_local_only_material_is_ignored_by_git_from_the_first_commit(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        """The `private/` ignore rule is the enforcement, not pipeline discipline."""
        root = tmp_path / "notes"
        init_vault(root, config_path=config_path, confirm=approve)

        ignored = git(root, "check-ignore", "--", "private/secret.md")

        assert ignored.strip() == "private/secret.md"

    def test_an_existing_gitignore_without_the_rule_is_reported_not_edited(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        """A repo that arrives with its own `.gitignore` keeps it — and then nobody has written
        the one rule `local_only` depends on, which is worth saying rather than assuming."""
        root = tmp_path / "notes"
        root.mkdir()
        (root / ".gitignore").write_text("*.log\n", encoding="utf-8")

        report = init_vault(root, config_path=config_path, confirm=approve)

        assert (root / ".gitignore").read_text(encoding="utf-8") == "*.log\n"
        assert any("private/" in note for note in report.notes)

    def test_a_directory_without_git_gets_a_repository_after_asking(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        asked: list[str] = []

        def remember(question: str) -> bool:
            asked.append(question)
            return True

        report = init_vault(tmp_path / "notes", config_path=config_path, confirm=remember)

        assert report.git_initialized
        assert (tmp_path / "notes" / ".git").exists()
        assert "git init" in asked[0]

    def test_an_existing_repository_is_not_re_initialized(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        root = make_repo(tmp_path / "notes")

        report = init_vault(root, config_path=config_path)

        assert not report.git_initialized


class TestVaultName:
    """The name becomes a YAML key and part of an index filename, so it is not free-form."""

    def test_it_comes_from_the_directory(self, tmp_path: Path) -> None:
        assert default_vault_name(tmp_path / "My Notes") == "my-notes"

    def test_a_name_that_would_parse_as_something_else_is_refused(self) -> None:
        """`2026:` in YAML is an integer key, and the vault would never be found by name."""
        assert normalize_vault_name("2026") == ""

    def test_an_unusable_explicit_name_is_refused_by_name(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        with pytest.raises(InitError, match="'2026' is not usable"):
            init_vault(
                tmp_path / "notes", config_path=config_path, vault_name="2026", confirm=approve
            )

    def test_a_directory_with_no_usable_name_falls_back(self, tmp_path: Path) -> None:
        assert default_vault_name(tmp_path / "...") == "vault"

    def test_the_written_config_quotes_the_root(self, tmp_path: Path) -> None:
        """A path holding a space or a `#` must survive the round trip into YAML."""
        root = tmp_path / "my notes # 2"
        rendered = render_config("notes", root)

        assert json.dumps(str(root)) in rendered


class TestIdempotence:
    """The common second run is a user who is not sure the first one worked."""

    def test_a_second_run_touches_nothing(self, tmp_path: Path, config_path: Path) -> None:
        root = tmp_path / "notes"
        init_vault(root, config_path=config_path, confirm=approve)
        before = snapshot(root)
        config_before = config_path.stat().st_mtime

        report = init_vault(root, config_path=config_path, confirm=approve)

        assert snapshot(root) == before
        assert config_path.stat().st_mtime == config_before
        assert not report.config_written
        assert report.created == ()

    def test_a_deleted_directory_is_restored(self, tmp_path: Path, config_path: Path) -> None:
        """The scaffold is a statement about what the vault needs, not a one-time event."""
        root = tmp_path / "notes"
        init_vault(root, config_path=config_path, confirm=approve)
        for path in sorted((root / "reflections").rglob("*")):
            path.unlink()
        (root / "reflections").rmdir()

        report = init_vault(root, config_path=config_path, confirm=approve)

        assert (root / "reflections" / ".gitkeep").is_file()
        assert "reflections/" in report.created


class TestRefusals:
    """Each of these is a no, and each one has to be a no that changed nothing."""

    def test_a_config_that_says_something_else_is_not_overwritten(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        config_path.parent.mkdir(parents=True)
        config_path.write_text("vaults: {other: {root: /somewhere}}\n", encoding="utf-8")

        with pytest.raises(InitError, match=str(config_path)):
            init_vault(tmp_path / "notes", config_path=config_path, confirm=approve)

        assert config_path.read_text(encoding="utf-8").startswith("vaults:")

    def test_a_refused_config_leaves_no_half_vault_behind(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        """The refusal is resolved before the first mkdir, so there is nothing to clean up."""
        config_path.parent.mkdir(parents=True)
        config_path.write_text("vaults: {other: {root: /somewhere}}\n", encoding="utf-8")

        with pytest.raises(InitError):
            init_vault(tmp_path / "notes", config_path=config_path, confirm=approve)

        assert not (tmp_path / "notes").exists()

    def test_force_replaces_the_config(self, tmp_path: Path, config_path: Path) -> None:
        config_path.parent.mkdir(parents=True)
        config_path.write_text("vaults: {other: {root: /somewhere}}\n", encoding="utf-8")

        report = init_vault(
            tmp_path / "notes", config_path=config_path, force=True, confirm=approve
        )

        assert report.config_written
        assert load_config(config_path).vault().root == tmp_path / "notes"

    def test_force_replaces_edited_starter_files(self, tmp_path: Path, config_path: Path) -> None:
        root = tmp_path / "notes"
        init_vault(root, config_path=config_path, confirm=approve)
        (root / "CLAUDE.md").write_text("mine now", encoding="utf-8")

        init_vault(root, config_path=config_path, force=True, confirm=approve)

        assert (root / "CLAUDE.md").read_text(encoding="utf-8").startswith("# This vault")

    def test_an_edited_starter_file_is_kept_without_force(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        root = tmp_path / "notes"
        init_vault(root, config_path=config_path, confirm=approve)
        (root / "CLAUDE.md").write_text("mine now", encoding="utf-8")

        report = init_vault(root, config_path=config_path, confirm=approve)

        assert (root / "CLAUDE.md").read_text(encoding="utf-8") == "mine now"
        assert "CLAUDE.md" in report.kept

    def test_declining_git_refuses_and_says_why(self, tmp_path: Path, config_path: Path) -> None:
        """Config validation rejects a non-repo vault, so writing one would only fail later."""
        root = tmp_path / "notes"
        root.mkdir()

        with pytest.raises(InitError, match="the commit history is the audit log"):
            init_vault(root, config_path=config_path)

        assert not config_path.exists()

    def test_a_file_where_the_vault_should_be_is_refused(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        target = tmp_path / "notes"
        target.write_text("not a directory", encoding="utf-8")

        with pytest.raises(InitError, match="is not a directory"):
            init_vault(target, config_path=config_path, confirm=approve)


class TestAdopt:
    """Somebody's real notes folder: dirty, opinionated, and not ours to edit."""

    @pytest.fixture
    def lived_in(self, tmp_path: Path) -> Path:
        root = make_repo(tmp_path / "notes")
        (root / "README.md").write_text("my notes\n", encoding="utf-8")
        (root / "journal").mkdir()
        (root / "journal" / "2020.md").write_text("years of this\n", encoding="utf-8")
        git(root, "add", "-A")
        git(root, "commit", "-qm", "notes")
        # The normal state of a working repo: an uncommitted edit and an untracked file.
        (root / "README.md").write_text("my notes, edited\n", encoding="utf-8")
        (root / "scratch.md").write_text("half a thought\n", encoding="utf-8")
        return root

    def test_it_is_inferred_from_material_that_is_already_there(self, lived_in: Path) -> None:
        assert looks_lived_in(lived_in)

    def test_a_scaffolded_vault_is_not_mistaken_for_a_lived_in_one(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        root = tmp_path / "notes"
        init_vault(root, config_path=config_path, confirm=approve)

        assert not looks_lived_in(root)

    def test_nothing_that_was_there_is_modified(self, lived_in: Path, config_path: Path) -> None:
        before = snapshot(lived_in)
        dirty_before = set(git(lived_in, "status", "--porcelain").splitlines())

        report = init_vault(lived_in, config_path=config_path)

        assert report.adopted
        # Content and mtime alike: an adopted tree may gain paths, never lose or change one.
        assert before.items() <= snapshot(lived_in).items()
        assert dirty_before <= set(git(lived_in, "status", "--porcelain").splitlines())

    def test_no_starter_files_are_written_over_an_adopted_tree(
        self, lived_in: Path, config_path: Path
    ) -> None:
        init_vault(lived_in, config_path=config_path)

        assert not (lived_in / "CLAUDE.md").exists()
        assert not (lived_in / ".gitignore").exists()

    def test_the_directories_the_engine_writes_into_are_created(
        self, lived_in: Path, config_path: Path
    ) -> None:
        report = init_vault(lived_in, config_path=config_path)

        assert (lived_in / "inbox").is_dir()
        assert "inbox/" in report.created

    def test_a_missing_private_ignore_rule_is_reported_not_fixed(
        self, lived_in: Path, config_path: Path
    ) -> None:
        """Editing somebody's `.gitignore` silently is worse than telling them to."""
        report = init_vault(lived_in, config_path=config_path)

        assert any("private/" in note for note in report.notes)

    def test_an_existing_ignore_rule_earns_no_warning(
        self, lived_in: Path, config_path: Path
    ) -> None:
        (lived_in / ".gitignore").write_text("private/\n", encoding="utf-8")

        report = init_vault(lived_in, config_path=config_path)

        assert not any("private/" in note for note in report.notes)

    def test_the_config_still_lands(self, lived_in: Path, config_path: Path) -> None:
        report = init_vault(lived_in, config_path=config_path)

        assert report.config_written
        assert load_config(config_path).vault().root == lived_in

    def test_adopting_a_directory_that_does_not_exist_is_refused(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        with pytest.raises(InitError, match="nothing to adopt"):
            init_vault(tmp_path / "ghost", config_path=config_path, adopt=True, confirm=approve)

    def test_adopting_a_non_repository_refuses_when_git_is_declined(
        self, tmp_path: Path, config_path: Path
    ) -> None:
        root = tmp_path / "notes"
        root.mkdir()
        (root / "README.md").write_text("my notes\n", encoding="utf-8")

        with pytest.raises(InitError, match="not a git repository"):
            init_vault(root, config_path=config_path)

        assert not config_path.exists()


class TestTemplateClone:
    """The GitHub-first path: the `memvault-template` repo cloned, then adopted.

    That template is deliberately dumb — a skeleton, starter docs, and a `.gitignore`, with no
    logic of its own — which only holds together while its layout is exactly what init would have
    scaffolded. A clone that still needs directories created, or that earns a privacy warning, is
    the two having drifted, and the adopter meets a half-vault on their very first command.
    """

    @pytest.fixture
    def clone(self, tmp_path: Path) -> Path:
        """A checkout of the template, built from what init itself calls the vault shape.

        Rebuilt here rather than copied from the repo so the test states the contract instead of
        following it: skeleton directories that survive a clone, the starter docs, and a README —
        the file that makes a template clone read as somebody's tree rather than a blank one.
        """
        root = make_repo(tmp_path / "my-vault")
        for relative in SKELETON_DIRS:
            (root / relative).mkdir(parents=True)
            (root / relative / ".gitkeep").write_text("", encoding="utf-8")
        for name, body in STARTER_FILES:
            (root / name).write_text(body, encoding="utf-8")
        (root / "README.md").write_text("# my vault\n", encoding="utf-8")
        git(root, "add", "-A")
        git(root, "commit", "-qm", "template")
        return root

    def test_a_clone_is_adopted_rather_than_scaffolded(
        self, clone: Path, config_path: Path
    ) -> None:
        report = init_vault(clone, config_path=config_path)

        assert report.adopted

    def test_the_engine_finds_nothing_left_to_create(self, clone: Path, config_path: Path) -> None:
        """The load-bearing one: every directory the engine writes into already travelled."""
        report = init_vault(clone, config_path=config_path)

        assert report.created == ()
        assert set(report.kept) == {f"{relative}/" for relative in SKELETON_DIRS}

    def test_the_privacy_rule_travelled_with_the_clone(
        self, clone: Path, config_path: Path
    ) -> None:
        """Adopt never writes a `.gitignore`, so if the template's did not arrive, nothing has
        written the one line `local_only` material depends on."""
        report = init_vault(clone, config_path=config_path)

        assert not any("private/" in note for note in report.notes)

    def test_the_clone_is_left_exactly_as_it_arrived(self, clone: Path, config_path: Path) -> None:
        before = snapshot(clone)

        init_vault(clone, config_path=config_path)

        assert snapshot(clone) == before
        assert git(clone, "status", "--porcelain") == ""

    def test_the_config_loads_against_the_clone(self, clone: Path, config_path: Path) -> None:
        report = init_vault(clone, config_path=config_path)

        vault = load_config(report.config_path).vault()
        assert vault.root == clone
        assert vault.inbox.is_dir()


class TestConfigLocation:
    """R9 says discovery starts at the XDG path, so that is where a new config belongs."""

    def test_the_default_is_the_xdg_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(XDG_CONFIG_HOME_VAR, str(tmp_path / "xdg"))

        assert default_config_path() == tmp_path / "xdg" / "memvault" / "config.yaml"

    def test_a_config_written_there_is_the_one_discovery_finds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(XDG_CONFIG_HOME_VAR, str(tmp_path / "xdg"))

        report = init_vault(tmp_path / "notes", confirm=approve)

        assert report.config_path == tmp_path / "xdg" / "memvault" / "config.yaml"
        assert load_config().source == report.config_path

    def test_a_config_searched_earlier_is_reported_rather_than_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A written config that never gets read is the quietest way for init to fail."""
        xdg = tmp_path / "xdg" / "memvault"
        xdg.mkdir(parents=True)
        (xdg / "config.yaml").write_text("vaults: {}\n", encoding="utf-8")
        monkeypatch.setenv(XDG_CONFIG_HOME_VAR, str(tmp_path / "xdg"))

        report = init_vault(
            tmp_path / "notes", config_path=tmp_path / "elsewhere.yaml", confirm=approve
        )

        assert any(str(xdg / "config.yaml") in note for note in report.notes)

    def test_an_environment_variable_pointing_elsewhere_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "somewhere-else.yaml"))

        report = init_vault(
            tmp_path / "notes", config_path=tmp_path / "written.yaml", confirm=approve
        )

        assert any("somewhere-else.yaml" in note for note in report.notes)

    def test_no_shadow_is_reported_when_the_written_file_is_the_one_that_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(XDG_CONFIG_HOME_VAR, str(tmp_path / "xdg"))

        report = init_vault(tmp_path / "notes", confirm=approve)

        assert report.notes == ()


class TestWiring:
    """A pasteable block that does not parse is worse than no block at all."""

    def test_every_harness_gets_an_entry(self, tmp_path: Path) -> None:
        harnesses = {entry.harness for entry in wiring(tmp_path / "notes")}

        assert harnesses == {"Claude Code", "Codex", "opencode", "Cursor"}

    @pytest.mark.parametrize("harness", ["opencode", "Cursor"])
    def test_the_json_blocks_parse_and_name_the_vault(self, tmp_path: Path, harness: str) -> None:
        root = tmp_path / "my notes"
        entry = next(e for e in wiring(root) if e.harness == harness)

        parsed = json.loads("\n".join(entry.body))

        assert str(root) in json.dumps(parsed)

    def test_the_command_line_carries_the_vault_path(self, tmp_path: Path) -> None:
        entry = next(e for e in wiring(tmp_path / "notes") if e.harness == "Claude Code")

        assert entry.where.endswith(f"memvault mcp {tmp_path / 'notes'}")

    def test_every_line_is_indented_under_its_harness(self, tmp_path: Path) -> None:
        assert all(line.startswith("  ") for line in wiring_lines(tmp_path / "notes"))


def test_the_suite_guard_covers_the_path_init_writes_to_by_default() -> None:
    """Init writes config files for a living, so a leak here would land on the real machine."""
    assert str(default_config_path()).startswith(ABSENT_HOME)
