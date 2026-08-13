"""Config loading and validation.

The theme of these tests: a bad config must fail at load, naming what is wrong, rather than
producing a Config object that misbehaves later during a pass.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from memvault.config import (
    CONFIG_ENV_VAR,
    XDG_CONFIG_HOME_VAR,
    ConfigError,
    VaultConfig,
    find_config,
    load_config,
)


def make_vault(tmp_path: Path, name: str, *, git: bool = True, inbox: bool = True) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    if git:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    if inbox:
        (root / "inbox").mkdir(exist_ok=True)
    return root


def write_config(tmp_path: Path, data: dict, name: str = "memvault.config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def two_vault_config(tmp_path: Path) -> Path:
    personal = make_vault(tmp_path, "personal")
    work = make_vault(tmp_path, "work")
    return write_config(
        tmp_path,
        {
            "default_vault": "personal",
            "vaults": {
                "personal": {"root": str(personal), "work_route": "work"},
                "work": {"root": str(work)},
            },
        },
    )


def test_loads_two_vaults_and_selects_default(two_vault_config: Path) -> None:
    config = load_config(two_vault_config)

    assert set(config.vaults) == {"personal", "work"}
    assert config.vault().name == "personal"
    assert config.vault("work").name == "work"


def test_work_route_resolves_to_the_other_vault(two_vault_config: Path) -> None:
    config = load_config(two_vault_config)

    target = config.work_target(config.vault("personal"))

    assert target is not None
    assert target.name == "work"
    assert config.work_target(config.vault("work")) is None


def test_unknown_vault_name_is_rejected(two_vault_config: Path) -> None:
    config = load_config(two_vault_config)

    with pytest.raises(ConfigError, match="unknown vault 'nope'"):
        config.vault("nope")


def test_missing_root_key_names_the_key(tmp_path: Path) -> None:
    path = write_config(tmp_path, {"vaults": {"personal": {"inbox": "inbox"}}})

    with pytest.raises(ConfigError, match="missing required key 'root'"):
        load_config(path)


def test_nonexistent_root_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, {"vaults": {"personal": {"root": str(tmp_path / "ghost")}}})

    with pytest.raises(ConfigError, match="does not exist"):
        load_config(path)


def test_root_that_is_not_a_git_repo_is_rejected(tmp_path: Path) -> None:
    plain = make_vault(tmp_path, "plain", git=False)
    path = write_config(tmp_path, {"vaults": {"personal": {"root": str(plain)}}})

    with pytest.raises(ConfigError, match="not a git repository"):
        load_config(path)


def test_tilde_paths_expand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    make_vault(home, "vault")
    monkeypatch.setenv("HOME", str(home))
    path = write_config(tmp_path, {"vaults": {"personal": {"root": "~/vault"}}})

    vault = load_config(path).vault()

    assert vault.root == (home / "vault").resolve()
    assert vault.inbox == (home / "vault" / "inbox").resolve()


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path, {"vaults": {"personal": {"root": str(root)}}, "clasifier": {"model": "x"}}
    )

    with pytest.raises(ConfigError, match="unknown key\\(s\\) in config root: clasifier"):
        load_config(path)


def test_unknown_vault_key_is_rejected(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path, {"vaults": {"personal": {"root": str(root), "wrok_route": "work"}}}
    )

    with pytest.raises(ConfigError, match="unknown key\\(s\\) in vaults.personal: wrok_route"):
        load_config(path)


def test_work_route_pointing_at_unknown_vault_is_rejected(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path, {"vaults": {"personal": {"root": str(root), "work_route": "ghost"}}}
    )

    with pytest.raises(ConfigError, match="work_route points at 'ghost'"):
        load_config(path)


def test_work_route_pointing_at_itself_is_rejected(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path, {"vaults": {"personal": {"root": str(root), "work_route": "personal"}}}
    )

    with pytest.raises(ConfigError, match="points at itself"):
        load_config(path)


def test_default_vault_required_when_several_configured(tmp_path: Path) -> None:
    a = make_vault(tmp_path, "a")
    b = make_vault(tmp_path, "b")
    path = write_config(tmp_path, {"vaults": {"a": {"root": str(a)}, "b": {"root": str(b)}}})

    with pytest.raises(ConfigError, match="'default_vault' is required"):
        load_config(path)


def test_single_vault_becomes_the_default_implicitly(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "solo")
    path = write_config(tmp_path, {"vaults": {"solo": {"root": str(root)}}})

    assert load_config(path).vault().name == "solo"


def test_default_vault_naming_a_missing_vault_is_rejected(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path, {"default_vault": "ghost", "vaults": {"personal": {"root": str(root)}}}
    )

    with pytest.raises(ConfigError, match="default_vault 'ghost' is not a configured vault"):
        load_config(path)


def test_empty_config_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "memvault.config.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="config file is empty"):
        load_config(path)


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "memvault.config.yaml"
    path.write_text("vaults: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_missing_config_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "absent.yaml")


def test_env_var_supplies_the_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(tmp_path, {"vaults": {"personal": {"root": str(root)}}}, "elsewhere.yaml")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    monkeypatch.chdir(tmp_path / "personal")

    assert load_config().vault().name == "personal"


def test_confidence_threshold_outside_range_is_rejected(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path,
        {"vaults": {"personal": {"root": str(root)}}, "classifier": {"confidence_threshold": 1.5}},
    )

    with pytest.raises(ConfigError, match="confidence_threshold must be between 0 and 1"):
        load_config(path)


def test_chunk_overlap_not_smaller_than_chunk_size_is_rejected(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path,
        {
            "vaults": {"personal": {"root": str(root)}},
            "index": {"chunk_max_chars": 500, "chunk_overlap_chars": 500},
        },
    )

    with pytest.raises(ConfigError, match="must be less than"):
        load_config(path)


def test_min_similarity_defaults_to_the_calibrated_floor(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(tmp_path, {"vaults": {"personal": {"root": str(root)}}})

    assert load_config(path).index.min_similarity == 0.82


def test_min_similarity_is_overridable(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path,
        {"vaults": {"personal": {"root": str(root)}}, "index": {"min_similarity": 0.9}},
    )

    assert load_config(path).index.min_similarity == 0.9


def test_min_similarity_outside_range_is_rejected(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path,
        {"vaults": {"personal": {"root": str(root)}}, "index": {"min_similarity": 1.4}},
    )

    with pytest.raises(ConfigError, match="min_similarity must be between 0 and 1"):
        load_config(path)


def test_unknown_index_backend_is_rejected(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path, {"vaults": {"personal": {"root": str(root)}}, "index": {"backend": "magic"}}
    )

    with pytest.raises(ConfigError, match="backend must be"):
        load_config(path)


def test_index_db_path_is_templated_per_vault(two_vault_config: Path) -> None:
    config = load_config(two_vault_config)

    assert config.index.db_path("personal").name == "personal.db"
    assert config.index.db_path("work").name == "work.db"


class TestReflectionTemplate:
    """Where the weekly pass files its reflection note (R6)."""

    def test_it_defaults_to_a_root_of_its_own(self, tmp_path: Path) -> None:
        """Its own root, so a vault can guard, index, or ignore reflections separately."""
        root = make_vault(tmp_path, "personal")
        path = write_config(tmp_path, {"vaults": {"personal": {"root": str(root)}}})

        vault = load_config(path).vault()

        assert vault.reflection_template == "reflections/{yyyy}/{date}-reflection.md"
        assert vault.reflection_template.split("/")[0] != vault.digest_template.split("/")[0]

    def test_a_vault_may_put_the_note_where_it_likes(self, tmp_path: Path) -> None:
        root = make_vault(tmp_path, "personal")
        path = write_config(
            tmp_path,
            {
                "vaults": {
                    "personal": {
                        "root": str(root),
                        "reflection_template": "repos/Vault/reflections/{date}.md",
                    }
                }
            },
        )

        assert load_config(path).vault().reflection_template == "repos/Vault/reflections/{date}.md"


class TestClaudeMemAliases:
    """The alias map is a boundary control: claude-mem mixes work and personal observations."""

    def test_canonical_name_maps_to_itself(self, tmp_path: Path) -> None:
        root = make_vault(tmp_path, "personal")
        path = write_config(
            tmp_path,
            {
                "vaults": {
                    "personal": {
                        "root": str(root),
                        "claude_mem_projects": {"tcg-vendor": ["tcg-vendor", "TCGVendor"]},
                    }
                }
            },
        )

        vault = load_config(path).vault()

        assert vault.canonical_project("tcg-vendor") == "tcg-vendor"

    def test_alias_folds_into_canonical_name(self, tmp_path: Path) -> None:
        root = make_vault(tmp_path, "personal")
        path = write_config(
            tmp_path,
            {
                "vaults": {
                    "personal": {
                        "root": str(root),
                        "claude_mem_projects": {"tcg-vendor": ["tcg-vendor", "TCGVendor"]},
                    }
                }
            },
        )

        vault = load_config(path).vault()

        assert vault.canonical_project("TCGVendor") == "tcg-vendor"

    def test_unlisted_project_returns_none_rather_than_defaulting(self, tmp_path: Path) -> None:
        root = make_vault(tmp_path, "personal")
        path = write_config(
            tmp_path,
            {
                "vaults": {
                    "personal": {
                        "root": str(root),
                        "claude_mem_projects": {"Homeworld": ["Homeworld"]},
                    }
                }
            },
        )

        vault = load_config(path).vault()

        assert vault.canonical_project("oryx") is None


def test_ignore_dirty_defaults_to_empty(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(tmp_path, {"vaults": {"personal": {"root": str(root)}}})

    assert load_config(path).vault().ignore_dirty == ()


def test_ignore_dirty_is_read_as_a_tuple_of_prefixes(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path,
        {"vaults": {"personal": {"root": str(root), "ignore_dirty": ["memory/auto", "config"]}}},
    )

    assert load_config(path).vault().ignore_dirty == ("memory/auto", "config")


# --- Segmentation and per-source policy -------------------------------------------------
#
# The theme here: a config that says nothing about segmentation must behave exactly as it did
# before segmentation existed. Every new dial defaults to "inherit", and the only direction a
# source policy can move the work boundary is narrower.


def _single(tmp_path: Path, extra: dict) -> Path:
    root = make_vault(tmp_path, "personal")
    return write_config(tmp_path, {"vaults": {"personal": {"root": str(root)}}, **extra})


def test_segment_and_sources_default_to_off_and_empty(tmp_path: Path) -> None:
    config = load_config(_single(tmp_path, {}))

    assert config.segment.enabled is False
    assert config.sources == {}
    assert config.should_segment("conversate") is False
    assert config.allows_work_note("conversate") is True
    assert config.confidence_threshold_for("conversate") == config.classifier.confidence_threshold


def test_segment_section_is_read(tmp_path: Path) -> None:
    config = load_config(
        _single(tmp_path, {"segment": {"enabled": True, "min_chars": 500, "max_segments": 4}})
    )

    assert config.segment.enabled is True
    assert config.segment.min_chars == 500
    assert config.segment.max_segments == 4
    assert config.should_segment("anything") is True


def test_unknown_key_in_segment_is_rejected_by_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="segement_enabled"):
        load_config(_single(tmp_path, {"segment": {"segement_enabled": True}}))


def test_unknown_key_in_a_source_policy_is_rejected_by_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="wrok_route"):
        load_config(_single(tmp_path, {"sources": {"conversate": {"wrok_route": "never"}}}))


def test_invalid_work_route_names_the_value_and_the_valid_set(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="nope"):
        load_config(_single(tmp_path, {"sources": {"conversate": {"work_route": "nope"}}}))


def test_segment_enabled_must_be_a_real_boolean(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="segment.enabled"):
        load_config(_single(tmp_path, {"segment": {"enabled": "yes"}}))


def test_negative_min_chars_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="min_chars"):
        load_config(_single(tmp_path, {"segment": {"min_chars": -1}}))


def test_max_segments_below_one_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="max_segments"):
        load_config(_single(tmp_path, {"segment": {"max_segments": 0}}))


def test_source_threshold_overrides_only_that_source(tmp_path: Path) -> None:
    config = load_config(
        _single(
            tmp_path,
            {
                "classifier": {"confidence_threshold": 0.7},
                "sources": {"conversate": {"confidence_threshold": 0.9}},
            },
        )
    )

    assert config.confidence_threshold_for("conversate") == 0.9
    assert config.confidence_threshold_for("manual") == 0.7
    assert config.confidence_threshold_for(None) == 0.7


def test_source_can_disable_segmentation_the_global_switch_enabled(tmp_path: Path) -> None:
    config = load_config(
        _single(
            tmp_path,
            {"segment": {"enabled": True}, "sources": {"manual": {"segment": False}}},
        )
    )

    assert config.should_segment("conversate") is True
    assert config.should_segment("manual") is False


def test_work_route_never_suppresses_the_note_for_that_source_only(tmp_path: Path) -> None:
    config = load_config(_single(tmp_path, {"sources": {"conversate": {"work_route": "never"}}}))

    assert config.allows_work_note("conversate") is False
    assert config.allows_work_note("manual") is True


def test_source_lookup_is_case_insensitive_and_trimmed(tmp_path: Path) -> None:
    config = load_config(_single(tmp_path, {"sources": {"Conversate": {"work_route": "never"}}}))

    for spelling in ("conversate", "Conversate", "  CONVERSATE  "):
        assert config.allows_work_note(spelling) is False


def test_duplicate_source_names_differing_only_in_case_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="more than once"):
        load_config(
            _single(
                tmp_path, {"sources": {"conversate": {}, "Conversate": {"work_route": "never"}}}
            )
        )


def test_a_source_with_no_entry_inherits_every_global(tmp_path: Path) -> None:
    config = load_config(
        _single(
            tmp_path,
            {
                "segment": {"enabled": True},
                "classifier": {"confidence_threshold": 0.8},
                "sources": {"conversate": {"work_route": "never"}},
            },
        )
    )

    assert config.should_segment("whatsapp") is True
    assert config.confidence_threshold_for("whatsapp") == 0.8
    assert config.allows_work_note("whatsapp") is True


# --- Areas ------------------------------------------------------------------------------
#
# Topic destinations a drop's distilled note can be filed into. Declared rather than derived:
# the engine holds no vault paths of its own.


def _vault_with(tmp_path: Path, **overrides: object) -> VaultConfig:
    """A single-vault config loaded through the real loader, with per-test vault overrides."""
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path,
        {"default_vault": "personal", "vaults": {"personal": {"root": str(root), **overrides}}},
    )
    return load_config(path).vault("personal")


def test_a_vault_with_no_areas_has_an_empty_tuple(tmp_path: Path) -> None:
    assert _vault_with(tmp_path).areas == ()


def test_an_area_carries_its_template_hint_and_aliases(tmp_path: Path) -> None:
    vault = _vault_with(
        tmp_path,
        areas=[
            {
                "name": "finance",
                "note_template": "finance/adhoc/{date}-{slug}.md",
                "when": "markets, macro, portfolio, invoicing",
                "workspaces": ["Finance"],
            }
        ],
    )

    area = vault.areas[0]
    assert area.name == "finance"
    assert area.note_template == "finance/adhoc/{date}-{slug}.md"
    assert area.when.startswith("markets")
    assert area.workspaces == ("Finance",)


def test_area_workspaces_are_optional(tmp_path: Path) -> None:
    vault = _vault_with(
        tmp_path,
        areas=[
            {"name": "realtor", "note_template": "realtor/adhoc/{date}-{slug}.md", "when": "house"}
        ],
    )

    assert vault.areas[0].workspaces == ()


def test_a_single_workspace_string_is_accepted(tmp_path: Path) -> None:
    vault = _vault_with(
        tmp_path,
        areas=[
            {
                "name": "finance",
                "note_template": "finance/adhoc/{date}-{slug}.md",
                "workspaces": "Finance",
            }
        ],
    )

    assert vault.areas[0].workspaces == ("Finance",)


def test_an_area_without_a_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="name"):
        _vault_with(tmp_path, areas=[{"note_template": "x/{date}-{slug}.md", "when": "y"}])


def test_an_area_without_a_note_template_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="note_template"):
        _vault_with(tmp_path, areas=[{"name": "finance", "when": "y"}])


def test_an_area_note_template_without_a_slug_placeholder_is_rejected(tmp_path: Path) -> None:
    """Without {slug} every note in an area files to the same path."""
    with pytest.raises(ConfigError, match="slug"):
        _vault_with(
            tmp_path,
            areas=[{"name": "finance", "note_template": "finance/adhoc/notes.md", "when": "y"}],
        )


def test_two_areas_with_the_same_name_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="duplicate"):
        _vault_with(
            tmp_path,
            areas=[
                {"name": "finance", "note_template": "a/{date}-{slug}.md", "when": "y"},
                {"name": "finance", "note_template": "b/{date}-{slug}.md", "when": "z"},
            ],
        )


def test_an_unknown_key_inside_an_area_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="colour"):
        _vault_with(
            tmp_path,
            areas=[
                {
                    "name": "finance",
                    "note_template": "a/{date}-{slug}.md",
                    "when": "y",
                    "colour": "blue",
                }
            ],
        )


def test_an_area_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        _vault_with(tmp_path, areas=["finance"])


# --- Config discovery (R9) --------------------------------------------------------------
#
# The order is a compatibility promise as much as a feature: an install that has always kept its
# config at `~/.memvault/config.yaml` must keep working after the XDG location exists, and the
# cwd lookup — which most of this suite and every scratch vault leans on — must stay last rather
# than disappear.


def write_config_naming(path: Path, vault_name: str, root: Path) -> Path:
    """Write a valid config at exactly `path` whose single vault is called `vault_name`.

    The vault name is the fingerprint: a discovery test asserts *which* file won by reading
    `load_config().vault().name`, which no path-shaped assertion could tell apart.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"vaults": {vault_name: {"root": str(root)}}}), encoding="utf-8")
    return path


@pytest.fixture
def every_discovery_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A valid config in every place `find_config` looks, each naming a different vault."""
    root = make_vault(tmp_path, "vault")
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)

    locations = {
        "explicit": write_config_naming(tmp_path / "explicit.yaml", "explicit", root),
        "env": write_config_naming(tmp_path / "env.yaml", "env", root),
        "xdg": write_config_naming(home / ".config" / "memvault" / "config.yaml", "xdg", root),
        "legacy": write_config_naming(home / ".memvault" / "config.yaml", "legacy", root),
        "cwd": write_config_naming(cwd / "memvault.config.yaml", "cwd", root),
    }
    monkeypatch.setenv(CONFIG_ENV_VAR, str(locations["env"]))
    return locations


def test_an_explicit_path_beats_every_other_location(
    every_discovery_location: dict[str, Path],
) -> None:
    assert load_config(every_discovery_location["explicit"]).vault().name == "explicit"


def test_the_env_var_beats_the_user_level_and_cwd_locations(
    every_discovery_location: dict[str, Path],
) -> None:
    assert load_config().vault().name == "env"


def test_xdg_beats_the_legacy_and_cwd_locations(
    every_discovery_location: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR)

    assert load_config().vault().name == "xdg"


def test_the_legacy_home_config_is_found_when_xdg_is_absent(
    every_discovery_location: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An install predating the XDG location keeps working without being moved."""
    monkeypatch.delenv(CONFIG_ENV_VAR)
    every_discovery_location["xdg"].unlink()

    assert load_config().vault().name == "legacy"


def test_the_cwd_config_is_the_last_resort(
    every_discovery_location: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR)
    every_discovery_location["xdg"].unlink()
    every_discovery_location["legacy"].unlink()

    assert load_config().vault().name == "cwd"


def test_xdg_config_home_relocates_the_xdg_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_vault(tmp_path, "vault")
    elsewhere = tmp_path / "elsewhere"
    write_config_naming(elsewhere / "memvault" / "config.yaml", "relocated", root)
    monkeypatch.setenv(XDG_CONFIG_HOME_VAR, str(elsewhere))
    monkeypatch.chdir(tmp_path)

    assert load_config().vault().name == "relocated"


def test_no_config_anywhere_names_every_place_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError) as caught:
        find_config()

    message = str(caught.value)
    assert ".config/memvault/config.yaml" in message
    assert ".memvault/config.yaml" in message
    assert "memvault.config.yaml" in message
    assert CONFIG_ENV_VAR in message


def test_the_suite_cannot_discover_a_real_user_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conftest guard, asserted: on a machine that runs MemVault for real, both home
    locations exist, and without the guard this call would load a live vault's config.
    """
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match="no config file found"):
        find_config()


# --- Ranking weights (R4) ---------------------------------------------------------------
#
# Like `index.min_similarity`, these defaults are calibrated against a model rather than derived,
# so the tests pin the shipped numbers: a silent drift would change every recall ordering while
# every other test kept passing.


def test_ranking_defaults_are_the_calibrated_values(tmp_path: Path) -> None:
    ranking = load_config(_single(tmp_path, {})).ranking

    assert (ranking.keyword, ranking.semantic) == (1.0, 1.0)
    assert (ranking.importance, ranking.recency, ranking.graph) == (0.01, 0.005, 0.005)
    assert ranking.half_life_days == 180.0


def test_ranking_weights_are_overridable(tmp_path: Path) -> None:
    config = load_config(
        _single(
            tmp_path,
            {
                "ranking": {
                    "keyword": 0.5,
                    "semantic": 2,
                    "importance": 0.02,
                    "recency": 0,
                    "graph": 0.004,
                    "half_life_days": 90,
                }
            },
        )
    )

    assert config.ranking.keyword == 0.5
    assert config.ranking.semantic == 2.0
    assert config.ranking.importance == 0.02
    assert config.ranking.recency == 0.0
    assert config.ranking.graph == 0.004
    assert config.ranking.half_life_days == 90.0


def test_a_negative_ranking_weight_is_rejected_by_name(tmp_path: Path) -> None:
    """A negative weight would rank a *better* signal lower, which no config author means."""
    with pytest.raises(ConfigError, match="ranking.importance"):
        load_config(_single(tmp_path, {"ranking": {"importance": -0.01}}))


def test_a_half_life_of_zero_is_rejected_by_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="half_life_days"):
        load_config(_single(tmp_path, {"ranking": {"half_life_days": 0}}))


def test_a_negative_half_life_is_rejected_by_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="half_life_days"):
        load_config(_single(tmp_path, {"ranking": {"half_life_days": -30}}))


def test_a_non_finite_ranking_weight_is_rejected_by_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="ranking.recency"):
        load_config(_single(tmp_path, {"ranking": {"recency": float("inf")}}))


def test_an_unknown_ranking_key_is_rejected_by_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="halflife_days"):
        load_config(_single(tmp_path, {"ranking": {"halflife_days": 180}}))


# --- Classifier presets (R11) -----------------------------------------------------------


def test_the_default_preset_is_claude(tmp_path: Path) -> None:
    assert load_config(_single(tmp_path, {})).classifier.preset == "claude"


def test_a_named_preset_parses(tmp_path: Path) -> None:
    config = load_config(_single(tmp_path, {"classifier": {"preset": "codex"}}))

    assert config.classifier.preset == "codex"


def test_a_preset_is_read_case_insensitively_and_trimmed(tmp_path: Path) -> None:
    config = load_config(_single(tmp_path, {"classifier": {"preset": "  OpenCode "}}))

    assert config.classifier.preset == "opencode"


def test_an_unknown_preset_names_the_value_and_the_valid_set(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="gemini"):
        load_config(_single(tmp_path, {"classifier": {"preset": "gemini"}}))


# --- The claude-mem toggle (R11) --------------------------------------------------------
#
# claude-mem is one adopter's plugin, not a component of this engine, so a vault must be able to
# say it has none. The default stays `true` because configs written before the key existed have
# a populated `claude_mem_projects` map and must keep reading their database.


def test_claude_mem_defaults_to_enabled_when_projects_are_configured(tmp_path: Path) -> None:
    vault = _vault_with(tmp_path, claude_mem_projects={"Homeworld": ["Homeworld"]})

    assert vault.claude_mem is True
    assert vault.claude_mem_enabled is True


def test_claude_mem_false_disables_the_source(tmp_path: Path) -> None:
    projects = {"Homeworld": ["Homeworld"]}
    vault = _vault_with(tmp_path, claude_mem=False, claude_mem_projects=projects)

    assert vault.claude_mem_enabled is False


def test_no_configured_projects_is_the_same_statement_as_disabled(tmp_path: Path) -> None:
    """An empty project map leaves the source nothing it is allowed to read anyway."""
    vault = _vault_with(tmp_path, claude_mem=True)

    assert vault.claude_mem_enabled is False


def test_claude_mem_must_be_a_real_boolean(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="claude_mem"):
        _vault_with(tmp_path, claude_mem="yes")


# --- The shipped example config -----------------------------------------------------------
#
# `memvault.config.example.yaml` is the annotated reference an adopter copies from, and it is the
# only config in the repo that documents every key. A key renamed in `config.py` without being
# renamed there ships a file that fails to load on first contact, which is the worst possible
# moment to meet a validation error.


def test_the_shipped_example_config_loads(tmp_path: Path) -> None:
    """Only the paths are rewritten — real vault roots would make this a test of one machine."""
    example = Path(__file__).resolve().parents[1] / "memvault.config.example.yaml"
    raw = yaml.safe_load(example.read_text(encoding="utf-8"))
    for name, settings in raw["vaults"].items():
        settings["root"] = str(make_vault(tmp_path, name))
    raw["index"]["path"] = str(tmp_path / "index" / "{vault}.db")

    config = load_config(write_config(tmp_path, raw, name="from-example.yaml"))

    assert set(config.vaults) == set(raw["vaults"])
    assert config.classifier.preset == "claude"


def test_the_example_config_names_nobody_in_particular(tmp_path: Path) -> None:
    """It ships to strangers (R13): a personal path or project name in it is a leak."""
    example = Path(__file__).resolve().parents[1] / "memvault.config.example.yaml"
    text = example.read_text(encoding="utf-8")

    for path in ("~/Developer", "/Users/", "olyx"):
        assert path not in text
