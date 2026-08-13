"""CLI dispatch and the `doctor` health check.

`doctor` reports and never repairs, following the vault shell scripts' `--check` convention:
problems print with their remedy and the exit status is non-zero.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from memvault import cli
from memvault.cli import main
from memvault.config import IndexConfig
from memvault.index import NULL_MODEL_ID, SEMANTIC_EXTRA_HINT, IndexBuildError, IndexStats
from memvault.ingest import EXIT_FAILED, EXIT_PARTIAL, HeldItem, IngestReport
from memvault.reflect import AppendedEntry, ReflectReport
from tests.test_config import make_vault, write_config


def fake_index(tmp_path: Path, *vault_names: str) -> None:
    """Stand in for a built index so `doctor` sees a set-up vault.

    A missing index is a real problem — `recall` returns nothing without one — so tests that
    assert a healthy vault have to provide one.
    """
    directory = tmp_path / "index"
    directory.mkdir(exist_ok=True)
    for name in vault_names:
        (directory / f"{name}.db").write_bytes(b"")


@pytest.fixture
def healthy_config(tmp_path: Path) -> Path:
    personal = make_vault(tmp_path, "personal")
    work = make_vault(tmp_path, "work")
    fake_index(tmp_path, "personal", "work")
    return write_config(
        tmp_path,
        {
            "default_vault": "personal",
            "vaults": {
                "personal": {"root": str(personal), "work_route": "work"},
                "work": {"root": str(work)},
            },
            # Pinned into tmp_path so the suite never reads or writes the real ~/.memvault.
            "index": {"path": str(tmp_path / "index" / "{vault}.db")},
        },
    )


def test_doctor_on_healthy_vault_exits_zero(
    healthy_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", str(healthy_config), "doctor"])

    assert code == 0
    assert "all checks healthy" in capsys.readouterr().out


def test_doctor_reports_the_work_route(
    healthy_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--config", str(healthy_config), "doctor"])

    assert "work route → 'work'" in capsys.readouterr().out


def test_doctor_flags_missing_inbox_and_names_the_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_vault(tmp_path, "personal", inbox=False)
    fake_index(tmp_path, "personal")
    config = write_config(
        tmp_path,
        {
            "vaults": {"personal": {"root": str(root)}},
            "index": {"path": str(tmp_path / "index" / "{vault}.db")},
        },
    )

    code = main(["--config", str(config), "doctor"])
    out = capsys.readouterr().out

    assert code == 1
    assert "MISSING" in out
    assert str(root / "inbox") in out
    assert "create it, then re-run" in out
    assert "problems: 1" in out


class TestDoctorIndexFreshness:
    """A stale index fails silently — recall keeps answering from an older vault."""

    def test_missing_index_is_a_problem_naming_the_remedy(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = make_vault(tmp_path, "personal")
        config = write_config(
            tmp_path,
            {
                "vaults": {"personal": {"root": str(root)}},
                "index": {"path": str(tmp_path / "index" / "{vault}.db")},
            },
        )

        code = main(["--config", str(config), "doctor"])
        out = capsys.readouterr().out

        assert code == 1
        assert "no index at" in out
        assert "run `memvault index`" in out

    def test_index_older_than_the_vault_is_stale(
        self, healthy_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        note = tmp_path / "personal" / "note.md"
        note.write_text("# later than the index", encoding="utf-8")
        os.utime(note, (time.time() + 60, time.time() + 60))

        code = main(["--config", str(healthy_config), "doctor"])
        out = capsys.readouterr().out

        assert code == 1
        assert "STALE" in out
        assert "run `memvault index`" in out

    def test_a_file_only_in_the_inbox_does_not_make_the_index_stale(
        self, healthy_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        drop = tmp_path / "personal" / "inbox" / "pending.md"
        drop.write_text("not filed yet", encoding="utf-8")
        os.utime(drop, (time.time() + 60, time.time() + 60))

        main(["--config", str(healthy_config), "doctor"])

        assert "STALE" not in capsys.readouterr().out


class TestDoctorStuckItems:
    """A held drop is the system working; a held drop nobody looks at is not."""

    def test_review_markers_are_reported_with_their_reason(
        self, healthy_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        inbox = tmp_path / "personal" / "inbox"
        (inbox / "murky.txt").write_text("ambiguous", encoding="utf-8")
        (inbox / "murky.txt.needs-review").write_text(
            "confidence 0.30 is below the 0.70 threshold", encoding="utf-8"
        )

        code = main(["--config", str(healthy_config), "doctor"])
        out = capsys.readouterr().out

        assert code == 1
        assert "STUCK" in out
        assert "confidence 0.30 is below the 0.70 threshold" in out

    def test_markers_are_not_counted_as_pending_drops(
        self, healthy_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        inbox = tmp_path / "personal" / "inbox"
        (inbox / "murky.txt").write_text("ambiguous", encoding="utf-8")
        (inbox / "murky.txt.needs-review").write_text("unclear", encoding="utf-8")

        main(["--config", str(healthy_config), "doctor"])
        out = capsys.readouterr().out

        # Two files sit in the inbox, but only one of them is a drop.
        assert "1 item(s)" in out

    def test_a_clean_inbox_reports_no_stuck_items(
        self, healthy_config: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--config", str(healthy_config), "doctor"])

        assert "STUCK" not in capsys.readouterr().out


def test_doctor_counts_pending_inbox_items(
    healthy_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "personal" / "inbox" / "note.txt").write_text("hi", encoding="utf-8")

    code = main(["--config", str(healthy_config), "doctor"])
    out = capsys.readouterr().out

    assert code == 0
    assert "1 item(s)" in out


def test_vault_flag_selects_the_named_vault(
    healthy_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--config", str(healthy_config), "--vault", "work", "doctor"])

    assert "vault 'work'" in capsys.readouterr().out


def test_unknown_vault_flag_exits_with_config_error(
    healthy_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", str(healthy_config), "--vault", "ghost", "doctor"])

    assert code == 1
    assert "unknown vault 'ghost'" in capsys.readouterr().err


def test_missing_config_exits_with_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", str(tmp_path / "absent.yaml"), "doctor"])

    assert code == 1
    assert "config error" in capsys.readouterr().err


class TestReflectCommand:
    """`reflect` is wired to U9's pass. A stub keeps both the `claude` CLI and the real
    claude-mem database out of the suite."""

    def _stub_pass(self, monkeypatch: pytest.MonkeyPatch, report: ReflectReport) -> list[dict]:
        seen: list[dict] = []

        def fake_run(_config: object, vault: object, **kwargs: object) -> ReflectReport:
            seen.append({"vault": getattr(vault, "name", "?"), **kwargs})
            return report

        monkeypatch.setattr(cli, "run_reflect", fake_run)
        return seen

    def _report(self, **overrides: object) -> ReflectReport:
        fields: dict = {
            "vault": "personal",
            "since": date(2026, 7, 26),
            "until": date(2026, 8, 1),
        }
        fields.update(overrides)
        return ReflectReport(**fields)

    def test_a_clean_pass_exits_zero_and_prints_what_it_wrote(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        report = self._report(
            appended=(
                AppendedEntry(
                    project="Sapik",
                    kind="learning",
                    relative_path="repos/Sapik/learnings.md",
                    heading="## 2026-08-01 — a claim",
                ),
            ),
            digest_path="repos/Vault/digest/2026-W31.md",
        )
        self._stub_pass(monkeypatch, report)

        code = main(["--config", str(healthy_config), "reflect"])
        out = capsys.readouterr().out

        assert code == 0
        assert "APPENDED repos/Sapik/learnings.md — 2026-08-01 — a claim" in out
        assert "DIGEST   repos/Vault/digest/2026-W31.md" in out

    def test_a_degraded_pass_exits_three_and_names_the_reason(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._stub_pass(monkeypatch, self._report(warnings=("claude-mem unavailable: not found",)))

        code = main(["--config", str(healthy_config), "reflect"])

        assert code == EXIT_PARTIAL
        assert "claude-mem unavailable" in capsys.readouterr().err

    def test_a_failed_pass_exits_one(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._stub_pass(monkeypatch, self._report(failure="the curator exited 1"))

        code = main(["--config", str(healthy_config), "reflect"])

        assert code == EXIT_FAILED
        assert "the curator exited 1" in capsys.readouterr().err

    def test_window_flags_reach_the_pass(
        self, healthy_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._stub_pass(monkeypatch, self._report())

        main(
            [
                "--config",
                str(healthy_config),
                "reflect",
                "--since",
                "2026-07-01",
                "--until",
                "2026-07-31",
                "--claude-mem-db",
                "/tmp/fixture.db",
            ]
        )

        assert seen[0]["since"] == date(2026, 7, 1)
        assert seen[0]["until"] == date(2026, 7, 31)
        assert seen[0]["db_path"] == "/tmp/fixture.db"

    def test_a_malformed_window_date_is_a_usage_error(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._stub_pass(monkeypatch, self._report())

        code = main(["--config", str(healthy_config), "reflect", "--since", "last tuesday"])

        assert code == 1
        assert "--since must be a YYYY-MM-DD date" in capsys.readouterr().err


class TestIngestCommand:
    """`ingest` is wired to U6's pass. A stub keeps the `claude` CLI out of the suite."""

    def _stub_pass(self, monkeypatch: pytest.MonkeyPatch, report: IngestReport) -> list[str]:
        seen: list[str] = []

        def fake_run(_config: object, vault: object, **_kwargs: object) -> IngestReport:
            seen.append(getattr(vault, "name", "?"))
            return report

        monkeypatch.setattr(cli, "run_ingest", fake_run)
        return seen

    def test_an_empty_pass_exits_zero(
        self,
        healthy_config: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen = self._stub_pass(monkeypatch, IngestReport(vault="personal"))

        code = main(["--config", str(healthy_config), "ingest"])

        assert code == 0
        assert seen == ["personal"]
        assert "0 filed" in capsys.readouterr().out

    def test_a_held_item_exits_with_the_partial_status_and_points_at_the_marker(
        self,
        healthy_config: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        report = IngestReport(
            vault="personal", held=(HeldItem("murky.md", "abc", "confidence 0.30 is too low"),)
        )
        self._stub_pass(monkeypatch, report)

        code = main(["--config", str(healthy_config), "ingest"])
        captured = capsys.readouterr()

        assert code == EXIT_PARTIAL
        assert "REVIEW   murky.md — confidence 0.30 is too low" in captured.out
        assert ".needs-review" in captured.err

    def test_a_failure_exits_one_and_names_itself(
        self,
        healthy_config: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_pass(
            monkeypatch, IngestReport(vault="personal", failure="vault has uncommitted changes")
        )

        code = main(["--config", str(healthy_config), "ingest"])

        assert code == EXIT_FAILED
        assert "uncommitted changes" in capsys.readouterr().err

    def test_the_log_path_reaches_the_pass(
        self, healthy_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def fake_run(_config: object, _vault: object, **kwargs: object) -> IngestReport:
            seen.update(kwargs)
            return IngestReport(vault="personal")

        monkeypatch.setattr(cli, "run_ingest", fake_run)

        main(["--config", str(healthy_config), "ingest", "--log-path", "/tmp/mv.log"])

        assert seen["log_path"] == "/tmp/mv.log"


class _NamedEmbedder:
    """The smallest thing the index command reads off an embedder: which model it is.

    Named rather than `object()` because the command now branches on `model_id` to decide
    whether the pass it is about to run can embed at all, so a stub without one is not a stub of
    an embedder.
    """

    def __init__(self, model_id: str = "stub-v1") -> None:
        self.model_id = model_id

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class TestIndexCommand:
    """`index` is wired to U7's builder. A stub keeps the model download out of the suite."""

    def _stub_build(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stats: IndexStats,
        *,
        model_id: str = "stub-v1",
    ) -> list[Path]:
        seen: list[Path] = []

        def fake_build(_vault: object, _cfg: object, _emb: object, *, db_path: Path) -> IndexStats:
            seen.append(db_path)
            return stats

        monkeypatch.setattr(cli, "build_index", fake_build)
        monkeypatch.setattr(cli, "passage_embedder", lambda _cfg: _NamedEmbedder(model_id))
        return seen

    def test_reports_file_and_chunk_counts(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stats = IndexStats(
            db_path=Path("/tmp/x.db"),
            files_scanned=7,
            files_indexed=2,
            files_unchanged=5,
            chunks_written=11,
            chunks_embedded=4,
            chunks_reused=7,
            elapsed_seconds=1.25,
        )
        self._stub_build(monkeypatch, stats)

        code = main(["--config", str(healthy_config), "index"])
        out = capsys.readouterr().out

        assert code == 0
        assert "7 file(s) scanned — 2 indexed, 5 unchanged, 0 removed" in out
        assert "11 chunk(s) — 4 embedded, 7 reused, in 1.2s" in out

    def test_reports_the_graph_counts_so_a_rebuild_can_be_compared_against_them(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Disposability is checked by hand this way: rebuild, and these numbers must match."""
        stats = IndexStats(
            db_path=Path("/tmp/x.db"),
            links_written=9,
            observations_written=4,
            links_pending=2,
        )
        self._stub_build(monkeypatch, stats)

        main(["--config", str(healthy_config), "index"])

        out = capsys.readouterr().out
        assert "9 link(s) and 4 observation(s) written — 2 link(s) pending across the vault" in out

    def test_warns_about_the_model_download_on_first_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = make_vault(tmp_path, "fresh")
        config = write_config(
            tmp_path,
            {
                "vaults": {"fresh": {"root": str(root)}},
                "index": {"path": str(tmp_path / "index" / "{vault}.db")},
            },
        )
        self._stub_build(monkeypatch, IndexStats(db_path=Path("/tmp/x.db")))

        main(["--config", str(config), "index"])

        assert "this is not a hang" in capsys.readouterr().out

    def test_stays_quiet_about_the_download_once_an_index_exists(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._stub_build(monkeypatch, IndexStats(db_path=Path("/tmp/x.db")))

        main(["--config", str(healthy_config), "index"])

        assert "this is not a hang" not in capsys.readouterr().out

    def test_a_keyword_only_pass_says_so_before_it_runs(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Said up front, because the index this builds answers no semantic query.

        Learning it later from a search that feels empty is how an adopter concludes the tool is
        bad rather than that one extra is missing.
        """
        self._stub_build(monkeypatch, IndexStats(db_path=Path("/tmp/x.db")), model_id=NULL_MODEL_ID)

        main(["--config", str(healthy_config), "index"])

        out = capsys.readouterr().out
        assert SEMANTIC_EXTRA_HINT in out
        # And not the model-download line: nothing is going to be downloaded.
        assert "this is not a hang" not in out

    def test_a_pass_that_can_embed_says_nothing_about_the_extra(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._stub_build(monkeypatch, IndexStats(db_path=Path("/tmp/x.db")))

        main(["--config", str(healthy_config), "index"])

        assert SEMANTIC_EXTRA_HINT not in capsys.readouterr().out

    def test_indexing_falls_back_to_the_null_backend_rather_than_refusing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an index there is no search at all, not merely no semantic search.

        So the light install indexes; it is the vectors that are missing, never the database.
        """
        monkeypatch.setattr(cli, "semantic_available", lambda: False)

        assert cli.passage_embedder(IndexConfig()).model_id == NULL_MODEL_ID

    def test_indexing_uses_the_real_backend_when_the_extra_is_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "semantic_available", lambda: True)
        config = IndexConfig()

        assert cli.passage_embedder(config).model_id == config.model

    def test_index_writes_to_the_configured_per_vault_path(
        self, healthy_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._stub_build(monkeypatch, IndexStats(db_path=Path("/tmp/x.db")))

        main(["--config", str(healthy_config), "--vault", "work", "index"])

        assert seen[0].name == "work.db"

    def test_build_failure_exits_non_zero(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fake_build(*_: object, **__: object) -> IndexStats:
            raise IndexBuildError("model 'ghost' is not available")

        monkeypatch.setattr(cli, "build_index", fake_build)
        monkeypatch.setattr(cli, "passage_embedder", lambda _cfg: _NamedEmbedder())

        code = main(["--config", str(healthy_config), "index"])

        assert code == 1
        assert "model 'ghost' is not available" in capsys.readouterr().err

    def test_skipped_files_are_reported_with_their_reason(
        self,
        healthy_config: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stats = IndexStats(
            db_path=Path("/tmp/x.db"), skipped=(("notes/binary.md", "not valid UTF-8"),)
        )
        self._stub_build(monkeypatch, stats)

        main(["--config", str(healthy_config), "index"])
        out = capsys.readouterr().out

        assert "SKIPPED" in out
        assert "notes/binary.md — not valid UTF-8" in out


class TestInitCommand:
    """`init` is the one subcommand that must work when there is no config to load.

    Every test here goes through `main`, because the thing being tested is the dispatch order:
    until this unit, `main` loaded config unconditionally, which made the command that creates a
    config unreachable without one.
    """

    def test_it_runs_with_no_config_anywhere(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)

        code = main(["--config", str(tmp_path / "conf.yaml"), "init", str(tmp_path / "v"), "--git"])

        assert code == 0
        assert "CONFIG" in capsys.readouterr().out
        assert (tmp_path / "conf.yaml").exists()

    def test_it_prints_wiring_for_every_harness(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--config", str(tmp_path / "conf.yaml"), "init", str(tmp_path / "v"), "--git"])

        out = capsys.readouterr().out
        for harness in ("Claude Code", "Codex", "opencode", "Cursor"):
            assert harness in out

    def test_no_git_refuses_instead_of_prompting(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A scheduler or a CI job has no terminal; `--no-git` is how it says so out loud."""
        code = main(
            [
                "--config",
                str(tmp_path / "conf.yaml"),
                "init",
                str(tmp_path / "v"),
                "--no-git",
            ]
        )

        assert code == 1
        assert "not a git repository" in capsys.readouterr().err
        assert not (tmp_path / "conf.yaml").exists()

    def test_a_conflicting_config_exits_one_naming_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        existing = tmp_path / "conf.yaml"
        existing.write_text("vaults: {other: {root: /somewhere}}\n", encoding="utf-8")

        code = main(["--config", str(existing), "init", str(tmp_path / "v"), "--git"])

        assert code == 1
        assert str(existing) in capsys.readouterr().err

    def test_a_second_run_reports_the_config_as_already_current(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = ["--config", str(tmp_path / "conf.yaml"), "init", str(tmp_path / "v"), "--git"]
        main(argv)
        capsys.readouterr()

        code = main(argv)

        assert code == 0
        assert "already current" in capsys.readouterr().out

    def test_the_vault_it_writes_is_one_doctor_can_read(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The point of the unit: first contact cannot dead-end. Only the index is still absent."""
        config = tmp_path / "conf.yaml"
        main(["--config", str(config), "init", str(tmp_path / "v"), "--git"])
        capsys.readouterr()

        main(["--config", str(config), "doctor"])
        out = capsys.readouterr().out

        assert "config valid" in out
        assert "vault 'v'" in out
        assert "0 item(s)" in out


def test_a_command_with_no_config_anywhere_offers_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare "no config file found" leaves a first-time adopter with nowhere to go."""
    monkeypatch.chdir(tmp_path)

    code = main(["doctor"])

    assert code == 1
    assert "memvault init" in capsys.readouterr().err


def test_doctor_does_not_count_hidden_files_as_drops(
    healthy_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`.gitkeep` keeps the inbox directory in git; the contract says it is not a drop."""
    (tmp_path / "personal" / "inbox" / ".gitkeep").write_text("", encoding="utf-8")

    code = main(["--config", str(healthy_config), "doctor"])
    out = capsys.readouterr().out

    assert code == 0
    assert "0 item(s)" in out
