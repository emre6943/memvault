"""Read-only queries against the claude-mem database.

Every test builds its own SQLite file with the live schema's shape. None of them reads the real
`~/.claude-mem/claude-mem.db`: it is a third-party file whose contents change under the suite,
and a test that read it would be asserting on someone's week rather than on this code.

The four verified traps each get a test that fails if the trap is walked into again — the
millisecond scale, the WAL-safe open, the absent `sdk_sessions` rows, and the shared work/
personal corpus. They are the tests that matter here; the rest is parsing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import date, datetime, time
from pathlib import Path

import pytest

from memvault.claude_mem import (
    ClaudeMemError,
    EpochScale,
    ObservationWindow,
    detect_epoch_scale,
    fetch_window,
    open_database,
    read_only_uri,
    resolve_db_path,
    window_bounds,
)
from memvault.config import VaultConfig

#: The live schema, reduced to the columns this module reads plus the ones whose absence would
#: change behavior (`merged_into_project`, the `sdk_sessions` foreign key). Column order and
#: nullability match the real file so a query that works here works there.
SCHEMA = """
CREATE TABLE sdk_sessions (
    memory_session_id TEXT PRIMARY KEY,
    project TEXT
);

CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_session_id TEXT NOT NULL,
    project TEXT NOT NULL,
    text TEXT,
    type TEXT NOT NULL,
    title TEXT,
    subtitle TEXT,
    facts TEXT,
    narrative TEXT,
    concepts TEXT,
    files_read TEXT,
    files_modified TEXT,
    prompt_number INTEGER,
    created_at TEXT NOT NULL,
    created_at_epoch INTEGER NOT NULL,
    content_hash TEXT,
    generated_by_model TEXT,
    merged_into_project TEXT,
    agent_type TEXT,
    agent_id TEXT,
    metadata TEXT,
    FOREIGN KEY(memory_session_id) REFERENCES sdk_sessions(memory_session_id)
);

CREATE TABLE session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_session_id TEXT NOT NULL,
    project TEXT NOT NULL,
    request TEXT,
    investigated TEXT,
    learned TEXT,
    completed TEXT,
    next_steps TEXT,
    files_read TEXT,
    files_edited TEXT,
    notes TEXT,
    prompt_number INTEGER,
    created_at TEXT NOT NULL,
    created_at_epoch INTEGER NOT NULL,
    merged_into_project TEXT
);
"""

WINDOW_SINCE = date(2026, 7, 26)
WINDOW_UNTIL = date(2026, 8, 1)


def noon(day: date) -> datetime:
    """Midday local time — unambiguously inside its own day, whatever the bound arithmetic does."""
    return datetime.combine(day, time(12, 0))


def epoch_ms(day: date) -> int:
    return int(noon(day).timestamp() * 1000)


def epoch_s(day: date) -> int:
    return int(noon(day).timestamp())


class Builder:
    """A fixture database, written the way the plugin writes one."""

    def __init__(self, path: Path, *, milliseconds: bool = True) -> None:
        self.path = path
        self.milliseconds = milliseconds
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self._sessions = 0

    def _epoch(self, day: date) -> int:
        return epoch_ms(day) if self.milliseconds else epoch_s(day)

    def session(self, project: str) -> str:
        """A session row, for the minority of observations that still have one."""
        self._sessions += 1
        session_id = f"session-{self._sessions}"
        self.conn.execute(
            "INSERT INTO sdk_sessions (memory_session_id, project) VALUES (?, ?)",
            (session_id, project),
        )
        return session_id

    def observation(
        self,
        project: str,
        day: date,
        *,
        session_id: str = "orphaned-session",
        type_: str = "discovery",
        title: str = "A thing happened",
        facts: str | None = None,
        concepts: str | None = None,
        merged_into: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO observations (memory_session_id, project, type, title, subtitle, "
            "facts, narrative, concepts, files_modified, created_at, created_at_epoch, "
            "merged_into_project) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                project,
                type_,
                title,
                "a subtitle",
                facts,
                "a narrative",
                concepts,
                '["src/app.py"]',
                noon(day).isoformat(),
                self._epoch(day),
                merged_into,
            ),
        )

    def summary(
        self,
        project: str,
        day: date,
        *,
        request: str = "Do the thing",
        merged_into: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO session_summaries (memory_session_id, project, request, investigated, "
            "learned, completed, next_steps, notes, created_at, created_at_epoch, "
            "merged_into_project) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "orphaned-session",
                project,
                request,
                "looked around",
                "learned a thing",
                "did the thing",
                "next: the other thing",
                "",
                noon(day).isoformat(),
                self._epoch(day),
                merged_into,
            ),
        )

    def close(self) -> Path:
        self.conn.commit()
        self.conn.close()
        return self.path


@pytest.fixture
def personal_vault(tmp_path: Path) -> VaultConfig:
    """A vault whose alias map claims personal projects and folds two spellings into one."""
    root = tmp_path / "personal"
    root.mkdir()
    return VaultConfig(
        name="personal",
        root=root,
        inbox=root / "inbox",
        claude_mem_projects={
            "Homeworld": ("Homeworld",),
            "TrackFusion": ("TrackFusion",),
            "tcg-vendor": ("tcg-vendor", "TCGVendor"),
        },
    )


@pytest.fixture
def work_vault(tmp_path: Path) -> VaultConfig:
    root = tmp_path / "work"
    root.mkdir()
    return VaultConfig(
        name="work",
        root=root,
        inbox=root / "inbox",
        claude_mem_projects={"oryx": ("oryx", "oryx-workspace-1"), "olyx": ("olyx",)},
    )


def build(tmp_path: Path, name: str = "claude-mem.db", *, milliseconds: bool = True) -> Builder:
    return Builder(tmp_path / name, milliseconds=milliseconds)


def read(vault: VaultConfig, path: Path) -> ObservationWindow:
    return fetch_window(vault, WINDOW_SINCE, WINDOW_UNTIL, db_path=path)


class TestConnection:
    """How the database is opened, which is the difference between fresh data and stale data."""

    def test_the_uri_is_read_only_and_never_immutable(self, tmp_path: Path) -> None:
        uri = read_only_uri(tmp_path / "claude-mem.db")

        assert "mode=ro" in uri
        # `immutable=1` makes SQLite skip the WAL, which on this database means silently
        # missing the newest sessions — the ones a weekly pass exists to read.
        assert "immutable" not in uri

    def test_a_path_with_a_question_mark_stays_part_of_the_path(self, tmp_path: Path) -> None:
        uri = read_only_uri(tmp_path / "why?.db")

        assert "why%3F.db" in uri
        assert uri.endswith("?mode=ro")

    def test_a_missing_database_raises_a_reason_naming_the_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.db"

        with pytest.raises(ClaudeMemError, match=str(missing)), open_database(missing):
            pass

    def test_a_wal_write_is_visible_to_a_read_only_reader(self, tmp_path: Path) -> None:
        """The property `immutable=1` would break, asserted directly rather than by inspection."""
        path = build(tmp_path).close()
        writer = sqlite3.connect(path)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute(
            "INSERT INTO observations (memory_session_id, project, type, created_at, "
            "created_at_epoch) VALUES ('s', 'Homeworld', 'discovery', '2026-08-01', ?)",
            (epoch_ms(date(2026, 8, 1)),),
        )
        writer.commit()

        with open_database(path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        writer.close()

        assert count == 1

    def test_the_reader_cannot_write(self, tmp_path: Path) -> None:
        path = build(tmp_path).close()

        with open_database(path) as conn, pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM observations")

    def test_resolve_expands_a_home_relative_path(self) -> None:
        assert resolve_db_path("~/x.db").is_absolute()


class TestTimestampScale:
    """The trap that makes a wrong window look like a very productive week."""

    def test_milliseconds_are_detected_and_only_in_window_rows_come_back(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", date(2026, 7, 28), title="inside")
        builder.observation("Homeworld", date(2026, 5, 1), title="long before")
        builder.observation("Homeworld", date(2026, 9, 9), title="long after")
        path = builder.close()

        window = read(personal_vault, path)

        assert window.scale is EpochScale.MILLISECONDS
        assert [item.title for item in window.observations] == ["inside"]

    def test_a_seconds_scale_database_is_detected_rather_than_matching_everything(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path, milliseconds=False)
        builder.observation("Homeworld", date(2026, 7, 28), title="inside")
        builder.observation("Homeworld", date(2026, 5, 1), title="long before")
        path = builder.close()

        window = read(personal_vault, path)

        assert window.scale is EpochScale.SECONDS
        # The point of detecting it: the out-of-window row stays out. Treating a seconds column
        # as milliseconds would put every row ever written inside the week.
        assert [item.title for item in window.observations] == ["inside"]

    def test_an_empty_database_reports_an_unknown_scale_without_failing(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        window = read(personal_vault, build(tmp_path).close())

        assert window.scale is EpochScale.UNKNOWN
        assert window.empty
        assert window.available

    def test_the_scale_is_sampled_from_the_data_not_assumed(self, tmp_path: Path) -> None:
        builder = build(tmp_path, milliseconds=False)
        builder.summary("Homeworld", date(2026, 7, 28))
        path = builder.close()

        with open_database(path) as conn:
            assert detect_epoch_scale(conn) is EpochScale.SECONDS

    def test_bounds_are_half_open_and_scaled(self) -> None:
        start_ms, end_ms = window_bounds(WINDOW_SINCE, WINDOW_UNTIL, EpochScale.MILLISECONDS)
        start_s, end_s = window_bounds(WINDOW_SINCE, WINDOW_UNTIL, EpochScale.SECONDS)

        assert start_ms < epoch_ms(WINDOW_SINCE) < epoch_ms(WINDOW_UNTIL) < end_ms
        assert (start_ms // 1000, end_ms // 1000) == (start_s, end_s)


class TestNoJoinThroughSessions:
    """~79% of observations have no `sdk_sessions` row. An inner join would drop all of them."""

    def test_orphaned_observations_are_returned(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        attached = builder.session("Homeworld")
        builder.observation("Homeworld", date(2026, 7, 28), session_id=attached, title="attached")
        builder.observation("Homeworld", date(2026, 7, 29), title="orphaned")
        builder.observation("Homeworld", date(2026, 7, 30), title="also orphaned")
        path = builder.close()

        window = read(personal_vault, path)

        assert [item.title for item in window.observations] == [
            "attached",
            "orphaned",
            "also orphaned",
        ]

    def test_orphaned_session_summaries_are_returned(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.summary("TrackFusion", date(2026, 7, 27), request="ship the thing")
        path = builder.close()

        window = read(personal_vault, path)

        assert [item.request for item in window.summaries] == ["ship the thing"]

    def test_a_table_with_no_sessions_at_all_still_returns_everything(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        for day in (date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)):
            builder.observation("Homeworld", day)
        path = builder.close()

        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM sdk_sessions").fetchone()[0] == 0

        assert len(read(personal_vault, path).observations) == 3


class TestProjectBoundary:
    """One database, two lives. The alias map is the wall between them."""

    def test_work_projects_are_excluded_from_a_personal_vault(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", date(2026, 7, 28), title="mine")
        builder.observation("oryx", date(2026, 7, 28), title="work")
        builder.observation("dashboard-workspace-2", date(2026, 7, 28), title="also work")
        builder.summary("olyx", date(2026, 7, 28))
        path = builder.close()

        window = read(personal_vault, path)

        assert [item.title for item in window.observations] == ["mine"]
        assert window.summaries == ()

    def test_personal_projects_are_excluded_from_a_work_vault(
        self, tmp_path: Path, work_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", date(2026, 7, 28), title="personal")
        builder.observation("oryx-workspace-1", date(2026, 7, 28), title="work")
        path = builder.close()

        window = read(work_vault, path)

        assert [item.title for item in window.observations] == ["work"]
        assert [item.project for item in window.observations] == ["oryx"]

    def test_an_unlisted_project_is_skipped_and_counted_never_defaulted_in(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("some-new-thing", date(2026, 7, 28))
        builder.observation("some-new-thing", date(2026, 7, 29))
        builder.summary("some-new-thing", date(2026, 7, 29))
        path = builder.close()

        window = read(personal_vault, path)

        assert window.observations == ()
        assert window.summaries == ()
        assert [(item.project, item.observations, item.summaries) for item in window.skipped] == [
            ("some-new-thing", 2, 1)
        ]
        assert "some-new-thing (3)" in window.skipped_line()

    def test_the_alias_map_folds_two_spellings_into_one_canonical_project(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("tcg-vendor", date(2026, 7, 27), title="one")
        builder.observation("TCGVendor", date(2026, 7, 28), title="two")
        builder.summary("TCGVendor", date(2026, 7, 28))
        path = builder.close()

        window = read(personal_vault, path)

        assert {item.project for item in window.observations} == {"tcg-vendor"}
        assert {item.raw_project for item in window.observations} == {"tcg-vendor", "TCGVendor"}
        assert window.projects == ("tcg-vendor",)
        assert window.skipped == ()

    def test_a_skipped_project_does_not_make_the_read_unavailable(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("olyx", date(2026, 7, 28))
        path = builder.close()

        window = read(personal_vault, path)

        assert window.available
        assert window.degraded is None


class TestRowFiltering:
    """Merged rows, and the JSON columns that can arrive malformed."""

    def test_rows_with_merged_into_project_set_are_excluded(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", date(2026, 7, 28), title="live")
        builder.observation("Homeworld", date(2026, 7, 28), title="merged", merged_into="Homeworld")
        builder.summary("Homeworld", date(2026, 7, 28), request="live")
        builder.summary("Homeworld", date(2026, 7, 28), request="merged", merged_into="Homeworld")
        path = builder.close()

        window = read(personal_vault, path)

        assert [item.title for item in window.observations] == ["live"]
        assert [item.request for item in window.summaries] == ["live"]

    def test_empty_string_merged_into_project_is_treated_as_unset(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", date(2026, 7, 28), title="live", merged_into="")
        path = builder.close()

        assert [item.title for item in read(personal_vault, path).observations] == ["live"]

    def test_json_columns_parse_into_lists(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation(
            "Homeworld",
            date(2026, 7, 28),
            facts='["the epoch columns are milliseconds", "the WAL matters"]',
            concepts='["claude-mem", "sqlite"]',
        )
        path = builder.close()

        observation = read(personal_vault, path).observations[0]

        assert observation.facts == ("the epoch columns are milliseconds", "the WAL matters")
        assert observation.concepts == ("claude-mem", "sqlite")
        assert observation.files_modified == ("src/app.py",)
        assert observation.malformed == ()

    def test_a_malformed_json_column_degrades_its_own_row_and_nothing_else(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", date(2026, 7, 27), title="broken", facts="{not json")
        builder.observation("Homeworld", date(2026, 7, 28), title="fine", facts='["a fact"]')
        path = builder.close()

        window = read(personal_vault, path)

        broken, fine = window.observations
        assert broken.title == "broken"
        assert (broken.facts, broken.malformed) == ((), ("facts",))
        assert fine.facts == ("a fact",)

    def test_a_json_object_where_a_list_belongs_degrades_the_row(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", date(2026, 7, 28), concepts='{"a": 1}')
        path = builder.close()

        observation = read(personal_vault, path).observations[0]

        assert observation.concepts == ()
        assert "concepts" in observation.malformed

    def test_null_json_columns_are_empty_not_malformed(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", date(2026, 7, 28), facts=None, concepts=None)
        path = builder.close()

        observation = read(personal_vault, path).observations[0]

        assert (observation.facts, observation.concepts, observation.malformed) == ((), (), ())


class TestTheToggle:
    """A source the vault turned off is not a source that failed (R11).

    Every adopter starts here: no claude-mem plugin, no project map, and a weekly pass that
    must not open by reporting a missing third-party database it was never asked to read.
    """

    def test_a_vault_with_claude_mem_off_does_not_read_the_database(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", WINDOW_SINCE, title="Would have been read")
        path = builder.close()

        window = read(replace(personal_vault, claude_mem=False), path)

        # The file is right there and full of rows for a mapped project. The toggle, not the
        # file's absence, is what decides.
        assert window.observations == ()
        assert window.summaries == ()

    def test_a_vault_with_no_project_map_does_not_read_the_database(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        builder = build(tmp_path)
        builder.observation("Homeworld", WINDOW_SINCE)
        path = builder.close()

        window = read(replace(personal_vault, claude_mem_projects={}), path)

        assert window.empty

    def test_a_disabled_source_is_available_rather_than_degraded(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        # The whole point: `degraded` is what turns a reflect run PARTIAL, and a source nobody
        # configured must not do that.
        window = read(replace(personal_vault, claude_mem=False), tmp_path / "absent.db")

        assert window.available
        assert window.degraded is None

    def test_nothing_is_reported_as_skipped_when_the_source_is_off(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        # A skip line names projects left outside the vault's boundary. With the source off
        # there is no boundary being drawn, and a digest that listed skips would be describing
        # a read that never happened.
        builder = build(tmp_path)
        builder.observation("oryx", WINDOW_SINCE)
        path = builder.close()

        window = read(replace(personal_vault, claude_mem=False), path)

        assert window.skipped == ()
        assert window.skipped_line() == ""


class TestDegradation:
    """A third-party database that cannot be read must cost a warning, not the pass."""

    def test_a_missing_database_degrades_with_a_reason(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        window = read(personal_vault, tmp_path / "absent.db")

        assert not window.available
        assert window.degraded is not None
        assert "was not found" in window.degraded
        assert window.empty

    def test_a_file_that_is_not_a_database_degrades_rather_than_raising(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        path = tmp_path / "corrupt.db"
        path.write_bytes(b"this is not a SQLite file, it is a note about one")

        window = read(personal_vault, path)

        assert not window.available
        assert window.observations == ()

    def test_a_database_missing_the_expected_tables_degrades(
        self, tmp_path: Path, personal_vault: VaultConfig
    ) -> None:
        path = tmp_path / "old-schema.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
        conn.commit()
        conn.close()

        window = read(personal_vault, path)

        assert not window.available
        assert window.degraded is not None
        assert "query failed" in window.degraded


def test_the_suite_guard_redirects_the_default_database_path() -> None:
    """The autouse guard in `conftest` is live: nothing here can reach the real corpus."""
    assert not resolve_db_path().exists()
