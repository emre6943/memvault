"""Idle-triggered ingestion: the markers, the decision, the runner, and the environment.

Three properties carry R7 and each has its own class below.

*The queue marker is the truth.* Every test that kills a runner asserts the marker survived it,
because the failure this design exists to prevent — a request that evaporates with the process
holding it — looks exactly like success from inside the process.

*Quiescence is a pair of markers.* A test that only ended sessions would never notice that
SessionEnd alone cannot see the three terminals still open.

*A refusal is not a failure.* The dirty guard fires far more often under idle-triggering than it
did at 23:00, so the tests pin that a refusal keeps the marker, backs off, and stays quiet until
it has repeated.

No test spawns a real runner except the last one, which spawns a real interpreter under a
deliberately impoverished environment — the honest-environment test the launchd learnings ask
for. Everywhere else the spawner, the clock, the sleep, and the notifier are injected.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import yaml

from memvault.cli import main
from memvault.config import Config, DebounceConfig, VaultConfig, load_config
from memvault.debounce import (
    PATH_ADDITIONS,
    Quiescence,
    RetryState,
    RunnerLock,
    StatePaths,
    Verdict,
    clear_queue,
    clear_session,
    decide,
    enqueue,
    ensure_runner,
    live_sessions,
    mark_session_start,
    next_retry,
    queued_at,
    read_retry,
    run_runner,
    runner_alive,
    runner_argv,
    runner_env,
    runtime_banner,
    should_notify,
    state_home,
    write_retry,
)
from memvault.ingest import IngestReport, IngestStatus
from tests.test_config import make_vault, write_config

SETTINGS = DebounceConfig()

#: A pid that is definitely alive and definitely not this process — the shell that started
#: pytest. Used wherever a test needs a lock somebody *else* is holding, since a lock naming this
#: process is the one case the runner is entitled to adopt.
ANOTHER_LIVE_PID = os.getppid()


@dataclass
class Recorded:
    """One spawn that was asked for, captured instead of performed."""

    argv: list[str]
    env: dict[str, str]
    log: Path


class FakeSpawner:
    """Stands in for a detached process. Returns a pid that is alive: this one."""

    def __init__(self) -> None:
        self.calls: list[Recorded] = []

    def __call__(self, argv: list[str], env: dict[str, str], log: Path) -> int:
        self.calls.append(Recorded(argv, env, log))
        return os.getpid()


class FakeIngest:
    """An ingestion pass that never touches a vault, answering with prepared reports."""

    def __init__(self, *reports: IngestReport) -> None:
        self.reports = list(reports)
        self.calls = 0

    def __call__(self, config: Config, vault: VaultConfig) -> IngestReport:
        self.calls += 1
        if not self.reports:
            return IngestReport(vault=vault.name)
        return self.reports.pop(0) if len(self.reports) > 1 else self.reports[0]


class FakeClock:
    """A clock that only moves when a test says so, and a sleep that moves it."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, job: str, reason: str, log_path: str) -> bool:
        self.calls.append((job, reason, log_path))
        return True


def refused_report(vault: str = "personal") -> IngestReport:
    """What the pass returns when the cleanliness guard stopped it."""
    reason = "vault 'personal' has uncommitted changes: transcripts/2026/08/stray.md"
    return IngestReport(vault=vault, dirty=reason, failure=reason)


@pytest.fixture
def paths(tmp_path: Path) -> StatePaths:
    return StatePaths(root=tmp_path / "state" / "personal")


@pytest.fixture
def scratch_config(tmp_path: Path) -> Config:
    """A loadable config over one real git vault, so the CLI paths are exercised for real."""
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path,
        {
            "vaults": {"personal": {"root": str(root)}},
            "index": {"path": str(tmp_path / "index" / "{vault}.db")},
        },
    )
    return load_config(path)


def fast(config: Config, **changes: float | int) -> Config:
    """The same config with debounce settings a test can reach in one cycle."""
    return replace(config, debounce=replace(config.debounce, **changes))


class TestWhereStateLives:
    """Markers live outside the vault, under a directory derived from the environment."""

    def test_the_state_directory_follows_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A scheduler and a hook see different homes; the marker path has to follow."""
        monkeypatch.delenv("MEMVAULT_STATE_DIR", raising=False)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        monkeypatch.setenv("HOME", "/tmp/some-home")

        assert state_home() == Path("/tmp/some-home/.local/state/memvault")

    def test_xdg_state_home_wins_over_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MEMVAULT_STATE_DIR", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", "/tmp/xdg")

        assert state_home() == Path("/tmp/xdg/memvault")

    def test_each_vault_gets_its_own_markers(self, tmp_path: Path) -> None:
        """Two vaults ingest independently, so one's quiet must not answer for the other."""
        personal = StatePaths.for_vault("personal", home=tmp_path)
        work = StatePaths.for_vault("work", home=tmp_path)

        assert personal.queue != work.queue

    def test_a_session_id_cannot_escape_the_sessions_directory(self, paths: StatePaths) -> None:
        """The id comes from somebody else's hook payload, so it names a file, never a path."""
        marker = paths.session("../../etc/passwd")

        assert marker.parent == paths.sessions


class TestTheQueueMarker:
    """A file says an ingest is owed. Losing the process must not lose the request."""

    def test_enqueue_creates_the_marker(self, paths: StatePaths) -> None:
        enqueue(paths)

        assert paths.queue.exists()
        assert queued_at(paths) is not None

    def test_enqueueing_again_pushes_the_clock_forward(self, paths: StatePaths) -> None:
        """This is the debounce: a burst of sessions costs one pass, not one pass each."""
        enqueue(paths, now=1000.0)
        first = queued_at(paths)

        enqueue(paths, now=2000.0)

        assert first == 1000.0
        assert queued_at(paths) == 2000.0

    def test_no_marker_reads_as_nothing_owed(self, paths: StatePaths) -> None:
        assert queued_at(paths) is None

    def test_clearing_removes_it(self, paths: StatePaths) -> None:
        enqueue(paths)

        assert clear_queue(paths) is True
        assert not paths.queue.exists()

    def test_a_marker_refreshed_during_the_pass_survives_it(self, paths: StatePaths) -> None:
        """A session that ended mid-pass has asked for a pass that has not happened yet."""
        enqueue(paths, now=1000.0)
        observed = queued_at(paths)
        enqueue(paths, now=2000.0)

        cleared = clear_queue(paths, unchanged_since=observed)

        assert cleared is False
        assert paths.queue.exists()


class TestActiveSessions:
    """SessionEnd alone cannot see the sessions still running, so SessionStart says so."""

    def test_a_started_session_is_live(self, paths: StatePaths) -> None:
        mark_session_start(paths, "abc")

        assert live_sessions(paths, now=time.time(), settings=SETTINGS) == 1

    def test_ending_it_clears_the_marker(self, paths: StatePaths) -> None:
        mark_session_start(paths, "abc")
        clear_session(paths, "abc")

        assert live_sessions(paths, now=time.time(), settings=SETTINGS) == 0

    def test_ending_a_session_that_never_started_is_not_an_error(self, paths: StatePaths) -> None:
        paths.ensure()

        assert clear_session(paths, "never-started") is False

    def test_a_stale_marker_is_disbelieved_and_removed(self, paths: StatePaths) -> None:
        """One crashed session must cost one stale window, not ingestion forever."""
        marker = mark_session_start(paths, "crashed")
        ancient = time.time() - SETTINGS.session_max_age_seconds - 60
        os.utime(marker, (ancient, ancient))

        assert live_sessions(paths, now=time.time(), settings=SETTINGS) == 0
        assert not marker.exists()

    def test_no_sessions_directory_reads_as_nobody_working(self, paths: StatePaths) -> None:
        assert live_sessions(paths, now=time.time(), settings=SETTINGS) == 0


class TestTheDecision:
    """`decide` is the only definition of quiet, and it is pure — no clock, no disk."""

    def test_nothing_queued_ends_the_runner(self) -> None:
        decision = decide(Quiescence(queued_at=None, live_sessions=0), now=100.0, settings=SETTINGS)

        assert decision.verdict is Verdict.IDLE

    def test_inside_the_quiet_window_it_waits(self) -> None:
        state = Quiescence(queued_at=1000.0, live_sessions=0)

        decision = decide(state, now=1060.0, settings=SETTINGS)

        assert decision.verdict is Verdict.WAIT
        assert decision.wait_seconds == pytest.approx(SETTINGS.quiet_seconds - 60.0)

    def test_a_refreshed_marker_restarts_the_clock(self) -> None:
        """The same instant is a run before the refresh and a wait after it."""
        late = 1000.0 + SETTINGS.quiet_seconds + 1

        before = decide(Quiescence(queued_at=1000.0, live_sessions=0), now=late, settings=SETTINGS)
        after = decide(
            Quiescence(queued_at=late - 10, live_sessions=0), now=late, settings=SETTINGS
        )

        assert before.verdict is Verdict.RUN
        assert after.verdict is Verdict.WAIT

    def test_a_live_session_blocks_a_run_the_window_would_allow(self) -> None:
        """A session started after the last enqueue is producing the material this would file."""
        state = Quiescence(queued_at=1000.0, live_sessions=1)

        decision = decide(state, now=1000.0 + SETTINGS.quiet_seconds + 3600, settings=SETTINGS)

        assert decision.verdict is Verdict.WAIT
        assert "still active" in decision.reason

    def test_quiet_with_nobody_working_runs(self) -> None:
        state = Quiescence(queued_at=1000.0, live_sessions=0)

        decision = decide(state, now=1000.0 + SETTINGS.quiet_seconds, settings=SETTINGS)

        assert decision.verdict is Verdict.RUN

    def test_a_pending_backoff_holds_the_run_back(self) -> None:
        state = Quiescence(queued_at=1000.0, live_sessions=0, not_before=9000.0)

        decision = decide(state, now=1000.0 + SETTINGS.quiet_seconds, settings=SETTINGS)

        assert decision.verdict is Verdict.WAIT
        assert "backing off" in decision.reason


class TestBackoff:
    """A refusal is an ordinary event under idle-triggering, so it escalates slowly and once."""

    def test_the_first_refusal_waits_the_base_interval(self) -> None:
        state = next_retry(RetryState(), now=0.0, settings=SETTINGS)

        assert state.refusals == 1
        assert state.not_before == pytest.approx(SETTINGS.retry_minutes * 60)

    def test_each_refusal_doubles_the_wait(self) -> None:
        first = next_retry(RetryState(), now=0.0, settings=SETTINGS)
        second = next_retry(first, now=0.0, settings=SETTINGS)

        assert second.not_before == pytest.approx(first.not_before * 2)

    def test_the_wait_is_capped(self) -> None:
        """A vault left half-committed a week ago clears when a human acts, not sooner."""
        state = RetryState()
        for _ in range(20):
            state = next_retry(state, now=0.0, settings=SETTINGS)

        assert state.not_before == pytest.approx(SETTINGS.max_retry_minutes * 60)

    def test_the_notification_fires_once_per_streak(self) -> None:
        state = RetryState()
        fired = 0
        for _ in range(6):
            after = next_retry(state, now=0.0, settings=SETTINGS)
            fired += should_notify(state, after)
            state = after

        assert fired == 1

    def test_a_streak_survives_the_runner_that_recorded_it(self, paths: StatePaths) -> None:
        """The lid closes mid-backoff; the next runner must not restart the count at one."""
        write_retry(paths, RetryState(refusals=2, not_before=99.0, notified=False))

        assert read_retry(paths) == RetryState(refusals=2, not_before=99.0, notified=False)

    def test_a_cleared_streak_leaves_no_file(self, paths: StatePaths) -> None:
        write_retry(paths, RetryState(refusals=2, not_before=99.0))
        write_retry(paths, RetryState())

        assert not paths.retry.exists()

    def test_unreadable_retry_state_reads_as_no_streak(self, paths: StatePaths) -> None:
        """A corrupted state file must not stop ingestion; it only forgets a count."""
        paths.ensure()
        paths.retry.write_text("{not json", encoding="utf-8")

        assert read_retry(paths) == RetryState()


class TestOneRunnerAtATime:
    """Two hooks firing in the same second must not produce two runners."""

    def test_a_free_lock_is_taken(self, paths: StatePaths) -> None:
        with RunnerLock(paths) as lock:
            assert lock.held is True
            assert paths.lock.read_text(encoding="utf-8").strip() == str(os.getpid())

    def test_a_lock_another_process_holds_is_not_taken(self, paths: StatePaths) -> None:
        paths.ensure()
        paths.lock.write_text(f"{ANOTHER_LIVE_PID}\n", encoding="utf-8")

        with RunnerLock(paths) as lock:
            assert lock.held is False

    def test_a_lock_registered_under_our_own_pid_is_adopted(self, paths: StatePaths) -> None:
        """`ensure_runner` registers the child before it starts, so the child finds its own."""
        paths.ensure()
        paths.lock.write_text(f"{os.getpid()}\n", encoding="utf-8")

        with RunnerLock(paths) as lock:
            assert lock.held is True

    def test_leaving_releases_it(self, paths: StatePaths) -> None:
        with RunnerLock(paths):
            pass

        assert not paths.lock.exists()

    def test_a_lock_held_by_a_dead_process_is_abandoned(self, paths: StatePaths) -> None:
        """A killed runner must not lock the vault out of ingestion until somebody notices."""
        paths.ensure()
        paths.lock.write_text("999999999\n", encoding="utf-8")

        assert runner_alive(paths) is False
        assert not paths.lock.exists()

    def test_a_lock_held_by_a_live_process_is_respected(self, paths: StatePaths) -> None:
        paths.ensure()
        paths.lock.write_text(f"{ANOTHER_LIVE_PID}\n", encoding="utf-8")

        assert runner_alive(paths) is True

    def test_enqueueing_twice_spawns_one_runner(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        vault = scratch_config.vault()
        spawner = FakeSpawner()

        enqueue(paths)
        ensure_runner(scratch_config, vault, paths, spawn=spawner)
        enqueue(paths)
        ensure_runner(scratch_config, vault, paths, spawn=spawner)

        assert len(spawner.calls) == 1

    def test_nothing_owed_spawns_nothing(self, scratch_config: Config, paths: StatePaths) -> None:
        """The runner exists to service a request; without one it would only burn a process."""
        spawner = FakeSpawner()

        assert ensure_runner(scratch_config, scratch_config.vault(), paths, spawn=spawner) is None
        assert spawner.calls == []

    def test_an_owed_pass_with_no_live_runner_is_respawned(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        """The lid-close case: the marker outlived the process that meant to service it."""
        enqueue(paths)
        spawner = FakeSpawner()

        pid = ensure_runner(scratch_config, scratch_config.vault(), paths, spawn=spawner)

        assert pid is not None
        assert len(spawner.calls) == 1


class TestTheRunnerLoop:
    """What the waiting process does, driven by an injected clock and a fake pass."""

    def run(
        self,
        config: Config,
        paths: StatePaths,
        *,
        ingest: FakeIngest,
        clock: FakeClock,
        notifier: FakeNotifier | None = None,
        cycles: int = 6,
    ) -> int:
        return run_runner(
            config,
            config.vault(),
            paths,
            ingest=ingest,
            notifier=notifier or FakeNotifier(),
            sleep=clock.sleep,
            clock=clock,
            max_cycles=cycles,
        )

    def test_it_exits_when_nothing_is_owed(self, scratch_config: Config, paths: StatePaths) -> None:
        ingest = FakeIngest()

        code = self.run(scratch_config, paths, ingest=ingest, clock=FakeClock())

        assert code == 0
        assert ingest.calls == 0

    def test_quiescence_reached_ingests_exactly_once(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        clock = FakeClock()
        enqueue(paths, now=clock.now)
        ingest = FakeIngest()

        self.run(fast(scratch_config, quiet_minutes=1), paths, ingest=ingest, clock=clock)

        assert ingest.calls == 1
        assert not paths.queue.exists()

    def test_it_waits_out_the_window_before_ingesting(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        clock = FakeClock()
        enqueue(paths, now=clock.now)
        ingest = FakeIngest()

        self.run(
            fast(scratch_config, quiet_minutes=2, poll_seconds=30),
            paths,
            ingest=ingest,
            clock=clock,
        )

        assert clock.slept != []
        assert ingest.calls == 1

    def test_a_live_session_keeps_it_waiting(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        clock = FakeClock()
        enqueue(paths, now=clock.now)
        mark_session_start(paths, "still-open")
        ingest = FakeIngest()

        self.run(fast(scratch_config, quiet_minutes=0), paths, ingest=ingest, clock=clock, cycles=4)

        assert ingest.calls == 0
        assert paths.queue.exists()

    def test_a_second_runner_exits_immediately(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        paths.ensure()
        paths.lock.write_text(f"{ANOTHER_LIVE_PID}\n", encoding="utf-8")
        enqueue(paths)
        ingest = FakeIngest()

        code = self.run(
            fast(scratch_config, quiet_minutes=0), paths, ingest=ingest, clock=FakeClock()
        )

        assert code == 0
        assert ingest.calls == 0

    def test_a_refusal_keeps_the_marker_and_backs_off(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        """R7's retry-later: nothing is broken, somebody has uncommitted work open."""
        clock = FakeClock()
        enqueue(paths, now=clock.now)
        ingest = FakeIngest(refused_report())

        self.run(fast(scratch_config, quiet_minutes=0), paths, ingest=ingest, clock=clock, cycles=1)

        assert paths.queue.exists()
        assert read_retry(paths).refusals == 1

    def test_a_refusal_is_silent_the_first_time(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        clock = FakeClock()
        enqueue(paths, now=clock.now)
        notifier = FakeNotifier()

        self.run(
            fast(scratch_config, quiet_minutes=0),
            paths,
            ingest=FakeIngest(refused_report()),
            clock=clock,
            notifier=notifier,
            cycles=1,
        )

        assert notifier.calls == []

    def test_repeated_refusals_notify_once(self, scratch_config: Config, paths: StatePaths) -> None:
        """Three in a row is no longer "somebody is editing"; it is something stuck."""
        clock = FakeClock()
        enqueue(paths, now=clock.now)
        notifier = FakeNotifier()
        config = fast(scratch_config, quiet_minutes=0, retry_minutes=0.01, max_retry_minutes=0.02)

        self.run(
            config,
            paths,
            ingest=FakeIngest(refused_report()),
            clock=clock,
            notifier=notifier,
            cycles=12,
        )

        assert len(notifier.calls) == 1
        assert "refused" in notifier.calls[0][1]

    def test_after_notifying_it_leaves_the_request_and_stops(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        """A streak clears when a human commits, which may be days. The marker waits; this does
        not — the next session start spawns a runner the moment there is a reason to.
        """
        clock = FakeClock()
        enqueue(paths, now=clock.now)
        ingest = FakeIngest(refused_report())
        config = fast(scratch_config, quiet_minutes=0, retry_minutes=0.01, max_retry_minutes=0.02)

        self.run(config, paths, ingest=ingest, clock=clock, cycles=50)

        assert ingest.calls == SETTINGS.notify_after_refusals
        assert paths.queue.exists()

    def test_a_marker_it_cannot_clear_stops_it_rather_than_looping(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        """Otherwise a serviced pass whose marker survives would be ingested again forever."""
        clock = FakeClock()
        paths.ensure()
        paths.queue.mkdir()  # a path that stats like a marker and refuses to be unlinked
        (paths.queue / "in-the-way").touch()
        os.utime(paths.queue, (clock.now, clock.now))
        ingest = FakeIngest()

        self.run(
            fast(scratch_config, quiet_minutes=0),
            paths,
            ingest=ingest,
            clock=clock,
            cycles=20,
        )

        assert ingest.calls == 1

    def test_a_success_after_refusals_clears_the_streak(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        clock = FakeClock()
        enqueue(paths, now=clock.now)
        write_retry(paths, RetryState(refusals=2, not_before=0.0, notified=True))

        self.run(
            fast(scratch_config, quiet_minutes=0),
            paths,
            ingest=FakeIngest(IngestReport(vault="personal")),
            clock=clock,
            cycles=2,
        )

        assert read_retry(paths) == RetryState()
        assert not paths.retry.exists()

    def test_a_genuine_failure_notifies_immediately(
        self, scratch_config: Config, paths: StatePaths
    ) -> None:
        """Only a refusal is retry-later. A broken commit is still a broken commit."""
        clock = FakeClock()
        enqueue(paths, now=clock.now)
        notifier = FakeNotifier()
        broken = IngestReport(vault="personal", failure="git commit failed")

        self.run(
            fast(scratch_config, quiet_minutes=0),
            paths,
            ingest=FakeIngest(broken),
            clock=clock,
            notifier=notifier,
            cycles=2,
        )

        assert len(notifier.calls) == 1
        assert broken.status is IngestStatus.FAILED


class TestTheSpawnedEnvironment:
    """Hook and scheduler environments are the third environment; nothing is inherited blind."""

    def test_the_runner_is_started_by_absolute_interpreter_path(
        self, scratch_config: Config
    ) -> None:
        """No PATH lookup at all — the failure mode this replaces was a PATH with nothing on it."""
        argv = runner_argv(scratch_config, scratch_config.vault())

        assert argv[0] == sys.executable
        assert argv[1:3] == ["-m", "memvault.cli"]

    def test_the_runner_is_told_which_config_and_vault(self, scratch_config: Config) -> None:
        """Discovery under a scheduler's HOME has been wrong before; the parent already knows."""
        argv = runner_argv(scratch_config, scratch_config.vault())

        assert str(scratch_config.source) in argv
        assert "--debounce-runner" in argv

    def test_the_config_path_is_pinned_into_the_environment(self, scratch_config: Config) -> None:
        env = runner_env(scratch_config, environ={})

        assert env["MEMVAULT_CONFIG"] == str(scratch_config.source)

    def test_the_standard_locations_are_added_to_an_impoverished_path(
        self, scratch_config: Config
    ) -> None:
        """The classifier is a CLI looked up by name, and a hook's PATH may hold nothing."""
        env = runner_env(scratch_config, environ={"PATH": "/usr/bin"})

        entries = env["PATH"].split(os.pathsep)
        assert "/usr/bin" in entries
        for addition in PATH_ADDITIONS:
            assert os.path.expanduser(addition) in entries

    def test_an_existing_path_entry_is_not_duplicated(self, scratch_config: Config) -> None:
        env = runner_env(scratch_config, environ={"PATH": "/usr/bin:/bin"})

        entries = env["PATH"].split(os.pathsep)
        assert entries.count("/usr/bin") == 1


class TestTheBanner:
    """Which code touched the vault is a question that has cost a debugging session."""

    def test_it_names_the_package_and_a_version(self) -> None:
        banner = runtime_banner()

        assert banner.startswith("memvault ")
        assert "(" in banner

    def test_it_says_where_the_code_came_from(self) -> None:
        """A checkout and an installed copy are different engines; the log must distinguish."""
        banner = runtime_banner()

        assert "checkout" in banner or "installed at" in banner


@pytest.fixture
def cli_config(tmp_path: Path) -> Path:
    """A config file on disk, for tests that go through `main` rather than the module."""
    root = make_vault(tmp_path, "personal")
    return write_config(
        tmp_path,
        {
            "vaults": {"personal": {"root": str(root)}},
            "index": {"path": str(tmp_path / "index" / "{vault}.db")},
        },
    )


@pytest.fixture
def cli_state() -> StatePaths:
    """The marker directory the CLI itself resolves — under the suite's redirected state dir."""
    return StatePaths.for_vault("personal")


@pytest.fixture
def recording_spawner(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture what the CLI would have spawned, instead of detaching a real process."""
    spawned: list[list[str]] = []

    def fake(argv: list[str], env: dict[str, str], log: Path) -> int:
        spawned.append(argv)
        return os.getpid()

    monkeypatch.setattr("memvault.debounce._spawn_detached", fake)
    return spawned


class TestTheCommandLine:
    """The three hook-facing modes, as `ingest` flags."""

    def test_enqueue_writes_the_marker_and_does_not_ingest(
        self,
        cli_config: Path,
        cli_state: StatePaths,
        recording_spawner: list[list[str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code = main(["--config", str(cli_config), "ingest", "--enqueue"])

        assert code == 0
        assert cli_state.queue.exists()
        assert "queued" in capsys.readouterr().out
        assert len(recording_spawner) == 1

    def test_enqueue_clears_the_session_that_just_ended(
        self, cli_config: Path, cli_state: StatePaths, recording_spawner: list[list[str]]
    ) -> None:
        mark_session_start(cli_state, "sess-1")

        main(["--config", str(cli_config), "ingest", "--enqueue", "--session", "sess-1"])

        assert live_sessions(cli_state, now=time.time(), settings=SETTINGS) == 0

    def test_session_start_marks_the_session_active(
        self, cli_config: Path, cli_state: StatePaths
    ) -> None:
        code = main(["--config", str(cli_config), "ingest", "--session-start", "sess-2"])

        assert code == 0
        assert live_sessions(cli_state, now=time.time(), settings=SETTINGS) == 1

    def test_session_start_respawns_a_runner_for_an_owed_pass(
        self,
        cli_config: Path,
        cli_state: StatePaths,
        recording_spawner: list[list[str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The lid closed during the wait; this is the earliest anything notices."""
        enqueue(cli_state)

        main(["--config", str(cli_config), "ingest", "--session-start", "sess-3"])

        assert len(recording_spawner) == 1
        assert "re-spawned" in capsys.readouterr().out

    def test_a_session_start_with_nothing_owed_spawns_nothing(
        self, cli_config: Path, recording_spawner: list[list[str]]
    ) -> None:
        main(["--config", str(cli_config), "ingest", "--session-start", "sess-4"])

        assert recording_spawner == []

    def test_the_nightly_entry_services_an_owed_pass(
        self, cli_config: Path, cli_state: StatePaths
    ) -> None:
        """Same entry point as always: a plain run answers what the queue was asking for."""
        enqueue(cli_state)

        code = main(["--config", str(cli_config), "ingest"])

        assert code == 0
        assert not cli_state.queue.exists()

    def test_a_refused_nightly_pass_leaves_the_request_standing(
        self,
        cli_config: Path,
        cli_state: StatePaths,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clearing the marker on a refusal would silently cancel the retry."""
        monkeypatch.setattr("memvault.notify.notify", lambda *args, **kwargs: False)
        enqueue(cli_state)
        root = tmp_path / "personal"
        (root / "inbox" / "note.md").write_text("a thought worth keeping\n", encoding="utf-8")
        transcripts = root / "transcripts" / "2026" / "08"
        transcripts.mkdir(parents=True)
        (transcripts / "stray.md").write_text("half a note\n", encoding="utf-8")

        code = main(["--config", str(cli_config), "ingest"])

        assert code != 0
        assert cli_state.queue.exists()


class TestTheHonestEnvironment:
    """The launchd learnings, as a test: run it with almost no environment and see.

    Both prior scheduler failures on this machine were environment divergence found only in
    production — a wrong `CLAUDE_CONFIG_DIR`, a `PATH` without brew on it. A hook is a third
    environment, so this spawns a real interpreter with `env -i`-shaped surroundings: a HOME
    that holds the config, a PATH of the four system directories, and nothing else at all.
    """

    def test_a_scrubbed_environment_still_finds_its_config_and_ingests(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        root = tmp_path / "vault"
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
        (root / "inbox").mkdir()

        config_path = home / ".config" / "memvault" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "vaults": {"personal": {"root": str(root)}},
                    "index": {"path": str(tmp_path / "index" / "{vault}.db")},
                    "debounce": {"quiet_minutes": 0, "poll_seconds": 1},
                }
            ),
            encoding="utf-8",
        )

        # An ingest already owed, queued where a runner under this HOME will look for it.
        state = home / ".local" / "state" / "memvault" / "personal"
        state.mkdir(parents=True)
        (state / "ingest.queued").touch()

        log = tmp_path / "ingest.log"
        bare = {
            "HOME": str(home),
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "memvault.cli",
                "ingest",
                "--debounce-runner",
                "--log-path",
                str(log),
            ],
            env=bare,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, result.stderr
        # It found the XDG config under the scrubbed HOME, said which engine was running, ran
        # the pass, and answered the request — all without an environment worth the name.
        written = log.read_text(encoding="utf-8")
        assert "memvault " in written
        assert "0 filed" in written
        assert not (state / "ingest.queued").exists()

    def test_a_scrubbed_enqueue_writes_its_marker_under_the_scrubbed_home(
        self, tmp_path: Path
    ) -> None:
        """The state directory has to follow HOME, or a hook queues into somebody else's vault."""
        home = tmp_path / "home"
        root = tmp_path / "vault"
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "inbox").mkdir()
        config_path = home / ".config" / "memvault" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "vaults": {"personal": {"root": str(root)}},
                    "index": {"path": str(tmp_path / "index" / "{vault}.db")},
                    # A window nothing will reach, so the spawned runner waits instead of
                    # ingesting while this test reads the marker out from under it.
                    "debounce": {"quiet_minutes": 600},
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "memvault.cli",
                "ingest",
                "--enqueue",
                "--log-path",
                str(tmp_path / "ingest.log"),
            ],
            env={"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            capture_output=True,
            text=True,
            timeout=60,
        )

        marker = home / ".local" / "state" / "memvault" / "personal" / "ingest.queued"
        try:
            assert result.returncode == 0, result.stderr
            assert marker.exists()
        finally:
            _stop_runner(marker.parent / "runner.lock")


def _stop_runner(lock: Path) -> None:
    """Kill the real detached runner a test started, so it does not wait out its window here."""
    with suppress(OSError, ValueError):
        pid = int(lock.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)


def test_the_config_section_parses(tmp_path: Path) -> None:
    """The window is config because 15 minutes is a working guess, not a measurement."""
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path,
        {
            "vaults": {"personal": {"root": str(root)}},
            "debounce": {"quiet_minutes": 5, "notify_after_refusals": 2},
        },
    )

    config = load_config(path)

    assert config.debounce.quiet_minutes == 5
    assert config.debounce.quiet_seconds == 300
    assert config.debounce.notify_after_refusals == 2


def test_an_unknown_debounce_key_is_rejected_by_name(tmp_path: Path) -> None:
    root = make_vault(tmp_path, "personal")
    path = write_config(
        tmp_path,
        {"vaults": {"personal": {"root": str(root)}}, "debounce": {"quiet_mins": 5}},
    )

    with pytest.raises(Exception) as exc:
        load_config(path)

    assert "quiet_mins" in str(exc.value)


def test_the_retry_state_file_is_plain_json(paths: StatePaths) -> None:
    """Readable by a human debugging a stuck vault, without this module in hand."""
    write_retry(paths, RetryState(refusals=3, not_before=1234.0, notified=True))

    assert json.loads(paths.retry.read_text(encoding="utf-8"))["refusals"] == 3
