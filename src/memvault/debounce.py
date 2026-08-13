"""Idle-triggered ingestion: markers on disk, a disposable runner, and one decision function.

Ingest used to run at 23:00. That is a guess about when a person stops working, and it was wrong
in both directions — a conversation captured at nine in the morning waited fourteen hours to
become memory, and a machine asleep at 23:00 lost the pass entirely, because launchd user agents
do not catch up. R7 replaces the clock with quiescence: sessions announce themselves, and the
pass runs a few minutes after the last one goes quiet.

**The queue marker is the source of truth, and the runner is disposable.** A file says an ingest
is owed; a process merely intends to perform one. That asymmetry is the whole design. Closing the
lid kills the runner mid-wait, and if the runner held the intention the ingest would evaporate
with it — the exact failure the nightly job was already suffering. Instead the marker survives,
and any later entry point (the next SessionStart, the nightly safety net) sees an owed pass with
no live runner and services it.

**Quiescence is a pair of markers, not one.** SessionEnd alone cannot tell whether other sessions
are still running, and one terminal closing while three stay open is an ordinary afternoon. So
SessionStart writes an active-session marker and SessionEnd removes it, and the runner waits for
the queue marker to go un-refreshed *and* for no live session to remain. A marker whose session
crashed would otherwise suspend ingestion forever, so markers past
`DebounceConfig.session_max_age_minutes` are disbelieved.

**A dirty-guard refusal is retry-later, not failure.** The pass now fires whenever work goes
quiet, which is precisely when somebody is most likely to have uncommitted vault edits open, so
refusals stop being exceptional. Each one reschedules with exponential backoff and stays silent;
only a run of them (`notify_after_refusals`) means something is genuinely stuck, and only that
notifies. Retry state lives in a file for the same reason the queue marker does: the runner it
belongs to may not survive the wait.

**The spawned runner builds its own environment.** Two scheduler failures on this machine were
environment divergence discovered live — launchd's minimal env had neither the right
`CLAUDE_CONFIG_DIR` nor brew's `bin` on `PATH`. Hook environments are a third environment with
the same class of quirk, so nothing here inherits and hopes: the runner is spawned as
`<this interpreter> -m memvault.cli`, which needs no `PATH` lookup at all, with `MEMVAULT_CONFIG`
pinned to the config file the parent actually resolved and `PATH` widened to the standard
locations the classifier's CLI is likely to live in. Each pass logs the running version and where
it came from, because "which checkout is the scheduler running" has cost a debugging session
before.

Nothing here pushes. The debounced pass commits exactly like the nightly one did; pushing stays
the vault's own backup job (KTD7).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from memvault.config import Config, DebounceConfig, VaultConfig
from memvault.ingest import (
    DEFAULT_LOG_PATH,
    IngestReport,
    IngestStatus,
    Notifier,
    run_ingest,
)
from memvault.notify import notify_failure

logger = logging.getLogger(__name__)

#: Where marker files live when nothing says otherwise: `$XDG_STATE_HOME/memvault/<vault>`,
#: falling back to the XDG default. Deliberately outside the vault — a marker file inside it
#: would be uncommitted state under the very tree the cleanliness guard inspects, so the queue
#: marker would refuse the pass it exists to request.
STATE_ENV_VAR = "MEMVAULT_STATE_DIR"
XDG_STATE_HOME_VAR = "XDG_STATE_HOME"
DEFAULT_STATE_HOME = "~/.local/state"
STATE_RELATIVE = "memvault"

QUEUE_MARKER = "ingest.queued"
RUNNER_LOCK = "runner.lock"
SESSIONS_DIR = "sessions"
RETRY_STATE = "retry.json"

#: Directories a spawned runner is given on `PATH` on top of whatever it inherited. The engine
#: itself needs none of them — it is started by absolute interpreter path — but the classifier is
#: a CLI looked up by name, and a hook's `PATH` has been observed to hold nothing but `/usr/bin`.
PATH_ADDITIONS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
    "~/.local/bin",
)

#: A lock whose owning process is gone and which is older than this is treated as abandoned. The
#: pid check catches almost every case; this catches the one where the pid has been recycled by
#: an unrelated process, which would otherwise block ingestion until a reboot.
LOCK_MAX_AGE_SECONDS = 6 * 60 * 60


# --------------------------------------------------------------------------------------
# Where state lives
# --------------------------------------------------------------------------------------


def state_home() -> Path:
    """The directory holding every vault's marker files.

    `$MEMVAULT_STATE_DIR` wins so a test — and a second install on one machine — can put state
    somewhere of its own; otherwise the XDG state location, which is derived from `$HOME` and so
    moves correctly under a scheduler or a hook that redirects it.
    """
    explicit = os.environ.get(STATE_ENV_VAR)
    if explicit:
        return Path(os.path.expanduser(explicit))
    base = os.environ.get(XDG_STATE_HOME_VAR) or DEFAULT_STATE_HOME
    return Path(os.path.expanduser(base)) / STATE_RELATIVE


@dataclass(frozen=True)
class StatePaths:
    """Every file the debounce machinery reads or writes, for one vault.

    A record rather than a set of path-building calls, so the layout is stated once and a test
    can point the whole thing at a temporary directory in one move.
    """

    root: Path

    @classmethod
    def for_vault(cls, vault: str, *, home: Path | None = None) -> StatePaths:
        return cls(root=(home or state_home()) / vault)

    @property
    def queue(self) -> Path:
        return self.root / QUEUE_MARKER

    @property
    def lock(self) -> Path:
        return self.root / RUNNER_LOCK

    @property
    def sessions(self) -> Path:
        return self.root / SESSIONS_DIR

    @property
    def retry(self) -> Path:
        return self.root / RETRY_STATE

    def session(self, session_id: str) -> Path:
        """The marker for one session. The id is folded to a safe filename, never a path.

        A session id arrives from a harness's hook payload, which is somebody else's string:
        allowing it to contain a separator would let a misbehaving client aim a marker write
        anywhere on the filesystem.
        """
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in session_id.strip())
        return self.sessions / (safe or "unnamed")

    def ensure(self) -> None:
        self.sessions.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# The decision, as a pure function
# --------------------------------------------------------------------------------------


class Verdict(StrEnum):
    RUN = "run"
    WAIT = "wait"
    IDLE = "idle"


@dataclass(frozen=True)
class Quiescence:
    """What the markers on disk currently say, reduced to the three facts the decision needs."""

    #: The queue marker's mtime, or None when no ingest is owed.
    queued_at: float | None
    #: How many active-session markers are young enough to be believed.
    live_sessions: int
    #: The earliest a retry may run, after a dirty-guard refusal. Zero when nothing is pending.
    not_before: float = 0.0


@dataclass(frozen=True)
class Decision:
    """What the runner does next, and the sentence explaining it to a log reader."""

    verdict: Verdict
    wait_seconds: float = 0.0
    reason: str = ""


def decide(state: Quiescence, *, now: float, settings: DebounceConfig) -> Decision:
    """Whether to ingest now, wait, or exit — from marker facts alone.

    Pure, and separated from every file read on purpose: this is the only place the definition
    of "quiet" lives, and it is the part worth pinning with tests that never touch a disk or a
    clock. Note the order — a live session outranks an expired quiet window, because a session
    that started after the last enqueue is producing exactly the material the pass would file
    halfway through writing it.
    """
    if state.queued_at is None:
        return Decision(Verdict.IDLE, reason="nothing queued")

    if state.live_sessions:
        return Decision(
            Verdict.WAIT,
            wait_seconds=settings.poll_seconds,
            reason=f"{state.live_sessions} session(s) still active",
        )

    quiet_until = state.queued_at + settings.quiet_seconds
    if now < quiet_until:
        return Decision(
            Verdict.WAIT,
            wait_seconds=quiet_until - now,
            reason=f"queued {now - state.queued_at:.0f}s ago; "
            f"waiting {settings.quiet_minutes:g} minute(s) of quiet",
        )

    if now < state.not_before:
        return Decision(
            Verdict.WAIT,
            wait_seconds=state.not_before - now,
            reason=f"backing off after a refusal; retrying in {state.not_before - now:.0f}s",
        )

    return Decision(Verdict.RUN, reason="quiet window elapsed with no active session")


@dataclass(frozen=True)
class RetryState:
    """How many times in a row the guard refused, and when the next attempt may happen."""

    refusals: int = 0
    not_before: float = 0.0
    notified: bool = False

    @property
    def clear(self) -> bool:
        return self.refusals == 0 and not self.notified


def next_retry(state: RetryState, *, now: float, settings: DebounceConfig) -> RetryState:
    """The retry state after one more refusal: backoff doubles, the notification fires once.

    Exponential rather than fixed because the two causes have different shapes. A session that
    happens to hold an edit open clears in minutes, and a short first retry catches it; a vault
    somebody left half-committed a week ago clears when a human acts, and hammering it every five
    minutes until then would be a log full of noise saying nothing new.
    """
    refusals = state.refusals + 1
    delay = min(settings.retry_minutes * (2 ** (refusals - 1)), settings.max_retry_minutes)
    return RetryState(
        refusals=refusals,
        not_before=now + delay * 60.0,
        notified=state.notified or refusals >= settings.notify_after_refusals,
    )


def should_notify(before: RetryState, after: RetryState) -> bool:
    """Whether this refusal is the one that earns a notification. Exactly once per streak."""
    return after.notified and not before.notified


# --------------------------------------------------------------------------------------
# Reading and writing the markers
# --------------------------------------------------------------------------------------


def enqueue(paths: StatePaths, *, now: float | None = None) -> Path:
    """Record that an ingest is owed, or refresh the request if one already is.

    Touch semantics are the debounce: every SessionEnd pushes the mtime forward, so the runner's
    quiet window restarts and a burst of sessions costs one pass rather than one pass each.
    """
    paths.ensure()
    paths.queue.touch()
    if now is not None:
        os.utime(paths.queue, (now, now))
    return paths.queue


def queued_at(paths: StatePaths) -> float | None:
    """When the owed ingest was last requested, or None when none is owed."""
    try:
        return paths.queue.stat().st_mtime
    except OSError:
        return None


def clear_queue(paths: StatePaths, *, unchanged_since: float | None = None) -> bool:
    """Mark the owed ingest as serviced. Returns whether the marker was removed.

    `unchanged_since` is the mtime observed before the pass started. A session that ended while
    the pass was running has already asked for another one, and removing the marker then would
    discard a request nobody will repeat — so a refreshed marker is left exactly where it is.
    """
    current = queued_at(paths)
    if current is None:
        return False
    if unchanged_since is not None and current > unchanged_since:
        logger.info("queue marker was refreshed during the pass; leaving it for the next round")
        return False
    try:
        paths.queue.unlink()
    except OSError as exc:
        logger.warning("could not clear the queue marker %s: %s", paths.queue, exc)
        return False
    return True


def mark_session_start(paths: StatePaths, session_id: str) -> Path:
    """Write the active-session marker whose absence means "nobody is working"."""
    paths.ensure()
    marker = paths.session(session_id)
    marker.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return marker


def clear_session(paths: StatePaths, session_id: str) -> bool:
    """Remove one session's marker. A marker that is already gone is not an error."""
    try:
        paths.session(session_id).unlink()
    except OSError:
        return False
    return True


def live_sessions(paths: StatePaths, *, now: float, settings: DebounceConfig) -> int:
    """How many sessions are believably still running.

    A marker older than `session_max_age_minutes` is counted as abandoned and deleted rather
    than merely ignored, so one crashed session costs one stale window instead of suspending
    ingestion for as long as the file survives.
    """
    if not paths.sessions.is_dir():
        return 0

    live = 0
    for marker in sorted(paths.sessions.iterdir()):
        try:
            age = now - marker.stat().st_mtime
        except OSError:
            continue
        if age > settings.session_max_age_seconds:
            logger.info(
                "session marker %s is %.0fs old — treating its session as crashed", marker.name, age
            )
            marker.unlink(missing_ok=True)
            continue
        live += 1
    return live


def read_retry(paths: StatePaths) -> RetryState:
    """The refusal streak, or a clean state when there is no file or it cannot be read."""
    try:
        payload = json.loads(paths.retry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return RetryState()
    if not isinstance(payload, dict):
        return RetryState()
    try:
        return RetryState(
            refusals=int(payload.get("refusals", 0)),
            not_before=float(payload.get("not_before", 0.0)),
            notified=bool(payload.get("notified", False)),
        )
    except (TypeError, ValueError):
        return RetryState()


def write_retry(paths: StatePaths, state: RetryState) -> None:
    """Persist the refusal streak, or delete the file once the streak is over."""
    if state.clear:
        paths.retry.unlink(missing_ok=True)
        return
    paths.ensure()
    paths.retry.write_text(
        json.dumps(
            {
                "refusals": state.refusals,
                "not_before": state.not_before,
                "notified": state.notified,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def read_quiescence(paths: StatePaths, *, now: float, settings: DebounceConfig) -> Quiescence:
    """Everything `decide` needs, read off the disk in one place."""
    return Quiescence(
        queued_at=queued_at(paths),
        live_sessions=live_sessions(paths, now=now, settings=settings),
        not_before=read_retry(paths).not_before,
    )


# --------------------------------------------------------------------------------------
# One runner at a time
# --------------------------------------------------------------------------------------


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Somebody else's process holds that pid. It is not this runner, but it is alive, and
        # stealing the lock from a live pid is the one thing worse than waiting.
        return True
    except OSError:
        return False
    return True


def runner_pid(paths: StatePaths) -> int | None:
    """The pid of the live runner, or None when no live one holds the lock.

    An abandoned lock — dead pid, or one old enough that the pid could have been recycled — is
    removed here, because the alternative is a vault that never ingests again after one crash
    and no message saying why.
    """
    try:
        raw = paths.lock.read_text(encoding="utf-8").strip()
        age = time.time() - paths.lock.stat().st_mtime
    except OSError:
        return None

    try:
        pid = int(raw)
    except ValueError:
        pid = -1

    if _process_alive(pid) and age <= LOCK_MAX_AGE_SECONDS:
        return pid

    logger.info("removing an abandoned runner lock (pid %s, %.0fs old)", raw or "?", age)
    paths.lock.unlink(missing_ok=True)
    return None


def runner_alive(paths: StatePaths) -> bool:
    return runner_pid(paths) is not None


class RunnerLock:
    """The one-runner guarantee, as a context manager.

    Acquisition is `O_CREAT | O_EXCL`, which is atomic on every filesystem this runs on, so two
    runners racing for the lock cannot both believe they won.

    One case is not a race and must not be treated as one: a runner spawned by `ensure_runner`
    finds a lock the parent wrote *on its behalf*, naming its own pid. It adopts that lock rather
    than exiting. The alternative — the parent registering the runner only after it has started —
    leaves the whole of a Python interpreter's startup as a window in which a second enqueue sees
    no runner and spawns another.
    """

    def __init__(self, paths: StatePaths) -> None:
        self.paths = paths
        self.held = False

    def __enter__(self) -> RunnerLock:
        self.paths.ensure()
        runner_pid(self.paths)  # clears an abandoned lock before we try for it
        try:
            handle = os.open(self.paths.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            self.held = self._registered_for_us()
            return self
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()}\n")
        self.held = True
        return self

    def _registered_for_us(self) -> bool:
        try:
            return self.paths.lock.read_text(encoding="utf-8").strip() == str(os.getpid())
        except OSError:
            return False

    def __exit__(self, *exc: object) -> None:
        if self.held:
            self.paths.lock.unlink(missing_ok=True)
            self.held = False


# --------------------------------------------------------------------------------------
# Spawning, with an environment of its own
# --------------------------------------------------------------------------------------

#: Injectable so no test spawns a process. Takes the argv and the environment; returns the pid.
Spawner = Callable[[list[str], dict[str, str], Path], int]


def runner_argv(
    config: Config, vault: VaultConfig, *, log_path: str = DEFAULT_LOG_PATH
) -> list[str]:
    """The command that runs a debounce runner for this vault.

    Deliberately `sys.executable -m memvault.cli` rather than the `memvault` console script: an
    absolute interpreter path resolves under any `PATH` at all, including the empty one a hook
    can hand us, and it is guaranteed to be the interpreter that already has this package
    importable. `--config` is passed explicitly for the same reason — discovery under a
    scheduler's `$HOME` has been wrong before, and the parent has already resolved the answer.
    """
    return [
        sys.executable,
        "-m",
        "memvault.cli",
        "--config",
        str(config.source),
        "--vault",
        vault.name,
        "ingest",
        "--debounce-runner",
        "--log-path",
        log_path,
    ]


def runner_env(config: Config, *, environ: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a spawned runner gets: the parent's, plus what a hook's is missing.

    Hook and scheduler environments are minimal in ways discovered only in production, so the
    two things a pass cannot run without are set rather than inherited — the config file the
    parent resolved, and a `PATH` wide enough to find the classifier's CLI.
    """
    env = dict(os.environ if environ is None else environ)
    env["MEMVAULT_CONFIG"] = str(config.source)

    entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    for addition in PATH_ADDITIONS:
        expanded = os.path.expanduser(addition)
        if expanded not in entries:
            entries.append(expanded)
    env["PATH"] = os.pathsep.join(entries)
    return env


def _spawn_detached(argv: list[str], env: dict[str, str], log: Path) -> int:
    """Start the runner in its own session so it outlives the hook that asked for it."""
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
    finally:
        stream.close()
    return process.pid


def ensure_runner(
    config: Config,
    vault: VaultConfig,
    paths: StatePaths,
    *,
    log_path: str = DEFAULT_LOG_PATH,
    spawn: Spawner | None = None,
) -> int | None:
    """Start a runner if one is owed and none is alive. Returns its pid, or None.

    Called from both legs of R7 — the SessionEnd enqueue and the SessionStart re-spawn — because
    the queue marker outliving its runner is normal, not exceptional. A lid closed during the
    wait leaves an owed pass and no process, and this is what notices.

    The child is registered in the lock file here, by the parent, rather than by the child when
    it eventually starts: two sessions ending a second apart would otherwise both look out on an
    empty lock and spawn. The child adopts the lock it finds under its own pid.
    """
    if queued_at(paths) is None:
        return None
    existing = runner_pid(paths)
    if existing is not None:
        logger.debug("a runner is already alive (pid %d); not spawning another", existing)
        return None

    log = Path(os.path.expanduser(log_path))
    spawner = spawn or _spawn_detached
    pid = spawner(runner_argv(config, vault, log_path=log_path), runner_env(config), log)
    paths.ensure()
    paths.lock.write_text(f"{pid}\n", encoding="utf-8")
    logger.info("spawned debounce runner (pid %d) for vault %r", pid, vault.name)
    return pid


# --------------------------------------------------------------------------------------
# What is actually running
# --------------------------------------------------------------------------------------


def _checkout_description(path: Path) -> str | None:
    """`<branch> @ <short sha>` when this code sits in a git checkout, else None."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    return f"{lines[0]} @ {lines[1]}"


def installed_version() -> str:
    """The distribution version, or a marker that it could not be determined."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("memvault")
    except PackageNotFoundError:
        return "unknown"
    except Exception:  # pragma: no cover - importlib failures are not worth failing a pass over
        return "unknown"


def runtime_banner() -> str:
    """One line naming the version and the checkout an unattended pass is running from.

    Written because the answer has been surprising before: the scheduled jobs ran the working
    checkout, so whichever feature branch happened to be checked out at 23:00 was the engine
    that touched the vault. A log that names the branch turns that from a debugging session into
    a glance.
    """
    module_root = Path(__file__).resolve().parent
    origin = _checkout_description(module_root)
    where = f"checkout {origin}" if origin else f"installed at {module_root}"
    return f"memvault {installed_version()} ({where})"


# --------------------------------------------------------------------------------------
# The runner loop
# --------------------------------------------------------------------------------------

#: Injectable so the loop is testable without a clock, a subprocess, or a notification. The
#: notifier seam is `ingest.Notifier`, reused rather than restated so both passes notify alike.
Ingester = Callable[[Config, VaultConfig], IngestReport]


def _default_ingest(config: Config, vault: VaultConfig) -> IngestReport:
    """Run the pass without its own notification.

    `run_ingest` notifies on any non-OK status, and under R7 the most common non-OK status is a
    dirty-guard refusal — which is explicitly not a failure. The runner takes the notification
    decision instead, so a refusal is silent until it has repeated.
    """
    return run_ingest(config, vault, notifier=None)


def run_runner(
    config: Config,
    vault: VaultConfig,
    paths: StatePaths,
    *,
    ingest: Ingester = _default_ingest,
    notifier: Notifier | None = notify_failure,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    log_path: str = DEFAULT_LOG_PATH,
    max_cycles: int | None = None,
) -> int:
    """Wait for quiescence, then ingest. Exits when nothing is owed.

    Disposable by design: it holds no state a crash could lose, and everything it decides on is
    re-read from disk each cycle, so being killed mid-wait costs a spawn and nothing else.

    `max_cycles` bounds the loop for tests. In production the exit condition is the queue marker
    disappearing, which is the same condition that makes a re-spawn unnecessary.
    """
    settings = config.debounce
    with RunnerLock(paths) as lock:
        if not lock.held:
            logger.info("another runner already holds %s; exiting", paths.lock)
            return 0

        logger.info("debounce runner for vault %r: %s", vault.name, runtime_banner())

        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            now = clock()
            decision = decide(
                read_quiescence(paths, now=now, settings=settings), now=now, settings=settings
            )

            if decision.verdict is Verdict.IDLE:
                logger.info("nothing owed — %s", decision.reason)
                return 0
            if decision.verdict is Verdict.WAIT:
                logger.debug("waiting %.0fs — %s", decision.wait_seconds, decision.reason)
                sleep(max(0.0, min(decision.wait_seconds, settings.poll_seconds)))
                continue

            if not _one_pass(
                config,
                vault,
                paths,
                settings=settings,
                ingest=ingest,
                notifier=notifier,
                clock=clock,
                log_path=log_path,
            ):
                return 0

        return 0


def _one_pass(
    config: Config,
    vault: VaultConfig,
    paths: StatePaths,
    *,
    settings: DebounceConfig,
    ingest: Ingester,
    notifier: Notifier | None,
    clock: Callable[[], float],
    log_path: str,
) -> bool:
    """Run one ingestion, fold its outcome into the markers, and say whether to keep waiting.

    Returning False never abandons the request — the queue marker is what carries it, and it is
    still there. It only ends *this* process, which is the disposable half.
    """
    observed = queued_at(paths)
    logger.info("vault %r: quiet — ingesting (%s)", vault.name, runtime_banner())
    report = ingest(config, vault)

    if report.refused:
        before = read_retry(paths)
        after = next_retry(before, now=clock(), settings=settings)
        write_retry(paths, after)
        logger.warning(
            "vault %r: guard refused (%d in a row) — retrying after %.0fs, the queue marker "
            "stands: %s",
            vault.name,
            after.refusals,
            max(0.0, after.not_before - clock()),
            report.dirty,
        )
        if notifier is not None and should_notify(before, after):
            notifier(
                "ingest",
                f"{after.refusals} ingest attempts refused: the vault has uncommitted changes",
                log_path,
            )
        if after.notified:
            # A streak this long clears when a human commits something, which may be days. A
            # process sitting here until then buys nothing the marker does not already hold, and
            # the next session start re-spawns a runner the moment there is a reason to.
            logger.info("vault %r: leaving the queued ingest for a later attempt", vault.name)
            return False
        return True

    write_retry(paths, RetryState())
    cleared = clear_queue(paths, unchanged_since=observed)
    logger.info("vault %r: %s", vault.name, report.summary_line())

    if notifier is not None and report.status is not IngestStatus.OK:
        notifier("ingest", report.summary_line(), log_path)

    if not cleared and queued_at(paths) == observed:
        # The pass ran and the marker is neither gone nor refreshed, so the next cycle would
        # decide to run it again, and the one after that. Stop rather than ingest in a loop.
        logger.error(
            "vault %r: the queue marker at %s survived a serviced pass; stopping this runner",
            vault.name,
            paths.queue,
        )
        return False
    return True


def configure_logging(log_path: str) -> Path:
    """Send this process's logs to a file, because a detached runner has no terminal.

    Returns the path so the caller can name it. Failing to open the log is not worth failing the
    pass over — the run still happens, it is merely unobserved.
    """
    path = Path(os.path.expanduser(log_path))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return path
