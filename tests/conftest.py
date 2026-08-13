"""Suite-wide guards.

Two rules with teeth, both about the same failure shape: a test that quietly reaches the
author's real machine still passes, still prints a plausible report, and only a reader of the
captured output would notice what it had just read or curated.

**No test reads the real claude-mem database.** It is a third-party file holding actual work and
personal history, it changes under the suite, and a test that read it would assert on someone's
week rather than on this code. The default path is redirected for every test, so reaching it
takes a deliberate `db_path=` — which the `claude_mem` tests supply, pointing at fixtures they
built themselves.

**No test discovers a real user-level config.** `find_config` searches the XDG and legacy home
locations before falling back to the current directory (R9), and on a machine that runs MemVault
for real both resolve to a config naming live vaults. Redirecting HOME points those two legs at
a home that does not exist, while leaving the discovery code itself under test.

**No test writes debounce markers into the real state directory.** The queue and session markers
say whether an ingestion pass is owed on this machine, so a test that touched them would either
suppress or provoke a real run against a real vault.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memvault import claude_mem
from memvault.config import CONFIG_ENV_VAR, XDG_CONFIG_HOME_VAR
from memvault.debounce import STATE_ENV_VAR

#: A path no filesystem holds. Not created and not cleaned up — the fixture needs somewhere that
#: is definitely absent, not somewhere temporary, and creating a directory per test would make
#: this guard cost more than the thing it guards.
ABSENT_DB_PATH = "/nonexistent/memvault-tests/claude-mem.db"

#: Likewise absent, and deliberately not `tmp_path`: a home that exists could be populated by an
#: earlier test and then discovered by a later one.
ABSENT_HOME = "/nonexistent/memvault-tests/home"


@pytest.fixture(autouse=True)
def never_the_real_claude_mem_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the claude-mem default at a path that does not exist, for every test."""
    monkeypatch.setattr(claude_mem, "DEFAULT_DB_PATH", ABSENT_DB_PATH)


@pytest.fixture(autouse=True)
def never_the_real_user_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every user-level config location somewhere absent, for every test.

    The environment is cleared rather than the lookup patched, so the tests that assert the
    discovery order exercise the shipped `find_config` and merely put files where it looks.
    """
    monkeypatch.setenv("HOME", ABSENT_HOME)
    monkeypatch.delenv(XDG_CONFIG_HOME_VAR, raising=False)
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def never_the_real_debounce_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Give every test its own state directory for queue and session markers.

    Unlike the two guards above this one points somewhere real and writable, because these
    markers are written rather than merely read — and a test that could not write one would pass
    for the wrong reason.
    """
    state = tmp_path / "state"
    monkeypatch.setenv(STATE_ENV_VAR, str(state))
    return state
