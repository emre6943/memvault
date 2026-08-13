"""Notification helper.

The governing property: a notification failing must never propagate. These tests mostly prove
that nothing escapes, because the one thing worse than a missed notification is an ingestion
pass that crashed while trying to send one.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from memvault import notify as notify_module
from memvault.notify import notify, notify_failure


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(notify_module.shutil, "which", lambda _: "/usr/bin/osascript")
    monkeypatch.setattr(notify_module.subprocess, "run", fake_run)
    return calls


def test_notify_invokes_osascript_with_title_and_message(captured_calls: list[list[str]]) -> None:
    delivered = notify("MemVault", "three items filed")

    assert delivered is True
    script = captured_calls[0][2]
    assert 'display notification "three items filed"' in script
    assert 'with title "MemVault"' in script


def test_quotes_in_text_are_escaped(captured_calls: list[list[str]]) -> None:
    notify('a "quoted" title', 'body with "quotes"')

    script = captured_calls[0][2]

    assert '\\"quoted\\"' in script
    assert '\\"quotes\\"' in script


def test_backslashes_in_text_are_escaped(captured_calls: list[list[str]]) -> None:
    notify("MemVault", "path C:\\temp")

    assert "C:\\\\temp" in captured_calls[0][2]


def test_failure_notification_names_the_log_path(captured_calls: list[list[str]]) -> None:
    notify_failure("ingest", "git commit rejected", "/tmp/memvault-ingest.log")

    script = captured_calls[0][2]

    assert "MemVault: ingest failed" in script
    assert "git commit rejected" in script
    assert "/tmp/memvault-ingest.log" in script


def test_missing_osascript_returns_false_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(notify_module.shutil, "which", lambda _: None)

    assert notify("MemVault", "anything") is False


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "osascript"),
        subprocess.TimeoutExpired("osascript", 10),
        OSError("no such process"),
    ],
)
def test_osascript_failures_are_swallowed(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def fake_run(*_: Any, **__: Any) -> subprocess.CompletedProcess[bytes]:
        raise failure

    monkeypatch.setattr(notify_module.shutil, "which", lambda _: "/usr/bin/osascript")
    monkeypatch.setattr(notify_module.subprocess, "run", fake_run)

    assert notify("MemVault", "anything") is False
