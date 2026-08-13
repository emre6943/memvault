"""macOS notifications for unattended runs.

Failures notify; success is silent. That asymmetry is deliberate — the scheduled passes run
several times a day, and a notification that fires on success trains the eye to ignore it.

A notification is never load-bearing: if `osascript` is missing, sandboxed, or fails for any
reason, the caller carries on. Losing a notification must never fail an ingestion pass.
"""

from __future__ import annotations

import shutil
import subprocess

NOTIFY_TIMEOUT_SECONDS = 10


def _escape(text: str) -> str:
    """Escape for embedding in an AppleScript double-quoted string."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, message: str, *, sound: str = "Basso") -> bool:
    """Post a macOS notification. Returns whether it was delivered.

    The return value exists for tests and callers that want to log a missed notification.
    No caller should branch on it to decide whether its own work succeeded.
    """
    binary = shutil.which("osascript")
    if binary is None:
        return False

    script = (
        f'display notification "{_escape(message)}" '
        f'with title "{_escape(title)}" '
        f'sound name "{_escape(sound)}"'
    )

    try:
        subprocess.run(
            [binary, "-e", script],
            check=True,
            capture_output=True,
            timeout=NOTIFY_TIMEOUT_SECONDS,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def notify_failure(job: str, reason: str, log_path: str) -> bool:
    """Notify about a failed unattended run, naming the log that explains it.

    Naming the log path in the message matches the vault's own backup job: the notification is
    only useful if it tells you where to look next.
    """
    return notify(f"MemVault: {job} failed", f"{reason} — see {log_path}")
