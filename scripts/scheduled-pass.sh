#!/usr/bin/env bash
#
# Run one MemVault pass unattended, for launchd.
#
# Wraps `uv run memvault <command>` so the plists can point at a single absolute path and so
# failures produce a visible notification rather than a silent non-zero exit in a log nobody
# reads. Success is quiet by design.
#
# Config: $MEMVAULT_CONFIG, else ~/.memvault/config.yaml. launchd jobs carry no environment,
# so the default is what normally applies.
#
# Usage: scripts/scheduled-pass.sh <ingest|reflect|index> [vault]
# Cadence: ingest daily 21:00, reflect weekly (Sun 20:00) — see config/launchd/

set -euo pipefail

# launchd runs with a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin), which contains neither uv
# nor the claude CLI the classifier shells out to. Prepend the usual install locations so a
# scheduled run behaves like an interactive one.
PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
export PATH

REPO="$(cd "$(dirname "$0")/.." && pwd)"
COMMAND="${1:-}"
VAULT="${2:-}"
CONFIG="${MEMVAULT_CONFIG:-$HOME/.memvault/config.yaml}"
LOG_PREFIX="[memvault ${COMMAND:-?} $(date +%F' '%T)]"
LOG_PATH="/tmp/memvault-${COMMAND:-unknown}.log"

notify() { # notify <title> <message>
  osascript -e "display notification \"$2\" with title \"$1\" sound name \"Basso\"" 2>/dev/null || true
}

case "$COMMAND" in
  ingest|reflect|index) ;;
  *)
    echo "$LOG_PREFIX ERROR: expected <ingest|reflect|index>, got '${COMMAND:-}'" >&2
    exit 2
    ;;
esac

if [ ! -f "$CONFIG" ]; then
  echo "$LOG_PREFIX ERROR: config not found at $CONFIG" >&2
  notify "MemVault: $COMMAND failed" "config not found at $CONFIG — see $LOG_PATH"
  exit 1
fi

if ! command -v uv &>/dev/null; then
  echo "$LOG_PREFIX ERROR: uv not on PATH" >&2
  notify "MemVault: $COMMAND failed" "uv not on PATH — see $LOG_PATH"
  exit 1
fi

ARGS=(run memvault --config "$CONFIG")
[ -n "$VAULT" ] && ARGS+=(--vault "$VAULT")
ARGS+=("$COMMAND")

echo "$LOG_PREFIX starting (config: $CONFIG${VAULT:+, vault: $VAULT})"

# Capture the status directly rather than reading $? after the `if`.
#
# `if cmd; then ...; fi` with a false condition and no else branch evaluates to
# 0, so `status=$?` after `fi` read 0 on EVERY failure. The log said
# "FAILED (exit 0)", launchd saw a clean exit, and the notification never fired.
# That is why reflect could fail on every attempt from 2026-08-02 onward and
# nothing anywhere said so — found 2026-08-07.
set +e
(cd "$REPO" && uv "${ARGS[@]}")
status=$?
set -e

if [ "$status" -eq 0 ]; then
  echo "$LOG_PREFIX done"
  exit 0
fi

echo "$LOG_PREFIX FAILED (exit $status)" >&2
notify "MemVault: $COMMAND failed" "exit $status — see $LOG_PATH"
exit "$status"
