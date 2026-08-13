# CLAUDE.md — MemVault

Guidance for Claude Code when working in this repo.

## What this is

MemVault is the **engine** for a file-first memory system: it drains a git-backed inbox,
classifies each drop with an LLM, files it into a vault of Markdown with provenance, routes work
material as distilled notes into a second vault, indexes everything for hybrid search, and runs a
weekly curation pass.

**The engine holds no content.** Vaults are separate repos — e.g. a personal vault and a work vault
and every path, route, and rule comes from `memvault.config.yaml`. Never hardcode a personal path
in `src/` — that is the property that makes this usable by anyone else.

Status: **implemented.** All ten plan units are built and verified end to end against a real
vault. 629 tests; `ruff` and `mypy --strict` clean.

- Requirements: `docs/brainstorms/2026-08-01-memvault-requirements.md`
- Plan: `docs/plans/2026-08-01-001-feat-memvault-engine-plan.md`
- Inbox contract (the interface feeders build against): `docs/inbox-contract.md`

## Commands

```bash
uv sync                       # install deps
uv run pytest                 # full test suite
uv run pytest tests/test_x.py # single file
uv run ruff check . && uv run ruff format .
uv run mypy src/

uv run memvault ingest        # drain the inbox into the vault
uv run memvault ingest --enqueue        # ask for one a few quiet minutes from now (SessionEnd)
uv run memvault ingest --session-start ID  # mark a session active (SessionStart)
uv run memvault index         # rebuild/update the search index
uv run memvault recall "..."  # hybrid search
uv run memvault reflect       # weekly curation pass
uv run memvault doctor        # health check; reports only, exits non-zero on problems
uv run memvault mcp <root>    # stdio MCP server: recall, write_inbox, write_note, vault_status
```

`--vault <name>` selects a configured vault; omitting it uses the default.

## Conventions

These mirror the vault repos so the two halves read as one system.

- **Dates are `YYYY-MM-DD`.** No exceptions, anywhere.
- **Em-dash `—` separates heading halves**, e.g. `## 2026-08-01 — <claim>`.
- **Append-only means append-only.** `learnings.md`, `quirks.md`, and `reflect-log.md` in a vault
  are never rewritten. Corrections append. A regression here destroys history with no other copy.
- **Skills** live at `skills/<name>/SKILL.md` and are installed into a vault's `.claude/skills/`.
  Frontmatter carries exactly two keys, `name` and `description`, with the description on one
  unwrapped line ending in explicit trigger phrases.
- **Shell scripts**: `#!/usr/bin/env bash`, `set -euo pipefail`, a header comment block stating
  purpose and usage, a `--check` flag that reports without repairing, and a fixed-width uppercase
  status column (`OK`, `STALE`, `STUCK`) with each problem line ending in its remedy.
- **Automated failures notify; success is silent.** `osascript display notification`, naming the
  log path in the message.
- **Python**: type hints throughout, `ruff` formatting, errors handled explicitly. This is a real
  Python project, so `uv` — unlike a vault's bare `python3` utility scripts.

## Non-negotiables

- **Raw transcripts never reach the work vault.** Only distilled notes cross. `distill.py` enforces
  this mechanically with a verbatim-overlap guard; do not weaken it to make a test pass.
- **Files are the source of truth; the index is disposable.** Deleting `index/vault.db` and
  rebuilding must reproduce equivalent results. Never store something only in the index.
- **Fail toward the inbox.** When classification is uncertain or anything errors, the item stays in
  the inbox with a marker. A stuck inbox is discovered today; a misfiled memory is discovered in
  six months.
- **Every pass commits.** The git diff is the review surface for all automated memory changes.

## claude-mem database

The weekly pass reads `~/.claude-mem/claude-mem.db` read-only. Four traps, all verified:

- `*_epoch` columns are **milliseconds**, not seconds.
- Open with `file:...?mode=ro`. **Never `?immutable=1`** — it ignores the WAL and returns stale data.
- **Never `INNER JOIN` through `sdk_sessions`** — only ~21% of observations still have a session
  row. Query `observations` and `session_summaries` standalone; both carry `project` and timestamps.
- **Work and personal observations share this one database.** Each vault config carries a project
  allowlist; an unknown project is skipped and logged, never defaulted into the vault.
