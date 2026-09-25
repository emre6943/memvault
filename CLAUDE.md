# CLAUDE.md — MemVault

MemVault is the **engine** for a file-first memory system: it drains a git-backed inbox,
classifies each drop with an LLM (shelling out to the `claude` CLI), files it into a vault of
Markdown with provenance, routes work material as distilled notes into a second vault, indexes
everything for hybrid search, and runs a weekly curation pass.

**The engine holds no content.** Vaults are separate repos, and every path, route and rule comes
from `memvault.config.yaml`. Never hardcode a personal path in `src/` — that is what makes this
usable by anyone else.

- Adopter setup: `SETUP.md`. Inbox contract (the interface feeders build against):
  `docs/inbox-contract.md`.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
uv run mypy src/              # --strict clean; keep it that way

uv run memvault ingest                     # drain the inbox into the vault
uv run memvault ingest --enqueue           # ask for a pass a few quiet minutes from now (SessionEnd hook)
uv run memvault ingest --session-start ID  # mark a session active so the pass waits (SessionStart hook)
uv run memvault index                      # build/update the search index
uv run memvault recall "..."
uv run memvault reflect                    # weekly curation pass
uv run memvault doctor                     # reports only, never repairs; exits non-zero on problems
uv run memvault mcp [root]                 # stdio MCP: recall, write_inbox, write_note, vault_status
```

`--vault <name>` selects a configured vault; omitting it uses `default_vault`. Without `--config`
the search order is `$MEMVAULT_CONFIG`, `~/.config/memvault/config.yaml`,
`~/.memvault/config.yaml`, then `./memvault.config.yaml`.

## Non-negotiables

- **Raw transcripts never reach the work vault.** Only distilled notes cross. `distill.py`
  enforces this with a verbatim-overlap guard; do not weaken it to make a test pass.
- **Files are the source of truth; the index is disposable.** The index lives outside the vault
  (`index.path`, default `~/.memvault/{vault}.db`); deleting and rebuilding it must reproduce
  equivalent results. Never store anything only in the index.
- **Fail toward the inbox.** When classification is uncertain or anything errors, the item stays
  in the inbox with a marker. A stuck inbox is discovered today; a misfiled memory in six months.
- **Every pass commits.** The git diff is the review surface for all automated memory changes.
- **Append-only means append-only.** A vault's `learnings.md`, `quirks.md` and `reflect-log.md`
  are never rewritten; corrections append. A regression here destroys history with no other copy.

## Conventions

These mirror the vault repos so the two halves read as one system.

- Dates are `YYYY-MM-DD`, everywhere. An em-dash separates heading halves:
  `## 2026-08-01 — <claim>`.
- Skills live at `skills/<name>/SKILL.md` and are installed into a vault's `.claude/skills/`.
  Frontmatter has exactly two keys, `name` and `description`; the description is one unwrapped
  line ending in explicit trigger phrases.
- Shell scripts: `#!/usr/bin/env bash`, `set -euo pipefail`, a header comment stating purpose and
  usage, a `--check` flag that reports without repairing, and a fixed-width uppercase status
  column (`OK`, `STALE`, `STUCK`) with each problem line ending in its remedy.
- Automated failures notify; success is silent. Notifications use `osascript` (`notify.py`), so
  they are macOS-only and a silent no-op elsewhere — the log path in the message is what matters.

## claude-mem database

The weekly pass reads `~/.claude-mem/claude-mem.db` read-only. Four verified traps:

- `*_epoch` columns are **milliseconds**, not seconds.
- Open with `file:...?mode=ro`. Never `?immutable=1` — it ignores the WAL and returns stale data.
- Never `INNER JOIN` through `sdk_sessions` — most observations no longer have a session row.
  Query `observations` and `session_summaries` standalone; both carry `project` and timestamps.
- Work and personal observations share this one database. Each vault config carries a project
  allowlist; an unknown project is skipped and logged, never defaulted into a vault.
