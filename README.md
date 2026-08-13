# MemVault

A file-first memory engine. Everything you learn, say, or are told enters through one inbox, gets
classified and filed into a vault of plain Markdown, and becomes retrievable by meaning instead of
by recency.

Every memory is a file. Every automated change is a git diff you can read and revert.

## How it works

```
SOURCES            INBOX           ENGINE                    VAULT
glasses ────┐                  ┌─ normalize                transcripts/
claude.ai ──┼──▶ inbox/ ──────▶├─ classify (LLM)   ──────▶ memory/
manual ─────┘    (git)         ├─ route                    repos/<name>/
                               ├─ distill                  index/vault.db
                               └─ index
```

Drop any text into `inbox/` and commit — from a laptop, or from a phone with an iOS git client.
The ingestion pass classifies it, files it into the vault with provenance, updates the search
index, and commits. The inbox drains to empty; a non-empty inbox means work is pending.

A weekly pass reads what arrived, curates it into durable memory, and writes a digest. A `/recall`
skill answers questions against the whole vault so sessions fetch depth on demand rather than
loading memory in bulk at startup.

## Engine and vault are separate

MemVault is the engine — code, no content. A **vault** is a separate git repo holding your
Markdown. The engine reads a config naming your vault's paths and routes, so one install serves
several vaults (for example, a personal one and a work one) with no code changes.

## Install

```bash
uv tool install "memvault-cli[semantic]"    # recommended
uv tool install memvault-cli                # light: keyword search only
```

The `semantic` extra pulls `fastembed` and downloads a ~130MB multilingual model on first index.
Without it everything still works and every file is still indexed — `recall` runs the keyword
half alone and says so in its own output, so half a search never reads like a whole one. Adding
the extra later is `memvault index` again.

Working on the engine itself instead? `uv sync` and prefix the commands below with `uv run`.

## Commands

```bash
memvault init ~/my-vault   # scaffold a vault (or adopt notes you already keep) + config
memvault doctor            # vault health; exits non-zero on problems
memvault ingest            # drain the inbox into the vault, then commit
memvault index             # build or update the search index
memvault recall "query"    # hybrid keyword + semantic search
memvault reflect           # weekly curation pass over recent material
memvault mcp ~/my-vault    # serve the vault over MCP (Codex, opencode, Cursor, …)
```

`memvault init` prints the exact wiring line for each harness once your vault exists.

Ingest can also run itself a few quiet minutes after you stop working, instead of on a clock. A
SessionStart hook runs `memvault ingest --session-start <id>` and a SessionEnd hook runs
`memvault ingest --enqueue`; a plain `memvault ingest` from a scheduled job stays the safety net
and services whatever the queue is still asking for. See `debounce:` in the example config.

`--vault <name>` picks a configured vault. Start with [`SETUP.md`](SETUP.md).

## Status

**Implemented and verified end to end**, and in daily use. `ruff` and `mypy --strict` clean.

Verified against a real vault rather than only in tests — a dropped note was classified, filed
with provenance, and committed with the inbox drained; a work-flavored drop produced a distilled
note in the second vault with no verbatim sentence crossing; re-running produced no duplicate.

- Requirements: [`docs/brainstorms/2026-08-01-memvault-requirements.md`](docs/brainstorms/2026-08-01-memvault-requirements.md)
- Plans: [`docs/plans/`](docs/plans/)
- Inbox contract, for building feeders: [`docs/inbox-contract.md`](docs/inbox-contract.md)
- Before this repo goes public: [`docs/release-audit-checklist.md`](docs/release-audit-checklist.md)

## License

MIT — see [`LICENSE`](LICENSE).
