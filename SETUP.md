# Setting up MemVault

From nothing to a working vault. Follow this end to end and you will have a memory system whose
every change is a git diff you can read.

MemVault is the **engine**. Your **vault** is a separate git repo holding your Markdown. The
engine holds no content and no paths of its own, so one install can drive several vaults.

## 1. Prerequisites

| Requirement | Why | Check |
|---|---|---|
| Python 3.11+ | the engine | `python3 --version` |
| [uv](https://docs.astral.sh/uv/) | dependency management | `uv --version` |
| git | vaults are git repos; the history is the audit log | `git --version` |
| Claude Code CLI | classification and distillation run on your subscription | `claude --version` |
| macOS | only for scheduling and notifications; the engine itself is portable | — |

The first `memvault index` downloads an embedding model (~130MB, `multilingual-e5-small`). It
runs locally from then on — no API key, no per-run cost, nothing leaves the machine. **On a slow
connection the first run looks like a hang.** It isn't.

## 2. Install the engine

```bash
git clone https://github.com/emre6943/memvault.git
cd MemVault
uv sync
uv run memvault --help
```

## 3. Create a vault and its config

One command does both:

```bash
uv run memvault init ~/my-vault
```

A vault is any git repo. An empty directory gets the skeleton — `inbox/`, `transcripts/`,
`notes/`, `repos/`, `reflections/`, `memory/auto/`, `private/` — plus starter `CLAUDE.md` and
`AGENTS.md`, a `.gitignore`, and a `git init` it asks for first. **If you already keep notes in a
repo, point it there**: init notices, creates only the directories the engine writes into, and
touches nothing that is already yours. It never overwrites without `--force`, so re-running it is
safe, and it prints the lines that wire the vault into Claude Code, Codex, opencode, and Cursor.

The config lands at `~/.config/memvault/config.yaml`, the first place the engine looks. Pass
`--config PATH` to put it somewhere else, `--name` to call the vault something other than its
directory, and `--git` to skip the prompt (a scheduler has no terminal to answer it).

The `private/` ignore rule init writes is load-bearing. Drops marked `local_only: true` are filed
under that prefix, and the ignore rule — not pipeline discipline — is what keeps them off your
remote. On an adopted repo init only *tells* you the rule is missing; editing somebody's
`.gitignore` behind their back is not its job.

## 4. Read the config

The generated file is deliberately minimal — everything not in it has a default:

```yaml
vaults:
  personal:
    root: ~/my-vault
    inbox: inbox
```

`memvault.config.example.yaml` in this repo is the annotated reference: every key, with the
reasoning behind its default. With a single vault it is also the default vault, so `--vault` is
optional. Add a second vault and a `work_route` when you want the personal/work split described
in the requirements doc.

Validation is eager and specific — a wrong path or a typo'd key fails immediately, naming the
offending value, rather than halfway through a pass:

```bash
uv run memvault doctor
```

Expect `all checks healthy`. Anything else prints the problem and its remedy, and exits non-zero.

The config lives outside the engine repo on purpose: the engine is shareable and the config is
yours. Discovery looks in this order, and the first hit wins:

| Where | When it is used |
|---|---|
| `--config PATH` | you named it |
| `$MEMVAULT_CONFIG` | set in the environment |
| `~/.config/memvault/config.yaml` | where `init` writes — the normal home |
| `~/.memvault/config.yaml` | installs predating the XDG path; still read, no rush to move |
| `./memvault.config.yaml` | a scratch vault, or working inside a checkout |

If two of them exist, `init` says so rather than leaving you with a config nothing reads.

## 5. Your first drop

Write anything into the inbox and commit it:

```bash
cd ~/my-vault
cat > inbox/first-note.txt <<'EOF'
Talked to Sam about the migration. We agreed to cut over on the 14th and keep the old
endpoint alive for two weeks. I need to confirm the DNS TTL before then.
EOF
git add inbox && git commit -m "inbox: first drop"
```

No frontmatter, no naming convention — a bare `.txt` is the normal case. Then drain it:

```bash
cd /path/to/MemVault
uv run memvault ingest
```

What happens: the drop is classified, filed into your vault with provenance frontmatter (source,
date, summary, participants, and the model that classified it), removed from the inbox, and both
changes land in **one commit**. Read it:

```bash
cd ~/my-vault && git show --stat HEAD
```

If a drop is still in the inbox afterwards, that is the design working: the classifier was not
confident enough, so the item was held rather than misfiled. A review marker beside it says why.
See [`docs/inbox-contract.md`](docs/inbox-contract.md) for the full drop format.

### Dropping from a phone

Install a git client with share-sheet support ([Working Copy](https://workingcopy.app) on iOS),
clone your vault, and share text into `inbox/`. An iOS Shortcut reduces it to about two taps.
This is how glasses transcripts and exported chats reach the vault.

## 6. Build the index and search

```bash
uv run memvault index
uv run memvault recall "dns cutover"
```

Results cite vault-relative paths so you (or a session) can open the source. The index lives
outside your vault at `~/.memvault/<vault>.db` and is **disposable** — delete it and re-run
`index` any time. Files are the only source of truth.

### Install the recall skill

So Claude Code sessions can query the vault instead of loading memory at startup:

```bash
mkdir -p ~/my-vault/.claude/skills
ln -s /path/to/MemVault/skills/recall ~/my-vault/.claude/skills/recall
```

Then `/recall <question>` works in any session that has the vault on its path.

## 7. Weekly curation

```bash
uv run memvault reflect
```

Reads what was filed since the last pass — plus your claude-mem observations if that plugin is
installed — appends to your durable memory files, writes a digest, and commits. Writes are
append-only: corrections append, history is never rewritten.

If you use claude-mem, list which projects belong to this vault under `claude_mem_projects`.
That list is a **boundary control, not a filter**: claude-mem stores work and personal
observations in one database, and an unlisted project is skipped rather than pulled in.

## 8. Schedule it

```bash
# Write a launchd plist (macOS) or cron/systemd timer that runs scripts/scheduled-pass.sh
# nightly for ingest and weekly for reflect — the script reads MEMVAULT_CONFIG or the
# default config path and is safe to run unattended.
launchctl load ~/Library/LaunchAgents/com.<you>.memvault-ingest.plist
```

Edit the plists first — they contain absolute paths and an author-specific `Label`. Ingest runs
daily at 21:00; reflect runs Sundays at 20:00. Logs go to `/tmp/memvault-ingest.log` and
`/tmp/memvault-reflect.log`.

Failures raise a macOS notification naming the log. Success is silent — a notification on every
successful pass trains the eye to ignore it.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `config error: ... is not a git repository` | A vault must be a git repo. `git init` it. |
| `MISSING inbox not found` | `mkdir inbox` inside the vault. |
| First `index` seems hung | The embedding model is downloading (~130MB). Let it finish once. |
| Items pile up in the inbox | The classifier is not confident. Read the review markers; add a `classification:` to the frontmatter to decide it yourself. |
| `doctor` reports `STALE` | Vault files changed since the last index. Run `memvault index`. |
| `recall` returns nothing you expected | Lower `index.min_similarity` (try 0.78). The default 0.82 is calibrated for `multilingual-e5-small` and must be re-tuned if you change `model`. |
| `recall` returns weak, irrelevant hits | Raise `index.min_similarity` toward 0.84. |
| Scheduled job never runs | `launchctl list | grep memvault`, then check `/tmp/memvault-*.log`. |
| Your shell blocks bare `claude` | Only affects interactive shells. If needed, set `classifier.command` to an absolute path such as `~/.local/bin/claude`. |

## Where things live

| Path | What |
|---|---|
| `<vault>/inbox/` | drops waiting to be filed; empty is the healthy state |
| `<vault>/transcripts/YYYY/MM/` | filed material with provenance frontmatter |
| `<vault>/private/` | `local_only` material, gitignored |
| `~/.config/memvault/config.yaml` | your config (`~/.memvault/config.yaml` still works) |
| `~/.memvault/<vault>.db` | the search index, disposable |
| `/tmp/memvault-*.log` | scheduled-run logs |
