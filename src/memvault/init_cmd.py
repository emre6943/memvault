"""`memvault init` — first contact with the engine.

Nothing else in this package runs without a config, and until this command existed the only way
to get one was to read SETUP.md and hand-write YAML against a schema that fails eagerly. That is
a dead end for a first-time adopter, which is what R8 exists to close: init must leave behind a
directory the other subcommands can immediately act on.

Two shapes of first contact, and they are different jobs:

**Scaffold** — an absent or empty directory becomes a vault: the directories the engine writes
into, starter `CLAUDE.md`/`AGENTS.md`, a `.gitignore` that excludes `private/`, and a git
repository, because config validation refuses a vault that is not one (the commit history is the
audit log, so this is a rule rather than a preference).

**Adopt** — a directory that already holds somebody's notes. There the tree belongs to its owner:
init creates the directories the engine writes into and touches nothing that already exists. No
starter files, no edit to `.gitignore`, and no assumption that the tree is clean — a lived-in repo
usually is not, and a command that refused to run over uncommitted work would be refusing the
normal case rather than protecting anything.

The mode is inferred from what is on disk and always printed, so it is never a surprise;
`--adopt` forces the careful one.

**Every refusal happens before the first write.** A config that already exists and says something
else, a git repository the user declined — both stop the pass while the disk still looks the way
it did. A refused init leaves no half-vault behind, which is what makes re-running it safe.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from memvault.config import CONFIG_ENV_VAR, ConfigError, load_config, user_config_paths
from memvault.git_ops import GitError, run_git

#: Directories the engine writes into, created when missing in both modes. They are the default
#: template roots from `VaultConfig`, one level deep: `transcripts/` for filed material, `notes/`
#: for distilled notes, `repos/` for per-project durable memory, `reflections/` for the weekly
#: pass's note, and `memory/auto/` for harness-managed auto-memory. A vault that re-templates any
#: of them keeps these as empty leftovers rather than as a broken assumption — the writers create
#: what they need on demand and never read this list.
SKELETON_DIRS = (
    "inbox",
    "transcripts",
    "notes",
    "repos",
    "reflections",
    "memory/auto",
)

#: Where `local_only` material is filed. Deliberately not `.gitkeep`-ed: the whole point of the
#: directory is that git ignores it, so a tracked placeholder inside it would be the one file
#: from that tree that reaches a remote.
PRIVATE_DIR = "private"

#: Empty directories do not exist in git, so a scaffolded vault would lose its shape on the first
#: clone without these.
GITKEEP = ".gitkeep"

#: The one rule the vault's `.gitignore` must carry. `local_only` enforcement is this line, not
#: pipeline discipline, which is why a scaffold writes it and an adopt reports its absence rather
#: than silently editing a file somebody else wrote.
GITIGNORE_BODY = """\
# Material dropped with `local_only: true` is filed here. This line is what keeps it off your
# remote — the engine relies on the ignore rule rather than on discipline.
private/

# Editor and OS noise.
.DS_Store
"""

#: Starter vault context, written to both `CLAUDE.md` and `AGENTS.md`. The duplication is
#: deliberate: the two files are read by different harnesses, neither reliably follows a pointer
#: to the other, and a generated pair is honest about being a starting point.
VAULT_CONTEXT = """\
# This vault

Everything in here is memory: plain Markdown files in one git repo. There is no database that
outlives them — the search index is disposable and rebuilt from these files at any time.

## How material arrives

Drop anything into `inbox/` and commit it. `memvault ingest` classifies each drop, files it with
provenance frontmatter, removes it from the inbox, and commits the whole move as one diff. The
inbox drains to empty, so a non-empty `inbox/` means work is pending.

## Where things live

| Path | What |
|---|---|
| `inbox/` | drops waiting to be filed; empty is the healthy state |
| `transcripts/` | filed raw material, with provenance |
| `notes/` | distilled notes |
| `repos/` | per-project durable memory (learnings, quirks, logs) |
| `reflections/` | what the weekly pass concluded, with citations |
| `memory/auto/` | harness-managed automatic memory |
| `private/` | `local_only` material — gitignored, never pushed |

## Retrieval

Search by meaning rather than reading files in bulk:

```bash
memvault recall "what did we decide about X"
```

Results cite vault-relative paths, so open what looks relevant instead of loading memory at
startup.

## Conventions

Dates are `YYYY-MM-DD`. Durable memory files are append-only — a correction appends, history is
never rewritten.

<!-- This file and AGENTS.md were generated by `memvault init` and hold the same text for two
     different harnesses. Edit both, or delete the one your tools do not read. -->
"""

#: Written into `CLAUDE.md` and `AGENTS.md` alike.
STARTER_FILES: tuple[tuple[str, str], ...] = (
    ("CLAUDE.md", VAULT_CONTEXT),
    ("AGENTS.md", VAULT_CONTEXT),
    (".gitignore", GITIGNORE_BODY),
)

#: What a vault is called when the directory name yields nothing usable as a YAML key.
FALLBACK_VAULT_NAME = "vault"

_VAULT_NAME = re.compile(r"^[a-z][a-z0-9-]*$")

#: The written config, kept minimal on purpose: every other key has a default, and a generated
#: file full of commented-out settings is a file nobody reads. The annotated example that ships
#: with the engine is the reference.
CONFIG_TEMPLATE = """\
# MemVault configuration, written by `memvault init`.
#
# The engine holds no vault paths of its own — everything it needs is here, which is what lets
# one install drive several vaults. Every key not shown has a default; see the annotated
# `memvault.config.example.yaml` that ships with the engine for the full surface.

default_vault: {name}

vaults:
  {name}:
    # A vault must be a git repository: its commit history is the audit log.
    root: {root}
    inbox: inbox
"""

#: Width of the harness-name column in the printed wiring block.
_HARNESS_COLUMN = 12

#: Answers a yes/no question. A seam rather than a direct `input()` call so the flags, the
#: prompt, and the tests are three callers of one decision instead of three code paths.
Confirm = Callable[[str], bool]


def decline(_question: str) -> bool:
    """The safe default: answer no to everything, mutate nothing."""
    return False


class InitError(Exception):
    """Raised when init refuses to proceed, always before anything has been written."""


@dataclass(frozen=True)
class InitReport:
    """What one init pass did, in the order a reader wants it."""

    root: Path
    vault_name: str
    config_path: Path
    #: True when the directory already held material of its own and was treated as somebody
    #: else's tree rather than as a blank one.
    adopted: bool
    created: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    git_initialized: bool = False
    config_written: bool = False

    def summary_line(self) -> str:
        mode = "adopted" if self.adopted else "scaffolded"
        config = "config written" if self.config_written else "config unchanged"
        return (
            f"{mode} {self.root} as vault {self.vault_name!r} — "
            f"{len(self.created)} path(s) created, {config}"
        )


def normalize_vault_name(raw: str) -> str:
    """Fold a directory name into a vault name, or return "" when nothing usable is left.

    The result doubles as a YAML mapping key and as a filename component (`{vault}.db`), so it is
    restricted to a leading letter plus lowercase alphanumerics and dashes. `2026` would parse
    back as an integer key and `my vault.db` is a path waiting to be mis-quoted; both are worth
    refusing at the one point where a better name can still be asked for.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", raw.strip().lower()).strip("-")
    return slug if _VAULT_NAME.match(slug) else ""


def default_vault_name(root: Path) -> str:
    """What to call the vault at `root` when nobody said."""
    return normalize_vault_name(root.name) or FALLBACK_VAULT_NAME


def default_config_path() -> Path:
    """Where a written config goes: the XDG location, the first place discovery looks (R9)."""
    return user_config_paths()[0]


def render_config(vault_name: str, root: Path) -> str:
    """The config file text for one vault.

    The root is JSON-quoted, which is valid YAML and survives a path containing a space, a `#`,
    or a colon. It is written absolute rather than as `~/...` because a scheduler and a hook do
    not necessarily agree with an interactive shell about what `~` means.
    """
    return CONFIG_TEMPLATE.format(name=vault_name, root=json.dumps(str(root)))


@dataclass(frozen=True)
class Wiring:
    """How one harness is pointed at the vault.

    `where` is the command to run or the file to edit; `body` is what goes in that file, already
    formatted. Structured rather than pre-joined text so the JSON entries can be parsed in a test
    — a config block that does not parse is worse than no block at all, and nobody notices a
    stray brace by reading.
    """

    harness: str
    where: str
    body: tuple[str, ...] = ()


def wiring(root: Path) -> tuple[Wiring, ...]:
    """How to point each harness at this vault, ready to paste.

    Every harness reaches the vault the same way — the stdio MCP server the `mcp` subcommand
    runs — so the differences are entirely about where each one keeps its config. Claude Code has
    a command for it; the other three want a file edited.
    """
    quoted = json.dumps(str(root))
    return (
        Wiring("Claude Code", f"claude mcp add memvault -- memvault mcp {root}"),
        Wiring(
            "Codex",
            "~/.codex/config.toml",
            (
                "[mcp_servers.memvault]",
                'command = "memvault"',
                f'args = ["mcp", {quoted}]',
            ),
        ),
        Wiring(
            "opencode",
            "~/.config/opencode/opencode.json",
            (
                '{"mcp": {"memvault": {"type": "local",',
                f'  "command": ["memvault", "mcp", {quoted}], "enabled": true}}}}}}',
            ),
        ),
        Wiring(
            "Cursor",
            "~/.cursor/mcp.json",
            (
                '{"mcpServers": {"memvault":',
                f'  {{"command": "memvault", "args": ["mcp", {quoted}]}}}}}}',
            ),
        ),
    )


def wiring_lines(root: Path) -> tuple[str, ...]:
    """`wiring` rendered for the terminal, one harness per block."""
    lines: list[str] = []
    for entry in wiring(root):
        lines.append(f"  {entry.harness:<{_HARNESS_COLUMN}} {entry.where}")
        lines.extend(f"  {'':<{_HARNESS_COLUMN}}   {line}" for line in entry.body)
    return tuple(lines)


def looks_lived_in(root: Path) -> bool:
    """Whether this directory already holds material that is not ours to write.

    Only the top level is examined, and only for entries the scaffold does not own. That keeps
    re-running init on its own output in scaffold mode (so a missing directory is restored) while
    a clone of a template repo, or somebody's years-old notes folder, is adopted on sight.
    """
    if not root.is_dir():
        return False
    ours = {name.split("/", 1)[0] for name in SKELETON_DIRS}
    ours.add(PRIVATE_DIR)
    ours.update(name for name, _ in STARTER_FILES)
    return any(
        entry.name not in ours and not entry.name.startswith(".") for entry in root.iterdir()
    )


def _same_path(left: Path, right: Path) -> bool:
    """Compare two paths that may or may not exist, past symlinks and `..`."""
    return left.resolve() == right.resolve()


def shadowed_by(written: Path) -> Path | None:
    """A config that discovery would find *before* the one init just wrote, if any.

    Writing a config that never gets read is the quietest way for init to fail, and it is easy to
    arrange: a `$MEMVAULT_CONFIG` left over from an earlier setup, or a legacy `~/.memvault`
    config being written while an XDG one already exists. Reported, never removed.
    """
    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        env_path = Path(env_value).expanduser()
        return None if _same_path(env_path, written) else env_path

    for candidate in user_config_paths():
        if _same_path(candidate, written):
            return None
        if candidate.exists():
            return candidate
    return None


def _config_action(config_path: Path, text: str, *, force: bool) -> bool:
    """Whether the config needs writing. Raises rather than overwrite somebody else's file.

    An identical file is a no-op and not an error, which is what makes a second `memvault init`
    against the same vault a safe thing to run — the common case is a user who is not sure
    whether the first one worked.
    """
    if not config_path.exists():
        return True
    if config_path.read_text(encoding="utf-8") == text:
        return False
    if force:
        return True
    raise InitError(
        f"{config_path} already exists and says something else. Re-run with --force to replace "
        f"it, pass --config to write somewhere else, or edit it by hand."
    )


def _ensure_git(root: Path, confirm: Confirm) -> bool:
    """Make `root` a git repository, asking first. Returns whether one was created."""
    if (root / ".git").exists():
        return False
    if not confirm(f"{root} is not a git repository. Run `git init` there?"):
        raise InitError(
            f"{root} is not a git repository, so a config naming it would not load. A vault must "
            f"be a git repo — the commit history is the audit log. Run `git init` there yourself, "
            f"or re-run with --git."
        )
    root.mkdir(parents=True, exist_ok=True)
    try:
        run_git(root, "init", "-q")
    except GitError as exc:
        raise InitError(str(exc)) from exc
    return True


def _scaffold(root: Path, *, adopt: bool, force: bool) -> tuple[list[str], list[str], list[str]]:
    """Create what is missing. Returns (created, kept, notes), all vault-relative."""
    created: list[str] = []
    kept: list[str] = []
    notes: list[str] = []

    for relative in SKELETON_DIRS:
        directory = root / relative
        if directory.is_dir():
            kept.append(f"{relative}/")
            continue
        directory.mkdir(parents=True, exist_ok=True)
        (directory / GITKEEP).write_text("", encoding="utf-8")
        created.append(f"{relative}/")

    # An adopted tree gets no starter files, and its `.gitignore` is its owner's to edit.
    if not adopt:
        private = root / PRIVATE_DIR
        if private.is_dir():
            kept.append(f"{PRIVATE_DIR}/")
        else:
            private.mkdir(parents=True, exist_ok=True)
            created.append(f"{PRIVATE_DIR}/")

        for name, body in STARTER_FILES:
            path = root / name
            if path.exists() and not force:
                kept.append(name)
                continue
            path.write_text(body, encoding="utf-8", newline="\n")
            created.append(name)

    # Asked of the finished tree rather than assumed from what was written, and asked in both
    # modes: a repo that arrived with a `.gitignore` of its own keeps it, in which case nothing
    # so far has established the one rule the privacy guarantee rests on.
    if not _ignores_private(root):
        notes.append(
            f"add `{PRIVATE_DIR}/` to .gitignore — it is what keeps `local_only` material "
            f"off your remote"
        )

    return created, kept, notes


def _ignores_private(root: Path) -> bool:
    """Whether git already excludes the local-only prefix in this repository.

    Asked of git rather than of `.gitignore`, because the rule may live in a global excludes file
    or in `.git/info/exclude`, and a false alarm about privacy is the kind of warning people
    learn to ignore.
    """
    probe = f"{PRIVATE_DIR}/probe.md"
    try:
        return bool(run_git(root, "check-ignore", "--", probe, check=False).strip())
    except GitError:
        return False


def init_vault(
    path: str | Path,
    *,
    config_path: Path | None = None,
    vault_name: str | None = None,
    adopt: bool | None = None,
    force: bool = False,
    confirm: Confirm = decline,
) -> InitReport:
    """Scaffold or adopt a vault at `path` and write a config naming it.

    `adopt=None` infers the mode from what is on disk. Refusals — a conflicting config, a
    declined `git init`, an unusable name — are raised before anything is created.
    """
    root = Path(path).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise InitError(f"{root} is not a directory")

    if vault_name is None:
        name = default_vault_name(root)
    else:
        name = normalize_vault_name(vault_name)
        if not name:
            raise InitError(
                f"{vault_name!r} is not usable as a vault name. Use a letter followed by "
                f"lowercase letters, digits, or dashes — it becomes a config key and a filename."
            )

    adopting = looks_lived_in(root) if adopt is None else adopt
    if adopting and not root.is_dir():
        raise InitError(f"nothing to adopt: {root} does not exist")

    target = (config_path or default_config_path()).expanduser()
    text = render_config(name, root)
    # Both refusals are resolved before the first write, so a refused run changes nothing.
    needs_config = _config_action(target, text, force=force)
    git_initialized = _ensure_git(root, confirm)

    created, kept, notes = _scaffold(root, adopt=adopting, force=force)

    if needs_config:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")

    try:
        load_config(target)
    except ConfigError as exc:
        raise InitError(f"the config at {target} does not load: {exc}") from exc

    shadow = shadowed_by(target)
    if shadow is not None:
        notes.append(f"{shadow} is discovered before {target} and will win — remove or move one")

    return InitReport(
        root=root,
        vault_name=name,
        config_path=target,
        adopted=adopting,
        created=tuple(created),
        kept=tuple(kept),
        notes=tuple(notes),
        git_initialized=git_initialized,
        config_written=needs_config,
    )
