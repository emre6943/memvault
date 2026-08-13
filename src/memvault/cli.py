"""Command-line entry point.

Subcommands are thin: they resolve config, then hand off to the module that owns the work.
`doctor` reports and never repairs, following the vault scripts' `--check` convention.

One subcommand does not resolve config: `init` exists precisely because there is none yet, so
dispatch splits before the load rather than after it. Anything else that finds no config says so
and points at `init`, because a first-time adopter meeting a bare "no config file found" has no
way to guess what comes next.

`ingest` carries three extra modes as flags rather than as subcommands of its own, because they
are all the same pass reached at different moments: a hook asking for one (`--enqueue`), a hook
announcing a session so the pass waits for it (`--session-start`), and the detached process that
eventually performs it (`--debounce-runner`). A plain `memvault ingest` stays the safety net and
answers whatever a queued run was still owed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from memvault.config import Config, ConfigError, IndexConfig, VaultConfig, load_config
from memvault.debounce import (
    StatePaths,
    clear_queue,
    clear_session,
    configure_logging,
    enqueue,
    ensure_runner,
    mark_session_start,
    run_runner,
)
from memvault.index import (
    NULL_MODEL_ID,
    SEMANTIC_EXTRA_HINT,
    Embedder,
    FastEmbedEmbedder,
    NullEmbedder,
    build_index,
    semantic_available,
)
from memvault.ingest import (
    DEFAULT_LOG_PATH,
    IngestReport,
    IngestStatus,
    run_ingest,
)
from memvault.ingest import (
    MARKER_SUFFIX as REVIEW_MARKER_SUFFIX,
)
from memvault.init_cmd import Confirm, InitError, InitReport, init_vault, wiring_lines
from memvault.recall import (
    DEFAULT_LIMIT,
    IndexBuildError,
    RecallError,
    format_results,
    query_embedder,
)
from memvault.recall import recall as run_recall
from memvault.reflect import DEFAULT_LOG_PATH as DEFAULT_REFLECT_LOG_PATH
from memvault.reflect import DEFAULT_WINDOW_DAYS, ReflectReport, run_reflect

STATUS_WIDTH = 8

#: What the index freshness check concluded, as a word two surfaces can both report.
INDEX_MISSING = "missing"
INDEX_STALE = "stale"
INDEX_CURRENT = "current"


def _status(keyword: str, message: str) -> str:
    """Fixed-width uppercase status column, matching the vault's shell scripts."""
    return f"{keyword.upper():<{STATUS_WIDTH}} {message}"


@dataclass(frozen=True)
class Check:
    """One health finding: its status word, its sentence, and whether it counts as a problem.

    `detail` is the indented block under a finding — the reason beside each held drop — kept
    apart from `message` so a structured consumer gets the lines as lines rather than as one
    string with newlines in it.
    """

    keyword: str
    message: str
    detail: tuple[str, ...] = ()
    problem: bool = False


@dataclass(frozen=True)
class HealthReport:
    """What `doctor` found, before anything is printed.

    Data rather than output, because two surfaces ask this same question and must not develop
    two opinions of healthy: the terminal, where the answer is status lines and an exit code,
    and the MCP `vault_status` tool, where it is a record a model reads. The renderer below owns
    the words; this owns the findings.
    """

    config_path: Path
    vault: str
    root: Path
    inbox: Path
    inbox_pending: int
    held_for_review: int
    index_path: Path
    index_state: str
    checks: tuple[Check, ...] = ()

    @property
    def problems(self) -> int:
        return sum(1 for check in self.checks if check.problem)

    @property
    def healthy(self) -> bool:
        return self.problems == 0


def _index_check(db_path: Path, vault: VaultConfig) -> tuple[str, Check]:
    """Compare the index's age against the newest vault file.

    Staleness is worth a non-zero exit because a stale index fails silently: recall keeps
    answering, just from an older vault, which is harder to notice than an error.
    """
    if not db_path.exists():
        return INDEX_MISSING, Check(
            "missing", f"no index at {db_path} — run `memvault index`", problem=True
        )

    newest = 0.0
    for path in vault.root.rglob("*.md"):
        if ".git" in path.parts or path.is_relative_to(vault.inbox):
            continue
        newest = max(newest, path.stat().st_mtime)

    if newest > db_path.stat().st_mtime:
        return INDEX_STALE, Check(
            "stale", "index older than the vault — run `memvault index`", problem=True
        )

    return INDEX_CURRENT, Check("ok", f"index current at {db_path}")


def check_health(config: Config, vault: VaultConfig) -> HealthReport:
    """Inspect a vault and return what is wrong with it. Inspects only — never repairs."""
    checks: list[Check] = [
        Check("ok", f"config valid: {config.source}"),
        Check("ok", f"vault '{vault.name}' at {vault.root}"),
    ]

    drops: list[Path] = []
    markers: list[Path] = []
    if vault.inbox.is_dir():
        # Hidden files are not drops — the inbox contract skips any path component starting
        # with a dot, so counting `.gitkeep` here would report work that will never happen.
        pending = [
            p
            for p in vault.inbox.rglob("*")
            if p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(vault.inbox).parts)
        ]
        markers = [p for p in pending if p.name.endswith(REVIEW_MARKER_SUFFIX)]
        drops = [p for p in pending if not p.name.endswith(REVIEW_MARKER_SUFFIX)]
        checks.append(
            Check("ok" if not drops else "PENDING", f"inbox {vault.inbox} — {len(drops)} item(s)")
        )
        if markers:
            checks.append(
                Check(
                    "stuck",
                    f"{len(markers)} item(s) held for review — read the "
                    f"*{REVIEW_MARKER_SUFFIX} file beside each, then fix or delete the drop",
                    detail=tuple(
                        f"{marker.name} — "
                        f"{marker.read_text(encoding='utf-8', errors='replace').strip()}"
                        for marker in sorted(markers)
                    ),
                    problem=True,
                )
            )
    else:
        checks.append(
            Check(
                "missing",
                f"inbox not found at {vault.inbox} — create it, then re-run",
                problem=True,
            )
        )

    db_path = config.index.db_path(vault.name)
    index_state, index_check = _index_check(db_path, vault)
    checks.append(index_check)

    target = config.work_target(vault)
    if vault.work_route:
        if target is None:
            checks.append(
                Check(
                    "missing",
                    f"work_route '{vault.work_route}' unresolved — fix config",
                    problem=True,
                )
            )
        else:
            checks.append(Check("ok", f"work route → '{target.name}' at {target.root}"))

    return HealthReport(
        config_path=config.source,
        vault=vault.name,
        root=vault.root,
        inbox=vault.inbox,
        inbox_pending=len(drops),
        held_for_review=len(markers),
        index_path=db_path,
        index_state=index_state,
        checks=tuple(checks),
    )


def doctor(config: Config, vault: VaultConfig) -> int:
    """Report vault health. Reports only — never repairs."""
    report = check_health(config, vault)

    for check in report.checks:
        print(_status(check.keyword, check.message))
        for line in check.detail:
            print(f"{'':<{STATUS_WIDTH}} {line}")

    if report.problems:
        print(f"problems: {report.problems}")
        return 1

    print("all checks healthy")
    return 0


def _print_ingest(report: IngestReport) -> None:
    """Print the pass as status lines, held items last so they end up nearest the prompt."""
    for item in report.filed:
        print(_status("filed", item.summary_line()))
    for item in report.notes:
        print(_status("note", f"{item.vault} — {item.summary_line()}"))
    for drained in report.recovered:
        print(_status("drained", f"{drained.filename} — {drained.reason}"))
    for reference in report.commits:
        print(_status("commit", f"{reference.vault} — {reference.short_sha}"))
    for blocked in report.note_failures:
        print(_status("blocked", f"{blocked.filename} — {blocked.reason}"))
    for held in report.held:
        print(_status("review", f"{held.filename} — {held.reason}"))

    if report.dirty is not None and report.failure is None:
        # The pass had nothing to file, so the dirt cost it nothing — but it will refuse the
        # moment a drop arrives, and saying so now is what stops that from being a surprise.
        print(_status("dirty", f"the next pass with work to do will refuse: {report.dirty}"))

    if report.failure is not None:
        print(_status("failed", report.failure), file=sys.stderr)
    print(report.summary_line())


def _debounce_command(config: Config, vault: VaultConfig, args: argparse.Namespace) -> int | None:
    """Handle the three idle-triggering modes, or return None when this is a normal ingest.

    All three are `ingest` flags rather than subcommands of their own because they are the same
    pass reached at different moments: a hook asking for one, a hook noticing an owed one, and
    the process that eventually performs it. A `memvault enqueue` sitting beside `memvault
    ingest` would suggest two different things happen.
    """
    paths = StatePaths.for_vault(vault.name)

    if args.session_start:
        marker = mark_session_start(paths, args.session_start)
        print(_status("session", f"active: {marker}"))
        # The same call the enqueue makes. A lid closed mid-wait leaves an owed pass and no
        # process, and the first session afterwards is the earliest anything notices.
        pid = ensure_runner(config, vault, paths, log_path=args.log_path)
        if pid is not None:
            print(_status("runner", f"re-spawned for a queued ingest (pid {pid})"))
        return 0

    if args.enqueue:
        if args.session:
            clear_session(paths, args.session)
        marker = enqueue(paths)
        pid = ensure_runner(config, vault, paths, log_path=args.log_path)
        print(_status("queued", str(marker)))
        if pid is not None:
            print(_status("runner", f"waiting for quiet (pid {pid})"))
        return 0

    if args.debounce_runner:
        # Nothing has configured logging in a detached process, and its stdout goes to a file
        # nobody reads. The banner and every decision belong in the log or they are lost.
        configure_logging(args.log_path)
        return run_runner(config, vault, paths, log_path=args.log_path)

    return None


def ingest_command(config: Config, vault: VaultConfig, args: argparse.Namespace) -> int:
    """Drain the inbox into the vault and commit the result.

    Exits 0 when everything filed, 3 when some drops were held for review, and 1 when the pass
    could not run or its commit failed. The three are distinct so a scheduler can tell "a human
    should look at one drop" from "the pass is broken".
    """
    handled = _debounce_command(config, vault, args)
    if handled is not None:
        return handled

    report = run_ingest(config, vault, log_path=args.log_path)
    _print_ingest(report)
    if not report.refused:
        # This pass is the safety net for whatever the debounced one was owed (R7): it just did
        # the work the queue marker was asking for, so the request is answered. A refusal
        # answers nothing, and clearing the marker there would silently cancel the retry.
        clear_queue(StatePaths.for_vault(vault.name))
    if report.status is IngestStatus.PARTIAL:
        print(
            "some items stayed in the inbox — read the .needs-review marker beside each",
            file=sys.stderr,
        )
    return report.exit_code


def _parse_day(value: str | None, flag: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{flag} must be a YYYY-MM-DD date, got {value!r}") from exc


def passage_embedder(index_config: IndexConfig) -> Embedder:
    """Embedder for indexing. E5 is asymmetric — passages and queries carry different prefixes,
    so this must not be the one `recall` uses.

    An install without the `[semantic]` extra gets `NullEmbedder`, which indexes every chunk for
    keyword search and stores no vectors (R12). Indexing must not be the thing that fails on a
    light install: without an index there is no search at all, not merely no semantic search.
    """
    if index_config.backend == "fastembed" and not semantic_available():
        return NullEmbedder()
    return FastEmbedEmbedder(index_config.model)


def index_command(config: Config, vault: VaultConfig) -> int:
    """Bring the search index in line with the vault's files.

    The first run downloads the embedding model (~130MB) and looks like a hang, so say what is
    about to happen before it starts rather than after it finishes.
    """
    db_path = config.index.db_path(vault.name)
    print(f"indexing '{vault.name}' → {db_path}")
    embedder = passage_embedder(config.index)
    if embedder.model_id == NULL_MODEL_ID:
        # Said before the pass, not after: the index this builds is keyword-only, and finding
        # that out from a later empty-feeling search is how an adopter concludes the tool is bad
        # rather than that the extra is missing.
        print(_status("note", SEMANTIC_EXTRA_HINT))
    elif not db_path.exists():
        print("first run: the embedding model downloads once (~130MB); this is not a hang")

    try:
        stats = build_index(vault, config.index, embedder, db_path=db_path)
    except IndexBuildError as exc:
        print(f"index error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{stats.files_scanned} file(s) scanned — {stats.files_indexed} indexed, "
        f"{stats.files_unchanged} unchanged, {stats.files_removed} removed"
    )
    print(
        f"{stats.chunks_written} chunk(s) — {stats.chunks_embedded} embedded, "
        f"{stats.chunks_reused} reused, in {stats.elapsed_seconds:.1f}s"
    )
    # The graph line is what makes the index's disposability checkable by hand: rebuild from
    # scratch and these counts must land on the same numbers.
    print(
        f"{stats.links_written} link(s) and {stats.observations_written} observation(s) "
        f"written — {stats.links_pending} link(s) pending across the vault"
    )
    for relative, reason in stats.skipped:
        print(_status("skipped", f"{relative} — {reason}"))
    return 0


def recall_command(config: Config, vault: VaultConfig, args: argparse.Namespace) -> int:
    """Run a hybrid search and print cited paths.

    No match exits 0: an honest "nothing in the vault about this" is an answer, and a skill
    that reads a non-zero status as breakage would report a failure instead of relaying it.
    """
    try:
        since = _parse_day(args.since, "--since")
        until = _parse_day(args.until, "--until")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        response = run_recall(
            vault,
            config.index,
            query_embedder(config.index),
            args.query,
            limit=args.limit,
            sources=tuple(args.source or ()),
            since=since,
            until=until,
            min_similarity=(
                config.index.min_similarity if args.min_similarity is None else args.min_similarity
            ),
            ranking=config.ranking,
        )
    except (RecallError, IndexBuildError) as exc:
        print(f"recall error: {exc}", file=sys.stderr)
        return 1

    print(format_results(response))
    return 0


def _print_reflect(report: ReflectReport) -> None:
    """Print the pass as status lines: what it wrote, then what it could not read."""
    for item in report.appended:
        print(_status("appended", f"{item.relative_path} — {item.heading.lstrip('# ').strip()}"))
    for item in report.duplicates:
        print(_status("present", f"{item.relative_path} — already recorded"))
    for project in report.created_projects:
        print(_status("created", f"memory directory for {project}"))
    if report.digest_path:
        print(_status("digest", report.digest_path))
    for reference in report.commits:
        print(_status("commit", f"{reference.vault} — {reference.short_sha}"))
    for project in report.unknown_projects:
        print(_status("dropped", f"{project} — not a project this vault claims"))
    if report.skipped_projects:
        print(_status("skipped", "outside this vault: " + ", ".join(report.skipped_projects)))
    for warning in report.warnings:
        print(_status("degraded", warning), file=sys.stderr)

    if report.failure is not None:
        print(_status("failed", report.failure), file=sys.stderr)
    print(report.summary_line())


def reflect_command(config: Config, vault: VaultConfig, args: argparse.Namespace) -> int:
    """Curate the window's material into durable memory and a digest, then commit.

    Exits 0 on a clean pass, 3 when it ran but on less material than it should have — a
    claude-mem database that could not be read — and 1 when it could not run at all. The three
    match `ingest` so one scheduler can read both.
    """
    try:
        since = _parse_day(args.since, "--since")
        until = _parse_day(args.until, "--until")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        report = run_reflect(
            config,
            vault,
            since=since,
            until=until,
            days=args.days,
            db_path=args.claude_mem_db,
            log_path=args.log_path,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_reflect(report)
    return report.exit_code


def _ask(question: str) -> bool:
    """Ask on a terminal; decline anywhere else.

    A scheduler, a hook, and a CI job all reach this with no terminal attached, and a prompt
    there would either hang forever or read a line of somebody else's input. Declining is the
    answer that changes nothing, and `--git` is how a non-interactive caller says yes.
    """
    if not sys.stdin.isatty():
        return False
    try:
        answer = input(f"{question} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _git_approval(args: argparse.Namespace) -> Confirm:
    """How `init` answers its one question: the flags first, then the terminal."""
    if args.git:
        return lambda _question: True
    if args.no_git:
        return lambda _question: False
    return _ask


def _print_init(report: InitReport) -> None:
    """Print what init did, then how to reach the vault from each harness."""
    for relative in report.created:
        print(_status("created", relative))
    for relative in report.kept:
        print(_status("kept", relative))
    if report.git_initialized:
        print(_status("git", f"initialized a repository at {report.root}"))
    print(
        _status(
            "config",
            f"{'wrote' if report.config_written else 'already current'} {report.config_path}",
        )
    )
    for note in report.notes:
        print(_status("note", note))

    print()
    print(f"Vault {report.vault_name!r} at {report.root}. Point your tools at it:")
    print()
    for line in wiring_lines(report.root):
        print(line)
    print()
    print("Then: `memvault doctor`, drop a file in inbox/ and `memvault ingest`.")


def init_command(args: argparse.Namespace) -> int:
    """Scaffold or adopt a vault. The one subcommand that runs without a config."""
    try:
        report = init_vault(
            args.path,
            config_path=None if args.config is None else Path(args.config),
            vault_name=args.name,
            adopt=True if args.adopt else None,
            force=args.force,
            confirm=_git_approval(args),
        )
    except InitError as exc:
        print(f"init error: {exc}", file=sys.stderr)
        return 1

    _print_init(report)
    print(report.summary_line())
    return 0


def mcp_command(config: Config, args: argparse.Namespace) -> int:
    """Serve this vault to any harness that speaks MCP, over stdio, until the client hangs up.

    The server module is imported here rather than at the top of the file for two reasons. It
    pulls in the MCP SDK — pydantic, starlette, a web server — which `recall` and `ingest` have
    no use for and should not wait on. And it imports this module for the health core behind
    `vault_status`, so a top-level import in both directions would be a cycle.
    """
    from memvault.mcp_server import MCPStartupError, serve

    try:
        serve(config, directory=args.path, requested_vault=args.vault)
    except MCPStartupError as exc:
        print(f"mcp error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memvault",
        description="File-first memory engine: inbox ingestion, hybrid retrieval, reflection.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "path to a config file — the one `init` writes, the one every other command reads. "
            "Without it, the search order is $MEMVAULT_CONFIG, ~/.config/memvault/config.yaml, "
            "~/.memvault/config.yaml, then ./memvault.config.yaml"
        ),
    )
    parser.add_argument("--vault", help="which configured vault to act on (default: default_vault)")

    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser(
        "init",
        help="scaffold or adopt a vault and write its config",
        description=(
            "Create a vault at PATH — skeleton, starter docs, git repo — or adopt a directory "
            "that already holds notes, touching nothing that is already there. Writes the config "
            "to ~/.config/memvault/config.yaml unless --config names another path."
        ),
    )
    init_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        metavar="PATH",
        help="the vault directory (default: the current one)",
    )
    init_parser.add_argument(
        "--name",
        metavar="NAME",
        help="what to call this vault in the config (default: the directory's name)",
    )
    init_parser.add_argument(
        "--adopt",
        action="store_true",
        help=(
            "treat the directory as somebody's existing notes: create only the directories the "
            "engine writes into, and modify nothing that already exists. Inferred when the "
            "directory already holds material of its own"
        ),
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing config and starter files instead of refusing",
    )
    git_group = init_parser.add_mutually_exclusive_group()
    git_group.add_argument(
        "--git",
        action="store_true",
        help="run `git init` without asking (a vault must be a git repository)",
    )
    git_group.add_argument(
        "--no-git",
        action="store_true",
        help="never run `git init`; refuse instead if the directory is not a repository",
    )
    sub.add_parser("doctor", help="report vault health; exits non-zero on problems")
    mcp_parser = sub.add_parser(
        "mcp",
        help="serve this vault over stdio MCP (recall, write_inbox, write_note, vault_status)",
        description=(
            "Run the vault's MCP server on stdin/stdout, for Codex, opencode, Cursor, Claude "
            "Desktop, or anything else that speaks MCP. PATH names which vault to serve and is "
            "checked against the config: a directory that is no configured vault's root is "
            "refused at startup rather than serving a different vault than the caller meant."
        ),
    )
    mcp_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="PATH",
        help="the vault directory to serve (default: --vault, else the config's default_vault)",
    )
    ingest_parser = sub.add_parser(
        "ingest",
        help="drain the inbox into the vault",
        description=(
            "Run the ingestion pass now, or take part in the idle-triggered one. Without a flag "
            "the pass runs immediately and answers whatever a queued run was owed — which is how "
            "the nightly safety net services a request whose runner did not survive."
        ),
    )
    ingest_parser.add_argument(
        "--log-path",
        default=DEFAULT_LOG_PATH,
        metavar="PATH",
        help=f"log named in the failure notification (default: {DEFAULT_LOG_PATH})",
    )
    debounce_group = ingest_parser.add_mutually_exclusive_group()
    debounce_group.add_argument(
        "--session-start",
        metavar="ID",
        help=(
            "mark a session active so a debounced ingest waits for it, and re-spawn a runner if "
            "one is owed. For a SessionStart hook"
        ),
    )
    debounce_group.add_argument(
        "--enqueue",
        action="store_true",
        help=(
            "ask for an ingest a few quiet minutes from now instead of running one, and start "
            "the waiting runner if none is alive. For a SessionEnd hook"
        ),
    )
    debounce_group.add_argument(
        "--debounce-runner",
        action="store_true",
        help="wait for quiescence and then ingest; the process --enqueue spawns",
    )
    ingest_parser.add_argument(
        "--session",
        metavar="ID",
        help="with --enqueue: the session that just ended, whose active marker is cleared",
    )
    sub.add_parser("index", help="build or update the search index")
    recall_parser = sub.add_parser("recall", help="hybrid search across the vault")
    recall_parser.add_argument("query", help="what to search for")
    recall_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help=f"how many passages to cite (default: {DEFAULT_LIMIT})",
    )
    recall_parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME",
        help="keep only files whose declared source or vault section matches (repeatable)",
    )
    recall_parser.add_argument(
        "--since", metavar="YYYY-MM-DD", help="keep only files dated on or after this day"
    )
    recall_parser.add_argument(
        "--until", metavar="YYYY-MM-DD", help="keep only files dated on or before this day"
    )
    recall_parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        metavar="X",
        help=(
            "absolute cosine floor for semantic hits, on top of the corpus median "
            "(default: index.min_similarity from the config)"
        ),
    )
    reflect_parser = sub.add_parser("reflect", help="run the curation pass over recent material")
    reflect_parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        metavar="N",
        help=f"how many days back the window reaches (default: {DEFAULT_WINDOW_DAYS})",
    )
    reflect_parser.add_argument(
        "--since", metavar="YYYY-MM-DD", help="first day of the window (overrides --days)"
    )
    reflect_parser.add_argument(
        "--until", metavar="YYYY-MM-DD", help="last day of the window (default: today)"
    )
    reflect_parser.add_argument(
        "--claude-mem-db",
        metavar="PATH",
        help="claude-mem database to read (default: ~/.claude-mem/claude-mem.db)",
    )
    reflect_parser.add_argument(
        "--log-path",
        default=DEFAULT_REFLECT_LOG_PATH,
        metavar="PATH",
        help=f"log named in the failure notification (default: {DEFAULT_REFLECT_LOG_PATH})",
    )

    return parser


#: Subcommands dispatched before the config is loaded, keyed to the function that runs them.
#: `init`'s whole starting state is "there is no config", so a load first would make the one
#: command that fixes that unreachable exactly when it is needed.
CONFIGLESS_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {"init": init_command}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    configless = CONFIGLESS_COMMANDS.get(args.command)
    if configless is not None:
        return configless(args)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    # `mcp` picks its own vault: the directory the harness was pointed at decides, and the whole
    # point of the reconciliation is to fail on a mismatch rather than to quietly serve whatever
    # `default_vault` happens to name.
    if args.command == "mcp":
        return mcp_command(config, args)

    try:
        vault = config.vault(args.vault)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    if args.command == "doctor":
        return doctor(config, vault)
    if args.command == "ingest":
        return ingest_command(config, vault, args)
    if args.command == "index":
        return index_command(config, vault)
    if args.command == "recall":
        return recall_command(config, vault, args)
    if args.command == "reflect":
        return reflect_command(config, vault, args)

    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
