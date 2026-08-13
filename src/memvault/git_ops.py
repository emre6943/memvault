"""Git operations for the ingestion pass.

Every automated change to memory is a commit, because the diff is the only review surface this
system has (R7). That makes git a load-bearing dependency rather than a convenience, and this
module exists so the rules about how it is used live in one readable place instead of being
spread across `subprocess.run` calls in the orchestrator.

Three rules, and each one is here because its opposite loses something.

**A pathspec is always named.** Staging is `git add -A -- <paths>` and committing is
`git commit -m ... -- <paths>`, never a bare `git add .`. The pass knows exactly which files it
touched, so a commit that contained anything else would be a commit whose message lies. The
`--` form also makes the commit a partial one, so anything a human happened to stage before the
pass ran stays staged rather than being swept in.

**A pathspec that matches nothing is dropped rather than passed.** `git add -A -- ghost.txt`
exits 128 when `ghost.txt` is neither on disk nor tracked, which is exactly the shape of a drop
that arrived uncommitted and was drained in the same pass. Filtering to paths git can actually
see turns that from a crash into a no-op.

**Nothing is pushed.** Pushing stays with the vault's own backup job, which is where
`local_only` handling already lives (KTD7); a second place that pushes is a second place that
can push the wrong thing.

Failure is an exception, never a return code a caller might forget to read: a git command that
did not do what it was asked must not be mistaken for one that did.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Untracked files are reported individually rather than by directory. A vault write into a
#: brand-new `transcripts/2026/08/` would otherwise show up as one directory entry, and the
#: cleanliness check would have no way to tell whose directory it is.
_STATUS_ARGS = ("status", "--porcelain", "-z", "--untracked-files=all")

#: The only status code that means "nothing to see here". Untracked (`?`) counts as a change:
#: every file this pass writes is untracked at the moment it lands, so a cleanliness check that
#: forgave untracked files would forgive exactly the state it exists to detect.
_UNCHANGED = frozenset({" "})


class GitError(Exception):
    """Raised when a git command fails, or when a repository is not in a usable state."""


class DirtyVaultError(GitError):
    """Raised when a vault carries uncommitted changes the pass did not make.

    Its own type because the ingestion pass treats it differently from a failed command: it is
    a refusal to start, raised before anything is written, rather than a failure part-way
    through.
    """


@dataclass(frozen=True)
class StatusEntry:
    """One line of `git status --porcelain`, split into its parts."""

    index_status: str
    worktree_status: str
    path: str

    @property
    def changed(self) -> bool:
        return not (self.index_status in _UNCHANGED and self.worktree_status in _UNCHANGED)

    def describe(self) -> str:
        return f"{self.index_status}{self.worktree_status} {self.path}"


@dataclass(frozen=True)
class CommitRef:
    """One commit this pass created, named by the vault it landed in."""

    vault: str
    sha: str

    @property
    def short_sha(self) -> str:
        return self.sha[:7]


def _flatten(value: object) -> str:
    """One-line rendering, so a multi-line git error still fits in a notification."""
    return " ".join(str(value).split())


def run_git(root: Path, *args: str, check: bool = True, stdin: str | None = None) -> str:
    """Run one git command in `root` and return its stdout.

    No shell, and no `env=`, so the child inherits this process's environment — a vault whose
    identity comes from a repository-local `user.email` and one whose identity comes from the
    global config both work without this module knowing which.

    `stdin` exists for the one command that needs it: `check-ignore --stdin`, which is the only
    way to hand git a path list it will not choke on.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"{root}: could not run git: {_flatten(exc)}") from exc

    if check and completed.returncode != 0:
        detail = _flatten(completed.stderr) or _flatten(completed.stdout) or "no error output"
        raise GitError(f"{root}: `git {' '.join(args)}` exited {completed.returncode}: {detail}")
    return completed.stdout


def parse_status(payload: str) -> tuple[StatusEntry, ...]:
    """Parse NUL-separated porcelain output.

    Renames and copies emit two records — the new path, then the old one — so the second is
    consumed rather than read as an entry of its own. NUL separation is what makes this safe
    for a vault holding filenames with spaces, quotes, or newlines in them, none of which the
    default quoted format survives.
    """
    tokens = payload.split("\0")
    entries: list[StatusEntry] = []

    position = 0
    while position < len(tokens):
        token = tokens[position]
        position += 1
        if len(token) < 4:
            continue
        index_status, worktree_status, path = token[0], token[1], token[3:]
        if index_status in ("R", "C"):
            position += 1
        entries.append(StatusEntry(index_status, worktree_status, path))

    return tuple(entries)


def status(root: Path) -> tuple[StatusEntry, ...]:
    """Every path git considers changed in this repository."""
    return parse_status(run_git(root, *_STATUS_ARGS))


def _under(path: str, prefix: str) -> bool:
    cleaned = prefix.strip("/")
    return bool(cleaned) and (path == cleaned or path.startswith(f"{cleaned}/"))


def template_root(template: str) -> str:
    """The top-level directory a path template writes into.

    Derived from the template rather than configured separately, so a guarded prefix cannot drift
    from where the pass actually writes. Deliberately coarse: `repos/{project}` and
    `repos/Vault/digest/{yyyy}-W{ww}.md` both yield `repos`, which is exactly what a pass writing
    through both needs guarded.

    A template whose first segment is itself a placeholder has no static root and yields "".
    `_under` matches nothing against an empty prefix, so a guard scoped to it considers no dirt
    rather than guessing at one.
    """
    head = template.split("/", 1)[0]
    return head if head and "{" not in head else ""


def dirty_paths(
    root: Path, *, ignore: Sequence[str] = (), limit_to: Sequence[str] | None = None
) -> tuple[str, ...]:
    """Repository-relative paths with uncommitted changes.

    `ignore` drops paths the pass does not own. The inbox always qualifies — an inbox holding an
    uncommitted drop is the ordinary manual case, and refusing to run over it would make the
    documented minimum drop unusable — as do any prefixes the vault declares in `ignore_dirty`.

    `limit_to` inverts the question: consider *only* dirt under these prefixes. That is what a
    route-target vault needs. Its commit contains one distilled note, staged by exact path, so
    unrelated work sitting uncommitted in that repo cannot ride along — and it is somebody's
    active working tree, which will rarely be clean. Guarding the whole repo there would refuse
    the pass for reasons that have nothing to do with it.
    """
    entries = (entry.path for entry in status(root) if entry.changed)
    if limit_to is not None:
        entries = (path for path in entries if any(_under(path, prefix) for prefix in limit_to))
    return tuple(path for path in entries if not any(_under(path, prefix) for prefix in ignore))


def ensure_clean(
    root: Path,
    *,
    label: str,
    ignore: Sequence[str] = (),
    limit_to: Sequence[str] | None = None,
) -> None:
    """Refuse to proceed when a vault carries changes the pass did not make.

    Raised before the first vault write, never after: the point is that the pass's commit
    contains the pass's work and nothing else, and that is only achievable up front.
    """
    dirty = dirty_paths(root, ignore=ignore, limit_to=limit_to)
    if not dirty:
        return

    shown = ", ".join(dirty[:5]) + (f", and {len(dirty) - 5} more" if len(dirty) > 5 else "")
    scope = f" (guarding {', '.join(p for p in limit_to if p)})" if limit_to else ""
    raise DirtyVaultError(
        f"vault {label!r} at {root} has uncommitted changes{scope}: {shown}. Commit or stash "
        "them first — the ingestion pass will not fold unrelated work into its own commit."
    )


def tracked(root: Path, paths: Sequence[str]) -> set[str]:
    """Which of these repository-relative paths git already tracks."""
    if not paths:
        return set()
    payload = run_git(root, "ls-files", "-z", "--", *paths)
    return {entry for entry in payload.split("\0") if entry}


def ignored(root: Path, paths: Sequence[str]) -> set[str]:
    """Which of these paths the repository's own ignore rules exclude.

    This is how `local_only` stays enforced by `.gitignore` rather than by discipline (KTD7).
    A file filed under the vault's gitignored prefix must be written and must not be staged, and
    `git add` on one exits 1 with a hint rather than skipping it — so the filtering happens here
    instead of turning one private drop into a failed pass.

    A tracked path is never reported: `check-ignore` consults the index first, so a file already
    in git stays committable even if a later ignore rule would have covered it.
    """
    if not paths:
        return set()
    payload = run_git(root, "check-ignore", "-z", "--stdin", stdin="\0".join(paths), check=False)
    return {entry for entry in payload.split("\0") if entry}


def stageable(root: Path, paths: Iterable[str]) -> tuple[str, ...]:
    """The subset of `paths` git can be asked to stage without erroring.

    A path that is neither on disk nor tracked matches no pathspec, and passing it to
    `git add` aborts the whole invocation. That is not a hypothetical: an uncommitted drop that
    the pass files and then deletes is exactly this case, and the whole commit would fail on it.
    Ignored paths are dropped for the same reason and a better one — they are meant to stay out.
    """
    unique = list(dict.fromkeys(paths))
    if not unique:
        return ()
    known = tracked(root, unique)
    excluded = ignored(root, unique)
    return tuple(
        path
        for path in unique
        if path not in excluded and ((root / path).exists() or path in known)
    )


def stage(root: Path, paths: Iterable[str]) -> tuple[str, ...]:
    """Stage exactly these paths, additions and deletions alike. Returns what was staged."""
    selected = stageable(root, paths)
    if selected:
        run_git(root, "add", "-A", "--", *selected)
    return selected


def has_staged_changes(root: Path, paths: Sequence[str]) -> bool:
    """Whether the index differs from HEAD for these paths.

    `git diff --cached` exits 1 when there is a difference, which is the answer rather than a
    failure — hence `check=False` and an explicit read of the return code.
    """
    if not paths:
        return False
    try:
        subprocess.run(
            ["git", "-C", str(root), "diff", "--cached", "--quiet", "--", *paths],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return True
    except OSError as exc:
        raise GitError(f"{root}: could not run git: {_flatten(exc)}") from exc
    return False


def head_sha(root: Path) -> str | None:
    """The current commit, or None on an unborn branch."""
    try:
        return run_git(root, "rev-parse", "HEAD").strip() or None
    except GitError:
        return None


def commit(root: Path, message: str, paths: Sequence[str], *, vault: str) -> CommitRef | None:
    """Commit exactly these paths, or return None when they hold nothing to commit.

    Nothing to commit is an ordinary outcome, not an error: a pass that only refreshed a review
    marker whose text did not change has done its work and owes the vault no commit.

    `paths` is taken as given rather than re-filtered through `stageable`. Staging a deletion
    removes the path from the index, so a second `stageable` pass would find it neither on disk
    nor tracked and quietly drop the one change the pass most needs to record. The index is the
    authority by this point, and `has_staged_changes` is what reads it.
    """
    selected = list(dict.fromkeys(paths))
    if not selected or not has_staged_changes(root, selected):
        logger.debug("vault %r: nothing staged to commit in %s", vault, root)
        return None

    run_git(root, "commit", "-q", "-m", message, "--", *selected)
    sha = head_sha(root)
    if sha is None:
        raise GitError(f"{root}: commit reported success but HEAD does not resolve")

    logger.info("vault %r: committed %s (%d path(s))", vault, sha[:7], len(selected))
    return CommitRef(vault=vault, sha=sha)
