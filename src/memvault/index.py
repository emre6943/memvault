"""Search index: vault Markdown chunked into FTS5 rows and packed embedding blobs.

Why this shape: the index is a cache, never a second source of truth (R17). Deleting the
database and rebuilding it from the files must reproduce equivalent rows, so nothing here
invents content — every chunk is a deterministic function of a file's bytes and the
configured chunk geometry. Nothing is ever stored here that is not recoverable from a file.

Why incremental: a vault grows monotonically, and re-embedding all of it on every pass makes
indexing something you avoid running. Files are skipped by SHA-256 over their text, and
inside a changed file each chunk reuses an existing embedding when its own hash is unchanged,
so editing one paragraph costs one embedding rather than a file's worth (R16).

Why the embedder is a Protocol: the real backend downloads a ~130MB ONNX model on first use.
Tests inject a deterministic stub instead, which keeps the suite fast and runnable offline —
and it leaves the OpenAI backend of KTD2 a matter of passing a different object.

Why brute force stays (KTD3): embeddings are stored as packed float32 blobs and read back as
one NumPy matrix. An exact matrix multiply beats an approximate index until the corpus is far
larger than this one; `build_index` returns the wall-clock measurement that decides when.

Why an embedding is optional (R12): `fastembed` lives behind the `[semantic]` extra, so a bare
install has no way to produce a vector at all. Rather than refusing to index — which would leave
that install with no index, and therefore no keyword search either — `NullEmbedder` writes chunks
with a NULL embedding. The FTS triggers fire on the insert either way, so keyword recall works
exactly as before, and the absence is a fact recall reads out of the database rather than a
capability it has to be told about.

Why the graph tables live here and not in a graph database (R1): edges, typed observations, and
the ranking facts in frontmatter are all parsed out of the files by `graph.py` on the same pass
that chunks them, so they inherit the same disposability — deleting this database and rebuilding
it reproduces the graph exactly. Resolution of a wikilink to a path is the one part that is not a
property of a single file, so it is recomputed over the whole `links` table at the end of every
run rather than at parse time: a note may point at a file that gets written a week later, and an
incremental index that never revisited the question would diverge from a fresh rebuild.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import re
import sqlite3
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from memvault import graph
from memvault.config import IndexConfig, VaultConfig

#: Bumped to "2" for the graph tables (R1). A changed version wipes and rebuilds rather than
#: migrating: chunk-hash embedding reuse makes the rebuild cost nothing but I/O, and an ALTER
#: path would be a second definition of the schema that only the upgrade route ever exercises.
SCHEMA_VERSION = "2"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"

#: The `model_id` an index built without embeddings records. A real id, not an empty string, so
#: the stored value still answers "which vector space is in here" honestly — and so switching the
#: extra on or off changes the id and takes the ordinary wipe-and-rebuild road that any other
#: model change takes. Nothing may reuse a vector across that boundary, in either direction.
NULL_MODEL_ID = "null"

#: What to say to someone whose install cannot embed. Named here rather than in `recall` because
#: both the indexer and the query path have to say the same thing, and because the fact — that
#: `fastembed` is an extra — belongs to the module that owns the backend.
SEMANTIC_EXTRA_HINT = (
    "semantic search is off: this install has no `fastembed`. Add it with "
    '`uv tool install "memvault-cli[semantic]"` (or `uv sync --extra semantic`) and re-run '
    "`memvault index`. Keyword search works without it."
)

# E5 models are trained with asymmetric prefixes: stored text is a passage, a search string is
# a query. Both sides must agree, so the strings live here rather than in each caller.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PARAM_BATCH = 400

#: A temp table holding `(content_hash, embedding)` pairs rescued from `chunks` immediately
#: before a schema wipe empties it. It exists only for the length of one connection, which is
#: exactly the length of one `build_index` run — the snapshot is a bridge across the wipe, not a
#: second store of vectors that could drift out of step with the real one.
_EMBEDDING_SNAPSHOT = "embedding_snapshot"

#: The tables a wipe has to clear. Every one of them is keyed by vault-relative path and rebuilt
#: from the files, so forgetting one leaves rows describing a corpus that no longer exists.
_PER_PATH_TABLES = ("chunks", "links", "observations", "file_meta", "files")


class IndexBuildError(Exception):
    """Raised when the index cannot be built or read — a bad embedder or an unusable database."""


class Embedder(Protocol):
    """What the indexer needs from an embedding backend.

    `model_id` is identity, not decoration: embeddings from two different models are not
    comparable, so a changed id invalidates the whole index rather than quietly mixing vector
    spaces that a cosine score would then rank against each other.
    """

    model_id: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""


@dataclass(frozen=True)
class HubModel:
    """How to register a Hub model that fastembed's built-in catalog does not carry."""

    dim: int
    pooling: str = "MEAN"
    normalize: bool = True


# fastembed 0.8 ships no entry for `multilingual-e5-small`, though its ONNX weights sit on the
# Hub. Registering it keeps the model KTD2 actually chose, rather than silently substituting a
# different one and leaving a stored vector space nobody can account for.
HUB_MODELS: dict[str, HubModel] = {
    "intfloat/multilingual-e5-small": HubModel(dim=384),
    "intfloat/multilingual-e5-base": HubModel(dim=768),
}


class FastEmbedEmbedder:
    """Local ONNX embeddings via `fastembed` — the default backend (KTD2).

    Local because sending vault text to a third-party API contradicts the privacy posture the
    rest of the engine enforces, and multilingual because this vault holds Turkish; an
    English-only model would rank Turkish content as noise.

    The model is loaded lazily so that constructing an engine, running `doctor`, or importing
    this module never triggers the first-run model download (~130MB).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, *, prefix: str = PASSAGE_PREFIX) -> None:
        self.model_id = model_id
        self.prefix = prefix
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on install shape
            raise IndexBuildError(
                "the 'fastembed' backend requires the fastembed package. "
                "Install it with `uv sync`, or configure index.backend explicitly."
            ) from exc

        try:
            self._model = TextEmbedding(model_name=self.model_id)
        except ValueError:
            self._register(TextEmbedding)
            self._model = TextEmbedding(model_name=self.model_id)
        return self._model

    def _register(self, text_embedding: Any) -> None:
        spec = HUB_MODELS.get(self.model_id)
        if spec is None:
            raise IndexBuildError(
                f"{self.model_id!r} is not in fastembed's catalog and has no registration entry "
                "in memvault.index.HUB_MODELS. Pick a model from "
                "`TextEmbedding.list_supported_models()` or add an entry naming its dimension "
                "and pooling."
            )
        from fastembed.common.model_description import ModelSource, PoolingType

        text_embedding.add_custom_model(
            model=self.model_id,
            pooling=PoolingType[spec.pooling],
            normalization=spec.normalize,
            sources=ModelSource(hf=self.model_id),
            dim=spec.dim,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        prefixed = [f"{self.prefix}{text}" for text in texts]
        return [[float(value) for value in vector] for vector in model.embed(prefixed)]


class NullEmbedder:
    """The backend for an install without the `[semantic]` extra: it embeds nothing, loudly.

    It returns an empty vector per text, which `build_index` writes as a NULL blob. That is the
    whole mechanism — the column is already nullable and the FTS triggers fire on the chunk
    insert regardless, so a null-embedded index is a complete keyword index and an incomplete
    semantic one, which is exactly what a bare install has.

    Refusing to index at all was the alternative and is worse: it would leave the light install
    with no index, so `recall` would fail on a missing database rather than answer with the half
    it can honestly serve.
    """

    model_id = NULL_MODEL_ID

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[] for _ in texts]


def semantic_available() -> bool:
    """Whether this install can embed at all — i.e. whether the `[semantic]` extra is present.

    Asked with `find_spec` rather than by importing: importing `fastembed` costs onnxruntime's
    load time, and the answer is needed on every `index` and `recall` invocation, including the
    ones that never reach a model.
    """
    return importlib.util.find_spec("fastembed") is not None


@dataclass(frozen=True)
class Chunk:
    """One indexable unit: a Markdown section, or a slice of one that was too long."""

    ordinal: int
    heading: str
    text: str

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded and hashed.

        The heading breadcrumb is prepended so a chunk carries its ancestry: a section titled
        `## Costs` means something different under `# Erfpacht` than under `# Mortgage`, and
        the section body alone does not say which.
        """
        return f"{self.heading}\n\n{self.text}" if self.heading else self.text

    @property
    def content_hash(self) -> str:
        return content_hash(self.embedding_text)


@dataclass(frozen=True)
class ChunkRow:
    """A stored chunk, as recall reads it back."""

    id: int
    path: str
    ordinal: int
    heading: str
    text: str


@dataclass(frozen=True)
class IndexStats:
    """What one indexing pass did. `elapsed_seconds` is KTD3's revisit measurement."""

    db_path: Path
    files_scanned: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    chunks_written: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    #: The graph side of the same pass. The two `written` counts are per-pass, like the chunk
    #: counts beside them; `links_pending` is corpus-wide, because whether a link resolves is a
    #: property of the whole vault and is recomputed for every stored row on every pass. Pending
    #: is a normal state — the target may simply not be written yet — not an error count.
    links_written: int = 0
    observations_written: int = 0
    links_pending: int = 0
    elapsed_seconds: float = 0.0
    skipped: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_embedding(values: Sequence[float]) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def unpack_embedding(blob: bytes) -> npt.NDArray[np.float32]:
    return np.frombuffer(blob, dtype=np.float32)


def strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block.

    Frontmatter is metadata *about* the text, not text a human wrote, and indexing it puts
    keys like `classification` and `content_id` into keyword results where they read as noise.
    An unterminated block is left alone: it is more likely a document that starts with a rule
    than a truncated header, and dropping the whole file would be the worse mistake.
    """
    if not text.startswith("---"):
        return text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text
    for position in range(1, len(lines)):
        if lines[position].strip() in {"---", "..."}:
            return "".join(lines[position + 1 :])
    return text


def _sections(body: str) -> list[tuple[str, str]]:
    """Split Markdown at ATX headings, returning (heading breadcrumb, section text).

    Headings inside fenced code blocks are not headings — a `# comment` line in a shell
    snippet would otherwise shred a section into fragments that read as nonsense.
    """
    stack: list[tuple[int, str]] = []
    heading = ""
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []
    fence: str | None = None

    for line in body.splitlines():
        stripped = line.strip()
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            buffer.append(line)
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            buffer.append(line)
            continue

        match = _HEADING_RE.match(line)
        if match is None:
            buffer.append(line)
            continue

        if buffer:
            sections.append((heading, "\n".join(buffer)))
        buffer = [line]
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))
        heading = " > ".join(title for _, title in stack)

    if buffer:
        sections.append((heading, "\n".join(buffer)))

    return [(head, text) for head, text in sections if text.strip()]


def _split_by_size(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Window an over-long section, overlapping so a sentence split across the seam survives.

    Without overlap, the one paragraph that answers a query is exactly the one most likely to
    straddle a boundary and match neither piece.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            floor = start + overlap_chars + 1
            boundary = text.rfind("\n", floor, end)
            if boundary == -1:
                boundary = text.rfind(" ", floor, end)
            if boundary != -1:
                end = boundary + 1
        piece = text[start:end].strip("\n")
        if piece.strip():
            pieces.append(piece)
        if end >= length:
            break
        start = max(end - overlap_chars, start + 1)
    return pieces


def chunk_markdown(text: str, *, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """Chunk on heading boundaries first, size ceiling second.

    Heading-first because a retrieved chunk should be readable on its own, and a Markdown
    heading is the author's own statement of where one idea stops.
    """
    chunks: list[Chunk] = []
    for heading, section in _sections(strip_frontmatter(text)):
        for piece in _split_by_size(section.strip(), max_chars, overlap_chars):
            chunks.append(Chunk(ordinal=len(chunks), heading=heading, text=piece))
    return chunks


def escape_fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Raw user input reaching MATCH is a syntax error waiting to happen — a stray `"` or a
    leading `-` raises rather than returning no results. Every word becomes a quoted token,
    which is also what keeps Turkish words with apostrophes from being read as operators.
    """
    tokens = _WORD_RE.findall(text)
    return " ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def open_index(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the index database and ensure its schema."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            indexed_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            heading TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding BLOB,
            UNIQUE(path, ordinal)
        );
        CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
        CREATE INDEX IF NOT EXISTS chunks_hash_idx ON chunks(content_hash);
        """
    )
    # The graph side (R1). `target` is what the file says and `resolved_path` is what the corpus
    # currently makes of it — two columns rather than one, because a link that resolves later
    # must stay distinguishable from a link that was always resolved, and because resolution is
    # recomputed globally while the target text is parsed per file.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS file_meta (
            path TEXT PRIMARY KEY,
            title TEXT,
            kind TEXT,
            importance INTEGER,
            date TEXT,
            superseded_by TEXT
        );
        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            predicate TEXT NOT NULL,
            target TEXT NOT NULL,
            resolved_path TEXT,
            UNIQUE(path, ordinal)
        );
        CREATE INDEX IF NOT EXISTS links_path_idx ON links(path);
        CREATE INDEX IF NOT EXISTS links_resolved_idx ON links(resolved_path);
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            category TEXT NOT NULL,
            text TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            UNIQUE(path, ordinal)
        );
        CREATE INDEX IF NOT EXISTS observations_path_idx ON observations(path);
        CREATE INDEX IF NOT EXISTS observations_category_idx ON observations(category);
        """
    )
    # An external-content FTS5 table keeps one copy of the text. Triggers keep it in sync so
    # no write path can forget to, which is the failure that makes a search index lie.
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            heading, text,
            content='chunks', content_rowid='id',
            tokenize="unicode61 remove_diacritics 2"
        );
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, heading, text) VALUES (new.id, new.heading, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, heading, text)
            VALUES ('delete', old.id, old.heading, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, heading, text)
            VALUES ('delete', old.id, old.heading, old.text);
            INSERT INTO chunks_fts(rowid, heading, text) VALUES (new.id, new.heading, new.text);
        END;
        """
    )
    conn.commit()
    return conn


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def index_model(conn: sqlite3.Connection) -> str | None:
    """Which embedding model produced the vectors in this database, if any are stored."""
    return _meta_get(conn, "model_id")


def _snapshot_embeddings(conn: sqlite3.Connection) -> None:
    """Park every stored vector in a temp table so the wipe about to happen does not lose it.

    `_embeddings_by_hash` reads its reuse cache out of `chunks`, which a schema bump empties —
    so without this, bumping the version would re-embed the entire corpus to arrive back at the
    vectors it just deleted. A chunk's hash covers its text and heading and nothing else, so a
    vector keyed by it is as valid after the rebuild as before.
    """
    conn.execute(
        f"CREATE TEMP TABLE IF NOT EXISTS {_EMBEDDING_SNAPSHOT} ("
        "content_hash TEXT PRIMARY KEY, embedding BLOB NOT NULL)"
    )
    conn.execute(
        f"INSERT OR IGNORE INTO {_EMBEDDING_SNAPSHOT}(content_hash, embedding) "
        "SELECT content_hash, embedding FROM chunks WHERE embedding IS NOT NULL"
    )


def _has_snapshot(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM temp.sqlite_master WHERE type = 'table' AND name = ?",
        (_EMBEDDING_SNAPSHOT,),
    ).fetchone()
    return row is not None


def _reset_for_model(conn: sqlite3.Connection, model_id: str) -> None:
    """Wipe stored rows when the embedding model or the schema version changed.

    Vectors from two models occupy unrelated spaces; scoring a query from one against
    passages from the other returns confident nonsense. Rebuilding is cheap, so the index
    discards rather than mixes.

    A schema bump takes the same road (KTD "index migration = SCHEMA_VERSION bump") — but for a
    different reason, and with a different cost. The vectors are still valid there, so they are
    snapshotted first and the rebuild reuses them; a changed *model* invalidates them, and the
    snapshot is deliberately not taken so nothing can reuse what is no longer comparable.
    """
    stored = _meta_get(conn, "model_id")
    stored_schema = _meta_get(conn, "schema_version")
    if stored == model_id and stored_schema == SCHEMA_VERSION:
        return
    if stored is not None or stored_schema is not None:
        if stored == model_id:
            _snapshot_embeddings(conn)
        else:
            # The stored width described vectors that are about to stop existing. Leaving it
            # behind would let a later run compare a fresh vector against a dimension no row in
            # the database has any more — and an index built by `NullEmbedder` would keep
            # claiming a width while holding nothing at all.
            conn.execute("DELETE FROM meta WHERE key = ?", ("embedding_dim",))
        for table in _PER_PATH_TABLES:
            conn.execute(f"DELETE FROM {table}")
    _meta_set(conn, "model_id", model_id)
    _meta_set(conn, "schema_version", SCHEMA_VERSION)
    conn.commit()


def _matches(relative: str, pattern: str) -> bool:
    """Match a vault-relative path against a glob, anchored at any directory level.

    `node_modules/**` should exclude a nested `packages/web/node_modules/x` too — an exclude
    that only holds at the root is an exclude that silently stops working one directory down.
    """
    if fnmatch.fnmatchcase(relative, pattern):
        return True
    parts = relative.split("/")
    return any(fnmatch.fnmatchcase("/".join(parts[i:]), pattern) for i in range(1, len(parts)))


def _is_excluded(relative: str, patterns: Sequence[str]) -> bool:
    return any(_matches(relative, pattern) for pattern in patterns)


def iter_vault_files(
    root: Path, include: Sequence[str], exclude: Sequence[str]
) -> Iterator[tuple[Path, str]]:
    """Yield (absolute path, vault-relative posix path) for every file the config selects."""
    seen: set[Path] = set()
    for pattern in include:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if _is_excluded(relative, exclude):
                continue
            seen.add(path)
            yield path, relative


def _embeddings_by_hash(conn: sqlite3.Connection, hashes: Sequence[str]) -> dict[str, bytes]:
    """Look up already-computed vectors for these chunk hashes, from anywhere in the index.

    Scoped to the whole table rather than the file being reindexed: identical text moved
    between files is still identical text, and re-embedding it would be work with a known
    answer.

    The snapshot left behind by a schema wipe is consulted second, so a rebuild that follows a
    version bump costs no embeddings at all while an ordinary pass never pays for the extra
    lookup.
    """
    sources = ["chunks"]
    if _has_snapshot(conn):
        sources.append(_EMBEDDING_SNAPSHOT)

    found: dict[str, bytes] = {}
    unique = list(dict.fromkeys(hashes))
    for source in sources:
        for start in range(0, len(unique), _PARAM_BATCH):
            batch = unique[start : start + _PARAM_BATCH]
            # The only interpolated values are a table name from this module and a run of `?`;
            # every actual value stays a bound parameter.
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT content_hash, embedding FROM {source} "
                f"WHERE embedding IS NOT NULL AND content_hash IN ({placeholders})",
                batch,
            )
            for chunk_hash, blob in rows:
                found.setdefault(str(chunk_hash), bytes(blob))
    return found


def _delete_graph_rows(conn: sqlite3.Connection, relative: str) -> None:
    """Drop everything the graph knows about one path, mirroring the chunk delete beside it."""
    conn.execute("DELETE FROM links WHERE path = ?", (relative,))
    conn.execute("DELETE FROM observations WHERE path = ?", (relative,))
    conn.execute("DELETE FROM file_meta WHERE path = ?", (relative,))


def _write_graph_rows(conn: sqlite3.Connection, relative: str, parsed: graph.ParsedFile) -> None:
    """Replace one file's graph rows with what its bytes currently say.

    `resolved_path` is left NULL here on purpose. Resolution runs once over the whole table at
    the end of the pass, because the answer depends on files this loop may not have reached yet
    — and on files it will never reach, since an unchanged file is skipped entirely.

    A metadata row is written even when the file declared nothing, so the table mirrors `files`
    and a reader can tell "indexed, said nothing" from "not indexed".
    """
    _delete_graph_rows(conn, relative)
    conn.execute(
        "INSERT INTO file_meta(path, title, kind, importance, date, superseded_by) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            relative,
            parsed.metadata.title,
            parsed.metadata.kind,
            parsed.metadata.importance,
            parsed.metadata.date,
            parsed.metadata.superseded_by,
        ),
    )
    conn.executemany(
        "INSERT INTO links(path, ordinal, predicate, target, resolved_path) "
        "VALUES (?, ?, ?, ?, NULL)",
        [
            (relative, ordinal, link.predicate, link.target)
            for ordinal, link in enumerate(parsed.links)
        ],
    )
    conn.executemany(
        "INSERT INTO observations(path, ordinal, category, text, tags) VALUES (?, ?, ?, ?, ?)",
        [
            (relative, ordinal, observation.category, observation.text, " ".join(observation.tags))
            for ordinal, observation in enumerate(parsed.observations)
        ],
    )


def _resolve_links(conn: sqlite3.Connection) -> int:
    """Recompute every link's target against the current corpus. Returns the pending count.

    Run at the end of every pass, over every row, including rows nothing touched: a link is
    pending because of what the vault did *not* contain, so the only thing that can un-pend it is
    a later pass noticing that it now does. Recomputing the resolved ones too is what makes a
    deleted or newly-ambiguous target demote a link back to pending instead of leaving the index
    asserting an edge to a file that is gone.
    """
    entries = [
        (str(path), None if title is None else str(title))
        for path, title in conn.execute(
            "SELECT files.path, file_meta.title FROM files "
            "LEFT JOIN file_meta ON file_meta.path = files.path"
        )
    ]
    paths = {path for path, _ in entries}
    aliases = graph.build_alias_index(entries)

    resolutions: dict[str, str | None] = {}
    pending = 0
    # Read the whole set before updating: modifying a table while stepping a cursor over it is
    # not a defined thing to do in SQLite, and this loop writes to the rows it is reading.
    for row_id, target in conn.execute("SELECT id, target FROM links").fetchall():
        text = str(target)
        if text not in resolutions:
            resolutions[text] = graph.resolve_link(text, paths=paths, aliases=aliases)
        resolved = resolutions[text]
        if resolved is None:
            pending += 1
        conn.execute("UPDATE links SET resolved_path = ? WHERE id = ?", (resolved, int(row_id)))
    return pending


def build_index(
    vault: VaultConfig,
    index_config: IndexConfig,
    embedder: Embedder,
    *,
    db_path: str | Path | None = None,
) -> IndexStats:
    """Bring the index in line with the vault's files, embedding only what changed.

    The database lives outside the vault's git tree and may be deleted at any time; this
    function is the only thing that has to exist to get it back.
    """
    started = time.monotonic()
    resolved = Path(db_path) if db_path is not None else index_config.db_path(vault.name)
    conn = open_index(resolved)
    try:
        _reset_for_model(conn, embedder.model_id)

        known: dict[str, str] = {
            str(path): str(file_hash)
            for path, file_hash in conn.execute("SELECT path, content_hash FROM files")
        }
        seen: set[str] = set()
        skipped: list[tuple[str, str]] = []
        scanned = indexed = unchanged = written = embedded = reused = 0
        links = observations = 0
        dimension = embedding_dimension(conn)

        for absolute, relative in iter_vault_files(
            vault.root, index_config.include, index_config.exclude
        ):
            scanned += 1
            try:
                text = absolute.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as exc:
                skipped.append((relative, f"unreadable as UTF-8 text: {exc}"))
                continue

            seen.add(relative)
            file_hash = content_hash(text)
            if known.get(relative) == file_hash:
                unchanged += 1
                continue

            # Parsed from the full text, before `chunk_markdown` strips the frontmatter the
            # ranking facts live in.
            parsed = graph.parse_file(text)

            chunks = chunk_markdown(
                text,
                max_chars=index_config.chunk_max_chars,
                overlap_chars=index_config.chunk_overlap_chars,
            )
            hashes = [chunk.content_hash for chunk in chunks]
            cached = _embeddings_by_hash(conn, hashes)
            # Widened to `bytes | None` because an install without the `[semantic]` extra stores
            # a chunk with no vector at all; the reuse cache never holds one of those, since it
            # only ever reads rows where the embedding is NOT NULL.
            blobs: dict[str, bytes | None] = dict(cached)
            pending = [chunk for chunk in chunks if chunk.content_hash not in cached]

            if pending:
                vectors = embedder.embed([chunk.embedding_text for chunk in pending])
                if len(vectors) != len(pending):
                    raise IndexBuildError(
                        f"embedder returned {len(vectors)} vectors for {len(pending)} chunks "
                        f"while indexing {relative}"
                    )
                for chunk, vector in zip(pending, vectors, strict=True):
                    if not vector:
                        # An embedder answering with no vector is saying it cannot embed
                        # (`NullEmbedder`). The chunk is still written and still reaches FTS;
                        # only the semantic half of recall is absent, and it says so.
                        blobs[chunk.content_hash] = None
                        continue
                    blob = pack_embedding(vector)
                    if dimension and len(blob) // 4 != dimension:
                        raise IndexBuildError(
                            f"embedder returned a {len(blob) // 4}-dimension vector while the "
                            f"index stores {dimension}-dimension vectors. Delete {resolved} "
                            "and rebuild if the model changed."
                        )
                    dimension = len(blob) // 4
                    blobs[chunk.content_hash] = blob
                    embedded += 1
            reused += len(chunks) - len(pending)

            conn.execute("DELETE FROM chunks WHERE path = ?", (relative,))
            conn.executemany(
                "INSERT INTO chunks(path, ordinal, heading, text, content_hash, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        relative,
                        chunk.ordinal,
                        chunk.heading,
                        chunk.text,
                        chunk.content_hash,
                        blobs[chunk.content_hash],
                    )
                    for chunk in chunks
                ],
            )
            _write_graph_rows(conn, relative, parsed)
            conn.execute(
                "INSERT INTO files(path, content_hash, indexed_at) VALUES (?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "content_hash = excluded.content_hash, indexed_at = excluded.indexed_at",
                (relative, file_hash, time.time()),
            )
            conn.commit()
            indexed += 1
            written += len(chunks)
            links += len(parsed.links)
            observations += len(parsed.observations)

        removed = 0
        for gone in sorted(set(known) - seen):
            conn.execute("DELETE FROM chunks WHERE path = ?", (gone,))
            conn.execute("DELETE FROM files WHERE path = ?", (gone,))
            _delete_graph_rows(conn, gone)
            removed += 1
        if dimension:
            _meta_set(conn, "embedding_dim", str(dimension))
        pending_links = _resolve_links(conn)
        conn.commit()

        return IndexStats(
            db_path=resolved,
            files_scanned=scanned,
            files_indexed=indexed,
            files_unchanged=unchanged,
            files_removed=removed,
            chunks_written=written,
            chunks_embedded=embedded,
            chunks_reused=reused,
            links_written=links,
            observations_written=observations,
            links_pending=pending_links,
            elapsed_seconds=time.monotonic() - started,
            skipped=tuple(skipped),
        )
    finally:
        conn.close()


def embedding_dimension(conn: sqlite3.Connection) -> int:
    """Vector width of the stored embeddings, or 0 when nothing is indexed yet."""
    value = _meta_get(conn, "embedding_dim")
    return int(value) if value else 0


def has_embeddings(conn: sqlite3.Connection) -> bool:
    """Whether any stored chunk carries a vector — i.e. whether semantic recall has anything.

    Asked of the rows rather than of the recorded `model_id`, because the two can disagree in the
    one direction that matters: an index whose model is recorded but whose chunks were all written
    NULL is still a keyword-only index, and a query that trusted the metadata would search a
    matrix with no rows in it and call the silence a result.
    """
    row = conn.execute("SELECT 1 FROM chunks WHERE embedding IS NOT NULL LIMIT 1").fetchone()
    return row is not None


def count_chunks(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return int(row[0])


def fetch_chunks(conn: sqlite3.Connection, ids: Sequence[int]) -> list[ChunkRow]:
    """Read chunk rows by id, in the order the ids were given."""
    if not ids:
        return []
    by_id: dict[int, ChunkRow] = {}
    unique = list(dict.fromkeys(ids))
    for start in range(0, len(unique), _PARAM_BATCH):
        batch = unique[start : start + _PARAM_BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT id, path, ordinal, heading, text FROM chunks WHERE id IN ({placeholders})",
            batch,
        )
        for chunk_id, path, ordinal, heading, text in rows:
            by_id[int(chunk_id)] = ChunkRow(
                id=int(chunk_id),
                path=str(path),
                ordinal=int(ordinal),
                heading=str(heading),
                text=str(text),
            )
    return [by_id[chunk_id] for chunk_id in unique if chunk_id in by_id]


def fetch_file_metadata(
    conn: sqlite3.Connection, paths: Sequence[str]
) -> dict[str, graph.FileMetadata]:
    """The ranking facts these files declared, as the last pass read them off their frontmatter.

    Read from the index rather than from disk because ranking asks this question once per
    candidate file per query, and the answer is a pure function of bytes the indexer already
    parsed. A path with no row is simply absent — a caller must decide what an unjudged file is
    worth, and this function will not invent a neutral one for it.
    """
    if not paths:
        return {}
    found: dict[str, graph.FileMetadata] = {}
    unique = list(dict.fromkeys(paths))
    for start in range(0, len(unique), _PARAM_BATCH):
        batch = unique[start : start + _PARAM_BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            "SELECT path, title, kind, importance, date, superseded_by FROM file_meta "
            f"WHERE path IN ({placeholders})",
            batch,
        )
        for path, title, kind, importance, day, superseded_by in rows:
            found[str(path)] = graph.FileMetadata(
                title=None if title is None else str(title),
                kind=None if kind is None else str(kind),
                importance=None if importance is None else int(importance),
                date=None if day is None else str(day),
                superseded_by=None if superseded_by is None else str(superseded_by),
            )
    return found


def fetch_neighbors(conn: sqlite3.Connection, paths: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Every file one resolved relation hop from each of these, in both directions.

    Both directions because a relation is a statement about a pair, and which end of it holds the
    wikilink is a fact about who wrote first: a note filed today that says `part_of [[project]]`
    makes the project relevant to the note *and* the note relevant to the project.

    Pending links contribute nothing — a target the corpus cannot resolve names no file to boost.
    Neighbours come back sorted so a hop discounted by fan-out is the same hop on every run.
    """
    if not paths:
        return {}
    collected: dict[str, set[str]] = {}
    unique = list(dict.fromkeys(paths))
    for start in range(0, len(unique), _PARAM_BATCH):
        batch = unique[start : start + _PARAM_BATCH]
        placeholders = ",".join("?" * len(batch))
        outgoing = conn.execute(
            "SELECT path, resolved_path FROM links "
            f"WHERE resolved_path IS NOT NULL AND path IN ({placeholders})",
            batch,
        )
        incoming = conn.execute(
            f"SELECT resolved_path, path FROM links WHERE resolved_path IN ({placeholders})",
            batch,
        )
        for source, neighbor in [*outgoing, *incoming]:
            if str(source) == str(neighbor):
                continue
            collected.setdefault(str(source), set()).add(str(neighbor))
    return {source: tuple(sorted(neighbors)) for source, neighbors in collected.items()}


def fetch_leading_chunks(conn: sqlite3.Connection, paths: Sequence[str]) -> list[ChunkRow]:
    """One citable chunk per file — its opening section — in the order the paths were given.

    A file pulled into results by a relation rather than by its own words has no chunk that
    matched anything, so something has to stand for it. The first chunk is the file's own
    beginning, which is the closest thing to a summary that costs no model call.
    """
    if not paths:
        return []
    best: dict[str, ChunkRow] = {}
    unique = list(dict.fromkeys(paths))
    for start in range(0, len(unique), _PARAM_BATCH):
        batch = unique[start : start + _PARAM_BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT id, path, ordinal, heading, text FROM chunks WHERE path IN ({placeholders}) "
            "ORDER BY path, ordinal",
            batch,
        )
        for chunk_id, path, ordinal, heading, text in rows:
            best.setdefault(
                str(path),
                ChunkRow(
                    id=int(chunk_id),
                    path=str(path),
                    ordinal=int(ordinal),
                    heading=str(heading),
                    text=str(text),
                ),
            )
    return [best[path] for path in unique if path in best]


def load_embeddings(conn: sqlite3.Connection) -> tuple[list[int], npt.NDArray[np.float32]]:
    """Every stored vector as one matrix, with the chunk id for each row.

    Returned raw rather than normalized: the caller decides the metric, and a stored vector
    that has been silently rescaled is a debugging trap.
    """
    ids: list[int] = []
    vectors: list[npt.NDArray[np.float32]] = []
    for chunk_id, blob in conn.execute(
        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"
    ):
        ids.append(int(chunk_id))
        vectors.append(unpack_embedding(bytes(blob)))
    if not vectors:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, np.vstack(vectors).astype(np.float32, copy=False)


def match_chunks(conn: sqlite3.Connection, fts_query: str, limit: int = 50) -> list[int]:
    """Chunk ids matching an FTS5 expression, best BM25 first.

    Ranking stops here: this is keyword retrieval, not the hybrid fusion that recall owns.
    """
    if not fts_query.strip():
        return []
    try:
        rows = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise IndexBuildError(
            f"invalid FTS5 query {fts_query!r}: {exc}. Pass user text through escape_fts_query."
        ) from exc
    return [int(row[0]) for row in rows]
