"""Search index building, incrementality, and disposability.

The theme of these tests: the index must be a pure function of the files. Every test either
proves that a change on disk is reflected in the database, or that an absence of change costs
nothing — and one proves the whole database can be thrown away without losing anything.

No test touches the network. The real backend downloads a ~130MB model on first use, so every
test injects `StubEmbedder`, whose vectors are a deterministic function of the input text.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from memvault import index as index_module
from memvault.config import IndexConfig, VaultConfig
from memvault.graph import FileMetadata
from memvault.index import (
    QUERY_PREFIX,
    Chunk,
    FastEmbedEmbedder,
    IndexBuildError,
    NullEmbedder,
    build_index,
    chunk_markdown,
    count_chunks,
    embedding_dimension,
    escape_fts_query,
    fetch_chunks,
    fetch_file_metadata,
    fetch_leading_chunks,
    fetch_neighbors,
    has_embeddings,
    load_embeddings,
    match_chunks,
    open_index,
    semantic_available,
    strip_frontmatter,
)

DIMENSION = 8


class StubEmbedder:
    """A deterministic embedder that records what it was asked to embed.

    Same text in, same vector out, with no model and no network — which is the only way the
    "re-indexing embeds nothing" assertions can be stated as `calls == []` rather than as a
    timing observation.
    """

    def __init__(self, model_id: str = "stub-v1") -> None:
        self.model_id = model_id
        self.calls: list[list[str]] = []

    @property
    def embedded_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(DIMENSION)]


def make_vault(tmp_path: Path, files: dict[str, str], name: str = "personal") -> VaultConfig:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return VaultConfig(name=name, root=root, inbox=root / "inbox")


def index_config(tmp_path: Path, **overrides: object) -> IndexConfig:
    defaults: dict[str, object] = {"path": str(tmp_path / "db" / "{vault}.db")}
    defaults.update(overrides)
    return IndexConfig(**defaults)  # type: ignore[arg-type]


def chunk_tuples(db_path: Path) -> list[tuple[str, int, str, str]]:
    conn = open_index(db_path)
    try:
        rows = conn.execute(
            "SELECT path, ordinal, heading, text FROM chunks ORDER BY path, ordinal"
        ).fetchall()
    finally:
        conn.close()
    return [(str(p), int(o), str(h), str(t)) for p, o, h, t in rows]


def link_tuples(db_path: Path) -> list[tuple[str, int, str, str, str | None]]:
    """Every stored edge, without its row id — ids are insertion order, edges are content."""
    conn = open_index(db_path)
    try:
        rows = conn.execute(
            "SELECT path, ordinal, predicate, target, resolved_path FROM links "
            "ORDER BY path, ordinal"
        ).fetchall()
    finally:
        conn.close()
    return [
        (str(p), int(o), str(pred), str(t), None if r is None else str(r))
        for p, o, pred, t, r in rows
    ]


def observation_tuples(db_path: Path) -> list[tuple[str, int, str, str, str]]:
    conn = open_index(db_path)
    try:
        rows = conn.execute(
            "SELECT path, ordinal, category, text, tags FROM observations ORDER BY path, ordinal"
        ).fetchall()
    finally:
        conn.close()
    return [(str(p), int(o), str(c), str(t), str(g)) for p, o, c, t, g in rows]


MetaRow = tuple[str, str | None, str | None, int | None, str | None, str | None]


def file_meta_tuples(db_path: Path) -> list[MetaRow]:
    conn = open_index(db_path)
    try:
        rows = conn.execute(
            "SELECT path, title, kind, importance, date, superseded_by FROM file_meta ORDER BY path"
        ).fetchall()
    finally:
        conn.close()
    return [
        (
            str(path),
            None if title is None else str(title),
            None if kind is None else str(kind),
            None if importance is None else int(importance),
            None if day is None else str(day),
            None if superseded is None else str(superseded),
        )
        for path, title, kind, importance, day, superseded in rows
    ]


def graph_snapshot(db_path: Path) -> tuple[list[object], list[object], list[object]]:
    """The whole graph side of the index, in one comparable value."""
    return (
        list(link_tuples(db_path)),
        list(observation_tuples(db_path)),
        list(file_meta_tuples(db_path)),
    )


def downgrade_to_v1(db_path: Path) -> None:
    """Make an index look like one written before the graph tables existed.

    Faithful to what a real v1 database holds: the schema marker of the old version and none of
    the new tables at all, so the rebuild has to create them as well as fill them.
    """
    conn = open_index(db_path)
    try:
        conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        for table in ("links", "observations", "file_meta"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
    finally:
        conn.close()


def fts_paths(db_path: Path, query: str) -> list[str]:
    conn = open_index(db_path)
    try:
        ids = match_chunks(conn, escape_fts_query(query))
        return [row.path for row in fetch_chunks(conn, ids)]
    finally:
        conn.close()


class TestChunking:
    def test_frontmatter_is_stripped_before_chunking(self) -> None:
        text = "---\ntitle: Note\nlocal_only: true\n---\n\n# Body\n\nReal content.\n"

        chunks = chunk_markdown(text, max_chars=1200, overlap_chars=150)

        assert len(chunks) == 1
        assert "local_only" not in chunks[0].text
        assert "Real content." in chunks[0].text

    def test_unterminated_frontmatter_leaves_the_document_intact(self) -> None:
        text = "---\nnot really frontmatter\n\n# Heading\n\nBody.\n"

        assert strip_frontmatter(text) == text

    def test_headings_split_a_long_file_into_one_chunk_per_section(self) -> None:
        body = "\n".join(f"## Section {n}\n\n{'word ' * 40}\n" for n in range(1, 5))
        text = f"# Title\n\nPreamble paragraph.\n\n{body}"

        chunks = chunk_markdown(text, max_chars=1200, overlap_chars=150)

        assert [chunk.heading for chunk in chunks] == [
            "Title",
            "Title > Section 1",
            "Title > Section 2",
            "Title > Section 3",
            "Title > Section 4",
        ]

    def test_headings_inside_fenced_code_do_not_start_a_section(self) -> None:
        text = "# Real\n\n```bash\n# not a heading\necho hi\n```\n\nAfter.\n"

        chunks = chunk_markdown(text, max_chars=1200, overlap_chars=150)

        assert len(chunks) == 1
        assert "# not a heading" in chunks[0].text

    def test_an_over_long_section_is_split_by_the_size_ceiling_with_overlap(self) -> None:
        paragraph = "\n".join(f"Line {n} of a very long single section." for n in range(200))
        text = f"## Long\n\n{paragraph}\n"

        chunks = chunk_markdown(text, max_chars=600, overlap_chars=120)

        assert len(chunks) > 1
        assert all(len(chunk.text) <= 600 for chunk in chunks)
        assert all(chunk.heading == "Long" for chunk in chunks)
        for earlier, later in zip(chunks, chunks[1:], strict=False):
            assert earlier.text[-60:] in later.text

    def test_ordinals_are_contiguous_from_zero(self) -> None:
        text = "# A\n\nalpha\n\n# B\n\nbeta\n\n# C\n\ngamma\n"

        chunks = chunk_markdown(text, max_chars=1200, overlap_chars=150)

        assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]

    def test_an_empty_document_yields_no_chunks(self) -> None:
        assert chunk_markdown("   \n\n\t\n", max_chars=1200, overlap_chars=150) == []

    def test_the_heading_breadcrumb_carries_ancestry_into_the_embedded_text(self) -> None:
        chunk = Chunk(ordinal=0, heading="Erfpacht > Costs", text="## Costs\n\n1200 per year.")

        assert chunk.embedding_text.startswith("Erfpacht > Costs")
        assert chunk.content_hash != Chunk(ordinal=0, heading="", text=chunk.text).content_hash


class TestTurkish:
    """Turkish is the reason KTD2 picked a multilingual model; it must survive the plumbing."""

    turkish = (
        "# Şiir Defteri\n\n"
        "Bugün İstanbul'da yağmur yağdı ve düşünceler ağırlaştı.\n"
        "Çiğdem, öğle vakti gülümsedi.\n"
    )

    def test_turkish_text_round_trips_through_chunking_unmangled(self) -> None:
        chunks = chunk_markdown(self.turkish, max_chars=1200, overlap_chars=150)

        assert len(chunks) == 1
        assert "İstanbul'da yağmur yağdı" in chunks[0].text
        assert chunks[0].heading == "Şiir Defteri"

    def test_turkish_text_round_trips_through_indexing_and_embedding(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"siir.md": self.turkish})
        embedder = StubEmbedder()

        stats = build_index(vault, index_config(tmp_path), embedder)

        assert "İstanbul'da yağmur yağdı" in embedder.embedded_texts[0]
        assert chunk_tuples(stats.db_path)[0][3].count("ğ") == self.turkish.count("ğ")

    def test_a_turkish_keyword_is_findable_through_fts(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"siir.md": self.turkish})

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert fts_paths(stats.db_path, "Çiğdem") == ["siir.md"]


class TestBuildIndex:
    def test_indexing_a_small_vault_populates_chunks_and_fts_rows(self, tmp_path: Path) -> None:
        vault = make_vault(
            tmp_path,
            {
                "a.md": "# Alpha\n\nThe alpha note.\n\n## Detail\n\nMore alpha.\n",
                "notes/b.md": "# Beta\n\nThe beta note.\n",
            },
        )

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert stats.files_scanned == 2
        assert stats.files_indexed == 2
        assert stats.chunks_written == 3
        assert stats.chunks_embedded == 3
        conn = open_index(stats.db_path)
        try:
            assert count_chunks(conn) == 3
            assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 3
        finally:
            conn.close()

    def test_every_chunk_stores_an_embedding_of_the_expected_width(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"a.md": "# Alpha\n\nBody.\n"})

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        conn = open_index(stats.db_path)
        try:
            ids, matrix = load_embeddings(conn)
        finally:
            conn.close()
        assert len(ids) == 1
        assert matrix.shape == (1, DIMENSION)

    def test_reindexing_unchanged_files_embeds_nothing(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"a.md": "# Alpha\n\nBody.\n", "b.md": "# Beta\n\nBody.\n"})
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder())
        second = StubEmbedder()

        stats = build_index(vault, config, second)

        assert second.calls == []
        assert stats.files_unchanged == 2
        assert stats.files_indexed == 0
        assert stats.chunks_embedded == 0

    def test_editing_one_file_re_embeds_only_that_files_chunks(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"a.md": "# Alpha\n\nBody.\n", "b.md": "# Beta\n\nBody.\n"})
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder())
        (vault.root / "a.md").write_text("# Alpha\n\nRewritten body.\n", encoding="utf-8")
        second = StubEmbedder()

        stats = build_index(vault, config, second)

        assert stats.files_indexed == 1
        assert stats.files_unchanged == 1
        assert len(second.embedded_texts) == 1
        assert "Rewritten body." in second.embedded_texts[0]
        assert not any("Beta" in text for text in second.embedded_texts)

    def test_an_unchanged_section_of_a_changed_file_reuses_its_embedding(
        self, tmp_path: Path
    ) -> None:
        original = "# Keep\n\nUntouched section.\n\n# Change\n\nOriginal wording.\n"
        vault = make_vault(tmp_path, {"a.md": original})
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder())
        (vault.root / "a.md").write_text(
            original.replace("Original wording.", "New wording."), encoding="utf-8"
        )
        second = StubEmbedder()

        stats = build_index(vault, config, second)

        assert stats.chunks_embedded == 1
        assert stats.chunks_reused == 1
        assert not any("Untouched section." in text for text in second.embedded_texts)

    def test_deleting_a_file_removes_its_chunks_from_both_tables(self, tmp_path: Path) -> None:
        vault = make_vault(
            tmp_path, {"a.md": "# Alpha\n\nRare token zolgensma.\n", "b.md": "# B\n"}
        )
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder())
        (vault.root / "a.md").unlink()

        stats = build_index(vault, config, StubEmbedder())

        assert stats.files_removed == 1
        assert [path for path, *_ in chunk_tuples(stats.db_path)] == ["b.md"]
        assert fts_paths(stats.db_path, "zolgensma") == []

    def test_deleting_the_db_and_rebuilding_reproduces_equivalent_chunk_rows(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(
            tmp_path,
            {
                "a.md": "# Alpha\n\nOne.\n\n## Deep\n\nTwo.\n",
                "sub/b.md": "# Beta\n\nThree.\n",
                "c.md": "No heading at all, just a paragraph of prose.\n",
            },
        )
        config = index_config(tmp_path)
        first = build_index(vault, config, StubEmbedder())
        before = chunk_tuples(first.db_path)
        first.db_path.unlink()

        second = build_index(vault, config, StubEmbedder())

        assert chunk_tuples(second.db_path) == before
        assert second.chunks_written == first.chunks_written

    def test_a_local_only_file_is_still_indexed(self, tmp_path: Path) -> None:
        vault = make_vault(
            tmp_path,
            {
                "private/secret.md": (
                    "---\nlocal_only: true\n---\n\n# Private\n\nSalary negotiation notes.\n"
                )
            },
        )

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert stats.chunks_written == 1
        assert fts_paths(stats.db_path, "salary") == ["private/secret.md"]

    def test_exclude_globs_are_honored(self, tmp_path: Path) -> None:
        vault = make_vault(
            tmp_path,
            {
                "keep.md": "# Keep\n\nIndexed.\n",
                ".git/COMMIT_EDITMSG.md": "# Git\n\nNot indexed.\n",
                "web/node_modules/pkg/readme.md": "# Dep\n\nNot indexed.\n",
            },
        )

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert [path for path, *_ in chunk_tuples(stats.db_path)] == ["keep.md"]
        assert stats.files_scanned == 1

    def test_include_globs_select_only_matching_files(self, tmp_path: Path) -> None:
        vault = make_vault(
            tmp_path, {"note.md": "# Note\n\nYes.\n", "data.csv": "a,b\n1,2\n", "raw.txt": "no\n"}
        )

        stats = build_index(vault, index_config(tmp_path, include=("**/*.md",)), StubEmbedder())

        assert [path for path, *_ in chunk_tuples(stats.db_path)] == ["note.md"]

    def test_a_non_utf8_file_is_skipped_with_a_reason_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(tmp_path, {"good.md": "# Good\n\nText.\n"})
        (vault.root / "binary.md").write_bytes(b"\xff\xfe\x00\x01 not text")

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert stats.files_indexed == 1
        assert [path for path, _ in stats.skipped] == ["binary.md"]
        assert "UTF-8" in stats.skipped[0][1]

    def test_an_empty_file_indexes_no_chunks_but_is_recorded_as_seen(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"empty.md": "\n\n", "real.md": "# Real\n\nBody.\n"})
        config = index_config(tmp_path)

        first = build_index(vault, config, StubEmbedder())
        second_embedder = StubEmbedder()
        second = build_index(vault, config, second_embedder)

        assert first.chunks_written == 1
        assert second.files_unchanged == 2
        assert second_embedder.calls == []

    def test_an_empty_vault_produces_an_empty_index_rather_than_an_error(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(tmp_path, {})

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert stats.files_scanned == 0
        assert stats.chunks_written == 0
        conn = open_index(stats.db_path)
        try:
            ids, matrix = load_embeddings(conn)
        finally:
            conn.close()
        assert ids == []
        assert matrix.size == 0

    def test_the_db_lives_where_the_config_says_and_not_inside_the_vault(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(tmp_path, {"a.md": "# A\n\nBody.\n"})

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert stats.db_path == (tmp_path / "db" / "personal.db").resolve()
        assert vault.root not in stats.db_path.parents

    def test_an_explicit_db_path_overrides_the_configured_template(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"a.md": "# A\n\nBody.\n"})
        elsewhere = tmp_path / "scratch" / "one-off.db"

        stats = build_index(vault, index_config(tmp_path), StubEmbedder(), db_path=elsewhere)

        assert stats.db_path == elsewhere
        assert elsewhere.exists()

    def test_changing_the_embedding_model_discards_incomparable_vectors(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(tmp_path, {"a.md": "# A\n\nBody.\n"})
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder("stub-v1"))
        replacement = StubEmbedder("stub-v2")

        stats = build_index(vault, config, replacement)

        assert stats.files_indexed == 1
        assert stats.chunks_embedded == 1
        assert replacement.calls != []

    def test_an_embedder_returning_the_wrong_number_of_vectors_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        class ShortEmbedder(StubEmbedder):
            def embed(self, texts: Sequence[str]) -> list[list[float]]:
                return super().embed(texts)[:-1]

        vault = make_vault(tmp_path, {"a.md": "# A\n\nOne.\n\n# B\n\nTwo.\n"})

        with pytest.raises(IndexBuildError, match="vectors for"):
            build_index(vault, index_config(tmp_path), ShortEmbedder())


class TestReadAccess:
    """What U8 consumes: chunk rows, the stacked embedding matrix, and an FTS5 MATCH."""

    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        vault = make_vault(
            tmp_path,
            {
                "a.md": "# Erfpacht\n\nThe ground lease runs to 2041.\n",
                "b.md": "# Mortgage\n\nNHG caps the loan at a set amount.\n",
            },
        )
        stats = build_index(vault, index_config(tmp_path), StubEmbedder())
        return open_index(stats.db_path)

    def test_load_embeddings_returns_one_row_per_chunk_in_id_order(
        self, conn: sqlite3.Connection
    ) -> None:
        ids, matrix = load_embeddings(conn)

        assert ids == sorted(ids)
        assert matrix.shape == (len(ids), DIMENSION)
        conn.close()

    def test_fetch_chunks_preserves_the_requested_order(self, conn: sqlite3.Connection) -> None:
        ids, _ = load_embeddings(conn)

        rows = fetch_chunks(conn, list(reversed(ids)))

        assert [row.id for row in rows] == list(reversed(ids))
        conn.close()

    def test_fetch_chunks_ignores_ids_that_no_longer_exist(self, conn: sqlite3.Connection) -> None:
        assert fetch_chunks(conn, [999_999]) == []
        assert fetch_chunks(conn, []) == []
        conn.close()

    def test_every_result_carries_a_vault_relative_path(self, conn: sqlite3.Connection) -> None:
        ids = match_chunks(conn, escape_fts_query("erfpacht"))

        rows = fetch_chunks(conn, ids)

        assert [row.path for row in rows] == ["a.md"]
        assert all(not Path(row.path).is_absolute() for row in rows)
        conn.close()

    def test_a_query_matching_nothing_returns_an_empty_list(self, conn: sqlite3.Connection) -> None:
        assert match_chunks(conn, escape_fts_query("zolgensma")) == []
        assert match_chunks(conn, "") == []
        conn.close()

    def test_a_malformed_fts_expression_names_the_escaping_helper(
        self, conn: sqlite3.Connection
    ) -> None:
        with pytest.raises(IndexBuildError, match="escape_fts_query"):
            match_chunks(conn, 'unbalanced "quote')
        conn.close()


class TestFastEmbedEmbedder:
    """The real backend, exercised without downloading a model or touching the network."""

    class FakeModel:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.seen.extend(texts)
            return [[0.0, 1.0] for _ in texts]

    def test_stored_text_is_embedded_with_the_passage_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = self.FakeModel()
        embedder = FastEmbedEmbedder()
        monkeypatch.setattr(embedder, "_load", lambda: fake)

        vectors = embedder.embed(["the ground lease runs to 2041"])

        assert fake.seen == ["passage: the ground lease runs to 2041"]
        assert vectors == [[0.0, 1.0]]

    def test_a_query_side_embedder_uses_the_other_half_of_the_e5_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = self.FakeModel()
        embedder = FastEmbedEmbedder(prefix=QUERY_PREFIX)
        monkeypatch.setattr(embedder, "_load", lambda: fake)

        embedder.embed(["erfpacht"])

        assert fake.seen == ["query: erfpacht"]

    def test_embedding_nothing_never_loads_the_model(self) -> None:
        assert FastEmbedEmbedder("nonexistent/model-xyz").embed([]) == []

    def test_an_unknown_model_names_the_registration_table(self) -> None:
        embedder = FastEmbedEmbedder("nonexistent/model-xyz")

        with pytest.raises(IndexBuildError, match="HUB_MODELS"):
            embedder.embed(["anything"])


class TestEscapeFtsQuery:
    def test_punctuation_that_would_be_fts_syntax_is_neutralized(self) -> None:
        assert escape_fts_query('erfpacht OR "x-') == '"erfpacht" "OR" "x"'

    def test_turkish_words_survive_escaping(self) -> None:
        assert escape_fts_query("Çiğdem öğle") == '"Çiğdem" "öğle"'

    def test_an_empty_query_escapes_to_an_empty_expression(self) -> None:
        assert escape_fts_query("   !!!  ") == ""


SOURCE_NOTE = (
    "---\n"
    "title: Ranking weights\n"
    "kind: reflection\n"
    "date: 2026-08-12\n"
    "importance: 8\n"
    "---\n\n"
    "# Ranking\n\n"
    "- [decision] Weights are additive on the RRF scale #ranking #memvault\n"
    "\n"
    "## Relations\n\n"
    "- part_of [[repos/MemVault/learnings.md]]\n"
)

TARGET_NOTE = "---\ntitle: MemVault learnings\n---\n\n# Learnings\n\nWhat the engine taught.\n"


class TestGraphIndexing:
    """The graph tables are written from the files, on the same pass that writes the chunks."""

    def test_a_note_contributes_its_edges_its_statements_and_its_facts(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(
            tmp_path, {"notes/ranking.md": SOURCE_NOTE, "repos/MemVault/learnings.md": TARGET_NOTE}
        )

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert link_tuples(stats.db_path) == [
            (
                "notes/ranking.md",
                0,
                "part_of",
                "repos/MemVault/learnings.md",
                "repos/MemVault/learnings.md",
            )
        ]
        assert observation_tuples(stats.db_path) == [
            (
                "notes/ranking.md",
                0,
                "decision",
                "Weights are additive on the RRF scale #ranking #memvault",
                "ranking memvault",
            )
        ]
        assert (
            "notes/ranking.md",
            "Ranking weights",
            "reflection",
            8,
            "2026-08-12",
            None,
        ) in file_meta_tuples(stats.db_path)

    def test_a_supersession_key_is_stored_as_a_fact_about_the_file(self, tmp_path: Path) -> None:
        vault = make_vault(
            tmp_path,
            {"old.md": "---\ntitle: Old\nsuperseded_by: new.md\n---\n\nBody.\n"},
        )

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert file_meta_tuples(stats.db_path) == [("old.md", "Old", None, None, None, "new.md")]

    def test_every_indexed_file_has_a_metadata_row_even_when_it_declares_nothing(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(tmp_path, {"plain.md": "# Plain\n\nNo frontmatter at all.\n"})

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert file_meta_tuples(stats.db_path) == [("plain.md", None, None, None, None, None)]

    def test_a_link_whose_target_is_not_in_the_vault_is_pending_rather_than_an_error(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(tmp_path, {"notes/ranking.md": SOURCE_NOTE})

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert link_tuples(stats.db_path) == [
            ("notes/ranking.md", 0, "part_of", "repos/MemVault/learnings.md", None)
        ]

    def test_editing_a_file_replaces_its_graph_rows(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"a.md": "# A\n\n- [decision] First call.\n"})
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder())
        (vault.root / "a.md").write_text(
            "# A\n\n- [decision] Reversed it.\n- about [[b]]\n", encoding="utf-8"
        )

        stats = build_index(vault, config, StubEmbedder())

        assert [row[3] for row in observation_tuples(stats.db_path)] == ["Reversed it."]
        assert [row[2] for row in link_tuples(stats.db_path)] == ["about"]

    def test_deleting_a_file_removes_its_graph_rows_with_its_chunks(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"notes/ranking.md": SOURCE_NOTE, "b.md": "# B\n\nBody.\n"})
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder())
        (vault.root / "notes" / "ranking.md").unlink()

        stats = build_index(vault, config, StubEmbedder())

        assert link_tuples(stats.db_path) == []
        assert observation_tuples(stats.db_path) == []
        assert [row[0] for row in file_meta_tuples(stats.db_path)] == ["b.md"]

    def test_a_pass_that_changes_nothing_leaves_the_graph_rows_standing(
        self, tmp_path: Path
    ) -> None:
        """Unchanged files are skipped for embedding; their edges must not be skipped away."""
        vault = make_vault(
            tmp_path, {"notes/ranking.md": SOURCE_NOTE, "repos/MemVault/learnings.md": TARGET_NOTE}
        )
        config = index_config(tmp_path)
        first = build_index(vault, config, StubEmbedder())
        before = graph_snapshot(first.db_path)

        second = build_index(vault, config, StubEmbedder())

        assert second.files_unchanged == 2
        assert graph_snapshot(second.db_path) == before

    def test_the_pass_reports_what_the_graph_gained(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"notes/ranking.md": SOURCE_NOTE})

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert stats.links_written == 1
        assert stats.observations_written == 1
        assert stats.links_pending == 1

    def test_an_edge_written_inside_a_code_fence_is_documentation_not_an_edge(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(
            tmp_path,
            {"docs/syntax.md": "# Syntax\n\n```\n- part_of [[repos/MemVault]]\n```\n\nThat.\n"},
        )

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert link_tuples(stats.db_path) == []


class TestLinkResolution:
    """Resolution is a whole-corpus property, so it is recomputed after every pass."""

    def test_a_target_filed_later_resolves_without_the_source_file_changing(
        self, tmp_path: Path
    ) -> None:
        """The load-bearing incremental case: a link pending today must resolve tomorrow.

        Resolving only while parsing the source file would leave this link pending forever, and
        the incremental index would then disagree with a rebuild — the one property the whole
        files-are-truth design rests on.
        """
        vault = make_vault(tmp_path, {"notes/ranking.md": SOURCE_NOTE})
        config = index_config(tmp_path)
        first = build_index(vault, config, StubEmbedder())
        assert link_tuples(first.db_path)[0][4] is None
        (vault.root / "repos" / "MemVault").mkdir(parents=True)
        (vault.root / "repos" / "MemVault" / "learnings.md").write_text(
            TARGET_NOTE, encoding="utf-8"
        )

        second = build_index(vault, config, StubEmbedder())

        assert second.files_unchanged == 1
        assert link_tuples(second.db_path) == [
            (
                "notes/ranking.md",
                0,
                "part_of",
                "repos/MemVault/learnings.md",
                "repos/MemVault/learnings.md",
            )
        ]

    def test_the_incrementally_resolved_index_matches_a_fresh_rebuild(self, tmp_path: Path) -> None:
        """Same corpus, two histories: built in two passes, and built once from scratch."""
        vault = make_vault(tmp_path, {"notes/ranking.md": SOURCE_NOTE})
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder())
        (vault.root / "repos" / "MemVault").mkdir(parents=True)
        (vault.root / "repos" / "MemVault" / "learnings.md").write_text(
            TARGET_NOTE, encoding="utf-8"
        )
        incremental = build_index(vault, config, StubEmbedder())
        incrementally_built = graph_snapshot(incremental.db_path)
        incremental.db_path.unlink()

        rebuilt = build_index(vault, config, StubEmbedder())

        assert graph_snapshot(rebuilt.db_path) == incrementally_built

    def test_a_title_resolves_a_link_that_names_no_path(self, tmp_path: Path) -> None:
        vault = make_vault(
            tmp_path,
            {
                "a.md": "# A\n\n- about [[MemVault learnings]]\n",
                "repos/MemVault/learnings.md": TARGET_NOTE,
            },
        )

        stats = build_index(vault, index_config(tmp_path), StubEmbedder())

        assert link_tuples(stats.db_path)[0][4] == "repos/MemVault/learnings.md"

    def test_a_second_file_claiming_the_title_demotes_the_link_to_pending(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(
            tmp_path,
            {
                "a.md": "# A\n\n- about [[MemVault learnings]]\n",
                "repos/MemVault/learnings.md": TARGET_NOTE,
            },
        )
        config = index_config(tmp_path)
        first = build_index(vault, config, StubEmbedder())
        assert link_tuples(first.db_path)[0][4] is not None
        (vault.root / "copy.md").write_text(TARGET_NOTE, encoding="utf-8")

        second = build_index(vault, config, StubEmbedder())

        assert link_tuples(second.db_path)[0][4] is None

    def test_deleting_a_target_returns_the_links_that_named_it_to_pending(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(
            tmp_path, {"notes/ranking.md": SOURCE_NOTE, "repos/MemVault/learnings.md": TARGET_NOTE}
        )
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder())
        (vault.root / "repos" / "MemVault" / "learnings.md").unlink()

        stats = build_index(vault, config, StubEmbedder())

        assert link_tuples(stats.db_path)[0][4] is None


class TestGraphDisposability:
    def test_deleting_the_db_and_rebuilding_reproduces_identical_graph_rows(
        self, tmp_path: Path
    ) -> None:
        """R1 stated as a test: the graph is a function of the files and nothing else."""
        vault = make_vault(
            tmp_path,
            {
                "notes/ranking.md": SOURCE_NOTE,
                "repos/MemVault/learnings.md": TARGET_NOTE,
                "notes/orphan.md": "# Orphan\n\n- about [[nothing/here.md]]\n",
            },
        )
        config = index_config(tmp_path)
        first = build_index(vault, config, StubEmbedder())
        before = graph_snapshot(first.db_path)
        first.db_path.unlink()

        second = build_index(vault, config, StubEmbedder())

        assert graph_snapshot(second.db_path) == before
        assert before[0] != []


class TestSchemaMigration:
    """A schema bump wipes and rebuilds. It must cost no embeddings and leave nothing behind."""

    vault_files = {
        "notes/ranking.md": SOURCE_NOTE,
        "repos/MemVault/learnings.md": TARGET_NOTE,
        "b.md": "# B\n\nAnother note entirely.\n",
    }

    def test_a_v1_index_rebuilds_into_v2_without_re_embedding_a_single_chunk(
        self, tmp_path: Path
    ) -> None:
        """The load-bearing migration case: the wipe empties the table the reuse cache reads.

        Without snapshotting `(content_hash, embedding)` before the delete, every schema bump
        would silently re-embed the whole corpus — minutes of work with a known answer, and on a
        real vault the reason nobody would run the migration twice.
        """
        vault = make_vault(tmp_path, self.vault_files)
        config = index_config(tmp_path)
        first = build_index(vault, config, StubEmbedder())
        chunks_before = chunk_tuples(first.db_path)
        downgrade_to_v1(first.db_path)
        embedder = StubEmbedder()

        stats = build_index(vault, config, embedder)

        assert embedder.calls == []
        assert stats.chunks_embedded == 0
        assert stats.chunks_reused == stats.chunks_written
        assert chunk_tuples(stats.db_path) == chunks_before

    def test_the_rebuilt_index_still_holds_a_vector_for_every_chunk(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, self.vault_files)
        config = index_config(tmp_path)
        first = build_index(vault, config, StubEmbedder())
        downgrade_to_v1(first.db_path)

        stats = build_index(vault, config, StubEmbedder())

        conn = open_index(stats.db_path)
        try:
            ids, matrix = load_embeddings(conn)
            assert len(ids) == count_chunks(conn)
        finally:
            conn.close()
        assert matrix.shape == (len(ids), DIMENSION)

    def test_the_bump_fills_the_graph_tables_the_old_database_never_had(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(tmp_path, self.vault_files)
        config = index_config(tmp_path)
        first = build_index(vault, config, StubEmbedder())
        expected = graph_snapshot(first.db_path)
        downgrade_to_v1(first.db_path)

        stats = build_index(vault, config, StubEmbedder())

        assert graph_snapshot(stats.db_path) == expected

    def test_the_bump_leaves_no_rows_for_files_that_are_gone(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, self.vault_files)
        config = index_config(tmp_path)
        first = build_index(vault, config, StubEmbedder())
        downgrade_to_v1(first.db_path)
        (vault.root / "notes" / "ranking.md").unlink()

        stats = build_index(vault, config, StubEmbedder())

        indexed = {path for path, *_ in file_meta_tuples(stats.db_path)}
        assert "notes/ranking.md" not in indexed
        assert link_tuples(stats.db_path) == []
        assert observation_tuples(stats.db_path) == []

    def test_a_changed_embedding_model_still_re_embeds_everything(self, tmp_path: Path) -> None:
        """The snapshot must not outlive its reason: vectors from another model are not reusable."""
        vault = make_vault(tmp_path, self.vault_files)
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder("stub-v1"))
        replacement = StubEmbedder("stub-v2")

        stats = build_index(vault, config, replacement)

        assert replacement.calls != []
        assert stats.chunks_embedded == stats.chunks_written


HUB_NOTE = (
    "---\n"
    "title: Hub\n"
    "kind: reflection\n"
    "importance: 8\n"
    "date: 2026-08-12\n"
    "---\n\n"
    "# Hub\n\n"
    "- lists [[a.md]]\n"
    "- lists [[b.md]]\n"
    "- lists [[nowhere.md]]\n"
    "- about [[hub.md]]\n"
)


class TestRankingReadAccess:
    """What U4's ranking asks the index for: declared facts, one relation hop, a citable chunk."""

    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        vault = make_vault(
            tmp_path,
            {
                "hub.md": HUB_NOTE,
                "a.md": "# Alpha\n\nFirst section.\n\n## Later\n\nSecond section.\n",
                "b.md": "# Beta\n\nOnly section.\n",
            },
        )
        stats = build_index(vault, index_config(tmp_path), StubEmbedder())
        return open_index(stats.db_path)

    def test_the_declared_facts_come_back_as_the_indexer_read_them(
        self, conn: sqlite3.Connection
    ) -> None:
        found = fetch_file_metadata(conn, ["hub.md", "a.md"])

        assert found["hub.md"] == FileMetadata(
            title="Hub", kind="reflection", importance=8, date="2026-08-12"
        )
        assert found["a.md"] == FileMetadata()
        conn.close()

    def test_a_path_the_index_never_saw_is_absent_rather_than_neutral(
        self, conn: sqlite3.Connection
    ) -> None:
        # Ranking has to decide what an unjudged file is worth; inventing a row here would make
        # "nobody said" and "this file does not exist" the same answer.
        assert fetch_file_metadata(conn, ["ghost.md"]) == {}
        assert fetch_file_metadata(conn, []) == {}
        conn.close()

    def test_an_edge_is_a_hop_from_either_end(self, conn: sqlite3.Connection) -> None:
        # Which end of a relation holds the wikilink is a fact about who wrote first, not about
        # which file is relevant to which.
        neighbours = fetch_neighbors(conn, ["hub.md", "a.md"])

        assert neighbours["hub.md"] == ("a.md", "b.md")
        assert neighbours["a.md"] == ("hub.md",)
        conn.close()

    def test_a_pending_link_names_no_neighbour(self, conn: sqlite3.Connection) -> None:
        # `nowhere.md` is not in the vault, so the link sits unresolved. Boosting a path that
        # matches no file would put a result in the list that nobody can open.
        assert "nowhere.md" not in fetch_neighbors(conn, ["hub.md"])["hub.md"]
        conn.close()

    def test_a_file_that_links_to_itself_is_not_its_own_neighbour(
        self, conn: sqlite3.Connection
    ) -> None:
        # Self-boosting is a ranking bug wearing a wikilink: it would let any file promote itself
        # by naming itself, and it also inflates the out-degree the hub guard divides by.
        assert "hub.md" not in fetch_neighbors(conn, ["hub.md"])["hub.md"]
        conn.close()

    def test_a_file_reached_only_through_the_graph_is_represented_by_its_opening(
        self, conn: sqlite3.Connection
    ) -> None:
        # Such a file matched nothing itself, so it has no winning chunk. Its beginning is the
        # closest thing to a summary that costs no model call.
        rows = fetch_leading_chunks(conn, ["b.md", "a.md"])

        assert [row.path for row in rows] == ["b.md", "a.md"]
        assert all(row.ordinal == 0 for row in rows)
        conn.close()

    def test_leading_chunks_skip_paths_with_nothing_indexed(self, conn: sqlite3.Connection) -> None:
        assert fetch_leading_chunks(conn, ["ghost.md"]) == []
        assert fetch_leading_chunks(conn, []) == []
        conn.close()


class TestIndexingWithoutTheSemanticExtra:
    """A bare install still gets a complete keyword index (R12).

    The premise these tests defend: refusing to index without `fastembed` would leave a light
    install with no database at all, so `recall` would fail on a missing index rather than
    answer with the half that needs no model.
    """

    VAULT = {
        "notes/erfpacht.md": "# Erfpacht\n\nThe canon is bought off for fifty years.\n",
        "notes/poem.md": "# Defter\n\nBir dize daha.\n",
    }

    def test_chunks_are_written_with_no_embedding_at_all(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, self.VAULT)
        config = index_config(tmp_path)

        stats = build_index(vault, config, NullEmbedder())

        conn = open_index(stats.db_path)
        try:
            stored = conn.execute("SELECT COUNT(*), COUNT(embedding) FROM chunks").fetchone()
            assert stored[0] == stats.chunks_written > 0
            # Every row present, not one vector among them — the column is nullable and this is
            # the state the whole degrade rests on.
            assert stored[1] == 0
            assert has_embeddings(conn) is False
            assert load_embeddings(conn)[0] == []
        finally:
            conn.close()

    def test_keyword_search_works_against_a_null_embedded_index(self, tmp_path: Path) -> None:
        # The FTS triggers fire on the chunk insert, not on the embedding, so the keyword half is
        # not degraded at all — it is complete.
        vault = make_vault(tmp_path, self.VAULT)
        config = index_config(tmp_path)
        stats = build_index(vault, config, NullEmbedder())

        conn = open_index(stats.db_path)
        try:
            hits = fetch_chunks(conn, match_chunks(conn, escape_fts_query("erfpacht")))
        finally:
            conn.close()

        assert [row.path for row in hits] == ["notes/erfpacht.md"]

    def test_nothing_is_counted_as_embedded(self, tmp_path: Path) -> None:
        # `chunks_embedded` is what the operator reads to know a pass did model work. Counting
        # the null answers would report a model that never ran.
        vault = make_vault(tmp_path, self.VAULT)

        stats = build_index(vault, index_config(tmp_path), NullEmbedder())

        assert stats.chunks_embedded == 0
        assert stats.chunks_written > 0

    def test_the_graph_is_built_exactly_as_it_would_be_with_a_model(self, tmp_path: Path) -> None:
        # Observations, relations and frontmatter are parsed from the files and have nothing to
        # do with embeddings, so a light install must get the whole graph.
        files = {
            "notes/a.md": "---\nimportance: 8\n---\n\n# A\n\n- part_of [[notes/b]]\n",
            "notes/b.md": "# B\n\n- [decision] we chose the cheap option #cost\n",
        }
        vault = make_vault(tmp_path, files)

        stats = build_index(vault, index_config(tmp_path), NullEmbedder())

        assert stats.links_written == 1
        assert stats.observations_written == 1
        assert link_tuples(stats.db_path) == [("notes/a.md", 0, "part_of", "notes/b", "notes/b.md")]

    def test_installing_the_extra_later_re_embeds_the_whole_corpus(self, tmp_path: Path) -> None:
        # `null` is a model id like any other, so switching to a real one takes the ordinary
        # wipe-and-rebuild road. Reusing a NULL "vector" is not a thing that could be attempted,
        # and the snapshot must not offer one.
        vault = make_vault(tmp_path, self.VAULT)
        config = index_config(tmp_path)
        build_index(vault, config, NullEmbedder())

        embedder = StubEmbedder()
        stats = build_index(vault, config, embedder)

        assert stats.chunks_embedded == stats.chunks_written > 0
        conn = open_index(stats.db_path)
        try:
            assert has_embeddings(conn) is True
        finally:
            conn.close()

    def test_dropping_the_extra_clears_the_recorded_vector_width(self, tmp_path: Path) -> None:
        # A stale `embedding_dim` would have the index claiming a width for vectors it no longer
        # holds, which is exactly the metadata a later query would trust over the rows.
        vault = make_vault(tmp_path, self.VAULT)
        config = index_config(tmp_path)
        build_index(vault, config, StubEmbedder())

        stats = build_index(vault, config, NullEmbedder())

        conn = open_index(stats.db_path)
        try:
            assert embedding_dimension(conn) == 0
        finally:
            conn.close()


class TestSemanticAvailability:
    """`semantic_available` is the one question the two embedder factories ask."""

    def test_it_reports_the_absence_of_the_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Asked with `find_spec` so the answer costs no onnxruntime import on the many calls
        # that never reach a model.
        monkeypatch.setattr(index_module.importlib.util, "find_spec", lambda _name: None)

        assert semantic_available() is False

    def test_it_reports_the_presence_of_the_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(index_module.importlib.util, "find_spec", lambda _name: object())

        assert semantic_available() is True
