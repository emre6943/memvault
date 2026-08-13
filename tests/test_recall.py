"""Hybrid recall: what each mode contributes, and what the fusion does with both.

The theme of these tests: a retrieval result must be explainable. Every case here names which
mode should have found a chunk and why it should place where it does, so a regression in the
fusion shows up as a specific reordering rather than as "the numbers changed".

No test touches the network or a real model. `ConceptEmbedder` maps text onto a handful of
declared topic axes, which makes semantic similarity something a fixture states outright:
unrelated text is orthogonal, a paraphrase shares an axis with no shared words, and a query
about nothing in the vault embeds to the zero vector. Those are the three geometries the
retrieval rules turn on, and a hash-based stub has none of them.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import pytest

from memvault import recall as recall_module
from memvault.cli import main
from memvault.config import IndexConfig, RankingConfig, VaultConfig
from memvault.index import (
    NULL_MODEL_ID,
    SEMANTIC_EXTRA_HINT,
    NullEmbedder,
    build_index,
    escape_fts_query,
    match_chunks,
    open_index,
)
from memvault.recall import (
    KEYWORD_ONLY_INDEX_HINT,
    RRF_K,
    RecallError,
    RecallResponse,
    build_fts_query,
    file_meta,
    format_results,
    importance_term,
    query_embedder,
    recall,
    recency_term,
)

#: Topic axes. A text's vector counts the terms of each axis it contains, so a file about one
#: topic is a unit vector on that axis and a file about two sits between them.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "housing": ("erfpacht", "canon", "leasehold", "grondpacht", "vve", "woz"),
    "medical": ("zolgensma", "dose", "clinic"),
    "poetry": ("şiir", "dize", "defter", "poem"),
}
AXES = tuple(CONCEPTS)


class ConceptEmbedder:
    """A deterministic embedder whose geometry a fixture can state in prose.

    Text unrelated to every axis embeds to zero, which is what makes "this query matches
    nothing" a real state rather than a threshold nobody can justify.
    """

    def __init__(self, model_id: str = "concept-v1") -> None:
        self.model_id = model_id
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        return [float(sum(lowered.count(term) for term in CONCEPTS[axis])) for axis in AXES]


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


def indexed(
    tmp_path: Path, files: dict[str, str]
) -> tuple[VaultConfig, IndexConfig, ConceptEmbedder]:
    vault = make_vault(tmp_path, files)
    config = index_config(tmp_path)
    build_index(vault, config, ConceptEmbedder())
    return vault, config, ConceptEmbedder()


def paths(response: RecallResponse) -> list[str]:
    return [result.path for result in response.results]


FILLER = {
    "notes/filler-a.md": "# Filler A\n\nA note about nothing in particular at all.\n",
    "notes/filler-b.md": "# Filler B\n\nAnother unremarkable paragraph of prose.\n",
    "notes/filler-c.md": "# Filler C\n\nGroceries, laundry, and a haircut.\n",
}

#: Ranking facts are ages, so every fixture that cares about recency is written relative to the
#: day the suite runs. A hardcoded date would decay a little further every week until a test that
#: passed on the day it was written failed for reasons nobody changed.
TODAY = date.today()

#: The bridge in `RankingConfig`'s docstring: with the three new terms at zero, recall must score
#: exactly what it scored before this plan, from exactly the same candidates.
PURE_RRF = RankingConfig(importance=0.0, recency=0.0, graph=0.0)


def note(
    body: str,
    *,
    importance: int | None = None,
    day: date | None = None,
    kind: str | None = None,
    superseded_by: str | None = None,
    relations: Sequence[tuple[str, str]] = (),
) -> str:
    """A vault file carrying the frontmatter and relation lines ranking reads.

    Written as text rather than built through the writer so a test states the file's ranking
    facts where a reader can see them, and so a hand-edited `importance:` is exercised the same
    way a classifier-written one is.
    """
    header: list[str] = []
    if kind is not None:
        header.append(f"kind: {kind}")
    if importance is not None:
        header.append(f"importance: {importance}")
    if day is not None:
        header.append(f"date: {day.isoformat()}")
    if superseded_by is not None:
        header.append(f"superseded_by: {superseded_by}")

    front = "---\n" + "\n".join(header) + "\n---\n\n" if header else ""
    tail = ""
    if relations:
        tail = "\n## Relations\n\n" + "".join(
            f"- {predicate} [[{target}]]\n" for predicate, target in relations
        )
    return f"{front}{body}{tail}"


def days_ago(count: int) -> date:
    return TODAY - timedelta(days=count)


def cited(rendered: str) -> list[str]:
    """The paths a rendered result block actually cites, ignoring text inside the snippets.

    A snippet can quote a relation line naming another file, so searching the whole output for a
    filename answers a different question than "did this file place".
    """
    lines = [line.strip() for line in rendered.splitlines()]
    return [line.split(". ", 1)[1] for line in lines if line[:1].isdigit() and ". " in line]


def rrf_score(*ranks: int) -> float:
    """What pure reciprocal rank fusion pays for these per-mode ranks, in scoring order."""
    total = 0.0
    for rank in ranks:
        total += 1.0 / (RRF_K + rank)
    return total


class TestKeywordSide:
    """The rare exact token is the thing embeddings smooth away."""

    files = {
        **FILLER,
        "research/rare.md": ("# Rare\n\nThe zolgensma dossier, filed beside a şiir and a dize.\n"),
        "clinic/strong-a.md": "# Dosing\n\nOne dose, then a second dose at the clinic.\n",
        "clinic/strong-b.md": "# More dosing\n\nA dose given at the clinic, then a dose.\n",
    }

    def test_a_rare_exact_term_ranks_first_despite_weak_semantic_similarity(
        self, tmp_path: Path
    ) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "zolgensma")

        # `research/rare.md` is the only chunk carrying the token, and it is the *weakest* of
        # the three semantic hits — the clinic notes are pure `medical`, it is two-thirds
        # `poetry`. Placing it first is the keyword side doing its job.
        assert paths(response)[0] == "research/rare.md"
        assert response.results[0].keyword_rank == 1
        assert response.results[0].semantic_rank == 3
        assert response.results[0].similarity < response.results[1].similarity

    def test_a_query_of_pure_punctuation_returns_nothing_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "   !!!   ")

        assert response.results == []
        assert "nothing in the vault" in response.message


class TestSemanticSide:
    """A paraphrase shares no words with the question, which is the whole point."""

    files = {
        **FILLER,
        "realtor/CLAUDE.md": "# Erfpacht\n\nAmsterdam ground rent, explained end to end.\n",
        "research/drugs.md": "# Drugs\n\nA dose at the clinic.\n",
    }

    def test_a_paraphrase_with_no_shared_terms_is_still_retrieved(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "leasehold canon obligations")

        assert "realtor/CLAUDE.md" in paths(response)
        found = next(r for r in response.results if r.path == "realtor/CLAUDE.md")
        assert found.keyword_rank is None
        assert found.semantic_rank == 1

    def test_the_keyword_side_alone_would_have_found_nothing(self, tmp_path: Path) -> None:
        vault, _, _ = indexed(tmp_path, self.files)

        conn = open_index(tmp_path / "db" / "personal.db")
        try:
            assert match_chunks(conn, build_fts_query("leasehold canon obligations")) == []
        finally:
            conn.close()
        assert vault.root.exists()


class TestFusion:
    """Reciprocal rank fusion exists to make "both modes agree" mean something."""

    files = {
        **FILLER,
        "both.md": "# Both\n\nThe erfpacht canon is paid yearly.\n",
        "keyword.md": "# Keyword\n\nErfpacht in passing, then a dose, a dose, and a clinic.\n",
        "semantic.md": "# Semantic\n\nThe leasehold canon obligations run with the property.\n",
    }

    def test_a_chunk_both_modes_found_outranks_a_chunk_either_found(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht")

        assert paths(response)[:3] == ["both.md", "keyword.md", "semantic.md"]
        by_path = {result.path: result for result in response.results}
        assert by_path["both.md"].modes == "keyword+semantic"
        assert by_path["keyword.md"].keyword_rank is not None
        assert by_path["semantic.md"].keyword_rank is None
        assert by_path["both.md"].score > by_path["keyword.md"].score
        assert by_path["keyword.md"].score > by_path["semantic.md"].score

    def test_the_limit_caps_the_number_of_citations(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        assert len(recall(vault, config, embedder, "erfpacht", limit=2).results) == 2

    def test_a_non_positive_limit_is_rejected_rather_than_returning_everything(
        self, tmp_path: Path
    ) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        with pytest.raises(RecallError, match="limit must be positive"):
            recall(vault, config, embedder, "erfpacht", limit=0)


class TestCitations:
    """R19: a result is only useful if the session can open what it points at."""

    files = {
        **FILLER,
        "realtor/CLAUDE.md": "# Erfpacht\n\n## Costs\n\nThe canon is 1200 per year.\n",
        "transcripts/2026/07/2026-07-12-viewing.md": (
            "---\nsource: whatsapp\ndate: 2026-07-12\n---\n\n"
            "# Viewing\n\nWe talked about the leasehold and the vve reserve.\n"
        ),
    }

    def test_every_result_cites_a_vault_relative_path_that_exists_on_disk(
        self, tmp_path: Path
    ) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon")

        assert response.results != []
        for result in response.results:
            assert not Path(result.path).is_absolute()
            assert (vault.root / result.path).exists()

    def test_a_result_carries_its_heading_breadcrumb_and_a_snippet(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "canon")

        top = next(r for r in response.results if r.path == "realtor/CLAUDE.md")
        assert top.heading == "Erfpacht > Costs"
        assert "1200 per year" in top.snippet

    def test_a_long_chunk_is_snipped_around_the_query_term(self, tmp_path: Path) -> None:
        padding = "Filler sentence with no bearing on anything. " * 30
        vault, config, embedder = indexed(
            tmp_path, {**FILLER, "long.md": f"# Long\n\n{padding}The erfpacht canon matters.\n"}
        )

        response = recall(vault, config, embedder, "erfpacht")

        snippet = response.results[0].snippet
        assert "erfpacht" in snippet.casefold()
        assert len(snippet) < len(padding)
        assert snippet.startswith("…")

    def test_the_rendered_output_leads_with_paths_and_tells_the_reader_to_open_them(
        self, tmp_path: Path
    ) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        rendered = format_results(recall(vault, config, embedder, "erfpacht canon"))

        assert "1. realtor/CLAUDE.md" in rendered
        assert "Read the files above for depth" in rendered


class TestNoMatch:
    """A query about nothing must say so — not answer with its nearest neighbour."""

    files = {**FILLER, "realtor/CLAUDE.md": "# Erfpacht\n\nThe canon is paid yearly.\n"}

    def test_a_query_matching_nothing_returns_an_empty_list_and_a_clear_message(
        self, tmp_path: Path
    ) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "quantum chromodynamics")

        assert response.results == []
        assert "quantum chromodynamics" in response.message
        assert "memvault index" in response.message
        assert response.chunks_searched > 0

    def test_the_message_is_what_a_caller_sees_when_there_is_nothing_to_render(
        self, tmp_path: Path
    ) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "quantum chromodynamics")

        assert format_results(response) == response.message

    def test_an_empty_index_names_the_command_that_would_fill_it(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, {})

        response = recall(vault, config, embedder, "erfpacht")

        assert response.results == []
        assert "memvault index" in response.message

    def test_a_missing_index_is_an_error_rather_than_an_empty_answer(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"a.md": "# A\n\nBody.\n"})

        with pytest.raises(RecallError, match="no search index"):
            recall(vault, index_config(tmp_path), ConceptEmbedder(), "erfpacht")

    def test_an_index_built_by_another_model_refuses_to_answer(self, tmp_path: Path) -> None:
        vault, config, _ = indexed(tmp_path, self.files)

        with pytest.raises(RecallError, match="not comparable"):
            recall(vault, config, ConceptEmbedder("concept-v2"), "erfpacht")


class TestFilters:
    """ "Transcripts from last month" must not need a second tool."""

    files = {
        **FILLER,
        "realtor/CLAUDE.md": (
            "---\nsource: mode\ndate: 2026-05-04\n---\n\n"
            "# Erfpacht\n\nThe canon and the vve, in general.\n"
        ),
        "transcripts/2026/07/2026-07-12-viewing.md": (
            "---\nsource: whatsapp\ndate: 2026-07-12\n---\n\n"
            "# Viewing\n\nThe erfpacht canon on this one is bought off.\n"
        ),
        "transcripts/2025/01/2025-01-05-old.md": (
            "---\nsource: whatsapp\ndate: 2025-01-05\n---\n\n"
            "# Old viewing\n\nAn erfpacht canon question from long ago.\n"
        ),
    }

    def test_a_since_window_excludes_older_material(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon", since=date(2026, 6, 1))

        assert paths(response) == ["transcripts/2026/07/2026-07-12-viewing.md"]

    def test_an_until_window_excludes_newer_material(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon", until=date(2025, 12, 31))

        assert paths(response) == ["transcripts/2025/01/2025-01-05-old.md"]

    def test_a_window_with_no_material_in_it_says_the_filter_is_why(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(
            vault,
            config,
            embedder,
            "erfpacht canon",
            since=date(2020, 1, 1),
            until=date(2020, 12, 31),
        )

        assert response.results == []
        assert "within the filters" in response.message
        assert "2020-01-01" in response.message

    def test_a_source_filter_matches_the_vault_section(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon", sources=("transcripts",))

        # Order is not asserted here: both survivors are transcripts and recency decides between
        # them, which is the ranking's business rather than the filter's.
        assert set(paths(response)) == {
            "transcripts/2026/07/2026-07-12-viewing.md",
            "transcripts/2025/01/2025-01-05-old.md",
        }
        assert "realtor/CLAUDE.md" not in paths(response)

    def test_a_source_filter_matches_the_declared_frontmatter_source(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon", sources=("mode",))

        assert paths(response) == ["realtor/CLAUDE.md"]

    def test_filters_combine(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(
            vault,
            config,
            embedder,
            "erfpacht canon",
            sources=("whatsapp",),
            since=date(2026, 1, 1),
        )

        assert paths(response) == ["transcripts/2026/07/2026-07-12-viewing.md"]


class TestFileMeta:
    """Dates come from the file when it declares one, and from the vault layout when it does not."""

    def test_frontmatter_wins_over_the_path(self, tmp_path: Path) -> None:
        vault = make_vault(
            tmp_path,
            {
                "transcripts/2026/07/2026-07-12-x.md": (
                    "---\nsource: whatsapp\ndate: 2024-02-02\n---\n\n# X\n\nBody.\n"
                )
            },
        )

        meta = file_meta(vault.root, "transcripts/2026/07/2026-07-12-x.md")

        assert meta == type(meta)(source="whatsapp", section="transcripts", date=date(2024, 2, 2))

    def test_a_dated_filename_is_used_when_frontmatter_is_absent(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {"transcripts/2026-07-12-x.md": "# X\n\nBody.\n"})

        meta = file_meta(vault.root, "transcripts/2026-07-12-x.md")

        assert meta.date == date(2026, 7, 12)
        assert meta.source is None

    def test_a_year_month_layout_dates_a_file_to_the_first_of_the_month(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(tmp_path, {"transcripts/2026/03/note.md": "# X\n\nBody.\n"})

        assert file_meta(vault.root, "transcripts/2026/03/note.md").date == date(2026, 3, 1)

    def test_malformed_frontmatter_degrades_instead_of_failing_the_search(
        self, tmp_path: Path
    ) -> None:
        vault = make_vault(tmp_path, {"a.md": "---\nsource: [unclosed\n---\n\n# A\n\nBody.\n"})

        meta = file_meta(vault.root, "a.md")

        assert meta.source is None
        assert meta.section == ""

    def test_a_file_that_vanished_since_indexing_still_resolves(self, tmp_path: Path) -> None:
        vault = make_vault(tmp_path, {})

        assert file_meta(vault.root, "gone/2026-01-01-x.md").date == date(2026, 1, 1)


class TestTurkish:
    """KTD2 picked a multilingual model for this vault; retrieval must honour it."""

    files = {
        **FILLER,
        "poems/defter.md": ("# Şiir Defteri\n\nBir dize yazdım, İstanbul'da yağmur yağarken.\n"),
        "realtor/CLAUDE.md": "# Erfpacht\n\nThe canon is paid yearly.\n",
    }

    def test_a_turkish_query_retrieves_turkish_content(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "şiir defteri dize")

        assert paths(response)[0] == "poems/defter.md"
        assert "yağmur" in response.results[0].snippet

    def test_a_turkish_word_with_an_apostrophe_is_not_read_as_syntax(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "İstanbul'da")

        assert paths(response) == ["poems/defter.md"]


class TestQuerySyntax:
    """Raw user text reaching MATCH is the failure mode `escape_fts_query` exists to prevent."""

    hostile = [
        'erfpacht "unbalanced',
        "erfpacht*",
        "NEAR(erfpacht canon)",
        "-erfpacht",
        'he said "no" then ^left',
        "erfpacht OR canon AND NOT vve",
        "(((",
        "*",
        "erfpacht: canon; vve!",
    ]

    @pytest.mark.parametrize("query", hostile)
    def test_fts_syntax_in_a_query_never_raises(self, tmp_path: Path, query: str) -> None:
        vault, config, embedder = indexed(
            tmp_path, {**FILLER, "realtor/CLAUDE.md": "# Erfpacht\n\nThe canon is yearly.\n"}
        )

        response = recall(vault, config, embedder, query)

        assert isinstance(response.message, str)

    def test_the_words_of_a_query_are_ored_rather_than_anded(self) -> None:
        # ANDing a natural-language question against a vault matches nothing at all, which is
        # the failure that makes a hybrid search feel broken for everything but single words.
        assert build_fts_query("erfpacht canon") == '("erfpacht") OR ("canon")'
        assert build_fts_query("NEAR(a") == '("NEAR" "a")'
        assert build_fts_query("*") == ""

    def test_a_hostile_query_still_matches_what_a_clean_one_would(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(
            tmp_path, {**FILLER, "realtor/CLAUDE.md": "# Erfpacht\n\nThe canon is yearly.\n"}
        )

        assert paths(recall(vault, config, embedder, 'erfpacht "')) == ["realtor/CLAUDE.md"]


class TestDisposableIndex:
    """AE7: the index is a cache, so a rebuild must not change the answer."""

    files = {
        **FILLER,
        "realtor/CLAUDE.md": "# Erfpacht\n\n## Costs\n\nThe canon is 1200 per year.\n",
        "transcripts/2026/07/2026-07-12-viewing.md": (
            "# Viewing\n\nThe leasehold and the vve reserve came up.\n"
        ),
        "research/drugs.md": "# Drugs\n\nOne dose at the clinic.\n",
    }

    def test_rebuilding_the_index_returns_the_same_top_three(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)
        before = recall(vault, config, embedder, "erfpacht canon vve", limit=3)

        config.db_path(vault.name).unlink()
        build_index(vault, config, ConceptEmbedder())
        after = recall(vault, config, ConceptEmbedder(), "erfpacht canon vve", limit=3)

        assert paths(after) == paths(before)
        assert [r.heading for r in after.results] == [r.heading for r in before.results]
        assert [round(r.score, 12) for r in after.results] == [
            round(r.score, 12) for r in before.results
        ]

    def test_a_new_file_is_findable_after_a_reindex_and_not_before(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)
        (vault.root / "new.md").write_text(
            "# New\n\nA hypotheekrenteaftrek note.\n", encoding="utf-8"
        )

        assert paths(recall(vault, config, embedder, "hypotheekrenteaftrek")) == []
        build_index(vault, config, ConceptEmbedder())
        after = recall(vault, config, ConceptEmbedder(), "hypotheekrenteaftrek")
        assert paths(after) == ["new.md"]


class TestRankingTerms:
    """The two scalar terms, on their own, before any fusion can hide what they do."""

    def test_an_unjudged_file_is_worth_exactly_a_five(self) -> None:
        # A corpus filed before `importance:` existed must rank as unjudged, not as unimportant —
        # otherwise the day the key ships, every old file in the vault sinks at once.
        assert importance_term(None) == importance_term(5)

    def test_the_band_runs_from_a_one_to_a_ten(self) -> None:
        assert (importance_term(1), importance_term(10)) == (0.0, 1.0)

    def test_a_score_outside_the_band_is_clamped_rather_than_trusted(self) -> None:
        # A hand-edited `importance: 99` is a typo, and a term above 1.0 would out-shout the
        # fusion it is supposed to nudge.
        assert importance_term(99) == importance_term(10)
        assert importance_term(-4) == importance_term(1)

    def test_a_file_one_half_life_old_is_worth_half_of_a_file_written_today(self) -> None:
        fresh = recency_term(TODAY, today=TODAY, half_life_days=180.0)
        aged = recency_term(days_ago(180), today=TODAY, half_life_days=180.0)

        assert fresh == 1.0
        assert aged == pytest.approx(0.5)

    def test_a_date_in_the_future_is_not_worth_more_than_today(self) -> None:
        # Frontmatter dates are declared, and a drop about next week's viewing carries next
        # week's date. Uncapped, `0.5 ** negative` would make a plan outrank everything filed.
        assert recency_term(TODAY + timedelta(days=90), today=TODAY, half_life_days=180.0) == 1.0

    def test_a_file_the_vault_cannot_date_makes_no_recency_claim(self) -> None:
        # `file_meta` falls back to frontmatter, then the path, then the mtime, so this is the
        # state of a file that is no longer on disk. Guessing a neutral age for it would invent
        # the fact the fallbacks failed to find.
        assert recency_term(None, today=TODAY, half_life_days=180.0) == 0.0


class TestBlendedRanking:
    """R4: what a file is worth, not only how well its words matched."""

    files = {
        **FILLER,
        "old-important.md": note(
            "# Erfpacht buyout\n\nThe erfpacht canon buyout, decided and paid.\n",
            importance=9,
            day=days_ago(730),
        ),
        "recent-trivial.md": note(
            "# Erfpacht mention\n\nThe erfpacht canon came up in passing again.\n",
            importance=2,
            day=TODAY,
        ),
    }

    def test_an_old_important_note_outranks_a_fresh_trivial_one_at_default_weights(
        self, tmp_path: Path
    ) -> None:
        # The calibration this plan is judged on: a two-year-old decision beats this week's
        # small talk, because someone said one of them mattered.
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon")

        assert paths(response)[:2] == ["old-important.md", "recent-trivial.md"]

    def test_between_equally_important_notes_the_recent_one_wins(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(
            tmp_path,
            {
                **FILLER,
                "stale.md": note(
                    "# Canon\n\nThe erfpacht canon, as it stood.\n", importance=5, day=days_ago(400)
                ),
                "fresh.md": note(
                    "# Canon\n\nThe erfpacht canon, as it stands.\n", importance=5, day=TODAY
                ),
            },
        )

        response = recall(vault, config, embedder, "erfpacht canon")

        assert paths(response)[:2] == ["fresh.md", "stale.md"]

    def test_a_file_with_no_importance_is_ranked_as_a_five_rather_than_as_a_one(
        self, tmp_path: Path
    ) -> None:
        # Same body, same day, so the only thing that can separate them is the declared score.
        body = "# Canon\n\nThe erfpacht canon, recorded.\n"
        vault, config, embedder = indexed(
            tmp_path,
            {
                **FILLER,
                "unjudged.md": note(body, day=TODAY),
                "trivial.md": note(body, importance=1, day=TODAY),
                "vital.md": note(body, importance=10, day=TODAY),
            },
        )

        response = recall(vault, config, embedder, "erfpacht canon")

        by_path = {result.path: result for result in response.results}
        assert by_path["trivial.md"].score < by_path["unjudged.md"].score
        assert by_path["unjudged.md"].score < by_path["vital.md"].score


class TestSupersession:
    """R5: a note that says it was replaced must not keep winning the query it used to."""

    files = {
        **FILLER,
        "old-canon.md": note(
            "# Erfpacht canon\n\nThe canon figure, before the buyout.\n",
            importance=8,
            superseded_by="new-canon.md",
        ),
        "new-canon.md": note("# Erfpacht canon\n\nThe canon figure, after the buyout.\n"),
    }

    def test_a_superseded_note_drops_below_its_successor(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon")

        assert paths(response).index("new-canon.md") < paths(response).index("old-canon.md")

    def test_the_result_carries_the_successor_so_a_reader_can_follow_it(
        self, tmp_path: Path
    ) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon")

        by_path = {result.path: result for result in response.results}
        assert by_path["old-canon.md"].superseded_by == "new-canon.md"
        assert by_path["new-canon.md"].superseded_by is None

    def test_the_rendered_output_says_so_rather_than_only_ranking_lower(
        self, tmp_path: Path
    ) -> None:
        # Rank alone is not an explanation: a reader who opens the second result should know it
        # was replaced before they act on what it says.
        vault, config, embedder = indexed(tmp_path, self.files)

        rendered = format_results(recall(vault, config, embedder, "erfpacht canon"))

        assert "superseded by new-canon.md" in rendered


class TestGraphHop:
    """R2: one relation hop from a strong hit, with the hub guards that keep it honest."""

    files = {
        **FILLER,
        "hit.md": note(
            "# Erfpacht\n\nThe erfpacht canon, in full.\n",
            relations=[("part_of", "neighbour.md")],
        ),
        "neighbour.md": "# Ground rent history\n\nA note sharing no words with the query.\n",
        "control.md": "# Unrelated history\n\nAnother note sharing no words with the query.\n",
    }

    def test_a_file_one_hop_from_a_hit_enters_the_results(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon")

        # `control.md` is the proof that the hop is what pulled the neighbour in: the two are
        # indistinguishable to both retrieval modes and differ only by the relation line.
        assert "neighbour.md" in paths(response)
        assert "control.md" not in paths(response)

    def test_the_neighbour_is_labelled_as_reached_through_the_graph(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon")

        by_path = {result.path: result for result in response.results}
        assert by_path["neighbour.md"].modes == "graph"
        assert by_path["hit.md"].graph_neighbour is False

    def test_a_hub_lifts_each_of_its_neighbours_less_than_a_focused_note_does(
        self, tmp_path: Path
    ) -> None:
        # A note that links to everything says nothing about any one of them. Without the
        # out-degree discount, an index page would decide half of every query.
        hub_targets = {
            f"hub/target-{n:02d}.md": f"# Target {n}\n\nRoutine paragraph {n}.\n" for n in range(30)
        }
        files = {
            **FILLER,
            **hub_targets,
            "hub.md": note(
                "# Erfpacht index\n\nThe erfpacht canon index page.\n",
                relations=[("lists", target) for target in sorted(hub_targets)],
            ),
            "focused.md": note(
                "# Erfpacht note\n\nThe erfpacht canon, one thought.\n",
                relations=[("part_of", f"hub/target-{n:02d}.md") for n in range(3)],
            ),
        }
        vault, config, embedder = indexed(tmp_path, files)

        response = recall(vault, config, embedder, "erfpacht canon", limit=60)

        by_path = {result.path: result for result in response.results}
        # `target-02` is reached from both the hub and the three-link note; `target-05` only from
        # the hub. Same body, same day, so the gap between them is the out-degree discount.
        assert by_path["hub/target-05.md"].score < by_path["hub/target-02.md"].score

    def test_a_reflection_note_boosts_nothing_it_cites(self, tmp_path: Path) -> None:
        # A reflection note cites everything it reflected on. Left as a boost source it would
        # radiate "recently reflected on" across files with nothing else in common.
        files = {
            **FILLER,
            "reflection.md": note(
                "# Weekly reflection\n\nThe erfpacht canon came up this week.\n",
                kind="reflection",
                relations=[("cites", "cited.md")],
            ),
            "plain.md": note(
                "# Erfpacht\n\nThe erfpacht canon, plainly.\n",
                relations=[("cites", "also-cited.md")],
            ),
            "cited.md": "# One\n\nA note sharing no words with the query.\n",
            "also-cited.md": "# Two\n\nAnother note sharing no words with the query.\n",
        }
        vault, config, embedder = indexed(tmp_path, files)

        response = recall(vault, config, embedder, "erfpacht canon")

        assert "also-cited.md" in paths(response)
        assert "cited.md" not in paths(response)


class TestZeroWeightBridge:
    """The regression bridge: zeroed weights must reproduce pre-v2 recall exactly."""

    files = {
        **FILLER,
        "hit.md": note(
            "# Erfpacht\n\nThe erfpacht canon, in full.\n",
            importance=10,
            day=days_ago(900),
            relations=[("part_of", "neighbour.md")],
        ),
        "neighbour.md": "# Ground rent history\n\nA note sharing no words with the query.\n",
        "also.md": note("# Canon\n\nThe canon, mentioned once.\n", importance=1, day=TODAY),
    }

    def test_the_scores_are_pure_reciprocal_rank_fusion(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon", ranking=PURE_RRF)

        assert response.results != []
        for result in response.results:
            ranks = [
                rank for rank in (result.keyword_rank, result.semantic_rank) if rank is not None
            ]
            assert result.score == rrf_score(*ranks)

    def test_the_candidate_pool_is_not_expanded_through_the_graph(self, tmp_path: Path) -> None:
        # Pool expansion is gated on the graph weight, so the bridge reproduces today's
        # *candidates*, not merely today's arithmetic over a wider set.
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon", limit=50, ranking=PURE_RRF)

        assert "neighbour.md" not in paths(response)
        assert all(result.modes != "graph" for result in response.results)

    def test_every_candidate_under_the_bridge_was_returned_by_a_retrieval_mode(
        self, tmp_path: Path
    ) -> None:
        # The pool claim stated positively, so it holds for a vault whose relations this test does
        # not know about: with the graph off, nothing may reach the results except through FTS or
        # cosine, and the difference against the default weights is exactly the graph-only files.
        vault, config, embedder = indexed(tmp_path, self.files)

        bridged = recall(vault, config, embedder, "erfpacht canon", limit=50, ranking=PURE_RRF)
        blended = recall(vault, config, embedder, "erfpacht canon", limit=50)

        assert bridged.results != []
        assert all(
            (result.keyword_rank, result.semantic_rank) != (None, None)
            for result in bridged.results
        )
        graph_only = {result.path for result in blended.results if result.modes == "graph"}
        assert set(paths(blended)) - graph_only == set(paths(bridged))

    def test_default_weights_do_expand_it(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "erfpacht canon", limit=50)

        assert "neighbour.md" in paths(response)

    def test_rebuilding_the_index_leaves_the_blended_ordering_identical(
        self, tmp_path: Path
    ) -> None:
        # AE7 extended to the new signals: importance, recency and the graph are all read out of
        # rebuilt tables, so a rebuild is the test that none of them depends on row ids.
        vault, config, embedder = indexed(tmp_path, self.files)
        before = recall(vault, config, embedder, "erfpacht canon", limit=10)

        config.db_path(vault.name).unlink()
        build_index(vault, config, ConceptEmbedder())
        after = recall(vault, config, ConceptEmbedder(), "erfpacht canon", limit=10)

        assert paths(after) == paths(before)
        assert [round(r.score, 12) for r in after.results] == [
            round(r.score, 12) for r in before.results
        ]

    def test_a_query_matching_nothing_stays_empty_however_the_weights_are_set(
        self, tmp_path: Path
    ) -> None:
        # The graph must not answer a question the vault has nothing on: with no hits there is
        # no source to hop from, and an empty answer stays an answer.
        vault, config, embedder = indexed(tmp_path, self.files)

        response = recall(vault, config, embedder, "quantum chromodynamics")

        assert response.results == []
        assert "nothing in the vault" in response.message


class TestCliWiring:
    """The subcommand is what the skill actually calls."""

    files = {
        **FILLER,
        "realtor/CLAUDE.md": (
            "---\nsource: mode\ndate: 2026-05-04\n---\n\n"
            "# Erfpacht\n\nThe canon is 1200 per year.\n"
        ),
    }

    @pytest.fixture(autouse=True)
    def stub_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the CLI off the real backend — it downloads ~130MB on first use."""
        monkeypatch.setattr("memvault.cli.query_embedder", lambda _config: ConceptEmbedder())

    def write_config(self, tmp_path: Path, vault: VaultConfig, extra: str = "") -> Path:
        (vault.root / ".git").mkdir(exist_ok=True)
        (vault.root / "inbox").mkdir(exist_ok=True)
        path = tmp_path / "memvault.config.yaml"
        path.write_text(
            "vaults:\n"
            "  personal:\n"
            f"    root: {vault.root}\n"
            "index:\n"
            f"  path: {tmp_path / 'db' / '{vault}.db'}\n" + extra,
            encoding="utf-8",
        )
        return path

    def test_recall_prints_cited_paths_and_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        vault, _, _ = indexed(tmp_path, self.files)
        config_path = self.write_config(tmp_path, vault)

        code = main(["--config", str(config_path), "recall", "erfpacht", "--limit", "3"])

        out = capsys.readouterr().out
        assert code == 0
        assert "realtor/CLAUDE.md" in out
        assert "Read the files above for depth" in out

    def test_a_no_match_query_exits_zero_with_the_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        vault, _, _ = indexed(tmp_path, self.files)
        config_path = self.write_config(tmp_path, vault)

        code = main(["--config", str(config_path), "recall", "quantum chromodynamics"])

        assert code == 0
        assert "nothing in the vault" in capsys.readouterr().out

    def test_filter_flags_reach_the_search(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        vault, _, _ = indexed(tmp_path, self.files)
        config_path = self.write_config(tmp_path, vault)

        code = main(
            [
                "--config",
                str(config_path),
                "recall",
                "erfpacht",
                "--source",
                "transcripts",
                "--since",
                "2026-01-01",
            ]
        )

        assert code == 0
        assert "within the filters" in capsys.readouterr().out

    def test_a_malformed_date_flag_is_rejected_before_any_search(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        vault, _, _ = indexed(tmp_path, self.files)
        config_path = self.write_config(tmp_path, vault)

        code = main(["--config", str(config_path), "recall", "erfpacht", "--since", "last week"])

        assert code == 1
        assert "--since must be a YYYY-MM-DD date" in capsys.readouterr().err

    def test_the_configured_ranking_weights_reach_the_search(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The weights are only a config surface if the command actually reads them: with the
        # graph weight zeroed, the neighbour the default config surfaces must disappear.
        files = {
            **FILLER,
            "hit.md": note(
                "# Erfpacht\n\nThe erfpacht canon, in full.\n",
                relations=[("part_of", "neighbour.md")],
            ),
            "neighbour.md": "# Ground rent history\n\nA note sharing no words with the query.\n",
        }
        vault, _, _ = indexed(tmp_path, files)
        default_config = self.write_config(tmp_path, vault)

        assert main(["--config", str(default_config), "recall", "erfpacht canon"]) == 0
        assert "neighbour.md" in cited(capsys.readouterr().out)

        zeroed = self.write_config(tmp_path, vault, extra="ranking:\n  graph: 0.0\n")
        assert main(["--config", str(zeroed), "recall", "erfpacht canon"]) == 0
        assert "neighbour.md" not in cited(capsys.readouterr().out)

    def test_a_missing_index_reports_the_command_that_builds_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        vault = make_vault(tmp_path, self.files)
        config_path = self.write_config(tmp_path, vault)

        code = main(["--config", str(config_path), "recall", "erfpacht"])

        assert code == 1
        assert "no search index" in capsys.readouterr().err


class TestScale:
    """KTD3 says brute force until it hurts; this is the shape of "does not hurt yet"."""

    def test_a_few_thousand_chunks_search_without_special_handling(self, tmp_path: Path) -> None:
        files = {
            f"notes/{n:04d}.md": f"# Note {n}\n\nRoutine paragraph number {n}.\n"
            for n in range(400)
        }
        files["realtor/CLAUDE.md"] = "# Erfpacht\n\nThe canon is 1200 per year.\n"
        vault, config, embedder = indexed(tmp_path, files)

        response = recall(vault, config, embedder, "erfpacht canon")

        assert paths(response) == ["realtor/CLAUDE.md"]
        assert response.chunks_searched == 401


class TestConcurrentReadOnly:
    """Recall must never write to the index it reads."""

    def test_searching_leaves_the_database_unchanged(self, tmp_path: Path) -> None:
        vault, config, embedder = indexed(
            tmp_path, {**FILLER, "realtor/CLAUDE.md": "# Erfpacht\n\nThe canon.\n"}
        )
        db_path = config.db_path(vault.name)
        conn: sqlite3.Connection = open_index(db_path)
        try:
            before = conn.execute(
                "SELECT id, path, ordinal, content_hash FROM chunks ORDER BY id"
            ).fetchall()
        finally:
            conn.close()

        recall(vault, config, embedder, "erfpacht")

        conn = open_index(db_path)
        try:
            after = conn.execute(
                "SELECT id, path, ordinal, content_hash FROM chunks ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        assert after == before


def test_escape_and_build_agree_on_a_single_word() -> None:
    assert build_fts_query("erfpacht") == f"({escape_fts_query('erfpacht')})"


class TestRecallWithoutTheSemanticHalf:
    """Keyword-only recall, and the two different reasons it happens (R12).

    The theme: half a search must never read like a whole one. Every case here checks both that
    the keyword half still answers and that the response says the other half did not run.
    """

    VAULT = {
        "notes/erfpacht.md": "# Erfpacht\n\nThe canon on this leasehold is bought off.\n",
        "notes/vve.md": "# VvE\n\nThe vve reserve fund is thin.\n",
        **FILLER,
    }

    def test_an_index_with_no_vectors_answers_on_keywords_and_says_so(self, tmp_path: Path) -> None:
        # The install can embed; the index simply was not built with a model. The fix is a
        # re-index, and the message has to name that one rather than an install.
        vault = make_vault(tmp_path, self.VAULT)
        config = index_config(tmp_path)
        build_index(vault, config, NullEmbedder())

        response = recall(vault, config, ConceptEmbedder(), "erfpacht")

        assert paths(response) == ["notes/erfpacht.md"]
        assert response.keyword_only is True
        assert KEYWORD_ONLY_INDEX_HINT in response.message
        assert SEMANTIC_EXTRA_HINT not in response.message

    def test_an_install_that_cannot_embed_answers_on_keywords_and_says_so(
        self, tmp_path: Path
    ) -> None:
        # The mirror image: a fully embedded index, queried by an install with no `fastembed`.
        # Naming the re-index here would send someone to rebuild an index that is already fine.
        vault, config, _ = indexed(tmp_path, self.VAULT)

        response = recall(vault, config, NullEmbedder(), "erfpacht")

        assert paths(response) == ["notes/erfpacht.md"]
        assert response.keyword_only is True
        assert SEMANTIC_EXTRA_HINT in response.message

    def test_a_null_embedder_is_not_read_as_a_model_mismatch(self, tmp_path: Path) -> None:
        # `null` is a model id, so the ordinary "two models are not comparable" guard would fire
        # on every query from a light install and turn the degrade into a hard failure.
        vault, config, _ = indexed(tmp_path, self.VAULT)

        response = recall(vault, config, NullEmbedder(), "erfpacht")

        assert "not comparable" not in response.message

    def test_a_paraphrase_is_the_thing_that_is_lost(self, tmp_path: Path) -> None:
        # The cost of the degrade, stated against its own control: `grondpacht` shares no word
        # with any file, so the keyword half cannot reach it and only the embedded index does.
        embedded_vault, embedded_config, embedder = indexed(tmp_path / "with", self.VAULT)
        assert "notes/erfpacht.md" in paths(
            recall(embedded_vault, embedded_config, embedder, "grondpacht")
        )

        vault = make_vault(tmp_path / "without", self.VAULT)
        config = index_config(tmp_path / "without")
        build_index(vault, config, NullEmbedder())

        response = recall(vault, config, ConceptEmbedder(), "grondpacht")

        assert paths(response) == []
        assert response.keyword_only is True
        assert KEYWORD_ONLY_INDEX_HINT in response.message

    def test_an_empty_result_still_carries_the_warning(self, tmp_path: Path) -> None:
        # The case where silence is most likely to be believed: a caller reads "nothing in the
        # vault" and stops. It must be told that only half the retrieval ran.
        vault = make_vault(tmp_path, self.VAULT)
        config = index_config(tmp_path)
        build_index(vault, config, NullEmbedder())

        response = recall(vault, config, NullEmbedder(), "zolgensma")

        assert response.results == []
        assert "nothing in the vault matches" in response.message
        assert SEMANTIC_EXTRA_HINT in response.message

    def test_a_complete_install_says_nothing_about_any_of_this(self, tmp_path: Path) -> None:
        # The guard against a warning that is always on: with both halves working, neither hint
        # appears and the flag stays false.
        vault, config, embedder = indexed(tmp_path, self.VAULT)

        response = recall(vault, config, embedder, "erfpacht")

        assert response.keyword_only is False
        assert SEMANTIC_EXTRA_HINT not in response.message
        assert KEYWORD_ONLY_INDEX_HINT not in response.message


class TestQueryEmbedderSelection:
    """Which backend a query gets is decided by whether the extra is installed at all."""

    def test_a_bare_install_gets_the_null_backend_rather_than_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Raising here would make every query on a light install a failure, when the keyword half
        # works perfectly well without a model.
        monkeypatch.setattr(recall_module, "semantic_available", lambda: False)

        assert query_embedder(index_config(tmp_path)).model_id == NULL_MODEL_ID

    def test_an_install_with_the_extra_gets_the_real_query_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(recall_module, "semantic_available", lambda: True)
        config = index_config(tmp_path)

        embedder = query_embedder(config)

        assert embedder.model_id == config.model
