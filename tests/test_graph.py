"""Parsing the graph back out of vault Markdown.

The theme of these tests: every edge and every fact the index stores must be recoverable from a
file's own bytes, because the index is disposable and the files are not (R1). So each test here
either proves a line in a file becomes a typed row, or proves that something which merely *looks*
like one — a checkbox, a heading inside a fence, a Markdown link — does not.

Resolution is tested as a pure function over a corpus snapshot rather than against a database:
whether a wikilink names anything is a whole-corpus question, and stating it in one place is what
lets the indexer recompute it globally without re-reading a single file.
"""

from __future__ import annotations

from memvault.classify import Relation
from memvault.graph import (
    FileMetadata,
    Link,
    Observation,
    build_alias_index,
    clean_target,
    parse_body,
    parse_file,
    parse_metadata,
    resolve_link,
    split_frontmatter,
)
from memvault.writer import render_document, render_relations


def body_of(*lines: str) -> str:
    return "\n".join(lines) + "\n"


class TestObservations:
    def test_a_typed_line_parses_into_category_text_and_tags(self) -> None:
        links, observations = parse_body(body_of("- [decision] Went with erfpacht buyout #house"))

        assert links == ()
        assert observations == (
            Observation(
                category="decision", text="Went with erfpacht buyout #house", tags=("house",)
            ),
        )

    def test_the_text_is_kept_as_written_so_a_result_renders_the_authors_sentence(self) -> None:
        _, observations = parse_body(body_of("- [fact] Rate #mortgage was 3.9% in August"))

        assert observations[0].text == "Rate #mortgage was 3.9% in August"
        assert observations[0].tags == ("mortgage",)

    def test_a_numeric_hash_is_a_number_not_a_tag(self) -> None:
        """`#2026-08` and `#3` are how people write months and counts, not how they tag."""
        _, observations = parse_body(body_of("- [fact] Filed in #2026-08, issue #3, tagged #real"))

        assert observations[0].tags == ("real",)

    def test_a_checkbox_is_a_task_not_an_observation(self) -> None:
        """`- [x] done` and `- [ ] todo` are Markdown, and a one-letter category is not a type."""
        _, observations = parse_body(body_of("- [x] shipped", "- [ ] pending", "* [X] done"))

        assert observations == ()

    def test_a_markdown_link_bullet_is_not_an_observation(self) -> None:
        _, observations = parse_body(body_of("- [the plan](docs/plan.md) is the spec"))

        assert observations == ()

    def test_lines_inside_a_fenced_block_are_not_parsed(self) -> None:
        text = body_of(
            "- [decision] Real one",
            "```markdown",
            "- [decision] Example from the docs",
            "- part_of [[repos/Example]]",
            "```",
            "- [note] Also real",
        )

        links, observations = parse_body(text)

        assert [observation.text for observation in observations] == ["Real one", "Also real"]
        assert links == ()

    def test_a_tilde_fence_hides_lines_too(self) -> None:
        text = body_of("~~~", "- [decision] Inside the fence", "~~~")

        assert parse_body(text)[1] == ()

    def test_an_indented_bullet_still_counts(self) -> None:
        _, observations = parse_body(body_of("  - [decision] Nested under a parent"))

        assert observations[0].category == "decision"

    def test_a_repeated_tag_is_recorded_once_in_order_of_appearance(self) -> None:
        _, observations = parse_body(body_of("- [fact] #gold and #oil and #gold again"))

        assert observations[0].tags == ("gold", "oil")

    def test_turkish_survives_both_the_text_and_its_tags(self) -> None:
        _, observations = parse_body(body_of("- [note] Çiğdem öğle vakti gülümsedi #şiir"))

        assert observations[0].text.endswith("#şiir")
        assert observations[0].tags == ("şiir",)


class TestRelations:
    def test_a_typed_wikilink_line_parses_into_predicate_and_target(self) -> None:
        links, observations = parse_body(body_of("- part_of [[repos/MemVault/learnings.md]]"))

        assert links == (Link(predicate="part_of", target="repos/MemVault/learnings.md"),)
        assert observations == ()

    def test_a_trailing_comment_after_the_link_does_not_stop_it_parsing(self) -> None:
        links, _ = parse_body(body_of("- follows_up [[2026-08-01-memvault]] — the first pass"))

        assert links == (Link(predicate="follows_up", target="2026-08-01-memvault"),)

    def test_relations_keep_their_order_of_appearance(self) -> None:
        links, _ = parse_body(body_of("- about [[gold]]", "- part_of [[research]]", "- x [[y]]"))

        assert [link.predicate for link in links] == ["about", "part_of", "x"]

    def test_a_bare_wikilink_without_a_predicate_is_not_an_edge(self) -> None:
        """Untyped links are prose. The graph carries only what somebody typed deliberately."""
        links, _ = parse_body(body_of("See [[repos/MemVault]] for the details."))

        assert links == ()

    def test_an_uppercase_predicate_is_not_a_predicate(self) -> None:
        links, _ = parse_body(body_of("- See [[repos/MemVault]]"))

        assert links == ()

    def test_an_empty_target_is_dropped_rather_than_stored(self) -> None:
        links, _ = parse_body(body_of("- part_of [[   ]]"))

        assert links == ()


class TestCleanTarget:
    def test_a_display_alias_is_dropped_so_the_link_names_the_file(self) -> None:
        assert clean_target("repos/MemVault|the engine") == "repos/MemVault"

    def test_a_heading_anchor_is_dropped(self) -> None:
        assert clean_target("notes/ranking#weights") == "notes/ranking"

    def test_a_leading_dot_slash_is_not_part_of_the_name(self) -> None:
        assert clean_target("./notes/ranking") == "notes/ranking"

    def test_an_anchor_only_target_keeps_nothing(self) -> None:
        assert clean_target("#weights") == ""


class TestMetadata:
    def test_the_keys_the_writers_emit_are_read_back(self) -> None:
        text = (
            "---\n"
            "title: Erfpacht buyout\n"
            "kind: reflection\n"
            "date: 2026-08-12\n"
            "importance: 8\n"
            "superseded_by: notes/erfpacht-final.md\n"
            "---\n\n"
            "# Body\n"
        )

        parsed = parse_file(text)

        assert parsed.metadata == FileMetadata(
            title="Erfpacht buyout",
            kind="reflection",
            importance=8,
            date="2026-08-12",
            superseded_by="notes/erfpacht-final.md",
        )

    def test_an_importance_outside_the_band_is_no_answer_rather_than_a_clamped_one(self) -> None:
        assert parse_metadata({"importance": 0}).importance is None
        assert parse_metadata({"importance": 99}).importance is None
        assert parse_metadata({"importance": "high"}).importance is None
        assert parse_metadata({"importance": True}).importance is None

    def test_a_numeric_string_importance_is_still_a_number(self) -> None:
        assert parse_metadata({"importance": "7"}).importance == 7

    def test_the_ingested_date_stands_in_when_no_material_date_was_written(self) -> None:
        assert parse_metadata({"ingested": "2026-07-04"}).date == "2026-07-04"

    def test_a_date_key_beats_the_ingested_day(self) -> None:
        meta = parse_metadata({"date": "2026-07-04", "ingested": "2026-08-12"})

        assert meta.date == "2026-07-04"

    def test_malformed_frontmatter_yields_no_metadata_and_keeps_the_body(self) -> None:
        text = "---\ntitle: [unclosed\n---\n\n- [decision] Still parsed.\n"

        parsed = parse_file(text)

        assert parsed.metadata == FileMetadata()
        assert parsed.observations[0].text == "Still parsed."

    def test_a_file_with_no_frontmatter_is_all_body(self) -> None:
        frontmatter, body = split_frontmatter("# Heading\n\n- part_of [[x]]\n")

        assert frontmatter == {}
        assert body.startswith("# Heading")

    def test_frontmatter_lines_never_reach_the_body_parser(self) -> None:
        """A `superseded_by:` value is a key, not a relation — the fence is where body starts."""
        text = "---\nrelations:\n  - part_of [[x]]\n---\n\nBody only.\n"

        assert parse_file(text).links == ()


class TestResolution:
    corpus = (
        ("notes/ranking.md", "Ranking weights"),
        ("repos/MemVault/learnings.md", None),
        ("transcripts/2026/08/2026-08-12-gold.md", "Gold and wars"),
    )

    @property
    def paths(self) -> set[str]:
        return {path for path, _ in self.corpus}

    @property
    def aliases(self) -> dict[str, tuple[str, ...]]:
        return build_alias_index(self.corpus)

    def test_a_vault_relative_path_resolves_to_itself(self) -> None:
        resolved = resolve_link("notes/ranking.md", paths=self.paths, aliases=self.aliases)

        assert resolved == "notes/ranking.md"

    def test_the_extension_may_be_left_off(self) -> None:
        resolved = resolve_link("notes/ranking", paths=self.paths, aliases=self.aliases)

        assert resolved == "notes/ranking.md"

    def test_a_unique_title_resolves(self) -> None:
        resolved = resolve_link("Gold and wars", paths=self.paths, aliases=self.aliases)

        assert resolved == "transcripts/2026/08/2026-08-12-gold.md"

    def test_a_title_match_ignores_case_and_surrounding_space(self) -> None:
        resolved = resolve_link("  gold AND wars ", paths=self.paths, aliases=self.aliases)

        assert resolved == "transcripts/2026/08/2026-08-12-gold.md"

    def test_a_unique_filename_stem_resolves(self) -> None:
        resolved = resolve_link("2026-08-12-gold", paths=self.paths, aliases=self.aliases)

        assert resolved == "transcripts/2026/08/2026-08-12-gold.md"

    def test_a_stem_shared_by_two_files_is_pending_rather_than_a_coin_flip(self) -> None:
        paths = {"a/learnings.md", "b/learnings.md"}
        aliases = build_alias_index([(path, None) for path in sorted(paths)])

        assert resolve_link("learnings", paths=paths, aliases=aliases) is None

    def test_an_exact_path_still_wins_when_its_stem_is_ambiguous(self) -> None:
        corpus = [("a/learnings.md", None), ("b/learnings.md", None)]

        resolved = resolve_link(
            "a/learnings.md",
            paths={path for path, _ in corpus},
            aliases=build_alias_index(corpus),
        )

        assert resolved == "a/learnings.md"

    def test_a_target_naming_nothing_is_pending_rather_than_an_error(self) -> None:
        assert resolve_link("notes/not-yet.md", paths=self.paths, aliases=self.aliases) is None

    def test_an_empty_target_resolves_to_nothing(self) -> None:
        assert resolve_link("", paths=self.paths, aliases=self.aliases) is None


class TestWhatTheWritersEmit:
    """The emitting side (U2) and this parser must agree, or the graph is write-only.

    Stated against the real rendering functions rather than against a hand-typed string: a change
    to how a relation block is laid out would otherwise pass every test on both sides while the
    edges quietly stopped being indexed.
    """

    def test_a_rendered_relation_block_parses_back_into_the_same_edges(self) -> None:
        relations = (
            Relation(predicate="part_of", target="repos/MemVault"),
            Relation(predicate="about", target="notes/ranking.md"),
        )

        links, observations = parse_body(render_relations(relations))

        assert links == (
            Link(predicate="part_of", target="repos/MemVault"),
            Link(predicate="about", target="notes/ranking.md"),
        )
        assert observations == ()

    def test_a_filed_document_gives_back_its_facts_its_edges_and_its_statements(self) -> None:
        document = render_document(
            {
                "title": "Erfpacht buyout",
                "date": "2026-08-12",
                "summary": "What we decided.",
                "importance": 8,
                "superseded_by": "notes/erfpacht-final.md",
            },
            "- [decision] Bought the lease out #house\n\n"
            + render_relations([Relation(predicate="about", target="notes/erfpacht.md")]),
        )

        parsed = parse_file(document)

        assert parsed.metadata == FileMetadata(
            title="Erfpacht buyout",
            importance=8,
            date="2026-08-12",
            superseded_by="notes/erfpacht-final.md",
        )
        assert parsed.links == (Link(predicate="about", target="notes/erfpacht.md"),)
        assert parsed.observations[0].tags == ("house",)
