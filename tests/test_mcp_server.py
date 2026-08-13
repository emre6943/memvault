"""The MCP surface: four tools, two of which write, and the bounds on what they may touch.

Every test drives the real server through the SDK's in-process client, so what is asserted is
what a harness would receive — the declared schemas, the annotations, the structured payload,
and the error text — rather than the return value of a Python function that happens to be
registered as a tool.

Two themes run through the write tests. The inbox tool may only ever create one new file
directly inside the configured inbox, so the traversal and collision cases are the point of the
tool rather than edge cases of it. The note tool may only ever write into an area the vault
itself names, so an unknown area is refused with the list of real ones instead of guessed at.

No test loads an embedding model. The embedder arrives through the factory seam, which is also
how the laziness is proved: a factory that raises still lets the server start, list its tools,
and answer everything that is not a semantic query.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from mcp.client import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, Tool

from memvault.cli import main
from memvault.config import Config, IndexConfig, load_config
from memvault.index import IndexBuildError, build_index
from memvault.ingest import run_ingest
from memvault.mcp_server import (
    MCPStartupError,
    build_server,
    quiet_stdout,
    reconcile_vault,
)
from memvault.recall import recall as run_recall
from tests.test_ingest import StubClassifier, git, make_repo

#: The one topic these fixtures are about, present in the vault and in every query that expects
#: a hit. A word with no English cognate keeps an accidental match from looking like a real one.
SUBJECT = "erfpacht"

VAULT_FILES = {
    "transcripts/2026/07/2026-07-02-canal-flat.md": (
        "---\ntitle: Canal flat\ndate: 2026-07-02\nsource: whatsapp\nimportance: 7\n---\n\n"
        f"# Canal flat\n\nThe {SUBJECT} on the canal flat is bought off until 2043.\n"
    ),
    "notes/2026/2026-07-03-groceries.md": (
        "---\ntitle: Groceries\ndate: 2026-07-03\n---\n\n"
        "# Groceries\n\nBread, olives, and a new kettle.\n"
    ),
}

#: The word axes the toy embedder counts. One axis per subject the fixtures talk about, so a
#: file about one of them is a unit vector and a query about none of them is the zero vector —
#: which is what makes "nothing in the vault matches this" a real state rather than a threshold.
AXES: dict[str, tuple[str, ...]] = {
    "housing": (SUBJECT, "flat", "canal"),
    "shopping": ("bread", "olives", "kettle", "groceries"),
}


class ToyEmbedder:
    """A deterministic embedder with a geometry a fixture can state in prose."""

    def __init__(self, model_id: str = "toy-v1") -> None:
        self.model_id = model_id
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        lowered = [text.casefold() for text in texts]
        return [
            [float(sum(text.count(term) for term in terms)) for terms in AXES.values()]
            for text in lowered
        ]


class RefusingEmbedderFactory:
    """Stands in for an install without the embedding extra: constructing one raises.

    `FastEmbedEmbedder` raises `IndexBuildError` from its lazy loader when fastembed is absent,
    so this is the same failure a bare install produces, moved one step earlier.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _index: IndexConfig) -> Any:
        self.calls += 1
        raise IndexBuildError("the 'fastembed' backend requires the fastembed package.")


class CountingEmbedderFactory:
    """Counts how many embedders were built, which is how laziness is observed."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _index: IndexConfig) -> ToyEmbedder:
        self.calls += 1
        return ToyEmbedder()


class NoisyEmbedder:
    """An embedder that prints while it works, the way a download bar would.

    It claims the toy model's id so a query gets as far as embedding: an index built by one
    model refuses a query from another, and the refusal would happen before the noise.
    """

    model_id = "toy-v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        print("downloading model 37%")
        return [[0.0, 0.0] for _ in texts]


def write_config(tmp_path: Path, data: dict[str, Any], name: str = "memvault.config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def vault_config(tmp_path: Path, **overrides: Any) -> Path:
    """A single-vault config whose index lives in `tmp_path`, never in `~/.memvault`."""
    root = make_repo(tmp_path, "personal")
    for relative, content in VAULT_FILES.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    # Committed, because ingest refuses to fold somebody else's uncommitted work into its own
    # commit — a fixture left dirty would make the ingest test fail for the wrong reason.
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fixture material")

    data: dict[str, Any] = {
        "default_vault": "personal",
        "vaults": {
            "personal": {
                "root": str(root),
                "areas": [
                    {
                        "name": "finance",
                        "note_template": "notes/finance/{date}-{slug}.md",
                        "when": "money, mortgages, and the flat",
                    }
                ],
            }
        },
        "index": {"path": str(tmp_path / "index" / "{vault}.db")},
    }
    data.update(overrides)
    return write_config(tmp_path, data)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return load_config(vault_config(tmp_path))


def served(config: Config, **kwargs: Any) -> MCPServer[Any]:
    """The real server for this config's default vault, with a toy embedder by default."""
    kwargs.setdefault("embedder_factory", lambda _index: ToyEmbedder())
    return build_server(config, config.vault(), **kwargs)


def tools(server: MCPServer[Any]) -> list[Tool]:
    async def go() -> list[Tool]:
        async with Client(server) as client:
            return list((await client.list_tools()).tools)

    return asyncio.run(go())


def call(server: MCPServer[Any], name: str, **arguments: Any) -> CallToolResult:
    async def go() -> CallToolResult:
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(go())


def payload(result: CallToolResult) -> dict[str, Any]:
    """The structured content of a successful call, insisting that it succeeded."""
    assert not result.is_error, message_of(result)
    assert result.structured_content is not None
    return dict(result.structured_content)


def message_of(result: CallToolResult) -> str:
    return " ".join(getattr(block, "text", "") for block in result.content)


def refusal(result: CallToolResult) -> str:
    """The error text of a call that was expected to be refused."""
    assert result.is_error, f"expected a refusal, got {result.structured_content!r}"
    return message_of(result)


def indexed(config: Config) -> None:
    """Build the vault's search index with the toy embedder, so recall has something to read."""
    vault = config.vault()
    build_index(vault, config.index, ToyEmbedder(), db_path=config.index.db_path(vault.name))


def index_of(config: Config) -> Path:
    return config.index.db_path(config.vault().name)


def filed_paths(root: Path) -> list[str]:
    """Every Markdown file the vault holds outside its inbox — what a write tool must not move."""
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.md")
        if ".git" not in path.parts and "inbox" not in path.parts
    )


# --- the tool surface ---------------------------------------------------------------------


def test_the_server_exposes_exactly_the_four_tools(config: Config) -> None:
    """R10 names four tools. A fifth appearing is an interface change, not a detail."""
    assert sorted(tool.name for tool in tools(served(config))) == [
        "recall",
        "vault_status",
        "write_inbox",
        "write_note",
    ]


def test_every_tool_carries_an_input_and_an_output_schema(config: Config) -> None:
    """Typed returns are the output schema — a client should not have to parse prose."""
    for tool in tools(served(config)):
        assert tool.input_schema["type"] == "object", tool.name
        assert tool.output_schema is not None, tool.name


def test_the_reading_tools_declare_themselves_read_only(config: Config) -> None:
    """A harness decides whether to prompt from the annotation, so it has to be honest."""
    by_name = {tool.name: tool for tool in tools(served(config))}

    for name in ("recall", "vault_status"):
        annotations = by_name[name].annotations
        assert annotations is not None and annotations.read_only_hint is True


def test_the_writing_tools_do_not_claim_to_be_read_only(config: Config) -> None:
    """The gate is the harness's prompt; a write tool that hid would walk straight through it."""
    by_name = {tool.name: tool for tool in tools(served(config))}

    for name in ("write_inbox", "write_note"):
        annotations = by_name[name].annotations
        assert annotations is not None and annotations.read_only_hint is False


# --- recall -------------------------------------------------------------------------------


def test_recall_returns_what_the_cli_would_return_for_the_same_query(config: Config) -> None:
    """The tool is a second surface on one implementation, not a second implementation."""
    indexed(config)
    expected = run_recall(
        config.vault(),
        config.index,
        ToyEmbedder(),
        SUBJECT,
        min_similarity=config.index.min_similarity,
        ranking=config.ranking,
    )

    got = payload(call(served(config), "recall", query=SUBJECT))

    assert [hit["path"] for hit in got["results"]] == [r.path for r in expected.results]
    assert [hit["score"] for hit in got["results"]] == [r.score for r in expected.results]
    assert got["chunks_searched"] == expected.chunks_searched


def test_recall_carries_the_pointer_fields_a_caller_chooses_by(config: Config) -> None:
    """Paths and snippets are the deliverable; a result without them is not actionable."""
    indexed(config)

    got = payload(call(served(config), "recall", query=SUBJECT))

    first = got["results"][0]
    assert first["path"] == "transcripts/2026/07/2026-07-02-canal-flat.md"
    assert SUBJECT in first["snippet"]
    assert "keyword" in first["modes"]


def test_recall_says_so_rather_than_inventing_a_nearest_neighbour(config: Config) -> None:
    """An empty result is an answer, and the message has to distinguish it from a failure."""
    indexed(config)

    got = payload(call(served(config), "recall", query="chromodynamics"))

    assert got["results"] == []
    assert "nothing in the vault" in got["message"]


def test_recall_refuses_a_date_it_cannot_read(config: Config) -> None:
    """A filter silently ignored would answer the wrong question and look like the right one."""
    indexed(config)

    assert "YYYY-MM-DD" in refusal(call(served(config), "recall", query=SUBJECT, since="last week"))


def test_recall_reports_a_missing_index_instead_of_answering_emptily(config: Config) -> None:
    """No index and no matches look identical from the outside, so they must not read alike."""
    assert "memvault index" in refusal(call(served(config), "recall", query=SUBJECT))


# --- write_inbox --------------------------------------------------------------------------


def test_write_inbox_lands_a_drop_that_a_later_ingest_files(config: Config, tmp_path: Path) -> None:
    """The tool queues material; the ordinary pass is what turns it into a memory."""
    vault = config.vault()
    before = filed_paths(vault.root)

    written = payload(
        call(
            served(config),
            "write_inbox",
            text="Deniz called about the notary appointment on Friday.",
            metadata={"source": "phone", "date": "2026-07-05"},
        )
    )
    report = run_ingest(
        config, vault, classifier=StubClassifier(), notifier=None, ingested=date(2026, 7, 5)
    )

    assert (vault.inbox / written["filename"]).exists() is False, "the inbox must drain"
    assert report.exit_code == 0
    filed = set(filed_paths(vault.root)) - set(before)
    assert len(filed) == 1
    body = (vault.root / filed.pop()).read_text(encoding="utf-8")
    assert "notary appointment" in body
    assert "source: phone" in body


def test_write_inbox_writes_into_the_inbox_and_nowhere_else(config: Config) -> None:
    """The narrow write surface is the mechanical bound if a harness auto-approves."""
    vault = config.vault()
    before = filed_paths(vault.root)

    written = payload(call(served(config), "write_inbox", text="A thought worth keeping."))

    assert filed_paths(vault.root) == before, "nothing outside the inbox may change"
    assert (vault.inbox / written["filename"]).is_file()
    assert written["path"].startswith("inbox/")


@pytest.mark.parametrize(
    "filename",
    ["../escape.md", "sub/../../escape.md", "/etc/escape.md", "notes/escape.md", "..", "."],
)
def test_write_inbox_refuses_a_filename_that_names_a_place(config: Config, filename: str) -> None:
    """A drop names a file. Anything with a path in it is refused before a byte is written."""
    result = call(served(config), "write_inbox", text="x", filename=filename)

    assert "plain file name" in refusal(result)
    assert list(config.vault().inbox.iterdir()) == []


def test_write_inbox_refuses_a_hidden_filename(config: Config) -> None:
    """The inbox skips hidden entries, so a dotted drop would wait there forever unread."""
    assert "hidden" in refusal(call(served(config), "write_inbox", text="x", filename=".secret.md"))


def test_write_inbox_refuses_a_metadata_key_the_inbox_does_not_read(config: Config) -> None:
    """A misspelled key is a mistake the caller can fix now; ignoring it teaches it nothing."""
    result = call(served(config), "write_inbox", text="x", metadata={"classifcation": "personal"})

    assert "classifcation" in refusal(result)


def test_write_inbox_refuses_a_declared_value_the_contract_rejects(config: Config) -> None:
    """`importance: "high"` would be held for review at ingest; refusing now names it sooner."""
    result = call(served(config), "write_inbox", text="x", metadata={"importance": "high"})

    assert "importance" in refusal(result)
    assert list(config.vault().inbox.iterdir()) == []


def test_write_inbox_refuses_an_empty_drop(config: Config) -> None:
    """A drop with no body is not a memory; ingest would skip it and nobody would be told."""
    assert "no content" in refusal(call(served(config), "write_inbox", text="   \n\n"))


def test_write_inbox_refuses_to_overwrite_what_is_already_waiting(config: Config) -> None:
    """Two drops are two memories. Overwriting one of them loses material silently."""
    server = served(config)
    call(server, "write_inbox", text="first", filename="note.md")

    second = call(server, "write_inbox", text="second", filename="note.md")

    assert "already waiting" in refusal(second)
    assert (config.vault().inbox / "note.md").read_text(encoding="utf-8").strip() == "first"


def test_write_inbox_names_generated_files_by_their_content(config: Config) -> None:
    """The same text queued twice must collide rather than double a memory quietly."""
    server = served(config)
    first = payload(call(server, "write_inbox", text="A repeated thought."))

    assert "already waiting" in refusal(call(server, "write_inbox", text="A repeated thought."))
    assert first["content_id"][:8] in first["filename"]


# --- write_note ---------------------------------------------------------------------------


def test_write_note_files_into_a_configured_area(config: Config) -> None:
    """The area's own template decides the path — the tool never picks one."""
    written = payload(
        call(
            served(config),
            "write_note",
            area="finance",
            title="Notary fee split",
            body="The buyer pays the transfer deed.",
            importance=8,
        )
    )

    path = config.vault().root / written["path"]
    assert written["path"].startswith("notes/finance/")
    text = path.read_text(encoding="utf-8")
    assert "title: Notary fee split" in text
    assert "importance: 8" in text
    assert "The buyer pays the transfer deed." in text


def test_write_note_records_that_a_tool_call_put_it_there(config: Config) -> None:
    """Provenance: a note that arrived through a tool is a different fact from a distilled one."""
    written = payload(
        call(served(config), "write_note", area="finance", title="Rates", body="Fixed for ten.")
    )

    text = (config.vault().root / written["path"]).read_text(encoding="utf-8")
    assert "kind: note" in text
    assert "source: mcp" in text
    assert "area: finance" in text


def test_write_note_refuses_an_area_the_vault_does_not_name(config: Config) -> None:
    """Named-area-only is the mechanical bound; the refusal lists what does exist."""
    result = call(served(config), "write_note", area="somewhere-else", title="X", body="y")

    text = refusal(result)
    assert "somewhere-else" in text
    assert "finance" in text


def test_write_note_reaches_a_project_directory_the_config_never_lists(config: Config) -> None:
    """Projects are areas too — the same list the ingestion pass builds, not a shorter one."""
    (config.vault().root / "repos" / "MemVault").mkdir(parents=True)

    written = payload(
        call(served(config), "write_note", area="MemVault", title="Index rebuild", body="Cheap.")
    )

    assert written["path"].startswith("repos/MemVault/adhoc/")


def test_write_note_never_overwrites_an_existing_note(config: Config) -> None:
    """Two notes filed on one day under one title are two notes, and both survive."""
    server = served(config)
    first = payload(call(server, "write_note", area="finance", title="Same day", body="one"))
    second = payload(call(server, "write_note", area="finance", title="Same day", body="two"))

    assert first["path"] != second["path"]
    assert (config.vault().root / first["path"]).read_text(encoding="utf-8").endswith("one\n")


def test_write_note_refuses_a_title_that_leaves_no_filename(config: Config) -> None:
    """An `untitled-` file from a tool call is a bug report nobody will ever open."""
    assert "letter or digit" in refusal(
        call(served(config), "write_note", area="finance", title="???", body="x")
    )


def test_write_note_refuses_an_importance_outside_the_band(config: Config) -> None:
    """The 1-10 band is what recall's neutral-5 default is calibrated against."""
    result = call(
        served(config), "write_note", area="finance", title="Loud", body="x", importance=99
    )

    assert "1 to 10" in refusal(result)


# --- vault_status -------------------------------------------------------------------------


def test_vault_status_reports_inbox_depth_and_a_missing_index(config: Config) -> None:
    """The two questions a caller asks before trusting an answer: what is queued, and is the
    index behind the files."""
    call(served(config), "write_inbox", text="waiting to be filed")

    got = payload(call(served(config), "vault_status"))

    assert got["inbox_pending"] == 1
    assert got["index_state"] == "missing"
    assert got["healthy"] is False
    assert got["indexed_chunks"] is None


def test_vault_status_reports_a_stale_index(config: Config) -> None:
    """A stale index answers, just from an older vault — the failure nobody notices."""
    indexed(config)
    later = (config.vault().root / "transcripts" / "later.md").resolve()
    later.write_text("# Later\n\nWritten after the index was built.\n", encoding="utf-8")

    got = payload(call(served(config), "vault_status"))

    assert got["index_state"] == "stale"


def test_vault_status_counts_the_chunks_the_index_holds(config: Config) -> None:
    """The one question the terminal report does not answer, and the reason the server holds a
    connection at all."""
    indexed(config)
    with sqlite3.connect(index_of(config)) as conn:
        expected = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    got = payload(call(served(config), "vault_status"))

    assert got["indexed_chunks"] == expected
    assert got["index_state"] == "current"


def test_vault_status_agrees_with_doctor_about_health(
    config: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two surfaces, one health core: they must not develop two opinions of healthy."""
    indexed(config)

    got = payload(call(served(config), "vault_status"))
    code = main(["--config", str(config.source), "doctor"])
    capsys.readouterr()

    assert got["healthy"] is (code == 0)


# --- vault reconciliation -------------------------------------------------------------------


def test_a_directory_that_is_no_configured_root_is_refused_by_name(
    config: Config, tmp_path: Path
) -> None:
    """The silent alternative is tools answering about one vault while ingest files into
    another, with nothing in the output saying so."""
    stranger = tmp_path / "not-a-vault"
    stranger.mkdir()

    with pytest.raises(MCPStartupError) as caught:
        reconcile_vault(config, stranger)

    assert str(stranger) in str(caught.value)
    assert "personal" in str(caught.value)


def test_the_passed_directory_picks_the_vault_rather_than_the_default(tmp_path: Path) -> None:
    """The harness's directory is the user's statement of what they want served."""
    personal = make_repo(tmp_path, "personal")
    work = make_repo(tmp_path, "work")
    path = write_config(
        tmp_path,
        {
            "default_vault": "personal",
            "vaults": {"personal": {"root": str(personal)}, "work": {"root": str(work)}},
            "index": {"path": str(tmp_path / "index" / "{vault}.db")},
        },
    )
    config = load_config(path)

    assert reconcile_vault(config, work).name == "work"


def test_an_explicit_vault_that_disagrees_with_the_directory_is_refused(tmp_path: Path) -> None:
    """Two answers to one question is not something to resolve in either one's favour."""
    personal = make_repo(tmp_path, "personal")
    work = make_repo(tmp_path, "work")
    config = load_config(
        write_config(
            tmp_path,
            {
                "default_vault": "personal",
                "vaults": {"personal": {"root": str(personal)}, "work": {"root": str(work)}},
                "index": {"path": str(tmp_path / "index" / "{vault}.db")},
            },
        )
    )

    with pytest.raises(MCPStartupError) as caught:
        reconcile_vault(config, work, requested="personal")

    assert "two different vaults" in str(caught.value)


def test_no_directory_falls_back_to_the_configured_default(config: Config) -> None:
    """`memvault mcp` with no argument is how the plan's own scratch-project wiring reads."""
    assert reconcile_vault(config, None).name == "personal"


def test_the_cli_refuses_a_mismatch_before_it_serves_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal that reached the transport would surface to the user as a client timeout."""
    path = vault_config(tmp_path)
    stranger = tmp_path / "elsewhere"
    stranger.mkdir()

    code = main(["--config", str(path), "mcp", str(stranger)])

    assert code == 1
    assert "is not the root of any vault" in capsys.readouterr().err


# --- startup cost and stdout hygiene ----------------------------------------------------------


def test_the_server_starts_and_serves_without_an_embedder(config: Config) -> None:
    """A bare install has no embedding model, and every non-semantic tool must still work."""
    factory = RefusingEmbedderFactory()
    server = served(config, embedder_factory=factory)

    assert len(tools(server)) == 4
    assert payload(call(server, "vault_status"))["vault"] == "personal"
    assert payload(call(server, "write_inbox", text="still works"))["filename"]
    assert factory.calls == 0, "nothing but a query may build an embedder"


def test_a_semantic_query_without_an_embedder_says_what_is_missing(config: Config) -> None:
    """The degrade has to name the package; "recall error" alone strands the adopter."""
    indexed(config)

    assert "fastembed" in refusal(
        call(served(config, embedder_factory=RefusingEmbedderFactory()), "recall", query=SUBJECT)
    )


def test_the_embedder_is_built_once_and_only_when_a_query_needs_it(config: Config) -> None:
    """Startup is bounded by the client, so the model load cannot happen in the lifespan."""
    indexed(config)
    factory = CountingEmbedderFactory()
    server = served(config, embedder_factory=factory)

    call(server, "vault_status")
    assert factory.calls == 0

    call(server, "recall", query=SUBJECT)
    call(server, "recall", query=SUBJECT)
    assert factory.calls == 1, "the memoized embedder must survive across calls"


def test_what_the_embedder_prints_goes_to_stderr_not_the_wire(
    config: Config, capsys: pytest.CaptureFixture[str]
) -> None:
    """One stray line on stdout corrupts the session, and a download bar is a lot of lines."""
    indexed(config)

    call(served(config, embedder_factory=lambda _index: NoisyEmbedder()), "recall", query=SUBJECT)

    captured = capsys.readouterr()
    assert "downloading model" in captured.err
    assert captured.out == ""


def test_quiet_stdout_puts_the_stream_back_afterwards() -> None:
    """A guard that leaked would silence the whole process, which is worse than the noise."""
    import sys

    before = sys.stdout
    with quiet_stdout():
        assert sys.stdout is sys.stderr
    assert sys.stdout is before


def test_the_lifespan_leaves_no_index_behind_when_there_is_none(config: Config) -> None:
    """A server is not an indexer: starting one must not create an empty database that
    `doctor` would then call current."""
    call(served(config), "vault_status")

    assert not index_of(config).exists()
