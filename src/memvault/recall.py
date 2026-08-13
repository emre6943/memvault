"""Hybrid recall: keyword and semantic retrieval over the index, fused and then ranked.

Why hybrid: keyword search finds the rare exact token — a name, an error code, `erfpacht` —
that an embedding smooths away, and semantic search finds the paraphrase that shares no words
with the question. Either alone misses a whole class of memory, so both run and their results
are merged (R18).

Why reciprocal rank fusion: the two modes produce scores on incomparable scales — a BM25
figure and a cosine — and any attempt to weight them against each other is a constant nobody
can justify. RRF throws the magnitudes away and keeps only each mode's *opinion of order*, so
neither can dominate by having the larger numbers. A chunk both modes like outranks a chunk
only one of them likes, which is the behaviour the fusion exists to produce.

Why the other signals are additive terms and not a blend (R4): how well a passage matches is not
the same question as how much it is worth, and a vault answers with both. Importance, age, and
one relation hop enter as small additions on the fused score's own scale — a first place in one
mode is worth `1/(60+1)`, so the terms are sized against that — rather than as a weighted average
of raw magnitudes. Magnitude blending is specifically what this module still refuses: the E5
model compresses related and unrelated text alike into a narrow band (the measurement behind
`IndexConfig.min_similarity`), so multiplying a cosine by a weight and adding it would drown
every other signal in a number that barely varies. Each term is normalised to 0-1 first and its
weight lives in `RankingConfig`, documented as calibrated against a model and a corpus. Zeroing
the three new weights reproduces pre-v2 ordering exactly, from exactly the pre-v2 candidates.

Why supersession is not one of those dials: `superseded_by:` is a statement that a file has been
replaced, not a preference about ranking, so it multiplies the score down whatever the weights
say and the result carries the successor's path (R5). Ranking never edits a file — the frontmatter
key is written by hand or by reflection, and read here.

Why half of it is optional (R12): the embedding backend lives behind the `[semantic]` extra, so
an install may have no way to embed a query, and an index may have been built with no vectors to
compare one against. Neither is an error — the keyword half is complete on its own — so the query
runs, `keyword_only` is set, and the message names which of the two is missing and how to fix it.
Silently returning half a search is the failure mode that matters here: a caller reads "nothing
matches" and concludes the vault is empty on a subject it holds a paraphrase of.

Why results carry paths and snippets rather than full text: a session should spend its context
on the two files that matter, not on twenty chunks (R19, R20). The snippet is a decision aid
for choosing what to open; the path is the answer.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from memvault.config import IndexConfig, RankingConfig, VaultConfig
from memvault.graph import IMPORTANCE_MAX, IMPORTANCE_MIN, FileMetadata
from memvault.inbox import FRONTMATTER_FENCE, FRONTMATTER_TERMINATORS
from memvault.index import (
    NULL_MODEL_ID,
    QUERY_PREFIX,
    SEMANTIC_EXTRA_HINT,
    ChunkRow,
    Embedder,
    FastEmbedEmbedder,
    IndexBuildError,
    NullEmbedder,
    count_chunks,
    embedding_dimension,
    escape_fts_query,
    fetch_chunks,
    fetch_file_metadata,
    fetch_leading_chunks,
    fetch_neighbors,
    has_embeddings,
    index_model,
    load_embeddings,
    match_chunks,
    open_index,
    semantic_available,
)

#: Standard reciprocal-rank-fusion constant. Large enough that the top few ranks of each mode
#: are close together, which is what stops one mode's first place from burying the other's.
RRF_K = 60

DEFAULT_LIMIT = 8

#: How many candidates each mode contributes before fusion. Wider than `limit` because a chunk
#: that places tenth in both modes should still be able to win on the sum.
POOL_FACTOR = 10
MIN_POOL = 50

#: Filters are applied after each mode has ranked, so a narrow filter can eat a whole pool.
#: Over-fetching when filters are active keeps "transcripts from last month" from coming back
#: empty merely because the unfiltered pool was full of mode files.
FILTER_POOL_FACTOR = 10

SNIPPET_CHARS = 240

#: What a file nobody scored is worth on the 1-10 band. The middle, not the bottom: a vault filed
#: before `importance:` existed is unjudged, and reading silence as "unimportant" would sink its
#: whole history the day the key ships.
NEUTRAL_IMPORTANCE = 5

#: How hard `superseded_by:` pushes a file down (R5). An order of magnitude, not removal — the
#: superseded note is still the record of what was believed, and a query for it must still find
#: it, below the file that replaced it.
SUPERSEDED_FACTOR = 0.1

#: How many of the strongest hits get to pull their neighbours in. A hop from the best few hits is
#: a related file; a hop from the fortieth is a coincidence with a wikilink in it.
GRAPH_SOURCE_LIMIT = 5

#: Reflection notes cite everything they reflected on (R6), so as boost sources they would radiate
#: "recently reflected on" across files with nothing else in common. They are boosted like any
#: other file; they simply never boost.
REFLECTION_KIND = "reflection"

#: Said when the install can embed but the index holds no vectors — someone indexed on a bare
#: install and has since added the extra, or never had it when the index was built. Distinct from
#: `SEMANTIC_EXTRA_HINT` because the fix is different: this one is a re-index, not an install.
KEYWORD_ONLY_INDEX_HINT = (
    "semantic search is off: this index was built without embeddings. "
    "Re-run `memvault index` to add meaning-based results."
)

_EPS = 1e-9
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_ISO_DAY_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_YEAR_MONTH_RE = re.compile(r"(?:^|/)(\d{4})/(\d{2})(?:/|$)")
_FRONTMATTER_HEAD_BYTES = 8192


class RecallError(Exception):
    """Raised when a query cannot be run at all — no index, or a mismatched embedding model.

    Distinct from returning no results: a query that matches nothing is an answer, and this is
    the absence of a working index to ask.
    """


@dataclass(frozen=True)
class FileMeta:
    """What recall knows about a vault file without reading all of it.

    `source` is the drop's declared origin (`whatsapp`, `manual`, …) and `section` is the
    top-level vault directory it was filed under (`transcripts`, `research`, …). Both answer
    "what kind of material is this", so the source filter accepts either.
    """

    source: str | None = None
    section: str = ""
    date: date | None = None


@dataclass(frozen=True)
class RecallResult:
    """One cited passage. The path is the deliverable; everything else helps choose it."""

    path: str
    heading: str
    snippet: str
    score: float
    chunk_id: int
    similarity: float = 0.0
    keyword_rank: int | None = None
    semantic_rank: int | None = None
    source: str | None = None
    date: date | None = None
    #: The path this file's frontmatter says replaced it, if it says so (R5). Carried on the
    #: result rather than only folded into the score, because a reader who opens a superseded
    #: note should learn that before they act on it.
    superseded_by: str | None = None
    #: True when a relation hop from a stronger hit is part of why this is here. Some of these
    #: files matched nothing themselves, so without the flag their placement is unexplainable.
    graph_neighbour: bool = False

    @property
    def modes(self) -> str:
        """Which retrieval modes found this — the one-glance explanation of its rank."""
        found = []
        if self.keyword_rank is not None:
            found.append("keyword")
        if self.semantic_rank is not None:
            found.append("semantic")
        if self.graph_neighbour:
            found.append("graph")
        return "+".join(found) or "none"


@dataclass(frozen=True)
class RecallResponse:
    """Results plus the sentence to show when there are none.

    A no-match query is not an error, so it returns here rather than raising — but it must not
    return silently either, or a caller reports "nothing in the vault" when the real cause was
    a filter that excluded everything.
    """

    query: str
    results: list[RecallResult] = field(default_factory=list)
    message: str = ""
    chunks_searched: int = 0
    #: True when only the keyword half ran (R12) — the install cannot embed, or the index holds
    #: no vectors. Carried as a flag as well as in `message` so a caller that renders its own
    #: text still knows the search was partial rather than having to match on a sentence.
    keyword_only: bool = False


def query_embedder(index_config: IndexConfig) -> Embedder:
    """The query half of the E5 pair, or the null backend when this install cannot embed.

    E5 is asymmetric: stored text was embedded as `passage: …`, so a query embedded without
    `query: …` lands in a slightly different place and quietly loses recall.

    A bare install gets `NullEmbedder` rather than an error (R12). `recall` reads its `model_id`
    and takes the keyword-only road; raising here instead would turn a light install's every
    query into a failure, when half of the search works perfectly well without a model.
    """
    if index_config.backend == "fastembed":
        if not semantic_available():
            return NullEmbedder()
        return FastEmbedEmbedder(index_config.model, prefix=QUERY_PREFIX)
    raise RecallError(
        f"index.backend {index_config.backend!r} has no recall implementation yet. "
        "Set index.backend to 'fastembed'."
    )


def build_fts_query(text: str) -> str:
    """Turn free text into an FTS5 expression that ORs the query's words.

    Every word goes through `escape_fts_query`, so quotes, `*`, `NEAR` and a leading `-` are
    literal tokens rather than syntax. The words are then ORed: FTS5 ANDs bare terms, and a
    natural-language question ANDed against a vault matches nothing at all. BM25 still ranks a
    chunk carrying several of the words above one carrying a single word, so OR loses ordering
    quality, not ordering.
    """
    groups = [escape_fts_query(word) for word in text.split()]
    return " OR ".join(f"({group})" for group in groups if group)


def _cosine(
    matrix: npt.NDArray[np.float32], query_vector: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Cosine similarity of every stored vector against the query (KTD3: exact, brute force)."""
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1) * query_norm
    dot = matrix @ query_vector
    similarities: npt.NDArray[np.float32] = np.divide(
        dot, denominator, out=np.zeros_like(dot), where=denominator > 0
    ).astype(np.float32, copy=False)
    return similarities


def _semantic_candidates(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    pool: int,
    min_similarity: float,
) -> tuple[list[int], dict[int, float]]:
    """Chunks whose meaning stands out for this query, best first.

    The admission rule is "more similar than the typical chunk in this vault": the corpus
    median is the floor. That is what lets a query about nothing at all return nothing —
    without it, brute-force scoring always has a best chunk and recall would answer every
    question with its nearest neighbour, however far away.

    `min_similarity` raises the floor for backends whose unrelated pairs still score high
    (E5 puts unrelated text around 0.75), where the median alone is not a useful cut.
    """
    ids, matrix = load_embeddings(conn)
    if not ids:
        return [], {}

    vectors = embedder.embed([query])
    if len(vectors) != 1:
        raise RecallError(f"the embedder returned {len(vectors)} vectors for one query")
    query_vector = np.asarray(vectors[0], dtype=np.float32)

    stored_width = embedding_dimension(conn) or int(matrix.shape[1])
    if query_vector.shape[0] != stored_width:
        raise RecallError(
            f"the query embedding is {query_vector.shape[0]}-dimensional while the index stores "
            f"{stored_width}-dimensional vectors. The index was built with a different model — "
            "delete it and re-run `memvault index`."
        )

    similarities = _cosine(matrix, query_vector)
    by_id = {chunk_id: float(value) for chunk_id, value in zip(ids, similarities, strict=True)}
    threshold = max(min_similarity, float(np.median(similarities)))

    chosen: list[int] = []
    for position in np.argsort(-similarities, kind="stable"):
        value = float(similarities[position])
        if value < threshold or value <= _EPS:
            break
        chosen.append(ids[int(position)])
        if len(chosen) >= pool:
            break
    return chosen, by_id


def _read_head(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(_FRONTMATTER_HEAD_BYTES)
    except OSError:
        return ""


def _frontmatter(text: str) -> dict[str, Any]:
    """Read a leading YAML frontmatter block, or nothing.

    Malformed frontmatter degrades to "no declared metadata" rather than failing the search:
    a broken header is a reason to rank a file lower, never a reason to refuse a query.
    """
    if not text.startswith(FRONTMATTER_FENCE):
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return {}
    for index in range(1, len(lines)):
        if lines[index].strip() in FRONTMATTER_TERMINATORS:
            try:
                parsed = yaml.safe_load("\n".join(lines[1:index]))
            except yaml.YAMLError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        match = _ISO_DAY_RE.search(value)
        if match is not None:
            try:
                return date(int(match[1]), int(match[2]), int(match[3]))
            except ValueError:
                return None
    return None


def _day_from_path(relative: str) -> date | None:
    """A date read off the filed path, for vault files that carry no frontmatter date.

    The vault's own layout dates its material — `transcripts/2026/07/2026-07-12-slug.md` — so
    the path is a better answer than an mtime that a `git clone` would rewrite.
    """
    name = relative.rsplit("/", 1)[-1]
    match = _ISO_DAY_RE.search(name)
    if match is not None:
        try:
            return date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            return None
    month = _YEAR_MONTH_RE.search(relative)
    if month is not None:
        try:
            return date(int(month[1]), int(month[2]), 1)
        except ValueError:
            return None
    return None


def file_meta(root: Path, relative: str) -> FileMeta:
    """Resolve a filed file's source and date: frontmatter first, then path, then mtime."""
    path = root / relative
    section = relative.split("/")[0] if "/" in relative else ""

    declared = _frontmatter(_read_head(path))
    raw_source = declared.get("source")
    source = raw_source.strip() if isinstance(raw_source, str) and raw_source.strip() else None

    day = _coerce_day(declared.get("date")) or _coerce_day(declared.get("ingested"))
    if day is None:
        day = _day_from_path(relative)
    if day is None:
        try:
            day = datetime.fromtimestamp(path.stat().st_mtime).date()
        except OSError:
            day = None
    return FileMeta(source=source, section=section, date=day)


def _passes(meta: FileMeta, sources: set[str], since: date | None, until: date | None) -> bool:
    if sources:
        candidates = {meta.section.casefold()} if meta.section else set()
        if meta.source:
            candidates.add(meta.source.casefold())
        if not candidates & sources:
            return False
    if since is not None and (meta.date is None or meta.date < since):
        return False
    return not (until is not None and (meta.date is None or meta.date > until))


def _snippet(text: str, terms: Sequence[str], width: int = SNIPPET_CHARS) -> str:
    """A readable window of the chunk, centred on the first query word it contains."""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat

    lowered = flat.casefold()
    position = -1
    for term in terms:
        found = lowered.find(term)
        if found != -1 and (position == -1 or found < position):
            position = found
    if position == -1:
        position = 0

    start = max(0, min(position - width // 3, len(flat) - width))
    end = min(len(flat), start + width)
    return ("…" if start > 0 else "") + flat[start:end].strip() + ("…" if end < len(flat) else "")


def importance_term(value: int | None) -> float:
    """A declared 1-10 importance as a 0-1 term, with silence worth the middle of the band.

    Clamped rather than trusted at the edges: a hand-edited `importance: 99` is a typo, and a term
    above 1.0 would out-shout the fusion it exists to nudge.
    """
    score = NEUTRAL_IMPORTANCE if value is None else value
    clamped = min(max(score, IMPORTANCE_MIN), IMPORTANCE_MAX)
    return (clamped - IMPORTANCE_MIN) / (IMPORTANCE_MAX - IMPORTANCE_MIN)


def recency_term(day: date | None, *, today: date, half_life_days: float) -> float:
    """Exponential decay on a file's age: 1.0 today, half of that one half-life later.

    Exponential rather than a window, because "older than N days" is a cliff that makes a file
    worthless the morning after and a smooth curve is what a memory actually feels like.

    A date in the future scores the same as today — frontmatter dates are declared, and a note
    about next week's viewing is not more relevant than one about this morning's. A file the vault
    cannot date at all makes no recency claim: `file_meta` falls back to frontmatter, then the
    path, then the mtime, so an undated file is one that is no longer on disk, and guessing a
    neutral age for it would invent the fact those fallbacks failed to find.
    """
    if day is None:
        return 0.0
    age = (today - day).days
    if age <= 0:
        return 1.0
    return float(0.5 ** (age / half_life_days))


def _graph_strength(
    conn: sqlite3.Connection,
    *,
    best_by_path: dict[str, float],
    facts: dict[str, FileMetadata],
) -> dict[str, float]:
    """How much each file is pulled up by one relation hop from the strongest hits (R2).

    Two guards keep a well-linked vault from turning into a vault where the links decide
    everything. A source's contribution is split across its neighbours, so a note that links to
    thirty files says a thirtieth as much about each of them as a note that links to three — an
    index page must not quietly own every query it matches. And a `kind: reflection` file never
    acts as a source at all, because it cites everything it reflected on.

    Returned as a 0-1 strength per path, capped: a file reached from several hits is more likely
    to be relevant, but not without limit.
    """
    ranked = sorted(best_by_path, key=lambda path: (-best_by_path[path], path))
    sources = [path for path in ranked if facts.get(path, FileMetadata()).kind != REFLECTION_KIND][
        :GRAPH_SOURCE_LIMIT
    ]
    if not sources:
        return {}

    neighbours = fetch_neighbors(conn, sources)
    strength: dict[str, float] = {}
    for source in sources:
        hop = neighbours.get(source, ())
        if not hop:
            continue
        share = 1.0 / len(hop)
        for target in hop:
            strength[target] = min(1.0, strength.get(target, 0.0) + share)
    return strength


def _keyword_only_note(*, can_embed: bool) -> str:
    """Which of the two keyword-only sentences applies, given what the embedder turned out to be.

    Always one or the other, never silence: a query answered by half the engine reads exactly
    like a query answered by all of it, and a caller who is not told will conclude the vault
    holds nothing on a subject it holds a paraphrase of.
    """
    return KEYWORD_ONLY_INDEX_HINT if can_embed else SEMANTIC_EXTRA_HINT


def _describe_filters(sources: set[str], since: date | None, until: date | None) -> str:
    parts = []
    if sources:
        parts.append("source " + "/".join(sorted(sources)))
    if since is not None:
        parts.append(f"on or after {since.isoformat()}")
    if until is not None:
        parts.append(f"on or before {until.isoformat()}")
    return ", ".join(parts)


def recall(
    vault: VaultConfig,
    index_config: IndexConfig,
    embedder: Embedder,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    sources: Sequence[str] = (),
    since: date | None = None,
    until: date | None = None,
    min_similarity: float = 0.0,
    ranking: RankingConfig | None = None,
    db_path: str | Path | None = None,
) -> RecallResponse:
    """Search the vault's index and return cited passages, best first.

    Both modes run against the same index and are merged by reciprocal rank fusion, so a chunk
    that only the keyword side found still places, and a chunk both sides found outranks it. The
    fused score is then adjusted by what the vault says about the file — how important it was
    called, how old it is, whether a strong hit points at it, whether something replaced it.

    `ranking` defaults to the shipped weights. Passing one with `importance`, `recency` and
    `graph` at zero is the bridge back to pre-v2 behaviour: same candidates, same scores.
    """
    if limit <= 0:
        raise RecallError(f"limit must be positive, got {limit}")

    weights = RankingConfig() if ranking is None else ranking
    today = date.today()

    wanted = {source.casefold() for source in sources if source.strip()}
    filtered = bool(wanted or since is not None or until is not None)
    filter_note = _describe_filters(wanted, since, until)

    resolved = Path(db_path) if db_path is not None else index_config.db_path(vault.name)
    if not resolved.exists():
        raise RecallError(
            f"no search index at {resolved}. Build one with `memvault index` --vault {vault.name}."
        )

    conn = open_index(resolved)
    try:
        total = count_chunks(conn)
        if total == 0:
            return RecallResponse(
                query=query,
                message=(
                    f"the index at {resolved} holds no chunks — run `memvault index "
                    f"--vault {vault.name}` before searching."
                ),
            )

        # Two independent ways the semantic half can be missing, and each has its own fix (R12):
        # this install cannot embed a query, or this index holds nothing to compare one against.
        # Either way the keyword half is untouched, so the query runs and the answer says which.
        can_embed = embedder.model_id != NULL_MODEL_ID
        semantic_on = can_embed and has_embeddings(conn)
        keyword_only_note = "" if semantic_on else _keyword_only_note(can_embed=can_embed)

        if semantic_on:
            stored_model = index_model(conn)
            if stored_model is not None and stored_model != embedder.model_id:
                raise RecallError(
                    f"the index was built with {stored_model!r} but this query would use "
                    f"{embedder.model_id!r}. Vectors from two models are not comparable — "
                    f"re-run `memvault index --vault {vault.name}`."
                )

        pool = max(limit * POOL_FACTOR, MIN_POOL) * (FILTER_POOL_FACTOR if filtered else 1)
        keyword_ids = match_chunks(conn, build_fts_query(query), limit=pool)
        semantic_ids: list[int] = []
        similarity: dict[int, float] = {}
        if semantic_on:
            semantic_ids, similarity = _semantic_candidates(
                conn, embedder, query, pool, min_similarity
            )

        rows: dict[int, ChunkRow] = {
            row.id: row for row in fetch_chunks(conn, [*keyword_ids, *semantic_ids])
        }

        meta_cache: dict[str, FileMeta] = {}

        def admits(relative: str) -> bool:
            if relative not in meta_cache:
                meta_cache[relative] = file_meta(vault.root, relative)
            return _passes(meta_cache[relative], wanted, since, until)

        def kept(candidates: Sequence[int]) -> list[int]:
            surviving = []
            for chunk_id in candidates:
                row = rows.get(chunk_id)
                if row is not None and admits(row.path):
                    surviving.append(chunk_id)
            return surviving

        keyword_kept = kept(keyword_ids)
        semantic_kept = kept(semantic_ids)

        # Ranks are taken after filtering so the fusion scores what the caller actually asked for,
        # rather than leaving gaps where excluded chunks used to sit.
        keyword_rank = {chunk_id: rank for rank, chunk_id in enumerate(keyword_kept, start=1)}
        semantic_rank = {chunk_id: rank for rank, chunk_id in enumerate(semantic_kept, start=1)}

        scores: dict[int, float] = defaultdict(float)
        for chunk_id, rank in keyword_rank.items():
            scores[chunk_id] += weights.keyword / (RRF_K + rank)
        for chunk_id, rank in semantic_rank.items():
            scores[chunk_id] += weights.semantic / (RRF_K + rank)

        candidates = set(scores)
        facts = fetch_file_metadata(conn, sorted({rows[cid].path for cid in candidates}))

        # The graph runs on the fused scores, before the additive terms, so a file cannot be
        # promoted into being a boost source by its own importance or freshness.
        strength: dict[str, float] = {}
        if weights.graph > 0 and candidates:
            best_by_path: dict[str, float] = {}
            for chunk_id in candidates:
                path = rows[chunk_id].path
                best_by_path[path] = max(best_by_path.get(path, 0.0), scores[chunk_id])
            strength = _graph_strength(conn, best_by_path=best_by_path, facts=facts)

            # Pool expansion is gated on the same weight: with the graph off, the candidate set
            # is exactly what the two modes returned, which is what the zero-weight bridge means.
            fresh = [path for path in sorted(strength) if path not in best_by_path]
            for row in fetch_leading_chunks(conn, fresh):
                if not admits(row.path):
                    continue
                rows[row.id] = row
                candidates.add(row.id)
            facts.update(fetch_file_metadata(conn, sorted({rows[cid].path for cid in candidates})))
    finally:
        conn.close()

    for chunk_id in sorted(candidates):
        path = rows[chunk_id].path
        declared = facts.get(path, FileMetadata())
        scores[chunk_id] += weights.importance * importance_term(declared.importance)
        scores[chunk_id] += weights.recency * recency_term(
            meta_cache[path].date, today=today, half_life_days=weights.half_life_days
        )
        scores[chunk_id] += weights.graph * strength.get(path, 0.0)
        if declared.superseded_by:
            scores[chunk_id] *= SUPERSEDED_FACTOR

    # Ties break on path then ordinal, never on chunk id: ids are reassigned by a rebuild, and
    # AE7 asks for the same top results from a rebuilt index.
    ordered = sorted(candidates, key=lambda cid: (-scores[cid], rows[cid].path, rows[cid].ordinal))

    terms = [term.casefold() for term in _WORD_RE.findall(query)]
    results = [
        RecallResult(
            path=rows[chunk_id].path,
            heading=rows[chunk_id].heading,
            snippet=_snippet(rows[chunk_id].text, terms),
            score=scores[chunk_id],
            chunk_id=chunk_id,
            similarity=similarity.get(chunk_id, 0.0),
            keyword_rank=keyword_rank.get(chunk_id),
            semantic_rank=semantic_rank.get(chunk_id),
            source=meta_cache[rows[chunk_id].path].source,
            date=meta_cache[rows[chunk_id].path].date,
            superseded_by=facts.get(rows[chunk_id].path, FileMetadata()).superseded_by,
            graph_neighbour=rows[chunk_id].path in strength,
        )
        for chunk_id in ordered[:limit]
    ]

    if results:
        message = f"{len(results)} result(s) for {query!r} across {total} indexed chunk(s)"
        if filter_note:
            message += f" (filtered: {filter_note})"
    elif filtered and (keyword_ids or semantic_ids):
        message = (
            f"nothing matches {query!r} within the filters ({filter_note}). "
            "Widen the window or drop --source."
        )
    else:
        message = (
            f"nothing in the vault matches {query!r}. Try fewer or different words, "
            "or re-run `memvault index` if the material is newer than the index."
        )

    # On its own line, and on every branch including the empty one: "nothing matches" means
    # something different when only half the retrieval ran, and that is precisely the case where
    # a reader is most likely to accept the emptiness as an answer.
    if keyword_only_note:
        message = f"{message}\n{keyword_only_note}"

    return RecallResponse(
        query=query,
        results=results,
        message=message,
        chunks_searched=total,
        keyword_only=bool(keyword_only_note),
    )


def format_results(response: RecallResponse) -> str:
    """Render results for a terminal, path first — the path is what the caller acts on."""
    if not response.results:
        return response.message

    lines = [response.message, ""]
    for position, result in enumerate(response.results, start=1):
        lines.append(f"{position}. {result.path}")
        if result.heading:
            lines.append(f"   {result.heading}")
        if result.superseded_by:
            # Above the snippet on purpose: a reader who acts on the text below this line needs
            # to know it was replaced before they read it, not after.
            lines.append(f"   superseded by {result.superseded_by}")
        lines.append(f"   {result.snippet}")
        meta = [f"score {result.score:.4f}", result.modes]
        if result.date is not None:
            meta.append(result.date.isoformat())
        if result.source:
            meta.append(result.source)
        lines.append("   " + " · ".join(meta))
        lines.append("")
    lines.append("Read the files above for depth — these snippets are only a pointer.")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_LIMIT",
    "GRAPH_SOURCE_LIMIT",
    "KEYWORD_ONLY_INDEX_HINT",
    "NEUTRAL_IMPORTANCE",
    "RRF_K",
    "SEMANTIC_EXTRA_HINT",
    "SUPERSEDED_FACTOR",
    "FileMeta",
    "IndexBuildError",
    "RecallError",
    "RecallResponse",
    "RecallResult",
    "build_fts_query",
    "file_meta",
    "format_results",
    "importance_term",
    "query_embedder",
    "recall",
    "recency_term",
]
