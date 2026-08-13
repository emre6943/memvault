"""Configuration schema and loader.

A config file describes one or more vaults completely. The engine holds no vault-specific
paths of its own — everything comes from here, which is what lets a single install drive
several vaults and what makes the engine usable by someone other than its author.

Validation is eager and specific: a bad path fails at load with the offending value named,
never at write time, halfway through a pass.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAMES = ("memvault.config.yaml", "memvault.config.yml")
CONFIG_ENV_VAR = "MEMVAULT_CONFIG"

#: The XDG base-directory variable, honoured when set. Named here rather than inlined because
#: the test suite clears it to keep discovery off the developer's own machine.
XDG_CONFIG_HOME_VAR = "XDG_CONFIG_HOME"
#: Where a user-level config lives under `$XDG_CONFIG_HOME` (default `~/.config`).
XDG_CONFIG_RELATIVE = "memvault/config.yaml"
#: Where installs predating the XDG location keep theirs. Searched after XDG so a move is
#: optional: an existing setup keeps working untouched, which is the whole promise of R9.
LEGACY_CONFIG_PATH = "~/.memvault/config.yaml"

_TOP_LEVEL_KEYS = {
    "default_vault",
    "vaults",
    "classifier",
    "debounce",
    "distill",
    "index",
    "ranking",
    "segment",
    "sources",
}
_VAULT_KEYS = {
    "root",
    "inbox",
    "transcript_template",
    "note_template",
    "digest_template",
    "memory_template",
    "reflection_template",
    "local_only_prefix",
    "work_route",
    "claude_mem",
    "claude_mem_projects",
    "ignore_dirty",
    "areas",
}
_AREA_KEYS = {"name", "note_template", "when", "workspaces"}
_DEBOUNCE_KEYS = {
    "quiet_minutes",
    "poll_seconds",
    "session_max_age_minutes",
    "retry_minutes",
    "max_retry_minutes",
    "notify_after_refusals",
}
_CLASSIFIER_KEYS = {"command", "preset", "model", "confidence_threshold", "timeout_seconds"}
_RANKING_KEYS = {"keyword", "semantic", "importance", "recency", "graph", "half_life_days"}
_DISTILL_KEYS = {"overlap_threshold", "shingle_size"}
_SEGMENT_KEYS = {"enabled", "min_chars", "max_segments", "timeout_seconds"}
_SOURCE_KEYS = {"segment", "confidence_threshold", "work_route"}

#: What `sources.<name>.work_route` may say. `inherit` leaves the vault's own `work_route` in
#: charge; `never` suppresses the distilled note for material from this source whatever the
#: classifier concluded. There is deliberately no value that *adds* a work route a vault does
#: not already have — a source policy may only ever narrow what crosses, never widen it.
WORK_ROUTE_POLICIES = ("inherit", "never")
_INDEX_KEYS = {
    "min_similarity",
    "path",
    "backend",
    "model",
    "chunk_max_chars",
    "chunk_overlap_chars",
    "include",
    "exclude",
}


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or describes an unusable vault."""


def _expand(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def _reject_unknown(section: str, got: dict[str, Any], allowed: set[str], source: Path) -> None:
    """Unknown keys are an error, not a silent no-op.

    A typo in a route name or template key would otherwise leave the engine quietly using a
    default while the author believes it is configured.
    """
    unknown = sorted(set(got) - allowed)
    if unknown:
        raise ConfigError(
            f"{source}: unknown key(s) in {section}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )


#: Which CLI the engine is talking to, and therefore how its arguments and its reply envelope
#: are shaped. `custom` means "run `command` exactly as written and read bare JSON back" — the
#: only behaviour that existed before presets, and the escape hatch for a CLI with no entry here.
#: The engine must run on whatever brain an adopter already pays for (R11), so this is config
#: rather than a branch in the classifier.
CLASSIFIER_PRESETS = ("claude", "codex", "opencode", "custom")


@dataclass(frozen=True)
class ClassifierConfig:
    """How the ingestion pass reaches an LLM.

    `command` is the executable name or absolute path. On a machine whose shell shadows
    `claude` with a guard function, that guard does not apply here — subprocess execution
    bypasses shell aliases and functions — but an absolute path can be set explicitly.

    `preset` says which CLI that command is, which is a different question from what it is
    called: an adopter may keep the binary under any name, and a wrapper script called `claude`
    may be anything at all.
    """

    command: str = "claude"
    preset: str = "claude"
    model: str | None = None
    confidence_threshold: float = 0.7
    timeout_seconds: int = 120

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path) -> ClassifierConfig:
        _reject_unknown("classifier", data, _CLASSIFIER_KEYS, source)
        preset = str(data.get("preset", "claude")).strip().lower()
        if preset not in CLASSIFIER_PRESETS:
            raise ConfigError(
                f"{source}: classifier.preset must be one of {', '.join(CLASSIFIER_PRESETS)}, "
                f"got {preset!r}"
            )
        threshold = float(data.get("confidence_threshold", 0.7))
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(
                f"{source}: classifier.confidence_threshold must be between 0 and 1, "
                f"got {threshold}"
            )
        timeout = int(data.get("timeout_seconds", 120))
        if timeout <= 0:
            raise ConfigError(
                f"{source}: classifier.timeout_seconds must be positive, got {timeout}"
            )
        return cls(
            command=str(data.get("command", "claude")),
            preset=preset,
            model=data.get("model"),
            confidence_threshold=threshold,
            timeout_seconds=timeout,
        )


@dataclass(frozen=True)
class DistillConfig:
    """The verbatim-leakage guard's two dials (R9).

    Both defaults restate `distill.DEFAULT_OVERLAP_THRESHOLD` and `distill.DEFAULT_SHINGLE_SIZE`
    rather than importing them — `distill` imports this module, so the arrow only points one
    way. A test asserts the two agree, because a silent drift here would loosen the strongest
    privacy claim the system makes without anyone editing the guard.

    `overlap_threshold` is the fraction of one raw sentence that may survive verbatim into a
    distilled note before the work write fails. Lower is stricter; `0.0` blocks on any matching
    run at all. `shingle_size` is how many consecutive words make a match, and it doubles as the
    guard's one deliberate hole: a sentence shorter than this can never trip it, which is what
    lets a short common phrase through. Raising it widens that hole.
    """

    overlap_threshold: float = 0.4
    shingle_size: int = 6

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path) -> DistillConfig:
        _reject_unknown("distill", data, _DISTILL_KEYS, source)
        threshold = float(data.get("overlap_threshold", 0.4))
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(
                f"{source}: distill.overlap_threshold must be between 0 and 1, got {threshold}"
            )
        size = int(data.get("shingle_size", 6))
        if size < 1:
            raise ConfigError(f"{source}: distill.shingle_size must be at least 1, got {size}")
        return cls(overlap_threshold=threshold, shingle_size=size)


@dataclass(frozen=True)
class SegmentConfig:
    """Whether and how a drop is split into topical segments before classification.

    Segmentation exists because ambient capture broke an assumption the engine was built on:
    that one drop is about one thing. A recorded conversation wanders across personal and work
    subjects in a single body, and every whole-drop verdict for such a body is wrong — `personal`
    loses the work material, `work` and `mixed` both send a distilled note about the personal
    material into the work vault.

    Splitting first and classifying each part restores the original assumption at a finer grain.
    It does not touch the privacy rule: raw content still files only to the personal vault, and
    the work vault still receives distilled notes only (R9). What changes is that the question is
    asked per topic rather than per file.

    `min_chars` is a floor below which splitting is not attempted at all, so short drops never
    pay for a model call. `max_segments` bounds the fan-out: every extra segment costs one
    classification, and a segmenter that returns forty fragments of a rambling conversation
    would be both expensive and useless. Exceeding it holds the drop rather than truncating,
    because dropping segments silently would lose material.
    """

    enabled: bool = False
    min_chars: int = 2000
    max_segments: int = 12
    timeout_seconds: int = 180

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path) -> SegmentConfig:
        _reject_unknown("segment", data, _SEGMENT_KEYS, source)
        enabled = data.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError(f"{source}: segment.enabled must be true or false, got {enabled!r}")
        min_chars = int(data.get("min_chars", 2000))
        if min_chars < 0:
            raise ConfigError(f"{source}: segment.min_chars must not be negative, got {min_chars}")
        max_segments = int(data.get("max_segments", 12))
        if max_segments < 1:
            raise ConfigError(
                f"{source}: segment.max_segments must be at least 1, got {max_segments}"
            )
        timeout = int(data.get("timeout_seconds", 180))
        if timeout <= 0:
            raise ConfigError(f"{source}: segment.timeout_seconds must be positive, got {timeout}")
        return cls(
            enabled=enabled,
            min_chars=min_chars,
            max_segments=max_segments,
            timeout_seconds=timeout,
        )


@dataclass(frozen=True)
class DebounceConfig:
    """When an idle-triggered ingestion pass is allowed to run (R7).

    Ingest used to happen at a fixed clock time, which meant a conversation captured at 9am
    waited fourteen hours to become memory. Idle-triggering replaces the clock with a question
    the machine can actually answer: has anything happened lately, and is anyone still working?

    `quiet_minutes` is that question's threshold — how long the queue marker must sit
    un-refreshed before the pass runs. Fifteen is a working default, not a measured one; the
    right value comes from lived experience, because too short interrupts a session's own
    material and too long recreates the nightly delay.

    `session_max_age_minutes` is the staleness escape. An active-session marker is written when
    a session starts and removed when it ends, so a crashed session — a closed lid, a killed
    terminal — leaves one behind forever. Without an age past which a marker is disbelieved, one
    crash would suspend ingestion permanently, which is exactly the failure mode a marker pair
    was introduced to avoid. Twelve hours is long enough that no real session is cut short.

    `retry_minutes`, `max_retry_minutes`, and `notify_after_refusals` govern the dirty-guard
    refusal, which under idle-triggering stops being an error and becomes an ordinary event: the
    pass now fires whenever work goes quiet, which is precisely when someone is most likely to
    have uncommitted vault edits open. A refusal reschedules with exponential backoff and stays
    silent; only a run of them means something is actually stuck, so only that notifies.
    """

    quiet_minutes: float = 15.0
    poll_seconds: float = 30.0
    session_max_age_minutes: float = 720.0
    retry_minutes: float = 5.0
    max_retry_minutes: float = 60.0
    notify_after_refusals: int = 3

    @property
    def quiet_seconds(self) -> float:
        return self.quiet_minutes * 60.0

    @property
    def session_max_age_seconds(self) -> float:
        return self.session_max_age_minutes * 60.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path) -> DebounceConfig:
        _reject_unknown("debounce", data, _DEBOUNCE_KEYS, source)

        def positive(name: str, default: float) -> float:
            value = float(data.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(
                    f"{source}: debounce.{name} must be a positive number, got {value}"
                )
            return value

        def non_negative(name: str, default: float) -> float:
            value = float(data.get(name, default))
            if not math.isfinite(value) or value < 0:
                raise ConfigError(f"{source}: debounce.{name} must not be negative, got {value}")
            return value

        # Zero is allowed here and nowhere else: a vault that wants the old fire-immediately
        # behaviour says so with `quiet_minutes: 0`, and a test needs it to run in one pass.
        quiet = non_negative("quiet_minutes", 15.0)
        retry = positive("retry_minutes", 5.0)
        max_retry = positive("max_retry_minutes", 60.0)
        if max_retry < retry:
            raise ConfigError(
                f"{source}: debounce.max_retry_minutes ({max_retry}) is below "
                f"debounce.retry_minutes ({retry}); the cap would shorten the first retry"
            )
        refusals = int(data.get("notify_after_refusals", 3))
        if refusals < 1:
            raise ConfigError(
                f"{source}: debounce.notify_after_refusals must be at least 1, got {refusals}"
            )
        return cls(
            quiet_minutes=quiet,
            poll_seconds=positive("poll_seconds", 30.0),
            session_max_age_minutes=positive("session_max_age_minutes", 720.0),
            retry_minutes=retry,
            max_retry_minutes=max_retry,
            notify_after_refusals=refusals,
        )


@dataclass(frozen=True)
class SourcePolicy:
    """Per-source overrides, keyed by a drop's declared `source` frontmatter value.

    A note someone deliberately pasted and a 90-minute conversation a pair of glasses recorded
    without being asked deserve different caution, and the difference is policy rather than
    code. Keeping it in config is what stops the engine from growing vault-specific opinions
    like "conversate is special", which would undo the property that makes one install usable
    by someone other than its author (R21).

    Every field is `None`/`inherit` by default, meaning "defer to the global setting". A source
    with no entry therefore behaves exactly as it did before this existed.
    """

    segment: bool | None = None
    confidence_threshold: float | None = None
    work_route: str = "inherit"

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any], source: Path) -> SourcePolicy:
        _reject_unknown(f"sources.{name}", data, _SOURCE_KEYS, source)

        segment = data.get("segment")
        if segment is not None and not isinstance(segment, bool):
            raise ConfigError(
                f"{source}: sources.{name}.segment must be true or false, got {segment!r}"
            )

        threshold = data.get("confidence_threshold")
        if threshold is not None:
            threshold = float(threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ConfigError(
                    f"{source}: sources.{name}.confidence_threshold must be between 0 and 1, "
                    f"got {threshold}"
                )

        route = str(data.get("work_route", "inherit")).strip().lower()
        if route not in WORK_ROUTE_POLICIES:
            raise ConfigError(
                f"{source}: sources.{name}.work_route must be one of "
                f"{', '.join(WORK_ROUTE_POLICIES)}, got {route!r}"
            )

        return cls(segment=segment, confidence_threshold=threshold, work_route=route)


#: The policy a source with no configured entry gets: defer to the global settings on every
#: dial. Shared rather than constructed per lookup so identity comparisons in tests are stable.
DEFAULT_SOURCE_POLICY = SourcePolicy()


def normalize_source(value: str | None) -> str:
    """Fold a declared `source` to its lookup key.

    Frontmatter is hand-written as often as it is generated, so `Conversate`, `conversate `, and
    `conversate` must all reach the same policy. Anything falsy becomes the empty string, which
    matches no configured entry and therefore inherits.
    """
    return (value or "").strip().lower()


@dataclass(frozen=True)
class IndexConfig:
    """Search index settings.

    The index is disposable — deleting it and rebuilding from files is always safe.

    `min_similarity` is an absolute cosine floor for semantic hits and is **model-specific**.
    E5 compresses everything into a narrow band, so a purely relative floor (above the corpus
    median) admits half the corpus for any query, including nonsense. Calibrated against a real
    1087-chunk vault with `multilingual-e5-small`:

    | query kind | best score |
    |---|---|
    | real ("erfpacht", "gold price analysis", "Coolify VPS deploy") | 0.84 - 0.90 |
    | nonsense ("quantum chromodynamics", "Premier League fixtures") | 0.81 - 0.82 |

    At 0.82 a real query kept 16 chunks and a nonsense query kept none. Raise it toward 0.84 for
    stricter recall; lower it toward 0.78 to surface weak matches. **Change it when you change
    `model`** — the band is a property of the model, not of the vault.
    """

    path: str = "~/.memvault/{vault}.db"
    backend: str = "fastembed"
    model: str = "intfloat/multilingual-e5-small"
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150
    include: tuple[str, ...] = ("**/*.md",)
    exclude: tuple[str, ...] = (".git/**", "node_modules/**", "index/**")
    min_similarity: float = 0.82

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path) -> IndexConfig:
        _reject_unknown("index", data, _INDEX_KEYS, source)
        backend = str(data.get("backend", "fastembed"))
        if backend not in {"fastembed", "openai"}:
            raise ConfigError(
                f"{source}: index.backend must be 'fastembed' or 'openai', got {backend!r}"
            )
        max_chars = int(data.get("chunk_max_chars", 1200))
        overlap = int(data.get("chunk_overlap_chars", 150))
        if overlap >= max_chars:
            raise ConfigError(
                f"{source}: index.chunk_overlap_chars ({overlap}) must be less than "
                f"chunk_max_chars ({max_chars})"
            )
        floor = float(data.get("min_similarity", 0.82))
        if not 0.0 <= floor <= 1.0:
            raise ConfigError(
                f"{source}: index.min_similarity must be between 0 and 1, got {floor}"
            )
        return cls(
            path=str(data.get("path", "~/.memvault/{vault}.db")),
            backend=backend,
            model=str(data.get("model", "intfloat/multilingual-e5-small")),
            chunk_max_chars=max_chars,
            chunk_overlap_chars=overlap,
            include=tuple(data.get("include", ("**/*.md",))),
            exclude=tuple(data.get("exclude", (".git/**", "node_modules/**", "index/**"))),
            min_similarity=floor,
        )

    def db_path(self, vault_name: str) -> Path:
        return _expand(self.path.format(vault=vault_name))


#: Each weight's shipped default, kept beside the dataclass so validation can iterate them
#: without restating the numbers. See `RankingConfig` for what each one scales.
_RANKING_WEIGHT_DEFAULTS = {
    "keyword": 1.0,
    "semantic": 1.0,
    "importance": 0.01,
    "recency": 0.005,
    "graph": 0.005,
}


@dataclass(frozen=True)
class RankingConfig:
    """How recall's signals combine into one score.

    Two families of dial, and the difference matters more than the numbers:

    `keyword` and `semantic` **scale the two existing candidate pools**, whose contributions are
    reciprocal-rank shaped — each hit adds `1/(60 + rank)`, so the top of a list is worth about
    0.0164 and the tenth entry about 0.0143. A weight of 1.0 is today's behaviour exactly.

    `importance`, `recency`, and `graph` are **additive terms on that same scale**, not blends of
    raw magnitudes. Cosine similarity is deliberately never mixed in by magnitude: the E5 model
    compresses real and nonsense queries alike into a ~0.73-0.92 band (the same measurement that
    produced `IndexConfig.min_similarity`), so a linear blend against raw similarity drowns every
    other signal. Each term is normalised to 0-1 first, then multiplied by its weight, which is
    why the defaults look small — the whole `importance` swing, from a 1 to a 10, is 0.01 against
    the 0.0164 a first place in one mode is worth: enough to reorder near-ties, not enough to
    overturn a decisive match.

    **The ratio between `importance` and `recency` is the calibrated part**, not the absolute
    sizes. Importance is twice recency because age is the weaker claim: a file being old is no
    evidence it is wrong, while an importance score is a judgement somebody made about the
    material. At these values a two-year-old 9 outranks a fresh 2, and among equals the recent
    one still wins — which is the behaviour `tests/test_recall.py` pins.

    Like `min_similarity`, **these defaults are calibrated against a model and a corpus**, not
    derived. Revisit them when the embedding model changes, and expect to tune them from real
    queries rather than from first principles. Setting `importance`, `recency`, and `graph` to
    zero reproduces pre-v2 pure-RRF ordering exactly, from the pre-v2 candidate pool, which is the
    bridge a vault stays on until its corpus actually carries `importance:` values.

    `half_life_days` is the recency decay: a file that old scores half the recency term of one
    written today. 180 days suits a vault whose oldest material is still worth surfacing; shorten
    it for a vault that is mostly operational churn.
    """

    keyword: float = 1.0
    semantic: float = 1.0
    importance: float = 0.01
    recency: float = 0.005
    graph: float = 0.005
    half_life_days: float = 180.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path) -> RankingConfig:
        _reject_unknown("ranking", data, _RANKING_KEYS, source)

        weights: dict[str, float] = {}
        for name, default in _RANKING_WEIGHT_DEFAULTS.items():
            value = float(data.get(name, default))
            if not math.isfinite(value):
                raise ConfigError(f"{source}: ranking.{name} must be a finite number, got {value}")
            if value < 0:
                # A negative weight inverts the signal — an important file would rank lower for
                # being important — which nobody means to write, so it is a typo worth naming.
                raise ConfigError(f"{source}: ranking.{name} must not be negative, got {value}")
            weights[name] = value

        half_life = float(data.get("half_life_days", 180.0))
        if not math.isfinite(half_life) or half_life <= 0:
            raise ConfigError(
                f"{source}: ranking.half_life_days must be a positive number, got {half_life}. "
                "Set ranking.recency to 0 to switch recency off instead."
            )

        return cls(
            keyword=weights["keyword"],
            semantic=weights["semantic"],
            importance=weights["importance"],
            recency=weights["recency"],
            graph=weights["graph"],
            half_life_days=half_life,
        )


@dataclass(frozen=True)
class AreaConfig:
    """One topic destination a drop's distilled note can be filed into.

    Declared rather than derived: the engine holds no vault paths of its own, so which areas
    exist — and what they are for — belongs to the vault's config file.

    `when` is prose written for the classifier, not for a human reader. It is the only thing that
    tells the model what "finance" means in this vault, so it should read like the sentence you
    would say to someone filing your post.
    """

    name: str
    note_template: str
    when: str = ""
    #: Feeder-declared workspace names that map here without asking the model.
    workspaces: tuple[str, ...] = ()


def _areas_from(name: str, data: Any, source: Path) -> tuple[AreaConfig, ...]:
    """Parse a vault's `areas:` block, rejecting anything a later pass would misfile on."""
    areas: list[AreaConfig] = []
    seen: set[str] = set()
    for index, raw in enumerate(data or ()):
        where = f"vaults.{name}.areas[{index}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{source}: {where} must be a mapping, got {type(raw).__name__}")
        _reject_unknown(where, raw, _AREA_KEYS, source)

        area_name = str(raw.get("name", "")).strip()
        if not area_name:
            raise ConfigError(f"{source}: {where} is missing required key 'name'")
        if area_name in seen:
            raise ConfigError(f"{source}: {where} is a duplicate area name: {area_name!r}")
        seen.add(area_name)

        template = str(raw.get("note_template", "")).strip()
        if not template:
            raise ConfigError(f"{source}: {where} is missing required key 'note_template'")
        if "{slug}" not in template:
            raise ConfigError(
                f"{source}: {where}.note_template must contain '{{slug}}' — without it every "
                f"note in this area files to the same path. Got {template!r}"
            )

        workspaces_raw = raw.get("workspaces") or ()
        workspaces = (
            (str(workspaces_raw),)
            if isinstance(workspaces_raw, str)
            else tuple(str(w) for w in workspaces_raw)
        )
        areas.append(
            AreaConfig(
                name=area_name,
                note_template=template,
                when=str(raw.get("when", "")),
                workspaces=workspaces,
            )
        )
    return tuple(areas)


@dataclass(frozen=True)
class VaultConfig:
    """One vault: where its content lives and how filed material is laid out inside it."""

    name: str
    root: Path
    inbox: Path
    transcript_template: str = "transcripts/{yyyy}/{mm}/{date}-{slug}.md"
    note_template: str = "notes/{yyyy}/{date}-{slug}.md"
    digest_template: str = "digest/{yyyy}-W{ww}.md"
    #: Where per-project durable memory lives, one directory per project. The weekly pass
    #: appends into `learnings.md`, `quirks.md`, and `reflect-log.md` under it, and creates the
    #: directory from a skeleton when a project has none yet. It is a template rather than a
    #: fixed `repos/` because the layout belongs to the vault, not to the engine.
    memory_template: str = "repos/{project}"
    #: Where the weekly pass writes its reflection note — the self-posed questions it answered
    #: with citations into the window's own filed material (R6). Kept apart from
    #: `digest_template` on purpose: the digest is a report of a window and the reflection is a
    #: piece of memory that recall is meant to return, so a vault that wants one indexed and the
    #: other out of the way needs two roots to point at. Rendered with the window's end date, so
    #: a second pass over the same window replaces its note rather than accumulating near-copies.
    reflection_template: str = "reflections/{yyyy}/{date}-reflection.md"
    local_only_prefix: str = "private"
    work_route: str | None = None
    ignore_dirty: tuple[str, ...] = ()
    #: Whether the weekly pass reads claude-mem observations for this vault at all. claude-mem is
    #: one adopter's plugin rather than a component of this engine, so a vault must be able to
    #: say it has none — without that, every adopter without it pays a degraded-source warning
    #: for something they never installed. The default is `true` because configs written before
    #: this key existed carry a populated `claude_mem_projects` map and must keep reading it.
    claude_mem: bool = True
    claude_mem_projects: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Topic destinations for distilled notes filed inside this vault. Empty means a drop files
    #: raw and gains no note — the behaviour before area routing existed.
    areas: tuple[AreaConfig, ...] = ()

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any], source: Path) -> VaultConfig:
        _reject_unknown(f"vaults.{name}", data, _VAULT_KEYS, source)
        if "root" not in data:
            raise ConfigError(f"{source}: vaults.{name} is missing required key 'root'")

        root = _expand(str(data["root"]))
        if not root.exists():
            raise ConfigError(f"{source}: vaults.{name}.root does not exist: {root}")
        if not root.is_dir():
            raise ConfigError(f"{source}: vaults.{name}.root is not a directory: {root}")
        if not (root / ".git").exists():
            raise ConfigError(
                f"{source}: vaults.{name}.root is not a git repository: {root}. "
                "A vault must be a git repo — the commit history is the audit log."
            )

        inbox_raw = str(data.get("inbox", "inbox"))
        inbox = root / inbox_raw if not os.path.isabs(inbox_raw) else _expand(inbox_raw)

        memory_template = str(data.get("memory_template", "repos/{project}"))
        if "{project}" not in memory_template:
            raise ConfigError(
                f"{source}: vaults.{name}.memory_template must contain '{{project}}' — every "
                f"project needs a directory of its own. Got {memory_template!r}"
            )

        claude_mem = data.get("claude_mem", True)
        if not isinstance(claude_mem, bool):
            raise ConfigError(
                f"{source}: vaults.{name}.claude_mem must be true or false, got {claude_mem!r}"
            )

        aliases: dict[str, tuple[str, ...]] = {}
        for canonical, values in (data.get("claude_mem_projects") or {}).items():
            if isinstance(values, str):
                aliases[str(canonical)] = (values,)
            else:
                aliases[str(canonical)] = tuple(str(v) for v in values)

        return cls(
            name=name,
            root=root,
            inbox=inbox,
            transcript_template=str(
                data.get("transcript_template", "transcripts/{yyyy}/{mm}/{date}-{slug}.md")
            ),
            note_template=str(data.get("note_template", "notes/{yyyy}/{date}-{slug}.md")),
            digest_template=str(data.get("digest_template", "digest/{yyyy}-W{ww}.md")),
            memory_template=memory_template,
            reflection_template=str(
                data.get("reflection_template", "reflections/{yyyy}/{date}-reflection.md")
            ),
            local_only_prefix=str(data.get("local_only_prefix", "private")),
            work_route=data.get("work_route"),
            ignore_dirty=tuple(str(p) for p in (data.get("ignore_dirty") or ())),
            claude_mem=claude_mem,
            claude_mem_projects=aliases,
            areas=_areas_from(name, data.get("areas"), source),
        )

    @property
    def claude_mem_enabled(self) -> bool:
        """Whether the reflection pass has a claude-mem source to read for this vault.

        Two ways to say no, and they mean the same thing: `claude_mem: false` is the explicit
        one, and an empty `claude_mem_projects` map is the one every adopter starts from — with
        no project named, the boundary control would let nothing through anyway. Callers use
        this rather than the raw flag so "off" and "nothing configured" cannot drift apart.
        """
        return self.claude_mem and bool(self.claude_mem_projects)

    def canonical_project(self, raw: str) -> str | None:
        """Map a claude-mem project key to this vault's canonical name.

        Returns None when the key belongs to no configured project. Callers skip those rather
        than defaulting them into the vault: the claude-mem database mixes work and personal
        observations, so an unrecognized key is a boundary question, not a naming nuisance.
        """
        for canonical, aliases in self.claude_mem_projects.items():
            if raw == canonical or raw in aliases:
                return canonical
        return None


@dataclass(frozen=True)
class Config:
    source: Path
    default_vault: str
    vaults: dict[str, VaultConfig]
    classifier: ClassifierConfig
    index: IndexConfig
    ranking: RankingConfig = field(default_factory=RankingConfig)
    debounce: DebounceConfig = field(default_factory=DebounceConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    sources: dict[str, SourcePolicy] = field(default_factory=dict)

    def policy_for(self, source: str | None) -> SourcePolicy:
        """The policy governing a drop from `source`, or the inherit-everything default."""
        return self.sources.get(normalize_source(source), DEFAULT_SOURCE_POLICY)

    def should_segment(self, source: str | None) -> bool:
        """Whether a drop from `source` is split before classification.

        The source's own answer wins when it has one; otherwise the global switch decides.
        """
        policy = self.policy_for(source)
        return self.segment.enabled if policy.segment is None else policy.segment

    def confidence_threshold_for(self, source: str | None) -> float:
        """The confidence a classification from `source` must clear to be acted on.

        Narrowing only, like `work_route`: a source may demand *more* certainty than the global
        setting, never less. The classifier has already applied the global threshold by the time
        this is consulted, so a lower value could not loosen anything anyway — returning the
        larger of the two makes that visible instead of leaving it as a silent no-op.
        """
        policy = self.policy_for(source)
        if policy.confidence_threshold is None:
            return self.classifier.confidence_threshold
        return max(self.classifier.confidence_threshold, policy.confidence_threshold)

    def allows_work_note(self, source: str | None) -> bool:
        """Whether material from `source` may produce a distilled note in the work vault.

        Narrowing only: `False` here suppresses a note the routing table would otherwise
        create, and there is no value that creates one the table would not.
        """
        return self.policy_for(source).work_route != "never"

    def vault(self, name: str | None = None) -> VaultConfig:
        chosen = name or self.default_vault
        if chosen not in self.vaults:
            known = ", ".join(sorted(self.vaults)) or "(none)"
            raise ConfigError(f"unknown vault {chosen!r}. Configured vaults: {known}")
        return self.vaults[chosen]

    def work_target(self, vault: VaultConfig) -> VaultConfig | None:
        """The vault that receives distilled notes from this one, if any."""
        return self.vaults[vault.work_route] if vault.work_route else None


def user_config_paths() -> tuple[Path, ...]:
    """The user-level config locations, in search order: XDG first, then the legacy home path.

    Resolved on every call rather than at import, because `$XDG_CONFIG_HOME` and `$HOME` are
    part of the answer and both move — under a scheduler, under a hook, and under the tests.
    """
    xdg_home = os.environ.get(XDG_CONFIG_HOME_VAR) or "~/.config"
    return (
        Path(os.path.expanduser(xdg_home)) / XDG_CONFIG_RELATIVE,
        Path(os.path.expanduser(LEGACY_CONFIG_PATH)),
    )


def find_config(explicit: str | Path | None = None) -> Path:
    """Resolve which config file to use (R9).

    In order: an explicit path, `$MEMVAULT_CONFIG`, `~/.config/memvault/config.yaml`,
    `~/.memvault/config.yaml`, then a `memvault.config.yaml` in the current directory.

    The two home locations are ordered so adopting the standard one is a move a user makes when
    they feel like it, never one an upgrade forces: an install that has always kept its config at
    `~/.memvault/config.yaml` keeps being found. The current directory stays last and stays
    supported — it is what a scratch vault and this suite rely on — but it is deliberately the
    weakest signal, because "which directory a scheduler happened to start in" is not a
    statement about which vault the user meant.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        return path.resolve()

    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        path = Path(env_value).expanduser()
        if not path.exists():
            raise ConfigError(f"{CONFIG_ENV_VAR} points at a missing file: {path}")
        return path.resolve()

    searched = list(user_config_paths())
    searched.extend(Path.cwd() / name for name in DEFAULT_CONFIG_NAMES)
    for candidate in searched:
        if candidate.exists():
            return candidate.resolve()

    places = "\n".join(f"  {candidate}" for candidate in searched)
    raise ConfigError(
        f"no config file found. Looked in:\n{places}\n"
        f"and at the {CONFIG_ENV_VAR} environment variable.\n"
        "Run `memvault init <directory>` to set up a vault and write one — it scaffolds an empty "
        "directory or adopts notes you already keep. Or pass --config to name a file explicitly."
    )


def load_config(path: str | Path | None = None) -> Config:
    source = find_config(path)

    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: invalid YAML: {exc}") from exc

    if raw is None:
        raise ConfigError(f"{source}: config file is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: config root must be a mapping, got {type(raw).__name__}")

    _reject_unknown("config root", raw, _TOP_LEVEL_KEYS, source)

    vaults_raw = raw.get("vaults")
    if not vaults_raw:
        raise ConfigError(f"{source}: config must define at least one vault under 'vaults'")
    if not isinstance(vaults_raw, dict):
        raise ConfigError(f"{source}: 'vaults' must be a mapping of name to vault settings")

    vaults = {
        name: VaultConfig.from_dict(name, settings or {}, source)
        for name, settings in vaults_raw.items()
    }

    default_vault = raw.get("default_vault")
    if default_vault is None:
        if len(vaults) == 1:
            default_vault = next(iter(vaults))
        else:
            raise ConfigError(
                f"{source}: 'default_vault' is required when more than one vault is configured. "
                f"Configured: {', '.join(sorted(vaults))}"
            )
    if default_vault not in vaults:
        raise ConfigError(
            f"{source}: default_vault {default_vault!r} is not a configured vault. "
            f"Configured: {', '.join(sorted(vaults))}"
        )

    for vault in vaults.values():
        if vault.work_route and vault.work_route not in vaults:
            raise ConfigError(
                f"{source}: vaults.{vault.name}.work_route points at {vault.work_route!r}, "
                f"which is not a configured vault. Configured: {', '.join(sorted(vaults))}"
            )
        if vault.work_route == vault.name:
            raise ConfigError(
                f"{source}: vaults.{vault.name}.work_route points at itself. Distilled notes "
                "must cross into a different vault."
            )

    sources_raw = raw.get("sources") or {}
    if not isinstance(sources_raw, dict):
        raise ConfigError(
            f"{source}: 'sources' must be a mapping of source name to policy, "
            f"got {type(sources_raw).__name__}"
        )
    sources: dict[str, SourcePolicy] = {}
    for name, settings in sources_raw.items():
        key = normalize_source(str(name))
        if not key:
            raise ConfigError(f"{source}: 'sources' contains an empty source name")
        if key in sources:
            raise ConfigError(
                f"{source}: 'sources' defines {key!r} more than once "
                "(names are compared case-insensitively)"
            )
        sources[key] = SourcePolicy.from_dict(key, settings or {}, source)

    return Config(
        source=source,
        default_vault=str(default_vault),
        vaults=vaults,
        classifier=ClassifierConfig.from_dict(raw.get("classifier") or {}, source),
        index=IndexConfig.from_dict(raw.get("index") or {}, source),
        ranking=RankingConfig.from_dict(raw.get("ranking") or {}, source),
        debounce=DebounceConfig.from_dict(raw.get("debounce") or {}, source),
        distill=DistillConfig.from_dict(raw.get("distill") or {}, source),
        segment=SegmentConfig.from_dict(raw.get("segment") or {}, source),
        sources=sources,
    )
