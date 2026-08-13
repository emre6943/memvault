"""Which CLI the classifier command is, and how to speak to it (R11).

`classifier.command` says what to run; `classifier.preset` says what it *is*. Those are
different questions: an adopter may keep any binary under any name, and a wrapper script called
`claude` may be anything at all.

Four call sites reach a model — the classifier, the distiller, the segmenter, and the weekly
curator — and until this module existed each carried its own copy of the same `claude`-shaped
command line and its own call into the same envelope reader. Four copies of one fact is three
chances to drift, and drift here is not visible: a stale argv does not crash, it holds an entire
ingest pass for review. Adding a CLI is now one row in `PRESETS`.

**Every preset feeds the prompt on stdin.** A transcript is far longer than a platform's
argument limit, so a CLI that cannot read its prompt from stdin cannot be a preset at all; it
needs a wrapper script and `preset: custom`.

Exactly two things vary per CLI:

*The arguments.* Documented per preset below and asserted in the tests, because these strings
are the whole contract with a program this repo does not own.

*The envelope.* What the CLI prints around the model's reply. `claude -p --output-format json`
wraps it in a result object; `codex exec --json` streams newline-delimited events and the reply
is the last assistant message; `opencode run` streams prose that carries the JSON object
somewhere inside it; a wrapper script prints the reply and nothing else. Every style unwraps
back to the same thing — the bare JSON object the prompt asked for — so nothing downstream of
`unwrap` knows which CLI answered.

Each style also falls back to "the whole output is the reply" when it finds no envelope it
recognizes. That fallback is what lets a stub in a test, or a naive CLI that prints only what
the model said, work under any preset — and it is why adding a preset cannot break the ones
already shipped.

Only `claude` is verified against a live CLI: it is the one this vault runs, and the plan's
acceptance for this change is that `preset: claude` behaves byte-identically to the code it
replaced. The other three are built from each CLI's documented non-interactive contract as of
2026-08 and are best-effort. When one of them drifts, `custom` plus a three-line wrapper is the
escape hatch that cannot drift, which is the reason it exists.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from memvault.config import ClassifierConfig

logger = logging.getLogger(__name__)

#: Fenced blocks a chatty CLI wraps JSON in. Only blocks that open with `{` are considered, so
#: a model that also shows a shell command in a fence does not have that read as its reply.
_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def flatten(value: object) -> str:
    """One-line rendering, so a multi-line error still fits in a review marker.

    Lives here rather than in `classify` because every module that reads a CLI's output needs
    it and `classify` cannot be imported from this side — the arrow points the other way.
    """
    return " ".join(str(value).split())


class Envelope(StrEnum):
    """What a CLI prints around the model's reply."""

    #: `{"result": "<reply>", "is_error": false, ...}` — the `claude -p --output-format json`
    #: result object.
    CLAUDE_JSON = "claude_json"
    #: Newline-delimited events; the reply is the last assistant message among them.
    CODEX_JSONL = "codex_jsonl"
    #: Prose with the JSON object somewhere inside it, fenced or not.
    TEXT = "text"
    #: The reply and nothing else. What a wrapper script is expected to print.
    BARE = "bare"


@dataclass(frozen=True)
class Preset:
    """One CLI's non-interactive contract: how to call it, and what it prints back.

    `args` come before the model flag and `trailing` after it, because a positional argument —
    codex's `-` stdin sentinel — has to stay last however the middle is built.
    """

    name: str
    args: tuple[str, ...] = ()
    model_flag: str | None = None
    trailing: tuple[str, ...] = ()
    envelope: Envelope = Envelope.BARE

    def argv(self, command: str, model: str | None = None) -> list[str]:
        """The full command line for this preset. Pure; no I/O, no config lookup."""
        argv = [command, *self.args]
        if model:
            if self.model_flag is None:
                # Not an error: a wrapper script is entitled to pick its own model, and a
                # config that names one anyway is stating an intent worth logging once per
                # call rather than a mistake worth holding a drop over.
                logger.info(
                    "classifier.preset %r has no model flag; classifier.model %r is left to "
                    "the command itself",
                    self.name,
                    model,
                )
            else:
                argv += [self.model_flag, model]
        return [*argv, *self.trailing]


#: The table. Keys are `config.CLASSIFIER_PRESETS`; a test asserts the two agree, because a
#: preset accepted at config load with no row here would fail at the first drop instead.
PRESETS: dict[str, Preset] = {
    # `claude -p --output-format json [--model M]`. Byte-for-byte the command line the four
    # call sites each built for themselves before this table existed.
    "claude": Preset(
        name="claude",
        args=("-p", "--output-format", "json"),
        model_flag="--model",
        envelope=Envelope.CLAUDE_JSON,
    ),
    # `codex exec --json --skip-git-repo-check [--model M] -`.
    #
    # `-` is the documented sentinel for "the prompt is on stdin". It is not optional padding:
    # codex hangs waiting for EOF when a prompt arrives as an argument from a non-TTY parent,
    # and this process is always somebody's non-TTY parent.
    #
    # `--skip-git-repo-check` because codex refuses to run outside a git repository and this
    # process's working directory is wherever the pass was launched from, which is not
    # necessarily the vault.
    "codex": Preset(
        name="codex",
        args=("exec", "--json", "--skip-git-repo-check"),
        model_flag="--model",
        trailing=("-",),
        envelope=Envelope.CODEX_JSONL,
    ),
    # `opencode run [--model M]`, prompt on stdin. opencode streams the assistant's text to
    # stdout rather than a machine envelope, so the JSON object is read back out of the prose.
    "opencode": Preset(
        name="opencode",
        args=("run",),
        model_flag="--model",
        envelope=Envelope.TEXT,
    ),
    # The command exactly as written, expecting the bare reply on stdout: the behaviour that
    # existed before presets, and the escape hatch for any CLI with no row above. No flags are
    # added because none can be guessed — a wrapper script owns its own command line.
    "custom": Preset(name="custom", envelope=Envelope.BARE),
}


def preset_for(name: str) -> Preset:
    """Look up a preset by name.

    `config` already rejects an unknown preset at load, so reaching the error here means a
    `ClassifierConfig` was constructed in code with a name no table row answers. It raises
    rather than falling back to `claude`: a silent fallback would run somebody's `codex` binary
    with claude's flags and report the failure as a bad reply.
    """
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown classifier preset {name!r}; expected one of {', '.join(PRESETS)}"
        ) from None


def build_argv(config: ClassifierConfig) -> list[str]:
    """The command line for this classifier config. The one implementation, four callers."""
    return preset_for(config.preset).argv(config.command, config.model)


def envelope_for(config: ClassifierConfig) -> Envelope:
    """Which envelope this classifier config's output arrives in."""
    return preset_for(config.preset).envelope


def unwrap(
    stdout: str, envelope: Envelope, *, subject: str = "the classifier"
) -> tuple[str | None, str | None]:
    """Read the model's reply out of one CLI's stdout, or say why it is not there.

    Returns `(reply, problem)` with exactly one of them set. `subject` is the noun a problem
    names — the four call sites describe themselves as the classifier, the distiller, the
    segmenter and the curator, and every message is built around it so one wording serves all
    four.
    """
    stripped = stdout.strip()
    if not stripped:
        return None, f"{subject} produced no output"

    if envelope is Envelope.BARE:
        return stripped, None
    if envelope is Envelope.CODEX_JSONL:
        return _codex_reply(stripped, subject)
    if envelope is Envelope.TEXT:
        return _text_reply(stripped, subject)
    return _claude_reply(stripped, subject)


def _claude_reply(stdout: str, subject: str) -> tuple[str | None, str | None]:
    """Unwrap `claude -p --output-format json`'s result object.

    A payload that is already the bare reply object is accepted too, so a differently
    configured command — or a stub — does not have to fake the envelope.
    """
    try:
        envelope: Any = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"{subject}'s output was not JSON: {flatten(exc)}"

    if isinstance(envelope, dict):
        if envelope.get("is_error"):
            subtype = envelope.get("subtype") or envelope.get("result") or "unspecified"
            return None, f"{subject} reported an error: {flatten(subtype)}"
        result = envelope.get("result")
        if isinstance(result, str):
            return result, None

    return stdout, None


def _codex_kind(body: dict[str, Any]) -> str:
    return str(body.get("type") or body.get("item_type") or "").strip()


def _codex_text(body: dict[str, Any]) -> str | None:
    for key in ("text", "message", "content"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _codex_event(event: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read one codex event as `(assistant text, reported error)`.

    Two shapes are recognized because codex has shipped both: an `item.completed` event
    carrying an `item`, and the older flat event carrying a `msg`. Anything else is a
    progress event and contributes nothing.
    """
    item = event.get("item")
    if isinstance(item, dict) and _codex_kind(item) == "agent_message":
        return _codex_text(item), None

    body = event.get("msg")
    if not isinstance(body, dict):
        body = event

    kind = _codex_kind(body)
    if kind == "agent_message":
        return _codex_text(body), None
    if kind == "error":
        return None, _codex_text(body) or "unspecified"
    return None, None


def _codex_reply(stdout: str, subject: str) -> tuple[str | None, str | None]:
    """Take the last assistant message out of a `codex exec --json` event stream.

    The last one, not the first: codex may narrate before it answers, and the answer is what
    the prompt asked for. An error event loses to a later message for the same reason — a run
    that recovered and answered has answered.
    """
    text: str | None = None
    error: str | None = None
    saw_event = False

    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            event: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if "type" in event or "msg" in event:
            saw_event = True

        message, failure = _codex_event(event)
        if message is not None:
            text, error = message, None
        elif failure is not None:
            error = failure

    if text is not None:
        return text, None
    if error is not None:
        return None, f"{subject} reported an error: {flatten(error)}"
    if saw_event:
        return None, f"{subject} streamed no assistant message"
    # No line looked like an event at all, so this was not an event stream: read the whole
    # output as the reply, the same fallback every other style makes.
    return stdout, None


def _text_reply(stdout: str, subject: str) -> tuple[str | None, str | None]:
    """Find the JSON object inside a CLI that streams the assistant's prose.

    A fenced block wins over a brace span: a model that explains itself before answering may
    well use a brace in the explanation, and the fence is the one place it says "this is the
    payload".
    """
    for match in _FENCE.finditer(stdout):
        block = match.group(1).strip()
        if block.startswith("{"):
            return block, None

    start, end = stdout.find("{"), stdout.rfind("}")
    if start != -1 and end > start:
        return stdout[start : end + 1], None

    return None, f"{subject}'s output carried no JSON object"
