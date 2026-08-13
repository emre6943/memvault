# Inbox contract

The interface between MemVault and anything that feeds it. A feeder — a manual paste, a
claude.ai sync job, an EvenClaude ambient capture — writes files into a vault's `inbox/` and
needs nothing else from the engine. This document is what it builds against.

Implemented by `src/memvault/inbox.py`. If the two disagree, that is a bug in one of them.

## A drop

A drop is one file in the vault's `inbox/`, at any depth. Subdirectories are walked, so a feeder
may organize by date or source (`inbox/2026-08/whatsapp/...`) without telling the engine.

A file is read as a drop when all of the following hold:

| Condition | Rule |
|---|---|
| Extension | `.md`, `.txt`, or none at all |
| Encoding | UTF-8, with or without a BOM |
| Content | at least one non-whitespace character after frontmatter is removed |
| Name | no path component begins with `.` |

Everything else is skipped with a logged reason and left in place: a `.pdf`, a photo, a
`.DS_Store`, an empty placeholder. Skipped files stay in the inbox, so `memvault doctor` keeps
reporting them until someone deals with them. **Nothing is deleted by discovery.**

There is no naming convention. `note.txt` is a valid drop.

## Optional frontmatter

A drop may open with a YAML block fenced by `---`, terminated by `---` or `...`. It is entirely
optional — a bare `.txt` is the normal case.

Every field is an **override**: declaring it suppresses the classifier's inference for that
field and nothing else. Declaring `classification` does not stop the classifier from writing a
title; declaring `date` does not stop it from classifying.

| Field | Type | Meaning |
|---|---|---|
| `source` | string | Where the drop came from — `whatsapp`, `conversate`, `g2`, `manual` |
| `date` | `YYYY-MM-DD` | When the material happened, not when it was dropped |
| `tags` | string or list of strings | Free-form labels carried into the filed item |
| `classification` | `personal`, `work`, or `mixed` | Skips classification entirely |
| `local_only` | `true` or `false` | Files under the vault's gitignored prefix; never pushed |
| `participants` | string or list of strings | Who is in the material, so it can be found and deleted on request |
| `importance` | whole number 1–10 | How much this deserves to resurface later. Recall ranks on it; an absent value is neutral, not low |
| `workspace` | string | The feeder's own workspace name, when it has one. A routing hint: it can send the drop's distilled note to a topic area without asking the classifier. A value nothing maps to is ignored, not rejected |

Notes on the parsing, all of which have tests:

- An unquoted `date: 2026-08-01` is a YAML date and is accepted; so is a quoted string. Both
  render as `YYYY-MM-DD`. Any other format is rejected by name.
- `local_only` must be a real boolean. `"no"` and `0` are rejected rather than coerced — this
  flag decides whether content reaches a hosted remote, so it is never guessed.
- `classification` is matched case-insensitively against the three valid values. Anything else
  is rejected rather than passed through to the router.
- `tags` and `participants` accept a bare string as a one-item list.
- `importance` must be a whole number in range; `"high"`, `0`, and `42` are rejected by name.
  Declaring it suppresses the classifier's own score, which is why it is not guessed at — the
  classifier's version of the same field is read tolerantly and silently dropped when unusable,
  because there nothing was overridden.
- **Unknown keys are ignored**, not rejected. A feeder may carry its own bookkeeping
  (`device_battery`, `sync_cursor`) without an engine change. They are logged and dropped, so
  do not use them to pass anything the vault needs to keep.

## Identity and idempotency

Each drop gets an `id`: the SHA-256 of its **body**, after these normalizations:

1. Frontmatter is removed.
2. `CRLF` and lone `CR` become `LF`.
3. A UTF-8 BOM is stripped.
4. Unicode is composed to NFC — so `İstanbul` typed on a Mac and on a phone agree.
5. Surrounding blank lines and trailing whitespace are removed. Interior blank lines and
   leading indentation are content and survive.

Two consequences a feeder should rely on:

- **Metadata is free.** Re-dropping the same transcript with a corrected `date` or added `tags`
  produces the same id, so the ingestion pass recognizes it as already filed and does nothing.
  A feeder that cannot tell whether it already sent something may safely send it again.
- **Content is identity.** A one-character edit to the body is a different drop and will be
  filed as a new item. A feeder that re-sends lightly-edited material will accumulate near
  duplicates; diff against your own last-sync marker if that matters.

Discovery does not de-duplicate within a single pass: two files holding the same body both come
back, sharing an id. Deciding what to do with that is the ingestion pass's job.

## Unparseable drops

A drop whose frontmatter cannot be used is **retained**, never skipped. It comes back as a
normal record with the reason attached, and the ingestion pass holds it in the inbox with a
review marker rather than filing it.

- Malformed YAML inside a well-formed fence: the body is still separated correctly, so fixing
  the frontmatter later leaves the id unchanged and the drop files normally.
- A fence that opens with `---` and never closes: the whole file is kept as the body, since
  there is no way to tell metadata from content. The id therefore includes the fence text and
  will change once the file is fixed.
- A `---` at the top of the file that was a Markdown horizontal rule rather than a fence: the
  block parses as YAML but not as a mapping, so the whole file is kept as the body and the
  first section is not swallowed. Quote or indent a leading rule if you want it left alone.
- An invalid field value (`date: soon`, `local_only: "no"`, `classification: wrok`): every bad
  field is reported at once, so one round of fixing is enough.

The reason is always a single line, so it fits in a review marker.

This is the general posture, stated once: **failure leaves material in the inbox.** A stuck
inbox is discovered today; a misfiled memory is discovered in six months.

## Review markers

When a pass cannot file a drop, it leaves the drop untouched and writes a sibling file named
`<drop filename>.needs-review`. `inbox/note.txt` gets `inbox/note.txt.needs-review`; a drop with
no extension gets one all the same. The marker holds **exactly one line**: the reason, phrased
for the person who will read it.

```
inbox/2026-08/whatsapp/call.md
inbox/2026-08/whatsapp/call.md.needs-review   ← "confidence 0.42 is below the configured threshold 0.70, …"
```

Three properties a feeder can rely on:

- **A marker is never a drop.** Its `.needs-review` extension is not one discovery accepts, so
  markers are skipped on every subsequent pass without a rule of their own.
- **A marker is rewritten, not accumulated.** Re-running after a failed fix refreshes the one
  line; it does not pile up a history. When the same reason comes back, the file does not change
  and the pass makes no commit.
- **A marker disappears with its drop.** Fix the drop — correct the frontmatter, or leave it for
  a better model — and the next successful pass removes both files together.

A drop that is still in the inbox with no marker beside it was not reached: either the pass has
not run, or it refused to start (see below).

## When the pass refuses to start

`memvault ingest` will not run over a vault whose working tree has uncommitted changes **outside**
the inbox, because its own commit is the review surface and must not carry someone else's work.
Uncommitted changes *inside* the inbox are fine — a hand-pasted `note.txt` is the documented
minimum drop, and the pass stages only the exact inbox paths it touched.

Exit statuses: `0` everything filed, `3` some items were held for review, `1` the pass could not
run or its commit failed.

## Worked example

A feeder writes `inbox/2026-08/conversate/2026-08-01-sam-call.md`:

```markdown
---
source: conversate
date: 2026-08-01
classification: work
participants: [Ada, Sam]
tags: [pdftoexcel, client-call]
---

Sam walked through the supplier price lists. He wants one Excel tab per supplier plus
comparison columns, and he expects to teach a template once per supplier rather than per file.

Action: confirm whether the read-back override step needs to match against last week's Excel.
```

Discovery produces one record:

| Field | Value |
|---|---|
| `id` | `sha256` of the body text below the frontmatter, e.g. `9f2c…` |
| `body` | the two paragraphs, verbatim, minus surrounding blank lines |
| `declared.source` | `conversate` |
| `declared.date` | `2026-08-01` |
| `declared.classification` | `work` |
| `declared.participants` | `("Ada", "Sam")` |
| `declared.tags` | `("pdftoexcel", "client-call")` |
| `declared.declared_keys` | `{source, date, classification, participants, tags}` |
| `filename` | `2026-08/conversate/2026-08-01-sam-call.md` |
| `size_bytes` | size of the file on disk |
| `unparseable` | `None` |

Because `classification` is declared, the classifier is not asked for one. Because the record is
`work`, the raw body is filed into the personal vault and only a distilled note crosses into the
work vault — the frontmatter cannot change that.

The minimum viable drop, for comparison, is a file called `note.txt` containing one line of
text. Everything above is optional.

## What a feeder must do

1. Write the file into `inbox/`, at any depth, following the rules under **A drop**.
2. Commit it. The inbox is git-backed; the commit is the audit trail of what arrived.
3. Nothing else. Do not write into the vault directly, and do not remove inbox entries —
   `memvault ingest` drains the inbox itself, only after the vault write has succeeded.

If a drop is still in the inbox after a pass, that is a signal, not a bug. Read the review
marker beside it.
