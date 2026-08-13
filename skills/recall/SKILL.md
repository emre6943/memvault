---
name: recall
description: Hybrid keyword + semantic search over the vault's index, returning cited vault-relative paths to open. Use before answering from memory, before starting work on a recurring topic, or whenever a past decision might already be written down. Triggers on "/recall", "what do I know about", "did I write anything about", "have we decided this before", "search my vault", "look this up in the vault".
---

# /recall — ask the vault what it already knows

Search the indexed vault for material related to `$ARGUMENTS` and come back with a
short list of cited files. The command returns pointers, not content — depth comes
from opening the two or three files that look right, not from the snippets.

## Steps

1. **Search**: run `memvault recall "$ARGUMENTS" --limit 8` from anywhere — the
   installed command finds its own config, so this does not need a checkout or a
   working directory. Pass the user's phrasing through as-is; quotes, `*`, and `NEAR`
   are escaped for you and never need stripping.
   (Working from a clone of the engine instead? `uv run memvault …` does the same
   thing; everything below is identical either way.)
2. **Narrow when the ask is scoped**: "transcripts from last month" is
   `--source transcripts --since YYYY-MM-DD`; a named origin is `--source whatsapp`.
   `--source` matches a file's declared frontmatter `source` or its top-level vault
   directory, and repeats. Dates are `YYYY-MM-DD`, both bounds inclusive.
3. **Read the cited files, not the snippets** — open the top 2-3 paths in full and
   answer from them. The snippet exists to choose what to open; a session that pastes
   result blocks around has spent its context on the index instead of the memory.
4. **Respect a `superseded by` line**: a result rendered with one has been replaced.
   Open the successor it names, and cite the old file only when the question is what
   was believed at the time.
5. **Handle an empty result honestly**: relay the command's message. Nothing found
   means nothing is filed, and inventing an answer from adjacent material is the one
   failure this skill exists to prevent. Retry once with different words before
   concluding.
6. **Read a `semantic search is off` line as a caveat on the emptiness**: only the
   keyword half ran, so a paraphrase that shares no word with the file was never
   searched for. Say so rather than reporting a clean "nothing filed", and relay the
   fix the message names — installing the `semantic` extra, or re-running
   `memvault index`.
7. **Check the index when material is missing**: if the user names something they know
   they wrote recently, the index may predate it — say so and suggest
   `memvault index`. The index is a rebuildable cache, never the source of truth.
8. **Cite in the answer**: name the vault-relative path of every file the answer leans
   on, so the reader can check it. Paths, never chunk ids.

## Notes

- Retrieval is reciprocal rank fusion over an FTS5 keyword pass and a brute-force cosine
  pass, so a rare exact token and a paraphrase with no shared words both surface. A
  result tagged `keyword+semantic` was found by both and is the strongest signal.
- Ranking then adjusts that fused score by what the vault says about the file, in small
  additive terms rather than a blend of raw magnitudes: a declared `importance: 1-10`
  (a file that declares none counts as a 5, not as a 1), an exponential recency decay
  with a 180-day half-life by default, and one relation hop. So an old note somebody
  called important can outrank this week's passing mention, which is the point.
- A result tagged `graph` matched neither mode itself — it is one typed relation hop
  from a file that did, and it is cited by its opening section. Treat it as a lead
  rather than as an answer. Two guards keep that honest: a note that links to thirty
  files lifts each of them a thirtieth as much as one that links to three, and a
  `kind: reflection` note never boosts what it cites.
- A file whose frontmatter carries `superseded_by:` is pushed far down and rendered
  with a `superseded by <path>` line. It is never hidden — the record of what was
  believed still has to be findable.
- The weights live under `ranking:` in the config. Zeroing `importance`, `recency` and
  `graph` restores pure fusion exactly, which is where a vault sits until its files
  actually carry `importance:` values.
- A semantic hit must beat the corpus median similarity to be admitted, which is what
  lets a query about nothing return nothing. With an E5 backend, whose unrelated text
  still scores around 0.75, add `--min-similarity` to make that cut bite.
- The semantic half is optional: embeddings ship in the `semantic` extra
  (`uv tool install "memvault[semantic]"`). Without it, or against an index built
  without it, recall still runs the keyword half in full and adds one line saying so.
  That line is load-bearing — half a search returns "nothing matches" exactly as
  confidently as a whole one.
- Recall never writes: it reads the index and the vault, and it is safe to run against
  a vault with uncommitted work in progress.
- No match exits 0 — the absence of a memory is an answer, not a failure.
