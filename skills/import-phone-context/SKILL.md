---
name: import-phone-context
description: Pull conversation transcripts and notes off a paired iPhone in bulk via a local backup, and stage them as vault inbox drops. Use for Even Realities Conversate transcripts, Apple Notes, and Voice Memos. Triggers on "/import-phone-context", "import from my phone", "sync my glasses conversations", "pull conversate transcripts", "get my notes off my phone".
---

# /import-phone-context — bulk-import phone material into the vault

Extract material the phone holds locally and stage it as inbox drops for `memvault ingest`.
Everything here is read-only with respect to the phone.

**The one rule:** this skill writes into `inbox/` and nowhere else. It never writes into the
vault proper, never classifies, and never removes an inbox entry — those belong to
`memvault ingest`, per `docs/inbox-contract.md`. A feeder that classified would create a second
opinion about the personal/work boundary, which is the one decision this system keeps in one
place.

## Why a backup rather than an app or an export

Settled on 2026-08-02, recorded in `docs/brainstorms/2026-08-02-conversate-extraction-findings.md`:

- **An Even Hub app cannot do this.** The whole bridge surface of `@evenrealities/even_hub_sdk`
  is sixteen display/input methods with no access to Conversate's data. A Hub app is also
  foreground-only.
- **The in-app export is per-conversation.** Fine once, unusable as a habit.
- **A local backup contains everything**, needs no interaction, and is incremental after the
  first run.

## Before starting

| Check | Command | Expected |
|---|---|---|
| Device visible | `idevice_id -l` | a UDID |
| Trusted | `idevicepair validate` | `SUCCESS` |
| Not force-encrypted | `ideviceinfo -q com.apple.mobile.backup` | `RequiresEncryption: 0` |
| Disk headroom | `df -h /` | comfortably more than the phone's used space |

If `idevice_id -l` is empty but the phone is plugged in, check the USB tree directly with
`ioreg -p IOUSB -w 0 | grep -i iphone` before concluding anything. A charge-only cable and a
not-yet-trusted device look identical from `idevice_id` alone.

**A macOS "Installation failed … not currently available from the Software Update server" dialog
is a red herring.** It appears when the phone's iOS is newer than the Mac's bundled device
support (seen with iOS 27.0 on macOS 26.5.2). Finder cannot browse the device, but pairing still
validates and `idevicebackup2` completes normally. Do not chase it.

## Steps

1. **Confirm with the user before backing up.** A full first backup is tens of gigabytes and
   takes roughly fifteen minutes over USB. Say the expected size and duration, and where it will
   be written. Never start one unasked.

2. **Back up** to a scratch directory, not the default location:
   ```
   idevicebackup2 -u <udid> backup --full <scratch-dir>
   ```
   Run it in the background and report progress. Finishes with `Backup Successful.`

3. **Locate the stores** in `<scratch-dir>/<udid>/Manifest.db`. Files are addressed by
   `fileID`, stored at `<fileID[:2]>/<fileID>`:

   ```sql
   SELECT fileID FROM Files WHERE domain=? AND relativePath=?;
   ```

   | Material | domain | relativePath |
   |---|---|---|
   | Conversate | `AppDomain-com.even.sg` | `Documents/conversate.db` |
   | Apple Notes | `AppDomainGroup-group.com.apple.notes` | `NoteStore.sqlite` |
   | Voice Memos | `AppDomainGroup-group.com.apple.VoiceMemos.shared` | `Recordings/CloudRecordings.db` |

4. **Copy the stores out** to `private/phone-import/<YYYY-MM-DD>/` in the vault. That prefix is
   gitignored — verify with `git check-ignore -v <path>` before copying anything, and stop if it
   is not ignored.

5. **Render drops into `inbox/`.** One file per conversation or note. Frontmatter carries
   `source`, `date`, and any `participants` the store knows. **Never write `classification`** —
   that is the engine's decision. See the Conversate schema below.

6. **Delete the backup** once the stores are copied, and say how much space was reclaimed.

7. **Commit the inbox**, then stop. Tell the user to run `memvault ingest` (or let the 21:00 job
   do it). Do not run the ingest as part of this skill: the inbox commit is the reviewable
   record of what arrived, and it should be reviewable before anything acts on it.

## Conversate schema

```sql
local_conversations(converse_id PK, title, create_time, detail_json, updated_at, ...)
conversation_messages(id PK, converse_id, speaker_name, content, timestamp, is_user, ...)
```

`create_time` and `timestamp` are Unix seconds.

Four things learned the hard way:

- **`detail_json` is unreliable.** AI summaries existed on 2 of 13 conversations, and the three
  largest had none. Treat the transcript as the payload and every `detail_json` field —
  `overall_summary_data`, `action_item_data`, `people_wiki_data` — as optional enrichment.
- **`topic_data.topic_seq` is always empty.** Even's own topic segmentation is never populated,
  so MemVault's segmenter does the work. Do not build on it.
- **`speaker_name` is `self` / `others` / empty**, not real names. It is still worth preserving:
  the per-conversation Txt export drops speaker attribution entirely.
- **Empty conversations are common** — 3 of 13 had zero messages. Skip them rather than emitting
  empty drops.

## Duplicates

Mostly handled by the engine, with one gap worth knowing.

A drop's identity is the SHA-256 of its normalized body, so re-importing an unchanged
conversation produces the same id, `ingest` recognizes it as filed, and nothing happens — no
duplicate and no commit, because the file is created and drained within one pass and git sees a
net-zero diff. Repeated full backups are therefore safe to feed straight back in.

The gap is a conversation that **grew** since the last import. More messages means a different
body, so a new id, so it files alongside the old one as a near-duplicate. Content hashing cannot
catch that by design. Keep a cursor at `~/.memvault/conversate-cursor.json` mapping `converse_id`
to the last message timestamp and the emitted hash: skip unchanged conversations, and re-emit
grown ones with a `supersedes:` field naming the earlier filing.

## What not to import

**Messages (`sms.db`) and WhatsApp (`ChatStorage.sqlite`) are deliberately excluded.** They are
overwhelmingly other people's words, and ingesting them would put third parties' private
messages into an indexed, searchable vault they never agreed to. Import them only if the user
asks explicitly, and say plainly what it means when they do.

**Claude app conversations are not extractable.** The iOS app stores only analytics, Firebase
state, cookies, and preferences — no message store. Conversations live server-side. The
claude.ai sync route is a separate, unbuilt feeder.

## Related

- `docs/inbox-contract.md` — what a drop must look like
- `docs/brainstorms/2026-08-02-conversate-extraction-findings.md` — how this was established
- `docs/plans/2026-08-02-001-feat-conversate-sync-and-segmentation-plan.md` — the plan this serves
