# Changelog

Notable changes. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-08-02

### Added

- **The agent distils, not just judges.** 0.2.0 let the agent decide the write gate
  but left consolidation on the offline extractor, which splits sentences and cannot
  find the invariant across several events — the entire point of the pass. New
  `memory_pending` and `memory_distil` MCP tools (and `Memory.pending()` /
  `Memory.distil()`) hand the raw episodes to the agent and take back what is durably
  true, through the same `_upsert_fact` path — so supersession, reinforcement and
  provenance behave exactly as the offline pass does. An empty proposition list is a
  valid answer that clears the backlog; unknown episode ids are refused, because a
  belief citing episodes that do not exist is worse than no belief.
- **The injected block now asks for writes.** Measured after a day of real use: the
  hook fired every session, consolidation ran every session, and the store held zero
  observations. The block invited reads and never asked for a write, so the agent
  read memory and never wrote it. A store nobody writes to is a config file.

### Fixed

### Fixed

- `SentenceTransformerEmbedder` no longer calls a deprecated method.
  sentence-transformers 5.x renamed `get_sentence_embedding_dimension` to
  `get_embedding_dimension`; ours emitted a `FutureWarning` on every load and would
  have broken outright when the old name goes. Prefers the new name and falls back,
  since the extra allows `>=3.0` where only the old one exists.

### Added

- The optional embedder backend is now covered by tests that load a real model —
  protocol conformance and a guard against our own deprecated call sites. Both skip
  cleanly without the extra, which is how CI runs.

## [0.2.0] — 2026-08-01

### Added

- **The agent can judge, so no API key is needed.** `encode()`, `memory_write` and
  `brainmem encode` now accept `verdict` and `target`. Inside Claude Code there is
  already a frontier model holding the memory block in its context; it supplies the
  verdict itself rather than brainmem paying for a second model to re-read the same
  two sentences. This closes the gap the offline heuristic structurally cannot:
  "Deploy approval moved to the security team" carries no state-change cue, so the
  entity-and-negation gate scores it `novel` against "Deploys are approved by the
  platform lead" — the agent scores it `contradiction` and the stale belief retires.
  A `contradiction` without a `target` is refused rather than silently downgraded.
- `BRAINMEM_LLM=anthropic` is demoted to what it is: the headless option for when no
  agent is present. It is no longer the recommended path.

## [0.1.1] — 2026-08-01

### Security

- **Stored memory could forge the envelope it is injected inside.** `context()` is
  rendered within `<memory source="brainmem">…</memory>`, and the "evidence, not
  instruction" caveat lives inside that block. A stored proposition containing a
  closing tag pushed everything after it outside the wrapper, where the caveat no
  longer applied. Because memory is replayed at every session start, the injection
  persisted rather than passing with the turn — and anything the agent summarises
  can reach the write path. Tag-like sequences are now neutralised on the assembled
  block, case- and whitespace-insensitively.
- **Writes are bounded.** One observation caps at `MAX_CONTENT` (4000 chars) and
  empty writes are refused, so a caller cannot push megabytes through the embedder
  into a store three processes read on every session start.
- Added `SECURITY.md` with the threat model, including what brainmem explicitly does
  *not* defend against.

### Fixed

- **Retrieval no longer degrades on a store shared across processes.** `HashEmbedder`
  used builtin `hash()`, which Python salts per process (PEP 456). The hook, the CLI
  and the MCP server are three processes over one database, so vectors written by one
  session were meaningless to the next. The store still returned rows in the right
  shape, so nothing looked broken while ranking was effectively random. Now blake2b.
- **The SessionStart hook no longer depends on `jq`.** It parsed its payload with jq,
  which is not a declared dependency and is absent on minimal images, while `python3`
  was already required. Where jq was missing the goal silently stopped being
  conditioned and the block still rendered.
- **`memory_write` no longer reports an empty write as "already known."**
- Retrieval is ~2x faster and holds less memory: the scan selected every column and
  decoded blobs one row at a time. At 100k facts a three-row query cost 946ms and
  peaked at 296MB; it now scans `(id, embedding)`, decodes one buffer, and hydrates
  only the winners.

### Changed

- File-backed stores use WAL with `synchronous=NORMAL`. Under the default rollback
  journal a reader blocks the writer, so a SessionEnd consolidation could stall the
  next session start. Best-effort: `:memory:` and network filesystems still run.
- Documented scaling threshold corrected from ~100k to ~20k live facts, measured
  rather than estimated. `bench.py` reproduces it.

### Added

- `bench.py` — retrieval latency against store size.
- Pluggable embedder with an optional `sentence-transformers` backend
  (`pip install 'brainmem[embeddings]'`, `BRAINMEM_EMBEDDER=sentence-transformers`).

## [0.1.0] — 2026-08-01

First public release. Surprise-gated writes, validity intervals with supersession
and point-in-time recall, failure-valence memory that leads the assembled context,
an outcome channel that is the only thing moving confidence, a SessionStart hook and
an MCP server.

[0.3.0]: https://github.com/Jimmycarroll2021/Brainmem/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Jimmycarroll2021/Brainmem/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Jimmycarroll2021/Brainmem/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Jimmycarroll2021/Brainmem/releases/tag/v0.1.0
