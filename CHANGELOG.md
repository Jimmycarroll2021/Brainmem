# Changelog

Notable changes. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [SemVer](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/Jimmycarroll2021/Brainmem/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Jimmycarroll2021/Brainmem/releases/tag/v0.1.0
