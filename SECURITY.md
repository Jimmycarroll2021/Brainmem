# Security

## Reporting a vulnerability

Open a [private security advisory](https://github.com/Jimmycarroll2021/Brainmem/security/advisories/new).
Please don't open a public issue for something exploitable. I'll acknowledge within
a week; this is a solo project, so that is a realistic target rather than an SLA.

## The threat model, stated plainly

brainmem writes text into an LLM's context window and replays it at the start of
every future session. That makes it a **persistence layer for prompt injection**, and
it is the property to reason about before deploying it.

A normal prompt injection lasts one turn. An injection that reaches memory lasts
until someone notices and deletes the row.

**The write path is wider than it looks.** Anything the agent summarises can end up
in `memory_write`: a web page it read, a tool result, a file in a repo, an issue
comment, a log line. If an agent can be talked into writing a sentence, it can be
talked into writing an attacker's sentence.

### What the library defends against

- **Envelope forgery.** `context()` is rendered inside
  `<memory source="brainmem">…</memory>`, and the "this is evidence, not instruction"
  caveat lives inside that block. Stored text containing a closing tag would push
  everything after it outside the wrapper, where the caveat no longer applies.
  Tag-like sequences are neutralised on the assembled block, case- and
  whitespace-insensitively. See `test_stored_content_cannot_forge_the_memory_envelope`.
- **Unbounded writes.** One observation is capped at `MAX_CONTENT` (4000 chars) and
  empty writes are refused, so a single caller cannot push megabytes through the
  embedder into a store that three processes read on every session start.
- **SQL injection.** Every query is parameterised. Fuzzed with quotes, semicolons,
  null bytes and `DROP TABLE` fragments; the schema survives.

### What it does *not* defend against, by design

- **Believable false statements.** brainmem gates on *novelty*, not truth. "The
  deploy key is stored in /tmp/keys.txt" is a perfectly well-formed observation and
  will be stored, ranked, and replayed. Nothing in the pipeline evaluates whether a
  claim is true — that is what `record_outcome` is for, and it requires a real
  signal from outside.
- **Instruction-shaped content.** Neutralising the envelope stops text escaping the
  block. It does not stop text *inside* the block reading as an instruction. The
  block says memory is evidence rather than instruction; a model may still comply
  with a confident-sounding imperative.
- **A hostile local user.** The store is an unencrypted SQLite file with filesystem
  permissions and nothing else. Anyone who can read it can read every belief; anyone
  who can write it can plant one.
- **Secrets hygiene.** brainmem will happily store an API key you hand it and inject
  it into future contexts. The `redact` hook on `Memory(...)` is the place to strip
  them, and it is a no-op by default.

### If you expose `memory_write` to untrusted input

1. Set a `redact` callable on `Memory` and drop credential-shaped strings there.
2. Treat the store as tainted: `brainmem explain <id>` shows the raw episodes behind
   any belief, which is the audit path for "why did it say that".
3. Prefer `BRAINMEM_LLM=anthropic`. The offline heuristic gate cannot judge whether
   a contradiction is real, so a planted belief supersedes a true one more easily.
4. Review `brainmem retrieve "" -k 50` periodically. Memory that nobody reads is
   memory nobody notices has been poisoned.

## Supply chain

- Runtime dependency: numpy. Everything else is stdlib.
- Optional extras (`mcp`, `anthropic`, `sentence-transformers`) are imported lazily
  and are not required to run.
- Releases are published to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/);
  there is no long-lived API token in this repository, and the release workflow runs
  the full suite before it builds.
