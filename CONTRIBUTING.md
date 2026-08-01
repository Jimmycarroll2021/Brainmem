# Contributing

## Running the suites

All four must pass, plus lint:

```bash
python test_brainmem.py   # library invariants (also runs under pytest)
bash smoke_test.sh        # install, CLI, both hooks, cross-process persistence
python e2e_mcp.py         # spawns the real MCP server over stdio
python demo.py            # full lifecycle, no API key needed
ruff check .
```

`smoke_test.sh` needs `bash` and `jq`. `e2e_mcp.py` needs `pip install 'brainmem[mcp]'`.

## The bar

- **Tests first.** Every fix lands with a test that fails before it and passes after.
- **Test the invariant, not the implementation.** These suites exist to catch things
  that fail *silently*: a hook that never fires, a vector space that changes between
  processes, a belief whose confidence rises because it was repeated. Bugs that
  fail loudly need less help.
- **A test that can pass by luck is worse than no test.** If a property holds by
  chance a meaningful fraction of the time, add distractors until it doesn't.
- **numpy is the only required dependency.** Optional backends import lazily inside
  the function that needs them and degrade to the offline default.
- Comments explain **why**. If the code already says what it does, the comment
  shouldn't repeat it.

## Why there are four suites

Most of the interesting bugs in this project were invisible to unit tests, because
the failure was in the seam rather than the function:

- The library imports fine while the installer copies from a directory that does
  not exist.
- The hook is wired correctly and never fires, because a Git Bash path does not
  resolve for the process that spawns it.
- Retrieval returns rows in the right shape while ranking them at random, because
  the embedding hash is salted per process and every caller is a separate process.

`smoke_test.sh` and `e2e_mcp.py` exist for that class of failure. If you change
anything about how the pieces are wired together — paths, processes, the MCP
schema, the hook contract — the test for it belongs there, not in
`test_brainmem.py`.

## The full list of silent failures

The README carries four of these. Here is the rest, because together they are the
argument for the four-suite structure above:

- **MCP SDK 2.0 removed `mcp.server.fastmcp`.** The server imported a module that no
  longer exists. `brainmem_mcp.py` now tries `MCPServer` and falls back to `FastMCP`.
- **`memory_write` misreported outcome conflicts.** It reported "already known" based
  on the gate's verdict while the episode had in fact been stored. The data was right
  and the tool was lying to the agent about it.
- **The token budget only governed facts.** Recent events and skills were unbounded,
  so `token_budget` was closer to a suggestion. Everything is fitted now, in priority
  order: failures, facts, procedures, raw events.
- **`install.sh` and `smoke_test.sh` copied from an `integration/` directory that
  does not exist.** Both die on that line under `set -e`, so neither the installer
  nor the shell suite could run at all. Nothing that merely imports the library
  notices.
- **The dead-hook failure has a Windows form.** Generated paths were `/c/Users/...`,
  which the process Claude Code spawns cannot resolve, and hooks relied on the
  executable bit rather than an explicit interpreter. Now `cygpath` plus
  `bash "<script>"`.
- **One `--success` pinned confidence to 1.00.** `conf = n_success/n_total` overwrote
  the distilled prior, so a belief labelled `unverified, n=1` jumped to maximum
  confidence and 1/1 ranked identically to 9/9. Now Laplace-smoothed: 0.67 and 0.91.
- **Distillation split clauses into claims.** `_extract` kept any 12-240 char
  fragment, so "...not yet load-tested at that volume" became a standalone fact whose
  antecedent stayed behind in L1. Once embedded, a fragment is indistinguishable from
  a fact: it ranks, it fits the budget, it gets injected.
- **`memory_write` silently coerced unrecognised outcomes.** `outcome` was a bare
  `str` in a dict lookup, so a near-miss like `"failure"` became `unknown` — the
  failure signal discarded, and with no outcome left to conflict the observation was
  swallowed as redundant and reported as "already known". It is a `Literal` now,
  enumerated in the tool schema.
- **A `cp` that failed silently.** `smoke_test.sh` runs without `set -e`, so when a
  renamed file stopped existing the copy failed, the install was half-done, and all
  29 checks still reported green. The install step now asserts each file landed.

## Stress testing

`python bench.py` measures retrieval against store size. Things already probed, so
you know what has and has not been checked:

- **8 concurrent processes writing one store**: no lost writes, no errors. SQLite's
  default 5s `busy_timeout` absorbs it. Note the same 128 inputs produce a *different*
  store serial vs parallel (18 facts vs 22) — the gate compares against what is
  already there, so interleaving changes what counts as novel. Not data loss, but
  don't expect byte-identical stores from identical input.
- **Adversarial content** — empty strings, 1MB blobs, null bytes, SQL fragments, RTL
  text, control characters, forged envelope tags: no crashes, no injection (queries
  are parameterised), table intact.
- **Bounded now**: `encode()` refuses empty content and truncates past
  `MAX_CONTENT` (4000 chars). Before this, one caller could push megabytes through
  the embedder into a store three processes read on every session start.
- **Envelope forgery is defended, instruction-shaped content is not.** See
  `SECURITY.md` — memory is a persistence layer for prompt injection, and that is
  the property to reason about before deploying it.

## Optional backends

```bash
pip install 'brainmem[embeddings]'   # then BRAINMEM_EMBEDDER=sentence-transformers
pip install 'brainmem[anthropic]'    # then BRAINMEM_LLM=anthropic
```

Switching embedders changes the vector dimension, so start a fresh store — the old
vectors are not comparable to the new ones.

The live judge test runs only when the `anthropic` package is installed *and*
credentials resolve — either `ANTHROPIC_API_KEY` or an OAuth profile from
`ant auth login`. It skips cleanly otherwise, which is what CI does.
