# brainmem

[![CI](https://github.com/Jimmycarroll2021/Brainmem/actions/workflows/ci.yml/badge.svg)](https://github.com/Jimmycarroll2021/Brainmem/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Agent memory built around the constraint that actually binds: storage is free, attention is not. Encoding is gated on surprise, retrieval is budgeted, and beliefs carry validity intervals and provenance so you can answer *what did the agent believe, when, and on what evidence*.

## Install

```bash
pip install brainmem                          # core; numpy is the only dependency
pip install 'brainmem[mcp,embeddings]'        # MCP tools + real semantic retrieval
./install.sh                                  # wire into Claude Code (PREFIX=… to relocate)
```

## 60 seconds

```python
from brainmem import Memory

m = Memory("memory.db")
m.encode("Validation of the 60MB CSV timed out.", outcome=False)
m.encode("Chunking the CSV to 20MB completed validation.", outcome=True)
m.consolidate()                      # distil episodes into durable facts

print(m.context("run the validation batch", token_budget=600))
# ## What has gone wrong before
# - [1] Avoid: Validation of the 60MB CSV timed out  (unverified, n=1)
# ## What I know
# - [2] Chunking the CSV to 20MB completed validation  (conf 0.60, n=1)

m.record_outcome(2, success=True)    # close the loop; this is what makes it learn
```

Or from the shell: `brainmem encode "..." --outcome fail`, `brainmem retrieve "..."`, `brainmem stats`.

`install.sh` writes `~/.brainmem/settings-brainmem.json` with **absolute paths baked
in**, then merge it into `~/.claude/settings.json`. Do not substitute `$HOME` or `~`:
Claude Code expands variables in `.mcp.json` but not in `settings.json`, so a
placeholder path leaves the hook silently dead. On Windows the same applies to Git
Bash paths — `install.sh` converts them with `cygpath`, so run it from the shell you
intend to use and merge what it generates rather than hand-writing the paths.

When merging, **append** to any `SessionStart` array you already have instead of
replacing it, and re-run `install.sh` after installing the `mcp` SDK — the
`mcpServers` block is only emitted when the SDK is importable at install time.

## Verify

```bash
python test_brainmem.py   # 44 invariants — gating, supersession, budget, utility
bash smoke_test.sh        # 29 checks — install, CLI, both hooks, cross-process persistence
python e2e_mcp.py         # 22 checks — spawns the real MCP server over stdio
python demo.py            # full lifecycle, no API key needed
ruff check .
```

All four pass on a clean checkout, on Linux, macOS and Windows across Python
3.10–3.13. `test_brainmem.py` also runs under `pytest`. `e2e_mcp.py` needs the
`mcp` extra; the rest need only numpy. See [CONTRIBUTING.md](CONTRIBUTING.md) for
why the shell and MCP suites exist separately from the unit tests.

## Layers

| Layer | Role | Key property |
|---|---|---|
| L0 working | assembled context | token-budgeted, never persisted |
| L1 episodic | append-only event log | immutable, carries outcome |
| L2 semantic | distilled propositions | validity intervals, provenance, valence |
| L3 procedural | cached action sequences | scored by success rate |
| core | pinned identity | always loaded |

Processes: `encode()` gates on novelty, `consolidate()` distils offline, `retrieve()` ranks and diversifies, `record_outcome()` closes the feedback loop, `decay()`/`prune_guidelines()` forget.

## Two ways memory reaches the agent

**SessionStart hook** — injects ~600 tokens before the first turn. A floor, not the whole store. It pre-commits against an unknown goal, so it stays small deliberately.

**MCP tools** — `memory_search`, `memory_write`, `memory_outcome`, `memory_explain`, `memory_status`. Defers retrieval to inference time, when the agent knows what it's doing. This is the ceiling.

`SessionEnd` runs `maintain`: consolidate, prune, decay. That's the sleep pass — LLM-expensive and batched, because finding the invariant across events needs several events at once.

## The outcome channel

`encode(..., outcome=True|False|None)` records whether acting on something worked. Three consequences:

1. **Failures distil separately** into `valence='failure'` lessons, and lead the assembled context. Ma et al. (2026) ablated their semantic memory: removing failure reasons cost 8 points, removing success patterns cost 2. Most memory systems store only successes.
2. **Outcome conflict defeats redundancy.** Identical text with a flipped outcome is stored, not collapsed — "I did this before and got a different result" is the strongest prediction error available.
3. **Utility ranks beliefs.** `0.7·confidence + 0.3·usage` after `record_outcome()`. Without it, ranking rewards beliefs that look relevant over beliefs that have been right.

**The catch.** In a simulator the oracle is free. In advisory, analytical, or consulting work there is no oracle — nothing emits `success=True`. You must supply outcomes: a human verdict, a downstream check, a test result. Everything degrades gracefully to `outcome=None`, but the mechanisms carrying most of the measured gain are exactly the ones that need the signal. Wiring `memory_outcome` into a real workflow is the difference between this being useful and being decoration.

## Defaults worth knowing

- `retrieve(k=3)` — retrieval quality saturates fast (74% at k=1, 82% at k=2, flat at k=3 and k=5 in Ma et al.). Raise only with evidence.
- `context()` orders each block in a serial-position V — best at head and tail, weakest in the middle (Liu et al., 2023, "Lost in the Middle").
- Failures are fitted to the token budget *before* facts, so under pressure they're the last thing dropped.
- `prune_guidelines(keep=20)` caps outcome-scored rules; anything ≥0.8 confidence with ≥5 successes is protected.

## Production swaps, in order of impact

1. **Real embedder** — shipped. `pip install 'brainmem[embeddings]'` and set
   `BRAINMEM_EMBEDDER=sentence-transformers`. The `HashEmbedder` default has no
   semantic generalisation (it is hashed n-grams), so "the batch aborted" and "the
   job failed" share no vector mass. Switching changes the vector dimension — start
   a fresh store, because the old vectors are not comparable to the new ones.
2. **`BRAINMEM_LLM=anthropic`** for the write gate and extractor. The heuristic judge cannot detect contradiction reliably; a cosine threshold structurally can't, since "X leads the project" and "X has left the project" embed almost identically.
3. **pgvector or FAISS** above ~100k rows. Only `_nearest_facts` changes.

## Found by testing, not by reading

Bugs the unit tests could not have caught, listed because they show what the
suites are for:

- **MCP SDK 2.0 removed `mcp.server.fastmcp`.** The server imported a module that
  no longer exists. `mcp_server.py` now tries `MCPServer` and falls back to
  `FastMCP`, so it runs on both.
- **`memory_write` misreported outcome conflicts.** It said "already known" based
  on the gate's verdict, while the episode had in fact been stored. The data was
  right and the tool was lying to the agent about it.
- **The token budget only governed facts.** Recent events and skills were
  unbounded, so `token_budget` was closer to a suggestion. Everything is fitted
  now, in priority order: failures, facts, procedures, raw events.
- **`$HOME` in `settings.json` is never expanded.** The original config would have
  installed cleanly and then never fired. Hence `install.sh`.
- **`install.sh` and `smoke_test.sh` copied from an `integration/` directory that
  does not exist.** Both die on that line under `set -e`, so neither the installer
  nor the shell suite could run at all. Nothing that only imports the library
  notices.
- **The same dead-hook failure has a Windows form.** Under Git Bash the generated
  paths were `/c/Users/...`, which the process Claude Code spawns cannot resolve,
  and the hooks relied on the executable bit rather than an explicit interpreter.
  Installed cleanly, never fired. Paths now go through `cygpath` and hooks are
  invoked as `bash "<script>"`.
- **The SessionStart hook read its goal from `$1`.** Claude Code delivers hook
  payloads as JSON on stdin and never passes argv, so in a live session the goal
  was always empty and silently fell back to the working directory — the block
  still rendered, it just stopped being goal-conditioned. Every test passed the
  goal as `$1`, which is exactly why none of them saw it.
- **The embedding hash was salted per process.** `HashEmbedder` used builtin
  `hash()`, which Python salts per process (PEP 456). Every deployment path here —
  the SessionStart hook, the CLI, the MCP server — is a separate process over one
  database, so vectors written by one session were meaningless to the next. The
  store still returned rows in the right shape, so nothing looked broken while
  ranking was quietly random; the same query that ranked a failure lesson first
  in-process ranked it fourth from a fresh process. Now blake2b.
- **Confidence rose on restatement, not evidence.** Both the `encode` redundancy
  path and the `_distil` "reinforced" path did `confidence + 0.05`. But the gate
  decides redundancy by entity overlap and token similarity — it measures how alike
  two *strings* are, not whether two *independent sources* agree, and it cannot tell
  a paraphrase from a caveat. Recording "the thirty percent rule is not a
  demonstrated optimum" made the store **more** certain of the thirty percent rule.
  Restatement now moves `support` only; `confidence` has exactly one mutator,
  `record_outcome`.
- **One `--success` pinned confidence to 1.00.** `conf = n_success/n_total`
  overwrote the distilled prior, so a belief the store itself labelled `unverified,
  n=1` jumped to maximum confidence and 1/1 ranked identically to 9/9 — breaking the
  very rule the docstring states. Now Laplace-smoothed: 0.67, 0.91.
- **Distillation split clauses into claims.** `_extract` kept any 12–240 char
  fragment, so "…not yet load-tested at that volume" became a standalone fact whose
  antecedent stayed behind in L1. A fragment is indistinguishable from a fact once
  embedded: it ranks, it fits the budget, it gets injected. Fragments that cannot be
  read alone are now dropped — the raw episode still holds them.
- **`memory_write` silently coerced unrecognised outcomes.** `outcome` was a bare
  `str` looked up in a dict, so a natural near-miss like `"failure"` became
  `unknown`: the failure signal was discarded, and with no outcome left to
  conflict, the observation was swallowed as redundant and reported as "already
  known". It is a `Literal` now, enumerated in the tool schema and rejected at the
  boundary. Losing the store's most valuable signal quietly is worse than failing
  loudly.

## What remains unproven

Surprisal-gated writes are principled but I know of no clean benchmark showing they beat write-everything at scale. Ma et al. don't test it either — their episodic store grows monotonically. If you find or run such a benchmark, it's the result most likely to change this design.
