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
