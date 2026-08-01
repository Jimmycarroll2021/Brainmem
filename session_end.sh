#!/usr/bin/env bash
# brainmem SessionEnd hook — run the offline consolidation pass.
#
# This is the "sleep" step: distil the session's episodes into semantic facts and
# failure lessons, prune low-utility guidelines, decay stale beliefs. It belongs
# here and not in SessionStart because it is LLM-expensive and because finding the
# invariant across events requires seeing several events at once.
set -euo pipefail
BRAINMEM_DIR="${BRAINMEM_DIR:-$HOME/.brainmem}"
export BRAINMEM_DB="${BRAINMEM_DB:-$BRAINMEM_DIR/memory.db}"
python3 "$BRAINMEM_DIR/brainmem_cli.py" maintain >>"$BRAINMEM_DIR/maintain.log" 2>&1 || true
