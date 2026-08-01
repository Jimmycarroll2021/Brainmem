#!/usr/bin/env bash
# brainmem installer.
#
#   ./install.sh                 # installs to ~/.brainmem
#   PREFIX=/opt/brainmem ./install.sh
#
# Generates settings-brainmem.json with ABSOLUTE, NATIVE paths baked in. This is
# not cosmetic: Claude Code expands environment variables in .mcp.json but NOT in
# settings.json, so a literal "$HOME/..." in a hook command is passed through
# unexpanded and the hook silently never runs. The same failure has a Windows
# form — a Git Bash /c/Users/... path does not resolve for the process Claude
# Code spawns — so paths are converted with cygpath where it exists.

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.brainmem}"
PY="${PY:-python3}"

echo "installing to $PREFIX"
mkdir -p "$PREFIX"
cp "$SRC/brainmem.py" "$SRC/brainmem_cli.py" "$SRC/brainmem_mcp.py" "$PREFIX/"
cp "$SRC/session_start.sh" "$SRC/session_end.sh" "$PREFIX/"
chmod +x "$PREFIX"/*.sh

# --- dependency check ---------------------------------------------------------
if ! "$PY" -c "import numpy" 2>/dev/null; then
  echo "  MISSING: numpy is required.  pip install numpy" >&2
  exit 1
fi
echo "  numpy ok"
if "$PY" -c "import mcp" 2>/dev/null; then
  echo "  mcp ok"
  MCP_OK=1
else
  echo "  note: mcp SDK not found — hooks will work, MCP tools will not."
  echo "        install with: pip install 'mcp[cli]'"
  MCP_OK=0
fi

# --- generate settings with real paths ----------------------------------------
# Resolve the interpreter to a real executable. `command -v python3` on Windows
# returns the WindowsApps Store alias, which is a launcher shim and not a usable
# mcpServers command.
PY_ABS="$("$PY" -c 'import sys; print(sys.executable)')"

# Claude Code spawns hook commands itself rather than through this shell, so the
# paths must be in the platform's native form. Under Git Bash a path like
# /c/Users/... never resolves for the spawned process and the hook silently never
# fires — the same class of failure as an unexpanded $HOME, just Windows-shaped.
# cygpath -m yields C:/Users/... : native drive, forward slashes, JSON-safe.
if command -v cygpath >/dev/null 2>&1; then
  PREFIX_NATIVE="$(cygpath -m "$PREFIX")"
  PY_ABS="$(cygpath -m "$PY_ABS")"
else
  PREFIX_NATIVE="$PREFIX"
fi

OUT="$PREFIX/settings-brainmem.json"

# Hooks are invoked as `bash "<script>"` rather than relying on the shebang and
# the executable bit — neither of which survives on Windows.
{
  printf '{\n  "hooks": {\n'
  printf '    "SessionStart": [\n      { "hooks": [ { "type": "command", "command": "bash \\"%s/session_start.sh\\"" } ] }\n    ],\n' "$PREFIX_NATIVE"
  printf '    "SessionEnd": [\n      { "hooks": [ { "type": "command", "command": "bash \\"%s/session_end.sh\\"" } ] }\n    ]\n' "$PREFIX_NATIVE"
  printf '  }'
  if [ "$MCP_OK" = "1" ]; then
    printf ',\n  "mcpServers": {\n    "brainmem": {\n'
    printf '      "command": "%s",\n' "$PY_ABS"
    printf '      "args": ["%s/brainmem_mcp.py"],\n' "$PREFIX_NATIVE"
    printf '      "env": { "BRAINMEM_DB": "%s/memory.db" }\n' "$PREFIX_NATIVE"
    printf '    }\n  }'
  fi
  printf '\n}\n'
} > "$OUT"

"$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$OUT" \
  && echo "  wrote $OUT (valid JSON, absolute paths)"

# --- verify the install actually works ----------------------------------------
export BRAINMEM_DB="$PREFIX/memory.db"
"$PY" "$PREFIX/brainmem_cli.py" stats >/dev/null && echo "  store initialised"
BRAINMEM_DIR="$PREFIX" bash "$PREFIX/session_start.sh" "install check" >/dev/null \
  && echo "  SessionStart hook runs"

cat <<EOF

Done. Merge the contents of:
    $OUT
into your ~/.claude/settings.json (or .claude/settings.json for one project).

Note: do NOT hand-edit those paths to use \$HOME or ~ — Claude Code does not
expand variables in settings.json, and the hook will silently stop firing.

No API key is needed. The agent judges contradictions itself via
memory_write(verdict=..., target=...) — see the README.

    export BRAINMEM_LLM=anthropic   # only for headless use with no agent present
EOF
