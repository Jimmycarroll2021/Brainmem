"""brainmem as an MCP server — memory the agent can pull, not just receive.

    pip install "mcp[cli]"
    python mcp_server.py

Why this exists alongside the SessionStart hook: the hook pre-commits to a
context block before the goal is known, using one query embedding. Exposing
retrieval as a tool defers the decision to inference time, when the agent
actually knows what it is doing. The hook is the floor; this is the ceiling.

Tools:
    memory_search   query facts and prior failures
    memory_write    record an observation (gated — may be dropped as redundant)
    memory_outcome  record whether acting on a fact worked
    memory_explain  provenance chain for a belief
    memory_status   store statistics
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brainmem import Memory, make_embedder  # noqa: E402

# The SDK renamed FastMCP -> MCPServer in 2.0 and dropped the old module
# entirely, so a single import breaks on one version or the other. Both expose
# the same .tool() decorator and .run() method, which is all this server uses.
try:  # SDK >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("brainmem")

_DB = os.environ.get("BRAINMEM_DB", str(Path.home() / ".brainmem" / "memory.db"))
Path(_DB).parent.mkdir(parents=True, exist_ok=True)


def _mem() -> Memory:
    # New connection per call: sqlite3 connections are not thread-safe, and MCP
    # servers are concurrent. Cheap enough that pooling isn't worth the risk.
    llm = None
    if os.environ.get("BRAINMEM_LLM") == "anthropic":
        from brainmem import AnthropicLLM

        llm = AnthropicLLM()
    return Memory(path=_DB, embedder=make_embedder(), llm=llm)


@mcp.tool()
def memory_search(query: str, k: int = 3, include_failures: bool = True) -> str:
    """Search long-term memory for what is known about a topic.

    Returns facts and, separately, prior failures. Failures are listed first
    because knowing what went wrong last time is worth more than knowing what
    worked. Each result carries an id — pass it to memory_outcome afterwards so
    the store learns which beliefs are actually load-bearing.

    k defaults to 3; retrieval quality saturates around there and deeper
    retrieval spends context budget it cannot repay.
    """
    m = _mem()
    out: list[str] = []
    if include_failures:
        fails = m.retrieve(query, k=min(k, 3), valence="failure")
        if fails:
            out.append("PRIOR FAILURES:")
            out += [f"  [{f.id}] {f.proposition}" for f in fails]
    facts = m.retrieve(query, k=k, valence="fact")
    if facts:
        out.append("KNOWN:")
        out += [
            f"  [{f.id}] {f.proposition}  (conf {f.confidence:.2f}, "
            f"utility {f.utility:.2f}, n={f.support})"
            for f in facts
        ]
    return "\n".join(out) if out else "No relevant memory found."


@mcp.tool()
def memory_write(
    content: str,
    actor: str = "agent",
    outcome: Literal["ok", "fail", "unknown"] = "unknown",
    session: str = "",
) -> str:
    """Record an observation. It may be gated out as redundant — that is correct.

    outcome: "ok" if acting on this worked, "fail" if it did not, "unknown"
    otherwise. Supply it whenever you can: failures are the single most valuable
    thing this store holds, and nothing else can infer them for you.
    """
    # Literal, not str: an unrecognised value used to fall through to None and be
    # recorded as unknown. A natural near-miss like "failure" therefore discarded
    # the failure signal, and — with no outcome to conflict — the observation was
    # then swallowed as redundant and reported as "already known". Silently losing
    # the most valuable signal in the store is worse than a loud rejection, so the
    # allowed values are enumerated in the tool schema and enforced at the boundary.
    flag = {"ok": True, "fail": False}.get(outcome)
    r = _mem().encode(content, actor=actor, session=session or None, outcome=flag)
    # Report what actually happened, not what the gate said. A "redundant"
    # verdict can still be stored when the outcome conflicts with the thing it
    # resembles, and telling the agent otherwise would be a lie it acts on.
    if r["episode_id"] is None:
        return "Already known — strengthened the existing entry rather than duplicating it."
    if r["verdict"] == "redundant":
        return (
            f"Same as something already recorded, but with a different outcome — "
            f"kept because that is new information (episode {r['episode_id']})."
        )
    return f"Recorded as {r['verdict']} (episode {r['episode_id']})."


@mcp.tool()
def memory_outcome(fact_id: int, worked: bool) -> str:
    """Record whether acting on a retrieved fact actually worked.

    This is the only channel by which memory learns which of its beliefs are
    reliable. Without it, ranking rewards beliefs that look relevant over beliefs
    that have been right — call it after acting on anything memory_search gave you.
    """
    u = _mem().record_outcome(fact_id, worked)
    return f"Fact {fact_id} utility now {u:.2f}."


@mcp.tool()
def memory_explain(fact_id: int) -> str:
    """Show the raw observations a belief was distilled from."""
    ev = _mem().explain(fact_id)
    if not ev:
        return f"No fact {fact_id}."
    lines = [
        f"{ev['proposition']}",
        f"confidence {ev['confidence']:.2f}, support {ev['support']}, live={ev['live']}",
    ]
    if ev.get("retired_reason"):
        lines.append(f"retired: {ev['retired_reason']}")
    lines += [f"  <- ep{e['id']} ({e.get('actor')}): {e['content']}" for e in ev["evidence"]]
    return "\n".join(lines)


@mcp.tool()
def memory_status() -> str:
    """Store statistics — episode counts, live facts, failure lessons."""
    return "\n".join(f"{k}: {v}" for k, v in _mem().stats().items())


if __name__ == "__main__":
    mcp.run()
