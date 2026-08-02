"""brainmem as an MCP server — memory the agent can pull, not just receive.

    pip install 'brainmem[mcp]'
    brainmem-mcp          # or: python brainmem_mcp.py

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
    memory_pending  raw episodes awaiting distillation
    memory_distil   turn those episodes into durable beliefs
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
        from brainmem import AnthropicLLM  # noqa: PLC0415

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
    verdict: Literal["auto", "novel", "redundant", "refinement", "contradiction"] = "auto",
    target: str = "",
) -> str:
    """Record an observation. It may be gated out as redundant — that is correct.

    outcome: "ok" if acting on this worked, "fail" if it did not, "unknown"
    otherwise. Supply it whenever you can: failures are the single most valuable
    thing this store holds, and nothing else can infer them for you.

    verdict: leave as "auto" and a cheap offline heuristic decides. Set it yourself
    when you can see the answer and the heuristic cannot — you are a language model
    reading both statements, it is entity overlap and a negation word list.

    Set verdict="contradiction" with target="f12" when this observation means a
    belief you were shown is no longer true: a role moved to someone else, a
    threshold changed, a decision was reversed. The heuristic misses these unless
    the new text contains an explicit state-change cue ("left", "cancelled"), so
    "Deploy approval moved to the security team" does NOT register against
    "Deploys are approved by the platform lead" — but it plainly should.
    Contradiction closes the old belief off and links this one as its successor;
    nothing is deleted, and point-in-time queries still return the old answer.

    Get the target id from the memory block you were given at session start, or
    from memory_search. Leave verdict="auto" if you are unsure — a wrong
    contradiction retires a true belief.
    """
    # Literal, not str: an unrecognised value used to fall through to None and be
    # recorded as unknown. A natural near-miss like "failure" therefore discarded
    # the failure signal, and — with no outcome to conflict — the observation was
    # then swallowed as redundant and reported as "already known". Silently losing
    # the most valuable signal in the store is worse than a loud rejection, so the
    # allowed values are enumerated in the tool schema and enforced at the boundary.
    flag = {"ok": True, "fail": False}.get(outcome)
    if verdict == "contradiction" and not target:
        return (
            "Refused: verdict='contradiction' needs target='f<id>' — "
            "which belief does it contradict?"
        )
    r = _mem().encode(
        content,
        actor=actor,
        session=session or None,
        outcome=flag,
        verdict=None if verdict == "auto" else verdict,
        target=target or None,
    )
    # Report what actually happened, not what the gate said. A "redundant"
    # verdict can still be stored when the outcome conflicts with the thing it
    # resembles, and telling the agent otherwise would be a lie it acts on.
    if r["verdict"] == "empty":
        # Distinct from "already known". Reporting an empty write as a successful
        # dedup is the same class of lie as misreporting an outcome conflict: the
        # agent believes something was recorded when nothing was.
        return "Nothing recorded — the observation was empty."
    if r["episode_id"] is None:
        return "Already known — strengthened the existing entry rather than duplicating it."
    if r["verdict"] == "redundant":
        return (
            f"Same as something already recorded, but with a different outcome — "
            f"kept because that is new information (episode {r['episode_id']})."
        )
    return f"Recorded as {r['verdict']} (episode {r['episode_id']})."


@mcp.tool()
def memory_pending(limit: int = 30) -> str:
    """Raw observations waiting to be distilled into durable beliefs.

    Call this when you have a moment at the end of a piece of work, or when
    memory_status shows a backlog. The offline extractor splits sentences; it
    cannot find the invariant across several events, which is the whole point of
    the pass. You can. Read these, decide what is durably true, and send it back
    with memory_distil.
    """
    rows = _mem().pending(limit=limit)
    if not rows:
        return "Nothing pending — everything has been distilled."
    out = [f"{len(rows)} episode(s) awaiting distillation:"]
    out += [
        f"  [{r['id']}] {r['content']}"
        + (f"  [{r['outcome'].upper()}]" if r["outcome"] else "")
        for r in rows
    ]
    out.append(
        "\nGroup the ones that are about the same thing and call memory_distil "
        "per group. Use valence='failure' for groups that record something going "
        "wrong — those are ranked separately and shown first."
    )
    return "\n".join(out)


@mcp.tool()
def memory_distil(
    episode_ids: list[int],
    propositions: list[str],
    valence: Literal["fact", "failure"] = "fact",
) -> str:
    """Turn raw episodes into durable beliefs, citing them as provenance.

    Write what remains true after the moment has passed, not a summary of what
    happened. "Validation fails on inputs above 20MB" is durable; "I ran the
    validation script" is not. Each proposition should stand on its own months
    later, with no other context on screen — no "it", no "that volume", no
    reference to anything outside the sentence.

    Pass an empty propositions list if nothing in the group is worth keeping.
    That is a real answer and it clears the backlog; leaving them pending means
    seeing them again every session.

    valence='failure' for lessons about something going wrong. They distil under
    their own rules, rank separately, and are fitted into the context budget
    before successes — so they are the last thing dropped when space runs out.
    """
    try:
        r = _mem().distil(episode_ids, propositions, valence=valence)
    except ValueError as e:
        return f"Refused: {e}"
    if not any((r["facts_new"], r["facts_reinforced"], r["superseded"])):
        return f"Cleared {r['episodes']} episode(s); nothing durable recorded."
    return (
        f"Distilled {r['episodes']} episode(s): {r['facts_new']} new, "
        f"{r['facts_reinforced']} reinforced, {r['superseded']} superseded."
    )


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


def main() -> None:
    """Console-script entry point (`brainmem-mcp`).

    Also runnable by path, which is how install.sh wires it into settings.json.
    """
    mcp.run()


if __name__ == "__main__":
    main()
