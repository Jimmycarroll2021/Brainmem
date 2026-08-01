"""End-to-end MCP test: spawn the real server over stdio, call every tool.

This is not a mock. It launches brainmem_mcp.py as a subprocess, performs the MCP
handshake, lists tools, and exercises the full write -> consolidate -> search ->
outcome -> explain loop through the protocol, then verifies the effects landed in
the database on disk.

    python e2e_mcp.py
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
FAILS: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILS.append(label)


def text_of(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


async def main() -> int:
    db = Path(tempfile.mkdtemp()) / "e2e.db"
    env = {**os.environ, "BRAINMEM_DB": str(db), "PYTHONPATH": str(HERE)}
    params = StdioServerParameters(
        command=sys.executable, args=[str(HERE / "brainmem_mcp.py")], env=env
    )

    print("=== MCP server over stdio ===")
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()

        tools = {t.name for t in (await s.list_tools()).tools}
        expected = {
            "memory_search",
            "memory_write",
            "memory_outcome",
            "memory_explain",
            "memory_status",
        }
        check(expected <= tools, "all five tools registered", f"got {sorted(tools)}")

        # -- empty store must not error --------------------------------
        r = text_of(await s.call_tool("memory_search", {"query": "anything"}))
        check("No relevant memory" in r, "empty store returns cleanly", r[:80])

        # -- write, including a failure --------------------------------
        r = text_of(
            await s.call_tool(
                "memory_write",
                {
                    "content": "Validation of the 60MB input CSV timed out.",
                    "outcome": "fail",
                    "actor": "system",
                },
            )
        )
        check("Recorded" in r, "memory_write records a failure", r[:80])

        r = text_of(
            await s.call_tool(
                "memory_write",
                {
                    "content": "Chunking the CSV to 20MB completed validation.",
                    "outcome": "ok",
                    "actor": "self",
                },
            )
        )
        check("Recorded" in r, "memory_write records a success", r[:80])

        # -- the outcome vocabulary must be enforced, not silently coerced --
        # "failure"/"success" are natural near-misses. When outcome was a bare
        # str they fell through to unknown, discarding the failure signal and
        # then getting swallowed as redundant with an "already known" reply.
        schema = next(
            t for t in (await s.list_tools()).tools if t.name == "memory_write"
        )
        props = (
            getattr(schema, "input_schema", None) or schema.inputSchema
        )["properties"]
        check(
            props["outcome"].get("enum") == ["ok", "fail", "unknown"],
            "outcome values are enumerated in the tool schema",
            str(props["outcome"]),
        )
        r = text_of(
            await s.call_tool(
                "memory_write",
                {"content": "A near-miss outcome vocabulary.", "outcome": "failure"},
            )
        )
        check(
            "validation error" in r.lower() or "error" in r.lower(),
            "an unrecognised outcome is rejected, not silently downgraded",
            r[:80],
        )

        # -- duplicate must be gated -----------------------------------
        r = text_of(
            await s.call_tool(
                "memory_write",
                {"content": "Chunking the CSV to 20MB completed validation."},
            )
        )
        check("Already known" in r, "duplicate gated through the protocol", r[:80])

        # -- outcome conflict must NOT be gated ------------------------
        r = text_of(
            await s.call_tool(
                "memory_write",
                {
                    "content": "Chunking the CSV to 20MB completed validation.",
                    "outcome": "fail",
                },
            )
        )
        check(
            "Already known" not in r,
            "same text + flipped outcome is stored, not collapsed",
            r[:80],
        )

        # -- consolidation happens out of band, via the CLI ------------
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(HERE / "brainmem_cli.py"),
            "consolidate",
            env=env,
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        check(b"facts_new" in out, "CLI consolidate runs against the same DB", out[:90])

        # -- search now returns something ------------------------------
        r = text_of(await s.call_tool("memory_search", {"query": "validation CSV"}))
        check("PRIOR FAILURES" in r, "failures surfaced first in search", r[:120])
        check("[" in r, "results carry ids for the outcome loop", r[:120])

        fact_id = int(r.split("[")[1].split("]")[0])

        # -- outcome loop ----------------------------------------------
        r = text_of(
            await s.call_tool("memory_outcome", {"fact_id": fact_id, "worked": True})
        )
        check("utility now" in r, "memory_outcome updates utility", r[:80])

        r2 = text_of(
            await s.call_tool("memory_outcome", {"fact_id": fact_id, "worked": True})
        )
        u1 = float(r.rsplit(" ", 1)[1].rstrip("."))
        u2 = float(r2.rsplit(" ", 1)[1].rstrip("."))
        check(u2 > u1, "repeated success raises utility", f"{u1} -> {u2}")

        # -- provenance -------------------------------------------------
        r = text_of(await s.call_tool("memory_explain", {"fact_id": fact_id}))
        check("<- ep" in r, "memory_explain resolves to raw episodes", r[:100])

        # -- status ------------------------------------------------------
        r = text_of(await s.call_tool("memory_status", {}))
        check("failure_lessons" in r, "memory_status reports", r[:80])

        # -- bad input must not crash the server -------------------------
        r = text_of(await s.call_tool("memory_explain", {"fact_id": 99999}))
        check("No fact" in r, "unknown id handled without crashing", r[:80])
        r = text_of(await s.call_tool("memory_status", {}))
        check("episodes" in r, "server still alive after bad input", r[:60])

        # -- concurrency: sqlite connections are per-call ----------------
        results = await asyncio.gather(
            *[s.call_tool("memory_search", {"query": f"topic {i}"}) for i in range(8)]
        )
        check(len(results) == 8, "8 concurrent tool calls all returned")

    # -- effects persisted to disk after the server exited -------------------
    check(db.exists(), "database written to disk")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    n_ep = con.execute("SELECT COUNT(*) c FROM episodes").fetchone()["c"]
    n_fail = con.execute(
        "SELECT COUNT(*) c FROM facts WHERE valence='failure'"
    ).fetchone()["c"]
    n_scored = con.execute("SELECT COUNT(*) c FROM facts WHERE n_total>0").fetchone()["c"]
    check(n_ep >= 3, "episodes persisted", f"n={n_ep}")
    check(n_fail >= 1, "failure lesson persisted", f"n={n_fail}")
    check(n_scored >= 1, "utility scores persisted", f"n={n_scored}")

    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all MCP checks passed'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
