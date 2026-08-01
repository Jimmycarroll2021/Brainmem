"""Where does this stop being fast enough? Run:  python bench.py

The vector scan is O(n) in live facts and deliberately so — it is the honest
zero-dependency default. This measures where that stops being acceptable, so the
"swap in pgvector" advice in the README is a measured threshold rather than a guess.

Facts are inserted directly, bypassing the write gate: the question here is how
retrieval scales with store size, not how well the gate rejects duplicates.
"""

from __future__ import annotations

import pathlib
import random
import sys
import tempfile
import time

from brainmem import Memory, _blob

WORDS = ["payroll", "telemetry", "roster", "invoice", "backup", "audit", "vendor", "capacity", "checkout", "search", "billing", "onboarding", "reporting", "ingest", "failover", "caching", "latency", "throughput", "quota", "shard", "replica", "index", "migration", "rollback", "deploy", "incident", "alert", "threshold", "pipeline", "schema", "partition", "region", "cluster", "tenant", "contract", "renewal", "approval", "cutoff", "retention", "sampling", "seed", "budget"]


def sentence(rng: random.Random) -> str:
    return " ".join(rng.sample(WORDS, 9)).capitalize() + "."


def build(n: int, path: str) -> Memory:
    m = Memory(path=path)
    rng = random.Random(7)
    now = time.time()
    rows = [
        (s, None, 0.6, 0.5, 1, now, None, now, 0, _blob(m._vec(s)), "fact", "[]")
        for s in (sentence(rng) for _ in range(n))
    ]
    m.db.executemany(
        "INSERT INTO facts (proposition, superseded_by, confidence, utility, support,"
        " valid_from, valid_to, last_used, n_total, embedding, valence, provenance)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    m.db.commit()
    return m


def main() -> int:
    sizes = [int(a) for a in sys.argv[1:]] or [1_000, 10_000, 50_000, 100_000]
    print(f"  {'facts':>8} {'retrieve p50':>13} {'p95':>9} {'context()':>11} {'db MB':>8}")
    print("  " + "-" * 54)
    for n in sizes:
        d = pathlib.Path(tempfile.mkdtemp()) / "bench.db"
        m = build(n, str(d))
        rng = random.Random(99)
        lat = []
        for _ in range(20):
            q = sentence(rng)
            t = time.perf_counter()
            m.retrieve(q, k=3)
            lat.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        m.context("run the validation batch", token_budget=600)
        ctx_ms = (time.perf_counter() - t) * 1000
        lat.sort()
        print(
            f"  {n:>8} {lat[len(lat) // 2]:>11.1f}ms {lat[int(len(lat) * 0.95)]:>7.1f}ms "
            f"{ctx_ms:>9.1f}ms {d.stat().st_size / 1e6:>8.1f}"
        )
        m.db.close()
    print(
        "\n  context() runs on every SessionStart, so that column is the one that\n"
        "  shows up as a stall. Past ~20k live facts it is worth moving\n"
        "  _nearest_facts onto pgvector or FAISS; nothing else changes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
