"""Command-line surface for brainmem, so hooks and scripts don't need Python.

    python brainmem_cli.py context "prepare the status report" --budget 800
    python brainmem_cli.py encode "Priya has left the Department" --outcome fail
    python brainmem_cli.py retrieve "who leads Education" -k 3
    python brainmem_cli.py consolidate
    python brainmem_cli.py maintain          # consolidate + prune + decay
    python brainmem_cli.py outcome 12 --success
    python brainmem_cli.py stats

Store location: $BRAINMEM_DB, else ~/.brainmem/memory.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brainmem import Memory, make_embedder  # noqa: E402


def db_path() -> str:
    p = Path(os.environ.get("BRAINMEM_DB", Path.home() / ".brainmem" / "memory.db"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def build() -> Memory:
    """Swap the embedder/LLM here to upgrade every caller at once.

    Set BRAINMEM_LLM=anthropic to use a real judge (needs ANTHROPIC_API_KEY).
    The offline default is deliberately weak; see brainmem.HeuristicLLM.
    """
    llm = None
    if os.environ.get("BRAINMEM_LLM") == "anthropic":
        from brainmem import AnthropicLLM  # noqa: PLC0415

        llm = AnthropicLLM()
    return Memory(path=db_path(), embedder=make_embedder(), llm=llm)


def main() -> int:
    ap = argparse.ArgumentParser(prog="brainmem")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("context", help="assemble a working set for a goal")
    c.add_argument("goal")
    c.add_argument("--budget", type=int, default=800)
    c.add_argument("--session", default=None)

    e = sub.add_parser("encode", help="write an observation through the gate")
    e.add_argument("content")
    e.add_argument("--actor", default=None)
    e.add_argument("--session", default=None)
    e.add_argument("--outcome", choices=["ok", "fail"], default=None)
    e.add_argument(
        "--verdict",
        choices=["novel", "redundant", "refinement", "contradiction"],
        default=None,
        help="judge it yourself instead of letting the offline gate guess",
    )
    e.add_argument("--target", default=None, help="belief this refers to, e.g. f12")

    r = sub.add_parser("retrieve", help="query the semantic store")
    r.add_argument("query")
    r.add_argument("-k", type=int, default=3)
    r.add_argument("--valence", choices=["fact", "failure"], default=None)

    o = sub.add_parser("outcome", help="record whether acting on a fact worked")
    o.add_argument("fact_id", type=int)
    g = o.add_mutually_exclusive_group(required=True)
    g.add_argument("--success", action="store_true")
    g.add_argument("--failure", action="store_true")

    x = sub.add_parser("explain", help="provenance chain for a fact")
    x.add_argument("fact_id", type=int)

    sub.add_parser("consolidate", help="offline episodic -> semantic pass")
    sub.add_parser("stats")
    mt = sub.add_parser("maintain", help="consolidate, prune guidelines, decay")
    mt.add_argument("--keep", type=int, default=20)

    a = ap.parse_args()
    m = build()

    if a.cmd == "context":
        print(m.context(a.goal, token_budget=a.budget, session=a.session))
    elif a.cmd == "encode":
        outcome = {"ok": True, "fail": False}.get(a.outcome)
        print(
            json.dumps(
                m.encode(
                    a.content,
                    actor=a.actor,
                    session=a.session,
                    outcome=outcome,
                    verdict=a.verdict,
                    target=a.target,
                )
            )
        )
    elif a.cmd == "retrieve":
        for f in m.retrieve(a.query, k=a.k, valence=a.valence):
            print(f"[{f.id}] {f.proposition}  (conf {f.confidence:.2f}, utility {f.utility:.2f})")
    elif a.cmd == "outcome":
        print(json.dumps({"utility": m.record_outcome(a.fact_id, a.success)}))
    elif a.cmd == "explain":
        print(json.dumps(m.explain(a.fact_id), indent=2))
    elif a.cmd == "consolidate":
        print(json.dumps(m.consolidate()))
    elif a.cmd == "maintain":
        out = m.consolidate()
        out["guidelines_pruned"] = m.prune_guidelines(keep=a.keep)
        out.update(m.decay())
        print(json.dumps(out))
    elif a.cmd == "stats":
        print(json.dumps(m.stats(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
