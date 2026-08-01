"""Walk the whole memory lifecycle with no API key required.

    python demo.py

Uses the offline fallbacks (HashEmbedder + HeuristicLLM). Swap in a real embedder
and AnthropicLLM for anything resembling production quality.
"""

import time

from brainmem import DAY, Memory

m = Memory(path=":memory:")
m.pin("I am a delivery assistant for an infrastructure consultancy.")
m.pin("I never state a compliance conclusion without citing its evidence.")

t0 = time.time() - 40 * DAY

# (day, actor, text, outcome)  outcome: True worked, False failed, None unknown
stream = [
    (0, "client", "The Department of Education engagement is led by Priya Raman.", None),
    (0, "client", "Priya Raman is the departmental lead on the Education engagement.", None),
    (1, "system", "The Education compliance checker cut processing time by 89%.", None),
    (2, "client", "Priya wants fortnightly status reports, not weekly.", None),
    (3, "system", "Validation of the 60MB input CSV timed out.", False),
    (4, "self", "Splitting the CSV into 20MB chunks before validation completed.", True),
    (5, "self", "Validation of the 60MB input CSV timed out.", False),  # same text, still kept
    (9, "client",
     "Priya Raman has left the Department; Tom Nguyen now leads the engagement.", None),
    (10, "client", "Tom Nguyen is the new departmental lead.", None),
]

print("=== ENCODE (write gate) ===")
for day, actor, text, outcome in stream:
    r = m.encode(text, actor=actor, session="edu", outcome=outcome, ts=t0 + day * DAY)
    flag = "stored" if r["episode_id"] else "dropped"
    tag = {True: "ok", False: "FAIL", None: "-"}[outcome]
    print(f"  {r['verdict']:<14} {flag:<8} {tag:<5} {text[:56]}")

print("\n=== CONSOLIDATE (offline replay) ===")
print(" ", m.consolidate(now=t0 + 11 * DAY))

m.learn_skill("a validation batch fails on input size", "split the CSV into 20MB chunks and re-run")
m.record_skill_use(1, success=True)
m.record_skill_use(1, success=True)

print("\n=== OUTCOME CHANNEL (utility scoring) ===")
for r in m.db.execute("SELECT id, proposition FROM facts WHERE valence='fact' LIMIT 2"):
    u = m.record_outcome(r["id"], True)
    u = m.record_outcome(r["id"], True)
    print(f"  fact {r['id']} utility {u:.2f}  {r['proposition'][:48]}")
print("  pruned:", m.prune_guidelines(keep=20))

print("\n=== RETRIEVE (who leads the engagement?) ===")
for f in m.retrieve("who is leading the Education engagement", k=4):
    print(f"  [{f.score:5.2f}] {f.proposition}")

print("\n=== POINT-IN-TIME (what did I believe on day 5?) ===")
for f in m.retrieve("who is leading the Education engagement", k=6, at=t0 + 5 * DAY):
    print(f"  {f.proposition}")

print("\n=== PROVENANCE ===")
ev = m.explain(1)
print(f"  {ev.get('proposition')}")
print(f"  live={ev.get('live')} reason={ev.get('retired_reason')}")
for e in ev.get("evidence", []):
    print(f"    <- ep{e['id']} ({e['actor']}): {e['content'][:58]}")

print("\n=== DECAY ===")
print(" ", m.decay(now=time.time()))

print("\n=== ASSEMBLED WORKING SET (failures first) ===")
print(m.context("Run the validation batch for the Education engagement",
                token_budget=600, session="edu"))

print("\n=== STATS ===")
print(" ", m.stats())
