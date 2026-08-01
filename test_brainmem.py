"""Invariants worth pinning. Run:  python test_brainmem.py   (or pytest)

These target the parts that are easy to get quietly wrong — temporal validity,
supersession, and the write gate — not the parts that fail loudly.
"""

import contextlib
import json
import pathlib
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = str(pathlib.Path(__file__).resolve().parent)

from brainmem import (
    DAY,
    AnthropicLLM,
    Fact,
    HashEmbedder,
    HeuristicLLM,
    Memory,
    _serial_position,
    make_embedder,
)

T0 = time.time() - 30 * DAY


def _seeded() -> Memory:
    m = Memory()
    m.encode("The Education engagement is led by Priya Raman.", ts=T0)
    m.encode("Priya Raman has left the Department.", ts=T0 + 10 * DAY)
    m.consolidate(now=T0 + 11 * DAY)
    return m


def test_duplicate_is_gated_not_stored():
    m = Memory()
    m.encode("Priya Raman leads the Education engagement.", ts=T0)
    r = m.encode("Priya Raman leads the Education engagement.", ts=T0 + 60)
    assert r["verdict"] == "redundant"
    assert r["episode_id"] is None, "redundant observations must not consume storage"


def test_contradiction_supersedes_rather_than_deletes():
    m = _seeded()
    rows = list(m.db.execute("SELECT * FROM facts ORDER BY id"))
    retired = [r for r in rows if r["valid_to"] is not None]
    assert retired, "a contradicted belief must be closed off"
    assert retired[0]["superseded_by"] is not None
    assert "led by Priya" in retired[0]["proposition"] or "Priya" in retired[0]["proposition"]


def test_superseded_fact_excluded_from_default_retrieval():
    m = _seeded()
    props = [f.proposition for f in m.retrieve("who leads the engagement", k=10)]
    assert all(m.db.execute(
        "SELECT valid_to FROM facts WHERE proposition = ?", (p,)
    ).fetchone()["valid_to"] is None for p in props)


def test_point_in_time_recovers_the_old_belief():
    m = _seeded()
    past = [f.proposition for f in m.retrieve("who leads", k=10, at=T0 + 5 * DAY)]
    now = [f.proposition for f in m.retrieve("who leads", k=10)]
    assert any("led by Priya" in p for p in past), past
    assert not any("led by Priya" in p for p in now), now


def test_valid_from_is_observation_time_not_distillation_time():
    m = _seeded()
    first = m.db.execute("SELECT valid_from FROM facts ORDER BY id LIMIT 1").fetchone()
    assert first["valid_from"] < T0 + DAY, "fact must date from when it was observed"


def test_provenance_resolves_to_raw_episodes():
    m = _seeded()
    ev = m.explain(1)
    assert ev["evidence"], "every fact must be traceable to source episodes"
    assert ev["live"] is False


def test_decay_retires_stale_unsupported_facts():
    m = Memory()
    m.encode("A transient build flag was set to verbose.", ts=T0)
    m.consolidate(now=T0)
    before = m.stats()["facts_live"]
    out = m.decay(now=time.time() + 400 * DAY)
    assert out["facts_retired"] >= 1
    assert m.stats()["facts_live"] < before


def test_high_support_facts_survive_decay():
    m = Memory()
    for i in range(4):
        m.encode("The client requires fortnightly reporting.", ts=T0 + i * DAY)
    m.consolidate(now=T0 + 5 * DAY)
    m.db.execute("UPDATE facts SET support = 6, confidence = 0.9")
    m.db.commit()
    assert m.decay(now=time.time() + 400 * DAY)["facts_retired"] == 0


def test_make_embedder_defaults_to_hash():
    assert isinstance(make_embedder(), HashEmbedder)


def test_make_embedder_honours_explicit_name():
    assert isinstance(make_embedder("hash"), HashEmbedder)


def test_unknown_embedder_fails_loudly():
    """A typo must not silently downgrade retrieval to the offline fallback.

    Falling back would leave a store whose vectors came from two different
    spaces — the same silent-corruption shape as the salted hash, so it errors.
    """
    try:
        make_embedder("sentence-transformer")  # missing plural
    except ValueError as e:
        assert "sentence-transformer" in str(e)
    else:
        raise AssertionError("unknown embedder must raise, not fall back")


def test_missing_optional_backend_names_the_extra():
    """The failure a user actually hits: asked for the real embedder, hasn't installed it."""
    import builtins

    real = builtins.__import__

    def blocked(name, *a, **k):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module named sentence_transformers")
        return real(name, *a, **k)

    builtins.__import__ = blocked
    try:
        make_embedder("sentence-transformers")
    except ImportError as e:
        assert "brainmem[embeddings]" in str(e), str(e)
    else:
        raise AssertionError("must raise ImportError naming the extra")
    finally:
        builtins.__import__ = real


def test_embedding_is_stable_across_processes():
    """Every deployment path is a separate process — hook, CLI, MCP server, all
    reading one database. Builtin hash() is salted per process (PEP 456), so a
    salted embedder silently makes yesterday's vectors meaningless. The failure
    is invisible from the outside because retrieval still returns rows."""
    code = (
        "from brainmem import HashEmbedder; import numpy as np; "
        "print(np.nonzero(HashEmbedder().embed(['validation timed out'])[0])[0].tolist())"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(runs) == 1, f"embedding differs across processes: {runs}"


def test_cross_process_ranking_matches_in_process():
    """The consequence of the above, at the level anyone would actually notice.

    Enough distractors that agreeing by chance is not plausible — with two facts
    a salted embedder still lands on the right order half the time, which would
    make this pass by luck and hide the very bug it exists to catch.
    """
    others = [
        "Rosters lock at midday Thursday and cannot be edited after.",
        "Telemetry retention is capped at ninety days by policy.",
        "Nightly backups run offsite to the Sydney region.",
        "New starters get hardware on their first Monday.",
        "Access reviews are quarterly and owned by the security lead.",
        "Vendor contracts auto-renew unless cancelled sixty days out.",
        "Payroll exports must be lodged before the Wednesday cutoff.",
        "Audit sampling uses a fixed seed for reproducibility.",
    ]
    with tempfile.TemporaryDirectory() as d:
        db = str(pathlib.Path(d) / "m.db")
        m = Memory(path=db)
        for i, c in enumerate(others):
            m.encode(c, ts=T0 + i)
        m.encode("Validation of the 60MB input CSV timed out.", ts=T0 + 50)
        m.consolidate(now=T0 + DAY)
        q = "validation CSV timed out"
        in_proc = [f.id for f in m.retrieve(q, k=5)]
        m.db.close()
        code = (
            "import sys;from brainmem import Memory;"
            "m=Memory(path=sys.argv[1]);"
            f"print([f.id for f in m.retrieve({q!r}, k=5)])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code, db], capture_output=True, text=True, cwd=HERE
        ).stdout.strip()
        assert str(in_proc) == out, f"in-process {in_proc} != cross-process {out}"


def test_serial_position_puts_best_at_the_edges():
    items = [
        Fact(i, f"p{i}", None, 0.5, 0.5, 1, T0, None, T0, 0) for i in range(1, 6)
    ]
    out = _serial_position(items)
    assert out[0].id == 1, "highest-ranked fact belongs at the head"
    assert out[-1].id == 2, "second-ranked belongs at the tail, not the middle"


def test_context_respects_token_budget():
    m = Memory()
    for i in range(40):
        m.encode(f"Finding number {i} concerns subsystem {i} and its failure mode.", ts=T0 + i)
    m.consolidate(now=T0 + 100)
    ctx = m.context("summarise the findings", token_budget=300)
    assert len(ctx) // 4 <= 400, "assembled context must stay near its budget"


# ---------------------------------------------------------------------------
# Outcome channel (added after Ma et al., 2026)
# ---------------------------------------------------------------------------


def test_failures_distil_to_their_own_valence():
    m = Memory()
    m.encode("The 60MB CSV timed out during validation.", outcome=False, ts=T0)
    m.encode("Chunking the CSV to 20MB completed validation.", outcome=True, ts=T0 + DAY)
    m.consolidate(now=T0 + 2 * DAY)
    valences = {r["valence"] for r in m.db.execute("SELECT valence FROM facts")}
    assert "failure" in valences, "failed events must produce failure-valence lessons"
    assert "fact" in valences


def test_failure_and_fact_are_not_merged():
    """Near-identical embeddings, opposite meaning — merging them destroys both."""
    m = Memory()
    m.encode("Validation timed out on the large CSV.", outcome=False, ts=T0)
    m.encode("Validation succeeded on the large CSV.", outcome=True, ts=T0 + 60)
    m.consolidate(now=T0 + DAY)
    rows = list(m.db.execute("SELECT valence FROM facts WHERE valid_to IS NULL"))
    assert len({r["valence"] for r in rows}) == 2


def test_utility_rewards_reliability_over_novelty():
    m = Memory()
    m.encode("Chunking the CSV to 20MB completes validation.", outcome=True, ts=T0)
    m.consolidate(now=T0 + DAY)
    fid = m.db.execute("SELECT id FROM facts LIMIT 1").fetchone()["id"]
    u1 = m.record_outcome(fid, True)
    for _ in range(4):
        u2 = m.record_outcome(fid, True)
    assert u2 > u1, "repeated success must raise utility"
    u3 = m.record_outcome(fid, False)
    assert u3 < u2, "a failure must lower utility"


def test_one_success_does_not_saturate_confidence():
    """A single thumbs-up must not make a belief maximally confident.

    conf = n_success/n_total sets confidence to 1.00 on the first success,
    discarding the distilled prior and ignoring sample size — so a fact the store
    itself flagged as unverified (n=1) outranks everything. This matters most in
    advisory work, where "it worked" is often an unverifiable judgement call.
    """
    m = Memory()
    m.encode("Planning headroom at thirty percent above peak held through the sale.", ts=T0)
    m.consolidate(now=T0 + DAY)
    fid = m.db.execute("SELECT id FROM facts LIMIT 1").fetchone()["id"]
    m.record_outcome(fid, True)
    conf = m.db.execute("SELECT confidence FROM facts WHERE id = ?", (fid,)).fetchone()[0]
    assert conf < 0.95, f"one success must not saturate confidence, got {conf:.2f}"


def test_confidence_separates_one_from_nine_successes():
    """The docstring's stated intent, applied to confidence and not just utility."""
    m = Memory()
    m.encode("Rosters lock at midday Thursday and cannot be edited after.", ts=T0)
    m.encode("Telemetry retention is capped at ninety days by policy.", ts=T0 + 60)
    m.consolidate(now=T0 + DAY)
    ids = [r["id"] for r in m.db.execute("SELECT id FROM facts ORDER BY id LIMIT 2")]
    once, nine = ids[0], ids[1]
    m.record_outcome(once, True)
    for _ in range(9):
        m.record_outcome(nine, True)
    c1, c9 = (
        m.db.execute("SELECT confidence FROM facts WHERE id = ?", (f,)).fetchone()[0]
        for f in (once, nine)
    )
    assert c9 > c1, f"9/9 must outrank 1/1 on confidence, got {c9:.2f} vs {c1:.2f}"


def test_sustained_failure_still_drives_confidence_down():
    m = Memory()
    m.encode("Reconciliation happens against the ledger rather than the bank feed.", ts=T0)
    m.consolidate(now=T0 + DAY)
    fid = m.db.execute("SELECT id FROM facts LIMIT 1").fetchone()["id"]
    for _ in range(8):
        m.record_outcome(fid, False)
    conf = m.db.execute("SELECT confidence FROM facts WHERE id = ?", (fid,)).fetchone()[0]
    assert conf < 0.25, f"repeated failure must collapse confidence, got {conf:.2f}"


# ---------------------------------------------------------------------------
# Restatement is not evidence
#
# The write gate decides redundancy by entity overlap and token similarity. That
# measures how alike two strings are, not whether two independent sources agree,
# so "redundant" is the signature of a restatement rather than of corroboration.
# Confidence is a claim about the world and must move only on observed outcomes.
# ---------------------------------------------------------------------------

_HEADROOM = "Redis capacity headroom is planned at thirty percent above peak."


def _believed(m: Memory) -> tuple[float, int]:
    r = m.db.execute(
        "SELECT confidence, support FROM facts WHERE valid_to IS NULL ORDER BY id LIMIT 1"
    ).fetchone()
    return r["confidence"], r["support"]


def _with_headroom_belief() -> tuple[Memory, float]:
    m = Memory()
    m.encode(_HEADROOM, ts=T0)
    m.consolidate(now=T0 + DAY)
    return m, _believed(m)[0]


def test_restatement_does_not_raise_confidence():
    """Saying the same thing again is evidence about the speaker, not the world."""
    m, seeded = _with_headroom_belief()
    for i, echo in enumerate([
        _HEADROOM,
        "Redis headroom for capacity is planned at thirty percent above the peak.",
        "Redis capacity headroom above peak is planned at thirty percent.",
        "Redis capacity headroom planned above peak is thirty percent.",
    ]):
        r = m.encode(echo, ts=T0 + (2 + i) * DAY)
        assert r["verdict"] == "redundant", f"gate did not call it redundant: {r}"
    conf, _ = _believed(m)
    assert conf == seeded, f"restatement moved confidence {seeded:.2f} -> {conf:.2f}"


def test_a_caveat_cannot_strengthen_the_belief_it_doubts():
    """The observed failure: an argument AGAINST a belief raised its confidence.

    The gate reads a caveat as redundant because it reuses the belief's own words,
    and generic negation is deliberately not a polarity cue (it is usually
    contrastive). So doubt is indistinguishable from agreement here — which is
    exactly why this channel must not be allowed to touch confidence at all.
    """
    m, seeded = _with_headroom_belief()
    r = m.encode(
        "Redis headroom at thirty percent above peak is not a demonstrated optimum.",
        ts=T0 + 2 * DAY,
    )
    assert r["verdict"] == "redundant", f"precondition: gate must mis-read the caveat: {r}"
    conf, _ = _believed(m)
    assert conf <= seeded, f"recording a doubt raised confidence {seeded:.2f} -> {conf:.2f}"


def test_restatement_still_counts_as_support():
    """Support is the retention channel, not the truth channel, and must keep
    counting: decay() spares any fact with support > 1, so a claim worth
    repeating stays reachable even though repeating it proves nothing."""
    m, _ = _with_headroom_belief()
    m.encode(_HEADROOM, ts=T0 + 2 * DAY)
    m.encode("Redis capacity headroom above peak is planned at thirty percent.", ts=T0 + 3 * DAY)
    _, support = _believed(m)
    assert support >= 3, f"restatement must still accrue support, got {support}"
    assert m.decay(now=T0 + 400 * DAY)["facts_retired"] == 0


def test_distilled_restatement_does_not_raise_confidence():
    """The same defect on the consolidation path: a proposition gated redundant
    against an existing fact is called 'reinforced', but one note restating
    itself three ways is still a sample size of one."""
    m = Memory()
    m.encode(
        "Redis capacity headroom is planned at thirty percent above peak. "
        "Redis capacity headroom above peak is planned at thirty percent. "
        "Redis headroom above peak capacity is planned at thirty percent.",
        ts=T0,
    )
    out = m.consolidate(now=T0 + DAY)
    assert out["facts_reinforced"] >= 2, f"precondition: distil must reinforce: {out}"
    conf, support = _believed(m)
    assert conf == 0.6, f"distilled restatement moved confidence to {conf:.2f}"
    assert support >= 3, f"support must still count the restatements, got {support}"


def test_restatement_cannot_outrank_observed_evidence():
    """Confidence multiplies the retrieval score, so whatever can raise it decides
    what a future agent sees. record_outcome caps one success at 0.67 precisely so
    evidence has to accumulate; a free restatement channel bypasses that entirely."""
    m = Memory()
    m.encode(_HEADROOM, ts=T0)
    m.encode("Rosters lock at midday Thursday and cannot be edited after.", ts=T0 + 60)
    m.consolidate(now=T0 + DAY)
    echoed = m.db.execute(
        "SELECT id FROM facts WHERE proposition LIKE '%headroom%'"
    ).fetchone()["id"]
    proven = m.db.execute(
        "SELECT id FROM facts WHERE proposition LIKE '%Rosters%'"
    ).fetchone()["id"]
    for i in range(10):
        m.encode(_HEADROOM, ts=T0 + (2 + i) * DAY)
    for _ in range(9):
        m.record_outcome(proven, True)
    conf = {
        f: m.db.execute("SELECT confidence FROM facts WHERE id = ?", (f,)).fetchone()[0]
        for f in (echoed, proven)
    }
    assert conf[proven] > conf[echoed], (
        f"a belief proven 9/9 must outrank one merely repeated 10 times, "
        f"got {conf[proven]:.2f} vs {conf[echoed]:.2f}"
    )


def test_redundant_but_failed_observation_is_not_dropped():
    m = Memory()
    m.encode("Run the nightly validation batch.", ts=T0)
    r = m.encode("Run the nightly validation batch.", outcome=False, ts=T0 + 60)
    assert r["episode_id"] is not None, (
        "repeating an action and having it fail is new information, not an echo"
    )


def test_pruning_protects_proven_guidelines():
    m = Memory()
    # Genuinely distinct claims. A shared sentence template would be collapsed by
    # the write gate — correctly — leaving nothing to prune.
    claims = [
        "Invoices above ten thousand dollars need a second approver.",
        "Rosters lock at midday Thursday and cannot be edited after.",
        "Telemetry retention is capped at ninety days by policy.",
        "Nightly backups run offsite to the Sydney region.",
        "New starters get hardware on their first Monday.",
        "Reconciliation happens against the bank feed, not the ledger.",
        "Access reviews are quarterly and owned by the security lead.",
        "Capacity headroom is planned at thirty percent above peak.",
        "Severity one incidents page the on-call engineer immediately.",
        "Vendor contracts auto-renew unless cancelled sixty days out.",
        "Payroll exports must be lodged before the Wednesday cutoff.",
        "Audit sampling uses a fixed seed for reproducibility.",
    ]
    for i, c in enumerate(claims):
        m.encode(c, outcome=True, ts=T0 + i)
    m.consolidate(now=T0 + 100)
    ids = [r["id"] for r in m.db.execute("SELECT id FROM facts WHERE valid_to IS NULL")]
    for fid in ids:
        m.record_outcome(fid, True)
    proven = ids[0]
    for _ in range(6):
        m.record_outcome(proven, True)
    assert m.prune_guidelines(keep=3) > 0, "pruning must retire low-utility entries"
    still_live = m.db.execute(
        "SELECT valid_to FROM facts WHERE id = ?", (proven,)
    ).fetchone()["valid_to"]
    assert still_live is None, "a rule proven 7/7 must survive pruning"
    assert m.stats()["facts_live"] < len(ids)


def test_context_carries_ids_so_outcomes_can_be_recorded():
    """Without ids the block is a dead end: memory_outcome takes a fact id, so an
    agent reading only the injected block has nothing to pass it."""
    m = Memory()
    m.encode("The 60MB CSV timed out during validation.", outcome=False, ts=T0)
    m.encode("Chunking the CSV to 20MB completed validation.", outcome=True, ts=T0 + DAY)
    m.consolidate(now=T0 + 2 * DAY)
    ctx = m.context("run the validation batch", token_budget=600)
    live = [r["id"] for r in m.db.execute("SELECT id FROM facts WHERE valid_to IS NULL")]
    shown = [i for i in live if f"[{i}]" in ctx]
    assert shown, f"no fact ids in the block; ids are {live}"


def test_context_puts_failures_before_facts():
    m = Memory()
    m.encode("The 60MB CSV timed out during validation.", outcome=False, ts=T0)
    m.encode("Chunking the CSV to 20MB completed validation.", outcome=True, ts=T0 + DAY)
    m.consolidate(now=T0 + 2 * DAY)
    ctx = m.context("run the validation batch", token_budget=600)
    assert "What has gone wrong before" in ctx
    assert ctx.index("What has gone wrong before") < ctx.index("What I know")


def test_failures_survive_a_tight_budget():
    m = Memory()
    m.encode("The 60MB CSV timed out during validation.", outcome=False, ts=T0)
    for i in range(10):
        m.encode(f"Subsystem {i} was configured with option {i}.", outcome=True, ts=T0 + i)
    m.consolidate(now=T0 + 50)
    ctx = m.context("validation batch configuration", token_budget=180)
    assert "What has gone wrong before" in ctx, "failures must be the last thing dropped"


def test_budget_binds_on_the_whole_block_not_just_facts():
    """Recent events and skills are budgeted too, or the budget is decorative."""
    m = Memory()
    m.encode("The 60MB CSV timed out during validation.", outcome=False, ts=T0)
    for i in range(12):
        m.encode(f"Subsystem {i} reported condition code {i} at handover.", ts=T0 + i)
    m.consolidate(now=T0 + 50)
    m.learn_skill("validation fails on size", "chunk the CSV to 20MB and re-run")
    m.record_skill_use(1, success=True)
    big = len(m.context("validation", token_budget=900))
    small = len(m.context("validation", token_budget=60))
    assert small < big, f"budget must bind: {small} vs {big}"
    assert len(m.context("validation", token_budget=60)) // 4 <= 120


# ---------------------------------------------------------------------------
# Distillation — a semantic fact is read back with its episode gone
# ---------------------------------------------------------------------------


def _props(text: str) -> list[str]:
    return json.loads(HeuristicLLM().complete("EXTRACT", f"- {text}"))["propositions"]


def test_distilled_proposition_carries_its_own_antecedent():
    """The L1 -> L2 step exists to produce claims that survive without their source.

    A clause split off mid-sentence keeps pointing at the sentence it came from,
    which by retrieval time is gone — so it embeds, ranks and gets injected while
    meaning nothing.
    """
    m = Memory()
    m.encode(
        "Q3 capacity recommendation set at 23,400 rps. Outcome unknown until Q3 "
        "actuals; not yet load-tested at that volume.",
        ts=T0,
    )
    m.consolidate(now=T0 + DAY)
    props = [r["proposition"] for r in m.db.execute("SELECT proposition FROM facts")]
    assert any("23,400 rps" in p for p in props), props
    assert not any("that volume" in p for p in props), props
    assert not any(p.startswith("Outcome unknown") for p in props), props


def test_fragment_opening_on_a_pronoun_is_dropped():
    out = _props("Deploys run on Fridays. It was rolled back after about an hour.")
    assert any("Deploys run on Fridays" in p for p in out), out
    assert not any(p.startswith("It ") for p in out), out


def test_unresolved_demonstrative_is_dropped():
    out = _props("The load test peaked at 5,000 rps. Throughput was never validated at that volume.")
    assert any("5,000 rps" in p for p in out), out
    assert not any("that volume" in p for p in out), out


def test_verbless_fragment_is_not_a_claim():
    """A bare noun phrase names a topic without asserting anything, so there is
    nothing a later session could act on, agree with or contradict."""
    out = _props("Owner of the capacity plan unclear.")
    assert out == [], out


def test_standalone_clause_after_a_semicolon_still_distils():
    """The fix must cost recall of genuine claims, not just noise — a semicolon
    routinely joins two independent assertions."""
    out = _props("Priya Raman has left the Department; Tom Nguyen now leads the engagement.")
    assert any("Priya Raman has left" in p for p in out), out
    assert any("Tom Nguyen now leads" in p for p in out), out


# ---------------------------------------------------------------------------
# AnthropicLLM — the production judge, exercised without an API key
# ---------------------------------------------------------------------------


class _Block:
    """A response block. Only text blocks carry `.text`, as in the real SDK, so
    dropping the type filter fails loudly instead of silently concatenating
    reasoning into the JSON."""

    def __init__(self, type_: str, text: str | None = None) -> None:
        self.type = type_
        if text is not None:
            self.text = text


class _FakeMessages:
    def __init__(self, blocks: list[_Block], calls: list[dict], error: Exception | None):
        self._blocks, self._calls, self._error = blocks, calls, error

    def create(self, **kwargs):
        self._calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(content=self._blocks)


@contextlib.contextmanager
def _fake_anthropic(blocks: list[_Block] | None = None, error: Exception | None = None):
    """Stand in for the SDK so the real AnthropicLLM path runs with no API key.

    Patching sys.modules rather than the class keeps the deferred `import
    anthropic` and the client construction inside the code under test.
    """
    calls: list[dict] = []
    client = SimpleNamespace(messages=_FakeMessages(blocks or [], calls, error))
    saved = sys.modules.get("anthropic")
    sys.modules["anthropic"] = SimpleNamespace(Anthropic=lambda *a, **k: client)
    try:
        yield calls
    finally:
        if saved is None:
            del sys.modules["anthropic"]
        else:
            sys.modules["anthropic"] = saved


def _live_judge_available() -> bool:
    """Credentials resolve from env vars OR an OAuth profile on disk, so probe the
    SDK rather than checking one variable and wrongly concluding there is no auth."""
    try:
        import anthropic
    except ImportError:
        return False
    try:
        anthropic.Anthropic()
        return True
    except Exception:
        return False


def test_live_judge_detects_contradiction_the_heuristic_misses():
    """The README's central claim about the real judge, actually exercised.

    Skips without credentials — CI on a public repo has none. HeuristicLLM keys
    contradiction off state-change cues (NEG); a role *transfer* carries none, so
    it reads as agreement. This is the case a real judge is supposed to catch.
    """
    user = (
        "OBSERVATION: The Education engagement is now run by Tom Nguyen.\n"
        "CANDIDATES:\n  [f1] The Education engagement is led by Priya Raman."
    )
    heuristic = json.loads(HeuristicLLM().complete(Memory.GATE_SYS, user))["verdict"]
    assert heuristic != "contradiction", (
        "baseline moved: the heuristic now catches this, so the test no longer "
        f"demonstrates the gap (got {heuristic!r})"
    )
    if not _live_judge_available():
        print("       (live judge skipped: no Anthropic credentials)")
        return
    raw = AnthropicLLM().complete(Memory.GATE_SYS, user).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    assert json.loads(raw).get("verdict") == "contradiction", f"got {raw!r}"


def test_anthropic_llm_joins_text_and_ignores_reasoning_blocks():
    """Thinking is on by default on current models, so every response carries
    non-text blocks. Concatenating them would corrupt the JSON."""
    blocks = [
        _Block("thinking"),  # no .text at all — a naive join raises
        _Block("text", '{"verdict": '),
        _Block("server_tool_use", "NOT JSON"),  # has .text, wrong type — must be skipped
        _Block("text", '"novel", "target": null}'),
    ]
    with _fake_anthropic(blocks):
        out = AnthropicLLM().complete("WRITE GATE.", "OBSERVATION: x")
    assert out == '{"verdict": "novel", "target": null}', out


def test_anthropic_llm_request_carries_model_and_prompts():
    with _fake_anthropic([_Block("text", "{}")]) as calls:
        AnthropicLLM().complete("WRITE GATE. sys", "OBSERVATION: obs")
    assert len(calls) == 1, calls
    kw = calls[0]
    assert kw["model"] == "claude-opus-5", kw["model"]
    assert kw["system"] == "WRITE GATE. sys"
    assert kw["messages"] == [{"role": "user", "content": "OBSERVATION: obs"}]
    assert kw["max_tokens"] >= 2048, (
        "max_tokens covers thinking as well as the reply; a JSON-sized budget truncates it"
    )


def test_write_gate_follows_the_anthropic_verdict():
    """Identical text twice — the heuristic calls that redundant. The point of a
    real judge is that its verdict, not the cosine fallback, drives the write."""
    m = Memory()
    m.encode("Priya Raman leads the Education engagement.", ts=T0)
    with _fake_anthropic([_Block("text", '{"verdict": "novel", "target": null}')]):
        m.llm = AnthropicLLM()
        r = m.encode("Priya Raman leads the Education engagement.", ts=T0 + 60)
    assert r["verdict"] == "novel", r
    assert r["episode_id"] is not None, "the judge said novel, so it must be stored"


def test_gate_degrades_to_novel_when_the_api_fails():
    m = Memory()
    m.encode("Priya Raman leads the Education engagement.", ts=T0)
    with _fake_anthropic(error=RuntimeError("529 overloaded")):
        m.llm = AnthropicLLM()
        r = m.encode("Priya Raman leads the Education engagement.", ts=T0 + 60)
    assert r["verdict"] == "novel", "an API outage must not take the encode path down"
    assert r["episode_id"] is not None


def test_consolidate_survives_an_extractor_outage():
    m = Memory()
    m.encode("The 60MB CSV timed out during validation.", outcome=False, ts=T0)
    with _fake_anthropic(error=RuntimeError("529 overloaded")):
        m.llm = AnthropicLLM()
        out = m.consolidate(now=T0 + DAY)
    assert out["facts_new"] == 0, out
    assert m.stats()["facts_live"] == 0


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
