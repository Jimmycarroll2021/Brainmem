"""
brainmem — a memory system for LLM agents, borrowing the *functions* of human
memory rather than its neural implementation.

Core premise: storage is free, attention is not. Memory is therefore a write-gating
and retrieval-budgeting problem, not a storage problem.

Layers
  L0 working   the context window (assembled per turn, never persisted)
  L1 episodic  append-only event log, immutable, timestamped        (fast learner)
  L2 semantic  distilled propositions with provenance + validity    (slow learner)
  L3 procedural cached action sequences scored by success rate
  core         pinned identity/constitution, always loaded

Processes
  encode()      surprisal-gated write  (novelty, not volume)
  retrieve()    multi-signal scoring + MMR + token budget
  consolidate() offline episodic -> semantic replay (the "sleep" pass)
  decay()       forgetting as a precision mechanism, not a bug

Theory anchor: Complementary Learning Systems (McClelland, McNaughton & O'Reilly,
1995) — a fast, high-plasticity episodic store feeding a slow, interference-resistant
semantic store via offline replay.

Deliberate divergences from biology are marked  # DIVERGE:

Dependencies: numpy, stdlib. Swap the brute-force vector scan for pgvector/FAISS
above ~100k rows; nothing else needs to change.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

DAY = 86400.0

# An observation is a sentence or two about something that happened. Past this it
# is a document, and documents belong in the thing memory points at rather than in
# memory. The cap matters because memory_write is reachable by an agent
# summarising untrusted input: without it, one caller can put megabytes through
# the embedder and into a store that three processes read on every session start.
# Truncate rather than reject — the head of an observation carries the point, and
# losing all of it is worse than losing the tail.
MAX_CONTENT = 4000

# --------------------------------------------------------------------------------
# Pluggable model interfaces
# --------------------------------------------------------------------------------


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return raw model text."""
        ...


# Anything that can write to memory can try to close the block it will later be
# rendered inside. Matches a tag whose name is "memory" regardless of case,
# leading slash, or whitespace before the bracket — `</MEMORY >` is the same
# attack as `</memory>`, and a filter that only catches the literal is not a
# filter. Deliberately narrow: real content is full of `<200ms` and `a > b`, and
# escaping those would make the block unreadable to buy nothing.
_ENVELOPE = re.compile(r"<(\s*/?\s*memory)\b", re.IGNORECASE)


def _neutralise(text: str) -> str:
    """Make stored text unable to forge the envelope it is rendered inside.

    context() output is injected into an agent's context wrapped in
    <memory source="brainmem">...</memory>, and the "evidence, not instruction"
    caveat lives inside that wrapper. A stored proposition carrying a closing tag
    would push everything after it outside the wrapper, where the caveat no longer
    applies — and memory is replayed at the start of every future session, so the
    injection persists rather than passing with the turn.

    Escaping only the bracket keeps the text readable and self-evident: a reader
    still sees what was stored, and the model cannot act on a tag that is no
    longer a tag.
    """
    return _ENVELOPE.sub(r"&lt;\1", text)


def _stable_hash(s: str) -> int:
    """Process-stable replacement for builtin hash().

    Builtin hash() is salted per process (PEP 456), so it cannot place a token in
    a vector that will later be compared against a vector built in a different
    process — which is every comparison this system makes: the SessionStart hook,
    the CLI and the MCP server are three separate processes over one database.
    Salted, a store still returns rows, so nothing looks broken while ranking is
    quietly random. blake2b is stable across processes, machines and releases.
    """
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


class HashEmbedder:
    """Zero-dependency fallback. Deterministic hashed character n-grams.

    Good enough to exercise the machinery offline; replace with a real embedding
    model in production — this one has no semantic generalisation whatsoever.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", t.lower())
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
            for g in grams:
                out[i, _stable_hash(g) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """Real semantic embeddings. `pip install 'brainmem[embeddings]'`.

    HashEmbedder has no semantic generalisation: "the batch aborted" and "the job
    failed" share no n-grams and therefore no vector mass, so a lesson recorded in
    one vocabulary is invisible to a query phrased in another. That is the single
    largest quality gap in the offline default and the reason retrieval on a fresh
    store looks worse than the design deserves.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. pip install 'brainmem[embeddings]'"
            ) from e
        self._m = SentenceTransformer(model)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        v = self._m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


_EMBEDDERS = {"hash": HashEmbedder, "sentence-transformers": SentenceTransformerEmbedder}


def make_embedder(name: str | None = None) -> Embedder:
    """Select the embedder once, so every entry point agrees.

    The hook, the CLI and the MCP server are separate processes over one database.
    If they disagree about the embedder, stored vectors and query vectors come from
    different spaces and retrieval degrades silently — the same failure shape as a
    per-process hash seed. An unknown name therefore raises rather than falling
    back to the offline default.
    """
    key = (name or os.environ.get("BRAINMEM_EMBEDDER") or "hash").strip()
    if key not in _EMBEDDERS:
        raise ValueError(f"unknown embedder {key!r}; choose one of {sorted(_EMBEDDERS)}")
    return _EMBEDDERS[key]()


class HeuristicLLM:
    """Offline stand-in so the system runs with no API key.

    It is entity-anchored rather than similarity-based, because a cosine threshold
    CANNOT detect contradiction: "X leads the project" and "X has left the project"
    embed almost identically. Contradiction detection is a polarity judgement about
    a shared subject, not a distance measurement. This heuristic approximates that
    badly but visibly; production needs a real judge (NLI model or LLM).
    """

    # Only STATE-CHANGE cues drive polarity. Generic negation ("not", "never") is
    # usually contrastive — "Priya wants fortnightly reports, not weekly" negates a
    # predicate, not Priya's status — and using it flips facts that never conflicted.
    NEG = {
        "left", "leaving", "leaves", "departed", "resigned", "quit", "stopped",
        "cancelled", "canceled", "former", "ex", "removed", "ended", "terminated",
        "replaced", "superseded", "retired", "closed",
    }

    # A clause that opens with one of these is finishing a thought someone else
    # started. Split off on its own it keeps pointing at a sentence that no longer
    # travels with it.
    DEPENDENT = {
        "and", "but", "or", "nor", "so", "yet", "then", "also", "plus", "not",
        "never", "because", "although", "though", "while", "whereas", "unless",
        "until", "which", "who", "whom", "whose", "where", "when",
        "it", "its", "this", "that", "these", "those", "they", "them", "their",
        "he", "him", "his", "she", "her", "there", "such", "both",
    }

    # Predicate detection is morphological (-ed/-ing/-s), which misses the common
    # irregulars. Missing them costs precision in exactly the wrong direction: a
    # real claim gets thrown away.
    IRREGULAR = {
        "set", "put", "cut", "led", "met", "ran", "made", "took", "gave", "got",
        "went", "said", "held", "left", "sent", "kept", "built", "found", "lost",
        "won", "paid", "read", "cost", "hit", "split", "spent", "chose", "broke",
        "rose", "fell", "began", "became", "came", "saw", "knew", "grew", "wrote",
    }

    AUX = {
        "is", "are", "was", "were", "be", "been", "being", "am", "has", "have",
        "had", "do", "does", "did", "will", "would", "shall", "should", "can",
        "cannot", "could", "may", "might", "must",
    }

    DET = {
        "the", "a", "an", "this", "that", "these", "those", "all", "each",
        "every", "any", "its", "their", "his", "her", "our", "your", "my", "no",
    }

    PREP = {
        "at", "in", "on", "of", "for", "to", "with", "by", "from", "into",
        "above", "below", "under", "over", "during", "after", "before", "than",
        "against", "beyond", "within", "about", "near", "past",
    }

    def complete(self, system: str, user: str) -> str:
        if "WRITE GATE" in system:
            return self._gate(user)
        if "EXTRACT FAILURE" in system:
            return self._extract(user, failure=True)
        if "EXTRACT" in system:
            return self._extract(user)
        return "{}"

    def _gate(self, user: str) -> str:
        obs, cands = self._split(user)
        best_key, best_ov, best_verdict = None, 0.0, "novel"
        for key, text in cands:
            shared = _entities(obs) & _entities(text)
            ov = _overlap(obs, text)
            if shared:
                # Earliest mention approximates the grammatical subject. Anchoring
                # on the wrong entity silently destroys contradiction detection.
                ent = min(shared, key=lambda e: obs.find(e))
                if _polarity(obs, ent) != _polarity(text, ent):
                    return json.dumps({"verdict": "contradiction", "target": key})
                verdict = "redundant" if ov > 0.60 else "refinement"
            else:
                verdict = "redundant" if ov > 0.80 else ("refinement" if ov > 0.45 else "novel")
            if ov > best_ov:
                best_key, best_ov, best_verdict = key, ov, verdict
        if best_verdict == "novel":
            return json.dumps({"verdict": "novel", "target": None})
        return json.dumps({"verdict": best_verdict, "target": best_key})

    def _extract(self, user: str, failure: bool = False) -> str:
        props: list[str] = []
        for line in user.splitlines():
            line = line.strip(" -\t")
            if len(line) < 12 or line.endswith("?"):
                continue
            for s in re.split(r"(?<=[.!;])\s+", line):
                s = s.strip().rstrip(".;")
                if 12 <= len(s) <= 240 and " " in s and self._standalone(s):
                    props.append(f"Avoid: {s}" if failure else s)
        seen, uniq = set(), []
        for p in props:
            k = p.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        return json.dumps({"propositions": uniq[:8]})

    @classmethod
    def _standalone(cls, s: str) -> bool:
        """Can this fragment be read years later with nothing else on screen?

        Splitting on clause boundaries is cheap but it manufactures fragments, and
        a fragment is indistinguishable from a fact once it has been embedded: it
        ranks, it fits the budget, it gets injected. The antecedent, meanwhile, is
        back in L1 where nobody is looking. Dropping it is the conservative move —
        the raw episode is still there, so nothing is actually lost.
        """
        toks = re.findall(r"[a-z0-9']+", s.lower())
        if not toks:
            return False
        # Sentence-initial capital is the writer's own signal that a new claim
        # starts here rather than the previous one continuing.
        if not (s[0].isupper() or s[0].isdigit()):
            return False
        if toks[0] in cls.DEPENDENT:
            return False
        for i, t in enumerate(toks[:-1]):
            # "this/these/those" are always determiners; "that" is only one after a
            # preposition — elsewhere it is a relativiser ("the run that failed"),
            # which points inside the fragment and resolves fine.
            demo = t in {"this", "these", "those"} or (
                t == "that" and i and toks[i - 1] in cls.PREP
            )
            if demo and toks[i + 1] not in toks[:i] + toks[i + 2:]:
                return False
        return cls._has_predicate(toks)

    @classmethod
    def _has_predicate(cls, toks: list[str]) -> bool:
        """A claim asserts something; a noun phrase merely names it.

        "Outcome unknown until Q3 actuals" survives every length and character
        filter while asserting nothing that a later session could act on, agree
        with or contradict.
        """
        for i, t in enumerate(toks):
            if t in cls.AUX or t in cls.IRREGULAR:
                return True
            if len(t) > 3 and t.endswith("ed"):
                return True
            if len(t) > 4 and t.endswith("ing"):
                return True
            # A trailing -s is a present-tense verb or a plural noun; only the verb
            # reading takes a complement, so a final -s token is not evidence.
            if i < len(toks) - 1 and len(t) > 2 and t.endswith("s") and not t.endswith(
                ("ss", "us", "is")
            ):
                return True
        # Imperatives have no subject to find a verb after: "chunk the CSV" is a
        # bare verb followed by a determiner.
        return len(toks) > 1 and toks[1] in cls.DET

    @staticmethod
    def _split(user: str) -> tuple[str, list[tuple[str, str]]]:
        obs, cands, mode = "", [], None
        for line in user.splitlines():
            if line.startswith("OBSERVATION:"):
                mode = "o"
                obs = line.split(":", 1)[1].strip()
                continue
            if line.startswith("CANDIDATES:"):
                mode = "c"
                continue
            if mode == "c":
                m = re.match(r"\s*\[([fe]\d+)\]\s*(.+)", line)
                if m:
                    cands.append((m.group(1), m.group(2)))
        return obs, cands


class AnthropicLLM:
    """Production judge. `pip install anthropic`.

    max_tokens has to cover thinking as well as the reply: thinking is on by
    default on current models and the cap applies to their sum, so a budget
    sized for the JSON alone truncates the JSON.
    """

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 4096) -> None:
        import anthropic  # noqa: PLC0415

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if b.type == "text")


def _overlap(a: str, b: str) -> float:
    """Overlap coefficient — more robust than Jaccard for paraphrase of unequal length."""
    ta = set(re.findall(r"[a-z0-9']+", a.lower()))
    tb = set(re.findall(r"[a-z0-9']+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _polarity(text: str, entity: str, window: int = 5) -> bool:
    """True if the clause attached to `entity` reads positive.

    Scoping the negation check to the tokens following the entity is what stops
    "Priya has left; Tom now leads" being read as negative about Tom.
    """
    toks = re.findall(r"[a-z0-9']+", text.lower())
    ent = entity.lower()
    if ent not in toks:
        return True
    i = toks.index(ent)
    return not (set(toks[i : i + window + 1]) & HeuristicLLM.NEG)


# --------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------


@dataclass
class Fact:
    id: int
    proposition: str
    subject: str | None
    confidence: float
    importance: float
    support: int
    valid_from: float
    valid_to: float | None
    last_used: float
    use_count: int
    valence: str = "fact"
    n_success: int = 0
    n_total: int = 0
    utility: float = 0.5
    provenance: list[int] = field(default_factory=list)
    score: float = 0.0


@dataclass
class Episode:
    id: int
    ts: float
    session: str | None
    actor: str | None
    kind: str
    content: str
    salience: float
    outcome: bool | None = None
    score: float = 0.0


@dataclass
class Weights:
    """Retrieval signal weights. Tune per deployment; these are sane defaults
    in the spirit of Park et al. (2023) 'Generative Agents'."""

    relevance: float = 1.0
    recency: float = 0.5
    importance: float = 0.5
    usage: float = 0.25
    utility: float = 0.5  # outcome-derived; see Memory.record_outcome
    tau_days: float = 30.0  # recency half-life-ish constant


# --------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  session TEXT, actor TEXT, kind TEXT,
  content TEXT NOT NULL,
  entities TEXT,
  salience REAL DEFAULT 0.5,
  outcome INTEGER,            -- 1 worked, 0 failed, NULL unknown
  consolidated INTEGER DEFAULT 0,
  archived INTEGER DEFAULT 0,
  embedding BLOB
);
CREATE INDEX IF NOT EXISTS ix_ep_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS ix_ep_cons ON episodes(consolidated);

CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY,
  subject TEXT,
  proposition TEXT NOT NULL,
  confidence REAL DEFAULT 0.6,
  importance REAL DEFAULT 0.5,
  support INTEGER DEFAULT 1,
  created_at REAL, valid_from REAL, valid_to REAL,
  superseded_by INTEGER,
  retired_reason TEXT,
  last_used REAL, use_count INTEGER DEFAULT 0,
  valence TEXT DEFAULT 'fact', -- fact | pattern | failure
  n_success INTEGER DEFAULT 0, n_total INTEGER DEFAULT 0,
  utility REAL DEFAULT 0.5,
  provenance TEXT,
  embedding BLOB
);
CREATE INDEX IF NOT EXISTS ix_fact_live ON facts(valid_to);
CREATE INDEX IF NOT EXISTS ix_fact_valence ON facts(valence);

CREATE TABLE IF NOT EXISTS skills (
  id INTEGER PRIMARY KEY,
  trigger TEXT NOT NULL, procedure TEXT NOT NULL,
  attempts INTEGER DEFAULT 0, successes INTEGER DEFAULT 0,
  last_used REAL, embedding BLOB
);

CREATE TABLE IF NOT EXISTS core (
  id INTEGER PRIMARY KEY, content TEXT NOT NULL
);
"""


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _unblob(b: bytes | None, dim: int) -> np.ndarray:
    if not b:
        return np.zeros(dim, dtype=np.float32)
    return np.frombuffer(b, dtype=np.float32)


# --------------------------------------------------------------------------------
# The memory system
# --------------------------------------------------------------------------------


class Memory:
    def __init__(
        self,
        path: str = ":memory:",
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        weights: Weights | None = None,
        redact: Any = None,
    ) -> None:
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        # Three processes share one store — the SessionStart hook, the CLI and the
        # MCP server. Under the default rollback journal a reader blocks the writer
        # and vice versa, so a SessionEnd consolidation can stall the next session's
        # start. WAL lets them proceed concurrently, and NORMAL stops fsyncing on
        # every commit (a crash can lose the last transaction, which for an
        # append-only observation log is a fair trade for not blocking a session).
        # Both are best-effort: `:memory:` cannot do WAL, and neither can a store on
        # a network filesystem. Failing to upgrade is not a reason to refuse to run.
        with contextlib.suppress(sqlite3.DatabaseError):
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.emb = embedder or HashEmbedder()
        self.llm = llm or HeuristicLLM()
        self.w = weights or Weights()
        # DIVERGE: biology has no PII problem. Agent memory accumulates personal
        # data indefinitely, so the write path gets a redaction hook.
        self.redact = redact or (lambda s: s)

    # -- helpers ------------------------------------------------------------

    def _vec(self, text: str) -> np.ndarray:
        return self.emb.embed([text])[0]

    def _rows(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.db.execute(sql, tuple(args)))

    # =======================================================================
    # ENCODE — the write gate
    # =======================================================================

    GATE_SYS = (
        "WRITE GATE. Decide how a new observation relates to what is already known.\n"
        "Reply with JSON only: {\"verdict\": one of "
        "[novel, redundant, refinement, contradiction], \"target\": <candidate key, "
        "e.g. \"f12\", or null>}.\n"
        "novel = adds information no candidate covers.\n"
        "redundant = already fully covered.\n"
        "refinement = sharpens or extends a candidate.\n"
        "contradiction = cannot both be true of the same time period."
    )

    def encode(
        self,
        content: str,
        *,
        actor: str | None = None,
        kind: str = "observation",
        session: str | None = None,
        importance: float = 0.5,
        outcome: bool | None = None,
        ts: float | None = None,
    ) -> dict[str, Any]:
        """Surprisal-gated write.

        `outcome` is the optional ground-truth channel: True if acting on this
        worked, False if it did not, None if unknown. Ma et al. (2026) found
        removing failure reasons cost more accuracy than removing success
        patterns (-8 vs -2 points), so failures are the more valuable signal.

        In a simulator the oracle is free. In advisory or analytical work there
        is no oracle, so `outcome` must be supplied by the caller — a human
        verdict, a downstream check, a test result. Everything below degrades
        gracefully to the None case.

        The single highest-leverage decision in an agent memory system is what NOT
        to store. Encoding is driven by prediction error: if the existing store
        already predicts the observation, we strengthen rather than duplicate.
        """
        content = self.redact(content)

        # Bound the write before anything expensive touches it. An empty
        # observation is not a memory: stored, it takes an episode row and later
        # renders as a blank bullet inside a budget that had to drop something
        # real to make room for it.
        content = content.strip()
        if not content:
            return {"verdict": "empty", "episode_id": None, "target": None}
        if len(content) > MAX_CONTENT:
            content = content[:MAX_CONTENT].rstrip() + " …[truncated]"

        now = ts if ts is not None else time.time()
        v = self._vec(content)

        # Gate against distilled facts AND recent raw episodes. Facts only exist
        # after a consolidation pass, so a facts-only gate is blind for the whole
        # of the first session — exactly when duplicate chatter arrives.
        candidates = self._gate_candidates(v)
        verdict, target = self._gate(content, candidates)

        if verdict == "redundant" and not self._outcome_conflict(target, outcome):
            # Strengthen instead of storing, but strengthen SUPPORT only — never
            # confidence. The gate calls things redundant on entity overlap and
            # token similarity, which measures how alike two strings are, not
            # whether two independent sources agree. Everything it flags is
            # therefore a restatement of wording we already hold, and it cannot
            # tell a paraphrase from a caveat about the same subject: "X is not a
            # demonstrated optimum" reuses X's own words, and generic negation is
            # deliberately not a polarity cue (see HeuristicLLM.NEG), so a
            # recorded doubt reads as agreement. Letting that raise confidence
            # meant an argument against a belief made the store more certain of
            # it. Confidence is a claim about the world and moves only on observed
            # outcomes (record_outcome); support counts assertions and drives
            # retention in decay(), which is the honest reading of a repeat.
            if target and target.startswith("f"):
                self.db.execute(
                    "UPDATE facts SET support = support + 1, last_used = ?"
                    " WHERE id = ?",
                    (now, int(target[1:])),
                )
            elif target and target.startswith("e"):
                self.db.execute(
                    "UPDATE episodes SET salience = MIN(1.0, salience + 0.1) WHERE id = ?",
                    (int(target[1:]),),
                )
            self.db.commit()
            return {"verdict": verdict, "episode_id": None, "target": target}

        salience = {
            "novel": 0.8,
            "refinement": 0.6,
            "contradiction": 0.95,
            "redundant": 0.9,  # reached only via outcome conflict — highly informative
        }[verdict]
        if outcome is False:
            salience = 1.0  # failures are the highest-value training signal
        cur = self.db.execute(
            "INSERT INTO episodes (ts, session, actor, kind, content, entities,"
            " salience, outcome, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                now,
                session,
                actor,
                kind,
                content,
                json.dumps(sorted(_entities(content))),
                max(salience, importance),
                None if outcome is None else int(outcome),
                _blob(v),
            ),
        )
        ep_id = int(cur.lastrowid or 0)

        if verdict == "contradiction" and target and target.startswith("f"):
            self._supersede(int(target[1:]), None, now, reason="contradicted by observation")

        self.db.commit()
        return {"verdict": verdict, "episode_id": ep_id, "target": target}

    def _outcome_conflict(self, target: str | None, outcome: bool | None) -> bool:
        """True if this observation's result differs from the thing it resembles.

        Surprisal is about prediction error, and the strongest prediction error
        available is: I did this before and got a different result. Textual
        redundancy is irrelevant when the outcome flips — that is precisely the
        case worth recording. Without this, every "it failed this time" is
        silently swallowed as a duplicate of "it worked last time".
        """
        if outcome is None or not target:
            return outcome is False
        if target.startswith("e"):
            r = self.db.execute(
                "SELECT outcome FROM episodes WHERE id = ?", (int(target[1:]),)
            ).fetchone()
            prior = None if r is None or r["outcome"] is None else bool(r["outcome"])
            if prior is not None and prior != outcome:
                return True
        return outcome is False

    def _gate_candidates(self, v: np.ndarray, k: int = 8) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = [
            (f"f{f.id}", f.proposition) for f in self._nearest_facts(v, k=k)
        ]
        rows = self._rows(
            "SELECT id, content, embedding FROM episodes WHERE consolidated = 0"
            " AND archived = 0 ORDER BY ts DESC LIMIT 100"
        )
        if rows:
            mat = np.stack([_unblob(r["embedding"], self.emb.dim) for r in rows])
            for i in np.argsort(-(mat @ v))[:k]:
                out.append((f"e{rows[i]['id']}", rows[i]["content"]))
        return out

    def _gate(self, content: str, candidates: list[tuple[str, str]]) -> tuple[str, str | None]:
        if not candidates:
            return "novel", None
        lines = "\n".join(f"  [{key}] {text}" for key, text in candidates)
        user = f"OBSERVATION: {content}\nCANDIDATES:\n{lines}"
        try:
            data = _json(self.llm.complete(self.GATE_SYS, user))
            verdict = str(data.get("verdict", "novel"))
            if verdict not in {"novel", "redundant", "refinement", "contradiction"}:
                verdict = "novel"
            target = data.get("target")
            return verdict, str(target) if target else None
        except Exception:
            return "novel", None

    # =======================================================================
    # CONSOLIDATE — the offline replay pass
    # =======================================================================

    EXTRACT_SYS = (
        "EXTRACT durable propositions from a cluster of related events.\n"
        "Keep only what remains true beyond the moment it was observed. Drop "
        "pleasantries, transient state and anything already implied.\n"
        'Reply with JSON only: {"propositions": ["...", "..."]}'
    )

    FAILURE_SYS = (
        "EXTRACT FAILURE lessons from a cluster of events that did not work.\n"
        "State each as an avoidable cause or a precondition that was missed, not as a "
        "description of what happened. Prefer the form 'X fails when Y' or "
        "'do Y before X'.\n"
        'Reply with JSON only: {"propositions": ["...", "..."]}'
    )

    def consolidate(self, *, batch: int = 200, now: float | None = None) -> dict[str, int]:
        """Episodic -> semantic distillation.

        Must run offline and batched: finding the invariant across events requires
        seeing several events at once. An online, per-observation extractor
        structurally cannot do this — it has a sample size of one.
        """
        now = now if now is not None else time.time()
        rows = self._rows(
            "SELECT * FROM episodes WHERE consolidated = 0 ORDER BY ts LIMIT ?", (batch,)
        )
        if not rows:
            return {"episodes": 0, "facts_new": 0, "facts_reinforced": 0, "superseded": 0}

        vecs = np.stack([_unblob(r["embedding"], self.emb.dim) for r in rows])
        clusters = _agglomerate(vecs, threshold=0.55)

        new = reinforced = superseded = failures = 0
        for idx in clusters:
            group = [rows[i] for i in idx]
            # Split each cluster by outcome before distilling. Mixing worked and
            # failed events into one summary produces mush: the LLM averages over
            # a contradiction. Failures get their own prompt and their own valence
            # so they can be surfaced separately at assembly time.
            fails = [g for g in group if g["outcome"] == 0]
            rest = [g for g in group if g["outcome"] != 0]
            for sub, sys_prompt, valence in (
                (rest, self.EXTRACT_SYS, "fact"),
                (fails, self.FAILURE_SYS, "failure"),
            ):
                if not sub:
                    continue
                new_, reinf_, sup_ = self._distil(sub, sys_prompt, valence, now)
                new += new_
                reinforced += reinf_
                superseded += sup_
                if valence == "failure":
                    failures += new_

        self.db.executemany(
            "UPDATE episodes SET consolidated = 1 WHERE id = ?",
            [(r["id"],) for r in rows],
        )
        self.db.commit()
        return {
            "episodes": len(rows),
            "facts_new": new,
            "facts_reinforced": reinforced,
            "superseded": superseded,
            "failure_lessons": failures,
        }

    def _distil(
        self, group: list[sqlite3.Row], sys_prompt: str, valence: str, now: float
    ) -> tuple[int, int, int]:
        body = "\n".join(f"- {g['content']}" for g in group)
        try:
            props = _json(self.llm.complete(sys_prompt, body)).get("propositions", [])
        except Exception:
            props = []
        src = [int(g["id"]) for g in group]
        # A fact was true from when it was first observed, not from when it was
        # distilled — otherwise point-in-time retrieval is systematically wrong.
        valid_from = min(float(g["ts"]) for g in group)
        new = reinforced = superseded = 0
        for prop in props:
            result = self._upsert_fact(str(prop), src, now, valid_from, valence)
            new += result == "new"
            reinforced += result == "reinforced"
            superseded += result == "superseded"
        return new, reinforced, superseded

    def _upsert_fact(
        self,
        proposition: str,
        provenance: list[int],
        now: float,
        valid_from: float,
        valence: str = "fact",
    ) -> str:
        v = self._vec(proposition)
        # Only compare like with like. "Do X before Y" (failure) and "X was done
        # before Y" (fact) are near-identical in embedding space but are not the
        # same claim, and merging them destroys both.
        near = [
            (f"f{f.id}", f.proposition)
            for f in self._nearest_facts(v, k=5)
            if f.valence == valence
        ]
        if near:
            verdict, target = self._gate(proposition, near)
            tid = int(target[1:]) if target and target.startswith("f") else None
            if verdict == "redundant" and tid:
                row = self.db.execute(
                    "SELECT provenance FROM facts WHERE id = ?", (tid,)
                ).fetchone()
                prov = sorted(set(json.loads(row["provenance"] or "[]")) | set(provenance))
                # Support and provenance only, for the same reason as the encode
                # path: "reinforced" here means the distilled wording collided
                # with wording we already hold, not that new evidence arrived. A
                # single note restating one claim three ways yields three hits and
                # is still a sample size of one — and because these propositions
                # are extracted from episodes that already passed the write gate,
                # the collisions are correlated by construction. Widening
                # provenance is the useful part: it records which episodes the
                # belief now answers for, and explain() surfaces that.
                self.db.execute(
                    "UPDATE facts SET support = support + 1, provenance = ?"
                    " WHERE id = ?",
                    (json.dumps(prov), tid),
                )
                self.db.commit()
                return "reinforced"
            if verdict == "contradiction" and tid:
                new_id = self._insert_fact(
                    proposition, provenance, now, v, valid_from, valence
                )
                self._supersede(tid, new_id, valid_from, reason="superseded by newer evidence")
                return "superseded"
        self._insert_fact(proposition, provenance, now, v, valid_from, valence)
        return "new"

    def _insert_fact(
        self,
        proposition: str,
        provenance: list[int],
        now: float,
        v: np.ndarray,
        valid_from: float | None = None,
        valence: str = "fact",
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO facts (subject, proposition, confidence, importance, support,"
            " created_at, valid_from, valid_to, last_used, use_count, valence,"
            " n_success, n_total, utility, provenance, embedding)"
            " VALUES (?,?,?,?,?,?,?,NULL,?,0,?,0,0,0.5,?,?)",
            (
                _subject(proposition),
                proposition,
                0.6,
                0.5,
                1,
                now,                                        # created_at (record time)
                valid_from if valid_from is not None else now,  # valid_from (world time)
                now,                                        # last_used
                valence,
                json.dumps(provenance),
                _blob(v),
            ),
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def _supersede(self, fact_id: int, by: int | None, now: float, reason: str) -> None:
        # DIVERGE: brains overwrite. We close the validity interval and keep the
        # record, so "what did the agent believe last March" stays answerable and
        # every live fact keeps a provenance chain back to raw episodes.
        self.db.execute(
            "UPDATE facts SET valid_to = ?, superseded_by = ?, retired_reason = ?"
            " WHERE id = ? AND valid_to IS NULL",
            (now, by, reason, fact_id),
        )
        self.db.commit()

    # =======================================================================
    # RETRIEVE
    # =======================================================================

    def _nearest_facts(self, v: np.ndarray, k: int, at: float | None = None) -> list[Fact]:
        """Brute-force cosine scan over live facts.

        Scan only (id, embedding), then hydrate the k winners. Profiled at 100k
        facts, the dot product is ~5ms and everything else is marshalling:
        `SELECT *` cost 640ms and decoding one row at a time another 305ms, for a
        296MB peak on a query that returns three rows. Selecting two columns and
        decoding the blobs as a single buffer is ~2x faster and holds only the
        embedding matrix. The maths is untouched — see
        test_nearest_facts_matches_a_brute_force_reference.

        This is still O(n) per query and is meant to be: it is the honest
        zero-dependency default. Past roughly 20k live facts the latency starts
        to show up in the SessionStart hook; that is where pgvector or FAISS
        earns its dependency, and only this method changes.
        """
        if at is None:
            ids = self._rows("SELECT id, embedding FROM facts WHERE valid_to IS NULL")
        else:
            ids = self._rows(
                "SELECT id, embedding FROM facts WHERE valid_from <= ?"
                " AND (valid_to IS NULL OR valid_to > ?)",
                (at, at),
            )
        if not ids:
            return []
        # One allocation for the whole matrix rather than one per row.
        mat = np.frombuffer(
            b"".join(r["embedding"] for r in ids), dtype=np.float32
        ).reshape(len(ids), self.emb.dim)
        sims = mat @ v
        order = np.argsort(-sims)[:k]

        want = [int(ids[i]["id"]) for i in order]
        score = {int(ids[i]["id"]): float(sims[i]) for i in order}
        rows = self._rows(
            f"SELECT * FROM facts WHERE id IN ({','.join('?' * len(want))})", want
        )
        by_id = {int(r["id"]): r for r in rows}
        # Re-impose similarity order: SQL IN makes no ordering promise.
        return [_fact(by_id[i], score[i]) for i in want if i in by_id]

    def retrieve(
        self,
        query: str,
        *,
        k: int = 3,
        valence: str | None = None,
        at: float | None = None,
        now: float | None = None,
        mmr_lambda: float = 0.7,
    ) -> list[Fact]:
        """Multi-signal ranking, then MMR.

        Cosine relevance alone is the standard failure: it optimises similarity
        when the objective is usefulness, and it happily returns eight paraphrases
        of one fact. MMR is what stops the context window filling with echoes.
        `at` gives point-in-time retrieval over the validity intervals.

        k defaults to 3 because retrieval depth saturates fast. Ma et al. (2026)
        measured 74% at k=1, 82% at k=2, and a flat 82% at k=3 and k=5 — the
        fourth and fifth retrieved items bought nothing. Deeper retrieval spends
        attention budget it cannot repay. Raise k only with evidence.
        """
        now = now if now is not None else time.time()
        v = self._vec(query)
        pool = self._nearest_facts(v, k=max(k * 4, 25), at=at)
        if not pool:
            return []

        tau = self.w.tau_days * DAY
        if valence:
            pool = [f for f in pool if f.valence == valence]
            if not pool:
                return []
        for f in pool:
            rel = max(0.0, f.score)
            rec = math.exp(-max(0.0, now - f.last_used) / tau)
            f.score = (
                self.w.relevance * rel
                + self.w.recency * rec
                + self.w.importance * f.importance
                + self.w.usage * math.log1p(f.use_count) / 5.0
                # Utility is the outcome channel. Without it, ranking rewards
                # beliefs that merely look relevant over beliefs that have
                # actually worked — which is how a plausible wrong answer
                # outranks a proven right one.
                + self.w.utility * (f.utility - 0.5) * 2.0
            ) * f.confidence

        vecs = {f.id: self._vec(f.proposition) for f in pool}
        chosen: list[Fact] = []
        remaining = sorted(pool, key=lambda f: -f.score)
        while remaining and len(chosen) < k:
            if not chosen:
                chosen.append(remaining.pop(0))
                continue
            best, best_val = None, -1e9
            for cand in remaining:
                redundancy = max(float(vecs[cand.id] @ vecs[c.id]) for c in chosen)
                val = mmr_lambda * cand.score - (1 - mmr_lambda) * redundancy
                if val > best_val:
                    best, best_val = cand, val
            remaining.remove(best)  # type: ignore[arg-type]
            chosen.append(best)  # type: ignore[arg-type]

        for f in chosen:
            self.db.execute(
                "UPDATE facts SET use_count = use_count + 1, last_used = ? WHERE id = ?",
                (now, f.id),
            )
        self.db.commit()
        return chosen

    def recent_episodes(self, n: int = 5, session: str | None = None) -> list[Episode]:
        if session:
            rows = self._rows(
                "SELECT * FROM episodes WHERE archived = 0 AND session = ?"
                " ORDER BY ts DESC LIMIT ?",
                (session, n),
            )
        else:
            rows = self._rows(
                "SELECT * FROM episodes WHERE archived = 0 ORDER BY ts DESC LIMIT ?", (n,)
            )
        return [_episode(r) for r in rows][::-1]

    def skills_for(self, query: str, k: int = 3, min_rate: float = 0.5) -> list[sqlite3.Row]:
        rows = self._rows("SELECT * FROM skills WHERE attempts > 0")
        rows = [r for r in rows if r["successes"] / max(1, r["attempts"]) >= min_rate]
        if not rows:
            return []
        v = self._vec(query)
        mat = np.stack([_unblob(r["embedding"], self.emb.dim) for r in rows])
        return [rows[i] for i in np.argsort(-(mat @ v))[:k]]

    # =======================================================================
    # ASSEMBLE — fit memory into the attention budget
    # =======================================================================

    def context(
        self,
        goal: str,
        *,
        token_budget: int = 1200,
        session: str | None = None,
        n_facts: int = 5,
        n_failures: int = 3,
        n_episodes: int = 5,
    ) -> str:
        """Build the L0 working set.

        Two ordering decisions, both load-bearing:

        1. Failures come before facts. Ma et al. (2026) ablated their semantic
           memory and found removing failure reasons cost 8 points while removing
           success patterns cost 2 — knowing what not to do is worth roughly four
           times knowing what worked. Most memory systems store only successes.

        2. Within each block, a serial-position V: best at the head, second-best
           at the tail, weakest buried in the middle, because transformers attend
           unevenly across long inputs (Liu et al., 2023, "Lost in the Middle").
        """
        core = [r["content"] for r in self._rows("SELECT content FROM core ORDER BY id")]
        failures = self.retrieve(goal, k=n_failures, valence="failure")
        facts = self.retrieve(goal, k=n_facts, valence="fact")
        eps = self.recent_episodes(n_episodes, session=session)
        skills = self.skills_for(goal)

        parts: list[str] = []
        if core:
            parts.append("## Identity\n" + "\n".join(f"- {c}" for c in core))
        parts.append(f"## Current goal\n{goal}")

        budget = token_budget - _tokens("\n\n".join(parts))

        def _fit(items: list[Fact]) -> list[Fact]:
            nonlocal budget
            kept: list[Fact] = []
            for f in items:
                cost = _tokens(f.proposition) + 12
                if cost > budget:
                    break
                budget -= cost
                kept.append(f)
            return kept

        # Failures are fitted first so that under a tight budget they are the
        # last thing dropped, not the first.
        kept_fail = _fit(failures)
        kept_fact = _fit(facts)

        # Ids are carried into the block, matching how retrieve() renders them.
        # Without them the block is a dead end for the feedback loop: record_outcome
        # takes a fact id, so an agent acting on what it read here would have to
        # re-query just to recover an id it was already shown.
        if kept_fail:
            parts.append(
                "## What has gone wrong before\n"
                + "\n".join(
                    f"- [{f.id}] {f.proposition}  ({_util(f)})"
                    for f in _serial_position(kept_fail)
                )
            )
        if kept_fact:
            parts.append(
                "## What I know\n"
                + "\n".join(
                    f"- [{f.id}] {f.proposition}  (conf {f.confidence:.2f}, n={f.support})"
                    for f in _serial_position(kept_fact)
                )
            )
        # Everything below is budgeted too. A budget that governs only facts is
        # not a budget: recent events alone can double the block, and they are
        # the least distilled material in it. Priority order is deliberate —
        # failures, then facts, then proven procedures, then raw events last.
        kept_skills = []
        for r in skills:
            line = (
                f"- when {r['trigger']}: {r['procedure']} "
                f"({r['successes']}/{r['attempts']})"
            )
            if _tokens(line) > budget:
                break
            budget -= _tokens(line)
            kept_skills.append(line)
        if kept_skills:
            parts.append("## How I have done this before\n" + "\n".join(kept_skills))

        kept_eps = []
        for e in eps:
            line = f"- [{_ago(e.ts)}] {e.actor or '?'}: {e.content}{_outcome_tag(e)}"
            if _tokens(line) > budget:
                break
            budget -= _tokens(line)
            kept_eps.append(line)
        if kept_eps:
            parts.append("## Recent events\n" + "\n".join(kept_eps))

        # Neutralise once, on the assembled block, rather than at each render site:
        # a section added later inherits the protection instead of quietly reopening
        # the hole. This also covers the goal string, which arrives from the hook
        # payload and is no more trustworthy than anything in the store.
        return _neutralise("\n\n".join(parts))

    # =======================================================================
    # FORGET
    # =======================================================================

    def decay(
        self, *, now: float | None = None, floor: float = 0.12, archive_after_days: float = 90.0
    ) -> dict[str, int]:
        """Forgetting is a precision mechanism.

        At fixed k, retrieval precision falls as the store grows: irrelevant items
        that happen to be similar crowd out relevant ones. A store that never
        forgets gets monotonically worse at being read. Retire weakly-supported,
        long-unused facts; archive consolidated episodes rather than deleting them.
        """
        now = now if now is not None else time.time()
        tau = self.w.tau_days * DAY
        retired = 0
        for r in self._rows("SELECT * FROM facts WHERE valid_to IS NULL"):
            strength = (
                r["confidence"]
                * math.exp(-max(0.0, now - r["last_used"]) / tau)
                * (1.0 + math.log1p(r["support"]))
            )
            if strength < floor and r["support"] <= 1 and r["importance"] < 0.75:
                self._supersede(int(r["id"]), None, now, reason="decayed")
                retired += 1
        cur = self.db.execute(
            "UPDATE episodes SET archived = 1 WHERE consolidated = 1 AND archived = 0"
            " AND ts < ?",
            (now - archive_after_days * DAY,),
        )
        self.db.commit()
        return {"facts_retired": retired, "episodes_archived": cur.rowcount}

    # =======================================================================
    # Writes to the other stores
    # =======================================================================

    def pin(self, content: str) -> int:
        cur = self.db.execute("INSERT INTO core (content) VALUES (?)", (content,))
        self.db.commit()
        return int(cur.lastrowid or 0)

    def learn_skill(self, trigger: str, procedure: str) -> int:
        cur = self.db.execute(
            "INSERT INTO skills (trigger, procedure, last_used, embedding) VALUES (?,?,?,?)",
            (trigger, procedure, time.time(), _blob(self._vec(trigger + " " + procedure))),
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def record_skill_use(self, skill_id: int, success: bool) -> None:
        # DIVERGE from pure similarity retrieval: procedural memory is scored by
        # outcome, not by resemblance. A confident wrong procedure is worse than none.
        self.db.execute(
            "UPDATE skills SET attempts = attempts + 1, successes = successes + ?,"
            " last_used = ? WHERE id = ?",
            (1 if success else 0, time.time(), skill_id),
        )
        self.db.commit()

    def record_outcome(self, fact_id: int, success: bool, *, now: float | None = None) -> float:
        """Feed the result of acting on a belief back into its ranking.

        utility = 0.7 * confidence + 0.3 * usage,  usage = min(n_success/10, 1)

        Taken directly from Ma et al. (2026) §B.3. The 0.7/0.3 split is their
        choice, not a derived optimum — treat it as a starting point. The shape
        matters more than the constants: reliability dominates, but a rule that
        has only ever been right once should not outrank one right nine times.

        Confidence is Laplace-smoothed — (n_success+1)/(n_total+2), not the raw
        ratio. The raw ratio applies that "right once vs right nine times" rule to
        usage but breaks it for confidence: the first success is 1/1, which pins
        confidence at 1.00, overwrites the distilled prior, and makes a belief the
        store itself flagged as unverified outrank everything. Smoothing puts one
        success at 0.67 and nine at 0.91, so evidence has to accumulate. This
        matters most in advisory work, where "it worked" is frequently an
        unverifiable judgement call rather than an observed result.
        """
        now = now if now is not None else time.time()
        r = self.db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if not r:
            return 0.0
        n_success = int(r["n_success"] or 0) + (1 if success else 0)
        n_total = int(r["n_total"] or 0) + 1
        conf = (n_success + 1) / (n_total + 2)
        usage = min(n_success / 10.0, 1.0)
        utility = 0.7 * conf + 0.3 * usage
        self.db.execute(
            "UPDATE facts SET n_success = ?, n_total = ?, utility = ?,"
            " confidence = ?, last_used = ? WHERE id = ?",
            (n_success, n_total, utility, max(0.05, conf), now, fact_id),
        )
        self.db.commit()
        return utility

    def prune_guidelines(self, *, keep: int = 20, now: float | None = None) -> int:
        """Cap the number of outcome-scored guidelines.

        Ma et al. inject their top-20 guidelines into every prompt wholesale, with
        no retrieval. That is a growing system prompt: fine at 20, broken at 2000.
        Here the cap is a storage bound and retrieval still applies on top, so the
        two mechanisms compose instead of one substituting for the other.

        Proven rules (conf >= 0.8 and >= 5 successes) are protected regardless of
        rank, so a reliable rule can't be evicted by a burst of novel ones.
        """
        now = now if now is not None else time.time()
        rows = self._rows(
            "SELECT * FROM facts WHERE valid_to IS NULL AND n_total > 0"
            " ORDER BY utility DESC"
        )
        pruned = 0
        for r in rows[keep:]:
            conf = (r["n_success"] or 0) / max(1, r["n_total"] or 1)
            if conf >= 0.8 and (r["n_success"] or 0) >= 5:
                continue
            self._supersede(int(r["id"]), None, now, reason="pruned: low utility")
            pruned += 1
        return pruned

    def explain(self, fact_id: int) -> dict[str, Any]:
        """Provenance chain for a belief — which raw episodes support it."""
        r = self.db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if not r:
            return {}
        ids = json.loads(r["provenance"] or "[]")
        eps = (
            self._rows(
                f"SELECT id, ts, actor, content FROM episodes WHERE id IN "
                f"({','.join('?' * len(ids))})",
                ids,
            )
            if ids
            else []
        )
        return {
            "proposition": r["proposition"],
            "confidence": r["confidence"],
            "support": r["support"],
            "live": r["valid_to"] is None,
            "retired_reason": r["retired_reason"],
            "evidence": [dict(e) for e in eps],
        }

    def stats(self) -> dict[str, int]:
        q = lambda s: int(self.db.execute(s).fetchone()[0])  # noqa: E731
        return {
            "episodes": q("SELECT COUNT(*) FROM episodes"),
            "episodes_unconsolidated": q("SELECT COUNT(*) FROM episodes WHERE consolidated=0"),
            "facts_live": q("SELECT COUNT(*) FROM facts WHERE valid_to IS NULL"),
            "facts_retired": q("SELECT COUNT(*) FROM facts WHERE valid_to IS NOT NULL"),
            "failure_lessons": q(
                "SELECT COUNT(*) FROM facts WHERE valence='failure' AND valid_to IS NULL"
            ),
            "facts_outcome_scored": q("SELECT COUNT(*) FROM facts WHERE n_total > 0"),
            "skills": q("SELECT COUNT(*) FROM skills"),
        }


# --------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------


def _fact(r: sqlite3.Row, score: float = 0.0) -> Fact:
    return Fact(
        id=int(r["id"]),
        proposition=r["proposition"],
        subject=r["subject"],
        confidence=r["confidence"],
        importance=r["importance"],
        support=int(r["support"]),
        valid_from=r["valid_from"],
        valid_to=r["valid_to"],
        last_used=r["last_used"] or r["created_at"],
        use_count=int(r["use_count"]),
        valence=r["valence"] or "fact",
        n_success=int(r["n_success"] or 0),
        n_total=int(r["n_total"] or 0),
        utility=float(r["utility"] if r["utility"] is not None else 0.5),
        provenance=json.loads(r["provenance"] or "[]"),
        score=score,
    )


def _episode(r: sqlite3.Row) -> Episode:
    return Episode(
        id=int(r["id"]),
        ts=r["ts"],
        session=r["session"],
        actor=r["actor"],
        kind=r["kind"],
        content=r["content"],
        salience=r["salience"],
        outcome=None if r["outcome"] is None else bool(r["outcome"]),
    )


def _json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


STOP_CAPS = {"The", "This", "That", "They", "There", "When", "What", "Batch", "Splitting"}


def _entities(text: str) -> set[str]:
    return {
        w for w in re.findall(r"\b[A-Z][a-zA-Z0-9&'-]{2,}\b", text) if w not in STOP_CAPS
    }


def _subject(text: str) -> str | None:
    ents = sorted(_entities(text))
    return ents[0] if ents else None


def _util(f: Fact) -> str:
    if f.n_total:
        return f"held {f.n_success}/{f.n_total} times, utility {f.utility:.2f}"
    return f"unverified, n={f.support}"


def _outcome_tag(e: Episode) -> str:
    if e.outcome is None:
        return ""
    return "  [worked]" if e.outcome else "  [FAILED]"


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _ago(ts: float) -> str:
    d = max(0.0, time.time() - ts)
    if d < 3600:
        return f"{int(d // 60)}m ago"
    if d < DAY:
        return f"{int(d // 3600)}h ago"
    return f"{int(d // DAY)}d ago"


def _serial_position(items: list[Fact]) -> list[Fact]:
    """Best material at the head and tail, weakest in the middle."""
    head: list[Fact] = []
    tail: list[Fact] = []
    for i, x in enumerate(items):
        (head if i % 2 == 0 else tail).append(x)
    return head + tail[::-1]


def _agglomerate(vecs: np.ndarray, threshold: float) -> list[list[int]]:
    """Single-pass greedy clustering on cosine similarity."""
    n = len(vecs)
    unassigned = set(range(n))
    clusters: list[list[int]] = []
    sims = vecs @ vecs.T
    while unassigned:
        seed = unassigned.pop()
        group = [seed]
        for j in list(unassigned):
            if sims[seed, j] >= threshold:
                group.append(j)
                unassigned.discard(j)
        clusters.append(sorted(group))
    return clusters
