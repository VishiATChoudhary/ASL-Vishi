# Supernode PoC (Entropy-Weighted PPR Retrieval) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lab-demo PoC showing that LLM-built knowledge graphs develop supernodes that corrupt PPR retrieval, and that an entropy-weighted degree-damped transition kernel fixes it.

**Architecture:** A self-contained Python pipeline (no Graphiti/Neo4j dependency): LLM triple extraction (Claude API) -> in-memory NetworkX knowledge graph with leakiness metrics -> sparse-matrix Personalized PageRank retrieval with a tunable damping kernel -> three demo artifacts: (1) an interactive graph visualization plus a live failing-query comparison, (2) diagnostic plots proving query-independent hub retrieval, (3) a Recall@5 table on MuSiQue comparing vanilla PPR vs the entropy-weighted kernel.

**Tech Stack:** Python 3.11+, uv, anthropic SDK (model `claude-opus-5`, structured outputs via `messages.parse`), networkx, scipy, numpy, sentence-transformers (local embeddings), matplotlib, pyvis, HF `datasets`, pytest.

## Global Constraints

- Model ID for extraction is exactly `claude-opus-5`, passed as a `--model` CLI flag default so the user can override per run.
- Claude API calls use `client.messages.parse(...)` with a Pydantic `output_format` and `max_tokens=16000`; always check `parsed_output` for `None` and `stop_reason == "refusal"` before use.
- Never use `tiktoken` or any client-side token estimator.
- No em dashes in any generated text, code comments, or commit messages.
- All scripts are runnable via `uv run python scripts/<name>.py` from the repo root.
- Every artifact (plots, HTML, JSON caches) is written to `artifacts/` (gitignored except `.gitkeep`).
- Extraction results are cached to JSONL under `data/cache/` so repeated runs cost zero API calls.
- Package name: `supernode_poc`, src layout (`src/supernode_poc/`).

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/supernode_poc/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `.gitignore`
- Create: `artifacts/.gitkeep`, `data/cache/.gitkeep`

**Interfaces:**
- Produces: an installable package `supernode_poc` importable in tests; `uv run pytest` works.

- [ ] **Step 1: Initialize git and uv project**

Run (from `/Users/vishi/repos/Agentic Systems Labs`):
```bash
git init
uv init --name supernode-poc --package --python 3.11
```
Note: `uv init --package` creates `pyproject.toml` and `src/supernode_poc/` may need renaming; ensure the package dir is `src/supernode_poc/` with an `__init__.py` containing `__version__ = "0.1.0"`.

- [ ] **Step 2: Set dependencies in pyproject.toml**

Replace the generated `pyproject.toml` content with:
```toml
[project]
name = "supernode-poc"
version = "0.1.0"
description = "PoC: entropy-weighted degree normalization fixes PPR supernode leak in LLM-built KGs"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.60",
    "networkx>=3.2",
    "scipy>=1.12",
    "numpy>=1.26",
    "sentence-transformers>=3.0",
    "matplotlib>=3.8",
    "pyvis>=0.3.2",
    "datasets>=2.19",
    "requests>=2.31",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/supernode_poc"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
Run: `uv sync`

- [ ] **Step 3: Write .gitignore**

```gitignore
.venv/
__pycache__/
*.pyc
artifacts/*
!artifacts/.gitkeep
data/cache/*
!data/cache/.gitkeep
data/raw/
.pytest_cache/
uv.lock
```

- [ ] **Step 4: Write the smoke test**

`tests/test_smoke.py`:
```python
def test_package_imports():
    import supernode_poc

    assert supernode_poc.__version__ == "0.1.0"
```

- [ ] **Step 5: Run test, verify it passes**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold supernode-poc package with uv"
```

---

### Task 2: Knowledge graph with leakiness metrics

**Files:**
- Create: `src/supernode_poc/models.py`
- Create: `src/supernode_poc/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Produces: `Triple(BaseModel)` with fields `subject: str`, `relation: str`, `object: str`; `TripleList(BaseModel)` with `triples: list[Triple]`.
- Produces: `KG` class with methods `add_triples(triples: list[Triple], source_id: str) -> None`, `normalize(name: str) -> str` (static), `nodes() -> list[str]`, `degree(v: str) -> int`, `relation_entropy(v: str) -> float`, `leakiness(v: str) -> float`, `edges() -> list[tuple[str, str, str]]` (u, v, relation), `save(path)` / `KG.load(path)` (JSON), and attribute `node_sources: dict[str, set[str]]`.

- [ ] **Step 1: Write the models**

`src/supernode_poc/models.py`:
```python
from pydantic import BaseModel


class Triple(BaseModel):
    subject: str
    relation: str
    object: str


class TripleList(BaseModel):
    triples: list[Triple]
```

- [ ] **Step 2: Write the failing tests**

`tests/test_graph.py`:
```python
import math

from supernode_poc.graph import KG
from supernode_poc.models import Triple


def make_triples(pairs):
    return [Triple(subject=s, relation=r, object=o) for s, r, o in pairs]


def test_normalize_collapses_case_and_whitespace():
    assert KG.normalize("  Alice   Smith ") == "alice smith"


def test_add_triples_merges_normalized_nodes_and_tracks_sources():
    kg = KG()
    kg.add_triples(make_triples([("Alice", "works_at", "Acme")]), source_id="ep1")
    kg.add_triples(make_triples([("alice ", "lives_in", "Berlin")]), source_id="ep2")
    assert set(kg.nodes()) == {"alice", "acme", "berlin"}
    assert kg.node_sources["alice"] == {"ep1", "ep2"}
    assert kg.degree("alice") == 2


def test_relation_entropy_zero_for_homogeneous_hub():
    kg = KG()
    pairs = [("acme", "employs", f"person{i}") for i in range(10)]
    kg.add_triples(make_triples(pairs), source_id="s")
    assert kg.relation_entropy("acme") == 0.0
    assert kg.leakiness("acme") == 0.0


def test_relation_entropy_positive_for_heterogeneous_hub():
    kg = KG()
    pairs = [("meeting", f"rel{i}", f"thing{i}") for i in range(10)]
    kg.add_triples(make_triples(pairs), source_id="s")
    assert math.isclose(kg.relation_entropy("meeting"), math.log(10))
    assert kg.leakiness("meeting") > kg.leakiness("thing0")


def test_save_load_roundtrip(tmp_path):
    kg = KG()
    kg.add_triples(make_triples([("a", "r1", "b"), ("a", "r2", "c")]), source_id="s1")
    path = tmp_path / "kg.json"
    kg.save(path)
    kg2 = KG.load(path)
    assert set(kg2.nodes()) == set(kg.nodes())
    assert sorted(kg2.edges()) == sorted(kg.edges())
    assert kg2.node_sources == kg.node_sources
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'supernode_poc.graph'`

- [ ] **Step 4: Implement KG**

`src/supernode_poc/graph.py`:
```python
import json
import math
from collections import Counter
from pathlib import Path

import networkx as nx

from supernode_poc.models import Triple


class KG:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self.node_sources: dict[str, set[str]] = {}

    @staticmethod
    def normalize(name: str) -> str:
        return " ".join(name.lower().split())

    def add_triples(self, triples: list[Triple], source_id: str) -> None:
        for t in triples:
            s = self.normalize(t.subject)
            o = self.normalize(t.object)
            r = self.normalize(t.relation).replace(" ", "_")
            if not s or not o:
                continue
            self.g.add_edge(s, o, relation=r)
            self.node_sources.setdefault(s, set()).add(source_id)
            self.node_sources.setdefault(o, set()).add(source_id)

    def nodes(self) -> list[str]:
        return list(self.g.nodes())

    def edges(self) -> list[tuple[str, str, str]]:
        return [(u, v, d["relation"]) for u, v, d in self.g.edges(data=True)]

    def degree(self, v: str) -> int:
        return self.g.degree(v)

    def relation_entropy(self, v: str) -> float:
        rels = [d["relation"] for _, _, d in self.g.in_edges(v, data=True)]
        rels += [d["relation"] for _, _, d in self.g.out_edges(v, data=True)]
        n = len(rels)
        if n == 0:
            return 0.0
        counts = Counter(rels)
        return -sum((c / n) * math.log(c / n) for c in counts.values())

    def leakiness(self, v: str) -> float:
        return self.degree(v) * self.relation_entropy(v)

    def save(self, path: str | Path) -> None:
        payload = {
            "edges": self.edges(),
            "node_sources": {k: sorted(v) for k, v in self.node_sources.items()},
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "KG":
        payload = json.loads(Path(path).read_text())
        kg = cls()
        for u, v, r in payload["edges"]:
            kg.g.add_edge(u, v, relation=r)
        kg.node_sources = {k: set(v) for k, v in payload["node_sources"].items()}
        return kg
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_graph.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add src/supernode_poc/models.py src/supernode_poc/graph.py tests/test_graph.py
git commit -m "feat: KG with relation-entropy leakiness metric"
```

---

### Task 3: PPR retrieval kernel (vanilla + entropy-damped)

**Files:**
- Create: `src/supernode_poc/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `KG` from Task 2 (`nodes()`, `edges()`, `leakiness()`, `degree()`, `node_sources`).
- Produces: `transition_matrix(kg: KG, beta: float = 0.0, kernel: str = "entropy") -> tuple[scipy.sparse.csr_matrix, list[str]]` (row-stochastic P over undirected edges; damping factor `(1 + x(target)) ** -beta` where `x` is `leakiness` for `kernel="entropy"` or raw `degree` for `kernel="degree"`; the degree kernel is the ablation baseline for the claim that entropy weighting protects legitimate homogeneous hubs).
- Produces: `ppr(P, seed_vec: np.ndarray, alpha: float = 0.15, iters: int = 60) -> np.ndarray` (power iteration, returns mass vector summing to ~1).
- Produces: `score_sources(pi: np.ndarray, nodes: list[str], kg: KG, spread_normalize: bool = True, exclude_top_leaky_pct: float | None = None) -> dict[str, float]`. A node's mass is divided by the number of sources containing it before being added to each (otherwise a hub appearing in 20 chunks boosts all 20 and recreates the supernode problem at the scoring stage); `exclude_top_leaky_pct` optionally drops the top-percentile leakiness nodes from scoring entirely (the "transparent hub" variant).
- Produces: `retrieve(kg, question: str, embedder, beta: float = 0.0, kernel: str = "entropy", k: int = 5, top_m_seeds: int = 10, spread_normalize: bool = True, exclude_top_leaky_pct: float | None = None) -> list[str]` (top-k source ids; embedder defined in Task 4).

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieval.py`:
```python
import numpy as np

from supernode_poc.graph import KG
from supernode_poc.models import Triple
from supernode_poc.retrieval import ppr, score_sources, transition_matrix


def star_kg(hub: str, n: int, heterogeneous: bool) -> KG:
    """Star plus one partner per leaf.

    The partner edges matter: in a pure star each leaf has exactly one
    neighbor, so row renormalization cancels any damping and beta has no
    effect. With a second neighbor, damping the hub shifts each leaf's
    outgoing mass toward its partner.
    """
    kg = KG()
    triples = [
        Triple(
            subject=hub,
            relation=(f"rel{i}" if heterogeneous else "employs"),
            object=f"leaf{i}",
        )
        for i in range(n)
    ]
    triples += [
        Triple(subject=f"leaf{i}", relation="knows", object=f"partner{i}")
        for i in range(n)
    ]
    kg.add_triples(triples, source_id="s")
    return kg


def run_ppr_from_leaf(kg: KG, beta: float) -> dict[str, float]:
    P, nodes = transition_matrix(kg, beta=beta)
    seed = np.zeros(len(nodes))
    seed[nodes.index("leaf0")] = 1.0
    pi = ppr(P, seed)
    return dict(zip(nodes, pi))


def test_ppr_mass_sums_to_one():
    kg = star_kg("hub", 5, heterogeneous=True)
    mass = run_ppr_from_leaf(kg, beta=0.0)
    assert np.isclose(sum(mass.values()), 1.0)


def test_beta_zero_matches_homogeneous_and_heterogeneous():
    het = run_ppr_from_leaf(star_kg("hub", 10, True), beta=0.0)
    hom = run_ppr_from_leaf(star_kg("hub", 10, False), beta=0.0)
    assert np.isclose(het["hub"], hom["hub"])


def test_damping_reduces_heterogeneous_hub_mass_only():
    het0 = run_ppr_from_leaf(star_kg("hub", 10, True), beta=0.0)
    het1 = run_ppr_from_leaf(star_kg("hub", 10, True), beta=1.0)
    hom0 = run_ppr_from_leaf(star_kg("hub", 10, False), beta=0.0)
    hom1 = run_ppr_from_leaf(star_kg("hub", 10, False), beta=1.0)
    assert het1["hub"] < het0["hub"]
    assert np.isclose(hom1["hub"], hom0["hub"])


def test_degree_kernel_damps_homogeneous_hub_too():
    # entropy kernel protects H=0 hubs; degree kernel does not: the contrast
    # is the ablation that shows entropy weighting is doing real work
    hom0 = run_ppr_from_leaf(star_kg("hub", 10, False), beta=0.0)
    P, nodes = transition_matrix(star_kg("hub", 10, False), beta=1.0, kernel="degree")
    seed = np.zeros(len(nodes))
    seed[nodes.index("leaf0")] = 1.0
    homd = dict(zip(nodes, ppr(P, seed)))
    assert homd["hub"] < hom0["hub"]


def test_score_sources_sums_node_mass():
    kg = KG()
    kg.add_triples([Triple(subject="a", relation="r", object="b")], source_id="s1")
    kg.add_triples([Triple(subject="c", relation="r", object="d")], source_id="s2")
    P, nodes = transition_matrix(kg)
    seed = np.zeros(len(nodes))
    seed[nodes.index("a")] = 1.0
    pi = ppr(P, seed)
    scores = score_sources(pi, nodes, kg)
    assert scores["s1"] > scores["s2"]


def test_spread_normalization_divides_multi_source_node_mass():
    kg = KG()
    kg.add_triples([Triple(subject="hub", relation="r1", object="a")], source_id="s1")
    kg.add_triples([Triple(subject="hub", relation="r2", object="b")], source_id="s2")
    P, nodes = transition_matrix(kg)
    seed = np.zeros(len(nodes))
    seed[nodes.index("a")] = 1.0
    pi = ppr(P, seed)
    spread = score_sources(pi, nodes, kg, spread_normalize=True)
    raw = score_sources(pi, nodes, kg, spread_normalize=False)
    # hub sits in both sources; spreading halves its contribution to each
    assert spread["s1"] < raw["s1"]


def test_exclude_top_leaky_drops_hub_from_scoring():
    kg = KG()
    triples = [Triple(subject="hub", relation=f"rel{i}", object=f"leaf{i}") for i in range(10)]
    kg.add_triples(triples, source_id="s1")
    kg.add_triples([Triple(subject="c", relation="r", object="d")], source_id="s2")
    P, nodes = transition_matrix(kg)
    seed = np.zeros(len(nodes))
    seed[nodes.index("leaf0")] = 1.0
    pi = ppr(P, seed)
    with_hub = score_sources(pi, nodes, kg, spread_normalize=False)
    without_hub = score_sources(pi, nodes, kg, spread_normalize=False, exclude_top_leaky_pct=0.05)
    assert without_hub["s1"] < with_hub["s1"]
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'supernode_poc.retrieval'`

- [ ] **Step 3: Implement the kernel**

`src/supernode_poc/retrieval.py`:
```python
import numpy as np
import scipy.sparse as sp

from supernode_poc.graph import KG


def transition_matrix(kg: KG, beta: float = 0.0, kernel: str = "entropy"):
    nodes = sorted(kg.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    if kernel == "entropy":
        raw = [kg.leakiness(v) for v in nodes]
    elif kernel == "degree":
        raw = [float(kg.degree(v)) for v in nodes]
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    damp = np.array([(1.0 + x) ** (-beta) for x in raw])
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v, _ in kg.edges():
        i, j = idx[u], idx[v]
        # undirected walk; mass entering a leaky node is damped
        rows += [i, j]
        cols += [j, i]
        vals += [damp[j], damp[i]]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    rowsum = np.asarray(A.sum(axis=1)).ravel()
    rowsum[rowsum == 0] = 1.0
    P = sp.diags(1.0 / rowsum) @ A
    return P.tocsr(), nodes


def ppr(P, seed_vec: np.ndarray, alpha: float = 0.15, iters: int = 60) -> np.ndarray:
    s = seed_vec / max(seed_vec.sum(), 1e-12)
    pi = s.copy()
    for _ in range(iters):
        pi = alpha * s + (1 - alpha) * (P.T @ pi)
    return pi


def score_sources(
    pi: np.ndarray,
    nodes: list[str],
    kg: KG,
    spread_normalize: bool = True,
    exclude_top_leaky_pct: float | None = None,
) -> dict[str, float]:
    excluded: set[str] = set()
    if exclude_top_leaky_pct:
        n_top = max(1, int(len(nodes) * exclude_top_leaky_pct))
        excluded = set(sorted(nodes, key=kg.leakiness, reverse=True)[:n_top])
    scores: dict[str, float] = {}
    for i, v in enumerate(nodes):
        if v in excluded:
            continue
        srcs = kg.node_sources.get(v, ())
        if not srcs:
            continue
        # dividing by source count stops a hub in 20 chunks from boosting
        # all 20; without this the scoring stage re-creates the leak the
        # kernel just removed
        share = float(pi[i]) / (len(srcs) if spread_normalize else 1)
        for sid in srcs:
            scores[sid] = scores.get(sid, 0.0) + share
    return scores


def retrieve(
    kg: KG,
    question: str,
    embedder,
    beta: float = 0.0,
    kernel: str = "entropy",
    k: int = 5,
    top_m_seeds: int = 10,
    spread_normalize: bool = True,
    exclude_top_leaky_pct: float | None = None,
) -> list[str]:
    P, nodes = transition_matrix(kg, beta=beta, kernel=kernel)
    node_embs = embedder.embed(nodes)
    q = embedder.embed([question])[0]
    sims = node_embs @ q
    order = np.argsort(-sims)[:top_m_seeds]
    seed = np.zeros(len(nodes))
    for i in order:
        seed[i] = max(float(sims[i]), 0.0)
    if seed.sum() == 0:
        seed[order[0]] = 1.0
    pi = ppr(P, seed)
    scores = score_sources(
        pi, nodes, kg,
        spread_normalize=spread_normalize,
        exclude_top_leaky_pct=exclude_top_leaky_pct,
    )
    return [sid for sid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/supernode_poc/retrieval.py tests/test_retrieval.py
git commit -m "feat: PPR retrieval with entropy-weighted degree damping"
```

---

### Task 4: Embedder and LLM triple extraction

**Files:**
- Create: `src/supernode_poc/embeddings.py`
- Create: `src/supernode_poc/extraction.py`
- Test: `tests/test_extraction.py`

**Interfaces:**
- Produces: `Embedder` class with `embed(texts: list[str]) -> np.ndarray` (L2-normalized rows; model `sentence-transformers/all-MiniLM-L6-v2`).
- Produces: `SYSTEM_NEUTRAL` (default everywhere), `SYSTEM_SPECIFIC` (anti-generic ablation prompt), `PROMPTS: dict[str, str]` registry keyed `"neutral"`/`"specific"`.
- Produces: `extract_triples(client, text: str, model: str = "claude-opus-5", system: str = SYSTEM_NEUTRAL) -> list[Triple]` using `client.messages.parse` with `output_format=TripleList`.
- Produces: `extract_corpus(client, items: list[tuple[str, str]], cache_path, model, system) -> dict[str, list[Triple]]` where items are `(source_id, text)`; caches to JSONL and skips cached ids on rerun. Callers must use a distinct cache path per prompt variant (cached rows do not record which prompt produced them).

- [ ] **Step 1: Write the embedder**

`src/supernode_poc/embeddings.py`:
```python
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])
```

- [ ] **Step 2: Write the failing extraction tests (stub client, no API)**

`tests/test_extraction.py`:
```python
import json

from supernode_poc.extraction import extract_corpus, extract_triples
from supernode_poc.models import Triple, TripleList


class FakeResponse:
    def __init__(self, triples):
        self.parsed_output = TripleList(triples=triples)
        self.stop_reason = "end_turn"


class FakeMessages:
    def __init__(self, triples):
        self._triples = triples
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        return FakeResponse(self._triples)


class FakeClient:
    def __init__(self, triples):
        self.messages = FakeMessages(triples)


def test_extract_triples_returns_parsed_list():
    t = Triple(subject="Alice", relation="works_at", object="Acme")
    client = FakeClient([t])
    result = extract_triples(client, "Alice works at Acme.")
    assert result == [t]


def test_extract_corpus_caches_and_skips(tmp_path):
    t = Triple(subject="Bob", relation="likes", object="coffee")
    client = FakeClient([t])
    cache = tmp_path / "cache.jsonl"

    first = extract_corpus(client, [("s1", "Bob likes coffee.")], cache, model="claude-opus-5")
    assert client.messages.calls == 1
    assert first["s1"] == [t]
    assert len(cache.read_text().strip().splitlines()) == 1

    second = extract_corpus(client, [("s1", "Bob likes coffee.")], cache, model="claude-opus-5")
    assert client.messages.calls == 1  # served from cache, no new call
    assert second["s1"] == [t]
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'supernode_poc.extraction'`

- [ ] **Step 4: Implement extraction**

`src/supernode_poc/extraction.py`:
```python
import json
from pathlib import Path

from supernode_poc.models import Triple, TripleList

# The neutral prompt is the DEFAULT for all main experiments. The hypothesis
# under test is that ordinary LLM extraction creates supernodes; a prompt that
# discourages generic entities would suppress the phenomenon before we can
# measure it. SYSTEM_SPECIFIC exists only as an extraction-side-mitigation
# ablation (compare degree distributions and retrieval under both prompts).
SYSTEM_NEUTRAL = (
    "You extract knowledge-graph triples from text. "
    "Extract every distinct fact as one (subject, relation, object) triple. "
    "Use lowercase entity names and snake_case relation names."
)

SYSTEM_SPECIFIC = SYSTEM_NEUTRAL + (
    " Prefer specific entities over generic ones: 'alice' not 'the user', "
    "'acme quarterly review' not 'meeting'."
)

PROMPTS = {"neutral": SYSTEM_NEUTRAL, "specific": SYSTEM_SPECIFIC}


def extract_triples(
    client, text: str, model: str = "claude-opus-5", system: str = SYSTEM_NEUTRAL
) -> list[Triple]:
    resp = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": text}],
        output_format=TripleList,
    )
    if getattr(resp, "stop_reason", None) == "refusal" or resp.parsed_output is None:
        return []
    return resp.parsed_output.triples


def extract_corpus(client, items, cache_path, model: str = "claude-opus-5", system: str = SYSTEM_NEUTRAL):
    cache_path = Path(cache_path)
    cached: dict[str, list[Triple]] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cached[row["source_id"]] = [Triple(**t) for t in row["triples"]]

    results: dict[str, list[Triple]] = {}
    with cache_path.open("a") as fh:
        for source_id, text in items:
            if source_id in cached:
                results[source_id] = cached[source_id]
                continue
            triples = extract_triples(client, text, model=model, system=system)
            results[source_id] = triples
            fh.write(
                json.dumps(
                    {"source_id": source_id, "triples": [t.model_dump() for t in triples]}
                )
                + "\n"
            )
            fh.flush()
    return results
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/supernode_poc/embeddings.py src/supernode_poc/extraction.py tests/test_extraction.py
git commit -m "feat: Claude structured-output triple extraction with JSONL cache"
```

---

### Task 5: LoCoMo ingest, visualization, and failing-query demo (demo piece 1)

**Files:**
- Create: `scripts/ingest_locomo.py`
- Create: `scripts/demo_query.py`

**Interfaces:**
- Consumes: `extract_corpus`, `KG`, `Embedder`, `retrieve`.
- Produces: `artifacts/locomo_kg.json` (saved KG), `artifacts/locomo_graph.html` (pyvis viz), and a side-by-side beta=0 vs beta=1 retrieval printout.
- Produces: `data/cache/locomo_sources.json` mapping `source_id -> original text` (needed by demo_query and diagnostics).

- [ ] **Step 1: Write the ingest script**

`scripts/ingest_locomo.py`:
```python
"""Build a KG from one LoCoMo conversation.

Usage: uv run python scripts/ingest_locomo.py [--model claude-opus-5] [--max-chunks 40]
Requires ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
import argparse
import json
from pathlib import Path

import anthropic
import requests

from supernode_poc.extraction import extract_corpus
from supernode_poc.graph import KG

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
RAW = Path("data/raw/locomo10.json")
SOURCES = Path("data/cache/locomo_sources.json")


def download() -> dict:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        RAW.write_bytes(requests.get(LOCOMO_URL, timeout=60).content)
    return json.loads(RAW.read_text())


def conversation_chunks(sample: dict, max_chunks: int) -> list[tuple[str, str]]:
    """Flatten one conversation into (source_id, text) chunks of ~6 turns."""
    conv = sample["conversation"]
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and isinstance(conv[k], list)),
        key=lambda k: int(k.split("_")[1]),
    )
    chunks: list[tuple[str, str]] = []
    for sk in session_keys:
        turns = conv[sk]
        for i in range(0, len(turns), 6):
            window = turns[i : i + 6]
            text = "\n".join(f"{t['speaker']}: {t['text']}" for t in window)
            chunks.append((f"{sk}-{i}", text))
    return chunks[:max_chunks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--max-chunks", type=int, default=40)
    ap.add_argument("--prompt", choices=["neutral", "specific"], default="neutral")
    args = ap.parse_args()

    data = download()
    sample = data[0]
    chunks = conversation_chunks(sample, args.max_chunks)
    print(f"{len(chunks)} chunks to extract (prompt={args.prompt})")
    SOURCES.write_text(json.dumps(dict(chunks)))

    from supernode_poc.extraction import PROMPTS

    client = anthropic.Anthropic()
    triples_by_source = extract_corpus(
        client,
        chunks,
        cache_path=f"data/cache/locomo_triples_{args.prompt}.jsonl",
        model=args.model,
        system=PROMPTS[args.prompt],
    )

    kg = KG()
    for source_id, triples in triples_by_source.items():
        kg.add_triples(triples, source_id=source_id)
    Path("artifacts").mkdir(exist_ok=True)
    kg.save("artifacts/locomo_kg.json")

    ranked = sorted(kg.nodes(), key=kg.leakiness, reverse=True)[:10]
    print(f"\nGraph: {len(kg.nodes())} nodes, {len(kg.edges())} edges")
    print("Top-10 leakiness (candidate supernodes):")
    for v in ranked:
        print(f"  {v:30s} deg={kg.degree(v):4d} H={kg.relation_entropy(v):.2f} leak={kg.leakiness(v):.1f}")

    # pyvis visualization: size by degree, red = top 1 percent leakiness
    from pyvis.network import Network

    threshold = sorted((kg.leakiness(v) for v in kg.nodes()), reverse=True)
    cutoff = threshold[max(0, len(threshold) // 100 - 1)] if threshold else 0.0
    net = Network(height="900px", width="100%", directed=False)
    for v in kg.nodes():
        leaky = kg.leakiness(v) >= cutoff and kg.leakiness(v) > 0
        net.add_node(
            v,
            label=v,
            size=6 + 2 * kg.degree(v) ** 0.5,
            color="#d62728" if leaky else "#1f77b4",
            title=f"deg={kg.degree(v)} leak={kg.leakiness(v):.1f}",
        )
    for u, v, r in kg.edges():
        net.add_edge(u, v, title=r)
    net.write_html("artifacts/locomo_graph.html", open_browser=False)
    print("\nWrote artifacts/locomo_graph.html")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run the schema assumption before spending API budget**

Run: `uv run python -c "
import json, requests
d = requests.get('https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json', timeout=60).json()
s = d[0]
print(type(d), len(d))
print(list(s.keys())[:10])
conv = s['conversation']
print([k for k in conv.keys()][:8])
"`
Expected: a list of samples with a `conversation` dict containing `session_1`, `session_2`, ... keys of turn lists with `speaker` and `text` fields. If the schema differs, adjust `conversation_chunks` to match the actual keys before proceeding (this is the one permitted deviation in this task; keep the `(source_id, text)` output contract identical).

- [ ] **Step 3: Run the ingest (costs API budget, roughly 40 extraction calls)**

Run: `uv run python scripts/ingest_locomo.py`
Expected: prints chunk count, then top-10 leakiness table, writes `artifacts/locomo_kg.json` and `artifacts/locomo_graph.html`. Open the HTML and confirm the red hairball center is visible.

**Manual inspection (required, feeds the entropy-is-a-hypothesis check):** go through the top-10 leakiness table and label each node "junk hub" (generic noun, coreference collapse) or "legitimate rich entity" (a real person or org with genuinely diverse relations). Record the labels in a comment at the bottom of `scripts/ingest_locomo.py`. If most top nodes are legitimate entities, the entropy metric is misfiring on this corpus and the demo narrative must present leakiness as a candidate metric under test, not a detector.

- [ ] **Step 4: Write the demo query script**

`scripts/demo_query.py`:
```python
"""Side-by-side retrieval: vanilla PPR (beta=0) vs entropy-damped (beta=1).

Usage: uv run python scripts/demo_query.py "What does Caroline do for work?"
"""
import json
import sys
import textwrap

from supernode_poc.embeddings import Embedder
from supernode_poc.graph import KG
from supernode_poc.retrieval import retrieve


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What does the speaker do for work?"
    kg = KG.load("artifacts/locomo_kg.json")
    sources = json.loads(open("data/cache/locomo_sources.json").read())
    embedder = Embedder()

    for beta in (0.0, 1.0):
        print(f"\n=== beta={beta} ===")
        for rank, sid in enumerate(retrieve(kg, question, embedder, beta=beta, k=3), 1):
            snippet = textwrap.shorten(sources.get(sid, "<missing>"), 160)
            print(f"{rank}. [{sid}] {snippet}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the demo query and record the comparison**

Run: `uv run python scripts/demo_query.py "What does Caroline do for work?"` (substitute a speaker name that exists in the ingested conversation, visible in the leakiness table).
Expected: both blocks print 3 sources. Inspect: the beta=0 block should include at least one off-topic chunk pulled through a hub; the beta=1 block should be more on-topic. If both look identical, try 2 or 3 other questions and keep the most contrastive one as the demo question, noting it in the script's docstring default.

- [ ] **Step 6: Commit**

```bash
git add scripts/ingest_locomo.py scripts/demo_query.py
git commit -m "feat: LoCoMo ingest, supernode viz, and beta comparison demo"
```

---

### Task 6: Diagnostics (demo piece 2)

**Files:**
- Create: `src/supernode_poc/diagnostics.py`
- Create: `scripts/run_diagnostics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `KG`, `transition_matrix`, `ppr`, `Embedder`.
- Produces: `degree_distribution(kg) -> tuple[np.ndarray, np.ndarray]` (degree values, counts, both sorted ascending by degree).
- Produces: `mass_through_top_leaky(kg, pi: np.ndarray, nodes: list[str], pct: float = 0.01) -> float` (fraction of PPR mass held by top-pct leakiness nodes).
- Produces: `retrieval_frequency(kg, questions: list[str], embedder, beta: float, k: int = 5) -> Counter` (how often each node appears among top-k mass nodes across queries).
- Produces: `random_seed_frequency(kg, n_queries: int, k: int = 5, rng_seed: int = 0) -> Counter` (same statistic but with uniformly random seed nodes instead of question embeddings; this is the clean query-independence control because it bypasses the embedder entirely, so embedder word-order insensitivity cannot confound it).
- Produces: `artifacts/degree_distribution.png` and `artifacts/bias_curve.png`.

- [ ] **Step 1: Write the failing tests**

`tests/test_diagnostics.py`:
```python
import numpy as np

from supernode_poc.diagnostics import degree_distribution, mass_through_top_leaky
from supernode_poc.graph import KG
from supernode_poc.models import Triple
from supernode_poc.retrieval import ppr, transition_matrix


def build_hub_graph() -> KG:
    kg = KG()
    triples = [Triple(subject="hub", relation=f"rel{i}", object=f"leaf{i}") for i in range(20)]
    triples += [Triple(subject="a", relation="knows", object="b")]
    kg.add_triples(triples, source_id="s")
    return kg


def test_degree_distribution_shape():
    kg = build_hub_graph()
    degrees, counts = degree_distribution(kg)
    assert degrees.max() == 20  # the hub
    assert counts.sum() == len(kg.nodes())


def test_mass_through_top_leaky_flags_hub():
    kg = build_hub_graph()
    P, nodes = transition_matrix(kg)
    seed = np.zeros(len(nodes))
    seed[nodes.index("leaf0")] = 1.0
    pi = ppr(P, seed)
    frac = mass_through_top_leaky(kg, pi, nodes, pct=0.05)
    assert frac > 0.2  # hub soaks a large share of the walk mass
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'supernode_poc.diagnostics'`

- [ ] **Step 3: Implement diagnostics**

`src/supernode_poc/diagnostics.py`:
```python
from collections import Counter

import numpy as np

from supernode_poc.graph import KG
from supernode_poc.retrieval import ppr, transition_matrix


def degree_distribution(kg: KG):
    degs = np.array([kg.degree(v) for v in kg.nodes()])
    values, counts = np.unique(degs, return_counts=True)
    return values, counts


def mass_through_top_leaky(kg: KG, pi: np.ndarray, nodes: list[str], pct: float = 0.01) -> float:
    n_top = max(1, int(len(nodes) * pct))
    top = set(sorted(nodes, key=kg.leakiness, reverse=True)[:n_top])
    return float(sum(pi[i] for i, v in enumerate(nodes) if v in top))


def retrieval_frequency(kg: KG, questions: list[str], embedder, beta: float, k: int = 5) -> Counter:
    P, nodes = transition_matrix(kg, beta=beta)
    node_embs = embedder.embed(nodes)
    freq: Counter = Counter()
    for q in questions:
        qv = embedder.embed([q])[0]
        sims = node_embs @ qv
        order = np.argsort(-sims)[:10]
        seed = np.zeros(len(nodes))
        for i in order:
            seed[i] = max(float(sims[i]), 0.0)
        if seed.sum() == 0:
            continue
        pi = ppr(P, seed)
        for i in np.argsort(-pi)[:k]:
            freq[nodes[i]] += 1
    return freq


def random_seed_frequency(kg: KG, n_queries: int, k: int = 5, rng_seed: int = 0) -> Counter:
    """Query-independence control that bypasses the embedder entirely."""
    P, nodes = transition_matrix(kg)
    rng = np.random.default_rng(rng_seed)
    freq: Counter = Counter()
    for _ in range(n_queries):
        picks = rng.choice(len(nodes), size=min(10, len(nodes)), replace=False)
        seed = np.zeros(len(nodes))
        seed[picks] = 1.0
        pi = ppr(P, seed)
        for i in np.argsort(-pi)[:k]:
            freq[nodes[i]] += 1
    return freq
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the diagnostics script**

`scripts/run_diagnostics.py`:
```python
"""Produce the two motivation plots from the LoCoMo KG.

Usage: uv run python scripts/run_diagnostics.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from supernode_poc.diagnostics import (
    degree_distribution,
    random_seed_frequency,
    retrieval_frequency,
)
from supernode_poc.embeddings import Embedder
from supernode_poc.graph import KG

REAL_QUESTIONS = [
    "What does the speaker do for work?",
    "Where does the speaker live?",
    "What hobbies were discussed?",
    "What health issues came up?",
    "What travel plans were mentioned?",
    "Who are the speaker's family members?",
    "What food does the speaker like?",
    "What happened at the last event they attended?",
]

# Genuinely unrelated topics, not shuffled words: a shuffled-word control
# shares vocabulary with the real questions, and a near-bag-of-words embedder
# would map it to almost the same seeds, proving nothing about the graph.
OFF_DOMAIN_QUESTIONS = [
    "How do volcanoes form?",
    "What is the boiling point of nitrogen?",
    "Explain the rules of chess castling.",
    "Which planet has the strongest winds?",
    "How is porcelain manufactured?",
    "What causes tides in the ocean?",
    "Describe photosynthesis in algae.",
    "What year was the printing press invented?",
]


def main() -> None:
    kg = KG.load("artifacts/locomo_kg.json")
    embedder = Embedder()

    # Plot 1: log-log degree distribution
    values, counts = degree_distribution(kg)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(values, counts)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree")
    ax.set_ylabel("node count")
    ax.set_title("LLM-built KG degree distribution (heavy tail = supernodes)")
    fig.tight_layout()
    fig.savefig("artifacts/degree_distribution.png", dpi=150)

    # Plot 2: bias curve with two controls.
    # Control A: off-domain questions (different topic, same pipeline).
    # Control B: uniformly random seed nodes (bypasses the embedder entirely,
    # so embedder insensitivity cannot explain agreement).
    freq_real = retrieval_frequency(kg, REAL_QUESTIONS, embedder, beta=0.0)
    freq_off = retrieval_frequency(kg, OFF_DOMAIN_QUESTIONS, embedder, beta=0.0)
    freq_rand = random_seed_frequency(kg, n_queries=len(REAL_QUESTIONS))

    common = sorted(set(freq_real) | set(freq_off) | set(freq_rand), key=lambda v: -kg.degree(v))[:30]
    x = np.array([freq_real.get(v, 0) for v in common])
    y_off = np.array([freq_off.get(v, 0) for v in common])
    y_rand = np.array([freq_rand.get(v, 0) for v in common])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y_off, label="off-domain questions", marker="o")
    ax.scatter(x, y_rand, label="random graph seeds", marker="x")
    lim = max(x.max(), y_off.max(), y_rand.max()) + 1
    ax.plot([0, lim], [0, lim], linestyle="--", color="gray")
    ax.set_xlabel("retrieval count, real questions")
    ax.set_ylabel("retrieval count, control")
    ax.set_title("Hubs are retrieved regardless of the query (points on diagonal)")
    ax.legend()
    for v in common[:5]:
        ax.annotate(v, (freq_real.get(v, 0), freq_rand.get(v, 0)), fontsize=7)
    fig.tight_layout()
    fig.savefig("artifacts/bias_curve.png", dpi=150)

    from scipy.stats import spearmanr

    rho_off, p_off = spearmanr(x, y_off)
    rho_rand, p_rand = spearmanr(x, y_rand)
    print(f"Spearman(real, off-domain):   rho={rho_off:.2f} p={p_off:.3f}")
    print(f"Spearman(real, random-seed):  rho={rho_rand:.2f} p={p_rand:.3f}")
    print("High rho on BOTH controls, especially random-seed, means hub retrieval")
    print("is query-independent at the graph level: the smoking gun.")
    print("Wrote artifacts/degree_distribution.png and artifacts/bias_curve.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the script**

Run: `uv run python scripts/run_diagnostics.py`
Expected: both PNGs written; two Spearman rhos printed. On a hub-dominated graph expect the random-seed rho clearly above 0.5 (that one is embedder-proof). If the off-domain rho is high but the random-seed rho is low, the observed bias lives in the embedder, not the graph, and the demo narrative must say so. Eyeball both plots.

- [ ] **Step 7: Commit**

```bash
git add src/supernode_poc/diagnostics.py scripts/run_diagnostics.py tests/test_diagnostics.py
git commit -m "feat: degree-distribution and query-independence diagnostics"
```

---

### Task 7: MuSiQue evaluation (demo piece 3)

**Files:**
- Create: `scripts/build_musique.py`
- Create: `scripts/run_eval.py`

**Interfaces:**
- Consumes: `extract_corpus`, `KG`, `Embedder`, `retrieve`.
- Produces: `artifacts/musique_kg.json`, `data/cache/musique_questions.json` (list of `{id, question, supporting: [canonical_ids], split: "dev"|"test"}`), and a printed eval report: dev-set sweep table (beta x kernel), then the dev-selected configuration evaluated once on the held-out test split with a bootstrap confidence interval next to the beta=0 baseline.
- Note: this is an **open-corpus variant** of MuSiQue, not the standard 20-candidate-per-question ranking task. Paragraphs are deduplicated by normalized text into canonical ids so the same passage appearing under several questions cannot cause false negatives. State the variant explicitly whenever the numbers are shown.

- [ ] **Step 1: Write the corpus builder**

`scripts/build_musique.py`:
```python
"""Build a KG over a MuSiQue dev subset (open-corpus variant).

Paragraphs from all sampled questions are merged into ONE corpus and
deduplicated by normalized text. This is deliberately NOT the standard
20-candidate-per-question MuSiQue ranking task; it is an open-corpus
experiment where cross-question generic entities can form supernodes,
which is exactly the phenomenon under study. Label all numbers accordingly.

Usage: uv run python scripts/build_musique.py [--n-questions 60] [--model claude-opus-5]
Requires ANTHROPIC_API_KEY. Roughly n_questions * 20 short extraction calls on
first run; cached afterwards. Start with a small n to gauge cost.
"""
import argparse
import hashlib
import json
from pathlib import Path

import anthropic
from datasets import load_dataset

from supernode_poc.extraction import PROMPTS, extract_corpus
from supernode_poc.graph import KG


def canonical_id(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return "p" + hashlib.sha1(normalized.encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=60)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--prompt", choices=["neutral", "specific"], default="neutral")
    args = ap.parse_args()

    # If this dataset id fails, fall back to downloading the official MuSiQue
    # release from github.com/StonyBrookNLP/musique and loading the dev jsonl.
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    rows = list(ds.select(range(args.n_questions)))

    items: list[tuple[str, str]] = []
    questions = []
    seen: set[str] = set()
    for qi, row in enumerate(rows):
        supporting = []
        for p in row["paragraphs"]:
            text = f"{p['title']}. {p['paragraph_text']}"
            cid = canonical_id(text)
            if cid not in seen:
                seen.add(cid)
                items.append((cid, text))
            if p["is_supporting"]:
                supporting.append(cid)
        questions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "supporting": sorted(set(supporting)),
                # deterministic alternating split so dev and test cover the
                # same difficulty range; beta is selected on dev only
                "split": "dev" if qi % 2 == 0 else "test",
            }
        )

    print(f"{len(items)} unique paragraphs, {len(questions)} questions")
    Path("data/cache/musique_questions.json").write_text(json.dumps(questions))

    client = anthropic.Anthropic()
    triples_by_source = extract_corpus(
        client,
        items,
        cache_path=f"data/cache/musique_triples_{args.prompt}.jsonl",
        model=args.model,
        system=PROMPTS[args.prompt],
    )

    kg = KG()
    for cid, triples in triples_by_source.items():
        kg.add_triples(triples, source_id=cid)
    kg.save("artifacts/musique_kg.json")
    print(f"Graph: {len(kg.nodes())} nodes, {len(kg.edges())} edges")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run the dataset schema before extraction**

Run: `uv run python -c "
from datasets import load_dataset
ds = load_dataset('dgslibisey/MuSiQue', split='validation')
r = ds[0]
print(r.keys())
print(type(r['paragraphs']), r['paragraphs'][0].keys() if isinstance(r['paragraphs'], list) else r['paragraphs'])
"`
Expected: keys include `id`, `question`, `paragraphs` with `idx`, `title`, `paragraph_text`, `is_supporting`. If `paragraphs` is a dict of parallel lists (a common HF layout), adapt the loop in `build_musique.py` to zip those lists; keep the output contracts (`items`, `musique_questions.json`) identical. If the dataset id 404s, download `musique_ans_v1.0_dev.jsonl` from the StonyBrookNLP/musique release into `data/raw/` and load it with `json.loads` per line.

- [ ] **Step 3: Run the build (costs API budget; start small)**

Run: `uv run python scripts/build_musique.py --n-questions 20`
Expected: paragraph/question counts printed, KG saved. If cost and time look acceptable, rerun with `--n-questions 60` (cache makes the first 20 free).

- [ ] **Step 4: Write the eval script**

`scripts/run_eval.py`:
```python
"""Recall@5 on the MuSiQue open-corpus subset with a dev/test protocol.

Sweep (beta, kernel) on the dev split only, select the best dev
configuration, then evaluate it exactly once on the held-out test split and
report a bootstrap confidence interval next to the beta=0 baseline. This
protocol exists because "try many betas and report the winner" finds a
positive result by chance.

Usage: uv run python scripts/run_eval.py [--betas 0 0.5 1.0 2.0] [--k 5]
"""
import argparse
import json

import numpy as np

from supernode_poc.embeddings import Embedder
from supernode_poc.graph import KG
from supernode_poc.retrieval import retrieve


def recall_at_k(retrieved: list[str], supporting: list[str]) -> float:
    if not supporting:
        return 0.0
    return len(set(retrieved) & set(supporting)) / len(supporting)


def eval_config(kg, questions, embedder, beta, kernel, k) -> list[float]:
    return [
        recall_at_k(
            retrieve(kg, q["question"], embedder, beta=beta, kernel=kernel, k=k),
            q["supporting"],
        )
        for q in questions
    ]


def bootstrap_ci(scores: list[float], n: int = 2000, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    arr = np.array(scores)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    kg = KG.load("artifacts/musique_kg.json")
    questions = json.loads(open("data/cache/musique_questions.json").read())
    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    embedder = Embedder()

    print(f"dev={len(dev)} test={len(test)} k={args.k} (open-corpus MuSiQue variant)")
    print("\nDev sweep (selection only, do not report these as results):")
    print(f"{'kernel':>8} {'beta':>6} | {'dev Recall@' + str(args.k):>14}")
    print("-" * 34)
    best = (None, None, -1.0)
    for kernel in ("entropy", "degree"):
        for beta in args.betas:
            scores = eval_config(kg, dev, embedder, beta, kernel, args.k)
            mean = sum(scores) / len(scores)
            print(f"{kernel:>8} {beta:>6.1f} | {mean:>14.3f}")
            if beta > 0 and mean > best[2]:
                best = (kernel, beta, mean)

    kernel, beta, _ = best
    print(f"\nSelected on dev: kernel={kernel} beta={beta}")
    print("\nHeld-out test (report these):")
    base_scores = eval_config(kg, test, embedder, 0.0, "entropy", args.k)
    sel_scores = eval_config(kg, test, embedder, beta, kernel, args.k)
    b_lo, b_hi = bootstrap_ci(base_scores)
    s_lo, s_hi = bootstrap_ci(sel_scores)
    print(f"  beta=0 baseline:        {np.mean(base_scores):.3f}  CI95 [{b_lo:.3f}, {b_hi:.3f}]")
    print(f"  {kernel} beta={beta}:   {np.mean(sel_scores):.3f}  CI95 [{s_lo:.3f}, {s_hi:.3f}]")
    wins = sum(s > b for s, b in zip(sel_scores, base_scores))
    losses = sum(s < b for s, b in zip(sel_scores, base_scores))
    print(f"  per-question win/loss/tie: {wins}/{losses}/{len(test) - wins - losses}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the eval**

Run: `uv run python scripts/run_eval.py`
Expected: dev sweep table, then held-out test comparison with confidence intervals and a per-question win/loss count. Success criterion for the PoC: the dev-selected configuration beats the beta=0 baseline on the held-out test split, and the win is visible in the per-question counts, not just the mean. Overlapping confidence intervals with a positive mean gap = "suggestive, needs more questions", and the demo must say exactly that. A flat or negative result is also a reportable finding; record it either way. Widening the beta grid is allowed, but only ever on the dev split.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_musique.py scripts/run_eval.py
git commit -m "feat: MuSiQue Recall@5 eval with beta sweep"
```

---

### Task 8: Demo README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: everything above; documents the three demo pieces in presentation order.

- [ ] **Step 1: Write the README**

`README.md`:
```markdown
# Supernode PoC: entropy-weighted PPR retrieval for LLM-built knowledge graphs

LLM extraction creates generic hub nodes ("supernodes") that soak up
query-independent PageRank mass. This PoC demonstrates the failure and tests
a fix: damp walk mass entering nodes by
`(1 + degree * relation_entropy) ** -beta`, with hub mass additionally
spread across (or excluded from) the chunks that contain the hub at scoring
time.

Scope of the claim: damping reroutes walk mass when alternative paths exist.
It cannot remove a false connection whose only route is the hub itself
(structural fixes like facet splitting address that; see the proposal).
Entropy-weighted leakiness is itself a hypothesis under test here, not a
known-good junk detector: the eval compares it against plain degree damping,
and the ingest step includes manual inspection of top-leakiness nodes.

Companion research proposal: `research-proposal-supernodes.md`.

## Setup

    uv sync
    export ANTHROPIC_API_KEY=...   # or `ant auth login`

## Demo (presentation order)

1. **The failure, live**

       uv run python scripts/ingest_locomo.py
       open artifacts/locomo_graph.html
       uv run python scripts/demo_query.py "What does Caroline do for work?"

   The graph view shows the hairball: red nodes are top-1% leakiness.
   The query comparison shows beta=0 retrieving off-topic chunks through hubs
   while beta=1 stays on topic.

2. **The evidence, quantified**

       uv run python scripts/run_diagnostics.py

   Produces `artifacts/degree_distribution.png` (heavy tail) and
   `artifacts/bias_curve.png` (hub retrieval frequency matches two controls:
   off-domain questions and embedder-free random graph seeds; agreement with
   the random-seed control is the graph-level query-independence proof).

3. **The fix, measured**

       uv run python scripts/build_musique.py --n-questions 60
       uv run python scripts/run_eval.py

   Open-corpus MuSiQue variant (deduplicated merged paragraphs, NOT the
   standard per-question candidate ranking task). Beta and kernel are
   selected on a dev split; the reported number is the held-out test split
   with bootstrap confidence intervals and per-question win/loss counts.

## Ablations

- Kernel: `entropy` vs `degree` damping (does entropy weighting protect
  legitimate homogeneous hubs?). Built into `run_eval.py`.
- Extraction prompt: rerun ingest/build with `--prompt specific`
  (anti-generic instructions) vs the default `--prompt neutral`, and compare
  degree distributions. Tests whether prompting alone suppresses supernodes.

## Tests

    uv run pytest
```

- [ ] **Step 2: Run the full test suite one final time**

Run: `uv run pytest -v`
Expected: all tests pass (smoke + graph + retrieval + extraction + diagnostics).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: demo walkthrough README"
```
