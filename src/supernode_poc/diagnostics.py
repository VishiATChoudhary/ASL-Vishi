"""Diagnostics for graph structure and query-independent retrieval bias."""

from collections import Counter

import numpy as np

from supernode_poc.graph import KG
from supernode_poc.retrieval import ppr, transition_matrix


def degree_distribution(kg: KG) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique degree values and their node counts."""
    degrees = np.asarray([kg.degree(node) for node in kg.nodes()], dtype=int)
    if degrees.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.unique(degrees, return_counts=True)


def mass_through_top_leaky(
    kg: KG,
    pi: np.ndarray,
    nodes: list[str],
    pct: float = 0.01,
) -> float:
    """Return PPR mass held by the highest-leakiness node fraction."""
    if not 0 < pct <= 1:
        raise ValueError("pct must be in (0, 1]")
    if len(pi) != len(nodes):
        raise ValueError("pi and nodes must have the same length")
    if not nodes:
        return 0.0
    n_top = max(1, int(np.ceil(len(nodes) * pct)))
    top = set(sorted(nodes, key=lambda node: (-kg.leakiness(node), node))[:n_top])
    return float(sum(pi[index] for index, node in enumerate(nodes) if node in top))


def retrieval_frequency(
    kg: KG,
    questions: list[str],
    embedder,
    beta: float,
    k: int = 5,
) -> Counter[str]:
    """Count nodes appearing in the highest-PPR set for semantic queries."""
    if k < 1:
        raise ValueError("k must be positive")
    matrix, nodes = transition_matrix(kg, beta=beta)
    if not nodes:
        return Counter()
    node_embeddings = embedder.embed(kg.labels(nodes))
    frequency: Counter[str] = Counter()
    for question in questions:
        similarities = node_embeddings @ embedder.embed([question])[0]
        order = np.argsort(-similarities, kind="stable")[: min(10, len(nodes))]
        seed = np.maximum(similarities[order], 0.0)
        if not np.any(seed):
            seed = np.ones_like(seed)
        seed_vector = np.zeros(len(nodes), dtype=float)
        seed_vector[order] = seed
        mass = ppr(matrix, seed_vector)
        frequency.update(nodes[index] for index in np.argsort(-mass, kind="stable")[:k])
    return frequency


def random_seed_frequency(
    kg: KG,
    n_queries: int,
    k: int = 5,
    rng_seed: int = 0,
) -> Counter[str]:
    """Count high-mass nodes from uniformly sampled graph seeds."""
    if n_queries < 0:
        raise ValueError("n_queries must be nonnegative")
    if k < 1:
        raise ValueError("k must be positive")
    matrix, nodes = transition_matrix(kg)
    if not nodes:
        return Counter()
    rng = np.random.default_rng(rng_seed)
    frequency: Counter[str] = Counter()
    for _ in range(n_queries):
        picks = rng.choice(len(nodes), size=min(10, len(nodes)), replace=False)
        seed = np.zeros(len(nodes), dtype=float)
        seed[picks] = 1.0
        mass = ppr(matrix, seed)
        frequency.update(nodes[index] for index in np.argsort(-mass, kind="stable")[:k])
    return frequency
