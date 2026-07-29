"""Sparse Personalized PageRank retrieval with supernode damping."""

import math
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from supernode_poc.graph import KG


def _nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def transition_matrix(
    kg: KG,
    beta: float = 0.0,
    kernel: str = "entropy",
    identity_mix: float | None = None,
) -> tuple[sp.csr_matrix, list[str]]:
    """Return a row-stochastic undirected walk matrix and its node order."""
    beta = _nonnegative_float(beta, "beta")
    if identity_mix is not None and (not math.isfinite(identity_mix) or not 0 <= identity_mix <= 1):
        raise ValueError("identity_mix must be in [0, 1]")
    nodes = kg.nodes()
    size = len(nodes)
    if kernel == "entropy":
        penalty = np.array([kg.leakiness(node) for node in nodes], dtype=float)
    elif kernel == "degree":
        penalty = np.array([kg.degree(node) for node in nodes], dtype=float)
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    if not size:
        return sp.csr_matrix((0, 0), dtype=float), nodes

    index = {node: i for i, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    log_values: list[float] = []
    records = kg.edge_records()
    walk_records = (
        [record for record in records if record[2] != "same_as"]
        if identity_mix is not None
        else records
    )
    for subject, object_, _, _, weight in walk_records:
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("edge weights must be finite and positive")
        i, j = index[subject], index[object_]
        rows.extend((i, j))
        cols.extend((j, i))
        log_weight = math.log(weight)
        log_values.extend(
            (
                log_weight - beta * math.log1p(penalty[j]),
                log_weight - beta * math.log1p(penalty[i]),
            )
        )

    if rows:
        # Per-row rescaling is algebraically cancelled by normalization and
        # prevents underflow for large beta values.
        row_max = np.full(size, -np.inf)
        np.maximum.at(row_max, rows, log_values)
        values = np.exp(np.asarray(log_values) - row_max[rows])
        adjacency = sp.csr_matrix((values, (rows, cols)), shape=(size, size))
    else:
        adjacency = sp.csr_matrix((size, size), dtype=float)

    row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    dangling = row_sum == 0
    if dangling.any():
        adjacency = adjacency + sp.diags(dangling.astype(float), format="csr")
        row_sum[dangling] = 1.0
    matrix = (sp.diags(1.0 / row_sum) @ adjacency).tocsr()
    if identity_mix is not None and identity_mix > 0:
        identity_rows: list[int] = []
        identity_cols: list[int] = []
        identity_values: list[float] = []
        for subject, object_, relation, _, weight in records:
            if relation != "same_as":
                continue
            i, j = index[subject], index[object_]
            identity_rows.extend((i, j))
            identity_cols.extend((j, i))
            identity_values.extend((weight, weight))
        identity = sp.csr_matrix(
            (identity_values, (identity_rows, identity_cols)), shape=(size, size)
        )
        identity_sum = np.asarray(identity.sum(axis=1)).ravel()
        has_identity = identity_sum > 0
        if has_identity.any():
            inverse = np.zeros(size, dtype=float)
            inverse[has_identity] = 1.0 / identity_sum[has_identity]
            identity = sp.diags(inverse) @ identity
            mix = identity_mix * has_identity.astype(float)
            matrix = sp.diags(1.0 - mix) @ matrix + sp.diags(mix) @ identity
    return matrix.tocsr(), nodes


def ppr(
    P: sp.spmatrix | np.ndarray,
    seed_vec: np.ndarray,
    alpha: float = 0.15,
    iters: int = 60,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute PPR until L1 convergence or the iteration limit."""
    alpha = float(alpha)
    tol = _nonnegative_float(tol, "tol")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not isinstance(iters, int) or isinstance(iters, bool) or iters <= 0:
        raise ValueError("iters must be a positive integer")

    matrix = sp.csr_matrix(P, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("P must be square")
    seed = np.asarray(seed_vec, dtype=float)
    if seed.ndim != 1 or seed.size != matrix.shape[0]:
        raise ValueError("seed_vec length must match P")
    if seed.size == 0:
        return seed.copy()
    if not np.all(np.isfinite(seed)) or np.any(seed < 0) or seed.sum() <= 0:
        raise ValueError("seed_vec must contain finite, non-negative values with positive mass")
    if matrix.data.size and (not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0)):
        raise ValueError("P must contain finite, non-negative values")

    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    dangling = row_sum == 0
    if np.any(~dangling & ~np.isclose(row_sum, 1.0, atol=1e-8)):
        raise ValueError("non-dangling rows of P must sum to one")

    seed = seed / seed.sum()
    mass = seed.copy()
    for _ in range(iters):
        propagated = np.asarray(matrix.T @ mass).ravel()
        if dangling.any():
            propagated += mass[dangling].sum() * seed
        updated = alpha * seed + (1.0 - alpha) * propagated
        if np.linalg.norm(updated - mass, ord=1) <= tol:
            mass = updated
            break
        mass = updated
    total = mass.sum()
    if total <= 0 or not math.isfinite(float(total)):
        raise RuntimeError("PPR produced invalid probability mass")
    return mass / total


def score_sources(
    pi: np.ndarray,
    nodes: Sequence[str],
    kg: KG,
    spread_normalize: bool = True,
    exclude_top_leaky_pct: float | None = None,
) -> dict[str, float]:
    """Aggregate node mass into sources without multiplying shared-node mass."""
    mass = np.asarray(pi, dtype=float)
    if mass.ndim != 1 or mass.size != len(nodes):
        raise ValueError("pi length must match nodes")
    if not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("pi must contain finite, non-negative values")

    excluded: set[str] = set()
    if exclude_top_leaky_pct is not None:
        pct = float(exclude_top_leaky_pct)
        if not math.isfinite(pct) or not 0 <= pct <= 1:
            raise ValueError("exclude_top_leaky_pct must be in [0, 1]")
        if pct and nodes:
            count = min(len(nodes), max(1, math.ceil(len(nodes) * pct)))
            ranked = sorted(nodes, key=lambda node: (-kg.leakiness(node), node))
            excluded = set(ranked[:count])

    scores: dict[str, float] = {}
    for value, node in zip(mass, nodes, strict=True):
        if node in excluded:
            continue
        sources = sorted(kg.node_sources.get(node, ()))
        if not sources:
            continue
        share = float(value) / (len(sources) if spread_normalize else 1)
        for source_id in sources:
            scores[source_id] = scores.get(source_id, 0.0) + share
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
    """Return source identifiers ranked by damped PPR mass."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    for value, name in ((k, "k"), (top_m_seeds, "top_m_seeds")):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    matrix, nodes = transition_matrix(kg, beta=beta, kernel=kernel)
    if not nodes:
        return []
    node_embeddings = np.asarray(embedder.embed(kg.labels(nodes)), dtype=float)
    query_embedding = np.asarray(embedder.embed([question]), dtype=float)
    if node_embeddings.ndim != 2 or node_embeddings.shape[0] != len(nodes):
        raise ValueError("embedder returned an invalid node embedding matrix")
    if query_embedding.shape != (1, node_embeddings.shape[1]):
        raise ValueError("embedder returned an invalid query embedding")
    if not np.all(np.isfinite(node_embeddings)) or not np.all(np.isfinite(query_embedding)):
        raise ValueError("embedder returned non-finite values")

    similarities = node_embeddings @ query_embedding[0]
    order = np.argsort(-similarities, kind="stable")[: min(top_m_seeds, len(nodes))]
    seed = np.zeros(len(nodes), dtype=float)
    seed[order] = np.maximum(similarities[order], 0.0)
    if seed.sum() == 0:
        seed[order[0]] = 1.0
    scores = score_sources(
        ppr(matrix, seed),
        nodes,
        kg,
        spread_normalize=spread_normalize,
        exclude_top_leaky_pct=exclude_top_leaky_pct,
    )
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [source_id for source_id, _ in ranked[:k]]
