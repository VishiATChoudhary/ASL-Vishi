"""Sparse Personalized PageRank with bounded soft-identity transitions."""

import math
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from identity_integrity.graph import KG


def _nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def transition_matrix(
    kg: KG,
    identity_mix: float = 0.0,
) -> tuple[sp.csr_matrix, list[str]]:
    """Return a row-stochastic graph walk with bounded identity flow.

    Ordinary graph edges define the base walk. At nodes with ``same_as``
    edges, ``identity_mix`` reserves that fraction of outgoing probability for
    identity transitions. The rest remains on the base graph walk.
    """
    identity_mix = float(identity_mix)
    if not math.isfinite(identity_mix) or not 0 <= identity_mix <= 1:
        raise ValueError("identity_mix must be finite and in [0, 1]")

    nodes = kg.nodes()
    size = len(nodes)
    if not size:
        return sp.csr_matrix((0, 0), dtype=float), nodes

    index = {node: position for position, node in enumerate(nodes)}
    graph_rows: list[int] = []
    graph_cols: list[int] = []
    graph_values: list[float] = []
    identity_rows: list[int] = []
    identity_cols: list[int] = []
    identity_values: list[float] = []

    for subject, object_, relation, _, weight in kg.edge_records():
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("edge weights must be finite and positive")
        subject_index = index[subject]
        object_index = index[object_]
        if relation == "same_as":
            identity_rows.extend((subject_index, object_index))
            identity_cols.extend((object_index, subject_index))
            identity_values.extend((weight, weight))
        else:
            graph_rows.extend((subject_index, object_index))
            graph_cols.extend((object_index, subject_index))
            graph_values.extend((weight, weight))

    graph = sp.csr_matrix(
        (graph_values, (graph_rows, graph_cols)),
        shape=(size, size),
        dtype=float,
    )
    graph = _row_normalize_with_self_loops(graph)

    if identity_mix == 0 or not identity_rows:
        return graph, nodes

    identity = sp.csr_matrix(
        (identity_values, (identity_rows, identity_cols)),
        shape=(size, size),
        dtype=float,
    )
    identity_sum = np.asarray(identity.sum(axis=1)).ravel()
    has_identity = identity_sum > 0
    inverse = np.zeros(size, dtype=float)
    inverse[has_identity] = 1.0 / identity_sum[has_identity]
    identity = sp.diags(inverse) @ identity

    reserved = identity_mix * has_identity.astype(float)
    matrix = sp.diags(1.0 - reserved) @ graph + sp.diags(reserved) @ identity
    return matrix.tocsr(), nodes


def _row_normalize_with_self_loops(matrix: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    dangling = row_sum == 0
    if dangling.any():
        matrix = matrix + sp.diags(dangling.astype(float), format="csr")
        row_sum[dangling] = 1.0
    return (sp.diags(1.0 / row_sum) @ matrix).tocsr()


def ppr(
    transition: sp.spmatrix | np.ndarray,
    seed_vec: np.ndarray,
    alpha: float = 0.15,
    iters: int = 60,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute Personalized PageRank until L1 convergence or the iteration limit."""
    alpha = float(alpha)
    tol = _nonnegative_float(tol, "tol")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not isinstance(iters, int) or isinstance(iters, bool) or iters <= 0:
        raise ValueError("iters must be a positive integer")

    matrix = sp.csr_matrix(transition, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be square")
    seed = np.asarray(seed_vec, dtype=float)
    if seed.ndim != 1 or seed.size != matrix.shape[0]:
        raise ValueError("seed_vec length must match transition")
    if seed.size == 0:
        return seed.copy()
    if not np.all(np.isfinite(seed)) or np.any(seed < 0) or seed.sum() <= 0:
        raise ValueError("seed_vec must contain finite, non-negative values with positive mass")
    if matrix.data.size and (not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0)):
        raise ValueError("transition must contain finite, non-negative values")

    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    dangling = row_sum == 0
    if np.any(~dangling & ~np.isclose(row_sum, 1.0, atol=1e-8)):
        raise ValueError("non-dangling rows of transition must sum to one")

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
    mass: np.ndarray,
    nodes: Sequence[str],
    kg: KG,
) -> dict[str, float]:
    """Aggregate node mass into sources without duplicating shared-node mass."""
    values = np.asarray(mass, dtype=float)
    if values.ndim != 1 or values.size != len(nodes):
        raise ValueError("mass length must match nodes")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("mass must contain finite, non-negative values")

    scores: dict[str, float] = {}
    for value, node in zip(values, nodes, strict=True):
        sources = sorted(kg.node_sources.get(node, ()))
        if not sources:
            continue
        share = float(value) / len(sources)
        for source_id in sources:
            scores[source_id] = scores.get(source_id, 0.0) + share
    return scores
