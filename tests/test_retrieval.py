import numpy as np
import pytest

from identity_integrity.graph import KG
from identity_integrity.models import Triple
from identity_integrity.retrieval import ppr, score_sources, transition_matrix


def test_transition_matrix_uses_ordinary_edges_when_identity_mix_is_zero():
    kg = KG()
    kg.g.add_edge("a", "b", relation="related_to", source_id="s1", weight=1.0)
    kg.g.add_edge("a", "c", relation="same_as", source_id=None, weight=1.0)

    matrix, nodes = transition_matrix(kg)
    row = matrix.getrow(nodes.index("a")).toarray()[0]

    assert np.isclose(row[nodes.index("b")], 1.0)
    assert np.isclose(row[nodes.index("c")], 0.0)


def test_identity_mix_reserves_bounded_transition_mass_for_same_as_edges():
    kg = KG()
    kg.g.add_edge("a", "b", relation="related_to", source_id="s1", weight=1.0)
    kg.g.add_edge("a", "c", relation="same_as", source_id=None, weight=1.0)

    matrix, nodes = transition_matrix(kg, identity_mix=0.3)
    row = matrix.getrow(nodes.index("a")).toarray()[0]

    assert np.isclose(row[nodes.index("b")], 0.7)
    assert np.isclose(row[nodes.index("c")], 0.3)


def test_identity_weights_distribute_reserved_mass_between_candidates():
    kg = KG()
    kg.g.add_edge("a", "b", relation="related_to", source_id="s1", weight=1.0)
    kg.g.add_edge("a", "c", relation="same_as", source_id=None, weight=1.0)
    kg.g.add_edge("a", "d", relation="same_as", source_id=None, weight=3.0)

    matrix, nodes = transition_matrix(kg, identity_mix=0.4)
    row = matrix.getrow(nodes.index("a")).toarray()[0]

    assert np.isclose(row[nodes.index("b")], 0.6)
    assert np.isclose(row[nodes.index("c")], 0.1)
    assert np.isclose(row[nodes.index("d")], 0.3)


def test_ppr_mass_sums_to_one():
    kg = KG()
    kg.add_triples([Triple(subject="a", relation="related_to", object="b")], "s1")
    matrix, nodes = transition_matrix(kg)
    seed = np.zeros(len(nodes))
    seed[nodes.index("a")] = 1.0

    assert np.isclose(ppr(matrix, seed).sum(), 1.0)


def test_source_scoring_spreads_shared_node_mass():
    kg = KG()
    kg.add_triples([Triple(subject="bridge", relation="r1", object="a")], "s1")
    kg.add_triples([Triple(subject="bridge", relation="r2", object="b")], "s2")
    matrix, nodes = transition_matrix(kg)
    seed = np.zeros(len(nodes))
    seed[nodes.index("a")] = 1.0

    scores = score_sources(ppr(matrix, seed), nodes, kg)

    assert set(scores) == {"s1", "s2"}
    assert all(score >= 0 for score in scores.values())


def test_empty_transition_and_dangling_ppr_are_well_defined():
    matrix, nodes = transition_matrix(KG())
    assert matrix.shape == (0, 0)
    assert nodes == []
    mass = ppr(np.zeros((2, 2)), np.array([1.0, 0.0]))
    assert np.allclose(mass, [1.0, 0.0])


@pytest.mark.parametrize("identity_mix", [-0.1, 1.1, np.inf])
def test_transition_rejects_invalid_identity_mix(identity_mix):
    with pytest.raises(ValueError):
        transition_matrix(KG(), identity_mix=identity_mix)


def test_ppr_rejects_invalid_seed():
    with pytest.raises(ValueError):
        ppr(np.eye(2), np.zeros(2))
