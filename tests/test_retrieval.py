import numpy as np
import pytest

from supernode_poc.graph import KG
from supernode_poc.models import Triple
from supernode_poc.retrieval import ppr, retrieve, score_sources, transition_matrix


def star_kg(hub: str, n: int, heterogeneous: bool) -> KG:
    """Build a star whose leaves have alternate neighbors."""
    kg = KG()
    triples = [
        Triple(
            subject=hub,
            relation=f"rel{i}" if heterogeneous else "employs",
            object=f"leaf{i}",
        )
        for i in range(n)
    ]
    triples.extend(
        Triple(subject=f"leaf{i}", relation="knows", object=f"partner{i}") for i in range(n)
    )
    kg.add_triples(triples, "s")
    return kg


def run_ppr_from_leaf(kg: KG, beta: float, kernel: str = "entropy") -> dict[str, float]:
    matrix, nodes = transition_matrix(kg, beta=beta, kernel=kernel)
    seed = np.zeros(len(nodes))
    seed[nodes.index("leaf0")] = 1.0
    return dict(zip(nodes, ppr(matrix, seed), strict=True))


def test_ppr_mass_sums_to_one():
    assert np.isclose(sum(run_ppr_from_leaf(star_kg("hub", 5, True), 0).values()), 1.0)


def test_beta_zero_matches_homogeneous_and_heterogeneous():
    heterogeneous = run_ppr_from_leaf(star_kg("hub", 10, True), 0)
    homogeneous = run_ppr_from_leaf(star_kg("hub", 10, False), 0)
    assert np.isclose(heterogeneous["hub"], homogeneous["hub"])


def test_entropy_damping_only_reduces_heterogeneous_hub_mass():
    heterogeneous_0 = run_ppr_from_leaf(star_kg("hub", 10, True), 0)
    heterogeneous_1 = run_ppr_from_leaf(star_kg("hub", 10, True), 1)
    homogeneous_0 = run_ppr_from_leaf(star_kg("hub", 10, False), 0)
    homogeneous_1 = run_ppr_from_leaf(star_kg("hub", 10, False), 1)
    assert heterogeneous_1["hub"] < heterogeneous_0["hub"]
    assert np.isclose(homogeneous_1["hub"], homogeneous_0["hub"])


def test_degree_kernel_damps_homogeneous_hub_too():
    kg = star_kg("hub", 10, False)
    assert run_ppr_from_leaf(kg, 1, "degree")["hub"] < run_ppr_from_leaf(kg, 0)["hub"]


def test_transition_respects_edge_weights():
    kg = KG()
    kg.g.add_edge("a", "b", relation="r", source_id="s", weight=1.0)
    kg.g.add_edge("a", "c", relation="same_as", source_id=None, weight=3.0)
    matrix, nodes = transition_matrix(kg)
    row = matrix.getrow(nodes.index("a")).toarray()[0]
    assert np.isclose(row[nodes.index("b")], 0.25)
    assert np.isclose(row[nodes.index("c")], 0.75)


def test_identity_mix_reserves_transition_mass_for_same_as_edges():
    kg = KG()
    kg.g.add_edge("a", "b", relation="r", source_id="s", weight=1.0)
    kg.g.add_edge("a", "c", relation="same_as", source_id=None, weight=1.0)
    matrix, nodes = transition_matrix(kg, identity_mix=0.3)
    row = matrix.getrow(nodes.index("a")).toarray()[0]
    assert np.isclose(row[nodes.index("b")], 0.7)
    assert np.isclose(row[nodes.index("c")], 0.3)


def test_source_scoring_spreads_shared_node_mass():
    kg = KG()
    kg.add_triples([Triple(subject="hub", relation="r1", object="a")], "s1")
    kg.add_triples([Triple(subject="hub", relation="r2", object="b")], "s2")
    matrix, nodes = transition_matrix(kg)
    seed = np.zeros(len(nodes))
    seed[nodes.index("a")] = 1
    mass = ppr(matrix, seed)
    spread = score_sources(mass, nodes, kg)
    raw = score_sources(mass, nodes, kg, spread_normalize=False)
    assert spread["s1"] < raw["s1"]


def test_exclude_top_leaky_drops_hub_from_scoring():
    kg = star_kg("hub", 10, True)
    matrix, nodes = transition_matrix(kg)
    seed = np.zeros(len(nodes))
    seed[nodes.index("leaf0")] = 1
    mass = ppr(matrix, seed)
    with_hub = score_sources(mass, nodes, kg, spread_normalize=False)
    without_hub = score_sources(mass, nodes, kg, spread_normalize=False, exclude_top_leaky_pct=0.05)
    assert without_hub["s"] < with_hub["s"]


def test_empty_transition_and_dangling_ppr_are_well_defined():
    matrix, nodes = transition_matrix(KG())
    assert matrix.shape == (0, 0)
    assert nodes == []
    mass = ppr(np.zeros((2, 2)), np.array([1.0, 0.0]))
    assert np.allclose(mass, [1.0, 0.0])


@pytest.mark.parametrize("beta", [-1, np.inf])
def test_transition_rejects_invalid_beta(beta):
    with pytest.raises(ValueError):
        transition_matrix(star_kg("hub", 2, True), beta=beta)


def test_ppr_rejects_invalid_seed():
    with pytest.raises(ValueError):
        ppr(np.eye(2), np.zeros(2))


def test_retrieve_breaks_score_ties_by_source_id():
    class ConstantEmbedder:
        @staticmethod
        def embed(texts):
            return np.ones((len(texts), 1))

    kg = KG()
    kg.add_triples([Triple(subject="a", relation="r", object="b")], "z_source")
    kg.add_triples([Triple(subject="c", relation="r", object="d")], "a_source")
    assert retrieve(kg, "query", ConstantEmbedder(), k=2) == ["a_source", "z_source"]
