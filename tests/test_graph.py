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
    kg.add_triples(make_triples([("acme", "employs", f"person{i}") for i in range(10)]), "s")
    assert kg.relation_entropy("acme") == 0.0
    assert kg.leakiness("acme") == 0.0


def test_relation_entropy_positive_for_heterogeneous_hub():
    kg = KG()
    kg.add_triples(make_triples([("meeting", f"rel{i}", f"thing{i}") for i in range(10)]), "s")
    assert math.isclose(kg.relation_entropy("meeting"), math.log(10))
    assert kg.leakiness("meeting") > kg.leakiness("thing0")


def test_save_load_roundtrip_creates_parent(tmp_path):
    kg = KG()
    kg.add_triples(make_triples([("a", "r1", "b"), ("a", "r2", "c")]), "s1")
    path = tmp_path / "nested" / "kg.json"
    kg.save(path)
    loaded = KG.load(path)
    assert loaded.nodes() == kg.nodes()
    assert loaded.edges() == kg.edges()
    assert loaded.node_sources == kg.node_sources
