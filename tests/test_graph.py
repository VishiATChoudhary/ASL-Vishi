from identity_integrity.graph import KG, fragment_by_source
from identity_integrity.models import Triple


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


def test_save_load_roundtrip_creates_parent(tmp_path):
    kg = KG()
    kg.metadata = {"provider": "claude", "model": "claude-opus-5"}
    kg.add_triples(make_triples([("a", "r1", "b"), ("a", "r2", "c")]), "s1")
    path = tmp_path / "nested" / "kg.json"
    kg.save(path)
    loaded = KG.load(path)
    assert loaded.nodes() == kg.nodes()
    assert loaded.edges() == kg.edges()
    assert loaded.node_sources == kg.node_sources
    assert loaded.metadata == kg.metadata
    assert loaded.edge_records() == kg.edge_records()


def test_fragment_by_source_preserves_semantics_and_adds_optional_identity_edge():
    kg = KG()
    kg.add_triples(make_triples([("Alice", "likes", "tea")]), "s1")
    kg.add_triples(make_triples([("Alice", "lives_in", "Paris")]), "s2")

    fragmented, groups = fragment_by_source(kg, {"alice"}, seed=0)
    fragments = groups["alice"]
    assert len(fragments) == 2
    assert fragmented.labels(fragments) == ["alice", "alice"]
    assert {frozenset(fragmented.node_sources[node]) for node in fragments} == {
        frozenset({"s1"}),
        frozenset({"s2"}),
    }
    assert all(record[3] in {"s1", "s2"} for record in fragmented.edge_records())

    repaired, repaired_groups = fragment_by_source(kg, {"alice"}, seed=0, identity_weight=0.3)
    identity = [record for record in repaired.edge_records() if record[2] == "same_as"]
    assert repaired_groups == groups
    assert identity == [(fragments[0], fragments[1], "same_as", None, 0.3)]
