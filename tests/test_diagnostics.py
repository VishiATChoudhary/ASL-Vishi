import importlib.util
from pathlib import Path

import numpy as np
import pytest

from supernode_poc.diagnostics import (
    degree_distribution,
    mass_through_top_leaky,
    random_seed_frequency,
)
from supernode_poc.graph import KG
from supernode_poc.models import Triple
from supernode_poc.retrieval import ppr, transition_matrix


def load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_musique = load_script("build_musique")
ingest_locomo = load_script("ingest_locomo")
run_eval = load_script("run_eval")


def build_hub_graph() -> KG:
    kg = KG()
    triples = [Triple(subject="hub", relation=f"rel{i}", object=f"leaf{i}") for i in range(20)]
    triples.append(Triple(subject="a", relation="knows", object="b"))
    kg.add_triples(triples, source_id="s")
    return kg


def test_degree_distribution_shape() -> None:
    kg = build_hub_graph()
    degrees, counts = degree_distribution(kg)
    assert degrees.max() == 20
    assert counts.sum() == len(kg.nodes())


def test_empty_degree_distribution() -> None:
    degrees, counts = degree_distribution(KG())
    assert degrees.size == counts.size == 0


def test_mass_through_top_leaky_flags_hub() -> None:
    kg = build_hub_graph()
    matrix, nodes = transition_matrix(kg)
    seed = np.zeros(len(nodes))
    seed[nodes.index("leaf0")] = 1.0
    fraction = mass_through_top_leaky(kg, ppr(matrix, seed), nodes, pct=0.05)
    assert fraction > 0.2


def test_mass_through_top_leaky_validates_inputs() -> None:
    with pytest.raises(ValueError, match="pct"):
        mass_through_top_leaky(KG(), np.array([]), [], pct=0.0)


def test_random_seed_frequency_is_deterministic() -> None:
    kg = build_hub_graph()
    first = random_seed_frequency(kg, n_queries=12, rng_seed=4)
    second = random_seed_frequency(kg, n_queries=12, rng_seed=4)
    assert first == second
    assert sum(first.values()) == 12 * 5


def test_locomo_chunking_is_numeric_and_deterministic() -> None:
    sample = {
        "conversation": {
            "session_10": [{"speaker": "B", "text": "later", "dia_id": "D2:1"}],
            "session_2": [{"speaker": "A", "text": "earlier", "dia_id": "D1:1"}],
            "session_2_date_time": "2024-01-01",
        }
    }
    chunks = ingest_locomo.conversation_chunks(sample, max_chunks=2, turns_per_chunk=1)
    assert chunks == [("session_2-0000", "A: earlier"), ("session_10-0000", "B: later")]


def test_musique_canonicalization_and_paragraph_layouts() -> None:
    assert build_musique.canonical_id("  Green.  Text ") == build_musique.canonical_id(
        "green. text"
    )
    assert build_musique.canonical_id("green. other") != build_musique.canonical_id("green. text")
    row = {"title": "T", "paragraph_text": "P", "is_supporting": True}
    assert list(build_musique.paragraph_rows([row])) == [row]
    columns = {
        "title": ["T1", "T2"],
        "paragraph_text": ["P1", "P2"],
        "is_supporting": [True, False],
    }
    assert [item["title"] for item in build_musique.paragraph_rows(columns)] == ["T1", "T2"]


def test_musique_sampling_is_nested_and_seeded() -> None:
    dataset = [{"id": f"q{index}"} for index in range(10)]
    small = build_musique.select_rows(dataset, 3, seed=7)
    large = build_musique.select_rows(dataset, 7, seed=7)
    assert {row["id"] for row in small} <= {row["id"] for row in large}
    assert small == build_musique.select_rows(dataset, 3, seed=7)


def test_eval_metrics_are_paired_and_reproducible() -> None:
    assert run_eval.recall_at_k(["a", "a", "b"], ["a", "c"]) == 0.5
    baseline = np.array([0.0, 0.5, 0.5, 1.0])
    selected = baseline + 0.25
    assert run_eval.bootstrap_mean_ci(baseline, 100, seed=3) == run_eval.bootstrap_mean_ci(
        baseline, 100, seed=3
    )
    low, high = run_eval.paired_delta_ci(baseline, selected, 100, seed=3)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)
