"""Run the controlled entity-fragmentation and oracle soft-repair experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from identity_integrity.extraction import DEFAULT_MODELS
from identity_integrity.graph import KG, fragment_by_source
from identity_integrity.models import Triple
from identity_integrity.retrieval import ppr, score_sources, transition_matrix

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
ARTIFACTS = ROOT / "artifacts"
QUESTIONS = CACHE / "musique_questions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", nargs="+", choices=sorted(DEFAULT_MODELS), default=None)
    parser.add_argument("--prompt", default="neutral")
    parser.add_argument("--identity-mixes", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_extracted_graph(provider: str, prompt: str) -> KG:
    """Rebuild a source-aware graph from one provider's validated extraction cache."""
    path = CACHE / f"musique_triples_{provider}_{prompt}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing extraction cache: {path}")

    graph = KG()
    provenance = None
    seen = set()
    with path.open(encoding="utf-8") as rows:
        for line in rows:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = row["source_id"]
            if source_id in seen:
                raise ValueError(f"duplicate extraction cache row for {source_id}")
            seen.add(source_id)
            graph.add_triples(
                [Triple.model_validate(triple) for triple in row["triples"]], source_id
            )
            current = {
                key: row[key]
                for key in (
                    "provider",
                    "model",
                    "effort",
                    "cli_version",
                    "system_sha256",
                    "schema_sha256",
                )
            }
            provenance = provenance or current
            if current != provenance:
                raise ValueError(f"mixed extraction provenance in {path}")

    if not graph.nodes():
        raise ValueError(f"extraction cache produced an empty graph: {path}")
    graph.metadata = {"corpus": "MuSiQue", "prompt": prompt, **(provenance or {})}
    return graph


def paired_ci(delta: np.ndarray, samples: int, seed: int) -> list[float]:
    """Return a paired percentile-bootstrap interval over questions."""
    if len(delta) == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    return [float(value) for value in np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])]


def anchor_recall(
    graph: KG,
    seed_node: str,
    anchor_source: str,
    targets: set[str],
    k: int,
    identity_mix: float = 0.0,
) -> float:
    """Measure whether a walk from one shard retrieves evidence on the other shard."""
    matrix, nodes = transition_matrix(graph, identity_mix=identity_mix)
    seed = np.zeros(len(nodes), dtype=float)
    seed[nodes.index(seed_node)] = 1.0
    source_scores = score_sources(ppr(matrix, seed), nodes, graph)
    source_scores.pop(anchor_source, None)
    retrieved = [
        source_id
        for source_id, _ in sorted(source_scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    ]
    return len(set(retrieved) & targets) / len(targets)


def run_bridge_probe(
    graph: KG,
    questions: list[dict[str, Any]],
    identity_mixes: list[float],
    trials: int,
    shards: int,
    k: int,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Inject one bridge split per eligible question and test a soft identity repair."""
    rows = []
    graph_nodes = graph.nodes()
    for question_index, question in enumerate(questions):
        supporting = set(question["supporting"])
        candidates = [
            node
            for node in graph_nodes
            if len(graph.node_sources.get(node, set()) & supporting) >= 2
        ]
        if not candidates:
            continue

        bridge = sorted(
            candidates,
            key=lambda node: (
                -len(graph.node_sources[node] & supporting),
                -graph.degree(node),
                node,
            ),
        )[0]

        for trial in range(trials):
            case = _fragmented_case(
                graph,
                bridge,
                supporting,
                question_index,
                trial,
                shards,
                seed,
            )
            if case is None:
                continue
            assignment_seed, fragmented, anchor_fragment, anchor_source, targets = case
            repaired, repaired_groups = fragment_by_source(
                graph,
                {bridge},
                shards=shards,
                seed=assignment_seed,
                identity_weight=1.0,
            )
            repaired_anchor = next(
                fragment
                for fragment in repaired_groups[bridge]
                if anchor_source in repaired.node_sources[fragment]
            )
            rows.append(
                {
                    "id": question["id"],
                    "split": question["split"],
                    "trial": trial,
                    "bridge": bridge,
                    "target_count": len(targets),
                    "original": anchor_recall(graph, bridge, anchor_source, targets, k),
                    "fragmented": anchor_recall(
                        fragmented,
                        anchor_fragment,
                        anchor_source,
                        targets,
                        k,
                    ),
                    "repaired": {
                        str(mix): anchor_recall(
                            repaired,
                            repaired_anchor,
                            anchor_source,
                            targets,
                            k,
                            identity_mix=mix,
                        )
                        for mix in identity_mixes
                    },
                }
            )

    dev_rows = [row for row in rows if row["split"] == "dev"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not dev_rows or not test_rows:
        raise ValueError("bridge probe requires eligible development and test questions")

    dev_means = {
        mix: float(np.mean([row["repaired"][str(mix)] for row in dev_rows]))
        for mix in identity_mixes
    }
    selected_mix = sorted(identity_mixes, key=lambda mix: (-dev_means[mix], mix))[0]
    original = _per_question_means(test_rows, "original")
    fragmented = _per_question_means(test_rows, "fragmented")
    repaired = _per_question_means(test_rows, "repaired", selected_mix)
    fragmentation_delta = fragmented - original
    repair_delta = repaired - fragmented
    gap = float(original.mean() - fragmented.mean())
    recovery = float(repaired.mean() - fragmented.mean())
    recovery_fraction = recovery / gap if gap > 0 else None
    fragmentation_ci = paired_ci(fragmentation_delta, bootstrap, seed)
    repair_ci = paired_ci(repair_delta, bootstrap, seed)
    fault_verified = fragmentation_ci[1] < 0
    poc_verified = (
        fault_verified
        and recovery_fraction is not None
        and recovery_fraction >= 0.5
        and repair_ci[0] > 0
    )

    return {
        "design": {
            "bridge_selection": "most supporting-source coverage, then degree, then name",
            "anchor": "one supporting source on one fragment",
            "metric": f"other-supporting-source Recall@{k}",
            "identity_mixes_selected_on_dev": identity_mixes,
            "oracle_identity_links": True,
        },
        "sample_sizes": {
            "dev_questions": len({row["id"] for row in dev_rows}),
            "test_questions": len({row["id"] for row in test_rows}),
            "trials_per_question": trials,
        },
        "dev_identity_mix_means": {str(mix): mean for mix, mean in dev_means.items()},
        "selected_identity_mix": selected_mix,
        "test": {
            "original_mean": float(original.mean()),
            "fragmented_mean": float(fragmented.mean()),
            "repaired_mean": float(repaired.mean()),
            "fragmentation_delta": float(fragmentation_delta.mean()),
            "fragmentation_delta_ci95": fragmentation_ci,
            "repair_delta": float(repair_delta.mean()),
            "repair_delta_ci95": repair_ci,
            "positive_gap_recovery_fraction": recovery_fraction,
            "fault_verified": fault_verified,
            "poc_verified": poc_verified,
        },
    }


def _fragmented_case(
    graph: KG,
    bridge: str,
    supporting: set[str],
    question_index: int,
    trial: int,
    shards: int,
    seed: int,
) -> tuple[int, KG, str, str, set[str]] | None:
    """Find a deterministic split that places supporting evidence on both sides."""
    for attempt in range(100):
        assignment_seed = seed + question_index * 10_000 + trial * 100 + attempt
        fragmented, groups = fragment_by_source(
            graph,
            {bridge},
            shards=shards,
            seed=assignment_seed,
        )
        supported_by_fragment = {
            fragment: fragmented.node_sources[fragment] & supporting
            for fragment in groups[bridge]
        }
        occupied = [fragment for fragment, sources in supported_by_fragment.items() if sources]
        if len(occupied) < 2:
            continue

        anchor_fragment = sorted(
            occupied,
            key=lambda fragment: sorted(supported_by_fragment[fragment])[0],
        )[0]
        anchor_source = sorted(supported_by_fragment[anchor_fragment])[0]
        targets = set().union(
            *(
                supported_by_fragment[fragment]
                for fragment in occupied
                if fragment != anchor_fragment
            )
        )
        return assignment_seed, fragmented, anchor_fragment, anchor_source, targets
    return None


def _per_question_means(
    rows: list[dict[str, Any]],
    field: str,
    mix: float | None = None,
) -> np.ndarray:
    values = []
    for question_id in sorted({row["id"] for row in rows}):
        question_rows = [row for row in rows if row["id"] == question_id]
        observations = (
            [row[field] for row in question_rows]
            if mix is None
            else [row[field][str(mix)] for row in question_rows]
        )
        values.append(float(np.mean(observations)))
    return np.asarray(values)


def run_provider(
    provider: str,
    questions: list[dict[str, Any]],
    identity_mixes: list[float],
    trials: int,
    shards: int,
    k: int,
    bootstrap: int,
    seed: int,
    prompt: str,
) -> dict[str, Any]:
    graph = load_extracted_graph(provider, prompt)
    bridge_probe = run_bridge_probe(
        graph,
        questions,
        identity_mixes,
        trials,
        shards,
        k,
        bootstrap,
        seed,
    )
    result = bridge_probe["test"]
    print(
        f"{provider}: original={result['original_mean']:.3f} "
        f"fragmented={result['fragmented_mean']:.3f} repaired={result['repaired_mean']:.3f}",
        flush=True,
    )
    return {
        "graph": {
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges()),
            "extraction": graph.metadata,
        },
        "bridge_probe": bridge_probe,
    }


def write_plot(results: dict[str, Any], path: Path) -> None:
    providers = list(results)
    x = np.arange(len(providers), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for offset, key, label in (
        (-width, "original_mean", "Original"),
        (0.0, "fragmented_mean", "Fragmented"),
        (width, "repaired_mean", "Soft identity repair"),
    ):
        values = [results[provider]["bridge_probe"]["test"][key] for provider in providers]
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x, [provider.title() for provider in providers])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Other-evidence Recall@5")
    ax.set_title("Identity-bridge failure and soft repair")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    identity_mixes = sorted(set(args.identity_mixes))
    if any(not 0 < mix <= 1 for mix in identity_mixes):
        raise ValueError("identity mixes must be in (0, 1]")
    if args.trials < 1 or args.shards < 2 or args.k < 1 or args.bootstrap < 1:
        raise ValueError("trials, k, bootstrap, and shards must be positive")
    if not QUESTIONS.exists():
        raise FileNotFoundError(f"missing questions cache: {QUESTIONS}")

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    providers = args.providers or sorted(DEFAULT_MODELS)
    results = {
        provider: run_provider(
            provider,
            questions,
            identity_mixes,
            args.trials,
            args.shards,
            args.k,
            args.bootstrap,
            args.seed,
            args.prompt,
        )
        for provider in providers
    }
    report = {
        "protocol": {
            "name": "source-consistent identity-bridge fragmentation PoC",
            "questions": len(questions),
            "dev": sum(question["split"] == "dev" for question in questions),
            "test": sum(question["split"] == "test" for question in questions),
            "identity_mixes_selected_on_dev": identity_mixes,
            "trials": args.trials,
            "shards": args.shards,
            "k": args.k,
            "bootstrap": args.bootstrap,
            "bootstrap_unit": "question after averaging fragmentation trials",
            "seed": args.seed,
            "fault_success": "paired fragmentation-loss interval below zero",
            "repair_success": (
                "recover at least half the gap with paired repair-gain interval above zero"
            ),
            "oracle_identity_links": True,
            "scope": "controlled mechanism test, not a natural duplicate-rate estimate",
        },
        "providers": results,
        "fault_verified_both": all(
            result["bridge_probe"]["test"]["fault_verified"] for result in results.values()
        ),
        "poc_verified_both": all(
            result["bridge_probe"]["test"]["poc_verified"] for result in results.values()
        ),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS / "fragmentation_poc.json"
    plot_path = ARTIFACTS / "fragmentation_dose_response.png"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_plot(results, plot_path)
    print(f"Wrote {report_path.relative_to(ROOT)} and {plot_path.relative_to(ROOT)}")
    print(
        f"Fault verified: {report['fault_verified_both']}; "
        f"soft-repair PoC verified: {report['poc_verified_both']}"
    )


if __name__ == "__main__":
    main()
