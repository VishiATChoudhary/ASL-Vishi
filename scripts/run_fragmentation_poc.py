"""Run the causal entity-fragmentation and oracle soft-repair PoC."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from supernode_poc.embeddings import Embedder
from supernode_poc.extraction import DEFAULT_MODELS
from supernode_poc.graph import KG, fragment_by_source
from supernode_poc.models import Triple
from supernode_poc.retrieval import ppr, score_sources, transition_matrix

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
ARTIFACTS = ROOT / "artifacts"
QUESTIONS = CACHE / "musique_questions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", nargs="+", choices=sorted(DEFAULT_MODELS), default=None)
    parser.add_argument("--prompt", default="neutral")
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[0, 0.1, 0.25, 0.35, 0.5, 0.75, 1]
    )
    parser.add_argument("--repair-fraction", type=float, default=0.35)
    parser.add_argument("--identity-weights", nargs="+", type=float, default=[0.1, 0.3, 1.0])
    parser.add_argument("--identity-mixes", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_extracted_graph(provider: str, prompt: str) -> KG:
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
    graph.metadata = {"corpus": "MuSiQue", "prompt": prompt, **(provenance or {})}
    return graph


def recall_at_k(retrieved: list[str], supporting: list[str]) -> float:
    return len(set(retrieved) & set(supporting)) / len(set(supporting))


def evaluate(
    graph: KG,
    questions: list[dict[str, Any]],
    embedder: Embedder,
    k: int,
) -> np.ndarray:
    matrix, nodes = transition_matrix(graph)
    labels = graph.labels(nodes)
    node_embeddings = embedder.embed(labels)
    scores = []
    for question in questions:
        query = embedder.embed([question["question"]])[0]
        similarities = node_embeddings @ query
        order = np.argsort(-similarities, kind="stable")[: min(10, len(nodes))]
        seed = np.zeros(len(nodes), dtype=float)
        seed[order] = np.maximum(similarities[order], 0.0)
        if seed.sum() == 0:
            seed[order[0]] = 1.0
        source_scores = score_sources(ppr(matrix, seed), nodes, graph)
        retrieved = [
            source_id
            for source_id, _ in sorted(source_scores.items(), key=lambda item: (-item[1], item[0]))[
                :k
            ]
        ]
        scores.append(recall_at_k(retrieved, question["supporting"]))
    return np.asarray(scores, dtype=float)


def selected_nodes(eligible: list[str], fraction: float, trial: int, seed: int) -> set[str]:
    count = round(len(eligible) * fraction)
    if count == 0:
        return set()
    permutation = np.random.default_rng(seed + trial).permutation(len(eligible))
    return {eligible[index] for index in permutation[:count]}


def split_scores(values: np.ndarray, questions: list[dict[str, Any]], split: str) -> np.ndarray:
    return values[[question["split"] == split for question in questions]]


def paired_ci(delta: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    return [float(value) for value in np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])]


def summarize_trials(matrix: np.ndarray) -> dict[str, Any]:
    trial_means = matrix.mean(axis=1)
    return {
        "mean": float(matrix.mean()),
        "trial_mean_sd": float(trial_means.std(ddof=1)) if len(trial_means) > 1 else 0.0,
        "trial_means": [float(value) for value in trial_means],
    }


def bridge_nodes(graph: KG, question: dict[str, Any], eligible: set[str]) -> set[str]:
    supporting = set(question["supporting"])
    return {node for node in eligible if len(graph.node_sources.get(node, set()) & supporting) >= 2}


def anchor_recall(
    graph: KG,
    seed_node: str,
    anchor_source: str,
    targets: set[str],
    k: int,
    identity_mix: float | None = None,
) -> float:
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
            case = None
            for attempt in range(100):
                assignment_seed = seed + question_index * 10_000 + trial * 100 + attempt
                fragmented, groups = fragment_by_source(
                    graph, {bridge}, shards=shards, seed=assignment_seed
                )
                supported_by_fragment = {
                    fragment: fragmented.node_sources[fragment] & supporting
                    for fragment in groups[bridge]
                }
                occupied = [
                    fragment for fragment, sources in supported_by_fragment.items() if sources
                ]
                if len(occupied) < 2:
                    continue
                anchor_fragment = sorted(
                    occupied, key=lambda fragment: sorted(supported_by_fragment[fragment])[0]
                )[0]
                anchor_source = sorted(supported_by_fragment[anchor_fragment])[0]
                targets = set().union(
                    *(
                        supported_by_fragment[fragment]
                        for fragment in occupied
                        if fragment != anchor_fragment
                    )
                )
                case = assignment_seed, fragmented, anchor_fragment, anchor_source, targets
                break
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
                        fragmented, anchor_fragment, anchor_source, targets, k
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


def _per_question_means(
    rows: list[dict[str, Any]], field: str, mix: float | None = None
) -> np.ndarray:
    ids = sorted({row["id"] for row in rows})
    values = []
    for question_id in ids:
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
    fractions: list[float],
    repair_fraction: float,
    identity_weights: list[float],
    identity_mixes: list[float],
    trials: int,
    shards: int,
    k: int,
    bootstrap: int,
    seed: int,
    prompt: str,
) -> dict[str, Any]:
    graph = load_extracted_graph(provider, prompt)
    eligible = sorted(
        node for node in graph.nodes() if len(graph.node_sources.get(node, ())) >= shards
    )
    eligible_set = set(eligible)
    embedder = Embedder()
    original = evaluate(graph, questions, embedder, k)
    masks = {
        split: np.asarray([question["split"] == split for question in questions])
        for split in ("dev", "test")
    }
    fragmented_scores: dict[tuple[float, int], np.ndarray] = {}
    dose = []
    for fraction in fractions:
        trial_values = []
        for trial in range(trials):
            if fraction == 0:
                values = original
            else:
                selected = selected_nodes(eligible, fraction, trial, seed)
                fragmented, _ = fragment_by_source(
                    graph, selected, shards=shards, seed=seed + trial
                )
                values = evaluate(fragmented, questions, embedder, k)
            fragmented_scores[(fraction, trial)] = values
            trial_values.append(values)
        matrix = np.stack(trial_values)
        dose.append(
            {
                "fraction": fraction,
                "all": summarize_trials(matrix),
                "dev": summarize_trials(matrix[:, masks["dev"]]),
                "test": summarize_trials(matrix[:, masks["test"]]),
            }
        )
        print(
            f"{provider} fraction={fraction:g} "
            f"dev={dose[-1]['dev']['mean']:.3f} test={dose[-1]['test']['mean']:.3f}",
            flush=True,
        )

    test_means = [row["test"]["mean"] for row in dose]
    rho = spearmanr(fractions, test_means)
    monotonic = all(right <= left + 1e-12 for left, right in pairwise(test_means))
    fault_verified = monotonic and test_means[-1] < test_means[0]

    dev_weight_means = {}
    repair_dev_scores = {}
    for weight in identity_weights:
        trials_for_weight = []
        for trial in range(trials):
            selected = selected_nodes(eligible, repair_fraction, trial, seed)
            repaired, _ = fragment_by_source(
                graph,
                selected,
                shards=shards,
                seed=seed + trial,
                identity_weight=weight,
            )
            trials_for_weight.append(evaluate(repaired, questions, embedder, k))
        matrix = np.stack(trials_for_weight)
        repair_dev_scores[weight] = matrix
        dev_weight_means[weight] = float(matrix[:, masks["dev"]].mean())
        print(
            f"{provider} identity_weight={weight:g} dev={dev_weight_means[weight]:.3f}",
            flush=True,
        )
    selected_weight = sorted(
        identity_weights, key=lambda weight: (-dev_weight_means[weight], weight)
    )[0]

    original_test = original[masks["test"]]
    fragmented_test = np.stack(
        [fragmented_scores[(repair_fraction, trial)][masks["test"]] for trial in range(trials)]
    )
    repaired_test = repair_dev_scores[selected_weight][:, masks["test"]]
    fragmented_per_question = fragmented_test.mean(axis=0)
    repaired_per_question = repaired_test.mean(axis=0)
    original_mean = float(original_test.mean())
    fragmented_mean = float(fragmented_test.mean())
    repaired_mean = float(repaired_test.mean())
    gap = original_mean - fragmented_mean
    recovered = repaired_mean - fragmented_mean
    recovery_fraction = recovered / gap if gap > 0 else None

    exposed_fragment_delta = []
    unexposed_fragment_delta = []
    exposed_repair_delta = []
    unexposed_repair_delta = []
    test_questions = [question for question in questions if question["split"] == "test"]
    bridges = [bridge_nodes(graph, question, eligible_set) for question in test_questions]
    for trial in range(trials):
        selected = selected_nodes(eligible, repair_fraction, trial, seed)
        for index, question_bridges in enumerate(bridges):
            exposed = bool(selected & question_bridges)
            fragment_delta = fragmented_test[trial, index] - original_test[index]
            repair_delta = repaired_test[trial, index] - fragmented_test[trial, index]
            (exposed_fragment_delta if exposed else unexposed_fragment_delta).append(fragment_delta)
            (exposed_repair_delta if exposed else unexposed_repair_delta).append(repair_delta)

    repair = {
        "fraction": repair_fraction,
        "dev_identity_weight_means": {
            str(weight): mean for weight, mean in dev_weight_means.items()
        },
        "selected_identity_weight": selected_weight,
        "test": {
            "original_mean": original_mean,
            "fragmented_mean": fragmented_mean,
            "repaired_mean": repaired_mean,
            "fragmentation_delta": fragmented_mean - original_mean,
            "fragmentation_delta_ci95": paired_ci(
                fragmented_per_question - original_test, bootstrap, seed
            ),
            "repair_delta": recovered,
            "repair_delta_ci95": paired_ci(
                repaired_per_question - fragmented_per_question, bootstrap, seed
            ),
            "positive_gap_recovery_fraction": recovery_fraction,
            "poc_verified": recovery_fraction is not None and recovery_fraction >= 0.5,
        },
        "cross_supporting_source_analysis": {
            "questions_with_bridge_nodes": sum(bool(nodes) for nodes in bridges),
            "exposed_fragmentation_delta": _mean_or_none(exposed_fragment_delta),
            "unexposed_fragmentation_delta": _mean_or_none(unexposed_fragment_delta),
            "exposed_repair_delta": _mean_or_none(exposed_repair_delta),
            "unexposed_repair_delta": _mean_or_none(unexposed_repair_delta),
        },
    }
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
    return {
        "graph": {
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges()),
            "eligible_multi_source_nodes": len(eligible),
            "eligible_fraction": len(eligible) / len(graph.nodes()),
            "extraction": graph.metadata,
        },
        "dose_response": dose,
        "dose_response_test_spearman": {
            "rho": _finite_or_none(rho.statistic),
            "p": _finite_or_none(rho.pvalue),
        },
        "strictly_nonincreasing_test_curve": monotonic,
        "fault_verified": fault_verified,
        "repair": repair,
        "bridge_probe": bridge_probe,
    }


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def write_plot(results: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for provider, result in results.items():
        dose = result["dose_response"]
        x = [row["fraction"] for row in dose]
        y = [row["test"]["mean"] for row in dose]
        error = [row["test"]["trial_mean_sd"] for row in dose]
        ax.errorbar(x, y, yerr=error, marker="o", capsize=3, label=provider)
    ax.set_xlabel("Fraction of multi-source entities split into two shards")
    ax.set_ylabel("Held-out Recall@5")
    ax.set_title("Synthetic entity fragmentation dose-response")
    ax.legend()

    ax = axes[1]
    providers = list(results)
    x = np.arange(len(providers), dtype=float)
    width = 0.24
    for offset, key, label in (
        (-width, "original_mean", "Original"),
        (0.0, "fragmented_mean", "Fragmented"),
        (width, "repaired_mean", "Soft identity repair"),
    ):
        values = [results[provider]["bridge_probe"]["test"][key] for provider in providers]
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x, providers)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Other-evidence Recall@5")
    ax.set_title("Anchor-seeded cross-source bridge probe")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    providers = args.providers or sorted(DEFAULT_MODELS)
    fractions = sorted(set(args.fractions) | {0.0, args.repair_fraction})
    if any(not 0 <= fraction <= 1 for fraction in fractions):
        raise ValueError("fragmentation fractions must be in [0, 1]")
    if any(weight <= 0 for weight in args.identity_weights):
        raise ValueError("identity weights must be positive")
    if any(not 0 < mix <= 1 for mix in args.identity_mixes):
        raise ValueError("identity mixes must be in (0, 1]")
    if args.trials < 1 or args.shards < 2 or args.k < 1 or args.bootstrap < 1:
        raise ValueError("trials, k, bootstrap, and shards must be positive")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    results = {
        provider: run_provider(
            provider,
            questions,
            fractions,
            args.repair_fraction,
            sorted(set(args.identity_weights)),
            sorted(set(args.identity_mixes)),
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
            "name": "source-consistent synthetic entity fragmentation PoC",
            "questions": len(questions),
            "dev": sum(question["split"] == "dev" for question in questions),
            "test": sum(question["split"] == "test" for question in questions),
            "fractions": fractions,
            "repair_fraction": args.repair_fraction,
            "identity_weights_selected_on_dev": sorted(set(args.identity_weights)),
            "bridge_identity_mixes_selected_on_dev": sorted(set(args.identity_mixes)),
            "trials": args.trials,
            "shards": args.shards,
            "k": args.k,
            "bootstrap": args.bootstrap,
            "bootstrap_unit": "question after averaging fragmentation trials",
            "seed": args.seed,
            "fault_success": "nonincreasing test curve with lower recall at full fragmentation",
            "repair_success": "recover at least half of a positive test recall gap",
            "bridge_fault_success": "paired fragmentation-loss interval below zero",
            "bridge_repair_success": (
                "recover at least half the gap with paired repair-gain interval above zero"
            ),
            "bridge_probe_adaptive_follow_up": (
                "designed after the broad null; criteria fixed before bridge outcomes"
            ),
            "oracle_identity_links": True,
        },
        "providers": results,
        "broad_dose_fault_verified_both": all(
            result["fault_verified"] for result in results.values()
        ),
        "bridge_fault_verified_both": all(
            result["bridge_probe"]["test"]["fault_verified"] for result in results.values()
        ),
        "bridge_poc_verified_both": all(
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
        f"Broad dose fault verified: {report['broad_dose_fault_verified_both']}; "
        f"bridge fault verified: {report['bridge_fault_verified_both']}; "
        f"bridge PoC verified: {report['bridge_poc_verified_both']}"
    )


if __name__ == "__main__":
    main()
