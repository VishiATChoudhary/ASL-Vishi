"""Evaluate damped PPR on the transductive open-corpus MuSiQue variant.

Entropy and degree configurations are selected independently on dev, then
compared with vanilla PPR on held-out test questions using paired bootstrap
resampling. Entropy is the primary hypothesis; degree is an ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from supernode_poc.embeddings import Embedder
from supernode_poc.extraction import DEFAULT_MODELS, PROMPTS
from supernode_poc.graph import KG
from supernode_poc.retrieval import retrieve

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
QUESTIONS_PATH = ROOT / "data/cache/musique_questions.json"


def recall_at_k(retrieved: list[str], supporting: list[str]) -> float:
    if not supporting:
        raise ValueError("supporting source IDs must not be empty")
    return len(set(retrieved) & set(supporting)) / len(set(supporting))


def evaluate_config(
    kg: KG,
    questions: list[dict[str, Any]],
    embedder: Embedder,
    *,
    beta: float,
    kernel: str,
    k: int,
) -> list[dict[str, Any]]:
    records = []
    for question in questions:
        retrieved = retrieve(
            kg,
            question["question"],
            embedder,
            beta=beta,
            kernel=kernel,
            k=k,
        )
        records.append(
            {
                "id": question["id"],
                "score": recall_at_k(retrieved, question["supporting"]),
                "retrieved": retrieved,
                "supporting": question["supporting"],
            }
        )
    return records


def scores(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([record["score"] for record in records], dtype=float)


def source_coverage(questions: list[dict[str, Any]], available: set[str]) -> float:
    supporting = [source for question in questions for source in set(question["supporting"])]
    return sum(source in available for source in supporting) / len(supporting)


def bootstrap_mean_ci(values: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty split")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_delta_ci(
    baseline: np.ndarray, selected: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float]:
    if baseline.shape != selected.shape or baseline.size == 0:
        raise ValueError("paired scores must have equal, nonzero length")
    return bootstrap_mean_ci(selected - baseline, n_resamples, seed)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="claude")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="neutral")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    betas = list(dict.fromkeys(args.betas))
    if args.k < 1 or args.bootstrap < 1:
        raise ValueError("--k and --bootstrap must be positive")
    if any(beta < 0 for beta in betas):
        raise ValueError("--betas must be non-negative")
    candidates = [beta for beta in betas if beta > 0]
    if not candidates:
        raise ValueError("provide at least one beta greater than zero")

    graph_path = ARTIFACTS / f"musique_{args.provider}_{args.prompt}_kg.json"
    if not graph_path.exists() or not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"run scripts/build_musique.py --provider {args.provider} --prompt {args.prompt} first"
        )
    kg = KG.load(graph_path)
    questions_sha256 = file_sha256(QUESTIONS_PATH)
    expected_questions_sha256 = kg.metadata.get("questions_sha256")
    if expected_questions_sha256 and expected_questions_sha256 != questions_sha256:
        raise ValueError("question metadata does not match the graph build")
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    dev = [question for question in questions if question.get("split") == "dev"]
    test = [question for question in questions if question.get("split") == "test"]
    if not dev or not test:
        raise ValueError("question metadata must contain non-empty dev and test splits")
    available_sources = {source for sources in kg.node_sources.values() for source in sources}
    coverage = {
        "dev": source_coverage(dev, available_sources),
        "test": source_coverage(test, available_sources),
    }
    embedder = Embedder()

    print(f"dev={len(dev)} test={len(test)} k={args.k} (transductive open-corpus MuSiQue variant)")
    print(
        f"Supporting-source graph coverage: dev={coverage['dev']:.3f} test={coverage['test']:.3f}"
    )
    print("\nDev sweep for configuration selection:")
    print(f"{'kernel':>8} {'beta':>7} | {'dev Recall@' + str(args.k):>14}")
    print("-" * 35)
    dev_results: list[dict[str, Any]] = []
    for kernel in ("entropy", "degree"):
        for beta in betas:
            records = evaluate_config(kg, dev, embedder, beta=beta, kernel=kernel, k=args.k)
            mean = float(scores(records).mean())
            result = {"kernel": kernel, "beta": beta, "mean": mean}
            dev_results.append(result)
            print(f"{kernel:>8} {beta:>7g} | {mean:>14.3f}")

    selected_configs = {}
    for kernel in ("entropy", "degree"):
        eligible = [
            result for result in dev_results if result["kernel"] == kernel and result["beta"] > 0
        ]
        selected_configs[kernel] = sorted(
            eligible, key=lambda result: (-result["mean"], result["beta"])
        )[0]
    print("\nSelected independently on dev:")
    for kernel, config in selected_configs.items():
        print(f"  {kernel}: beta={config['beta']:g} Recall@{args.k}={config['mean']:.3f}")

    test_records = {
        "baseline": evaluate_config(kg, test, embedder, beta=0.0, kernel="entropy", k=args.k)
    }
    for kernel, config in selected_configs.items():
        test_records[kernel] = evaluate_config(
            kg,
            test,
            embedder,
            beta=config["beta"],
            kernel=kernel,
            k=args.k,
        )
    baseline_scores = scores(test_records["baseline"])
    baseline_ci = bootstrap_mean_ci(baseline_scores, args.bootstrap, args.seed)
    baseline_mean = float(baseline_scores.mean())
    print("\nHeld-out test results:")
    print(
        f"  vanilla beta=0:       {baseline_mean:.3f}  "
        f"CI95 [{baseline_ci[0]:.3f}, {baseline_ci[1]:.3f}]"
    )
    summaries = {}
    for kernel in ("entropy", "degree"):
        candidate_scores = scores(test_records[kernel])
        candidate_ci = bootstrap_mean_ci(candidate_scores, args.bootstrap, args.seed)
        delta = candidate_scores - baseline_scores
        delta_ci = paired_delta_ci(baseline_scores, candidate_scores, args.bootstrap, args.seed)
        if delta_ci[0] > 0:
            interpretation = "positive paired interval on this sampled corpus"
        elif delta_ci[1] < 0:
            interpretation = "negative paired interval on this sampled corpus"
        else:
            interpretation = "paired interval includes zero; result is inconclusive"
        summaries[kernel] = {
            "mean": float(candidate_scores.mean()),
            "ci95": list(candidate_ci),
            "delta_mean": float(delta.mean()),
            "delta_ci95": list(delta_ci),
            "wins": int(np.sum(delta > 0)),
            "losses": int(np.sum(delta < 0)),
            "ties": int(np.sum(delta == 0)),
            "interpretation": interpretation,
        }
        beta = selected_configs[kernel]["beta"]
        summary = summaries[kernel]
        print(
            f"  {kernel} beta={beta:g}: {summary['mean']:.3f}  "
            f"delta={summary['delta_mean']:+.3f} "
            f"delta CI95 [{summary['delta_ci95'][0]:+.3f}, "
            f"{summary['delta_ci95'][1]:+.3f}] "
            f"W/L/T={summary['wins']}/{summary['losses']}/{summary['ties']}"
        )
    print(f"  primary interpretation: {summaries['entropy']['interpretation']}")

    report = {
        "protocol": {
            "name": "MuSiQue transductive open-corpus variant",
            "extraction": kg.metadata,
            "standard_musique_task": False,
            "selection_split": "dev",
            "evaluation_split": "test",
            "k": args.k,
            "bootstrap_resamples": args.bootstrap,
            "random_seed": args.seed,
            "graph_sha256": file_sha256(graph_path),
            "questions_sha256": questions_sha256,
        },
        "sample_sizes": {"dev": len(dev), "test": len(test)},
        "supporting_source_coverage": coverage,
        "dev_sweep": dev_results,
        "selected_on_dev": selected_configs,
        "test": {
            "baseline": {
                "kernel": "entropy",
                "beta": 0.0,
                "mean": baseline_mean,
                "ci95": list(baseline_ci),
                "records": test_records["baseline"],
            },
            "entropy": {**summaries["entropy"], "records": test_records["entropy"]},
            "degree": {**summaries["degree"], "records": test_records["degree"]},
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS / f"musique_{args.provider}_{args.prompt}_eval.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
