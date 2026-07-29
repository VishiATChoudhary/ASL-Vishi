"""Evaluate damped PPR on the transductive open-corpus MuSiQue variant.

Configurations are selected on dev. The selected nonzero configuration is
then compared with vanilla PPR on held-out test questions using paired
bootstrap resampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from supernode_poc.embeddings import Embedder
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

    graph_path = ARTIFACTS / "musique_kg.json"
    if not graph_path.exists() or not QUESTIONS_PATH.exists():
        raise FileNotFoundError("run scripts/build_musique.py first")
    kg = KG.load(graph_path)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    dev = [question for question in questions if question.get("split") == "dev"]
    test = [question for question in questions if question.get("split") == "test"]
    if not dev or not test:
        raise ValueError("question metadata must contain non-empty dev and test splits")
    embedder = Embedder()

    print(f"dev={len(dev)} test={len(test)} k={args.k} (transductive open-corpus MuSiQue variant)")
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

    eligible = [result for result in dev_results if result["beta"] > 0]
    selected_config = sorted(
        eligible,
        key=lambda result: (-result["mean"], result["kernel"], result["beta"]),
    )[0]
    kernel = selected_config["kernel"]
    beta = selected_config["beta"]
    print(f"\nSelected on dev: kernel={kernel} beta={beta:g}")

    baseline_records = evaluate_config(kg, test, embedder, beta=0.0, kernel="entropy", k=args.k)
    selected_records = evaluate_config(kg, test, embedder, beta=beta, kernel=kernel, k=args.k)
    baseline_scores = scores(baseline_records)
    selected_scores = scores(selected_records)
    baseline_ci = bootstrap_mean_ci(baseline_scores, args.bootstrap, args.seed)
    selected_ci = bootstrap_mean_ci(selected_scores, args.bootstrap, args.seed)
    delta = selected_scores - baseline_scores
    delta_ci = paired_delta_ci(baseline_scores, selected_scores, args.bootstrap, args.seed)
    wins = int(np.sum(delta > 0))
    losses = int(np.sum(delta < 0))
    ties = int(np.sum(delta == 0))

    baseline_mean = float(baseline_scores.mean())
    selected_mean = float(selected_scores.mean())
    delta_mean = float(delta.mean())
    print("\nHeld-out test results:")
    print(
        f"  vanilla beta=0:       {baseline_mean:.3f}  "
        f"CI95 [{baseline_ci[0]:.3f}, {baseline_ci[1]:.3f}]"
    )
    print(
        f"  {kernel} beta={beta:g}:      {selected_mean:.3f}  "
        f"CI95 [{selected_ci[0]:.3f}, {selected_ci[1]:.3f}]"
    )
    print(
        f"  paired delta:         {delta_mean:+.3f}  CI95 [{delta_ci[0]:+.3f}, {delta_ci[1]:+.3f}]"
    )
    print(f"  per-question W/L/T:   {wins}/{losses}/{ties}")
    if delta_ci[0] > 0:
        interpretation = "positive paired interval on this sampled corpus"
    elif delta_ci[1] < 0:
        interpretation = "negative paired interval on this sampled corpus"
    else:
        interpretation = "paired interval includes zero; result is inconclusive"
    print(f"  interpretation:       {interpretation}")

    report = {
        "protocol": {
            "name": "MuSiQue transductive open-corpus variant",
            "standard_musique_task": False,
            "selection_split": "dev",
            "evaluation_split": "test",
            "k": args.k,
            "bootstrap_resamples": args.bootstrap,
            "random_seed": args.seed,
            "graph_sha256": file_sha256(graph_path),
            "questions_sha256": file_sha256(QUESTIONS_PATH),
        },
        "sample_sizes": {"dev": len(dev), "test": len(test)},
        "dev_sweep": dev_results,
        "selected": selected_config,
        "test": {
            "baseline": {
                "kernel": "entropy",
                "beta": 0.0,
                "mean": baseline_mean,
                "ci95": list(baseline_ci),
                "records": baseline_records,
            },
            "selected": {
                "kernel": kernel,
                "beta": beta,
                "mean": selected_mean,
                "ci95": list(selected_ci),
                "records": selected_records,
            },
            "paired_delta": {
                "mean": delta_mean,
                "ci95": list(delta_ci),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "interpretation": interpretation,
            },
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS / "musique_eval.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
