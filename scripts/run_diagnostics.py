"""Generate cautious graph-bias diagnostics from the LoCoMo KG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from supernode_poc.diagnostics import (
    degree_distribution,
    random_seed_frequency,
    retrieval_frequency,
)
from supernode_poc.embeddings import Embedder
from supernode_poc.extraction import DEFAULT_MODELS, PROMPTS
from supernode_poc.graph import KG

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

REAL_QUESTIONS = [
    "What does the speaker do for work?",
    "Where does the speaker live?",
    "What hobbies were discussed?",
    "What health issues came up?",
    "What travel plans were mentioned?",
    "Who are the speaker's family members?",
    "What food does the speaker like?",
    "What happened at the last event they attended?",
]

OFF_DOMAIN_QUESTIONS = [
    "How do volcanoes form?",
    "What is the boiling point of nitrogen?",
    "Explain the rules of chess castling.",
    "Which planet has the strongest winds?",
    "How is porcelain manufactured?",
    "What causes tides in the ocean?",
    "Describe photosynthesis in algae.",
    "When was the printing press invented?",
]


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return Spearman rho and p, or NaNs when either input is constant."""
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-nodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="claude")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="neutral")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1 or args.top_nodes < 1:
        raise ValueError("--k and --top-nodes must be positive")
    run_name = f"locomo_{args.provider}_{args.prompt}"
    graph_path = ARTIFACTS / f"{run_name}_kg.json"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"run scripts/ingest_locomo.py --provider {args.provider} --prompt {args.prompt} first"
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    kg = KG.load(graph_path)
    if not kg.nodes():
        raise ValueError("LoCoMo KG is empty")
    embedder = Embedder()

    degrees, counts = degree_distribution(kg)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(degrees, counts, color="#1f77b4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node count")
    ax.set_title("LoCoMo KG degree distribution")
    ax.text(
        0.02,
        0.03,
        "A heavy tail motivates inspection; it does not by itself identify bad hubs.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    degree_path = ARTIFACTS / f"{run_name}_degree_distribution.png"
    fig.savefig(degree_path, dpi=150)
    plt.close(fig)

    freq_real = retrieval_frequency(kg, REAL_QUESTIONS, embedder, beta=0.0, k=args.k)
    freq_off = retrieval_frequency(kg, OFF_DOMAIN_QUESTIONS, embedder, beta=0.0, k=args.k)
    freq_random = random_seed_frequency(
        kg, n_queries=len(REAL_QUESTIONS), k=args.k, rng_seed=args.seed
    )
    candidates = set(freq_real) | set(freq_off) | set(freq_random)
    common = sorted(candidates, key=lambda node: (-kg.degree(node), node))[: args.top_nodes]
    if not common:
        raise ValueError("diagnostic retrieval returned no nodes")

    real = np.array([freq_real.get(node, 0) for node in common])
    off = np.array([freq_off.get(node, 0) for node in common])
    random = np.array([freq_random.get(node, 0) for node in common])
    rho_off, p_off = correlation(real, off)
    rho_random, p_random = correlation(real, random)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(real, off, label="Off-domain questions", marker="o", alpha=0.8)
    ax.scatter(real, random, label="Random graph seeds", marker="x", alpha=0.8)
    limit = int(max(real.max(), off.max(), random.max())) + 1
    ax.plot([0, limit], [0, limit], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(-0.2, limit)
    ax.set_ylim(-0.2, limit)
    ax.set_xlabel("Retrieval count for in-domain questions")
    ax.set_ylabel("Retrieval count for control")
    ax.set_title("Node retrieval frequency across query controls")
    ax.legend()
    for node in sorted(common, key=lambda value: (-kg.leakiness(value), value))[:5]:
        ax.annotate(node, (freq_real.get(node, 0), freq_random.get(node, 0)), fontsize=7)
    fig.tight_layout()
    bias_path = ARTIFACTS / f"{run_name}_bias_curve.png"
    fig.savefig(bias_path, dpi=150)
    plt.close(fig)

    metrics = {
        "protocol": {
            "extraction": kg.metadata,
            "k": args.k,
            "top_nodes": len(common),
            "questions_per_condition": len(REAL_QUESTIONS),
            "random_seed": args.seed,
        },
        "spearman": {
            "in_domain_vs_off_domain": {
                "rho": rho_off if np.isfinite(rho_off) else None,
                "p": p_off if np.isfinite(p_off) else None,
            },
            "in_domain_vs_random_seed": {
                "rho": rho_random if np.isfinite(rho_random) else None,
                "p": p_random if np.isfinite(p_random) else None,
            },
        },
        "nodes": [
            {
                "node": node,
                "degree": kg.degree(node),
                "leakiness": kg.leakiness(node),
                "in_domain": freq_real.get(node, 0),
                "off_domain": freq_off.get(node, 0),
                "random_seed": freq_random.get(node, 0),
            }
            for node in common
        ],
    }
    metrics_path = ARTIFACTS / f"{run_name}_diagnostic_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Spearman(in-domain, off-domain):  rho={rho_off:.2f} p={p_off:.3g}")
    print(f"Spearman(in-domain, random-seed): rho={rho_random:.2f} p={p_random:.3g}")
    if np.isfinite(rho_random) and rho_random >= 0.5:
        print("The random-seed agreement is consistent with graph-level retrieval bias.")
    else:
        print("The random-seed control does not show strong graph-level retrieval bias.")
    print("With eight questions per condition, treat these diagnostics as exploratory.")
    print(
        "Wrote "
        f"{degree_path.relative_to(ROOT)}, {bias_path.relative_to(ROOT)}, and "
        f"{metrics_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
