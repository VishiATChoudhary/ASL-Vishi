"""Compare vanilla and damped PPR for one LoCoMo query."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from supernode_poc.embeddings import Embedder
from supernode_poc.graph import KG
from supernode_poc.retrieval import retrieve

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="What does the speaker do for work?",
    )
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--kernel", choices=["entropy", "degree"], default="entropy")
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1 or any(beta < 0 for beta in args.betas):
        raise ValueError("--k must be positive and --betas must be non-negative")

    graph_path = ROOT / "artifacts/locomo_kg.json"
    sources_path = ROOT / "data/cache/locomo_sources.json"
    if not graph_path.exists() or not sources_path.exists():
        raise FileNotFoundError("run scripts/ingest_locomo.py first")
    kg = KG.load(graph_path)
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    embedder = Embedder()

    print(f"Question: {args.question}")
    for beta in args.betas:
        print(f"\n=== kernel={args.kernel} beta={beta:g} ===")
        retrieved = retrieve(
            kg,
            args.question,
            embedder,
            beta=beta,
            kernel=args.kernel,
            k=args.k,
        )
        if not retrieved:
            print("No source was retrieved.")
            continue
        for rank, source_id in enumerate(retrieved, 1):
            snippet = textwrap.shorten(
                sources.get(source_id, "<source text unavailable>").replace("\n", " "),
                width=180,
                placeholder="...",
            )
            print(f"{rank}. [{source_id}] {snippet}")


if __name__ == "__main__":
    main()
