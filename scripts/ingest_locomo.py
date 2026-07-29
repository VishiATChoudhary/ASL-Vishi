"""Build and visualize a LoCoMo KG through Claude or Codex CLI extraction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import requests
from pyvis.network import Network

from supernode_poc.extraction import DEFAULT_MODELS, EFFORTS, PROMPTS, CLIExtractor, extract_corpus
from supernode_poc.graph import KG

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/locomo10.json"
CACHE = ROOT / "data/cache"
ARTIFACTS = ROOT / "artifacts"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def download() -> list[dict[str, Any]]:
    """Return the local LoCoMo data, downloading it once if needed."""
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        response = requests.get(LOCOMO_URL, timeout=60)
        response.raise_for_status()
        RAW.write_bytes(response.content)
    payload = json.loads(RAW.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"expected a non-empty JSON list in {RAW}")
    return payload


def conversation_chunks(
    sample: dict[str, Any], max_chunks: int, turns_per_chunk: int = 6
) -> list[tuple[str, str]]:
    """Flatten one conversation into stable ``(source_id, text)`` chunks."""
    conversation = sample.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo sample has no conversation object")

    def session_number(key: str) -> int:
        try:
            return int(key.removeprefix("session_").split("_")[0])
        except ValueError:
            return 10**9

    session_keys = sorted(
        (
            key
            for key, turns in conversation.items()
            if key.startswith("session_") and isinstance(turns, list)
        ),
        key=lambda key: (session_number(key), key),
    )
    chunks: list[tuple[str, str]] = []
    for session in session_keys:
        turns = conversation[session]
        for offset in range(0, len(turns), turns_per_chunk):
            window = turns[offset : offset + turns_per_chunk]
            lines = []
            for turn in window:
                if not isinstance(turn, dict) or "speaker" not in turn or "text" not in turn:
                    raise ValueError(f"unexpected turn schema in {session}: {turn!r}")
                lines.append(f"{turn['speaker']}: {turn['text']}")
            if lines:
                chunks.append((f"{session}-{offset:04d}", "\n".join(lines)))
    return chunks[:max_chunks]


def write_visualization(kg: KG, path: Path) -> None:
    """Write a self-contained graph view with the top 1 percent highlighted."""
    nodes = sorted(kg.nodes())
    top_count = max(1, math.ceil(len(nodes) * 0.01)) if nodes else 0
    highlighted = set(sorted(nodes, key=lambda node: (-kg.leakiness(node), node))[:top_count])
    net = Network(
        height="900px",
        width="100%",
        directed=False,
        cdn_resources="in_line",
    )
    for node in nodes:
        net.add_node(
            node,
            label=node,
            size=6 + 2 * math.sqrt(kg.degree(node)),
            color="#d62728" if node in highlighted else "#1f77b4",
            title=(
                f"degree={kg.degree(node)}<br>"
                f"entropy={kg.relation_entropy(node):.3f}<br>"
                f"leakiness={kg.leakiness(node):.3f}"
            ),
        )
    for source, target, relation in sorted(kg.edges()):
        net.add_edge(source, target, title=relation)
    path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(path), open_browser=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="claude")
    parser.add_argument("--model", help="exact model ID; defaults to the provider's pinned model")
    parser.add_argument("--effort", choices=sorted(EFFORTS), default="low")
    parser.add_argument("--max-chunks", type=int, default=40)
    parser.add_argument("--turns-per-chunk", type=int, default=6)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="neutral")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="download, validate, and chunk LoCoMo without invoking a model CLI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_chunks < 1 or args.turns_per_chunk < 1:
        raise ValueError("--max-chunks and --turns-per-chunk must be positive")

    data = download()
    if not 0 <= args.sample_index < len(data):
        raise IndexError(f"--sample-index must be between 0 and {len(data) - 1}")
    chunks = conversation_chunks(data[args.sample_index], args.max_chunks, args.turns_per_chunk)
    chunks = [
        (f"sample-{args.sample_index}-c{args.turns_per_chunk:03d}-{source_id}", text)
        for source_id, text in chunks
    ]
    if not chunks:
        raise ValueError("selected conversation produced no chunks")

    CACHE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sources_path = CACHE / "locomo_sources.json"
    sources_path.write_text(
        json.dumps(dict(chunks), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(chunks)} chunks with provider={args.provider} prompt={args.prompt}")
    if args.prepare_only:
        print("No model CLI calls were made.")
        return

    extractor = CLIExtractor(
        provider=args.provider,
        model=args.model,
        effort=args.effort,
    )
    print(
        f"Starting extraction with {extractor.provider} model={extractor.model} "
        f"effort={extractor.effort}."
    )
    triples_by_source = extract_corpus(
        extractor,
        chunks,
        cache_path=CACHE / f"locomo_triples_{args.provider}_{args.prompt}.jsonl",
        system=PROMPTS[args.prompt],
    )
    kg = KG()
    for source_id, triples in triples_by_source.items():
        kg.add_triples(triples, source_id)
    if not kg.nodes():
        raise ValueError("triple extraction produced an empty graph")
    kg.metadata = {
        "corpus": "LoCoMo",
        "sample_index": args.sample_index,
        "turns_per_chunk": args.turns_per_chunk,
        "chunk_count": len(chunks),
        "prompt": args.prompt,
        **extractor.provenance(PROMPTS[args.prompt]),
    }

    run_name = f"locomo_{args.provider}_{args.prompt}"
    graph_path = ARTIFACTS / f"{run_name}_kg.json"
    html_path = ARTIFACTS / f"{run_name}_graph.html"
    kg.save(graph_path)
    write_visualization(kg, html_path)

    ranked = sorted(kg.nodes(), key=lambda node: (-kg.leakiness(node), node))[:10]
    print(f"Graph: {len(kg.nodes())} nodes, {len(kg.edges())} edges")
    print("Top leakiness candidates. Inspect each before calling it a junk hub:")
    for node in ranked:
        print(
            f"  {node[:40]:40s} degree={kg.degree(node):4d} "
            f"H={kg.relation_entropy(node):.2f} leak={kg.leakiness(node):.1f}"
        )
    print(f"Wrote {graph_path.relative_to(ROOT)} and {html_path.relative_to(ROOT)}")
    print("Record whether each top candidate is generic or a legitimate rich entity.")


if __name__ == "__main__":
    main()
