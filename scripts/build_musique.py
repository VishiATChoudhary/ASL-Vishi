"""Build a MuSiQue KG through Claude or Codex CLI extraction.

All paragraphs from sampled questions share one deduplicated corpus. This is
not the standard per-question candidate-ranking task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset

from identity_integrity.extraction import (
    DEFAULT_MODELS,
    EFFORTS,
    PROMPTS,
    CLIExtractor,
    extract_corpus,
)
from identity_integrity.graph import KG

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
ARTIFACTS = ROOT / "artifacts"


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def canonical_id(text: str) -> str:
    """Return a stable content ID after whitespace and case normalization."""
    digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]
    return f"p{digest}"


def paragraph_rows(value: Any) -> Iterable[Mapping[str, Any]]:
    """Support both row-oriented and HF column-oriented paragraph records."""
    if isinstance(value, list):
        for paragraph in value:
            if not isinstance(paragraph, Mapping):
                raise ValueError("paragraph list contains a non-object value")
            yield paragraph
        return
    if isinstance(value, Mapping):
        columns = dict(value)
        if not columns:
            return
        if not all(isinstance(column, list) for column in columns.values()):
            raise ValueError("paragraph columns must be lists")
        lengths = {len(column) for column in columns.values()}
        if len(lengths) != 1:
            raise ValueError("paragraph columns must be equally sized lists")
        for index in range(lengths.pop()):
            yield {key: column[index] for key, column in columns.items()}
        return
    raise ValueError(f"unexpected paragraphs representation: {type(value).__name__}")


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def select_rows(dataset: Any, count: int, seed: int) -> list[Mapping[str, Any]]:
    """Select a reproducible, order-independent subset, nested as count grows."""
    ranked: list[tuple[str, int]] = []
    for index in range(len(dataset)):
        row_id = str(dataset[index].get("id", index))
        key = hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()
        ranked.append((key, index))
    indices = [index for _, index in sorted(ranked)[:count]]
    return [dataset[index] for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-questions", type=int, default=60)
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="claude")
    parser.add_argument("--model", help="exact model ID; defaults to the provider's pinned model")
    parser.add_argument("--effort", choices=sorted(EFFORTS), default="low")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="neutral")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write question metadata and report corpus size without invoking a model CLI",
    )
    return parser.parse_args()


def report_progress(completed: int, total: int) -> None:
    step = max(1, total // 20)
    if completed == 0 or completed == total or completed % step == 0:
        print(f"Extracted or loaded {completed}/{total}", flush=True)


def main() -> None:
    args = parse_args()
    if args.n_questions < 2 or args.workers < 1:
        raise ValueError("--n-questions must be at least 2 and --workers must be positive")

    dataset = load_dataset("dgslibisey/MuSiQue", split="validation")
    if args.n_questions > len(dataset):
        raise ValueError(
            f"requested {args.n_questions} questions, but validation has {len(dataset)}"
        )
    rows = select_rows(dataset, args.n_questions, args.seed)

    corpus: dict[str, str] = {}
    questions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id", index))
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {row_id} has no text")
        supporting: set[str] = set()
        for paragraph in paragraph_rows(row.get("paragraphs")):
            title = str(paragraph.get("title") or "").strip()
            body = str(paragraph.get("paragraph_text") or "").strip()
            text = f"{title}. {body}" if title else body
            if not text:
                continue
            source_id = canonical_id(text)
            previous = corpus.setdefault(source_id, text)
            if normalize_text(previous) != normalize_text(text):
                raise RuntimeError(f"content hash collision for {source_id}")
            if as_bool(paragraph.get("is_supporting", False)):
                supporting.add(source_id)
        if not supporting:
            raise ValueError(f"question {row_id} has no supporting paragraphs")
        questions.append(
            {
                "id": row_id,
                "question": question,
                "supporting": sorted(supporting),
                "split": "dev" if index % 2 == 0 else "test",
            }
        )

    items = sorted(corpus.items())
    CACHE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    questions_path = CACHE / "musique_questions.json"
    questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Prepared {len(items)} unique paragraphs and {len(questions)} questions "
        f"({sum(q['split'] == 'dev' for q in questions)} dev, "
        f"{sum(q['split'] == 'test' for q in questions)} test)"
    )
    print("Protocol: transductive open-corpus MuSiQue variant")
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
        items,
        cache_path=CACHE / f"musique_triples_{args.provider}_{args.prompt}.jsonl",
        system=PROMPTS[args.prompt],
        workers=args.workers,
        progress=report_progress,
    )
    kg = KG()
    for source_id, triples in triples_by_source.items():
        kg.add_triples(triples, source_id)
    if not kg.nodes():
        raise ValueError("triple extraction produced an empty graph")
    kg.metadata = {
        "corpus": "MuSiQue transductive open-corpus variant",
        "question_count": len(questions),
        "seed": args.seed,
        "prompt": args.prompt,
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        **extractor.provenance(PROMPTS[args.prompt]),
    }
    graph_path = ARTIFACTS / f"musique_{args.provider}_{args.prompt}_kg.json"
    kg.save(graph_path)
    print(f"Graph: {len(kg.nodes())} nodes, {len(kg.edges())} edges")
    print(f"Wrote {graph_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
