"""In-memory knowledge graph and supernode diagnostics."""

import json
import math
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx

from supernode_poc.models import Triple


class KG:
    """Directed property multigraph with source provenance on nodes."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self.node_sources: dict[str, set[str]] = {}
        self.node_labels: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def normalize(name: str) -> str:
        """Case-fold an entity and collapse all runs of whitespace."""
        return " ".join(name.casefold().split())

    def add_triples(self, triples: list[Triple], source_id: str) -> None:
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        for triple in triples:
            subject = self.normalize(triple.subject)
            object_ = self.normalize(triple.object)
            relation = self.normalize(triple.relation).replace(" ", "_")
            if not subject or not object_ or not relation:
                continue
            self.g.add_edge(
                subject,
                object_,
                relation=relation,
                source_id=source_id,
                weight=1.0,
            )
            self.node_sources.setdefault(subject, set()).add(source_id)
            self.node_sources.setdefault(object_, set()).add(source_id)
            self.node_labels.setdefault(subject, subject)
            self.node_labels.setdefault(object_, object_)

    def nodes(self) -> list[str]:
        return sorted(self.g.nodes())

    def edges(self) -> list[tuple[str, str, str]]:
        return sorted((u, v, data["relation"]) for u, v, data in self.g.edges(data=True))

    def edge_records(self) -> list[tuple[str, str, str, str | None, float]]:
        records = [
            (
                subject,
                object_,
                data["relation"],
                data.get("source_id"),
                float(data.get("weight", 1.0)),
            )
            for subject, object_, data in self.g.edges(data=True)
        ]
        return sorted(records, key=lambda row: (row[0], row[1], row[2], row[3] or "", row[4]))

    def label(self, node: str) -> str:
        return self.node_labels.get(node, node)

    def labels(self, nodes: Iterable[str]) -> list[str]:
        return [self.label(node) for node in nodes]

    def degree(self, node: str) -> int:
        return int(self.g.degree(node))

    def relation_entropy(self, node: str) -> float:
        relations = [data["relation"] for *_, data in self.g.in_edges(node, data=True)]
        relations.extend(data["relation"] for *_, data in self.g.out_edges(node, data=True))
        if not relations:
            return 0.0
        counts = Counter(relations)
        total = len(relations)
        return -sum((count / total) * math.log(count / total) for count in counts.values())

    def leakiness(self, node: str) -> float:
        return self.degree(node) * self.relation_entropy(node)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 2,
            "edges": [
                {
                    "subject": subject,
                    "object": object_,
                    "relation": relation,
                    "source_id": source_id,
                    "weight": weight,
                }
                for subject, object_, relation, source_id, weight in self.edge_records()
            ],
            "metadata": self.metadata,
            "node_labels": self.node_labels,
            "node_sources": {
                node: sorted(sources) for node, sources in sorted(self.node_sources.items())
            },
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "KG":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls()
        for edge in payload["edges"]:
            if isinstance(edge, dict):
                kg.g.add_edge(
                    edge["subject"],
                    edge["object"],
                    relation=edge["relation"],
                    source_id=edge.get("source_id"),
                    weight=float(edge.get("weight", 1.0)),
                )
            else:
                subject, object_, relation = edge
                kg.g.add_edge(subject, object_, relation=relation, weight=1.0)
        kg.node_sources = {node: set(sources) for node, sources in payload["node_sources"].items()}
        kg.node_labels = payload.get("node_labels", {node: node for node in kg.nodes()})
        kg.metadata = payload.get("metadata", {})
        return kg


def fragment_by_source(
    kg: KG,
    nodes_to_fragment: Iterable[str],
    *,
    shards: int = 2,
    seed: int = 0,
    identity_weight: float | None = None,
) -> tuple[KG, dict[str, list[str]]]:
    """Split each selected entity consistently by source document."""
    if shards < 2:
        raise ValueError("shards must be at least 2")
    if identity_weight is not None and (not math.isfinite(identity_weight) or identity_weight <= 0):
        raise ValueError("identity_weight must be finite and positive")

    selected = {
        node
        for node in nodes_to_fragment
        if node in kg.g and len(kg.node_sources.get(node, ())) >= shards
    }
    assignments: dict[str, dict[str, int]] = {}
    groups: dict[str, list[str]] = {}
    for node in sorted(selected):
        sources = sorted(kg.node_sources[node])
        random.Random(f"{seed}:{node}").shuffle(sources)
        assignments[node] = {source: index % shards for index, source in enumerate(sources)}
        fragments = [f"{node} ::fragment::{index}" for index in range(shards)]
        if any(fragment in kg.g for fragment in fragments):
            raise ValueError(f"fragment identifier collision for node: {node}")
        groups[node] = fragments

    fragmented = KG()
    for subject, object_, relation, source_id, weight in kg.edge_records():
        if source_id is None:
            raise ValueError("fragmentation requires source provenance on every input edge")
        mapped_subject = _fragment_id(subject, source_id, assignments, groups)
        mapped_object = _fragment_id(object_, source_id, assignments, groups)
        fragmented.g.add_edge(
            mapped_subject,
            mapped_object,
            relation=relation,
            source_id=source_id,
            weight=weight,
        )
        fragmented.node_sources.setdefault(mapped_subject, set()).add(source_id)
        fragmented.node_sources.setdefault(mapped_object, set()).add(source_id)
        fragmented.node_labels[mapped_subject] = kg.label(subject)
        fragmented.node_labels[mapped_object] = kg.label(object_)

    if identity_weight is not None:
        for fragments in groups.values():
            for fragment in fragments[1:]:
                fragmented.g.add_edge(
                    fragments[0],
                    fragment,
                    relation="same_as",
                    source_id=None,
                    weight=identity_weight,
                )
    fragmented.metadata = {
        **kg.metadata,
        "fragmentation": {
            "selected_nodes": len(groups),
            "shards": shards,
            "seed": seed,
            "identity_weight": identity_weight,
        },
    }
    return fragmented, groups


def _fragment_id(
    node: str,
    source_id: str,
    assignments: dict[str, dict[str, int]],
    groups: dict[str, list[str]],
) -> str:
    if node not in assignments:
        return node
    return groups[node][assignments[node][source_id]]
