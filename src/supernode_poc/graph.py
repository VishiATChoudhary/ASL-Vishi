"""In-memory knowledge graph and supernode diagnostics."""

import json
import math
from collections import Counter
from pathlib import Path

import networkx as nx

from supernode_poc.models import Triple


class KG:
    """Directed property multigraph with source provenance on nodes."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self.node_sources: dict[str, set[str]] = {}

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
            self.g.add_edge(subject, object_, relation=relation)
            self.node_sources.setdefault(subject, set()).add(source_id)
            self.node_sources.setdefault(object_, set()).add(source_id)

    def nodes(self) -> list[str]:
        return sorted(self.g.nodes())

    def edges(self) -> list[tuple[str, str, str]]:
        return sorted((u, v, data["relation"]) for u, v, data in self.g.edges(data=True))

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
            "edges": self.edges(),
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
        for subject, object_, relation in payload["edges"]:
            kg.g.add_edge(subject, object_, relation=relation)
        kg.node_sources = {node: set(sources) for node, sources in payload["node_sources"].items()}
        return kg
