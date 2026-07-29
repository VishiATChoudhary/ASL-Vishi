import hashlib
import json
from pathlib import Path
from typing import Any

from supernode_poc.models import Triple, TripleList

SYSTEM_NEUTRAL = (
    "You extract knowledge-graph triples from text. "
    "Extract every distinct fact as one (subject, relation, object) triple. "
    "Use lowercase entity names and snake_case relation names."
)

SYSTEM_SPECIFIC = SYSTEM_NEUTRAL + (
    " Prefer specific entities over generic ones: 'alice' not 'the user', "
    "'acme quarterly review' not 'meeting'."
)

PROMPTS = {"neutral": SYSTEM_NEUTRAL, "specific": SYSTEM_SPECIFIC}


class ExtractionError(RuntimeError):
    """Raised when Claude does not return a complete structured extraction."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_triples(
    client: Any,
    text: str,
    model: str = "claude-opus-5",
    system: str = SYSTEM_NEUTRAL,
) -> list[Triple]:
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": text}],
        output_format=TripleList,
    )
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        raise ExtractionError("Claude refused the triple-extraction request")
    if stop_reason == "max_tokens":
        raise ExtractionError("Claude reached max_tokens before completing extraction")
    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        raise ExtractionError("Claude returned no parsed triple extraction")
    return parsed.triples


def extract_corpus(
    client: Any,
    items: list[tuple[str, str]],
    cache_path: str | Path,
    model: str = "claude-opus-5",
    system: str = SYSTEM_NEUTRAL,
) -> dict[str, list[Triple]]:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    system_sha256 = _sha256(system)
    cached: dict[tuple[str, str, str, str], list[Triple]] = {}

    if path.exists():
        with path.open(encoding="utf-8") as cache_file:
            for line_number, line in enumerate(cache_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    key = (
                        row["source_id"],
                        row["text_sha256"],
                        row["model"],
                        row["system_sha256"],
                    )
                    cached[key] = [Triple.model_validate(triple) for triple in row["triples"]]
                except (KeyError, TypeError, ValueError) as error:
                    message = f"Invalid extraction cache row {line_number} in {path}"
                    raise ValueError(message) from error

    results: dict[str, list[Triple]] = {}
    with path.open("a", encoding="utf-8") as cache_file:
        for source_id, text in items:
            text_sha256 = _sha256(text)
            key = (source_id, text_sha256, model, system_sha256)
            triples = cached.get(key)
            if triples is None:
                triples = extract_triples(client, text, model=model, system=system)
                row = {
                    "source_id": source_id,
                    "text_sha256": text_sha256,
                    "model": model,
                    "system_sha256": system_sha256,
                    "triples": [triple.model_dump() for triple in triples],
                }
                cache_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache_file.flush()
                cached[key] = triples
            results[source_id] = triples
    return results
