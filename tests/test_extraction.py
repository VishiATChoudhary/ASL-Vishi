import json

import pytest
from supernode_poc.models import Triple, TripleList

from supernode_poc.extraction import (
    PROMPTS,
    SYSTEM_NEUTRAL,
    ExtractionError,
    extract_corpus,
    extract_triples,
)


class FakeResponse:
    def __init__(self, triples=(), *, stop_reason="end_turn", parsed=True):
        self.parsed_output = TripleList(triples=list(triples)) if parsed else None
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FakeClient:
    def __init__(self, *responses):
        self.messages = FakeMessages(responses)


def test_extract_triples_uses_structured_parse_contract():
    triple = Triple(subject="Alice", relation="works_at", object="Acme")
    client = FakeClient(FakeResponse([triple]))

    assert extract_triples(client, "Alice works at Acme.") == [triple]
    assert client.messages.calls == [
        {
            "model": "claude-opus-5",
            "max_tokens": 16000,
            "system": SYSTEM_NEUTRAL,
            "messages": [{"role": "user", "content": "Alice works at Acme."}],
            "output_format": TripleList,
        }
    ]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (FakeResponse(stop_reason="refusal"), "refused"),
        (FakeResponse(stop_reason="max_tokens"), "max_tokens"),
        (FakeResponse(parsed=False), "no parsed"),
    ],
)
def test_extract_triples_rejects_incomplete_results(response, message):
    with pytest.raises(ExtractionError, match=message):
        extract_triples(FakeClient(response), "text")


def test_extract_corpus_caches_complete_results_with_provenance(tmp_path):
    triple = Triple(subject="Bob", relation="likes", object="coffee")
    client = FakeClient(FakeResponse([triple]))
    cache = tmp_path / "nested" / "cache.jsonl"

    first = extract_corpus(client, [("s1", "Bob likes coffee.")], cache)
    second = extract_corpus(client, [("s1", "Bob likes coffee.")], cache)

    assert first == second == {"s1": [triple]}
    assert len(client.messages.calls) == 1
    row = json.loads(cache.read_text(encoding="utf-8"))
    assert set(row) == {"source_id", "text_sha256", "model", "system_sha256", "triples"}
    assert len(row["text_sha256"]) == len(row["system_sha256"]) == 64


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("same text", "model-a", "prompt-a"), ("changed text", "model-a", "prompt-a")),
        (("same text", "model-a", "prompt-a"), ("same text", "model-b", "prompt-a")),
        (("same text", "model-a", "prompt-a"), ("same text", "model-a", "prompt-b")),
    ],
)
def test_extract_corpus_invalidates_cache_when_inputs_change(tmp_path, first, second):
    triple = Triple(subject="a", relation="r", object="b")
    client = FakeClient(FakeResponse([triple]), FakeResponse([triple]))
    cache = tmp_path / "cache.jsonl"

    extract_corpus(client, [("s1", first[0])], cache, model=first[1], system=first[2])
    extract_corpus(client, [("s1", second[0])], cache, model=second[1], system=second[2])

    assert len(client.messages.calls) == 2
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(stop_reason="refusal"),
        FakeResponse(stop_reason="max_tokens"),
        FakeResponse(parsed=False),
    ],
)
def test_extract_corpus_does_not_cache_incomplete_results(tmp_path, response):
    cache = tmp_path / "cache.jsonl"

    with pytest.raises(ExtractionError):
        extract_corpus(FakeClient(response), [("s1", "text")], cache)

    assert cache.read_text(encoding="utf-8") == ""


def test_prompt_registry_keeps_neutral_default_and_specific_ablation():
    assert PROMPTS["neutral"] == SYSTEM_NEUTRAL
    assert "specific entities" in PROMPTS["specific"]
