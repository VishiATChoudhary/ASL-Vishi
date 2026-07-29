import json
import subprocess

import pytest

from supernode_poc.extraction import (
    PROMPTS,
    SYSTEM_NEUTRAL,
    CLIExtractor,
    ExtractionError,
    extract_corpus,
    extract_triples,
)
from supernode_poc.models import Triple


class FakeRunner:
    def __init__(self, *outputs: str, version: str = "fake 1.0", returncode: int = 0):
        self.outputs = iter(outputs)
        self.cli_version = version
        self.returncode = returncode
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, self.cli_version, "")
        output = next(self.outputs, "")
        return subprocess.CompletedProcess(command, self.returncode, output, "failure detail")

    @property
    def extraction_calls(self):
        return [call for call in self.calls if call[0][-1] != "--version"]


def triples_payload(triples: list[Triple]) -> dict:
    return {"triples": [triple.model_dump() for triple in triples]}


def claude_output(triples: list[Triple]) -> str:
    payload = triples_payload(triples)
    return json.dumps(
        {
            "is_error": False,
            "subtype": "success",
            "result": json.dumps(payload),
            "structured_output": payload,
        }
    )


@pytest.mark.parametrize(
    ("provider", "output"),
    [
        ("claude", claude_output([Triple(subject="alice", relation="works_at", object="acme")])),
        (
            "codex",
            json.dumps(
                triples_payload([Triple(subject="alice", relation="works_at", object="acme")])
            ),
        ),
    ],
)
def test_extract_triples_uses_each_cli_structured_output_contract(provider, output):
    expected = Triple(subject="alice", relation="works_at", object="acme")
    runner = FakeRunner(output)
    extractor = CLIExtractor(provider=provider, runner=runner)

    assert extract_triples(extractor, "Alice works at Acme.") == [expected]
    command, options = runner.extraction_calls[0]
    assert options["input"]
    assert options["capture_output"] is True
    assert options["cwd"].name.startswith("supernode-poc-")
    if provider == "claude":
        assert "--json-schema" in command
        assert command[command.index("--tools") + 1] == ""
        assert options["input"] == "Alice works at Acme."
    else:
        assert command[:2] == ["codex", "exec"]
        assert "--output-schema" in command
        assert "--ignore-user-config" in command
        assert "<input_text>" in options["input"]


def test_claude_accepts_result_fallback_when_structured_field_is_absent():
    payload = triples_payload([Triple(subject="a", relation="r", object="b")])
    output = json.dumps({"is_error": False, "subtype": "success", "result": json.dumps(payload)})
    extractor = CLIExtractor(provider="claude", runner=FakeRunner(output))

    assert extract_triples(extractor, "text") == [Triple(subject="a", relation="r", object="b")]


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("not json", "invalid JSON"),
        (json.dumps({"triples": [{"subject": "a"}]}), "does not match"),
    ],
)
def test_extract_triples_rejects_invalid_codex_results(output, message):
    extractor = CLIExtractor(provider="codex", runner=FakeRunner(output))
    with pytest.raises(ExtractionError, match=message):
        extract_triples(extractor, "text")


def test_extract_triples_reports_cli_failure():
    extractor = CLIExtractor(provider="codex", runner=FakeRunner(returncode=2))
    with pytest.raises(ExtractionError, match="status 2"):
        extract_triples(extractor, "text")


def test_extract_corpus_caches_complete_results_with_provenance(tmp_path):
    triple = Triple(subject="bob", relation="likes", object="coffee")
    runner = FakeRunner(claude_output([triple]))
    extractor = CLIExtractor(provider="claude", runner=runner)
    cache = tmp_path / "nested" / "cache.jsonl"

    first = extract_corpus(extractor, [("s1", "Bob likes coffee.")], cache)
    second = extract_corpus(extractor, [("s1", "Bob likes coffee.")], cache)

    assert first == second == {"s1": [triple]}
    assert len(runner.extraction_calls) == 1
    row = json.loads(cache.read_text(encoding="utf-8"))
    assert row["provider"] == "claude"
    assert row["model"] == "claude-opus-5"
    assert row["effort"] == "low"
    assert row["cli_version"] == "fake 1.0"
    assert len(row["text_sha256"]) == len(row["system_sha256"]) == 64
    assert len(row["schema_sha256"]) == 64


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("same text", "model-a", "low", "prompt-a"), ("changed", "model-a", "low", "prompt-a")),
        (("same text", "model-a", "low", "prompt-a"), ("same text", "model-b", "low", "prompt-a")),
        (("same text", "model-a", "low", "prompt-a"), ("same text", "model-a", "high", "prompt-a")),
        (("same text", "model-a", "low", "prompt-a"), ("same text", "model-a", "low", "prompt-b")),
    ],
)
def test_extract_corpus_invalidates_cache_when_inputs_change(tmp_path, first, second):
    triple = Triple(subject="a", relation="r", object="b")
    runner = FakeRunner(claude_output([triple]), claude_output([triple]))
    cache = tmp_path / "cache.jsonl"

    for text, model, effort, system in (first, second):
        extractor = CLIExtractor(provider="claude", model=model, effort=effort, runner=runner)
        extract_corpus(extractor, [("s1", text)], cache, system=system)

    assert len(runner.extraction_calls) == 2
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2


def test_extract_corpus_does_not_cache_failed_results(tmp_path):
    cache = tmp_path / "cache.jsonl"
    extractor = CLIExtractor(provider="codex", runner=FakeRunner("not json"))

    with pytest.raises(ExtractionError):
        extract_corpus(extractor, [("s1", "text")], cache)

    assert cache.read_text(encoding="utf-8") == ""


def test_defaults_and_prompt_registry():
    assert CLIExtractor(provider="claude").model == "claude-opus-5"
    assert CLIExtractor(provider="codex").model == "gpt-5.6-sol"
    assert PROMPTS["neutral"] == SYSTEM_NEUTRAL
    assert "specific entities" in PROMPTS["specific"]
