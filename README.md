# Soft Identity Constraints for LLM-Written Knowledge Graphs

This repository contains the controlled proof of concept for
[Soft Identity Constraints for LLM-Written Knowledge Graphs](research/identity-integrity-proposal.pdf).
It tests one causal mechanism: splitting a shared entity disconnects evidence
across source documents, while a reversible `same_as` link can restore most of
that retrieval path.

The experiment does **not** estimate how often natural incremental ingestion
creates duplicate entities. It injects the fault and uses oracle-correct links
to establish the mechanism before evaluating an automatic resolver.

## Method

Claude and Codex independently extract triples from the same 1,048 paragraphs
in a fixed 60-question MuSiQue sample. Every graph edge retains its source
paragraph.

For each eligible question, the evaluation:

1. Finds an entity shared by at least two required passages.
2. Splits that entity into two source-consistent graph nodes without changing
   its triples, labels, or provenance.
3. Seeds Personalized PageRank on one fragment and measures whether the other
   required passage appears in the top five results.
4. Adds an oracle-correct `same_as` edge and reserves a bounded fraction of the
   graph walk for that identity transition.
5. Selects the identity mixture on development questions and evaluates it once
   on held-out test questions, using paired bootstrap confidence intervals.

The retrieval operator is:

```text
P(v, ·) = (1 - λcᵥ) P_graph(v, ·) + λcᵥ P_identity(v, ·)
```

Here, `λ` bounds identity influence and `cᵥ` represents link confidence. The
current controlled experiment uses known-correct identity links.

## Results

| Extractor | Eligible test questions | Original | Fragmented | Soft repair | Loss recovered |
|---|---:|---:|---:|---:|---:|
| Claude Opus 5 | 24 | 75.0% | 15.0% | 64.2% | 81.9% |
| Codex GPT-5.6-sol | 22 | 77.3% | 10.0% | 71.8% | 91.9% |

For both independently extracted graphs, the paired 95% intervals show a loss
after fragmentation and a gain after repair. These results verify the narrow
causal mechanism; they do not yet validate automatic entity resolution or the
prevalence of the fault in production graphs.

Full machine-readable results are in
[`artifacts/fragmentation_poc.json`](artifacts/fragmentation_poc.json), with a
summary plot in
[`artifacts/fragmentation_dose_response.png`](artifacts/fragmentation_dose_response.png).

## Reproduce

Python 3.11 or newer plus authenticated Claude and Codex CLIs are required for
fresh extraction. Embeddings are not used by the focused bridge probe.

```bash
uv sync
uv run pytest

# Prepare the fixed MuSiQue sample without model calls.
uv run python scripts/build_musique.py --n-questions 60 --prepare-only

# Build one graph per extraction provider. Cached rows are reused.
uv run python scripts/build_musique.py --n-questions 60 --provider claude --prompt neutral
uv run python scripts/build_musique.py --n-questions 60 --provider codex --prompt neutral

# Run both held-out bridge experiments and regenerate the tracked results.
uv run python scripts/run_fragmentation_poc.py
```

Extraction caches under `data/cache/` are intentionally ignored. Cache keys
include the provider, exact model, reasoning effort, CLI version, prompt hash,
input hash, and validation-schema hash.

## Important files

- `src/identity_integrity/graph.py` preserves provenance and implements
  source-consistent fragmentation plus reversible identity edges.
- `src/identity_integrity/retrieval.py` implements bounded identity transitions,
  Personalized PageRank, and source scoring.
- `scripts/run_fragmentation_poc.py` defines bridge selection, development-only
  tuning, held-out evaluation, bootstrap intervals, and result generation.
- `scripts/build_musique.py` prepares the reproducible corpus and runs validated
  triple extraction through isolated model CLI processes.
