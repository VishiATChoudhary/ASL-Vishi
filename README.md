# Supernode PoC

This lab tests whether generic hubs in LLM-extracted knowledge graphs attract
query-independent Personalized PageRank mass, and whether damping transitions
into high-degree, high-relation-entropy nodes improves source retrieval.

The damped kernel weights entry into node `v` by:

```text
(1 + degree(v) * relation_entropy(v)) ** -beta
```

This is a hypothesis test, not a claim that every high-leakiness node is bad.
Damping can reroute a walk when another path exists. It cannot remove a false
connection when the hub is the only path. See
[`research-proposal-supernodes.md`](research-proposal-supernodes.md) for the
research motivation and structural follow-up ideas.

## Setup

Python 3.11 or newer and an Anthropic API key are required for extraction.
Embeddings run locally.

```bash
uv sync
export ANTHROPIC_API_KEY=...
uv run pytest
```

An authenticated `ant auth login` profile is also supported by the pinned SDK.

Extraction is cached under `data/cache/`. Repeating the same build reuses the
cache and makes no extraction calls for cached inputs. Generated reports and
visuals go to `artifacts/`.

## 1. Inspect the LoCoMo graph

```bash
uv run python scripts/ingest_locomo.py --prepare-only
uv run python scripts/ingest_locomo.py --prompt neutral
open artifacts/locomo_graph.html
uv run python scripts/demo_query.py "What does the speaker do for work?"
```

The graph highlights the top 1 percent of nodes by candidate leakiness. Review
the printed top ten manually and classify each as a generic hub or a legitimate
rich entity before interpreting the metric. The live query is illustrative,
not evaluation evidence, and should not be selected after the fact solely for
a favorable result.

## 2. Measure retrieval bias

```bash
uv run python scripts/run_diagnostics.py
```

This writes:

- `artifacts/degree_distribution.png`
- `artifacts/bias_curve.png`
- `artifacts/diagnostic_metrics.json`

The controls use off-domain questions and uniformly random graph seeds. The
random-seed control bypasses the embedder, which helps separate graph bias from
embedding behavior. The default eight questions per condition make this an
exploratory diagnostic, not proof of query independence.

## 3. Evaluate retrieval on MuSiQue

Check corpus size without making Claude calls:

```bash
uv run python scripts/build_musique.py --n-questions 20 --prepare-only
```

Then build and evaluate. Start small to estimate extraction cost before using
the default 60-question sample.

```bash
uv run python scripts/build_musique.py --n-questions 60 --prompt neutral
uv run python scripts/run_eval.py --betas 0 0.5 1 2 --k 5
```

This is a transductive, open-corpus MuSiQue variant. Paragraphs from all sampled
questions are deduplicated by normalized content and merged into one graph. It
is not the standard per-question, 20-candidate MuSiQue ranking task.

Entropy and degree kernels are swept on the dev split. The best nonzero dev
configuration is compared once with vanilla PPR on held-out test questions.
The report includes a paired bootstrap confidence interval, per-question
win/loss/tie counts, exact retrieved IDs, and input hashes:

```text
artifacts/musique_eval.json
```

If the paired interval includes zero, report the result as inconclusive. A flat
or negative result is also a valid finding.

## Ablations

- `--prompt neutral` is the main condition. `--prompt specific` tests whether
  extraction instructions that discourage generic entities suppress hubs.
- `run_eval.py` compares entropy-weighted damping with degree-only damping.
- `beta=0` is vanilla PPR for either kernel.

Use separate, predeclared runs when comparing prompt conditions. Do not widen
the beta grid after looking at held-out test results.
