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
[`research/research-proposal-supernodes.md`](research/research-proposal-supernodes.md) (local, untracked) for the
research motivation and structural follow-up ideas.

## Setup

Python 3.11 or newer plus authenticated Codex and Claude CLIs are required for
the two extraction conditions. Embeddings run locally.

```bash
uv sync
codex login status
claude auth status
uv run pytest
```

The default pinned extraction models are `gpt-5.6-sol` for Codex and
`claude-opus-5` for Claude. Override either with `--model`. Both default to low
reasoning effort because extraction is a constrained structured task. Override
that with `--effort` and treat it as a separate experimental condition.

Extraction is cached under `data/cache/`. Repeating the same build reuses the
cache and makes no model calls for cached inputs. Provider, exact model, effort,
CLI version, prompt hash, input hash, and schema hash are part of the cache
provenance. Generated reports and visuals go to `artifacts/`.

Claude runs in safe mode with tools disabled and no session persistence. Codex
runs ephemerally in read-only mode with user configuration and repository rules
disabled. Each process starts in an empty temporary directory, and every result
is validated against the same Pydantic JSON schema.

## 1. Inspect the LoCoMo graph

```bash
uv run python scripts/ingest_locomo.py --prepare-only
uv run python scripts/ingest_locomo.py --provider claude --prompt neutral
uv run python scripts/ingest_locomo.py --provider codex --prompt neutral
open artifacts/locomo_claude_neutral_graph.html
open artifacts/locomo_codex_neutral_graph.html
uv run python scripts/demo_query.py --provider claude "What does the speaker do for work?"
uv run python scripts/demo_query.py --provider codex "What does the speaker do for work?"
```

The graph highlights the top 1 percent of nodes by candidate leakiness. Review
the printed top ten manually and classify each as a generic hub or a legitimate
rich entity before interpreting the metric. The live query is illustrative,
not evaluation evidence, and should not be selected after the fact solely for
a favorable result.

## 2. Measure retrieval bias

```bash
uv run python scripts/run_diagnostics.py --provider claude
uv run python scripts/run_diagnostics.py --provider codex
```

For each provider this writes provider-tagged versions of:

- `artifacts/locomo_<provider>_neutral_degree_distribution.png`
- `artifacts/locomo_<provider>_neutral_bias_curve.png`
- `artifacts/locomo_<provider>_neutral_diagnostic_metrics.json`

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
uv run python scripts/build_musique.py --n-questions 60 --provider claude --prompt neutral
uv run python scripts/run_eval.py --provider claude --betas 0 0.5 1 2 --k 5
uv run python scripts/build_musique.py --n-questions 60 --provider codex --prompt neutral
uv run python scripts/run_eval.py --provider codex --betas 0 0.5 1 2 --k 5
```

This is a transductive, open-corpus MuSiQue variant. Paragraphs from all sampled
questions are deduplicated by normalized content and merged into one graph. It
is not the standard per-question, 20-candidate MuSiQue ranking task.

Entropy and degree kernels are swept on the dev split. The best nonzero dev
configuration is compared once with vanilla PPR on held-out test questions.
The report includes a paired bootstrap confidence interval, per-question
win/loss/tie counts, exact retrieved IDs, and input hashes:

```text
artifacts/musique_<provider>_neutral_eval.json
```

If the paired interval includes zero, report the result as inconclusive. A flat
or negative result is also a valid finding.

## Ablations

- `--prompt neutral` is the main condition. `--prompt specific` tests whether
  extraction instructions that discourage generic entities suppress hubs.
- `run_eval.py` compares entropy-weighted damping with degree-only damping.
- `beta=0` is vanilla PPR for either kernel.
- Claude and Codex are separate extraction conditions. Do not merge their
  caches, graphs, tuning decisions, or held-out results.

Use separate, predeclared runs when comparing prompt conditions. Do not widen
the beta grid after looking at held-out test results.

## Observed neutral-prompt result, 2026-07-29

This run used Claude Code 2.1.220 with `claude-opus-5` and Codex CLI 0.146.0
with `gpt-5.6-sol`, both at low effort. All 40 LoCoMo chunks and all 1,048
unique paragraphs in the fixed 60-question MuSiQue sample were extracted by
each provider without empty or invalid results.

LoCoMo developed a heavy tail. Caroline and Melanie were the two dominant
nodes and appeared among the top five PPR nodes for all eight in-domain, all
eight off-domain, and all eight random-seed trials under vanilla PPR. They are
real conversation participants, however, not generic junk entities. The metric
therefore detects structural concentration but does not distinguish harmful
hubs from legitimate richly described people.

Entropy damping at beta 1 reduced mean PPR mass in the top 1 percent leakiness
nodes from 0.368 to 0.176 for Claude and from 0.287 to 0.134 for Codex. In the
live work query, Claude changed one of three retrieved sources without an
obvious quality improvement; Codex returned the same three sources.

The primary held-out MuSiQue result was:

| Extractor | Vanilla Recall@5 | Entropy Recall@5 | Paired delta, 95% CI | W/L/T |
|---|---:|---:|---:|---:|
| Claude | 0.431 | 0.436 | +0.006 [-0.033, +0.050] | 1/1/28 |
| Codex | 0.397 | 0.419 | +0.022 [+0.000, +0.056] | 2/0/28 |

Beta 1 was selected independently for entropy on each dev graph. Both paired
intervals include zero. Degree-only damping selected beta 2 and was at least as
good as entropy: Recall@5 was 0.447 for Claude and 0.419 for Codex. The highest
leakiness MuSiQue nodes were mostly legitimate entities such as United States,
India, named films, cities, and football clubs.

The honest conclusion is that the run demonstrates query-independent PPR mass
concentration and shows that damping reduces it. It does not establish that the
entropy metric specifically identifies bad supernodes, nor that entropy
damping improves held-out retrieval. The retrieval result is inconclusive and
the degree ablation gives no evidence that relation entropy is the useful part
of the kernel.

## Entity-fragmentation PoC result, 2026-07-29

The updated research proposal identifies entity-resolution failure as the more
important candidate fault: the same real entity can become separate graph nodes
when sources are ingested independently. The controlled PoC injects that fault
without changing triples, source provenance, or semantic embedding labels. This
isolates graph identity from extraction quality.

Run the recorded experiment from the cached Claude and Codex extractions:

```bash
uv run python scripts/run_fragmentation_poc.py
```

The broad test split 0 to 100 percent of entities appearing in multiple sources
into two source-consistent shards. Ordinary multi-seed Recall@5 did not decline
monotonically. At full fragmentation it changed from 0.431 to 0.448 for Claude
and from 0.397 to 0.396 for Codex. This broad hypothesis was not verified. The
retriever can seed semantically identical shards independently, which masks the
lost graph connection.

The narrower bridge probe tests the exact proposed mechanism. For each eligible
MuSiQue question, it selects one entity shared by supporting sources, seeds one
source-side shard, and measures recall of the other supporting evidence. The
fault is a source-consistent split. The repair is an oracle-correct `same_as`
edge with a reserved fraction of transition mass. Identity-mix candidates were
selected on 30 dev questions, and the selected value of 0.5 was evaluated once
on the 30-question test split. Each case used five deterministic fragmentation
trials.

| Extractor | Eligible test questions | Original | Fragmented | Soft repair | Fragmentation delta, 95% CI | Repair delta, 95% CI | Gap recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude | 24 | 0.750 | 0.150 | 0.642 | -0.600 [-0.783, -0.408] | +0.492 [+0.308, +0.675] | 81.9% |
| Codex | 22 | 0.773 | 0.100 | 0.718 | -0.673 [-0.845, -0.473] | +0.618 [+0.418, +0.800] | 91.9% |

This verifies a narrow causal claim on two independently extracted graphs:
splitting an identity bridge destroys cross-source evidence traversal, and a
confidence-weighted soft identity transition restores most of it. It does not
show how often natural incremental ingestion creates these splits, whether an
automatic resolver can find the correct links, or whether repair improves the
broad end-to-end retriever. The repair uses oracle-correct links, and the chosen
mix is at the upper edge of the tested grid. The bridge probe was designed after
observing the broad null result, although its criteria were fixed before its
outcomes were run, so an independent confirmatory replication remains necessary.

The appropriate PoC is therefore the interactive identity-bridge failure and
soft-repair demonstration, backed by the held-out table above. The next research
stage should measure natural duplicate rate under batch versus incremental
ingestion, evaluate a real resolver against manually adjudicated aliases, inject
wrong links to measure merge risk, and then rerun end-to-end question retrieval.
Full results and per-condition provenance are in
`artifacts/fragmentation_poc.json`; the broad negative result and verified bridge
result are plotted in `artifacts/fragmentation_dose_response.png`.
