# NLP evaluation

This directory owns repository-level quality evaluation for `nlp-stack`. It is development and evidence infrastructure, not a runtime package and not a publication surface.

## Contract

Evaluation inputs are versioned JSONL corpora under `evaluation/corpora/`. Predictions use task-specific JSONL records with the same case IDs. Reports use `schemaVersion: 1` and record the suite, system, aggregate metrics, and per-case evidence when useful.

The initial sentence-boundary suite uses UTF-8 half-open byte ends so it evaluates the same span coordinate system exposed by `text-core`. The committed corpus is deliberately small: it is a deterministic smoke/regression corpus covering ordinary English, decimals and built-in abbreviations, caller-supplied citation abbreviations, internal-period abbreviations, Unicode case folding, and CJK terminators. It is not presented as a production-quality benchmark.

`evaluation.metrics` provides reusable deterministic metrics for later suites:

- precision / recall / F1 over exact events or spans;
- accuracy and macro F1 for classification-style tasks;
- reciprocal rank, MRR, Recall@k, and nDCG@k for retrieval;
- Spearman correlation for semantic similarity scores.

## Sentence-boundary baseline

Run the live `text-core` baseline with:

```bash
bash scripts/check-nlp-evaluation.sh
```

The script generates predictions through the public `text-core` sentence-boundary API, evaluates them against `sentence-boundaries-v1`, writes a report under `.artifacts/nlp-eval/`, and requires F1 to remain at least `NLP_EVAL_MIN_BOUNDARY_F1` (default `1.0`).

The committed prediction and aggregate report files under `evaluation/baselines/` are reproducible fixtures for the current baseline. The Python test suite verifies that the aggregate report is derivable from those committed inputs. The live script is the proof that current Rust behavior still produces predictions through the public capability seam.

## Semantic-analysis smoke suites

`semantic_similarity_smoke_v1.jsonl` and `topic_shift_smoke_v1.jsonl` establish the first versioned H7 fixtures. They are deliberately small contract/evaluator fixtures, not semantic-quality benchmarks.

The evaluator exposes two report-only commands:

```bash
python3 -m evaluation.runner semantic-similarity \
  --gold evaluation/corpora/semantic_similarity_smoke_v1.jsonl \
  --predictions path/to/predictions.jsonl \
  --suite semantic-similarity-smoke-v1 \
  --system your-system

python3 -m evaluation.runner topic-shifts \
  --gold evaluation/corpora/topic_shift_smoke_v1.jsonl \
  --predictions path/to/predictions.jsonl \
  --suite topic-shift-smoke-v1 \
  --system your-system
```

Semantic similarity reports Spearman rank correlation and mean absolute error. Topic shifts use exact index precision/recall/F1. No minimum semantic-quality threshold is enforced yet: hashed embeddings are useful deterministic interoperability fixtures, but they are not a meaningful quality target for semantic similarity or topic segmentation.

### Grouped and multilingual evidence

Semantic-similarity and topic-shift gold cases may include an optional non-empty `group` string. Group metadata is owned by the gold corpus: it does not participate in case identity, prediction matching, or score calculation. When at least one gold case is grouped, the report adds a deterministic `groups` section with per-group case counts and the same metrics used by the aggregate report. Ungrouped cases continue to contribute to aggregate metrics but are intentionally absent from the grouped breakdown.

`semantic_similarity_multilingual_v1.jsonl` adds small English, German, Spanish, and cross-language pairs. Its groups separate monolingual language pairs from English–German, English–Spanish, and German–Spanish comparisons. `topic_shift_multilingual_v1.jsonl` adds monolingual topic trajectories plus language-switch cases, including a language switch that keeps the same topic so systems can expose language sensitivity rather than silently treating every language change as a semantic topic boundary.

These fixtures are still calibration evidence, not production benchmarks. They make language-specific regressions visible without introducing a preferred embedding model or a minimum quality threshold. Larger labelled corpora and model-backed comparisons should be added with explicit provenance before semantic quality gates are considered.

## Adding a capability suite

Keep each suite independently runnable and independently interpretable. Add a versioned corpus, a prediction shape, a deterministic evaluator using the reusable metrics where possible, a baseline report that states which system produced it, and the narrowest adapter needed to obtain predictions through the capability's stable public seam.

Do not mix model downloads, training, or hosted inference into the deterministic smoke suites. Larger external datasets and model-backed evaluations may be opt-in, but their provenance and setup requirements must be explicit.
