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

## Adding a capability suite

Keep each suite independently runnable and independently interpretable. Add a versioned corpus, a prediction shape, a deterministic evaluator using the reusable metrics where possible, a baseline report that states which system produced it, and the narrowest adapter needed to obtain predictions through the capability's stable public seam.

Do not mix model downloads, training, or hosted inference into the deterministic smoke suites. Larger external datasets and model-backed evaluations may be opt-in, but their provenance and setup requirements must be explicit.
