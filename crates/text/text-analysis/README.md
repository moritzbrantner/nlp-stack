# text-analysis

Unified text analysis orchestration for `moritzbrantner-video-analysis`.

The crate composes the lower-level text packages into document and corpus
reports. Defaults are deterministic, local-first, and do not download models or
invoke native inference runtimes. Local model-backed NER and embeddings are
available through explicit options and feature flags.

For an end-to-end walkthrough of how `CorpusAnalysisReport` relates to
`TextCorpus`, lexical scoring, model-backed semantic search, and retrieval indexes,
see [`docs/TEXT_CORPUS_GUIDE.md`](../../../docs/TEXT_CORPUS_GUIDE.md).

## Example

```rust,no_run
use text_analysis::{analyze_text, DocumentAnalysisOptions};

let report = analyze_text(
    "doc-1",
    "Alice presented the tokenizer roadmap in Berlin.",
    &DocumentAnalysisOptions::default(),
)?;

assert!(!report.lexical.keywords.is_empty());
# Ok::<(), text_core::DetectError>(())
```

## Semantic maps

`analyze_document_semantics` and `analyze_conversation_semantics` build a typed,
multi-scale semantic map rather than reducing a whole source to one vector.
Documents keep sentence, paragraph, and document units; conversations keep
speaker turns plus sentence children. The primary sequence is then used to
produce deterministic semantic-neighbor edges, concept clusters, a semantic
shift/activation timeline, concept hotspots, and per-speaker concept shares.

The `*_with` functions take the caller's `TextEmbeddingBackend`, so local
Candle/ONNX or future embedding implementations share the same semantic-map
contracts and deterministic aggregation algorithms. There is no hashed default
(#186); the package surface needs `importedEmbeddings` and otherwise fails with
`missing_dependency`.

The initial exact similarity matrix is intentionally O(n²). It is a transparent
baseline, not a claim that all future large-corpus analysis should stay
pairwise. See
[`docs/SEMANTIC_ANALYSIS_HORIZON.md`](../../../docs/SEMANTIC_ANALYSIS_HORIZON.md)
for the implementation horizon and ownership constraints.

## Feature flags

- `tokenizers`: enables tokenizer-backed model bundle helpers.
- `candle`: enables local Candle-backed NER and embedding integration.
- `onnx`: enables local ONNX-backed embedding integration.
- `external-tests`: enables model/runtime integration test features.

## Stable contract

The stable surface is high-level document and corpus report orchestration over
the focused text crates. Report sections may grow additively. Default analysis is
local-first and no-download: it requests the local linguistic models (a missing
model is a `linguistics_unavailable` diagnostic, never heuristic output) and no
embedding. `LinguisticDepth::ModelFree` runs only the model-free linguistic
operations.

## Quality and limits

This crate composes lower-level lexical statistics and model outputs. It provides
consistent report structure, not a production-grade NLP quality guarantee.

Semantic-map clusters and heat-map-oriented metrics are structural signals over
embedding similarity. They must not be interpreted as psychological judgments,
truth assessments, or lossless representations of meaning.

## Model-backed analysis

`DocumentAnalysisOptions::model_backed()` selects local-model analysis but does
not auto-download model bundles. Use
`DocumentAnalysisOptions::model_backed_with_downloads()` only when download
behavior is explicitly desired. Package-surface callers must pass
`autoDownload: true` in the `linguistics` request to request model downloads.

## Package surface

- Primary workflow: `analysis.document` analyzes one text with core, lexical,
  similarity, linguistic, and embedding sections.
- Workflow operations: `analysis.document`, `analysis.corpus`, and
  `analysis.similarity`.
- Debug operations: `analysis.describe` and `describe` inspect package metadata
  and operation support.
- Runtime support: default package-surface operations are deterministic and
  available through library, CLI, server, and WASM wrappers; model-backed
  behavior remains opt-in through explicit options and feature flags.
- Sample output includes `title`, `message`, `summary`, `result`, diagnostics,
  and operation-specific fields such as `core`, `lexical`, `documents`, or
  `similarity`.
- The package surface does not download models or invoke native inference by
  default.

## Related crates

- `text-core`
- `text-lexical`
- `text-linguistics`
- `text-embeddings`
- `text-retrieval`
