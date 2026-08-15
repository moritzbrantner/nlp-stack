# text-classification

Concrete text classification, sentiment, and zero-shot classification contracts
for `moritzbrantner-video-analysis`.

Model download and bundle handling belong to `moritzbrantner-model-runtime`; this crate owns
the text-facing request/response types, imported prediction handling, and
deterministic fallback behavior.

## Stable contract

The stable surface is task selection, model catalog metadata, request/response
DTOs, imported prediction postprocessing, explicit fallback policies, and
runtime broker traits supplied by callers.

## Quality and limits

Default classification and sentiment fallbacks are deterministic heuristics.
Catalog entries that are not `loadable` are reference metadata, not runnable
native models.

## Package surface

- Primary workflow: `classification.classify` runs imported-prediction or
  explicit lexical fallback text classification.
- Workflow operations: `classification.classify`, `classification.sentiment`,
  and `classification.zeroShot`.
- Debug operations: `classification.models` inspects the model catalog, and
  `describe` inspects package metadata and operation support.
- Runtime support: pure Rust default behavior is available through library, CLI,
  server, and WASM wrappers.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific fields from classification, sentiment, or zero-shot
  responses.
- The package surface does not download model bundles or execute native model
  runtimes.
