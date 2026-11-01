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

`CandleSequenceClassifier::from_bundle` (or `from_bundle_on` for an explicit
device) loads the weights and parses the tokenizer once; every later `classify`
call reuses them, and clones share them across threads. The local classify and
sentiment requests keep one loaded classifier per bundle root and model id.
`classify_batch` pads up to 16 texts into one forward pass: labels and their
ranking match per-text `classify`, scores can differ in the last float bits.

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
