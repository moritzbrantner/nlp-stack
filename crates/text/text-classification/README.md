# text-classification

Concrete text classification, sentiment, and zero-shot classification contracts
for `moritzbrantner-video-analysis`.

Model download and bundle handling belong to `moritzbrantner-model-runtime`; this crate owns
the text-facing request/response types, imported prediction handling, and
local model execution.

## Stable contract

The stable surface is task selection, model catalog metadata, request/response
DTOs, imported prediction postprocessing, local models, and runtime broker
traits supplied by callers.

## Quality and limits

There are no heuristic fallbacks: without imported predictions, a caller
backend or a local model, classification, sentiment and zero-shot requests fail
with `missing_model`. The former `fast_fallback`/`lexical_fallback` policies were
removed (#186).
Catalog entries that are not `loadable` are reference metadata, not runnable
native models.

`CandleSequenceClassifier::from_bundle` (or `from_bundle_on` for an explicit
device) loads the weights and parses the tokenizer once; every later `classify`
call reuses them, and clones share them across threads. The local classify and
sentiment requests keep one loaded classifier per bundle root and model id.
`classify_batch` pads up to 16 texts into one forward pass: labels and their
ranking match per-text `classify`, scores can differ in the last float bits.
Local zero-shot requests likewise keep one loaded `OnnxZeroShotClassifier`
(ONNX session and tokenizer) per bundle root and model id.

The ONNX paths load ONNX Runtime >= 1.24 from `ORT_DYLIB_PATH` (or the loader
path). A missing or older library is an `ONNX Runtime unavailable: … set
ORT_DYLIB_PATH …` error, not a hang.

## Package surface

- Primary workflow: `classification.classify` runs imported-prediction or
  local-model text classification and fails without a model.
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
