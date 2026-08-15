# text-question-answering

Concrete extractive question-answering contracts for `moritzbrantner-video-analysis`.

Model download and bundle handling belong to `moritzbrantner-model-runtime`; this crate owns
the question/context request shape, answer predictions, imported span
postprocessing, and fallback behavior.

## Stable contract

The stable surface is extractive QA request/response DTOs, imported span
postprocessing, model catalog metadata, and runtime broker traits supplied by
callers.

## Quality and limits

Imported span workflows remain deterministic and WASM-compatible. Native
server/CLI builds with `local-onnx` use
`onnx-community/roberta-base-squad2-ONNX` by default when no imported
predictions or caller backend are supplied. Missing bundles are resolved through
`moritzbrantner-model-runtime` and materialized under `.model-runtime` unless
the request disables `auto_download`.

## Package surface

- Primary workflow: `qa.answer` postprocesses imported extractive QA spans when
  supplied, uses a caller backend when provided, otherwise runs the default
  local ONNX QA model on native server/CLI builds with `local-onnx`.
- Primary cited document-QA path: `qa.answerWithIndex` builds a deterministic
  request-scoped in-memory `TextIndex` and returns cited extractive answers.
- Deprecated compatibility cited document-QA path: `qa.answerWithRetrieval`
  keeps the soft-legacy compatibility `RetrievalIndex` path callable for
  existing consumers, but new package consumers should use
  `qa.answerWithIndex`.
- Retrieval-backed QA paths assemble Retrieved Context before answer extraction:
  full chunk text is used for extraction when available, while citations keep
  the existing display snippet shape.
- Workflow operations: `qa.answer`, `qa.answerWithIndex`, and
  `qa.answerBatch`.
- Support operations: `qa.answerWithRetrieval` remains runnable as a deprecated
  compatibility path.
- Debug operations: `qa.models` inspects the model catalog, and `describe`
  inspects package metadata and operation support.
- Runtime support: imported/fallback behavior is available through library,
  CLI, server, and WASM wrappers. Local ONNX execution is server/CLI-only.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific answer, citation, batch-result, indexed chunk, and
  compatibility retrieval fields.
- Model-backed calls declare filesystem reads, filesystem writes, and network
  access in their execution plan. Use `fallbackPolicy:
  "heuristicIfUnavailable"` only when deterministic heuristic fallback is
  explicitly acceptable.
