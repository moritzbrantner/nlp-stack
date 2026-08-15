# text-generation-linguistics

Adapters from `moritzbrantner-text-linguistics` analysis outputs into `moritzbrantner-text-generation`
Markov training and template synthesis.

## Stable contract

The stable surface is conversion from linguistic analysis outputs into
generation terms, deterministic synthesis inputs, and Markov training data.

## Quality and limits

This crate inherits the heuristic quality limits of `text-linguistics` and the
deterministic generation limits of `text-generation`. It does not execute hosted
or native LLM inference.

## Package surface

- Primary workflow: `generationLinguistics.synthesizeFromAnalysis` analyzes
  text and synthesizes a deterministic document from linguistic terms.
- Workflow operations: `generationLinguistics.analysisTerms`,
  `generationLinguistics.synthesizeFromAnalysis`, and
  `generationLinguistics.trainAnalysis`.
- Debug operations: `describe` inspects package metadata and operation support.
- Runtime support: pure Rust, available through library, CLI, server, and WASM
  wrappers.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific fields such as `terms`, `entities`, `value`, `trace`, or
  `tokens`.
- Package-surface operations use deterministic local analysis and do not
  download model bundles or execute hosted generation.
