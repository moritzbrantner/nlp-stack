# text-core

Shared text documents, tokenization, spans, and statistics for `moritzbrantner-video-analysis`.

Default builds are deterministic, local-first, and do not download models or
invoke native inference/runtime tools.

## Feature flags

- No optional feature flags today.

## Stable contract

`TextDocument`, `OwnedTextDocument`, `TextSpan`, token/sentence/paragraph
records, text processing options, and the portable document/segment contracts
are the stable data boundary for other crates. The portable contracts also
carry optional rich metadata: source references, media timing, provenance, and
annotation spans.

## Quality and limits

Segmentation is deterministic and Unicode-aware, but it is not a statistical NLP
tokenizer. Higher-level linguistic quality belongs in `text-linguistics`.

## Example

```rust,no_run
use text_core::{build_annotation_graph, TextDocument, TextProcessingOptions};

let document = TextDocument::new("doc-1", "Rust-first multimodal analysis.");
let graph = build_annotation_graph(document.text, &TextProcessingOptions::default());

assert!(!graph.tokens.is_empty());
assert_eq!(graph.sentences.len(), 1);
```

## Package surface

- Primary workflow: `text.tokenize` returns tokens, spans, script profile, and
  text statistics.
- Workflow operations: `text.statistics`, `text.normalize`, `text.tokenize`,
  and `text.boundaries`.
- Debug operations: `describe` inspects package metadata and operation support.
- Runtime support: pure Rust, available through library, CLI, server, and WASM
  wrappers.
- Sample output includes `title`, `message`, `summary`, `result`, and the
  operation-specific fields such as `tokens`, `stats`, `text`, or `words`.
- This crate does not download models, run native inference, or scan files.

## Related crates

- `text-lexical`
- `text-linguistics`
- `media-core`
