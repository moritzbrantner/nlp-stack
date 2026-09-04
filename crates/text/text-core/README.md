# text-core

Shared deterministic text documents, tokenization, spans, and statistics for the
NLP stack.

Default builds are local-first and do not download models or invoke native
inference/runtime tools.

## Feature flags

- No optional feature flags today.

## Stable kernel contract

`TextDocument`, `OwnedTextDocument`, `TextSpan`, token/sentence/paragraph
records, and text processing options are the intended text-kernel boundary.

A2 is actively removing extraction-era responsibilities that do not belong in
the kernel. `TextDocumentContract` / `TextSegmentContract`, media timing,
analysis-event/error re-exports, analyzer lifecycle, and package-surface/JSON
dispatch remain compatibility debt during that migration; do not add new
consumers or new parallel contract types to those seams.

### Span coordinates

UTF-8 half-open byte offsets are authoritative. The current `char_start` /
`char_end` members remain only as migration compatibility fields and must not be
used as a second source of truth.

When a boundary needs another coordinate system, derive it from the byte range:

```rust,no_run
use text_core::{TextProcessingOptions, TextSpan};

let text = "e\u{301}👍🏽a";
let span = TextSpan {
    byte_start: 3,
    byte_end: 11,
    // Legacy compatibility values are ignored by explicit conversions.
    char_start: 0,
    char_end: 0,
};

let utf16 = span.to_utf16(text)?;
let graphemes = span.to_grapheme(text)?;
assert_eq!((utf16.start, utf16.end), (2, 6));
assert_eq!((graphemes.start, graphemes.end), (1, 2));

# let _ = TextProcessingOptions::default();
# Ok::<(), text_core::contracts::TextSpanConversionError>(())
```

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

## Migration status

The extraction-era package surface still exposes `text.statistics`,
`text.normalize`, `text.tokenize`, `text.boundaries`, and `describe` through the
existing adapters. Those erased transport operations are not part of the target
kernel API and will move outward as the architecture migration proceeds.

The repository-level A2 boundary check is monotonic: current media/runtime/
transport debt may shrink, but new dependencies, new cross-domain source uses,
or new mirror `*Contract` types are rejected.

## Related crates

- `text-lexical`
- `text-linguistics`
