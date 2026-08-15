# text-embeddings

Lightweight semantic text embeddings and search for `moritzbrantner-video-analysis`.

Default builds use deterministic hashed/local embedding behavior. Native model
execution and tokenizer-backed model support are opt-in through explicit feature
flags.

## Feature flags

- `tokenizers`: tokenizer loading and model bundle tokenizer support
- `onnx`: ONNX-backed text embedding runtime support
- `candle`: Candle-backed text embedding runtime support
- `external-tests`: opt-in model/runtime checks

## Stable contract

The stable surface is `TextEmbeddingBackend`, deterministic hashed embeddings,
embedding metadata, vector search helpers, and backend catalog reporting.
Default package operations use hashed embeddings only.

## Quality and limits

Hashed vectors are reproducible interoperability baselines, not production
semantic embedding quality claims. Native ONNX and Candle embeddings remain
feature-gated and setup-dependent.

## Example

```rust,no_run
use text_lexical::CorpusOptions;
use text_embeddings::{HashedTextEmbedder, SemanticTextIndex, TextEmbeddingConfig};

let embedder = HashedTextEmbedder::new(
    TextEmbeddingConfig {
        dimensions: 128,
        use_idf: true,
    },
    CorpusOptions::default(),
) .unwrap();
let mut index = SemanticTextIndex::new(embedder);

index.add_document("scene-1", "opening scene with presenter").unwrap();
let matches = index.search("presenter on screen", 5).unwrap();

assert_eq!(matches[0].metadata.model_name.as_deref(), Some("hashed-text-embedder"));
```

## Package surface

- Primary workflow: `embeddings.embed` builds deterministic hashed embeddings.
- Workflow operations: `embeddings.embed`, `embeddings.similarity`,
  `embeddings.semanticSearch`, and `embeddings.relatedTerms`.
- Debug operations: `embeddings.backends` inspects backend catalog metadata, and
  `describe` inspects package metadata and operation support.
- Runtime support: default package-surface operations are pure Rust and
  available through library, CLI, server, and WASM wrappers; tokenizer, ONNX, and
  Candle runtime paths remain opt-in library features.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific fields such as `embeddings`, `similarity`, `results`, or
  `relatedTerms`.
- The package surface does not download model bundles or invoke native ONNX or
  Candle inference. Backend catalog inspection reports support metadata without
  loading model bundles.

## Model Loading

The catalog treats hashed embeddings as the default loadable path. MiniLM Candle
and ONNX entries are loadable only when the matching `.model-runtime` bundles
exist and the crate is built with `candle` or `onnx` plus `model-bundles`.

```bash
scripts/sync_model_bundles.sh
cargo test -p text-embeddings --features external-tests -- --ignored
```

Use `bun run text-native:bench` for Criterion benches and
`bun run text-wasm:bench:all` for local browser WASM timings.

## Related crates

- `text-core`
- `vector-analysis-index`
- `text-lexical`
