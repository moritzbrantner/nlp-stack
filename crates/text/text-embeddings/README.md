# text-embeddings

Text embedding backends (ONNX, Candle), embedding search helpers and
rule-based co-occurrence term statistics (`embeddings.relatedTerms`: counts of
terms within a window, not semantic similarity).

There is no default embedding backend: callers supply a real
`TextEmbeddingBackend`. Native model execution and tokenizer-backed model support
are opt-in through explicit feature flags. The deterministic feature-hashing
embedder lives in `testing` as `HashedFixtureEmbedder` for tests that need
reproducible vectors; it is not a semantic model and never a production default
(#186).

## Multilingual presets (#188)

`MultilingualEmbeddingPreset` pins three multilingual models by Hugging Face
commit and per-file SHA-256. `load_embedding_preset` (features
`candle,model-bundles`) resolves the bundle through `model-runtime`, checks the
revision and digests, and returns a `CandleTextEmbedder` with the model's
pooling, prefixes and L2 normalization:

| Preset | Repository | Dims | Pooling | Query / passage format |
| --- | --- | --- | --- | --- |
| `multilingual-e5-base` (default) | `intfloat/multilingual-e5-base` | 768 | mean | `query: ` / `passage: ` (symmetric: `query: `) |
| `qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | last token | `Instruct: …\nQuery:` / none |
| `bge-m3` | `BAAI/bge-m3` | 1024 | CLS | none |

Callers never write prefixes. `embed_query` embeds a search query,
`embed_passages` embeds stored passages, and `embed_text` / `embed_batch` embed
symmetric text (similarity). `EmbeddingModelInfo` reports model id, revision and
pooling. Bundles live in `TEXT_EMBEDDINGS_MODEL_DIR` (default `.model-runtime`);
downloads are off unless `EmbeddingPresetLoadOptions::auto_download` is set.

`ImportedEmbedder` serves caller-computed vectors (per text, optionally per
query/passage role) for surfaces that cannot load a model, such as WASM.

## Feature flags

- `tokenizers`: tokenizer loading and model bundle tokenizer support
- `onnx`: ONNX-backed text embedding runtime support
- `candle`: Candle-backed text embedding runtime support
- `external-tests`: opt-in model/runtime checks

## Stable contract

The stable surface is `TextEmbeddingBackend`, embedding metadata, vector search
helpers (`EmbeddingSearchIndex<E>`, `text_similarity`), `NoEmbedder` for
model-free indexes, and backend catalog reporting. The package surface cannot
load a model, so `embeddings.embed`, `embeddings.similarity` and
`embeddings.semanticSearch` fail with a `missing_dependency` error.

## Quality and limits

Native ONNX and Candle embeddings are feature-gated and setup-dependent.
`testing::HashedFixtureEmbedder` vectors measure term overlap only.

## Example

```rust,no_run
use text_embeddings::{EmbeddingSearchIndex, TextEmbeddingBackend};

fn search(embedder: impl TextEmbeddingBackend) {
    let mut index = EmbeddingSearchIndex::new(embedder);
    index.add_document("scene-1", "opening scene with presenter").unwrap();
    let matches = index.search("presenter on screen", 5).unwrap();
    assert_eq!(matches[0].id, "scene-1");
}
```

## Package surface

- Primary workflow: `embeddings.embed`, which needs a configured embedding model
  and otherwise fails with `missing_dependency`.
- Workflow operations: `embeddings.embed`, `embeddings.similarity`,
  `embeddings.semanticSearch`, and `embeddings.relatedTerms`.
- Debug operations: `embeddings.backends` inspects backend catalog metadata, and
  `describe` inspects package metadata and operation support.
- Runtime support: `embeddings.relatedTerms` and `embeddings.backends` are pure
  Rust everywhere; tokenizer, ONNX, and Candle runtime paths remain opt-in library
  features.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific fields such as `embeddings`, `similarity`, `results`, or
  `relatedTerms`.
- The package surface does not download model bundles or invoke native ONNX or
  Candle inference. Backend catalog inspection reports support metadata without
  loading model bundles.

## Model Loading

The catalog has no default loadable backend. MiniLM Candle and ONNX entries are
loadable only when the matching `.model-runtime` bundles exist and the crate is
built with `candle` or `onnx` plus `model-bundles`.

`CandleTextEmbedder::from_bundle` (or `from_bundle_on` for an explicit device)
loads the weights and parses the tokenizer once; every later call reuses them,
and clones share them across threads. `embed_batch` pads up to 16 texts into one
forward pass: vectors match per-text `embed_text` except in the last float bits.

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
