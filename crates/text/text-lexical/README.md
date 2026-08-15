# text-lexical

Text feature extraction and analyzer adapters for `moritzbrantner-video-analysis`.

Default builds are deterministic, local-first, and limited to classical lexical
analysis; they do not download models or invoke native inference/runtime tools.

## Feature flags

- No optional feature flags today.

## Stable contract

The stable surface is deterministic lexical analysis, `TextCorpus`, TF-IDF and
BM25 scoring, corpus snapshots, analyzers for generic text pipelines, and
portable `text-core` document/segment ingestion. Transcript-specific analyzers
live in `text-transcripts`.

## Quality and limits

Readability, sentiment, stemming, keyword extraction, and summaries are
classical local heuristics. Their API shape is stable, but output quality should
be treated as best-effort outside covered languages and fixtures.

## Example

```rust,no_run
use text_core::TextProcessingOptions;
use text_lexical::{keywords, token_shingle_similarity, KeywordOptions};

let keywords = keywords(
    "Video analysis surfaces scenes, transcript events, and metrics.",
    &KeywordOptions::default(),
);

let similarity = token_shingle_similarity(
    "scene transitions follow motion",
    "scene transitions follow dialogue",
    2,
    &TextProcessingOptions::default(),
)?;

let _ = (keywords, similarity);
# Ok::<(), text_core::DetectError>(())
```

## Corpus APIs

Use `TextCorpus` when you want to assemble a lexical corpus from raw text,
portable `text-core` document contracts, or segment contracts while keeping
document language and metadata available for later workflows. It owns the raw
text and can be serialized through snapshots after conversion to deterministic
lexical term statistics.

For the high-level workflow across contracts, `TextCorpus`, retrieval, analysis,
classification, and cited QA, see
[`docs/TEXT_WORKSPACE_GUIDE.md`](../../../docs/TEXT_WORKSPACE_GUIDE.md). For
the lower-level corpus APIs, see
[`docs/TEXT_CORPUS_GUIDE.md`](../../../docs/TEXT_CORPUS_GUIDE.md).

`TfIdfCorpus` and `Bm25Corpus` are scoring/index structures. They preserve the
existing direct construction APIs and are still the right choice when you only
need local lexical search or term statistics. `TextCorpus` converts into both
without changing the scoring behavior.

`text-retrieval::RetrievalIndex` is separate: it owns chunked, metadata-rich
retrieval workflows that combine full-text, vector, and hybrid search.

```rust,no_run
use text_core::{TextDocumentContract, TextSegmentContract};
use text_lexical::{Bm25Options, CorpusOptions, TextCorpus};

let mut document = TextDocumentContract::new("doc-1", "Rust cargo builds packages.");
document.language = Some("en".to_string());
document
    .attributes
    .insert("source".to_string(), "readme".to_string());

let mut segment = TextSegmentContract::new(2, "Scene reports mention cargo.");
segment.stream_id = Some("subs".to_string());

let corpus = TextCorpus::from_document_contracts([document], CorpusOptions::default())?;
let tfidf = corpus.to_tfidf_corpus()?;
let bm25 = corpus.to_bm25_corpus(Bm25Options::default())?;
let snapshot_json = serde_json::to_string_pretty(&corpus.snapshot()?)?;

let _ = (segment, tfidf.search("cargo", 5)?, bm25.search("cargo", 5)?, snapshot_json);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Package surface

- Primary workflow: `lexical.analyze` computes deterministic lexical features,
  summaries, readability, sentiment, and rule entities.
- Workflow operations: `lexical.analyze`, `lexical.keywords`,
  `lexical.corpusSearch`, and `lexical.corpusStats`.
- Debug operations: `describe` inspects package metadata and operation support.
- Runtime support: pure Rust, available through library, CLI, server, and WASM
  wrappers.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific fields such as `keywords`, `phraseKeywords`, `mode`,
  `results`, `stats`, `terms`, `documentTfidf`, or sparse matrix previews.
- This crate does not download models, execute native inference, or persist
  corpus indexes from package-surface operations.

## Related crates

- `text-core`
- `text-embeddings`
- `media-core`
