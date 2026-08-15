# text-lexical

Corpus indexing, TF-IDF, and BM25 utilities for `video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,no_run
use text_lexical::{Bm25Corpus, Bm25Options, CorpusOptions, TfIdfCorpus};
use text_core::OwnedTextSegment;

let tfidf = TfIdfCorpus::from_texts(
    ["multimodal analysis with rust", "scene graphs and video reports"],
    CorpusOptions::default(),
)?;

assert_eq!(tfidf.documents()[0].id, "doc-0");

let mut corpus = Bm25Corpus::new(Bm25Options::default());
corpus.add_document("doc-1", "multimodal analysis with rust")?;

let _results = corpus.search("analysis rust", 5)?;

let mut subtitle_corpus = TfIdfCorpus::new(CorpusOptions::default());
for segment in [
    OwnedTextSegment::new(0, "Rust cargo crates"),
    OwnedTextSegment::new(1, "Cargo build pipeline"),
] {
    subtitle_corpus.add_text_segment("subs", &segment.as_segment())?;
}

assert_eq!(subtitle_corpus.documents()[0].id, "subs:0");
# Ok::<(), text_core::DetectError>(())
```

## Related crates

- `text-core`
- `text-transcripts`
- `text-embeddings`
