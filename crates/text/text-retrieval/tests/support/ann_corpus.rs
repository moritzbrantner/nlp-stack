//! Deterministic topic-structured corpus for the HNSW retrieval evidence (#119).
//!
//! Each document is one short chunk: words drawn mostly from one of `TOPICS`
//! synthetic topic vocabularies plus shared filler words, so the hashed fixture
//! embedder produces clustered, overlapping vectors rather than random noise.

use text_embeddings::testing::HashedFixtureEmbedder;
use text_embeddings::TextEmbeddingConfig;
use text_lexical::CorpusOptions;
use text_retrieval::{IngestionOptions, RetrievalIndex, SearchDocument};

pub const DIMENSIONS: usize = 128;
pub const TOPICS: usize = 48;
const TOPIC_WORDS: usize = 40;
const SHARED_WORDS: usize = 200;
const WORDS_PER_DOCUMENT: usize = 16;
pub const SEED: u64 = 0x1190_c0de;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// `document_count` documents, identical for every call with the same count.
pub fn documents(document_count: usize) -> Vec<SearchDocument> {
    let mut rng = Rng(SEED);
    (0..document_count)
        .map(|index| {
            let topic = rng.below(TOPICS);
            let words = (0..WORDS_PER_DOCUMENT)
                .map(|_| {
                    if rng.below(4) == 0 {
                        format!("shared{}", rng.below(SHARED_WORDS))
                    } else {
                        format!("topic{topic}word{}", rng.below(TOPIC_WORDS))
                    }
                })
                .collect::<Vec<_>>();
            SearchDocument::new(format!("doc-{index:06}"), words.join(" "))
        })
        .collect()
}

pub fn embedder() -> HashedFixtureEmbedder {
    HashedFixtureEmbedder::new(
        TextEmbeddingConfig {
            dimensions: DIMENSIONS,
            use_idf: false,
        },
        CorpusOptions::default(),
    )
    .unwrap()
}

/// One chunk per document: every document is shorter than the chunk window.
pub fn index(document_count: usize) -> RetrievalIndex<HashedFixtureEmbedder> {
    let mut index = RetrievalIndex::new(embedder());
    let report = index
        .ingest_documents(&documents(document_count), &IngestionOptions::default())
        .unwrap();
    assert_eq!(report.chunks_indexed, document_count);
    index
}
