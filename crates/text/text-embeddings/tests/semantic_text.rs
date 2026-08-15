use text_core::TextDocument;
use text_embeddings::{
    text_similarity, CooccurrenceConfig, CooccurrenceGraph, HashedTextEmbedder, SemanticTextIndex,
    TextEmbeddingConfig,
};
use text_lexical::CorpusOptions;

#[test]
fn semantic_index_embeds_text_documents_and_searches_public_results() {
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 256,
            use_idf: true,
        },
        CorpusOptions::default(),
    )
    .unwrap();
    let mut index = SemanticTextIndex::new(embedder);

    index
        .add_text_document(&TextDocument::new(
            "rust",
            "rust cargo crates ownership compiler",
        ))
        .unwrap();
    index
        .add_text_document(&TextDocument::new("fruit", "oranges bananas apples citrus"))
        .unwrap();
    index
        .add_text_document(&TextDocument::new(
            "video",
            "video frames scenes detection ffmpeg",
        ))
        .unwrap();

    assert_eq!(index.corpus().len(), 3);
    let results = index.search("cargo compiler crate", 2).unwrap();

    assert_eq!(results[0].id, "rust");
    assert!(results[0].score > results[1].score);
    assert!(results[0].distance <= results[1].distance);
}

#[test]
fn hashed_embeddings_support_similarity_and_related_terms() {
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 256,
            use_idf: false,
        },
        CorpusOptions::default(),
    )
    .unwrap();

    let rust_similarity =
        text_similarity("rust cargo crates", "cargo compiler rust", &embedder).unwrap();
    let fruit_similarity =
        text_similarity("rust cargo crates", "banana citrus apple", &embedder).unwrap();
    assert!(rust_similarity > fruit_similarity);

    let mut graph = CooccurrenceGraph::new(CooccurrenceConfig {
        window_size: 2,
        min_term_len: 3,
    })
    .unwrap();
    graph.train_text("rust cargo build rust cargo test rust ownership");

    let related = graph.related_terms("rust", 3);
    assert_eq!(related[0].term, "cargo");
    assert!(related[0].score > 0.0);
}
