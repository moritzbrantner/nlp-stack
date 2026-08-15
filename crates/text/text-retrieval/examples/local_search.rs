use std::collections::BTreeMap;

use text_embeddings::{HashedTextEmbedder, TextEmbeddingConfig};
use text_lexical::CorpusOptions;
use text_retrieval::{
    HybridConfig, IngestionOptions, RetrievalIndex, SearchDocument, SearchFilter, SearchQuery,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let documents = vec![
        search_document(
            "doc-architecture",
            "Architecture notes describe Rust crates, public surfaces, and package boundaries.",
            [("source", "guide"), ("kind", "overview"), ("language", "en")],
        ),
        search_document(
            "doc-lexical",
            "Lexical search ranks TF-IDF and BM25 terms for local document collections.",
            [("source", "guide"), ("kind", "lexical"), ("language", "en")],
        ),
        search_document(
            "doc-semantic",
            "Semantic search uses deterministic hashed embeddings for fast local similarity.",
            [("source", "guide"), ("kind", "semantic"), ("language", "en")],
        ),
        search_document(
            "doc-hybrid",
            "Hybrid retrieval combines full-text signals with semantic vector scores and metadata filters.",
            [("source", "guide"), ("kind", "retrieval"), ("language", "en")],
        ),
    ];

    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 64,
            use_idf: false,
        },
        CorpusOptions::default(),
    )?;
    let mut index = RetrievalIndex::new(embedder);
    let report = index.ingest_documents(
        &documents,
        &IngestionOptions {
            chunk_tokens: 12,
            chunk_overlap_tokens: 3,
            store_raw_text: true,
        },
    )?;
    println!(
        "Ingested {} documents into {} chunks",
        report.documents_received, report.chunks_indexed
    );

    print_results(
        "Full-text search",
        &index.search(&SearchQuery::full_text("BM25 lexical document ranking", 3))?,
    );
    print_results(
        "Semantic search",
        &index.search(&SearchQuery::semantic("local similarity embeddings", 3))?,
    );
    print_results(
        "Hybrid search",
        &index.search(&SearchQuery::hybrid(
            "metadata aware hybrid retrieval",
            3,
            HybridConfig {
                semantic_weight: 0.65,
                lexical_weight: 0.35,
                rerank_window: 8,
                rerank: false,
            },
        ))?,
    );

    let mut metadata_equals = BTreeMap::new();
    metadata_equals.insert("kind".to_string(), "retrieval".to_string());
    let filtered = SearchQuery::hybrid(
        "vector metadata filters",
        3,
        HybridConfig {
            semantic_weight: 0.5,
            lexical_weight: 0.5,
            rerank_window: 8,
            rerank: false,
        },
    )
    .filter(SearchFilter {
        metadata_equals,
        ..SearchFilter::default()
    });
    print_results("Metadata-filtered search", &index.search(&filtered)?);

    Ok(())
}

fn search_document(
    id: &str,
    body: &str,
    metadata: impl IntoIterator<Item = (&'static str, &'static str)>,
) -> SearchDocument {
    let mut document = SearchDocument::new(id, body);
    document.metadata = metadata
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    document
}

fn print_results(label: &str, results: &[text_retrieval::SearchResult]) {
    println!("\n{label}:");
    for result in results {
        println!(
            "  chunk={} document={} score={:.3} semantic={:.3} lexical={:.3}",
            result.chunk_id,
            result.document_id,
            result.score,
            result.semantic_score,
            result.lexical_score
        );
        println!("    snippet: {}", result.snippet);
        println!("    metadata: {:?}", result.metadata);
    }
}
