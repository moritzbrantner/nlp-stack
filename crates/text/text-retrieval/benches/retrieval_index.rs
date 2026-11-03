use criterion::{criterion_group, criterion_main, Criterion};
use runtime_core::{OperationId, SurfaceRequest};
use std::hint::black_box;
use text_embeddings::testing::HashedFixtureEmbedder;
use text_embeddings::TextEmbeddingConfig;
use text_lexical::CorpusOptions;
use text_retrieval::surface::run_surface_operation;
use text_retrieval::{
    IngestionOptions, RetrievalIndex, RetrievalMode, SearchDocument, SearchQuery,
};

fn bench_retrieval_index(c: &mut Criterion) {
    let chunk_input = serde_json::json!({
        "documents": [
            {
                "id": "doc-1",
                "body": "Rust text retrieval chunks transcript content for search. ".repeat(64)
            }
        ],
        "options": {
            "strategy": "TokenWindow",
            "ingestion": {
                "chunk_tokens": 32,
                "chunk_overlap_tokens": 4,
                "store_raw_text": true
            }
        }
    });
    // Index mechanics only: the hashed fixture stands in for an embedding model.
    let mut index = RetrievalIndex::new(
        HashedFixtureEmbedder::new(
            TextEmbeddingConfig {
                dimensions: 128,
                use_idf: false,
            },
            CorpusOptions::default(),
        )
        .unwrap(),
    );
    index
        .ingest_documents(
            &[
                SearchDocument::new("doc-1", "rust text retrieval"),
                SearchDocument::new("doc-2", "video scene reports"),
                SearchDocument::new("doc-3", "transcript search and chunks"),
            ],
            &IngestionOptions::default(),
        )
        .unwrap();
    let query = SearchQuery::new("text search", 10).mode(RetrievalMode::Hybrid);

    c.bench_function("retrieval_chunk", |b| {
        b.iter(|| {
            run_surface_operation(SurfaceRequest {
                operation: OperationId::new("retrieval.chunk"),
                input: black_box(chunk_input.clone()),
            })
            .unwrap()
        })
    });
    c.bench_function("retrieval_hybrid_search", |b| {
        b.iter(|| index.search(black_box(&query)).unwrap())
    });
}

criterion_group!(benches, bench_retrieval_index);
criterion_main!(benches);
