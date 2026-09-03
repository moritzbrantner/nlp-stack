use criterion::{black_box, criterion_group, criterion_main, Criterion};
use runtime_core::{OperationId, SurfaceRequest};
use text_analysis::semantic::{
    analyze_document_semantics, compare_semantic_neighborhoods, SemanticAnalysisOptions,
};
use text_analysis::surface::run_surface_operation;
use text_core::TextDocument;

fn bench_document_and_corpus(c: &mut Criterion) {
    let document_input = serde_json::json!({
        "id": "bench-doc",
        "text": "Alice presented the tokenizer roadmap in Berlin. Rust crates analyze text with deterministic local features. ".repeat(12),
        "profile": "deterministic",
        "keywordLimit": 12,
        "summarySentences": 3,
        "embedding": {"mode": "hashed", "dimensions": 128, "useIdf": false}
    });
    let corpus_input = serde_json::json!({
        "documents": [
            {"id": "doc-1", "text": "rust text analysis"},
            {"id": "doc-2", "text": "video scene analysis"},
            {"id": "doc-3", "text": "semantic search over transcripts"}
        ],
        "query": "text analysis",
        "topK": 5,
        "includeSemanticNeighbors": true,
        "embedding": {"mode": "hashed", "dimensions": 128, "useIdf": true}
    });
    let semantic_text = [
        "Semantic search improves retrieval.",
        "Semantic search improves retrieval.",
        "Tomatoes grow in soil.",
        "Vector indexes accelerate nearest-neighbor search.",
        "Semantic search improves retrieval.",
        "Garden soil supports tomato roots.",
    ]
    .join(" ")
    .repeat(12);
    let semantic_document = TextDocument::new("semantic-bench", &semantic_text);
    let semantic_options = SemanticAnalysisOptions::default();
    let semantic_report =
        analyze_document_semantics(&semantic_document, &semantic_options).unwrap();

    c.bench_function("analysis_document", |b| {
        b.iter(|| {
            run_surface_operation(SurfaceRequest {
                operation: OperationId::new("analysis.document"),
                input: black_box(document_input.clone()),
            })
            .unwrap()
        })
    });
    c.bench_function("analysis_corpus", |b| {
        b.iter(|| {
            run_surface_operation(SurfaceRequest {
                operation: OperationId::new("analysis.corpus"),
                input: black_box(corpus_input.clone()),
            })
            .unwrap()
        })
    });
    c.bench_function("semantic_neighborhood_parity", |b| {
        b.iter(|| {
            compare_semantic_neighborhoods(
                black_box(&semantic_report),
                semantic_options.neighbors_per_unit,
                semantic_options.neighbor_threshold,
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_document_and_corpus);
criterion_main!(benches);
