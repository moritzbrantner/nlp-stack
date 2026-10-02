use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use runtime_core::{OperationId, SurfaceRequest};
use text_analysis::semantic::{
    analyze_document_semantics, analyze_document_semantics_with, compare_semantic_neighborhoods,
    SemanticAnalysisOptions,
};
use text_analysis::surface::run_surface_operation;
use text_core::TextDocument;
use text_embeddings::{DenseVector, TextEmbeddingBackend};

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

// Identical vectors force every possible merge and exposed the cubic clustering regression.
// Criterion samples end-to-end latency; lower is better. Keep sizes stable for baseline comparisons.
fn bench_repeated_semantic_sentences(c: &mut Criterion) {
    struct RepeatedEmbedding;
    impl TextEmbeddingBackend for RepeatedEmbedding {
        fn embed_text(&self, _text: &str) -> text_core::Result<DenseVector> {
            DenseVector::new(vec![1.0, 0.0])
        }
    }

    let mut group = c.benchmark_group("semantic_repeated_sentences");
    group.sample_size(10);
    for count in [500, 1_000, 2_000] {
        let text = "Cats sleep. ".repeat(count);
        let document = TextDocument::new("repeated", &text);
        let options = SemanticAnalysisOptions::default();
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                analyze_document_semantics_with(black_box(&document), &options, &RepeatedEmbedding)
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_document_and_corpus,
    bench_repeated_semantic_sentences
);
criterion_main!(benches);
