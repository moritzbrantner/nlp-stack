use criterion::{black_box, criterion_group, criterion_main, Criterion};
use runtime_core::{OperationId, SurfaceRequest};
use text_lexical::surface::run_surface_operation;

fn bench_lexical_corpus(c: &mut Criterion) {
    let keyword_input = serde_json::json!({
        "text": "Rust text analysis supports transcript retrieval and lexical search. ".repeat(64),
        "maxTerms": 16
    });
    let search_input = serde_json::json!({
        "documents": [
            {"id": "doc-1", "text": "rust text analysis"},
            {"id": "doc-2", "text": "video scene analysis"},
            {"id": "doc-3", "text": "transcript retrieval and search"}
        ],
        "query": "text search",
        "mode": "bm25"
    });

    c.bench_function("lexical_keywords", |b| {
        b.iter(|| {
            run_surface_operation(SurfaceRequest {
                operation: OperationId::new("lexical.keywords"),
                input: black_box(keyword_input.clone()),
            })
            .unwrap()
        })
    });
    c.bench_function("lexical_corpus_search", |b| {
        b.iter(|| {
            run_surface_operation(SurfaceRequest {
                operation: OperationId::new("lexical.corpusSearch"),
                input: black_box(search_input.clone()),
            })
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_lexical_corpus);
criterion_main!(benches);
