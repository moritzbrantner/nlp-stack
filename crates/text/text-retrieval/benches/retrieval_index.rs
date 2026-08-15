use criterion::{black_box, criterion_group, criterion_main, Criterion};
use runtime_core::{OperationId, SurfaceRequest};
use text_retrieval::surface::run_surface_operation;

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
    let search_input = serde_json::json!({
        "documents": [
            {"id": "doc-1", "body": "rust text retrieval"},
            {"id": "doc-2", "body": "video scene reports"},
            {"id": "doc-3", "body": "transcript search and chunks"}
        ],
        "query": "text search",
        "mode": "hybrid"
    });

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
        b.iter(|| {
            run_surface_operation(SurfaceRequest {
                operation: OperationId::new("retrieval.search"),
                input: black_box(search_input.clone()),
            })
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_retrieval_index);
criterion_main!(benches);
