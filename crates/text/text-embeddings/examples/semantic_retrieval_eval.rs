use runtime_core::{OperationId, SurfaceRequest};
use text_embeddings::surface::run_surface_operation;

fn main() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("embeddings.semanticSearch"),
        input: serde_json::json!({
            "documents": [
                {"id": "doc-1", "text": "rust text analysis"},
                {"id": "doc-2", "text": "semantic search over transcripts"},
                {"id": "doc-3", "text": "camera calibration and reconstruction"}
            ],
            "query": "transcript text search",
            "dimensions": 128
        }),
    })
    .expect("semantic search");

    println!("{}", serde_json::to_string_pretty(&response.value).unwrap());
}
