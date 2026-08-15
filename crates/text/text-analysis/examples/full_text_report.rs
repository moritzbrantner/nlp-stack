use runtime_core::{OperationId, SurfaceRequest};
use text_analysis::surface::run_surface_operation;

fn main() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("analysis.document"),
        input: serde_json::json!({
            "id": "report-demo",
            "text": "Alice presented the tokenizer roadmap in Berlin. Rust crates analyze text with deterministic local features.",
            "profile": "deterministic",
            "keywordLimit": 8,
            "summarySentences": 2,
            "embedding": {"mode": "hashed", "dimensions": 128, "useIdf": false}
        }),
    })
    .expect("document report");

    println!("{}", serde_json::to_string_pretty(&response.value).unwrap());
}
