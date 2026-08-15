use runtime_core::{OperationId, SurfaceRequest};
use text_classification::surface::run_surface_operation;

fn main() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("classification.classify"),
        input: serde_json::json!({
            "text": "Rust text workflows are reliable.",
            "labels": ["positive", "negative"],
            "model": {"fallbackPolicy": "lexical_fallback"}
        }),
    })
    .expect("fallback classification");

    println!("{}", serde_json::to_string_pretty(&response.value).unwrap());
}
