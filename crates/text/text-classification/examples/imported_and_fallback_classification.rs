use runtime_core::{OperationId, SurfaceRequest};
use text_classification::surface::run_surface_operation;

fn main() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("classification.classify"),
        input: serde_json::json!({
            "text": "Rust text workflows are reliable.",
            "labels": ["positive", "negative"],
            "importedPredictions": [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.1}]
        }),
    })
    .expect("imported classification");

    println!("{}", serde_json::to_string_pretty(&response.value).unwrap());
}
