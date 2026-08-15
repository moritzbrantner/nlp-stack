use runtime_core::{OperationId, SurfaceRequest};
use text_generation::surface::run_surface_operation;

fn main() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("generation.markovGenerate"),
        input: serde_json::json!({
            "trainingTexts": [
                "rust text analysis supports transcript search",
                "rust crates expose deterministic generation"
            ],
            "order": 2,
            "maxTokens": 12
        }),
    })
    .expect("markov generation");

    println!("{}", serde_json::to_string_pretty(&response.value).unwrap());
}
