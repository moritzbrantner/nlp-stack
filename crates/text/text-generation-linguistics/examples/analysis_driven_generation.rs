use runtime_core::{OperationId, SurfaceRequest};
use text_generation_linguistics::surface::run_surface_operation;

fn main() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("generationLinguistics.synthesizeFromAnalysis"),
        input: serde_json::json!({
            "id": "analysis-demo",
            "text": "Alice presented tokenizer evidence in Berlin. Rust search summarized transcript topics."
        }),
    })
    .expect("analysis-driven generation");

    println!("{}", serde_json::to_string_pretty(&response.value).unwrap());
}
