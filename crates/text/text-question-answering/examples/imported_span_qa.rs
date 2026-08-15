use runtime_core::{OperationId, SurfaceRequest};
use text_question_answering::surface::run_surface_operation;

fn main() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("qa.answer"),
        input: serde_json::json!({
            "question": "What is reliable?",
            "context": "Rust is reliable for deterministic text analysis.",
            "importedPredictions": [
                {"text": "Rust", "score": 0.91, "attributes": {"byte_start": "0", "byte_end": "4"}}
            ]
        }),
    })
    .expect("imported span QA");

    println!("{}", serde_json::to_string_pretty(&response.value).unwrap());
}
