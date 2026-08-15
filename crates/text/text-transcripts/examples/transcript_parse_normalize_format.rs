use runtime_core::{OperationId, SurfaceRequest};
use text_transcripts::surface::run_surface_operation;

fn main() {
    let parsed = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("transcripts.parse"),
        input: serde_json::json!({
            "format": "srt",
            "content": "1\n00:00:01,000 --> 00:00:02,000\nHello transcript.\n"
        }),
    })
    .expect("parse transcript");
    let formatted = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("transcripts.formatSrt"),
        input: parsed.value,
    })
    .expect("format transcript");

    println!(
        "{}",
        serde_json::to_string_pretty(&formatted.value).unwrap()
    );
}
