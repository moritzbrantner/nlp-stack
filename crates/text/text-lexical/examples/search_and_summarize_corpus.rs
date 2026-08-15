use runtime_core::{OperationId, SurfaceRequest};
use text_lexical::surface::run_surface_operation;

fn main() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("lexical.corpusSearch"),
        input: serde_json::json!({
            "documents": [
                {"id": "doc-1", "text": "rust text analysis and tokenizer design"},
                {"id": "doc-2", "text": "video scenes and transcript evidence"},
                {"id": "doc-3", "text": "lexical search over captions"}
            ],
            "query": "text search",
            "mode": "bm25"
        }),
    })
    .expect("search corpus");

    println!("{}", serde_json::to_string_pretty(&response.value).unwrap());
}
