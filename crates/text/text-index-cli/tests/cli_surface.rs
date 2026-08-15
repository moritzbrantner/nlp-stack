#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(text_index_cli::LIBRARY_CRATE, "text-index");
    let surface = text_index_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-index");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_index_cli::run_operation(
        "index.search",
        serde_json::json!({
            "documents": [
                {"id": "doc-1", "body": "Rust durable text indexing needs stable adapters"},
                {"id": "doc-2", "body": "Video scene reports mention adapters separately"}
            ],
            "query": {
                "text": "text indexing stable adapters",
                "topK": 2,
                "requiredPhrases": ["stable adapters"]
            }
        }),
    )
    .expect("search");
    assert_eq!(response.value["operation"], "index.search");
    let results = response.value["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["documentId"], serde_json::json!("doc-1"));
    assert_eq!(
        results[0]["matchedPhrases"],
        serde_json::json!(["stable adapters"])
    );
}
