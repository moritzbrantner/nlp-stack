#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(text_retrieval_cli::LIBRARY_CRATE, "text-retrieval");
    let surface = text_retrieval_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-retrieval");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_retrieval_cli::run_operation(
        "retrieval.chunk",
        serde_json::json!({"documents": [{"id": "doc-1", "body": "Rust text retrieval. Hybrid search."}]}),
    )
    .expect("chunk");
    assert_eq!(response.value["operation"], "retrieval.chunk");
}

#[test]
fn cli_adapter_reports_model_required_for_search() {
    let error = text_retrieval_cli::run_operation(
        "retrieval.search",
        serde_json::json!({"documents": [{"id": "doc-1", "body": "Rust text retrieval"}], "query": "text", "mode": "hybrid"}),
    )
    .expect_err("no embedding model is configured");
    assert!(error.to_string().contains("missing_dependency"), "{error}");
}
