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
        "retrieval.search",
        serde_json::json!({"documents": [{"id": "doc-1", "body": "Rust text retrieval"}, {"id": "doc-2", "body": "Video scene reports"}], "query": "text", "mode": "hybrid"}),
    )
    .expect("search");
    assert_eq!(response.value["operation"], "retrieval.search");
    assert!(!response.value["results"].as_array().unwrap().is_empty());
}
