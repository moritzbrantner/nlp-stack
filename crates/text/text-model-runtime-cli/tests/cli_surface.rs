#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(text_model_runtime_cli::LIBRARY_CRATE, "text-model-runtime");
    let surface = text_model_runtime_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-model-runtime");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_model_runtime_cli::run_operation(
        "runtime.tokenizeSummary",
        serde_json::json!({"text": "Rust text runtime", "maxTokens": 8}),
    )
    .expect("tokenize summary");
    assert_eq!(response.value["operation"], "runtime.tokenizeSummary");
    assert!(!response.value["tokens"].as_array().unwrap().is_empty());
}
