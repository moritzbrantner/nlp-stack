#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(text_core_cli::LIBRARY_CRATE, "text-core");
    let surface = text_core_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-core");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_core_cli::run_operation(
        "text.tokenize",
        serde_json::json!({"text": "Hello Berlin.", "includePunctuation": true}),
    )
    .expect("tokenize");
    assert_eq!(response.value["operation"], "text.tokenize");
    assert!(response.value["tokens"].as_array().unwrap().len() >= 2);
}
