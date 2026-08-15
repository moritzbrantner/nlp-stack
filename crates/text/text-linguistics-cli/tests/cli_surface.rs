#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(text_linguistics_cli::LIBRARY_CRATE, "text-linguistics");
    let surface = text_linguistics_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-linguistics");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_linguistics_cli::run_operation(
        "linguistics.analyze",
        serde_json::json!({"text": "Alice presented the tokenizer roadmap in Berlin.", "profile": "fast"}),
    )
    .expect("analyze");
    assert_eq!(response.value["operation"], "linguistics.analyze");
    assert!(!response.value["tokens"].as_array().unwrap().is_empty());
}
