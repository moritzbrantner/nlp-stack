#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(text_generation_cli::LIBRARY_CRATE, "text-generation");
    let surface = text_generation_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-generation");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_generation_cli::run_operation(
        "generation.markovGenerate",
        serde_json::json!({"trainingTexts": ["rust text analysis supports crates"], "order": 2, "maxTokens": 6}),
    )
    .expect("generate");
    assert_eq!(response.value["operation"], "generation.markovGenerate");
    assert!(response.value["summary"].is_object());
}
