#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        text_classification_cli::LIBRARY_CRATE,
        "text-classification"
    );
    let surface = text_classification_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-classification");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_classification_cli::run_operation(
        "classification.classify",
        serde_json::json!({"text": "rust is reliable", "labels": ["positive", "negative"], "model": {"fallbackPolicy": "lexical_fallback"}}),
    )
    .expect("classify");
    assert_eq!(response.value["operation"], "classification.classify");
    assert!(response.value["summary"].is_object());
}
