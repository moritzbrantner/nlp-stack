#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(text_lexical_cli::LIBRARY_CRATE, "text-lexical");
    let surface = text_lexical_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-lexical");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_lexical_cli::run_operation(
        "lexical.analyze",
        serde_json::json!({"text": "Rust crates make text analysis reliable.", "maxTerms": 5}),
    )
    .expect("analyze");
    assert_eq!(response.value["operation"], "lexical.analyze");
    assert!(!response.value["keywords"].as_array().unwrap().is_empty());
}
