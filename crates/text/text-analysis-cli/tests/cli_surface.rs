#[test]
fn cli_adapter_reports_analysis_operations() {
    assert_eq!(text_analysis_cli::LIBRARY_CRATE, "text-analysis");
    let surface = text_analysis_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-analysis");
    assert!(surface
        .operations
        .iter()
        .any(|operation| operation.id.as_str() == "analysis.document"));
}

#[test]
fn run_operation_returns_document_sections() {
    let response = text_analysis_cli::run_operation(
        "analysis.document",
        serde_json::json!({
            "id": "doc-1",
            "text": "Rust crates analyze text."
        }),
    )
    .unwrap();
    assert_eq!(response.value["operation"], "analysis.document");
    assert!(response.value.get("core").is_some());
    assert!(response.value.get("lexical").is_some());
    assert!(response.value.get("similarity").is_some());
}
