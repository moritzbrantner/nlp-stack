#[test]
fn health_endpoint_reports_wrapped_library() {
    let response = text_analysis_server::response_for("GET", "/health", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-analysis"));
    assert!(response.body.contains("candleDevice"));
}

#[test]
fn operations_endpoint_lists_document_analysis() {
    let response = text_analysis_server::response_for("GET", "/api/operations", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("analysis.document"));
}

#[test]
fn document_endpoint_calls_library_surface() {
    let response = text_analysis_server::response_for(
        "POST",
        "/api/analysis.document",
        r#"{"id":"doc-1","text":"Rust crates analyze text."}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("lexical"));
}

#[test]
fn malformed_json_returns_diagnostic() {
    let response = text_analysis_server::response_for("POST", "/api/run", "{");
    assert_eq!(response.status_code, 400);
    assert!(response.body.contains("invalid_request"));
}
