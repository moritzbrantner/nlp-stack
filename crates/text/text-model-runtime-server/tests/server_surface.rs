#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_model_runtime_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-model-runtime"));
    assert!(response.body.contains("candleDevice"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_model_runtime_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"runtime.tokenizeSummary","input":{"text":"Rust text runtime","maxTokens":8}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("runtime.tokenizeSummary"));
}
