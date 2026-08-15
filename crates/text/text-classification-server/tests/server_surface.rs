#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_classification_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-classification"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_classification_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"classification.classify","input":{"text":"rust is reliable","labels":["positive","negative"],"model":{"fallbackPolicy":"lexical_fallback"}}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("classification.classify"));
}
