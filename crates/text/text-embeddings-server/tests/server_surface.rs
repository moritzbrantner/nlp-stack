#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_embeddings_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-embeddings"));
    assert!(response.body.contains("candleDevice"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_embeddings_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"embeddings.embed","input":{"texts":["rust text"],"dimensions":16}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("embeddings.embed"));
}
