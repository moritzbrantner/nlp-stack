#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_retrieval_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-retrieval"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_retrieval_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"retrieval.search","input":{"documents":[{"id":"doc-1","body":"Rust text retrieval"},{"id":"doc-2","body":"Video scene reports"}],"query":"text","mode":"hybrid"}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("retrieval.search"));
}
