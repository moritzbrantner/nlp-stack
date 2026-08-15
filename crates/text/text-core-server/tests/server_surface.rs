#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_core_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-core"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_core_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"text.tokenize","input":{"text":"Hello Berlin.","includePunctuation":true}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text.tokenize"));
}
