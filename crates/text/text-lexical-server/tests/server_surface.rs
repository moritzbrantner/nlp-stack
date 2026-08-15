#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_lexical_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-lexical"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_lexical_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"lexical.analyze","input":{"text":"Rust crates make text analysis reliable.","maxTerms":5}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("lexical.analyze"));
}
