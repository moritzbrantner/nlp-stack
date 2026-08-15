#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_linguistics_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-linguistics"));
    assert!(response.body.contains("candleDevice"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_linguistics_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"linguistics.analyze","input":{"text":"Alice presented the tokenizer roadmap in Berlin.","profile":"fast"}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("linguistics.analyze"));
}
