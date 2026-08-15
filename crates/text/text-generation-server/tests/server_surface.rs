#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_generation_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-generation"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_generation_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"generation.markovGenerate","input":{"trainingTexts":["rust text analysis supports crates"],"order":2,"maxTokens":6}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("generation.markovGenerate"));
}
