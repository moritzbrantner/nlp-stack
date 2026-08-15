#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_generation_linguistics_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-generation-linguistics"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_generation_linguistics_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"generationLinguistics.synthesizeFromAnalysis","input":{"id":"analysis-doc","text":"Alice presented the tokenizer roadmap in Berlin."}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response
        .body
        .contains("generationLinguistics.synthesizeFromAnalysis"));
}
