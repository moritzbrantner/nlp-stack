#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_transcripts_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-transcripts"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_transcripts_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"transcripts.parse","input":{"format":"srt","content":"1\n00:00:01,000 --> 00:00:02,000\nHello.\n"}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("transcripts.parse"));
}
