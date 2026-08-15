#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(text_transcripts_cli::LIBRARY_CRATE, "text-transcripts");
    let surface = text_transcripts_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-transcripts");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_transcripts_cli::run_operation(
        "transcripts.parse",
        serde_json::json!({"format": "srt", "content": "1\n00:00:01,000 --> 00:00:02,000\nHello.\n"}),
    )
    .expect("parse");
    assert_eq!(response.value["operation"], "transcripts.parse");
    assert_eq!(response.value["segments"].as_array().unwrap().len(), 1);
}
