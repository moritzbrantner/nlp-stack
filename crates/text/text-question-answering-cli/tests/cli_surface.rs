#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        text_question_answering_cli::LIBRARY_CRATE,
        "text-question-answering"
    );
    let surface = text_question_answering_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-question-answering");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_question_answering_cli::run_operation(
        "qa.answer",
        serde_json::json!({"question": "What is reliable?", "context": "Rust is reliable.", "importedPredictions": [{"text": "Rust", "score": 0.9}]}),
    )
    .expect("answer");
    assert_eq!(response.value["operation"], "qa.answer");
    assert!(response.value["summary"].is_object());
}
