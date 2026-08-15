#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_question_answering_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-question-answering"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_question_answering_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"qa.answer","input":{"question":"What is reliable?","context":"Rust is reliable.","importedPredictions":[{"text":"Rust","score":0.9}]}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("qa.answer"));
}
