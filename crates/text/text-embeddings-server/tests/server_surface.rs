#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = text_embeddings_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("text-embeddings"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = text_embeddings_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"embeddings.relatedTerms","input":{"text":"rust text crates make text analysis reliable","term":"text"}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("embeddings.relatedTerms"));
}

#[test]
fn run_endpoint_reports_model_required() {
    let response = text_embeddings_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"embeddings.embed","input":{"texts":["rust text"]}}"#,
    );
    assert_ne!(response.status_code, 200);
    assert!(
        response.body.contains("missing_dependency"),
        "{}",
        response.body
    );
}
