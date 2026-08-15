#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(text_embeddings_cli::LIBRARY_CRATE, "text-embeddings");
    let surface = text_embeddings_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-embeddings");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    let response = text_embeddings_cli::run_operation(
        "embeddings.embed",
        serde_json::json!({"texts": ["rust text"], "dimensions": 16}),
    )
    .expect("embed");
    assert_eq!(response.value["operation"], "embeddings.embed");
    assert_eq!(
        response.value["embeddings"][0].as_array().unwrap().len(),
        16
    );
}
