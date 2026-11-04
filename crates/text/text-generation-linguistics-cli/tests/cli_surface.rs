#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        text_generation_linguistics_cli::LIBRARY_CRATE,
        "text-generation-linguistics"
    );
    let surface = text_generation_linguistics_cli::package_surface();
    assert_eq!(surface.library, "moenarch-text-generation-linguistics");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_default_workflow() {
    // The analysis needs the linguistic models, which this build does not include.
    let error = text_generation_linguistics_cli::run_operation(
        "generationLinguistics.synthesizeFromAnalysis",
        serde_json::json!({"id": "analysis-doc", "text": "Alice presented the tokenizer roadmap in Berlin."}),
    )
    .expect_err("no linguistic models in this build");
    assert!(error.to_string().contains("missing_dependency"), "{error}");
}
