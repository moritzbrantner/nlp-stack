use runtime_core::{OperationId, SurfaceRequest};
use text_analysis::surface::{package_surface, run_surface_operation};

fn run(operation: &str, input: serde_json::Value) -> Result<serde_json::Value, String> {
    run_surface_operation(SurfaceRequest {
        operation: OperationId::new(operation),
        input,
    })
    .map(|response| response.value)
}

#[test]
fn describe_alias_reports_operation_inventory() {
    let value = run("describe", serde_json::json!({"includeOperations": true})).unwrap();

    assert_eq!(value["library"], "moenarch-text-analysis");
    assert_eq!(value["operationCount"], package_surface().operations.len());
    assert!(value["operations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|operation| operation == "analysis.corpus"));
}

#[test]
fn document_surface_parses_modes_and_option_overrides() {
    let value = run(
        "analysis.document",
        serde_json::json!({
            "id": "doc-surface",
            "text": "Rust packages expose text APIs.",
            "languageHint": "en",
            "keywordLimit": 2,
            "summarySentences": 1,
            "ngramSizes": [2],
            "shingleSizes": [2],
            "includeAnnotationGraph": false,
            "linguistics": {"mode": "off"},
            "embedding": {"mode": "off"}
        }),
    )
    .unwrap();

    assert_eq!(value["id"], "doc-surface");
    assert_eq!(value["language"], "en");
    assert!(value["core"]["annotationGraph"].is_null());
    assert!(value["linguistic"].is_null());
    assert!(value["embedding"].is_null());
    assert_eq!(value["similarity"]["tokenShingleCounts"][0]["n"], 2);
    assert_eq!(value["summary"]["id"], "doc-surface");
    assert_eq!(value["summary"]["language"], "en");
    assert!(value["summary"]["tokenCount"].as_u64().unwrap() > 0);
    assert!(value["summary"]["sentenceCount"].as_u64().unwrap() > 0);
    assert_eq!(value["summary"]["keywordCount"], 2);
    assert_eq!(
        value["summary"]["embeddingDimensions"],
        serde_json::Value::Null
    );
    assert_eq!(value["summary"]["diagnosticCount"], 0);
}

#[test]
fn local_model_surface_defaults_do_not_auto_download() {
    let value = run(
        "analysis.document",
        serde_json::json!({
            "id": "doc",
            "text": "Alice works at OpenAI in Berlin.",
            "profile": "modelBacked",
            "linguistics": {
                "mode": "localModel",
                "bundleDir": std::env::temp_dir()
                    .join("text-analysis-missing-ner-bundle")
                    .to_string_lossy()
                    .to_string()
            },
            "embedding": {"mode": "off"}
        }),
    )
    .unwrap();

    assert!(value["linguistic"].is_null());
    assert!(value["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .any(|diagnostic| diagnostic["code"] == "linguistics_unavailable"));
}

#[test]
fn corpus_surface_generates_missing_ids_and_honors_toggles() {
    let value = run(
        "analysis.corpus",
        serde_json::json!({
            "documents": [
                {"text": "rust text analysis"},
                {"id": "provided", "text": "video scene analysis"}
            ],
            "includeNearDuplicates": false,
            "includeSemanticNeighbors": false,
            "embedding": {"mode": "off"}
        }),
    )
    .unwrap();

    assert_eq!(value["documents"][0]["id"], "doc-0");
    assert_eq!(value["documents"][1]["id"], "provided");
    assert!(value["tfidfSearch"].is_null());
    assert!(value["bm25Search"].is_null());
    assert_eq!(value["nearDuplicates"].as_array().unwrap().len(), 0);
    assert_eq!(value["semanticNeighbors"].as_array().unwrap().len(), 0);
    assert_eq!(value["summary"]["documentCount"], 2);
    assert_eq!(
        value["summary"]["termCount"].as_u64().unwrap(),
        value["termStats"].as_array().unwrap().len() as u64
    );
    assert_eq!(value["summary"]["resultCount"], 0);
    assert_eq!(value["summary"]["nearDuplicateCount"], 0);
    assert_eq!(value["summary"]["semanticNeighborCount"], 0);
    assert_eq!(value["summary"]["diagnosticCount"], 0);
}

#[test]
fn similarity_surface_supports_character_alias_and_clamps_n() {
    let value = run(
        "analysis.similarity",
        serde_json::json!({
            "left": "scene",
            "right": "scene cut",
            "n": 0,
            "mode": "char"
        }),
    )
    .unwrap();

    assert_eq!(value["mode"], "character");
    assert_eq!(value["n"], 1);
    assert!(value["similarity"]["jaccard"].as_f64().unwrap() > 0.0);
    assert_eq!(value["summary"]["mode"], "character");
    assert_eq!(value["summary"]["n"], 1);
    assert!(value["summary"]["score"].as_f64().unwrap() > 0.0);
    assert_eq!(
        value["summary"]["leftCount"],
        value["similarity"]["leftCount"]
    );
    assert_eq!(
        value["summary"]["rightCount"],
        value["similarity"]["rightCount"]
    );
    assert_eq!(
        value["summary"]["intersectionCount"],
        value["similarity"]["intersectionCount"]
    );
    assert_eq!(
        value["summary"]["unionCount"],
        value["similarity"]["unionCount"]
    );
}

#[test]
fn workflow_operation_examples_return_structured_values() {
    let surface = package_surface();
    for operation_id in [
        "analysis.document",
        "analysis.corpus",
        "analysis.similarity",
    ] {
        let operation = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() == operation_id)
            .expect("workflow operation exists");
        let value = run(operation_id, operation.example_request.clone()).unwrap();

        assert!(
            value["title"].is_string(),
            "{operation_id} should include a title"
        );
        assert!(
            value["message"].is_string(),
            "{operation_id} should include a message"
        );
        assert!(
            value["summary"].is_object(),
            "{operation_id} should include summary"
        );
        assert!(
            value["result"].is_object(),
            "{operation_id} should include result"
        );
        assert_eq!(value["summary"]["status"], "ok");
    }
}

#[test]
fn surface_errors_are_actionable() {
    let invalid_request = run("analysis.document", serde_json::json!({})).unwrap_err();
    assert!(invalid_request.contains("invalid request"));

    let unsupported_operation = run("analysis.missing", serde_json::json!({})).unwrap_err();
    assert!(unsupported_operation.contains("unsupported operation"));

    let unsupported_mode = run(
        "analysis.similarity",
        serde_json::json!({
            "left": "a",
            "right": "b",
            "mode": "unsupported"
        }),
    )
    .unwrap_err();
    assert!(unsupported_mode.contains("unsupported similarity mode"));
}
