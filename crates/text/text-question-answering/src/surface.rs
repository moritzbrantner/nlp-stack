//! Library-owned runtime surface for `text-question-answering`.

use runtime_core::{
    add_surface_operation_schema_extension, structured_surface_value, OperationId, PackageSurface,
    RuntimeCapabilities, SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{
    answer_question, answer_question_batch, answer_question_with_retrieval,
    answer_question_with_text_index, model_catalog,
};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation("describe", "Inspect package metadata", "Extractive question-answering contracts, imported span postprocessing, and backend execution hooks.", serde_json::json!({"includeOperations": true})),
            operation("qa.models", "Inspect QA model catalog", "Lists registered extractive question-answering presets.", serde_json::json!({})),
            operation("qa.answer", "Answer question", "Postprocesses imported span predictions for extractive QA.", serde_json::json!({"question": "What is reliable?", "context": "Rust is reliable.", "importedPredictions": [{"text": "Rust", "score": 0.9}]})),
            operation("qa.answerWithIndex", "Answer from text index", "Builds a deterministic in-memory Text Index and returns cited extractive answers.", serde_json::json!({"question": "What language has ownership?", "documents": [{"id": "doc-rust", "body": "Rust has ownership and deterministic package workflows.", "language": "en", "metadata": {"attributes": {"kind": "language"}}}], "indexOptions": {"chunkingStrategy": "tokenWindow", "chunkTokens": 16, "chunkOverlapTokens": 0, "storeRawText": true}, "topKChunks": 2, "topKAnswers": 1, "localModel": {"autoDownload": false}, "fallbackPolicy": "heuristicIfUnavailable"})),
            deprecated_support_operation("qa.answerWithRetrieval", "Answer from compatibility retrieval", "Builds a deterministic in-memory compatibility retrieval index and returns cited extractive answers.", serde_json::json!({"question": "What language has ownership?", "documents": [{"id": "doc-rust", "body": "Rust has ownership and deterministic package workflows."}], "topKChunks": 2, "topKAnswers": 1, "localModel": {"autoDownload": false}, "fallbackPolicy": "heuristicIfUnavailable"})),
            operation("qa.answerBatch", "Answer question batch", "Runs multiple imported-span QA requests and returns item-level successes and errors.", serde_json::json!({"requests": [{"question": "Who presented the roadmap?", "context": "Alice presented the roadmap.", "importedPredictions": [{"text": "Alice", "score": 0.94}]}]})),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    runtime_core::surface_operation(id, name, description, example_request)
}

fn deprecated_support_operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    let mut operation = operation(id, name, description, example_request);
    add_surface_operation_schema_extension(
        &mut operation,
        "xOperationCategory",
        serde_json::json!("support"),
    );
    add_surface_operation_schema_extension(&mut operation, "xDeprecated", serde_json::json!(true));
    add_surface_operation_schema_extension(
        &mut operation,
        "xReplacementOperation",
        serde_json::json!("qa.answerWithIndex"),
    );
    add_surface_operation_schema_extension(
        &mut operation,
        "xDeprecationReason",
        serde_json::json!(
            "Compatibility retrieval remains callable for existing consumers; new cited document QA should use qa.answerWithIndex."
        ),
    );
    operation
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "qa.models" => serde_json::json!({ "models": model_catalog() }),
        "qa.answer" => serde_json::to_value(
            answer_question(runtime_core::parse_surface_input(None, request.input)?)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?,
        "qa.answerWithIndex" => serde_json::to_value(
            answer_question_with_text_index(runtime_core::parse_surface_input(
                None,
                request.input,
            )?)
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?,
        "qa.answerWithRetrieval" => serde_json::to_value(
            answer_question_with_retrieval(runtime_core::parse_surface_input(None, request.input)?)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?,
        "qa.answerBatch" => serde_json::to_value(
            answer_question_batch(runtime_core::parse_surface_input(None, request.input)?)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?,
        operation => {
            return Err(runtime_core::SurfaceError::unsupported_operation(
                operation,
                env!("CARGO_PKG_NAME"),
            )
            .to_error_string())
        }
    };
    let value = annotated_value(&operation, value);
    Ok(SurfaceResponse {
        operation,
        value,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    })
}

fn describe_value(input: serde_json::Value) -> serde_json::Value {
    let surface = package_surface();
    serde_json::json!({
        "library": surface.library,
        "version": surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface.operations.iter().map(|operation| operation.id.as_str()).collect::<Vec<_>>(),
        "input": input
    })
}

fn annotated_value(operation: &OperationId, value: serde_json::Value) -> serde_json::Value {
    let (title, message, summary) = match operation.as_str() {
        "describe" => (
            "Package surface metadata",
            "Inspected the text-question-answering package operations and runtime support.",
            serde_json::json!({
                "status": "ok",
                "operationCount": value["operationCount"]
            }),
        ),
        "qa.models" => (
            "QA model catalog",
            "Inspected registered extractive question-answering model presets without executing a model.",
            serde_json::json!({
                "status": "ok",
                "modelCount": value["models"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "qa.answer" => (
            "Question answering result",
            "Postprocessed imported extractive QA span predictions for the supplied question and context.",
            serde_json::json!({
                "status": "ok",
                "answerCount": value["answers"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "qa.answerWithRetrieval" => (
            "Compatibility retrieval question answering result",
            "Built a deterministic compatibility retrieval index from supplied documents and returned cited answers.",
            serde_json::json!({
                "status": "ok",
                "backend": "compatibility-retrieval",
                "retrievedChunkCount": value["retrievedChunks"].as_array().map(Vec::len).unwrap_or(0),
                "answerCount": value["answers"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "qa.answerWithIndex" => (
            "Text Index question answering result",
            "Built a deterministic in-memory Text Index from supplied documents and returned cited answers.",
            serde_json::json!({
                "status": "ok",
                "backend": "text-index",
                "retrievedChunkCount": value["retrievedChunks"].as_array().map(Vec::len).unwrap_or(0),
                "answerCount": value["answers"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "qa.answerBatch" => (
            "Batch question answering result",
            "Ran imported-span QA requests with item-level success and error reporting.",
            serde_json::json!({
                "status": "ok",
                "successes": value["successes"],
                "failures": value["failures"]
            }),
        ),
        _ => (
            "Question answering result",
            "Ran a text-question-answering package operation.",
            serde_json::json!({"status": "ok"}),
        ),
    };
    structured_surface_value(operation, title, message, summary, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_qa_operations_only() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"qa.models".to_string()));
        assert!(ids.contains(&"qa.answer".to_string()));
        assert!(ids.contains(&"qa.answerWithIndex".to_string()));
        assert!(ids.contains(&"qa.answerWithRetrieval".to_string()));
        assert!(ids.contains(&"qa.answerBatch".to_string()));
        assert!(!ids.contains(&"nlp.sentiment".to_string()));
    }

    #[test]
    fn answer_with_index_surface_returns_cited_answers() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "qa.answerWithIndex".into(),
            input: serde_json::json!({
                "question": "What language has ownership?",
                "documents": [
                    {
                        "id": "doc-rust",
                        "body": "Rust has ownership and deterministic package workflows.",
                        "language": "en",
                        "metadata": {
                            "attributes": {
                                "kind": "language"
                            }
                        }
                    }
                ],
                "indexOptions": {
                    "chunkingStrategy": "tokenWindow",
                    "chunkTokens": 16,
                    "chunkOverlapTokens": 0,
                    "storeRawText": true
                },
                "topKChunks": 2,
                "topKAnswers": 1,
                "localModel": { "autoDownload": false },
                "fallbackPolicy": "heuristicIfUnavailable"
            }),
        })
        .expect("answer with text index");

        assert_eq!(response.operation.as_str(), "qa.answerWithIndex");
        assert_eq!(response.value["operation"], "qa.answerWithIndex");
        assert_eq!(response.value["summary"]["status"], "ok");
        assert_eq!(response.value["summary"]["backend"], "text-index");
        assert!(
            response.value["summary"]["retrievedChunkCount"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(
            response.value["result"]["answers"][0]["citations"][0]["documentId"],
            "doc-rust"
        );
    }

    #[test]
    fn answer_with_retrieval_surface_is_deprecated_support() {
        let operation = package_surface()
            .operations
            .into_iter()
            .find(|operation| operation.id.as_str() == "qa.answerWithRetrieval")
            .expect("retrieval operation");

        assert_eq!(operation.input_schema["xOperationCategory"], "support");
        assert_eq!(operation.output_schema["xOperationCategory"], "support");
        assert_eq!(operation.input_schema["xDeprecated"], true);
        assert_eq!(operation.output_schema["xDeprecated"], true);
        assert_eq!(
            operation.input_schema["xReplacementOperation"],
            "qa.answerWithIndex"
        );
        assert_eq!(
            operation.output_schema["xReplacementOperation"],
            "qa.answerWithIndex"
        );

        let response = run_surface_operation(SurfaceRequest {
            operation: "qa.answerWithRetrieval".into(),
            input: operation.example_request,
        })
        .expect("answer with retrieval remains runnable");
        assert_eq!(response.operation.as_str(), "qa.answerWithRetrieval");
        assert_eq!(
            response.value["summary"]["backend"],
            "compatibility-retrieval"
        );
    }
}
