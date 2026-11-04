//! Library-owned runtime surface for `text-embeddings`.

use runtime_core::SurfaceError;
use runtime_core::{
    structured_surface_value, OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation,
    SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;

use crate::{CooccurrenceConfig, CooccurrenceGraph};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Inspect package metadata",
                "Text embedding backends and co-occurrence statistics. Embedding operations need a configured embedding model.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "embeddings.backends",
                "Inspect embedding backends",
                "Lists feature-gated embedding model backends without loading models. No backend is available by default.",
                serde_json::json!({}),
            ),
            operation(
                "embeddings.embed",
                "Embed text",
                "Embeds text with a configured embedding model; fails with `missing_dependency` when none is configured.",
                serde_json::json!({"texts": ["rust text analysis"], "dimensions": 64}),
            ),
            operation(
                "embeddings.similarity",
                "Text similarity",
                "Computes embedding similarity with a configured embedding model; fails with `missing_dependency` when none is configured.",
                serde_json::json!({"left": "rust text", "right": "text crates"}),
            ),
            operation(
                "embeddings.semanticSearch",
                "Semantic search",
                "Searches documents with a configured embedding model; fails with `missing_dependency` when none is configured.",
                serde_json::json!({"documents": [{"id": "doc-1", "text": "rust text analysis"}, {"id": "doc-2", "text": "video scenes"}], "query": "text"}),
            ),
            operation(
                "embeddings.relatedTerms",
                "Related terms",
                "Scores co-occurring related terms from local text.",
                serde_json::json!({"text": "rust text crates make text analysis reliable", "term": "text"}),
            ),
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

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "embeddings.backends" => backends_value(),
        "embeddings.embed" | "embeddings.similarity" | "embeddings.semanticSearch" => {
            return Err(embedding_model_required(request.operation.as_str()))
        }
        "embeddings.relatedTerms" => related_terms_value(parse_input(request.input)?)?,
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
            "Inspected the text-embeddings package operations and runtime support.",
            serde_json::json!({
                "status": "ok",
                "operationCount": value["operationCount"]
            }),
        ),
        "embeddings.backends" => (
            "Embedding backend catalog",
            "Inspected feature-gated embedding model backends without loading models.",
            serde_json::json!({
                "status": "ok",
                "defaultBackend": value["defaultBackend"],
                "backendCount": value["backends"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "embeddings.relatedTerms" => (
            "Related terms result",
            "Scored local co-occurring terms for the requested term.",
            serde_json::json!({
                "status": "ok",
                "term": value["term"],
                "relatedTermCount": value["relatedTerms"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        _ => (
            "Text embeddings result",
            "Ran a text-embeddings package operation.",
            serde_json::json!({"status": "ok"}),
        ),
    };
    structured_surface_value(operation, title, message, summary, value)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RelatedTermsRequest {
    text: String,
    term: Option<String>,
    #[serde(default = "default_window")]
    window_size: usize,
    #[serde(default = "default_top_k")]
    limit: usize,
}

/// The surface has no way to configure an embedding model, so every embedding
/// operation fails with a typed `missing_dependency` error instead of falling back
/// to a non-semantic stand-in.
fn embedding_model_required(operation: &str) -> String {
    SurfaceError::missing_dependency(
        Some(operation),
        "embedding model",
        "configure an ONNX or Candle embedding model (features `onnx,model-bundles` or `candle,model-bundles`) and call the typed text-embeddings API; no heuristic embedding fallback exists",
    )
    .to_error_string()
}

fn backends_value() -> serde_json::Value {
    serde_json::json!({
        "defaultBackend": null,
        "backends": [
            {
                "backend": "onnx",
                "loadable": false,
                "default": false,
                "requiredFeature": "onnx,model-bundles",
                "requiredSetup": "Provide a local model bundle and enable ONNX runtime features."
            },
            {
                "backend": "candle",
                "loadable": false,
                "default": false,
                "requiredFeature": "candle,model-bundles",
                "requiredSetup": "Provide a local model bundle and enable Candle features."
            }
        ]
    })
}

fn related_terms_value(request: RelatedTermsRequest) -> Result<serde_json::Value, String> {
    let mut graph = CooccurrenceGraph::new(CooccurrenceConfig {
        window_size: request.window_size,
        ..CooccurrenceConfig::default()
    })
    .map_err(|error| error.to_string())?;
    graph.train_text(&request.text);
    let term = request
        .term
        .or_else(|| {
            graph
                .term_counts()
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(term, _)| term.clone())
        })
        .unwrap_or_default();
    let related = graph.related_terms(&term, request.limit);
    Ok(serde_json::json!({
        "term": term,
        "relatedTerms": related.into_iter().map(|term| serde_json::json!({
            "term": term.term,
            "count": term.count,
            "score": term.score
        })).collect::<Vec<_>>()
    }))
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    runtime_core::parse_surface_input(None, input)
}

fn default_top_k() -> usize {
    10
}
fn default_window() -> usize {
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_embedding_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"embeddings.embed".to_string()));
        assert!(ids.contains(&"embeddings.backends".to_string()));
        assert!(ids.contains(&"embeddings.semanticSearch".to_string()));
    }

    #[test]
    fn embedding_operations_require_a_configured_model() {
        for (operation, input) in [
            (
                "embeddings.embed",
                serde_json::json!({"texts": ["rust text"]}),
            ),
            (
                "embeddings.similarity",
                serde_json::json!({"left": "a", "right": "b"}),
            ),
            (
                "embeddings.semanticSearch",
                serde_json::json!({"documents": [{"text": "a"}], "query": "a"}),
            ),
        ] {
            let error = run_surface_operation(SurfaceRequest {
                operation: OperationId::new(operation),
                input,
            })
            .expect_err("no embedding model is configured");
            let error = runtime_core::parse_surface_error(&error).expect("typed error");
            assert_eq!(error.code, "missing_dependency");
            assert_eq!(error.details["dependency"], "embedding model");
        }
    }

    #[test]
    fn backends_operation_lists_no_default_backend() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("embeddings.backends"),
            input: serde_json::json!({}),
        })
        .expect("backends");
        assert!(response.value["defaultBackend"].is_null());
        let backends = response.value["backends"].as_array().unwrap();
        assert!(backends.iter().all(|backend| backend["loadable"] == false));
        assert_eq!(response.value["summary"]["backendCount"], 2);
    }

    #[test]
    fn malformed_input_returns_typed_error_string() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("embeddings.relatedTerms"),
            input: serde_json::json!({"text": ["not-a-string"]}),
        })
        .expect_err("invalid request");
        assert!(error.contains("invalid request"));
    }
}
