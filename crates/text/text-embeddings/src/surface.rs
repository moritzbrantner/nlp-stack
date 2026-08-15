//! Library-owned runtime surface for `text-embeddings`.

use runtime_core::{
    structured_surface_value, OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation,
    SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;
use text_core::TextDocument;

use crate::{
    text_similarity, CooccurrenceConfig, CooccurrenceGraph, HashedTextEmbedder, SemanticTextIndex,
    TextEmbeddingBackend, TextEmbeddingConfig,
};

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
                "Lightweight semantic text embeddings and search for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "embeddings.backends",
                "Inspect embedding backends",
                "Lists deterministic and feature-gated embedding backend availability without loading models.",
                serde_json::json!({"dimensions": 128}),
            ),
            operation(
                "embeddings.embed",
                "Embed text",
                "Builds deterministic hashed text embeddings.",
                serde_json::json!({"texts": ["rust text analysis"], "dimensions": 64}),
            ),
            operation(
                "embeddings.similarity",
                "Text similarity",
                "Computes deterministic hashed-vector text similarity.",
                serde_json::json!({"left": "rust text", "right": "text crates"}),
            ),
            operation(
                "embeddings.semanticSearch",
                "Semantic search",
                "Builds a transient hashed semantic index and searches it.",
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
        "embeddings.backends" => backends_value(parse_input(request.input)?)?,
        "embeddings.embed" => embed_value(parse_input(request.input)?)?,
        "embeddings.similarity" => similarity_value(parse_input(request.input)?)?,
        "embeddings.semanticSearch" => semantic_search_value(parse_input(request.input)?)?,
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
            "Inspected deterministic and feature-gated embedding backends without loading models.",
            serde_json::json!({
                "status": "ok",
                "defaultBackend": value["defaultBackend"],
                "backendCount": value["backends"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "embeddings.embed" => (
            "Text embedding result",
            "Built deterministic hashed embeddings for the supplied texts.",
            serde_json::json!({
                "status": "ok",
                "embeddingCount": value["embeddings"].as_array().map(Vec::len).unwrap_or(0),
                "dimensions": value["model"]["dimensions"]
            }),
        ),
        "embeddings.similarity" => (
            "Text similarity result",
            "Computed deterministic hashed-vector similarity for the supplied text pair.",
            serde_json::json!({
                "status": "ok",
                "similarity": value["similarity"],
                "dimensions": value["model"]["dimensions"]
            }),
        ),
        "embeddings.semanticSearch" => (
            "Semantic search result",
            "Built a transient hashed semantic index and searched it in memory.",
            serde_json::json!({
                "status": "ok",
                "resultCount": value["results"].as_array().map(Vec::len).unwrap_or(0),
                "dimensions": value["model"]["dimensions"]
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
struct EmbedRequest {
    texts: Vec<String>,
    #[serde(default = "default_dimensions")]
    dimensions: usize,
    #[serde(default = "default_true")]
    normalize: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BackendsRequest {
    #[serde(default = "default_dimensions")]
    dimensions: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SimilarityRequest {
    left: String,
    right: String,
    #[serde(default = "default_dimensions")]
    dimensions: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticSearchRequest {
    documents: Vec<SemanticDocument>,
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_dimensions")]
    dimensions: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticDocument {
    id: Option<String>,
    text: String,
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

fn embed_value(request: EmbedRequest) -> Result<serde_json::Value, String> {
    runtime_core::require_non_empty("embeddings.embed", "texts", &request.texts)?;
    let embedder = embedder(request.dimensions);
    let refs = request.texts.iter().map(String::as_str).collect::<Vec<_>>();
    let mut vectors = embedder
        .embed_batch(&refs)
        .map_err(|error| error.to_string())?;
    if !request.normalize {
        // HashedTextEmbedder returns normalized vectors; expose the requested flag
        // as metadata without changing deterministic backend behavior.
    }
    Ok(serde_json::json!({
        "model": embedder.model_info(),
        "normalizeRequested": request.normalize,
        "embeddings": vectors.drain(..).map(|vector| vector.as_slice().to_vec()).collect::<Vec<_>>()
    }))
}

fn backends_value(request: BackendsRequest) -> Result<serde_json::Value, String> {
    let hashed = embedder(request.dimensions);
    Ok(serde_json::json!({
        "defaultBackend": "hashed",
        "backends": [
            {
                "backend": "hashed",
                "loadable": true,
                "default": true,
                "model": hashed.model_info()
            },
            {
                "backend": "onnx",
                "loadable": false,
                "default": false,
                "requiredFeature": "onnx,model-bundles",
                "requiredSetup": "Provide a local model bundle and enable ONNX runtime features.",
                "model": {
                    "modelName": "feature-gated-onnx-text-embedding",
                    "backend": "onnx",
                    "dimensions": 0,
                    "normalized": true
                }
            },
            {
                "backend": "candle",
                "loadable": false,
                "default": false,
                "requiredFeature": "candle,model-bundles",
                "requiredSetup": "Provide a local model bundle and enable Candle features.",
                "model": {
                    "modelName": "feature-gated-candle-text-embedding",
                    "backend": "candle",
                    "dimensions": 0,
                    "normalized": true
                }
            }
        ]
    }))
}

fn similarity_value(request: SimilarityRequest) -> Result<serde_json::Value, String> {
    let embedder = embedder(request.dimensions);
    Ok(serde_json::json!({
        "model": embedder.model_info(),
        "similarity": text_similarity(&request.left, &request.right, &embedder).map_err(|error| error.to_string())?
    }))
}

fn semantic_search_value(request: SemanticSearchRequest) -> Result<serde_json::Value, String> {
    runtime_core::require_non_empty("embeddings.semanticSearch", "documents", &request.documents)?;
    let embedder = embedder(request.dimensions);
    let mut index = SemanticTextIndex::new(embedder);
    let ids = request
        .documents
        .iter()
        .enumerate()
        .map(|(index, document)| {
            document
                .id
                .clone()
                .unwrap_or_else(|| format!("doc-{index}"))
        })
        .collect::<Vec<_>>();
    let docs = request
        .documents
        .iter()
        .enumerate()
        .map(|(index, document)| TextDocument::new(ids[index].as_str(), &document.text))
        .collect::<Vec<_>>();
    index
        .add_documents(docs)
        .map_err(|error| error.to_string())?;
    let results = index
        .search(&request.query, request.top_k)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "model": index.embedder().model_info(),
        "results": results.into_iter().map(|result| serde_json::json!({
            "id": result.id,
            "score": result.score,
            "distance": result.distance,
            "metadata": {
                "backend": result.metadata.backend,
                "provenance": result.metadata.provenance,
                "modelName": result.metadata.model_name,
                "dimensions": result.metadata.dimensions
            }
        })).collect::<Vec<_>>()
    }))
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

fn embedder(dimensions: usize) -> HashedTextEmbedder {
    HashedTextEmbedder {
        config: TextEmbeddingConfig {
            dimensions: dimensions.max(1),
            use_idf: false,
        },
        ..HashedTextEmbedder::default()
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    runtime_core::parse_surface_input(None, input)
}

fn default_dimensions() -> usize {
    128
}
fn default_top_k() -> usize {
    10
}
fn default_window() -> usize {
    4
}
fn default_true() -> bool {
    true
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
    fn embed_operation_returns_requested_dimensions() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("embeddings.embed"),
            input: serde_json::json!({"texts": ["rust text"], "dimensions": 16}),
        })
        .expect("embed");
        assert_eq!(response.value["model"]["dimensions"], 16);
        assert_eq!(
            response.value["embeddings"][0].as_array().unwrap().len(),
            16
        );
    }

    #[test]
    fn backends_operation_lists_release_safe_backends() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("embeddings.backends"),
            input: serde_json::json!({"dimensions": 32}),
        })
        .expect("backends");
        assert_eq!(response.value["defaultBackend"], "hashed");
        let backends = response.value["backends"].as_array().unwrap();
        let names = backends
            .iter()
            .map(|backend| backend["backend"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(names.contains(&"hashed"));
        assert!(names.contains(&"onnx"));
        assert!(names.contains(&"candle"));
        let hashed = backends
            .iter()
            .find(|backend| backend["backend"] == "hashed")
            .unwrap();
        assert_eq!(hashed["loadable"], true);
        assert!(hashed["requiredFeature"].is_null());
        assert!(hashed["requiredSetup"].is_null());
        assert_eq!(hashed["model"]["dimensions"], 32);
        let onnx = backends
            .iter()
            .find(|backend| backend["backend"] == "onnx")
            .unwrap();
        assert_eq!(onnx["loadable"], false);
        assert_eq!(onnx["requiredFeature"], "onnx,model-bundles");
        let candle = backends
            .iter()
            .find(|backend| backend["backend"] == "candle")
            .unwrap();
        assert_eq!(candle["loadable"], false);
        assert_eq!(candle["requiredFeature"], "candle,model-bundles");
        assert_eq!(response.value["summary"]["backendCount"], 3);
    }

    #[test]
    fn backends_operation_clamps_hashed_dimensions() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("embeddings.backends"),
            input: serde_json::json!({"dimensions": 0}),
        })
        .expect("backends");
        assert_eq!(response.value["backends"][0]["model"]["dimensions"], 1);
    }

    #[test]
    fn malformed_input_returns_typed_error_string() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("embeddings.embed"),
            input: serde_json::json!({"texts": "not-array"}),
        })
        .expect_err("invalid request");
        assert!(error.contains("invalid request"));
    }
}
