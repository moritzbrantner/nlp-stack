//! Library-owned runtime surface for `text-retrieval`.

use runtime_core::{
    structured_surface_value, OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation,
    SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;
use text_core::TextProcessingOptions;

use crate::{
    chunk_search_document, rerank_documents, ChunkingOptions, IngestReport, RerankRequest,
    SearchDocument, SearchFilter,
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
                "Library-first retrieval. Index-backed operations need a caller-supplied embedding model; the surface has none and reports `missing_dependency`.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "retrieval.chunk",
                "Chunk documents",
                "Chunks search documents without writing persistence artifacts.",
                serde_json::json!({"documents": [{"id": "doc-1", "body": "Rust text retrieval. Hybrid search."}]}),
            ),
            operation(
                "retrieval.search",
                "Search documents",
                "Searches an embedding-backed retrieval index. The surface cannot load an embedding model, so this fails with `missing_dependency`; use text-index `index.search` in lexical mode for model-free search.",
                serde_json::json!({"documents": [{"id": "doc-1", "body": "Rust text retrieval"}, {"id": "doc-2", "body": "Video scene reports"}], "query": "text", "mode": "hybrid"}),
            ),
            operation(
                "retrieval.rerank",
                "Rerank documents",
                "Reranks query/document pairs using imported scores or deterministic lexical overlap.",
                serde_json::json!({"query": "rust", "documents": ["rust text", "video scenes"]}),
            ),
            operation(
                "retrieval.snapshotPlan",
                "Plan persisted snapshot",
                "Plans an embedding-backed retrieval snapshot. The surface cannot load an embedding model, so this fails with `missing_dependency`.",
                serde_json::json!({"documents": [{"id": "doc-1", "body": "Rust text retrieval"}, {"id": "doc-2", "body": "Video scene reports"}], "previewLimit": 3}),
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
    let mut operation = runtime_core::surface_operation(id, name, description, example_request);
    if let Some(contract) = landscape_contract(id) {
        runtime_core::attach_landscape_contract(&mut operation, contract);
    }
    operation
}

fn landscape_contract(id: &str) -> Option<runtime_core::landscape::LandscapeOperationContract> {
    match id {
        "retrieval.search" => Some(runtime_core::landscape::LandscapeOperationContract::new(
            runtime_core::landscape::LandscapeFunction::new(
                "text.retrieval.search",
                env!("CARGO_PKG_NAME"),
            )
            .input(runtime_core::landscape::LandscapePort::new(
                "query",
                runtime_core::landscape::well_known::text_retrieval_query(),
            ))
            .input(
                runtime_core::landscape::LandscapePort::new(
                    "documents",
                    runtime_core::landscape::well_known::text_document(),
                )
                .many(),
            )
            .output(
                runtime_core::landscape::LandscapePort::new(
                    "results",
                    runtime_core::landscape::well_known::text_search_result(),
                )
                .many(),
            ),
        )),
        _ => None,
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "retrieval.chunk" => chunk_value(parse_input(request.input)?)?,
        "retrieval.search" => search_value(parse_input(request.input)?)?,
        "retrieval.rerank" => serde_json::to_value(rerank_value(parse_input(request.input)?)?)
            .map_err(|error| error.to_string())?,
        "retrieval.snapshotPlan" => snapshot_plan_value(parse_input(request.input)?)?,
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
            "Inspected the text-retrieval package operations and runtime support.",
            serde_json::json!({
                "status": "ok",
                "operationCount": value["operationCount"]
            }),
        ),
        "retrieval.chunk" => (
            "Document chunking result",
            "Chunked search documents in memory without writing persistence artifacts.",
            serde_json::json!({
                "status": "ok",
                "documentCount": value["report"]["documentsReceived"],
                "chunkCount": value["chunks"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "retrieval.search" => (
            "Retrieval search result",
            "Built a transient in-memory retrieval index and searched it.",
            serde_json::json!({
                "status": "ok",
                "mode": value["mode"],
                "indexedChunks": value["report"]["chunksIndexed"],
                "resultCount": value["results"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "retrieval.rerank" => (
            "Document reranking result",
            "Reranked query/document pairs using imported scores or deterministic lexical overlap.",
            serde_json::json!({
                "status": "ok",
                "query": value["query"],
                "resultCount": value["results"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "retrieval.snapshotPlan" => (
            "Retrieval snapshot plan",
            "Built a transient retrieval index and returned manifest, file metadata, and preview records without writing files.",
            serde_json::json!({
                "status": "ok",
                "chunkCount": value["manifest"]["chunkCount"],
                "vectorCount": value["manifest"]["vectorCount"],
                "dimensions": value["manifest"]["dimensions"],
                "fileCount": value["files"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        _ => (
            "Text retrieval result",
            "Ran a text-retrieval package operation.",
            serde_json::json!({"status": "ok"}),
        ),
    };
    structured_surface_value(operation, title, message, summary, value)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChunkRequest {
    documents: Vec<SearchDocument>,
    #[serde(default)]
    options: ChunkingOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RetrievalSearchRequest {
    documents: Vec<SearchDocument>,
    query: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    filters: Vec<SearchFilter>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SnapshotPlanRequest {
    documents: Vec<SearchDocument>,
    #[serde(default)]
    options: ChunkingOptions,
    #[serde(default = "default_preview_limit")]
    preview_limit: usize,
}

fn chunk_value(request: ChunkRequest) -> Result<serde_json::Value, String> {
    runtime_core::require_non_empty("retrieval.chunk", "documents", &request.documents)?;
    let processing = TextProcessingOptions::default();
    let mut chunks = Vec::new();
    for document in &request.documents {
        chunks.extend(
            chunk_search_document(document, &request.options, &processing)
                .map_err(|error| error.to_string())?,
        );
    }
    let report = IngestReport {
        documents_received: request.documents.len(),
        documents_replaced: 0,
        documents_skipped: request
            .documents
            .iter()
            .filter(|document| document.body.trim().is_empty())
            .count(),
        chunks_indexed: chunks.len(),
    };
    Ok(serde_json::json!({ "chunks": chunks, "report": report }))
}

fn search_value(request: RetrievalSearchRequest) -> Result<serde_json::Value, String> {
    runtime_core::require_non_empty("retrieval.search", "documents", &request.documents)?;
    let _ = (request.query, request.top_k, request.filters);
    parse_mode(&request.mode)?;
    Err(embedding_model_required("retrieval.search"))
}

/// The retrieval index embeds every chunk, and the surface has no way to configure
/// an embedding model, so index-backed operations fail instead of falling back to
/// a non-semantic stand-in.
fn embedding_model_required(operation: &str) -> String {
    runtime_core::SurfaceError::missing_dependency(
        Some(operation),
        "embedding model",
        "build a RetrievalIndex with a real embedding backend (for example `text_embeddings::load_embedding_preset`) through the typed text-retrieval API, or use text-index `index.search` (lexical, or semantic/hybrid with caller-computed `importedEmbeddings`); no heuristic embedding fallback exists",
    )
    .to_error_string()
}

fn rerank_value(request: RerankRequest) -> Result<crate::RerankResponse, String> {
    runtime_core::require_non_empty("retrieval.rerank", "documents", &request.documents)?;
    rerank_documents(request).map_err(|error| error.to_string())
}

fn snapshot_plan_value(request: SnapshotPlanRequest) -> Result<serde_json::Value, String> {
    runtime_core::require_non_empty("retrieval.snapshotPlan", "documents", &request.documents)?;
    let _ = (request.options, request.preview_limit);
    Err(embedding_model_required("retrieval.snapshotPlan"))
}

fn parse_mode(mode: &str) -> Result<(), String> {
    match mode {
        "full_text" | "fullText" | "full-text" | "semantic" | "hybrid" => Ok(()),
        other => Err(format!("unsupported retrieval mode `{other}`")),
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    runtime_core::parse_surface_input(None, input)
}

fn default_mode() -> String {
    "hybrid".to_string()
}
fn default_top_k() -> usize {
    10
}
fn default_preview_limit() -> usize {
    5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_retrieval_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"retrieval.chunk".to_string()));
        assert!(ids.contains(&"retrieval.search".to_string()));
        assert!(ids.contains(&"retrieval.rerank".to_string()));
        assert!(ids.contains(&"retrieval.snapshotPlan".to_string()));
    }

    #[test]
    fn index_backed_operations_require_an_embedding_model() {
        for (operation, input) in [
            (
                "retrieval.search",
                serde_json::json!({
                    "documents": [{"id": "doc-1", "body": "rust text retrieval"}],
                    "query": "text",
                    "mode": "full_text"
                }),
            ),
            (
                "retrieval.snapshotPlan",
                serde_json::json!({"documents": [{"id": "doc-1", "body": "Rust text retrieval"}]}),
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
    fn malformed_input_returns_typed_error_string() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("retrieval.search"),
            input: serde_json::json!({"documents": []}),
        })
        .expect_err("invalid request");
        assert!(error.contains("invalid request"));
    }

    #[test]
    fn snapshot_plan_rejects_empty_documents() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("retrieval.snapshotPlan"),
            input: serde_json::json!({"documents": []}),
        })
        .expect_err("invalid request");
        assert!(error.contains("invalid_request"));
    }
}
