//! Library-owned runtime surface for `text-index`.

mod fuzzy;

use runtime_core::{
    primary_workflow_operation, set_surface_operation_curation, structured_surface_value,
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceError, SurfaceOperation,
    SurfaceOperationCuration, SurfaceRequest, SurfaceResponse, SurfaceRuntimeContext,
};
use serde::Deserialize;
use std::path::PathBuf;

use crate::{
    IndexBuildOptions, IndexDocument, IndexInspectReport, IndexMutationReport, IndexQuery,
    IndexSearchMode, IndexSearchResult, IndexSnapshotPlan, MemoryIndexStore, TextIndex,
    TextIndexStore,
};
use text_embeddings::{ImportedEmbedder, ImportedEmbedding, ImportedEmbeddingModel};

pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation("describe", "Inspect package metadata", "Durable local text indexing with lexical search. Semantic and hybrid search need an embedding model; this surface takes caller-computed `importedEmbeddings`.", serde_json::json!({"includeOperations": true})),
            operation("index.build", "Build index", "Builds a transient in-memory text index unless an explicit committed SQLite backend is requested.", serde_json::json!({"backend": "memory", "documents": [{"id": "doc-1", "body": "Rust text index"}]})),
            operation("index.open", "Open index", "Opens or describes an index backend.", serde_json::json!({"backend": "memory"})),
            operation("index.addDocuments", "Add documents", "Adds documents to a transient memory index or an explicit committed SQLite backend.", serde_json::json!({"backend": "memory", "documents": [{"id": "doc-1", "body": "Rust text index"}]})),
            operation("index.removeDocuments", "Remove documents", "Removes documents only from an explicit committed SQLite backend; memory execution returns a side-effect-free plan.", serde_json::json!({"backend": "memory", "documentIds": ["doc-1"]})),
            operation("index.search", "Search index", "Builds or opens a requested backend and searches it. Lexical mode (BM25, with optional bounded typo-tolerant expansion) needs no model. The surface cannot load an embedding model, so semantic and hybrid modes need caller-computed vectors in `importedEmbeddings`: a `passage` vector for every chunk text and a `query` vector for the query text, plus `embeddingModel` provenance (id, revision, pooling). Without them they fail with `missing_dependency`. Lexical mode ignores imported vectors.", serde_json::json!({"backend": "memory", "documents": [{"id": "doc-1", "body": "Rust text index supports required phrases and fuzzy retrieval"}], "query": {"text": "retrieavl required phrases", "mode": "lexical", "requiredPhrases": ["required phrases"], "fuzzy": {"maxEditDistance": 1, "minTermLength": 4, "maxExpansionsPerTerm": 3, "maxQueryVariants": 16, "maxVocabularyTerms": 20000, "fuzzyWeight": 0.8}}, "embeddingModel": {"name": "intfloat/multilingual-e5-base", "revision": "d128750597153bb5987e10b1c3493a34e5a4502a", "pooling": "mean", "dimensions": 3}, "importedEmbeddings": [{"text": "Rust text index supports required phrases and fuzzy retrieval", "kind": "passage", "vector": [0.9, 0.1, 0.0]}, {"text": "retrieavl required phrases", "kind": "query", "vector": [0.8, 0.2, 0.0]}]})),
            operation("index.inspect", "Inspect index", "Builds or opens a requested backend and returns counts.", serde_json::json!({"backend": "memory", "documents": [{"id": "doc-1", "body": "Rust text index"}]})),
            operation("index.snapshotPlan", "Plan snapshot", "Builds or opens a requested backend and returns a snapshot plan.", serde_json::json!({"backend": "memory", "documents": [{"id": "doc-1", "body": "Rust text index"}]})),
        ],
    }
}

pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    run_surface_operation_with_context(
        request,
        &SurfaceRuntimeContext::compatibility_no_side_effects(),
    )
}

pub fn run_surface_operation_with_context(
    request: SurfaceRequest,
    context: &SurfaceRuntimeContext,
) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "index.open" => {
            let input: OpenRequest = parse_input(request.input)?;
            let index = open_index(input.backend, context)?;
            serde_json::json!({
                "backend": index.backend_name(),
                "inspect": index.inspect()?,
                "snapshotPlan": index.snapshot_plan()?
            })
        }
        "index.build" | "index.addDocuments" => {
            let input: BuildRequest = parse_input(request.input)?;
            let (index, report) = build_index(input, context)?;
            serde_json::json!({"backend": index.backend_name(), "report": report, "inspect": index.inspect()?})
        }
        "index.removeDocuments" => {
            let input: RemoveRequest = parse_input(request.input)?;
            remove_documents(input, context)?
        }
        "index.search" => {
            let input: SearchRequest = parse_input(request.input)?;
            validate_search_request(&input)?;
            let embedder = search_embedder(&input)?;
            let (index, _) = build_index_with(
                BuildRequest {
                    documents: input.documents,
                    options: input.options,
                    backend: input.backend,
                },
                embedder,
                context,
            )?;
            let results = if let Some(options) = input.query.fuzzy.as_ref() {
                fuzzy::search(&index, &input.query.query, options)?
            } else {
                serde_json::to_value(index.search(&input.query.query)?)
                    .map_err(|error| error.to_string())?
            };
            serde_json::json!({"backend": index.backend_name(), "results": results})
        }
        "index.inspect" => {
            let input: BuildRequest = parse_input(request.input)?;
            let (index, _) = build_index(input, context)?;
            serde_json::json!(index.inspect()?)
        }
        "index.snapshotPlan" => {
            let input: BuildRequest = parse_input(request.input)?;
            let (index, _) = build_index(input, context)?;
            serde_json::json!(index.snapshot_plan()?)
        }
        other => {
            return Err(runtime_core::SurfaceError::unsupported_operation(
                other,
                env!("CARGO_PKG_NAME"),
            )
            .to_error_string())
        }
    };
    Ok(SurfaceResponse {
        operation: operation.clone(),
        value: annotated_value(&operation, value),
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    })
}

enum SurfaceIndex {
    Memory(TextIndex<ImportedEmbedder, MemoryIndexStore>),
    #[cfg(feature = "sqlite")]
    Sqlite(TextIndex<ImportedEmbedder, crate::SqliteIndexStore>),
}

impl SurfaceIndex {
    fn backend_name(&self) -> &'static str {
        match self {
            Self::Memory(index) => index.store().backend_name(),
            #[cfg(feature = "sqlite")]
            Self::Sqlite(index) => index.store().backend_name(),
        }
    }

    fn upsert_documents(
        &mut self,
        documents: &[IndexDocument],
    ) -> Result<IndexMutationReport, String> {
        match self {
            Self::Memory(index) => index
                .upsert_documents(documents)
                .map_err(|error| error.to_string()),
            #[cfg(feature = "sqlite")]
            Self::Sqlite(index) => index
                .upsert_documents(documents)
                .map_err(|error| error.to_string()),
        }
    }

    fn remove_documents(&mut self, document_ids: &[String]) -> Result<IndexMutationReport, String> {
        match self {
            Self::Memory(index) => index
                .remove_documents(document_ids)
                .map_err(|error| error.to_string()),
            #[cfg(feature = "sqlite")]
            Self::Sqlite(index) => index
                .remove_documents(document_ids)
                .map_err(|error| error.to_string()),
        }
    }

    fn search(&self, query: &IndexQuery) -> Result<Vec<IndexSearchResult>, String> {
        match self {
            Self::Memory(index) => index.search(query).map_err(|error| error.to_string()),
            #[cfg(feature = "sqlite")]
            Self::Sqlite(index) => index.search(query).map_err(|error| error.to_string()),
        }
    }

    fn inspect(&self) -> Result<IndexInspectReport, String> {
        match self {
            Self::Memory(index) => index.inspect().map_err(|error| error.to_string()),
            #[cfg(feature = "sqlite")]
            Self::Sqlite(index) => index.inspect().map_err(|error| error.to_string()),
        }
    }

    fn snapshot_plan(&self) -> Result<IndexSnapshotPlan, String> {
        match self {
            Self::Memory(index) => index.snapshot_plan().map_err(|error| error.to_string()),
            #[cfg(feature = "sqlite")]
            Self::Sqlite(index) => index.snapshot_plan().map_err(|error| error.to_string()),
        }
    }
}

fn build_index(
    input: BuildRequest,
    context: &SurfaceRuntimeContext,
) -> Result<(SurfaceIndex, IndexMutationReport), String> {
    build_index_with(input, None, context)
}

/// `embedder` is `None` for a lexical-only index, which stores no vectors.
fn build_index_with(
    input: BuildRequest,
    embedder: Option<ImportedEmbedder>,
    context: &SurfaceRuntimeContext,
) -> Result<(SurfaceIndex, IndexMutationReport), String> {
    let documents = input.documents;
    runtime_core::validate_max_items("index.search", "documents", documents.len(), 1024)?;
    let mut index = open_configured_index(input.backend, input.options, embedder, context)?;
    let report = index.upsert_documents(&documents)?;
    Ok((index, report))
}

fn open_index(
    backend: BackendRequest,
    context: &SurfaceRuntimeContext,
) -> Result<SurfaceIndex, String> {
    open_configured_index(backend, IndexBuildOptions::default(), None, context)
}

fn open_configured_index(
    backend: BackendRequest,
    mut options: IndexBuildOptions,
    embedder: Option<ImportedEmbedder>,
    context: &SurfaceRuntimeContext,
) -> Result<SurfaceIndex, String> {
    let backend_kind = backend.kind();
    options.commit = backend.commit.unwrap_or(options.commit);
    match backend_kind.as_str() {
        "memory" => Ok(SurfaceIndex::Memory(
            TextIndex::from_parts(embedder, MemoryIndexStore::new()).with_options(options),
        )),
        "sqlite" => {
            if !context.side_effects.allow_writes {
                return Err(SurfaceError::permission_denied(
                    Some("index.search"),
                    "writes",
                    "SQLite text-index backends require explicit runtime write permission",
                )
                .to_error_string());
            }
            open_sqlite_index(backend, options, embedder)
        }
        other => Err(SurfaceError::unsupported_value(
            Some("index.search"),
            "backend",
            other,
            &["memory", "sqlite"],
        )
        .to_error_string()),
    }
}

#[cfg(feature = "sqlite")]
fn open_sqlite_index(
    backend: BackendRequest,
    options: IndexBuildOptions,
    embedder: Option<ImportedEmbedder>,
) -> Result<SurfaceIndex, String> {
    let path = backend.path.ok_or_else(|| {
        SurfaceError::invalid_request(
            Some("index.search"),
            "SQLite text-index backend requires `path` plus `commit: true`",
        )
        .to_error_string()
    })?;
    if backend.commit != Some(true) {
        return Err(SurfaceError::invalid_request(
            Some("index.search"),
            "SQLite text-index backend requires `commit: true`",
        )
        .to_error_string());
    }
    let store = crate::SqliteIndexStore::open(path, true).map_err(|error| error.to_string())?;
    Ok(SurfaceIndex::Sqlite(
        TextIndex::from_parts(embedder, store).with_options(options),
    ))
}

#[cfg(not(feature = "sqlite"))]
fn open_sqlite_index(
    backend: BackendRequest,
    _options: IndexBuildOptions,
    _embedder: Option<ImportedEmbedder>,
) -> Result<SurfaceIndex, String> {
    if backend.path.is_none() || backend.commit != Some(true) {
        return Err(SurfaceError::invalid_request(
            Some("index.search"),
            "SQLite text-index backend requires `path` plus `commit: true`",
        )
        .to_error_string());
    }
    Err(SurfaceError::missing_dependency(
        Some("index.search"),
        "sqlite feature",
        "enable the text-index `sqlite` feature in a native/server adapter",
    )
    .to_error_string())
}

fn remove_documents(
    input: RemoveRequest,
    context: &SurfaceRuntimeContext,
) -> Result<serde_json::Value, String> {
    let backend_kind = input.backend.kind();
    if backend_kind == "memory" {
        return Ok(serde_json::json!({
            "accepted": true,
            "backend": "memory",
            "documentIds": input.document_ids,
            "commitRequiredForDurableWrites": true,
            "report": {
                "documentsReceived": 0,
                "documentsReplaced": 0,
                "documentsRemoved": 0,
                "documentsSkipped": 0,
                "chunksIndexed": 0,
                "vectorsIndexed": 0
            }
        }));
    }
    let mut index = open_index(input.backend, context)?;
    let report = index.remove_documents(&input.document_ids)?;
    Ok(serde_json::json!({
        "accepted": true,
        "backend": index.backend_name(),
        "documentIds": input.document_ids,
        "commitRequiredForDurableWrites": true,
        "report": report,
        "inspect": index.inspect()?
    }))
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
        "describe" | "index.open" => (
            "Package surface metadata",
            "Inspected the text-index package operations and runtime support.",
            serde_json::json!({
                "status": "ok",
                "operationCount": value["operationCount"]
            }),
        ),
        "index.build" | "index.addDocuments" => (
            "Index build result",
            "Built a transient in-memory text index from the provided documents.",
            serde_json::json!({
                "status": "ok",
                "documentsReceived": value["report"]["documentsReceived"],
                "documentsReplaced": value["report"]["documentsReplaced"],
                "chunkCount": value["inspect"]["chunkCount"],
                "vectorCount": value["inspect"]["vectorCount"]
            }),
        ),
        "index.removeDocuments" => (
            "Index removal plan",
            "Planned document removal for a durable text index; browser package execution stays side-effect free.",
            serde_json::json!({
                "status": "ok",
                "documentCount": value["documentIds"].as_array().map(Vec::len).unwrap_or(0),
                "commitRequiredForDurableWrites": value["commitRequiredForDurableWrites"]
            }),
        ),
        "index.search" => (
            "Index search result",
            "Built a transient in-memory text index and searched it.",
            serde_json::json!({
                "status": "ok",
                "resultCount": value["results"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "index.inspect" => (
            "Index inspection result",
            "Inspected transient text index counts without durable writes.",
            serde_json::json!({
                "status": "ok",
                "documentCount": value["documentCount"],
                "chunkCount": value["chunkCount"],
                "vectorCount": value["vectorCount"],
                "facetCount": value["facetCount"]
            }),
        ),
        "index.snapshotPlan" => (
            "Index snapshot plan",
            "Planned transient index snapshot metadata without writing files.",
            serde_json::json!({
                "status": "ok",
                "backend": value["backend"],
                "documentCount": value["documentCount"],
                "chunkCount": value["chunkCount"],
                "vectorCount": value["vectorCount"]
            }),
        ),
        _ => (
            "Text index result",
            "Ran a text-index package operation.",
            serde_json::json!({"status": "ok"}),
        ),
    };
    structured_surface_value(operation, title, message, summary, value)
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    let mut operation = if id == "index.search" {
        primary_workflow_operation(
            id,
            name,
            description,
            example_request,
            None,
            &[
                "moenarch-text-core",
                "moenarch-text-lexical",
                "moenarch-text-embeddings",
                "moenarch-vector-analysis-index",
            ],
        )
    } else {
        runtime_core::surface_operation(id, name, description, example_request)
    };
    let curation = match id {
        "index.search" => SurfaceOperationCuration::workflow(10).primary(),
        "index.build" => SurfaceOperationCuration::workflow(20),
        "index.addDocuments" => SurfaceOperationCuration::workflow(30),
        "index.snapshotPlan" => SurfaceOperationCuration::workflow(40),
        "index.removeDocuments" => SurfaceOperationCuration::support(500),
        "describe" => SurfaceOperationCuration::debug(900),
        "index.open" => SurfaceOperationCuration::debug(910),
        "index.inspect" => SurfaceOperationCuration::debug(920),
        _ => SurfaceOperationCuration::from_operation_id(id),
    };
    set_surface_operation_curation(&mut operation, curation);
    operation
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BuildRequest {
    documents: Vec<IndexDocument>,
    #[serde(default)]
    options: IndexBuildOptions,
    #[serde(flatten)]
    backend: BackendRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest {
    #[serde(default)]
    documents: Vec<IndexDocument>,
    query: SurfaceSearchQuery,
    #[serde(default)]
    options: IndexBuildOptions,
    #[serde(flatten)]
    backend: BackendRequest,
    /// Provenance of `imported_embeddings`.
    #[serde(default)]
    embedding_model: Option<ImportedEmbeddingModel>,
    /// Caller-computed vectors for semantic and hybrid search.
    #[serde(default)]
    imported_embeddings: Vec<ImportedEmbedding>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SurfaceSearchQuery {
    #[serde(flatten)]
    query: IndexQuery,
    #[serde(default)]
    fuzzy: Option<fuzzy::FuzzySearchOptions>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoveRequest {
    document_ids: Vec<String>,
    #[serde(flatten)]
    backend: BackendRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenRequest {
    #[serde(flatten)]
    backend: BackendRequest,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BackendRequest {
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    commit: Option<bool>,
}

impl BackendRequest {
    fn kind(&self) -> String {
        self.backend.trim().to_ascii_lowercase()
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    runtime_core::parse_surface_input(None, input)
}

fn validate_search_request(input: &SearchRequest) -> Result<(), String> {
    if input.query.query.text.trim().is_empty() {
        return Err(SurfaceError::invalid_request(
            Some("index.search"),
            "index.search requires a non-empty query.text",
        )
        .to_error_string());
    }
    if input.query.query.mode != IndexSearchMode::Lexical {
        if input.imported_embeddings.is_empty() {
            return Err(SurfaceError::missing_dependency(
                Some("index.search"),
                "embedding model",
                "semantic and hybrid index.search need an embedding model, which this surface cannot load; supply `importedEmbeddings` (a `passage` vector per chunk text and a `query` vector for the query) computed with a real model, use `\"mode\": \"lexical\"`, or use the typed text-index API with a real embedder",
            )
            .to_error_string());
        }
        if input.query.fuzzy.is_some() {
            return Err(SurfaceError::invalid_request(
                Some("index.search"),
                "fuzzy expansion is lexical-only; use `\"mode\": \"lexical\"` with `fuzzy`",
            )
            .to_error_string());
        }
    }
    runtime_core::validate_max_items("index.search", "documents", input.documents.len(), 1024)?;
    runtime_core::validate_max_items(
        "index.search",
        "importedEmbeddings",
        input.imported_embeddings.len(),
        16_384,
    )
}

/// The caller's vectors for semantic and hybrid search; lexical search ignores them.
fn search_embedder(input: &SearchRequest) -> Result<Option<ImportedEmbedder>, String> {
    if input.query.query.mode == IndexSearchMode::Lexical {
        return Ok(None);
    }
    ImportedEmbedder::new(&input.imported_embeddings, input.embedding_model.as_ref())
        .map(Some)
        .map_err(|error| {
            SurfaceError::invalid_request(Some("index.search"), error.to_string()).to_error_string()
        })
}

fn default_backend() -> String {
    "memory".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_core::OperationId;

    fn request(operation: &str, input: serde_json::Value) -> SurfaceRequest {
        SurfaceRequest {
            operation: OperationId::new(operation),
            input,
        }
    }

    #[test]
    fn sqlite_backend_requires_explicit_path_and_commit() {
        let error = run_surface_operation(request(
            "index.build",
            serde_json::json!({
                "backend": "sqlite",
                "documents": [{"id": "doc-1", "body": "SQLite requires explicit writes."}]
            }),
        ))
        .expect_err("sqlite must not write without explicit path and commit");
        let parsed = runtime_core::parse_surface_error(&error).expect("typed error");
        assert_eq!(parsed.code, "permission_denied");
        assert_eq!(parsed.details["permission"], "writes");
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn sqlite_backend_builds_when_path_and_commit_are_explicit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("text-index.sqlite");
        let response = run_surface_operation_with_context(
            request(
                "index.build",
                serde_json::json!({
                    "backend": "sqlite",
                    "path": path,
                    "commit": true,
                    "documents": [{"id": "doc-1", "body": "SQLite text indexes persist chunks."}]
                }),
            ),
            &write_context(),
        )
        .expect("sqlite build");
        assert_eq!(response.value["summary"]["status"], serde_json::json!("ok"));
        assert_eq!(
            response.value["result"]["backend"],
            serde_json::json!("sqlite")
        );
        assert_eq!(
            response.value["result"]["inspect"]["documentCount"],
            serde_json::json!(1)
        );
    }

    #[test]
    fn primary_search_schema_is_strict_and_proves_lower_contracts() {
        let surface = package_surface();
        let operation = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() == "index.search")
            .expect("index.search operation");
        assert_eq!(operation.input_schema["additionalProperties"], false);
        assert_eq!(operation.input_schema["xOperationCategory"], "workflow");
        assert_eq!(
            operation.input_schema["xLowerContractProof"]["crates"][0],
            "moenarch-text-core"
        );
    }

    #[test]
    fn search_builds_transient_memory_index() {
        let response = run_surface_operation(request(
            "index.search",
            serde_json::json!({
                "backend": "memory",
                "documents": [{"id": "doc-1", "body": "Rust text index search supports transient memory indexes."}],
                "query": {"text": "transient memory", "mode": "lexical", "topK": 1}
            }),
        ))
        .expect("memory search");
        assert_eq!(response.value["operation"], "index.search");
        assert_eq!(response.value["result"]["backend"], "memory");
        assert_eq!(response.value["summary"]["resultCount"], 1);
    }

    #[test]
    fn semantic_and_hybrid_search_require_an_embedding_model() {
        for mode in ["hybrid", "semantic"] {
            let query = serde_json::json!({"text": "transient memory", "mode": mode, "topK": 1});
            let error = run_surface_operation(request(
                "index.search",
                serde_json::json!({
                    "backend": "memory",
                    "documents": [{"id": "doc-1", "body": "transient memory index"}],
                    "query": query
                }),
            ))
            .expect_err("no embedding model is configured");
            let parsed = runtime_core::parse_surface_error(&error).expect("typed error");
            assert_eq!(parsed.code, "missing_dependency");
            assert_eq!(parsed.details["dependency"], "embedding model");
        }
    }

    #[test]
    fn semantic_search_uses_imported_query_and_passage_vectors() {
        let response = run_surface_operation(request(
            "index.search",
            serde_json::json!({
                "documents": [
                    {"id": "en", "body": "The capital of Germany is Berlin."},
                    {"id": "food", "body": "Tomatoes grow in garden soil."}
                ],
                "query": {"text": "Was ist die Hauptstadt von Deutschland?", "mode": "semantic", "topK": 2},
                "options": {"chunkingStrategy": "paragraph", "chunkTokens": 256, "chunkOverlapTokens": 32, "storeRawText": true},
                "embeddingModel": {"name": "intfloat/multilingual-e5-base", "revision": "d128750", "pooling": "mean", "dimensions": 2},
                "importedEmbeddings": [
                    {"text": "The capital of Germany is Berlin.", "kind": "passage", "vector": [0.9, 0.1]},
                    {"text": "Tomatoes grow in garden soil.", "kind": "passage", "vector": [0.1, 0.9]},
                    {"text": "Was ist die Hauptstadt von Deutschland?", "kind": "query", "vector": [0.8, 0.2]}
                ]
            }),
        ))
        .expect("semantic search with imported vectors");
        let results = &response.value["result"]["results"];
        assert_eq!(results[0]["documentId"], "en");
        assert!(
            results[0]["scoreBreakdown"]["semanticScore"]
                .as_f64()
                .unwrap()
                > 0.9
        );
    }

    #[test]
    fn semantic_search_rejects_missing_passage_vectors_and_fuzzy() {
        let base = serde_json::json!({
            "documents": [{"id": "en", "body": "The capital of Germany is Berlin."}],
            "query": {"text": "Hauptstadt", "mode": "hybrid", "topK": 1},
            "importedEmbeddings": [{"text": "Hauptstadt", "kind": "query", "vector": [1.0, 0.0]}]
        });
        let error = run_surface_operation(request("index.search", base.clone()))
            .expect_err("the chunk has no passage vector");
        assert!(error.contains("(as a passage)"), "{error}");

        let mut fuzzy = base;
        fuzzy["query"]["fuzzy"] = serde_json::json!({"maxEditDistance": 1});
        let error = run_surface_operation(request("index.search", fuzzy)).expect_err("fuzzy");
        let parsed = runtime_core::parse_surface_error(&error).expect("typed error");
        assert_eq!(parsed.code, "invalid_request");
    }

    #[test]
    fn search_requires_an_explicit_mode() {
        let error = run_surface_operation(request(
            "index.search",
            serde_json::json!({
                "documents": [{"id": "doc-1", "body": "transient memory index"}],
                "query": {"text": "transient memory", "topK": 1}
            }),
        ))
        .expect_err("mode is required");
        let parsed = runtime_core::parse_surface_error(&error).expect("typed error");
        assert_eq!(parsed.code, "invalid_request");
        assert!(parsed.message.contains("mode"), "{}", parsed.message);
    }

    #[test]
    fn surface_indexes_store_no_vectors() {
        let response = run_surface_operation(request(
            "index.build",
            serde_json::json!({"documents": [{"id": "doc-1", "body": "Rust text index"}]}),
        ))
        .expect("build");
        assert_eq!(response.value["summary"]["vectorCount"], 0);
    }

    #[test]
    fn search_supports_bounded_fuzzy_lexical_queries() {
        let response = run_surface_operation(request(
            "index.search",
            serde_json::json!({
                "backend": "memory",
                "documents": [
                    {"id": "strategy", "body": "medieval strategy combat formations"},
                    {"id": "recipe", "body": "kitchen recipe ingredients"}
                ],
                "query": {
                    "text": "stratgey formations",
                    "mode": "lexical",
                    "topK": 3,
                    "fuzzy": {
                        "maxEditDistance": 1,
                        "minTermLength": 4,
                        "maxExpansionsPerTerm": 3,
                        "maxQueryVariants": 16,
                        "maxVocabularyTerms": 20000,
                        "fuzzyWeight": 0.8
                    }
                }
            }),
        ))
        .expect("fuzzy memory search");
        assert_eq!(
            response.value["result"]["results"][0]["documentId"],
            "strategy"
        );
        assert_eq!(
            response.value["result"]["results"][0]["fuzzyMatches"][0]["matchedTerm"],
            "strategy"
        );
    }

    #[test]
    fn search_reports_typed_invalid_query_and_unknown_backend() {
        let invalid_query = run_surface_operation(request(
            "index.search",
            serde_json::json!({
                "backend": "memory",
                "documents": [{"id": "doc-1", "body": "Body"}],
                "query": {"text": "", "mode": "lexical"}
            }),
        ))
        .expect_err("invalid query");
        let parsed = runtime_core::parse_surface_error(&invalid_query).expect("typed query");
        assert_eq!(parsed.code, "invalid_request");

        let unknown_backend = run_surface_operation(request(
            "index.search",
            serde_json::json!({
                "backend": "missing",
                "documents": [{"id": "doc-1", "body": "Body"}],
                "query": {"text": "body", "mode": "lexical"}
            }),
        ))
        .expect_err("unknown backend");
        let parsed = runtime_core::parse_surface_error(&unknown_backend).expect("typed backend");
        assert_eq!(parsed.code, "unsupported_value");
        assert_eq!(parsed.details["field"], "backend");
    }

    #[cfg(feature = "sqlite")]
    fn write_context() -> SurfaceRuntimeContext {
        SurfaceRuntimeContext {
            runtime: runtime_core::SurfaceRuntimeKind::NativeServer,
            side_effects: runtime_core::SurfaceSideEffectPolicy {
                allow_reads: true,
                allow_writes: true,
                allow_network: false,
                allow_external_process: false,
                max_download_bytes: None,
            },
            storage: runtime_core::SurfaceStorageContext::default(),
            model: runtime_core::SurfaceModelContext::plan_only(),
        }
    }
}
