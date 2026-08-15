//! Library-owned runtime surface for `text-index`.

use runtime_core::{
    primary_workflow_operation, set_surface_operation_curation, structured_surface_value,
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceError, SurfaceOperation,
    SurfaceOperationCuration, SurfaceRequest, SurfaceResponse, SurfaceRuntimeContext,
};
use serde::Deserialize;
use std::path::PathBuf;
use text_embeddings::{HashedTextEmbedder, TextEmbeddingConfig};
use text_lexical::CorpusOptions;

use crate::{
    IndexBuildOptions, IndexDocument, IndexInspectReport, IndexMutationReport, IndexQuery,
    IndexSearchResult, IndexSnapshotPlan, MemoryIndexStore, TextIndex, TextIndexStore,
};

pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation("describe", "Inspect package metadata", "Durable local text indexing and hybrid search.", serde_json::json!({"includeOperations": true})),
            operation("index.build", "Build index", "Builds a transient in-memory text index unless an explicit committed SQLite backend is requested.", serde_json::json!({"backend": "memory", "documents": [{"id": "doc-1", "body": "Rust text index"}]})),
            operation("index.open", "Open index", "Opens or describes an index backend.", serde_json::json!({"backend": "memory"})),
            operation("index.addDocuments", "Add documents", "Adds documents to a transient memory index or an explicit committed SQLite backend.", serde_json::json!({"backend": "memory", "documents": [{"id": "doc-1", "body": "Rust text index"}]})),
            operation("index.removeDocuments", "Remove documents", "Removes documents only from an explicit committed SQLite backend; memory execution returns a side-effect-free plan.", serde_json::json!({"backend": "memory", "documentIds": ["doc-1"]})),
            operation("index.search", "Search index", "Builds or opens a requested backend and searches it.", serde_json::json!({"backend": "memory", "documents": [{"id": "doc-1", "body": "Rust text index supports required phrases"}], "query": {"text": "text index required phrases", "requiredPhrases": ["required phrases"]}})),
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
            let (index, _) = build_index(
                BuildRequest {
                    documents: input.documents,
                    options: input.options,
                    dimensions: input.dimensions,
                    backend: input.backend,
                },
                context,
            )?;
            serde_json::json!({"backend": index.backend_name(), "results": index.search(&input.query)?})
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
    Memory(TextIndex<HashedTextEmbedder, MemoryIndexStore>),
    #[cfg(feature = "sqlite")]
    Sqlite(TextIndex<HashedTextEmbedder, crate::SqliteIndexStore>),
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
    let documents = input.documents;
    runtime_core::validate_max_items("index.search", "documents", documents.len(), 1024)?;
    let mut index = open_configured_index(input.backend, input.options, input.dimensions, context)?;
    let report = index.upsert_documents(&documents)?;
    Ok((index, report))
}

fn open_index(
    backend: BackendRequest,
    context: &SurfaceRuntimeContext,
) -> Result<SurfaceIndex, String> {
    open_configured_index(
        backend,
        IndexBuildOptions::default(),
        default_dimensions(),
        context,
    )
}

fn open_configured_index(
    backend: BackendRequest,
    mut options: IndexBuildOptions,
    dimensions: usize,
    context: &SurfaceRuntimeContext,
) -> Result<SurfaceIndex, String> {
    let backend_kind = backend.kind();
    options.commit = backend.commit.unwrap_or(options.commit);
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: dimensions.max(1),
            use_idf: false,
        },
        CorpusOptions::default(),
    )
    .map_err(|error| error.to_string())?;
    match backend_kind.as_str() {
        "memory" => Ok(SurfaceIndex::Memory(
            TextIndex::with_store(embedder, MemoryIndexStore::new()).with_options(options),
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
            open_sqlite_index(backend, embedder, options)
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
    embedder: HashedTextEmbedder,
    options: IndexBuildOptions,
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
        TextIndex::with_store(embedder, store).with_options(options),
    ))
}

#[cfg(not(feature = "sqlite"))]
fn open_sqlite_index(
    backend: BackendRequest,
    _embedder: HashedTextEmbedder,
    _options: IndexBuildOptions,
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
            "Built a transient in-memory text index and searched it with deterministic lexical and semantic scoring.",
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
    #[serde(default = "default_dimensions")]
    dimensions: usize,
    #[serde(flatten)]
    backend: BackendRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest {
    #[serde(default)]
    documents: Vec<IndexDocument>,
    query: IndexQuery,
    #[serde(default)]
    options: IndexBuildOptions,
    #[serde(default = "default_dimensions")]
    dimensions: usize,
    #[serde(flatten)]
    backend: BackendRequest,
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
    if input.query.text.trim().is_empty() {
        return Err(SurfaceError::invalid_request(
            Some("index.search"),
            "index.search requires a non-empty query.text",
        )
        .to_error_string());
    }
    runtime_core::validate_max_items("index.search", "documents", input.documents.len(), 1024)
}

fn default_backend() -> String {
    "memory".to_string()
}

fn default_dimensions() -> usize {
    128
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
                "query": {"text": "transient memory", "topK": 1}
            }),
        ))
        .expect("memory search");
        assert_eq!(response.value["operation"], "index.search");
        assert_eq!(response.value["result"]["backend"], "memory");
        assert_eq!(response.value["summary"]["resultCount"], 1);
    }

    #[test]
    fn search_reports_typed_invalid_query_and_unknown_backend() {
        let invalid_query = run_surface_operation(request(
            "index.search",
            serde_json::json!({
                "backend": "memory",
                "documents": [{"id": "doc-1", "body": "Body"}],
                "query": {"text": ""}
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
                "query": {"text": "body"}
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
