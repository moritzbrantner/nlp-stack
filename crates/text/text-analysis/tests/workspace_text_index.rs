use text_analysis::{
    TextWorkspace, TextWorkspaceOptions, WorkspaceDocument, WorkspaceIndexOptions,
    WorkspaceIndexStorage,
};
use text_core::TextDocumentContract;
use text_index::{IndexBuildOptions, IndexQuery};

#[test]
fn workspace_builds_and_searches_memory_text_index() {
    let mut workspace = TextWorkspace::new(TextWorkspaceOptions {
        index: WorkspaceIndexOptions {
            storage: WorkspaceIndexStorage::Memory,
            build: IndexBuildOptions {
                chunk_tokens: 8,
                chunk_overlap_tokens: 0,
                ..IndexBuildOptions::default()
            },
            embedding_dimensions: 32,
            commit: false,
        },
        ..TextWorkspaceOptions::default()
    });
    workspace
        .ingest_documents([WorkspaceDocument::DocumentContract(
            TextDocumentContract::new(
                "doc-1",
                "Workspace text index stores analysis-ready chunks.",
            ),
        )])
        .unwrap();
    let report = workspace.build_text_index().unwrap();
    assert_eq!(report.documents_received, 1);
    let results = workspace
        .search_text_index(IndexQuery::new("analysis chunks", 1))
        .unwrap();
    assert_eq!(results.results[0].document_id, "doc-1");
    assert_eq!(workspace.inspect_text_index().unwrap().document_count, 1);
}
