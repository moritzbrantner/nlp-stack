#![cfg(feature = "sqlite")]

use std::collections::BTreeMap;

use tempfile::tempdir;
use text_embeddings::{HashedTextEmbedder, TextEmbeddingConfig};
use text_lexical::CorpusOptions;
use text_retrieval::{
    build_memory_text_index, IngestionOptions, PersistedSearchIndex, RetrievalIndex, SearchDocument,
};

fn embedder() -> HashedTextEmbedder {
    HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 32,
            use_idf: false,
        },
        CorpusOptions::default(),
    )
    .unwrap()
}

#[test]
fn legacy_documents_build_memory_text_index() {
    let (index, report) = build_memory_text_index(
        embedder(),
        &[SearchDocument {
            id: "doc-1".to_string(),
            title: Some("Hybrid Search".to_string()),
            body: "Durable text index compatibility.".to_string(),
            metadata: BTreeMap::from([("language".to_string(), "en".to_string())]),
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
        }],
        &IngestionOptions {
            chunk_tokens: 8,
            chunk_overlap_tokens: 0,
            store_raw_text: true,
        },
    )
    .unwrap();
    assert_eq!(report.documents_received, 1);
    assert_eq!(
        index
            .search(&text_index::IndexQuery::new("durable index", 1))
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn persisted_search_index_imports_into_sqlite_text_index() {
    let mut retrieval = RetrievalIndex::new(embedder());
    retrieval
        .ingest_documents(
            &[SearchDocument::new(
                "doc-1",
                "Persisted retrieval snapshots import into SQLite text index.",
            )],
            &IngestionOptions {
                chunk_tokens: 8,
                chunk_overlap_tokens: 0,
                store_raw_text: true,
            },
        )
        .unwrap();
    let dir = tempdir().unwrap();
    let sqlite_path = dir.path().join("index.sqlite");
    let report = PersistedSearchIndex::from_index(&retrieval)
        .import_into_sqlite_path(&sqlite_path)
        .unwrap();
    assert_eq!(report.documents_received, 1);

    let store = text_index::SqliteIndexStore::open(&sqlite_path, true).unwrap();
    let index = text_index::TextIndex::with_store(embedder(), store);
    let results = index
        .search(&text_index::IndexQuery::new("sqlite text index", 1))
        .unwrap();
    assert_eq!(results[0].document_id, "doc-1");
}
