#![cfg(feature = "sqlite")]

use std::collections::{BTreeMap, BTreeSet};

use text_core::TextSourceRef;
use text_embeddings::{HashedTextEmbedder, TextEmbeddingConfig};
use text_index::{
    IndexBuildOptions, IndexDocument, IndexFilter, IndexQuery, MemoryIndexStore, SemanticFacet,
    SemanticFacetFilter, SqliteIndexStore, TextIndex, TextIndexStore,
};
use text_lexical::CorpusOptions;

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

fn documents() -> Vec<IndexDocument> {
    let mut report = IndexDocument::new(
        "report-1",
        "The multimodal report cites a transcript about durable hybrid search.",
    );
    report
        .metadata
        .attributes
        .insert("source".to_string(), "report".to_string());
    report.metadata.source = Some(TextSourceRef {
        source_id: Some("report".to_string()),
        source_kind: Some("markdown".to_string()),
        uri: None,
        media_timestamp: Some(text_core::Timestamp::new(10, text_core::Timebase::new(1, 1)).into()),
        duration_seconds: None,
    });
    report.semantic_facets.push(SemanticFacet {
        kind: "topic".to_string(),
        key: "name".to_string(),
        value: "hybrid-search".to_string(),
        score: Some(0.9),
        provenance: Vec::new(),
    });

    vec![
        report,
        IndexDocument::new("note-1", "A playlist note about music recommendations."),
    ]
}

#[test]
fn sqlite_schema_creation_reports_fts5() {
    let store = SqliteIndexStore::in_memory().unwrap();
    let inspect = store.inspect().unwrap();
    assert_eq!(inspect.document_count, 0);
    assert_eq!(inspect.sqlite_fts5_available, Some(true));
}

#[test]
fn memory_and_sqlite_search_parity() {
    let options = IndexBuildOptions {
        chunk_tokens: 12,
        chunk_overlap_tokens: 0,
        ..IndexBuildOptions::default()
    };
    let mut memory =
        TextIndex::with_store(embedder(), MemoryIndexStore::new()).with_options(options.clone());
    let mut sqlite = TextIndex::with_store(embedder(), SqliteIndexStore::in_memory().unwrap())
        .with_options(options);
    memory.upsert_documents(&documents()).unwrap();
    sqlite.upsert_documents(&documents()).unwrap();

    let query = IndexQuery {
        text: "durable hybrid search".to_string(),
        explain: true,
        ..IndexQuery::default()
    };
    let memory_results = memory.search(&query).unwrap();
    let sqlite_results = sqlite.search(&query).unwrap();
    assert_eq!(memory_results[0].document_id, sqlite_results[0].document_id);
    assert_eq!(sqlite_results[0].score_breakdown.semantic_weight, 0.6);
    assert_eq!(sqlite_results[0].score_breakdown.lexical_weight, 0.4);
}

#[test]
fn sqlite_filters_metadata_source_timestamp_and_facets() {
    let mut index = TextIndex::with_store(embedder(), SqliteIndexStore::in_memory().unwrap())
        .with_options(IndexBuildOptions {
            chunk_tokens: 12,
            chunk_overlap_tokens: 0,
            ..IndexBuildOptions::default()
        });
    index.upsert_documents(&documents()).unwrap();
    let results = index
        .search(&IndexQuery {
            text: "transcript durable".to_string(),
            filter: IndexFilter {
                metadata_equals: BTreeMap::from([("source".to_string(), "report".to_string())]),
                source_kinds: BTreeSet::from(["markdown".to_string()]),
                source_ids: BTreeSet::from(["report".to_string()]),
                timestamp_seconds_min: Some(9.0),
                timestamp_seconds_max: Some(11.0),
                semantic_facets: vec![SemanticFacetFilter {
                    kind: "topic".to_string(),
                    key: "name".to_string(),
                    value: "hybrid-search".to_string(),
                }],
                ..IndexFilter::default()
            },
            ..IndexQuery::default()
        })
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document_id, "report-1");
}

#[test]
fn sqlite_upsert_replaces_vectors_and_facets() {
    let mut index = TextIndex::with_store(embedder(), SqliteIndexStore::in_memory().unwrap())
        .with_options(IndexBuildOptions {
            chunk_tokens: 8,
            chunk_overlap_tokens: 0,
            ..IndexBuildOptions::default()
        });
    let mut first = IndexDocument::new("doc", "old hybrid search");
    first.semantic_facets.push(SemanticFacet {
        kind: "topic".to_string(),
        key: "name".to_string(),
        value: "old".to_string(),
        score: None,
        provenance: Vec::new(),
    });
    index.upsert_documents(&[first]).unwrap();
    let mut second = IndexDocument::new("doc", "new durable index");
    second.semantic_facets.push(SemanticFacet {
        kind: "topic".to_string(),
        key: "name".to_string(),
        value: "new".to_string(),
        score: None,
        provenance: Vec::new(),
    });
    let report = index.upsert_documents(&[second]).unwrap();
    assert_eq!(report.documents_replaced, 1);
    let inspect = index.inspect().unwrap();
    assert_eq!(inspect.document_count, 1);
    assert_eq!(inspect.facet_count, 1);
}
