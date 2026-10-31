use super::*;
use crate::ingestion::chunk_document;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use text_core::{TextProcessingOptions, TextSegmentContract};
use text_embeddings::{
    DenseVector, HashedTextEmbedder, TextEmbeddingBackend, TextEmbeddingBackendKind,
    TextEmbeddingConfig, TextEmbeddingMetadata,
};
use text_lexical::{Bm25Options, CorpusOptions};
use vector_analysis_index::{SerializableVectorRecord, VectorRecordMetadata};

fn embedder() -> HashedTextEmbedder {
    HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 32,
            use_idf: true,
        },
        CorpusOptions::default(),
    )
    .unwrap()
}

#[derive(Debug)]
struct FlaggedEmbedder {
    panic_on_embed: Cell<bool>,
}

impl FlaggedEmbedder {
    fn new() -> Self {
        Self {
            panic_on_embed: Cell::new(false),
        }
    }
}

impl TextEmbeddingBackend for FlaggedEmbedder {
    fn embed_text(&self, text: &str) -> text_core::Result<DenseVector> {
        if self.panic_on_embed.get() {
            panic!("query embedding should have been skipped");
        }
        DenseVector::new([
            text.bytes().filter(|byte| byte % 2 == 0).count() as f32 + 1.0,
            text.bytes().filter(|byte| byte % 2 == 1).count() as f32 + 1.0,
        ])
    }

    fn metadata(&self) -> TextEmbeddingMetadata {
        TextEmbeddingMetadata {
            backend: TextEmbeddingBackendKind::Custom,
            model_name: Some("flagged".to_string()),
            dimensions: Some(2),
            ..TextEmbeddingMetadata::default()
        }
    }
}

struct FakeReranker;

impl TextReranker for FakeReranker {
    fn rerank(&mut self, _query: &str, documents: &[String]) -> text_core::Result<Vec<f32>> {
        Ok(documents
            .iter()
            .map(|document| {
                if document.contains("second") {
                    0.95
                } else {
                    0.1
                }
            })
            .collect())
    }

    fn runtime(&self) -> RerankerRuntime {
        RerankerRuntime::External
    }
}

fn query(text: &str) -> SearchQuery {
    SearchQuery {
        text: text.to_string(),
        top_k: 3,
        filter: None,
        hybrid: HybridConfig::default(),
    }
}

#[test]
fn chunking_preserves_order_and_ids() {
    let document = SearchDocument {
        id: "doc-1".to_string(),
        title: None,
        body: (0..10)
            .map(|index| format!("token{index}"))
            .collect::<Vec<_>>()
            .join(" "),
        metadata: BTreeMap::new(),
        source: None,
        provenance: Vec::new(),
        annotations: Vec::new(),
    };

    let chunks = chunk_document(
        &document,
        &IngestionOptions {
            chunk_tokens: 4,
            chunk_overlap_tokens: 1,
            store_raw_text: true,
        },
        &TextProcessingOptions::default(),
    )
    .unwrap();

    assert_eq!(chunks[0].chunk_id, "doc-1:0");
    assert_eq!(chunks[1].chunk_id, "doc-1:1");
    assert_eq!(chunks[0].ordinal, 0);
    assert_eq!(chunks[1].ordinal, 1);
}

#[test]
fn explicit_chunking_can_use_sentences() {
    let document = SearchDocument::new("doc-1", "First sentence. Second sentence.");
    let chunks = chunk_search_document(
        &document,
        &ChunkingOptions {
            strategy: ChunkingStrategy::Sentence,
            ..ChunkingOptions::default()
        },
        &TextProcessingOptions::default(),
    )
    .unwrap();

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, "First sentence.");
    assert_eq!(chunks[1].text, "Second sentence.");
}

#[test]
fn rerank_context_uses_backend_scores() {
    let mut reranker = FakeReranker;
    let mut context = RerankExecutionContext {
        reranker: Some(&mut reranker),
        model_id: Some("fake-reranker".to_string()),
    };

    let response = rerank_documents_with_context(
        RerankRequest {
            query: "pick second".to_string(),
            documents: vec!["first document".to_string(), "second document".to_string()],
            top_k: 2,
            imported_scores: Vec::new(),
        },
        &mut context,
    )
    .unwrap();

    assert_eq!(response.results[0].index, 1);
    assert_eq!(response.results[0].runtime.as_deref(), Some("external"));
    assert_eq!(
        response.results[0].model_id.as_deref(),
        Some("fake-reranker")
    );
}

#[test]
fn empty_query_returns_error() {
    let index = RetrievalIndex::new(embedder());
    let err = index.search(&query("   ")).unwrap_err();
    assert!(matches!(err, DetectError::InvalidArgument(message) if message.contains("query")));
}

#[test]
fn search_document_from_text_segment_contract_preserves_metadata() {
    let mut segment = TextSegmentContract::new(7, "hello retrieval");
    segment.stream_id = Some("subs".to_string());
    segment.language = Some("en".to_string());
    segment.timestamp =
        Some(text_core::Timestamp::new(1250, text_core::Timebase::new(1, 1000)).into());
    segment.duration_seconds = Some(1.25);
    segment
        .attributes
        .insert("speaker".to_string(), "narrator".to_string());
    segment
        .attributes
        .insert("confidence".to_string(), "0.8".to_string());
    segment
        .attributes
        .insert("source".to_string(), "fixture.srt".to_string());

    let document = SearchDocument::from_text_segment_contract(&segment);

    assert_eq!(document.id, "subs:7");
    assert_eq!(document.body, "hello retrieval");
    assert_eq!(document.metadata["language"], "en");
    assert_eq!(document.metadata["speaker"], "narrator");
    assert_eq!(document.metadata["confidence"], "0.8");
    assert_eq!(document.metadata["timestamp_seconds"], "1.25");
    assert_eq!(document.metadata["duration_seconds"], "1.25");
    assert_eq!(document.metadata["source"], "fixture.srt");

    let chunk = chunk_search_document(
        &document,
        &ChunkingOptions::default(),
        &TextProcessingOptions::default(),
    )
    .unwrap()
    .remove(0);
    let chunk_document = chunk.to_text_document_contract();
    let chunk_segment = chunk.to_text_segment_contract();
    assert_eq!(chunk_document.source.unwrap().duration_seconds, Some(1.25));
    assert_eq!(chunk_segment.duration_seconds, Some(1.25));
    assert_eq!(chunk_segment.attributes["speaker"], "narrator");
}

#[test]
fn replacing_document_rebuilds_chunk_set() {
    let mut index = RetrievalIndex::new(embedder());
    let options = IngestionOptions {
        chunk_tokens: 3,
        chunk_overlap_tokens: 1,
        store_raw_text: true,
    };

    index
        .ingest_documents(
            &[SearchDocument {
                id: "doc".to_string(),
                title: None,
                body: "rust cargo crates docs search".to_string(),
                metadata: BTreeMap::new(),
                source: None,
                provenance: Vec::new(),
                annotations: Vec::new(),
            }],
            &options,
        )
        .unwrap();
    let original_texts = index
        .chunks
        .values()
        .map(|chunk| chunk.text.clone())
        .collect::<Vec<_>>();

    let report = index
        .ingest_documents(
            &[SearchDocument {
                id: "doc".to_string(),
                title: None,
                body: "music playlists and recommendations".to_string(),
                metadata: BTreeMap::new(),
                source: None,
                provenance: Vec::new(),
                annotations: Vec::new(),
            }],
            &options,
        )
        .unwrap();

    assert_eq!(report.documents_replaced, 1);
    assert!(index
        .chunks
        .keys()
        .all(|chunk_id| chunk_id.starts_with("doc:")));
    assert!(index.chunks.values().all(|chunk| !original_texts
        .iter()
        .any(|original| original == &chunk.text)));
}

#[test]
fn hybrid_search_combines_bm25_and_vectors() {
    let mut index = RetrievalIndex::new(embedder());
    index
        .ingest_documents(
            &[
                SearchDocument {
                    id: "doc-1".to_string(),
                    title: Some("Rust Search".to_string()),
                    body: "Rust cargo crates enable semantic search over documentation."
                        .to_string(),
                    metadata: BTreeMap::from([
                        ("lang".to_string(), "en".to_string()),
                        ("tags".to_string(), "docs rust".to_string()),
                    ]),
                    source: None,
                    provenance: Vec::new(),
                    annotations: Vec::new(),
                },
                SearchDocument {
                    id: "doc-2".to_string(),
                    title: None,
                    body: "Berlin travel plans and restaurant notes.".to_string(),
                    metadata: BTreeMap::from([("lang".to_string(), "en".to_string())]),
                    source: None,
                    provenance: Vec::new(),
                    annotations: Vec::new(),
                },
            ],
            &IngestionOptions::default(),
        )
        .unwrap();

    let results = index.search(&query("rust documentation search")).unwrap();

    assert_eq!(results[0].document_id, "doc-1");
    assert!(results[0].semantic_score > 0.0);
}

#[test]
fn full_text_retrieval_skips_query_embedding() {
    let mut index = RetrievalIndex::new(FlaggedEmbedder::new());
    index
        .ingest_documents(
            &[SearchDocument {
                id: "doc-1".to_string(),
                title: None,
                body: "rust cargo full text search".to_string(),
                metadata: BTreeMap::new(),
                source: None,
                provenance: Vec::new(),
                annotations: Vec::new(),
            }],
            &IngestionOptions::default(),
        )
        .unwrap();
    index.embedder().panic_on_embed.set(true);

    let results = index
        .search(&SearchQuery::full_text("cargo search", 1))
        .unwrap();

    assert_eq!(results[0].document_id, "doc-1");
    assert_eq!(results[0].semantic_score, 0.0);
    assert!(results[0].lexical_score > 0.0);
}

#[test]
fn semantic_search_does_not_require_lexical_matches() {
    let mut index = RetrievalIndex::new(embedder());
    index
        .ingest_documents(
            &[SearchDocument {
                id: "doc-1".to_string(),
                title: None,
                body: "alpha beta gamma".to_string(),
                metadata: BTreeMap::new(),
                source: None,
                provenance: Vec::new(),
                annotations: Vec::new(),
            }],
            &IngestionOptions::default(),
        )
        .unwrap();

    let results = index
        .search(&SearchQuery::semantic("unshared tokens", 1))
        .unwrap();

    assert_eq!(results[0].document_id, "doc-1");
    assert_eq!(results[0].lexical_score, 0.0);
    assert!(results[0].semantic_score > 0.0);
}

#[test]
fn post_filter_search_expands_candidates_before_filtering() {
    let mut index = RetrievalIndex::new(embedder());
    let mut docs = Vec::new();
    for number in 0..40 {
        docs.push(SearchDocument {
            id: format!("doc-{number:02}"),
            title: None,
            body: "common search text".to_string(),
            metadata: BTreeMap::from([(
                "note".to_string(),
                if number == 39 {
                    "contains-special-target".to_string()
                } else {
                    "ordinary".to_string()
                },
            )]),
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
        });
    }
    index
        .ingest_documents(&docs, &IngestionOptions::default())
        .unwrap();

    let filter = SearchFilter {
        metadata_contains: BTreeMap::from([("note".to_string(), "special".to_string())]),
        document_ids: BTreeSet::from(["doc-39".to_string()]),
        ..SearchFilter::default()
    };
    let results = index
        .search(&SearchQuery::full_text("common", 1).filter(filter))
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document_id, "doc-39");
}

#[test]
fn filters_exclude_non_matching_chunks() {
    let mut index = RetrievalIndex::new(embedder());
    index
        .ingest_documents(
            &[
                SearchDocument {
                    id: "doc-1".to_string(),
                    title: None,
                    body: "English cargo docs".to_string(),
                    metadata: BTreeMap::from([
                        ("lang".to_string(), "en".to_string()),
                        ("tags".to_string(), "docs".to_string()),
                    ]),
                    source: None,
                    provenance: Vec::new(),
                    annotations: Vec::new(),
                },
                SearchDocument {
                    id: "doc-2".to_string(),
                    title: None,
                    body: "German cargo docs".to_string(),
                    metadata: BTreeMap::from([
                        ("lang".to_string(), "de".to_string()),
                        ("tags".to_string(), "docs".to_string()),
                    ]),
                    source: None,
                    provenance: Vec::new(),
                    annotations: Vec::new(),
                },
            ],
            &IngestionOptions::default(),
        )
        .unwrap();

    let mut query = query("cargo docs");
    query.filter = Some(SearchFilter {
        metadata_equals: BTreeMap::from([("lang".to_string(), "en".to_string())]),
        metadata_contains: BTreeMap::new(),
        required_tags: vec!["docs".to_string()],
        document_ids: BTreeSet::new(),
    });

    let results = index.search(&query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document_id, "doc-1");
}

#[test]
fn related_chunks_excludes_self() {
    let mut index = RetrievalIndex::new(embedder());
    index
        .ingest_documents(
            &[SearchDocument {
                id: "doc-1".to_string(),
                title: None,
                body: "rust cargo crates semantic search documentation indexing retrieval"
                    .to_string(),
                metadata: BTreeMap::new(),
                source: None,
                provenance: Vec::new(),
                annotations: Vec::new(),
            }],
            &IngestionOptions {
                chunk_tokens: 4,
                chunk_overlap_tokens: 1,
                store_raw_text: true,
            },
        )
        .unwrap();

    let related = index.related_chunks("doc-1:0", 2).unwrap();
    assert!(related.iter().all(|result| result.chunk_id != "doc-1:0"));
}

#[test]
fn from_parts_validates_dimension_mismatch() {
    let chunks = vec![DocumentChunk {
        chunk_id: "doc:0".to_string(),
        document_id: "doc".to_string(),
        text: "rust cargo docs".to_string(),
        ordinal: 0,
        metadata: BTreeMap::new(),
        source: None,
        provenance: Vec::new(),
        annotations: Vec::new(),
    }];
    let records = vec![SerializableVectorRecord {
        id: "doc:0".into(),
        vector: vec![1.0, 0.0],
        payload: VectorRecordMetadata::default(),
    }];

    let err = RetrievalIndex::from_parts(
        embedder(),
        CorpusOptions::default(),
        Bm25Options::default(),
        chunks,
        BTreeMap::new(),
        records,
    )
    .unwrap_err();

    assert!(matches!(err, DetectError::InvalidArgument(message) if message.contains("dimensions")));
}
