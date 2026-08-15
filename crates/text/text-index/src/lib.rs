#![doc = include_str!("../README.md")]

pub mod surface;

use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "sqlite")]
use std::path::Path;

use serde::{Deserialize, Serialize};
use text_core::DetectError;
use text_core::{
    split_paragraphs, split_sentence_spans, tokenize, TextAnnotationSpan, TextDocumentContract,
    TextProcessingOptions, TextProvenance, TextSegmentContract, TextSourceRef, TextSpan,
    TimestampContract, TokenKind,
};
use text_embeddings::{
    DenseVector, EmbeddingModelInfo, HashedTextEmbedder, TextEmbedderBackend, TextEmbeddingConfig,
};
use text_lexical::{Bm25Corpus, Bm25Options, CorpusOptions, TextCorpus, TextCorpusDocument};
use thiserror::Error;
use vector_analysis_index::{
    VectorRecord, VectorRecordId, VectorRecordMetadata, VectorSearchIndex,
};

const TITLE_METADATA_KEY: &str = "__title";

#[derive(Debug, Error)]
pub enum TextIndexError {
    #[error("invalid index argument: {0}")]
    InvalidArgument(String),
    #[error("index state error: {0}")]
    InvalidState(String),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[cfg(feature = "sqlite")]
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error(transparent)]
    Core(#[from] DetectError),
}

pub type Result<T> = std::result::Result<T, TextIndexError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct IndexDocumentMetadata {
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexDocument {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    pub body: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub metadata: IndexDocumentMetadata,
    #[serde(default)]
    pub analysis_attachments: Vec<IndexAnalysisAttachment>,
    #[serde(default)]
    pub semantic_facets: Vec<SemanticFacet>,
}

impl IndexDocument {
    pub fn new(id: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: None,
            body: body.into(),
            language: None,
            metadata: IndexDocumentMetadata::default(),
            analysis_attachments: Vec::new(),
            semantic_facets: Vec::new(),
        }
    }

    pub fn from_text_document_contract(document: &TextDocumentContract) -> Self {
        Self {
            id: document.id.clone(),
            title: None,
            body: document.text.clone(),
            language: document.language.clone(),
            metadata: IndexDocumentMetadata {
                attributes: document.attributes.clone(),
                source: source_with_timestamp(
                    document.source.clone(),
                    document.timestamp,
                    "text_document",
                    None,
                ),
                provenance: document.provenance.clone(),
                annotations: document.annotations.clone(),
            },
            analysis_attachments: Vec::new(),
            semantic_facets: Vec::new(),
        }
    }

    pub fn from_text_segment_contract(segment: &TextSegmentContract) -> Self {
        let contract = segment.to_text_document_contract();
        Self::from_text_document_contract(&contract)
    }

    pub fn from_text_corpus_document(document: &TextCorpusDocument) -> Self {
        Self {
            id: document.id.clone(),
            title: None,
            body: document.text.clone(),
            language: document.language.clone(),
            metadata: IndexDocumentMetadata {
                attributes: document.metadata.clone(),
                source: source_with_timestamp(
                    document.source.clone(),
                    document.timestamp,
                    "text_corpus_document",
                    None,
                ),
                provenance: document.provenance.clone(),
                annotations: document.annotations.clone(),
            },
            analysis_attachments: Vec::new(),
            semantic_facets: Vec::new(),
        }
    }

    pub fn from_text_corpus(corpus: &TextCorpus) -> Vec<Self> {
        corpus
            .documents
            .iter()
            .map(Self::from_text_corpus_document)
            .collect()
    }
}

fn source_with_timestamp(
    source: Option<TextSourceRef>,
    timestamp: Option<TimestampContract>,
    default_kind: &str,
    source_id: Option<String>,
) -> Option<TextSourceRef> {
    match (source, timestamp) {
        (Some(mut source), Some(timestamp)) => {
            source.media_timestamp.get_or_insert(timestamp);
            Some(source)
        }
        (source @ Some(_), None) => source,
        (None, Some(timestamp)) => Some(TextSourceRef {
            source_id,
            source_kind: Some(default_kind.to_string()),
            uri: None,
            media_timestamp: Some(timestamp),
            duration_seconds: None,
        }),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexChunk {
    pub id: String,
    pub document_id: String,
    pub ordinal: usize,
    pub text: String,
    pub byte_start: usize,
    pub byte_end: usize,
    #[serde(default)]
    pub token_start: Option<usize>,
    #[serde(default)]
    pub token_end: Option<usize>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
    #[serde(default)]
    pub semantic_facets: Vec<SemanticFacet>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexAnalysisAttachment {
    #[serde(default)]
    pub document_id: Option<String>,
    #[serde(default)]
    pub chunk_id: Option<String>,
    pub kind: String,
    pub producer: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticFacet {
    pub kind: String,
    pub key: String,
    pub value: String,
    #[serde(default)]
    pub score: Option<f32>,
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ChunkingStrategy {
    TokenWindow,
    Sentence,
    Paragraph,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexBuildOptions {
    pub chunking_strategy: ChunkingStrategy,
    pub chunk_tokens: usize,
    pub chunk_overlap_tokens: usize,
    pub store_raw_text: bool,
    #[serde(default)]
    pub commit: bool,
    #[serde(default)]
    pub processing: TextProcessingOptions,
}

impl Default for IndexBuildOptions {
    fn default() -> Self {
        Self {
            chunking_strategy: ChunkingStrategy::TokenWindow,
            chunk_tokens: 256,
            chunk_overlap_tokens: 32,
            store_raw_text: true,
            commit: false,
            processing: TextProcessingOptions::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexMutationReport {
    pub documents_received: usize,
    pub documents_replaced: usize,
    pub documents_removed: usize,
    pub documents_skipped: usize,
    pub chunks_indexed: usize,
    pub vectors_indexed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexInspectReport {
    pub document_count: usize,
    pub chunk_count: usize,
    pub vector_count: usize,
    pub facet_count: usize,
    pub attachment_count: usize,
    #[serde(default)]
    pub sqlite_fts5_available: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexSnapshotPlan {
    pub backend: String,
    pub document_count: usize,
    pub chunk_count: usize,
    pub vector_count: usize,
    pub facet_count: usize,
    pub attachment_count: usize,
    pub chunking_strategy: ChunkingStrategy,
    pub commit_required_for_writes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum IndexSearchMode {
    Lexical,
    Semantic,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexQuery {
    pub text: String,
    #[serde(default = "default_mode")]
    pub mode: IndexSearchMode,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_candidate_limit")]
    pub candidate_limit: usize,
    #[serde(default)]
    pub filter: IndexFilter,
    #[serde(default = "default_semantic_weight")]
    pub semantic_weight: f32,
    #[serde(default = "default_lexical_weight")]
    pub lexical_weight: f32,
    #[serde(default)]
    pub required_phrases: Vec<String>,
    #[serde(default)]
    pub explain: bool,
}

impl IndexQuery {
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            top_k,
            ..Self::default()
        }
    }

    pub fn lexical(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            mode: IndexSearchMode::Lexical,
            semantic_weight: 0.0,
            lexical_weight: 1.0,
            ..Self::new(text, top_k)
        }
    }

    pub fn semantic(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            mode: IndexSearchMode::Semantic,
            semantic_weight: 1.0,
            lexical_weight: 0.0,
            ..Self::new(text, top_k)
        }
    }
}

impl Default for IndexQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            mode: default_mode(),
            top_k: default_top_k(),
            candidate_limit: default_candidate_limit(),
            filter: IndexFilter::default(),
            semantic_weight: default_semantic_weight(),
            lexical_weight: default_lexical_weight(),
            required_phrases: Vec::new(),
            explain: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct IndexFilter {
    #[serde(default)]
    pub metadata_equals: BTreeMap<String, String>,
    #[serde(default)]
    pub metadata_contains: BTreeMap<String, String>,
    #[serde(default)]
    pub required_tags: Vec<String>,
    #[serde(default)]
    pub document_ids: BTreeSet<String>,
    #[serde(default)]
    pub semantic_facets: Vec<SemanticFacetFilter>,
    #[serde(default)]
    pub source_kinds: BTreeSet<String>,
    #[serde(default)]
    pub source_ids: BTreeSet<String>,
    #[serde(default)]
    pub timestamp_seconds_min: Option<f64>,
    #[serde(default)]
    pub timestamp_seconds_max: Option<f64>,
    #[serde(default)]
    pub provenance_operations: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticFacetFilter {
    pub kind: String,
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexSearchResult {
    pub chunk_id: String,
    pub document_id: String,
    pub score: f32,
    pub snippet: String,
    #[serde(default)]
    pub matched_phrases: Vec<String>,
    pub chunk: IndexChunk,
    pub score_breakdown: IndexScoreBreakdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexScoreBreakdown {
    pub semantic_score: f32,
    pub lexical_score: f32,
    pub normalized_semantic_score: f32,
    pub normalized_lexical_score: f32,
    pub semantic_weight: f32,
    pub lexical_weight: f32,
    #[serde(default)]
    pub explanation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredVector {
    pub chunk_id: String,
    pub backend: String,
    pub model_name: String,
    pub dimensions: usize,
    pub vector: Vec<f32>,
}

pub trait TextIndexStore {
    fn backend_name(&self) -> &'static str;
    fn upsert_document_state(
        &mut self,
        document: IndexDocument,
        chunks: Vec<IndexChunk>,
        vectors: Vec<StoredVector>,
        options: &IndexBuildOptions,
    ) -> Result<IndexMutationReport>;
    fn remove_documents(&mut self, document_ids: &[String]) -> Result<usize>;
    fn documents(&self) -> Result<Vec<IndexDocument>>;
    fn chunks(&self) -> Result<Vec<IndexChunk>>;
    fn vectors(&self) -> Result<Vec<StoredVector>>;
    fn lexical_candidates(&self, query: &str, limit: usize) -> Result<Option<Vec<(String, f32)>>>;
    fn inspect(&self) -> Result<IndexInspectReport>;
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryIndexStore {
    documents: BTreeMap<String, IndexDocument>,
    chunks: BTreeMap<String, IndexChunk>,
    vectors: BTreeMap<String, StoredVector>,
    document_chunks: BTreeMap<String, Vec<String>>,
}

impl MemoryIndexStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TextIndexStore for MemoryIndexStore {
    fn backend_name(&self) -> &'static str {
        "memory"
    }

    fn upsert_document_state(
        &mut self,
        document: IndexDocument,
        chunks: Vec<IndexChunk>,
        vectors: Vec<StoredVector>,
        _options: &IndexBuildOptions,
    ) -> Result<IndexMutationReport> {
        let replaced = self.documents.contains_key(&document.id) as usize;
        remove_document_state(
            &mut self.documents,
            &mut self.chunks,
            &mut self.vectors,
            &mut self.document_chunks,
            &document.id,
        );
        let document_id = document.id.clone();
        let vector_count = vectors.len();
        let chunk_count = chunks.len();
        self.documents.insert(document.id.clone(), document);
        self.document_chunks.insert(
            document_id,
            chunks.iter().map(|chunk| chunk.id.clone()).collect(),
        );
        for chunk in chunks {
            self.chunks.insert(chunk.id.clone(), chunk);
        }
        for vector in vectors {
            self.vectors.insert(vector.chunk_id.clone(), vector);
        }
        Ok(IndexMutationReport {
            documents_received: 1,
            documents_replaced: replaced,
            documents_removed: 0,
            documents_skipped: 0,
            chunks_indexed: chunk_count,
            vectors_indexed: vector_count,
        })
    }

    fn remove_documents(&mut self, document_ids: &[String]) -> Result<usize> {
        let mut removed = 0;
        for document_id in document_ids {
            if self.documents.contains_key(document_id) {
                remove_document_state(
                    &mut self.documents,
                    &mut self.chunks,
                    &mut self.vectors,
                    &mut self.document_chunks,
                    document_id,
                );
                removed += 1;
            }
        }
        Ok(removed)
    }

    fn documents(&self) -> Result<Vec<IndexDocument>> {
        Ok(self.documents.values().cloned().collect())
    }

    fn chunks(&self) -> Result<Vec<IndexChunk>> {
        Ok(self.chunks.values().cloned().collect())
    }

    fn vectors(&self) -> Result<Vec<StoredVector>> {
        Ok(self.vectors.values().cloned().collect())
    }

    fn lexical_candidates(
        &self,
        _query: &str,
        _limit: usize,
    ) -> Result<Option<Vec<(String, f32)>>> {
        Ok(None)
    }

    fn inspect(&self) -> Result<IndexInspectReport> {
        Ok(IndexInspectReport {
            document_count: self.documents.len(),
            chunk_count: self.chunks.len(),
            vector_count: self.vectors.len(),
            facet_count: self
                .chunks
                .values()
                .map(|chunk| chunk.semantic_facets.len())
                .sum::<usize>(),
            attachment_count: self
                .documents
                .values()
                .map(|document| document.analysis_attachments.len())
                .sum::<usize>(),
            sqlite_fts5_available: None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TextIndex<B, S> {
    embedder: B,
    store: S,
    options: IndexBuildOptions,
    corpus_options: CorpusOptions,
    bm25_options: Bm25Options,
}

pub type MemoryTextIndex<B = HashedTextEmbedder> = TextIndex<B, MemoryIndexStore>;
#[cfg(feature = "sqlite")]
pub type SqliteTextIndex<B = HashedTextEmbedder> = TextIndex<B, SqliteIndexStore>;

impl MemoryTextIndex<HashedTextEmbedder> {
    pub fn new_memory() -> Result<Self> {
        let corpus_options = CorpusOptions::default();
        let embedder = HashedTextEmbedder::new(
            TextEmbeddingConfig {
                dimensions: 128,
                use_idf: false,
            },
            corpus_options.clone(),
        )
        .map_err(TextIndexError::Core)?;
        Ok(TextIndex::with_store(embedder, MemoryIndexStore::new()))
    }
}

impl<B: TextEmbedderBackend, S: TextIndexStore> TextIndex<B, S> {
    pub fn with_store(embedder: B, store: S) -> Self {
        Self {
            embedder,
            store,
            options: IndexBuildOptions::default(),
            corpus_options: CorpusOptions::default(),
            bm25_options: Bm25Options::default(),
        }
    }

    pub fn with_options(mut self, options: IndexBuildOptions) -> Self {
        self.options = options;
        self
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    pub fn embedder_info(&self) -> EmbeddingModelInfo {
        self.embedder.model_info()
    }

    pub fn upsert_documents(&mut self, documents: &[IndexDocument]) -> Result<IndexMutationReport> {
        validate_build_options(&self.options)?;
        let mut total = IndexMutationReport {
            documents_received: documents.len(),
            documents_replaced: 0,
            documents_removed: 0,
            documents_skipped: 0,
            chunks_indexed: 0,
            vectors_indexed: 0,
        };
        for document in documents {
            validate_document(document)?;
            let chunks = chunk_index_document(document, &self.options)?;
            if chunks.is_empty() {
                let _ = self
                    .store
                    .remove_documents(std::slice::from_ref(&document.id))?;
                total.documents_skipped += 1;
                continue;
            }
            let vectors = embed_chunks(&self.embedder, &chunks)?;
            let report = self.store.upsert_document_state(
                document.clone(),
                chunks,
                vectors,
                &self.options,
            )?;
            total.documents_replaced += report.documents_replaced;
            total.documents_skipped += report.documents_skipped;
            total.chunks_indexed += report.chunks_indexed;
            total.vectors_indexed += report.vectors_indexed;
        }
        Ok(total)
    }

    pub fn remove_documents(&mut self, document_ids: &[String]) -> Result<IndexMutationReport> {
        let removed = self.store.remove_documents(document_ids)?;
        let inspect = self.store.inspect()?;
        Ok(IndexMutationReport {
            documents_received: 0,
            documents_replaced: 0,
            documents_removed: removed,
            documents_skipped: 0,
            chunks_indexed: inspect.chunk_count,
            vectors_indexed: inspect.vector_count,
        })
    }

    pub fn search(&self, query: &IndexQuery) -> Result<Vec<IndexSearchResult>> {
        validate_query(query)?;
        let normalized_query = normalize_query(&query.text, &self.corpus_options.processing)?;
        let required_phrases =
            normalize_required_phrases(&query.required_phrases, &self.corpus_options.processing);
        let limit = query.candidate_limit.max(query.top_k).max(1);
        let chunks = self
            .store
            .chunks()?
            .into_iter()
            .map(|chunk| (chunk.id.clone(), chunk))
            .collect::<BTreeMap<_, _>>();
        if chunks.is_empty() {
            return Err(TextIndexError::InvalidArgument(
                "search index is empty".to_string(),
            ));
        }
        let vectors = self
            .store
            .vectors()?
            .into_iter()
            .map(|vector| (vector.chunk_id.clone(), vector))
            .collect::<BTreeMap<_, _>>();

        let lexical_scores = if query.mode != IndexSearchMode::Semantic {
            match self.store.lexical_candidates(&normalized_query, limit)? {
                Some(candidates) => candidates.into_iter().collect::<BTreeMap<_, _>>(),
                None => bm25_candidates(
                    chunks.values(),
                    &normalized_query,
                    limit,
                    &self.bm25_options,
                )?,
            }
        } else {
            BTreeMap::new()
        };

        let semantic_scores = if query.mode != IndexSearchMode::Lexical {
            let query_vector = self.embedder.embed_text(&normalized_query)?;
            vector_candidates(&vectors, &chunks, query_vector.as_slice(), limit)?
        } else {
            BTreeMap::new()
        };

        let normalized_semantic = normalize_scores(&semantic_scores);
        let normalized_lexical = normalize_scores(&lexical_scores);
        let mut ids = BTreeSet::new();
        ids.extend(normalized_semantic.keys().cloned());
        ids.extend(normalized_lexical.keys().cloned());
        if !required_phrases.is_empty() {
            ids.extend(chunks.values().filter_map(|chunk| {
                let matched_phrases = matched_required_phrases(
                    &chunk.text,
                    &required_phrases,
                    &self.corpus_options.processing,
                );
                (matched_phrases.len() == required_phrases.len()).then(|| chunk.id.clone())
            }));
        }

        let (semantic_weight, lexical_weight) = effective_weights(query);
        let mut results = ids
            .into_iter()
            .filter_map(|chunk_id| {
                let chunk = chunks.get(&chunk_id)?;
                if !matches_filter(chunk, &query.filter) {
                    return None;
                }
                let matched_phrases =
                    matched_required_phrases(&chunk.text, &required_phrases, &self.corpus_options.processing);
                if matched_phrases.len() != required_phrases.len() {
                    return None;
                }
                let semantic_raw = semantic_scores.get(&chunk_id).copied().unwrap_or(0.0);
                let lexical_raw = lexical_scores.get(&chunk_id).copied().unwrap_or(0.0);
                let semantic = normalized_semantic.get(&chunk_id).copied().unwrap_or(0.0);
                let lexical = normalized_lexical.get(&chunk_id).copied().unwrap_or(0.0);
                let score = semantic * semantic_weight + lexical * lexical_weight;
                Some(IndexSearchResult {
                    chunk_id: chunk.id.clone(),
                    document_id: chunk.document_id.clone(),
                    score,
                    snippet: build_snippet(&chunk.text, &normalized_query),
                    matched_phrases,
                    chunk: chunk.clone(),
                    score_breakdown: IndexScoreBreakdown {
                        semantic_score: semantic_raw,
                        lexical_score: lexical_raw,
                        normalized_semantic_score: semantic,
                        normalized_lexical_score: lexical,
                        semantic_weight,
                        lexical_weight,
                        explanation: query.explain.then(|| {
                            format!(
                                "score={score:.4}; semantic={semantic:.4}*{semantic_weight:.2}; lexical={lexical:.4}*{lexical_weight:.2}"
                            )
                        }),
                    },
                })
            })
            .collect::<Vec<_>>();
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| {
                    right
                        .score_breakdown
                        .normalized_semantic_score
                        .total_cmp(&left.score_breakdown.normalized_semantic_score)
                })
                .then_with(|| left.chunk_id.cmp(&right.chunk_id))
        });
        results.truncate(query.top_k);
        Ok(results)
    }

    pub fn inspect(&self) -> Result<IndexInspectReport> {
        self.store.inspect()
    }

    pub fn snapshot_plan(&self) -> Result<IndexSnapshotPlan> {
        let inspect = self.store.inspect()?;
        Ok(IndexSnapshotPlan {
            backend: self.store.backend_name().to_string(),
            document_count: inspect.document_count,
            chunk_count: inspect.chunk_count,
            vector_count: inspect.vector_count,
            facet_count: inspect.facet_count,
            attachment_count: inspect.attachment_count,
            chunking_strategy: self.options.chunking_strategy,
            commit_required_for_writes: self.store.backend_name() == "sqlite",
        })
    }
}

pub fn chunk_index_document(
    document: &IndexDocument,
    options: &IndexBuildOptions,
) -> Result<Vec<IndexChunk>> {
    validate_document(document)?;
    validate_build_options(options)?;
    match options.chunking_strategy {
        ChunkingStrategy::TokenWindow => chunk_token_windows(document, options),
        ChunkingStrategy::Sentence => chunk_text_spans(
            document,
            split_sentence_spans(&document.body, &options.processing)
                .into_iter()
                .map(|sentence| sentence.span),
            options,
        ),
        ChunkingStrategy::Paragraph => chunk_text_spans(
            document,
            split_paragraphs(&document.body)
                .into_iter()
                .map(|paragraph| paragraph.span),
            options,
        ),
    }
}

#[cfg(feature = "sqlite")]
pub struct SqliteIndexStore {
    connection: rusqlite::Connection,
}

#[cfg(feature = "sqlite")]
impl std::fmt::Debug for SqliteIndexStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteIndexStore").finish_non_exhaustive()
    }
}

#[cfg(feature = "sqlite")]
impl SqliteIndexStore {
    pub fn open(path: impl AsRef<Path>, commit: bool) -> Result<Self> {
        if !commit {
            return Err(TextIndexError::InvalidArgument(
                "SQLite writes require commit: true plus a path".to_string(),
            ));
        }
        let connection = rusqlite::Connection::open(path)?;
        let store = Self { connection };
        store.create_schema()?;
        Ok(store)
    }

    pub fn in_memory() -> Result<Self> {
        let connection = rusqlite::Connection::open_in_memory()?;
        let store = Self { connection };
        store.create_schema()?;
        Ok(store)
    }

    pub fn fts5_available(&self) -> bool {
        self.connection
            .execute_batch("CREATE VIRTUAL TABLE IF NOT EXISTS __text_index_fts5_probe USING fts5(value); DROP TABLE __text_index_fts5_probe;")
            .is_ok()
    }

    fn create_schema(&self) -> Result<()> {
        self.connection.execute_batch(
            r#"
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                body TEXT NOT NULL,
                language TEXT,
                attributes_json TEXT NOT NULL,
                source_json TEXT,
                provenance_json TEXT NOT NULL,
                annotations_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                token_start INTEGER,
                token_end INTEGER,
                metadata_json TEXT NOT NULL,
                source_json TEXT,
                provenance_json TEXT NOT NULL,
                annotations_json TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                text
            );
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                backend TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_blob BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunk_metadata_key_value
                ON chunk_metadata(key, value);
            CREATE TABLE IF NOT EXISTS semantic_facets (
                chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                facet_kind TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                score REAL,
                provenance_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS semantic_facets_lookup
                ON semantic_facets(facet_kind, key, value);
            CREATE TABLE IF NOT EXISTS analysis_attachments (
                document_id TEXT,
                chunk_id TEXT,
                kind TEXT NOT NULL,
                producer TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            "#,
        )?;
        Ok(())
    }
}

#[cfg(feature = "sqlite")]
impl TextIndexStore for SqliteIndexStore {
    fn backend_name(&self) -> &'static str {
        "sqlite"
    }

    fn upsert_document_state(
        &mut self,
        document: IndexDocument,
        chunks: Vec<IndexChunk>,
        vectors: Vec<StoredVector>,
        _options: &IndexBuildOptions,
    ) -> Result<IndexMutationReport> {
        let replaced = document_exists(&self.connection, &document.id)? as usize;
        let tx = self.connection.transaction()?;
        delete_document_tx(&tx, &document.id)?;
        tx.execute(
            "INSERT INTO documents (id, title, body, language, attributes_json, source_json, provenance_json, annotations_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                document.id,
                document.title,
                document.body,
                document.language,
                serde_json::to_string(&document.metadata.attributes)?,
                option_json(&document.metadata.source)?,
                serde_json::to_string(&document.metadata.provenance)?,
                serde_json::to_string(&document.metadata.annotations)?,
            ],
        )?;
        for attachment in &document.analysis_attachments {
            tx.execute(
                "INSERT INTO analysis_attachments (document_id, chunk_id, kind, producer, payload_json)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    attachment.document_id.as_deref().unwrap_or(&document.id),
                    attachment.chunk_id,
                    attachment.kind,
                    attachment.producer,
                    serde_json::to_string(&attachment.payload)?,
                ],
            )?;
        }
        for chunk in &chunks {
            tx.execute(
                "INSERT INTO chunks (id, document_id, ordinal, text, byte_start, byte_end, token_start, token_end, metadata_json, source_json, provenance_json, annotations_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                rusqlite::params![
                    chunk.id,
                    chunk.document_id,
                    chunk.ordinal as i64,
                    chunk.text,
                    chunk.byte_start as i64,
                    chunk.byte_end as i64,
                    chunk.token_start.map(|value| value as i64),
                    chunk.token_end.map(|value| value as i64),
                    serde_json::to_string(&chunk.metadata)?,
                    option_json(&chunk.source)?,
                    serde_json::to_string(&chunk.provenance)?,
                    serde_json::to_string(&chunk.annotations)?,
                ],
            )?;
            tx.execute(
                "INSERT INTO chunk_fts (chunk_id, document_id, text) VALUES (?1, ?2, ?3)",
                rusqlite::params![chunk.id, chunk.document_id, chunk.text],
            )?;
            for (key, value) in &chunk.metadata {
                tx.execute(
                    "INSERT INTO chunk_metadata (chunk_id, key, value) VALUES (?1, ?2, ?3)",
                    rusqlite::params![chunk.id, key, value],
                )?;
            }
            for facet in &chunk.semantic_facets {
                tx.execute(
                    "INSERT INTO semantic_facets (chunk_id, facet_kind, key, value, score, provenance_json)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![
                        chunk.id,
                        facet.kind,
                        facet.key,
                        facet.value,
                        facet.score,
                        serde_json::to_string(&facet.provenance)?,
                    ],
                )?;
            }
        }
        for vector in &vectors {
            tx.execute(
                "INSERT INTO vectors (chunk_id, backend, model_name, dimensions, vector_blob)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    vector.chunk_id,
                    vector.backend,
                    vector.model_name,
                    vector.dimensions as i64,
                    f32s_to_le_blob(&vector.vector),
                ],
            )?;
        }
        tx.commit()?;
        Ok(IndexMutationReport {
            documents_received: 1,
            documents_replaced: replaced,
            documents_removed: 0,
            documents_skipped: 0,
            chunks_indexed: chunks.len(),
            vectors_indexed: vectors.len(),
        })
    }

    fn remove_documents(&mut self, document_ids: &[String]) -> Result<usize> {
        let tx = self.connection.transaction()?;
        let mut removed = 0;
        for document_id in document_ids {
            if document_exists_tx(&tx, document_id)? {
                delete_document_tx(&tx, document_id)?;
                removed += 1;
            }
        }
        tx.commit()?;
        Ok(removed)
    }

    fn documents(&self) -> Result<Vec<IndexDocument>> {
        let mut stmt = self.connection.prepare(
            "SELECT id, title, body, language, attributes_json, source_json, provenance_json, annotations_json
             FROM documents ORDER BY id",
        )?;
        let rows = stmt.query_map([], document_from_row)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(TextIndexError::Sqlite)
    }

    fn chunks(&self) -> Result<Vec<IndexChunk>> {
        let mut stmt = self.connection.prepare(
            "SELECT id, document_id, ordinal, text, byte_start, byte_end, token_start, token_end,
                    metadata_json, source_json, provenance_json, annotations_json
             FROM chunks ORDER BY document_id, ordinal",
        )?;
        let rows = stmt.query_map([], |row| chunk_from_row(&self.connection, row))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(TextIndexError::Sqlite)
    }

    fn vectors(&self) -> Result<Vec<StoredVector>> {
        let mut stmt = self.connection.prepare(
            "SELECT chunk_id, backend, model_name, dimensions, vector_blob FROM vectors ORDER BY chunk_id",
        )?;
        let rows = stmt.query_map([], vector_from_row)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(TextIndexError::Sqlite)
    }

    fn lexical_candidates(&self, query: &str, limit: usize) -> Result<Option<Vec<(String, f32)>>> {
        if !self.fts5_available() {
            return Ok(None);
        }
        let fts_query = fts5_query(query);
        if fts_query.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let sql = "SELECT chunk_id, bm25(chunk_fts) AS rank
                   FROM chunk_fts
                   WHERE chunk_fts MATCH ?1
                   ORDER BY rank
                   LIMIT ?2";
        let mut stmt = self.connection.prepare(sql)?;
        let rows = stmt.query_map(rusqlite::params![fts_query, limit as i64], |row| {
            let chunk_id: String = row.get(0)?;
            let rank: f32 = row.get(1)?;
            Ok((chunk_id, -rank))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map(Some)
            .map_err(TextIndexError::Sqlite)
    }

    fn inspect(&self) -> Result<IndexInspectReport> {
        Ok(IndexInspectReport {
            document_count: count_table(&self.connection, "documents")?,
            chunk_count: count_table(&self.connection, "chunks")?,
            vector_count: count_table(&self.connection, "vectors")?,
            facet_count: count_table(&self.connection, "semantic_facets")?,
            attachment_count: count_table(&self.connection, "analysis_attachments")?,
            sqlite_fts5_available: Some(self.fts5_available()),
        })
    }
}

pub fn validate_index_document(document: &IndexDocument) -> Result<()> {
    if document.id.trim().is_empty() {
        return Err(TextIndexError::InvalidArgument(
            "document id must not be empty".to_string(),
        ));
    }
    if document.body.trim().is_empty() {
        return Err(TextIndexError::InvalidArgument(
            "document body must not be empty".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_index_build_options(options: &IndexBuildOptions) -> Result<()> {
    if options.chunk_tokens == 0 {
        return Err(TextIndexError::InvalidArgument(
            "chunk token size must be greater than zero".to_string(),
        ));
    }
    if options.chunk_overlap_tokens >= options.chunk_tokens {
        return Err(TextIndexError::InvalidArgument(
            "chunk overlap must be smaller than chunk token size".to_string(),
        ));
    }
    Ok(())
}

fn validate_document(document: &IndexDocument) -> Result<()> {
    validate_index_document(document)
}

fn validate_build_options(options: &IndexBuildOptions) -> Result<()> {
    validate_index_build_options(options)
}

fn validate_query(query: &IndexQuery) -> Result<()> {
    if query.top_k == 0 {
        return Err(TextIndexError::InvalidArgument(
            "search limit must be greater than zero".to_string(),
        ));
    }
    if query.candidate_limit == 0 {
        return Err(TextIndexError::InvalidArgument(
            "candidate limit must be greater than zero".to_string(),
        ));
    }
    let (semantic_weight, lexical_weight) = effective_weights(query);
    if !semantic_weight.is_finite()
        || !lexical_weight.is_finite()
        || semantic_weight < 0.0
        || lexical_weight < 0.0
        || semantic_weight + lexical_weight <= f32::EPSILON
    {
        return Err(TextIndexError::InvalidArgument(
            "search weights must be finite non-negative values and not both zero".to_string(),
        ));
    }
    let _ = normalize_query(&query.text, &TextProcessingOptions::default())?;
    Ok(())
}

fn chunk_token_windows(
    document: &IndexDocument,
    options: &IndexBuildOptions,
) -> Result<Vec<IndexChunk>> {
    let tokens = searchable_tokens(&document.body, &options.processing);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let mut chunks = Vec::new();
    let step = options.chunk_tokens - options.chunk_overlap_tokens;
    let mut start = 0;
    let mut ordinal = 0;
    while start < tokens.len() {
        let end = (start + options.chunk_tokens).min(tokens.len());
        let span = TextSpan {
            byte_start: tokens[start].span.byte_start,
            byte_end: tokens[end - 1].span.byte_end,
            char_start: tokens[start].span.char_start,
            char_end: tokens[end - 1].span.char_end,
        };
        push_chunk(
            &mut chunks,
            document,
            ordinal,
            span,
            Some(start),
            Some(end),
            options,
        )?;
        ordinal += 1;
        if end == tokens.len() {
            break;
        }
        start += step;
    }
    Ok(chunks)
}

fn chunk_text_spans(
    document: &IndexDocument,
    spans: impl IntoIterator<Item = TextSpan>,
    options: &IndexBuildOptions,
) -> Result<Vec<IndexChunk>> {
    let mut chunks = Vec::new();
    for (ordinal, span) in spans.into_iter().enumerate() {
        push_chunk(&mut chunks, document, ordinal, span, None, None, options)?;
    }
    Ok(chunks)
}

fn push_chunk(
    chunks: &mut Vec<IndexChunk>,
    document: &IndexDocument,
    ordinal: usize,
    span: TextSpan,
    token_start: Option<usize>,
    token_end: Option<usize>,
    _options: &IndexBuildOptions,
) -> Result<()> {
    let text = document
        .body
        .get(span.byte_start..span.byte_end)
        .ok_or_else(|| {
            TextIndexError::InvalidArgument(
                "chunk span did not align to valid UTF-8 boundaries".to_string(),
            )
        })?
        .trim()
        .to_string();
    if text.is_empty() {
        return Ok(());
    }
    let mut metadata = document.metadata.attributes.clone();
    if let Some(language) = &document.language {
        metadata
            .entry("language".to_string())
            .or_insert(language.clone());
    }
    if let Some(title) = &document.title {
        metadata
            .entry(TITLE_METADATA_KEY.to_string())
            .or_insert_with(|| title.clone());
    }
    chunks.push(IndexChunk {
        id: format!("{}:{ordinal}", document.id),
        document_id: document.id.clone(),
        ordinal,
        text,
        byte_start: span.byte_start,
        byte_end: span.byte_end,
        token_start,
        token_end,
        metadata,
        source: document.metadata.source.clone(),
        provenance: document.metadata.provenance.clone(),
        annotations: document.metadata.annotations.clone(),
        semantic_facets: document.semantic_facets.clone(),
    });
    Ok(())
}

fn searchable_tokens(text: &str, processing: &TextProcessingOptions) -> Vec<text_core::Token> {
    tokenize(text, processing)
        .into_iter()
        .filter(|token| {
            matches!(
                token.kind,
                TokenKind::Word
                    | TokenKind::Number
                    | TokenKind::Url
                    | TokenKind::Email
                    | TokenKind::Mention
                    | TokenKind::Hashtag
            )
        })
        .collect()
}

fn embed_chunks<B: TextEmbedderBackend>(
    embedder: &B,
    chunks: &[IndexChunk],
) -> Result<Vec<StoredVector>> {
    let texts = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<Vec<_>>();
    let vectors = embedder.embed_batch(&texts)?;
    if vectors.len() != chunks.len() {
        return Err(TextIndexError::InvalidState(
            "embedder returned a different number of vectors than input chunks".to_string(),
        ));
    }
    let info = embedder.model_info();
    chunks
        .iter()
        .zip(vectors)
        .map(|(chunk, vector)| {
            let dimensions = vector.dimensions();
            if info.dimensions > 0 && info.dimensions != dimensions {
                return Err(TextIndexError::InvalidState(format!(
                    "embedder produced vectors that did not match advertised dimensions {}",
                    info.dimensions
                )));
            }
            Ok(StoredVector {
                chunk_id: chunk.id.clone(),
                backend: format!("{:?}", info.backend),
                model_name: info.model_name.clone(),
                dimensions,
                vector: vector.as_slice().to_vec(),
            })
        })
        .collect()
}

fn bm25_candidates<'a>(
    chunks: impl Iterator<Item = &'a IndexChunk>,
    query: &str,
    limit: usize,
    options: &Bm25Options,
) -> Result<BTreeMap<String, f32>> {
    let mut bm25 = Bm25Corpus::new(options.clone());
    for chunk in chunks {
        bm25.add_document(chunk.id.clone(), &chunk.text)?;
    }
    Ok(bm25
        .search(query, limit)?
        .into_iter()
        .map(|hit| (hit.id, hit.score))
        .collect())
}

fn vector_candidates(
    vectors: &BTreeMap<String, StoredVector>,
    chunks: &BTreeMap<String, IndexChunk>,
    query_vector: &[f32],
    limit: usize,
) -> Result<BTreeMap<String, f32>> {
    let mut index = VectorSearchIndex::new();
    for vector in vectors.values() {
        if !chunks.contains_key(&vector.chunk_id) {
            continue;
        }
        index.add(VectorRecord::with_payload(
            VectorRecordId::from(vector.chunk_id.clone()),
            DenseVector::new(vector.vector.clone())?,
            VectorRecordMetadata::default(),
        ))?;
    }
    if index.records().is_empty() {
        return Ok(BTreeMap::new());
    }
    let hits = index.search_filtered(query_vector, limit, None)?;
    Ok(hits
        .into_iter()
        .map(|hit| (hit.id.into_string(), hit.score))
        .collect())
}

fn normalize_query(query: &str, processing: &TextProcessingOptions) -> Result<String> {
    let terms = searchable_tokens(query, processing)
        .into_iter()
        .map(|token| token.normalized)
        .collect::<Vec<_>>();
    if terms.is_empty() {
        return Err(TextIndexError::InvalidArgument(
            "query must contain at least one searchable term".to_string(),
        ));
    }
    Ok(terms.join(" "))
}

fn normalize_phrase_text(text: &str, processing: &TextProcessingOptions) -> Option<String> {
    let terms = searchable_tokens(text, processing)
        .into_iter()
        .map(|token| token.normalized)
        .collect::<Vec<_>>();
    (!terms.is_empty()).then(|| terms.join(" "))
}

fn normalize_required_phrases(
    phrases: &[String],
    processing: &TextProcessingOptions,
) -> Vec<String> {
    phrases
        .iter()
        .filter_map(|phrase| normalize_phrase_text(phrase, processing))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn matched_required_phrases(
    text: &str,
    required_phrases: &[String],
    processing: &TextProcessingOptions,
) -> Vec<String> {
    if required_phrases.is_empty() {
        return Vec::new();
    }
    let Some(normalized_text) = normalize_phrase_text(text, processing) else {
        return Vec::new();
    };
    required_phrases
        .iter()
        .filter(|phrase| normalized_text.contains(phrase.as_str()))
        .cloned()
        .collect()
}

fn normalize_scores(scores: &BTreeMap<String, f32>) -> BTreeMap<String, f32> {
    if scores.is_empty() {
        return BTreeMap::new();
    }
    let min = scores.values().copied().fold(f32::INFINITY, f32::min);
    let max = scores.values().copied().fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() <= f32::EPSILON {
        return scores.keys().cloned().map(|id| (id, 1.0)).collect();
    }
    scores
        .iter()
        .map(|(id, score)| (id.clone(), (score - min) / (max - min)))
        .collect()
}

fn effective_weights(query: &IndexQuery) -> (f32, f32) {
    match query.mode {
        IndexSearchMode::Lexical => (0.0, 1.0),
        IndexSearchMode::Semantic => (1.0, 0.0),
        IndexSearchMode::Hybrid => (query.semantic_weight, query.lexical_weight),
    }
}

fn matches_filter(chunk: &IndexChunk, filter: &IndexFilter) -> bool {
    if !filter.document_ids.is_empty() && !filter.document_ids.contains(&chunk.document_id) {
        return false;
    }
    if !filter
        .metadata_equals
        .iter()
        .all(|(key, value)| chunk.metadata.get(key) == Some(value))
    {
        return false;
    }
    if !filter.metadata_contains.iter().all(|(key, needle)| {
        chunk
            .metadata
            .get(key)
            .is_some_and(|value| value.contains(needle))
    }) {
        return false;
    }
    let tags = metadata_tags(&chunk.metadata);
    if !filter
        .required_tags
        .iter()
        .all(|tag| tags.iter().any(|candidate| candidate == tag))
    {
        return false;
    }
    if !filter.semantic_facets.iter().all(|required| {
        chunk.semantic_facets.iter().any(|facet| {
            facet.kind == required.kind
                && facet.key == required.key
                && facet.value == required.value
        })
    }) {
        return false;
    }
    if !filter.source_kinds.is_empty()
        && !chunk
            .source
            .as_ref()
            .and_then(|source| source.source_kind.as_ref())
            .is_some_and(|kind| filter.source_kinds.contains(kind))
    {
        return false;
    }
    if !filter.source_ids.is_empty()
        && !chunk
            .source
            .as_ref()
            .and_then(|source| source.source_id.as_ref())
            .is_some_and(|source_id| filter.source_ids.contains(source_id))
    {
        return false;
    }
    if !timestamp_matches(
        chunk
            .source
            .as_ref()
            .and_then(|source| source.media_timestamp),
        filter.timestamp_seconds_min,
        filter.timestamp_seconds_max,
    ) {
        return false;
    }
    if !filter.provenance_operations.is_empty()
        && !chunk.provenance.iter().any(|provenance| {
            provenance
                .operation
                .as_ref()
                .is_some_and(|operation| filter.provenance_operations.contains(operation))
        })
    {
        return false;
    }
    true
}

fn timestamp_matches(
    timestamp: Option<TimestampContract>,
    min: Option<f64>,
    max: Option<f64>,
) -> bool {
    let Some(timestamp) = timestamp else {
        return min.is_none() && max.is_none();
    };
    let seconds = timestamp.seconds();
    min.is_none_or(|min| seconds >= min) && max.is_none_or(|max| seconds <= max)
}

fn metadata_tags(metadata: &BTreeMap<String, String>) -> Vec<String> {
    metadata
        .get("tags")
        .map(|tags| {
            tags.split([',', ';'])
                .flat_map(|group| group.split_whitespace())
                .map(str::trim)
                .filter(|tag| !tag.is_empty())
                .map(ToString::to_string)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default()
}

fn build_snippet(text: &str, normalized_query: &str) -> String {
    let terms = normalized_query
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>();
    let text_lower = text.to_lowercase();
    let match_start = terms.iter().filter_map(|term| text_lower.find(term)).min();
    let snippet = match match_start {
        Some(byte_index) => {
            let start = text[..byte_index]
                .char_indices()
                .rev()
                .nth(60)
                .map(|(index, _)| index)
                .unwrap_or(0);
            let end = text[byte_index..]
                .char_indices()
                .nth(160)
                .map(|(index, _)| byte_index + index)
                .unwrap_or(text.len());
            text[start..end].trim()
        }
        None => text
            .char_indices()
            .nth(160)
            .map(|(index, _)| &text[..index])
            .unwrap_or(text)
            .trim(),
    };
    snippet.to_string()
}

fn remove_document_state(
    documents: &mut BTreeMap<String, IndexDocument>,
    chunks: &mut BTreeMap<String, IndexChunk>,
    vectors: &mut BTreeMap<String, StoredVector>,
    document_chunks: &mut BTreeMap<String, Vec<String>>,
    document_id: &str,
) {
    documents.remove(document_id);
    if let Some(chunk_ids) = document_chunks.remove(document_id) {
        for chunk_id in chunk_ids {
            chunks.remove(&chunk_id);
            vectors.remove(&chunk_id);
        }
    }
}

fn default_mode() -> IndexSearchMode {
    IndexSearchMode::Hybrid
}

fn default_top_k() -> usize {
    10
}

fn default_candidate_limit() -> usize {
    64
}

fn default_semantic_weight() -> f32 {
    0.6
}

fn default_lexical_weight() -> f32 {
    0.4
}

#[cfg(feature = "sqlite")]
fn f32s_to_le_blob(values: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(values.len() * 4);
    for value in values {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

#[cfg(feature = "sqlite")]
fn f32s_from_le_blob(blob: &[u8]) -> Result<Vec<f32>> {
    if blob.len() % 4 != 0 {
        return Err(TextIndexError::InvalidState(
            "vector blob length is not a multiple of four".to_string(),
        ));
    }
    Ok(blob
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(feature = "sqlite")]
fn option_json<T: Serialize>(value: &Option<T>) -> Result<Option<String>> {
    value
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(TextIndexError::Serde)
}

#[cfg(feature = "sqlite")]
fn document_exists(connection: &rusqlite::Connection, document_id: &str) -> Result<bool> {
    use rusqlite::OptionalExtension;
    Ok(connection
        .query_row(
            "SELECT 1 FROM documents WHERE id = ?1",
            [document_id],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

#[cfg(feature = "sqlite")]
fn document_exists_tx(tx: &rusqlite::Transaction<'_>, document_id: &str) -> Result<bool> {
    use rusqlite::OptionalExtension;
    Ok(tx
        .query_row(
            "SELECT 1 FROM documents WHERE id = ?1",
            [document_id],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

#[cfg(feature = "sqlite")]
fn delete_document_tx(tx: &rusqlite::Transaction<'_>, document_id: &str) -> Result<()> {
    let chunk_ids = {
        let mut stmt = tx.prepare("SELECT id FROM chunks WHERE document_id = ?1")?;
        let rows = stmt.query_map([document_id], |row| row.get::<_, String>(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    for chunk_id in chunk_ids {
        tx.execute("DELETE FROM chunk_fts WHERE chunk_id = ?1", [&chunk_id])?;
    }
    tx.execute("DELETE FROM documents WHERE id = ?1", [document_id])?;
    tx.execute(
        "DELETE FROM analysis_attachments WHERE document_id = ?1",
        [document_id],
    )?;
    Ok(())
}

#[cfg(feature = "sqlite")]
fn count_table(connection: &rusqlite::Connection, table: &str) -> Result<usize> {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    let count: i64 = connection.query_row(&sql, [], |row| row.get(0))?;
    Ok(count as usize)
}

#[cfg(feature = "sqlite")]
fn document_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<IndexDocument> {
    let attributes_json: String = row.get(4)?;
    let source_json: Option<String> = row.get(5)?;
    let provenance_json: String = row.get(6)?;
    let annotations_json: String = row.get(7)?;
    Ok(IndexDocument {
        id: row.get(0)?,
        title: row.get(1)?,
        body: row.get(2)?,
        language: row.get(3)?,
        metadata: IndexDocumentMetadata {
            attributes: serde_json::from_str(&attributes_json).map_err(json_to_sqlite)?,
            source: source_json
                .map(|json| serde_json::from_str(&json).map_err(json_to_sqlite))
                .transpose()?,
            provenance: serde_json::from_str(&provenance_json).map_err(json_to_sqlite)?,
            annotations: serde_json::from_str(&annotations_json).map_err(json_to_sqlite)?,
        },
        analysis_attachments: Vec::new(),
        semantic_facets: Vec::new(),
    })
}

#[cfg(feature = "sqlite")]
fn chunk_from_row(
    connection: &rusqlite::Connection,
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<IndexChunk> {
    let chunk_id: String = row.get(0)?;
    let metadata_json: String = row.get(8)?;
    let source_json: Option<String> = row.get(9)?;
    let provenance_json: String = row.get(10)?;
    let annotations_json: String = row.get(11)?;
    Ok(IndexChunk {
        id: chunk_id.clone(),
        document_id: row.get(1)?,
        ordinal: row.get::<_, i64>(2)? as usize,
        text: row.get(3)?,
        byte_start: row.get::<_, i64>(4)? as usize,
        byte_end: row.get::<_, i64>(5)? as usize,
        token_start: row.get::<_, Option<i64>>(6)?.map(|value| value as usize),
        token_end: row.get::<_, Option<i64>>(7)?.map(|value| value as usize),
        metadata: serde_json::from_str(&metadata_json).map_err(json_to_sqlite)?,
        source: source_json
            .map(|json| serde_json::from_str(&json).map_err(json_to_sqlite))
            .transpose()?,
        provenance: serde_json::from_str(&provenance_json).map_err(json_to_sqlite)?,
        annotations: serde_json::from_str(&annotations_json).map_err(json_to_sqlite)?,
        semantic_facets: load_facets(connection, &chunk_id)?,
    })
}

#[cfg(feature = "sqlite")]
fn load_facets(
    connection: &rusqlite::Connection,
    chunk_id: &str,
) -> rusqlite::Result<Vec<SemanticFacet>> {
    let mut stmt = connection.prepare(
        "SELECT facet_kind, key, value, score, provenance_json
         FROM semantic_facets WHERE chunk_id = ?1 ORDER BY facet_kind, key, value",
    )?;
    let rows = stmt.query_map([chunk_id], |row| {
        let provenance_json: String = row.get(4)?;
        Ok(SemanticFacet {
            kind: row.get(0)?,
            key: row.get(1)?,
            value: row.get(2)?,
            score: row.get(3)?,
            provenance: serde_json::from_str(&provenance_json).map_err(json_to_sqlite)?,
        })
    })?;
    rows.collect()
}

#[cfg(feature = "sqlite")]
fn vector_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredVector> {
    let blob: Vec<u8> = row.get(4)?;
    let vector = f32s_from_le_blob(&blob).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            blob.len(),
            rusqlite::types::Type::Blob,
            Box::new(error),
        )
    })?;
    Ok(StoredVector {
        chunk_id: row.get(0)?,
        backend: row.get(1)?,
        model_name: row.get(2)?,
        dimensions: row.get::<_, i64>(3)? as usize,
        vector,
    })
}

#[cfg(feature = "sqlite")]
fn json_to_sqlite(error: serde_json::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
}

#[cfg(feature = "sqlite")]
fn fts5_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use text_core::TextProvenance;

    fn index() -> MemoryTextIndex<HashedTextEmbedder> {
        MemoryTextIndex::new_memory()
            .unwrap()
            .with_options(IndexBuildOptions {
                chunk_tokens: 8,
                chunk_overlap_tokens: 0,
                ..IndexBuildOptions::default()
            })
    }

    #[test]
    fn memory_index_builds_and_searches_hybrid_with_default_weights() {
        let mut index = index();
        index
            .upsert_documents(&[
                IndexDocument::new("doc-1", "Rust crates support durable text indexing."),
                IndexDocument::new("doc-2", "Music playlists and recommendations."),
            ])
            .unwrap();
        let results = index
            .search(&IndexQuery {
                text: "durable text index".to_string(),
                explain: true,
                ..IndexQuery::default()
            })
            .unwrap();
        assert_eq!(results[0].document_id, "doc-1");
        assert_eq!(results[0].score_breakdown.semantic_weight, 0.6);
        assert_eq!(results[0].score_breakdown.lexical_weight, 0.4);
        assert!(results[0].score_breakdown.explanation.is_some());
    }

    #[test]
    fn search_reports_and_requires_matched_phrases() {
        let mut index = index();
        index
            .upsert_documents(&[
                IndexDocument::new(
                    "doc-1",
                    "Climate policy needs public funding. Markets price risk.",
                ),
                IndexDocument::new(
                    "doc-2",
                    "Climate policy needs private finance and separate funding.",
                ),
            ])
            .unwrap();
        let results = index
            .search(&IndexQuery {
                text: "climate policy public funding".to_string(),
                required_phrases: vec!["public funding".to_string()],
                ..IndexQuery::default()
            })
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, "doc-1");
        assert_eq!(results[0].matched_phrases, vec!["public funding"]);
    }

    #[test]
    fn required_phrases_are_filtered_before_top_k_truncation() {
        let mut index = index();
        index
            .upsert_documents(&[
                IndexDocument::new("doc-1", "Climate policy mentions funding separately."),
                IndexDocument::new("doc-2", "Public funding supports climate adaptation."),
                IndexDocument::new("doc-3", "Public funding expands climate resilience."),
            ])
            .unwrap();
        let results = index
            .search(&IndexQuery {
                text: "climate funding".to_string(),
                top_k: 2,
                candidate_limit: 1,
                required_phrases: vec!["public funding".to_string()],
                ..IndexQuery::default()
            })
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|result| result.matched_phrases == ["public funding"]));
    }

    #[test]
    fn exact_term_match_stays_above_semantic_neighbor_in_balanced_hybrid() {
        let mut index = index();
        index
            .upsert_documents(&[
                IndexDocument::new("doc-1", "The archive describes neural ranking experiments."),
                IndexDocument::new("doc-2", "Garden notes mention tomatoes and irrigation."),
            ])
            .unwrap();
        let results = index
            .search(&IndexQuery {
                text: "neural ranking".to_string(),
                semantic_weight: 0.5,
                lexical_weight: 0.5,
                ..IndexQuery::default()
            })
            .unwrap();
        assert_eq!(results[0].document_id, "doc-1");
        assert!(results[0].score_breakdown.lexical_score > 0.0);
    }

    #[test]
    fn equal_scores_sort_by_chunk_id_for_stable_results() {
        let mut index = index();
        index
            .upsert_documents(&[
                IndexDocument::new("doc-b", "Shared passage text."),
                IndexDocument::new("doc-a", "Shared passage text."),
            ])
            .unwrap();
        let results = index
            .search(&IndexQuery {
                text: "shared passage".to_string(),
                mode: IndexSearchMode::Lexical,
                top_k: 2,
                ..IndexQuery::default()
            })
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|result| result.chunk_id.as_str())
                .collect::<Vec<_>>(),
            vec!["doc-a:0", "doc-b:0"]
        );
    }

    #[test]
    fn upsert_replaces_chunks_vectors_and_facets_atomically() {
        let mut index = index();
        let mut first = IndexDocument::new("doc-1", "old durable text");
        first.semantic_facets.push(SemanticFacet {
            kind: "topic".to_string(),
            key: "name".to_string(),
            value: "old".to_string(),
            score: Some(0.8),
            provenance: Vec::new(),
        });
        index.upsert_documents(&[first]).unwrap();
        let mut second = IndexDocument::new("doc-1", "new semantic index");
        second.semantic_facets.push(SemanticFacet {
            kind: "topic".to_string(),
            key: "name".to_string(),
            value: "new".to_string(),
            score: Some(0.9),
            provenance: Vec::new(),
        });
        let report = index.upsert_documents(&[second]).unwrap();
        assert_eq!(report.documents_replaced, 1);
        let inspect = index.inspect().unwrap();
        assert_eq!(inspect.document_count, 1);
        assert_eq!(inspect.facet_count, 1);
        let results = index.search(&IndexQuery::new("new", 1)).unwrap();
        assert_eq!(results[0].chunk.semantic_facets[0].value, "new");
    }

    #[test]
    fn filters_metadata_source_timestamp_provenance_and_facets() {
        let mut index = index();
        let mut document = IndexDocument::new("doc-1", "Alice explains hybrid search.");
        document
            .metadata
            .attributes
            .insert("speaker".to_string(), "alice".to_string());
        document.metadata.source = Some(TextSourceRef {
            source_id: Some("transcript".to_string()),
            source_kind: Some("audio".to_string()),
            uri: None,
            media_timestamp: Some(
                text_core::Timestamp::new(10, text_core::Timebase::new(1, 1)).into(),
            ),
            duration_seconds: Some(3.0),
        });
        document.metadata.provenance.push(TextProvenance {
            crate_name: Some("text-analysis".to_string()),
            operation: Some("analysis.document".to_string()),
            model_id: None,
            runtime: None,
            confidence: Some(1.0),
        });
        document.semantic_facets.push(SemanticFacet {
            kind: "entity".to_string(),
            key: "person".to_string(),
            value: "Alice".to_string(),
            score: None,
            provenance: Vec::new(),
        });
        index.upsert_documents(&[document]).unwrap();
        let results = index
            .search(&IndexQuery {
                text: "hybrid search".to_string(),
                required_phrases: vec!["hybrid search".to_string()],
                filter: IndexFilter {
                    metadata_equals: BTreeMap::from([("speaker".to_string(), "alice".to_string())]),
                    semantic_facets: vec![SemanticFacetFilter {
                        kind: "entity".to_string(),
                        key: "person".to_string(),
                        value: "Alice".to_string(),
                    }],
                    source_kinds: BTreeSet::from(["audio".to_string()]),
                    source_ids: BTreeSet::from(["transcript".to_string()]),
                    timestamp_seconds_min: Some(9.0),
                    timestamp_seconds_max: Some(11.0),
                    provenance_operations: BTreeSet::from(["analysis.document".to_string()]),
                    ..IndexFilter::default()
                },
                ..IndexQuery::default()
            })
            .unwrap();
        assert_eq!(results.len(), 1);
    }
}
