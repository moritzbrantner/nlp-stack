#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::{BTreeMap, BTreeSet};

mod storage;

use serde::{Deserialize, Serialize};
use text_core::{
    split_paragraphs, split_sentence_spans, tokenize, TextAnnotationSpan, TextDocument,
    TextDocumentContract, TextProcessingOptions, TextProvenance, TextSegmentContract,
    TextSourceRef, TextSpan, TokenKind,
};
use text_core::{DetectError, Result as CoreResult};
use text_embeddings::{EmbeddingModelInfo, TextEmbedderBackend};
use text_index::{
    ChunkingStrategy as IndexChunkingStrategy, IndexBuildOptions, IndexDocument,
    IndexDocumentMetadata, IndexMutationReport, MemoryIndexStore, TextIndex, TextIndexError,
};
use text_lexical::{Bm25Corpus, Bm25Options, CorpusOptions, TextCorpus, TextCorpusDocument};
use text_model_runtime::{TextReranker, TextRuntimeBackend};
use vector_analysis_index::{
    SerializableVectorRecord, VectorRecord, VectorRecordMetadata, VectorSearchFilter,
    VectorSearchIndex,
};

pub use storage::*;

const TITLE_METADATA_KEY: &str = "__title";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for search document.
pub struct SearchDocument {
    /// Identifier for this value.
    pub id: String,
    /// The title value.
    #[serde(default)]
    pub title: Option<String>,
    /// The body value.
    pub body: String,
    /// Metadata associated with this value.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// Optional rich source reference.
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    /// Provenance records for this document.
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    /// Annotation spans associated with this document.
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
}

impl SearchDocument {
    /// Creates a new value.
    pub fn new(id: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: None,
            body: body.into(),
            metadata: BTreeMap::new(),
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
        }
    }

    /// Builds this value from text document.
    pub fn from_text_document(document: &TextDocument<'_>) -> Self {
        let mut metadata = BTreeMap::new();
        if let Some(language) = document.language {
            metadata.insert("language".to_string(), language.to_string());
        }
        if let Some(timestamp) = document.timestamp {
            metadata.insert(
                "timestamp_seconds".to_string(),
                timestamp.seconds().to_string(),
            );
        }
        Self {
            id: document.id.to_string(),
            title: None,
            body: document.text.to_string(),
            metadata,
            source: document.timestamp.map(|timestamp| TextSourceRef {
                source_id: None,
                source_kind: Some("text_document".to_string()),
                uri: None,
                media_timestamp: Some(timestamp.into()),
                duration_seconds: None,
            }),
            provenance: Vec::new(),
            annotations: Vec::new(),
        }
    }

    /// Builds this value from a portable text document contract.
    pub fn from_text_document_contract(document: &TextDocumentContract) -> Self {
        let mut metadata = document.attributes.clone();
        if let Some(language) = &document.language {
            metadata.insert("language".to_string(), language.clone());
        }
        if let Some(timestamp) = document.timestamp {
            metadata.insert(
                "timestamp_seconds".to_string(),
                timestamp.seconds().to_string(),
            );
        }
        Self {
            id: document.id.clone(),
            title: None,
            body: document.text.clone(),
            metadata,
            source: document.source.clone(),
            provenance: document.provenance.clone(),
            annotations: document.annotations.clone(),
        }
    }

    /// Builds this value from a portable text segment contract.
    pub fn from_text_segment_contract(segment: &TextSegmentContract) -> Self {
        let mut metadata = segment.attributes.clone();
        if let Some(language) = &segment.language {
            metadata.insert("language".to_string(), language.clone());
        }
        if let Some(timestamp) = segment.timestamp {
            metadata.insert(
                "timestamp_seconds".to_string(),
                timestamp.seconds().to_string(),
            );
        }
        if let Some(duration_seconds) = segment.duration_seconds {
            metadata.insert("duration_seconds".to_string(), duration_seconds.to_string());
        }
        Self {
            id: segment
                .document_id()
                .unwrap_or_else(|| segment.segment_index.to_string()),
            title: None,
            body: segment.text.clone(),
            metadata,
            source: segment.source.clone().or_else(|| {
                (segment.timestamp.is_some() || segment.duration_seconds.is_some()).then(|| {
                    TextSourceRef {
                        source_id: segment.stream_id.clone(),
                        source_kind: Some("text_segment".to_string()),
                        uri: None,
                        media_timestamp: segment.timestamp,
                        duration_seconds: segment.duration_seconds,
                    }
                })
            }),
            provenance: segment.provenance.clone(),
            annotations: segment.annotations.clone(),
        }
    }

    /// Builds this value from a lexical corpus document without losing rich metadata.
    pub fn from_text_corpus_document(document: &TextCorpusDocument) -> Self {
        let mut metadata = document.metadata.clone();
        if let Some(language) = &document.language {
            metadata.insert("language".to_string(), language.clone());
        }
        if let Some(timestamp) = document.timestamp {
            metadata.insert(
                "timestamp_seconds".to_string(),
                timestamp.seconds().to_string(),
            );
        }
        if let Some(duration_seconds) = document
            .source
            .as_ref()
            .and_then(|source| source.duration_seconds)
        {
            metadata.insert("duration_seconds".to_string(), duration_seconds.to_string());
        }
        Self {
            id: document.id.clone(),
            title: None,
            body: document.text.clone(),
            metadata,
            source: document.source.clone(),
            provenance: document.provenance.clone(),
            annotations: document.annotations.clone(),
        }
    }

    /// Builds retrieval documents from a lexical corpus.
    pub fn from_text_corpus(corpus: &TextCorpus) -> Vec<Self> {
        corpus
            .documents
            .iter()
            .map(Self::from_text_corpus_document)
            .collect()
    }
}

impl From<&SearchDocument> for IndexDocument {
    fn from(document: &SearchDocument) -> Self {
        let mut attributes = document.metadata.clone();
        if let Some(language) = attributes.get("language").cloned() {
            attributes.entry("language".to_string()).or_insert(language);
        }
        Self {
            id: document.id.clone(),
            title: document.title.clone(),
            body: document.body.clone(),
            language: document.metadata.get("language").cloned(),
            metadata: IndexDocumentMetadata {
                attributes,
                source: document.source.clone(),
                provenance: document.provenance.clone(),
                annotations: document.annotations.clone(),
            },
            analysis_attachments: Vec::new(),
            semantic_facets: Vec::new(),
        }
    }
}

pub trait IntoSearchDocument {
    fn into_search_document(self) -> SearchDocument;
}

impl IntoSearchDocument for TextDocument<'_> {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_document(&self)
    }
}

impl IntoSearchDocument for &TextDocument<'_> {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_document(self)
    }
}

impl IntoSearchDocument for TextDocumentContract {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_document_contract(&self)
    }
}

impl IntoSearchDocument for &TextDocumentContract {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_document_contract(self)
    }
}

impl IntoSearchDocument for TextSegmentContract {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_segment_contract(&self)
    }
}

impl IntoSearchDocument for &TextSegmentContract {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_segment_contract(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for document chunk.
pub struct DocumentChunk {
    /// The chunk identifier value.
    pub chunk_id: String,
    /// The document identifier value.
    pub document_id: String,
    /// Text content for this value.
    pub text: String,
    /// The ordinal value.
    pub ordinal: usize,
    /// Metadata associated with this value.
    pub metadata: BTreeMap<String, String>,
    /// Optional rich source reference.
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    /// Provenance records for this chunk.
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    /// Annotation spans associated with this chunk.
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
}

impl DocumentChunk {
    /// Converts this retrieval chunk into a portable document contract.
    pub fn to_text_document_contract(&self) -> TextDocumentContract {
        TextDocumentContract {
            id: self.chunk_id.clone(),
            text: self.text.clone(),
            language: self.metadata.get("language").cloned(),
            timestamp: self
                .source
                .as_ref()
                .and_then(|source| source.media_timestamp),
            attributes: self.metadata.clone(),
            source: self.source.clone(),
            provenance: self.provenance.clone(),
            annotations: self.annotations.clone(),
        }
    }

    /// Converts this retrieval chunk into a portable segment contract.
    pub fn to_text_segment_contract(&self) -> TextSegmentContract {
        TextSegmentContract {
            stream_id: Some(self.document_id.clone()),
            segment_index: self.ordinal as u64,
            text: self.text.clone(),
            language: self.metadata.get("language").cloned(),
            timestamp: self
                .source
                .as_ref()
                .and_then(|source| source.media_timestamp),
            duration_seconds: self
                .source
                .as_ref()
                .and_then(|source| source.duration_seconds),
            is_final: true,
            attributes: self.metadata.clone(),
            source: self.source.clone(),
            provenance: self.provenance.clone(),
            annotations: self.annotations.clone(),
        }
    }
}

impl From<&DocumentChunk> for text_index::IndexChunk {
    fn from(chunk: &DocumentChunk) -> Self {
        text_index::IndexChunk {
            id: chunk.chunk_id.clone(),
            document_id: chunk.document_id.clone(),
            ordinal: chunk.ordinal,
            text: chunk.text.clone(),
            byte_start: 0,
            byte_end: chunk.text.len(),
            token_start: None,
            token_end: None,
            metadata: chunk.metadata.clone(),
            source: chunk.source.clone(),
            provenance: chunk.provenance.clone(),
            annotations: chunk.annotations.clone(),
            semantic_facets: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for ingestion options.
pub struct IngestionOptions {
    /// The chunk tokens value.
    pub chunk_tokens: usize,
    /// The chunk overlap tokens value.
    pub chunk_overlap_tokens: usize,
    /// The store raw text value.
    pub store_raw_text: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Strategy for constructing retrieval chunks.
pub enum ChunkingStrategy {
    /// Fixed token windows with overlap.
    TokenWindow,
    /// One sentence per chunk.
    Sentence,
    /// One paragraph per chunk.
    Paragraph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Options for explicit chunk construction.
pub struct ChunkingOptions {
    /// Chunking strategy.
    pub strategy: ChunkingStrategy,
    /// Token-window options used when `strategy` is `TokenWindow`.
    pub ingestion: IngestionOptions,
}

impl Default for ChunkingOptions {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::TokenWindow,
            ingestion: IngestionOptions::default(),
        }
    }
}

impl Default for IngestionOptions {
    fn default() -> Self {
        Self {
            chunk_tokens: 256,
            chunk_overlap_tokens: 32,
            store_raw_text: true,
        }
    }
}

impl From<IngestionOptions> for IndexBuildOptions {
    fn from(options: IngestionOptions) -> Self {
        Self {
            chunking_strategy: IndexChunkingStrategy::TokenWindow,
            chunk_tokens: options.chunk_tokens,
            chunk_overlap_tokens: options.chunk_overlap_tokens,
            store_raw_text: options.store_raw_text,
            commit: false,
            processing: TextProcessingOptions::default(),
        }
    }
}

impl From<ChunkingOptions> for IndexBuildOptions {
    fn from(options: ChunkingOptions) -> Self {
        let chunking_strategy = match options.strategy {
            ChunkingStrategy::TokenWindow => IndexChunkingStrategy::TokenWindow,
            ChunkingStrategy::Sentence => IndexChunkingStrategy::Sentence,
            ChunkingStrategy::Paragraph => IndexChunkingStrategy::Paragraph,
        };
        Self {
            chunking_strategy,
            chunk_tokens: options.ingestion.chunk_tokens,
            chunk_overlap_tokens: options.ingestion.chunk_overlap_tokens,
            store_raw_text: options.ingestion.store_raw_text,
            commit: false,
            processing: TextProcessingOptions::default(),
        }
    }
}

/// Builds the new in-memory `text-index` from legacy retrieval documents.
pub fn build_memory_text_index<B: TextEmbedderBackend>(
    embedder: B,
    documents: &[SearchDocument],
    options: &IngestionOptions,
) -> CoreResult<(TextIndex<B, MemoryIndexStore>, IndexMutationReport)> {
    let index_documents = documents
        .iter()
        .map(IndexDocument::from)
        .collect::<Vec<_>>();
    let mut index =
        TextIndex::with_store(embedder, MemoryIndexStore::new()).with_options((*options).into());
    let report = index
        .upsert_documents(&index_documents)
        .map_err(text_index_error)?;
    Ok((index, report))
}

/// Chunks one search document with explicit strategy options.
pub fn chunk_search_document(
    document: &SearchDocument,
    options: &ChunkingOptions,
    processing: &TextProcessingOptions,
) -> CoreResult<Vec<DocumentChunk>> {
    validate_document(document)?;
    match options.strategy {
        ChunkingStrategy::TokenWindow => chunk_document(document, &options.ingestion, processing),
        ChunkingStrategy::Sentence => chunk_spans(
            document,
            split_sentence_spans(&document.body, processing)
                .into_iter()
                .map(|sentence| sentence.span),
        ),
        ChunkingStrategy::Paragraph => chunk_spans(
            document,
            split_paragraphs(&document.body)
                .into_iter()
                .map(|paragraph| paragraph.span),
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
/// Data type for search filter.
pub struct SearchFilter {
    /// The metadata equals value.
    pub metadata_equals: BTreeMap<String, String>,
    /// The metadata contains value.
    pub metadata_contains: BTreeMap<String, String>,
    /// The required tags value.
    pub required_tags: Vec<String>,
    /// The document identifiers value.
    pub document_ids: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Variants describing retrieval mode.
pub enum RetrievalMode {
    /// The full text variant.
    FullText,
    /// The semantic variant.
    Semantic,
    /// The hybrid variant.
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// Data type for hybrid config.
pub struct HybridConfig {
    /// The semantic weight value.
    pub semantic_weight: f32,
    /// The lexical weight value.
    pub lexical_weight: f32,
    /// The rerank window value.
    pub rerank_window: usize,
    /// Whether search results should be reranked after initial retrieval.
    pub rerank: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.8,
            lexical_weight: 0.2,
            rerank_window: 64,
            rerank: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for search query.
pub struct SearchQuery {
    /// Text content for this value.
    pub text: String,
    /// The top k value.
    pub top_k: usize,
    /// The filter value.
    pub filter: Option<SearchFilter>,
    /// The hybrid value.
    pub hybrid: HybridConfig,
}

impl SearchQuery {
    /// Creates a new value.
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            top_k,
            filter: None,
            hybrid: HybridConfig::default(),
        }
    }

    /// Returns full text.
    pub fn full_text(text: impl Into<String>, top_k: usize) -> Self {
        Self::new(text, top_k).mode(RetrievalMode::FullText)
    }

    /// Returns semantic.
    pub fn semantic(text: impl Into<String>, top_k: usize) -> Self {
        Self::new(text, top_k).mode(RetrievalMode::Semantic)
    }

    /// Returns hybrid.
    pub fn hybrid(text: impl Into<String>, top_k: usize, config: HybridConfig) -> Self {
        Self {
            text: text.into(),
            top_k,
            filter: None,
            hybrid: config,
        }
    }

    /// Returns filter.
    pub fn filter(mut self, filter: SearchFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Returns mode.
    pub fn mode(mut self, mode: RetrievalMode) -> Self {
        match mode {
            RetrievalMode::FullText => {
                self.hybrid.semantic_weight = 0.0;
                self.hybrid.lexical_weight = 1.0;
            }
            RetrievalMode::Semantic => {
                self.hybrid.semantic_weight = 1.0;
                self.hybrid.lexical_weight = 0.0;
            }
            RetrievalMode::Hybrid => {
                if self.hybrid.semantic_weight <= f32::EPSILON {
                    self.hybrid.semantic_weight = HybridConfig::default().semantic_weight;
                }
                if self.hybrid.lexical_weight <= f32::EPSILON {
                    self.hybrid.lexical_weight = HybridConfig::default().lexical_weight;
                }
            }
        }
        self
    }

    /// Returns retrieval mode.
    pub fn retrieval_mode(&self) -> RetrievalMode {
        let semantic = self.hybrid.semantic_weight > f32::EPSILON;
        let lexical = self.hybrid.lexical_weight > f32::EPSILON;
        match (semantic, lexical) {
            (false, true) => RetrievalMode::FullText,
            (true, false) => RetrievalMode::Semantic,
            _ => RetrievalMode::Hybrid,
        }
    }
}

/// Request for query/document reranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RerankRequest {
    /// Query text.
    pub query: String,
    /// Documents to rerank.
    pub documents: Vec<String>,
    /// Maximum output count.
    #[serde(default = "default_rerank_top_k")]
    pub top_k: usize,
    /// Optional caller-supplied document scores.
    #[serde(default)]
    pub imported_scores: Vec<f32>,
}

/// One reranked document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RerankResult {
    /// Original document index.
    pub index: usize,
    /// Document text.
    pub document: String,
    /// Relevance score.
    pub score: f32,
    /// Optional runtime used to produce this score.
    #[serde(default)]
    pub runtime: Option<String>,
    /// Optional model identifier used to produce this score.
    #[serde(default)]
    pub model_id: Option<String>,
}

/// Response for query/document reranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RerankResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Query text.
    pub query: String,
    /// Ranked documents.
    pub results: Vec<RerankResult>,
    /// Optional runtime used by a rerank backend.
    #[serde(default)]
    pub runtime: Option<String>,
    /// Optional model identifier used by a rerank backend.
    #[serde(default)]
    pub model_id: Option<String>,
}

/// Caller-supplied runtime context for reranking.
#[derive(Default)]
pub struct RerankExecutionContext<'a> {
    /// Optional reranker backend.
    pub reranker: Option<&'a mut dyn TextReranker>,
    /// Optional model identifier for response metadata.
    pub model_id: Option<String>,
}

/// Reranks documents from imported scores or deterministic lexical overlap.
pub fn rerank_documents(request: RerankRequest) -> CoreResult<RerankResponse> {
    rerank_documents_with_context(request, &mut RerankExecutionContext::default())
}

/// Reranks documents using a runtime backend when supplied, otherwise falls back.
pub fn rerank_documents_with_context(
    request: RerankRequest,
    context: &mut RerankExecutionContext<'_>,
) -> CoreResult<RerankResponse> {
    if request.query.trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "rerank request must include a non-empty query".to_string(),
        ));
    }
    if request.documents.is_empty() {
        return Err(DetectError::InvalidArgument(
            "rerank request must include at least one document".to_string(),
        ));
    }
    let backend_scores = if let Some(reranker) = context.reranker.as_deref_mut() {
        let runtime = rerank_runtime_name(reranker.runtime_backend());
        Some((
            runtime,
            reranker.rerank(&request.query, &request.documents)?,
        ))
    } else {
        None
    };
    let mut results = request
        .documents
        .iter()
        .enumerate()
        .map(|(index, document)| {
            let (score, runtime) = if let Some((runtime, scores)) = &backend_scores {
                (
                    scores.get(index).copied().unwrap_or(0.0),
                    Some(runtime.clone()),
                )
            } else {
                (
                    request
                        .imported_scores
                        .get(index)
                        .copied()
                        .unwrap_or_else(|| lexical_overlap_score(&request.query, document)),
                    None,
                )
            };
            RerankResult {
                index,
                document: document.clone(),
                score,
                runtime,
                model_id: context.model_id.clone(),
            }
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.index.cmp(&right.index))
    });
    results.truncate(request.top_k.max(1));
    Ok(RerankResponse {
        accepted: true,
        operation: "rerank".to_string(),
        query: request.query,
        results,
        runtime: backend_scores.map(|(runtime, _)| runtime),
        model_id: context.model_id.clone(),
    })
}

fn rerank_runtime_name(runtime: TextRuntimeBackend) -> String {
    match runtime {
        TextRuntimeBackend::Candle => "candle",
        TextRuntimeBackend::Onnx => "onnx",
        TextRuntimeBackend::Tokenizers => "tokenizers",
        TextRuntimeBackend::CudaOxide => "cuda_oxide",
        TextRuntimeBackend::External => "external",
        TextRuntimeBackend::Heuristic => "heuristic",
    }
    .to_string()
}

fn lexical_overlap_score(query: &str, document: &str) -> f32 {
    let processing = TextProcessingOptions::default();
    let query_terms = tokenize(query, &processing)
        .into_iter()
        .filter_map(|token| match token.kind {
            TokenKind::Word => Some(token.normalized),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if query_terms.is_empty() {
        return 0.0;
    }
    let document_terms = tokenize(document, &processing)
        .into_iter()
        .filter_map(|token| match token.kind {
            TokenKind::Word => Some(token.normalized),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let overlap = query_terms
        .iter()
        .filter(|term| document_terms.contains(term.as_str()))
        .count();
    overlap as f32 / query_terms.len() as f32
}

fn rerank_search_results(
    query: &str,
    results: &mut Vec<SearchResult>,
    top_k: usize,
    context: &mut RerankExecutionContext<'_>,
) -> CoreResult<()> {
    let candidate_count = results.len();
    if candidate_count == 0 {
        return Ok(());
    }
    let request = RerankRequest {
        query: query.to_string(),
        documents: results
            .iter()
            .map(|result| result.snippet.clone())
            .collect(),
        top_k: top_k.max(1),
        imported_scores: results.iter().map(|result| result.score).collect(),
    };
    let response = rerank_documents_with_context(request, context)?;
    let mut ranked = Vec::new();
    for reranked in response.results {
        if let Some(mut result) = results.get(reranked.index).cloned() {
            result.score = reranked.score;
            if let Some(runtime) = reranked.runtime {
                result
                    .metadata
                    .insert("rerank_runtime".to_string(), runtime);
            }
            if let Some(model_id) = reranked.model_id {
                result
                    .metadata
                    .insert("rerank_model_id".to_string(), model_id);
            }
            ranked.push(result);
        }
    }
    *results = ranked;
    if results.len() > candidate_count {
        results.truncate(candidate_count);
    }
    Ok(())
}

fn default_rerank_top_k() -> usize {
    3
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for search result.
pub struct SearchResult {
    /// The chunk identifier value.
    pub chunk_id: String,
    /// The document identifier value.
    pub document_id: String,
    /// Score assigned to this value.
    pub score: f32,
    /// The semantic score value.
    pub semantic_score: f32,
    /// The lexical score value.
    pub lexical_score: f32,
    /// The snippet value.
    pub snippet: String,
    /// Metadata associated with this value.
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for ingest report.
pub struct IngestReport {
    /// The documents received value.
    pub documents_received: usize,
    /// The documents replaced value.
    pub documents_replaced: usize,
    /// The documents skipped value.
    pub documents_skipped: usize,
    /// The chunks indexed value.
    pub chunks_indexed: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for retrieval index.
pub struct RetrievalIndex<B> {
    embedder: B,
    corpus_options: CorpusOptions,
    bm25_options: Bm25Options,
    chunks: BTreeMap<String, DocumentChunk>,
    raw_text_by_chunk_id: BTreeMap<String, String>,
    document_chunks: BTreeMap<String, Vec<String>>,
    vectors: VectorSearchIndex,
    bm25: Bm25Corpus,
}

impl<B: TextEmbedderBackend> RetrievalIndex<B> {
    /// Creates a new value.
    pub fn new(embedder: B) -> Self {
        Self::with_options(embedder, CorpusOptions::default(), Bm25Options::default())
    }

    /// Returns this value with options.
    pub fn with_options(
        embedder: B,
        corpus_options: CorpusOptions,
        bm25_options: Bm25Options,
    ) -> Self {
        Self {
            embedder,
            bm25: Bm25Corpus::new(bm25_options.clone()),
            corpus_options,
            bm25_options,
            chunks: BTreeMap::new(),
            raw_text_by_chunk_id: BTreeMap::new(),
            document_chunks: BTreeMap::new(),
            vectors: VectorSearchIndex::new(),
        }
    }

    /// Builds this value from parts.
    pub fn from_parts(
        embedder: B,
        corpus_options: CorpusOptions,
        bm25_options: Bm25Options,
        chunks: Vec<DocumentChunk>,
        raw_text_by_chunk_id: BTreeMap<String, String>,
        vector_records: Vec<SerializableVectorRecord>,
    ) -> CoreResult<Self> {
        let mut index = Self::with_options(embedder, corpus_options, bm25_options);
        for chunk in chunks {
            index.insert_chunk_state(chunk);
        }
        index.raw_text_by_chunk_id = raw_text_by_chunk_id;
        index.vectors = VectorSearchIndex::import_records(vector_records)?;
        index.rebuild_bm25()?;
        index.validate_vector_state()?;
        Ok(index)
    }

    /// Returns embedder.
    pub fn embedder(&self) -> &B {
        &self.embedder
    }

    /// Returns embedder info.
    pub fn embedder_info(&self) -> EmbeddingModelInfo {
        self.embedder.model_info()
    }

    /// Returns corpus options.
    pub fn corpus_options(&self) -> &CorpusOptions {
        &self.corpus_options
    }

    /// Returns bm25 options.
    pub fn bm25_options(&self) -> &Bm25Options {
        &self.bm25_options
    }

    /// Returns chunks.
    pub fn chunks(&self) -> Vec<&DocumentChunk> {
        self.chunks.values().collect()
    }

    /// Returns chunks iter.
    pub fn chunks_iter(&self) -> impl Iterator<Item = &DocumentChunk> {
        self.chunks.values()
    }

    /// Returns chunk.
    pub fn chunk(&self, chunk_id: &str) -> Option<&DocumentChunk> {
        self.chunks.get(chunk_id)
    }

    /// Returns raw text.
    pub fn raw_text(&self, chunk_id: &str) -> Option<&str> {
        self.raw_text_by_chunk_id.get(chunk_id).map(String::as_str)
    }

    /// Returns vector records.
    pub fn vector_records(&self) -> Vec<SerializableVectorRecord> {
        self.vectors.export_records()
    }

    /// Returns ingest documents.
    pub fn ingest_documents(
        &mut self,
        docs: &[SearchDocument],
        options: &IngestionOptions,
    ) -> CoreResult<IngestReport> {
        validate_ingestion_options(options)?;

        let mut replaced = 0;
        let mut skipped = 0;
        for document in docs {
            validate_document(document)?;
            if self.document_chunks.contains_key(&document.id) {
                self.remove_document(&document.id);
                replaced += 1;
            }

            let chunks = chunk_document(document, options, &self.corpus_options.processing)?;
            if chunks.is_empty() {
                skipped += 1;
                continue;
            }

            for chunk in chunks {
                if options.store_raw_text {
                    self.raw_text_by_chunk_id
                        .insert(chunk.chunk_id.clone(), chunk.text.clone());
                }
                self.insert_chunk_state(chunk);
            }
        }

        self.rebuild_indices()?;
        Ok(IngestReport {
            documents_received: docs.len(),
            documents_replaced: replaced,
            documents_skipped: skipped,
            chunks_indexed: self.chunks.len(),
        })
    }

    /// Returns search.
    pub fn search(&self, query: &SearchQuery) -> CoreResult<Vec<SearchResult>> {
        self.search_with_context(query, &mut RerankExecutionContext::default())
    }

    /// Returns search, optionally reranking the initial candidates with context.
    pub fn search_with_context(
        &self,
        query: &SearchQuery,
        context: &mut RerankExecutionContext<'_>,
    ) -> CoreResult<Vec<SearchResult>> {
        validate_query(query)?;
        self.ensure_non_empty()?;

        let normalized_query = normalize_query(&query.text, &self.corpus_options.processing)?;
        let filter = query.filter.as_ref();
        let overfetch = query
            .hybrid
            .rerank_window
            .max(query.top_k)
            .max(query.top_k * 4);
        let post_filter_limit = if filter.is_some_and(requires_post_filter) {
            self.chunks.len()
        } else {
            overfetch
        };

        let semantic_hits = if query.hybrid.semantic_weight > f32::EPSILON {
            let query_vector = self.embedder.embed_text(&normalized_query)?;
            self.validate_query_dimensions(query_vector.dimensions())?;
            self.vectors.search_filtered(
                query_vector.as_slice(),
                post_filter_limit,
                filter.map(search_filter_to_vector_filter).as_ref(),
            )?
        } else {
            Vec::new()
        };
        let lexical_hits = if query.hybrid.lexical_weight > f32::EPSILON {
            self.bm25.search(&normalized_query, post_filter_limit)?
        } else {
            Vec::new()
        };

        let semantic_scores = semantic_hits
            .iter()
            .filter(|hit| self.matches_filter(hit.id.as_str(), filter))
            .map(|hit| (hit.id.as_str().to_string(), hit.score))
            .collect::<BTreeMap<_, _>>();
        let lexical_scores = lexical_hits
            .into_iter()
            .filter(|hit| self.matches_filter(&hit.id, filter))
            .map(|hit| (hit.id, hit.score))
            .collect::<BTreeMap<_, _>>();

        let normalized_semantic = normalize_scores(&semantic_scores);
        let normalized_lexical = normalize_scores(&lexical_scores);

        let mut merged_ids = normalized_semantic.keys().cloned().collect::<BTreeSet<_>>();
        merged_ids.extend(normalized_lexical.keys().cloned());

        let mut results = merged_ids
            .into_iter()
            .filter_map(|chunk_id| {
                let chunk = self.chunks.get(&chunk_id)?;
                let semantic_score = normalized_semantic.get(&chunk_id).copied().unwrap_or(0.0);
                let lexical_score = normalized_lexical.get(&chunk_id).copied().unwrap_or(0.0);
                let score = semantic_score * query.hybrid.semantic_weight
                    + lexical_score * query.hybrid.lexical_weight;
                Some(SearchResult {
                    chunk_id: chunk.chunk_id.clone(),
                    document_id: chunk.document_id.clone(),
                    score,
                    semantic_score,
                    lexical_score,
                    snippet: self.build_snippet(chunk, &normalized_query),
                    metadata: chunk.metadata.clone(),
                })
            })
            .collect::<Vec<_>>();

        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.semantic_score.total_cmp(&left.semantic_score))
                .then_with(|| right.lexical_score.total_cmp(&left.lexical_score))
                .then_with(|| left.chunk_id.cmp(&right.chunk_id))
        });
        if query.hybrid.rerank {
            rerank_search_results(&query.text, &mut results, query.top_k, context)?;
        } else {
            results.truncate(query.top_k);
        }
        Ok(results)
    }

    /// Returns related chunks.
    pub fn related_chunks(&self, chunk_id: &str, top_k: usize) -> CoreResult<Vec<SearchResult>> {
        if top_k == 0 {
            return Err(invalid_argument("search limit must be greater than zero"));
        }
        self.ensure_non_empty()?;

        let query_record = self
            .vectors
            .records()
            .iter()
            .find(|record| record.id == chunk_id)
            .ok_or_else(|| invalid_argument(format!("chunk `{chunk_id}` was not indexed")))?;
        let hits = self
            .vectors
            .search_filtered(query_record.vector.as_slice(), top_k + 1, None)?;

        let mut results = hits
            .into_iter()
            .filter(|hit| hit.id.as_str() != chunk_id)
            .filter_map(|hit| {
                let chunk = self.chunks.get(hit.id.as_str())?;
                Some(SearchResult {
                    chunk_id: chunk.chunk_id.clone(),
                    document_id: chunk.document_id.clone(),
                    score: hit.score,
                    semantic_score: hit.score,
                    lexical_score: 0.0,
                    snippet: self.build_snippet(chunk, &chunk.text),
                    metadata: chunk.metadata.clone(),
                })
            })
            .collect::<Vec<_>>();
        results.truncate(top_k);
        Ok(results)
    }

    fn insert_chunk_state(&mut self, chunk: DocumentChunk) {
        self.document_chunks
            .entry(chunk.document_id.clone())
            .or_default()
            .push(chunk.chunk_id.clone());
        self.chunks.insert(chunk.chunk_id.clone(), chunk);
    }

    fn remove_document(&mut self, document_id: &str) {
        if let Some(chunk_ids) = self.document_chunks.remove(document_id) {
            for chunk_id in chunk_ids {
                self.chunks.remove(&chunk_id);
                self.raw_text_by_chunk_id.remove(&chunk_id);
            }
        }
    }

    fn rebuild_indices(&mut self) -> CoreResult<()> {
        self.rebuild_bm25()?;
        if self.chunks.is_empty() {
            self.vectors = VectorSearchIndex::new();
            return Ok(());
        }

        let texts = self
            .chunks
            .values()
            .map(|chunk| {
                self.raw_text_by_chunk_id
                    .get(&chunk.chunk_id)
                    .map(String::as_str)
                    .unwrap_or(chunk.text.as_str())
            })
            .collect::<Vec<_>>();
        let vectors = self.embedder.embed_batch(&texts)?;
        if vectors.len() != self.chunks.len() {
            return Err(invalid_argument(
                "embedder returned a different number of vectors than input texts",
            ));
        }
        self.validate_embedded_dimensions(vectors.iter().map(|vector| vector.dimensions()))?;

        let mut index = VectorSearchIndex::new();
        for (chunk, vector) in self.chunks.values().zip(vectors) {
            index.add(VectorRecord::with_payload(
                chunk.chunk_id.clone(),
                vector,
                VectorRecordMetadata {
                    tags: metadata_tags(&chunk.metadata),
                    metadata: chunk.metadata.clone(),
                },
            ))?;
        }
        self.vectors = index;
        Ok(())
    }

    fn rebuild_bm25(&mut self) -> CoreResult<()> {
        let mut bm25 = Bm25Corpus::new(self.bm25_options.clone());
        for chunk in self.chunks.values() {
            bm25.add_document(chunk.chunk_id.clone(), &chunk.text)?;
        }
        self.bm25 = bm25;
        Ok(())
    }

    fn validate_embedded_dimensions(
        &self,
        vector_dimensions: impl IntoIterator<Item = usize>,
    ) -> CoreResult<()> {
        let expected = self.embedder.model_info().dimensions;
        if expected == 0 {
            return Ok(());
        }
        if vector_dimensions
            .into_iter()
            .any(|dimensions| dimensions != expected)
        {
            return Err(invalid_argument(format!(
                "embedder produced vectors that did not match advertised dimensions {expected}"
            )));
        }
        Ok(())
    }

    fn validate_vector_state(&self) -> CoreResult<()> {
        let chunk_ids = self.chunks.keys().cloned().collect::<BTreeSet<_>>();
        let vector_ids = self
            .vectors
            .records()
            .iter()
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>();
        if chunk_ids != vector_ids {
            return Err(invalid_argument(
                "persisted vector records and chunk catalog do not describe the same ids",
            ));
        }
        if let Some(dimensions) = self.vectors.dimensions() {
            let expected = self.embedder.model_info().dimensions;
            if expected > 0 && expected != dimensions {
                return Err(invalid_argument(format!(
                    "embedder dimensions {expected} did not match persisted index dimensions {dimensions}"
                )));
            }
        }
        Ok(())
    }

    fn validate_query_dimensions(&self, dimensions: usize) -> CoreResult<()> {
        if let Some(index_dimensions) = self.vectors.dimensions() {
            if dimensions != index_dimensions {
                return Err(invalid_argument(format!(
                    "query vector dimensions {dimensions} did not match index dimensions {index_dimensions}"
                )));
            }
        }
        Ok(())
    }

    fn matches_filter(&self, chunk_id: &str, filter: Option<&SearchFilter>) -> bool {
        let Some(filter) = filter else {
            return true;
        };
        let Some(chunk) = self.chunks.get(chunk_id) else {
            return false;
        };
        if !filter.document_ids.is_empty() && !filter.document_ids.contains(&chunk.document_id) {
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
        filter
            .required_tags
            .iter()
            .all(|tag| tags.iter().any(|candidate| candidate == tag))
            && filter
                .metadata_equals
                .iter()
                .all(|(key, value)| chunk.metadata.get(key) == Some(value))
    }

    fn build_snippet(&self, chunk: &DocumentChunk, normalized_query: &str) -> String {
        let text = self
            .raw_text_by_chunk_id
            .get(&chunk.chunk_id)
            .map(String::as_str)
            .unwrap_or(chunk.text.as_str());
        build_snippet(text, normalized_query)
    }

    fn ensure_non_empty(&self) -> CoreResult<()> {
        if self.chunks.is_empty() {
            return Err(invalid_argument("search index is empty"));
        }
        Ok(())
    }
}

fn validate_document(document: &SearchDocument) -> CoreResult<()> {
    if document.id.trim().is_empty() {
        return Err(invalid_argument("document id must not be empty"));
    }
    if document.body.trim().is_empty() {
        return Err(invalid_argument("document body must not be empty"));
    }
    Ok(())
}

fn validate_ingestion_options(options: &IngestionOptions) -> CoreResult<()> {
    if options.chunk_tokens == 0 {
        return Err(invalid_argument(
            "chunk token size must be greater than zero",
        ));
    }
    if options.chunk_overlap_tokens >= options.chunk_tokens {
        return Err(invalid_argument(
            "chunk overlap must be smaller than the chunk token size",
        ));
    }
    Ok(())
}

fn validate_query(query: &SearchQuery) -> CoreResult<()> {
    if query.top_k == 0 {
        return Err(invalid_argument("search limit must be greater than zero"));
    }
    if !query.hybrid.semantic_weight.is_finite()
        || !query.hybrid.lexical_weight.is_finite()
        || query.hybrid.semantic_weight < 0.0
        || query.hybrid.lexical_weight < 0.0
    {
        return Err(invalid_argument(
            "hybrid weights must be finite and non-negative",
        ));
    }
    if query.hybrid.semantic_weight + query.hybrid.lexical_weight <= f32::EPSILON {
        return Err(invalid_argument("hybrid weights must not both be zero"));
    }
    if query.hybrid.rerank_window == 0 {
        return Err(invalid_argument("rerank window must be greater than zero"));
    }
    let _ = normalize_query(&query.text, &TextProcessingOptions::default())?;
    Ok(())
}

fn chunk_document(
    document: &SearchDocument,
    options: &IngestionOptions,
    processing: &TextProcessingOptions,
) -> CoreResult<Vec<DocumentChunk>> {
    let tokens = tokenize(document.body.as_str(), processing)
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
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let mut chunks = Vec::new();
    let step = options.chunk_tokens - options.chunk_overlap_tokens;
    let metadata = document_metadata(document);
    let mut start = 0;
    let mut ordinal = 0;
    while start < tokens.len() {
        let end = (start + options.chunk_tokens).min(tokens.len());
        let byte_start = tokens[start].span.byte_start;
        let byte_end = tokens[end - 1].span.byte_end;
        let text = document
            .body
            .get(byte_start..byte_end)
            .ok_or_else(|| invalid_argument("token spans did not align to valid UTF-8 boundaries"))?
            .trim()
            .to_string();
        if !text.is_empty() {
            chunks.push(DocumentChunk {
                chunk_id: format!("{}:{ordinal}", document.id),
                document_id: document.id.clone(),
                text,
                ordinal,
                metadata: metadata.clone(),
                source: document.source.clone(),
                provenance: document.provenance.clone(),
                annotations: document.annotations.clone(),
            });
            ordinal += 1;
        }
        if end == tokens.len() {
            break;
        }
        start += step;
    }
    Ok(chunks)
}

fn chunk_spans(
    document: &SearchDocument,
    spans: impl IntoIterator<Item = TextSpan>,
) -> CoreResult<Vec<DocumentChunk>> {
    let metadata = document_metadata(document);
    let mut chunks = Vec::new();
    for (ordinal, span) in spans.into_iter().enumerate() {
        let text = document
            .body
            .get(span.byte_start..span.byte_end)
            .ok_or_else(|| invalid_argument("chunk span did not align to valid UTF-8 boundaries"))?
            .trim()
            .to_string();
        if text.is_empty() {
            continue;
        }
        chunks.push(DocumentChunk {
            chunk_id: format!("{}:{ordinal}", document.id),
            document_id: document.id.clone(),
            text,
            ordinal,
            metadata: metadata.clone(),
            source: document.source.clone(),
            provenance: document.provenance.clone(),
            annotations: document.annotations.clone(),
        });
    }
    Ok(chunks)
}

fn document_metadata(document: &SearchDocument) -> BTreeMap<String, String> {
    let mut metadata = document.metadata.clone();
    if let Some(title) = &document.title {
        metadata
            .entry(TITLE_METADATA_KEY.to_string())
            .or_insert_with(|| title.clone());
    }
    metadata
}

fn normalize_query(query: &str, processing: &TextProcessingOptions) -> CoreResult<String> {
    let tokens = tokenize(query, processing)
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
        .map(|token| token.normalized)
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return Err(invalid_argument(
            "query must contain at least one searchable term",
        ));
    }
    Ok(tokens.join(" "))
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
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn search_filter_to_vector_filter(filter: &SearchFilter) -> VectorSearchFilter {
    VectorSearchFilter {
        required_tags: filter.required_tags.clone(),
        metadata_equals: filter.metadata_equals.clone(),
    }
}

fn requires_post_filter(filter: &SearchFilter) -> bool {
    !filter.document_ids.is_empty() || !filter.metadata_contains.is_empty()
}

fn normalize_scores(scores: &BTreeMap<String, f32>) -> BTreeMap<String, f32> {
    if scores.is_empty() {
        return BTreeMap::new();
    }
    let min = scores.values().copied().fold(f32::INFINITY, f32::min);
    let max = scores.values().copied().fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() <= f32::EPSILON {
        return scores
            .keys()
            .cloned()
            .map(|id| (id, 1.0))
            .collect::<BTreeMap<_, _>>();
    }
    scores
        .iter()
        .map(|(id, score)| (id.clone(), (score - min) / (max - min)))
        .collect()
}

fn build_snippet(text: &str, normalized_query: &str) -> String {
    let terms = tokenize(normalized_query, &TextProcessingOptions::default())
        .into_iter()
        .map(|token| token.normalized)
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

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

fn text_index_error(error: TextIndexError) -> DetectError {
    match error {
        TextIndexError::InvalidArgument(message) => DetectError::InvalidArgument(message),
        other => DetectError::Source(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use text_embeddings::{
        DenseVector, HashedTextEmbedder, TextEmbeddingBackend, TextEmbeddingBackendKind,
        TextEmbeddingConfig, TextEmbeddingMetadata,
    };

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

        fn runtime_backend(&self) -> TextRuntimeBackend {
            TextRuntimeBackend::External
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

        assert!(
            matches!(err, DetectError::InvalidArgument(message) if message.contains("dimensions"))
        );
    }
}
