//! Ingestion options, chunking, and in-memory `text-index` construction.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use text_core::{
    split_paragraphs, split_sentence_spans, tokenize, DetectError, Result as CoreResult,
    TextProcessingOptions, TextSpan, TokenKind,
};
use text_embeddings::TextEmbedderBackend;
use text_index::{
    ChunkingStrategy as IndexChunkingStrategy, IndexBuildOptions, IndexDocument,
    IndexMutationReport, MemoryIndexStore, TextIndex, TextIndexError,
};

use crate::{invalid_argument, DocumentChunk, SearchDocument};

const TITLE_METADATA_KEY: &str = "__title";

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

pub(crate) fn validate_document(document: &SearchDocument) -> CoreResult<()> {
    if document.id.trim().is_empty() {
        return Err(invalid_argument("document id must not be empty"));
    }
    if document.body.trim().is_empty() {
        return Err(invalid_argument("document body must not be empty"));
    }
    Ok(())
}

pub(crate) fn validate_ingestion_options(options: &IngestionOptions) -> CoreResult<()> {
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

pub(crate) fn chunk_document(
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

fn text_index_error(error: TextIndexError) -> DetectError {
    match error {
        TextIndexError::InvalidArgument(message) => DetectError::InvalidArgument(message),
        other => DetectError::Source(other.to_string()),
    }
}
