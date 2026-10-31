#![doc = include_str!("../README.md")]

mod document;
mod index;
mod ingestion;
mod query;
mod rerank;
mod storage;
pub mod surface;
#[cfg(test)]
mod tests;

use text_core::DetectError;

pub use document::{DocumentChunk, IntoSearchDocument, SearchDocument};
pub use index::{IngestReport, RetrievalIndex};
pub use ingestion::{
    build_memory_text_index, chunk_search_document, ChunkingOptions, ChunkingStrategy,
    IngestionOptions,
};
pub use query::{HybridConfig, RetrievalMode, SearchFilter, SearchQuery, SearchResult};
pub use rerank::{
    rerank_documents, rerank_documents_with_context, RerankExecutionContext, RerankRequest,
    RerankResponse, RerankResult, RerankerRuntime, TextReranker,
};
pub use storage::*;

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}
