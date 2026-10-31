//! Query-time request and result contracts, validation, and normalization.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use text_core::{tokenize, Result as CoreResult, TextProcessingOptions, TokenKind};
use vector_analysis_index::VectorSearchFilter;

use crate::invalid_argument;

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

pub(crate) fn validate_query(query: &SearchQuery) -> CoreResult<()> {
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

pub(crate) fn normalize_query(
    query: &str,
    processing: &TextProcessingOptions,
) -> CoreResult<String> {
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

pub(crate) fn search_filter_to_vector_filter(filter: &SearchFilter) -> VectorSearchFilter {
    VectorSearchFilter {
        required_tags: filter.required_tags.clone(),
        metadata_equals: filter.metadata_equals.clone(),
    }
}

pub(crate) fn requires_post_filter(filter: &SearchFilter) -> bool {
    !filter.document_ids.is_empty() || !filter.metadata_contains.is_empty()
}
