//! Query/document reranking contracts and the deterministic fallback scorer.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use text_core::{tokenize, DetectError, Result as CoreResult, TextProcessingOptions, TokenKind};

use crate::SearchResult;

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

/// Runtime family of a reranking backend, reported in rerank responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerRuntime {
    /// Candle-backed native model.
    Candle,
    /// ONNX Runtime-backed native model.
    Onnx,
    /// Caller-supplied external backend.
    External,
    /// Deterministic heuristic backend.
    Heuristic,
}

impl RerankerRuntime {
    /// Stable runtime name used in response metadata.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Onnx => "onnx",
            Self::External => "external",
            Self::Heuristic => "heuristic",
        }
    }
}

/// Query-document reranking backend.
pub trait TextReranker {
    /// Scores each document for the query, in document order.
    fn rerank(&mut self, query: &str, documents: &[String]) -> CoreResult<Vec<f32>>;

    /// Runtime family that produced the scores.
    fn runtime(&self) -> RerankerRuntime;
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
        let runtime = reranker.runtime().as_str().to_string();
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

pub(crate) fn rerank_search_results(
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
