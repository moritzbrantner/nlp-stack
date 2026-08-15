#![doc = include_str!("../README.md")]

mod retrieved_context;
pub mod surface;

use std::collections::BTreeMap;
use std::path::PathBuf;

use media_core::{DetectError, Result};
use retrieved_context::RetrievedContext;
use serde::{Deserialize, Serialize};
use text_embeddings::{HashedTextEmbedder, TextEmbedderBackend, TextEmbeddingConfig};
use text_index::{
    IndexBuildOptions, IndexDocument, IndexQuery, IndexSearchResult, MemoryIndexStore, TextIndex,
};
#[cfg(feature = "local-onnx")]
use text_model_runtime::{OnnxQuestionAnswerer, QuestionAnsweringDecodeOptions};
use text_model_runtime::{QuestionAnsweringBackend, TextRuntimeBackend};
use text_retrieval::{IngestionOptions, RetrievalIndex, SearchDocument, SearchQuery, SearchResult};

/// Runtime families for extractive question answering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionAnsweringRuntime {
    /// Candle-backed native inference.
    Candle,
    /// ONNX Runtime-backed native inference.
    Onnx,
    /// Browser-safe WASM postprocessing.
    WasmPostprocess,
    /// Deterministic or heuristic fallback.
    Heuristic,
    /// Caller-supplied model predictions.
    ImportedPredictions,
}

/// Model selection supplied by API, CLI, or UI callers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelSelection {
    /// Optional preset or Hugging Face model identifier.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Optional preferred runtime.
    #[serde(default)]
    pub runtime: Option<QuestionAnsweringRuntime>,
}

/// Local model options for default ONNX question answering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct QuestionAnsweringLocalModelOptions {
    /// Optional preset id; defaults to `roberta-base-squad2-onnx`.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Bundle root; defaults to `.model-runtime`.
    #[serde(default)]
    pub bundle_root: Option<PathBuf>,
    /// Whether missing bundles may be downloaded.
    #[serde(default)]
    pub auto_download: Option<bool>,
    /// Whether downloads should report progress.
    #[serde(default)]
    pub download_progress: Option<bool>,
    /// Optional Hugging Face cache directory.
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    /// Optional Hugging Face token.
    #[serde(default)]
    pub hf_token: Option<String>,
    /// Maximum download retries.
    #[serde(default)]
    pub max_retries: Option<usize>,
    /// Whether to overwrite materialized bundle files.
    #[serde(default)]
    pub overwrite: Option<bool>,
}

/// Fallback behavior when local QA cannot run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum QuestionAnsweringFallbackPolicy {
    /// Return a typed error.
    #[default]
    Error,
    /// Use deterministic heuristic answers if the local model is unavailable.
    HeuristicIfUnavailable,
}

/// Caller-supplied native or external runtime backend for QA execution.
#[derive(Default)]
pub struct QuestionAnsweringExecutionContext<'a> {
    /// Optional question-answering backend.
    pub qa: Option<&'a mut dyn QuestionAnsweringBackend>,
}

/// Model metadata for UI and CLI discovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionAnsweringModelMetadata {
    /// Stable local preset id.
    pub id: String,
    /// Upstream model id or strategy id.
    pub model_id: String,
    /// Preferred runtime.
    pub runtime: QuestionAnsweringRuntime,
    /// Whether the runtime is available in the default contributor build.
    pub supported: bool,
    /// Whether this entry has an implemented native load/run path.
    pub loadable: bool,
    /// Optional fallback preset id.
    pub fallback: Option<String>,
    /// Cargo feature required for native loading.
    pub required_feature: Option<String>,
    /// Setup command or note required before loading.
    pub required_setup: Option<String>,
    /// Surface operation that smokes the implemented path.
    pub smoke_operation: Option<String>,
    /// Human-readable note for unsupported or fallback-only paths.
    pub note: Option<String>,
}

/// Text span used by QA responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextSpanRef {
    /// Byte start offset.
    pub byte_start: usize,
    /// Byte end offset.
    pub byte_end: usize,
    /// Character start offset.
    pub char_start: usize,
    /// Character end offset.
    pub char_end: usize,
}

/// Imported span prediction used by server, CLI, and WASM postprocessing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedAnswerPrediction {
    /// Optional raw prediction kind.
    #[serde(default)]
    pub kind: Option<String>,
    /// Label emitted by the model.
    #[serde(default)]
    pub label: Option<String>,
    /// Answer text or span text.
    #[serde(default)]
    pub text: Option<String>,
    /// Confidence score.
    pub score: f32,
    /// Arbitrary model attributes, including offsets.
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

/// Request for extractive question answering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionAnsweringRequest {
    /// Question text.
    pub question: String,
    /// Context text.
    pub context: String,
    /// Maximum answer count.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Model selection.
    #[serde(default)]
    pub model: ModelSelection,
    /// Imported span predictions.
    #[serde(default)]
    pub imported_predictions: Vec<ImportedAnswerPrediction>,
    /// Local model configuration for default native execution.
    #[serde(default)]
    pub local_model: Option<QuestionAnsweringLocalModelOptions>,
    /// Fallback policy when native execution is unavailable.
    #[serde(default)]
    pub fallback_policy: Option<QuestionAnsweringFallbackPolicy>,
}

/// One QA answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnswerPrediction {
    /// Answer text.
    pub answer: String,
    /// Confidence score.
    pub score: f32,
    /// Optional context span.
    pub span: Option<TextSpanRef>,
}

/// Response for extractive question answering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionAnsweringResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Question text.
    pub question: String,
    /// Selected model id.
    pub model_id: String,
    /// Runtime used.
    pub runtime: QuestionAnsweringRuntime,
    /// Candidate answers.
    pub answers: Vec<AnswerPrediction>,
}

/// Citation for an answer produced from retrieved context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnswerCitation {
    /// Source document id.
    pub document_id: String,
    /// Source retrieval chunk id.
    pub chunk_id: String,
    /// Snippet used as answer context.
    pub snippet: String,
    /// Retrieval or rerank score.
    pub score: f32,
    /// Optional span inside the cited snippet.
    #[serde(default)]
    pub span: Option<TextSpanRef>,
}

/// One cited QA answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitedAnswer {
    /// Answer text.
    pub answer: String,
    /// Answer confidence score.
    pub score: f32,
    /// Optional span in the concatenated retrieval context.
    #[serde(default)]
    pub span: Option<TextSpanRef>,
    /// Source citations.
    pub citations: Vec<AnswerCitation>,
}

/// Request for retrieval-backed question answering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalQuestionAnsweringRequest {
    /// Question text.
    pub question: String,
    /// Documents used to build a transient deterministic retrieval index.
    #[serde(default)]
    pub documents: Vec<SearchDocument>,
    /// Retrieved chunk count.
    #[serde(default = "default_top_k_chunks")]
    pub top_k_chunks: usize,
    /// Answer count.
    #[serde(default = "default_top_k")]
    pub top_k_answers: usize,
    /// Imported span predictions mapped back to retrieved chunks.
    #[serde(default)]
    pub imported_predictions: Vec<ImportedAnswerPrediction>,
    /// Model selection for backend execution.
    #[serde(default)]
    pub model: ModelSelection,
    /// Local model configuration for default native execution.
    #[serde(default)]
    pub local_model: Option<QuestionAnsweringLocalModelOptions>,
    /// Fallback policy when native execution is unavailable.
    #[serde(default)]
    pub fallback_policy: Option<QuestionAnsweringFallbackPolicy>,
}

/// Response for retrieval-backed question answering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalQuestionAnsweringResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Question text.
    pub question: String,
    /// Runtime used for answer extraction.
    pub runtime: QuestionAnsweringRuntime,
    /// Retrieved chunks.
    pub retrieved_chunks: Vec<SearchResult>,
    /// Cited answers.
    pub answers: Vec<CitedAnswer>,
}

/// Request for `text-index`-backed question answering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextIndexQuestionAnsweringRequest {
    /// Question text.
    pub question: String,
    /// Documents used to build a transient deterministic text index.
    #[serde(default)]
    pub documents: Vec<IndexDocument>,
    /// Text index build options.
    #[serde(default)]
    pub index_options: IndexBuildOptions,
    /// Hashed embedding dimensions for the transient index.
    #[serde(default = "default_embedding_dimensions")]
    pub embedding_dimensions: usize,
    /// Retrieved chunk count.
    #[serde(default = "default_top_k_chunks")]
    pub top_k_chunks: usize,
    /// Answer count.
    #[serde(default = "default_top_k")]
    pub top_k_answers: usize,
    /// Imported span predictions mapped back to indexed chunks.
    #[serde(default)]
    pub imported_predictions: Vec<ImportedAnswerPrediction>,
    /// Model selection for backend execution.
    #[serde(default)]
    pub model: ModelSelection,
    /// Local model configuration for default native execution.
    #[serde(default)]
    pub local_model: Option<QuestionAnsweringLocalModelOptions>,
    /// Fallback policy when native execution is unavailable.
    #[serde(default)]
    pub fallback_policy: Option<QuestionAnsweringFallbackPolicy>,
}

/// Response for `text-index`-backed question answering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextIndexQuestionAnsweringResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Question text.
    pub question: String,
    /// Runtime used for answer extraction.
    pub runtime: QuestionAnsweringRuntime,
    /// Retrieved text-index chunks.
    pub retrieved_chunks: Vec<IndexSearchResult>,
    /// Cited answers.
    pub answers: Vec<CitedAnswer>,
}

/// Request for running multiple imported-span QA requests in one call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionAnsweringBatchRequest {
    /// Question-answering requests to run.
    pub requests: Vec<QuestionAnsweringRequest>,
}

/// One item result from a batch QA run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionAnsweringBatchItem {
    /// Zero-based request index.
    pub index: usize,
    /// Whether the item succeeded.
    pub accepted: bool,
    /// Successful item response.
    #[serde(default)]
    pub response: Option<QuestionAnsweringResponse>,
    /// Item-level error message.
    #[serde(default)]
    pub error: Option<String>,
}

/// Response for a batch QA run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionAnsweringBatchResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Successful item count.
    pub successes: usize,
    /// Failed item count.
    pub failures: usize,
    /// Per-item results in request order.
    pub results: Vec<QuestionAnsweringBatchItem>,
}

/// Backward-compatible alias for the batch item DTO.
pub type QuestionAnsweringBatchItemResult = QuestionAnsweringBatchItem;

/// Returns the question-answering model catalog.
pub fn model_catalog() -> Vec<QuestionAnsweringModelMetadata> {
    vec![QuestionAnsweringModelMetadata {
        id: "roberta-base-squad2-onnx".to_string(),
        model_id: "onnx-community/roberta-base-squad2-ONNX".to_string(),
        runtime: QuestionAnsweringRuntime::Onnx,
        supported: cfg!(feature = "local-onnx"),
        loadable: cfg!(feature = "local-onnx"),
        fallback: None,
        required_feature: Some("local-onnx".to_string()),
        required_setup: Some("First run downloads the ONNX bundle into .model-runtime".to_string()),
        smoke_operation: Some("qa.answer".to_string()),
        note: Some("Local ONNX extractive QA default when built with local-onnx.".to_string()),
    }]
}

/// Answers questions with an optional QA backend.
pub fn answer_question_with_context(
    request: QuestionAnsweringRequest,
    context: &mut QuestionAnsweringExecutionContext<'_>,
) -> Result<QuestionAnsweringResponse> {
    ensure_non_empty(&request.question, "question")?;
    ensure_non_empty(&request.context, "context")?;
    if !request.imported_predictions.is_empty() {
        return answer_question(request);
    }
    let model_id = request
        .model
        .model_id
        .clone()
        .unwrap_or_else(|| "roberta-base-squad2".to_string());
    if let Some(qa) = context.qa.as_deref_mut() {
        let runtime = runtime_from_backend(qa.runtime_backend());
        let mut answers = qa
            .answer(&request.question, &request.context)?
            .into_iter()
            .filter_map(|prediction| {
                let answer = prediction.text?;
                Some(AnswerPrediction {
                    answer,
                    score: prediction.score.unwrap_or(0.0),
                    span: span_from_attributes(&prediction.attributes),
                })
            })
            .collect::<Vec<_>>();
        answers.sort_by(|left, right| right.score.total_cmp(&left.score));
        answers.truncate(request.top_k.max(1));
        return Ok(QuestionAnsweringResponse {
            accepted: true,
            operation: "question-answer".to_string(),
            question: request.question,
            model_id,
            runtime,
            answers,
        });
    }
    answer_question(request)
}

/// Handles QA from imported predictions or returns a typed unsupported-runtime error.
pub fn answer_question(request: QuestionAnsweringRequest) -> Result<QuestionAnsweringResponse> {
    ensure_non_empty(&request.question, "question")?;
    ensure_non_empty(&request.context, "context")?;
    let model_id = request
        .model
        .model_id
        .clone()
        .unwrap_or_else(|| "roberta-base-squad2".to_string());

    if !request.imported_predictions.is_empty() {
        let mut answers = request
            .imported_predictions
            .into_iter()
            .filter_map(|prediction| {
                let answer = prediction.text?;
                Some(AnswerPrediction {
                    answer,
                    score: prediction.score,
                    span: span_from_attributes(&prediction.attributes),
                })
            })
            .collect::<Vec<_>>();
        answers.sort_by(|left, right| right.score.total_cmp(&left.score));
        answers.truncate(request.top_k.max(1));
        return Ok(QuestionAnsweringResponse {
            accepted: true,
            operation: "question-answer".to_string(),
            question: request.question,
            model_id,
            runtime: QuestionAnsweringRuntime::ImportedPredictions,
            answers,
        });
    }

    if request.fallback_policy.unwrap_or_default()
        == QuestionAnsweringFallbackPolicy::HeuristicIfUnavailable
        && request
            .local_model
            .as_ref()
            .and_then(|options| options.auto_download)
            == Some(false)
    {
        return Ok(heuristic_answer_response(
            request,
            "autoDownload=false".to_string(),
        ));
    }

    match run_local_qa(&request) {
        Ok(response) => Ok(response),
        Err(error)
            if request.fallback_policy.unwrap_or_default()
                == QuestionAnsweringFallbackPolicy::HeuristicIfUnavailable =>
        {
            Ok(heuristic_answer_response(request, error.to_string()))
        }
        Err(error) => Err(error),
    }
}

/// Runs multiple imported-span QA requests and returns item-level errors.
pub fn answer_question_batch(
    request: QuestionAnsweringBatchRequest,
) -> Result<QuestionAnsweringBatchResponse> {
    if request.requests.is_empty() {
        return Err(DetectError::InvalidArgument(
            "batch QA request must include at least one request".to_string(),
        ));
    }

    let mut successes = 0;
    let mut failures = 0;
    let results = request
        .requests
        .into_iter()
        .enumerate()
        .map(|(index, item)| match answer_question(item) {
            Ok(response) => {
                successes += 1;
                QuestionAnsweringBatchItem {
                    index,
                    accepted: true,
                    response: Some(response),
                    error: None,
                }
            }
            Err(error) => {
                failures += 1;
                QuestionAnsweringBatchItem {
                    index,
                    accepted: false,
                    response: None,
                    error: Some(error.to_string()),
                }
            }
        })
        .collect();

    Ok(QuestionAnsweringBatchResponse {
        accepted: true,
        operation: "question-answer-batch".to_string(),
        successes,
        failures,
        results,
    })
}

/// Answers a question by building a deterministic retrieval index from documents.
pub fn answer_question_with_retrieval(
    request: RetrievalQuestionAnsweringRequest,
) -> Result<RetrievalQuestionAnsweringResponse> {
    let mut context = QuestionAnsweringExecutionContext::default();
    answer_question_with_retrieval_context(request, &mut context)
}

/// Answers a retrieval-backed question with an optional QA backend.
pub fn answer_question_with_retrieval_context(
    request: RetrievalQuestionAnsweringRequest,
    context: &mut QuestionAnsweringExecutionContext<'_>,
) -> Result<RetrievalQuestionAnsweringResponse> {
    ensure_non_empty(&request.question, "question")?;
    if request.documents.is_empty() {
        return Err(DetectError::InvalidArgument(
            "retrieval QA request must include documents or use an existing retrieval index"
                .to_string(),
        ));
    }
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 64,
            use_idf: false,
        },
        text_lexical::CorpusOptions::default(),
    )?;
    let mut index = RetrievalIndex::new(embedder);
    index.ingest_documents(&request.documents, &IngestionOptions::default())?;
    answer_question_with_retrieval_index(request, &index, context)
}

/// Answers a retrieval-backed question using an existing retrieval index.
pub fn answer_question_with_retrieval_index<B: TextEmbedderBackend>(
    request: RetrievalQuestionAnsweringRequest,
    index: &RetrievalIndex<B>,
    context: &mut QuestionAnsweringExecutionContext<'_>,
) -> Result<RetrievalQuestionAnsweringResponse> {
    ensure_non_empty(&request.question, "question")?;
    let top_k_chunks = request.top_k_chunks.max(1);
    let retrieved_chunks = index.search(&SearchQuery::hybrid(
        request.question.clone(),
        top_k_chunks,
        text_retrieval::HybridConfig {
            rerank_window: top_k_chunks.max(4),
            ..text_retrieval::HybridConfig::default()
        },
    ))?;
    if retrieved_chunks.is_empty() {
        return Ok(RetrievalQuestionAnsweringResponse {
            accepted: true,
            operation: "retrieval-question-answer".to_string(),
            question: request.question,
            runtime: QuestionAnsweringRuntime::Heuristic,
            retrieved_chunks,
            answers: Vec::new(),
        });
    }
    let retrieved_context = RetrievedContext::from_search_results(&retrieved_chunks, index);
    let context_text = retrieved_context.context_text.clone();

    let (runtime, answers) = if !request.imported_predictions.is_empty() || context.qa.is_some() {
        let qa_response = if context.qa.is_some() && request.imported_predictions.is_empty() {
            answer_question_with_context(
                QuestionAnsweringRequest {
                    question: request.question.clone(),
                    context: context_text.clone(),
                    top_k: request.top_k_answers,
                    model: request.model.clone(),
                    imported_predictions: Vec::new(),
                    local_model: request.local_model.clone(),
                    fallback_policy: request.fallback_policy,
                },
                context,
            )?
        } else {
            answer_question(QuestionAnsweringRequest {
                question: request.question.clone(),
                context: context_text.clone(),
                top_k: request.top_k_answers,
                model: request.model.clone(),
                imported_predictions: request.imported_predictions.clone(),
                local_model: request.local_model.clone(),
                fallback_policy: request.fallback_policy,
            })?
        };
        let runtime = qa_response.runtime;
        let answers = qa_response
            .answers
            .into_iter()
            .map(|answer| retrieved_context.cited_answer_from_prediction(answer))
            .collect::<Vec<_>>();
        (runtime, answers)
    } else {
        let qa_response = answer_question(QuestionAnsweringRequest {
            question: request.question.clone(),
            context: context_text.clone(),
            top_k: request.top_k_answers,
            model: request.model.clone(),
            imported_predictions: Vec::new(),
            local_model: request.local_model.clone(),
            fallback_policy: request.fallback_policy,
        })?;
        let runtime = qa_response.runtime;
        let answers = qa_response
            .answers
            .into_iter()
            .map(|answer| retrieved_context.cited_answer_from_prediction(answer))
            .collect::<Vec<_>>();
        (runtime, answers)
    };

    Ok(RetrievalQuestionAnsweringResponse {
        accepted: true,
        operation: "retrieval-question-answer".to_string(),
        question: request.question,
        runtime,
        retrieved_chunks,
        answers,
    })
}

/// Answers a question by building a deterministic memory `TextIndex` from documents.
pub fn answer_question_with_text_index(
    request: TextIndexQuestionAnsweringRequest,
) -> Result<TextIndexQuestionAnsweringResponse> {
    let mut context = QuestionAnsweringExecutionContext::default();
    answer_question_with_text_index_context(request, &mut context)
}

/// Answers a `text-index`-backed question with an optional QA backend.
pub fn answer_question_with_text_index_context(
    request: TextIndexQuestionAnsweringRequest,
    context: &mut QuestionAnsweringExecutionContext<'_>,
) -> Result<TextIndexQuestionAnsweringResponse> {
    ensure_non_empty(&request.question, "question")?;
    if request.documents.is_empty() {
        return Err(DetectError::InvalidArgument(
            "text-index QA request must include documents".to_string(),
        ));
    }
    let top_k_chunks = request.top_k_chunks.max(1);
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: request.embedding_dimensions.max(1),
            use_idf: false,
        },
        text_lexical::CorpusOptions::default(),
    )?;
    let mut index = TextIndex::with_store(embedder, MemoryIndexStore::new())
        .with_options(request.index_options.clone());
    index
        .upsert_documents(&request.documents)
        .map_err(|error| DetectError::Source(error.to_string()))?;
    let retrieved_chunks = index
        .search(&IndexQuery::new(request.question.clone(), top_k_chunks))
        .map_err(|error| DetectError::Source(error.to_string()))?;
    if retrieved_chunks.is_empty() {
        return Ok(TextIndexQuestionAnsweringResponse {
            accepted: true,
            operation: "text-index-question-answer".to_string(),
            question: request.question,
            runtime: QuestionAnsweringRuntime::Heuristic,
            retrieved_chunks,
            answers: Vec::new(),
        });
    }
    let retrieved_context = RetrievedContext::from_index_results(&retrieved_chunks);
    let context_text = retrieved_context.context_text.clone();
    let qa_response = if context.qa.is_some() && request.imported_predictions.is_empty() {
        answer_question_with_context(
            QuestionAnsweringRequest {
                question: request.question.clone(),
                context: context_text.clone(),
                top_k: request.top_k_answers,
                model: request.model.clone(),
                imported_predictions: Vec::new(),
                local_model: request.local_model.clone(),
                fallback_policy: request.fallback_policy,
            },
            context,
        )?
    } else {
        answer_question(QuestionAnsweringRequest {
            question: request.question.clone(),
            context: context_text,
            top_k: request.top_k_answers,
            model: request.model.clone(),
            imported_predictions: request.imported_predictions.clone(),
            local_model: request.local_model.clone(),
            fallback_policy: request.fallback_policy,
        })?
    };
    let runtime = qa_response.runtime;
    let answers = qa_response
        .answers
        .into_iter()
        .map(|answer| retrieved_context.cited_answer_from_prediction(answer))
        .collect::<Vec<_>>();
    Ok(TextIndexQuestionAnsweringResponse {
        accepted: true,
        operation: "text-index-question-answer".to_string(),
        question: request.question,
        runtime,
        retrieved_chunks,
        answers,
    })
}

/// Returns preset ids registered for question answering.
pub fn registered_question_answering_presets() -> Vec<String> {
    model_catalog().into_iter().map(|model| model.id).collect()
}

fn default_top_k() -> usize {
    3
}

fn default_top_k_chunks() -> usize {
    5
}

fn default_embedding_dimensions() -> usize {
    64
}

fn ensure_non_empty(value: &str, name: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(DetectError::InvalidArgument(format!(
            "request body must include a non-empty `{name}` string"
        )));
    }
    Ok(())
}

#[cfg(not(feature = "local-onnx"))]
fn unsupported_runtime<T>(message: &str) -> Result<T> {
    Err(DetectError::InvalidArgument(format!(
        "unsupported_runtime: {message}"
    )))
}

fn run_local_qa(request: &QuestionAnsweringRequest) -> Result<QuestionAnsweringResponse> {
    #[cfg(feature = "local-onnx")]
    {
        use model_runtime::{resolve_or_download_bundle, ModelBundleResolveOptions, ModelPreset};

        let model_id = request
            .local_model
            .as_ref()
            .and_then(|options| options.model_id.clone())
            .or_else(|| request.model.model_id.clone())
            .unwrap_or_else(|| {
                ModelPreset::OnnxCommunityRobertaBaseSquad2
                    .as_str()
                    .to_string()
            });
        let preset = model_id
            .parse::<ModelPreset>()
            .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
        let mut resolve_options = ModelBundleResolveOptions::default();
        if let Some(local) = &request.local_model {
            if let Some(bundle_root) = &local.bundle_root {
                resolve_options.bundle_root = bundle_root.clone();
            }
            if let Some(auto_download) = local.auto_download {
                resolve_options.auto_download = auto_download;
            }
            if let Some(download_progress) = local.download_progress {
                resolve_options.download_progress = download_progress;
            }
            resolve_options.cache_dir = local.cache_dir.clone();
            resolve_options.hf_token = local.hf_token.clone();
            if let Some(max_retries) = local.max_retries {
                resolve_options.max_retries = max_retries;
            }
            if let Some(overwrite) = local.overwrite {
                resolve_options.overwrite = overwrite;
            }
        }
        let spec = preset.spec();
        let bundle = resolve_or_download_bundle(&spec, &resolve_options)
            .map_err(|error| DetectError::Source(error.to_string()))?;
        let mut qa = OnnxQuestionAnswerer::from_bundle(bundle)?.with_decode_options(
            QuestionAnsweringDecodeOptions {
                max_answer_len: 30,
                top_k: request.top_k.max(1),
                min_score: None,
            },
        );
        let mut answers = qa
            .answer(&request.question, &request.context)?
            .into_iter()
            .filter_map(|prediction| {
                let answer = prediction.text?;
                Some(AnswerPrediction {
                    answer,
                    score: prediction.score.unwrap_or(0.0),
                    span: span_from_attributes(&prediction.attributes),
                })
            })
            .collect::<Vec<_>>();
        answers.sort_by(|left, right| right.score.total_cmp(&left.score));
        answers.truncate(request.top_k.max(1));
        Ok(QuestionAnsweringResponse {
            accepted: true,
            operation: "question-answer".to_string(),
            question: request.question.clone(),
            model_id,
            runtime: QuestionAnsweringRuntime::Onnx,
            answers,
        })
    }
    #[cfg(not(feature = "local-onnx"))]
    {
        let _ = request;
        unsupported_runtime(
            "native question answering is server-only and requires the `local-onnx` feature",
        )
    }
}

fn heuristic_answer_response(
    request: QuestionAnsweringRequest,
    diagnostic: String,
) -> QuestionAnsweringResponse {
    let _ = diagnostic;
    let span = lexical_answer_span(&request.question, &request.context);
    let answer = span
        .map(|span| request.context[span.byte_start..span.byte_end].to_string())
        .unwrap_or_else(|| request.context.clone());
    QuestionAnsweringResponse {
        accepted: true,
        operation: "question-answer".to_string(),
        question: request.question,
        model_id: request
            .model
            .model_id
            .unwrap_or_else(|| "heuristic-unavailable-fallback".to_string()),
        runtime: QuestionAnsweringRuntime::Heuristic,
        answers: vec![AnswerPrediction {
            answer,
            score: 0.0,
            span,
        }],
    }
}

fn runtime_from_backend(backend: TextRuntimeBackend) -> QuestionAnsweringRuntime {
    match backend {
        TextRuntimeBackend::Candle => QuestionAnsweringRuntime::Candle,
        TextRuntimeBackend::Onnx => QuestionAnsweringRuntime::Onnx,
        TextRuntimeBackend::External => QuestionAnsweringRuntime::ImportedPredictions,
        TextRuntimeBackend::Tokenizers
        | TextRuntimeBackend::CudaOxide
        | TextRuntimeBackend::Heuristic => QuestionAnsweringRuntime::Heuristic,
    }
}

fn span_from_attributes(attributes: &BTreeMap<String, String>) -> Option<TextSpanRef> {
    let byte_start = attributes.get("byte_start")?.parse().ok()?;
    let byte_end = attributes.get("byte_end")?.parse().ok()?;
    let char_start = attributes
        .get("char_start")
        .and_then(|value| value.parse().ok())
        .unwrap_or(byte_start);
    let char_end = attributes
        .get("char_end")
        .and_then(|value| value.parse().ok())
        .unwrap_or(byte_end);
    Some(TextSpanRef {
        byte_start,
        byte_end,
        char_start,
        char_end,
    })
}

fn lexical_answer_span(question: &str, text: &str) -> Option<TextSpanRef> {
    let terms = question
        .split_whitespace()
        .map(|term| {
            term.trim_matches(|ch: char| !ch.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|term| term.len() > 2)
        .collect::<Vec<_>>();
    let lower_text = text.to_lowercase();
    for term in terms {
        if let Some(byte_start) = lower_text.find(&term) {
            let byte_end = (byte_start + term.len()).min(text.len());
            return Some(byte_span_to_ref(text, byte_start, byte_end));
        }
    }
    None
}

fn byte_span_to_ref(text: &str, byte_start: usize, byte_end: usize) -> TextSpanRef {
    let char_start = text[..byte_start].chars().count();
    let char_end = char_start + text[byte_start..byte_end].chars().count();
    TextSpanRef {
        byte_start,
        byte_end,
        char_start,
        char_end,
    }
}

/// Returns a JSON value for the question-answering schema catalog.
pub fn schema_summary() -> serde_json::Value {
    serde_json::json!({
        "task": "question_answering",
        "path": "/api/question-answer",
        "models": model_catalog(),
        "registeredPresets": registered_question_answering_presets(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use text_model_runtime::{RawPrediction, TextRuntimeBackend};

    struct FakeQa;

    impl QuestionAnsweringBackend for FakeQa {
        fn answer(&mut self, _question: &str, _context: &str) -> Result<Vec<RawPrediction>> {
            let mut attributes = BTreeMap::new();
            attributes.insert("byte_start".to_string(), "0".to_string());
            attributes.insert("byte_end".to_string(), "5".to_string());
            Ok(vec![RawPrediction {
                text: Some("Alice".to_string()),
                score: Some(0.8),
                attributes,
                ..RawPrediction::default()
            }])
        }

        fn runtime_backend(&self) -> TextRuntimeBackend {
            TextRuntimeBackend::External
        }
    }

    struct ContextAnswerQa {
        expected_context: Option<String>,
        answer: Option<String>,
    }

    impl QuestionAnsweringBackend for ContextAnswerQa {
        fn answer(&mut self, _question: &str, context: &str) -> Result<Vec<RawPrediction>> {
            if let Some(expected_context) = &self.expected_context {
                assert!(
                    context.contains(expected_context),
                    "QA context did not include expected text `{expected_context}`: {context}"
                );
            }
            Ok(vec![RawPrediction {
                text: Some(self.answer.clone().unwrap_or_else(|| context.to_string())),
                score: Some(0.8),
                attributes: BTreeMap::new(),
                ..RawPrediction::default()
            }])
        }

        fn runtime_backend(&self) -> TextRuntimeBackend {
            TextRuntimeBackend::External
        }
    }

    #[test]
    fn imported_answers_are_sorted() {
        let response = answer_question(QuestionAnsweringRequest {
            question: "Who?".to_string(),
            context: "Alice wrote it.".to_string(),
            top_k: 1,
            model: ModelSelection::default(),
            imported_predictions: vec![
                ImportedAnswerPrediction {
                    text: Some("Bob".to_string()),
                    score: 0.2,
                    kind: None,
                    label: None,
                    attributes: BTreeMap::new(),
                },
                ImportedAnswerPrediction {
                    text: Some("Alice".to_string()),
                    score: 0.9,
                    kind: None,
                    label: None,
                    attributes: BTreeMap::new(),
                },
            ],
            local_model: None,
            fallback_policy: None,
        })
        .expect("answer");
        assert_eq!(response.answers[0].answer, "Alice");
    }

    #[test]
    fn context_qa_uses_supplied_backend() {
        let mut qa = FakeQa;
        let mut context = QuestionAnsweringExecutionContext { qa: Some(&mut qa) };
        let response = answer_question_with_context(
            QuestionAnsweringRequest {
                question: "Who?".to_string(),
                context: "Alice wrote it.".to_string(),
                top_k: 1,
                model: ModelSelection::default(),
                imported_predictions: Vec::new(),
                local_model: None,
                fallback_policy: None,
            },
            &mut context,
        )
        .expect("context answer");
        assert_eq!(response.answers[0].answer, "Alice");
        assert_eq!(
            response.runtime,
            QuestionAnsweringRuntime::ImportedPredictions
        );
    }

    #[test]
    fn retrieval_qa_returns_cited_heuristic_answer() {
        let response = answer_question(QuestionAnsweringRequest {
            question: "Which ownership model?".to_string(),
            context: "Rust has ownership and deterministic package workflows.".to_string(),
            top_k: 1,
            model: ModelSelection::default(),
            imported_predictions: Vec::new(),
            local_model: Some(QuestionAnsweringLocalModelOptions {
                auto_download: Some(false),
                ..QuestionAnsweringLocalModelOptions::default()
            }),
            fallback_policy: Some(QuestionAnsweringFallbackPolicy::HeuristicIfUnavailable),
        })
        .expect("fallback qa");

        assert_eq!(response.runtime, QuestionAnsweringRuntime::Heuristic);
        assert_eq!(response.answers.len(), 1);
        assert_eq!(response.answers[0].answer, "ownership");
    }

    #[test]
    fn imported_retrieval_qa_prediction_maps_to_chunk_citation() {
        let response = answer_question_with_retrieval(RetrievalQuestionAnsweringRequest {
            question: "What has ownership?".to_string(),
            documents: vec![SearchDocument::new(
                "doc-rust",
                "Rust has ownership and deterministic package workflows.",
            )],
            top_k_chunks: 1,
            top_k_answers: 1,
            imported_predictions: vec![ImportedAnswerPrediction {
                kind: None,
                label: None,
                text: Some("Rust".to_string()),
                score: 0.9,
                attributes: BTreeMap::new(),
            }],
            model: ModelSelection::default(),
            local_model: None,
            fallback_policy: None,
        })
        .expect("retrieval qa");

        assert_eq!(
            response.runtime,
            QuestionAnsweringRuntime::ImportedPredictions
        );
        assert_eq!(response.answers[0].answer, "Rust");
        assert_eq!(response.answers[0].citations[0].document_id, "doc-rust");
        assert!(response.answers[0].citations[0].span.is_some());
    }

    #[test]
    fn imported_text_index_qa_prediction_maps_to_chunk_citation() {
        let response = answer_question_with_text_index(TextIndexQuestionAnsweringRequest {
            question: "What has ownership?".to_string(),
            documents: vec![IndexDocument::new(
                "doc-rust",
                "Rust has ownership and deterministic package workflows.",
            )],
            index_options: IndexBuildOptions {
                chunk_tokens: 8,
                chunk_overlap_tokens: 0,
                ..IndexBuildOptions::default()
            },
            embedding_dimensions: 32,
            top_k_chunks: 1,
            top_k_answers: 1,
            imported_predictions: vec![ImportedAnswerPrediction {
                kind: None,
                label: None,
                text: Some("Rust".to_string()),
                score: 0.9,
                attributes: BTreeMap::new(),
            }],
            model: ModelSelection::default(),
            local_model: None,
            fallback_policy: None,
        })
        .expect("text-index qa");

        assert_eq!(
            response.runtime,
            QuestionAnsweringRuntime::ImportedPredictions
        );
        assert_eq!(response.answers[0].answer, "Rust");
        assert_eq!(response.answers[0].citations[0].document_id, "doc-rust");
        assert!(response.answers[0].citations[0].span.is_some());
        assert_eq!(response.retrieved_chunks[0].document_id, "doc-rust");
    }

    #[test]
    fn text_index_qa_uses_full_chunk_text_for_answer_extraction() {
        let far_answer = "tail-only-answer";
        let body = format!(
            "Rust starts the chunk.{} {far_answer}",
            " filler".repeat(80)
        );
        let mut qa = ContextAnswerQa {
            expected_context: Some(far_answer.to_string()),
            answer: Some(far_answer.to_string()),
        };
        let mut context = QuestionAnsweringExecutionContext { qa: Some(&mut qa) };

        let response = answer_question_with_text_index_context(
            TextIndexQuestionAnsweringRequest {
                question: "What starts the chunk?".to_string(),
                documents: vec![IndexDocument::new("doc-rust", body)],
                index_options: IndexBuildOptions {
                    chunk_tokens: 512,
                    chunk_overlap_tokens: 0,
                    ..IndexBuildOptions::default()
                },
                embedding_dimensions: 32,
                top_k_chunks: 1,
                top_k_answers: 1,
                imported_predictions: Vec::new(),
                model: ModelSelection::default(),
                local_model: None,
                fallback_policy: None,
            },
            &mut context,
        )
        .expect("text-index qa");

        assert_eq!(response.answers[0].answer, far_answer);
        assert!(!response.retrieved_chunks[0].snippet.contains(far_answer));
        assert_eq!(
            response.answers[0].citations[0].snippet,
            response.retrieved_chunks[0].snippet
        );
        assert!(response.answers[0].citations[0].span.is_none());
    }

    #[test]
    fn full_text_answer_cites_matching_chunk_even_when_outside_snippet() {
        let answer = "second-tail-answer";
        let second_body = format!(
            "Rust starts the second chunk.{} {answer}",
            " filler".repeat(80)
        );
        let mut qa = ContextAnswerQa {
            expected_context: Some(answer.to_string()),
            answer: Some(answer.to_string()),
        };
        let mut context = QuestionAnsweringExecutionContext { qa: Some(&mut qa) };

        let response = answer_question_with_text_index_context(
            TextIndexQuestionAnsweringRequest {
                question: "What starts the chunk?".to_string(),
                documents: vec![
                    IndexDocument::new("doc-first", "Rust starts the first chunk."),
                    IndexDocument::new("doc-second", second_body),
                ],
                index_options: IndexBuildOptions {
                    chunk_tokens: 512,
                    chunk_overlap_tokens: 0,
                    ..IndexBuildOptions::default()
                },
                embedding_dimensions: 32,
                top_k_chunks: 2,
                top_k_answers: 1,
                imported_predictions: Vec::new(),
                model: ModelSelection::default(),
                local_model: None,
                fallback_policy: None,
            },
            &mut context,
        )
        .expect("text-index qa");

        assert_eq!(response.answers[0].answer, answer);
        assert_eq!(response.answers[0].citations[0].document_id, "doc-second");
        assert!(!response.answers[0].citations[0].snippet.contains(answer));
        assert!(response.answers[0].citations[0].span.is_none());
    }

    #[test]
    fn retrieval_qa_falls_back_to_snippet_when_raw_chunk_text_is_unavailable() {
        let raw_only = "raw-only-answer";
        let body = format!(
            "Rust starts the compatibility chunk.{} {raw_only}",
            " filler".repeat(80)
        );
        let embedder = HashedTextEmbedder::new(
            TextEmbeddingConfig {
                dimensions: 32,
                use_idf: false,
            },
            text_lexical::CorpusOptions::default(),
        )
        .expect("embedder");
        let mut index = RetrievalIndex::new(embedder);
        index
            .ingest_documents(
                &[SearchDocument::new("doc-rust", body)],
                &IngestionOptions {
                    chunk_tokens: 512,
                    chunk_overlap_tokens: 0,
                    store_raw_text: false,
                },
            )
            .expect("ingest");
        let mut qa = ContextAnswerQa {
            expected_context: None,
            answer: None,
        };
        let mut context = QuestionAnsweringExecutionContext { qa: Some(&mut qa) };

        let response = answer_question_with_retrieval_index(
            RetrievalQuestionAnsweringRequest {
                question: "What starts the compatibility chunk?".to_string(),
                documents: Vec::new(),
                top_k_chunks: 1,
                top_k_answers: 1,
                imported_predictions: Vec::new(),
                model: ModelSelection::default(),
                local_model: None,
                fallback_policy: None,
            },
            &index,
            &mut context,
        )
        .expect("retrieval qa");

        assert_eq!(
            response.answers[0].answer,
            response.retrieved_chunks[0].snippet
        );
        assert!(!response.answers[0].answer.contains(raw_only));
    }

    #[test]
    fn batch_qa_returns_item_level_results() {
        let response = answer_question_batch(QuestionAnsweringBatchRequest {
            requests: vec![
                QuestionAnsweringRequest {
                    question: "Who?".to_string(),
                    context: "Alice wrote it.".to_string(),
                    top_k: 1,
                    model: ModelSelection::default(),
                    imported_predictions: vec![ImportedAnswerPrediction {
                        text: Some("Alice".to_string()),
                        score: 0.9,
                        kind: None,
                        label: None,
                        attributes: BTreeMap::new(),
                    }],
                    local_model: None,
                    fallback_policy: None,
                },
                QuestionAnsweringRequest {
                    question: "".to_string(),
                    context: "Missing question.".to_string(),
                    top_k: 1,
                    model: ModelSelection::default(),
                    imported_predictions: Vec::new(),
                    local_model: None,
                    fallback_policy: None,
                },
            ],
        })
        .expect("batch");

        assert_eq!(response.successes, 1);
        assert_eq!(response.failures, 1);
        assert!(response.results[0].accepted);
        assert!(!response.results[1].accepted);
    }
}
