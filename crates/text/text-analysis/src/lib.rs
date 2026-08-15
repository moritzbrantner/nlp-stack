#![doc = include_str!("../README.md")]

pub mod corpus;
pub mod document;
pub mod fingerprint;
pub mod stats;
pub mod surface;
pub mod workspace;

use std::path::PathBuf;

pub use corpus::analyze_corpus;
pub use document::{analyze_document, analyze_text};
pub use fingerprint::{shingle_hamming_distance, simhash64, DocumentSimilarityPair};
use serde::{Deserialize, Serialize};
pub use stats::enriched_text_stats;
use text_classification::TextClassificationLocalModelOptions;
use text_core::{
    DetailedTextStats, Paragraph, ScriptProfile, Sentence, TextAnnotationGraph, TextDocument,
    TextProcessingOptions, Token,
};
use text_embeddings::{EmbeddingModelInfo, PoolingStrategy};
use text_lexical::{
    Bm25SearchResult, CorpusStats, CorpusTermStats, DocumentSearchResult, EntityMention, Keyword,
    ReadabilitySummary, SentimentSummary, SummarySentence, TermFrequency, TextFeatureSummary,
    TfIdfTerm,
};
pub use workspace::{
    TextWorkspace, TextWorkspaceOptions, WorkspaceDocument, WorkspaceIndexOptions,
    WorkspaceIndexSearchReport, WorkspaceIndexStorage, WorkspaceIngestReport,
    WorkspaceSearchReport, WorkspaceSnapshot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnalysisProfile {
    Deterministic,
    ModelBacked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinguisticDepth {
    Off,
    HeuristicFast,
    HeuristicBalanced,
    HeuristicRich,
    LocalModel {
        bundle_dir: PathBuf,
        auto_download: bool,
        download_progress: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingDepth {
    Off,
    Hashed {
        dimensions: usize,
        use_idf: bool,
    },
    CandleBundle {
        bundle_dir: PathBuf,
        pooling: PoolingStrategy,
    },
    OnnxBundle {
        bundle_dir: PathBuf,
        pooling: PoolingStrategy,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ClassificationDepth {
    Off,
    LexicalFallback,
    Imported,
    Backend,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentAnalysisOptions {
    pub profile: AnalysisProfile,
    pub processing: TextProcessingOptions,
    pub language_hint: Option<String>,
    pub keyword_limit: usize,
    pub summary_sentences: usize,
    pub ngram_sizes: Vec<usize>,
    pub shingle_sizes: Vec<usize>,
    pub include_annotation_graph: bool,
    pub linguistic_depth: LinguisticDepth,
    pub embedding_depth: EmbeddingDepth,
    pub include_sparse_embedding: bool,
    pub classification_depth: ClassificationDepth,
    pub classification_local_model: Option<TextClassificationLocalModelOptions>,
    pub classification_labels: Vec<String>,
    pub zero_shot_labels: Vec<String>,
}

impl Default for DocumentAnalysisOptions {
    fn default() -> Self {
        Self {
            profile: AnalysisProfile::Deterministic,
            processing: TextProcessingOptions::default(),
            language_hint: None,
            keyword_limit: 10,
            summary_sentences: 3,
            ngram_sizes: vec![2, 3],
            shingle_sizes: vec![3, 5],
            include_annotation_graph: true,
            linguistic_depth: LinguisticDepth::HeuristicBalanced,
            embedding_depth: EmbeddingDepth::Hashed {
                dimensions: 128,
                use_idf: false,
            },
            include_sparse_embedding: false,
            classification_depth: ClassificationDepth::Off,
            classification_local_model: None,
            classification_labels: vec![
                "technical".to_string(),
                "media".to_string(),
                "documentation".to_string(),
            ],
            zero_shot_labels: vec![
                "workflow".to_string(),
                "reference".to_string(),
                "analysis".to_string(),
            ],
        }
    }
}

impl DocumentAnalysisOptions {
    pub fn model_backed() -> Self {
        Self {
            profile: AnalysisProfile::ModelBacked,
            linguistic_depth: LinguisticDepth::LocalModel {
                bundle_dir: PathBuf::from(".model-runtime"),
                auto_download: false,
                download_progress: false,
            },
            classification_depth: ClassificationDepth::Backend,
            classification_local_model: Some(TextClassificationLocalModelOptions {
                bundle_root: Some(PathBuf::from(".model-runtime")),
                auto_download: Some(false),
                download_progress: Some(false),
                ..TextClassificationLocalModelOptions::default()
            }),
            ..Self::default()
        }
    }

    pub fn model_backed_with_downloads() -> Self {
        Self {
            profile: AnalysisProfile::ModelBacked,
            linguistic_depth: LinguisticDepth::LocalModel {
                bundle_dir: PathBuf::from(".model-runtime"),
                auto_download: true,
                download_progress: true,
            },
            classification_depth: ClassificationDepth::Backend,
            classification_local_model: Some(TextClassificationLocalModelOptions {
                bundle_root: Some(PathBuf::from(".model-runtime")),
                auto_download: Some(true),
                download_progress: Some(true),
                ..TextClassificationLocalModelOptions::default()
            }),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CorpusAnalysisOptions {
    pub document: DocumentAnalysisOptions,
    pub query: Option<String>,
    pub top_k: usize,
    pub tfidf_terms_per_document: usize,
    pub include_near_duplicates: bool,
    pub include_semantic_neighbors: bool,
    pub near_duplicate_shingle_size: usize,
    pub near_duplicate_threshold: f32,
}

impl Default for CorpusAnalysisOptions {
    fn default() -> Self {
        let document = DocumentAnalysisOptions {
            embedding_depth: EmbeddingDepth::Hashed {
                dimensions: 128,
                use_idf: true,
            },
            ..DocumentAnalysisOptions::default()
        };
        Self {
            document,
            query: None,
            top_k: 10,
            tfidf_terms_per_document: 10,
            include_near_duplicates: true,
            include_semantic_neighbors: true,
            near_duplicate_shingle_size: 5,
            near_duplicate_threshold: 0.6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextAnalysisDiagnostic {
    pub code: String,
    pub message: String,
    pub severity: String,
    pub source: Option<String>,
}

impl TextAnalysisDiagnostic {
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            severity: "warning".to_string(),
            source: Some("text-analysis".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentAnalysisReport {
    pub id: String,
    pub language: Option<String>,
    pub core: CoreAnalysisSection,
    pub enriched_stats: EnrichedTextStats,
    pub lexical: LexicalAnalysisSection,
    pub similarity: SimilarityAnalysisSection,
    pub linguistic: Option<LinguisticAnalysisSection>,
    pub embedding: Option<EmbeddingAnalysisSection>,
    #[serde(default)]
    pub classification: Option<ClassificationAnalysisSection>,
    pub diagnostics: Vec<TextAnalysisDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusAnalysisReport {
    pub stats: CorpusStats,
    pub term_stats: Vec<CorpusTermStats>,
    pub documents: Vec<CorpusDocumentAnalysis>,
    pub tfidf_search: Option<Vec<DocumentSearchResult>>,
    pub bm25_search: Option<Vec<Bm25SearchResult>>,
    pub near_duplicates: Vec<DocumentSimilarityPair>,
    pub semantic_neighbors: Vec<DocumentSimilarityPair>,
    #[serde(default)]
    pub classification: Option<CorpusClassificationSection>,
    pub diagnostics: Vec<TextAnalysisDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CoreAnalysisSection {
    pub stats: DetailedTextStats,
    pub script_profile: ScriptProfile,
    pub tokens: Vec<Token>,
    pub sentences: Vec<Sentence>,
    pub paragraphs: Vec<Paragraph>,
    pub annotation_graph: Option<TextAnnotationGraph>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnrichedTextStats {
    pub lexical_density: f32,
    pub stopword_ratio: f32,
    pub hapax_ratio: f32,
    pub shannon_entropy: f32,
    pub punctuation_token_ratio: f32,
    pub uppercase_token_ratio: f32,
    pub numeric_token_ratio: f32,
    pub url_count: usize,
    pub email_count: usize,
    pub mention_count: usize,
    pub hashtag_count: usize,
    pub sentence_token_min: usize,
    pub sentence_token_max: usize,
    pub sentence_token_mean: f32,
    pub sentence_token_p50: usize,
    pub sentence_token_p90: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LexicalAnalysisSection {
    pub summary: TextFeatureSummary,
    pub top_terms: Vec<TermFrequency>,
    pub keywords: Vec<Keyword>,
    pub phrase_keywords: Vec<Keyword>,
    pub readability: ReadabilitySummary,
    pub sentiment: SentimentSummary,
    pub extractive_summary: Vec<SummarySentence>,
    pub diverse_extractive_summary: Vec<SummarySentence>,
    pub rule_entities: Vec<EntityMention>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NgramFrequencyReport {
    pub n: usize,
    pub terms: Vec<text_lexical::NgramFrequency>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShingleCountReport {
    pub n: usize,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimilarityAnalysisSection {
    pub character_ngram_frequencies: Vec<NgramFrequencyReport>,
    pub token_ngram_frequencies: Vec<NgramFrequencyReport>,
    pub character_shingle_counts: Vec<ShingleCountReport>,
    pub token_shingle_counts: Vec<ShingleCountReport>,
    pub token_shingle_simhash: String,
    pub character_shingle_simhash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinguisticAnalysisSection {
    pub language: serde_json::Value,
    pub tokenizer: serde_json::Value,
    pub lemmas: serde_json::Value,
    pub morphology: serde_json::Value,
    pub pos: serde_json::Value,
    pub chunks: serde_json::Value,
    pub dependencies: serde_json::Value,
    pub entities: serde_json::Value,
    pub canonical_entities: serde_json::Value,
    pub coreference: serde_json::Value,
    pub events: serde_json::Value,
    pub relations: serde_json::Value,
    pub discourse: serde_json::Value,
    pub outline: serde_json::Value,
    pub topics: serde_json::Value,
    pub style: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SparseEmbeddingReport {
    pub dimensions: usize,
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingAnalysisSection {
    pub dimensions: usize,
    pub preview: Vec<f32>,
    pub vector: Vec<f32>,
    pub sparse: Option<SparseEmbeddingReport>,
    pub model: EmbeddingModelInfo,
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationAnalysisSection {
    pub sentiment: text_classification::SentimentResponse,
    pub classification: text_classification::TextClassificationResponse,
    pub zero_shot: text_classification::ZeroShotClassificationResponse,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelDistribution {
    pub label: String,
    pub count: usize,
    pub score_sum: f32,
    pub score_mean: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusClassificationSection {
    pub label_distribution: Vec<LabelDistribution>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusDocumentAnalysis {
    pub id: String,
    pub stats: DetailedTextStats,
    pub tfidf_terms: Vec<TfIdfTerm>,
    pub keywords: Vec<Keyword>,
    pub embedding_preview: Option<Vec<f32>>,
    #[serde(default)]
    pub classification: Option<ClassificationAnalysisSection>,
}

pub fn document_from_text<'a>(id: &'a str, text: &'a str) -> TextDocument<'a> {
    TextDocument::new(id, text)
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> text_core::DetectError {
    text_core::DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_enable_deterministic_sections() {
        let options = DocumentAnalysisOptions::default();
        assert_eq!(options.profile, AnalysisProfile::Deterministic);
        assert!(matches!(
            options.embedding_depth,
            EmbeddingDepth::Hashed { .. }
        ));
        assert!(matches!(
            options.linguistic_depth,
            LinguisticDepth::HeuristicBalanced
        ));
    }
}
