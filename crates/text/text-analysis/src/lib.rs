#![doc = include_str!("../README.md")]

pub mod corpus;
pub mod document;
pub mod fingerprint;
pub mod semantic;
pub mod stats;
pub mod surface;
pub mod workspace;

use std::path::PathBuf;

pub use corpus::analyze_corpus;
pub use document::{analyze_document, analyze_text};
pub use fingerprint::{shingle_hamming_distance, simhash64, DocumentSimilarityPair};
pub use semantic::{
    analyze_conversation_semantics, analyze_conversation_semantics_with,
    analyze_document_semantics, analyze_document_semantics_with, compare_semantic_neighborhoods,
    compose_linguistic_semantic_graph, interpret_semantic_report, ConceptAdoption, ConceptHandoff,
    ConceptIntroduction, ConversationSemanticDynamics, ConversationTurn, RecurringConcept,
    SemanticAnalysisOptions, SemanticAnalysisReport, SemanticCluster, SemanticConceptInterpretation,
    SemanticConceptInterpretationContent, SemanticConceptInterpretationRequest, SemanticGraphEdge,
    SemanticGraphEdgeKind, SemanticGraphNode, SemanticGraphNodeKind, SemanticHotspot,
    SemanticInterpretationBackend, SemanticInterpretationMetadata, SemanticInterpretationReport,
    SemanticLinguisticGraph, SemanticNeighbor, SemanticNeighborhoodEvidence, SemanticTimelinePoint,
    SemanticUnit, SemanticUnitKind, SpeakerConceptShare, SpeakerPairDynamics, SpeakerSemanticProfile,
};
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
    pub include_sparse_embedding: bool,
    pub linguistic_depth: LinguisticDepth,
    pub embedding_depth: EmbeddingDepth,
    pub classification_depth: ClassificationDepth,
    pub classification_local_model: Option<TextClassificationLocalModelOptions>,
}

impl Default for DocumentAnalysisOptions {
    fn default() -> Self {
        Self::deterministic()
    }
}

impl DocumentAnalysisOptions {
    pub fn deterministic() -> Self {
        Self {
            profile: AnalysisProfile::Deterministic,
            processing: TextProcessingOptions::default(),
            language_hint: None,
            keyword_limit: 10,
            summary_sentences: 3,
            ngram_sizes: vec![2, 3],
            shingle_sizes: vec![3, 5],
            include_annotation_graph: false,
            include_sparse_embedding: false,
            linguistic_depth: LinguisticDepth::HeuristicBalanced,
            embedding_depth: EmbeddingDepth::Hashed {
                dimensions: 128,
                use_idf: false,
            },
            classification_depth: ClassificationDepth::LexicalFallback,
            classification_local_model: None,
        }
    }

    pub fn model_backed() -> Self {
        Self {
            profile: AnalysisProfile::ModelBacked,
            linguistic_depth: LinguisticDepth::LocalModel {
                bundle_dir: PathBuf::from(".model-runtime/text-analysis/ner"),
                auto_download: false,
                download_progress: false,
            },
            embedding_depth: EmbeddingDepth::CandleBundle {
                bundle_dir: PathBuf::from(".model-runtime/text-analysis/embeddings"),
                pooling: PoolingStrategy::Mean,
            },
            classification_depth: ClassificationDepth::Backend,
            classification_local_model: Some(TextClassificationLocalModelOptions::default()),
            ..Self::deterministic()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextAnalysisDiagnostic {
    pub severity: String,
    pub code: String,
    pub message: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CoreAnalysisSection {
    pub stats: DetailedTextStats,
    pub paragraphs: Vec<Paragraph>,
    pub sentences: Vec<Sentence>,
    pub tokens: Vec<Token>,
    pub script_profile: ScriptProfile,
    pub annotation_graph: Option<TextAnnotationGraph>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LexicalAnalysisSection {
    pub frequencies: Vec<TermFrequency>,
    pub keywords: Vec<Keyword>,
    pub summary: Vec<SummarySentence>,
    pub sentiment: SentimentSummary,
    pub readability: ReadabilitySummary,
    pub feature_summary: TextFeatureSummary,
    pub rule_entities: Vec<EntityMention>,
    pub ngrams: Vec<NgramAnalysis>,
    pub shingles: Vec<ShingleAnalysis>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NgramAnalysis {
    pub n: usize,
    pub counts: Vec<(Vec<String>, usize)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShingleAnalysis {
    pub n: usize,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinguisticAnalysisSection {
    pub mode: String,
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
pub struct EmbeddingAnalysisSection {
    pub mode: String,
    pub model: EmbeddingModelInfo,
    pub dimensions: usize,
    pub normalized: bool,
    pub dense: Vec<f32>,
    pub sparse: Option<Vec<(usize, f32)>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationAnalysisSection {
    pub mode: String,
    pub sentiment: serde_json::Value,
    pub zero_shot: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentAnalysisReport {
    pub id: String,
    pub language: Option<String>,
    pub core: CoreAnalysisSection,
    pub lexical: LexicalAnalysisSection,
    pub linguistic: LinguisticAnalysisSection,
    pub embedding: EmbeddingAnalysisSection,
    pub classification: ClassificationAnalysisSection,
    pub diagnostics: Vec<TextAnalysisDiagnostic>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CorpusAnalysisOptions {
    pub document: DocumentAnalysisOptions,
    pub query: Option<String>,
    pub top_k: usize,
    pub tfidf_terms_per_document: usize,
    pub include_near_duplicates: bool,
    pub include_semantic_neighbors: bool,
}

impl Default for CorpusAnalysisOptions {
    fn default() -> Self {
        Self {
            document: DocumentAnalysisOptions::default(),
            query: None,
            top_k: 10,
            tfidf_terms_per_document: 10,
            include_near_duplicates: true,
            include_semantic_neighbors: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusDocumentAnalysis {
    pub report: DocumentAnalysisReport,
    pub tfidf: Vec<TfIdfTerm>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusAnalysisReport {
    pub documents: Vec<CorpusDocumentAnalysis>,
    pub stats: CorpusStats,
    pub term_stats: Vec<CorpusTermStats>,
    pub tfidf_search: Vec<DocumentSearchResult>,
    pub bm25_search: Vec<Bm25SearchResult>,
    pub near_duplicates: Vec<DocumentSimilarityPair>,
    pub semantic_neighbors: Vec<DocumentSimilarityPair>,
    pub diagnostics: Vec<TextAnalysisDiagnostic>,
}
