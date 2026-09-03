use serde::{Deserialize, Serialize};
use text_core::{TextProcessingOptions, TextSpan};
use text_embeddings::EmbeddingModelInfo;

/// Granularity of one meaning-bearing unit in a semantic analysis report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SemanticUnitKind {
    Sentence,
    Paragraph,
    Document,
    SpeakerTurn,
}

/// Structural and algorithmic options for semantic-map analysis.
#[derive(Debug, Clone)]
pub struct SemanticAnalysisOptions {
    pub processing: TextProcessingOptions,
    pub neighbors_per_unit: usize,
    pub neighbor_threshold: f32,
    pub cluster_threshold: f32,
}

impl Default for SemanticAnalysisOptions {
    fn default() -> Self {
        Self {
            processing: TextProcessingOptions::default(),
            neighbors_per_unit: 4,
            neighbor_threshold: 0.25,
            cluster_threshold: 0.60,
        }
    }
}

/// One ordered speaker turn supplied to conversation semantic analysis.
#[derive(Debug, Clone, Copy)]
pub struct ConversationTurn<'a> {
    pub id: &'a str,
    pub speaker: Option<&'a str>,
    pub text: &'a str,
    pub start_seconds: Option<f64>,
    pub end_seconds: Option<f64>,
}

impl<'a> ConversationTurn<'a> {
    /// Creates a speaker turn without timing metadata.
    pub fn new(id: &'a str, speaker: Option<&'a str>, text: &'a str) -> Self {
        Self {
            id,
            speaker,
            text,
            start_seconds: None,
            end_seconds: None,
        }
    }
}

/// Embedded semantic unit with source and hierarchy provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticUnit {
    pub id: String,
    pub source_id: String,
    pub kind: SemanticUnitKind,
    pub parent_id: Option<String>,
    pub sequence_index: usize,
    pub span: TextSpan,
    pub speaker: Option<String>,
    pub start_seconds: Option<f64>,
    pub end_seconds: Option<f64>,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// Undirected semantic-neighborhood edge between two primary units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticNeighbor {
    pub source_unit_id: String,
    pub target_unit_id: String,
    pub similarity: f32,
}

/// Deterministic concept cluster over primary semantic units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCluster {
    pub id: String,
    pub member_unit_ids: Vec<String>,
    pub representative_unit_id: String,
    pub representative_text: String,
    pub mean_similarity: f32,
}

/// Ordered semantic state for one primary unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticTimelinePoint {
    pub unit_id: String,
    pub sequence_index: usize,
    pub cluster_id: String,
    pub semantic_shift: f32,
    pub cluster_activation: f32,
}

/// Aggregate concentration of one concept across the primary semantic sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticHotspot {
    pub cluster_id: String,
    pub coverage: f32,
    pub persistence: f32,
    pub mean_activation: f32,
    pub peak_sequence_index: usize,
}

/// Share of one speaker's turns assigned to one semantic concept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerConceptShare {
    pub cluster_id: String,
    pub unit_count: usize,
    pub share: f32,
}

/// Conversation-level semantic distribution for one speaker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerSemanticProfile {
    pub speaker: String,
    pub unit_count: usize,
    pub concepts: Vec<SpeakerConceptShare>,
}

/// Multi-scale semantic map derived from a document or ordered conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticAnalysisReport {
    pub source_ids: Vec<String>,
    pub primary_unit_kind: SemanticUnitKind,
    pub embedding_model: EmbeddingModelInfo,
    pub units: Vec<SemanticUnit>,
    pub neighbors: Vec<SemanticNeighbor>,
    pub clusters: Vec<SemanticCluster>,
    pub timeline: Vec<SemanticTimelinePoint>,
    pub hotspots: Vec<SemanticHotspot>,
    pub speaker_profiles: Vec<SpeakerSemanticProfile>,
}
