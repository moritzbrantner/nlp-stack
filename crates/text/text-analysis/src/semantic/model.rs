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

/// Similarity evidence for adjacent turns spoken by a pair of speakers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerPairDynamics {
    pub left_speaker: String,
    pub right_speaker: String,
    pub adjacent_turn_count: usize,
    pub mean_similarity: f32,
    pub first_similarity: f32,
    pub last_similarity: f32,
    pub similarity_delta: f32,
}

/// First observed use of a deterministic concept by a speaker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConceptIntroduction {
    pub cluster_id: String,
    pub speaker: String,
    pub sequence_index: usize,
}

/// First later use of an introduced concept by another speaker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConceptAdoption {
    pub cluster_id: String,
    pub introduced_by: String,
    pub adopted_by: String,
    pub sequence_index: usize,
}

/// Adjacent speaker change accompanied by a deterministic concept change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConceptHandoff {
    pub from_cluster_id: String,
    pub to_cluster_id: String,
    pub from_speaker: String,
    pub to_speaker: String,
    pub sequence_index: usize,
}

/// Deterministic concept that returns after at least one intervening turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecurringConcept {
    pub cluster_id: String,
    pub occurrence_count: usize,
    pub non_adjacent_return_count: usize,
    pub first_sequence_index: usize,
    pub last_sequence_index: usize,
}

/// One contiguous active run of a deterministic conversation concept thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationSemanticThreadSegment {
    pub start_sequence_index: usize,
    pub end_sequence_index: usize,
    pub unit_count: usize,
}

/// One deterministic concept tracked as a potentially re-entered conversation thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationSemanticThread {
    pub cluster_id: String,
    pub occurrence_count: usize,
    pub segment_count: usize,
    pub reentry_count: usize,
    pub first_sequence_index: usize,
    pub last_sequence_index: usize,
    pub segments: Vec<ConversationSemanticThreadSegment>,
}

/// Evidence that two deterministic concept threads alternate across nearby turns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationThreadInterleaving {
    pub left_cluster_id: String,
    pub right_cluster_id: String,
    pub alternation_count: usize,
    pub first_sequence_index: usize,
    pub last_sequence_index: usize,
}

/// Observable semantic structure derived from an ordered multi-speaker conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationSemanticDynamics {
    pub speaker_pairs: Vec<SpeakerPairDynamics>,
    pub introductions: Vec<ConceptIntroduction>,
    pub adoptions: Vec<ConceptAdoption>,
    pub handoffs: Vec<ConceptHandoff>,
    pub recurring_concepts: Vec<RecurringConcept>,
    #[serde(default)]
    pub threads: Vec<ConversationSemanticThread>,
    #[serde(default)]
    pub thread_interleavings: Vec<ConversationThreadInterleaving>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_dynamics: Option<ConversationSemanticDynamics>,
}
