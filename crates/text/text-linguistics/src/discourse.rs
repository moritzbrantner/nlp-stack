//! Result types for discourse segments, topics and style. No producer fills them:
//! the cue-word section classifier, lemma-count topics and count-based style
//! profile were removed (#186), and requesting those layers fails with "model
//! required" until model backends exist.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing discourse relation.
pub enum DiscourseRelation {
    /// The sequence variant.
    Sequence,
    /// The elaboration variant.
    Elaboration,
    /// The contrast variant.
    Contrast,
    /// The cause variant.
    Cause,
    /// The question answer variant.
    QuestionAnswer,
    /// The transition variant.
    Transition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing discourse segment kind.
pub enum DiscourseSegmentKind {
    /// The intro variant.
    Intro,
    /// The body variant.
    Body,
    /// The conclusion variant.
    Conclusion,
    /// The question variant.
    Question,
    /// The answer variant.
    Answer,
    /// The claim variant.
    Claim,
    /// The evidence variant.
    Evidence,
    /// The topic shift variant.
    TopicShift,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for discourse segment.
pub struct DiscourseSegment {
    /// The index value.
    pub index: usize,
    /// The kind value.
    pub kind: DiscourseSegmentKind,
    /// The sentence start value.
    pub sentence_start: usize,
    /// The sentence end value.
    pub sentence_end: usize,
    /// Text content for this value.
    pub text: String,
    /// The cues value.
    pub cues: Vec<String>,
    /// The relation to previous value.
    pub relation_to_previous: Option<DiscourseRelation>,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for document outline.
pub struct DocumentOutline {
    /// The segments value.
    pub segments: Vec<DiscourseSegment>,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for topic descriptor.
pub struct TopicDescriptor {
    /// Label assigned to this value.
    pub label: String,
    /// The terms value.
    pub terms: Vec<String>,
    /// Score assigned to this value.
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for topic cluster.
pub struct TopicCluster {
    /// Identifier for this value.
    pub id: String,
    /// The descriptor value.
    pub descriptor: TopicDescriptor,
    /// The sentence indices value.
    pub sentence_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for topic model.
pub struct TopicModel {
    /// The descriptors value.
    pub descriptors: Vec<TopicDescriptor>,
    /// The clusters value.
    pub clusters: Vec<TopicCluster>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing register estimate.
pub enum RegisterEstimate {
    /// The formal variant.
    Formal,
    /// The neutral variant.
    Neutral,
    /// The informal variant.
    Informal,
    /// The technical variant.
    Technical,
    /// The conversational variant.
    Conversational,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for complexity metrics.
pub struct ComplexityMetrics {
    /// The average sentence tokens value.
    pub average_sentence_tokens: f32,
    /// The clause density value.
    pub clause_density: f32,
    /// The type token ratio value.
    pub type_token_ratio: f32,
    /// The lemma type token ratio value.
    pub lemma_type_token_ratio: f32,
    /// The disfluency rate value.
    pub disfluency_rate: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for style profile.
pub struct StyleProfile {
    /// The register value.
    pub register: RegisterEstimate,
    /// The complexity value.
    pub complexity: ComplexityMetrics,
    /// The question count value.
    pub question_count: usize,
    /// The exclamation count value.
    pub exclamation_count: usize,
    /// The passive voice estimate value.
    pub passive_voice_estimate: f32,
    /// The formality score value.
    pub formality_score: f32,
    /// The repetitiveness value.
    pub repetitiveness: f32,
    /// The disfluency markers value.
    pub disfluency_markers: usize,
}

impl Default for StyleProfile {
    fn default() -> Self {
        Self {
            register: RegisterEstimate::Neutral,
            complexity: ComplexityMetrics {
                average_sentence_tokens: 0.0,
                clause_density: 0.0,
                type_token_ratio: 0.0,
                lemma_type_token_ratio: 0.0,
                disfluency_rate: 0.0,
            },
            question_count: 0,
            exclamation_count: 0,
            passive_voice_estimate: 0.0,
            formality_score: 0.0,
            repetitiveness: 0.0,
            disfluency_markers: 0,
        }
    }
}
