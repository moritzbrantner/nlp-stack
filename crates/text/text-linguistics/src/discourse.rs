use std::collections::{BTreeMap, BTreeSet};

use text_core::{Sentence, Token, TokenKind};

use crate::syntax::sentence_token_ranges;
use crate::{
    ChunkKind, DependencyRelation, DependencyTree, LanguagePrediction, Lemma, LemmatizationResult,
    LexiconStore, PhraseChunk,
};

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

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for section classifier.
pub struct SectionClassifier;

impl SectionClassifier {
    /// Returns classify.
    pub fn classify(&self, sentences: &[Sentence]) -> DocumentOutline {
        let mut segments = Vec::new();
        for (index, sentence) in sentences.iter().enumerate() {
            let lower = sentence.text.to_lowercase();
            let mut cues = Vec::new();
            let mut kind = if index == 0 {
                DiscourseSegmentKind::Intro
            } else if index + 1 == sentences.len() {
                DiscourseSegmentKind::Conclusion
            } else {
                DiscourseSegmentKind::Body
            };
            let mut relation_to_previous = if index > 0 {
                Some(DiscourseRelation::Sequence)
            } else {
                None
            };
            if lower.trim_end().ends_with('?') {
                kind = DiscourseSegmentKind::Question;
                relation_to_previous = Some(DiscourseRelation::QuestionAnswer);
                cues.push("question_mark".to_string());
            } else if lower.starts_with("yes") || lower.starts_with("no") {
                kind = DiscourseSegmentKind::Answer;
                relation_to_previous = Some(DiscourseRelation::QuestionAnswer);
                cues.push("answer_cue".to_string());
            } else if lower.contains("because") || lower.contains("since") {
                kind = DiscourseSegmentKind::Evidence;
                relation_to_previous = Some(DiscourseRelation::Cause);
                cues.push("causal_marker".to_string());
            } else if lower.contains("however") || lower.contains("but ") {
                kind = DiscourseSegmentKind::Claim;
                relation_to_previous = Some(DiscourseRelation::Contrast);
                cues.push("contrast_marker".to_string());
            } else if lower.starts_with("first")
                || lower.starts_with("next")
                || lower.starts_with("finally")
            {
                kind = DiscourseSegmentKind::TopicShift;
                relation_to_previous = Some(DiscourseRelation::Transition);
                cues.push("transition_marker".to_string());
            }
            segments.push(DiscourseSegment {
                index,
                kind,
                sentence_start: index,
                sentence_end: index + 1,
                text: sentence.text.clone(),
                cues,
                relation_to_previous,
                confidence: 0.7,
            });
        }
        DocumentOutline { segments }
    }
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

pub(crate) fn build_topic_model(
    sentences: &[Sentence],
    tokens: &[Token],
    lemmas: &LemmatizationResult,
    chunks: &[PhraseChunk],
    language: Option<&LanguagePrediction>,
) -> TopicModel {
    let stop_words = LexiconStore::default()
        .stop_words_for(
            language
                .map(|prediction| prediction.language.as_str())
                .unwrap_or("en"),
        )
        .map(|lexicon| lexicon.entries.clone())
        .unwrap_or_default();

    let mut lemma_counts = BTreeMap::<String, usize>::new();
    for lemma in &lemmas.lemmas {
        let token = &tokens[lemma.token_index];
        if matches!(token.kind, TokenKind::Word | TokenKind::Number)
            && lemma.value.chars().count() >= 3
            && !stop_words.contains(&lemma.value)
        {
            *lemma_counts.entry(lemma.value.clone()).or_insert(0) += 1;
        }
    }

    let mut ranked_terms = lemma_counts.into_iter().collect::<Vec<_>>();
    ranked_terms.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let descriptors = ranked_terms
        .into_iter()
        .take(3)
        .map(|(term, count)| TopicDescriptor {
            label: term.clone(),
            terms: vec![term],
            score: count as f32 / sentences.len().max(1) as f32,
        })
        .collect::<Vec<_>>();
    let clusters = descriptors
        .iter()
        .enumerate()
        .map(|(index, descriptor)| {
            let sentence_indices = chunks
                .iter()
                .filter(|chunk| {
                    chunk.kind == ChunkKind::NounPhrase
                        && descriptor
                            .terms
                            .iter()
                            .any(|term| chunk.text.to_lowercase().contains(term))
                })
                .map(|chunk| chunk.sentence_index)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            TopicCluster {
                id: format!("topic-{index}"),
                descriptor: descriptor.clone(),
                sentence_indices,
            }
        })
        .collect();
    TopicModel {
        descriptors,
        clusters,
    }
}

pub(crate) fn build_style_profile(
    text: &str,
    sentences: &[Sentence],
    tokens: &[Token],
    lemmas: &[Lemma],
    dependencies: &[DependencyTree],
) -> StyleProfile {
    let question_count = text.matches('?').count();
    let exclamation_count = text.matches('!').count();
    let sentence_token_ranges = sentence_token_ranges(sentences, tokens);
    let sentence_lengths = sentence_token_ranges
        .iter()
        .map(|(start, end)| end.saturating_sub(*start))
        .collect::<Vec<_>>();
    let average_sentence_tokens = if sentence_lengths.is_empty() {
        0.0
    } else {
        sentence_lengths.iter().sum::<usize>() as f32 / sentence_lengths.len() as f32
    };
    let clause_markers = tokens
        .iter()
        .filter(|token| {
            matches!(
                token.normalized.as_str(),
                "and" | "but" | "because" | "that" | "which"
            )
        })
        .count();
    let clause_density = if sentences.is_empty() {
        0.0
    } else {
        clause_markers as f32 / sentences.len() as f32
    };
    let lexical_tokens = tokens
        .iter()
        .filter(|token| matches!(token.kind, TokenKind::Word | TokenKind::Number))
        .collect::<Vec<_>>();
    let unique_tokens = lexical_tokens
        .iter()
        .map(|token| token.normalized.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let unique_lemmas = lemmas
        .iter()
        .map(|lemma| lemma.value.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let type_token_ratio = if lexical_tokens.is_empty() {
        0.0
    } else {
        unique_tokens as f32 / lexical_tokens.len() as f32
    };
    let lemma_type_token_ratio = if lemmas.is_empty() {
        0.0
    } else {
        unique_lemmas as f32 / lemmas.len() as f32
    };
    let disfluency_markers = lexical_tokens
        .iter()
        .filter(|token| matches!(token.normalized.as_str(), "uh" | "um" | "er" | "ah" | "hmm"))
        .count();
    let disfluency_rate = if lexical_tokens.is_empty() {
        0.0
    } else {
        disfluency_markers as f32 / lexical_tokens.len() as f32
    };
    let passive_voice_hits = dependencies
        .iter()
        .filter(|tree| {
            tree.root_token_index.is_some_and(|root| {
                tokens[root].normalized.ends_with("ed")
                    && tree.edges.iter().any(|edge| {
                        edge.relation == DependencyRelation::Aux
                            && matches!(
                                tokens[edge.dependent_token_index].normalized.as_str(),
                                "was" | "were" | "is" | "been"
                            )
                    })
            })
        })
        .count();
    let passive_voice_estimate = if dependencies.is_empty() {
        0.0
    } else {
        passive_voice_hits as f32 / dependencies.len() as f32
    };
    let contractions = lexical_tokens
        .iter()
        .filter(|token| token.text.contains('\''))
        .count();
    let technical_terms = lexical_tokens
        .iter()
        .filter(|token| {
            matches!(
                token.normalized.as_str(),
                "api" | "schema" | "tokenizer" | "pipeline"
            )
        })
        .count();
    let formality_score = ((average_sentence_tokens / 20.0) + (technical_terms as f32 / 10.0)
        - (contractions as f32 / lexical_tokens.len().max(1) as f32))
        .clamp(0.0, 1.0);
    let repetitiveness = 1.0 - type_token_ratio;
    let register = if technical_terms >= 2 {
        RegisterEstimate::Technical
    } else if contractions >= 2 {
        RegisterEstimate::Conversational
    } else if formality_score > 0.65 {
        RegisterEstimate::Formal
    } else if formality_score < 0.25 {
        RegisterEstimate::Informal
    } else {
        RegisterEstimate::Neutral
    };

    StyleProfile {
        register,
        complexity: ComplexityMetrics {
            average_sentence_tokens,
            clause_density,
            type_token_ratio,
            lemma_type_token_ratio,
            disfluency_rate,
        },
        question_count,
        exclamation_count,
        passive_voice_estimate,
        formality_score,
        repetitiveness,
        disfluency_markers,
    }
}
