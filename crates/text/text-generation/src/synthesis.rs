#![doc = include_str!("../README.md")]

use std::collections::BTreeMap;

use data_inversion_core::{Generated, InformationFidelity, InversionMethod, InversionTrace};
use serde::{Deserialize, Serialize};
use text_core::{normalize_whitespace, OwnedTextDocument};
use text_core::{AnalysisEvent, DetectError, OwnedTextSegment, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for term prompt.
pub struct TermPrompt {
    /// The term value.
    pub term: String,
    /// The weight value.
    pub weight: f32,
}

impl TermPrompt {
    /// Creates a new value.
    pub fn new(term: impl Into<String>, weight: f32) -> Self {
        Self {
            term: term.into(),
            weight,
        }
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if normalize_whitespace(&self.term).is_empty() {
            return Err(invalid_argument("term prompt text must not be empty"));
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(invalid_argument(
                "term prompt weight must be finite and greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for text synthesis options.
pub struct TextSynthesisOptions {
    /// The sentence count value.
    pub sentence_count: usize,
    /// The min terms per sentence value.
    pub min_terms_per_sentence: usize,
    /// The max terms per sentence value.
    pub max_terms_per_sentence: usize,
    /// Language tag for this value.
    pub language: Option<String>,
}

/// Trait for deterministic or model-backed text synthesis backends.
pub trait TextSynthesisBackend {
    /// Synthesizes a text document from weighted term prompts.
    fn synthesize_from_terms(
        &mut self,
        id: &str,
        terms: &[TermPrompt],
        options: TextSynthesisOptions,
    ) -> Result<Generated<OwnedTextDocument>>;
}

#[derive(Debug, Default, Clone)]
/// Deterministic template-based synthesis backend.
pub struct TemplateTextSynthesisBackend;

impl TextSynthesisBackend for TemplateTextSynthesisBackend {
    fn synthesize_from_terms(
        &mut self,
        id: &str,
        terms: &[TermPrompt],
        options: TextSynthesisOptions,
    ) -> Result<Generated<OwnedTextDocument>> {
        synthesize_from_terms(id, terms, options)
    }
}

impl Default for TextSynthesisOptions {
    fn default() -> Self {
        Self {
            sentence_count: 3,
            min_terms_per_sentence: 2,
            max_terms_per_sentence: 4,
            language: None,
        }
    }
}

impl TextSynthesisOptions {
    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if self.sentence_count == 0 {
            return Err(invalid_argument("sentence_count must be greater than zero"));
        }
        if self.min_terms_per_sentence == 0
            || self.max_terms_per_sentence < self.min_terms_per_sentence
        {
            return Err(invalid_argument(
                "term count bounds must be non-zero and increasing",
            ));
        }
        Ok(())
    }
}

/// Returns synthesize from terms.
pub fn synthesize_from_terms(
    id: impl Into<String>,
    terms: &[TermPrompt],
    options: TextSynthesisOptions,
) -> Result<Generated<OwnedTextDocument>> {
    options.validate()?;
    let ranked = ranked_terms(terms)?;
    if ranked.is_empty() {
        return Err(invalid_argument("at least one positive term is required"));
    }

    let mut sentences = Vec::with_capacity(options.sentence_count);
    for sentence_index in 0..options.sentence_count {
        let available = ranked.len().min(options.max_terms_per_sentence);
        let span = available.max(options.min_terms_per_sentence.min(ranked.len()));
        let mut selected = Vec::with_capacity(span);
        for offset in 0..span {
            selected.push(
                ranked[(sentence_index + offset) % ranked.len()]
                    .term
                    .as_str(),
            );
        }
        sentences.push(compose_sentence(&selected));
    }

    let mut document = OwnedTextDocument::new(id, sentences.join(" "));
    if let Some(language) = options.language {
        document = document.language(language);
    }
    let confidence = (0.25 + (ranked.len().min(10) as f32 * 0.025)).min(0.5);
    let trace = InversionTrace::new(
        "weighted_terms",
        "owned_text_document",
        InformationFidelity::Heuristic,
    )
    .confidence(confidence)?
    .assumption("term order is ranked by weight and then text")
    .note(
        "syntax",
        InversionMethod::Template,
        "sentences are generated from deterministic relation templates",
    )
    .note(
        "semantics",
        InversionMethod::Inferred,
        "relationships between terms are inferred, not recovered",
    );
    Ok(Generated::new(document, trace))
}

/// Returns synthesize segment from terms.
pub fn synthesize_segment_from_terms(
    segment_index: u64,
    terms: &[TermPrompt],
    options: TextSynthesisOptions,
) -> Result<Generated<OwnedTextSegment>> {
    let generated = synthesize_from_terms(format!("segment-{segment_index}"), terms, options)?;
    let segment = OwnedTextSegment::new(segment_index, generated.value.text);
    Ok(Generated::new(segment, generated.trace))
}

/// Returns terms from counts.
pub fn terms_from_counts(counts: &BTreeMap<String, usize>) -> Vec<TermPrompt> {
    counts
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(term, count)| TermPrompt::new(term.clone(), *count as f32))
        .collect()
}

/// Returns terms from events.
pub fn terms_from_events(events: &[AnalysisEvent]) -> Vec<TermPrompt> {
    let mut weights = BTreeMap::<String, f32>::new();
    for event in events {
        let weight = event.score.unwrap_or(1.0).max(0.0);
        for term in label_terms(&event.label) {
            *weights.entry(term).or_default() += weight;
        }
    }
    weights
        .into_iter()
        .filter_map(|(term, weight)| (weight > 0.0).then(|| TermPrompt::new(term, weight)))
        .collect()
}

fn ranked_terms(terms: &[TermPrompt]) -> Result<Vec<TermPrompt>> {
    for term in terms {
        term.validate()?;
    }
    let mut ranked = terms
        .iter()
        .map(|term| TermPrompt::new(normalize_whitespace(&term.term), term.weight))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| left.term.cmp(&right.term))
    });
    Ok(ranked)
}

fn compose_sentence(terms: &[&str]) -> String {
    match terms {
        [] => String::new(),
        [term] => format!("{}.", capitalize(term)),
        [first, second] => format!("{} relates to {}.", capitalize(first), second),
        [first, rest @ ..] => {
            let tail = if rest.len() == 2 {
                format!("{} and {}", rest[0], rest[1])
            } else {
                let mut terms = rest[..rest.len() - 1].join(", ");
                terms.push_str(", and ");
                terms.push_str(rest[rest.len() - 1]);
                terms
            };
            format!(
                "{} connects {}, with context inferred.",
                capitalize(first),
                tail
            )
        }
    }
}

fn capitalize(term: &str) -> String {
    let mut chars = term.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    first.to_uppercase().chain(chars).collect()
}

fn label_terms(label: &str) -> Vec<String> {
    label
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|part| part.len() > 1)
        .filter(|part| !matches!(*part, "audio" | "text" | "video" | "analysis"))
        .map(str::to_lowercase)
        .collect()
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesizes_document_from_terms() {
        let generated = synthesize_from_terms(
            "doc",
            &[
                TermPrompt::new("rust", 4.0),
                TermPrompt::new("video", 2.0),
                TermPrompt::new("analysis", 1.0),
            ],
            TextSynthesisOptions {
                sentence_count: 2,
                ..TextSynthesisOptions::default()
            },
        )
        .unwrap();
        assert!(generated.value.text.contains("Rust"));
        assert_eq!(generated.trace.fidelity, InformationFidelity::Heuristic);
    }

    #[test]
    fn extracts_terms_from_events() {
        let terms = terms_from_events(&[AnalysisEvent::new("audio_pitch", "audio:pitch:440.00hz")]);
        assert!(terms.iter().any(|term| term.term == "pitch"));
    }
}
