#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::BTreeMap;

use data_inversion_core::{Generated, InversionMethod};
use text_core::OwnedTextDocument;
use text_core::Result;
use text_generation::{
    synthesize_from_terms, MarkovChain, MarkovInputMode, TermPrompt, TextSynthesisOptions,
};
use text_linguistics::LinguisticAnalysis;

pub trait LinguisticMarkovTraining {
    fn train_analysis(&mut self, analysis: &LinguisticAnalysis, mode: MarkovInputMode);

    fn train_analyses<'a>(
        &mut self,
        analyses: impl IntoIterator<Item = &'a LinguisticAnalysis>,
        mode: MarkovInputMode,
    ) {
        for analysis in analyses {
            self.train_analysis(analysis, mode);
        }
    }
}

impl LinguisticMarkovTraining for MarkovChain {
    fn train_analysis(&mut self, analysis: &LinguisticAnalysis, mode: MarkovInputMode) {
        let tokens = analysis_tokens(analysis, mode);
        self.train_tokens(&tokens);
    }
}

pub fn analysis_tokens(analysis: &LinguisticAnalysis, mode: MarkovInputMode) -> Vec<String> {
    match mode {
        MarkovInputMode::Surface => analysis
            .tokens
            .iter()
            .map(|token| token.text.clone())
            .collect(),
        MarkovInputMode::Normalized => analysis
            .tokens
            .iter()
            .map(|token| token.normalized.clone())
            .collect(),
        MarkovInputMode::Lemma => analysis
            .lemmas
            .iter()
            .map(|lemma| lemma.value.clone())
            .collect(),
        MarkovInputMode::EntityAware => entity_aware_tokens(analysis),
    }
}

pub fn synthesize_from_analysis(
    id: impl Into<String>,
    analysis: &LinguisticAnalysis,
    options: TextSynthesisOptions,
) -> Result<Generated<OwnedTextDocument>> {
    let terms = terms_from_analysis(analysis);
    let mut generated = synthesize_from_terms(id, &terms, options)?;
    generated.trace = generated
        .trace
        .assumption("analysis terms include entities, relations, topics, and salient lemmas")
        .note(
            "analysis",
            InversionMethod::Inferred,
            "linguistic annotations are condensed into weighted prompts before generation",
        );
    Ok(generated)
}

pub fn terms_from_analysis(analysis: &LinguisticAnalysis) -> Vec<TermPrompt> {
    let mut weights = BTreeMap::<String, f32>::new();

    for lemma in &analysis.lemmas {
        if lemma.value.len() > 2 {
            *weights.entry(lemma.value.to_lowercase()).or_default() += lemma.confidence.max(0.1);
        }
    }
    for entity in &analysis.entities {
        *weights.entry(entity.normalized.to_lowercase()).or_default() += entity.confidence + 0.5;
    }
    for relation in &analysis.relations {
        *weights.entry(relation.relation.to_lowercase()).or_default() += relation.confidence;
        *weights.entry(relation.subject.to_lowercase()).or_default() += relation.confidence * 0.5;
        *weights.entry(relation.object.to_lowercase()).or_default() += relation.confidence * 0.5;
    }
    for descriptor in &analysis.topics.descriptors {
        *weights.entry(descriptor.label.to_lowercase()).or_default() += descriptor.score.max(0.1);
        for term in &descriptor.terms {
            *weights.entry(term.to_lowercase()).or_default() += descriptor.score * 0.5;
        }
    }
    for segment in &analysis.discourse {
        for cue in &segment.cues {
            *weights.entry(cue.to_lowercase()).or_default() += segment.confidence * 0.25;
        }
    }

    weights
        .into_iter()
        .filter_map(|(term, weight)| (weight > 0.0).then(|| TermPrompt::new(term, weight)))
        .collect()
}

fn entity_aware_tokens(analysis: &LinguisticAnalysis) -> Vec<String> {
    let mut entity_starts = BTreeMap::new();
    for entity in &analysis.entities {
        entity_starts.insert(
            entity.token_start,
            (
                entity.token_end,
                format!(
                    "entity:{:?}:{}",
                    entity.entity_type,
                    entity.normalized.to_lowercase()
                ),
            ),
        );
    }

    let mut tokens = Vec::new();
    let mut index = 0;
    while index < analysis.tokens.len() {
        if let Some((token_end, label)) = entity_starts.get(&index) {
            tokens.push(label.clone());
            index = (*token_end).max(index + 1);
        } else {
            tokens.push(analysis.tokens[index].normalized.clone());
            index += 1;
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model-free analysis of `text` with model annotations supplied by hand:
    /// `lemmas` as `(token, lemma)` and `person` as the token of a PER entity.
    fn fixture_analysis(text: &str, lemmas: &[(&str, &str)], person: &str) -> LinguisticAnalysis {
        let mut analysis = text_linguistics::analyze_text(
            text,
            &text_linguistics::LinguisticAnalysisOptions::without_models(),
        )
        .unwrap();
        let index = |word: &str| {
            analysis
                .tokens
                .iter()
                .position(|token| token.text == word)
                .unwrap()
        };
        analysis.lemmas = lemmas
            .iter()
            .map(|&(word, lemma)| text_linguistics::Lemma {
                token_index: index(word),
                value: lemma.to_string(),
                language: Some("en".to_string()),
                confidence: 0.9,
            })
            .collect();
        let start = index(person);
        let token = analysis.tokens[start].clone();
        analysis.entities = vec![text_linguistics::NamedEntity {
            id: "entity-0".to_string(),
            entity_type: text_linguistics::EntityType::Person,
            mention: text_linguistics::EntityMentionSpan {
                text: token.text.clone(),
                span: token.span,
            },
            normalized: token.text,
            sentence_index: 0,
            token_start: start,
            token_end: start + 1,
            confidence: 0.9,
        }];
        analysis
    }

    #[test]
    fn trains_from_linguistic_analysis_using_shared_outputs() {
        let analysis = fixture_analysis(
            "Alice launched the roadmap in Berlin.",
            &[
                ("Alice", "Alice"),
                ("launched", "launch"),
                ("roadmap", "roadmap"),
            ],
            "Alice",
        );

        let mut chain = MarkovChain::new(1).unwrap();
        chain.train_analysis(&analysis, MarkovInputMode::Lemma);
        assert!(chain
            .transitions()
            .keys()
            .flatten()
            .any(|token| token == "launch"));

        let entity_tokens = analysis_tokens(&analysis, MarkovInputMode::EntityAware);
        assert!(entity_tokens
            .iter()
            .any(|token| token.starts_with("entity:Person:alice")));
    }

    #[test]
    fn synthesizes_document_from_linguistic_analysis() {
        let analysis = fixture_analysis(
            "Alice presented the roadmap in Berlin.",
            &[("presented", "present"), ("roadmap", "roadmap")],
            "Alice",
        );
        let terms = terms_from_analysis(&analysis);
        assert!(terms.iter().any(|term| term.term.contains("alice")));

        let generated = synthesize_from_analysis(
            "analysis-doc",
            &analysis,
            TextSynthesisOptions {
                sentence_count: 1,
                ..TextSynthesisOptions::default()
            },
        )
        .unwrap();
        assert!(!generated.value.text.trim().is_empty());
    }
}
