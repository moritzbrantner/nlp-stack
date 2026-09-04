use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use text_core::{split_sentence_spans, tokenize, Result, TextSpan, TokenKind};
use text_embeddings::{HashedTextEmbedder, TextEmbeddingBackend, TextEmbeddingConfig};
use text_lexical::CorpusOptions;

use crate::invalid_argument;

use super::corpus::SemanticCorpusItem;
use super::derive::build_report;
use super::{SemanticAnalysisOptions, SemanticAnalysisReport, SemanticUnit, SemanticUnitKind};

/// Structural options for deterministic word-in-context sense analysis.
#[derive(Debug, Clone, Default)]
pub struct SemanticWordSenseAnalysisOptions {
    pub semantic: SemanticAnalysisOptions,
}

/// One exact occurrence of a target term and the sentence context embedded for clustering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticWordOccurrence {
    pub unit_id: String,
    pub source_id: String,
    pub author: Option<String>,
    pub source: Option<String>,
    pub timestamp_millis: Option<i64>,
    pub occurrence_span: TextSpan,
    pub context_span: TextSpan,
    pub context_text: String,
}

/// One deterministic cluster of usage contexts for the target term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticWordSense {
    pub cluster_id: String,
    pub occurrence_count: usize,
    pub source_item_count: usize,
    pub author_count: usize,
    pub source_item_ids: Vec<String>,
    pub authors: Vec<String>,
    pub representative: SemanticWordOccurrence,
    pub occurrences: Vec<SemanticWordOccurrence>,
}

/// Corpus-level evidence for the different contexts in which one normalized term is used.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticWordSenseReport {
    pub target: String,
    pub normalized_target: String,
    pub occurrence_count: usize,
    pub source_item_count: usize,
    pub author_count: usize,
    pub senses: Vec<SemanticWordSense>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic: Option<SemanticAnalysisReport>,
}

/// Clusters sentence contexts for every occurrence of `target` with the deterministic hashed baseline.
pub fn analyze_word_senses(
    target: &str,
    items: &[SemanticCorpusItem<'_>],
    options: &SemanticWordSenseAnalysisOptions,
) -> Result<SemanticWordSenseReport> {
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 128,
            use_idf: false,
        },
        CorpusOptions::default(),
    )?;
    analyze_word_senses_with(target, items, options, &embedder)
}

/// Clusters sentence contexts for every occurrence of `target` with a caller-supplied embedder.
pub fn analyze_word_senses_with<E: TextEmbeddingBackend + ?Sized>(
    target: &str,
    items: &[SemanticCorpusItem<'_>],
    options: &SemanticWordSenseAnalysisOptions,
    embedder: &E,
) -> Result<SemanticWordSenseReport> {
    super::validate_options(&options.semantic)?;
    validate_items(items)?;
    let normalized_target = normalized_target(target, &options.semantic)?;
    let (units, occurrences) = occurrence_contexts(items, &normalized_target, &options.semantic);

    let source_item_ids = occurrences
        .iter()
        .map(|occurrence| occurrence.source_id.clone())
        .collect::<BTreeSet<_>>();
    let authors = occurrences
        .iter()
        .filter_map(|occurrence| occurrence.author.clone())
        .collect::<BTreeSet<_>>();

    if units.is_empty() {
        return Ok(SemanticWordSenseReport {
            target: target.to_string(),
            normalized_target,
            occurrence_count: 0,
            source_item_count: 0,
            author_count: 0,
            senses: Vec::new(),
            semantic: None,
        });
    }

    let semantic = build_report(
        items.iter().map(|item| item.id.to_string()).collect(),
        SemanticUnitKind::Sentence,
        units,
        &options.semantic,
        embedder,
    )?;
    let occurrences_by_unit = occurrences
        .into_iter()
        .map(|occurrence| (occurrence.unit_id.clone(), occurrence))
        .collect::<BTreeMap<_, _>>();
    let senses = semantic
        .clusters
        .iter()
        .filter_map(|cluster| {
            let representative = occurrences_by_unit
                .get(cluster.representative_unit_id.as_str())?
                .clone();
            let cluster_occurrences = cluster
                .member_unit_ids
                .iter()
                .filter_map(|unit_id| occurrences_by_unit.get(unit_id).cloned())
                .collect::<Vec<_>>();
            let cluster_source_ids = cluster_occurrences
                .iter()
                .map(|occurrence| occurrence.source_id.clone())
                .collect::<BTreeSet<_>>();
            let cluster_authors = cluster_occurrences
                .iter()
                .filter_map(|occurrence| occurrence.author.clone())
                .collect::<BTreeSet<_>>();
            Some(SemanticWordSense {
                cluster_id: cluster.id.clone(),
                occurrence_count: cluster_occurrences.len(),
                source_item_count: cluster_source_ids.len(),
                author_count: cluster_authors.len(),
                source_item_ids: cluster_source_ids.into_iter().collect(),
                authors: cluster_authors.into_iter().collect(),
                representative,
                occurrences: cluster_occurrences,
            })
        })
        .collect();

    Ok(SemanticWordSenseReport {
        target: target.to_string(),
        normalized_target,
        occurrence_count: occurrences_by_unit.len(),
        source_item_count: source_item_ids.len(),
        author_count: authors.len(),
        senses,
        semantic: Some(semantic),
    })
}

fn normalized_target(target: &str, options: &SemanticAnalysisOptions) -> Result<String> {
    if target.trim().is_empty() {
        return Err(invalid_argument("word-sense target must not be empty"));
    }
    let mut tokens = tokenize(target.trim(), &options.processing)
        .into_iter()
        .filter(|token| token.kind != TokenKind::Punctuation)
        .collect::<Vec<_>>();
    if tokens.len() != 1 {
        return Err(invalid_argument(
            "word-sense target must normalize to exactly one token",
        ));
    }
    Ok(tokens.remove(0).normalized)
}

fn occurrence_contexts(
    items: &[SemanticCorpusItem<'_>],
    normalized_target: &str,
    options: &SemanticAnalysisOptions,
) -> (Vec<SemanticUnit>, Vec<SemanticWordOccurrence>) {
    let mut units = Vec::new();
    let mut occurrences = Vec::new();
    let mut sequence_index = 0usize;

    for item in items {
        let sentences = split_sentence_spans(item.text, &options.processing);
        for token in tokenize(item.text, &options.processing) {
            if token.kind == TokenKind::Punctuation || token.normalized != normalized_target {
                continue;
            }
            let context = sentences
                .iter()
                .find(|sentence| contains_span(sentence.span, token.span));
            let (context_span, context_text) = context.map_or_else(
                || (token.span, token.text.clone()),
                |sentence| (sentence.span, sentence.text.clone()),
            );
            let unit_id = format!("{}:word-context:{sequence_index}", item.id);
            let author = item.author.map(str::trim).map(ToString::to_string);
            units.push(SemanticUnit {
                id: unit_id.clone(),
                source_id: item.id.to_string(),
                kind: SemanticUnitKind::Sentence,
                parent_id: None,
                sequence_index,
                span: context_span,
                speaker: author.clone(),
                start_seconds: None,
                end_seconds: None,
                text: context_text.clone(),
                embedding: Vec::new(),
            });
            occurrences.push(SemanticWordOccurrence {
                unit_id,
                source_id: item.id.to_string(),
                author,
                source: item.source.map(ToString::to_string),
                timestamp_millis: item.timestamp_millis,
                occurrence_span: token.span,
                context_span,
                context_text,
            });
            sequence_index += 1;
        }
    }

    (units, occurrences)
}

fn validate_items(items: &[SemanticCorpusItem<'_>]) -> Result<()> {
    if items.is_empty() {
        return Err(invalid_argument(
            "word-sense corpus must contain at least one item",
        ));
    }
    let mut seen = BTreeSet::new();
    for item in items {
        if item.id.trim().is_empty() {
            return Err(invalid_argument("word-sense corpus item id must not be empty"));
        }
        if !seen.insert(item.id) {
            return Err(invalid_argument(format!(
                "duplicate word-sense corpus item id `{}`",
                item.id
            )));
        }
        if item.text.trim().is_empty() {
            return Err(invalid_argument(format!(
                "word-sense corpus item `{}` text must not be empty",
                item.id
            )));
        }
        if item.author.is_some_and(|author| author.trim().is_empty()) {
            return Err(invalid_argument(format!(
                "word-sense corpus item `{}` author must not be empty when present",
                item.id
            )));
        }
        if item.source.is_some_and(|source| source.trim().is_empty()) {
            return Err(invalid_argument(format!(
                "word-sense corpus item `{}` source must not be empty when present",
                item.id
            )));
        }
    }
    Ok(())
}

fn contains_span(outer: TextSpan, inner: TextSpan) -> bool {
    outer.byte_start <= inner.byte_start && outer.byte_end >= inner.byte_end
}
