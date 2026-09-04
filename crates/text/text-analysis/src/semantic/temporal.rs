use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use text_core::Result;
use text_embeddings::{HashedTextEmbedder, TextEmbeddingBackend, TextEmbeddingConfig};
use text_lexical::CorpusOptions;

use crate::invalid_argument;

use super::corpus::{
    analyze_corpus_semantics_with, lexical_profile, passage, SemanticCorpusAnalysisOptions,
    SemanticCorpusItem, SemanticCorpusLexicalProfile, SemanticCorpusPassage, SemanticCorpusReport,
    SemanticCorpusSource,
};
use super::model::SemanticUnit;
use super::senses::{
    analyze_word_senses_with, SemanticWordOccurrence, SemanticWordSenseAnalysisOptions,
    SemanticWordSenseReport,
};

/// One caller-defined ordered corpus window.
///
/// Window membership is explicit: the semantic core does not infer calendar buckets.
#[derive(Debug, Clone, Copy)]
pub struct SemanticCorpusTemporalWindow<'a> {
    pub id: &'a str,
    pub item_ids: &'a [&'a str],
}

impl<'a> SemanticCorpusTemporalWindow<'a> {
    pub fn new(id: &'a str, item_ids: &'a [&'a str]) -> Self {
        Self { id, item_ids }
    }
}

/// Options for deterministic temporal projection over a semantic corpus.
#[derive(Debug, Clone)]
pub struct SemanticCorpusTemporalAnalysisOptions {
    pub corpus: SemanticCorpusAnalysisOptions,
    pub word_sense_target: Option<String>,
}

impl Default for SemanticCorpusTemporalAnalysisOptions {
    fn default() -> Self {
        Self {
            corpus: SemanticCorpusAnalysisOptions::default(),
            word_sense_target: None,
        }
    }
}

/// Share of one globally stable corpus concept inside one explicit window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusTemporalConceptShare {
    pub cluster_id: String,
    pub unit_count: usize,
    pub share: f32,
    pub evidence: SemanticCorpusPassage,
}

/// Share of one globally stable word-in-context sense inside one explicit window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusTemporalSenseShare {
    pub cluster_id: String,
    pub occurrence_count: usize,
    pub share: f32,
    pub evidence: SemanticWordOccurrence,
}

/// Optional word-sense distribution for one explicit corpus window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusTemporalWordSenseWindow {
    pub target: String,
    pub normalized_target: String,
    pub occurrence_count: usize,
    pub senses: Vec<SemanticCorpusTemporalSenseShare>,
}

/// Lexical and semantic evidence projected into one caller-defined corpus window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusTemporalWindowReport {
    pub id: String,
    pub item_ids: Vec<String>,
    pub item_count: usize,
    pub semantic_unit_count: usize,
    pub lexical: SemanticCorpusLexicalProfile,
    pub concepts: Vec<SemanticCorpusTemporalConceptShare>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_senses: Option<SemanticCorpusTemporalWordSenseWindow>,
}

/// Exact structural classification of one adjacent-window distribution change.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum SemanticTemporalChangeKind {
    Emerging,
    Persisting,
    Increased,
    Declined,
    Disappeared,
    Reentered,
    MissingEvidence,
}

/// Adjacent-window change for one globally stable corpus concept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusTemporalConceptChange {
    pub from_window_id: String,
    pub to_window_id: String,
    pub cluster_id: String,
    pub kind: SemanticTemporalChangeKind,
    pub previous_unit_count: Option<usize>,
    pub current_unit_count: Option<usize>,
    pub previous_share: Option<f32>,
    pub current_share: Option<f32>,
    pub share_delta: Option<f32>,
    pub previous_evidence: Option<SemanticCorpusPassage>,
    pub current_evidence: Option<SemanticCorpusPassage>,
}

/// Adjacent-window change for one globally stable word-in-context sense.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusTemporalSenseChange {
    pub from_window_id: String,
    pub to_window_id: String,
    pub cluster_id: String,
    pub kind: SemanticTemporalChangeKind,
    pub previous_occurrence_count: Option<usize>,
    pub current_occurrence_count: Option<usize>,
    pub previous_share: Option<f32>,
    pub current_share: Option<f32>,
    pub share_delta: Option<f32>,
    pub previous_evidence: Option<SemanticWordOccurrence>,
    pub current_evidence: Option<SemanticWordOccurrence>,
}

/// Deterministic longitudinal evidence over an attributed semantic corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusTemporalReport {
    pub corpus: SemanticCorpusReport,
    pub windows: Vec<SemanticCorpusTemporalWindowReport>,
    pub concept_changes: Vec<SemanticCorpusTemporalConceptChange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_senses: Option<SemanticWordSenseReport>,
    pub sense_changes: Vec<SemanticCorpusTemporalSenseChange>,
}

/// Projects globally stable semantic concepts into explicit ordered corpus windows.
pub fn analyze_corpus_temporal(
    items: &[SemanticCorpusItem<'_>],
    windows: &[SemanticCorpusTemporalWindow<'_>],
    options: &SemanticCorpusTemporalAnalysisOptions,
) -> Result<SemanticCorpusTemporalReport> {
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 128,
            use_idf: false,
        },
        CorpusOptions::default(),
    )?;
    analyze_corpus_temporal_with(items, windows, options, &embedder)
}

/// Projects globally stable semantic concepts using a caller-supplied embedding backend.
pub fn analyze_corpus_temporal_with<E: TextEmbeddingBackend + ?Sized>(
    items: &[SemanticCorpusItem<'_>],
    windows: &[SemanticCorpusTemporalWindow<'_>],
    options: &SemanticCorpusTemporalAnalysisOptions,
    embedder: &E,
) -> Result<SemanticCorpusTemporalReport> {
    let corpus = analyze_corpus_semantics_with(items, &options.corpus, embedder)?;
    validate_windows(&corpus, windows)?;

    let word_senses = options
        .word_sense_target
        .as_deref()
        .map(|target| {
            analyze_word_senses_with(
                target,
                items,
                &SemanticWordSenseAnalysisOptions {
                    semantic: options.corpus.semantic.clone(),
                },
                embedder,
            )
        })
        .transpose()?;

    let items_by_id = items
        .iter()
        .copied()
        .map(|item| (item.id.to_string(), item))
        .collect::<BTreeMap<_, _>>();
    let unit_by_id = corpus
        .semantic
        .units
        .iter()
        .map(|unit| (unit.id.as_str(), unit))
        .collect::<BTreeMap<_, _>>();
    let source_by_id = corpus
        .sources
        .iter()
        .map(|source| (source.id.as_str(), source))
        .collect::<BTreeMap<_, _>>();

    let window_reports = windows
        .iter()
        .map(|window| {
            window_report(
                window,
                &items_by_id,
                &corpus,
                &unit_by_id,
                &source_by_id,
                options.corpus.top_terms,
                word_senses.as_ref(),
            )
        })
        .collect::<Vec<_>>();

    let concept_changes = concept_changes(&corpus, &window_reports);
    let sense_changes = word_senses
        .as_ref()
        .map(|report| sense_changes(report, &window_reports))
        .unwrap_or_default();

    Ok(SemanticCorpusTemporalReport {
        corpus,
        windows: window_reports,
        concept_changes,
        word_senses,
        sense_changes,
    })
}

fn validate_windows(
    corpus: &SemanticCorpusReport,
    windows: &[SemanticCorpusTemporalWindow<'_>],
) -> Result<()> {
    if windows.len() < 2 {
        return Err(invalid_argument(
            "semantic temporal analysis requires at least two windows",
        ));
    }

    let source_ids = corpus
        .sources
        .iter()
        .map(|source| source.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut covered = BTreeSet::new();
    let mut window_ids = BTreeSet::new();

    for window in windows {
        if window.id.trim().is_empty() {
            return Err(invalid_argument(
                "semantic temporal window id must not be empty",
            ));
        }
        if !window_ids.insert(window.id) {
            return Err(invalid_argument(format!(
                "duplicate semantic temporal window id `{}`",
                window.id
            )));
        }

        let mut seen_items = BTreeSet::new();
        for item_id in window.item_ids {
            if !seen_items.insert(*item_id) {
                return Err(invalid_argument(format!(
                    "semantic temporal window `{}` contains duplicate item id `{item_id}`",
                    window.id
                )));
            }
            if !source_ids.contains(*item_id) {
                return Err(invalid_argument(format!(
                    "semantic temporal window `{}` references unknown item id `{item_id}`",
                    window.id
                )));
            }
            covered.insert(*item_id);
        }
    }

    if covered.len() != source_ids.len() {
        let missing = source_ids
            .difference(&covered)
            .copied()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(invalid_argument(format!(
            "semantic temporal windows must cover every corpus item; missing: {missing}"
        )));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn window_report<'a>(
    window: &SemanticCorpusTemporalWindow<'_>,
    items_by_id: &BTreeMap<String, SemanticCorpusItem<'a>>,
    corpus: &SemanticCorpusReport,
    unit_by_id: &BTreeMap<&str, &SemanticUnit>,
    source_by_id: &BTreeMap<&str, &SemanticCorpusSource>,
    top_terms: usize,
    word_senses: Option<&SemanticWordSenseReport>,
) -> SemanticCorpusTemporalWindowReport {
    let selected = window.item_ids.iter().copied().collect::<BTreeSet<_>>();
    let selected_items = window
        .item_ids
        .iter()
        .filter_map(|item_id| items_by_id.get(*item_id).copied())
        .collect::<Vec<_>>();
    let lexical = lexical_profile(selected_items.iter().copied(), top_terms);

    let mut counts = BTreeMap::<&str, usize>::new();
    for point in &corpus.semantic.timeline {
        let Some(unit) = unit_by_id.get(point.unit_id.as_str()).copied() else {
            continue;
        };
        if selected.contains(unit.source_id.as_str()) {
            *counts.entry(point.cluster_id.as_str()).or_default() += 1;
        }
    }
    let semantic_unit_count = counts.values().sum::<usize>();
    let concepts = corpus
        .semantic
        .clusters
        .iter()
        .filter_map(|cluster| {
            let unit_count = counts.get(cluster.id.as_str()).copied()?;
            let evidence_unit = cluster
                .member_unit_ids
                .iter()
                .filter_map(|unit_id| unit_by_id.get(unit_id.as_str()).copied())
                .filter(|unit| selected.contains(unit.source_id.as_str()))
                .min_by_key(|unit| unit.sequence_index)?;
            let source = source_by_id
                .get(evidence_unit.source_id.as_str())
                .copied();
            Some(SemanticCorpusTemporalConceptShare {
                cluster_id: cluster.id.clone(),
                unit_count,
                share: unit_count as f32 / semantic_unit_count as f32,
                evidence: passage(evidence_unit, source),
            })
        })
        .collect();

    SemanticCorpusTemporalWindowReport {
        id: window.id.to_string(),
        item_ids: window.item_ids.iter().map(|item_id| (*item_id).to_string()).collect(),
        item_count: selected_items.len(),
        semantic_unit_count,
        lexical,
        concepts,
        word_senses: word_senses.map(|report| word_sense_window(report, &selected)),
    }
}

fn word_sense_window(
    report: &SemanticWordSenseReport,
    selected: &BTreeSet<&str>,
) -> SemanticCorpusTemporalWordSenseWindow {
    let selected_senses = report
        .senses
        .iter()
        .filter_map(|sense| {
            let occurrence_count = sense
                .occurrences
                .iter()
                .filter(|occurrence| selected.contains(occurrence.source_id.as_str()))
                .count();
            if occurrence_count == 0 {
                return None;
            }
            let evidence = sense
                .occurrences
                .iter()
                .find(|occurrence| selected.contains(occurrence.source_id.as_str()))
                .cloned()?;
            Some((sense.cluster_id.clone(), occurrence_count, evidence))
        })
        .collect::<Vec<_>>();
    let occurrence_count = selected_senses
        .iter()
        .map(|(_, count, _)| *count)
        .sum::<usize>();
    let senses = selected_senses
        .into_iter()
        .map(|(cluster_id, count, evidence)| SemanticCorpusTemporalSenseShare {
            cluster_id,
            occurrence_count: count,
            share: count as f32 / occurrence_count as f32,
            evidence,
        })
        .collect();

    SemanticCorpusTemporalWordSenseWindow {
        target: report.target.clone(),
        normalized_target: report.normalized_target.clone(),
        occurrence_count,
        senses,
    }
}

fn concept_changes(
    corpus: &SemanticCorpusReport,
    windows: &[SemanticCorpusTemporalWindowReport],
) -> Vec<SemanticCorpusTemporalConceptChange> {
    let cluster_ids = corpus
        .semantic
        .clusters
        .iter()
        .map(|cluster| cluster.id.as_str())
        .collect::<Vec<_>>();
    let mut changes = Vec::new();

    for index in 1..windows.len() {
        let previous = &windows[index - 1];
        let current = &windows[index];
        for cluster_id in &cluster_ids {
            let previous_state = concept_state(previous, cluster_id);
            let current_state = concept_state(current, cluster_id);
            if !should_report_change(previous_state.0, current_state.0) {
                continue;
            }
            let prior_presence = windows[..index - 1]
                .iter()
                .any(|window| concept_state(window, cluster_id).0.is_some_and(|count| count > 0));
            let prior_missing = windows[..index - 1]
                .iter()
                .any(|window| window.item_count == 0);
            let kind = classify_change(
                previous_state.0,
                current_state.0,
                previous_state.1,
                current_state.1,
                prior_presence,
                prior_missing,
            );
            changes.push(SemanticCorpusTemporalConceptChange {
                from_window_id: previous.id.clone(),
                to_window_id: current.id.clone(),
                cluster_id: (*cluster_id).to_string(),
                kind,
                previous_unit_count: previous_state.0,
                current_unit_count: current_state.0,
                previous_share: previous_state.1,
                current_share: current_state.1,
                share_delta: distribution_delta(previous_state.1, current_state.1),
                previous_evidence: previous_state.2.cloned(),
                current_evidence: current_state.2.cloned(),
            });
        }
    }

    changes
}

fn sense_changes(
    report: &SemanticWordSenseReport,
    windows: &[SemanticCorpusTemporalWindowReport],
) -> Vec<SemanticCorpusTemporalSenseChange> {
    let cluster_ids = report
        .senses
        .iter()
        .map(|sense| sense.cluster_id.as_str())
        .collect::<Vec<_>>();
    let mut changes = Vec::new();

    for index in 1..windows.len() {
        let previous = &windows[index - 1];
        let current = &windows[index];
        for cluster_id in &cluster_ids {
            let previous_state = sense_state(previous, cluster_id);
            let current_state = sense_state(current, cluster_id);
            if !should_report_change(previous_state.0, current_state.0) {
                continue;
            }
            let prior_presence = windows[..index - 1]
                .iter()
                .any(|window| sense_state(window, cluster_id).0.is_some_and(|count| count > 0));
            let prior_missing = windows[..index - 1]
                .iter()
                .any(|window| window.item_count == 0);
            let kind = classify_change(
                previous_state.0,
                current_state.0,
                previous_state.1,
                current_state.1,
                prior_presence,
                prior_missing,
            );
            changes.push(SemanticCorpusTemporalSenseChange {
                from_window_id: previous.id.clone(),
                to_window_id: current.id.clone(),
                cluster_id: (*cluster_id).to_string(),
                kind,
                previous_occurrence_count: previous_state.0,
                current_occurrence_count: current_state.0,
                previous_share: previous_state.1,
                current_share: current_state.1,
                share_delta: distribution_delta(previous_state.1, current_state.1),
                previous_evidence: previous_state.2.cloned(),
                current_evidence: current_state.2.cloned(),
            });
        }
    }

    changes
}

fn concept_state<'a>(
    window: &'a SemanticCorpusTemporalWindowReport,
    cluster_id: &str,
) -> (Option<usize>, Option<f32>, Option<&'a SemanticCorpusPassage>) {
    if window.item_count == 0 {
        return (None, None, None);
    }
    let share = window
        .concepts
        .iter()
        .find(|concept| concept.cluster_id == cluster_id);
    (
        Some(share.map_or(0, |concept| concept.unit_count)),
        Some(share.map_or(0.0, |concept| concept.share)),
        share.map(|concept| &concept.evidence),
    )
}

fn sense_state<'a>(
    window: &'a SemanticCorpusTemporalWindowReport,
    cluster_id: &str,
) -> (Option<usize>, Option<f32>, Option<&'a SemanticWordOccurrence>) {
    if window.item_count == 0 {
        return (None, None, None);
    }
    let share = window
        .word_senses
        .as_ref()
        .and_then(|word_senses| {
            word_senses
                .senses
                .iter()
                .find(|sense| sense.cluster_id == cluster_id)
        });
    (
        Some(share.map_or(0, |sense| sense.occurrence_count)),
        Some(share.map_or(0.0, |sense| sense.share)),
        share.map(|sense| &sense.evidence),
    )
}

fn should_report_change(previous: Option<usize>, current: Option<usize>) -> bool {
    previous.is_some_and(|count| count > 0) || current.is_some_and(|count| count > 0)
}

fn classify_change(
    previous_count: Option<usize>,
    current_count: Option<usize>,
    previous_share: Option<f32>,
    current_share: Option<f32>,
    prior_presence: bool,
    prior_missing: bool,
) -> SemanticTemporalChangeKind {
    let (Some(previous_count), Some(current_count)) = (previous_count, current_count) else {
        return SemanticTemporalChangeKind::MissingEvidence;
    };

    match (previous_count, current_count) {
        (0, 0) => SemanticTemporalChangeKind::Persisting,
        (0, _) if prior_presence => SemanticTemporalChangeKind::Reentered,
        (0, _) if prior_missing => SemanticTemporalChangeKind::MissingEvidence,
        (0, _) => SemanticTemporalChangeKind::Emerging,
        (_, 0) => SemanticTemporalChangeKind::Disappeared,
        _ => match (
            previous_share.expect("observed previous share"),
            current_share.expect("observed current share"),
        ) {
            (previous, current) if current > previous => SemanticTemporalChangeKind::Increased,
            (previous, current) if current < previous => SemanticTemporalChangeKind::Declined,
            _ => SemanticTemporalChangeKind::Persisting,
        },
    }
}

fn distribution_delta(previous: Option<f32>, current: Option<f32>) -> Option<f32> {
    Some(current? - previous?)
}
