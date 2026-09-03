use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use text_core::{
    split_paragraphs, split_sentence_spans, Result, TextDocument, TextProcessingOptions, TextSpan,
};
use text_embeddings::{
    EmbeddingModelInfo, HashedTextEmbedder, TextEmbeddingBackend, TextEmbeddingConfig,
};
use text_lexical::CorpusOptions;

use crate::invalid_argument;

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
    pub fn new(id: &'a str, speaker: impl Into<Option<&'a str>>, text: &'a str) -> Self {
        Self {
            id,
            speaker: speaker.into(),
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

/// Analyzes one document with the deterministic hashed embedding baseline.
pub fn analyze_document_semantics(
    document: &TextDocument<'_>,
    options: &SemanticAnalysisOptions,
) -> Result<SemanticAnalysisReport> {
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 128,
            use_idf: false,
        },
        CorpusOptions::default(),
    )?;
    analyze_document_semantics_with(document, options, &embedder)
}

/// Analyzes one document with a caller-supplied embedding backend.
pub fn analyze_document_semantics_with<E: TextEmbeddingBackend + ?Sized>(
    document: &TextDocument<'_>,
    options: &SemanticAnalysisOptions,
    embedder: &E,
) -> Result<SemanticAnalysisReport> {
    validate_options(options)?;
    if document.id.trim().is_empty() {
        return Err(invalid_argument("semantic document id must not be empty"));
    }

    let units = document_units(document, options);
    analyze_units(
        vec![document.id.to_string()],
        SemanticUnitKind::Sentence,
        units,
        options,
        embedder,
    )
}

/// Analyzes an ordered conversation with the deterministic hashed embedding baseline.
pub fn analyze_conversation_semantics(
    turns: &[ConversationTurn<'_>],
    options: &SemanticAnalysisOptions,
) -> Result<SemanticAnalysisReport> {
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 128,
            use_idf: false,
        },
        CorpusOptions::default(),
    )?;
    analyze_conversation_semantics_with(turns, options, &embedder)
}

/// Analyzes an ordered conversation with a caller-supplied embedding backend.
pub fn analyze_conversation_semantics_with<E: TextEmbeddingBackend + ?Sized>(
    turns: &[ConversationTurn<'_>],
    options: &SemanticAnalysisOptions,
    embedder: &E,
) -> Result<SemanticAnalysisReport> {
    validate_options(options)?;
    if turns.is_empty() {
        return Err(invalid_argument(
            "semantic conversation must contain at least one turn",
        ));
    }

    let mut seen = BTreeSet::new();
    for turn in turns {
        if turn.id.trim().is_empty() {
            return Err(invalid_argument("conversation turn id must not be empty"));
        }
        if !seen.insert(turn.id) {
            return Err(invalid_argument(format!(
                "duplicate conversation turn id `{}`",
                turn.id
            )));
        }
    }

    let source_ids = turns.iter().map(|turn| turn.id.to_string()).collect();
    let units = conversation_units(turns, options);
    analyze_units(
        source_ids,
        SemanticUnitKind::SpeakerTurn,
        units,
        options,
        embedder,
    )
}

fn validate_options(options: &SemanticAnalysisOptions) -> Result<()> {
    if options.neighbors_per_unit == 0 {
        return Err(invalid_argument(
            "semantic neighbors_per_unit must be greater than zero",
        ));
    }
    validate_similarity_threshold("neighbor_threshold", options.neighbor_threshold)?;
    validate_similarity_threshold("cluster_threshold", options.cluster_threshold)
}

fn validate_similarity_threshold(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err(invalid_argument(format!(
            "semantic {name} must be finite and between -1 and 1"
        )));
    }
    Ok(())
}

fn document_units(
    document: &TextDocument<'_>,
    options: &SemanticAnalysisOptions,
) -> Vec<SemanticUnit> {
    let sentences = split_sentence_spans(document.text, &options.processing);
    let paragraphs = split_paragraphs(document.text);
    let document_unit_id = format!("{}:document", document.id);

    let mut units = Vec::new();
    for (sentence_index, sentence) in sentences.into_iter().enumerate() {
        let parent_id = paragraphs
            .iter()
            .enumerate()
            .find(|(_, paragraph)| contains_span(paragraph.span, sentence.span))
            .map(|(paragraph_index, _)| format!("{}:paragraph:{paragraph_index}", document.id));
        units.push(SemanticUnit {
            id: format!("{}:sentence:{sentence_index}", document.id),
            source_id: document.id.to_string(),
            kind: SemanticUnitKind::Sentence,
            parent_id,
            sequence_index: sentence_index,
            span: sentence.span,
            speaker: None,
            start_seconds: None,
            end_seconds: None,
            text: sentence.text,
            embedding: Vec::new(),
        });
    }

    for (paragraph_index, paragraph) in paragraphs.into_iter().enumerate() {
        units.push(SemanticUnit {
            id: format!("{}:paragraph:{paragraph_index}", document.id),
            source_id: document.id.to_string(),
            kind: SemanticUnitKind::Paragraph,
            parent_id: Some(document_unit_id.clone()),
            sequence_index: paragraph_index,
            span: paragraph.span,
            speaker: None,
            start_seconds: None,
            end_seconds: None,
            text: paragraph.text,
            embedding: Vec::new(),
        });
    }

    units.push(SemanticUnit {
        id: document_unit_id,
        source_id: document.id.to_string(),
        kind: SemanticUnitKind::Document,
        parent_id: None,
        sequence_index: 0,
        span: span_for_text(document.text),
        speaker: None,
        start_seconds: None,
        end_seconds: None,
        text: document.text.to_string(),
        embedding: Vec::new(),
    });

    units
}

fn conversation_units(
    turns: &[ConversationTurn<'_>],
    options: &SemanticAnalysisOptions,
) -> Vec<SemanticUnit> {
    let mut units = Vec::new();
    for (turn_index, turn) in turns.iter().enumerate() {
        let turn_unit_id = format!("{}:turn", turn.id);
        units.push(SemanticUnit {
            id: turn_unit_id.clone(),
            source_id: turn.id.to_string(),
            kind: SemanticUnitKind::SpeakerTurn,
            parent_id: None,
            sequence_index: turn_index,
            span: span_for_text(turn.text),
            speaker: turn.speaker.map(ToString::to_string),
            start_seconds: turn.start_seconds,
            end_seconds: turn.end_seconds,
            text: turn.text.to_string(),
            embedding: Vec::new(),
        });

        for (sentence_index, sentence) in
            split_sentence_spans(turn.text, &options.processing).into_iter().enumerate()
        {
            units.push(SemanticUnit {
                id: format!("{}:sentence:{sentence_index}", turn.id),
                source_id: turn.id.to_string(),
                kind: SemanticUnitKind::Sentence,
                parent_id: Some(turn_unit_id.clone()),
                sequence_index: sentence_index,
                span: sentence.span,
                speaker: turn.speaker.map(ToString::to_string),
                start_seconds: turn.start_seconds,
                end_seconds: turn.end_seconds,
                text: sentence.text,
                embedding: Vec::new(),
            });
        }
    }
    units
}

fn analyze_units<E: TextEmbeddingBackend + ?Sized>(
    source_ids: Vec<String>,
    primary_unit_kind: SemanticUnitKind,
    mut units: Vec<SemanticUnit>,
    options: &SemanticAnalysisOptions,
    embedder: &E,
) -> Result<SemanticAnalysisReport> {
    if units.is_empty() {
        return Err(invalid_argument("semantic analysis produced no units"));
    }

    embed_units(&mut units, embedder)?;
    let primary_indices = units
        .iter()
        .enumerate()
        .filter_map(|(index, unit)| (unit.kind == primary_unit_kind).then_some(index))
        .collect::<Vec<_>>();
    if primary_indices.is_empty() {
        return Err(invalid_argument(
            "semantic analysis produced no primary units",
        ));
    }

    let similarities = similarity_matrix(&units, &primary_indices);
    let neighbors = neighborhood_graph(
        &units,
        &primary_indices,
        &similarities,
        options.neighbors_per_unit,
        options.neighbor_threshold,
    );
    let clusters = concept_clusters(
        &units,
        &primary_indices,
        &similarities,
        options.cluster_threshold,
    );
    let timeline = semantic_timeline(&units, &primary_indices, &similarities, &clusters);
    let hotspots = semantic_hotspots(&timeline, &clusters);
    let speaker_profiles = speaker_profiles(&units, &primary_indices, &timeline);

    let mut embedding_model = embedder.model_info();
    if embedding_model.dimensions == 0 {
        embedding_model.dimensions = units.first().map_or(0, |unit| unit.embedding.len());
    }

    Ok(SemanticAnalysisReport {
        source_ids,
        primary_unit_kind,
        embedding_model,
        units,
        neighbors,
        clusters,
        timeline,
        hotspots,
        speaker_profiles,
    })
}

fn embed_units<E: TextEmbeddingBackend + ?Sized>(
    units: &mut [SemanticUnit],
    embedder: &E,
) -> Result<()> {
    let texts = units.iter().map(|unit| unit.text.as_str()).collect::<Vec<_>>();
    let vectors = embedder.embed_batch(&texts)?;
    if vectors.len() != units.len() {
        return Err(invalid_argument(format!(
            "embedding backend returned {} vectors for {} semantic units",
            vectors.len(),
            units.len()
        )));
    }

    let mut dimensions = None;
    for (unit, vector) in units.iter_mut().zip(vectors) {
        let values = vector.as_slice();
        if values.is_empty() {
            return Err(invalid_argument(
                "embedding backend returned an empty semantic vector",
            ));
        }
        if let Some(expected) = dimensions {
            if values.len() != expected {
                return Err(invalid_argument(format!(
                    "embedding backend returned inconsistent dimensions: expected {expected}, got {}",
                    values.len()
                )));
            }
        } else {
            dimensions = Some(values.len());
        }
        unit.embedding = values.to_vec();
    }
    Ok(())
}

fn similarity_matrix(units: &[SemanticUnit], primary_indices: &[usize]) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0; primary_indices.len()]; primary_indices.len()];
    for left in 0..primary_indices.len() {
        matrix[left][left] = 1.0;
        for right in (left + 1)..primary_indices.len() {
            let similarity = cosine(
                &units[primary_indices[left]].embedding,
                &units[primary_indices[right]].embedding,
            );
            matrix[left][right] = similarity;
            matrix[right][left] = similarity;
        }
    }
    matrix
}

fn neighborhood_graph(
    units: &[SemanticUnit],
    primary_indices: &[usize],
    similarities: &[Vec<f32>],
    neighbors_per_unit: usize,
    threshold: f32,
) -> Vec<SemanticNeighbor> {
    let mut edges = BTreeMap::<(usize, usize), f32>::new();
    for source in 0..primary_indices.len() {
        let mut candidates = (0..primary_indices.len())
            .filter(|target| *target != source)
            .map(|target| (target, similarities[source][target]))
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| {
                    units[primary_indices[*left_index]]
                        .id
                        .cmp(&units[primary_indices[*right_index]].id)
                })
        });

        for (target, similarity) in candidates.into_iter().take(neighbors_per_unit) {
            if similarity < threshold {
                continue;
            }
            let pair = if source < target {
                (source, target)
            } else {
                (target, source)
            };
            edges
                .entry(pair)
                .and_modify(|score| *score = score.max(similarity))
                .or_insert(similarity);
        }
    }

    edges
        .into_iter()
        .map(|((source, target), similarity)| SemanticNeighbor {
            source_unit_id: units[primary_indices[source]].id.clone(),
            target_unit_id: units[primary_indices[target]].id.clone(),
            similarity,
        })
        .collect()
}

fn concept_clusters(
    units: &[SemanticUnit],
    primary_indices: &[usize],
    similarities: &[Vec<f32>],
    threshold: f32,
) -> Vec<SemanticCluster> {
    let mut visited = vec![false; primary_indices.len()];
    let mut clusters = Vec::new();

    for seed in 0..primary_indices.len() {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        let mut queue = VecDeque::new();
        queue.push_back(seed);
        let mut members = Vec::new();
        while let Some(current) = queue.pop_front() {
            members.push(current);
            for candidate in 0..primary_indices.len() {
                if !visited[candidate] && similarities[current][candidate] >= threshold {
                    visited[candidate] = true;
                    queue.push_back(candidate);
                }
            }
        }
        members.sort_unstable();

        let representative = cluster_medoid(&members, similarities);
        let mean_similarity = cluster_mean_similarity(&members, similarities);
        let id = format!("concept-{}", clusters.len() + 1);
        clusters.push(SemanticCluster {
            id,
            member_unit_ids: members
                .iter()
                .map(|member| units[primary_indices[*member]].id.clone())
                .collect(),
            representative_unit_id: units[primary_indices[representative]].id.clone(),
            representative_text: units[primary_indices[representative]].text.clone(),
            mean_similarity,
        });
    }

    clusters
}

fn cluster_medoid(members: &[usize], similarities: &[Vec<f32>]) -> usize {
    if members.len() == 1 {
        return members[0];
    }

    members
        .iter()
        .copied()
        .map(|candidate| {
            let score = members
                .iter()
                .copied()
                .filter(|other| *other != candidate)
                .map(|other| similarities[candidate][other])
                .sum::<f32>()
                / (members.len() - 1) as f32;
            (candidate, score)
        })
        .max_by(|(left_index, left_score), (right_index, right_score)| {
            left_score
                .total_cmp(right_score)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(candidate, _)| candidate)
        .unwrap_or(members[0])
}

fn cluster_mean_similarity(members: &[usize], similarities: &[Vec<f32>]) -> f32 {
    if members.len() <= 1 {
        return 1.0;
    }
    let mut total = 0.0;
    let mut pairs = 0usize;
    for left in 0..members.len() {
        for right in (left + 1)..members.len() {
            total += similarities[members[left]][members[right]];
            pairs += 1;
        }
    }
    total / pairs.max(1) as f32
}

fn semantic_timeline(
    units: &[SemanticUnit],
    primary_indices: &[usize],
    similarities: &[Vec<f32>],
    clusters: &[SemanticCluster],
) -> Vec<SemanticTimelinePoint> {
    let mut cluster_by_unit = BTreeMap::<&str, (&str, &str)>::new();
    for cluster in clusters {
        for member in &cluster.member_unit_ids {
            cluster_by_unit.insert(
                member.as_str(),
                (cluster.id.as_str(), cluster.representative_unit_id.as_str()),
            );
        }
    }
    let primary_position = primary_indices
        .iter()
        .enumerate()
        .map(|(position, unit_index)| (units[*unit_index].id.as_str(), position))
        .collect::<BTreeMap<_, _>>();

    primary_indices
        .iter()
        .enumerate()
        .map(|(position, unit_index)| {
            let unit = &units[*unit_index];
            let (cluster_id, representative_id) = cluster_by_unit[unit.id.as_str()];
            let representative_position = primary_position[representative_id];
            let cluster_activation = similarities[position][representative_position].clamp(0.0, 1.0);
            let semantic_shift = if position == 0 {
                0.0
            } else {
                ((1.0 - similarities[position - 1][position].clamp(-1.0, 1.0)) / 2.0)
                    .clamp(0.0, 1.0)
            };
            SemanticTimelinePoint {
                unit_id: unit.id.clone(),
                sequence_index: unit.sequence_index,
                cluster_id: cluster_id.to_string(),
                semantic_shift,
                cluster_activation,
            }
        })
        .collect()
}

fn semantic_hotspots(
    timeline: &[SemanticTimelinePoint],
    clusters: &[SemanticCluster],
) -> Vec<SemanticHotspot> {
    let total_units = timeline.len().max(1) as f32;
    let mut hotspots = clusters
        .iter()
        .filter_map(|cluster| {
            let points = timeline
                .iter()
                .filter(|point| point.cluster_id == cluster.id)
                .collect::<Vec<_>>();
            let first = points.first()?;
            let last = points.last()?;
            let mean_activation = points
                .iter()
                .map(|point| point.cluster_activation)
                .sum::<f32>()
                / points.len() as f32;
            let peak_sequence_index = points
                .iter()
                .max_by(|left, right| {
                    left.cluster_activation
                        .total_cmp(&right.cluster_activation)
                        .then_with(|| right.sequence_index.cmp(&left.sequence_index))
                })
                .map(|point| point.sequence_index)
                .unwrap_or(first.sequence_index);
            Some(SemanticHotspot {
                cluster_id: cluster.id.clone(),
                coverage: points.len() as f32 / total_units,
                persistence: (last.sequence_index - first.sequence_index + 1) as f32 / total_units,
                mean_activation,
                peak_sequence_index,
            })
        })
        .collect::<Vec<_>>();
    hotspots.sort_by(|left, right| {
        right
            .coverage
            .total_cmp(&left.coverage)
            .then_with(|| right.mean_activation.total_cmp(&left.mean_activation))
            .then_with(|| left.cluster_id.cmp(&right.cluster_id))
    });
    hotspots
}

fn speaker_profiles(
    units: &[SemanticUnit],
    primary_indices: &[usize],
    timeline: &[SemanticTimelinePoint],
) -> Vec<SpeakerSemanticProfile> {
    let cluster_by_unit = timeline
        .iter()
        .map(|point| (point.unit_id.as_str(), point.cluster_id.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut counts = BTreeMap::<String, BTreeMap<String, usize>>::new();
    for unit_index in primary_indices {
        let unit = &units[*unit_index];
        let Some(speaker) = unit.speaker.as_ref() else {
            continue;
        };
        let Some(cluster_id) = cluster_by_unit.get(unit.id.as_str()) else {
            continue;
        };
        *counts
            .entry(speaker.clone())
            .or_default()
            .entry((*cluster_id).to_string())
            .or_default() += 1;
    }

    counts
        .into_iter()
        .map(|(speaker, concept_counts)| {
            let unit_count = concept_counts.values().sum::<usize>();
            let mut concepts = concept_counts
                .into_iter()
                .map(|(cluster_id, count)| SpeakerConceptShare {
                    cluster_id,
                    unit_count: count,
                    share: count as f32 / unit_count.max(1) as f32,
                })
                .collect::<Vec<_>>();
            concepts.sort_by(|left, right| {
                right
                    .unit_count
                    .cmp(&left.unit_count)
                    .then_with(|| left.cluster_id.cmp(&right.cluster_id))
            });
            SpeakerSemanticProfile {
                speaker,
                unit_count,
                concepts,
            }
        })
        .collect()
}

fn contains_span(outer: TextSpan, inner: TextSpan) -> bool {
    outer.byte_start <= inner.byte_start && outer.byte_end >= inner.byte_end
}

fn span_for_text(text: &str) -> TextSpan {
    TextSpan {
        byte_start: 0,
        byte_end: text.len(),
        char_start: 0,
        char_end: text.chars().count(),
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for index in 0..len {
        dot += left[index] * right[index];
        left_norm += left[index] * left[index];
        right_norm += right[index] * right[index];
    }
    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strict_options() -> SemanticAnalysisOptions {
        SemanticAnalysisOptions {
            neighbors_per_unit: 2,
            neighbor_threshold: 0.80,
            cluster_threshold: 0.90,
            ..SemanticAnalysisOptions::default()
        }
    }

    #[test]
    fn document_semantics_preserve_multi_scale_hierarchy() {
        let text = "Semantic search improves retrieval. Semantic search improves retrieval.\n\nTomatoes grow in soil.";
        let document = TextDocument::new("doc-1", text);
        let report = analyze_document_semantics(&document, &strict_options()).unwrap();

        assert_eq!(report.primary_unit_kind, SemanticUnitKind::Sentence);
        assert_eq!(
            report
                .units
                .iter()
                .filter(|unit| unit.kind == SemanticUnitKind::Sentence)
                .count(),
            3
        );
        assert_eq!(
            report
                .units
                .iter()
                .filter(|unit| unit.kind == SemanticUnitKind::Paragraph)
                .count(),
            2
        );
        assert_eq!(
            report
                .units
                .iter()
                .filter(|unit| unit.kind == SemanticUnitKind::Document)
                .count(),
            1
        );
        assert!(report
            .units
            .iter()
            .filter(|unit| unit.kind == SemanticUnitKind::Sentence)
            .all(|unit| unit.parent_id.is_some()));
    }

    #[test]
    fn identical_sentences_form_a_deterministic_concept_cluster() {
        let text = "Semantic search improves retrieval. Semantic search improves retrieval. Tomatoes grow in soil.";
        let document = TextDocument::new("doc-1", text);
        let report = analyze_document_semantics(&document, &strict_options()).unwrap();

        let repeated_cluster = report
            .clusters
            .iter()
            .find(|cluster| cluster.member_unit_ids.len() == 2)
            .expect("repeated semantic cluster");
        assert!(repeated_cluster.mean_similarity > 0.99);
        assert!(report
            .neighbors
            .iter()
            .any(|edge| edge.similarity > 0.99));
    }

    #[test]
    fn timeline_marks_a_large_shift_after_repeated_meaning() {
        let text = "Semantic search improves retrieval. Semantic search improves retrieval. Tomatoes grow in soil.";
        let document = TextDocument::new("doc-1", text);
        let report = analyze_document_semantics(&document, &strict_options()).unwrap();

        assert_eq!(report.timeline.len(), 3);
        assert!(report.timeline[1].semantic_shift < 0.01);
        assert!(report.timeline[2].semantic_shift > report.timeline[1].semantic_shift);
    }

    #[test]
    fn conversation_profiles_aggregate_concepts_by_speaker() {
        let turns = [
            ConversationTurn::new("turn-1", Some("Alice"), "Semantic search improves retrieval."),
            ConversationTurn::new("turn-2", Some("Bob"), "Semantic search improves retrieval."),
            ConversationTurn::new("turn-3", Some("Alice"), "Tomatoes grow in soil."),
        ];
        let report = analyze_conversation_semantics(&turns, &strict_options()).unwrap();

        assert_eq!(report.primary_unit_kind, SemanticUnitKind::SpeakerTurn);
        assert_eq!(report.timeline.len(), 3);
        let alice = report
            .speaker_profiles
            .iter()
            .find(|profile| profile.speaker == "Alice")
            .expect("Alice profile");
        assert_eq!(alice.unit_count, 2);
        assert_eq!(alice.concepts.iter().map(|concept| concept.unit_count).sum::<usize>(), 2);
    }
}
