use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use text_core::Result;
use text_embeddings::{DenseVector, EmbeddingSearchIndex, TextEmbeddingBackend};

use crate::invalid_argument;

use super::{SemanticAnalysisReport, SemanticNeighbor, SemanticUnit};

/// Evidence comparing the deterministic pairwise neighborhood baseline with the
/// existing Foundation-backed embedding search index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticNeighborhoodEvidence {
    pub primary_unit_count: usize,
    pub neighbors_per_unit: usize,
    pub threshold: f32,
    pub exact_edges: Vec<SemanticNeighbor>,
    pub indexed_edges: Vec<SemanticNeighbor>,
    pub shared_edge_count: usize,
    pub exact_only_edge_count: usize,
    pub indexed_only_edge_count: usize,
    pub max_similarity_delta: f32,
}

/// Compares the exact semantic-neighborhood baseline with the existing
/// Foundation-backed vector index while preserving the same edge semantics.
///
/// This is an evidence path, not an automatic algorithm switch. Callers can use
/// it to demonstrate parity before choosing an indexed implementation for scale.
pub fn compare_semantic_neighborhoods(
    report: &SemanticAnalysisReport,
    neighbors_per_unit: usize,
    threshold: f32,
) -> Result<SemanticNeighborhoodEvidence> {
    if neighbors_per_unit == 0 {
        return Err(invalid_argument(
            "semantic neighbors_per_unit must be greater than zero",
        ));
    }
    if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
        return Err(invalid_argument(
            "semantic neighborhood threshold must be finite and between -1 and 1",
        ));
    }

    let primary = report
        .units
        .iter()
        .filter(|unit| unit.kind == report.primary_unit_kind)
        .collect::<Vec<_>>();
    let exact_edges = exact_edges(&primary, neighbors_per_unit, threshold);
    let indexed_edges = indexed_edges(&primary, neighbors_per_unit, threshold)?;

    let exact_by_pair = edge_map(&exact_edges);
    let indexed_by_pair = edge_map(&indexed_edges);
    let exact_pairs = exact_by_pair.keys().cloned().collect::<BTreeSet<_>>();
    let indexed_pairs = indexed_by_pair.keys().cloned().collect::<BTreeSet<_>>();
    let shared_pairs = exact_pairs
        .intersection(&indexed_pairs)
        .cloned()
        .collect::<Vec<_>>();
    let max_similarity_delta = shared_pairs
        .iter()
        .map(|pair| (exact_by_pair[pair] - indexed_by_pair[pair]).abs())
        .fold(0.0_f32, f32::max);

    Ok(SemanticNeighborhoodEvidence {
        primary_unit_count: primary.len(),
        neighbors_per_unit,
        threshold,
        exact_edges,
        indexed_edges,
        shared_edge_count: shared_pairs.len(),
        exact_only_edge_count: exact_pairs.difference(&indexed_pairs).count(),
        indexed_only_edge_count: indexed_pairs.difference(&exact_pairs).count(),
        max_similarity_delta,
    })
}

fn exact_edges(
    units: &[&SemanticUnit],
    neighbors_per_unit: usize,
    threshold: f32,
) -> Vec<SemanticNeighbor> {
    let mut edges = BTreeMap::<(String, String), f32>::new();
    for (source_index, source) in units.iter().enumerate() {
        let mut candidates = units
            .iter()
            .enumerate()
            .filter(|(target_index, _)| *target_index != source_index)
            .map(|(_, target)| (*target, cosine(&source.embedding, &target.embedding)))
            .collect::<Vec<_>>();
        candidates.sort_by(|(left, left_score), (right, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left.id.cmp(&right.id))
        });
        for (target, similarity) in candidates.into_iter().take(neighbors_per_unit) {
            if similarity < threshold {
                continue;
            }
            insert_edge(&mut edges, &source.id, &target.id, similarity);
        }
    }
    render_edges(edges)
}

fn indexed_edges(
    units: &[&SemanticUnit],
    neighbors_per_unit: usize,
    threshold: f32,
) -> Result<Vec<SemanticNeighbor>> {
    let mut vectors = BTreeMap::<String, Vec<f32>>::new();
    for unit in units {
        vectors
            .entry(unit.text.clone())
            .or_insert_with(|| unit.embedding.clone());
    }
    let embedder = ReplayEmbeddingBackend { vectors };
    let mut index = EmbeddingSearchIndex::new(embedder);
    for unit in units {
        index.add_document(unit.id.clone(), &unit.text)?;
    }

    let mut edges = BTreeMap::<(String, String), f32>::new();
    for source in units {
        let matches = index.search(&source.text, neighbors_per_unit.saturating_add(1))?;
        for hit in matches
            .into_iter()
            .filter(|hit| hit.id != source.id)
            .take(neighbors_per_unit)
        {
            if hit.score < threshold {
                continue;
            }
            insert_edge(&mut edges, &source.id, &hit.id, hit.score);
        }
    }
    Ok(render_edges(edges))
}

fn insert_edge(
    edges: &mut BTreeMap<(String, String), f32>,
    source: &str,
    target: &str,
    similarity: f32,
) {
    let pair = if source <= target {
        (source.to_string(), target.to_string())
    } else {
        (target.to_string(), source.to_string())
    };
    edges
        .entry(pair)
        .and_modify(|score| *score = score.max(similarity))
        .or_insert(similarity);
}

fn render_edges(edges: BTreeMap<(String, String), f32>) -> Vec<SemanticNeighbor> {
    edges
        .into_iter()
        .map(|((source_unit_id, target_unit_id), similarity)| SemanticNeighbor {
            source_unit_id,
            target_unit_id,
            similarity,
        })
        .collect()
}

fn edge_map(edges: &[SemanticNeighbor]) -> BTreeMap<(String, String), f32> {
    edges
        .iter()
        .map(|edge| {
            (
                (edge.source_unit_id.clone(), edge.target_unit_id.clone()),
                edge.similarity,
            )
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
struct ReplayEmbeddingBackend {
    vectors: BTreeMap<String, Vec<f32>>,
}

impl TextEmbeddingBackend for ReplayEmbeddingBackend {
    fn embed_text(&self, text: &str) -> Result<DenseVector> {
        let values = self
            .vectors
            .get(text)
            .ok_or_else(|| invalid_argument("semantic replay embedding text was not indexed"))?;
        DenseVector::new(values.clone())
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
