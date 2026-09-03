use std::collections::{BTreeMap, VecDeque};

use text_core::Result;
use text_embeddings::TextEmbeddingBackend;

use crate::invalid_argument;

use super::{
    SemanticAnalysisOptions, SemanticAnalysisReport, SemanticCluster, SemanticHotspot,
    SemanticNeighbor, SemanticTimelinePoint, SemanticUnit, SemanticUnitKind, SpeakerConceptShare,
    SpeakerSemanticProfile,
};

pub(super) fn build_report<E: TextEmbeddingBackend + ?Sized>(
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
    let texts = units
        .iter()
        .map(|unit| unit.text.as_str())
        .collect::<Vec<_>>();
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
    for (source, source_similarities) in similarities
        .iter()
        .enumerate()
        .take(primary_indices.len())
    {
        let mut candidates = (0..primary_indices.len())
            .filter(|target| *target != source)
            .map(|target| (target, source_similarities[target]))
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score.total_cmp(left_score).then_with(|| {
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
        let mut queue = VecDeque::from([seed]);
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
        clusters.push(SemanticCluster {
            id: format!("concept-{}", clusters.len() + 1),
            member_unit_ids: members
                .iter()
                .map(|member| units[primary_indices[*member]].id.clone())
                .collect(),
            representative_unit_id: units[primary_indices[representative]].id.clone(),
            representative_text: units[primary_indices[representative]].text.clone(),
            mean_similarity: cluster_mean_similarity(&members, similarities),
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
    total / pairs as f32
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
            let cluster_activation =
                similarities[position][representative_position].clamp(0.0, 1.0);
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
                .map_or(first.sequence_index, |point| point.sequence_index);
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
