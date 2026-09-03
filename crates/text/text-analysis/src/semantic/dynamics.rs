use std::collections::{BTreeMap, BTreeSet};

use super::{
    ConceptAdoption, ConceptHandoff, ConceptIntroduction, ConversationSemanticDynamics,
    RecurringConcept, SemanticTimelinePoint, SemanticUnit, SpeakerPairDynamics,
};

pub(super) fn conversation_dynamics(
    units: &[SemanticUnit],
    primary_indices: &[usize],
    timeline: &[SemanticTimelinePoint],
) -> ConversationSemanticDynamics {
    let primary = primary_indices
        .iter()
        .map(|index| &units[*index])
        .collect::<Vec<_>>();
    let cluster_by_unit = timeline
        .iter()
        .map(|point| (point.unit_id.as_str(), point.cluster_id.as_str()))
        .collect::<BTreeMap<_, _>>();

    ConversationSemanticDynamics {
        speaker_pairs: speaker_pair_dynamics(&primary),
        introductions: concept_introductions(&primary, &cluster_by_unit),
        adoptions: concept_adoptions(&primary, &cluster_by_unit),
        handoffs: concept_handoffs(&primary, &cluster_by_unit),
        recurring_concepts: recurring_concepts(&primary, &cluster_by_unit),
    }
}

fn speaker_pair_dynamics(units: &[&SemanticUnit]) -> Vec<SpeakerPairDynamics> {
    let mut similarities = BTreeMap::<(String, String), Vec<f32>>::new();
    for pair in units.windows(2) {
        let left = pair[0];
        let right = pair[1];
        let (Some(left_speaker), Some(right_speaker)) =
            (left.speaker.as_ref(), right.speaker.as_ref())
        else {
            continue;
        };
        if left_speaker == right_speaker {
            continue;
        }
        let speakers = if left_speaker <= right_speaker {
            (left_speaker.clone(), right_speaker.clone())
        } else {
            (right_speaker.clone(), left_speaker.clone())
        };
        similarities
            .entry(speakers)
            .or_default()
            .push(cosine(&left.embedding, &right.embedding));
    }

    similarities
        .into_iter()
        .map(|((left_speaker, right_speaker), scores)| {
            let first_similarity = scores.first().copied().unwrap_or(0.0);
            let last_similarity = scores.last().copied().unwrap_or(first_similarity);
            SpeakerPairDynamics {
                left_speaker,
                right_speaker,
                adjacent_turn_count: scores.len(),
                mean_similarity: scores.iter().sum::<f32>() / scores.len().max(1) as f32,
                first_similarity,
                last_similarity,
                similarity_delta: last_similarity - first_similarity,
            }
        })
        .collect()
}

fn concept_introductions(
    units: &[&SemanticUnit],
    cluster_by_unit: &BTreeMap<&str, &str>,
) -> Vec<ConceptIntroduction> {
    let mut seen = BTreeSet::<String>::new();
    let mut introductions = Vec::new();
    for unit in units {
        let (Some(speaker), Some(cluster_id)) =
            (unit.speaker.as_ref(), cluster_by_unit.get(unit.id.as_str()))
        else {
            continue;
        };
        if seen.insert((*cluster_id).to_string()) {
            introductions.push(ConceptIntroduction {
                cluster_id: (*cluster_id).to_string(),
                speaker: speaker.clone(),
                sequence_index: unit.sequence_index,
            });
        }
    }
    introductions
}

fn concept_adoptions(
    units: &[&SemanticUnit],
    cluster_by_unit: &BTreeMap<&str, &str>,
) -> Vec<ConceptAdoption> {
    let mut introduced_by = BTreeMap::<String, String>::new();
    let mut adopted_by = BTreeSet::<(String, String)>::new();
    let mut adoptions = Vec::new();
    for unit in units {
        let (Some(speaker), Some(cluster_id)) =
            (unit.speaker.as_ref(), cluster_by_unit.get(unit.id.as_str()))
        else {
            continue;
        };
        let cluster_id = (*cluster_id).to_string();
        let Some(introducer) = introduced_by.get(&cluster_id) else {
            introduced_by.insert(cluster_id, speaker.clone());
            continue;
        };
        if introducer == speaker {
            continue;
        }
        if adopted_by.insert((cluster_id.clone(), speaker.clone())) {
            adoptions.push(ConceptAdoption {
                cluster_id,
                introduced_by: introducer.clone(),
                adopted_by: speaker.clone(),
                sequence_index: unit.sequence_index,
            });
        }
    }
    adoptions
}

fn concept_handoffs(
    units: &[&SemanticUnit],
    cluster_by_unit: &BTreeMap<&str, &str>,
) -> Vec<ConceptHandoff> {
    units
        .windows(2)
        .filter_map(|pair| {
            let left = pair[0];
            let right = pair[1];
            let left_speaker = left.speaker.as_ref()?;
            let right_speaker = right.speaker.as_ref()?;
            if left_speaker == right_speaker {
                return None;
            }
            let from_cluster_id = *cluster_by_unit.get(left.id.as_str())?;
            let to_cluster_id = *cluster_by_unit.get(right.id.as_str())?;
            if from_cluster_id == to_cluster_id {
                return None;
            }
            Some(ConceptHandoff {
                from_cluster_id: from_cluster_id.to_string(),
                to_cluster_id: to_cluster_id.to_string(),
                from_speaker: left_speaker.clone(),
                to_speaker: right_speaker.clone(),
                sequence_index: right.sequence_index,
            })
        })
        .collect()
}

fn recurring_concepts(
    units: &[&SemanticUnit],
    cluster_by_unit: &BTreeMap<&str, &str>,
) -> Vec<RecurringConcept> {
    let mut occurrences = BTreeMap::<String, Vec<usize>>::new();
    for unit in units {
        if let Some(cluster_id) = cluster_by_unit.get(unit.id.as_str()) {
            occurrences
                .entry((*cluster_id).to_string())
                .or_default()
                .push(unit.sequence_index);
        }
    }

    occurrences
        .into_iter()
        .filter_map(|(cluster_id, indices)| {
            if indices.len() < 2 {
                return None;
            }
            let non_adjacent_return_count = indices
                .windows(2)
                .filter(|pair| pair[1].saturating_sub(pair[0]) > 1)
                .count();
            Some(RecurringConcept {
                cluster_id,
                occurrence_count: indices.len(),
                non_adjacent_return_count,
                first_sequence_index: indices[0],
                last_sequence_index: *indices.last().unwrap_or(&indices[0]),
            })
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
