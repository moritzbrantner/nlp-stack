use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use text_core::TextSpan;
use text_linguistics::LinguisticAnalysis;

use super::{SemanticAnalysisReport, SemanticUnit};

/// Node kinds in the composed linguistic semantic graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SemanticGraphNodeKind {
    Unit,
    Concept,
    EntityMention,
    CanonicalEntity,
    CoreferenceCluster,
    CoreferenceMention,
    Event,
    EventArgument,
    Relation,
    RelationEndpoint,
    Discourse,
    Topic,
}

/// Edge kinds joining embedding-derived and linguistic semantic evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SemanticGraphEdgeKind {
    SemanticNeighbor,
    ConceptMembership,
    UnitContainsMention,
    MentionCanonical,
    MentionCoreference,
    UnitContainsEvent,
    EventArgument,
    UnitContainsRelation,
    RelationSubject,
    RelationObject,
    ResolvesToCanonical,
    UnitContainsDiscourse,
    DiscourseTransition,
    UnitTopic,
}

/// One typed node in a composed semantic graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticGraphNode {
    pub id: String,
    pub kind: SemanticGraphNodeKind,
    pub label: String,
    pub span: Option<TextSpan>,
    pub sequence_index: Option<usize>,
    pub confidence: Option<f32>,
}

/// One typed edge in a composed semantic graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticGraphEdge {
    pub source_id: String,
    pub target_id: String,
    pub kind: SemanticGraphEdgeKind,
    pub label: Option<String>,
    pub weight: Option<f32>,
}

/// Graph joining the deterministic semantic map to existing `text-linguistics`
/// outputs without moving linguistic extraction into `text-analysis`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticLinguisticGraph {
    pub nodes: Vec<SemanticGraphNode>,
    pub edges: Vec<SemanticGraphEdge>,
}

/// Composes an already-built semantic map with an already-built linguistic
/// analysis. The function performs only deterministic graph projection.
pub fn compose_linguistic_semantic_graph(
    report: &SemanticAnalysisReport,
    analysis: &LinguisticAnalysis,
) -> SemanticLinguisticGraph {
    let primary_units = report
        .units
        .iter()
        .filter(|unit| unit.kind == report.primary_unit_kind)
        .collect::<Vec<_>>();
    let unit_by_sequence = primary_units
        .iter()
        .map(|unit| (unit.sequence_index, *unit))
        .collect::<BTreeMap<_, _>>();

    let mut nodes = primary_units
        .iter()
        .map(|unit| SemanticGraphNode {
            id: unit.id.clone(),
            kind: SemanticGraphNodeKind::Unit,
            label: unit.text.clone(),
            span: Some(unit.span),
            sequence_index: Some(unit.sequence_index),
            confidence: None,
        })
        .collect::<Vec<_>>();
    let mut edges = report
        .neighbors
        .iter()
        .map(|neighbor| SemanticGraphEdge {
            source_id: neighbor.source_unit_id.clone(),
            target_id: neighbor.target_unit_id.clone(),
            kind: SemanticGraphEdgeKind::SemanticNeighbor,
            label: None,
            weight: Some(neighbor.similarity),
        })
        .collect::<Vec<_>>();

    for cluster in &report.clusters {
        let concept_id = format!("concept:{}", cluster.id);
        nodes.push(SemanticGraphNode {
            id: concept_id.clone(),
            kind: SemanticGraphNodeKind::Concept,
            label: cluster.representative_text.clone(),
            span: None,
            sequence_index: None,
            confidence: Some(cluster.mean_similarity),
        });
        for member in &cluster.member_unit_ids {
            edges.push(SemanticGraphEdge {
                source_id: member.clone(),
                target_id: concept_id.clone(),
                kind: SemanticGraphEdgeKind::ConceptMembership,
                label: None,
                weight: None,
            });
        }
    }

    let mut mention_node_ids = BTreeMap::<String, String>::new();
    for entity in &analysis.entities {
        let node_id = format!("mention:{}", entity.id);
        mention_node_ids.insert(entity.id.clone(), node_id.clone());
        nodes.push(SemanticGraphNode {
            id: node_id.clone(),
            kind: SemanticGraphNodeKind::EntityMention,
            label: entity.mention.text.clone(),
            span: Some(entity.mention.span),
            sequence_index: Some(entity.sentence_index),
            confidence: Some(entity.confidence),
        });
        if let Some(unit) = unit_by_sequence.get(&entity.sentence_index) {
            edges.push(SemanticGraphEdge {
                source_id: unit.id.clone(),
                target_id: node_id,
                kind: SemanticGraphEdgeKind::UnitContainsMention,
                label: Some(format!("{:?}", entity.entity_type)),
                weight: Some(entity.confidence),
            });
        }
    }

    let mut canonical_by_name = BTreeMap::<String, String>::new();
    for canonical in &analysis.canonical_entities {
        let node_id = format!("canonical:{}", canonical.id);
        canonical_by_name.insert(canonical.canonical_name.to_lowercase(), node_id.clone());
        for alias in &canonical.aliases {
            canonical_by_name.insert(alias.to_lowercase(), node_id.clone());
        }
        nodes.push(SemanticGraphNode {
            id: node_id.clone(),
            kind: SemanticGraphNodeKind::CanonicalEntity,
            label: canonical.canonical_name.clone(),
            span: None,
            sequence_index: None,
            confidence: None,
        });
        for mention in &canonical.mentions {
            if let Some(mention_id) = mention_node_ids.get(&mention.id) {
                edges.push(SemanticGraphEdge {
                    source_id: mention_id.clone(),
                    target_id: node_id.clone(),
                    kind: SemanticGraphEdgeKind::MentionCanonical,
                    label: Some(format!("{:?}", canonical.entity_type)),
                    weight: Some(mention.confidence),
                });
            }
        }
    }

    for cluster in &analysis.coreference {
        let cluster_id = format!("coref:{}", cluster.id);
        nodes.push(SemanticGraphNode {
            id: cluster_id.clone(),
            kind: SemanticGraphNodeKind::CoreferenceCluster,
            label: cluster.canonical_text.clone(),
            span: None,
            sequence_index: None,
            confidence: None,
        });
        for (mention_index, mention) in cluster.mentions.iter().enumerate() {
            let mention_id = format!("{}:mention:{}", cluster_id, mention_index);
            let span = token_range_span(analysis, mention.token_start, mention.token_end);
            nodes.push(SemanticGraphNode {
                id: mention_id.clone(),
                kind: SemanticGraphNodeKind::CoreferenceMention,
                label: mention.text.clone(),
                span,
                sequence_index: Some(mention.sentence_index),
                confidence: Some(mention.confidence),
            });
            edges.push(SemanticGraphEdge {
                source_id: mention_id.clone(),
                target_id: cluster_id.clone(),
                kind: SemanticGraphEdgeKind::MentionCoreference,
                label: mention.entity_type.map(|kind| format!("{:?}", kind)),
                weight: Some(mention.confidence),
            });
            if let Some(unit) = unit_by_sequence.get(&mention.sentence_index) {
                edges.push(SemanticGraphEdge {
                    source_id: unit.id.clone(),
                    target_id: mention_id,
                    kind: SemanticGraphEdgeKind::UnitContainsMention,
                    label: Some("coreference".to_string()),
                    weight: Some(mention.confidence),
                });
            }
        }
    }

    for (event_index, event) in analysis.events.iter().enumerate() {
        let event_id = format!("event:{}", event_index);
        nodes.push(SemanticGraphNode {
            id: event_id.clone(),
            kind: SemanticGraphNodeKind::Event,
            label: event.predicate.clone(),
            span: None,
            sequence_index: Some(event.sentence_index),
            confidence: Some(event.confidence),
        });
        if let Some(unit) = unit_by_sequence.get(&event.sentence_index) {
            edges.push(SemanticGraphEdge {
                source_id: unit.id.clone(),
                target_id: event_id.clone(),
                kind: SemanticGraphEdgeKind::UnitContainsEvent,
                label: Some(format!("{:?}", event.relation_type)),
                weight: Some(event.confidence),
            });
        }
        for (argument_index, argument) in event.arguments.iter().enumerate() {
            let argument_id = format!("{}:argument:{}", event_id, argument_index);
            nodes.push(SemanticGraphNode {
                id: argument_id.clone(),
                kind: SemanticGraphNodeKind::EventArgument,
                label: argument.text.clone(),
                span: None,
                sequence_index: Some(event.sentence_index),
                confidence: Some(argument.confidence),
            });
            edges.push(SemanticGraphEdge {
                source_id: event_id.clone(),
                target_id: argument_id.clone(),
                kind: SemanticGraphEdgeKind::EventArgument,
                label: Some(argument.role.clone()),
                weight: Some(argument.confidence),
            });
            if let Some(canonical_id) = canonical_by_name.get(&argument.text.to_lowercase()) {
                edges.push(SemanticGraphEdge {
                    source_id: argument_id,
                    target_id: canonical_id.clone(),
                    kind: SemanticGraphEdgeKind::ResolvesToCanonical,
                    label: None,
                    weight: Some(argument.confidence),
                });
            }
        }
    }

    for (relation_index, relation) in analysis.relations.iter().enumerate() {
        let relation_id = format!("relation:{}", relation_index);
        nodes.push(SemanticGraphNode {
            id: relation_id.clone(),
            kind: SemanticGraphNodeKind::Relation,
            label: format!(
                "{} {} {}",
                relation.subject, relation.relation, relation.object
            ),
            span: None,
            sequence_index: None,
            confidence: Some(relation.confidence),
        });
        if let Some(unit) =
            unit_containing_relation(&primary_units, &relation.subject, &relation.object)
        {
            edges.push(SemanticGraphEdge {
                source_id: unit.id.clone(),
                target_id: relation_id.clone(),
                kind: SemanticGraphEdgeKind::UnitContainsRelation,
                label: Some(format!("{:?}", relation.relation_type)),
                weight: Some(relation.confidence),
            });
        }
        add_relation_endpoint(
            &mut nodes,
            &mut edges,
            &canonical_by_name,
            &relation_id,
            "subject",
            &relation.subject,
            SemanticGraphEdgeKind::RelationSubject,
            relation.confidence,
        );
        add_relation_endpoint(
            &mut nodes,
            &mut edges,
            &canonical_by_name,
            &relation_id,
            "object",
            &relation.object,
            SemanticGraphEdgeKind::RelationObject,
            relation.confidence,
        );
    }

    for discourse in &analysis.discourse {
        let discourse_id = format!("discourse:{}", discourse.index);
        nodes.push(SemanticGraphNode {
            id: discourse_id.clone(),
            kind: SemanticGraphNodeKind::Discourse,
            label: discourse.text.clone(),
            span: None,
            sequence_index: Some(discourse.sentence_start),
            confidence: Some(discourse.confidence),
        });
        for sentence_index in discourse.sentence_start..discourse.sentence_end {
            if let Some(unit) = unit_by_sequence.get(&sentence_index) {
                edges.push(SemanticGraphEdge {
                    source_id: unit.id.clone(),
                    target_id: discourse_id.clone(),
                    kind: SemanticGraphEdgeKind::UnitContainsDiscourse,
                    label: Some(format!("{:?}", discourse.kind)),
                    weight: Some(discourse.confidence),
                });
            }
        }
        if discourse.index > 0 {
            if let Some(relation) = discourse.relation_to_previous {
                edges.push(SemanticGraphEdge {
                    source_id: format!("discourse:{}", discourse.index - 1),
                    target_id: discourse_id,
                    kind: SemanticGraphEdgeKind::DiscourseTransition,
                    label: Some(format!("{:?}", relation)),
                    weight: Some(discourse.confidence),
                });
            }
        }
    }

    for topic in &analysis.topics.clusters {
        let topic_id = format!("topic:{}", topic.id);
        nodes.push(SemanticGraphNode {
            id: topic_id.clone(),
            kind: SemanticGraphNodeKind::Topic,
            label: topic.descriptor.label.clone(),
            span: None,
            sequence_index: topic.sentence_indices.first().copied(),
            confidence: Some(topic.descriptor.score),
        });
        for sentence_index in &topic.sentence_indices {
            if let Some(unit) = unit_by_sequence.get(sentence_index) {
                edges.push(SemanticGraphEdge {
                    source_id: unit.id.clone(),
                    target_id: topic_id.clone(),
                    kind: SemanticGraphEdgeKind::UnitTopic,
                    label: Some(topic.descriptor.terms.join(", ")),
                    weight: Some(topic.descriptor.score),
                });
            }
        }
    }

    SemanticLinguisticGraph { nodes, edges }
}

fn token_range_span(
    analysis: &LinguisticAnalysis,
    token_start: usize,
    token_end: usize,
) -> Option<TextSpan> {
    if token_start >= token_end {
        return None;
    }
    let first = analysis.tokens.get(token_start)?;
    let last = analysis.tokens.get(token_end - 1)?;
    Some(TextSpan {
        byte_start: first.span.byte_start,
        byte_end: last.span.byte_end,
        char_start: first.span.char_start,
        char_end: last.span.char_end,
    })
}

fn unit_containing_relation<'a>(
    units: &[&'a SemanticUnit],
    subject: &str,
    object: &str,
) -> Option<&'a SemanticUnit> {
    let subject = subject.to_lowercase();
    let object = object.to_lowercase();
    units.iter().copied().find(|unit| {
        let text = unit.text.to_lowercase();
        text.contains(&subject) && text.contains(&object)
    })
}

#[allow(clippy::too_many_arguments)]
fn add_relation_endpoint(
    nodes: &mut Vec<SemanticGraphNode>,
    edges: &mut Vec<SemanticGraphEdge>,
    canonical_by_name: &BTreeMap<String, String>,
    relation_id: &str,
    role: &str,
    text: &str,
    edge_kind: SemanticGraphEdgeKind,
    confidence: f32,
) {
    let endpoint_id = format!("{}:{}", relation_id, role);
    nodes.push(SemanticGraphNode {
        id: endpoint_id.clone(),
        kind: SemanticGraphNodeKind::RelationEndpoint,
        label: text.to_string(),
        span: None,
        sequence_index: None,
        confidence: Some(confidence),
    });
    edges.push(SemanticGraphEdge {
        source_id: relation_id.to_string(),
        target_id: endpoint_id.clone(),
        kind: edge_kind,
        label: Some(role.to_string()),
        weight: Some(confidence),
    });
    if let Some(canonical_id) = canonical_by_name.get(&text.to_lowercase()) {
        edges.push(SemanticGraphEdge {
            source_id: endpoint_id,
            target_id: canonical_id.clone(),
            kind: SemanticGraphEdgeKind::ResolvesToCanonical,
            label: None,
            weight: Some(confidence),
        });
    }
}
