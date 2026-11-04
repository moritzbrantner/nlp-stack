use text_core::TextDocument;
use text_embeddings::testing::HashedFixtureEmbedder;
use text_linguistics::{LinguisticAnalysisOptions, TextNlpConfig, TextNlpPipeline};

/// The hashed fixture stands in for an embedding model so the test is deterministic.
fn analyze_document_semantics(
    document: &TextDocument<'_>,
    options: &SemanticAnalysisOptions,
) -> text_core::Result<super::SemanticAnalysisReport> {
    analyze_document_semantics_with(document, options, &HashedFixtureEmbedder::default())
}

/// The hashed fixture stands in for an embedding model so the test is deterministic.
fn analyze_conversation_semantics(
    turns: &[ConversationTurn<'_>],
    options: &SemanticAnalysisOptions,
) -> text_core::Result<super::SemanticAnalysisReport> {
    analyze_conversation_semantics_with(turns, options, &HashedFixtureEmbedder::default())
}

use super::{
    analyze_conversation_semantics_with, analyze_document_semantics_with,
    compare_semantic_neighborhoods, compose_linguistic_semantic_graph, interpret_semantic_report,
    ConversationTurn, SemanticAnalysisOptions, SemanticConceptInterpretationContent,
    SemanticConceptInterpretationRequest, SemanticGraphEdgeKind, SemanticGraphNodeKind,
    SemanticInterpretationBackend, SemanticInterpretationMetadata, SemanticUnitKind,
};

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
    assert!(report.conversation_dynamics.is_none());
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
    assert!(report.neighbors.iter().any(|edge| edge.similarity > 0.99));
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
        ConversationTurn::new(
            "turn-1",
            Some("Alice"),
            "Semantic search improves retrieval.",
        ),
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
    assert_eq!(
        alice
            .concepts
            .iter()
            .map(|concept| concept.unit_count)
            .sum::<usize>(),
        2
    );
    assert!(report.conversation_dynamics.is_some());
}

#[test]
fn foundation_index_matches_exact_semantic_neighborhood_fixture() {
    let text = "Semantic search improves retrieval. Semantic search improves retrieval. Tomatoes grow in soil. Semantic search improves retrieval.";
    let document = TextDocument::new("doc-scale", text);
    let report = analyze_document_semantics(&document, &strict_options()).unwrap();
    let evidence = compare_semantic_neighborhoods(&report, 2, 0.80).unwrap();

    assert_eq!(evidence.primary_unit_count, 4);
    assert_eq!(evidence.exact_only_edge_count, 0);
    assert_eq!(evidence.indexed_only_edge_count, 0);
    assert_eq!(evidence.shared_edge_count, evidence.exact_edges.len());
    assert!(evidence.max_similarity_delta < 0.000_01);
}

#[test]
fn linguistic_graph_joins_existing_linguistic_evidence_to_semantic_units() {
    let text = "Alice visited Berlin. She presented the roadmap.";
    let document = TextDocument::new("doc-graph", text);
    let semantic = analyze_document_semantics(&document, &strict_options()).unwrap();
    // A fixed labeler stands in for the NER model.
    struct FixtureNer;
    impl text_linguistics::SequenceLabeler for FixtureNer {
        fn label_text(
            &mut self,
            _text: &str,
        ) -> text_core::Result<Vec<text_linguistics::RawPrediction>> {
            Ok([("B-PER", 0, 5), ("B-LOC", 14, 20)]
                .into_iter()
                .map(|(label, start, end)| {
                    let mut prediction = text_linguistics::RawPrediction {
                        kind: Some("token".to_string()),
                        label: Some(label.to_string()),
                        score: Some(0.9),
                        ..text_linguistics::RawPrediction::default()
                    };
                    prediction
                        .attributes
                        .insert("byte_start".to_string(), start.to_string());
                    prediction
                        .attributes
                        .insert("byte_end".to_string(), end.to_string());
                    prediction
                })
                .collect())
        }
    }
    let linguistic = TextNlpPipeline::new(TextNlpConfig::from_options(
        LinguisticAnalysisOptions::without_models(),
    ))
    .analyze_text_with_entity_labeler(text, &mut FixtureNer)
    .unwrap();
    let graph = compose_linguistic_semantic_graph(&semantic, &linguistic);

    assert!(graph
        .nodes
        .iter()
        .any(|node| node.kind == SemanticGraphNodeKind::Concept));
    assert!(graph
        .nodes
        .iter()
        .any(|node| node.kind == SemanticGraphNodeKind::EntityMention));
    assert!(graph
        .edges
        .iter()
        .any(|edge| edge.kind == SemanticGraphEdgeKind::ConceptMembership));
    assert!(graph
        .edges
        .iter()
        .any(|edge| edge.kind == SemanticGraphEdgeKind::UnitContainsMention));
}

#[test]
fn conversation_dynamics_track_adoption_handoffs_and_recurring_concepts() {
    let turns = [
        ConversationTurn::new(
            "turn-1",
            Some("Alice"),
            "Semantic search improves retrieval.",
        ),
        ConversationTurn::new("turn-2", Some("Bob"), "Tomatoes grow in soil."),
        ConversationTurn::new("turn-3", Some("Bob"), "Semantic search improves retrieval."),
        ConversationTurn::new("turn-4", Some("Alice"), "Tomatoes grow in soil."),
        ConversationTurn::new(
            "turn-5",
            Some("Alice"),
            "Semantic search improves retrieval.",
        ),
    ];
    let report = analyze_conversation_semantics(&turns, &strict_options()).unwrap();
    let dynamics = report
        .conversation_dynamics
        .as_ref()
        .expect("conversation dynamics");

    assert!(!dynamics.speaker_pairs.is_empty());
    assert!(!dynamics.adoptions.is_empty());
    assert!(!dynamics.handoffs.is_empty());
    assert!(dynamics
        .recurring_concepts
        .iter()
        .any(|concept| concept.non_adjacent_return_count > 0));
}

#[derive(Debug)]
struct FixtureInterpreter;

impl SemanticInterpretationBackend for FixtureInterpreter {
    fn metadata(&self) -> SemanticInterpretationMetadata {
        SemanticInterpretationMetadata {
            backend: "fixture".to_string(),
            model: Some("deterministic-test".to_string()),
        }
    }

    fn interpret_concept(
        &self,
        request: &SemanticConceptInterpretationRequest<'_>,
    ) -> text_core::Result<SemanticConceptInterpretationContent> {
        Ok(SemanticConceptInterpretationContent {
            label: Some(format!("{} members", request.members.len())),
            summary: Some(request.representative.text.clone()),
            confidence: Some(1.0),
        })
    }
}

#[test]
fn interpretation_is_annotation_over_deterministic_clusters() {
    let text = "Semantic search improves retrieval. Semantic search improves retrieval. Tomatoes grow in soil.";
    let document = TextDocument::new("doc-interpret", text);
    let report = analyze_document_semantics(&document, &strict_options()).unwrap();
    let interpreted = interpret_semantic_report(&report, &FixtureInterpreter).unwrap();

    assert_eq!(interpreted.concepts.len(), report.clusters.len());
    assert!(interpreted
        .concepts
        .iter()
        .all(|concept| concept.metadata.backend == "fixture"));
    assert!(interpreted.concepts.iter().all(|concept| report
        .clusters
        .iter()
        .any(|cluster| cluster.id == concept.cluster_id)));
}
