use text_core::TextDocument;

use super::{
    analyze_conversation_semantics, analyze_document_semantics, ConversationTurn,
    SemanticAnalysisOptions, SemanticUnitKind,
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
}
