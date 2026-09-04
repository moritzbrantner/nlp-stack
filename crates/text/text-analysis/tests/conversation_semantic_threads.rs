use text_analysis::semantic::{
    analyze_conversation_semantics, ConversationTurn, SemanticAnalysisOptions,
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
fn alternating_concepts_form_reentered_threads_and_interleaving_evidence() {
    let turns = [
        ConversationTurn::new(
            "turn-1",
            Some("Alice"),
            "Semantic search improves retrieval.",
        ),
        ConversationTurn::new("turn-2", Some("Bob"), "Tomatoes grow in soil."),
        ConversationTurn::new(
            "turn-3",
            Some("Alice"),
            "Semantic search improves retrieval.",
        ),
        ConversationTurn::new("turn-4", Some("Bob"), "Tomatoes grow in soil."),
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

    assert_eq!(dynamics.threads.len(), 2);
    let mut occurrence_counts = dynamics
        .threads
        .iter()
        .map(|thread| {
            (
                thread.occurrence_count,
                thread.segment_count,
                thread.reentry_count,
            )
        })
        .collect::<Vec<_>>();
    occurrence_counts.sort_unstable();
    assert_eq!(occurrence_counts, vec![(2, 2, 1), (3, 3, 2)]);
    assert!(dynamics
        .threads
        .iter()
        .flat_map(|thread| &thread.segments)
        .all(|segment| segment.unit_count == 1));

    assert_eq!(dynamics.thread_interleavings.len(), 1);
    let interleaving = &dynamics.thread_interleavings[0];
    assert_eq!(interleaving.alternation_count, 3);
    assert_eq!(interleaving.first_sequence_index, 0);
    assert_eq!(interleaving.last_sequence_index, 4);
    assert_ne!(interleaving.left_cluster_id, interleaving.right_cluster_id);
}

#[test]
fn contiguous_concept_stays_one_thread_segment() {
    let turns = [
        ConversationTurn::new(
            "turn-1",
            Some("Alice"),
            "Semantic search improves retrieval.",
        ),
        ConversationTurn::new("turn-2", Some("Bob"), "Semantic search improves retrieval."),
        ConversationTurn::new(
            "turn-3",
            Some("Alice"),
            "Semantic search improves retrieval.",
        ),
    ];
    let report = analyze_conversation_semantics(&turns, &strict_options()).unwrap();
    let dynamics = report
        .conversation_dynamics
        .as_ref()
        .expect("conversation dynamics");

    assert_eq!(dynamics.threads.len(), 1);
    let thread = &dynamics.threads[0];
    assert_eq!(thread.occurrence_count, 3);
    assert_eq!(thread.segment_count, 1);
    assert_eq!(thread.reentry_count, 0);
    assert_eq!(thread.segments.len(), 1);
    assert_eq!(thread.segments[0].start_sequence_index, 0);
    assert_eq!(thread.segments[0].end_sequence_index, 2);
    assert_eq!(thread.segments[0].unit_count, 3);
    assert!(dynamics.thread_interleavings.is_empty());
}
