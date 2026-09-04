use text_analysis::semantic::{
    analyze_word_senses, SemanticAnalysisOptions, SemanticCorpusItem,
    SemanticWordSenseAnalysisOptions,
};

fn strict_options() -> SemanticWordSenseAnalysisOptions {
    SemanticWordSenseAnalysisOptions {
        semantic: SemanticAnalysisOptions {
            neighbors_per_unit: 2,
            neighbor_threshold: 0.80,
            cluster_threshold: 0.90,
            ..SemanticAnalysisOptions::default()
        },
    }
}

#[test]
fn word_senses_cluster_distinct_usage_contexts_and_keep_exact_occurrences() {
    let river = "The river bank borders the water.";
    let finance = "The bank approved the loan.";
    let items = [
        SemanticCorpusItem::new("river-1", Some("Alice"), river).with_source("notes/river-1.txt"),
        SemanticCorpusItem::new("river-2", Some("Alice"), river).with_source("notes/river-2.txt"),
        SemanticCorpusItem::new("finance-1", Some("Bob"), finance)
            .with_source("notes/finance-1.txt"),
        SemanticCorpusItem::new("finance-2", Some("Bob"), finance)
            .with_source("notes/finance-2.txt"),
    ];

    let report = analyze_word_senses("BANK", &items, &strict_options()).unwrap();

    assert_eq!(report.target, "BANK");
    assert_eq!(report.normalized_target, "bank");
    assert_eq!(report.occurrence_count, 4);
    assert_eq!(report.source_item_count, 4);
    assert_eq!(report.author_count, 2);
    assert_eq!(report.senses.len(), 2);
    assert!(report.semantic.is_some());

    for sense in &report.senses {
        assert_eq!(sense.occurrence_count, 2);
        assert_eq!(sense.source_item_count, 2);
        assert_eq!(sense.author_count, 1);
        assert_eq!(sense.occurrences.len(), 2);
        for occurrence in &sense.occurrences {
            let source_text = if occurrence.source_id.starts_with("river") {
                river
            } else {
                finance
            };
            assert_eq!(
                &source_text[occurrence.occurrence_span.byte_start..occurrence.occurrence_span.byte_end],
                "bank"
            );
            assert_eq!(
                &source_text[occurrence.context_span.byte_start..occurrence.context_span.byte_end],
                occurrence.context_text
            );
        }
    }

    let river_sense = report
        .senses
        .iter()
        .find(|sense| sense.representative.context_text == river)
        .expect("river-bank sense");
    assert_eq!(river_sense.authors, vec!["Alice"]);
    assert_eq!(
        river_sense.representative.source.as_deref(),
        Some("notes/river-1.txt")
    );

    let finance_sense = report
        .senses
        .iter()
        .find(|sense| sense.representative.context_text == finance)
        .expect("financial-bank sense");
    assert_eq!(finance_sense.authors, vec!["Bob"]);
}

#[test]
fn word_senses_return_empty_evidence_when_target_is_absent() {
    let items = [SemanticCorpusItem::new(
        "doc-1",
        Some("Alice"),
        "Semantic retrieval finds related passages.",
    )];

    let report = analyze_word_senses("bank", &items, &strict_options()).unwrap();

    assert_eq!(report.occurrence_count, 0);
    assert_eq!(report.source_item_count, 0);
    assert_eq!(report.author_count, 0);
    assert!(report.senses.is_empty());
    assert!(report.semantic.is_none());
}

#[test]
fn word_senses_require_one_normalized_target_token() {
    let items = [SemanticCorpusItem::new(
        "doc-1",
        Some("Alice"),
        "The river bank borders the water.",
    )];

    let error = analyze_word_senses("river bank", &items, &strict_options()).unwrap_err();
    assert!(error
        .to_string()
        .contains("word-sense target must normalize to exactly one token"));
}
