use text_analysis::semantic::{
    analyze_corpus_temporal, SemanticAnalysisOptions, SemanticCorpusAnalysisOptions,
    SemanticCorpusItem, SemanticCorpusTemporalAnalysisOptions, SemanticCorpusTemporalWindow,
    SemanticTemporalChangeKind,
};

fn strict_options(target: Option<&str>) -> SemanticCorpusTemporalAnalysisOptions {
    SemanticCorpusTemporalAnalysisOptions {
        corpus: SemanticCorpusAnalysisOptions {
            semantic: SemanticAnalysisOptions {
                neighbors_per_unit: 2,
                neighbor_threshold: 0.80,
                cluster_threshold: 0.90,
                ..SemanticAnalysisOptions::default()
            },
            top_terms: 8,
        },
        word_sense_target: target.map(ToString::to_string),
    }
}

fn cluster_id_for_text(
    report: &text_analysis::semantic::SemanticCorpusTemporalReport,
    text: &str,
) -> String {
    report
        .corpus
        .concepts
        .iter()
        .find(|concept| concept.representative.text == text)
        .expect("concept representative")
        .cluster_id
        .clone()
}

#[test]
fn temporal_windows_project_global_concepts_and_classify_structural_changes() {
    let stable = "Semantic search improves retrieval.";
    let returning = "Tomatoes grow in soil.";
    let emerging = "River banks contain sediment.";
    let items = [
        SemanticCorpusItem::new("w1-stable", Some("Alice"), stable)
            .with_source("notes/w1-stable.txt")
            .with_timestamp_millis(1_000),
        SemanticCorpusItem::new("w1-returning", Some("Alice"), returning)
            .with_source("notes/w1-returning.txt")
            .with_timestamp_millis(1_100),
        SemanticCorpusItem::new("w2-stable", Some("Alice"), stable)
            .with_source("notes/w2-stable.txt")
            .with_timestamp_millis(2_000),
        SemanticCorpusItem::new("w2-emerging", Some("Alice"), emerging)
            .with_source("notes/w2-emerging.txt")
            .with_timestamp_millis(2_100),
        SemanticCorpusItem::new("w3-stable", Some("Alice"), stable)
            .with_source("notes/w3-stable.txt")
            .with_timestamp_millis(3_000),
        SemanticCorpusItem::new("w3-returning", Some("Alice"), returning)
            .with_source("notes/w3-returning.txt")
            .with_timestamp_millis(3_100),
        SemanticCorpusItem::new("w3-emerging", Some("Alice"), emerging)
            .with_source("notes/w3-emerging.txt")
            .with_timestamp_millis(3_200),
    ];
    let w1 = ["w1-stable", "w1-returning"];
    let w2 = ["w2-stable", "w2-emerging"];
    let w3 = ["w3-stable", "w3-returning", "w3-emerging"];
    let windows = [
        SemanticCorpusTemporalWindow::new("early", &w1),
        SemanticCorpusTemporalWindow::new("middle", &w2),
        SemanticCorpusTemporalWindow::new("late", &w3),
    ];

    let report = analyze_corpus_temporal(&items, &windows, &strict_options(None)).unwrap();

    assert_eq!(report.corpus.semantic.clusters.len(), 3);
    assert_eq!(report.windows.len(), 3);
    assert_eq!(report.windows[0].item_count, 2);
    assert_eq!(report.windows[1].item_count, 2);
    assert_eq!(report.windows[2].item_count, 3);

    let stable_id = cluster_id_for_text(&report, stable);
    let returning_id = cluster_id_for_text(&report, returning);
    let emerging_id = cluster_id_for_text(&report, emerging);

    let change = |from: &str, to: &str, cluster_id: &str| {
        report
            .concept_changes
            .iter()
            .find(|change| {
                change.from_window_id == from
                    && change.to_window_id == to
                    && change.cluster_id == cluster_id
            })
            .expect("temporal concept change")
    };

    assert_eq!(
        change("early", "middle", &stable_id).kind,
        SemanticTemporalChangeKind::Persisting
    );
    assert_eq!(
        change("early", "middle", &returning_id).kind,
        SemanticTemporalChangeKind::Disappeared
    );
    let emerged = change("early", "middle", &emerging_id);
    assert_eq!(emerged.kind, SemanticTemporalChangeKind::Emerging);
    assert_eq!(
        emerged
            .current_evidence
            .as_ref()
            .and_then(|evidence| evidence.source.as_deref()),
        Some("notes/w2-emerging.txt")
    );
    assert_eq!(
        emerged
            .current_evidence
            .as_ref()
            .and_then(|evidence| evidence.timestamp_millis),
        Some(2_100)
    );

    assert_eq!(
        change("middle", "late", &returning_id).kind,
        SemanticTemporalChangeKind::Reentered
    );
    assert_eq!(
        change("middle", "late", &stable_id).kind,
        SemanticTemporalChangeKind::Declined
    );
}

#[test]
fn empty_window_is_missing_evidence_instead_of_an_observed_zero() {
    let text = "Semantic search improves retrieval.";
    let items = [
        SemanticCorpusItem::new("early", Some("Alice"), text).with_timestamp_millis(1_000),
        SemanticCorpusItem::new("late", Some("Alice"), text).with_timestamp_millis(3_000),
    ];
    let early = ["early"];
    let none: [&str; 0] = [];
    let late = ["late"];
    let windows = [
        SemanticCorpusTemporalWindow::new("early", &early),
        SemanticCorpusTemporalWindow::new("missing", &none),
        SemanticCorpusTemporalWindow::new("late", &late),
    ];

    let report = analyze_corpus_temporal(&items, &windows, &strict_options(None)).unwrap();
    let cluster_id = cluster_id_for_text(&report, text);

    let first = report
        .concept_changes
        .iter()
        .find(|change| {
            change.from_window_id == "early"
                && change.to_window_id == "missing"
                && change.cluster_id == cluster_id
        })
        .expect("change into missing window");
    assert_eq!(first.kind, SemanticTemporalChangeKind::MissingEvidence);
    assert_eq!(first.previous_unit_count, Some(1));
    assert_eq!(first.current_unit_count, None);

    let second = report
        .concept_changes
        .iter()
        .find(|change| {
            change.from_window_id == "missing"
                && change.to_window_id == "late"
                && change.cluster_id == cluster_id
        })
        .expect("change out of missing window");
    assert_eq!(second.kind, SemanticTemporalChangeKind::MissingEvidence);
    assert_eq!(second.previous_unit_count, None);
    assert_eq!(second.current_unit_count, Some(1));
}

#[test]
fn temporal_word_senses_reuse_global_sense_identity_across_windows() {
    let river = "The river bank borders the water.";
    let finance = "The bank approved the loan.";
    let items = [
        SemanticCorpusItem::new("river-early", Some("Alice"), river)
            .with_source("notes/river-early.txt")
            .with_timestamp_millis(1_000),
        SemanticCorpusItem::new("river-early-2", Some("Alice"), river)
            .with_source("notes/river-early-2.txt")
            .with_timestamp_millis(1_100),
        SemanticCorpusItem::new("finance-middle", Some("Bob"), finance)
            .with_source("notes/finance-middle.txt")
            .with_timestamp_millis(2_000),
        SemanticCorpusItem::new("finance-middle-2", Some("Bob"), finance)
            .with_source("notes/finance-middle-2.txt")
            .with_timestamp_millis(2_100),
        SemanticCorpusItem::new("river-late", Some("Alice"), river)
            .with_source("notes/river-late.txt")
            .with_timestamp_millis(3_000),
        SemanticCorpusItem::new("river-late-2", Some("Alice"), river)
            .with_source("notes/river-late-2.txt")
            .with_timestamp_millis(3_100),
    ];
    let early = ["river-early", "river-early-2"];
    let middle = ["finance-middle", "finance-middle-2"];
    let late = ["river-late", "river-late-2"];
    let windows = [
        SemanticCorpusTemporalWindow::new("early", &early),
        SemanticCorpusTemporalWindow::new("middle", &middle),
        SemanticCorpusTemporalWindow::new("late", &late),
    ];

    let report = analyze_corpus_temporal(&items, &windows, &strict_options(Some("bank"))).unwrap();
    let word_senses = report.word_senses.as_ref().expect("global word senses");
    assert_eq!(word_senses.senses.len(), 2);

    let river_id = word_senses
        .senses
        .iter()
        .find(|sense| sense.representative.context_text == river)
        .expect("river sense")
        .cluster_id
        .clone();
    let finance_id = word_senses
        .senses
        .iter()
        .find(|sense| sense.representative.context_text == finance)
        .expect("finance sense")
        .cluster_id
        .clone();

    let change = |from: &str, to: &str, cluster_id: &str| {
        report
            .sense_changes
            .iter()
            .find(|change| {
                change.from_window_id == from
                    && change.to_window_id == to
                    && change.cluster_id == cluster_id
            })
            .expect("temporal sense change")
    };

    assert_eq!(
        change("early", "middle", &river_id).kind,
        SemanticTemporalChangeKind::Disappeared
    );
    assert_eq!(
        change("early", "middle", &finance_id).kind,
        SemanticTemporalChangeKind::Emerging
    );
    let returned = change("middle", "late", &river_id);
    assert_eq!(returned.kind, SemanticTemporalChangeKind::Reentered);
    assert_eq!(
        returned
            .current_evidence
            .as_ref()
            .and_then(|evidence| evidence.source.as_deref()),
        Some("notes/river-late.txt")
    );
    assert_eq!(
        change("middle", "late", &finance_id).kind,
        SemanticTemporalChangeKind::Disappeared
    );
}

#[test]
fn temporal_windows_reject_unknown_items() {
    let items = [
        SemanticCorpusItem::new("known", Some("Alice"), "Known text."),
        SemanticCorpusItem::new("uncovered", Some("Alice"), "Other text."),
    ];
    let first = ["known"];
    let unknown = ["missing"];
    let windows = [
        SemanticCorpusTemporalWindow::new("one", &first),
        SemanticCorpusTemporalWindow::new("two", &unknown),
    ];

    let error = analyze_corpus_temporal(&items, &windows, &strict_options(None)).unwrap_err();
    assert!(error.to_string().contains("references unknown item id"));
}
