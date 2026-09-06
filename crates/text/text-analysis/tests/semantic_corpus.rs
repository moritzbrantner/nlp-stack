use text_analysis::semantic::{
    analyze_corpus_semantics, analyze_corpus_semantics_with, SemanticAnalysisOptions,
    SemanticCorpusAnalysisOptions, SemanticCorpusItem, SemanticUnitKind,
};
use text_embeddings::{DenseVector, TextEmbeddingBackend};

fn strict_options() -> SemanticCorpusAnalysisOptions {
    SemanticCorpusAnalysisOptions {
        semantic: SemanticAnalysisOptions {
            neighbors_per_unit: 2,
            neighbor_threshold: 0.80,
            cluster_threshold: 0.90,
            ..SemanticAnalysisOptions::default()
        },
        top_terms: 8,
        min_concept_units: 1,
    }
}

#[test]
fn corpus_profiles_aggregate_lexical_and_semantic_evidence_by_author() {
    let items = [
        SemanticCorpusItem::new(
            "alice-1",
            Some("Alice"),
            "Semantic search improves retrieval.",
        ),
        SemanticCorpusItem::new(
            "alice-2",
            Some("Alice"),
            "Semantic search improves retrieval.",
        ),
        SemanticCorpusItem::new("bob-1", Some("Bob"), "Tomatoes grow in soil."),
        SemanticCorpusItem::new("bob-2", Some("Bob"), "Tomatoes grow in soil."),
    ];

    let report = analyze_corpus_semantics(&items, &strict_options()).unwrap();

    assert_eq!(report.item_count, 4);
    assert_eq!(report.author_count, 2);
    assert_eq!(report.lexical.word_count, 16);
    assert_eq!(report.non_concept_unit_count, 0);
    assert_eq!(
        report.semantic.primary_unit_kind,
        SemanticUnitKind::Sentence
    );
    assert_eq!(report.semantic.timeline.len(), 4);
    assert_eq!(report.semantic.clusters.len(), 2);
    assert_eq!(
        report.semantic.embedding_model.model_name,
        "hashed-tfidf-sentence-baseline"
    );
    assert_eq!(report.semantic.embedding_model.dimensions, 512);

    let alice = report
        .authors
        .iter()
        .find(|profile| profile.author == "Alice")
        .expect("Alice profile");
    assert_eq!(alice.item_count, 2);
    assert_eq!(alice.semantic_unit_count, 2);
    assert_eq!(alice.lexical.word_count, 8);
    assert_eq!(alice.concepts.len(), 1);
    assert_eq!(alice.concepts[0].unit_count, 2);
    assert!((alice.concepts[0].share - 1.0).abs() < f32::EPSILON);

    assert!(report.concepts.iter().all(|concept| {
        concept.member_unit_count == 2
            && concept.source_item_count == 2
            && concept.author_count == 1
            && !concept.key_terms.is_empty()
            && !concept.label.is_empty()
            && concept.coherence > 0.99
    }));
}

#[test]
fn corpus_concept_representatives_retain_item_source_and_span_provenance() {
    let text = "Semantic search improves retrieval. Tomatoes grow in soil.";
    let items = [SemanticCorpusItem::new("letter-1", Some("Alice"), text)
        .with_source("letters/1.txt")
        .with_timestamp_millis(1_700_000_000_000)];

    let report = analyze_corpus_semantics(&items, &strict_options()).unwrap();

    assert_eq!(report.sources.len(), 1);
    assert_eq!(report.sources[0].source.as_deref(), Some("letters/1.txt"));
    assert_eq!(report.sources[0].timestamp_millis, Some(1_700_000_000_000));
    assert_eq!(report.concepts.len(), 2);

    for concept in &report.concepts {
        let passage = &concept.representative;
        assert_eq!(passage.source_id, "letter-1");
        assert_eq!(passage.author.as_deref(), Some("Alice"));
        assert_eq!(passage.source.as_deref(), Some("letters/1.txt"));
        assert_eq!(passage.timestamp_millis, Some(1_700_000_000_000));
        assert_eq!(
            &text[passage.span.byte_start..passage.span.byte_end],
            passage.text
        );
    }
}

#[test]
fn corpus_clustering_does_not_chain_through_one_bridge_unit() {
    let items = [
        SemanticCorpusItem::new("a", None, "Alpha."),
        SemanticCorpusItem::new("bridge", None, "Bridge."),
        SemanticCorpusItem::new("c", None, "Gamma."),
    ];
    let options = SemanticCorpusAnalysisOptions {
        semantic: SemanticAnalysisOptions {
            neighbors_per_unit: 2,
            neighbor_threshold: 0.0,
            cluster_threshold: 0.80,
            ..SemanticAnalysisOptions::default()
        },
        top_terms: 8,
        min_concept_units: 1,
    };

    let report = analyze_corpus_semantics_with(&items, &options, &ChainEmbedder).unwrap();

    assert_eq!(report.semantic.clusters.len(), 2);
    assert_eq!(report.semantic.clusters[0].member_unit_ids.len(), 2);
    assert_eq!(report.semantic.clusters[1].member_unit_ids.len(), 1);
}

#[test]
fn default_minimum_support_keeps_singletons_out_of_corpus_concepts() {
    let items = [
        SemanticCorpusItem::new("a", None, "Alpha."),
        SemanticCorpusItem::new("bridge", None, "Bridge."),
        SemanticCorpusItem::new("c", None, "Gamma."),
    ];
    let options = SemanticCorpusAnalysisOptions {
        semantic: SemanticAnalysisOptions {
            neighbors_per_unit: 2,
            neighbor_threshold: 0.0,
            cluster_threshold: 0.80,
            ..SemanticAnalysisOptions::default()
        },
        top_terms: 8,
        ..SemanticCorpusAnalysisOptions::default()
    };

    let report = analyze_corpus_semantics_with(&items, &options, &ChainEmbedder).unwrap();

    assert_eq!(report.concepts.len(), 1);
    assert_eq!(report.concepts[0].member_unit_count, 2);
    assert_eq!(report.non_concept_unit_count, 1);
}

#[test]
fn corpus_rejects_duplicate_item_identity() {
    let items = [
        SemanticCorpusItem::new("same", Some("Alice"), "First text."),
        SemanticCorpusItem::new("same", Some("Alice"), "Second text."),
    ];

    let error = analyze_corpus_semantics(&items, &strict_options()).unwrap_err();
    assert!(error
        .to_string()
        .contains("duplicate semantic corpus item id"));
}

struct ChainEmbedder;

impl TextEmbeddingBackend for ChainEmbedder {
    fn embed_text(&self, text: &str) -> text_core::Result<DenseVector> {
        let values = match text {
            "Alpha." => vec![1.0, 0.0],
            "Bridge." => vec![0.809_017, 0.587_785],
            "Gamma." => vec![0.309_017, 0.951_057],
            _ => vec![1.0, 0.0],
        };
        DenseVector::new(values)
    }
}
