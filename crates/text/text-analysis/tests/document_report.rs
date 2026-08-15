use text_analysis::{
    analyze_text, document_from_text, AnalysisProfile, ClassificationDepth,
    DocumentAnalysisOptions, EmbeddingDepth, LinguisticDepth,
};

#[test]
fn deterministic_document_analysis_returns_all_default_sections() {
    let report = analyze_text(
        "doc-1",
        "Alice presented the tokenizer roadmap in Berlin. Rust crates analyze text.",
        &DocumentAnalysisOptions::default(),
    )
    .unwrap();

    assert_eq!(report.id, "doc-1");
    assert!(report.core.stats.basic.words > 0);
    assert!(!report.lexical.keywords.is_empty());
    assert!(!report.similarity.token_shingle_counts.is_empty());
    assert!(report.linguistic.is_some());
    assert!(report.embedding.is_some());
}

#[test]
fn empty_text_reports_embedding_diagnostic_without_panic() {
    let report = analyze_text("empty", "   ", &DocumentAnalysisOptions::default()).unwrap();
    assert!(report.embedding.is_none());
    assert!(report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "embedding_unavailable"));
}

#[test]
fn zero_ngram_size_is_invalid() {
    let options = DocumentAnalysisOptions {
        ngram_sizes: vec![0],
        ..DocumentAnalysisOptions::default()
    };
    let error = analyze_text("doc", "text", &options).unwrap_err();
    assert!(error.to_string().contains("ngram"));
}

#[test]
fn optional_sections_can_be_disabled() {
    let options = DocumentAnalysisOptions {
        linguistic_depth: LinguisticDepth::Off,
        embedding_depth: EmbeddingDepth::Off,
        ..DocumentAnalysisOptions::default()
    };
    let report = analyze_text("doc", "Rust crates analyze text.", &options).unwrap();
    assert!(report.linguistic.is_none());
    assert!(report.embedding.is_none());
}

#[test]
fn lexical_fallback_classification_is_included_when_requested() {
    let options = DocumentAnalysisOptions {
        classification_depth: ClassificationDepth::LexicalFallback,
        classification_labels: vec!["rust".to_string(), "travel".to_string()],
        zero_shot_labels: vec!["code".to_string(), "holiday".to_string()],
        ..DocumentAnalysisOptions::default()
    };

    let report = analyze_text(
        "doc-classification",
        "Rust cargo crates provide reliable code workflows.",
        &options,
    )
    .unwrap();
    let classification = report.classification.expect("classification section");

    assert_eq!(
        classification.sentiment.runtime,
        text_classification::TextClassificationRuntime::Lexical
    );
    assert!(classification
        .classification
        .predictions
        .iter()
        .any(|prediction| prediction.label == "rust"));
    assert!(!classification.zero_shot.predictions.is_empty());
}

#[test]
fn document_options_control_report_shape_and_embedding_detail() {
    let options = DocumentAnalysisOptions {
        language_hint: Some("en".to_string()),
        keyword_limit: 3,
        summary_sentences: 1,
        ngram_sizes: vec![2],
        shingle_sizes: vec![2],
        include_annotation_graph: false,
        linguistic_depth: LinguisticDepth::Off,
        embedding_depth: EmbeddingDepth::Hashed {
            dimensions: 16,
            use_idf: false,
        },
        include_sparse_embedding: true,
        ..DocumentAnalysisOptions::default()
    };
    let report = analyze_text(
        "doc-options",
        "OPENAI shipped 2 Rust crates. Email dev@example.com or visit https://example.com #rust @team.",
        &options,
    )
    .unwrap();

    assert_eq!(report.language.as_deref(), Some("en"));
    assert!(report.core.annotation_graph.is_none());
    assert!(report.lexical.keywords.len() <= 3);
    assert_eq!(report.lexical.extractive_summary.len(), 1);
    assert_eq!(report.similarity.character_ngram_frequencies[0].n, 2);
    assert_eq!(report.similarity.token_shingle_counts[0].n, 2);
    assert_eq!(report.enriched_stats.url_count, 1);
    assert_eq!(report.enriched_stats.email_count, 1);
    assert_eq!(report.enriched_stats.mention_count, 1);
    assert_eq!(report.enriched_stats.hashtag_count, 1);
    assert!(report.enriched_stats.uppercase_token_ratio > 0.0);

    let embedding = report.embedding.expect("hashed embedding");
    assert_eq!(embedding.dimensions, 16);
    assert_eq!(embedding.vector.len(), 16);
    assert!(embedding.preview.len() <= 16);
    let sparse = embedding.sparse.expect("sparse embedding preview");
    assert_eq!(sparse.dimensions, 16);
    assert_eq!(sparse.indices.len(), sparse.values.len());
    assert!(!sparse.indices.is_empty());
}

#[test]
fn analyze_document_uses_document_language_before_hint() {
    let document = document_from_text("doc-language", "Bonjour le monde.");
    let document = text_core::TextDocument {
        language: Some("fr"),
        ..document
    };
    let options = DocumentAnalysisOptions {
        language_hint: Some("en".to_string()),
        linguistic_depth: LinguisticDepth::Off,
        embedding_depth: EmbeddingDepth::Off,
        ..DocumentAnalysisOptions::default()
    };

    let report = text_analysis::analyze_document(&document, &options).unwrap();

    assert_eq!(report.id, "doc-language");
    assert_eq!(report.language.as_deref(), Some("fr"));
}

#[test]
fn invalid_keyword_and_shingle_options_are_rejected() {
    let keyword_options = DocumentAnalysisOptions {
        keyword_limit: 0,
        ..DocumentAnalysisOptions::default()
    };
    let keyword_error = analyze_text("doc", "text", &keyword_options).unwrap_err();
    assert!(keyword_error.to_string().contains("keyword limit"));

    let shingle_options = DocumentAnalysisOptions {
        shingle_sizes: vec![0],
        ..DocumentAnalysisOptions::default()
    };
    let shingle_error = analyze_text("doc", "text", &shingle_options).unwrap_err();
    assert!(shingle_error.to_string().contains("shingle sizes"));
}

#[test]
fn model_backed_helper_does_not_auto_download_by_default() {
    let options = DocumentAnalysisOptions::model_backed();

    assert_eq!(options.profile, AnalysisProfile::ModelBacked);
    match options.linguistic_depth {
        LinguisticDepth::LocalModel {
            auto_download,
            download_progress,
            ..
        } => {
            assert!(!auto_download);
            assert!(!download_progress);
        }
        other => panic!("expected local model depth, got {other:?}"),
    }
}

#[test]
fn model_backed_with_downloads_is_explicit_download_opt_in() {
    let options = DocumentAnalysisOptions::model_backed_with_downloads();

    assert_eq!(options.profile, AnalysisProfile::ModelBacked);
    match options.linguistic_depth {
        LinguisticDepth::LocalModel {
            auto_download,
            download_progress,
            ..
        } => {
            assert!(auto_download);
            assert!(download_progress);
        }
        other => panic!("expected local model depth, got {other:?}"),
    }
}

#[test]
fn model_backed_profile_upgrade_does_not_auto_download() {
    let options = DocumentAnalysisOptions {
        profile: AnalysisProfile::ModelBacked,
        linguistic_depth: LinguisticDepth::HeuristicBalanced,
        ..DocumentAnalysisOptions::default()
    };

    let report = analyze_text("doc", "Alice works at OpenAI in Berlin.", &options).unwrap();

    assert!(report.linguistic.is_none());
    assert!(report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "linguistics_unavailable"));
}

#[test]
fn model_backed_profile_uses_local_model_options_or_reports_diagnostic() {
    let options = DocumentAnalysisOptions {
        profile: AnalysisProfile::ModelBacked,
        linguistic_depth: LinguisticDepth::LocalModel {
            bundle_dir: std::env::temp_dir().join("text-analysis-missing-ner-bundle"),
            auto_download: false,
            download_progress: false,
        },
        ..DocumentAnalysisOptions::default()
    };
    let report = analyze_text("doc", "Alice works at OpenAI in Berlin.", &options).unwrap();
    assert!(report.linguistic.is_none());
    assert!(report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "linguistics_unavailable"));
}
