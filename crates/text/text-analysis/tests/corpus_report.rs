use text_analysis::{
    analyze_corpus, CorpusAnalysisOptions, DocumentAnalysisOptions, EmbeddingDepth,
};
use text_core::TextDocument;

#[test]
fn corpus_report_includes_tfidf_bm25_duplicates_and_semantic_neighbors() {
    let documents = [
        TextDocument::new("doc-1", "rust cargo crates text analysis"),
        TextDocument::new("doc-2", "rust cargo crates text analysis pipeline"),
        TextDocument::new("doc-3", "video scene reports and camera motion"),
    ];
    let options = CorpusAnalysisOptions {
        query: Some("rust text analysis".to_string()),
        near_duplicate_threshold: 0.4,
        ..CorpusAnalysisOptions::default()
    };

    let report = analyze_corpus(documents, &options).unwrap();

    assert_eq!(report.stats.documents, 3);
    assert!(report.documents[0]
        .tfidf_terms
        .iter()
        .any(|term| term.term == "rust"));
    assert_eq!(report.tfidf_search.as_ref().unwrap()[0].id, "doc-1");
    assert_eq!(report.bm25_search.as_ref().unwrap()[0].id, "doc-1");
    assert!(!report.near_duplicates.is_empty());
    // Semantic neighbors are not requested by default (they need a model).
    assert!(report.semantic_neighbors.is_empty());
    assert!(!report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "semantic_neighbors_unavailable"));

    let requested = CorpusAnalysisOptions {
        include_semantic_neighbors: true,
        ..options
    };
    let report = analyze_corpus(
        [
            TextDocument::new("doc-1", "rust cargo crates text analysis"),
            TextDocument::new("doc-2", "rust cargo crates text analysis pipeline"),
        ],
        &requested,
    )
    .unwrap();
    assert!(report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "semantic_neighbors_unavailable"));
}

#[test]
fn corpus_options_can_disable_optional_pair_reports() {
    let documents = [
        TextDocument::new("doc-1", "rust text analysis"),
        TextDocument::new("doc-2", "video scene analysis"),
    ];
    let options = CorpusAnalysisOptions {
        query: None,
        include_near_duplicates: false,
        include_semantic_neighbors: false,
        document: DocumentAnalysisOptions {
            embedding_depth: EmbeddingDepth::Off,
            ..DocumentAnalysisOptions::default()
        },
        ..CorpusAnalysisOptions::default()
    };

    let report = analyze_corpus(documents, &options).unwrap();

    assert!(report.tfidf_search.is_none());
    assert!(report.bm25_search.is_none());
    assert!(report.near_duplicates.is_empty());
    assert!(report.semantic_neighbors.is_empty());
    assert!(report
        .documents
        .iter()
        .all(|document| document.embedding_preview.is_none()));
}

#[test]
fn corpus_validation_rejects_zero_limits() {
    let documents = [TextDocument::new("doc-1", "rust text analysis")];

    let top_k_options = CorpusAnalysisOptions {
        top_k: 0,
        ..CorpusAnalysisOptions::default()
    };
    let top_k_error = analyze_corpus(documents, &top_k_options).unwrap_err();
    assert!(top_k_error.to_string().contains("top_k"));

    let shingle_options = CorpusAnalysisOptions {
        near_duplicate_shingle_size: 0,
        ..CorpusAnalysisOptions::default()
    };
    let shingle_error = analyze_corpus(documents, &shingle_options).unwrap_err();
    assert!(shingle_error
        .to_string()
        .contains("near duplicate shingle size"));
}

#[test]
fn semantic_neighbors_report_diagnostic_for_non_hashed_embeddings() {
    let documents = [
        TextDocument::new("doc-1", "rust text analysis"),
        TextDocument::new("doc-2", "rust text analysis pipeline"),
    ];
    let options = CorpusAnalysisOptions {
        document: DocumentAnalysisOptions {
            embedding_depth: EmbeddingDepth::OnnxBundle {
                bundle_dir: std::env::temp_dir().join("missing-text-analysis-onnx-bundle"),
                pooling: text_embeddings::PoolingStrategy::Mean,
            },
            ..DocumentAnalysisOptions::default()
        },
        include_semantic_neighbors: true,
        ..CorpusAnalysisOptions::default()
    };

    let report = analyze_corpus(documents, &options).unwrap();

    assert!(report.semantic_neighbors.is_empty());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "onnx_embedding_unavailable"
            || diagnostic.code == "semantic_neighbors_model_provider"
    }));
}
