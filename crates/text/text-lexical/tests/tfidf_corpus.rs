use std::collections::BTreeSet;

use text_core::TextDocument;
use text_lexical::{CorpusOptions, TfIdfCorpus};

#[test]
fn indexes_text_documents_and_searches_ranked_results() {
    let mut stop_words = BTreeSet::new();
    stop_words.insert("the".to_string());
    stop_words.insert("and".to_string());
    let mut corpus = TfIdfCorpus::new(CorpusOptions {
        min_term_len: 3,
        stop_words,
        ..CorpusOptions::default()
    });

    let documents = [
        TextDocument::new("rust", "The Rust cargo tool builds crates and workspaces"),
        TextDocument::new("video", "Video frames need scene detection and indexing"),
        TextDocument::new("audio", "Audio frames contain rhythm pitch and spectra"),
    ];
    for document in &documents {
        corpus.add_text_document(document).unwrap();
    }

    let stats = corpus.stats();
    assert_eq!(stats.documents, 3);
    assert_eq!(corpus.document_count("frames"), 2);
    assert_eq!(
        corpus.document("rust").unwrap().term_counts.get("the"),
        None
    );

    let top_terms = corpus.term_stats(2);
    assert_eq!(top_terms[0].term, "frames");
    assert_eq!(top_terms[0].document_count, 2);

    let tfidf = corpus.document_tfidf("rust", 3).unwrap();
    assert!(tfidf.iter().any(|term| term.term == "cargo"));

    let results = corpus.search("cargo workspace crates", 2).unwrap();
    assert_eq!(results[0].id, "rust");
    assert!(results[0].score > 0.0);
    assert!(results.iter().all(|result| result.matched_terms > 0));
}

#[test]
fn ranks_external_text_against_existing_corpus_statistics() {
    let mut corpus = TfIdfCorpus::default();
    corpus
        .add_document("common", "shared shared common")
        .unwrap();
    corpus.add_document("rare", "shared unique").unwrap();

    let terms = corpus.text_tfidf("shared unique unique", 2);

    assert_eq!(terms[0].term, "unique");
    assert!(terms[0].score > terms[1].score);
    assert_eq!(terms[0].document_count, 1);
}
