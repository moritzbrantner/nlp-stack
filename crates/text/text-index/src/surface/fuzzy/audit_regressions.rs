//! These tests can be transplanted unchanged onto aa425fc to reproduce the bugs.
use super::*;
use crate::{IndexDocument, MemoryTextIndex};

fn indexes(documents: &[IndexDocument]) -> Vec<SurfaceIndex> {
    let mut memory = MemoryTextIndex::new_memory().expect("memory index");
    memory.upsert_documents(documents).expect("index documents");
    #[cfg(not(feature = "sqlite"))]
    let sqlite: Option<SurfaceIndex> = None;
    #[cfg(feature = "sqlite")]
    let sqlite = {
        let embedder = crate::HashedTextEmbedder::new(
            crate::TextEmbeddingConfig {
                dimensions: 128,
                use_idf: false,
            },
            crate::CorpusOptions::default(),
        )
        .expect("embedder");
        let mut index = crate::SqliteTextIndex::with_store(
            embedder,
            crate::SqliteIndexStore::in_memory().expect("SQLite store"),
        );
        index.upsert_documents(documents).expect("SQLite documents");
        Some(SurfaceIndex::Sqlite(index))
    };
    std::iter::once(SurfaceIndex::Memory(memory))
        .chain(sqlite)
        .collect()
}

#[test]
fn zero_top_k_is_rejected_even_when_corrections_exist() {
    for index in indexes(&[IndexDocument::new("doc", "strategy")]) {
        let query = IndexQuery::lexical("stratgey", 0);
        let expected = index.search(&query).expect_err("invalid original query");
        assert_eq!(
            search(&index, &query, &FuzzySearchOptions::default()).unwrap_err(),
            expected
        );
    }
}

#[test]
fn zero_candidate_limit_is_rejected_even_when_corrections_exist() {
    for index in indexes(&[IndexDocument::new("doc", "strategy")]) {
        let mut query = IndexQuery::lexical("stratgey", 3);
        query.candidate_limit = 0;
        let expected = index.search(&query).expect_err("invalid original query");
        assert_eq!(
            search(&index, &query, &FuzzySearchOptions::default()).unwrap_err(),
            expected
        );
    }
}

#[test]
fn validation_does_not_depend_on_whether_vocabulary_has_a_correction() {
    for body in ["strategy", "kitchen"] {
        for index in indexes(&[IndexDocument::new("doc", body)]) {
            let query = IndexQuery::lexical("stratgey", 0);
            assert!(search(&index, &query, &FuzzySearchOptions::default()).is_err());
        }
    }
}

#[test]
fn unrelated_expansion_does_not_erase_required_phrase_hits() {
    for with_expansion in [false, true] {
        let mut documents = vec![IndexDocument::new("target", "locked phrase")];
        if with_expansion {
            documents.push(IndexDocument::new("unrelated", "strategy"));
        }
        for index in indexes(&documents) {
            let mut query = IndexQuery::lexical("stratgey", 3);
            query.required_phrases = vec!["locked phrase".to_string()];
            let ordinary = index.search(&query).expect("ordinary phrase search");
            assert_eq!(ordinary.len(), 1);
            assert_eq!(ordinary[0].score_breakdown.lexical_score, 0.0);
            let mut fuzzy = search(&index, &query, &FuzzySearchOptions::default())
                .expect("fuzzy phrase search");
            let results = fuzzy.as_array_mut().expect("results");
            assert_eq!(results.len(), 1, "with_expansion={with_expansion}");
            assert_eq!(results[0]["fuzzyMatches"], serde_json::json!([]));
            results[0]
                .as_object_mut()
                .expect("result object")
                .remove("fuzzyMatches");
            assert_eq!(fuzzy, serde_json::to_value(ordinary).unwrap());
        }
    }
}

#[test]
fn retained_zero_score_hits_still_obey_every_phrase_and_metadata_filter() {
    let mut documents = vec![
        IndexDocument::new("target", "locked phrase approval"),
        IndexDocument::new("missing-approval", "locked phrase strategy"),
        IndexDocument::new("missing-phrase", "approval strategy"),
        IndexDocument::new("denied", "locked phrase approval strategy"),
    ];
    for document in &mut documents {
        document.metadata.attributes.insert(
            "access".to_string(),
            if document.id == "denied" {
                "denied"
            } else {
                "allowed"
            }
            .to_string(),
        );
    }
    for index in indexes(&documents) {
        let mut query = IndexQuery::lexical("stratgey", 10);
        query.required_phrases = vec!["locked phrase".to_string(), "approval".to_string()];
        query
            .filter
            .metadata_equals
            .insert("access".to_string(), "allowed".to_string());
        let fuzzy = search(&index, &query, &FuzzySearchOptions::default()).expect("search");
        let results = fuzzy.as_array().expect("results");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["documentId"], "target");
        assert_eq!(results[0]["fuzzyMatches"], serde_json::json!([]));
    }
}

#[test]
fn positive_fuzzy_hits_rank_above_retained_phrase_only_hits() {
    for index in indexes(&[
        IndexDocument::new("positive", "strategy locked phrase"),
        IndexDocument::new("zero", "locked phrase"),
    ]) {
        let mut query = IndexQuery::lexical("stratgey", 10);
        query.required_phrases = vec!["locked phrase".to_string()];
        let fuzzy = search(&index, &query, &FuzzySearchOptions::default()).expect("search");
        let results = fuzzy.as_array().expect("results");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["documentId"], "positive");
        assert_eq!(results[0]["score"].as_f64(), Some(1.0));
        assert_eq!(results[1]["documentId"], "zero");
        assert_eq!(results[1]["score"].as_f64(), Some(0.0));
        assert_eq!(results[1]["fuzzyMatches"], serde_json::json!([]));
    }
}

#[test]
fn retaining_phrase_hits_does_not_fabricate_unmatched_results() {
    for index in indexes(&[IndexDocument::new("doc", "strategy")]) {
        let mut query = IndexQuery::lexical("stratgey", 3);
        query.required_phrases = vec!["absent phrase".to_string()];
        assert_eq!(
            search(&index, &query, &FuzzySearchOptions::default()).unwrap(),
            serde_json::json!([])
        );
    }
}
