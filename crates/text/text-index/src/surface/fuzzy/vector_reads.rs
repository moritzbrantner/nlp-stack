//! Query-only vector-read regressions and opt-in cross-revision measurements.
use super::*;
use crate::{
    HashedTextEmbedder, IndexBuildOptions, IndexDocument, IndexInspectReport,
    IndexMutationReport, MemoryIndexStore, StoredVector, TextIndex, TextIndexError,
};
use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;

#[derive(Default)]
struct ObservedStore {
    inner: MemoryIndexStore,
    reads: Cell<usize>,
    scalars: Cell<usize>,
    unavailable: bool,
}

impl TextIndexStore for ObservedStore {
    fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    fn upsert_document_state(
        &mut self,
        document: IndexDocument,
        chunks: Vec<IndexChunk>,
        vectors: Vec<StoredVector>,
        options: &IndexBuildOptions,
    ) -> crate::Result<IndexMutationReport> {
        self.inner.upsert_document_state(document, chunks, vectors, options)
    }

    fn remove_documents(&mut self, ids: &[String]) -> crate::Result<usize> {
        self.inner.remove_documents(ids)
    }

    fn documents(&self) -> crate::Result<Vec<IndexDocument>> {
        self.inner.documents()
    }

    fn chunks(&self) -> crate::Result<Vec<IndexChunk>> {
        self.inner.chunks()
    }

    fn vectors(&self) -> crate::Result<Vec<StoredVector>> {
        self.reads.set(self.reads.get() + 1);
        if self.unavailable {
            return Err(TextIndexError::InvalidState("vector storage unavailable".to_string()));
        }
        let vectors = self.inner.vectors()?;
        self.scalars.set(self.scalars.get() + vectors.iter().map(|v| v.vector.len()).sum::<usize>());
        Ok(vectors)
    }

    fn lexical_candidates(&self, query: &str, limit: usize) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.inner.lexical_candidates(query, limit)
    }

    fn inspect(&self) -> crate::Result<IndexInspectReport> {
        self.inner.inspect()
    }
}

fn embedder(dimensions: usize) -> HashedTextEmbedder {
    HashedTextEmbedder::new(
        crate::TextEmbeddingConfig { dimensions, use_idf: false },
        crate::CorpusOptions::default(),
    ).expect("embedder")
}

fn documents(count: usize) -> Vec<IndexDocument> {
    (0..count).map(|i| {
        let mut document = IndexDocument::new(
            format!("doc-{i:05}"),
            format!("strategy retrieval archive shared evidence {i}"),
        );
        document.metadata.attributes.insert("access".to_string(),
            if i % 2 == 0 { "allowed" } else { "denied" }.to_string());
        document
    }).collect()
}

fn observed_index(count: usize, dimensions: usize) -> TextIndex<HashedTextEmbedder, ObservedStore> {
    let mut index = TextIndex::with_store(embedder(dimensions), ObservedStore::default());
    index.upsert_documents(&documents(count)).expect("documents");
    index
}

#[test]
fn lexical_performs_no_vector_reads_or_materialization() {
    let index = observed_index(8, 128);
    assert!(!index.search(&IndexQuery::lexical("strategy", 3)).unwrap().is_empty());
    assert_eq!(index.store().reads.get(), 0);
    assert_eq!(index.store().scalars.get(), 0);
}

#[test]
fn lexical_survives_unavailable_vector_storage() {
    let mut index = observed_index(8, 128);
    index.store_mut().unavailable = true;
    assert_eq!(index.search(&IndexQuery::lexical("strategy", 3)).unwrap().len(), 3);
    assert_eq!(index.store().reads.get(), 0);
}

#[test]
fn semantic_and_hybrid_read_and_use_vectors_once() {
    for query in [IndexQuery::semantic("strategy retrieval", 3), IndexQuery::new("strategy retrieval", 3)] {
        let index = observed_index(8, 128);
        let results = index.search(&query).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|r| r.score_breakdown.semantic_score > 0.0));
        assert_eq!(index.store().reads.get(), 1);
        assert_eq!(index.store().scalars.get(), 8 * 128);
    }
}

#[test]
fn semantic_and_hybrid_propagate_vector_storage_errors() {
    for query in [IndexQuery::semantic("strategy", 3), IndexQuery::new("strategy", 3)] {
        let mut index = observed_index(8, 128);
        index.store_mut().unavailable = true;
        assert!(matches!(index.search(&query), Err(TextIndexError::InvalidState(message))
            if message == "vector storage unavailable"));
        assert_eq!(index.store().reads.get(), 1);
    }
}

#[test]
fn lexical_preserves_scores_phrases_filters_and_explanations_without_vectors() {
    let mut index = observed_index(8, 128);
    let mut query = IndexQuery::lexical("strategy retrieval", 3);
    query.explain = true;
    query.required_phrases = vec!["shared evidence".to_string()];
    query.filter.metadata_equals.insert("access".to_string(), "allowed".to_string());
    let expected = index.search(&query).unwrap();
    assert_eq!(expected.len(), 3);
    assert!(expected.iter().all(|r| r.matched_phrases == ["shared evidence"]));
    index.store_mut().unavailable = true;
    assert_eq!(index.search(&query).unwrap(), expected);
}

#[test]
fn invalid_query_is_rejected_before_vector_storage() {
    let mut index = observed_index(8, 128);
    index.store_mut().unavailable = true;
    assert!(matches!(index.search(&IndexQuery::lexical("strategy", 0)),
        Err(TextIndexError::InvalidArgument(_))));
    assert_eq!(index.store().reads.get(), 0);
}

#[cfg(feature = "sqlite")]
#[test]
fn fuzzy_sqlite_variants_do_not_read_corrupt_vector_payloads() {
    let mut index = TextIndex::with_store(embedder(128), crate::SqliteIndexStore::in_memory().unwrap());
    index.upsert_documents(&documents(8)).unwrap();
    let surface = SurfaceIndex::Sqlite(index);
    let options = FuzzySearchOptions::default();
    let query = IndexQuery::lexical("stratgey retrieavl", 3);
    let vocabulary = build_vocabulary(&index_chunks(&surface).unwrap(), &options).unwrap();
    assert_eq!(build_query_variants(&normalized_query_tokens(&query.text), &vocabulary, &options).len(), 4);
    let expected = search(&surface, &query, &options).unwrap();
    assert_eq!(expected.as_array().unwrap().len(), 3);
    if let SurfaceIndex::Sqlite(index) = &surface {
        let changed = index.store().connection.execute("UPDATE vectors SET vector_blob = X'01'", []).unwrap();
        assert_eq!(changed, 8);
        assert!(index.store().vectors().is_err(), "positive control must detect corrupt vectors");
    }
    assert_eq!(search(&surface, &query, &options).unwrap(), expected);
    assert!(surface.search(&IndexQuery::semantic("strategy", 3)).is_err());
    assert!(surface.search(&IndexQuery::new("strategy", 3)).is_err());
}

#[test]
#[ignore = "paired native timing evidence; no wall-clock CI threshold"]
fn query_only_benchmark() {
    let iterations: usize = std::env::var("NLP_VECTOR_BENCH_ITERATIONS")
        .unwrap_or_else(|_| "10".to_string()).parse().expect("positive iterations");
    assert!(iterations > 0);
    for count in [32, 256] {
        for dimensions in [128, 1024] {
            let observed = observed_index(count, dimensions);
            let index = TextIndex::with_store(embedder(dimensions), observed.store().inner.clone());
            let surface = SurfaceIndex::Memory(index);
            let options = FuzzySearchOptions::default();
            for (kind, text) in [("lexical", "strategy retrieval"),
                ("fuzzy-one", "stratgey retrieval"), ("fuzzy-two", "stratgey retrieavl")] {
                let query = IndexQuery::lexical(text, 8);
                let vocabulary = build_vocabulary(&index_chunks(&surface).unwrap(), &options).unwrap();
                let variants = build_query_variants(&normalized_query_tokens(text), &vocabulary, &options).len();
                let operation = || {
                    if kind == "lexical" {
                        serde_json::to_value(surface.search(black_box(&query)).unwrap()).unwrap()
                    } else {
                        search(&surface, black_box(&query), &options).unwrap()
                    }
                };
                let expected = operation();
                assert_eq!(expected.as_array().unwrap().len(), 8);
                // Observe work outside the timed region; time the real unwrapped store.
                let work = if kind == "lexical" {
                    observed.store().reads.set(0);
                    observed.store().scalars.set(0);
                    assert_eq!(serde_json::to_value(observed.search(&query).unwrap()).unwrap(), expected);
                    serde_json::json!({"vector_reads": observed.store().reads.get(),
                        "vector_scalars": observed.store().scalars.get()})
                } else { serde_json::Value::Null };
                black_box(operation());
                let started = Instant::now();
                for _ in 0..iterations { black_box(operation()); }
                let nanos = started.elapsed().as_nanos() as f64 / iterations as f64;
                println!("NLP_VECTOR_BENCH {}", serde_json::json!({
                    "kind": kind, "documents": count, "dimensions": dimensions,
                    "query": text, "variants": variants, "iterations": iterations,
                    "ns_per_query": nanos, "work": work, "results": expected,
                }));
            }
        }
    }
}
