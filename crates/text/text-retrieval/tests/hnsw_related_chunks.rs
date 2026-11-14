//! The `hnsw` feature routes unfiltered semantic neighbours through Foundation's
//! external HNSW index on large corpora and keeps exact search elsewhere (#119).
#![cfg(feature = "hnsw")]

#[path = "support/ann_corpus.rs"]
mod ann_corpus;

use std::collections::BTreeSet;

use text_retrieval::RetrievalIndex;
use vector_analysis_index::VectorSearchIndex;

const TOP_K: usize = 10;

fn exact_related(exact: &VectorSearchIndex, chunk_id: &str) -> Vec<String> {
    let record = exact
        .records()
        .iter()
        .find(|record| record.id == chunk_id)
        .unwrap();
    exact
        .search_filtered(record.vector.as_slice(), TOP_K + 1, None)
        .unwrap()
        .into_iter()
        .map(|hit| hit.id.into_string())
        .filter(|id| id != chunk_id)
        .take(TOP_K)
        .collect()
}

fn related<B: text_embeddings::TextEmbedderBackend>(
    index: &RetrievalIndex<B>,
    chunk_id: &str,
) -> Vec<String> {
    index
        .related_chunks(chunk_id, TOP_K)
        .unwrap()
        .into_iter()
        .map(|result| result.chunk_id)
        .collect()
}

#[test]
fn large_corpus_related_chunks_keep_exact_recall() {
    let index = ann_corpus::index(4_096);
    let exact = VectorSearchIndex::import_records(index.vector_records()).unwrap();
    let chunk_ids = index
        .chunks_iter()
        .map(|chunk| chunk.chunk_id.clone())
        .step_by(41)
        .collect::<Vec<_>>();

    let mut hits = 0usize;
    let mut expected_total = 0usize;
    for chunk_id in &chunk_ids {
        let expected = exact_related(&exact, chunk_id);
        let actual = related(&index, chunk_id);
        assert_eq!(actual.len(), expected.len());
        let expected_ids = expected.iter().collect::<BTreeSet<_>>();
        hits += actual.iter().filter(|id| expected_ids.contains(id)).count();
        expected_total += expected.len();
    }
    let recall = hits as f64 / expected_total as f64;
    assert!(
        recall >= 0.95,
        "HNSW related-chunk recall@{TOP_K} was {recall:.4}"
    );

    // The lazily built graph is derived state: a clone rebuilds it identically.
    let clone = index.clone();
    for chunk_id in chunk_ids.iter().take(10) {
        assert_eq!(related(&clone, chunk_id), related(&index, chunk_id));
    }
}

#[test]
fn small_corpus_stays_exact() {
    let index = ann_corpus::index(512);
    let exact = VectorSearchIndex::import_records(index.vector_records()).unwrap();
    for chunk in index.chunks_iter().step_by(17) {
        assert_eq!(
            related(&index, &chunk.chunk_id),
            exact_related(&exact, &chunk.chunk_id)
        );
    }
}
