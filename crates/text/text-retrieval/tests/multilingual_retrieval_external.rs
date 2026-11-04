//! Opt-in cross-lingual retrieval with the real default embedding model (#188).
//!
//! Run with `cargo test -p moenarch-text-retrieval --features external-tests --release
//! --test multilingual_retrieval_external -- --ignored --nocapture`. The pinned
//! `intfloat/multilingual-e5-base` bundle lives in `TEXT_EMBEDDINGS_MODEL_DIR`
//! (default `.model-runtime`) and is downloaded when missing.
#![cfg(feature = "external-tests")]

use std::sync::OnceLock;

use text_embeddings::{
    load_embedding_preset, CandleTextEmbedder, EmbeddingPresetLoadOptions,
    MultilingualEmbeddingPreset, PoolingStrategy,
};
use text_index::{IndexDocument, IndexQuery, MemoryIndexStore, TextIndex};
use text_retrieval::{
    HybridConfig, IngestionOptions, PersistedSearchIndex, RetrievalIndex, SearchDocument,
    SearchQuery,
};

fn e5() -> CandleTextEmbedder {
    static EMBEDDER: OnceLock<CandleTextEmbedder> = OnceLock::new();
    EMBEDDER
        .get_or_init(|| {
            load_embedding_preset(
                MultilingualEmbeddingPreset::DEFAULT,
                &EmbeddingPresetLoadOptions {
                    auto_download: true,
                    ..EmbeddingPresetLoadOptions::default()
                },
            )
            .unwrap()
        })
        .clone()
}

const PASSAGES: &[(&str, &str)] = &[
    (
        "en-eiffel",
        "The Eiffel Tower in Paris is about 330 metres tall and was finished in 1889.",
    ),
    (
        "en-tomatoes",
        "Tomatoes grow best in warm, sunny gardens with well-drained soil.",
    ),
    (
        "en-rust",
        "Rust's borrow checker prevents data races at compile time.",
    ),
    (
        "es-economy",
        "La economía española creció el año pasado gracias al turismo.",
    ),
    (
        "de-train",
        "Der Zug nach Hamburg hat heute zwanzig Minuten Verspätung.",
    ),
];

#[test]
#[ignore = "downloads intfloat/multilingual-e5-base (1.1 GB) and retrieves an English passage for a German query"]
fn text_index_german_query_retrieves_english_passage() {
    let mut index = TextIndex::with_store(e5(), MemoryIndexStore::new());
    let documents = PASSAGES
        .iter()
        .map(|(id, body)| IndexDocument::new(*id, *body))
        .collect::<Vec<_>>();
    let report = index.upsert_documents(&documents).unwrap();
    assert_eq!(report.vectors_indexed, PASSAGES.len());

    let info = index.embedder_info().expect("a real embedder");
    assert_eq!(info.model_name, "intfloat/multilingual-e5-base");
    assert_eq!(info.dimensions, 768);
    assert_eq!(
        info.revision.as_deref(),
        Some(MultilingualEmbeddingPreset::DEFAULT.revision())
    );
    assert_eq!(info.pooling, Some(PoolingStrategy::Mean));

    let results = index
        .search(&IndexQuery::semantic("Wie hoch ist der Eiffelturm?", 3))
        .unwrap();
    println!(
        "semantic: {:?}",
        results
            .iter()
            .map(|result| (&result.document_id, result.score_breakdown.semantic_score))
            .collect::<Vec<_>>()
    );
    assert_eq!(results[0].document_id, "en-eiffel");

    // No query term occurs in the English passage, so lexical search alone cannot
    // find it; hybrid search still ranks it first through the semantic score.
    let hybrid = index
        .search(&IndexQuery::hybrid(
            "Welche Pflanzen mögen warme Gärten?",
            3,
        ))
        .unwrap();
    assert_eq!(hybrid[0].document_id, "en-tomatoes");
}

#[test]
#[ignore = "downloads intfloat/multilingual-e5-base (1.1 GB) and runs RetrievalIndex in every mode"]
fn retrieval_index_runs_every_mode_with_a_real_embedder() {
    let mut index = RetrievalIndex::new(e5());
    let documents = PASSAGES
        .iter()
        .map(|(id, body)| SearchDocument::new(*id, *body))
        .collect::<Vec<_>>();
    index
        .ingest_documents(&documents, &IngestionOptions::default())
        .unwrap();

    let semantic = index
        .search(&SearchQuery::semantic(
            "¿Qué altura tiene la Torre Eiffel?",
            1,
        ))
        .unwrap();
    assert_eq!(semantic[0].document_id, "en-eiffel");

    let full_text = index
        .search(&SearchQuery::full_text("borrow checker", 1))
        .unwrap();
    assert_eq!(full_text[0].document_id, "en-rust");
    assert_eq!(full_text[0].semantic_score, 0.0);

    let hybrid = index
        .search(&SearchQuery::hybrid(
            "Verspätung des Zuges",
            1,
            HybridConfig::default(),
        ))
        .unwrap();
    assert_eq!(hybrid[0].document_id, "de-train");

    // The snapshot records the model id, revision and pooling and reloads with the
    // same embedder.
    let snapshot = PersistedSearchIndex::from_index(&index);
    assert_eq!(
        snapshot.manifest.embedder.revision.as_deref(),
        Some(MultilingualEmbeddingPreset::DEFAULT.revision())
    );
    let reloaded = snapshot.into_index(e5()).unwrap();
    assert_eq!(
        reloaded
            .search(&SearchQuery::semantic("Wie hoch ist der Eiffelturm?", 1))
            .unwrap()[0]
            .document_id,
        "en-eiffel"
    );
}
