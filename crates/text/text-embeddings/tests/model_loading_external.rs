#![cfg(feature = "external-tests")]

use std::path::PathBuf;

use model_runtime::{HuggingFaceDownloader, ModelBundle, ModelBundleStore, ModelPreset};
use text_embeddings::{CandleTextEmbedder, OnnxTextEmbedder, TextEmbeddingBackend};

fn bundle_dir() -> PathBuf {
    std::env::var_os("TEXT_EMBEDDINGS_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".model-runtime"))
}

fn load_preset(preset: ModelPreset) -> ModelBundle {
    let spec = preset.spec();
    let revision = spec.revision_value().unwrap_or("main").to_string();
    let store = ModelBundleStore::new(bundle_dir())
        .downloader(HuggingFaceDownloader::new().progress(false).max_retries(2));
    store
        .load(&spec.name, &revision)
        .or_else(|_| store.download(&spec))
        .unwrap()
}

#[test]
#[ignore = "downloads the MiniLM Candle bundle and compares batched with per-text embeddings"]
fn batched_embeddings_match_per_text() {
    let embedder = CandleTextEmbedder::from_bundle(load_preset(ModelPreset::MiniLmL6V2)).unwrap();
    let texts = [
        "hello world",
        "Short.",
        "A Rust package ranks transcript passages for semantic search.",
        "Semantic search embeds every transcript segment, then ranks segments by cosine similarity to the query vector.",
        "Ein deutscher Satz über Suchmaschinen.",
    ];
    let batched = embedder.embed_batch(&texts).unwrap();
    for (text, batch) in texts.iter().zip(&batched) {
        let single = embedder.embed_text(text).unwrap();
        assert_eq!(single.dimensions(), batch.dimensions(), "{text}");
        // Padding changes the matrix-multiply shapes, so components may differ in the last
        // float bits; the vectors still point the same way.
        for (single, batch) in single.as_slice().iter().zip(batch.as_slice()) {
            assert!(
                (single - batch).abs() <= 1e-5,
                "{text}: {single} vs {batch}"
            );
        }
    }
}

#[test]
#[ignore = "loads opt-in MiniLM embedding bundles synced by scripts/sync_model_bundles.sh text"]
fn minilm_embedding_bundles_load_and_embed() {
    let cache = PathBuf::from(".model-runtime");

    let candle_bundle = ModelBundle::load(cache.join("minilm-l6-v2").join("main")).unwrap();
    let candle = CandleTextEmbedder::from_bundle(candle_bundle).unwrap();
    assert!(candle.embed_text("hello world").unwrap().dimensions() > 0);

    let onnx_bundle =
        ModelBundle::load(cache.join("xenova-minilm-l6-v2-onnx").join("main")).unwrap();
    let onnx = OnnxTextEmbedder::from_bundle(onnx_bundle).unwrap();
    assert!(onnx.embed_text("hello world").unwrap().dimensions() > 0);
}
