#![cfg(feature = "external-tests")]

use std::path::PathBuf;

use model_runtime::ModelBundle;
use text_embeddings::{CandleTextEmbedder, OnnxTextEmbedder, TextEmbeddingBackend};

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
