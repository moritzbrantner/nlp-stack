#![cfg(feature = "external-tests")]

use std::path::PathBuf;

use text_model_runtime::{validate_text_model_bundle, TextModelBundleCheck, TextModelCapability};

#[test]
#[ignore = "validates opt-in MiniLM embedding bundles from .model-runtime"]
fn minilm_embedding_bundles_report_loadability() {
    let cache = PathBuf::from(".model-runtime");
    let candle = validate_text_model_bundle(
        TextModelBundleCheck::new(
            "minilm-l6-v2",
            TextModelCapability::Embedding,
            cache.join("minilm-l6-v2").join("main"),
            ["config.json", "tokenizer.json", "model.safetensors"],
        )
        .required_feature("candle,model-bundles")
        .required_setup("scripts/sync_model_bundles.sh text")
        .smoke_operation("embeddings.embed"),
    );
    let onnx = validate_text_model_bundle(
        TextModelBundleCheck::new(
            "xenova-minilm-l6-v2-onnx",
            TextModelCapability::Embedding,
            cache.join("xenova-minilm-l6-v2-onnx").join("main"),
            ["config.json", "tokenizer.json", "onnx/model.onnx"],
        )
        .required_feature("onnx,model-bundles")
        .required_setup("scripts/sync_model_bundles.sh text")
        .smoke_operation("embeddings.embed"),
    );

    assert_eq!(candle.loadable, candle.bundle_present());
    assert_eq!(onnx.loadable, onnx.bundle_present());
}
