//! Fake zero-shot bundle for the ONNX Runtime load-error tests (#176). Each test lives in its
//! own binary because ONNX Runtime is loaded once per process.

use std::path::{Path, PathBuf};

use text_classification::{
    zero_shot_classify, FallbackPolicy, ModelSelection, TextClassificationLocalModelOptions,
    ZeroShotClassificationRequest,
};

/// A bundle root holding `xenova-bart-large-mnli-onnx/main` with a placeholder model; the
/// runtime load fails before the model bytes are read.
pub fn fake_zero_shot_bundle_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("nlp-zero-shot-{tag}-{}", std::process::id()));
    let bundle = root.join("xenova-bart-large-mnli-onnx").join("main");
    std::fs::create_dir_all(bundle.join("files/onnx")).unwrap();
    for (path, bytes) in [
        ("config.json", &b"{}"[..]),
        ("tokenizer.json", b"{}"),
        ("onnx/model.onnx", b"not a model"),
    ] {
        std::fs::write(bundle.join("files").join(path), bytes).unwrap();
    }
    let file = |path: &str| {
        let size = std::fs::metadata(bundle.join("files").join(path))
            .unwrap()
            .len();
        serde_json::json!({
            "remote_path": path,
            "local_path": format!("files/{path}"),
            "size_bytes": size,
        })
    };
    let manifest = serde_json::json!({
        "schema_version": 1,
        "name": "xenova-bart-large-mnli-onnx",
        "repo_id": "Xenova/bart-large-mnli",
        "revision": "main",
        "task": "zero_shot_classification",
        "files": {
            "config.json": file("config.json"),
            "tokenizer.json": file("tokenizer.json"),
            "onnx/model.onnx": file("onnx/model.onnx"),
        }
    });
    std::fs::write(
        bundle.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    root
}

/// Runs local zero-shot against `bundle_root` with fallbacks disabled and returns the error.
pub fn zero_shot_error(bundle_root: &Path) -> String {
    zero_shot_classify(ZeroShotClassificationRequest {
        text: "Rust package workflows".to_string(),
        labels: vec!["software".to_string(), "music".to_string()],
        hypothesis_template: "This text is about {}.".to_string(),
        model: ModelSelection {
            fallback_policy: FallbackPolicy::Error,
            ..ModelSelection::default()
        },
        imported_predictions: Vec::new(),
        local_model: Some(TextClassificationLocalModelOptions {
            model_id: Some("xenova-bart-large-mnli-onnx".to_string()),
            bundle_root: Some(bundle_root.to_path_buf()),
            auto_download: Some(false),
            ..TextClassificationLocalModelOptions::default()
        }),
    })
    .expect_err("zero-shot must fail without a usable ONNX Runtime")
    .to_string()
}
