//! Fake QA bundle for the ONNX Runtime load-error tests (#176). Each test lives in its own
//! binary because ONNX Runtime is loaded once per process.

use std::path::{Path, PathBuf};

use text_question_answering::{
    answer_question, QuestionAnsweringFallbackPolicy, QuestionAnsweringLocalModelOptions,
    QuestionAnsweringRequest,
};

/// A bundle root holding `roberta-base-squad2-onnx/main` with a placeholder model; the
/// runtime load fails before the model bytes are read.
pub fn fake_qa_bundle_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("nlp-qa-{tag}-{}", std::process::id()));
    let bundle = root.join("roberta-base-squad2-onnx").join("main");
    std::fs::create_dir_all(bundle.join("files/onnx")).unwrap();
    for (path, bytes) in [
        ("config.json", &b"{}"[..]),
        ("tokenizer.json", b"{}"),
        ("onnx/model_quantized.onnx", b"not a model"),
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
        "name": "roberta-base-squad2-onnx",
        "repo_id": "onnx-community/roberta-base-squad2-ONNX",
        "revision": "main",
        "task": "question_answering",
        "files": {
            "config.json": file("config.json"),
            "tokenizer.json": file("tokenizer.json"),
            "onnx/model_quantized.onnx": file("onnx/model_quantized.onnx"),
        }
    });
    std::fs::write(
        bundle.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    root
}

/// Runs local QA against `bundle_root` with the heuristic fallback disabled and returns the
/// error.
pub fn qa_error(bundle_root: &Path) -> String {
    answer_question(QuestionAnsweringRequest {
        question: "Who wrote the parser?".to_string(),
        context: "Ada wrote the parser in Rust.".to_string(),
        top_k: 1,
        model: Default::default(),
        imported_predictions: Vec::new(),
        local_model: Some(QuestionAnsweringLocalModelOptions {
            model_id: Some("roberta-base-squad2-onnx".to_string()),
            bundle_root: Some(bundle_root.to_path_buf()),
            auto_download: Some(false),
            ..QuestionAnsweringLocalModelOptions::default()
        }),
        fallback_policy: Some(QuestionAnsweringFallbackPolicy::Error),
    })
    .expect_err("QA must fail without a usable ONNX Runtime")
    .to_string()
}
