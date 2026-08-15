#![cfg(feature = "external-tests")]

use text_classification::{
    analyze_sentiment, classify_text, zero_shot_classify, SentimentRequest,
    TextClassificationLocalModelOptions, TextClassificationRequest, TextClassificationRuntime,
    ZeroShotClassificationRequest,
};

fn workspace_model_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(".model-runtime")
}

fn local_model(model_id: &str) -> TextClassificationLocalModelOptions {
    TextClassificationLocalModelOptions {
        model_id: Some(model_id.to_string()),
        bundle_root: Some(workspace_model_root()),
        auto_download: Some(false),
        download_progress: Some(false),
        ..TextClassificationLocalModelOptions::default()
    }
}

fn require_local_bundle(model_id: &str) {
    let manifest = workspace_model_root()
        .join(model_id)
        .join("main")
        .join("manifest.json");
    assert!(
        manifest.exists(),
        "missing model bundle `{model_id}` at `{}` and autoDownload is false",
        manifest.display()
    );
}

#[test]
#[ignore = "requires a local DistilBERT SST-2 Candle bundle"]
fn distilbert_sst2_classification_smoke() {
    require_local_bundle("distilbert-sst2");
    let response = classify_text(TextClassificationRequest {
        text: "Rust text workflows are reliable and useful.".to_string(),
        labels: vec!["positive".to_string(), "negative".to_string()],
        top_k: 2,
        multi_label: false,
        model: Default::default(),
        imported_predictions: Vec::new(),
        local_model: Some(local_model("distilbert-sst2")),
    })
    .expect("local classification");

    assert_eq!(response.runtime, TextClassificationRuntime::Candle);
    assert!(response
        .predictions
        .iter()
        .all(|prediction| prediction.score.is_finite()));
    assert!(response
        .predictions
        .iter()
        .any(|prediction| prediction.label.eq_ignore_ascii_case("positive")));
}

#[test]
#[ignore = "requires a local DistilBERT SST-2 Candle bundle"]
fn distilbert_sst2_sentiment_smoke() {
    require_local_bundle("distilbert-sst2");
    let response = analyze_sentiment(SentimentRequest {
        text: "The transcript search results were accurate.".to_string(),
        model: Default::default(),
        imported_predictions: Vec::new(),
        local_model: Some(local_model("distilbert-sst2")),
    })
    .expect("local sentiment");

    assert_eq!(response.runtime, TextClassificationRuntime::Candle);
    assert!(response.positive_score.is_finite());
    assert!(response.negative_score.is_finite());
}

#[test]
#[ignore = "requires a local Xenova BART MNLI ONNX bundle"]
fn bart_mnli_zero_shot_smoke() {
    require_local_bundle("xenova-bart-large-mnli-onnx");
    let response = zero_shot_classify(ZeroShotClassificationRequest {
        text: "A Rust package ranks transcript passages for semantic search.".to_string(),
        labels: vec![
            "software engineering".to_string(),
            "sports recap".to_string(),
            "music metadata".to_string(),
        ],
        hypothesis_template: "This text is about {}.".to_string(),
        model: Default::default(),
        imported_predictions: Vec::new(),
        local_model: Some(local_model("xenova-bart-large-mnli-onnx")),
    })
    .expect("local zero-shot");

    assert_eq!(response.runtime, TextClassificationRuntime::Onnx);
    assert!(response
        .predictions
        .iter()
        .all(|prediction| prediction.score.is_finite()));
    assert!(response
        .predictions
        .iter()
        .any(|prediction| prediction.label == "software engineering"));
}
