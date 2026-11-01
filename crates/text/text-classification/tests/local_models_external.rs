#![cfg(feature = "external-tests")]

use model_runtime::{HuggingFaceDownloader, ModelBundleStore, ModelPreset};
use text_classification::{
    analyze_sentiment, classify_text, zero_shot_classify, CandleSequenceClassifier,
    SentimentRequest, TextClassificationLocalModelOptions, TextClassificationRequest,
    TextClassificationRuntime, ZeroShotClassificationRequest,
};

#[test]
#[ignore = "downloads the DistilBERT SST-2 bundle and compares batched with per-text inference"]
fn batched_classification_matches_per_text() {
    let spec = ModelPreset::DistilbertSst2.spec();
    let revision = spec.revision_value().unwrap_or("main").to_string();
    let store = ModelBundleStore::new(
        std::env::var_os("TEXT_CLASSIFICATION_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(workspace_model_root),
    )
    .downloader(HuggingFaceDownloader::new().progress(false).max_retries(2));
    let bundle = store
        .load(&spec.name, &revision)
        .or_else(|_| store.download(&spec))
        .unwrap();
    let classifier = CandleSequenceClassifier::from_bundle(bundle).unwrap();
    let texts = [
        "Short.",
        "This movie was a complete waste of time.",
        "Rust text workflows are reliable and useful.",
        "I expected little, but the second half surprised me with sharp writing and a terrific score.",
        "Not bad at all, though the ending dragged.",
    ];
    let batched = classifier.classify_batch(&texts).unwrap();
    for (text, batch) in texts.iter().zip(&batched) {
        let single = classifier.classify(text).unwrap();
        assert_eq!(single.len(), batch.len(), "{text}");
        for (single, batch) in single.iter().zip(batch) {
            // Labels and their ranking (argmax first) match exactly; scores may differ in the
            // last float bits because a batch changes the matrix-multiply shapes.
            assert_eq!(single.label, batch.label, "{text}");
            let (single, batch) = (single.score.unwrap(), batch.score.unwrap());
            assert!(
                (single - batch).abs() <= 1e-5,
                "{text}: {single} vs {batch}"
            );
        }
    }
}

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
