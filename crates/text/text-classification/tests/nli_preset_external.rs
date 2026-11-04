//! Opt-in check of the pinned multilingual NLI preset.
//!
//! Run with `cargo test -p moenarch-text-classification --features onnx --release
//! --test nli_preset_external -- --ignored --nocapture`. The bundle lives in
//! `TEXT_CLASSIFICATION_MODEL_DIR` (default `.model-runtime`) and is downloaded at the
//! pinned revision when missing. ONNX Runtime comes from `ORT_DYLIB_PATH`.
#![cfg(feature = "onnx")]

use text_classification::{load_nli_preset, NliLabel, NliPreset, NliPresetLoadOptions};

#[test]
#[ignore = "downloads MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (1.1 GB ONNX) at the pinned revision"]
fn mdeberta_nli_judges_english_german_and_spanish_pairs() {
    let options = NliPresetLoadOptions {
        auto_download: true,
        download_progress: true,
        ..NliPresetLoadOptions::default()
    };
    let mut nli = load_nli_preset(NliPreset::MdebertaV3BaseXnliMultilingual, &options).unwrap();
    let cases = [
        (
            "The world has no explanation.",
            "The world has a cause or explanation beyond itself.",
            NliLabel::Contradiction,
        ),
        ("God exists.", "God exists.", NliLabel::Entailment),
        (
            "Die Welt hat keine Ursache.",
            "Die Welt hat eine Ursache.",
            NliLabel::Contradiction,
        ),
        (
            "Gott existiert, und man kann es beweisen.",
            "Gott existiert.",
            NliLabel::Entailment,
        ),
        (
            "El mundo tiene una causa.",
            "Hoy llueve en Madrid.",
            NliLabel::Neutral,
        ),
    ];
    for (premise, hypothesis, expected) in cases {
        let prediction = nli.classify(premise, hypothesis).unwrap();
        println!("{premise} => {hypothesis}: {prediction:?}");
        assert_eq!(prediction.label, expected, "{premise} / {hypothesis}");
        let sum = prediction.entailment + prediction.neutral + prediction.contradiction;
        assert!((sum - 1.0).abs() < 1e-4);
    }
}

#[test]
fn a_missing_bundle_fails_without_downloading() {
    let root = tempfile::tempdir().unwrap();
    let options = NliPresetLoadOptions {
        bundle_root: root.path().to_path_buf(),
        ..NliPresetLoadOptions::default()
    };
    let error = load_nli_preset(NliPreset::MdebertaV3BaseXnliMultilingual, &options).unwrap_err();
    assert!(error.to_string().contains("is not available"), "{error}");
}
