#![cfg(feature = "external-tests")]

use std::path::PathBuf;

use text_model_runtime::{validate_text_model_bundle, TextModelBundleCheck, TextModelCapability};

#[test]
#[ignore = "validates opt-in dslim/bert-base-NER bundle from .model-runtime"]
fn bert_base_ner_reports_loadability() {
    let report = validate_text_model_bundle(
        TextModelBundleCheck::new(
            "bert-base-ner",
            TextModelCapability::TokenClassification,
            PathBuf::from(".model-runtime")
                .join("bert-base-ner")
                .join("main"),
            ["config.json", "vocab.txt", "model.safetensors"],
        )
        .required_feature("candle,model-bundles")
        .required_setup("scripts/sync_model_bundles.sh text")
        .smoke_operation("linguistics.entities"),
    );

    assert_eq!(report.supported, true);
    assert_eq!(report.loadable, report.bundle_present());
}
