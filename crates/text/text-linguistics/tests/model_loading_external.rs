#![cfg(feature = "external-tests")]

use std::path::PathBuf;

use model_runtime::ModelBundle;
use text_linguistics::{CandleTokenClassifier, SequenceLabeler};

#[test]
#[ignore = "loads the opt-in dslim/bert-base-NER bundle synced by scripts/sync_model_bundles.sh text"]
fn bert_base_ner_loads_and_labels_tokens() {
    let bundle = ModelBundle::load(
        PathBuf::from(".model-runtime")
            .join("bert-base-ner")
            .join("main"),
    )
    .unwrap();
    let mut labeler = CandleTokenClassifier::from_bundle(bundle).unwrap();
    assert!(!labeler.labels().is_empty());
    assert!(!labeler
        .label_text("Ada Lovelace lived in London.")
        .unwrap()
        .is_empty());
}
