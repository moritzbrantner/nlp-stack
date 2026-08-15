#![cfg(feature = "external-tests")]

use std::path::PathBuf;

use text_model_runtime::{
    validate_text_model_bundle, TextModelBundleCheck, TextModelCapability, TokenizerPreset,
};

#[test]
#[ignore = "validates opt-in tokenizer bundles from .model-runtime"]
fn tokenizer_presets_have_honest_load_reports() {
    let cache = PathBuf::from(".model-runtime");
    for preset in TokenizerPreset::ALL {
        let tokenizer_file = match preset {
            TokenizerPreset::DistilbertSst2 => "vocab.txt",
            _ => "tokenizer.json",
        };
        let report = validate_text_model_bundle(
            TextModelBundleCheck::new(
                preset.as_str(),
                TextModelCapability::Tokenizer,
                cache.join(preset.as_str()).join("main"),
                [tokenizer_file],
            )
            .required_feature("tokenizers,model-bundles")
            .required_setup("scripts/sync_model_bundles.sh text")
            .smoke_operation("runtime.tokenizeSummary"),
        );
        assert_eq!(report.supported, true);
        assert_eq!(report.loadable, report.bundle_present());
    }
}
