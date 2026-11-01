//! Opt-in sentence language detection scores on UD 2.17 test splits.
//!
//! `UD_TREEBANK_DIR` holds any of `en_ewt-ud-test.conllu`, `de_gsd-ud-test.conllu`,
//! `es_ancora-ud-test.conllu`, `fr_gsd-ud-test.conllu`, `it_isdt-ud-test.conllu` and
//! `pt_bosque-ud-test.conllu`; missing files are skipped.

use std::path::PathBuf;

use text_linguistics::{LanguageDecision, LanguageDetector};

/// Minimum share of decided en/de/es sentences that are decided correctly.
const MIN_DECIDED_PRECISION: f64 = 0.98;

#[test]
#[ignore = "needs UD 2.17 test splits in UD_TREEBANK_DIR"]
fn sentence_languages_on_ud_test_splits() {
    let dir = PathBuf::from(std::env::var_os("UD_TREEBANK_DIR").expect("UD_TREEBANK_DIR"));
    let detector = LanguageDetector::default();
    let mut scored = 0;
    for (language, file) in [
        ("en", "en_ewt-ud-test.conllu"),
        ("de", "de_gsd-ud-test.conllu"),
        ("es", "es_ancora-ud-test.conllu"),
        ("fr", "fr_gsd-ud-test.conllu"),
        ("it", "it_isdt-ud-test.conllu"),
        ("pt", "pt_bosque-ud-test.conllu"),
    ] {
        let Ok(conllu) = std::fs::read_to_string(dir.join(file)) else {
            eprintln!("{file}: missing, skipped");
            continue;
        };
        let (mut sentences, mut decided, mut correct) = (0usize, 0usize, 0usize);
        for text in conllu
            .lines()
            .filter_map(|line| line.strip_prefix("# text = "))
        {
            sentences += 1;
            // Each UD sentence as one unit, independent of this crate's splitter.
            let (decision, candidates) = detector.classify(text);
            if decision == LanguageDecision::Detected {
                decided += 1;
                correct += usize::from(candidates[0].language == language);
            }
        }
        let precision = correct as f64 / decided.max(1) as f64;
        eprintln!(
            "{language} {file}: sentences={sentences} decided={:.3} decided-correct={precision:.4} misrouted={}",
            decided as f64 / sentences.max(1) as f64,
            decided - correct
        );
        if ["en", "de", "es"].contains(&language) {
            assert!(precision >= MIN_DECIDED_PRECISION, "{language} precision");
        }
        scored += 1;
    }
    assert!(scored > 0, "no UD test split found in UD_TREEBANK_DIR");
}
