#![cfg(feature = "stanza")]

//! Opt-in checks against a running Stanza sidecar (`sidecars/stanza/serve`). They are
//! ignored by default so CI stays offline:
//!
//! ```sh
//! bash sidecars/stanza/setup.sh && sidecars/stanza/serve &
//! cargo test -p moenarch-text-linguistics --features stanza \
//!     --test stanza_sidecar_external -- --ignored
//! ```
//!
//! `NLP_STANZA_URL` overrides the endpoint. With `NLP_STANZA_RECORD=1` the sample's
//! verified response is written to `tests/fixtures/stanza/sample-en-de-es.json`, the
//! recording the offline tests replay.

use text_linguistics::{
    analyze_text_with_syntax_backend, syntax_backend_requests, DependencyParsingOptions,
    LinguisticAnalysisOptions, LinguisticOperation, MorphologyOptions, OperationSupport, PosTag,
    StanzaSidecar, SyntaxBackend, SyntaxRequest, STANZA_VERSION,
};

const SAMPLE: &str = include_str!("fixtures/stanza/sample-en-de-es.txt");

fn options() -> LinguisticAnalysisOptions {
    LinguisticAnalysisOptions {
        morphology: MorphologyOptions::syntax_backend(),
        dependency_parsing: DependencyParsingOptions::syntax_backend(),
        ..LinguisticAnalysisOptions::without_models()
    }
}

#[test]
#[ignore = "needs a running Stanza sidecar (sidecars/stanza/serve)"]
fn sidecar_reports_pinned_versions() {
    let health = StanzaSidecar::from_env().unwrap().health().unwrap();
    assert_eq!(health.stanza_version, STANZA_VERSION);
    for language in ["en", "de", "es"] {
        let language = &health.languages[language];
        assert!(language.ready, "{:?}", language.problems);
        assert!(language.revision.is_some());
    }
}

#[test]
#[ignore = "needs a running Stanza sidecar (sidecars/stanza/serve)"]
fn sidecar_analyzes_the_en_de_es_sample() {
    let sidecar = StanzaSidecar::from_env().unwrap();
    let options = options();
    let requests = syntax_backend_requests(SAMPLE, &options, &|language| {
        sidecar.supports_language(language)
    });
    let languages = requests
        .iter()
        .map(|request| request.language.as_str())
        .collect::<Vec<_>>();
    assert_eq!(languages, ["en", "de", "es"]);
    let (_, body) = sidecar
        .analyze_with_body(&SyntaxRequest {
            text: SAMPLE,
            sentences: &requests,
        })
        .unwrap();
    if std::env::var_os("NLP_STANZA_RECORD").is_some() {
        let pretty = serde_json::to_string_pretty(
            &serde_json::from_str::<serde_json::Value>(&body).unwrap(),
        )
        .unwrap();
        std::fs::write(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/stanza/sample-en-de-es.json"
            ),
            pretty + "\n",
        )
        .unwrap();
    }

    let analysis = analyze_text_with_syntax_backend(SAMPLE, &options, &sidecar).unwrap();
    assert_eq!(analysis.syntax.len(), 3);
    for sentence in &analysis.sentence_analyses {
        for operation in [
            LinguisticOperation::Morphology,
            LinguisticOperation::PartOfSpeech,
            LinguisticOperation::DependencySyntax,
        ] {
            assert_eq!(
                sentence.operation(operation).unwrap().support,
                OperationSupport::Supported
            );
        }
    }
    let zum = analysis
        .tokens
        .iter()
        .position(|token| token.text == "zum")
        .unwrap();
    let pos = analysis
        .pos
        .iter()
        .find(|annotation| annotation.token_index == zum)
        .unwrap();
    assert_eq!(pos.tag, PosTag::Adp);
}
