#![cfg(feature = "stanza")]

//! Offline replay of a recorded Stanza sidecar response through the pipeline
//! (`tests/fixtures/stanza`, recorded by `stanza_sidecar_external`).

use text_linguistics::{
    analyze_text, analyze_text_with_syntax_backend, syntax_backend_requests,
    DependencyParsingOptions, DependencyRelation, LinguisticAnalysis, LinguisticAnalysisError,
    LinguisticAnalysisOptions, LinguisticOperation, MorphologyOptions, OperationSupport, PosTag,
    RecordedStanzaResponse, SyntaxBackend, SyntaxBackendError, STANZA_LANGUAGES,
};

const SAMPLE: &str = include_str!("fixtures/stanza/sample-en-de-es.txt");
const RECORDED: &str = include_str!("fixtures/stanza/sample-en-de-es.json");

fn options() -> LinguisticAnalysisOptions {
    LinguisticAnalysisOptions {
        morphology: MorphologyOptions::syntax_backend(),
        dependency_parsing: DependencyParsingOptions::syntax_backend(),
        ..LinguisticAnalysisOptions::without_models()
    }
}

fn analyze(text: &str) -> Result<LinguisticAnalysis, LinguisticAnalysisError> {
    analyze_text_with_syntax_backend(text, &options(), &RecordedStanzaResponse::new(RECORDED))
}

fn token(analysis: &LinguisticAnalysis, text: &str) -> usize {
    analysis
        .tokens
        .iter()
        .position(|token| token.text == text)
        .unwrap_or_else(|| panic!("no token {text:?}"))
}

#[test]
fn recorded_sample_gives_real_syntax_for_en_de_es() {
    let analysis = analyze(SAMPLE).unwrap();
    let languages = analysis
        .syntax
        .iter()
        .map(|sentence| sentence.language.as_str())
        .collect::<Vec<_>>();
    assert_eq!(languages, ["en", "de", "es"]);
    for (sentence, language) in analysis.sentence_analyses.iter().zip(["en", "de", "es"]) {
        for operation in [
            LinguisticOperation::Morphology,
            LinguisticOperation::PartOfSpeech,
            LinguisticOperation::DependencySyntax,
        ] {
            let report = sentence.operation(operation).unwrap();
            assert_eq!(report.support, OperationSupport::Supported);
            assert_eq!(
                report.producer.as_deref(),
                Some(format!("stanza-1.15.0:{language}/combined_charlm").as_str())
            );
        }
    }
    for sentence in &analysis.syntax {
        assert_eq!(sentence.provenance.backend_version, "1.15.0");
        assert!(sentence
            .provenance
            .models
            .iter()
            .any(|model| model.stage == "depparse" && model.revision.len() == 40));
    }

    // `zum` is one pipeline token over the words `zu` + `dem`.
    let zum = token(&analysis, "zum");
    let pos = analysis
        .pos
        .iter()
        .find(|pos| pos.token_index == zum)
        .unwrap();
    assert_eq!(pos.tag, PosTag::Adp);
    assert!(pos.reason.ends_with("words zu+dem"), "{}", pos.reason);
    let tree = &analysis.dependencies[1];
    let node = tree
        .nodes
        .iter()
        .find(|node| node.token_index == zum)
        .unwrap();
    assert_eq!(node.head_token_index, Some(token(&analysis, "Bahnhof")));
    assert_eq!(node.relation, DependencyRelation::Case);
    assert_eq!(tree.root_token_index, Some(token(&analysis, "glaubt")));

    // Non-ASCII spans stay exact UTF-8 byte spans.
    let spanish = &analysis.syntax[2];
    let nino = spanish.tokens.iter().find(|t| t.text == "niño").unwrap();
    assert_eq!(&SAMPLE[nino.span.byte_start..nino.span.byte_end], "niño");
    let maria = token(&analysis, "María");
    let lemma = analysis
        .lemmas
        .iter()
        .find(|l| l.token_index == maria)
        .unwrap();
    assert_eq!(lemma.value, "María");
    assert_eq!(lemma.language.as_deref(), Some("es"));
}

#[test]
fn sentences_in_unserved_languages_are_unsupported_and_not_sent() {
    let text = format!("{SAMPLE} Il pleut beaucoup aujourd'hui à Paris, dit-elle.");
    let requests = syntax_backend_requests(&text, &options(), &|language| {
        STANZA_LANGUAGES.contains(&language)
    });
    assert_eq!(requests.len(), 3);
    let analysis = analyze(&text).unwrap();
    let french = analysis.sentence_analyses.last().unwrap();
    assert_eq!(french.language.as_deref(), Some("fr"));
    let report = french.operation(LinguisticOperation::PartOfSpeech).unwrap();
    assert_eq!(report.support, OperationSupport::Unsupported);
    assert_eq!(
        report.reason,
        "the stanza syntax backend does not serve `fr`"
    );
}

#[test]
fn backend_mode_without_a_backend_requires_a_model() {
    let error = analyze_text(SAMPLE, &options()).unwrap_err();
    assert!(matches!(
        error,
        LinguisticAnalysisError::ModelRequired { ref capability, .. } if capability == "syntax backend"
    ));
}

#[test]
fn a_recording_of_other_text_is_refused() {
    let error = analyze(&SAMPLE.replace("Sarah", "Sandra")).unwrap_err();
    assert!(
        matches!(
            error,
            LinguisticAnalysisError::SyntaxBackend(SyntaxBackendError::Protocol { .. })
        ),
        "{error}"
    );
}

#[test]
fn recorded_backend_names_itself() {
    let backend = RecordedStanzaResponse::new(RECORDED);
    assert_eq!(backend.name(), "stanza");
    assert!(backend.supports_language("de") && !backend.supports_language("fr"));
}
