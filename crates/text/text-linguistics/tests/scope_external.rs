#![cfg(feature = "external-tests")]

//! Opt-in: negation, uncertainty and hypothetical scope read from the en/de/es UD
//! model parses. Bundles come from `TEXT_LINGUISTICS_MODEL_DIR` (default
//! `.model-runtime`).

use std::path::PathBuf;

use text_linguistics::{
    analyze_text, reporting_constructions, scope_operators, DependencyParsingOptions,
    LinguisticAnalysisOptions, MorphologyOptions, OperationSupport, ReportingEvidence,
    ScopeEvidence, ScopeOperatorKind, ScopeResolution,
};

fn analyze(text: &str) -> (ReportingEvidence, ScopeEvidence) {
    let bundle_dir = std::env::var_os("TEXT_LINGUISTICS_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".model-runtime"));
    let options = LinguisticAnalysisOptions {
        morphology: MorphologyOptions {
            bundle_dir: bundle_dir.clone(),
            ..MorphologyOptions::local_model()
        },
        dependency_parsing: DependencyParsingOptions {
            bundle_dir,
            ..DependencyParsingOptions::local_model()
        },
        ..LinguisticAnalysisOptions::default()
    };
    let analysis = analyze_text(text, &options).unwrap();
    let reporting = reporting_constructions(&analysis);
    let scope = scope_operators(&analysis, &reporting);
    for item in &scope.operators {
        for passage in [
            Some(&item.cue),
            item.head.as_ref(),
            item.scope.as_ref(),
            item.consequent.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            assert_eq!(
                &text[passage.span.byte_start..passage.span.byte_end],
                passage.text
            );
        }
    }
    (reporting, scope)
}

/// (kind, cue, head, scope) of one operator.
fn shape(
    scope: &ScopeEvidence,
) -> Vec<(ScopeOperatorKind, String, Option<String>, Option<String>)> {
    scope
        .operators
        .iter()
        .map(|item| {
            (
                item.kind,
                item.cue.text.clone(),
                item.head.as_ref().map(|head| head.text.clone()),
                item.scope.as_ref().map(|scope| scope.text.clone()),
            )
        })
        .collect()
}

#[test]
fn english_negation_relative_to_reports() {
    let (reporting, scope) = analyze("Sarah says John does not believe the river is polluted.");
    assert_eq!(scope.sentences[0].support, OperationSupport::Supported);
    let [negation] = scope.operators.as_slice() else {
        panic!("{:?}", shape(&scope));
    };
    assert_eq!(negation.kind, ScopeOperatorKind::Negation);
    assert_eq!(negation.head.as_ref().unwrap().text, "believe");
    let target = &reporting.constructions[negation.reporting_target.unwrap()];
    assert_eq!(target.cue.text, "believe");
    let within = &reporting.constructions[negation.reported_within.unwrap()];
    assert_eq!(within.cue.text, "says");

    let (reporting, scope) = analyze("Sarah does not say John believes the river is polluted.");
    let [negation] = scope.operators.as_slice() else {
        panic!("{:?}", shape(&scope));
    };
    assert_eq!(negation.head.as_ref().unwrap().text, "say");
    assert_eq!(
        reporting.constructions[negation.reporting_target.unwrap()]
            .cue
            .text,
        "say"
    );
    assert_eq!(negation.reported_within, None);
}

#[test]
fn english_uncertainty_and_conditional() {
    let (_, scope) = analyze("John might leave.");
    assert_eq!(
        shape(&scope),
        vec![(
            ScopeOperatorKind::Uncertainty,
            "might".into(),
            Some("leave".into()),
            Some("John might leave".into())
        )]
    );

    let (_, scope) = analyze("If John leaves, Sarah will call.");
    let [conditional] = scope.operators.as_slice() else {
        panic!("{:?}", shape(&scope));
    };
    assert_eq!(conditional.kind, ScopeOperatorKind::Hypothetical);
    assert_eq!(conditional.resolution, ScopeResolution::Resolved);
    assert_eq!(conditional.scope.as_ref().unwrap().text, "John leaves");
    assert_eq!(
        conditional.consequent.as_ref().unwrap().text,
        "Sarah will call"
    );
}

#[test]
fn german_and_spanish_negation_and_hedges() {
    let (_, scope) = analyze("Jürgen glaubt nicht, dass der Fluss verschmutzt ist.");
    let negation = scope
        .operators
        .iter()
        .find(|item| item.kind == ScopeOperatorKind::Negation)
        .unwrap_or_else(|| panic!("{:?}", shape(&scope)));
    assert_eq!(negation.cue.text, "nicht");
    assert_eq!(negation.head.as_ref().unwrap().text, "glaubt");
    assert!(negation.reporting_target.is_some(), "{negation:?}");

    let (_, scope) = analyze("Quizás José no salga.");
    let kinds = shape(&scope);
    assert!(
        kinds
            .iter()
            .any(|item| item.0 == ScopeOperatorKind::Uncertainty && item.1 == "Quizás"),
        "{kinds:?}"
    );
    assert!(
        kinds
            .iter()
            .any(|item| item.0 == ScopeOperatorKind::Negation && item.1 == "no"),
        "{kinds:?}"
    );
}
