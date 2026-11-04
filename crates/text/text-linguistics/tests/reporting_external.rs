#![cfg(feature = "external-tests")]

//! Opt-in: nested reporting constructions read from the en/de/es UD model parses.
//! Bundles come from `TEXT_LINGUISTICS_MODEL_DIR` (default `.model-runtime`).

use std::path::PathBuf;

use text_linguistics::{
    analyze_text, reporting_constructions, DependencyParsingOptions, LinguisticAnalysisOptions,
    MorphologyOptions, OperationSupport, ReportingEvidence, ReportingForm, ReportingKind,
};

fn analyze(text: &str) -> ReportingEvidence {
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
        ..LinguisticAnalysisOptions::without_models()
    };
    let analysis = analyze_text(text, &options).unwrap();
    let evidence = reporting_constructions(&analysis);
    for item in &evidence.constructions {
        assert_eq!(
            &text[item.content.span.byte_start..item.content.span.byte_end],
            item.content.text
        );
        assert_eq!(
            &text[item.cue.span.byte_start..item.cue.span.byte_end],
            item.cue.text
        );
    }
    evidence
}

/// (kind, source, cue, content, parent) of one construction.
type Shape = (ReportingKind, Option<String>, String, String, Option<usize>);

fn shape(evidence: &ReportingEvidence) -> Vec<Shape> {
    evidence
        .constructions
        .iter()
        .map(|item| {
            (
                item.kind,
                item.source.as_ref().map(|source| source.text.clone()),
                item.cue.text.clone(),
                item.content.text.clone(),
                item.parent,
            )
        })
        .collect()
}

#[test]
fn english_nested_report_and_belief() {
    let evidence = analyze("Sarah reports that John believes the river is polluted.");
    assert_eq!(evidence.sentences[0].support, OperationSupport::Supported);
    assert_eq!(
        shape(&evidence),
        vec![
            (
                ReportingKind::Speech,
                Some("Sarah".into()),
                "reports".into(),
                "John believes the river is polluted".into(),
                None
            ),
            (
                ReportingKind::Belief,
                Some("John".into()),
                "believes".into(),
                "the river is polluted".into(),
                Some(0)
            ),
        ]
    );
    assert_eq!(evidence.constructions[0].children, vec![1]);
    assert_eq!(
        evidence.constructions[0]
            .complementizer
            .as_ref()
            .unwrap()
            .text,
        "that"
    );
}

#[test]
fn english_plain_statement_and_direct_quote() {
    assert!(analyze("The river is polluted.").constructions.is_empty());
    let evidence = analyze("John said, \"The river is polluted.\"");
    let shape = shape(&evidence);
    assert_eq!(shape.len(), 1, "{shape:?}");
    assert_eq!(shape[0].0, ReportingKind::Speech);
    assert_eq!(
        evidence.constructions[0].form,
        ReportingForm::Direct,
        "{:?}",
        evidence.constructions[0]
    );
}

#[test]
fn german_and_spanish_nested_reports() {
    let german = analyze("Jürgen sagt, dass Zoë glaubt, dass der Fluss verschmutzt ist.");
    let shape_de = shape(&german);
    assert_eq!(shape_de.len(), 2, "{shape_de:?}");
    assert_eq!(shape_de[0].0, ReportingKind::Speech);
    assert_eq!(shape_de[1].0, ReportingKind::Belief);
    assert_eq!(shape_de[1].1.as_deref(), Some("Zoë"));
    assert_eq!(shape_de[1].3, "der Fluss verschmutzt ist");
    assert_eq!(shape_de[1].4, Some(0));

    let spanish = analyze("María dice que José cree que el río está contaminado.");
    let shape_es = shape(&spanish);
    assert_eq!(shape_es.len(), 2, "{shape_es:?}");
    assert_eq!(shape_es[1].1.as_deref(), Some("José"));
    assert_eq!(shape_es[1].3, "el río está contaminado");
    assert_eq!(shape_es[1].4, Some(0));
}
