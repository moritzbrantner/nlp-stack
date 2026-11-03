//! Events and relations have no model backend yet (#186 removed the heuristic
//! extractor), so requesting them fails instead of returning rule-based guesses.

use text_linguistics::{analyze_text, LinguisticAnalysisOptions};

#[test]
fn event_requests_fail_until_a_model_backend_exists() {
    let options = LinguisticAnalysisOptions {
        enable_events: true,
        ..LinguisticAnalysisOptions::without_models()
    };
    let error = analyze_text("Alice visited Berlin.", &options).expect_err("model required");
    assert!(error.to_string().contains("model required"), "{error}");
}

#[test]
fn analyses_without_event_requests_carry_no_events_or_relations() {
    let analysis = analyze_text(
        "Alice visited Berlin.",
        &LinguisticAnalysisOptions::without_models(),
    )
    .expect("model-free analysis");
    assert!(analysis.events.is_empty());
    assert!(analysis.relations.is_empty());
}
