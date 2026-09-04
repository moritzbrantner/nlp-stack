use runtime_core::{OperationId, SurfaceRequest};
use text_core::TextSpan;
use text_linguistics::{surface::run_surface_operation, TextNlpConfig, TextNlpPipeline};

fn assert_span(text: &str, span: TextSpan, expected: &str) {
    assert_eq!(&text[span.byte_start..span.byte_end], expected);
    let chars = text
        .chars()
        .skip(span.char_start)
        .take(span.char_end - span.char_start)
        .collect::<String>();
    assert_eq!(chars, expected);
}

#[test]
fn events_and_relations_retain_source_spans() {
    let text = "Alice visited Berlin.";
    let analysis = TextNlpPipeline::new(TextNlpConfig::rich())
        .analyze_text(text)
        .expect("linguistic analysis");

    let event = analysis.events.first().expect("event");
    assert_eq!(event.sentence_index, 0);
    assert_span(text, event.predicate_span, &event.predicate);
    for argument in &event.arguments {
        assert_span(text, argument.span, &argument.text);
    }

    let relation = analysis.relations.first().expect("relation");
    assert_eq!(relation.sentence_index, event.sentence_index);
    assert_span(text, relation.subject_span, &relation.subject);
    assert_span(text, relation.relation_span, &event.predicate);
    assert_span(text, relation.object_span, &relation.object);
}

#[test]
fn entity_surface_exposes_event_and_relation_provenance() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("linguistics.entities"),
        input: serde_json::json!({"text": "Alice visited Berlin.", "profile": "rich"}),
    })
    .expect("entity surface");
    let result = &response.value["result"];
    assert!(result["events"][0]["predicateSpan"].is_object());
    assert!(result["events"][0]["arguments"][0]["span"].is_object());
    assert_eq!(result["relations"][0]["sentenceIndex"], 0);
    assert!(result["relations"][0]["subjectSpan"].is_object());
    assert!(result["relations"][0]["relationSpan"].is_object());
    assert!(result["relations"][0]["objectSpan"].is_object());
}
