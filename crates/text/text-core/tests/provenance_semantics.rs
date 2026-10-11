//! Acceptance for issue #16 (A2): core provenance is semantic, not
//! implementation-specific. Concrete ML runtimes/backends (ONNX, Candle,
//! CUDA, model tokenizers, ...) belong to the producing capability's
//! execution metadata, never to `text_core::AnnotationProvenance`.

use text_core::{build_annotation_graph, AnnotationProvenance, TextProcessingOptions};

const SEMANTIC_VARIANTS: &[(AnnotationProvenance, &str)] = &[
    (AnnotationProvenance::Observed, "Observed"),
    (AnnotationProvenance::Heuristic, "Heuristic"),
    (AnnotationProvenance::Model, "Model"),
    (AnnotationProvenance::Derived, "Derived"),
    (AnnotationProvenance::Imported, "Imported"),
];

const REMOVED_RUNTIME_VARIANTS: &[&str] = &["Onnx", "Candle", "CudaOxide", "Tokenizer", "External"];

/// Compile-time exhaustiveness: adding any variant (for example a runtime
/// name) to `AnnotationProvenance` breaks this match and must be a
/// deliberate architecture decision.
fn semantic_name(provenance: AnnotationProvenance) -> &'static str {
    match provenance {
        AnnotationProvenance::Observed => "Observed",
        AnnotationProvenance::Heuristic => "Heuristic",
        AnnotationProvenance::Model => "Model",
        AnnotationProvenance::Derived => "Derived",
        AnnotationProvenance::Imported => "Imported",
    }
}

#[test]
fn semantic_provenance_variants_round_trip_as_plain_json_strings() {
    for &(variant, name) in SEMANTIC_VARIANTS {
        assert_eq!(semantic_name(variant), name);

        let json = serde_json::to_string(&variant).unwrap();
        assert_eq!(json, format!("\"{name}\""), "serialized form of {name}");

        let decoded: AnnotationProvenance = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, variant, "round trip of {name}");
    }
}

#[test]
fn runtime_specific_provenance_names_are_rejected() {
    for name in REMOVED_RUNTIME_VARIANTS {
        let json = format!("\"{name}\"");
        assert!(
            serde_json::from_str::<AnnotationProvenance>(&json).is_err(),
            "runtime/backend provenance `{name}` must not deserialize into core provenance"
        );
    }
}

#[test]
fn deterministic_kernel_annotation_graph_is_not_model_or_imported_provenance() {
    // The text-core graph comes from the deterministic rule tokenizer; it must
    // not claim model or imported provenance once `Tokenizer` is gone.
    let graph = build_annotation_graph(
        "Alice launched the API.\n\nBerlin hosted the event.",
        &TextProcessingOptions::default(),
    );
    assert_ne!(graph.provenance, AnnotationProvenance::Model);
    assert_ne!(graph.provenance, AnnotationProvenance::Imported);
}
