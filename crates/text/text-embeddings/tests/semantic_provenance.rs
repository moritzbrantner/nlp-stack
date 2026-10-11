//! Acceptance for issue #16 (A2): embedding producers report semantic core
//! provenance while concrete runtime/backend facts stay in
//! `TextEmbeddingMetadata.backend`.
//!
//! The ONNX/Candle/CUDA embedders only implement `TextEmbeddingBackend` behind
//! the non-default `tokenizers` feature and need model bundles, so their
//! `Model` mapping is not exercised here.

use text_core::{AnnotationProvenance, Result};
use text_embeddings::{
    DenseVector, HashedTextEmbedder, TextEmbeddingBackend, TextEmbeddingBackendKind,
    TextEmbeddingMetadata,
};

#[test]
fn hashed_embedder_reports_heuristic_provenance_with_backend_fact_in_metadata() {
    let metadata = HashedTextEmbedder::default().metadata();
    assert_eq!(metadata.provenance, AnnotationProvenance::Heuristic);
    assert_eq!(metadata.backend, TextEmbeddingBackendKind::Hashed);
}

#[test]
fn default_embedding_metadata_stays_derived() {
    assert_eq!(
        TextEmbeddingMetadata::default().provenance,
        AnnotationProvenance::Derived
    );
}

/// A backend serving vectors supplied from outside the process.
struct SuppliedVectors;

impl TextEmbeddingBackend for SuppliedVectors {
    fn embed_text(&self, _text: &str) -> Result<DenseVector> {
        DenseVector::new([1.0, 0.0])
    }

    fn metadata(&self) -> TextEmbeddingMetadata {
        TextEmbeddingMetadata {
            backend: TextEmbeddingBackendKind::External,
            provenance: AnnotationProvenance::Imported,
            model_name: Some("supplied".to_string()),
            dimensions: Some(2),
        }
    }
}

#[test]
fn externally_supplied_vectors_use_imported_provenance_and_external_backend_fact() {
    // The "external" fact lives in the capability's backend kind; core
    // provenance only says the vectors were imported.
    let metadata = SuppliedVectors.metadata();
    assert_eq!(metadata.provenance, AnnotationProvenance::Imported);
    assert_eq!(metadata.backend, TextEmbeddingBackendKind::External);
}

#[test]
fn concrete_runtime_names_remain_capability_backend_kinds() {
    for (kind, name) in [
        (TextEmbeddingBackendKind::Onnx, "\"onnx\""),
        (TextEmbeddingBackendKind::Candle, "\"candle\""),
        (TextEmbeddingBackendKind::CudaOxide, "\"cuda_oxide\""),
        (TextEmbeddingBackendKind::External, "\"external\""),
    ] {
        assert_eq!(serde_json::to_string(&kind).unwrap(), name);
    }
}
