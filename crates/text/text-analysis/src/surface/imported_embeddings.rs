pub(super) use text_embeddings::{
    ImportedEmbedder, ImportedEmbedding as ImportedSemanticEmbedding, ImportedEmbeddingModel,
};

/// The surface cannot load an embedding model; semantic operations need the
/// caller's vectors and fail without them instead of using a hashed baseline.
pub(super) fn embedding_model_required(operation: &str) -> String {
    runtime_core::SurfaceError::missing_dependency(
        Some(operation),
        "embedding model",
        "supply `importedEmbeddings` (one vector per semantic unit) from a real embedding model; the hashed baseline was removed",
    )
    .to_error_string()
}
