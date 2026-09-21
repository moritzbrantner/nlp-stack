use std::collections::BTreeMap;

use serde::Deserialize;
use text_core::AnnotationProvenance;
use text_embeddings::{
    DenseVector, EmbeddingModelInfo, TextEmbeddingBackend, TextEmbeddingBackendKind,
    TextEmbeddingMetadata,
};

use crate::invalid_argument;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ImportedSemanticEmbedding {
    text: String,
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ImportedEmbeddingModel {
    name: String,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Debug)]
pub(super) struct ImportedSemanticEmbedder {
    vectors: BTreeMap<String, Vec<f32>>,
    model_name: String,
    dimensions: usize,
    max_tokens: Option<usize>,
}

impl ImportedSemanticEmbedder {
    pub(super) fn new(
        embeddings: &[ImportedSemanticEmbedding],
        model: Option<&ImportedEmbeddingModel>,
    ) -> Result<Self, String> {
        let dimensions = embeddings
            .first()
            .map(|embedding| embedding.vector.len())
            .unwrap_or_default();
        if dimensions == 0 {
            return Err("imported semantic embeddings must contain non-empty vectors".to_string());
        }
        if let Some(declared) = model.and_then(|model| model.dimensions) {
            if declared != dimensions {
                return Err(format!(
                    "imported semantic embedding dimensions declared {declared} but vectors contain {dimensions} values"
                ));
            }
        }

        let mut vectors = BTreeMap::new();
        for embedding in embeddings {
            if embedding.vector.len() != dimensions {
                return Err(format!(
                    "imported semantic embeddings must all use {dimensions} dimensions"
                ));
            }
            let vector = normalize_imported_vector(&embedding.vector)?;
            if let Some(existing) = vectors.get(&embedding.text) {
                if existing != &vector {
                    return Err(
                        "duplicate imported semantic text supplied with conflicting vectors"
                            .to_string(),
                    );
                }
            } else {
                vectors.insert(embedding.text.clone(), vector);
            }
        }

        let model_name = model
            .map(|model| model.name.trim())
            .filter(|name| !name.is_empty())
            .unwrap_or("external-semantic-embedder")
            .to_string();
        Ok(Self {
            vectors,
            model_name,
            dimensions,
            max_tokens: model.and_then(|model| model.max_tokens),
        })
    }
}

fn normalize_imported_vector(values: &[f32]) -> Result<Vec<f32>, String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("imported semantic embeddings must contain only finite values".to_string());
    }
    // Every finite f32 square (including subnormals) fits in f64. This avoids
    // both overflow and an arbitrary lower magnitude cutoff before normalization.
    let squared_norm = values
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>();
    if !squared_norm.is_finite() || squared_norm == 0.0 {
        return Err("imported semantic embeddings must have a finite non-zero norm".to_string());
    }
    let norm = squared_norm.sqrt();
    Ok(values
        .iter()
        .map(|&value| (f64::from(value) / norm) as f32)
        .collect())
}

impl TextEmbeddingBackend for ImportedSemanticEmbedder {
    fn embed_text(&self, text: &str) -> text_core::Result<DenseVector> {
        let vector = self.vectors.get(text).ok_or_else(|| {
            invalid_argument(format!(
                "imported semantic embeddings are missing an exact vector for `{text}`"
            ))
        })?;
        DenseVector::new(vector.clone())
    }

    fn metadata(&self) -> TextEmbeddingMetadata {
        TextEmbeddingMetadata {
            backend: TextEmbeddingBackendKind::External,
            provenance: AnnotationProvenance::Derived,
            model_name: Some(self.model_name.clone()),
            dimensions: Some(self.dimensions),
        }
    }

    fn model_info(&self) -> EmbeddingModelInfo {
        EmbeddingModelInfo {
            model_name: self.model_name.clone(),
            backend: TextEmbeddingBackendKind::External,
            dimensions: self.dimensions,
            normalized: true,
            max_tokens: self.max_tokens,
        }
    }
}
