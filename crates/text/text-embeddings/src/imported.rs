//! Caller-supplied embeddings as a [`TextEmbeddingBackend`].
//!
//! Stateless surfaces (WASM, CLI JSON) cannot load an embedding model. Their callers
//! embed texts with a real model elsewhere (host-side, or a browser runtime) and pass
//! the vectors in; this backend looks each text up and never computes a vector itself.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use text_core::{AnnotationProvenance, Result};

use crate::{
    invalid_argument, DenseVector, EmbeddingInputKind, EmbeddingModelInfo, PoolingStrategy,
    TextEmbeddingBackend, TextEmbeddingBackendKind, TextEmbeddingMetadata,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// One caller-computed vector for an exact text.
pub struct ImportedEmbedding {
    /// The exact text the vector was computed for (without model prefixes).
    pub text: String,
    /// The vector.
    pub vector: Vec<f32>,
    /// The role the vector was computed for. `None` serves every role, which is
    /// right for symmetric models; asymmetric models (e5, Qwen3-Embedding) need
    /// `query` vectors for queries and `passage` vectors for indexed texts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<EmbeddingInputKind>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Provenance of caller-supplied vectors.
pub struct ImportedEmbeddingModel {
    /// Model id, for example `intfloat/multilingual-e5-base`.
    pub name: String,
    /// Declared vector dimensions; checked against the vectors.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Maximum token count of the model.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Model revision the vectors were computed with.
    #[serde(default)]
    pub revision: Option<String>,
    /// Pooling the vectors were computed with.
    #[serde(default)]
    pub pooling: Option<PoolingStrategy>,
}

#[derive(Debug, Clone, PartialEq)]
/// A backend that serves caller-supplied vectors and fails for any other text.
pub struct ImportedEmbedder {
    vectors: BTreeMap<(Option<EmbeddingInputKind>, String), Vec<f32>>,
    model: ImportedEmbeddingModel,
    dimensions: usize,
}

impl ImportedEmbedder {
    /// Validates and L2-normalizes `embeddings`. All vectors must be finite, non-zero
    /// and of one dimension, which must match `model.dimensions` when declared.
    pub fn new(
        embeddings: &[ImportedEmbedding],
        model: Option<&ImportedEmbeddingModel>,
    ) -> Result<Self> {
        let dimensions = embeddings
            .first()
            .map(|embedding| embedding.vector.len())
            .unwrap_or_default();
        if dimensions == 0 {
            return Err(invalid_argument(
                "imported semantic embeddings must contain non-empty vectors",
            ));
        }
        if let Some(declared) = model.and_then(|model| model.dimensions) {
            if declared != dimensions {
                return Err(invalid_argument(format!(
                    "imported semantic embedding dimensions declared {declared} but vectors contain {dimensions} values"
                )));
            }
        }
        let mut vectors = BTreeMap::new();
        for embedding in embeddings {
            if embedding.vector.len() != dimensions {
                return Err(invalid_argument(format!(
                    "imported semantic embeddings must all use {dimensions} dimensions"
                )));
            }
            let vector = normalize_imported_vector(&embedding.vector)?;
            let key = (embedding.kind, embedding.text.clone());
            match vectors.get(&key) {
                Some(existing) if existing != &vector => {
                    return Err(invalid_argument(
                        "duplicate imported semantic text supplied with conflicting vectors",
                    ));
                }
                Some(_) => {}
                None => {
                    vectors.insert(key, vector);
                }
            }
        }
        let mut model = model.cloned().unwrap_or_default();
        if model.name.trim().is_empty() {
            model.name = "external-semantic-embedder".to_string();
        } else {
            model.name = model.name.trim().to_string();
        }
        Ok(Self {
            vectors,
            model,
            dimensions,
        })
    }

    /// Vector dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn lookup(&self, text: &str, kind: EmbeddingInputKind) -> Result<DenseVector> {
        let vector = self
            .vectors
            .get(&(Some(kind), text.to_string()))
            .or_else(|| self.vectors.get(&(None, text.to_string())))
            .ok_or_else(|| {
                invalid_argument(format!(
                    "imported semantic embeddings are missing an exact vector for `{text}`{}",
                    match kind {
                        EmbeddingInputKind::Symmetric => String::new(),
                        EmbeddingInputKind::Query => " (as a query)".to_string(),
                        EmbeddingInputKind::Passage => " (as a passage)".to_string(),
                    }
                ))
            })?;
        DenseVector::new(vector.clone())
    }
}

fn normalize_imported_vector(values: &[f32]) -> Result<Vec<f32>> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_argument(
            "imported semantic embeddings must contain only finite values",
        ));
    }
    // Every finite f32 square (including subnormals) fits in f64. This avoids
    // both overflow and an arbitrary lower magnitude cutoff before normalization.
    let squared_norm = values
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>();
    if !squared_norm.is_finite() || squared_norm == 0.0 {
        return Err(invalid_argument(
            "imported semantic embeddings must have a finite non-zero norm",
        ));
    }
    let norm = squared_norm.sqrt();
    Ok(values
        .iter()
        .map(|&value| (f64::from(value) / norm) as f32)
        .collect())
}

impl TextEmbeddingBackend for ImportedEmbedder {
    fn embed_text(&self, text: &str) -> Result<DenseVector> {
        self.lookup(text, EmbeddingInputKind::Symmetric)
    }

    fn embed_inputs(&self, texts: &[&str], kind: EmbeddingInputKind) -> Result<Vec<DenseVector>> {
        texts.iter().map(|text| self.lookup(text, kind)).collect()
    }

    fn metadata(&self) -> TextEmbeddingMetadata {
        TextEmbeddingMetadata {
            backend: TextEmbeddingBackendKind::External,
            provenance: AnnotationProvenance::Imported,
            model_name: Some(self.model.name.clone()),
            dimensions: Some(self.dimensions),
        }
    }

    fn model_info(&self) -> EmbeddingModelInfo {
        EmbeddingModelInfo {
            model_name: self.model.name.clone(),
            backend: TextEmbeddingBackendKind::External,
            dimensions: self.dimensions,
            normalized: true,
            max_tokens: self.model.max_tokens,
            revision: self.model.revision.clone(),
            pooling: self.model.pooling,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(
        text: &str,
        vector: &[f32],
        kind: Option<EmbeddingInputKind>,
    ) -> ImportedEmbedding {
        ImportedEmbedding {
            text: text.to_string(),
            vector: vector.to_vec(),
            kind,
        }
    }

    #[test]
    fn serves_role_specific_vectors_before_shared_ones() {
        let embedder = ImportedEmbedder::new(
            &[
                embedding("berlin", &[1.0, 0.0], Some(EmbeddingInputKind::Query)),
                embedding("berlin", &[0.0, 2.0], None),
            ],
            Some(&ImportedEmbeddingModel {
                name: "intfloat/multilingual-e5-base".to_string(),
                dimensions: Some(2),
                revision: Some("d128750".to_string()),
                pooling: Some(PoolingStrategy::Mean),
                ..ImportedEmbeddingModel::default()
            }),
        )
        .unwrap();
        assert_eq!(
            embedder.embed_query("berlin").unwrap().as_slice(),
            &[1.0, 0.0]
        );
        assert_eq!(
            embedder.embed_passages(&["berlin"]).unwrap()[0].as_slice(),
            &[0.0, 1.0]
        );
        let info = embedder.model_info();
        assert_eq!(info.revision.as_deref(), Some("d128750"));
        assert_eq!(info.pooling, Some(PoolingStrategy::Mean));
        let error = embedder.embed_query("paris").unwrap_err().to_string();
        assert!(error.contains("(as a query)"), "{error}");
    }

    #[test]
    fn rejects_mismatched_dimensions_and_zero_vectors() {
        assert!(ImportedEmbedder::new(
            &[
                embedding("a", &[1.0, 0.0], None),
                embedding("b", &[1.0], None)
            ],
            None
        )
        .is_err());
        assert!(ImportedEmbedder::new(&[embedding("a", &[0.0, 0.0], None)], None).is_err());
        assert!(ImportedEmbedder::new(
            &[embedding("a", &[1.0, 0.0], None)],
            Some(&ImportedEmbeddingModel {
                name: "m".to_string(),
                dimensions: Some(3),
                ..ImportedEmbeddingModel::default()
            })
        )
        .is_err());
    }
}
