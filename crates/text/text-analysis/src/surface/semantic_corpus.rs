use std::collections::BTreeMap;

use runtime_core::SurfaceOperation;
use serde::Deserialize;
use text_core::AnnotationProvenance;
use text_embeddings::{
    DenseVector, EmbeddingModelInfo, TextEmbeddingBackend, TextEmbeddingBackendKind,
    TextEmbeddingMetadata,
};

use crate::{
    invalid_argument,
    semantic::{
        analyze_corpus_semantics, analyze_corpus_semantics_with, SemanticCorpusAnalysisOptions,
        SemanticCorpusItem,
    },
};

pub(super) fn operation() -> SurfaceOperation {
    super::operation(
        "analysis.semantic-corpus",
        "Build semantic corpus profile",
        "Aggregates lexical statistics and model- or baseline-backed corpus themes across attributed items, retaining source provenance and explicit embedding evidence.",
        serde_json::json!({
            "items": [
                {
                    "id": "alice-1",
                    "author": "Alice",
                    "source": "letters/1.txt",
                    "timestampMillis": 1700000000000_i64,
                    "text": "Semantic search improves retrieval. Embedding indexes support semantic search."
                },
                {
                    "id": "alice-2",
                    "author": "Alice",
                    "source": "letters/2.txt",
                    "timestampMillis": 1710000000000_i64,
                    "text": "Semantic retrieval finds related passages. Vector indexes accelerate retrieval."
                },
                {
                    "id": "bob-1",
                    "author": "Bob",
                    "source": "notes/1.txt",
                    "timestampMillis": 1720000000000_i64,
                    "text": "Tomatoes grow in garden soil. Healthy soil supports tomato roots."
                }
            ],
            "topTerms": 12,
            "minConceptUnits": 2,
            "neighborsPerUnit": 4,
            "neighborThreshold": 0.25,
            "clusterThreshold": 0.60
        }),
    )
}

pub(super) fn run(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let input = super::parse_input::<SemanticCorpusRequest>(input)?;
    let options = input.options();
    let items = input
        .items
        .iter()
        .map(|item| SemanticCorpusItem {
            id: item.id.as_str(),
            author: item.author.as_deref(),
            text: item.text.as_str(),
            source: item.source.as_deref(),
            timestamp_millis: item.timestamp_millis,
        })
        .collect::<Vec<_>>();
    let report = if input.imported_embeddings.is_empty() {
        analyze_corpus_semantics(&items, &options).map_err(|error| error.to_string())?
    } else {
        let embedder = ImportedSemanticEmbedder::new(
            &input.imported_embeddings,
            input.embedding_model.as_ref(),
        )?;
        analyze_corpus_semantics_with(&items, &options, &embedder)
            .map_err(|error| error.to_string())?
    };
    serde_json::to_value(report).map_err(|error| error.to_string())
}

pub(super) fn annotation(
    value: &serde_json::Value,
) -> (&'static str, &'static str, serde_json::Value) {
    (
        "Semantic corpus profile",
        "Corpus-aware theme evidence across attributed items, retaining representative passages and explicit embedding provenance.",
        serde_json::json!({
            "status": "ok",
            "embeddingModel": value["semantic"]["embeddingModel"],
            "itemCount": value["itemCount"],
            "authorCount": value["authorCount"],
            "wordCount": value["lexical"]["wordCount"],
            "uniqueTermCount": value["lexical"]["uniqueTerms"],
            "conceptCount": value["concepts"].as_array().map(Vec::len).unwrap_or(0),
            "nonConceptUnitCount": value["nonConceptUnitCount"],
            "semanticUnitCount": value["semantic"]["timeline"].as_array().map(Vec::len).unwrap_or(0),
            "representativePassageCount": value["concepts"].as_array().map(Vec::len).unwrap_or(0)
        }),
    )
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticCorpusRequest {
    items: Vec<SemanticCorpusItemRequest>,
    #[serde(default)]
    top_terms: Option<usize>,
    #[serde(default)]
    min_concept_units: Option<usize>,
    #[serde(default)]
    neighbors_per_unit: Option<usize>,
    #[serde(default)]
    neighbor_threshold: Option<f32>,
    #[serde(default)]
    cluster_threshold: Option<f32>,
    #[serde(default)]
    imported_embeddings: Vec<ImportedSemanticEmbedding>,
    #[serde(default)]
    embedding_model: Option<ImportedEmbeddingModel>,
}

impl SemanticCorpusRequest {
    fn options(&self) -> SemanticCorpusAnalysisOptions {
        let mut options = SemanticCorpusAnalysisOptions::default();
        if let Some(top_terms) = self.top_terms {
            options.top_terms = top_terms;
        }
        if let Some(min_concept_units) = self.min_concept_units {
            options.min_concept_units = min_concept_units;
        }
        if let Some(neighbors_per_unit) = self.neighbors_per_unit {
            options.semantic.neighbors_per_unit = neighbors_per_unit;
        }
        if let Some(neighbor_threshold) = self.neighbor_threshold {
            options.semantic.neighbor_threshold = neighbor_threshold;
        }
        if let Some(cluster_threshold) = self.cluster_threshold {
            options.semantic.cluster_threshold = cluster_threshold;
        }
        options
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticCorpusItemRequest {
    id: String,
    #[serde(default)]
    author: Option<String>,
    text: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    timestamp_millis: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImportedSemanticEmbedding {
    text: String,
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImportedEmbeddingModel {
    name: String,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Debug)]
struct ImportedSemanticEmbedder {
    vectors: BTreeMap<String, Vec<f32>>,
    model_name: String,
    dimensions: usize,
    max_tokens: Option<usize>,
}

impl ImportedSemanticEmbedder {
    fn new(
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
    let squared_norm = values.iter().map(|value| value * value).sum::<f32>();
    if !squared_norm.is_finite() || squared_norm <= f32::EPSILON {
        return Err("imported semantic embeddings must have a finite non-zero norm".to_string());
    }
    let norm = squared_norm.sqrt();
    Ok(values.iter().map(|value| value / norm).collect())
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
