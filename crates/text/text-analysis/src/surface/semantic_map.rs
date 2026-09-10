use std::collections::BTreeMap;

use runtime_core::SurfaceOperation;
use serde::Deserialize;
use text_core::{AnnotationProvenance, TextDocument};
use text_embeddings::{
    DenseVector, EmbeddingModelInfo, TextEmbeddingBackend, TextEmbeddingBackendKind,
    TextEmbeddingMetadata,
};
use text_linguistics::{TextNlpConfig, TextNlpPipeline};

use crate::{
    analyze_document_semantics, analyze_document_semantics_with, compare_semantic_neighborhoods,
    compose_linguistic_semantic_graph, invalid_argument, SemanticAnalysisOptions,
};

pub(super) fn operation() -> SurfaceOperation {
    super::operation(
        "analysis.semantic-map",
        "Build semantic map",
        "Builds semantic units, model- or baseline-backed concept neighborhoods, trajectories, linguistic graph evidence, and optional exact-vs-index neighborhood parity evidence.",
        serde_json::json!({
            "id": "semantic-doc",
            "text": "The search team began with exact lexical ranking because every match was easy to audit. A later evaluation showed that paraphrased questions often missed relevant passages. Sentence embeddings recovered many of those passages even when they shared few words with the query. The team therefore compared lexical and semantic retrieval on the same judged queries. Engineers also measured latency because model inference adds work before nearest-neighbor search. Product reviewers insisted that every result retain its source passage and model provenance. After tuning the candidate, the team returned to the original audit requirement and kept exact evidence beside semantic ranking. The launch decision was based on reproducible retrieval gains rather than novelty.",
            "neighborsPerUnit": 4,
            "neighborThreshold": 0.25,
            "clusterThreshold": 0.60,
            "includeLinguisticGraph": true,
            "includeNeighborhoodEvidence": true
        }),
    )
}

pub(super) fn run(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let input = super::parse_input::<SemanticMapRequest>(input)?;
    let options = input.options();
    let id = input
        .id
        .clone()
        .unwrap_or_else(|| "semantic-doc".to_string());
    let document = TextDocument::new(&id, &input.text);
    let semantic = if input.imported_embeddings.is_empty() {
        analyze_document_semantics(&document, &options).map_err(|error| error.to_string())?
    } else {
        let embedder = ImportedSemanticEmbedder::new(
            &input.imported_embeddings,
            input.embedding_model.as_ref(),
        )?;
        analyze_document_semantics_with(&document, &options, &embedder)
            .map_err(|error| error.to_string())?
    };

    let linguistic_graph = if input.include_linguistic_graph.unwrap_or(true) {
        let linguistic = TextNlpPipeline::new(TextNlpConfig::rich())
            .analyze_document(&document)
            .map_err(|error| error.to_string())?;
        Some(compose_linguistic_semantic_graph(&semantic, &linguistic))
    } else {
        None
    };
    let neighborhood_evidence = if input.include_neighborhood_evidence.unwrap_or(false) {
        Some(
            compare_semantic_neighborhoods(
                &semantic,
                options.neighbors_per_unit,
                options.neighbor_threshold,
            )
            .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };

    Ok(serde_json::json!({
        "semantic": semantic,
        "linguisticGraph": linguistic_graph,
        "neighborhoodEvidence": neighborhood_evidence
    }))
}

pub(super) fn annotation(
    value: &serde_json::Value,
) -> (&'static str, &'static str, serde_json::Value) {
    (
        "Semantic map result",
        "Built semantic structure from the supplied embedding evidence and projected existing linguistic evidence onto the same source units.",
        serde_json::json!({
            "status": "ok",
            "embeddingModel": value["semantic"]["embeddingModel"],
            "unitCount": value["semantic"]["units"].as_array().map(Vec::len).unwrap_or(0),
            "conceptCount": value["semantic"]["clusters"].as_array().map(Vec::len).unwrap_or(0),
            "neighborCount": value["semantic"]["neighbors"].as_array().map(Vec::len).unwrap_or(0),
            "hotspotCount": value["semantic"]["hotspots"].as_array().map(Vec::len).unwrap_or(0),
            "graphNodeCount": value["linguisticGraph"]["nodes"].as_array().map(Vec::len).unwrap_or(0),
            "graphEdgeCount": value["linguisticGraph"]["edges"].as_array().map(Vec::len).unwrap_or(0),
            "neighborhoodSharedEdgeCount": value["neighborhoodEvidence"]["sharedEdgeCount"],
            "neighborhoodExactOnlyEdgeCount": value["neighborhoodEvidence"]["exactOnlyEdgeCount"],
            "neighborhoodIndexedOnlyEdgeCount": value["neighborhoodEvidence"]["indexedOnlyEdgeCount"]
        }),
    )
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticMapRequest {
    id: Option<String>,
    text: String,
    #[serde(default)]
    neighbors_per_unit: Option<usize>,
    #[serde(default)]
    neighbor_threshold: Option<f32>,
    #[serde(default)]
    cluster_threshold: Option<f32>,
    #[serde(default)]
    include_linguistic_graph: Option<bool>,
    #[serde(default)]
    include_neighborhood_evidence: Option<bool>,
    #[serde(default)]
    imported_embeddings: Vec<ImportedSemanticEmbedding>,
    #[serde(default)]
    embedding_model: Option<ImportedEmbeddingModel>,
}

impl SemanticMapRequest {
    fn options(&self) -> SemanticAnalysisOptions {
        let mut options = SemanticAnalysisOptions::default();
        if let Some(neighbors_per_unit) = self.neighbors_per_unit {
            options.neighbors_per_unit = neighbors_per_unit;
        }
        if let Some(neighbor_threshold) = self.neighbor_threshold {
            options.neighbor_threshold = neighbor_threshold;
        }
        if let Some(cluster_threshold) = self.cluster_threshold {
            options.cluster_threshold = cluster_threshold;
        }
        options
    }
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
