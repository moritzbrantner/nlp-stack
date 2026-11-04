use super::imported_embeddings::{
    ImportedEmbeddingModel, ImportedSemanticEmbedder, ImportedSemanticEmbedding,
};

use runtime_core::SurfaceOperation;
use serde::Deserialize;
use text_core::TextDocument;
use text_linguistics::{TextNlpConfig, TextNlpPipeline};

use super::imported_embeddings::embedding_model_required;
use crate::{
    analyze_document_semantics_with, compare_semantic_neighborhoods,
    compose_linguistic_semantic_graph, SemanticAnalysisOptions,
};

pub(super) fn operation() -> SurfaceOperation {
    super::operation(
        "analysis.semantic-map",
        "Build semantic map",
        "Builds semantic units, model-backed concept neighborhoods from supplied `importedEmbeddings`, trajectories, linguistic graph evidence (needs the linguistic models), and optional exact-vs-index neighborhood parity evidence. Fails with `missing_dependency` without embeddings; there is no hashed baseline.",
        serde_json::json!({
            "id": "semantic-doc",
            "text": "A cat sleeps on the sofa. A feline rests on the couch. Database indexes accelerate lookup.",
            "neighborsPerUnit": 2,
            "neighborThreshold": 0.25,
            "clusterThreshold": 0.60,
            "includeLinguisticGraph": false,
            "includeNeighborhoodEvidence": true,
            "embeddingModel": {"name": "example-sentence-model", "dimensions": 3},
            "importedEmbeddings": [
                {"text": "A cat sleeps on the sofa.", "vector": [1.0, 0.0, 0.0]},
                {"text": "A feline rests on the couch.", "vector": [0.98, 0.02, 0.0]},
                {"text": "Database indexes accelerate lookup.", "vector": [0.0, 1.0, 0.0]},
                {"text": "A cat sleeps on the sofa. A feline rests on the couch. Database indexes accelerate lookup.", "vector": [0.7, 0.3, 0.0]}
            ]
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
    if input.imported_embeddings.is_empty() {
        return Err(embedding_model_required("analysis.semantic-map"));
    }
    let embedder =
        ImportedSemanticEmbedder::new(&input.imported_embeddings, input.embedding_model.as_ref())?;
    let semantic = analyze_document_semantics_with(&document, &options, &embedder)
        .map_err(|error| error.to_string())?;

    let linguistic_graph = if input.include_linguistic_graph.unwrap_or(true) {
        if !text_linguistics::local_models_compiled() {
            return Err(runtime_core::SurfaceError::missing_dependency(
                Some("analysis.semantic-map"),
                "linguistic models",
                "the linguistic graph needs the NER and UD models (text-linguistics `candle,model-bundles`); set `includeLinguisticGraph: false` to skip it",
            )
            .to_error_string());
        }
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
