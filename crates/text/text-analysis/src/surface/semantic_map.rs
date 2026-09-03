use runtime_core::SurfaceOperation;
use serde::Deserialize;
use text_core::TextDocument;
use text_linguistics::{TextNlpConfig, TextNlpPipeline};

use crate::{
    analyze_document_semantics, compare_semantic_neighborhoods, compose_linguistic_semantic_graph,
    SemanticAnalysisOptions,
};

pub(super) fn operation() -> SurfaceOperation {
    super::operation(
        "analysis.semantic-map",
        "Build semantic map",
        "Builds deterministic semantic units, concepts, trajectories, linguistic graph evidence, and optional exact-vs-index neighborhood parity evidence.",
        serde_json::json!({
            "id": "semantic-doc",
            "text": "Semantic search improves retrieval. Embedding indexes support semantic search. Tomatoes grow in garden soil. Healthy soil supports tomato roots.",
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
    let id = input.id.unwrap_or_else(|| "semantic-doc".to_string());
    let document = TextDocument::new(&id, &input.text);
    let options = input.options();
    let semantic = analyze_document_semantics(&document, &options)
        .map_err(|error| error.to_string())?;

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
        "Built deterministic semantic structure and projected existing linguistic evidence onto the same source units.",
        serde_json::json!({
            "status": "ok",
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
