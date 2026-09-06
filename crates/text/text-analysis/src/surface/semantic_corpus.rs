use runtime_core::SurfaceOperation;
use serde::Deserialize;

use crate::semantic::{
    analyze_corpus_semantics, SemanticCorpusAnalysisOptions, SemanticCorpusItem,
};

pub(super) fn operation() -> SurfaceOperation {
    super::operation(
        "analysis.semantic-corpus",
        "Build semantic corpus profile",
        "Aggregates lexical statistics and deterministic corpus themes across attributed items, retaining source provenance and explicit embedding evidence.",
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
    let report = analyze_corpus_semantics(&items, &options).map_err(|error| error.to_string())?;
    serde_json::to_value(report).map_err(|error| error.to_string())
}

pub(super) fn annotation(
    value: &serde_json::Value,
) -> (&'static str, &'static str, serde_json::Value) {
    (
        "Semantic corpus profile",
        "Corpus-aware deterministic theme evidence across attributed items, retaining representative passages and explicit embedding provenance.",
        serde_json::json!({
            "status": "ok",
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
