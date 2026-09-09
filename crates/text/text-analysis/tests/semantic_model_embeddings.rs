use runtime_core::{OperationId, SurfaceRequest};
use text_analysis::surface::run_surface_operation;

#[test]
fn semantic_map_accepts_external_model_embeddings() {
    let text = "A cat sleeps on the sofa. A feline rests on the couch. Database indexes accelerate lookup.";
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("analysis.semantic-map"),
        input: serde_json::json!({
            "id": "model-backed",
            "text": text,
            "neighborsPerUnit": 2,
            "neighborThreshold": 0.75,
            "clusterThreshold": 0.75,
            "includeLinguisticGraph": false,
            "includeNeighborhoodEvidence": false,
            "embeddingModel": {
                "name": "fixture/semantic-model",
                "dimensions": 3,
                "maxTokens": 128
            },
            "importedEmbeddings": [
                {"text": "A cat sleeps on the sofa.", "vector": [1.0, 0.0, 0.0]},
                {"text": "A feline rests on the couch.", "vector": [0.98, 0.02, 0.0]},
                {"text": "Database indexes accelerate lookup.", "vector": [0.0, 1.0, 0.0]},
                {"text": text, "vector": [0.7, 0.3, 0.0]}
            ]
        }),
    })
    .unwrap();

    let semantic = &response.value["result"]["semantic"];
    assert_eq!(semantic["embeddingModel"]["model_name"], "fixture/semantic-model");
    assert_eq!(semantic["embeddingModel"]["backend"], "external");
    assert_eq!(semantic["embeddingModel"]["dimensions"], 3);
    assert!(semantic["clusters"].as_array().unwrap().iter().any(|cluster| {
        cluster["memberUnitIds"].as_array().is_some_and(|members| members.len() == 2)
    }));
}
