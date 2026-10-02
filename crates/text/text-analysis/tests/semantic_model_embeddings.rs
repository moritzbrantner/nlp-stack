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
    assert_eq!(
        semantic["embeddingModel"]["model_name"],
        "fixture/semantic-model"
    );
    assert_eq!(semantic["embeddingModel"]["backend"], "external");
    assert_eq!(semantic["embeddingModel"]["dimensions"], 3);
    assert_eq!(semantic["embeddingModel"]["normalized"], true);
    for unit in semantic["units"].as_array().unwrap() {
        let squared_norm = unit["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap().powi(2))
            .sum::<f64>();
        assert!((squared_norm.sqrt() - 1.0).abs() < 0.000_01);
    }
    assert!(semantic["clusters"]
        .as_array()
        .unwrap()
        .iter()
        .any(|cluster| {
            cluster["memberUnitIds"]
                .as_array()
                .is_some_and(|members| members.len() == 2)
        }));
}

#[test]
fn semantic_corpus_accepts_external_model_embeddings() {
    let response = run_surface_operation(SurfaceRequest {
        operation: OperationId::new("analysis.semantic-corpus"),
        input: serde_json::json!({
            "items": [
                {
                    "id": "cat-1",
                    "author": "Alice",
                    "source": "notes/cat-1.txt",
                    "text": "A cat sleeps on the sofa."
                },
                {
                    "id": "cat-2",
                    "author": "Bob",
                    "source": "notes/cat-2.txt",
                    "text": "A feline rests on the couch."
                },
                {
                    "id": "database-1",
                    "author": "Cara",
                    "source": "notes/database-1.txt",
                    "text": "Database indexes accelerate lookup."
                }
            ],
            "minConceptUnits": 2,
            "neighborsPerUnit": 2,
            "neighborThreshold": 0.8,
            "clusterThreshold": 0.8,
            "embeddingModel": {
                "name": "fixture/corpus-semantic-model",
                "dimensions": 3,
                "maxTokens": 128
            },
            "importedEmbeddings": [
                {"text": "A cat sleeps on the sofa.", "vector": [2.0, 0.0, 0.0]},
                {"text": "A feline rests on the couch.", "vector": [1.9, 0.1, 0.0]},
                {"text": "Database indexes accelerate lookup.", "vector": [0.0, 2.0, 0.0]}
            ]
        }),
    })
    .unwrap();

    let report = &response.value["result"];
    let semantic = &report["semantic"];
    assert_eq!(
        semantic["embeddingModel"]["model_name"],
        "fixture/corpus-semantic-model"
    );
    assert_eq!(semantic["embeddingModel"]["backend"], "external");
    assert_eq!(semantic["embeddingModel"]["dimensions"], 3);
    assert_eq!(semantic["embeddingModel"]["normalized"], true);
    for unit in semantic["units"].as_array().unwrap() {
        let squared_norm = unit["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap().powi(2))
            .sum::<f64>();
        assert!((squared_norm.sqrt() - 1.0).abs() < 0.000_01);
    }
    assert!(report["concepts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|concept| { concept["memberUnitCount"] == 2 && concept["sourceItemCount"] == 2 }));
}

#[test]
fn imported_vectors_are_normalized_independently_of_scale() {
    for operation in ["analysis.semantic-map", "analysis.semantic-corpus"] {
        for scale in [
            1.0_f32,
            0.0001,
            1e20,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
        ] {
            let response = run_surface_operation(SurfaceRequest {
                operation: OperationId::new(operation),
                input: serde_json::json!({
                    "text": "Cats sleep.",
                    "items": [{"id": "cats", "text": "Cats sleep."}],
                    "includeLinguisticGraph": false,
                    "importedEmbeddings": [{"text": "Cats sleep.", "vector": [scale, -scale]}]
                }),
            })
            .unwrap_or_else(|error| panic!("{operation} rejected scale {scale}: {error}"));
            for unit in response.value["result"]["semantic"]["units"]
                .as_array()
                .unwrap()
            {
                let vector = unit["embedding"].as_array().unwrap();
                assert!(
                    (vector[0].as_f64().unwrap() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6
                );
                assert!(
                    (vector[1].as_f64().unwrap() + std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6
                );
            }
        }
    }
}

#[test]
fn invalid_imported_embedding_evidence_is_rejected_by_both_surfaces() {
    for operation in ["analysis.semantic-map", "analysis.semantic-corpus"] {
        for evidence in [
            serde_json::json!([{"text": "Cats sleep.", "vector": [0.0, 0.0]}]),
            serde_json::json!([{"text": "Cats sleep.", "vector": []}]),
            serde_json::json!([
                {"text": "Cats sleep.", "vector": [1.0, 0.0]},
                {"text": "Cats sleep.", "vector": [0.0, 1.0]}
            ]),
            serde_json::json!([
                {"text": "Cats sleep.", "vector": [1.0, 0.0]},
                {"text": "Dogs bark.", "vector": [1.0]}
            ]),
        ] {
            assert!(
                run_surface_operation(SurfaceRequest {
                    operation: OperationId::new(operation),
                    input: serde_json::json!({
                        "text": "Cats sleep.",
                        "items": [{"id": "cats", "text": "Cats sleep."}],
                        "includeLinguisticGraph": false,
                        "importedEmbeddings": evidence
                    }),
                })
                .is_err(),
                "{operation} accepted invalid evidence"
            );
        }
    }
}
