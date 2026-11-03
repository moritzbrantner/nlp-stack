//! Opt-in checks of the pinned multilingual embedding presets (#188).
//!
//! Run with `cargo test -p moenarch-text-embeddings --features candle,model-bundles
//! --release --test multilingual_presets_external -- --ignored --nocapture`.
//! Bundles live in `TEXT_EMBEDDINGS_MODEL_DIR` (default `.model-runtime`) and are
//! downloaded at the pinned revision when missing. Set `NLP_EVAL_OUTPUT_DIR` to also
//! write semantic-similarity predictions for `python3 -m evaluation.runner`, and
//! `TEXT_EMBEDDINGS_EVAL_PRESET` to report a preset other than the default.
#![cfg(all(feature = "candle", feature = "model-bundles"))]

use std::path::PathBuf;
use std::time::Instant;

use text_embeddings::{
    load_embedding_preset, text_similarity, CandleTextEmbedder, EmbeddingPresetLoadOptions,
    MultilingualEmbeddingPreset, PoolingStrategy, TextEmbeddingBackend,
};

fn load(preset: MultilingualEmbeddingPreset) -> CandleTextEmbedder {
    let options = EmbeddingPresetLoadOptions {
        auto_download: true,
        download_progress: true,
        ..EmbeddingPresetLoadOptions::default()
    };
    load_embedding_preset(preset, &options).unwrap()
}

fn corpus(name: &str) -> Vec<serde_json::Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../evaluation/corpora")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

fn assert_provenance(embedder: &CandleTextEmbedder, preset: MultilingualEmbeddingPreset) {
    let info = embedder.model_info();
    assert_eq!(info.model_name, preset.repo_id());
    assert_eq!(info.revision.as_deref(), Some(preset.revision()));
    assert_eq!(info.pooling, Some(preset.pooling()));
    assert_eq!(info.dimensions, preset.dimensions());
    assert!(info.normalized);
    let vector = embedder.embed_text("Berlin ist die Hauptstadt.").unwrap();
    assert_eq!(vector.dimensions(), preset.dimensions());
    let norm = cosine(vector.as_slice(), vector.as_slice()).sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "{norm}");
}

/// A German query must be closer to its English translation than to unrelated
/// English and Spanish passages.
fn assert_cross_lingual_retrieval(embedder: &CandleTextEmbedder) {
    let query = embedder
        .embed_query("Wie hoch ist der Eiffelturm?")
        .unwrap();
    let passages = [
        "The Eiffel Tower in Paris is about 330 metres tall.",
        "Tomatoes grow best in warm, sunny gardens.",
        "La economía española creció el año pasado.",
    ];
    let vectors = embedder.embed_passages(&passages).unwrap();
    let scores = vectors
        .iter()
        .map(|vector| cosine(query.as_slice(), vector.as_slice()))
        .collect::<Vec<_>>();
    assert!(
        scores[0] > scores[1] && scores[0] > scores[2],
        "German query should retrieve the English passage: {scores:?}"
    );
}

#[test]
#[ignore = "downloads intfloat/multilingual-e5-base (1.1 GB) at the pinned revision"]
fn e5_base_loads_pinned_and_reports_provenance() {
    let preset = MultilingualEmbeddingPreset::MultilingualE5Base;
    let embedder = load(preset);
    assert_provenance(&embedder, preset);
    assert_eq!(embedder.model_info().max_tokens, Some(512));
    assert_eq!(embedder.model_info().pooling, Some(PoolingStrategy::Mean));
    assert_cross_lingual_retrieval(&embedder);
    // The prefixes are applied inside the backend: a query and the same text as a
    // passage embed differently, and `embed_text` (symmetric) uses the query prefix.
    let text = "Berlin ist die Hauptstadt.";
    let as_query = embedder.embed_query(text).unwrap();
    let as_passage = embedder.embed_passages(&[text]).unwrap().remove(0);
    let symmetric = embedder.embed_text(text).unwrap();
    assert!(cosine(as_query.as_slice(), as_passage.as_slice()) < 0.9999);
    assert!(cosine(as_query.as_slice(), symmetric.as_slice()) > 0.9999);
}

#[test]
#[ignore = "downloads the preset (default multilingual-e5-base; TEXT_EMBEDDINGS_EVAL_PRESET selects another) and reports STS, retrieval and throughput on the small evaluation corpora"]
fn preset_semantic_similarity_report() {
    let preset = std::env::var("TEXT_EMBEDDINGS_EVAL_PRESET")
        .map(|value| value.parse::<MultilingualEmbeddingPreset>().unwrap())
        .unwrap_or(MultilingualEmbeddingPreset::DEFAULT);
    let embedder = load(preset);
    let output_dir = std::env::var_os("NLP_EVAL_OUTPUT_DIR").map(PathBuf::from);
    for name in [
        "semantic_similarity_smoke_v1.jsonl",
        "semantic_similarity_multilingual_v1.jsonl",
    ] {
        let cases = corpus(name);
        let mut predictions = String::new();
        for case in &cases {
            let similarity = text_similarity(
                case["left"].as_str().unwrap(),
                case["right"].as_str().unwrap(),
                &embedder,
            )
            .unwrap();
            println!(
                "{name} {:<22} gold={:.2} {preset}={similarity:.4}",
                case["id"].as_str().unwrap(),
                case["similarity"].as_f64().unwrap()
            );
            predictions.push_str(
                &serde_json::json!({"id": case["id"], "similarity": similarity}).to_string(),
            );
            predictions.push('\n');
        }
        if let Some(dir) = &output_dir {
            std::fs::create_dir_all(dir).unwrap();
            let path = dir.join(name.replace(".jsonl", &format!(".{preset}.predictions.jsonl")));
            std::fs::write(&path, predictions).unwrap();
            println!("wrote {}", path.display());
        }
    }

    // Cross-lingual retrieval over the multilingual corpus: each translation pair's
    // left text is a query; every distinct text in the corpus is a passage.
    let cases = corpus("semantic_similarity_multilingual_v1.jsonl");
    let mut passages = cases
        .iter()
        .flat_map(|case| {
            [
                case["left"].as_str().unwrap(),
                case["right"].as_str().unwrap(),
            ]
        })
        .collect::<Vec<_>>();
    passages.sort_unstable();
    passages.dedup();
    let passage_vectors = embedder.embed_passages(&passages).unwrap();
    let mut reciprocal_ranks = Vec::new();
    for case in cases
        .iter()
        .filter(|case| case["id"].as_str().unwrap().ends_with("-translation"))
    {
        let left = case["left"].as_str().unwrap();
        let right = case["right"].as_str().unwrap();
        let query = embedder.embed_query(left).unwrap();
        let mut ranked = passages
            .iter()
            .zip(&passage_vectors)
            .filter(|(text, _)| **text != left)
            .map(|(text, vector)| (*text, cosine(query.as_slice(), vector.as_slice())))
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let rank = ranked.iter().position(|(text, _)| *text == right).unwrap() + 1;
        println!("retrieval {}: rank {rank} of {}", case["id"], ranked.len());
        reciprocal_ranks.push(1.0 / rank as f64);
    }
    let mrr = reciprocal_ranks.iter().sum::<f64>() / reciprocal_ranks.len() as f64;
    println!(
        "cross-lingual retrieval: {} queries, MRR {mrr:.3}",
        reciprocal_ranks.len()
    );

    // Throughput: one batched pass over short sentences on this machine.
    let texts = (0..64)
        .map(|index| {
            format!("Satz {index}: Semantische Suche findet passende Abschnitte in Transkripten.")
        })
        .collect::<Vec<_>>();
    let texts = texts.iter().map(String::as_str).collect::<Vec<_>>();
    let started = Instant::now();
    embedder.embed_passages(&texts).unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "throughput: {} short sentences in {elapsed:.2}s = {:.1} embeddings/s",
        texts.len(),
        texts.len() as f64 / elapsed
    );
}

#[test]
#[ignore = "downloads Qwen/Qwen3-Embedding-0.6B (1.2 GB, about 2.4 GB in f32 memory)"]
fn qwen3_embedding_loads_pinned_and_retrieves_cross_lingually() {
    let preset = MultilingualEmbeddingPreset::Qwen3Embedding0_6B;
    let embedder = load(preset);
    assert_provenance(&embedder, preset);
    assert_cross_lingual_retrieval(&embedder);
}

#[test]
#[ignore = "downloads BAAI/bge-m3 (2.3 GB, about 2.3 GB in f32 memory)"]
fn bge_m3_loads_pinned_and_retrieves_cross_lingually() {
    let preset = MultilingualEmbeddingPreset::BgeM3;
    let embedder = load(preset);
    assert_provenance(&embedder, preset);
    assert_cross_lingual_retrieval(&embedder);
}
