#![cfg(feature = "external-tests")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use model_runtime::{HuggingFaceDownloader, ModelBundle, ModelBundleStore};
use text_core::{split_sentence_spans, tokenize, TextProcessingOptions};
use text_linguistics::{
    extract_named_entities_with_labeler, CandleTokenClassifier, EntityType, ModelPreset,
    SequenceLabeler,
};

/// Minimum exact-span micro F1 per supported language on the gold fixtures.
const MIN_GOLD_F1: f64 = 0.8;

fn bundle_dir() -> PathBuf {
    std::env::var_os("TEXT_LINGUISTICS_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".model-runtime"))
}

fn load_preset(preset: ModelPreset) -> ModelBundle {
    let spec = preset.spec();
    let revision = spec.revision_value().unwrap_or("main").to_string();
    let store = ModelBundleStore::new(bundle_dir())
        .downloader(HuggingFaceDownloader::new().progress(false).max_retries(2));
    store
        .load(&spec.name, &revision)
        .or_else(|_| store.download(&spec))
        .unwrap()
}

#[test]
#[ignore = "loads the opt-in dslim/bert-base-NER bundle synced by scripts/sync_model_bundles.sh text"]
fn bert_base_ner_loads_and_labels_tokens() {
    let bundle = ModelBundle::load(bundle_dir().join("bert-base-ner").join("main")).unwrap();
    let mut labeler = CandleTokenClassifier::from_bundle(bundle).unwrap();
    assert!(!labeler.labels().is_empty());
    assert!(!labeler
        .label_text("Ada Lovelace lived in London.")
        .unwrap()
        .is_empty());
}

type Span = (usize, usize, &'static str);

struct GoldCase {
    id: String,
    language: String,
    text: String,
    entities: Vec<Span>,
}

fn gold_cases() -> Vec<GoldCase> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../evaluation/corpora/named-entities-multilingual-v1.jsonl");
    std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            GoldCase {
                id: value["id"].as_str().unwrap().to_string(),
                language: value["language"].as_str().unwrap().to_string(),
                text: value["text"].as_str().unwrap().to_string(),
                entities: value["entities"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|entity| {
                        (
                            entity["byteStart"].as_u64().unwrap() as usize,
                            entity["byteEnd"].as_u64().unwrap() as usize,
                            gold_label(entity["label"].as_str().unwrap()),
                        )
                    })
                    .collect(),
            }
        })
        .collect()
}

fn gold_label(label: &str) -> &'static str {
    match label {
        "PER" => "PER",
        "ORG" => "ORG",
        "LOC" => "LOC",
        other => panic!("unexpected gold label {other}"),
    }
}

/// Only PER/ORG/LOC are scored: they are the labels every preset shares.
fn scored_label(entity_type: &EntityType) -> Option<&'static str> {
    match entity_type {
        EntityType::Person => Some("PER"),
        EntityType::Organization => Some("ORG"),
        EntityType::Location => Some("LOC"),
        _ => None,
    }
}

fn describe(text: &str, spans: &[Span], other: &[Span]) -> Vec<String> {
    spans
        .iter()
        .filter(|span| !other.contains(span))
        .map(|(start, end, label)| format!("{label}:{}", &text[*start..*end]))
        .collect()
}

/// Runs `preset` over the gold fixtures and returns per-language (tp, fp, fn).
fn evaluate(preset: ModelPreset) -> BTreeMap<String, (usize, usize, usize)> {
    let mut labeler = CandleTokenClassifier::from_bundle(load_preset(preset)).unwrap();
    let processing = TextProcessingOptions {
        include_punctuation: true,
        ..TextProcessingOptions::default()
    };
    let mut counts = BTreeMap::<String, (usize, usize, usize)>::new();
    for case in gold_cases() {
        let sentences = split_sentence_spans(&case.text, &processing);
        let tokens = tokenize(&case.text, &processing);
        let predicted =
            extract_named_entities_with_labeler(&case.text, &sentences, &tokens, &mut labeler)
                .unwrap()
                .into_iter()
                .filter_map(|entity| {
                    Some((
                        entity.mention.span.byte_start,
                        entity.mention.span.byte_end,
                        scored_label(&entity.entity_type)?,
                    ))
                })
                .collect::<Vec<_>>();
        let tp = predicted
            .iter()
            .filter(|span| case.entities.contains(span))
            .count();
        let missed = describe(&case.text, &case.entities, &predicted);
        let extra = describe(&case.text, &predicted, &case.entities);
        if !missed.is_empty() || !extra.is_empty() {
            eprintln!(
                "{} {}: missed {missed:?}, extra {extra:?}",
                preset.as_str(),
                case.id
            );
        }
        let entry = counts.entry(case.language.clone()).or_default();
        entry.0 += tp;
        entry.1 += predicted.len() - tp;
        entry.2 += case.entities.len() - tp;
    }
    counts
}

fn f1((tp, fp, fn_): (usize, usize, usize)) -> f64 {
    if tp == 0 {
        return 0.0;
    }
    let precision = tp as f64 / (tp + fp) as f64;
    let recall = tp as f64 / (tp + fn_) as f64;
    2.0 * precision * recall / (precision + recall)
}

#[test]
#[ignore = "downloads every pinned NER preset and scores it on the en/de/es gold fixtures"]
fn entity_presets_reach_gold_f1_for_their_languages() {
    for &preset in ModelPreset::ALL {
        for (language, counts) in evaluate(preset) {
            let score = f1(counts);
            eprintln!(
                "{} {language}: tp={} fp={} fn={} f1={score:.3}",
                preset.as_str(),
                counts.0,
                counts.1,
                counts.2
            );
            if preset.supports_language(&language) {
                assert!(
                    score >= MIN_GOLD_F1,
                    "{} {language} F1 {score:.3} < {MIN_GOLD_F1}",
                    preset.as_str()
                );
            }
        }
    }
}

#[test]
#[ignore = "downloads the multilingual NER preset and compares batched with per-text inference"]
fn batched_inference_matches_per_text_labels() {
    let labeler =
        CandleTokenClassifier::from_bundle(load_preset(ModelPreset::DistilbertMultilingualNerHrl))
            .unwrap();
    let cases = gold_cases();
    let texts = cases.iter().map(|case| case.text.as_str()).collect::<Vec<_>>();
    let batched = labeler.classify_batch(&texts).unwrap();
    for (text, batch) in texts.iter().zip(&batched) {
        let single = labeler.classify(text).unwrap();
        assert_eq!(single.len(), batch.len(), "{text}");
        for (single, batch) in single.iter().zip(batch) {
            // Labels and offsets match exactly; scores may differ in the last float bit
            // because a batch changes the matrix-multiply shapes.
            assert_eq!(single.label, batch.label, "{text}");
            assert_eq!(single.attributes, batch.attributes, "{text}");
            let (single, batch) = (single.score.unwrap(), batch.score.unwrap());
            assert!((single - batch).abs() <= 1e-5, "{text}: {single} vs {batch}");
        }
    }
}
