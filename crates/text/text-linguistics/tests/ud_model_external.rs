#![cfg(feature = "external-tests")]

//! Opt-in scores for the de/es UD model presets: UPOS, UFeats, lemma, UAS and LAS.

use std::collections::BTreeMap;
use std::path::PathBuf;

use model_runtime::{HuggingFaceDownloader, HuggingFaceModelSpec, ModelBundle, ModelBundleStore};
use text_linguistics::{
    analyze_text, parse_ud_feats, DependencyParsingOptions, LinguisticAnalysisOptions,
    MorphologyOptions, PosTag, UdLemmatizer, UdModel, UdModelPreset,
};

/// Minimum accuracies on the UD test splits (gold word segmentation).
const MIN_UD_UPOS: f64 = 0.95;
const MIN_UD_UFEATS: f64 = 0.85;
const MIN_UD_LEMMA: f64 = 0.9;
const MIN_UD_UAS: f64 = 0.85;
const MIN_UD_LAS: f64 = 0.82;

fn bundle_dir() -> PathBuf {
    std::env::var_os("TEXT_LINGUISTICS_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".model-runtime"))
}

fn load(spec: &HuggingFaceModelSpec) -> ModelBundle {
    let revision = spec.revision_value().unwrap_or("main").to_string();
    let store = ModelBundleStore::new(bundle_dir())
        .downloader(HuggingFaceDownloader::new().progress(false).max_retries(2));
    store
        .load(&spec.name, &revision)
        .or_else(|_| store.download(spec))
        .unwrap()
}

fn model(preset: UdModelPreset) -> UdModel {
    UdModel::from_bundles(
        preset,
        &load(&preset.spec()),
        &load(&UdModelPreset::tokenizer_spec()),
    )
    .unwrap()
}

/// The universal part of a UD DEPREL; LAS ignores subtypes as in the CoNLL 2018 scorer.
fn universal(deprel: &str) -> &str {
    deprel.split(':').next().unwrap_or(deprel)
}

/// Every word has one head, exactly one word attaches to the root, and every word
/// reaches the root.
fn assert_well_formed(heads: &[usize]) {
    assert_eq!(
        heads.iter().filter(|&&head| head == 0).count(),
        1,
        "{heads:?}"
    );
    for start in 0..heads.len() {
        let mut node = start + 1;
        for _ in 0..=heads.len() {
            if node == 0 {
                break;
            }
            node = heads[node - 1];
        }
        assert_eq!(node, 0, "cycle in {heads:?}");
    }
}

struct GoldWord {
    form: String,
    lemma: String,
    upos: String,
    feats: String,
    head: usize,
    deprel: String,
}

fn read_conllu(path: &PathBuf) -> Vec<Vec<GoldWord>> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
    let mut sentences = Vec::new();
    let mut current = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                sentences.push(std::mem::take(&mut current));
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let cols = line.split('\t').collect::<Vec<_>>();
        if cols[0].contains('-') || cols[0].contains('.') {
            continue;
        }
        current.push(GoldWord {
            form: cols[1].to_string(),
            lemma: cols[2].to_string(),
            upos: cols[3].to_string(),
            feats: parse_ud_feats(cols[5]).to_ud_string(),
            head: cols[6].parse().unwrap(),
            deprel: cols[7].to_string(),
        });
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    sentences
}

fn ud_test_file(preset: UdModelPreset) -> PathBuf {
    let dir = std::env::var_os("UD_TREEBANK_DIR")
        .map(PathBuf::from)
        .expect("set UD_TREEBANK_DIR to a directory holding the UD 2.17 test splits");
    dir.join(match preset {
        UdModelPreset::ComboGermanGsd => "de_gsd-ud-test.conllu",
        UdModelPreset::ComboSpanishAncora => "es_ancora-ud-test.conllu",
    })
}

#[test]
#[ignore = "downloads the pinned COMBO checkpoints and needs UD_TREEBANK_DIR with the UD 2.17 test splits"]
fn ud_presets_reach_ud_test_accuracy() {
    for &preset in UdModelPreset::ALL {
        let model = model(preset);
        let lemmatizer = UdLemmatizer::for_language(preset.language()).unwrap();
        let (mut words, mut upos, mut feats, mut lemma, mut all) = (0usize, 0, 0, 0, 0);
        let (mut uas, mut las, mut las_full) = (0usize, 0, 0);
        for sentence in read_conllu(&ud_test_file(preset)) {
            let forms = sentence
                .iter()
                .map(|word| word.form.as_str())
                .collect::<Vec<_>>();
            let predicted = model.analyze_words(&forms).unwrap();
            assert_well_formed(&predicted.iter().map(|word| word.head).collect::<Vec<_>>());
            for (gold, predicted) in sentence.iter().zip(&predicted) {
                let upos_ok = predicted.upos.as_ud() == gold.upos;
                let feats_ok = predicted.feats.to_ud_string() == gold.feats;
                let lemma_ok = lemmatizer.lemmatize(&gold.form, predicted.upos) == gold.lemma;
                words += 1;
                upos += usize::from(upos_ok);
                feats += usize::from(feats_ok);
                lemma += usize::from(lemma_ok);
                all += usize::from(upos_ok && feats_ok && lemma_ok);
                let head_ok = predicted.head == gold.head;
                uas += usize::from(head_ok);
                las +=
                    usize::from(head_ok && universal(&predicted.deprel) == universal(&gold.deprel));
                las_full += usize::from(head_ok && predicted.deprel == gold.deprel);
            }
        }
        let rate = |count: usize| count as f64 / words as f64;
        eprintln!(
            "{} UD test: words={words} UPOS={:.4} UFeats={:.4} Lemma={:.4} UPOS+UFeats+Lemma={:.4} \
             UAS={:.4} LAS={:.4} LAS(with subtypes)={:.4}",
            preset.as_str(),
            rate(upos),
            rate(feats),
            rate(lemma),
            rate(all),
            rate(uas),
            rate(las),
            rate(las_full)
        );
        assert!(rate(upos) >= MIN_UD_UPOS, "{} UPOS", preset.as_str());
        assert!(rate(feats) >= MIN_UD_UFEATS, "{} UFeats", preset.as_str());
        assert!(rate(lemma) >= MIN_UD_LEMMA, "{} lemma", preset.as_str());
        assert!(rate(uas) >= MIN_UD_UAS, "{} UAS", preset.as_str());
        assert!(rate(las) >= MIN_UD_LAS, "{} LAS", preset.as_str());
    }
}

/// Minimum token accuracies on the hand-annotated de/es fixtures, through the full
/// pipeline (language detection, tokenization, contraction handling).
const MIN_GOLD_UPOS: f64 = 0.9;
const MIN_GOLD_UFEATS: f64 = 0.75;
const MIN_GOLD_LEMMA: f64 = 0.9;

#[test]
#[ignore = "downloads the pinned COMBO checkpoints and scores them on the de/es gold fixtures"]
fn ud_presets_reach_gold_fixture_accuracy() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../evaluation/corpora/morphology-multilingual-v1.jsonl");
    let mut options = LinguisticAnalysisOptions::heuristic();
    options.morphology = MorphologyOptions {
        bundle_dir: bundle_dir(),
        download_progress: false,
        max_retries: 2,
        ..MorphologyOptions::local_model_with_downloads()
    };
    let mut counts = BTreeMap::<String, [usize; 4]>::new();
    let mut detected_right = BTreeMap::<String, (usize, usize)>::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        let language = case["language"].as_str().unwrap().to_string();
        // Score the morphology operation itself; language detection is reported separately.
        options.morphology.language = Some(language.clone());
        let analysis = analyze_text(case["text"].as_str().unwrap(), &options).unwrap();
        let detected = analysis
            .language
            .primary
            .as_ref()
            .map(|p| p.language.clone());
        let hits = detected_right.entry(language.clone()).or_default();
        hits.0 += usize::from(detected.as_deref() == Some(language.as_str()));
        hits.1 += 1;
        let gold = case["tokens"].as_array().unwrap();
        let texts = analysis
            .tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>();
        let gold_texts = gold
            .iter()
            .map(|t| t["text"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(texts, gold_texts, "{} tokenization", case["id"]);
        let entry = counts.entry(language.clone()).or_default();
        for (index, gold) in gold.iter().enumerate() {
            let upos = analysis.pos[index].tag.as_ud();
            let lemma = analysis.lemmas[index].value.as_str();
            let feats = analysis.morphology[index].tags.to_ud_string();
            let ok = [
                upos == gold["upos"].as_str().unwrap(),
                feats == gold["feats"].as_str().unwrap(),
                lemma == gold["lemma"].as_str().unwrap(),
            ];
            if ok.contains(&false) {
                eprintln!(
                    "{} (detected {detected:?}) {}: got {upos} {lemma} {feats}, gold {} {} {}",
                    case["id"], gold["text"], gold["upos"], gold["lemma"], gold["feats"]
                );
            }
            entry[0] += 1;
            for (slot, ok) in ok.into_iter().enumerate() {
                entry[slot + 1] += usize::from(ok);
            }
        }
    }
    for (language, [tokens, upos, feats, lemma]) in counts {
        let rate = |count: usize| count as f64 / tokens as f64;
        let (hits, cases) = detected_right[&language];
        eprintln!(
            "{language} gold fixtures: tokens={tokens} UPOS={:.3} UFeats={:.3} Lemma={:.3} \
             (language detected correctly in {hits}/{cases} sentences)",
            rate(upos),
            rate(feats),
            rate(lemma)
        );
        assert!(rate(upos) >= MIN_GOLD_UPOS, "{language} UPOS");
        assert!(rate(feats) >= MIN_GOLD_UFEATS, "{language} UFeats");
        assert!(rate(lemma) >= MIN_GOLD_LEMMA, "{language} lemma");
    }
}

/// Minimum attachment scores on the hand-annotated de/es dependency fixtures,
/// through the full pipeline (tokenization, contraction handling, token trees).
const MIN_GOLD_UAS: f64 = 0.9;
const MIN_GOLD_LAS: f64 = 0.85;

#[test]
#[ignore = "downloads the pinned COMBO checkpoints and scores them on the de/es dependency fixtures"]
fn ud_presets_reach_gold_dependency_accuracy() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../evaluation/corpora/dependencies-multilingual-v1.jsonl");
    let mut options = LinguisticAnalysisOptions::heuristic();
    options.morphology = MorphologyOptions {
        bundle_dir: bundle_dir(),
        download_progress: false,
        max_retries: 2,
        ..MorphologyOptions::local_model_with_downloads()
    };
    options.dependency_parsing = DependencyParsingOptions {
        bundle_dir: bundle_dir(),
        download_progress: false,
        max_retries: 2,
        ..DependencyParsingOptions::local_model_with_downloads()
    };
    let mut counts = BTreeMap::<String, [usize; 3]>::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        let language = case["language"].as_str().unwrap().to_string();
        options.morphology.language = Some(language.clone());
        options.dependency_parsing.language = Some(language.clone());
        let analysis = analyze_text(case["text"].as_str().unwrap(), &options).unwrap();
        let gold = case["tokens"].as_array().unwrap();
        let texts = analysis
            .tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>();
        let gold_texts = gold
            .iter()
            .map(|t| t["text"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(texts, gold_texts, "{} tokenization", case["id"]);
        let mut predicted = vec![(usize::MAX, String::new()); texts.len()];
        for tree in &analysis.dependencies {
            for node in &tree.nodes {
                predicted[node.token_index] = (
                    node.head_token_index.map_or(0, |head| head + 1),
                    node.relation.to_ud_deprel(node.subtype.as_deref()),
                );
            }
        }
        assert_well_formed(&predicted.iter().map(|(head, _)| *head).collect::<Vec<_>>());
        let entry = counts.entry(language).or_default();
        for (gold, (head, deprel)) in gold.iter().zip(&predicted) {
            let gold_head = gold["head"].as_u64().unwrap() as usize;
            let gold_deprel = gold["deprel"].as_str().unwrap();
            let head_ok = *head == gold_head;
            let label_ok = universal(deprel) == universal(gold_deprel);
            if !(head_ok && label_ok) {
                eprintln!(
                    "{} {}: got {head} {deprel}, gold {gold_head} {gold_deprel}",
                    case["id"], gold["text"]
                );
            }
            entry[0] += 1;
            entry[1] += usize::from(head_ok);
            entry[2] += usize::from(head_ok && label_ok);
        }
    }
    for (language, [tokens, uas, las]) in counts {
        let rate = |count: usize| count as f64 / tokens as f64;
        eprintln!(
            "{language} dependency fixtures: tokens={tokens} UAS={:.3} LAS={:.3}",
            rate(uas),
            rate(las)
        );
        assert!(rate(uas) >= MIN_GOLD_UAS, "{language} UAS");
        assert!(rate(las) >= MIN_GOLD_LAS, "{language} LAS");
    }
}

#[test]
fn pos_tags_round_trip_ud_labels() {
    for &tag in PosTag::ALL {
        assert_eq!(PosTag::from_ud(tag.as_ud()), Some(tag));
    }
}
