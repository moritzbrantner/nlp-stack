#![cfg(feature = "external-tests")]

//! Opt-in scores for the de/es UD morphology presets (UPOS, UFeats, lemma).

use std::collections::BTreeMap;
use std::path::PathBuf;

use model_runtime::{HuggingFaceDownloader, HuggingFaceModelSpec, ModelBundle, ModelBundleStore};
use text_linguistics::{
    analyze_text, parse_ud_feats, LinguisticAnalysisOptions, MorphologyOptions, PosTag,
    UdLemmatizer, UdMorphoTagger, UdMorphologyPreset,
};

/// Minimum accuracies on the UD test splits (gold word segmentation).
const MIN_UD_UPOS: f64 = 0.95;
const MIN_UD_UFEATS: f64 = 0.85;
const MIN_UD_LEMMA: f64 = 0.9;

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

fn tagger(preset: UdMorphologyPreset) -> UdMorphoTagger {
    UdMorphoTagger::from_bundles(
        preset,
        &load(&preset.spec()),
        &load(&UdMorphologyPreset::tokenizer_spec()),
    )
    .unwrap()
}

struct GoldWord {
    form: String,
    lemma: String,
    upos: String,
    feats: String,
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
        });
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    sentences
}

fn ud_test_file(preset: UdMorphologyPreset) -> PathBuf {
    let dir = std::env::var_os("UD_TREEBANK_DIR")
        .map(PathBuf::from)
        .expect("set UD_TREEBANK_DIR to a directory holding the UD 2.17 test splits");
    dir.join(match preset {
        UdMorphologyPreset::ComboGermanGsd => "de_gsd-ud-test.conllu",
        UdMorphologyPreset::ComboSpanishAncora => "es_ancora-ud-test.conllu",
    })
}

#[test]
#[ignore = "downloads the pinned COMBO checkpoints and needs UD_TREEBANK_DIR with the UD 2.17 test splits"]
fn ud_presets_reach_ud_test_accuracy() {
    for &preset in UdMorphologyPreset::ALL {
        let tagger = tagger(preset);
        let lemmatizer = UdLemmatizer::for_language(preset.language()).unwrap();
        let (mut words, mut upos, mut feats, mut lemma, mut all) = (0usize, 0, 0, 0, 0);
        for sentence in read_conllu(&ud_test_file(preset)) {
            let forms = sentence
                .iter()
                .map(|word| word.form.as_str())
                .collect::<Vec<_>>();
            let predicted = tagger.tag_words(&forms).unwrap();
            for (gold, predicted) in sentence.iter().zip(&predicted) {
                let upos_ok = predicted.upos.as_ud() == gold.upos;
                let feats_ok = predicted.feats.to_ud_string() == gold.feats;
                let lemma_ok = lemmatizer.lemmatize(&gold.form, predicted.upos) == gold.lemma;
                words += 1;
                upos += usize::from(upos_ok);
                feats += usize::from(feats_ok);
                lemma += usize::from(lemma_ok);
                all += usize::from(upos_ok && feats_ok && lemma_ok);
            }
        }
        let rate = |count: usize| count as f64 / words as f64;
        eprintln!(
            "{} UD test: words={words} UPOS={:.4} UFeats={:.4} Lemma={:.4} UPOS+UFeats+Lemma={:.4}",
            preset.as_str(),
            rate(upos),
            rate(feats),
            rate(lemma),
            rate(all)
        );
        assert!(rate(upos) >= MIN_UD_UPOS, "{} UPOS", preset.as_str());
        assert!(rate(feats) >= MIN_UD_UFEATS, "{} UFeats", preset.as_str());
        assert!(rate(lemma) >= MIN_UD_LEMMA, "{} lemma", preset.as_str());
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

#[test]
fn pos_tags_round_trip_ud_labels() {
    for &tag in PosTag::ALL {
        assert_eq!(PosTag::from_ud(tag.as_ud()), Some(tag));
    }
}
