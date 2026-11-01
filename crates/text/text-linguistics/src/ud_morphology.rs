//! UD-treebank-trained part-of-speech and morphological feature tagging.
//!
//! The German and Spanish taggers run the morphological head of the
//! [COMBO-NLP](https://gitlab.clarin-pl.eu/syntactic-tools/combo-nlp) UD 2.17
//! checkpoints (XLM-RoBERTa base, first-subword pooling) through Candle. Lemmas
//! come from [`UdLemmatizer`](crate::UdLemmatizer), not from the checkpoint.

use std::collections::BTreeMap;
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use std::collections::HashMap;
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use std::path::{Path, PathBuf};

#[cfg(all(feature = "candle", feature = "model-bundles"))]
use candle_core::{DType, Device, Tensor, D};
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use candle_transformers::models::xlm_roberta::{Config as XlmRobertaConfig, XLMRobertaModel};
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use model_runtime::ModelBundle;
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use serde_json::Value;
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use text_core::{DetectError, Result};

use crate::morphology::{MorphTagSet, PosTag};

/// Pinned Hugging Face commit of `clarin-pl/combo-nlp-xlm-roberta-base-german-gsd-ud2.17`.
pub const COMBO_GERMAN_GSD_REVISION: &str = "55234be1713fb4ae47a598e43325185248e4d651";
/// Pinned Hugging Face commit of `clarin-pl/combo-nlp-xlm-roberta-base-spanish-ancora-ud2.17`.
pub const COMBO_SPANISH_ANCORA_REVISION: &str = "3f5885ee238c6fd772c1dd0481ed71b40e042049";
/// Pinned Hugging Face commit of `FacebookAI/xlm-roberta-base`, whose tokenizer the
/// COMBO checkpoints were trained with.
pub const XLM_ROBERTA_BASE_REVISION: &str = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// UD-treebank-trained UPOS/feature taggers understood by text-linguistics.
pub enum UdMorphologyPreset {
    /// COMBO-NLP trained on UD_German-GSD 2.17 (CC BY-SA 4.0).
    ComboGermanGsd,
    /// COMBO-NLP trained on UD_Spanish-AnCora 2.17 (CC BY 4.0).
    ComboSpanishAncora,
}

impl UdMorphologyPreset {
    /// Every UD morphology preset.
    pub const ALL: &'static [Self] = &[Self::ComboGermanGsd, Self::ComboSpanishAncora];

    /// Stable preset id.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ComboGermanGsd => "combo-german-gsd-ud2.17",
            Self::ComboSpanishAncora => "combo-spanish-ancora-ud2.17",
        }
    }

    /// Hugging Face repository the preset loads.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::ComboGermanGsd => "clarin-pl/combo-nlp-xlm-roberta-base-german-gsd-ud2.17",
            Self::ComboSpanishAncora => {
                "clarin-pl/combo-nlp-xlm-roberta-base-spanish-ancora-ud2.17"
            }
        }
    }

    /// Pinned Hugging Face commit.
    pub fn revision(self) -> &'static str {
        match self {
            Self::ComboGermanGsd => COMBO_GERMAN_GSD_REVISION,
            Self::ComboSpanishAncora => COMBO_SPANISH_ANCORA_REVISION,
        }
    }

    /// ISO 639-1 language the preset is trained and evaluated for.
    pub fn language(self) -> &'static str {
        match self {
            Self::ComboGermanGsd => "de",
            Self::ComboSpanishAncora => "es",
        }
    }

    /// The preset for the primary subtag of `language` (`de-AT` -> German), if any.
    pub fn for_language(language: &str) -> Option<Self> {
        let primary = language
            .trim()
            .split(['-', '_'])
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        Self::ALL
            .iter()
            .copied()
            .find(|preset| preset.language() == primary)
    }

    /// The Hugging Face bundle spec for the checkpoint and its label encoders.
    #[cfg(all(feature = "candle", feature = "model-bundles"))]
    pub fn spec(self) -> model_runtime::HuggingFaceModelSpec {
        model_runtime::HuggingFaceModelSpec::new(
            self.repo_id(),
            model_runtime::ModelTask::TokenClassification,
        )
        .name(self.as_str())
        .revision(self.revision())
        .file("config.json")
        .file("encoders/morpho_encoder.json")
        .file("pytorch_model.bin")
    }

    /// The Hugging Face bundle spec for the shared XLM-RoBERTa tokenizer and encoder config.
    #[cfg(all(feature = "candle", feature = "model-bundles"))]
    pub fn tokenizer_spec() -> model_runtime::HuggingFaceModelSpec {
        model_runtime::HuggingFaceModelSpec::new(
            "FacebookAI/xlm-roberta-base",
            model_runtime::ModelTask::TokenClassification,
        )
        .name("xlm-roberta-base-tokenizer")
        .revision(XLM_ROBERTA_BASE_REVISION)
        .file("config.json")
        .file("tokenizer.json")
    }
}

#[derive(Debug, Clone, PartialEq)]
/// UPOS and UD features predicted for one syntactic word.
pub struct UdWordAnalysis {
    /// Universal part of speech.
    pub upos: PosTag,
    /// UD morphological features.
    pub feats: MorphTagSet,
    /// Softmax probability of the predicted UPOS.
    pub confidence: f32,
}

/// Splits surface contractions into the syntactic words UD treebanks tag
/// (`zum` -> `zu dem`, `del` -> `de el`). Returns `None` for ordinary words.
pub fn split_contraction(language: &str, word: &str) -> Option<[&'static str; 2]> {
    let lower = word.to_lowercase();
    let table: &[(&str, [&str; 2])] = match language {
        "de" => &[
            ("am", ["an", "dem"]),
            ("ans", ["an", "das"]),
            ("aufs", ["auf", "das"]),
            ("beim", ["bei", "dem"]),
            ("durchs", ["durch", "das"]),
            ("fürs", ["für", "das"]),
            ("hinterm", ["hinter", "dem"]),
            ("im", ["in", "dem"]),
            ("ins", ["in", "das"]),
            ("übers", ["über", "das"]),
            ("überm", ["über", "dem"]),
            ("ums", ["um", "das"]),
            ("unterm", ["unter", "dem"]),
            ("vom", ["von", "dem"]),
            ("vorm", ["vor", "dem"]),
            ("vors", ["vor", "das"]),
            ("zum", ["zu", "dem"]),
            ("zur", ["zu", "der"]),
        ],
        "es" => &[("al", ["a", "el"]), ("del", ["de", "el"])],
        _ => &[],
    };
    table
        .iter()
        .find(|(surface, _)| *surface == lower)
        .map(|(_, words)| *words)
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
struct FeatCategory {
    name: String,
    /// Value per label id; ids 0 (`<PAD>`, not applicable) and 1 (`<UNK>`) are `None`.
    values: Vec<Option<String>>,
    head: Linear,
    refine_head: Linear,
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
/// Candle runner for the COMBO morphological head (UPOS + UD features).
pub struct UdMorphoTagger {
    preset: UdMorphologyPreset,
    encoder: XLMRobertaModel,
    backbone: Linear,
    backbone_norm: LayerNorm,
    upos_head: Linear,
    upos_labels: Vec<Option<PosTag>>,
    feats: Vec<FeatCategory>,
    refine_in: Linear,
    refine_hidden: Linear,
    valid_feat_vectors: HashMap<usize, Vec<Vec<usize>>>,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
impl UdMorphoTagger {
    /// Loads a tagger from a checkpoint bundle ([`UdMorphologyPreset::spec`]) and the
    /// tokenizer bundle ([`UdMorphologyPreset::tokenizer_spec`]).
    pub fn from_bundles(
        preset: UdMorphologyPreset,
        model: &ModelBundle,
        tokenizer: &ModelBundle,
    ) -> Result<Self> {
        Self::from_paths(
            preset,
            &bundle_file(model, "pytorch_model.bin")?,
            &bundle_file(model, "encoders/morpho_encoder.json")?,
            &bundle_file(tokenizer, "config.json")?,
            &bundle_file(tokenizer, "tokenizer.json")?,
        )
    }

    fn from_paths(
        preset: UdMorphologyPreset,
        weights: &Path,
        morpho_encoder: &Path,
        encoder_config: &Path,
        tokenizer: &Path,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let config: XlmRobertaConfig = serde_json::from_value(read_json(encoder_config)?)
            .map_err(|err| DetectError::Source(format!("invalid XLM-R config: {err}")))?;
        let encoders = read_json(morpho_encoder)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer)
            .map_err(|err| DetectError::Source(format!("failed to load tokenizer: {err}")))?;

        // COMBO saves `{"model_state_dict": ..., "epoch": ..., "metrics": ...}`.
        let vb = VarBuilder::from_pth_with_state(weights, DType::F32, "model_state_dict", &device)
            .map_err(candle_error)?;
        let encoder = XLMRobertaModel::new(&config, vb.pp("encoder")).map_err(candle_error)?;
        let head = vb.pp("morpho_head");
        let hidden = config.hidden_size;
        let mid = hidden / 2;
        let backbone =
            candle_nn::linear(hidden, mid, head.pp("backbone.1")).map_err(candle_error)?;
        let backbone_norm =
            candle_nn::layer_norm(mid, 1e-5, head.pp("backbone.2")).map_err(candle_error)?;

        let upos_ids = label_ids(&encoders["upos"]["label_to_id"])?;
        let upos_labels = upos_ids
            .iter()
            .map(|label| PosTag::from_ud(label))
            .collect::<Vec<_>>();
        let upos_head =
            candle_nn::linear(mid, upos_ids.len(), head.pp("upos_head")).map_err(candle_error)?;

        let categories = encoders["feat_categories"]
            .as_array()
            .ok_or_else(|| DetectError::Source("morpho encoder lacks feat_categories".into()))?;
        let mut feats = Vec::new();
        let mut total_logits = upos_ids.len();
        for category in categories {
            let name = category
                .as_str()
                .ok_or_else(|| DetectError::Source("invalid feat category".into()))?
                .to_string();
            let ids = label_ids(&encoders["feats"][name.as_str()]["label_to_id"])?;
            total_logits += ids.len();
            let values = ids
                .iter()
                .enumerate()
                .map(|(index, value)| (index >= 2).then(|| value.clone()))
                .collect();
            feats.push(FeatCategory {
                head: candle_nn::linear(mid, ids.len(), head.pp(format!("feat_heads.{name}")))
                    .map_err(candle_error)?,
                refine_head: candle_nn::linear(
                    256,
                    ids.len(),
                    head.pp(format!("feat_refine_heads.{name}")),
                )
                .map_err(candle_error)?,
                name,
                values,
            });
        }
        let refine_in =
            candle_nn::linear(total_logits, 256, head.pp("feat_refine.0")).map_err(candle_error)?;
        let refine_hidden =
            candle_nn::linear(256, 256, head.pp("feat_refine.3")).map_err(candle_error)?;

        let mut valid_feat_vectors = HashMap::new();
        if let Some(map) = encoders["valid_feat_vectors"].as_object() {
            for (upos, vectors) in map {
                let Ok(upos) = upos.parse::<usize>() else {
                    continue;
                };
                let vectors = vectors
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|vector| {
                        vector.as_array().map(|ids| {
                            ids.iter()
                                .map(|id| id.as_u64().unwrap_or(0) as usize)
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect::<Vec<_>>();
                valid_feat_vectors.insert(upos, vectors);
            }
        }

        Ok(Self {
            preset,
            encoder,
            backbone,
            backbone_norm,
            upos_head,
            upos_labels,
            feats,
            refine_in,
            refine_hidden,
            valid_feat_vectors,
            tokenizer,
            device,
        })
    }

    /// The preset this tagger was loaded for.
    pub fn preset(&self) -> UdMorphologyPreset {
        self.preset
    }

    /// Tags one sentence of pre-split syntactic words.
    pub fn tag_words(&self, words: &[&str]) -> Result<Vec<UdWordAnalysis>> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let owned = words
            .iter()
            .map(|word| word.to_string())
            .collect::<Vec<_>>();
        let encoding = self
            .tokenizer
            .encode(owned, true)
            .map_err(|err| DetectError::Source(format!("tokenization failed: {err}")))?;
        if encoding.get_ids().len() > 512 {
            let (left, right) = words.split_at(words.len() / 2);
            let mut out = self.tag_words(left)?;
            out.extend(self.tag_words(right)?);
            return Ok(out);
        }

        let mut first_subword = vec![None; words.len()];
        for (position, word_id) in encoding.get_word_ids().iter().enumerate() {
            if let Some(word_id) = word_id {
                let slot = &mut first_subword[*word_id as usize];
                if slot.is_none() {
                    *slot = Some(position as u32);
                }
            }
        }

        let ids = encoding.get_ids();
        let input_ids = Tensor::new(ids, &self.device)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(candle_error)?;
        let attention =
            Tensor::ones((1, ids.len()), DType::U32, &self.device).map_err(candle_error)?;
        let token_types =
            Tensor::zeros((1, ids.len()), DType::U32, &self.device).map_err(candle_error)?;
        let hidden = self
            .encoder
            .forward(&input_ids, &attention, &token_types, None, None, None)
            .map_err(candle_error)?
            .squeeze(0)
            .map_err(candle_error)?;

        // Words that lost every subword (empty strings) reuse position 0 and get X.
        let positions = first_subword
            .iter()
            .map(|position| position.unwrap_or(0))
            .collect::<Vec<_>>();
        let index = Tensor::new(positions.as_slice(), &self.device).map_err(candle_error)?;
        let word_repr = hidden.index_select(&index, 0).map_err(candle_error)?;

        let h = self
            .backbone_norm
            .forward(&self.backbone.forward(&word_repr).map_err(candle_error)?)
            .and_then(|tensor| tensor.gelu_erf())
            .map_err(candle_error)?;
        let upos_logits = self.upos_head.forward(&h).map_err(candle_error)?;
        let initial = self
            .feats
            .iter()
            .map(|category| category.head.forward(&h))
            .collect::<candle_core::Result<Vec<_>>>()
            .map_err(candle_error)?;
        let mut combined = vec![upos_logits.clone()];
        combined.extend(initial.iter().cloned());
        let refined_h = Tensor::cat(&combined, D::Minus1)
            .and_then(|tensor| self.refine_in.forward(&tensor))
            .and_then(|tensor| tensor.gelu_erf())
            .and_then(|tensor| self.refine_hidden.forward(&tensor))
            .and_then(|tensor| tensor.gelu_erf())
            .map_err(candle_error)?;

        let upos_rows = upos_logits.to_vec2::<f32>().map_err(candle_error)?;
        let mut feat_rows = Vec::with_capacity(self.feats.len());
        for (category, initial) in self.feats.iter().zip(&initial) {
            let logits = (initial
                + category
                    .refine_head
                    .forward(&refined_h)
                    .map_err(candle_error)?)
            .map_err(candle_error)?;
            feat_rows.push(logits.to_vec2::<f32>().map_err(candle_error)?);
        }

        let mut out = Vec::with_capacity(words.len());
        for word in 0..words.len() {
            let (upos_id, confidence) = best_upos(&upos_rows[word], &self.upos_labels);
            let mut feat_ids = feat_rows
                .iter()
                .map(|rows| argmax(&rows[word]))
                .collect::<Vec<_>>();
            if let Some(valid) = self.valid_feat_vectors.get(&upos_id) {
                feat_ids = nearest_valid(&feat_ids, valid);
            }
            let mut feats = MorphTagSet::new();
            for (category, id) in self.feats.iter().zip(feat_ids) {
                if let Some(Some(value)) = category.values.get(id) {
                    feats.insert(category.name.clone(), value.clone());
                }
            }
            let upos = if first_subword[word].is_none() {
                PosTag::X
            } else {
                self.upos_labels[upos_id].unwrap_or(PosTag::X)
            };
            out.push(UdWordAnalysis {
                upos,
                feats,
                confidence,
            });
        }
        Ok(out)
    }
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn best_upos(logits: &[f32], labels: &[Option<PosTag>]) -> (usize, f32) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = logits.iter().map(|value| (value - max).exp()).sum();
    let (index, value) = logits
        .iter()
        .enumerate()
        .filter(|(index, _)| labels.get(*index).copied().flatten().is_some())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, value)| (index, *value))
        .unwrap_or((0, max));
    (index, (value - max).exp() / total)
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// Closest valid feature vector for the UPOS by Hamming distance (COMBO's
/// morphological constraint step).
#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn nearest_valid(ids: &[usize], valid: &[Vec<usize>]) -> Vec<usize> {
    if valid.is_empty() || valid.iter().any(|vector| vector == ids) {
        return ids.to_vec();
    }
    valid
        .iter()
        .min_by_key(|vector| vector.iter().zip(ids).filter(|(a, b)| a != b).count())
        .cloned()
        .unwrap_or_else(|| ids.to_vec())
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn label_ids(value: &Value) -> Result<Vec<String>> {
    let map = value
        .as_object()
        .ok_or_else(|| DetectError::Source("morpho encoder lacks label_to_id".into()))?;
    let mut labels = vec![String::new(); map.len()];
    for (label, id) in map {
        let id = id.as_u64().unwrap_or(u64::MAX) as usize;
        let slot = labels
            .get_mut(id)
            .ok_or_else(|| DetectError::Source(format!("label id {id} out of range")))?;
        *slot = label.clone();
    }
    Ok(labels)
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn bundle_file(bundle: &ModelBundle, remote_path: &str) -> Result<PathBuf> {
    bundle.file_path(remote_path).ok_or_else(|| {
        DetectError::InvalidArgument(format!("model bundle is missing `{remote_path}`"))
    })
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text)
        .map_err(|err| DetectError::Source(format!("invalid JSON `{}`: {err}", path.display())))
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn candle_error(error: candle_core::Error) -> DetectError {
    DetectError::Source(error.to_string())
}

/// Parses a CoNLL-U FEATS column (`Case=Nom|Number=Sing`, `_` for none).
pub fn parse_ud_feats(value: &str) -> MorphTagSet {
    let mut features = BTreeMap::new();
    for pair in value.split('|') {
        if let Some((key, value)) = pair.split_once('=') {
            features.insert(key.to_string(), value.to_string());
        }
    }
    MorphTagSet { features }
}
