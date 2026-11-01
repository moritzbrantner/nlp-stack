//! UD-treebank-trained part-of-speech, morphological feature and dependency analysis.
//!
//! The German and Spanish models run the morphological and biaffine dependency
//! heads of the [COMBO-NLP](https://gitlab.clarin-pl.eu/syntactic-tools/combo-nlp)
//! UD 2.17 checkpoints (XLM-RoBERTa base, first-subword pooling) through Candle.
//! Arcs are decoded into a single-rooted tree with Chu-Liu-Edmonds. Lemmas come
//! from [`UdLemmatizer`](crate::UdLemmatizer), not from the checkpoint.

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
pub enum UdModelPreset {
    /// COMBO-NLP trained on UD_German-GSD 2.17 (CC BY-SA 4.0).
    ComboGermanGsd,
    /// COMBO-NLP trained on UD_Spanish-AnCora 2.17 (CC BY 4.0).
    ComboSpanishAncora,
}

impl UdModelPreset {
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
        .file("encoders/deprel_encoder.json")
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
/// UPOS, UD features and dependency attachment predicted for one syntactic word.
pub struct UdWordAnalysis {
    /// Universal part of speech.
    pub upos: PosTag,
    /// UD morphological features.
    pub feats: MorphTagSet,
    /// Softmax probability of the predicted UPOS.
    pub confidence: f32,
    /// 1-based index of the head word in the sentence; 0 for the root.
    pub head: usize,
    /// Full UD dependency relation (`nsubj:pass`); `root` exactly when `head` is 0.
    pub deprel: String,
    /// Softmax probability of the chosen head among all candidate heads.
    pub head_confidence: f32,
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

/// `Linear -> LayerNorm -> ReLU`, the projection in front of each biaffine scorer.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
struct Mlp {
    linear: Linear,
    norm: LayerNorm,
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
impl Mlp {
    fn load(vb: VarBuilder, input: usize, output: usize) -> candle_core::Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(input, output, vb.pp("0"))?,
            norm: candle_nn::layer_norm(output, 1e-5, vb.pp("1"))?,
        })
    }

    /// Projects and appends the constant bias feature the biaffine weights expect.
    fn forward_with_bias(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let projected = self.norm.forward(&self.linear.forward(input)?)?.relu()?;
        let ones = Tensor::ones(
            (projected.dim(0)?, 1),
            projected.dtype(),
            projected.device(),
        )?;
        Tensor::cat(&[projected, ones], D::Minus1)
    }
}

/// COMBO's biaffine arc and relation scorers over a learned root embedding.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
struct DependencyHead {
    root: Tensor,
    arc_head: Mlp,
    arc_dep: Mlp,
    rel_head: Mlp,
    rel_dep: Mlp,
    /// `[arc + 1, arc + 1]`.
    arc_weight: Tensor,
    /// `[rel + 1, labels * (rel + 1)]`: `W[o, i, j]` laid out as `[i, (o, j)]`.
    rel_weight: Tensor,
    rel_dim: usize,
    /// Label per relation id; `<PAD>` and `<UNK>` are `None`.
    labels: Vec<Option<String>>,
    root_label: usize,
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
impl DependencyHead {
    /// Loads the head; `labels` are the deprel encoder's labels by id, `arc`/`rel`
    /// the checkpoint's `arc_hidden_dim`/`rel_hidden_dim`.
    fn load(
        vb: VarBuilder,
        hidden: usize,
        arc: usize,
        rel: usize,
        labels: Vec<String>,
    ) -> Result<Self> {
        let count = labels.len();
        let root_label = labels
            .iter()
            .position(|label| label == "root")
            .ok_or_else(|| DetectError::Source("deprel encoder lacks `root`".into()))?;
        let labels = labels
            .into_iter()
            .enumerate()
            .map(|(index, label)| (index >= 2).then_some(label))
            .collect();
        let head = vb.pp("dependency_head");
        let rel_weight = head
            .get((count, rel + 1, rel + 1), "rel_biaffine.weight")
            .and_then(|weight| weight.permute((1, 0, 2)))
            .and_then(|weight| weight.reshape((rel + 1, count * (rel + 1))))
            .map_err(candle_error)?;
        let arc_weight = head
            .get((1, arc + 1, arc + 1), "arc_biaffine.weight")
            .and_then(|weight| weight.reshape((arc + 1, arc + 1)))
            .map_err(candle_error)?;
        let root = vb
            .get((1, 1, hidden), "root_embedding")
            .and_then(|root| root.reshape((1, hidden)))
            .map_err(candle_error)?;
        let mlp = |name: &str, output: usize| {
            Mlp::load(head.pp(name), hidden, output).map_err(candle_error)
        };
        Ok(Self {
            root,
            arc_head: mlp("arc_head_mlp", arc)?,
            arc_dep: mlp("arc_dep_mlp", arc)?,
            rel_head: mlp("rel_head_mlp", rel)?,
            rel_dep: mlp("rel_dep_mlp", rel)?,
            arc_weight,
            rel_weight,
            rel_dim: rel + 1,
            labels,
            root_label,
        })
    }

    /// Heads (1-based, 0 = root), relations and head probabilities for `words`
    /// (`[n, hidden]` first-subword representations).
    fn parse(&self, words: &Tensor) -> candle_core::Result<Vec<(usize, String, f32)>> {
        let n = words.dim(0)?;
        let nodes = Tensor::cat(&[&self.root, words], 0)?;

        // arc[i][j]: score of node j heading word i + 1 (j = 0 is the root).
        let arc_dep = self.arc_dep.forward_with_bias(words)?;
        let arc_head = self.arc_head.forward_with_bias(&nodes)?;
        let arc = arc_dep
            .matmul(&self.arc_weight)?
            .matmul(&arc_head.t()?)?
            .to_vec2::<f32>()?;
        let heads = decode_dependency_tree(&arc);

        let rel_dep = self.rel_dep.forward_with_bias(words)?;
        let rel_head = self.rel_head.forward_with_bias(&nodes)?;
        let head_index = Tensor::new(
            heads.iter().map(|&head| head as u32).collect::<Vec<_>>(),
            words.device(),
        )?;
        let labels = self.labels.len();
        let rel = rel_dep
            .matmul(&self.rel_weight)?
            .reshape((n, labels, self.rel_dim))?
            .broadcast_mul(&rel_head.index_select(&head_index, 0)?.unsqueeze(1)?)?
            .sum(D::Minus1)?
            .to_vec2::<f32>()?;

        Ok(heads
            .iter()
            .enumerate()
            .map(|(word, &head)| {
                let label = if head == 0 {
                    self.root_label
                } else {
                    rel[word]
                        .iter()
                        .enumerate()
                        .filter(|(id, _)| *id != self.root_label && self.labels[*id].is_some())
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(id, _)| id)
                        .unwrap_or(self.root_label)
                };
                let row = &arc[word];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = row.iter().map(|score| (score - max).exp()).sum();
                (
                    head,
                    self.labels[label].clone().unwrap_or_else(|| "dep".into()),
                    (row[head] - max).exp() / total,
                )
            })
            .collect())
    }
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
/// Candle runner for a COMBO checkpoint: XLM-R encoder, morphological head
/// (UPOS + UD features) and biaffine dependency head.
pub struct UdModel {
    preset: UdModelPreset,
    encoder: XLMRobertaModel,
    backbone: Linear,
    backbone_norm: LayerNorm,
    upos_head: Linear,
    upos_labels: Vec<Option<PosTag>>,
    feats: Vec<FeatCategory>,
    refine_in: Linear,
    refine_hidden: Linear,
    valid_feat_vectors: HashMap<usize, Vec<Vec<usize>>>,
    dependency: DependencyHead,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
impl UdModel {
    /// Loads a tagger from a checkpoint bundle ([`UdModelPreset::spec`]) and the
    /// tokenizer bundle ([`UdModelPreset::tokenizer_spec`]).
    pub fn from_bundles(
        preset: UdModelPreset,
        model: &ModelBundle,
        tokenizer: &ModelBundle,
    ) -> Result<Self> {
        Self::from_paths(
            preset,
            &CheckpointPaths {
                weights: bundle_file(model, "pytorch_model.bin")?,
                config: bundle_file(model, "config.json")?,
                morpho_encoder: bundle_file(model, "encoders/morpho_encoder.json")?,
                deprel_encoder: bundle_file(model, "encoders/deprel_encoder.json")?,
            },
            &bundle_file(tokenizer, "config.json")?,
            &bundle_file(tokenizer, "tokenizer.json")?,
        )
    }

    fn from_paths(
        preset: UdModelPreset,
        checkpoint: &CheckpointPaths,
        encoder_config: &Path,
        tokenizer: &Path,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let weights = checkpoint.weights.as_path();
        let config: XlmRobertaConfig = serde_json::from_value(read_json(encoder_config)?)
            .map_err(|err| DetectError::Source(format!("invalid XLM-R config: {err}")))?;
        let checkpoint_config = read_json(&checkpoint.config)?;
        let dimension = |key: &str| {
            checkpoint_config["model"][key]
                .as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| DetectError::Source(format!("checkpoint config lacks `{key}`")))
        };
        let encoders = read_json(&checkpoint.morpho_encoder)?;
        let deprels = label_ids(&read_json(&checkpoint.deprel_encoder)?["label_to_id"])?;
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

        let dependency = DependencyHead::load(
            vb,
            hidden,
            dimension("arc_hidden_dim")?,
            dimension("rel_hidden_dim")?,
            deprels,
        )?;

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
            dependency,
            tokenizer,
            device,
        })
    }

    /// The preset this model was loaded for.
    pub fn preset(&self) -> UdModelPreset {
        self.preset
    }

    /// Tags and parses one sentence of pre-split syntactic words. Sentences longer
    /// than the encoder's 512 subwords are analyzed in halves; the second half's
    /// root then attaches to the first half's root as `parataxis`.
    pub fn analyze_words(&self, words: &[&str]) -> Result<Vec<UdWordAnalysis>> {
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
            let mut out = self.analyze_words(left)?;
            let left_root = out.iter().position(|word| word.head == 0).unwrap_or(0) + 1;
            out.extend(self.analyze_words(right)?.into_iter().map(|mut word| {
                if word.head == 0 {
                    word.head = left_root;
                    word.deprel = "parataxis".into();
                } else {
                    word.head += left.len();
                }
                word
            }));
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
        let arcs = self.dependency.parse(&word_repr).map_err(candle_error)?;

        let mut out = Vec::with_capacity(words.len());
        for (word, (head, deprel, head_confidence)) in arcs.into_iter().enumerate() {
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
                head,
                deprel,
                head_confidence,
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

#[cfg(all(feature = "candle", feature = "model-bundles"))]
struct CheckpointPaths {
    weights: PathBuf,
    config: PathBuf,
    morpho_encoder: PathBuf,
    deprel_encoder: PathBuf,
}

/// Decodes biaffine arc scores into a single-rooted dependency tree.
///
/// `scores[i][j]` is the score of node `j` heading word `i + 1`; node 0 is the
/// artificial root, so each row has `scores.len() + 1` entries. Returns each
/// word's head (0 = root). Chu-Liu-Edmonds finds the maximum spanning
/// arborescence; if several words then attach to the root, the one with the best
/// root score keeps it and the others attach to that word, as in COMBO's decoder.
pub fn decode_dependency_tree(scores: &[Vec<f32>]) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    // weights[head][dependent] over nodes 0..=n; nothing heads the root.
    let mut weights = vec![vec![f64::NEG_INFINITY; n + 1]; n + 1];
    for (word, row) in scores.iter().enumerate() {
        for (head, &score) in row.iter().enumerate().take(n + 1) {
            if head != word + 1 {
                weights[head][word + 1] = f64::from(score);
            }
        }
    }
    let mut heads = chu_liu_edmonds(&weights)[1..].to_vec();
    let roots = (0..n).filter(|&word| heads[word] == 0).collect::<Vec<_>>();
    let best = roots
        .iter()
        .copied()
        .max_by(|&a, &b| scores[a][0].total_cmp(&scores[b][0]))
        .unwrap_or(0);
    for word in roots {
        if word != best {
            heads[word] = best + 1;
        }
    }
    heads
}

/// Maximum spanning arborescence rooted at node 0 of `weights[head][dependent]`.
fn chu_liu_edmonds(weights: &[Vec<f64>]) -> Vec<usize> {
    let n = weights.len();
    let mut parent = vec![0; n];
    for dependent in 1..n {
        parent[dependent] = (0..n)
            .filter(|&head| head != dependent)
            .max_by(|&a, &b| weights[a][dependent].total_cmp(&weights[b][dependent]))
            .unwrap_or(0);
    }
    let Some(cycle) = find_cycle(&parent) else {
        return parent;
    };

    // Contract the cycle into one node `c`, solve, then expand.
    let mut in_cycle = vec![false; n];
    for &node in &cycle {
        in_cycle[node] = true;
    }
    let outside = (0..n).filter(|&node| !in_cycle[node]).collect::<Vec<_>>();
    let c = outside.len();
    let mut contracted = vec![vec![f64::NEG_INFINITY; c + 1]; c + 1];
    let mut enters = vec![0; c + 1];
    let mut leaves = vec![0; c + 1];
    for (a, &u) in outside.iter().enumerate() {
        for (b, &v) in outside.iter().enumerate() {
            if a != b {
                contracted[a][b] = weights[u][v];
            }
        }
        let best_by = |score: &dyn Fn(usize) -> f64| {
            cycle
                .iter()
                .copied()
                .map(|x| (x, score(x)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((cycle[0], f64::NEG_INFINITY))
        };
        // u -> cycle replaces the cycle arc into x.
        let (x, score) = best_by(&|x| weights[u][x] - weights[parent[x]][x]);
        contracted[a][c] = score;
        enters[a] = x;
        if u != 0 {
            let (x, score) = best_by(&|x| weights[x][u]);
            contracted[c][a] = score;
            leaves[a] = x;
        }
    }
    let solved = chu_liu_edmonds(&contracted);
    let mut result = parent;
    for (b, &v) in outside.iter().enumerate().skip(1) {
        result[v] = if solved[b] == c {
            leaves[b]
        } else {
            outside[solved[b]]
        };
    }
    let entry = solved[c];
    result[enters[entry]] = outside[entry];
    result
}

fn find_cycle(parent: &[usize]) -> Option<Vec<usize>> {
    // 0 = unvisited, 1 = on the current path, 2 = known acyclic.
    let mut state = vec![0u8; parent.len()];
    state[0] = 2;
    for start in 1..parent.len() {
        let mut path = Vec::new();
        let mut node = start;
        while state[node] == 0 {
            state[node] = 1;
            path.push(node);
            node = parent[node];
        }
        if state[node] == 1 {
            let from = path.iter().position(|&v| v == node).unwrap_or(0);
            return Some(path[from..].to_vec());
        }
        for visited in path {
            state[visited] = 2;
        }
    }
    None
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
