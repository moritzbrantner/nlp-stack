//! Pinned multilingual natural-language inference (NLI) preset.
//!
//! The preset names a Hugging Face repository, a commit revision and the SHA-256
//! digest of every file the ONNX backend reads. Loading resolves the bundle through
//! foundation `model-runtime`, refuses a bundle whose revision or file digests differ
//! from the pins, and never downloads unless the caller asks for it. There is no
//! default model and no fallback: a missing or mismatched bundle is an error (owner
//! decision, belief-lab#63, 2026-11-04).

use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
/// Multilingual (en/de/es and more) three-way NLI models with pinned weights.
pub enum NliPreset {
    /// `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`: mDeBERTa-v3 base
    /// fine-tuned on XNLI and multilingual-nli-2mil7, labels entailment / neutral /
    /// contradiction, MIT licence. Run through its own ONNX export.
    #[serde(rename = "mdeberta-v3-base-xnli-multilingual")]
    MdebertaV3BaseXnliMultilingual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// One pinned file of an NLI preset bundle.
pub struct NliPresetFile {
    /// Path inside the Hugging Face repository.
    pub path: &'static str,
    /// Lower-case hex SHA-256 digest of the file content.
    pub sha256: &'static str,
    /// File size in bytes.
    pub size_bytes: u64,
}

const MDEBERTA_V3_BASE_XNLI_FILES: &[NliPresetFile] = &[
    NliPresetFile {
        path: "config.json",
        sha256: "bc7b85f164a17c1b007d87fb99d3676f4a4d2e6511d2288dd3f5334dc0f34e6b",
        size_bytes: 1_093,
    },
    NliPresetFile {
        path: "tokenizer.json",
        sha256: "e23095eb61ba944c7be3a5d3e8ec19e37ce7ced0daa03550bde03e83c21b3f8a",
        size_bytes: 16_316_151,
    },
    NliPresetFile {
        path: "onnx/model.onnx",
        sha256: "e9238a33e07411eae1d1cef1d5f22e5f556717ce586c827581bb59598a45fe37",
        size_bytes: 1_116_115_065,
    },
];

impl NliPreset {
    /// Every preset.
    pub const ALL: &'static [Self] = &[Self::MdebertaV3BaseXnliMultilingual];

    /// Stable preset id (also the bundle directory name).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MdebertaV3BaseXnliMultilingual => "mdeberta-v3-base-xnli-multilingual",
        }
    }

    /// Hugging Face repository id; reported as the model id.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::MdebertaV3BaseXnliMultilingual => {
                "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
            }
        }
    }

    /// Pinned Hugging Face commit.
    pub fn revision(self) -> &'static str {
        match self {
            Self::MdebertaV3BaseXnliMultilingual => "b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
        }
    }

    /// Licence of the pinned weights.
    pub fn license(self) -> &'static str {
        match self {
            Self::MdebertaV3BaseXnliMultilingual => "MIT",
        }
    }

    /// Pinned files with their digests.
    pub fn files(self) -> &'static [NliPresetFile] {
        match self {
            Self::MdebertaV3BaseXnliMultilingual => MDEBERTA_V3_BASE_XNLI_FILES,
        }
    }

    /// The `model-runtime` bundle spec, pinned to [`Self::revision`].
    #[cfg(feature = "model-bundles")]
    pub fn spec(self) -> model_runtime::HuggingFaceModelSpec {
        self.files().iter().fold(
            model_runtime::HuggingFaceModelSpec::new(
                self.repo_id(),
                model_runtime::ModelTask::ZeroShotClassification,
            )
            .name(self.as_str())
            .revision(self.revision()),
            |spec, file| spec.file(file.path),
        )
    }
}

impl std::fmt::Display for NliPreset {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for NliPreset {
    type Err = String;

    /// Parses a preset id or Hugging Face repository id.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        Self::ALL
            .iter()
            .copied()
            .find(|preset| {
                preset.as_str().eq_ignore_ascii_case(value)
                    || preset.repo_id().eq_ignore_ascii_case(value)
            })
            .ok_or_else(|| {
                format!(
                    "unknown NLI preset `{value}`; expected one of {}",
                    Self::ALL
                        .iter()
                        .map(|preset| preset.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Relation of a hypothesis to a premise.
pub enum NliLabel {
    /// The premise entails the hypothesis.
    Entailment,
    /// Neither entailment nor contradiction.
    Neutral,
    /// The premise contradicts the hypothesis.
    Contradiction,
}

impl NliLabel {
    /// Maps a model label name (`entailment`, `neutral`, `contradiction`, any case).
    pub fn from_model_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "entailment" => Some(Self::Entailment),
            "neutral" => Some(Self::Neutral),
            "contradiction" => Some(Self::Contradiction),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// One NLI judgment with the model's probabilities as its receipt.
pub struct NliPrediction {
    /// The most probable label.
    pub label: NliLabel,
    /// Probability of entailment.
    pub entailment: f32,
    /// Probability of neutral.
    pub neutral: f32,
    /// Probability of contradiction.
    pub contradiction: f32,
}

impl NliPrediction {
    /// Builds a prediction from raw logits ordered by `labels` (the model's `id2label`).
    pub fn from_logits(logits: &[f32], labels: &[NliLabel]) -> Option<Self> {
        if logits.len() != labels.len() || logits.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp = logits
            .iter()
            .map(|value| (value - max).exp())
            .collect::<Vec<_>>();
        let total: f32 = exp.iter().sum();
        let probability = |wanted: NliLabel| {
            labels
                .iter()
                .position(|label| *label == wanted)
                .map(|index| exp[index] / total)
        };
        let entailment = probability(NliLabel::Entailment)?;
        let neutral = probability(NliLabel::Neutral)?;
        let contradiction = probability(NliLabel::Contradiction)?;
        let label = if entailment >= neutral && entailment >= contradiction {
            NliLabel::Entailment
        } else if contradiction >= neutral {
            NliLabel::Contradiction
        } else {
            NliLabel::Neutral
        };
        Some(Self {
            label,
            entailment,
            neutral,
            contradiction,
        })
    }
}

/// Reads the three NLI labels from a model's `id2label`; anything else is an error.
pub fn nli_labels(model_labels: &[String]) -> Result<Vec<NliLabel>, String> {
    let labels = model_labels
        .iter()
        .map(|label| {
            NliLabel::from_model_label(label)
                .ok_or_else(|| format!("model label `{label}` is not an NLI label"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for wanted in [
        NliLabel::Entailment,
        NliLabel::Neutral,
        NliLabel::Contradiction,
    ] {
        if labels.iter().filter(|label| **label == wanted).count() != 1 {
            return Err(format!(
                "model labels {model_labels:?} must name entailment, neutral and contradiction once each"
            ));
        }
    }
    Ok(labels)
}

#[cfg(feature = "onnx")]
pub use loading::{load_nli_preset, NliPresetLoadOptions, OnnxNliClassifier};

#[cfg(feature = "onnx")]
mod loading {
    use std::fs::File;
    use std::io::Read;
    use std::path::{Path, PathBuf};

    use model_runtime::{resolve_or_download_bundle, ModelBundle, ModelBundleResolveOptions};
    use sha2::{Digest, Sha256};
    use text_core::{DetectError, Result};

    use super::{nli_labels, NliLabel, NliPredictionResult, NliPreset, NliPresetFile};
    use crate::models::{
        labels_from_config, read_json, required_bundle_file, run_onnx_pair_classifier,
        runtime_onnx_error, tokenizer_with_model_limit,
    };
    use crate::tokenizer::TokenizerBundle;

    #[derive(Debug, Clone, PartialEq, Eq)]
    /// Where NLI bundles live and whether they may be downloaded.
    pub struct NliPresetLoadOptions {
        /// Root directory of `model-runtime` bundles.
        pub bundle_root: PathBuf,
        /// Whether a missing bundle may be downloaded from Hugging Face.
        pub auto_download: bool,
        /// Whether downloads report progress.
        pub download_progress: bool,
        /// Optional Hugging Face token.
        pub hf_token: Option<String>,
    }

    impl Default for NliPresetLoadOptions {
        /// `TEXT_CLASSIFICATION_MODEL_DIR` (default `.model-runtime`), downloads off.
        fn default() -> Self {
            Self {
                bundle_root: std::env::var_os("TEXT_CLASSIFICATION_MODEL_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from(".model-runtime")),
                auto_download: false,
                download_progress: false,
                hf_token: None,
            }
        }
    }

    #[derive(Debug)]
    /// ONNX Runtime-backed three-way NLI classifier for one pinned preset.
    pub struct OnnxNliClassifier {
        preset: NliPreset,
        tokenizer: TokenizerBundle,
        session: runtime_onnx::OnnxSession,
        labels: Vec<NliLabel>,
    }

    impl OnnxNliClassifier {
        /// The preset this classifier was loaded from (model id and revision provenance).
        pub fn preset(&self) -> NliPreset {
            self.preset
        }

        /// Judges whether `premise` entails, contradicts or is neutral towards `hypothesis`.
        pub fn classify(&mut self, premise: &str, hypothesis: &str) -> NliPredictionResult {
            let tokens = self.tokenizer.tokenize_pair(premise, hypothesis)?;
            let logits = run_onnx_pair_classifier(&mut self.session, &tokens)?;
            super::NliPrediction::from_logits(&logits, &self.labels).ok_or_else(|| {
                DetectError::Source(format!(
                    "NLI model `{}` returned {} non-finite or mislabelled logits",
                    self.preset.repo_id(),
                    logits.len()
                ))
            })
        }
    }

    /// Resolves the pinned bundle, verifies its revision and file digests, and loads
    /// the ONNX model with the labels from its `config.json`.
    pub fn load_nli_preset(
        preset: NliPreset,
        options: &NliPresetLoadOptions,
    ) -> Result<OnnxNliClassifier> {
        let resolve = ModelBundleResolveOptions {
            bundle_root: options.bundle_root.clone(),
            auto_download: options.auto_download,
            download_progress: options.download_progress,
            hf_token: options.hf_token.clone(),
            ..ModelBundleResolveOptions::default()
        };
        let bundle = resolve_or_download_bundle(&preset.spec(), &resolve).map_err(|error| {
            DetectError::Source(format!(
                "NLI model `{}` is not available: {error}",
                preset.repo_id()
            ))
        })?;
        if bundle.manifest.revision != preset.revision() {
            return Err(DetectError::InvalidArgument(format!(
                "bundle `{}` has revision `{}`, but the preset pins `{}`",
                preset.as_str(),
                bundle.manifest.revision,
                preset.revision()
            )));
        }
        verify_bundle_digests(&bundle, preset.files())?;
        let config = read_json(&required_bundle_file(&bundle, "config.json")?)?;
        let labels =
            nli_labels(&labels_from_config(&config)).map_err(DetectError::InvalidArgument)?;
        let tokenizer = tokenizer_with_model_limit(TokenizerBundle::from_bundle(&bundle)?, &config);
        let model = required_bundle_file(&bundle, "onnx/model.onnx")?;
        let session = runtime_onnx::OnnxSession::from_file(model).map_err(runtime_onnx_error)?;
        Ok(OnnxNliClassifier {
            preset,
            tokenizer,
            session,
            labels,
        })
    }

    fn verify_bundle_digests(bundle: &ModelBundle, files: &[NliPresetFile]) -> Result<()> {
        for file in files {
            let path = bundle.file_path(file.path).ok_or_else(|| {
                DetectError::InvalidArgument(format!(
                    "model bundle `{}` is missing pinned file `{}`",
                    bundle.manifest.name, file.path
                ))
            })?;
            let size = std::fs::metadata(&path)?.len();
            if size != file.size_bytes {
                return Err(DetectError::InvalidArgument(format!(
                    "`{}` in bundle `{}` has {size} bytes, but the pin expects {}",
                    file.path, bundle.manifest.name, file.size_bytes
                )));
            }
            let digest = sha256_file(&path)?;
            if digest != file.sha256 {
                return Err(DetectError::InvalidArgument(format!(
                    "`{}` in bundle `{}` has SHA-256 {digest}, but the pin expects {}",
                    file.path, bundle.manifest.name, file.sha256
                )));
            }
        }
        Ok(())
    }

    fn sha256_file(path: &Path) -> Result<String> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1 << 20];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }
}

#[cfg(feature = "onnx")]
type NliPredictionResult = text_core::Result<NliPrediction>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_parse_by_id_and_repo_and_pin_digests() {
        for preset in NliPreset::ALL {
            assert_eq!(preset.as_str().parse::<NliPreset>(), Ok(*preset));
            assert_eq!(preset.repo_id().parse::<NliPreset>(), Ok(*preset));
            assert_eq!(preset.revision().len(), 40);
            let paths = preset
                .files()
                .iter()
                .map(|file| file.path)
                .collect::<Vec<_>>();
            assert_eq!(paths, ["config.json", "tokenizer.json", "onnx/model.onnx"]);
            for file in preset.files() {
                assert_eq!(file.sha256.len(), 64);
                assert!(file
                    .sha256
                    .chars()
                    .all(|c| matches!(c, '0'..='9' | 'a'..='f')));
                assert!(file.size_bytes > 0);
            }
        }
        assert!("bart-large-mnli".parse::<NliPreset>().is_err());
    }

    #[test]
    fn labels_must_be_the_three_nli_labels() {
        let labels = ["entailment", "neutral", "contradiction"].map(String::from);
        assert_eq!(
            nli_labels(&labels),
            Ok(vec![
                NliLabel::Entailment,
                NliLabel::Neutral,
                NliLabel::Contradiction
            ])
        );
        assert!(nli_labels(&["LABEL_0", "LABEL_1", "LABEL_2"].map(String::from)).is_err());
        assert!(nli_labels(&["entailment", "not_entailment"].map(String::from)).is_err());
    }

    #[test]
    fn predictions_follow_the_model_label_order() {
        // joeddav-style order: contradiction, neutral, entailment.
        let order = [
            NliLabel::Contradiction,
            NliLabel::Neutral,
            NliLabel::Entailment,
        ];
        let prediction = NliPrediction::from_logits(&[4.0, 0.0, -2.0], &order).unwrap();
        assert_eq!(prediction.label, NliLabel::Contradiction);
        assert!(prediction.contradiction > 0.9);
        let sum = prediction.entailment + prediction.neutral + prediction.contradiction;
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(NliPrediction::from_logits(&[f32::NAN, 0.0, 0.0], &order).is_none());
        assert!(NliPrediction::from_logits(&[0.0, 0.0], &order).is_none());
    }

    #[cfg(feature = "model-bundles")]
    #[test]
    fn spec_requests_the_pinned_revision_and_files() {
        let preset = NliPreset::MdebertaV3BaseXnliMultilingual;
        let spec = preset.spec();
        assert_eq!(spec.revision_value(), Some(preset.revision()));
        assert_eq!(spec.repo_id_value(), Some(preset.repo_id()));
        assert_eq!(spec.files.len(), 3);
    }
}
