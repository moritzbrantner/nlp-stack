//! Linguistics-owned execution glue for local token-classification models.

use std::collections::BTreeMap;
#[cfg(feature = "model-bundles")]
use std::path::Path;
use std::path::PathBuf;

#[cfg(feature = "candle")]
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};
#[cfg(feature = "candle")]
use candle_nn::{Linear as CandleLinear, Module as CandleModule, VarBuilder as CandleVarBuilder};
#[cfg(feature = "candle")]
use candle_transformers::models::{bert as candle_bert, distilbert as candle_distilbert};
#[cfg(feature = "model-bundles")]
use model_runtime::ModelBundle;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use text_core::{DetectError, Result};

use crate::model_tokenizer::{TokenizedText, TokenizerBundle};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
/// Text-oriented raw prediction shared by local and imported NLP runtimes.
pub struct RawPrediction {
    /// Raw prediction kind.
    pub kind: Option<String>,
    /// Label assigned by the model.
    pub label: Option<String>,
    /// Optional text span or document text.
    pub text: Option<String>,
    /// Confidence score.
    pub score: Option<f32>,
    #[serde(default)]
    /// Arbitrary runtime attributes, including offsets.
    pub attributes: BTreeMap<String, String>,
}

/// Sequence labeling backend, used for NER and token classification.
pub trait SequenceLabeler {
    /// Labels text.
    fn label_text(&mut self, text: &str) -> Result<Vec<RawPrediction>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum CandleTokenClassifierArchitecture {
    Bert,
    DistilBert,
}

#[derive(Debug, Clone)]
/// Candle token-classification facade.
pub struct CandleTokenClassifier {
    tokenizer: TokenizerBundle,
    labels: Vec<String>,
    config: Value,
    model_paths: Vec<PathBuf>,
    architecture: CandleTokenClassifierArchitecture,
    #[cfg(feature = "candle")]
    device: CandleDevice,
}

impl CandleTokenClassifier {
    /// Builds from a model bundle.
    #[cfg(feature = "model-bundles")]
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self> {
        let config_path = required_bundle_file(&bundle, "config.json")?;
        let config = read_json(&config_path)?;
        let tokenizer = tokenizer_with_model_limit(TokenizerBundle::from_bundle(&bundle)?, &config);
        let model_paths = bundle_files_with_extension(&bundle, "safetensors");
        if model_paths.is_empty() {
            return Err(invalid_argument(
                "Candle token classification bundles must contain a `.safetensors` model file",
            ));
        }
        let architecture = token_classifier_architecture_from_config(&config)?;
        Ok(Self {
            tokenizer,
            labels: labels_from_config(&config),
            config,
            model_paths,
            architecture,
            #[cfg(feature = "candle")]
            // CUDA when this build and machine support it; otherwise the CPU.
            device: CandleDevice::cuda_if_available(0).unwrap_or(CandleDevice::Cpu),
        })
    }

    /// Runs the model on `device` instead of the default (CUDA if available, else CPU).
    #[cfg(feature = "candle")]
    pub fn on_device(mut self, device: CandleDevice) -> Self {
        self.device = device;
        self
    }

    /// Returns labels.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Returns tokenizer.
    pub fn tokenizer(&self) -> &TokenizerBundle {
        &self.tokenizer
    }

    /// Classifies text.
    pub fn classify(&mut self, text: &str) -> Result<Vec<RawPrediction>> {
        let tokens = self.tokenizer.tokenize(text)?;
        self.classify_tokenized(text, &tokens)
    }

    /// Classifies tokenized text.
    pub fn classify_tokenized(
        &self,
        text: &str,
        tokens: &TokenizedText,
    ) -> Result<Vec<RawPrediction>> {
        #[cfg(feature = "candle")]
        {
            let (logits, shape) = run_candle_token_classifier(
                &self.config,
                &self.model_paths,
                self.architecture,
                tokens,
                &self.device,
            )?;
            token_predictions_from_logits(text, tokens, &logits, &shape, &self.labels)
        }
        #[cfg(not(feature = "candle"))]
        {
            let _ = (
                text,
                tokens,
                &self.config,
                &self.model_paths,
                self.architecture,
            );
            Err(invalid_argument(
                "native Candle token classification requires the `candle` feature",
            ))
        }
    }
}

impl SequenceLabeler for CandleTokenClassifier {
    fn label_text(&mut self, text: &str) -> Result<Vec<RawPrediction>> {
        self.classify(text)
    }
}

/// Converts token-classification logits into token predictions.
pub fn token_predictions_from_logits(
    text: &str,
    tokens: &TokenizedText,
    logits: &[f32],
    shape: &[usize],
    labels: &[String],
) -> Result<Vec<RawPrediction>> {
    let (sequence, label_count) = match shape {
        [sequence, labels] => (*sequence, *labels),
        [batch, sequence, labels] if *batch == 1 => (*sequence, *labels),
        _ => {
            return Err(invalid_argument(format!(
                "unsupported token classification output shape `{shape:?}`"
            )));
        }
    };
    if label_count == 0 || logits.len() != sequence * label_count {
        return Err(invalid_argument(
            "token classification output shape does not match logits",
        ));
    }

    let mut predictions = Vec::new();
    for token_index in 0..sequence {
        if tokens.attention_mask.get(token_index).copied().unwrap_or(0) == 0 {
            continue;
        }
        let Some((start, end)) = tokens.offsets.get(token_index).copied().flatten() else {
            continue;
        };
        if start >= end || (!text.is_empty() && end > text.len()) {
            continue;
        }
        let offset = token_index * label_count;
        let scores = softmax(&logits[offset..offset + label_count]);
        let Some((label_index, score)) = scores
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
        else {
            continue;
        };
        let label = labels
            .get(label_index)
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{label_index}"));
        let mut prediction = RawPrediction {
            kind: Some("token".to_string()),
            label: Some(label),
            text: (end <= text.len()).then(|| text[start..end].to_string()),
            score: Some(score),
            ..RawPrediction::default()
        };
        prediction
            .attributes
            .insert("byte_start".to_string(), start.to_string());
        prediction
            .attributes
            .insert("byte_end".to_string(), end.to_string());
        prediction
            .attributes
            .insert("token_index".to_string(), token_index.to_string());
        predictions.push(prediction);
    }
    Ok(predictions)
}

/// Numerically stable softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f32>();
    if total <= f32::EPSILON || !total.is_finite() {
        return vec![0.0; logits.len()];
    }
    for value in &mut values {
        *value /= total;
    }
    values
}

#[cfg(feature = "model-bundles")]
fn tokenizer_with_model_limit(tokenizer: TokenizerBundle, config: &Value) -> TokenizerBundle {
    match model_max_tokens_from_config(config) {
        Some(max_tokens) => tokenizer.max_length(max_tokens),
        None => tokenizer,
    }
}

#[cfg(feature = "model-bundles")]
fn model_max_tokens_from_config(config: &Value) -> Option<usize> {
    config
        .get("max_position_embeddings")
        .or_else(|| config.get("max_seq_len"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

#[cfg(feature = "model-bundles")]
fn token_classifier_architecture_from_config(
    config: &Value,
) -> Result<CandleTokenClassifierArchitecture> {
    let architectures = architectures_from_config(config);
    if architectures.contains(&"DistilBertForTokenClassification") {
        return Ok(CandleTokenClassifierArchitecture::DistilBert);
    }
    if architectures.contains(&"BertForTokenClassification") {
        return Ok(CandleTokenClassifierArchitecture::Bert);
    }
    Err(invalid_argument(format!(
        "unsupported Candle token classification architecture {}; supported: DistilBertForTokenClassification, BertForTokenClassification",
        if architectures.is_empty() {
            "<missing>".to_string()
        } else {
            architectures.join(", ")
        },
    )))
}

#[cfg(feature = "model-bundles")]
fn architectures_from_config(config: &Value) -> Vec<&str> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
}

#[cfg(feature = "model-bundles")]
fn required_bundle_file(bundle: &ModelBundle, remote_path: &str) -> Result<PathBuf> {
    bundle.file_path(remote_path).ok_or_else(|| {
        invalid_argument(format!(
            "model bundle `{}` is missing required file `{remote_path}`",
            bundle.manifest.name
        ))
    })
}

#[cfg(feature = "model-bundles")]
fn bundle_files_with_extension(bundle: &ModelBundle, extension: &str) -> Vec<PathBuf> {
    bundle
        .manifest
        .files
        .keys()
        .filter(|path| {
            Path::new(path).extension().and_then(|value| value.to_str()) == Some(extension)
        })
        .filter_map(|path| bundle.file_path(path))
        .collect::<Vec<_>>()
}

#[cfg(feature = "model-bundles")]
fn read_json(path: &Path) -> Result<Value> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

#[cfg(feature = "model-bundles")]
fn labels_from_config(config: &Value) -> Vec<String> {
    let Some(id2label) = config.get("id2label") else {
        return Vec::new();
    };
    if let Some(map) = id2label.as_object() {
        let mut labels = map
            .iter()
            .filter_map(|(key, value)| {
                Some((key.parse::<usize>().ok()?, value.as_str()?.to_string()))
            })
            .collect::<BTreeMap<_, _>>();
        if labels.is_empty() {
            return Vec::new();
        }
        let max = labels.keys().copied().max().unwrap_or(0);
        return (0..=max)
            .map(|index| {
                labels
                    .remove(&index)
                    .unwrap_or_else(|| format!("LABEL_{index}"))
            })
            .collect();
    }
    if let Some(values) = id2label.as_array() {
        return values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_str()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| format!("LABEL_{index}"))
            })
            .collect();
    }
    Vec::new()
}

#[cfg(feature = "candle")]
fn run_candle_token_classifier(
    config: &Value,
    model_paths: &[PathBuf],
    architecture: CandleTokenClassifierArchitecture,
    tokens: &TokenizedText,
    device: &CandleDevice,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let vb = candle_var_builder(model_paths, device)?;
    let prefixes = model_prefix_candidates(config);

    let (sequence_output, used_prefix) = match architecture {
        CandleTokenClassifierArchitecture::Bert => {
            let config: candle_bert::Config =
                serde_json::from_value(config.clone()).map_err(|err| {
                    invalid_argument(format!("failed to parse BERT config for Candle: {err}"))
                })?;
            let (model, used_prefix) = load_candle_bert_model(&vb, &config, &prefixes)?;
            let input_ids = candle_input_ids(tokens, device)?;
            let token_type_ids = candle_token_type_ids(tokens, device)?;
            let attention_mask = candle_attention_mask_keep(tokens, device)?;
            (
                model
                    .forward(&input_ids, &token_type_ids, Some(&attention_mask))
                    .map_err(candle_error)?,
                used_prefix,
            )
        }
        CandleTokenClassifierArchitecture::DistilBert => {
            let config: candle_distilbert::Config = serde_json::from_value(config.clone())
                .map_err(|err| {
                    invalid_argument(format!(
                        "failed to parse DistilBERT config for Candle: {err}"
                    ))
                })?;
            let (model, used_prefix) = load_candle_distilbert_model(&vb, &config, &prefixes)?;
            let input_ids = candle_input_ids(tokens, device)?;
            let attention_mask = candle_attention_mask_distil(tokens, device)?;
            (
                model
                    .forward(&input_ids, &attention_mask)
                    .map_err(candle_error)?,
                used_prefix,
            )
        }
    };

    let classifier_candidates = prioritized_layer_candidates(&used_prefix, "classifier");
    let classifier = load_required_candle_linear(&vb, &classifier_candidates, "classifier")?;
    let logits = classifier.forward(&sequence_output).map_err(candle_error)?;
    let shape = logits.dims().to_vec();
    let values = logits
        .flatten_all()
        .map_err(candle_error)?
        .to_vec1::<f32>()
        .map_err(candle_error)?;
    Ok((values, shape))
}

#[cfg(feature = "candle")]
fn candle_var_builder<'a>(
    model_paths: &'a [PathBuf],
    device: &CandleDevice,
) -> Result<CandleVarBuilder<'a>> {
    let paths = model_paths
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<_>>();
    unsafe { CandleVarBuilder::from_mmaped_safetensors(&paths, CandleDType::F32, device) }
        .map_err(candle_error)
}

#[cfg(feature = "candle")]
fn load_candle_bert_model(
    vb: &CandleVarBuilder<'_>,
    config: &candle_bert::Config,
    prefixes: &[String],
) -> Result<(candle_bert::BertModel, String)> {
    let mut last_error = None;
    for prefix in prefixes {
        let model_vb = if prefix.is_empty() {
            vb.clone()
        } else {
            vb.pp(prefix)
        };
        match candle_bert::BertModel::load(model_vb, config) {
            Ok(model) => return Ok((model, prefix.clone())),
            Err(err) => last_error = Some(err.to_string()),
        }
    }
    Err(DetectError::Source(format!(
        "failed to load Candle BERT model for prefixes [{}]{}",
        prefixes.join(", "),
        last_error.map(|err| format!(": {err}")).unwrap_or_default()
    )))
}

#[cfg(feature = "candle")]
fn load_candle_distilbert_model(
    vb: &CandleVarBuilder<'_>,
    config: &candle_distilbert::Config,
    prefixes: &[String],
) -> Result<(candle_distilbert::DistilBertModel, String)> {
    let mut last_error = None;
    for prefix in prefixes {
        let model_vb = if prefix.is_empty() {
            vb.clone()
        } else {
            vb.pp(prefix)
        };
        match candle_distilbert::DistilBertModel::load(model_vb, config) {
            Ok(model) => return Ok((model, prefix.clone())),
            Err(err) => last_error = Some(err.to_string()),
        }
    }
    Err(DetectError::Source(format!(
        "failed to load Candle DistilBERT model for prefixes [{}]{}",
        prefixes.join(", "),
        last_error.map(|err| format!(": {err}")).unwrap_or_default()
    )))
}

#[cfg(feature = "candle")]
fn load_first_candle_linear(
    vb: &CandleVarBuilder<'_>,
    layer_paths: &[String],
) -> Result<Option<CandleLinear>> {
    for layer_path in layer_paths {
        if let Some(linear) = load_candle_linear(vb, layer_path)? {
            return Ok(Some(linear));
        }
    }
    Ok(None)
}

#[cfg(feature = "candle")]
fn load_required_candle_linear(
    vb: &CandleVarBuilder<'_>,
    layer_paths: &[String],
    layer_name: &str,
) -> Result<CandleLinear> {
    load_first_candle_linear(vb, layer_paths)?.ok_or_else(|| {
        DetectError::Source(format!(
            "failed to load Candle `{layer_name}` layer from [{}]",
            layer_paths.join(", ")
        ))
    })
}

#[cfg(feature = "candle")]
fn load_candle_linear(vb: &CandleVarBuilder<'_>, layer_path: &str) -> Result<Option<CandleLinear>> {
    let layer_vb = vb.pp(layer_path);
    if !layer_vb.contains_tensor("weight") {
        return Ok(None);
    }
    let weight = layer_vb.get_unchecked("weight").map_err(candle_error)?;
    let bias = if layer_vb.contains_tensor("bias") {
        Some(layer_vb.get_unchecked("bias").map_err(candle_error)?)
    } else {
        None
    };
    Ok(Some(CandleLinear::new(weight, bias)))
}

#[cfg(feature = "candle")]
fn prioritized_layer_candidates(primary_prefix: &str, suffix: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    if !primary_prefix.is_empty() {
        push_unique_string(&mut candidates, format!("{primary_prefix}.{suffix}"));
    }
    push_unique_string(&mut candidates, suffix.to_string());
    candidates
}

#[cfg(feature = "candle")]
fn model_prefix_candidates(config: &Value) -> Vec<String> {
    let mut prefixes = Vec::new();
    push_unique_string(&mut prefixes, String::new());
    if let Some(model_type) = config.get("model_type").and_then(Value::as_str) {
        push_unique_string(&mut prefixes, model_type.to_string());
    }
    push_unique_string(&mut prefixes, "bert".to_string());
    push_unique_string(&mut prefixes, "distilbert".to_string());
    push_unique_string(&mut prefixes, "0.auto_model".to_string());
    push_unique_string(&mut prefixes, "auto_model".to_string());
    push_unique_string(&mut prefixes, "model".to_string());
    prefixes
}

#[cfg(feature = "candle")]
fn push_unique_string(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

#[cfg(feature = "candle")]
fn candle_input_ids(tokens: &TokenizedText, device: &CandleDevice) -> Result<CandleTensor> {
    let values = tokens
        .input_ids
        .iter()
        .map(|value| {
            u32::try_from(*value).map_err(|_| {
                invalid_argument(format!(
                    "tokenizer produced an out-of-range input id for Candle: {value}"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    CandleTensor::from_vec(values, (1, tokens.input_ids.len()), device).map_err(candle_error)
}

#[cfg(feature = "candle")]
fn candle_token_type_ids(tokens: &TokenizedText, device: &CandleDevice) -> Result<CandleTensor> {
    let values = match &tokens.token_type_ids {
        Some(values) => values
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| {
                    invalid_argument(format!(
                        "tokenizer produced an out-of-range token type id for Candle: {value}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?,
        None => vec![0_u32; tokens.input_ids.len()],
    };
    CandleTensor::from_vec(values, (1, tokens.input_ids.len()), device).map_err(candle_error)
}

#[cfg(feature = "candle")]
fn candle_attention_mask_keep(
    tokens: &TokenizedText,
    device: &CandleDevice,
) -> Result<CandleTensor> {
    let values = tokens
        .attention_mask
        .iter()
        .map(|value| if *value == 0 { 0_u32 } else { 1_u32 })
        .collect::<Vec<_>>();
    CandleTensor::from_vec(values, (1, tokens.attention_mask.len()), device).map_err(candle_error)
}

#[cfg(feature = "candle")]
fn candle_attention_mask_distil(
    tokens: &TokenizedText,
    device: &CandleDevice,
) -> Result<CandleTensor> {
    let values = tokens
        .attention_mask
        .iter()
        .map(|value| if *value == 0 { 1_u8 } else { 0_u8 })
        .collect::<Vec<_>>();
    CandleTensor::from_vec(values, (1, tokens.attention_mask.len()), device).map_err(candle_error)
}

#[cfg(feature = "candle")]
fn candle_error(error: candle_core::Error) -> DetectError {
    DetectError::Source(format!("Candle runtime error: {error}"))
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}
