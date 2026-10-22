//! Classification-owned model execution glue for local Candle and ONNX classifiers.

use std::collections::BTreeMap;
#[cfg(any(feature = "candle", feature = "onnx"))]
use std::path::{Path, PathBuf};

#[cfg(feature = "candle")]
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};
#[cfg(feature = "candle")]
use candle_nn::{Linear as CandleLinear, Module as CandleModule, VarBuilder as CandleVarBuilder};
#[cfg(feature = "candle")]
use candle_transformers::models::{bert as candle_bert, distilbert as candle_distilbert};
#[cfg(any(feature = "candle", feature = "onnx"))]
use model_runtime::{ModelBundle, ModelTask};
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "candle", feature = "onnx"))]
use serde_json::Value;
#[cfg(any(feature = "candle", feature = "onnx"))]
use text_core::DetectError;
use text_core::Result;

#[cfg(feature = "onnx")]
use crate::tokenizer::PairTokens;
#[cfg(feature = "candle")]
use crate::tokenizer::TokenizedText;
#[cfg(any(feature = "candle", feature = "onnx"))]
use crate::tokenizer::TokenizerBundle;
use crate::TextClassificationRuntime;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
/// Raw label prediction emitted by a classification backend.
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

/// Text or pair classification backend.
pub trait SequenceClassifier {
    /// Classifies text against optional labels.
    fn classify_text(&mut self, text: &str, labels: &[String]) -> Result<Vec<RawPrediction>>;

    /// Runtime family that produced the predictions.
    fn runtime(&self) -> TextClassificationRuntime;
}

/// Premise/hypothesis pair classifier for zero-shot and NLI-style workflows.
pub trait PairSequenceClassifier {
    /// Scores a premise against candidate hypotheses.
    fn classify_pairs(
        &mut self,
        premise: &str,
        hypotheses: &[String],
    ) -> Result<Vec<RawPrediction>>;

    /// Runtime family that produced the predictions.
    fn runtime(&self) -> TextClassificationRuntime;
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandleSequenceClassifierArchitecture {
    Bert,
    DistilBert,
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone)]
/// Candle sequence-classification facade for local text classification.
pub struct CandleSequenceClassifier {
    tokenizer: TokenizerBundle,
    labels: Vec<String>,
    config: Value,
    model_paths: Vec<PathBuf>,
    architecture: CandleSequenceClassifierArchitecture,
    #[cfg(feature = "candle")]
    device: CandleDevice,
}

#[cfg(feature = "candle")]
impl CandleSequenceClassifier {
    /// Builds from a model bundle.
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self> {
        if bundle.manifest.task != ModelTask::TextClassification {
            return Err(invalid_argument(format!(
                "Candle sequence-classification bundle task must be text_classification, got {:?}",
                bundle.manifest.task
            )));
        }
        let config_path = required_bundle_file(&bundle, "config.json")?;
        let config = read_json(&config_path)?;
        let tokenizer = tokenizer_with_model_limit(TokenizerBundle::from_bundle(&bundle)?, &config);
        let model_paths = bundle_files_with_extension(&bundle, "safetensors");
        if model_paths.is_empty() {
            return Err(invalid_argument(
                "Candle sequence-classification bundles must contain a `.safetensors` model file",
            ));
        }
        let architecture = sequence_classifier_architecture_from_config(&config)?;
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

    /// Classifies text.
    pub fn classify(&mut self, text: &str) -> Result<Vec<RawPrediction>> {
        #[cfg(feature = "candle")]
        {
            let tokens = self.tokenizer.tokenize(text)?;
            let logits = run_candle_sequence_classifier(
                &self.config,
                &self.model_paths,
                self.architecture,
                &tokens,
                &self.device,
            )?;
            sequence_predictions_from_logits(&logits, &self.labels)
        }
        #[cfg(not(feature = "candle"))]
        {
            let _ = (
                text,
                &self.tokenizer,
                &self.config,
                &self.model_paths,
                self.architecture,
            );
            Err(invalid_argument(
                "native Candle sequence classification requires the `candle` feature",
            ))
        }
    }
}

#[cfg(feature = "candle")]
impl SequenceClassifier for CandleSequenceClassifier {
    fn classify_text(&mut self, text: &str, _labels: &[String]) -> Result<Vec<RawPrediction>> {
        self.classify(text)
    }

    fn runtime(&self) -> TextClassificationRuntime {
        TextClassificationRuntime::Candle
    }
}

#[cfg(feature = "onnx")]
#[derive(Debug)]
/// ONNX Runtime-backed pair classifier for zero-shot/NLI scoring.
pub struct OnnxZeroShotClassifier {
    tokenizer: TokenizerBundle,
    session: runtime_onnx::OnnxSession,
    labels: Vec<String>,
}

#[cfg(feature = "onnx")]
impl OnnxZeroShotClassifier {
    /// Builds from a model bundle.
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self> {
        if bundle.manifest.task != ModelTask::ZeroShotClassification {
            return Err(invalid_argument(format!(
                "ONNX zero-shot bundle task must be zero_shot_classification, got {:?}",
                bundle.manifest.task
            )));
        }
        let config_path = required_bundle_file(&bundle, "config.json")?;
        let config = read_json(&config_path)?;
        let tokenizer = tokenizer_with_model_limit(TokenizerBundle::from_bundle(&bundle)?, &config);
        let model_path = preferred_onnx_model_path(&bundle)?;
        let session =
            runtime_onnx::OnnxSession::from_file(model_path).map_err(runtime_onnx_error)?;
        Ok(Self {
            tokenizer,
            session,
            labels: labels_from_config(&config),
        })
    }
}

#[cfg(feature = "onnx")]
impl PairSequenceClassifier for OnnxZeroShotClassifier {
    fn classify_pairs(
        &mut self,
        premise: &str,
        hypotheses: &[String],
    ) -> Result<Vec<RawPrediction>> {
        let mut predictions = Vec::new();
        for hypothesis in hypotheses {
            let tokens = self.tokenizer.tokenize_pair(premise, hypothesis)?;
            let logits = run_onnx_pair_classifier(&mut self.session, &tokens)?;
            let score = entailment_score_from_logits(&logits, &self.labels);
            predictions.push(RawPrediction {
                kind: Some("zero_shot_pair".to_string()),
                label: Some(hypothesis.clone()),
                text: Some(premise.to_string()),
                score: Some(score),
                ..RawPrediction::default()
            });
        }
        Ok(predictions)
    }

    fn runtime(&self) -> TextClassificationRuntime {
        TextClassificationRuntime::Onnx
    }
}

#[cfg(any(feature = "candle", feature = "onnx"))]
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

#[cfg(any(feature = "candle", feature = "onnx"))]
fn tokenizer_with_model_limit(tokenizer: TokenizerBundle, config: &Value) -> TokenizerBundle {
    match model_max_tokens_from_config(config) {
        Some(max_tokens) => tokenizer.max_length(max_tokens),
        None => tokenizer,
    }
}

#[cfg(any(feature = "candle", feature = "onnx"))]
fn model_max_tokens_from_config(config: &Value) -> Option<usize> {
    config
        .get("max_position_embeddings")
        .or_else(|| config.get("max_seq_len"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

#[cfg(feature = "candle")]
fn sequence_classifier_architecture_from_config(
    config: &Value,
) -> Result<CandleSequenceClassifierArchitecture> {
    let architectures = architectures_from_config(config);
    if architectures.contains(&"DistilBertForSequenceClassification") {
        return Ok(CandleSequenceClassifierArchitecture::DistilBert);
    }
    if architectures.contains(&"BertForSequenceClassification") {
        return Ok(CandleSequenceClassifierArchitecture::Bert);
    }
    Err(invalid_argument(format!(
        "unsupported Candle sequence classification architecture {}; supported: DistilBertForSequenceClassification, BertForSequenceClassification",
        if architectures.is_empty() {
            "<missing>".to_string()
        } else {
            architectures.join(", ")
        },
    )))
}

#[cfg(feature = "candle")]
fn architectures_from_config(config: &Value) -> Vec<&str> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
}

#[cfg(any(feature = "candle", feature = "onnx"))]
fn required_bundle_file(bundle: &ModelBundle, remote_path: &str) -> Result<PathBuf> {
    bundle.file_path(remote_path).ok_or_else(|| {
        invalid_argument(format!(
            "model bundle `{}` is missing required file `{remote_path}`",
            bundle.manifest.name
        ))
    })
}

#[cfg(any(feature = "candle", feature = "onnx"))]
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

#[cfg(feature = "onnx")]
fn preferred_onnx_model_path(bundle: &ModelBundle) -> Result<PathBuf> {
    let candidates = [
        "onnx/model.onnx",
        "onnx/model_quantized.onnx",
        "onnx/model_int8.onnx",
        "onnx/model_uint8.onnx",
    ];
    for candidate in candidates {
        if let Some(path) = bundle.file_path(candidate) {
            return Ok(path);
        }
    }
    bundle_files_with_extension(bundle, "onnx")
        .into_iter()
        .next()
        .ok_or_else(|| {
            invalid_argument("ONNX classification bundle must contain an `.onnx` model file")
        })
}

#[cfg(any(feature = "candle", feature = "onnx"))]
fn read_json(path: &Path) -> Result<Value> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

#[cfg(any(feature = "candle", feature = "onnx"))]
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
/// Converts sequence-classification logits into ranked predictions.
pub fn sequence_predictions_from_logits(
    logits: &[f32],
    labels: &[String],
) -> Result<Vec<RawPrediction>> {
    if logits.is_empty() {
        return Err(invalid_argument(
            "sequence classification logits must not be empty",
        ));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(invalid_argument(
            "sequence classification logits must be finite",
        ));
    }
    let scores = softmax(logits);
    let mut predictions = scores
        .into_iter()
        .enumerate()
        .map(|(index, score)| RawPrediction {
            kind: Some("sequence_classification".to_string()),
            label: Some(
                labels
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{index}")),
            ),
            score: Some(score),
            ..RawPrediction::default()
        })
        .collect::<Vec<_>>();
    predictions.sort_by(|left, right| {
        right
            .score
            .unwrap_or(0.0)
            .total_cmp(&left.score.unwrap_or(0.0))
    });
    Ok(predictions)
}

#[cfg(feature = "onnx")]
/// Returns the entailment probability from NLI logits.
pub fn entailment_score_from_logits(logits: &[f32], labels: &[String]) -> f32 {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return 0.0;
    }
    let scores = softmax(logits);
    let entailment_index = labels
        .iter()
        .position(|label| label.to_ascii_lowercase().contains("entail"))
        .unwrap_or_else(|| scores.len().saturating_sub(1));
    scores.get(entailment_index).copied().unwrap_or(0.0)
}

#[cfg(feature = "onnx")]
fn run_onnx_pair_classifier(
    session: &mut runtime_onnx::OnnxSession,
    tokens: &PairTokens,
) -> Result<Vec<f32>> {
    use runtime_onnx::{OnnxRunner, OnnxTensor, OnnxTensorValue};

    let metadata = session.metadata().map_err(runtime_onnx_error)?;
    let shape = vec![1, tokens.input_ids.len()];
    let mut inputs = Vec::new();
    for input in &metadata.inputs {
        let name = input.name.clone();
        let values = if name.contains("attention") || name.contains("mask") {
            tokens.attention_mask.clone()
        } else if name.contains("token_type") || name.contains("type_ids") {
            tokens
                .token_type_ids
                .clone()
                .unwrap_or_else(|| vec![0_i64; tokens.input_ids.len()])
        } else {
            tokens.input_ids.clone()
        };
        inputs.push(runtime_onnx::OnnxNamedTensor {
            name,
            tensor: OnnxTensorValue::I64(
                OnnxTensor::new(shape.clone(), values).map_err(runtime_onnx_error)?,
            ),
        });
    }
    if inputs.is_empty() {
        return Err(invalid_argument("ONNX pair classifier exposes no inputs"));
    }
    let outputs = session.run(inputs).map_err(runtime_onnx_error)?;
    let logits = runtime_onnx::f32_output_by_preferred_name_or_index(&outputs, &["logits"], 0)
        .map_err(runtime_onnx_error)?;
    extract_sequence_logits(logits)
}

#[cfg(feature = "onnx")]
fn extract_sequence_logits(value: &runtime_onnx::OnnxF32Tensor) -> Result<Vec<f32>> {
    match value.shape.as_slice() {
        [labels] => Ok(value.values[..*labels].to_vec()),
        [1, labels] => Ok(value.values[..*labels].to_vec()),
        _ => Err(invalid_argument(format!(
            "unsupported ONNX sequence classification output shape `{:?}`",
            value.shape
        ))),
    }
}

#[cfg(feature = "onnx")]
fn runtime_onnx_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message)
        | runtime_onnx::OnnxRuntimeError::InvalidTensorShape(message) => invalid_argument(message),
        runtime_onnx::OnnxRuntimeError::Io(error) => DetectError::Io(error),
        other => DetectError::Source(other.to_string()),
    }
}

#[cfg(feature = "candle")]
fn run_candle_sequence_classifier(
    config: &Value,
    model_paths: &[PathBuf],
    architecture: CandleSequenceClassifierArchitecture,
    tokens: &TokenizedText,
    device: &CandleDevice,
) -> Result<Vec<f32>> {
    let vb = candle_var_builder(model_paths, device)?;
    let prefixes = model_prefix_candidates(config);

    let (sequence_output, used_prefix) = match architecture {
        CandleSequenceClassifierArchitecture::Bert => {
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
        CandleSequenceClassifierArchitecture::DistilBert => {
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

    let pooled = sequence_output
        .narrow(1, 0, 1)
        .map_err(candle_error)?
        .squeeze(1)
        .map_err(candle_error)?;
    let pre_classifier_candidates = prioritized_layer_candidates(&used_prefix, "pre_classifier");
    let hidden = match load_first_candle_linear(&vb, &pre_classifier_candidates)? {
        Some(pre_classifier) => pre_classifier
            .forward(&pooled)
            .map_err(candle_error)?
            .relu()
            .map_err(candle_error)?,
        None => pooled,
    };
    let classifier_candidates = prioritized_layer_candidates(&used_prefix, "classifier");
    let classifier = load_required_candle_linear(&vb, &classifier_candidates, "classifier")?;
    let logits = classifier.forward(&hidden).map_err(candle_error)?;
    logits
        .flatten_all()
        .map_err(candle_error)?
        .to_vec1::<f32>()
        .map_err(candle_error)
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

#[cfg(any(feature = "candle", feature = "onnx"))]
fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(all(test, any(feature = "candle", feature = "onnx")))]
mod tests {
    use super::*;

    #[test]
    fn softmax_normalizes_scores() {
        let scores = softmax(&[1.0, 2.0, 3.0]);
        let total = scores.iter().sum::<f32>();
        assert!((total - 1.0).abs() < 0.0001);
        assert!(scores[2] > scores[1]);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn sequence_logits_decode_to_ranked_predictions() {
        let predictions =
            sequence_predictions_from_logits(&[0.0, 2.0], &["NEGATIVE".into(), "POSITIVE".into()])
                .unwrap();
        assert_eq!(predictions[0].label.as_deref(), Some("POSITIVE"));
        assert!(predictions[0].score.unwrap() > predictions[1].score.unwrap());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn entailment_score_prefers_named_label() {
        let labels = vec![
            "contradiction".to_string(),
            "neutral".to_string(),
            "entailment".to_string(),
        ];
        let score = entailment_score_from_logits(&[0.0, 0.0, 2.0], &labels);
        assert!(score > 0.7);
    }
}
