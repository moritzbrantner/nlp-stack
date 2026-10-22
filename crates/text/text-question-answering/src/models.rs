//! Extractive-QA-owned model execution glue for the local ONNX answerer.

use std::collections::BTreeMap;
#[cfg(feature = "local-onnx")]
use std::fs;
#[cfg(feature = "local-onnx")]
use std::path::{Path, PathBuf};

#[cfg(feature = "local-onnx")]
use media_core::DetectError;
use media_core::Result;
#[cfg(feature = "local-onnx")]
use model_runtime::{ModelBundle, ModelTask};
use serde::{Deserialize, Serialize};
#[cfg(feature = "local-onnx")]
use serde_json::Value;

#[cfg(feature = "local-onnx")]
use crate::tokenizer::TokenizerBundle;
use crate::QuestionAnsweringRuntime;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
/// Raw answer span emitted by an extractive QA backend.
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

/// Extractive question-answering backend.
pub trait QuestionAnsweringBackend {
    /// Answers a question from context text.
    fn answer(&mut self, question: &str, context: &str) -> Result<Vec<RawPrediction>>;

    /// Runtime family that produced the answers.
    fn runtime(&self) -> QuestionAnsweringRuntime;
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Decode options for extractive question-answering logits.
pub struct QuestionAnsweringDecodeOptions {
    /// Maximum token span length for one answer.
    pub max_answer_len: usize,
    /// Number of answers to return.
    pub top_k: usize,
    /// Optional minimum span probability score.
    pub min_score: Option<f32>,
}

impl Default for QuestionAnsweringDecodeOptions {
    fn default() -> Self {
        Self {
            max_answer_len: 30,
            top_k: 3,
            min_score: None,
        }
    }
}

#[cfg(feature = "local-onnx")]
#[derive(Debug, Clone, PartialEq)]
/// Tokenized QA pair plus sequence ids needed to isolate context spans.
pub struct QuestionAnsweringTokens {
    /// Token ids.
    pub input_ids: Vec<i64>,
    /// Attention mask values.
    pub attention_mask: Vec<i64>,
    /// Optional token type ids.
    pub token_type_ids: Option<Vec<i64>>,
    /// Token offsets relative to their source sequence.
    pub offsets: Vec<Option<(usize, usize)>>,
    /// Tokenizer sequence ids; context tokens use sequence id 1.
    pub sequence_ids: Vec<Option<usize>>,
}

#[cfg(feature = "local-onnx")]
/// Decodes RoBERTa-style extractive QA logits into context spans.
pub fn decode_question_answering_spans(
    context: &str,
    tokens: &QuestionAnsweringTokens,
    start_logits: &[f32],
    end_logits: &[f32],
    options: QuestionAnsweringDecodeOptions,
) -> Result<Vec<RawPrediction>> {
    if start_logits.len() != tokens.input_ids.len() || end_logits.len() != tokens.input_ids.len() {
        return Err(invalid_argument(
            "QA logits must match the tokenized sequence length",
        ));
    }
    if start_logits
        .iter()
        .chain(end_logits.iter())
        .any(|value| !value.is_finite())
    {
        return Err(invalid_argument("QA logits must be finite"));
    }
    let start_probabilities = softmax(start_logits);
    let end_probabilities = softmax(end_logits);
    let top_k = options.top_k.max(1);
    let max_answer_len = options.max_answer_len.max(1);
    let mut spans = Vec::new();
    for (start, start_probability) in start_probabilities
        .iter()
        .copied()
        .enumerate()
        .take(tokens.input_ids.len())
    {
        if tokens.sequence_ids.get(start).copied().flatten() != Some(1) {
            continue;
        }
        let Some((byte_start, _)) = tokens.offsets.get(start).copied().flatten() else {
            continue;
        };
        for (end, end_probability) in end_probabilities
            .iter()
            .copied()
            .enumerate()
            .take(tokens.input_ids.len().min(start + max_answer_len))
            .skip(start)
        {
            if tokens.sequence_ids.get(end).copied().flatten() != Some(1) {
                continue;
            }
            let Some((_, byte_end)) = tokens.offsets.get(end).copied().flatten() else {
                continue;
            };
            if byte_start >= byte_end || byte_end > context.len() {
                continue;
            }
            if !context.is_char_boundary(byte_start) || !context.is_char_boundary(byte_end) {
                continue;
            }
            let score = start_probability * end_probability;
            if options.min_score.is_some_and(|min_score| score < min_score) {
                continue;
            }
            let char_start = context[..byte_start].chars().count();
            let char_end = char_start + context[byte_start..byte_end].chars().count();
            let mut prediction = RawPrediction {
                kind: Some("answer_span".to_string()),
                text: Some(context[byte_start..byte_end].to_string()),
                score: Some(score),
                ..RawPrediction::default()
            };
            prediction
                .attributes
                .insert("byte_start".to_string(), byte_start.to_string());
            prediction
                .attributes
                .insert("byte_end".to_string(), byte_end.to_string());
            prediction
                .attributes
                .insert("char_start".to_string(), char_start.to_string());
            prediction
                .attributes
                .insert("char_end".to_string(), char_end.to_string());
            spans.push(prediction);
        }
    }
    spans.sort_by(|left, right| {
        right
            .score
            .unwrap_or(0.0)
            .total_cmp(&left.score.unwrap_or(0.0))
    });
    spans.truncate(top_k);
    Ok(spans)
}

#[cfg(feature = "local-onnx")]
#[derive(Debug)]
/// ONNX Runtime-backed extractive question answerer.
pub struct OnnxQuestionAnswerer {
    tokenizer: TokenizerBundle,
    session: runtime_onnx::OnnxSession,
    decode_options: QuestionAnsweringDecodeOptions,
}

#[cfg(feature = "local-onnx")]
impl OnnxQuestionAnswerer {
    /// Builds from a model bundle.
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self> {
        if bundle.manifest.task != ModelTask::QuestionAnswering {
            return Err(invalid_argument(format!(
                "ONNX QA bundle task must be question_answering, got {:?}",
                bundle.manifest.task
            )));
        }
        let config_path = required_bundle_file(&bundle, "config.json")?;
        let config = read_json(&config_path)?;
        let tokenizer = tokenizer_with_model_limit(TokenizerBundle::from_bundle(&bundle)?, &config);
        let model_path = bundle_files_with_extension(&bundle, "onnx")
            .into_iter()
            .next()
            .ok_or_else(|| invalid_argument("ONNX QA bundle must contain an `.onnx` model file"))?;
        let session =
            runtime_onnx::OnnxSession::from_file(model_path).map_err(runtime_onnx_error)?;
        Ok(Self {
            tokenizer,
            session,
            decode_options: QuestionAnsweringDecodeOptions::default(),
        })
    }

    /// Sets decode options.
    pub fn with_decode_options(mut self, options: QuestionAnsweringDecodeOptions) -> Self {
        self.decode_options = options;
        self
    }
}

#[cfg(feature = "local-onnx")]
impl QuestionAnsweringBackend for OnnxQuestionAnswerer {
    fn answer(&mut self, question: &str, context: &str) -> Result<Vec<RawPrediction>> {
        let tokens = self.tokenizer.tokenize_pair(question, context)?;
        let (start_logits, end_logits) = run_onnx_question_answerer(&mut self.session, &tokens)?;
        decode_question_answering_spans(
            context,
            &tokens,
            &start_logits,
            &end_logits,
            self.decode_options,
        )
    }

    fn runtime(&self) -> QuestionAnsweringRuntime {
        QuestionAnsweringRuntime::Onnx
    }
}

#[cfg(feature = "local-onnx")]
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

#[cfg(feature = "local-onnx")]
fn tokenizer_with_model_limit(tokenizer: TokenizerBundle, config: &Value) -> TokenizerBundle {
    match model_max_tokens_from_config(config) {
        Some(max_tokens) => tokenizer.max_length(max_tokens),
        None => tokenizer,
    }
}

#[cfg(feature = "local-onnx")]
fn model_max_tokens_from_config(config: &Value) -> Option<usize> {
    config
        .get("max_position_embeddings")
        .or_else(|| config.get("max_seq_len"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

#[cfg(feature = "local-onnx")]
fn required_bundle_file(bundle: &ModelBundle, remote_path: &str) -> Result<PathBuf> {
    bundle.file_path(remote_path).ok_or_else(|| {
        invalid_argument(format!(
            "model bundle `{}` is missing required file `{remote_path}`",
            bundle.manifest.name
        ))
    })
}

#[cfg(feature = "local-onnx")]
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

#[cfg(feature = "local-onnx")]
fn read_json(path: &Path) -> Result<Value> {
    let data = fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

#[cfg(feature = "local-onnx")]
fn run_onnx_question_answerer(
    session: &mut runtime_onnx::OnnxSession,
    tokens: &QuestionAnsweringTokens,
) -> Result<(Vec<f32>, Vec<f32>)> {
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
        return Err(invalid_argument("ONNX QA model exposes no inputs"));
    }
    let outputs = session.run(inputs).map_err(runtime_onnx_error)?;
    if outputs.len() < 2 {
        return Err(invalid_argument(
            "ONNX QA model must return start and end logits",
        ));
    }
    let mut start = None;
    let mut end = None;
    for output in &outputs {
        let logits = extract_qa_logits(
            runtime_onnx::f32_output_by_name_or_index(&outputs, &output.name, 0)
                .map_err(runtime_onnx_error)?,
            tokens.input_ids.len(),
        )?;
        let lower = output.name.to_lowercase();
        if lower.contains("start") {
            start = Some(logits);
        } else if lower.contains("end") {
            end = Some(logits);
        }
    }
    if start.is_none() || end.is_none() {
        start = Some(extract_qa_logits(
            runtime_onnx::f32_output_by_name_or_index(&outputs, "", 0)
                .map_err(runtime_onnx_error)?,
            tokens.input_ids.len(),
        )?);
        end = Some(extract_qa_logits(
            runtime_onnx::f32_output_by_name_or_index(&outputs, "", 1)
                .map_err(runtime_onnx_error)?,
            tokens.input_ids.len(),
        )?);
    }
    Ok((start.unwrap(), end.unwrap()))
}

#[cfg(feature = "local-onnx")]
fn extract_qa_logits(value: &runtime_onnx::OnnxF32Tensor, sequence_len: usize) -> Result<Vec<f32>> {
    match value.shape.as_slice() {
        [seq] if *seq == sequence_len => Ok(value.values.clone()),
        [1, seq] if *seq == sequence_len => Ok(value.values.clone()),
        _ => Err(invalid_argument(format!(
            "unsupported ONNX QA output shape `{:?}`",
            value.shape
        ))),
    }
}

#[cfg(feature = "local-onnx")]
fn runtime_onnx_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message)
        | runtime_onnx::OnnxRuntimeError::InvalidTensorShape(message) => invalid_argument(message),
        runtime_onnx::OnnxRuntimeError::Io(error) => DetectError::Io(error),
        other => DetectError::Source(other.to_string()),
    }
}

#[cfg(feature = "local-onnx")]
fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}
