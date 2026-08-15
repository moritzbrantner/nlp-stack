#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::BTreeMap;
#[cfg(feature = "tokenizers")]
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "candle")]
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};
#[cfg(feature = "candle")]
use candle_nn::{Linear as CandleLinear, Module as CandleModule, VarBuilder as CandleVarBuilder};
#[cfg(feature = "candle")]
use candle_transformers::models::{bert as candle_bert, distilbert as candle_distilbert};
#[cfg(feature = "model-bundles")]
use jobs_core::BackgroundJobRunner;
use media_core::{DetectError, Result};
#[cfg(feature = "model-bundles")]
use model_runtime::{
    jobs::spawn_model_download_job, HuggingFaceDownloader, HuggingFaceModelSpec, ModelBundle,
    ModelBundleStore, ModelTask,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
/// Device preference for Candle-backed native text execution.
pub enum CandleDevicePreference {
    /// Run Candle models on CPU.
    #[serde(rename = "cpu")]
    #[default]
    Cpu,
    /// Run Candle models on a CUDA device.
    #[serde(rename = "cuda")]
    Cuda {
        /// CUDA device index.
        #[serde(rename = "deviceIndex")]
        device_index: usize,
    },
}

static CANDLE_DEVICE_PREFERENCE: OnceLock<RwLock<CandleDevicePreference>> = OnceLock::new();

fn candle_device_preference_lock() -> &'static RwLock<CandleDevicePreference> {
    CANDLE_DEVICE_PREFERENCE.get_or_init(|| RwLock::new(CandleDevicePreference::Cpu))
}

/// Sets the process-wide Candle device preference.
pub fn set_candle_device_preference(preference: CandleDevicePreference) {
    let mut guard = candle_device_preference_lock()
        .write()
        .unwrap_or_else(|err| err.into_inner());
    *guard = preference;
}

/// Returns the process-wide Candle device preference.
pub fn candle_device_preference() -> CandleDevicePreference {
    *candle_device_preference_lock()
        .read()
        .unwrap_or_else(|err| err.into_inner())
}

/// Validates that the requested Candle device can be initialized.
#[cfg(feature = "candle")]
pub fn validate_candle_device_preference(preference: CandleDevicePreference) -> Result<()> {
    match preference {
        CandleDevicePreference::Cpu => Ok(()),
        CandleDevicePreference::Cuda { device_index } => {
            #[cfg(feature = "cuda")]
            {
                if !candle_core::utils::cuda_is_available() {
                    return Err(candle_cuda_feature_error());
                }
                CandleDevice::new_cuda(device_index)
                    .map(|_| ())
                    .map_err(|err| candle_cuda_error(device_index, err))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device_index;
                Err(candle_cuda_feature_error())
            }
        }
    }
}

/// Builds the Candle device selected by the process-wide preference.
#[cfg(feature = "candle")]
pub fn candle_device_from_preference() -> Result<CandleDevice> {
    match candle_device_preference() {
        CandleDevicePreference::Cpu => Ok(CandleDevice::Cpu),
        CandleDevicePreference::Cuda { device_index } => {
            #[cfg(feature = "cuda")]
            {
                if !candle_core::utils::cuda_is_available() {
                    return Err(candle_cuda_feature_error());
                }
                CandleDevice::new_cuda(device_index)
                    .map_err(|err| candle_cuda_error(device_index, err))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device_index;
                Err(candle_cuda_feature_error())
            }
        }
    }
}

#[cfg(feature = "candle")]
fn candle_cuda_feature_error() -> DetectError {
    DetectError::Source(
        "Candle CUDA requested but this binary was built without the `cuda` feature".to_string(),
    )
}

#[cfg(all(feature = "candle", feature = "cuda"))]
fn candle_cuda_error(device_index: usize, error: candle_core::Error) -> DetectError {
    DetectError::Source(format!(
        "failed to initialize Candle CUDA device {device_index}: {error}"
    ))
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing truncation strategy.
pub enum TruncationStrategy {
    /// Does not truncate tokenized input.
    None,
    /// Keeps the first sequence up to the configured maximum.
    LongestFirst,
    /// Keeps only the first sequence up to the configured maximum.
    OnlyFirst,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Tokenized text prepared for native text model runtimes.
pub struct TokenizedText {
    /// Token ids.
    pub input_ids: Vec<i64>,
    /// Attention mask values.
    pub attention_mask: Vec<i64>,
    /// Optional token type ids.
    pub token_type_ids: Option<Vec<i64>>,
    /// Byte offsets for each token, when available.
    pub offsets: Vec<Option<(usize, usize)>>,
}

impl TokenizedText {
    /// Truncates every token field to the same maximum length.
    pub fn truncate(&mut self, max_length: usize) {
        self.input_ids.truncate(max_length);
        self.attention_mask.truncate(max_length);
        if let Some(token_type_ids) = &mut self.token_type_ids {
            token_type_ids.truncate(max_length);
        }
        self.offsets.truncate(max_length);
    }
}

/// User-visible text model capabilities that can be validated by package surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TextModelCapability {
    /// Tokenizer-only loading and tokenization.
    Tokenizer,
    /// Token classification, including NER.
    TokenClassification,
    /// Text embedding.
    Embedding,
    /// Sequence or sentiment classification.
    SequenceClassification,
    /// Extractive question answering.
    QuestionAnswering,
    /// Native whisper.cpp transcription.
    WhisperCpp,
}

/// Report describing whether a text model bundle can be loaded by this build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextModelLoadReport {
    /// Stable model or preset id.
    pub model_id: String,
    /// Capability being validated.
    pub capability: TextModelCapability,
    /// Whether the model is a supported package surface.
    pub supported: bool,
    /// Whether all required local files are present for the requested runtime.
    pub loadable: bool,
    /// Required cargo feature for real loading/running.
    pub required_feature: Option<String>,
    /// User-facing setup command or note.
    pub required_setup: Option<String>,
    /// Package surface operation that smokes this path.
    pub smoke_operation: Option<String>,
    /// Bundle root inspected.
    pub bundle_root: PathBuf,
    /// Required bundle files.
    pub required_files: Vec<String>,
    /// Present required bundle files.
    pub present_files: Vec<String>,
    /// Missing required bundle files.
    pub missing_files: Vec<String>,
    /// Whether tokenizer loading was validated.
    pub tokenizer_loaded: bool,
    /// Whether model weight loading or presence was validated.
    pub weights_loaded: bool,
    /// Non-fatal notes from validation.
    pub diagnostics: Vec<String>,
}

impl TextModelLoadReport {
    /// Returns true when the report has no missing required files.
    pub fn bundle_present(&self) -> bool {
        self.missing_files.is_empty()
    }
}

/// Report from a minimal run through a loaded or deterministic text runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextModelRunReport {
    /// Stable model or preset id.
    pub model_id: String,
    /// Capability being exercised.
    pub capability: TextModelCapability,
    /// Smoke operation name.
    pub operation: String,
    /// Whether the sample run executed.
    pub ran: bool,
    /// Count of output records, tokens, embeddings, or spans.
    pub output_count: usize,
    /// Non-fatal notes from the run.
    pub diagnostics: Vec<String>,
}

/// Input for validating local text model bundle presence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextModelBundleCheck {
    /// Stable model or preset id.
    pub model_id: String,
    /// Capability being validated.
    pub capability: TextModelCapability,
    /// Bundle root to inspect.
    pub bundle_root: PathBuf,
    /// Required file paths relative to `bundle_root`.
    pub required_files: Vec<String>,
    /// Optional feature required for native loading.
    pub required_feature: Option<String>,
    /// Optional setup guidance.
    pub required_setup: Option<String>,
    /// Optional surface operation used as smoke coverage.
    pub smoke_operation: Option<String>,
    /// Whether this path has an implemented runner in this workspace.
    pub supported: bool,
}

impl TextModelBundleCheck {
    /// Creates a new bundle check.
    pub fn new(
        model_id: impl Into<String>,
        capability: TextModelCapability,
        bundle_root: impl Into<PathBuf>,
        required_files: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            capability,
            bundle_root: bundle_root.into(),
            required_files: required_files.into_iter().map(Into::into).collect(),
            required_feature: None,
            required_setup: None,
            smoke_operation: None,
            supported: true,
        }
    }

    /// Sets required feature guidance.
    pub fn required_feature(mut self, feature: impl Into<String>) -> Self {
        self.required_feature = Some(feature.into());
        self
    }

    /// Sets setup guidance.
    pub fn required_setup(mut self, setup: impl Into<String>) -> Self {
        self.required_setup = Some(setup.into());
        self
    }

    /// Sets the smoke operation id.
    pub fn smoke_operation(mut self, operation: impl Into<String>) -> Self {
        self.smoke_operation = Some(operation.into());
        self
    }

    /// Marks this catalog entry as reference-only.
    pub fn reference_only(mut self) -> Self {
        self.supported = false;
        self
    }
}

/// Validates required files for a local text model bundle without downloading anything.
pub fn validate_text_model_bundle(check: TextModelBundleCheck) -> TextModelLoadReport {
    let mut present_files = Vec::new();
    let mut missing_files = Vec::new();
    for file in &check.required_files {
        if check.bundle_root.join(file).exists() {
            present_files.push(file.clone());
        } else {
            missing_files.push(file.clone());
        }
    }
    let loadable = missing_files.is_empty() && check.supported;
    let diagnostics = if loadable {
        Vec::new()
    } else if check.supported {
        vec![format!(
            "model bundle `{}` is missing {} required file(s)",
            check.model_id,
            missing_files.len()
        )]
    } else {
        vec![format!(
            "model `{}` is reference-only until a native runner is implemented",
            check.model_id
        )]
    };
    TextModelLoadReport {
        model_id: check.model_id,
        capability: check.capability,
        supported: check.supported,
        loadable,
        required_feature: check.required_feature,
        required_setup: check.required_setup,
        smoke_operation: check.smoke_operation,
        bundle_root: check.bundle_root,
        required_files: check.required_files,
        present_files,
        missing_files,
        tokenizer_loaded: false,
        weights_loaded: loadable,
        diagnostics,
    }
}

/// Validates tokenizer loading and a minimal tokenization run for a local bundle.
pub fn validate_tokenizer_bundle(
    model_id: impl Into<String>,
    tokenizer_path: impl Into<PathBuf>,
    sample: &str,
) -> (TextModelLoadReport, Option<TextModelRunReport>) {
    let model_id = model_id.into();
    let tokenizer_path = tokenizer_path.into();
    let bundle_root = tokenizer_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let file_name = tokenizer_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("tokenizer.json")
        .to_string();
    let mut report = validate_text_model_bundle(
        TextModelBundleCheck::new(
            model_id.clone(),
            TextModelCapability::Tokenizer,
            bundle_root,
            [file_name],
        )
        .required_feature("tokenizers")
        .smoke_operation("runtime.tokenizeSummary"),
    );

    if !report.bundle_present() {
        return (report, None);
    }

    match TokenizerBundle::new(&tokenizer_path).tokenize(sample) {
        Ok(tokens) => {
            report.tokenizer_loaded = true;
            report.loadable = true;
            let output_count = tokens.input_ids.len();
            (
                report,
                Some(TextModelRunReport {
                    model_id,
                    capability: TextModelCapability::Tokenizer,
                    operation: "tokenize".to_string(),
                    ran: true,
                    output_count,
                    diagnostics: Vec::new(),
                }),
            )
        }
        Err(error) => {
            report.loadable = false;
            report
                .diagnostics
                .push(format!("failed to load or run tokenizer: {error}"));
            (report, None)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Built-in tokenizer presets.
pub enum TokenizerPreset {
    /// BERT base uncased.
    BertBaseUncased,
    /// DistilBERT SST-2.
    DistilbertSst2,
    #[default]
    /// MiniLM L6 v2.
    MiniLmL6V2,
}

impl TokenizerPreset {
    /// All built-in presets.
    pub const ALL: &'static [Self] = &[
        Self::BertBaseUncased,
        Self::DistilbertSst2,
        Self::MiniLmL6V2,
    ];

    /// Stable preset id.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BertBaseUncased => "bert-base-uncased",
            Self::DistilbertSst2 => "distilbert-sst2",
            Self::MiniLmL6V2 => "minilm-l6-v2",
        }
    }

    fn source(self) -> TokenizerSource {
        match self {
            Self::BertBaseUncased => TokenizerSource::huggingface("bert-base-uncased"),
            Self::DistilbertSst2 => TokenizerSource::huggingface_file(
                "distilbert-base-uncased-finetuned-sst-2-english",
                "vocab.txt",
            ),
            Self::MiniLmL6V2 => {
                TokenizerSource::huggingface("sentence-transformers/all-MiniLM-L6-v2")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Source for a tokenizer bundle.
pub enum TokenizerSource {
    /// Local tokenizer file.
    Local(PathBuf),
    /// Built-in preset.
    Preset(TokenizerPreset),
    /// Hugging Face tokenizer file.
    HuggingFace {
        /// Repository id.
        repo_id: String,
        /// Revision.
        revision: String,
        /// Tokenizer file.
        tokenizer_file: String,
    },
}

impl TokenizerSource {
    /// Uses a local tokenizer file.
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local(path.into())
    }

    /// Uses a built-in preset.
    pub fn preset(preset: TokenizerPreset) -> Self {
        Self::Preset(preset)
    }

    /// Uses a Hugging Face tokenizer.
    pub fn huggingface(repo_id: impl Into<String>) -> Self {
        Self::huggingface_file(repo_id, "tokenizer.json")
    }

    fn huggingface_file(repo_id: impl Into<String>, tokenizer_file: impl Into<String>) -> Self {
        Self::HuggingFace {
            repo_id: repo_id.into(),
            revision: "main".to_string(),
            tokenizer_file: tokenizer_file.into(),
        }
    }

    fn resolve_path(&self, options: &TokenizerDownloadOptions) -> Result<PathBuf> {
        match self {
            Self::Local(path) => Ok(path.clone()),
            Self::Preset(preset) => preset.source().resolve_path(options),
            Self::HuggingFace {
                repo_id,
                revision,
                tokenizer_file,
            } => {
                #[cfg(feature = "model-bundles")]
                {
                    let spec = HuggingFaceModelSpec::new(
                        repo_id.clone(),
                        ModelTask::Custom("tokenizer".to_string()),
                    )
                    .name(format!("{repo_id}-tokenizer"))
                    .revision(revision.clone())
                    .file(tokenizer_file.clone());
                    let bundle_root = options
                        .cache_dir
                        .clone()
                        .unwrap_or_else(|| PathBuf::from(".model-runtime"));
                    let store = ModelBundleStore::new(bundle_root).downloader(options.downloader());
                    let runner = BackgroundJobRunner::default();
                    let mut handle = spawn_model_download_job(&runner, spec, store)
                        .map_err(|err| DetectError::Source(err.to_string()))?;
                    let bundle = handle
                        .join_result()
                        .map_err(|err| DetectError::Source(err.to_string()))?;
                    bundle.file_path(tokenizer_file).ok_or_else(|| {
                        DetectError::Source(format!(
                            "downloaded tokenizer `{repo_id}` did not contain `{tokenizer_file}`"
                        ))
                    })
                }
                #[cfg(not(feature = "model-bundles"))]
                {
                    let _ = (repo_id, revision, tokenizer_file, options);
                    Err(invalid_argument(
                        "Hugging Face tokenizer downloads require the `model-bundles` feature",
                    ))
                }
            }
        }
    }
}

impl Default for TokenizerSource {
    fn default() -> Self {
        Self::Preset(TokenizerPreset::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Tokenizer download options.
pub struct TokenizerDownloadOptions {
    /// Optional cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Optional Hugging Face token.
    pub token: Option<String>,
    /// Whether to emit progress.
    pub progress: bool,
    /// Maximum download retries.
    pub max_retries: usize,
}

impl TokenizerDownloadOptions {
    /// Builds the configured downloader.
    #[cfg(feature = "model-bundles")]
    pub fn downloader(&self) -> HuggingFaceDownloader {
        let mut downloader = HuggingFaceDownloader::new()
            .progress(self.progress)
            .max_retries(self.max_retries);
        if let Some(cache_dir) = &self.cache_dir {
            downloader = downloader.cache_dir(cache_dir.clone());
        }
        if let Some(token) = &self.token {
            downloader = downloader.token(token.clone());
        }
        downloader
    }
}

impl Default for TokenizerDownloadOptions {
    fn default() -> Self {
        Self {
            cache_dir: None,
            token: None,
            progress: true,
            max_retries: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Tokenizer file plus runtime truncation settings.
pub struct TokenizerBundle {
    tokenizer_path: PathBuf,
    /// Optional maximum token count.
    pub max_length: Option<usize>,
    /// Truncation strategy.
    pub truncation: TruncationStrategy,
}

impl TokenizerBundle {
    /// Creates a new tokenizer bundle.
    pub fn new(tokenizer_path: impl Into<PathBuf>) -> Self {
        Self {
            tokenizer_path: tokenizer_path.into(),
            max_length: None,
            truncation: TruncationStrategy::None,
        }
    }

    /// Builds from a model bundle containing `tokenizer.json` or `vocab.txt`.
    #[cfg(feature = "model-bundles")]
    pub fn from_bundle(bundle: &ModelBundle) -> Result<Self> {
        let tokenizer_path = bundle
            .file_path("tokenizer.json")
            .or_else(|| bundle.file_path("vocab.txt"))
            .ok_or_else(|| {
                invalid_argument(format!(
                    "model bundle `{}` is missing required tokenizer file `tokenizer.json` or `vocab.txt`",
                    bundle.manifest.name
                ))
            })?;
        Ok(Self::new(tokenizer_path))
    }

    /// Builds from the default cached source.
    pub fn from_default_cached() -> Result<Self> {
        Self::from_cached_source(TokenizerSource::default())
    }

    /// Builds from a cached source.
    pub fn from_cached_source(source: TokenizerSource) -> Result<Self> {
        Self::from_cached_source_with_options(source, &TokenizerDownloadOptions::default())
    }

    /// Builds from a cached source with options.
    pub fn from_cached_source_with_options(
        source: TokenizerSource,
        options: &TokenizerDownloadOptions,
    ) -> Result<Self> {
        Ok(Self::new(source.resolve_path(options)?))
    }

    /// Sets the maximum token count.
    pub fn max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Sets truncation behavior.
    pub fn truncation(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation = strategy;
        self
    }

    /// Returns the tokenizer path.
    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer_path
    }

    #[cfg(feature = "tokenizers")]
    /// Tokenizes text with the configured tokenizer.
    pub fn tokenize(&self, text: &str) -> Result<TokenizedText> {
        let tokenizer = load_tokenizer(&self.tokenizer_path)?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|err| DetectError::Source(format!("failed to tokenize text: {err}")))?;
        let mut tokenized = TokenizedText {
            input_ids: encoding
                .get_ids()
                .iter()
                .map(|value| i64::from(*value))
                .collect(),
            attention_mask: encoding
                .get_attention_mask()
                .iter()
                .map(|value| i64::from(*value))
                .collect(),
            token_type_ids: Some(
                encoding
                    .get_type_ids()
                    .iter()
                    .map(|value| i64::from(*value))
                    .collect(),
            ),
            offsets: encoding
                .get_offsets()
                .iter()
                .map(|(start, end)| Some((*start, *end)))
                .collect(),
        };
        if let Some(max_length) = self.max_length {
            tokenized.truncate(max_length);
        }
        Ok(tokenized)
    }

    #[cfg(not(feature = "tokenizers"))]
    /// Tokenizes text when the tokenizer feature is available.
    pub fn tokenize(&self, _text: &str) -> Result<TokenizedText> {
        Err(invalid_argument(
            "tokenizer execution requires the `tokenizers` feature",
        ))
    }
}

#[cfg(feature = "tokenizers")]
fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer> {
    if path.file_name().and_then(|value| value.to_str()) == Some("vocab.txt") {
        let vocab_path = path.to_str().ok_or_else(|| {
            invalid_argument(format!(
                "tokenizer vocab path is not valid UTF-8: {}",
                path.display()
            ))
        })?;
        let model = tokenizers::models::wordpiece::WordPiece::from_file(vocab_path)
            .build()
            .map_err(|err| {
                DetectError::Source(format!(
                    "failed to load WordPiece vocab `{}`: {err}",
                    path.display()
                ))
            })?;
        let vocab = tokenizers::Model::get_vocab(&model);
        let cls_id = *vocab.get("[CLS]").unwrap_or(&101);
        let sep_id = *vocab.get("[SEP]").unwrap_or(&102);
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_normalizer(Some(bert_normalizer_for_vocab(path)));
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::bert::BertPreTokenizer));
        tokenizer.with_post_processor(Some(tokenizers::processors::bert::BertProcessing::new(
            ("[SEP]".to_string(), sep_id),
            ("[CLS]".to_string(), cls_id),
        )));
        return Ok(tokenizer);
    }

    tokenizers::Tokenizer::from_file(path).map_err(|err| {
        DetectError::Source(format!(
            "failed to load tokenizer `{}`: {err}",
            path.display()
        ))
    })
}

#[cfg(feature = "tokenizers")]
fn bert_normalizer_for_vocab(path: &Path) -> tokenizers::normalizers::bert::BertNormalizer {
    let config = path
        .parent()
        .map(|parent| parent.join("tokenizer_config.json"))
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|contents| serde_json::from_str::<serde_json::Value>(&contents).ok());
    let do_lower_case = config
        .as_ref()
        .and_then(|value| value.get("do_lower_case"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let strip_accents = config
        .as_ref()
        .and_then(|value| value.get("strip_accents"))
        .and_then(serde_json::Value::as_bool);
    tokenizers::normalizers::bert::BertNormalizer::new(true, true, strip_accents, do_lower_case)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Runtime backend family.
pub enum TextRuntimeBackend {
    /// Tokenizer-only runtime.
    Tokenizers,
    /// ONNX Runtime backend.
    Onnx,
    /// Candle backend.
    Candle,
    /// cuda-oxide backend.
    CudaOxide,
    /// Caller-supplied external backend.
    External,
    /// Deterministic heuristic backend.
    Heuristic,
}

/// Sequence labeling backend, used for NER and token classification.
pub trait SequenceLabeler {
    /// Labels text.
    fn label_text(&mut self, text: &str) -> Result<Vec<RawPrediction>>;

    /// Runtime backend.
    fn runtime_backend(&self) -> TextRuntimeBackend;
}

/// Text or pair classification backend.
pub trait SequenceClassifier {
    /// Classifies text against optional labels.
    fn classify_text(&mut self, text: &str, labels: &[String]) -> Result<Vec<RawPrediction>>;

    /// Runtime backend.
    fn runtime_backend(&self) -> TextRuntimeBackend;
}

/// Premise/hypothesis pair classifier for zero-shot and NLI-style workflows.
pub trait PairSequenceClassifier {
    /// Scores a premise against candidate hypotheses.
    fn classify_pairs(
        &mut self,
        premise: &str,
        hypotheses: &[String],
    ) -> Result<Vec<RawPrediction>>;

    /// Runtime backend.
    fn runtime_backend(&self) -> TextRuntimeBackend;
}

/// Query-document reranking backend.
pub trait TextReranker {
    /// Reranks documents for a query.
    fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>>;

    /// Runtime backend.
    fn runtime_backend(&self) -> TextRuntimeBackend;
}

/// Extractive question-answering backend.
pub trait QuestionAnsweringBackend {
    /// Answers a question from context text.
    fn answer(&mut self, question: &str, context: &str) -> Result<Vec<RawPrediction>>;

    /// Runtime backend.
    fn runtime_backend(&self) -> TextRuntimeBackend;
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

#[cfg(feature = "tokenizers")]
impl TokenizerBundle {
    /// Tokenizes a `(question, context)` pair.
    pub fn tokenize_pair(&self, question: &str, context: &str) -> Result<QuestionAnsweringTokens> {
        let tokenizer = load_tokenizer(&self.tokenizer_path)?;
        let encoding = tokenizer
            .encode((question, context), true)
            .map_err(|err| DetectError::Source(format!("failed to tokenize QA pair: {err}")))?;
        let mut tokens = QuestionAnsweringTokens {
            input_ids: encoding
                .get_ids()
                .iter()
                .map(|value| i64::from(*value))
                .collect(),
            attention_mask: encoding
                .get_attention_mask()
                .iter()
                .map(|value| i64::from(*value))
                .collect(),
            token_type_ids: Some(
                encoding
                    .get_type_ids()
                    .iter()
                    .map(|value| i64::from(*value))
                    .collect(),
            ),
            offsets: encoding
                .get_offsets()
                .iter()
                .map(|(start, end)| Some((*start, *end)))
                .collect(),
            sequence_ids: encoding.get_sequence_ids().to_vec(),
        };
        if let Some(max_length) = self.max_length {
            tokens.input_ids.truncate(max_length);
            tokens.attention_mask.truncate(max_length);
            if let Some(token_type_ids) = &mut tokens.token_type_ids {
                token_type_ids.truncate(max_length);
            }
            tokens.offsets.truncate(max_length);
            tokens.sequence_ids.truncate(max_length);
        }
        Ok(tokens)
    }
}

#[cfg(not(feature = "tokenizers"))]
impl TokenizerBundle {
    /// Tokenizes a `(question, context)` pair when the tokenizer feature is available.
    pub fn tokenize_pair(
        &self,
        _question: &str,
        _context: &str,
    ) -> Result<QuestionAnsweringTokens> {
        Err(invalid_argument(
            "QA pair tokenization requires the `tokenizers` feature",
        ))
    }
}

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

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
#[derive(Debug)]
/// ONNX Runtime-backed extractive question answerer.
pub struct OnnxQuestionAnswerer {
    tokenizer: TokenizerBundle,
    session: runtime_onnx::OnnxSession,
    decode_options: QuestionAnsweringDecodeOptions,
}

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
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

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
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

    fn runtime_backend(&self) -> TextRuntimeBackend {
        TextRuntimeBackend::Onnx
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum CandleSequenceClassifierArchitecture {
    Bert,
    DistilBert,
}

#[cfg(feature = "model-bundles")]
#[derive(Debug, Clone)]
/// Candle sequence-classification facade for local text classification.
pub struct CandleSequenceClassifier {
    tokenizer: TokenizerBundle,
    labels: Vec<String>,
    config: Value,
    model_paths: Vec<PathBuf>,
    architecture: CandleSequenceClassifierArchitecture,
}

#[cfg(feature = "model-bundles")]
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
        })
    }

    /// Returns labels.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Classifies text.
    pub fn classify(&mut self, text: &str) -> Result<Vec<RawPrediction>> {
        let tokens = self.tokenizer.tokenize(text)?;
        self.classify_tokenized(&tokens)
    }

    /// Classifies tokenized text.
    pub fn classify_tokenized(&self, tokens: &TokenizedText) -> Result<Vec<RawPrediction>> {
        #[cfg(feature = "candle")]
        {
            let logits = run_candle_sequence_classifier(
                &self.config,
                &self.model_paths,
                self.architecture,
                tokens,
            )?;
            sequence_predictions_from_logits(&logits, &self.labels)
        }
        #[cfg(not(feature = "candle"))]
        {
            let _ = (tokens, &self.config, &self.model_paths, self.architecture);
            Err(invalid_argument(
                "native Candle sequence classification requires the `candle` feature",
            ))
        }
    }
}

#[cfg(feature = "model-bundles")]
impl SequenceClassifier for CandleSequenceClassifier {
    fn classify_text(&mut self, text: &str, _labels: &[String]) -> Result<Vec<RawPrediction>> {
        self.classify(text)
    }

    fn runtime_backend(&self) -> TextRuntimeBackend {
        TextRuntimeBackend::Candle
    }
}

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
#[derive(Debug)]
/// ONNX Runtime-backed pair classifier for zero-shot/NLI scoring.
pub struct OnnxZeroShotClassifier {
    tokenizer: TokenizerBundle,
    session: runtime_onnx::OnnxSession,
    labels: Vec<String>,
}

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
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

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
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

    fn runtime_backend(&self) -> TextRuntimeBackend {
        TextRuntimeBackend::Onnx
    }
}

/// Token classification backend.
pub trait TokenClassifier {
    /// Classifies tokenized text.
    fn classify_tokenized_text(&mut self, tokens: &TokenizedText) -> Result<Vec<RawPrediction>>;

    /// Runtime backend.
    fn runtime_backend(&self) -> TextRuntimeBackend;
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
        })
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

    fn runtime_backend(&self) -> TextRuntimeBackend {
        TextRuntimeBackend::Candle
    }
}

impl TokenClassifier for CandleTokenClassifier {
    fn classify_tokenized_text(&mut self, tokens: &TokenizedText) -> Result<Vec<RawPrediction>> {
        self.classify_tokenized("", tokens)
    }

    fn runtime_backend(&self) -> TextRuntimeBackend {
        TextRuntimeBackend::Candle
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

#[allow(dead_code)]
fn tokenizer_with_model_limit(tokenizer: TokenizerBundle, config: &Value) -> TokenizerBundle {
    match model_max_tokens_from_config(config) {
        Some(max_tokens) => tokenizer.max_length(max_tokens),
        None => tokenizer,
    }
}

#[allow(dead_code)]
fn model_max_tokens_from_config(config: &Value) -> Option<usize> {
    config
        .get("max_position_embeddings")
        .or_else(|| config.get("max_seq_len"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
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

#[allow(dead_code)]
fn read_json(path: &Path) -> Result<Value> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

#[allow(dead_code)]
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

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
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

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
fn run_onnx_pair_classifier(
    session: &mut runtime_onnx::OnnxSession,
    tokens: &QuestionAnsweringTokens,
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

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
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

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
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

#[cfg(all(feature = "onnx", feature = "model-bundles"))]
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
) -> Result<Vec<f32>> {
    let device = candle_device_from_preference()?;
    let vb = candle_var_builder(model_paths, &device)?;
    let prefixes = model_prefix_candidates(config);

    let (sequence_output, used_prefix) = match architecture {
        CandleSequenceClassifierArchitecture::Bert => {
            let config: candle_bert::Config =
                serde_json::from_value(config.clone()).map_err(|err| {
                    invalid_argument(format!("failed to parse BERT config for Candle: {err}"))
                })?;
            let (model, used_prefix) = load_candle_bert_model(&vb, &config, &prefixes)?;
            let input_ids = candle_input_ids(tokens, &device)?;
            let token_type_ids = candle_token_type_ids(tokens, &device)?;
            let attention_mask = candle_attention_mask_keep(tokens, &device)?;
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
            let input_ids = candle_input_ids(tokens, &device)?;
            let attention_mask = candle_attention_mask_distil(tokens, &device)?;
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
fn run_candle_token_classifier(
    config: &Value,
    model_paths: &[PathBuf],
    architecture: CandleTokenClassifierArchitecture,
    tokens: &TokenizedText,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let device = candle_device_from_preference()?;
    let vb = candle_var_builder(model_paths, &device)?;
    let prefixes = model_prefix_candidates(config);

    let (sequence_output, used_prefix) = match architecture {
        CandleTokenClassifierArchitecture::Bert => {
            let config: candle_bert::Config =
                serde_json::from_value(config.clone()).map_err(|err| {
                    invalid_argument(format!("failed to parse BERT config for Candle: {err}"))
                })?;
            let (model, used_prefix) = load_candle_bert_model(&vb, &config, &prefixes)?;
            let input_ids = candle_input_ids(tokens, &device)?;
            let token_type_ids = candle_token_type_ids(tokens, &device)?;
            let attention_mask = candle_attention_mask_keep(tokens, &device)?;
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
            let input_ids = candle_input_ids(tokens, &device)?;
            let attention_mask = candle_attention_mask_distil(tokens, &device)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenized_text_truncates_fields_together() {
        let mut tokenized = TokenizedText {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: Some(vec![0, 0, 0]),
            offsets: vec![Some((0, 1)), Some((1, 2)), Some((2, 3))],
        };

        tokenized.truncate(2);

        assert_eq!(tokenized.input_ids, vec![1, 2]);
        assert_eq!(tokenized.attention_mask, vec![1, 1]);
        assert_eq!(tokenized.token_type_ids, Some(vec![0, 0]));
        assert_eq!(tokenized.offsets, vec![Some((0, 1)), Some((1, 2))]);
    }

    #[test]
    fn softmax_normalizes_scores() {
        let scores = softmax(&[1.0, 2.0, 3.0]);
        let total = scores.iter().sum::<f32>();
        assert!((total - 1.0).abs() < 0.0001);
        assert!(scores[2] > scores[1]);
    }

    #[test]
    fn sequence_logits_decode_to_ranked_predictions() {
        let predictions =
            sequence_predictions_from_logits(&[0.0, 2.0], &["NEGATIVE".into(), "POSITIVE".into()])
                .unwrap();
        assert_eq!(predictions[0].label.as_deref(), Some("POSITIVE"));
        assert!(predictions[0].score.unwrap() > predictions[1].score.unwrap());
    }

    #[test]
    fn distilbert_sst2_tokenizer_preset_uses_vocab_file() {
        match TokenizerPreset::DistilbertSst2.source() {
            TokenizerSource::HuggingFace {
                repo_id,
                tokenizer_file,
                ..
            } => {
                assert_eq!(repo_id, "distilbert-base-uncased-finetuned-sst-2-english");
                assert_eq!(tokenizer_file, "vocab.txt");
            }
            other => panic!("unexpected DistilBERT tokenizer source: {other:?}"),
        }
    }

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

    #[test]
    fn candle_device_preference_defaults_to_cpu() {
        assert_eq!(
            CandleDevicePreference::default(),
            CandleDevicePreference::Cpu
        );
    }

    #[test]
    fn candle_device_preference_serializes_cpu_metadata() {
        set_candle_device_preference(CandleDevicePreference::Cpu);
        let value = serde_json::to_value(candle_device_preference()).unwrap();
        assert_eq!(value, serde_json::json!({"kind": "cpu"}));
    }

    #[cfg(all(feature = "candle", not(feature = "cuda")))]
    #[test]
    fn candle_cuda_validation_reports_missing_cuda_feature() {
        let error =
            validate_candle_device_preference(CandleDevicePreference::Cuda { device_index: 0 })
                .unwrap_err()
                .to_string();
        assert!(error.contains("built without the `cuda` feature"));
    }
}
