//! Linguistics-owned tokenizer selection and loading for subword alignment and local models.

#[cfg(feature = "tokenizers")]
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "model-bundles")]
use model_runtime::{
    HuggingFaceDownloader, HuggingFaceModelSpec, ModelBundle, ModelBundleStore, ModelTask,
};
use serde::{Deserialize, Serialize};
use text_core::{DetectError, Result};

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
                    let bundle = store
                        .download(&spec)
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
        tokenizer
            .with_normalizer(Some(bert_normalizer_for_vocab(path)))
            .map_err(|err| {
                DetectError::Source(format!(
                    "failed to configure tokenizer normalizer for `{}`: {err}",
                    path.display()
                ))
            })?;
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
}
