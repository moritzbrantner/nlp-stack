//! Embedding-local tokenizer glue for native embedding models.

#[cfg(feature = "tokenizers")]
use std::fs;
#[cfg(feature = "tokenizers")]
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
#[cfg(feature = "tokenizers")]
use text_core::{DetectError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Tokenized text prepared for native embedding models.
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

#[cfg(feature = "tokenizers")]
#[derive(Debug, Clone, PartialEq, Eq)]
/// Tokenizer file plus the model's maximum token count.
pub(crate) struct TokenizerBundle {
    tokenizer_path: PathBuf,
    pub(crate) max_length: Option<usize>,
}

#[cfg(feature = "tokenizers")]
impl TokenizerBundle {
    #[cfg(feature = "model-bundles")]
    pub(crate) fn new(tokenizer_path: impl Into<PathBuf>) -> Self {
        Self {
            tokenizer_path: tokenizer_path.into(),
            max_length: None,
        }
    }

    #[cfg(feature = "model-bundles")]
    pub(crate) fn max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    pub(crate) fn tokenize(&self, text: &str) -> Result<TokenizedText> {
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
}

#[cfg(feature = "tokenizers")]
fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer> {
    if path.file_name().and_then(|value| value.to_str()) == Some("vocab.txt") {
        let vocab_path = path.to_str().ok_or_else(|| {
            DetectError::InvalidArgument(format!(
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
