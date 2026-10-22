//! Extractive-QA-local tokenizer glue for question/context pairs.

use std::fs;
use std::path::{Path, PathBuf};

use media_core::{DetectError, Result};
use model_runtime::ModelBundle;

use crate::models::QuestionAnsweringTokens;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Tokenizer file plus the model's maximum token count.
pub(crate) struct TokenizerBundle {
    tokenizer_path: PathBuf,
    max_length: Option<usize>,
}

impl TokenizerBundle {
    /// Uses the bundle's `tokenizer.json`, falling back to a WordPiece `vocab.txt`.
    pub(crate) fn from_bundle(bundle: &ModelBundle) -> Result<Self> {
        let tokenizer_path = bundle
            .file_path("tokenizer.json")
            .or_else(|| bundle.file_path("vocab.txt"))
            .ok_or_else(|| {
                DetectError::InvalidArgument(format!(
                    "model bundle `{}` is missing required tokenizer file `tokenizer.json` or `vocab.txt`",
                    bundle.manifest.name
                ))
            })?;
        Ok(Self {
            tokenizer_path,
            max_length: None,
        })
    }

    pub(crate) fn max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Tokenizes a `(question, context)` pair.
    pub(crate) fn tokenize_pair(
        &self,
        question: &str,
        context: &str,
    ) -> Result<QuestionAnsweringTokens> {
        let encoding = load_tokenizer(&self.tokenizer_path)?
            .encode((question, context), true)
            .map_err(|err| DetectError::Source(format!("failed to tokenize QA pair: {err}")))?;
        let limit = self.max_length.unwrap_or(usize::MAX);
        let widen = |values: &[u32]| {
            values
                .iter()
                .take(limit)
                .map(|value| i64::from(*value))
                .collect::<Vec<_>>()
        };
        Ok(QuestionAnsweringTokens {
            input_ids: widen(encoding.get_ids()),
            attention_mask: widen(encoding.get_attention_mask()),
            token_type_ids: Some(widen(encoding.get_type_ids())),
            offsets: encoding
                .get_offsets()
                .iter()
                .take(limit)
                .map(|(start, end)| Some((*start, *end)))
                .collect(),
            sequence_ids: encoding
                .get_sequence_ids()
                .into_iter()
                .take(limit)
                .collect(),
        })
    }
}

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
