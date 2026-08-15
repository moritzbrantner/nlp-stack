use std::collections::BTreeMap;

use text_core::Result;
use text_core::{TextSpan, Token};
pub use text_model_runtime::{
    TokenizedText, TokenizerBundle, TokenizerDownloadOptions, TokenizerPreset, TokenizerSource,
    TruncationStrategy,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing tokenization mode.
pub enum TokenizationMode {
    /// The word variant.
    Word,
    /// The subword variant.
    Subword,
    /// The mixed variant.
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for tokenizer policy.
pub struct TokenizerPolicy {
    /// The mode value.
    pub mode: TokenizationMode,
    /// The default source value.
    pub default_source: TokenizerSource,
    /// The language overrides value.
    pub language_overrides: BTreeMap<String, TokenizerSource>,
    /// The task overrides value.
    pub task_overrides: BTreeMap<String, TokenizerSource>,
    /// The model family overrides value.
    pub model_family_overrides: BTreeMap<String, TokenizerSource>,
}

impl Default for TokenizerPolicy {
    fn default() -> Self {
        Self {
            mode: TokenizationMode::Mixed,
            default_source: TokenizerSource::default(),
            language_overrides: BTreeMap::new(),
            task_overrides: BTreeMap::new(),
            model_family_overrides: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for tokenizer selection.
pub struct TokenizerSelection {
    /// The mode value.
    pub mode: TokenizationMode,
    /// The source value.
    pub source: Option<TokenizerSource>,
    /// Language tag for this value.
    pub language: Option<String>,
    /// The task value.
    pub task: Option<String>,
    /// The model family value.
    pub model_family: Option<String>,
    /// The reason value.
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Data type for tokenizer registry.
pub struct TokenizerRegistry {
    /// The policy value.
    pub policy: TokenizerPolicy,
}

impl TokenizerRegistry {
    /// Returns select.
    pub fn select(
        &self,
        language: Option<&str>,
        task: Option<&str>,
        model_family: Option<&str>,
    ) -> TokenizerSelection {
        let source = match self.policy.mode {
            TokenizationMode::Word => None,
            TokenizationMode::Subword | TokenizationMode::Mixed => Some(
                task.and_then(|task| self.policy.task_overrides.get(task))
                    .or_else(|| {
                        model_family
                            .and_then(|family| self.policy.model_family_overrides.get(family))
                    })
                    .or_else(|| {
                        language.and_then(|language| self.policy.language_overrides.get(language))
                    })
                    .cloned()
                    .unwrap_or_else(|| self.policy.default_source.clone()),
            ),
        };
        let reason = if let Some(task) =
            task.filter(|task| self.policy.task_overrides.contains_key(*task))
        {
            format!("task override for `{task}`")
        } else if let Some(family) =
            model_family.filter(|family| self.policy.model_family_overrides.contains_key(*family))
        {
            format!("model family override for `{family}`")
        } else if let Some(language) =
            language.filter(|language| self.policy.language_overrides.contains_key(*language))
        {
            format!("language override for `{language}`")
        } else {
            "default tokenizer policy".to_string()
        };
        TokenizerSelection {
            mode: self.policy.mode,
            source,
            language: language.map(ToString::to_string),
            task: task.map(ToString::to_string),
            model_family: model_family.map(ToString::to_string),
            reason,
        }
    }

    /// Returns align.
    pub fn align(
        &self,
        text: &str,
        tokens: &[Token],
        selection: &TokenizerSelection,
    ) -> Result<Option<TokenAlignmentMap>> {
        let Some(source) = selection.source.clone() else {
            return Ok(None);
        };
        let bundle = TokenizerBundle::from_cached_source(source)?;
        let tokenized = bundle.tokenize(text)?;
        Ok(Some(align_tokenized_text(
            text,
            tokens,
            selection.clone(),
            &tokenized,
        )?))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for subword span.
pub struct SubwordSpan {
    /// The index value.
    pub index: usize,
    /// The input identifier value.
    pub input_id: i64,
    /// The span value.
    pub span: Option<TextSpan>,
    /// Text content for this value.
    pub text: Option<String>,
    /// The token type identifier value.
    pub token_type_id: Option<i64>,
    /// The attention value.
    pub attention: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for aligned token.
pub struct AlignedToken {
    /// The token index value.
    pub token_index: usize,
    /// The token value.
    pub token: Token,
    /// The subword indices value.
    pub subword_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for token alignment map.
pub struct TokenAlignmentMap {
    /// The selection value.
    pub selection: TokenizerSelection,
    /// The subwords value.
    pub subwords: Vec<SubwordSpan>,
    /// The aligned tokens value.
    pub aligned_tokens: Vec<AlignedToken>,
}

/// Returns align tokenized text.
pub fn align_tokenized_text(
    text: &str,
    tokens: &[Token],
    selection: TokenizerSelection,
    tokenized: &TokenizedText,
) -> Result<TokenAlignmentMap> {
    let subwords = tokenized
        .input_ids
        .iter()
        .enumerate()
        .map(|(index, input_id)| {
            let span = tokenized
                .offsets
                .get(index)
                .copied()
                .flatten()
                .and_then(|(start, end)| {
                    if start >= end {
                        None
                    } else {
                        byte_span_to_text_span(text, start, end)
                    }
                });
            let subword_text = span
                .map(|span| text[span.byte_start..span.byte_end].to_string())
                .filter(|value| !value.is_empty());
            SubwordSpan {
                index,
                input_id: *input_id,
                span,
                text: subword_text,
                token_type_id: tokenized
                    .token_type_ids
                    .as_ref()
                    .and_then(|values| values.get(index).copied()),
                attention: tokenized
                    .attention_mask
                    .get(index)
                    .copied()
                    .unwrap_or_default()
                    != 0,
            }
        })
        .collect::<Vec<_>>();

    let aligned_tokens = tokens
        .iter()
        .cloned()
        .enumerate()
        .map(|(token_index, token)| {
            let subword_indices = subwords
                .iter()
                .filter_map(|subword| {
                    let span = subword.span?;
                    spans_overlap(token.span, span).then_some(subword.index)
                })
                .collect();
            AlignedToken {
                token_index,
                token,
                subword_indices,
            }
        })
        .collect();

    Ok(TokenAlignmentMap {
        selection,
        subwords,
        aligned_tokens,
    })
}

fn spans_overlap(left: TextSpan, right: TextSpan) -> bool {
    left.byte_start < right.byte_end && right.byte_start < left.byte_end
}

fn byte_span_to_text_span(text: &str, byte_start: usize, byte_end: usize) -> Option<TextSpan> {
    if byte_start > byte_end || byte_end > text.len() {
        return None;
    }
    let char_start = text[..byte_start].chars().count();
    let char_end = text[..byte_end].chars().count();
    Some(TextSpan {
        byte_start,
        byte_end,
        char_start,
        char_end,
    })
}
