//! Backend-neutral word-level syntax: the [`SyntaxBackend`] trait and the typed
//! tokens, syntactic words and provenance every backend returns.
//!
//! A backend tokenizes each sentence itself, including multi-word token expansion
//! (`zum` → `zu` + `dem`, `del` → `de` + `el`). Its surface tokens carry exact UTF-8
//! half-open byte spans into the analyzed text; its syntactic words point at the
//! token they belong to. The pipeline aligns these tokens to its own by span.
//! Backends never fill in missing output heuristically: a missing model, an
//! unsupported language or an unreachable service is a [`SyntaxBackendError`].

use serde::Serialize;
use text_core::TextSpan;

use crate::{DependencyRelation, MorphTagSet, PosTag};

#[derive(Debug, Clone, PartialEq, Eq)]
/// One sentence to analyze.
pub struct SyntaxSentenceRequest {
    /// Index into the caller's sentences; echoed in [`SyntaxSentence::sentence_index`].
    pub sentence_index: usize,
    /// UTF-8 half-open byte span of the sentence in [`SyntaxRequest::text`].
    pub span: TextSpan,
    /// ISO 639-1 language the sentence is analyzed in (routed per sentence).
    pub language: String,
}

#[derive(Debug, Clone, Copy)]
/// The sentences of one text to analyze.
pub struct SyntaxRequest<'a> {
    /// The full source text the sentence spans point into.
    pub text: &'a str,
    /// The sentences, each with its own language.
    pub sentences: &'a [SyntaxSentenceRequest],
}

impl SyntaxRequest<'_> {
    /// The text of `sentence`.
    pub fn sentence_text(&self, sentence: &SyntaxSentenceRequest) -> &str {
        &self.text[sentence.span.byte_start..sentence.span.byte_end]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// One model package a backend ran for a sentence.
pub struct SyntaxModelVersion {
    /// The backend stage (`tokenize`, `mwt`, `pos`, `lemma`, `depparse`).
    pub stage: String,
    /// The model package (`combined_charlm`).
    pub package: String,
    /// Where the model comes from (`stanfordnlp/stanza-de`).
    pub source: String,
    /// The pinned revision of `source`.
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// Which backend, backend version and models produced a sentence's syntax.
pub struct SyntaxProvenance {
    /// The backend (`stanza`).
    pub backend: String,
    /// The backend's version (`1.15.0`).
    pub backend_version: String,
    /// The models that ran, one per stage.
    pub models: Vec<SyntaxModelVersion>,
}

impl SyntaxProvenance {
    /// A compact producer label, e.g. `stanza-1.15.0:de/combined_charlm`, naming the
    /// backend version and the POS/dependency package.
    pub fn producer(&self, language: &str) -> String {
        let package = self
            .models
            .iter()
            .find(|model| model.stage == "depparse")
            .or_else(|| self.models.iter().find(|model| model.stage == "pos"))
            .map(|model| model.package.as_str())
            .unwrap_or("default");
        format!(
            "{}-{}:{language}/{package}",
            self.backend, self.backend_version
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// A surface token as the backend tokenized it.
pub struct SyntaxToken {
    /// UTF-8 half-open byte span in the analyzed text.
    pub span: TextSpan,
    /// The token's text (`zum`).
    pub text: String,
    /// Index of this token's first word in [`SyntaxSentence::words`].
    pub first_word: usize,
    /// How many syntactic words the token expands to (2 for `zum`).
    pub word_count: usize,
}

impl SyntaxToken {
    /// Indices of this token's words in [`SyntaxSentence::words`].
    pub fn words(&self) -> std::ops::Range<usize> {
        self.first_word..self.first_word + self.word_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// A syntactic word (a CoNLL-U word line).
pub struct SyntaxWord {
    /// Index of the surface token in [`SyntaxSentence::tokens`].
    pub token: usize,
    /// Word form (`dem` within `zum`).
    pub form: String,
    /// Lemma, when the backend produced one.
    pub lemma: Option<String>,
    /// Universal POS tag, when produced and a valid UPOS label.
    pub upos: Option<PosTag>,
    /// Language-specific POS tag.
    pub xpos: Option<String>,
    /// UD morphological features.
    pub feats: MorphTagSet,
    /// Index of the head word in [`SyntaxSentence::words`]; `None` for the root.
    pub head: Option<usize>,
    /// Universal dependency relation.
    pub relation: DependencyRelation,
    /// Language-specific relation subtype (`pass` in `nsubj:pass`).
    pub subtype: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// One sentence's word-level syntax from a backend.
pub struct SyntaxSentence {
    /// The request's sentence index.
    pub sentence_index: usize,
    /// The language the sentence was analyzed in.
    pub language: String,
    /// Backend and model versions.
    pub provenance: SyntaxProvenance,
    /// Surface tokens in text order.
    pub tokens: Vec<SyntaxToken>,
    /// Syntactic words in sentence order.
    pub words: Vec<SyntaxWord>,
}

impl SyntaxSentence {
    /// Checks the invariants every consumer relies on: tokens lie inside `span` in
    /// order and cover all of its non-whitespace text, tokens cover their words
    /// exactly, and heads point at words of this sentence.
    pub fn validate(&self, text: &str, span: TextSpan) -> Result<(), String> {
        let mut previous_end = span.byte_start;
        let mut next_word = 0;
        for (index, token) in self.tokens.iter().enumerate() {
            let (start, end) = (token.span.byte_start, token.span.byte_end);
            if start < previous_end || end > span.byte_end || start >= end {
                return Err(format!(
                    "token {index} span {start}..{end} is out of order or outside the sentence"
                ));
            }
            if !text[previous_end..start].trim().is_empty() {
                return Err(format!(
                    "text before token {index} ({previous_end}..{start}) is not covered by any token"
                ));
            }
            if text.get(start..end) != Some(token.text.as_str()) {
                return Err(format!(
                    "token {index} text {:?} differs from the source span",
                    token.text
                ));
            }
            if token.first_word != next_word || token.word_count == 0 {
                return Err(format!("token {index} does not cover the next words"));
            }
            next_word += token.word_count;
            previous_end = end;
        }
        if !text[previous_end..span.byte_end].trim().is_empty() {
            return Err(format!(
                "text after the last token ({previous_end}..{}) is not covered by any token",
                span.byte_end
            ));
        }
        if next_word != self.words.len() {
            return Err("words are not all covered by tokens".to_string());
        }
        for (index, word) in self.words.iter().enumerate() {
            if !self
                .tokens
                .get(word.token)
                .is_some_and(|token| token.words().contains(&index))
            {
                return Err(format!("word {index} points at the wrong token"));
            }
            if word
                .head
                .is_some_and(|head| head >= self.words.len() || head == index)
            {
                return Err(format!("word {index} has an invalid head"));
            }
        }
        Ok(())
    }
}

/// A backend producing tokens, lemmas, POS, morphology and dependency syntax.
///
/// Implementations are outer-edge adapters (the Stanza sidecar, a future native
/// parser); the request and results are backend-neutral. Execution policy such as
/// endpoints, timeouts or model locations belongs to the implementation's own
/// configuration.
pub trait SyntaxBackend {
    /// A stable name for reports (`stanza`).
    fn name(&self) -> &str;

    /// Whether the backend serves `language` (ISO 639-1). Sentences in other
    /// languages are reported unsupported and never sent.
    fn supports_language(&self, language: &str) -> bool;

    /// Analyzes every requested sentence, one result per request entry in order.
    fn analyze(
        &self,
        request: &SyntaxRequest<'_>,
    ) -> Result<Vec<SyntaxSentence>, SyntaxBackendError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Why a syntax backend produced no result. There is no fallback.
pub enum SyntaxBackendError {
    /// The backend service is not reachable (for example the sidecar is not running).
    Unavailable {
        /// The backend.
        backend: String,
        /// Where it was expected.
        endpoint: String,
        /// The underlying failure.
        message: String,
    },
    /// The backend runs, but the pinned model for `language` is missing or corrupt.
    ModelMissing {
        /// The backend.
        backend: String,
        /// The language whose model is missing.
        language: String,
        /// What is missing and how to install it.
        message: String,
    },
    /// The backend does not serve `language`.
    UnsupportedLanguage {
        /// The backend.
        backend: String,
        /// The requested language.
        language: String,
    },
    /// The backend answered with something that violates its protocol or the span
    /// invariants (wrong version, malformed body, misaligned offsets).
    Protocol {
        /// The backend.
        backend: String,
        /// What was wrong.
        message: String,
    },
    /// The backend reported an internal failure.
    Failed {
        /// The backend.
        backend: String,
        /// The reported failure.
        message: String,
    },
}

impl std::fmt::Display for SyntaxBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable {
                backend,
                endpoint,
                message,
            } => write!(
                f,
                "{backend} syntax backend is not reachable at {endpoint}: {message}"
            ),
            Self::ModelMissing {
                backend,
                language,
                message,
            } => write!(
                f,
                "{backend} has no usable model for `{language}`: {message}"
            ),
            Self::UnsupportedLanguage { backend, language } => {
                write!(f, "{backend} does not serve `{language}`")
            }
            Self::Protocol { backend, message } => {
                write!(f, "{backend} protocol error: {message}")
            }
            Self::Failed { backend, message } => write!(f, "{backend} failed: {message}"),
        }
    }
}

impl std::error::Error for SyntaxBackendError {}

/// Token-level layers derived from backend sentences, aligned to `tokens` by span.
#[derive(Debug, Default)]
pub(crate) struct BackendLayers {
    pub(crate) lemmas: Vec<crate::Lemma>,
    pub(crate) pos: Vec<crate::PosAnnotation>,
    pub(crate) morphology: Vec<crate::MorphAnnotation>,
    pub(crate) dependencies: Vec<crate::DependencyTree>,
}

/// Backends such as Stanza report no probabilities, so token-level annotations
/// carry this fixed confidence. It is not a calibrated score.
pub(crate) const UNSCORED_CONFIDENCE: f32 = 1.0;

/// Projects backend words onto the pipeline's tokens. Each backend token belongs
/// to the pipeline token containing its first byte. A pipeline token covering
/// several words (`zum` = `zu` + `dem`) takes the attachment, UPOS, lemma and
/// features of the word closest to the root among them (the adposition for `zum`),
/// so the token tree stays a tree; the full word-level analysis stays in
/// [`SyntaxSentence`].
pub(crate) fn backend_token_layers(
    results: &[SyntaxSentence],
    tokens: &[text_core::Token],
    morphology: bool,
    dependencies: bool,
) -> BackendLayers {
    use crate::{
        DependencyEdge, DependencyNode, DependencyTree, Lemma, MorphAnnotation, PosAnnotation,
    };

    let token_at = |byte: usize| {
        let index = tokens.partition_point(|token| token.span.byte_end <= byte);
        tokens
            .get(index)
            .filter(|token| token.span.byte_start <= byte)
            .map(|_| index)
    };
    let mut layers = BackendLayers::default();
    for sentence in results {
        let words = &sentence.words;
        let producer = sentence.provenance.producer(&sentence.language);
        let word_token = words
            .iter()
            .map(|word| token_at(sentence.tokens[word.token].span.byte_start))
            .collect::<Vec<_>>();
        let dominates = |ancestor: usize, word: usize| {
            let mut node = words[word].head;
            for _ in 0..words.len() {
                match node {
                    Some(head) if head == ancestor => return true,
                    Some(head) => node = words[head].head,
                    None => return false,
                }
            }
            false
        };
        // Pipeline tokens of this sentence in order, each with its words.
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        for (word, token) in word_token.iter().enumerate() {
            let Some(token) = *token else { continue };
            match groups.last_mut() {
                Some((last, members)) if *last == token => members.push(word),
                _ => groups.push((token, vec![word])),
            }
        }
        let mut tree = DependencyTree {
            sentence_index: sentence.sentence_index,
            root_token_index: None,
            nodes: Vec::new(),
            edges: Vec::new(),
        };
        for (token_index, members) in &groups {
            let token_index = *token_index;
            let Some(&top) = members.iter().find(|&&word| {
                !members
                    .iter()
                    .any(|&other| other != word && dominates(other, word))
            }) else {
                continue;
            };
            let word = &words[top];
            let reason = if members.len() > 1 {
                let forms = members
                    .iter()
                    .map(|&member| words[member].form.as_str())
                    .collect::<Vec<_>>()
                    .join("+");
                format!("{producer} words {forms}")
            } else {
                producer.clone()
            };
            if morphology {
                if let Some(lemma) = &word.lemma {
                    layers.lemmas.push(Lemma {
                        token_index,
                        value: lemma.clone(),
                        language: Some(sentence.language.clone()),
                        confidence: UNSCORED_CONFIDENCE,
                    });
                }
                if let Some(tag) = word.upos {
                    layers.pos.push(PosAnnotation {
                        token_index,
                        tag,
                        confidence: UNSCORED_CONFIDENCE,
                        reason: reason.clone(),
                    });
                }
                layers.morphology.push(MorphAnnotation {
                    token_index,
                    lemma: word.lemma.clone(),
                    tags: word.feats.clone(),
                    confidence: UNSCORED_CONFIDENCE,
                });
            }
            if dependencies {
                // Climb past words inside this token or outside every pipeline token
                // (for example punctuation the pipeline did not tokenize).
                let mut head = word.head;
                for _ in 0..words.len() {
                    match head {
                        Some(candidate)
                            if word_token[candidate].is_none_or(|token| token == token_index) =>
                        {
                            head = words[candidate].head;
                        }
                        _ => break,
                    }
                }
                let head_token = head.and_then(|head| word_token[head]);
                if head_token.is_none() && (word.head.is_none() || tree.root_token_index.is_none())
                {
                    tree.root_token_index = Some(token_index);
                }
                tree.nodes.push(DependencyNode {
                    token_index,
                    head_token_index: head_token,
                    relation: word.relation,
                    subtype: word.subtype.clone(),
                });
                if let Some(head_token) = head_token {
                    tree.edges.push(DependencyEdge {
                        head_token_index: head_token,
                        dependent_token_index: token_index,
                        relation: word.relation,
                        subtype: word.subtype.clone(),
                        confidence: UNSCORED_CONFIDENCE,
                    });
                }
            }
        }
        if dependencies {
            layers.dependencies.push(tree);
        }
    }
    layers
}
