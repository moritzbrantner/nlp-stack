//! Per-sentence language routing: which operation ran for each sentence, with which
//! producer, and which operations are unsupported or undecided for its language.

use serde::Serialize;
use text_core::TextSpan;

use crate::language::SentenceLanguage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// The core linguistic operations every sentence reports on.
pub enum LinguisticOperation {
    /// Language identification.
    LanguageDetection,
    /// Sentence boundaries.
    SentenceSegmentation,
    /// Word tokens.
    Tokenization,
    /// Named entity recognition.
    NamedEntities,
    /// Lemmas and morphological features.
    Morphology,
    /// Part-of-speech tags.
    PartOfSpeech,
    /// Dependency trees.
    DependencySyntax,
}

impl LinguisticOperation {
    /// Every operation, in report order.
    pub const ALL: [Self; 7] = [
        Self::LanguageDetection,
        Self::SentenceSegmentation,
        Self::Tokenization,
        Self::NamedEntities,
        Self::Morphology,
        Self::PartOfSpeech,
        Self::DependencySyntax,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// How well an operation covers a sentence.
pub enum OperationSupport {
    /// Ran with a producer evaluated for the sentence's language, or with
    /// language-neutral rules.
    Supported,
    /// Ran, but the producer is unscored for this language (English heuristics,
    /// capitalization NER, a caller-supplied labeler). Results are kept and labelled.
    Partial,
    /// The sentence's language is known but no configured producer supports it.
    /// The analysis omits this operation's results for the sentence.
    Unsupported,
    /// The sentence's language is undecided, so no language-specific producer ran.
    /// The analysis omits this operation's results for the sentence.
    Unknown,
}

impl OperationSupport {
    /// Whether the operation's results for the sentence are kept in the analysis.
    pub fn keeps_results(self) -> bool {
        matches!(self, Self::Supported | Self::Partial)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// One operation's outcome for one sentence.
pub struct OperationReport {
    /// The operation.
    pub operation: LinguisticOperation,
    /// Coverage of the sentence.
    pub support: OperationSupport,
    /// Producer that ran (for example `ud-model:combo-german-gsd-ud2.17`); `None` when
    /// nothing ran.
    pub producer: Option<String>,
    /// Why the operation has this support level.
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// The routing record of one sentence.
pub struct SentenceAnalysis {
    /// Index into [`crate::LinguisticAnalysis::sentences`].
    pub sentence_index: usize,
    /// UTF-8 half-open byte span of the sentence in the source text.
    pub span: TextSpan,
    /// The language every language-specific operation used, if decided (ISO 639-1).
    pub language: Option<String>,
    /// Whether `language` was given by the caller instead of detected.
    pub language_pinned: bool,
    /// The sentence's language detection result.
    pub detection: SentenceLanguage,
    /// One report per [`LinguisticOperation::ALL`] entry, in that order.
    pub operations: Vec<OperationReport>,
}

impl SentenceAnalysis {
    /// The report for `operation`.
    pub fn operation(&self, operation: LinguisticOperation) -> Option<&OperationReport> {
        self.operations
            .iter()
            .find(|report| report.operation == operation)
    }

    /// Whether `operation`'s results for this sentence are kept.
    pub fn keeps(&self, operation: LinguisticOperation) -> bool {
        self.operation(operation)
            .is_some_and(|report| report.support.keeps_results())
    }
}

impl OperationReport {
    pub(crate) fn new(
        operation: LinguisticOperation,
        support: OperationSupport,
        producer: Option<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            operation,
            support,
            producer,
            reason: reason.into(),
        }
    }
}
