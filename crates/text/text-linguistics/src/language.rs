use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde::Serialize;
use text_core::{
    detect_script_profile, split_sentence_spans, tokenize_words, Sentence, TextProcessingOptions,
    TextSpan,
};

/// ISO 639-1 languages the detector distinguishes. English, German and Spanish are
/// the first-class targets; the others keep French, Italian, Portuguese and Dutch
/// text from being forced into one of them.
pub const DETECTABLE_LANGUAGES: &[&str] = &["en", "de", "es", "fr", "it", "pt", "nl"];

/// Producer id recorded for detected languages.
pub const LANGUAGE_DETECTOR_PRODUCER: &str = "lingua-1.8-high-accuracy";

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// One ranked language candidate.
pub struct LanguagePrediction {
    /// ISO 639-1 language code (`und` when there is no evidence).
    pub language: String,
    /// Detector confidence relative to the other [`DETECTABLE_LANGUAGES`] (0..=1). This
    /// is language-identification confidence only, not model or extraction confidence.
    pub confidence: f32,
    /// Dominant Unicode script of the text unit.
    pub script: Option<String>,
    /// Where the candidate came from.
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// Whether a text unit's language was decided.
pub enum LanguageDecision {
    /// The top candidate leads the runner-up by at least
    /// [`LanguageDetectionOptions::min_margin`].
    Detected,
    /// Candidates exist, but none reached the confidence threshold.
    Ambiguous,
    /// Fewer than [`LanguageDetectionOptions::min_tokens_for_decision`] words.
    TooShort,
    /// The dominant script is not covered by any detector language (for example
    /// Cyrillic). Candidates hold at most the script's usual language, at zero
    /// confidence.
    UnsupportedScript,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// The language decision for one sentence.
pub struct SentenceLanguage {
    /// Index of the sentence in the caller's sentence list.
    pub sentence_index: usize,
    /// UTF-8 half-open byte span of the sentence in the source text.
    pub span: TextSpan,
    /// Whether the language was decided.
    pub decision: LanguageDecision,
    /// Ranked candidates, best first (at most `1 + max_alternatives`).
    pub candidates: Vec<LanguagePrediction>,
}

impl SentenceLanguage {
    /// The detected language, only when [`LanguageDecision::Detected`].
    pub fn language(&self) -> Option<&LanguagePrediction> {
        if self.decision == LanguageDecision::Detected {
            self.candidates.first()
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// Document- and sentence-level language detection result.
pub struct LanguageProfile {
    /// Best candidate for the whole text (`und` when there is no evidence).
    pub primary: Option<LanguagePrediction>,
    /// Whether the whole text's language was decided.
    pub decision: LanguageDecision,
    /// Further whole-text candidates.
    pub alternatives: Vec<LanguagePrediction>,
    /// Dominant Unicode script.
    pub dominant_script: Option<String>,
    /// Whether scripts mix, sentences were detected as different languages, or the
    /// two best whole-text candidates are within `mixed_threshold`.
    pub is_mixed: bool,
    /// One decision per sentence (empty when sentence-level detection is off).
    pub sentences: Vec<SentenceLanguage>,
    /// Word tokens in the whole text.
    pub token_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Language detection thresholds and output limits.
pub struct LanguageDetectionOptions {
    /// Minimum words (tokens containing a letter) before a language is decided.
    pub min_tokens_for_decision: usize,
    /// Minimum lead of the top candidate's relative confidence over the runner-up
    /// before a language is decided. 0.25 decides 90.5% of the 6,840 UD 2.17 test
    /// sentences in en/de/es/fr/it/pt, and 99.7% of those decisions are right.
    pub min_margin: f32,
    /// Candidates kept after the best one.
    pub max_alternatives: usize,
    /// Whole-text candidates closer than this mark the text as mixed.
    pub mixed_threshold: f32,
    /// Whether [`LanguageDetector::detect_text`] also decides each sentence.
    pub sentence_level: bool,
}

impl Default for LanguageDetectionOptions {
    fn default() -> Self {
        Self {
            min_tokens_for_decision: 2,
            min_margin: 0.25,
            max_alternatives: 3,
            mixed_threshold: 0.12,
            sentence_level: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for language lexicon.
pub struct LanguageLexicon<T> {
    /// Language tag for this value.
    pub language: String,
    /// The entries value.
    pub entries: BTreeSet<T>,
}

impl<T: Ord> LanguageLexicon<T> {
    /// Creates a new value.
    pub fn new(language: impl Into<String>, entries: impl IntoIterator<Item = T>) -> Self {
        Self {
            language: language.into(),
            entries: entries.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for lexicon entry.
pub struct LexiconEntry {
    /// The term value.
    pub term: String,
    /// The category value.
    pub category: String,
    /// The weight value.
    pub weight: f32,
}

impl LexiconEntry {
    /// Creates a new value.
    pub fn new(term: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            term: term.into(),
            category: category.into(),
            weight: 1.0,
        }
    }

    /// Returns weight.
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for lexicon store.
pub struct LexiconStore {
    /// The stop words value.
    pub stop_words: BTreeMap<String, LanguageLexicon<String>>,
    /// The abbreviations value.
    pub abbreviations: BTreeMap<String, LanguageLexicon<String>>,
    /// The lexical classes value.
    pub lexical_classes: BTreeMap<String, Vec<LexiconEntry>>,
    /// The gazetteers value.
    pub gazetteers: BTreeMap<String, Vec<LexiconEntry>>,
    /// The valency hints value.
    pub valency_hints: BTreeMap<String, Vec<LexiconEntry>>,
    /// The sentiment terms value.
    pub sentiment_terms: BTreeMap<String, Vec<LexiconEntry>>,
}

impl Default for LexiconStore {
    fn default() -> Self {
        Self::multilingual_defaults()
    }
}

impl LexiconStore {
    /// Returns multilingual defaults.
    pub fn multilingual_defaults() -> Self {
        let stop_words = BTreeMap::from([
            (
                "en".to_string(),
                LanguageLexicon::new(
                    "en",
                    [
                        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
                        "has", "have", "he", "i", "in", "is", "it", "of", "on", "or", "she",
                        "that", "the", "they", "this", "to", "was", "we", "were", "with", "you",
                    ]
                    .into_iter()
                    .map(String::from),
                ),
            ),
            (
                "de".to_string(),
                LanguageLexicon::new(
                    "de",
                    [
                        "aber", "als", "am", "auch", "auf", "bei", "das", "der", "die", "ein",
                        "eine", "er", "es", "für", "ich", "im", "in", "ist", "mit", "nicht", "sie",
                        "und", "von", "wir", "zu",
                    ]
                    .into_iter()
                    .map(String::from),
                ),
            ),
            (
                "es".to_string(),
                LanguageLexicon::new(
                    "es",
                    [
                        "a", "con", "de", "del", "el", "ella", "en", "es", "la", "las", "lo",
                        "los", "para", "por", "que", "se", "su", "un", "una", "y", "yo",
                    ]
                    .into_iter()
                    .map(String::from),
                ),
            ),
            (
                "fr".to_string(),
                LanguageLexicon::new(
                    "fr",
                    [
                        "à", "avec", "ce", "de", "des", "du", "elle", "en", "est", "et", "il",
                        "je", "la", "le", "les", "mais", "nous", "pas", "pour", "que", "qui", "un",
                        "une", "vous",
                    ]
                    .into_iter()
                    .map(String::from),
                ),
            ),
        ]);

        let abbreviations = BTreeMap::from([(
            "en".to_string(),
            LanguageLexicon::new(
                "en",
                [
                    "mr.", "mrs.", "ms.", "dr.", "prof.", "inc.", "corp.", "ltd.",
                ]
                .into_iter()
                .map(String::from),
            ),
        )]);

        let lexical_classes = BTreeMap::from([
            (
                "discourse_markers".to_string(),
                vec![
                    LexiconEntry::new("however", "contrast"),
                    LexiconEntry::new("therefore", "cause"),
                    LexiconEntry::new("because", "cause"),
                    LexiconEntry::new("first", "transition"),
                    LexiconEntry::new("finally", "transition"),
                ],
            ),
            (
                "disfluencies".to_string(),
                vec![
                    LexiconEntry::new("uh", "disfluency"),
                    LexiconEntry::new("um", "disfluency"),
                    LexiconEntry::new("er", "disfluency"),
                    LexiconEntry::new("ah", "disfluency"),
                    LexiconEntry::new("hmm", "disfluency"),
                ],
            ),
        ]);

        let gazetteers = BTreeMap::from([
            (
                "months".to_string(),
                [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ]
                .into_iter()
                .map(|month| LexiconEntry::new(month, "month"))
                .collect(),
            ),
            (
                "org_suffixes".to_string(),
                ["inc", "corp", "ltd", "llc", "gmbh", "ag"]
                    .into_iter()
                    .map(|suffix| LexiconEntry::new(suffix, "organization_suffix"))
                    .collect(),
            ),
        ]);

        let valency_hints = BTreeMap::from([(
            "verbs".to_string(),
            vec![
                LexiconEntry::new("say", "attribution"),
                LexiconEntry::new("announce", "attribution"),
                LexiconEntry::new("launch", "action"),
                LexiconEntry::new("visit", "movement"),
                LexiconEntry::new("present", "communication"),
            ],
        )]);

        let sentiment_terms = BTreeMap::from([(
            "en".to_string(),
            vec![
                LexiconEntry::new("excellent", "positive").weight(1.5),
                LexiconEntry::new("reliable", "positive").weight(1.0),
                LexiconEntry::new("terrible", "negative").weight(2.0),
                LexiconEntry::new("broken", "negative").weight(1.5),
            ],
        )]);

        Self {
            stop_words,
            abbreviations,
            lexical_classes,
            gazetteers,
            valency_hints,
            sentiment_terms,
        }
    }

    /// Returns stop words for.
    pub fn stop_words_for(&self, language: &str) -> Option<&LanguageLexicon<String>> {
        self.stop_words.get(language)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Per-sentence language detection with [lingua](https://github.com/pemistahl/lingua-rs)
/// (Apache-2.0) in high-accuracy mode over [`DETECTABLE_LANGUAGES`].
///
/// Each sentence gets its own [`LanguageDecision`]; only [`LanguageDecision::Detected`]
/// sentences carry a language that callers may route language-specific models by.
pub struct LanguageDetector {
    /// Decision thresholds and output limits.
    pub options: LanguageDetectionOptions,
}

impl LanguageDetector {
    /// Detects the document language and, when `options.sentence_level` is set, the
    /// language of every sentence from the default sentence splitter.
    pub fn detect_text(&self, text: &str) -> LanguageProfile {
        let sentences = if self.options.sentence_level {
            split_sentence_spans(text, &TextProcessingOptions::default())
        } else {
            Vec::new()
        };
        self.detect_sentences(text, &sentences)
    }

    /// Detects the document language and the language of each of the caller's
    /// `sentences` (spans into `text`). `options.sentence_level` is ignored.
    pub fn detect_sentences(&self, text: &str, sentences: &[Sentence]) -> LanguageProfile {
        let token_count = tokenize_words(text).len();
        let script_profile = detect_script_profile(text);
        let (decision, mut candidates) = self.classify(text);
        if candidates.is_empty() {
            candidates.push(LanguagePrediction {
                language: "und".to_string(),
                confidence: 0.0,
                script: script_profile.dominant_script.clone(),
                reason: if token_count == 0 {
                    "no text".to_string()
                } else {
                    "no language evidence".to_string()
                },
            });
        }
        let primary = candidates.first().cloned();
        let alternatives = candidates.into_iter().skip(1).collect::<Vec<_>>();

        let sentences = sentences
            .iter()
            .enumerate()
            .map(|(sentence_index, sentence)| {
                let (decision, candidates) = self.classify(&sentence.text);
                SentenceLanguage {
                    sentence_index,
                    span: sentence.span,
                    decision,
                    candidates,
                }
            })
            .collect::<Vec<_>>();
        let sentence_languages = sentences
            .iter()
            .filter_map(|sentence| sentence.language())
            .map(|prediction| prediction.language.as_str())
            .collect::<BTreeSet<_>>();
        let close_alternative =
            primary
                .as_ref()
                .zip(alternatives.first())
                .is_some_and(|(first, second)| {
                    first.confidence - second.confidence <= self.options.mixed_threshold
                });
        let is_mixed = script_profile.is_mixed || sentence_languages.len() > 1 || close_alternative;

        LanguageProfile {
            primary,
            decision,
            alternatives,
            dominant_script: script_profile.dominant_script,
            is_mixed,
            sentences,
            token_count,
        }
    }

    /// Decides the language of one text unit (a sentence, a cue): the decision plus at
    /// most `1 + max_alternatives` ranked candidates, best first.
    pub fn classify(&self, text: &str) -> (LanguageDecision, Vec<LanguagePrediction>) {
        let script = detect_script_profile(text).dominant_script;
        if let Some(script) = script.as_deref().filter(|script| *script != "Latin") {
            let candidates = script_default_language(script)
                .map(|language| LanguagePrediction {
                    language: language.to_string(),
                    confidence: 0.0,
                    script: Some(script.to_string()),
                    reason: format!("{script} script; no language model covers it"),
                })
                .into_iter()
                .collect();
            return (LanguageDecision::UnsupportedScript, candidates);
        }

        let words = tokenize_words(text)
            .iter()
            .filter(|word| word.chars().any(char::is_alphabetic))
            .count();
        if words == 0 {
            return (LanguageDecision::TooShort, Vec::new());
        }
        let values = lingua_detector().compute_language_confidence_values(text);
        let margin = match values.as_slice() {
            [] => 0.0,
            [(_, top)] => *top,
            [(_, top), (_, second), ..] => top - second,
        };
        let candidates = values
            .into_iter()
            .filter(|(_, confidence)| *confidence > 0.0)
            .take(1 + self.options.max_alternatives)
            .map(|(language, confidence)| LanguagePrediction {
                language: language.iso_code_639_1().to_string(),
                confidence: confidence as f32,
                script: script.clone(),
                reason: "lingua relative confidence".to_string(),
            })
            .collect::<Vec<_>>();
        let decision = if words < self.options.min_tokens_for_decision {
            LanguageDecision::TooShort
        } else if !candidates.is_empty() && margin as f32 >= self.options.min_margin {
            LanguageDecision::Detected
        } else {
            LanguageDecision::Ambiguous
        };
        (decision, candidates)
    }
}

fn lingua_detector() -> &'static lingua::LanguageDetector {
    use lingua::Language;
    static DETECTOR: OnceLock<lingua::LanguageDetector> = OnceLock::new();
    DETECTOR.get_or_init(|| {
        lingua::LanguageDetectorBuilder::from_languages(&[
            Language::English,
            Language::German,
            Language::Spanish,
            Language::French,
            Language::Italian,
            Language::Portuguese,
            Language::Dutch,
        ])
        .build()
    })
}

/// Primary subtag of a BCP 47 tag, lowercased (`de-AT` -> `de`).
pub(crate) fn primary_language_subtag(language: &str) -> String {
    language
        .trim()
        .split(['-', '_'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn script_default_language(script: &str) -> Option<&'static str> {
    match script {
        "Han" | "Hiragana" | "Katakana" => Some("ja"),
        "Hangul" => Some("ko"),
        "Arabic" => Some("ar"),
        "Cyrillic" => Some("ru"),
        "Hebrew" => Some("he"),
        "Greek" => Some("el"),
        "Devanagari" => Some("hi"),
        _ => None,
    }
}
