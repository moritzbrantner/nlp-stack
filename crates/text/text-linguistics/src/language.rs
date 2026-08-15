use std::collections::{BTreeMap, BTreeSet};

use text_core::{
    detect_script_profile, split_sentence_spans, tokenize_words, TextProcessingOptions,
};

#[derive(Debug, Clone, PartialEq)]
/// Data type for language prediction.
pub struct LanguagePrediction {
    /// Language tag for this value.
    pub language: String,
    /// Confidence score for this value.
    pub confidence: f32,
    /// The script value.
    pub script: Option<String>,
    /// The reason value.
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for language profile.
pub struct LanguageProfile {
    /// The primary value.
    pub primary: Option<LanguagePrediction>,
    /// The alternatives value.
    pub alternatives: Vec<LanguagePrediction>,
    /// The dominant script value.
    pub dominant_script: Option<String>,
    /// The is mixed value.
    pub is_mixed: bool,
    /// The sentence predictions value.
    pub sentence_predictions: Vec<Option<LanguagePrediction>>,
    /// The token count value.
    pub token_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for language detection options.
pub struct LanguageDetectionOptions {
    /// The min tokens for decision value.
    pub min_tokens_for_decision: usize,
    /// The max alternatives value.
    pub max_alternatives: usize,
    /// The mixed threshold value.
    pub mixed_threshold: f32,
    /// The sentence level value.
    pub sentence_level: bool,
}

impl Default for LanguageDetectionOptions {
    fn default() -> Self {
        Self {
            min_tokens_for_decision: 2,
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
/// Data type for language detector.
pub struct LanguageDetector {
    /// The options value.
    pub options: LanguageDetectionOptions,
    /// The lexicons value.
    pub lexicons: LexiconStore,
}

impl LanguageDetector {
    /// Returns detect text.
    pub fn detect_text(&self, text: &str) -> LanguageProfile {
        let word_tokens = tokenize_words(text);
        let token_count = word_tokens.len();
        let script_profile = detect_script_profile(text);
        let dominant_script = script_profile.dominant_script.clone();

        let mut scored = self.language_scores(&word_tokens, dominant_script.as_deref());
        scored.sort_by(|left, right| {
            right
                .confidence
                .total_cmp(&left.confidence)
                .then_with(|| left.language.cmp(&right.language))
        });
        if scored.is_empty() {
            scored.push(LanguagePrediction {
                language: dominant_script
                    .as_deref()
                    .and_then(script_default_language)
                    .unwrap_or("und")
                    .to_string(),
                confidence: if token_count == 0 { 0.0 } else { 0.2 },
                script: dominant_script.clone(),
                reason: "script fallback".to_string(),
            });
        }

        let primary = scored.first().cloned();
        let alternatives = scored
            .iter()
            .skip(1)
            .take(self.options.max_alternatives)
            .cloned()
            .collect::<Vec<_>>();
        let is_mixed = primary
            .as_ref()
            .zip(alternatives.first())
            .map(|(first, second)| {
                (first.confidence - second.confidence).abs() <= self.options.mixed_threshold
            })
            .unwrap_or(script_profile.is_mixed)
            || script_profile.is_mixed;

        let sentence_predictions = if self.options.sentence_level {
            split_sentence_spans(text, &TextProcessingOptions::default())
                .into_iter()
                .map(|sentence| {
                    self.language_scores(
                        &tokenize_words(&sentence.text),
                        dominant_script.as_deref(),
                    )
                    .into_iter()
                    .max_by(|left, right| left.confidence.total_cmp(&right.confidence))
                })
                .collect()
        } else {
            Vec::new()
        };

        LanguageProfile {
            primary,
            alternatives,
            dominant_script,
            is_mixed,
            sentence_predictions,
            token_count,
        }
    }

    fn language_scores(
        &self,
        tokens: &[String],
        dominant_script: Option<&str>,
    ) -> Vec<LanguagePrediction> {
        let mut predictions = Vec::new();
        let lower_tokens = tokens
            .iter()
            .map(|token| token.to_lowercase())
            .collect::<Vec<_>>();
        for (language, lexicon) in &self.lexicons.stop_words {
            let hits = lower_tokens
                .iter()
                .filter(|token| lexicon.entries.contains(*token))
                .count();
            let lexical_score = if lower_tokens.is_empty() {
                0.0
            } else {
                hits as f32 / lower_tokens.len() as f32
            };
            let script_bonus = dominant_script
                .and_then(script_default_language)
                .map(|candidate| if candidate == language { 0.15 } else { 0.0 })
                .unwrap_or_default();
            let confidence = (lexical_score + script_bonus).clamp(0.0, 1.0);
            if confidence > 0.0 {
                predictions.push(LanguagePrediction {
                    language: language.clone(),
                    confidence,
                    script: dominant_script.map(ToString::to_string),
                    reason: if hits > 0 {
                        format!("matched {hits} stop words")
                    } else {
                        "script fallback".to_string()
                    },
                });
            }
        }
        if predictions.is_empty() {
            if let Some(language) = dominant_script.and_then(script_default_language) {
                predictions.push(LanguagePrediction {
                    language: language.to_string(),
                    confidence: 0.35,
                    script: dominant_script.map(ToString::to_string),
                    reason: "script fallback".to_string(),
                });
            }
        }
        predictions
    }
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
        "Latin" => Some("en"),
        _ => None,
    }
}
