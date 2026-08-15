#![doc = include_str!("../README.md")]

mod corpus;
pub mod surface;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use text_core::{
    detailed_text_stats, split_sentence_spans, text_stats, tokenize, tokenize_words, word_counts,
    TextProcessingOptions, TextSpan, TextStats, TokenKind,
};
use text_core::{AnalysisEvent, DetectError, Result, TextAnalyzer, TextSegment, Timestamp};

pub use corpus::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for term frequency.
pub struct TermFrequency {
    /// The term value.
    pub term: String,
    /// Number of items represented by this value.
    pub count: usize,
    /// The frequency value.
    pub frequency: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for text feature summary.
pub struct TextFeatureSummary {
    /// The stats value.
    pub stats: TextStats,
    /// The unique terms value.
    pub unique_terms: usize,
    /// The lexical diversity value.
    pub lexical_diversity: f32,
    /// The top terms value.
    pub top_terms: Vec<TermFrequency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for stop words.
pub struct StopWords {
    /// Language tag for this value.
    pub language: Option<String>,
    /// The terms value.
    pub terms: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for keyword options.
pub struct KeywordOptions {
    /// The max terms value.
    pub max_terms: usize,
    /// The min term len value.
    pub min_term_len: usize,
    /// The stop words value.
    pub stop_words: StopWords,
}

impl Default for KeywordOptions {
    fn default() -> Self {
        Self {
            max_terms: 10,
            min_term_len: 3,
            stop_words: english_stop_words(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for keyword.
pub struct Keyword {
    /// Text content for this value.
    pub text: String,
    /// Score assigned to this value.
    pub score: f32,
    /// Number of items represented by this value.
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for ngram frequency.
pub struct NgramFrequency {
    /// The terms value.
    pub terms: Vec<String>,
    /// Number of items represented by this value.
    pub count: usize,
    /// The frequency value.
    pub frequency: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for shingle similarity.
pub struct ShingleSimilarity {
    /// The left count value.
    pub left_count: usize,
    /// The right count value.
    pub right_count: usize,
    /// The intersection count value.
    pub intersection_count: usize,
    /// The union count value.
    pub union_count: usize,
    /// The jaccard value.
    pub jaccard: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for readability summary.
pub struct ReadabilitySummary {
    /// The sentence count value.
    pub sentence_count: usize,
    /// The word count value.
    pub word_count: usize,
    /// The average sentence words value.
    pub average_sentence_words: f32,
    /// The average word chars value.
    pub average_word_chars: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for stem options.
pub struct StemOptions {
    /// The min term len value.
    pub min_term_len: usize,
    /// The stop words value.
    pub stop_words: StopWords,
}

impl Default for StemOptions {
    fn default() -> Self {
        Self {
            min_term_len: 1,
            stop_words: StopWords {
                language: Some("en".to_string()),
                terms: BTreeSet::new(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for extractive summary options.
pub struct ExtractiveSummaryOptions {
    /// The max sentences value.
    pub max_sentences: usize,
    /// The min sentence words value.
    pub min_sentence_words: usize,
    /// The stop words value.
    pub stop_words: StopWords,
}

impl Default for ExtractiveSummaryOptions {
    fn default() -> Self {
        Self {
            max_sentences: 3,
            min_sentence_words: 3,
            stop_words: english_stop_words(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for summary sentence.
pub struct SummarySentence {
    /// The index value.
    pub index: usize,
    /// Text content for this value.
    pub text: String,
    /// The span value.
    pub span: TextSpan,
    /// Score assigned to this value.
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for phrase keyword options.
pub struct PhraseKeywordOptions {
    /// Maximum phrases to return.
    pub max_phrases: usize,
    /// Minimum token count per phrase.
    pub min_phrase_terms: usize,
    /// Minimum term length.
    pub min_term_len: usize,
    /// Stop words.
    pub stop_words: StopWords,
}

impl Default for PhraseKeywordOptions {
    fn default() -> Self {
        Self {
            max_phrases: 10,
            min_phrase_terms: 1,
            min_term_len: 3,
            stop_words: english_stop_words(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for sentiment lexicon.
pub struct SentimentLexicon {
    /// The terms value.
    pub terms: BTreeMap<String, f32>,
    /// The neutral threshold value.
    pub neutral_threshold: f32,
}

impl Default for SentimentLexicon {
    fn default() -> Self {
        let terms = [
            ("amazing", 2.0),
            ("bad", -1.5),
            ("boring", -1.0),
            ("broken", -1.5),
            ("delight", 1.5),
            ("excellent", 2.0),
            ("fail", -1.5),
            ("failure", -1.5),
            ("fast", 0.8),
            ("good", 1.2),
            ("great", 1.7),
            ("happy", 1.4),
            ("hate", -2.0),
            ("love", 2.0),
            ("negative", -1.0),
            ("poor", -1.5),
            ("positive", 1.0),
            ("reliable", 1.0),
            ("sad", -1.2),
            ("slow", -0.8),
            ("terrible", -2.0),
            ("useful", 1.0),
            ("worst", -2.0),
        ]
        .into_iter()
        .map(|(term, score)| (term.to_string(), score))
        .collect();
        Self {
            terms,
            neutral_threshold: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for sentiment summary.
pub struct SentimentSummary {
    /// The positive score value.
    pub positive_score: f32,
    /// The negative score value.
    pub negative_score: f32,
    /// The compound value.
    pub compound: f32,
    /// The token count value.
    pub token_count: usize,
    /// The matched terms value.
    pub matched_terms: usize,
    /// Label assigned to this value.
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for entity rule set.
pub struct EntityRuleSet {
    /// The emails value.
    pub emails: bool,
    /// The URLs value.
    pub urls: bool,
    /// The mentions value.
    pub mentions: bool,
    /// The hashtags value.
    pub hashtags: bool,
    /// The numbers value.
    pub numbers: bool,
    /// The capitalized phrases value.
    pub capitalized_phrases: bool,
}

impl Default for EntityRuleSet {
    fn default() -> Self {
        Self {
            emails: true,
            urls: true,
            mentions: true,
            hashtags: true,
            numbers: true,
            capitalized_phrases: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for entity mention.
pub struct EntityMention {
    /// The kind value.
    pub kind: String,
    /// Text content for this value.
    pub text: String,
    /// The normalized value.
    pub normalized: String,
    /// The span value.
    pub span: TextSpan,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for text analysis options.
pub struct TextAnalysisOptions {
    /// The processing value.
    pub processing: TextProcessingOptions,
    /// The keywords value.
    pub keywords: KeywordOptions,
    /// The emit top terms value.
    pub emit_top_terms: usize,
    /// The emit patterns value.
    pub emit_patterns: bool,
}

impl Default for TextAnalysisOptions {
    fn default() -> Self {
        Self {
            processing: TextProcessingOptions::default(),
            keywords: KeywordOptions::default(),
            emit_top_terms: 5,
            emit_patterns: true,
        }
    }
}

/// Returns summarize text.
pub fn summarize_text(text: &str, max_terms: usize) -> TextFeatureSummary {
    let stats = text_stats(text);
    let stop_words = english_stop_words();
    let top_terms = top_terms(text, max_terms, &stop_words.terms);
    let unique_terms = word_counts(text).len();
    let lexical_diversity = if stats.words == 0 {
        0.0
    } else {
        unique_terms as f32 / stats.words as f32
    };
    TextFeatureSummary {
        stats,
        unique_terms,
        lexical_diversity,
        top_terms,
    }
}

/// Returns term frequencies.
pub fn term_frequencies(text: &str) -> Vec<TermFrequency> {
    let counts = word_counts(text);
    let total = counts.values().sum::<usize>().max(1) as f32;
    let mut terms = counts
        .into_iter()
        .map(|(term, count)| TermFrequency {
            term,
            count,
            frequency: count as f32 / total,
        })
        .collect::<Vec<_>>();
    terms.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.term.cmp(&right.term))
    });
    terms
}

/// Returns top terms.
pub fn top_terms(text: &str, limit: usize, stop_words: &BTreeSet<String>) -> Vec<TermFrequency> {
    term_frequencies(text)
        .into_iter()
        .filter(|term| !stop_words.contains(&term.term))
        .take(limit)
        .collect()
}

/// Returns english stop words.
pub fn english_stop_words() -> StopWords {
    let terms = [
        "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be",
        "because", "been", "but", "by", "can", "do", "for", "from", "had", "has", "have", "he",
        "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "more",
        "my", "no", "not", "of", "on", "or", "our", "out", "she", "so", "some", "than", "that",
        "the", "their", "them", "then", "there", "these", "they", "this", "to", "up", "us", "was",
        "we", "were", "what", "when", "which", "who", "will", "with", "would", "you", "your",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    StopWords {
        language: Some("en".to_string()),
        terms,
    }
}

/// Returns stop words for a supported language tag.
pub fn stop_words_for_language(language: &str) -> StopWords {
    match language
        .split(['-', '_'])
        .next()
        .unwrap_or(language)
        .to_ascii_lowercase()
        .as_str()
    {
        "de" => stop_words_from_terms(
            "de",
            [
                "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "das", "dem", "den", "der",
                "des", "die", "ein", "eine", "einem", "einen", "einer", "es", "für", "hat", "ich",
                "im", "in", "ist", "mit", "nicht", "oder", "sich", "sie", "und", "von", "war",
                "wie", "wir", "zu",
            ],
        ),
        "fr" => stop_words_from_terms(
            "fr",
            [
                "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "est",
                "et", "eux", "il", "ils", "je", "la", "le", "les", "leur", "mais", "nous", "ou",
                "par", "pas", "pour", "que", "qui", "sur", "un", "une", "vous",
            ],
        ),
        "es" => stop_words_from_terms(
            "es",
            [
                "a", "al", "como", "con", "de", "del", "el", "ella", "en", "es", "esa", "ese",
                "esta", "este", "la", "las", "lo", "los", "mas", "no", "nos", "o", "para", "pero",
                "por", "que", "se", "sin", "su", "sus", "un", "una", "y", "yo",
            ],
        ),
        _ => english_stop_words(),
    }
}

fn stop_words_from_terms<const N: usize>(language: &str, terms: [&str; N]) -> StopWords {
    StopWords {
        language: Some(language.to_string()),
        terms: terms.into_iter().map(String::from).collect(),
    }
}

/// Returns keywords.
pub fn keywords(text: &str, options: &KeywordOptions) -> Vec<Keyword> {
    let processing = TextProcessingOptions::default();
    let mut counts = BTreeMap::<String, usize>::new();
    for token in tokenize(text, &processing) {
        if !matches!(
            token.kind,
            TokenKind::Word | TokenKind::Number | TokenKind::Email | TokenKind::Url
        ) {
            continue;
        }
        if token.normalized.chars().count() < options.min_term_len {
            continue;
        }
        if options.stop_words.terms.contains(&token.normalized) {
            continue;
        }
        *counts.entry(token.normalized).or_insert(0) += 1;
    }

    let total = counts.values().sum::<usize>().max(1) as f32;
    let mut terms = counts
        .into_iter()
        .map(|(text, count)| Keyword {
            text,
            score: count as f32 / total,
            count,
        })
        .collect::<Vec<_>>();
    terms.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| right.score.total_cmp(&left.score))
            .then_with(|| left.text.cmp(&right.text))
    });
    terms.truncate(options.max_terms);
    terms
}

/// Extracts RAKE-like phrase keywords from contiguous non-stop-word terms.
pub fn phrase_keywords(text: &str, options: &PhraseKeywordOptions) -> Vec<Keyword> {
    let mut phrases = Vec::<Vec<String>>::new();
    let mut current = Vec::<String>::new();
    for token in tokenize(text, &TextProcessingOptions::default()) {
        if !matches!(token.kind, TokenKind::Word | TokenKind::Number) {
            if !current.is_empty() {
                phrases.push(std::mem::take(&mut current));
            }
            continue;
        }
        if token.normalized.chars().count() < options.min_term_len
            || options.stop_words.terms.contains(&token.normalized)
        {
            if !current.is_empty() {
                phrases.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(token.normalized);
    }
    if !current.is_empty() {
        phrases.push(current);
    }

    let mut term_frequency = BTreeMap::<String, usize>::new();
    let mut term_degree = BTreeMap::<String, usize>::new();
    for phrase in &phrases {
        let degree = phrase.len().saturating_sub(1);
        for term in phrase {
            *term_frequency.entry(term.clone()).or_insert(0) += 1;
            *term_degree.entry(term.clone()).or_insert(0) += degree;
        }
    }
    let term_scores = term_frequency
        .iter()
        .map(|(term, frequency)| {
            let degree = term_degree.get(term).copied().unwrap_or(0) + frequency;
            (term.clone(), degree as f32 / (*frequency).max(1) as f32)
        })
        .collect::<BTreeMap<_, _>>();

    let mut scored = phrases
        .into_iter()
        .filter(|phrase| phrase.len() >= options.min_phrase_terms)
        .map(|phrase| {
            let score = phrase
                .iter()
                .map(|term| term_scores.get(term).copied().unwrap_or(0.0))
                .sum::<f32>();
            Keyword {
                text: phrase.join(" "),
                count: phrase.len(),
                score,
            }
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.text.cmp(&right.text))
    });
    scored.truncate(options.max_phrases);
    scored
}

/// Returns character ngrams.
pub fn character_ngrams(text: &str, n: usize) -> Result<Vec<String>> {
    if n == 0 {
        return Err(DetectError::InvalidArgument(
            "ngram size must be greater than zero".to_string(),
        ));
    }
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() < n {
        return Ok(Vec::new());
    }
    Ok(chars
        .windows(n)
        .map(|window| window.iter().collect::<String>())
        .collect())
}

/// Returns character ngram frequencies.
pub fn character_ngram_frequencies(text: &str, n: usize) -> Result<Vec<NgramFrequency>> {
    let ngrams = character_ngrams(text, n)?;
    Ok(ngram_frequencies(
        ngrams
            .into_iter()
            .map(|ngram| vec![ngram])
            .collect::<Vec<_>>(),
    ))
}

/// Returns token ngrams.
pub fn token_ngrams(text: &str, n: usize) -> Result<Vec<Vec<String>>> {
    if n == 0 {
        return Err(DetectError::InvalidArgument(
            "ngram size must be greater than zero".to_string(),
        ));
    }
    let tokens = tokenize_words(text);
    if tokens.len() < n {
        return Ok(Vec::new());
    }
    Ok(tokens.windows(n).map(|window| window.to_vec()).collect())
}

/// Returns token ngram frequencies.
pub fn token_ngram_frequencies(
    text: &str,
    n: usize,
    options: &TextProcessingOptions,
) -> Result<Vec<NgramFrequency>> {
    if n == 0 {
        return Err(DetectError::InvalidArgument(
            "ngram size must be greater than zero".to_string(),
        ));
    }
    let tokens = tokenize(text, options)
        .into_iter()
        .filter(|token| token.kind != TokenKind::Punctuation)
        .map(|token| token.normalized)
        .collect::<Vec<_>>();
    if tokens.len() < n {
        return Ok(Vec::new());
    }
    let ngrams = tokens
        .windows(n)
        .map(|window| window.to_vec())
        .collect::<Vec<_>>();
    Ok(ngram_frequencies(ngrams))
}

/// Returns character shingles.
pub fn character_shingles(text: &str, n: usize) -> Result<BTreeSet<String>> {
    Ok(character_ngrams(text, n)?.into_iter().collect())
}

/// Returns token shingles.
pub fn token_shingles(
    text: &str,
    n: usize,
    options: &TextProcessingOptions,
) -> Result<BTreeSet<Vec<String>>> {
    if n == 0 {
        return Err(DetectError::InvalidArgument(
            "shingle size must be greater than zero".to_string(),
        ));
    }
    let tokens = tokenize(text, options)
        .into_iter()
        .filter(|token| token.kind != TokenKind::Punctuation)
        .map(|token| token.normalized)
        .collect::<Vec<_>>();
    if tokens.len() < n {
        return Ok(BTreeSet::new());
    }
    Ok(tokens.windows(n).map(|window| window.to_vec()).collect())
}

/// Returns shingle jaccard similarity.
pub fn shingle_jaccard_similarity<T>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> ShingleSimilarity
where
    T: Ord,
{
    let intersection_count = left.intersection(right).count();
    let union_count = left.union(right).count();
    let jaccard = if union_count == 0 {
        1.0
    } else {
        intersection_count as f32 / union_count as f32
    };
    ShingleSimilarity {
        left_count: left.len(),
        right_count: right.len(),
        intersection_count,
        union_count,
        jaccard,
    }
}

/// Returns character shingle similarity.
pub fn character_shingle_similarity(
    text: &str,
    other: &str,
    n: usize,
) -> Result<ShingleSimilarity> {
    let left = character_shingles(text, n)?;
    let right = character_shingles(other, n)?;
    Ok(shingle_jaccard_similarity(&left, &right))
}

/// Returns token shingle similarity.
pub fn token_shingle_similarity(
    text: &str,
    other: &str,
    n: usize,
    options: &TextProcessingOptions,
) -> Result<ShingleSimilarity> {
    let left = token_shingles(text, n, options)?;
    let right = token_shingles(other, n, options)?;
    Ok(shingle_jaccard_similarity(&left, &right))
}

/// Returns readability summary.
pub fn readability_summary(text: &str, options: &TextProcessingOptions) -> ReadabilitySummary {
    let stats = detailed_text_stats(text, options);
    ReadabilitySummary {
        sentence_count: stats.basic.sentences,
        word_count: stats.basic.words,
        average_sentence_words: stats.average_words_per_sentence,
        average_word_chars: stats.average_chars_per_word,
    }
}

/// Returns stem terms.
pub fn stem_terms(text: &str, options: &StemOptions) -> Vec<String> {
    tokenize_words(text)
        .into_iter()
        .filter(|term| term.chars().count() >= options.min_term_len)
        .filter(|term| !options.stop_words.terms.contains(term))
        .map(|term| stem_english(&term))
        .filter(|term| !term.is_empty())
        .collect()
}

/// Returns extractive summary.
pub fn extractive_summary(
    text: &str,
    options: &ExtractiveSummaryOptions,
) -> Result<Vec<SummarySentence>> {
    if options.max_sentences == 0 {
        return Err(invalid_argument(
            "summary max_sentences must be greater than zero",
        ));
    }
    let sentences = split_sentence_spans(text, &TextProcessingOptions::default());
    if sentences.is_empty() {
        return Ok(Vec::new());
    }

    let mut document_counts = BTreeMap::<String, usize>::new();
    let mut sentence_terms = Vec::with_capacity(sentences.len());
    for sentence in &sentences {
        let terms = tokenize_words(&sentence.text)
            .into_iter()
            .filter(|term| term.chars().count() >= 3)
            .filter(|term| !options.stop_words.terms.contains(term))
            .collect::<Vec<_>>();
        for term in &terms {
            *document_counts.entry(term.clone()).or_insert(0) += 1;
        }
        sentence_terms.push(terms);
    }

    let mut ranked = sentences
        .into_iter()
        .enumerate()
        .filter_map(|(index, sentence)| {
            let terms = &sentence_terms[index];
            if sentence.token_count < options.min_sentence_words || terms.is_empty() {
                return None;
            }
            let raw_score = terms
                .iter()
                .map(|term| document_counts.get(term).copied().unwrap_or(0) as f32)
                .sum::<f32>();
            Some(SummarySentence {
                index,
                text: sentence.text,
                span: sentence.span,
                score: raw_score / terms.len() as f32,
            })
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.index.cmp(&right.index))
    });
    ranked.truncate(options.max_sentences);
    ranked.sort_by_key(|sentence| sentence.index);
    Ok(ranked)
}

/// Returns an extractive summary with a simple MMR diversity penalty.
pub fn diverse_extractive_summary(
    text: &str,
    options: &ExtractiveSummaryOptions,
    diversity_weight: f32,
) -> Result<Vec<SummarySentence>> {
    let mut candidates = extractive_summary(
        text,
        &ExtractiveSummaryOptions {
            max_sentences: options
                .max_sentences
                .saturating_mul(4)
                .max(options.max_sentences),
            ..options.clone()
        },
    )?;
    if candidates.len() <= options.max_sentences {
        return Ok(candidates);
    }
    let diversity_weight = diversity_weight.clamp(0.0, 1.0);
    let mut selected = Vec::<SummarySentence>::new();
    while selected.len() < options.max_sentences && !candidates.is_empty() {
        let best_index = candidates
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                let left_score = mmr_score(
                    left,
                    &selected,
                    diversity_weight,
                    &TextProcessingOptions::default(),
                );
                let right_score = mmr_score(
                    right,
                    &selected,
                    diversity_weight,
                    &TextProcessingOptions::default(),
                );
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right.index.cmp(&left.index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        selected.push(candidates.remove(best_index));
    }
    selected.sort_by_key(|sentence| sentence.index);
    Ok(selected)
}

/// Returns sentiment.
pub fn sentiment(text: &str, lexicon: &SentimentLexicon) -> SentimentSummary {
    let mut positive_score = 0.0_f32;
    let mut negative_score = 0.0_f32;
    let mut matched_terms = 0;
    let tokens = tokenize_words(text);
    for (index, token) in tokens.iter().enumerate() {
        if let Some(score) = lexicon.terms.get(token) {
            matched_terms += 1;
            let previous = index
                .checked_sub(1)
                .and_then(|index| tokens.get(index))
                .map(String::as_str);
            let before_previous = index
                .checked_sub(2)
                .and_then(|index| tokens.get(index))
                .map(String::as_str);
            let mut adjusted = *score;
            if previous.is_some_and(is_intensifier) || before_previous.is_some_and(is_intensifier) {
                adjusted *= 1.5;
            }
            if previous.is_some_and(is_negation) || before_previous.is_some_and(is_negation) {
                adjusted *= -0.75;
            }
            if adjusted >= 0.0 {
                positive_score += adjusted;
            } else {
                negative_score += adjusted.abs();
            }
        }
    }
    let total = positive_score + negative_score;
    let compound = if total <= f32::EPSILON {
        0.0
    } else {
        (positive_score - negative_score) / total
    };
    let label = if compound > lexicon.neutral_threshold {
        "positive"
    } else if compound < -lexicon.neutral_threshold {
        "negative"
    } else {
        "neutral"
    }
    .to_string();

    SentimentSummary {
        positive_score,
        negative_score,
        compound,
        token_count: tokens.len(),
        matched_terms,
        label,
    }
}

/// Returns rule entities.
pub fn rule_entities(text: &str, rules: &EntityRuleSet) -> Vec<EntityMention> {
    let tokens = tokenize(text, &TextProcessingOptions::default());
    let mut mentions = Vec::new();
    for token in &tokens {
        let kind = match token.kind {
            TokenKind::Email if rules.emails => Some("email"),
            TokenKind::Url if rules.urls => Some("url"),
            TokenKind::Mention if rules.mentions => Some("mention"),
            TokenKind::Hashtag if rules.hashtags => Some("hashtag"),
            TokenKind::Number if rules.numbers => Some("number"),
            _ => None,
        };
        if let Some(kind) = kind {
            mentions.push(EntityMention {
                kind: kind.to_string(),
                text: token.text.clone(),
                normalized: token.normalized.clone(),
                span: token.span,
            });
        }
    }

    if rules.capitalized_phrases {
        mentions.extend(capitalized_phrase_mentions(text, &tokens));
    }
    mentions.sort_by(|left, right| {
        left.span
            .byte_start
            .cmp(&right.span.byte_start)
            .then_with(|| left.span.byte_end.cmp(&right.span.byte_end))
            .then_with(|| left.kind.cmp(&right.kind))
    });
    mentions
}

#[derive(Debug, Default, Clone)]
/// Data type for text stats analyzer.
pub struct TextStatsAnalyzer;

impl TextAnalyzer for TextStatsAnalyzer {
    fn name(&self) -> &str {
        "text_stats"
    }

    fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>> {
        let stats = text_stats(segment.text);
        Ok(vec![
            event_at(self.name(), "text:stats", segment.timestamp).score(stats.words as f32)
        ])
    }
}

#[derive(Debug, Clone, Default)]
/// Data type for keyword analyzer.
pub struct KeywordAnalyzer {
    /// The options value.
    pub options: KeywordOptions,
}

impl KeywordAnalyzer {
    /// Creates a new value.
    pub fn new(options: KeywordOptions) -> Self {
        Self { options }
    }
}

impl TextAnalyzer for KeywordAnalyzer {
    fn name(&self) -> &str {
        "keywords"
    }

    fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>> {
        Ok(keywords(segment.text, &self.options)
            .into_iter()
            .map(|keyword| {
                event_at(
                    self.name(),
                    &format!("text:keyword:{}", keyword.text),
                    segment.timestamp,
                )
                .score(keyword.score)
            })
            .collect())
    }
}

#[derive(Debug, Default, Clone)]
/// Data type for pattern analyzer.
pub struct PatternAnalyzer;

impl TextAnalyzer for PatternAnalyzer {
    fn name(&self) -> &str {
        "text_patterns"
    }

    fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>> {
        let mut events = pattern_events(self.name(), segment.text, segment.timestamp);
        if segment.text.trim_end().ends_with(['?', '؟', '？']) {
            events.push(event_at(
                self.name(),
                "text:pattern:question",
                segment.timestamp,
            ));
        }
        Ok(events)
    }
}

#[derive(Debug, Clone, Default)]
/// Data type for extractive summary analyzer.
pub struct ExtractiveSummaryAnalyzer {
    /// The options value.
    pub options: ExtractiveSummaryOptions,
}

impl ExtractiveSummaryAnalyzer {
    /// Creates a new value.
    pub fn new(options: ExtractiveSummaryOptions) -> Self {
        Self { options }
    }
}

impl TextAnalyzer for ExtractiveSummaryAnalyzer {
    fn name(&self) -> &str {
        "extractive_summary"
    }

    fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>> {
        Ok(extractive_summary(segment.text, &self.options)?
            .into_iter()
            .map(|sentence| {
                event_at(
                    self.name(),
                    &format!("text:summary:{}", sentence.index),
                    segment.timestamp,
                )
                .score(sentence.score)
            })
            .collect())
    }
}

#[derive(Debug, Clone, Default)]
/// Data type for sentiment analyzer.
pub struct SentimentAnalyzer {
    /// The lexicon value.
    pub lexicon: SentimentLexicon,
}

impl SentimentAnalyzer {
    /// Creates a new value.
    pub fn new(lexicon: SentimentLexicon) -> Self {
        Self { lexicon }
    }
}

impl TextAnalyzer for SentimentAnalyzer {
    fn name(&self) -> &str {
        "sentiment"
    }

    fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>> {
        let summary = sentiment(segment.text, &self.lexicon);
        Ok(vec![event_at(
            self.name(),
            &format!("text:sentiment:{}", summary.label),
            segment.timestamp,
        )
        .score(summary.compound)])
    }
}

#[derive(Debug, Clone, Default)]
/// Data type for entity rule analyzer.
pub struct EntityRuleAnalyzer {
    /// The rules value.
    pub rules: EntityRuleSet,
}

impl EntityRuleAnalyzer {
    /// Creates a new value.
    pub fn new(rules: EntityRuleSet) -> Self {
        Self { rules }
    }
}

impl TextAnalyzer for EntityRuleAnalyzer {
    fn name(&self) -> &str {
        "rule_entities"
    }

    fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>> {
        Ok(rule_entities(segment.text, &self.rules)
            .into_iter()
            .map(|mention| {
                event_at(
                    self.name(),
                    &format!("text:entity:{}:{}", mention.kind, mention.normalized),
                    segment.timestamp,
                )
            })
            .collect())
    }
}

fn ngram_frequencies(ngrams: Vec<Vec<String>>) -> Vec<NgramFrequency> {
    let total = ngrams.len().max(1) as f32;
    let mut counts = BTreeMap::<Vec<String>, usize>::new();
    for ngram in ngrams {
        *counts.entry(ngram).or_insert(0) += 1;
    }
    let mut frequencies = counts
        .into_iter()
        .map(|(terms, count)| NgramFrequency {
            terms,
            count,
            frequency: count as f32 / total,
        })
        .collect::<Vec<_>>();
    frequencies.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.terms.cmp(&right.terms))
    });
    frequencies
}

fn mmr_score(
    candidate: &SummarySentence,
    selected: &[SummarySentence],
    diversity_weight: f32,
    options: &TextProcessingOptions,
) -> f32 {
    let redundancy = selected
        .iter()
        .map(|sentence| {
            token_shingle_similarity(&candidate.text, &sentence.text, 2, options)
                .map(|similarity| similarity.jaccard)
                .unwrap_or(0.0)
        })
        .fold(0.0_f32, f32::max);
    (1.0 - diversity_weight) * candidate.score - diversity_weight * redundancy
}

fn is_negation(term: &str) -> bool {
    matches!(
        term,
        "not" | "no" | "never" | "none" | "nicht" | "kein" | "pas"
    )
}

fn is_intensifier(term: &str) -> bool {
    matches!(
        term,
        "very" | "really" | "extremely" | "so" | "too" | "sehr" | "tres" | "très" | "muy"
    )
}

fn pattern_events(analyzer: &str, text: &str, timestamp: Option<Timestamp>) -> Vec<AnalysisEvent> {
    let mut seen = BTreeSet::new();
    let mut events = Vec::new();
    for token in tokenize(text, &TextProcessingOptions::default()) {
        let label = match token.kind {
            TokenKind::Url => Some("text:pattern:url"),
            TokenKind::Email => Some("text:pattern:email"),
            TokenKind::Mention => Some("text:pattern:mention"),
            TokenKind::Hashtag => Some("text:pattern:hashtag"),
            TokenKind::Number => Some("text:pattern:number"),
            _ => None,
        };
        if let Some(label) = label {
            if seen.insert(label) {
                events.push(event_at(analyzer, label, timestamp));
            }
        }
    }
    events
}

fn stem_english(term: &str) -> String {
    let mut stem = term.to_lowercase();
    if stem.len() <= 3 {
        return stem;
    }
    for (suffix, replacement) in [
        ("ization", "ize"),
        ("ational", "ate"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("tional", "tion"),
        ("biliti", "ble"),
        ("ing", ""),
        ("edly", ""),
        ("edly", ""),
        ("ed", ""),
        ("ies", "y"),
        ("sses", "ss"),
        ("s", ""),
    ] {
        if stem.ends_with(suffix) && stem.len() > suffix.len() + 2 {
            stem.truncate(stem.len() - suffix.len());
            stem.push_str(replacement);
            break;
        }
    }
    if stem.ends_with("nn") || stem.ends_with("tt") || stem.ends_with("pp") {
        stem.pop();
    }
    stem
}

fn capitalized_phrase_mentions(text: &str, tokens: &[text_core::Token]) -> Vec<EntityMention> {
    let mut mentions = Vec::new();
    let mut start = None::<usize>;
    let mut end = None::<usize>;
    for (index, token) in tokens.iter().enumerate() {
        let is_capitalized = token.kind == TokenKind::Word
            && token
                .text
                .chars()
                .next()
                .is_some_and(|ch| ch.is_uppercase())
            && token.text.chars().any(|ch| ch.is_lowercase());
        if is_capitalized {
            start.get_or_insert(index);
            end = Some(index);
        } else if let (Some(start_index), Some(end_index)) = (start.take(), end.take()) {
            push_capitalized_phrase(text, tokens, start_index, end_index, &mut mentions);
        }
    }
    if let (Some(start_index), Some(end_index)) = (start, end) {
        push_capitalized_phrase(text, tokens, start_index, end_index, &mut mentions);
    }
    mentions
}

fn push_capitalized_phrase(
    text: &str,
    tokens: &[text_core::Token],
    start_index: usize,
    end_index: usize,
    mentions: &mut Vec<EntityMention>,
) {
    if end_index < start_index {
        return;
    }
    let span = TextSpan {
        byte_start: tokens[start_index].span.byte_start,
        byte_end: tokens[end_index].span.byte_end,
        char_start: tokens[start_index].span.char_start,
        char_end: tokens[end_index].span.char_end,
    };
    let raw = text[span.byte_start..span.byte_end].to_string();
    mentions.push(EntityMention {
        kind: "capitalized_phrase".to_string(),
        normalized: raw.to_lowercase(),
        text: raw,
        span,
    });
}

fn event_at(analyzer: &str, label: &str, timestamp: Option<Timestamp>) -> AnalysisEvent {
    let event = AnalysisEvent::new(analyzer, label);
    if let Some(timestamp) = timestamp {
        event.at_timestamp(timestamp)
    } else {
        event
    }
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use text_core::{OwnedTextSegment, TextPipeline};

    #[test]
    fn ranks_terms_by_count() {
        let terms = term_frequencies("red blue red green blue red");
        assert_eq!(terms[0].term, "red");
        assert_eq!(terms[0].count, 3);
    }

    #[test]
    fn builds_token_ngrams() {
        let ngrams = token_ngrams("a b c", 2).unwrap();
        assert_eq!(
            ngrams,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["b".to_string(), "c".to_string()]
            ]
        );
    }

    #[test]
    fn filters_english_stop_words_for_keywords() {
        let terms = keywords(
            "the rust code and the rust tests",
            &KeywordOptions::default(),
        );
        assert_eq!(terms[0].text, "rust");
        assert!(!terms.iter().any(|term| term.text == "the"));
    }

    #[test]
    fn selects_language_stop_words() {
        let german = stop_words_for_language("de-DE");
        assert_eq!(german.language.as_deref(), Some("de"));
        assert!(german.terms.contains("und"));
    }

    #[test]
    fn extracts_phrase_keywords() {
        let phrases = phrase_keywords(
            "Fast video analysis, reliable transcript analysis, and semantic search.",
            &PhraseKeywordOptions::default(),
        );
        assert!(phrases
            .iter()
            .any(|phrase| phrase.text.contains("video analysis")));
    }

    #[test]
    fn counts_token_ngram_frequencies() {
        let ngrams =
            token_ngram_frequencies("red blue red blue green", 2, &Default::default()).unwrap();
        assert_eq!(ngrams[0].terms, vec!["red", "blue"]);
        assert_eq!(ngrams[0].count, 2);
    }

    #[test]
    fn counts_character_ngram_frequencies() {
        let ngrams = character_ngram_frequencies("ababa", 2).unwrap();
        assert_eq!(ngrams[0].terms, vec!["ab"]);
        assert_eq!(ngrams[0].count, 2);
    }

    #[test]
    fn builds_unique_token_shingles_from_normalized_tokens() {
        let shingles = token_shingles(
            "Rust, cargo, rust tests",
            2,
            &TextProcessingOptions::default(),
        )
        .unwrap();
        assert_eq!(shingles.len(), 3);
        assert!(shingles.contains(&vec!["rust".to_string(), "cargo".to_string()]));
        assert!(shingles.contains(&vec!["cargo".to_string(), "rust".to_string()]));
        assert!(shingles.contains(&vec!["rust".to_string(), "tests".to_string()]));
    }

    #[test]
    fn computes_token_shingle_jaccard_similarity() {
        let similarity = token_shingle_similarity(
            "rust cargo builds crates",
            "rust cargo runs tests",
            2,
            &TextProcessingOptions::default(),
        )
        .unwrap();
        assert_eq!(similarity.left_count, 3);
        assert_eq!(similarity.right_count, 3);
        assert_eq!(similarity.intersection_count, 1);
        assert_eq!(similarity.union_count, 5);
        assert!((similarity.jaccard - 0.2).abs() < 0.001);
    }

    #[test]
    fn computes_character_shingle_jaccard_similarity() {
        let similarity = character_shingle_similarity("banana", "bandana", 2).unwrap();
        assert_eq!(similarity.intersection_count, 3);
        assert_eq!(similarity.union_count, 5);
        assert!((similarity.jaccard - 0.6).abs() < 0.001);
    }

    #[test]
    fn computes_readability_summary() {
        let summary = readability_summary("One sentence. Two words here.", &Default::default());
        assert_eq!(summary.sentence_count, 2);
        assert_eq!(summary.word_count, 5);
    }

    #[test]
    fn stems_terms_after_stop_word_filtering() {
        let mut options = StemOptions {
            min_term_len: 3,
            ..StemOptions::default()
        };
        options.stop_words.terms.insert("running".to_string());
        let stems = stem_terms("running tested tests cities", &options);
        assert_eq!(stems, vec!["test", "test", "city"]);
    }

    #[test]
    fn ranks_extractive_summary_sentences() {
        let summary = extractive_summary(
            "Rust builds reliable tools. Bananas are yellow. Rust tools ship reliable crates.",
            &ExtractiveSummaryOptions {
                max_sentences: 1,
                min_sentence_words: 3,
                ..ExtractiveSummaryOptions::default()
            },
        )
        .unwrap();
        assert_eq!(summary.len(), 1);
        assert!(summary[0].text.contains("Rust"));
        assert!(summary[0].score > 1.0);
    }

    #[test]
    fn diversity_summary_limits_redundant_sentences() {
        let summary = diverse_extractive_summary(
            "Rust video analysis finds scenes. Rust video analysis finds motion. Audio cues help reports.",
            &ExtractiveSummaryOptions {
                max_sentences: 2,
                min_sentence_words: 3,
                ..ExtractiveSummaryOptions::default()
            },
            0.7,
        )
        .unwrap();
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn scores_sentiment_polarity_and_neutral_text() {
        let lexicon = SentimentLexicon::default();
        assert_eq!(
            sentiment("excellent reliable work", &lexicon).label,
            "positive"
        );
        assert_eq!(
            sentiment("terrible broken failure", &lexicon).label,
            "negative"
        );
        assert_eq!(sentiment("table chair window", &lexicon).label, "neutral");
        assert_eq!(sentiment("not good", &lexicon).label, "negative");
    }

    #[test]
    fn extracts_rule_entity_spans() {
        let text =
            "Contact Jane Doe at jane@example.com, visit https://example.com @team #Rust 42.";
        let entities = rule_entities(text, &EntityRuleSet::default());
        let kinds = entities
            .iter()
            .map(|entity| entity.kind.as_str())
            .collect::<Vec<_>>();
        assert!(kinds.contains(&"capitalized_phrase"));
        assert!(kinds.contains(&"email"));
        assert!(kinds.contains(&"url"));
        assert!(kinds.contains(&"mention"));
        assert!(kinds.contains(&"hashtag"));
        assert!(kinds.contains(&"number"));
        let email = entities
            .iter()
            .find(|entity| entity.kind == "email")
            .unwrap();
        assert_eq!(
            &text[email.span.byte_start..email.span.byte_end],
            "jane@example.com"
        );
    }

    #[test]
    fn pattern_analyzer_emits_labels() {
        let mut analyzer = PatternAnalyzer;
        let segment =
            OwnedTextSegment::new(0, "Mail hi@example.com @team #rust https://example.com 42?");
        let labels = analyzer
            .process_segment(&segment.as_segment())
            .unwrap()
            .into_iter()
            .map(|event| event.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"text:pattern:url".to_string()));
        assert!(labels.contains(&"text:pattern:number".to_string()));
        assert!(labels.contains(&"text:pattern:question".to_string()));
        assert!(labels.contains(&"text:pattern:email".to_string()));
        assert!(labels.contains(&"text:pattern:mention".to_string()));
        assert!(labels.contains(&"text:pattern:hashtag".to_string()));
    }

    #[test]
    fn analyzers_run_inside_text_pipeline() {
        let mut pipeline = TextPipeline::builder()
            .analyzer(TextStatsAnalyzer)
            .analyzer(KeywordAnalyzer::default())
            .analyzer(PatternAnalyzer)
            .build()
            .unwrap();

        pipeline
            .process_segment(OwnedTextSegment::new(
                0,
                "Visit https://example.com with rust rust?",
            ))
            .unwrap();
        let result = pipeline.finish_analysis().unwrap();
        let labels = result
            .events
            .into_iter()
            .map(|event| event.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"text:stats".to_string()));
        assert!(labels.contains(&"text:keyword:rust".to_string()));
        assert!(labels.contains(&"text:pattern:question".to_string()));
        assert!(labels.contains(&"text:pattern:url".to_string()));
    }

    #[test]
    fn tfidf_and_bm25_share_token_processing() {
        let options = CorpusOptions::default();
        let mut tfidf = TfIdfCorpus::new(options.clone());
        let mut bm25 = Bm25Corpus::new(Bm25Options {
            min_term_len: options.min_term_len,
            stop_words: options.stop_words.clone(),
            ..Bm25Options::default()
        });

        tfidf
            .add_document("doc-1", "Rust, rust! Cargo builds.")
            .unwrap();
        bm25.add_document("doc-1", "Rust, rust! Cargo builds.")
            .unwrap();

        assert_eq!(tfidf.documents()[0].term_counts["rust"], 2);
        assert_eq!(bm25.documents()[0].term_counts["rust"], 2);
    }
}
