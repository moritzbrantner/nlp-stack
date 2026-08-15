use std::collections::BTreeMap;

use text_core::{split_sentence_spans, tokenize, TextProcessingOptions, Token, TokenKind};
use text_lexical::english_stop_words;

use crate::EnrichedTextStats;

pub fn enriched_text_stats(text: &str, options: &TextProcessingOptions) -> EnrichedTextStats {
    let tokens = tokenize(text, options);
    enriched_text_stats_from_tokens(text, options, &tokens)
}

pub fn enriched_text_stats_from_tokens(
    text: &str,
    options: &TextProcessingOptions,
    tokens: &[Token],
) -> EnrichedTextStats {
    let lexical_tokens = tokens
        .iter()
        .filter(|token| {
            matches!(
                token.kind,
                TokenKind::Word
                    | TokenKind::Number
                    | TokenKind::Url
                    | TokenKind::Email
                    | TokenKind::Mention
                    | TokenKind::Hashtag
            )
        })
        .collect::<Vec<_>>();
    let word_tokens = tokens
        .iter()
        .filter(|token| token.kind == TokenKind::Word)
        .collect::<Vec<_>>();
    let total_tokens = tokens.len().max(1) as f32;
    let lexical_total = lexical_tokens.len().max(1) as f32;
    let stop_words = english_stop_words();
    let stopword_count = lexical_tokens
        .iter()
        .filter(|token| stop_words.terms.contains(&token.normalized))
        .count();
    let mut counts = BTreeMap::<String, usize>::new();
    for token in &lexical_tokens {
        *counts.entry(token.normalized.clone()).or_insert(0) += 1;
    }
    let hapax_count = counts.values().filter(|count| **count == 1).count();
    let entropy = counts
        .values()
        .map(|count| *count as f32 / lexical_total)
        .filter(|p| *p > 0.0)
        .map(|p| -p * p.log2())
        .sum::<f32>();
    let uppercase_tokens = word_tokens
        .iter()
        .filter(|token| {
            let has_alpha = token.text.chars().any(char::is_alphabetic);
            has_alpha
                && token
                    .text
                    .chars()
                    .filter(|ch| ch.is_alphabetic())
                    .all(|ch| ch.is_uppercase())
        })
        .count();
    let sentence_lengths = split_sentence_spans(text, options)
        .into_iter()
        .map(|sentence| sentence.token_count)
        .collect::<Vec<_>>();
    let (
        sentence_token_min,
        sentence_token_max,
        sentence_token_mean,
        sentence_token_p50,
        sentence_token_p90,
    ) = summarize_lengths(sentence_lengths);

    EnrichedTextStats {
        lexical_density: word_tokens.len() as f32 / total_tokens,
        stopword_ratio: stopword_count as f32 / lexical_total,
        hapax_ratio: hapax_count as f32 / counts.len().max(1) as f32,
        shannon_entropy: entropy,
        punctuation_token_ratio: tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Punctuation)
            .count() as f32
            / total_tokens,
        uppercase_token_ratio: uppercase_tokens as f32 / word_tokens.len().max(1) as f32,
        numeric_token_ratio: tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Number)
            .count() as f32
            / total_tokens,
        url_count: tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Url)
            .count(),
        email_count: tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Email)
            .count(),
        mention_count: tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Mention)
            .count(),
        hashtag_count: tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Hashtag)
            .count(),
        sentence_token_min,
        sentence_token_max,
        sentence_token_mean,
        sentence_token_p50,
        sentence_token_p90,
    }
}

fn summarize_lengths(mut lengths: Vec<usize>) -> (usize, usize, f32, usize, usize) {
    if lengths.is_empty() {
        return (0, 0, 0.0, 0, 0);
    }
    lengths.sort_unstable();
    let min = lengths[0];
    let max = *lengths.last().unwrap_or(&0);
    let mean = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
    let p50 = percentile(&lengths, 0.5);
    let p90 = percentile(&lengths, 0.9);
    (min, max, mean, p50, p90)
}

fn percentile(sorted: &[usize], percentile: f32) -> usize {
    let index = ((sorted.len() - 1) as f32 * percentile.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}
