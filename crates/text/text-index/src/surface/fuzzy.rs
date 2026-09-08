use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use text_core::{tokenize, TextProcessingOptions, TokenKind};

use super::SurfaceIndex;
use crate::{IndexChunk, IndexQuery, IndexSearchMode, IndexSearchResult, TextIndexStore};

const MAX_FUZZY_QUERY_TERMS: usize = 16;
const MAX_FUZZY_TERM_CHARS: usize = 64;
const HARD_MAX_EXPANSIONS_PER_TERM: usize = 8;
const HARD_MAX_QUERY_VARIANTS: usize = 64;
const HARD_MAX_VOCABULARY_TERMS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct FuzzySearchOptions {
    #[serde(default = "default_max_edit_distance")]
    pub max_edit_distance: usize,
    #[serde(default = "default_min_term_length")]
    pub min_term_length: usize,
    #[serde(default = "default_max_expansions_per_term")]
    pub max_expansions_per_term: usize,
    #[serde(default = "default_max_query_variants")]
    pub max_query_variants: usize,
    #[serde(default = "default_max_vocabulary_terms")]
    pub max_vocabulary_terms: usize,
    #[serde(default = "default_fuzzy_weight")]
    pub fuzzy_weight: f32,
}

impl Default for FuzzySearchOptions {
    fn default() -> Self {
        Self {
            max_edit_distance: default_max_edit_distance(),
            min_term_length: default_min_term_length(),
            max_expansions_per_term: default_max_expansions_per_term(),
            max_query_variants: default_max_query_variants(),
            max_vocabulary_terms: default_max_vocabulary_terms(),
            fuzzy_weight: default_fuzzy_weight(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct FuzzyTermMatch {
    query_term: String,
    matched_term: String,
    edit_distance: usize,
    similarity: f32,
}

#[derive(Debug, Clone)]
struct QueryToken {
    normalized: String,
    fuzzy_eligible: bool,
}

#[derive(Debug, Clone)]
struct VocabularyTerm {
    term: String,
    document_frequency: usize,
}

#[derive(Debug, Clone)]
struct QueryVariant {
    terms: Vec<String>,
    fuzzy_matches: Vec<FuzzyTermMatch>,
}

impl QueryVariant {
    fn weight(&self, options: &FuzzySearchOptions) -> f32 {
        if self.fuzzy_matches.is_empty() {
            return 1.0;
        }
        let mean_similarity = self
            .fuzzy_matches
            .iter()
            .map(|matched| matched.similarity)
            .sum::<f32>()
            / self.fuzzy_matches.len() as f32;
        options.fuzzy_weight * mean_similarity
    }

    fn text(&self) -> String {
        self.terms.join(" ")
    }
}

#[derive(Debug)]
struct AggregatedResult {
    result: IndexSearchResult,
    adjusted_lexical_score: f32,
    fuzzy_matches: Vec<FuzzyTermMatch>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SurfaceFuzzyResult {
    #[serde(flatten)]
    result: IndexSearchResult,
    fuzzy_matches: Vec<FuzzyTermMatch>,
}

pub(super) fn search(
    index: &SurfaceIndex,
    query: &IndexQuery,
    options: &FuzzySearchOptions,
) -> Result<serde_json::Value, String> {
    validate_options(query, options)?;

    let chunks = index_chunks(index)?;
    let query_tokens = normalized_query_tokens(&query.text);
    if query_tokens.is_empty() {
        return Err("fuzzy search requires at least one searchable query term".to_string());
    }
    if query_tokens.len() > MAX_FUZZY_QUERY_TERMS {
        return Err(format!(
            "fuzzy search supports at most {MAX_FUZZY_QUERY_TERMS} query terms"
        ));
    }

    let vocabulary = build_vocabulary(&chunks, options)?;
    let variants = build_query_variants(&query_tokens, &vocabulary, options);
    let limit = query.candidate_limit.max(query.top_k).max(1);

    if variants.len() == 1 {
        let results = index.search(query)?;
        return serialize_results(
            results
                .into_iter()
                .map(|result| SurfaceFuzzyResult {
                    result,
                    fuzzy_matches: Vec::new(),
                })
                .collect(),
        );
    }

    let mut best_by_chunk = BTreeMap::<String, AggregatedResult>::new();
    for variant in variants {
        let mut variant_query = query.clone();
        variant_query.text = variant.text();
        variant_query.top_k = limit;
        variant_query.candidate_limit = limit;
        let variant_weight = variant.weight(options);

        for result in index.search(&variant_query)? {
            let adjusted = result.score_breakdown.lexical_score * variant_weight;
            if adjusted <= 0.0 || !adjusted.is_finite() {
                continue;
            }
            let replace = best_by_chunk
                .get(&result.chunk_id)
                .is_none_or(|existing| {
                    adjusted > existing.adjusted_lexical_score
                        || (adjusted == existing.adjusted_lexical_score
                            && variant.fuzzy_matches.len() < existing.fuzzy_matches.len())
                });
            if replace {
                best_by_chunk.insert(
                    result.chunk_id.clone(),
                    AggregatedResult {
                        result,
                        adjusted_lexical_score: adjusted,
                        fuzzy_matches: variant.fuzzy_matches.clone(),
                    },
                );
            }
        }
    }

    let max_score = best_by_chunk
        .values()
        .map(|candidate| candidate.adjusted_lexical_score)
        .fold(0.0_f32, f32::max)
        .max(f32::EPSILON);
    let mut results = best_by_chunk
        .into_values()
        .map(|mut candidate| {
            let normalized = candidate.adjusted_lexical_score / max_score;
            candidate.result.score = normalized;
            candidate.result.score_breakdown.lexical_score = candidate.adjusted_lexical_score;
            candidate.result.score_breakdown.normalized_lexical_score = normalized;
            if query.explain {
                candidate.result.score_breakdown.explanation = Some(if candidate.fuzzy_matches.is_empty() {
                    format!("fuzzy lexical score={:.4}; exact query variant", candidate.adjusted_lexical_score)
                } else {
                    format!(
                        "fuzzy lexical score={:.4}; corrections={}",
                        candidate.adjusted_lexical_score,
                        candidate
                            .fuzzy_matches
                            .iter()
                            .map(|matched| format!(
                                "{}→{}(d={})",
                                matched.query_term, matched.matched_term, matched.edit_distance
                            ))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                });
            }
            SurfaceFuzzyResult {
                result: candidate.result,
                fuzzy_matches: candidate.fuzzy_matches,
            }
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .result
            .score
            .total_cmp(&left.result.score)
            .then_with(|| left.fuzzy_matches.len().cmp(&right.fuzzy_matches.len()))
            .then_with(|| left.result.chunk_id.cmp(&right.result.chunk_id))
    });
    results.truncate(query.top_k);
    serialize_results(results)
}

fn serialize_results(results: Vec<SurfaceFuzzyResult>) -> Result<serde_json::Value, String> {
    serde_json::to_value(results).map_err(|error| error.to_string())
}

fn validate_options(query: &IndexQuery, options: &FuzzySearchOptions) -> Result<(), String> {
    if query.mode != IndexSearchMode::Lexical {
        return Err("fuzzy search currently requires query.mode = lexical".to_string());
    }
    if !(1..=2).contains(&options.max_edit_distance) {
        return Err("fuzzy maxEditDistance must be 1 or 2".to_string());
    }
    if !(2..=MAX_FUZZY_TERM_CHARS).contains(&options.min_term_length) {
        return Err(format!(
            "fuzzy minTermLength must be between 2 and {MAX_FUZZY_TERM_CHARS}"
        ));
    }
    if !(1..=HARD_MAX_EXPANSIONS_PER_TERM).contains(&options.max_expansions_per_term) {
        return Err(format!(
            "fuzzy maxExpansionsPerTerm must be between 1 and {HARD_MAX_EXPANSIONS_PER_TERM}"
        ));
    }
    if !(2..=HARD_MAX_QUERY_VARIANTS).contains(&options.max_query_variants) {
        return Err(format!(
            "fuzzy maxQueryVariants must be between 2 and {HARD_MAX_QUERY_VARIANTS}"
        ));
    }
    if !(1..=HARD_MAX_VOCABULARY_TERMS).contains(&options.max_vocabulary_terms) {
        return Err(format!(
            "fuzzy maxVocabularyTerms must be between 1 and {HARD_MAX_VOCABULARY_TERMS}"
        ));
    }
    if !options.fuzzy_weight.is_finite() || !(0.0..=1.0).contains(&options.fuzzy_weight) || options.fuzzy_weight == 0.0 {
        return Err("fuzzy fuzzyWeight must be finite and greater than 0 and at most 1".to_string());
    }
    Ok(())
}

fn index_chunks(index: &SurfaceIndex) -> Result<Vec<IndexChunk>, String> {
    match index {
        SurfaceIndex::Memory(index) => index.store().chunks().map_err(|error| error.to_string()),
        #[cfg(feature = "sqlite")]
        SurfaceIndex::Sqlite(index) => index.store().chunks().map_err(|error| error.to_string()),
    }
}

fn normalized_query_tokens(text: &str) -> Vec<QueryToken> {
    tokenize(text, &TextProcessingOptions::default())
        .into_iter()
        .filter_map(|token| {
            let searchable = matches!(
                token.kind,
                TokenKind::Word
                    | TokenKind::Number
                    | TokenKind::Email
                    | TokenKind::Url
                    | TokenKind::Mention
                    | TokenKind::Hashtag
            );
            searchable.then(|| QueryToken {
                fuzzy_eligible: token.kind == TokenKind::Word,
                normalized: token.normalized,
            })
        })
        .collect()
}

fn build_vocabulary(
    chunks: &[IndexChunk],
    options: &FuzzySearchOptions,
) -> Result<Vec<VocabularyTerm>, String> {
    let mut frequencies = BTreeMap::<String, usize>::new();
    for chunk in chunks {
        let terms = tokenize(&chunk.text, &TextProcessingOptions::default())
            .into_iter()
            .filter(|token| token.kind == TokenKind::Word)
            .map(|token| token.normalized)
            .filter(|term| {
                let len = term.chars().count();
                len >= options.min_term_length && len <= MAX_FUZZY_TERM_CHARS
            })
            .collect::<BTreeSet<_>>();
        for term in terms {
            *frequencies.entry(term).or_insert(0) += 1;
            if frequencies.len() > options.max_vocabulary_terms {
                return Err(format!(
                    "fuzzy vocabulary exceeds configured maxVocabularyTerms ({})",
                    options.max_vocabulary_terms
                ));
            }
        }
    }
    Ok(frequencies
        .into_iter()
        .map(|(term, document_frequency)| VocabularyTerm {
            term,
            document_frequency,
        })
        .collect())
}

fn build_query_variants(
    tokens: &[QueryToken],
    vocabulary: &[VocabularyTerm],
    options: &FuzzySearchOptions,
) -> Vec<QueryVariant> {
    let original_terms = tokens
        .iter()
        .map(|token| token.normalized.clone())
        .collect::<Vec<_>>();
    let mut variants = vec![QueryVariant {
        terms: original_terms,
        fuzzy_matches: Vec::new(),
    }];

    for (index, token) in tokens.iter().enumerate() {
        if !token.fuzzy_eligible {
            continue;
        }
        let term_len = token.normalized.chars().count();
        if term_len < options.min_term_length || term_len > MAX_FUZZY_TERM_CHARS {
            continue;
        }
        let expansions = fuzzy_expansions(&token.normalized, vocabulary, options);
        if expansions.is_empty() {
            continue;
        }
        let existing = variants.clone();
        for variant in existing {
            for matched in &expansions {
                let mut next = variant.clone();
                next.terms[index] = matched.matched_term.clone();
                next.fuzzy_matches.push(matched.clone());
                variants.push(next);
            }
        }
        variants = deduplicate_and_bound_variants(variants, options);
    }

    variants
}

fn deduplicate_and_bound_variants(
    variants: Vec<QueryVariant>,
    options: &FuzzySearchOptions,
) -> Vec<QueryVariant> {
    let mut unique = BTreeMap::<String, QueryVariant>::new();
    for variant in variants {
        let key = variant.text();
        match unique.get(&key) {
            Some(existing) if existing.weight(options) >= variant.weight(options) => {}
            _ => {
                unique.insert(key, variant);
            }
        }
    }
    let mut variants = unique.into_values().collect::<Vec<_>>();
    variants.sort_by(|left, right| {
        right
            .weight(options)
            .total_cmp(&left.weight(options))
            .then_with(|| left.fuzzy_matches.len().cmp(&right.fuzzy_matches.len()))
            .then_with(|| left.text().cmp(&right.text()))
    });
    variants.truncate(options.max_query_variants);
    variants
}

fn fuzzy_expansions(
    query_term: &str,
    vocabulary: &[VocabularyTerm],
    options: &FuzzySearchOptions,
) -> Vec<FuzzyTermMatch> {
    let query_len = query_term.chars().count();
    let mut matches = vocabulary
        .iter()
        .filter_map(|candidate| {
            if candidate.term == query_term {
                return None;
            }
            let candidate_len = candidate.term.chars().count();
            if query_len.abs_diff(candidate_len) > options.max_edit_distance {
                return None;
            }
            let distance = bounded_damerau_levenshtein(
                query_term,
                &candidate.term,
                options.max_edit_distance,
            )?;
            if distance == 0 {
                return None;
            }
            let similarity = 1.0 - distance as f32 / query_len.max(candidate_len).max(1) as f32;
            Some((
                FuzzyTermMatch {
                    query_term: query_term.to_string(),
                    matched_term: candidate.term.clone(),
                    edit_distance: distance,
                    similarity,
                },
                candidate.document_frequency,
            ))
        })
        .collect::<Vec<_>>();
    matches.sort_by(|(left, left_frequency), (right, right_frequency)| {
        left.edit_distance
            .cmp(&right.edit_distance)
            .then_with(|| right_frequency.cmp(left_frequency))
            .then_with(|| left.matched_term.cmp(&right.matched_term))
    });
    matches.truncate(options.max_expansions_per_term);
    matches.into_iter().map(|(matched, _)| matched).collect()
}

fn bounded_damerau_levenshtein(left: &str, right: &str, max_distance: usize) -> Option<usize> {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    if left.len().abs_diff(right.len()) > max_distance {
        return None;
    }
    if left == right {
        return Some(0);
    }

    let mut previous_previous = vec![0; right.len() + 1];
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_char) in left.iter().enumerate() {
        let row = left_index + 1;
        let mut current = vec![row; right.len() + 1];
        for (right_index, right_char) in right.iter().enumerate() {
            let column = right_index + 1;
            let substitution_cost = usize::from(left_char != right_char);
            let mut distance = (previous[column] + 1)
                .min(current[column - 1] + 1)
                .min(previous[column - 1] + substitution_cost);
            if row > 1
                && column > 1
                && left[row - 1] == right[column - 2]
                && left[row - 2] == right[column - 1]
            {
                distance = distance.min(previous_previous[column - 2] + 1);
            }
            current[column] = distance;
        }
        previous_previous = previous;
        previous = current;
    }
    let distance = previous[right.len()];
    (distance <= max_distance).then_some(distance)
}

fn default_max_edit_distance() -> usize {
    1
}

fn default_min_term_length() -> usize {
    4
}

fn default_max_expansions_per_term() -> usize {
    3
}

fn default_max_query_variants() -> usize {
    16
}

fn default_max_vocabulary_terms() -> usize {
    20_000
}

fn default_fuzzy_weight() -> f32 {
    0.8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IndexDocument, MemoryTextIndex};

    fn index(documents: &[(&str, &str)]) -> SurfaceIndex {
        let mut index = MemoryTextIndex::new_memory().expect("memory index");
        let documents = documents
            .iter()
            .map(|(id, body)| IndexDocument::new(*id, *body))
            .collect::<Vec<_>>();
        index.upsert_documents(&documents).expect("index documents");
        SurfaceIndex::Memory(index)
    }

    #[test]
    fn damerau_distance_treats_adjacent_transposition_as_one_edit() {
        assert_eq!(bounded_damerau_levenshtein("strategy", "stratgey", 1), Some(1));
        assert_eq!(bounded_damerau_levenshtein("search", "searh", 1), Some(1));
        assert_eq!(bounded_damerau_levenshtein("search", "fetch", 1), None);
    }

    #[test]
    fn fuzzy_search_recovers_a_transposed_query_term() {
        let index = index(&[
            ("strategy", "medieval strategy combat formations"),
            ("recipe", "kitchen recipe ingredients"),
        ]);
        let query = IndexQuery::lexical("stratgey formations", 3);
        let results = search(&index, &query, &FuzzySearchOptions::default()).expect("fuzzy search");
        let results = results.as_array().expect("result array");
        assert_eq!(results[0]["documentId"], "strategy");
        assert_eq!(results[0]["fuzzyMatches"][0]["queryTerm"], "stratgey");
        assert_eq!(results[0]["fuzzyMatches"][0]["matchedTerm"], "strategy");
    }

    #[test]
    fn exact_term_match_is_not_penalized_by_fuzzy_mode() {
        let index = index(&[
            ("exact", "strategy strategy formations"),
            ("typo", "stratgey formations"),
        ]);
        let query = IndexQuery::lexical("strategy formations", 2);
        let results = search(&index, &query, &FuzzySearchOptions::default()).expect("fuzzy search");
        let results = results.as_array().expect("result array");
        assert_eq!(results[0]["documentId"], "exact");
        assert!(results[0]["fuzzyMatches"].as_array().expect("matches").is_empty());
    }
}
