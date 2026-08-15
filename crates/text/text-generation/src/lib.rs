#![doc = include_str!("../README.md")]

pub mod surface;
mod synthesis;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use text_core::tokenize_words;
use text_core::{DetectError, Result};

pub use synthesis::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for markov prediction.
pub struct MarkovPrediction {
    /// The token value.
    pub token: String,
    /// Number of items represented by this value.
    pub count: usize,
    /// The probability value.
    pub probability: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for markov generation.
pub struct MarkovGeneration {
    /// The tokens value.
    pub tokens: Vec<String>,
    /// Text content for this value.
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for markov chain.
pub struct MarkovChain {
    order: usize,
    transitions: BTreeMap<Vec<String>, BTreeMap<String, usize>>,
    starts: BTreeMap<Vec<String>, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
/// Variants describing markov input mode.
pub enum MarkovInputMode {
    /// The surface variant.
    Surface,
    #[default]
    /// The normalized variant.
    Normalized,
    /// The lemma variant.
    Lemma,
    /// The entity aware variant.
    EntityAware,
}

impl MarkovChain {
    /// Creates a new value.
    pub fn new(order: usize) -> Result<Self> {
        if order == 0 {
            return Err(invalid_argument("markov order must be greater than zero"));
        }
        Ok(Self {
            order,
            transitions: BTreeMap::new(),
            starts: BTreeMap::new(),
        })
    }

    /// Returns order.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Returns whether is empty.
    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    /// Returns total transitions.
    pub fn total_transitions(&self) -> usize {
        self.transitions
            .values()
            .map(|counts| counts.values().sum::<usize>())
            .sum()
    }

    /// Returns contexts.
    pub fn contexts(&self) -> usize {
        self.transitions.len()
    }

    /// Returns starts.
    pub fn starts(&self) -> &BTreeMap<Vec<String>, usize> {
        &self.starts
    }

    /// Returns transitions.
    pub fn transitions(&self) -> &BTreeMap<Vec<String>, BTreeMap<String, usize>> {
        &self.transitions
    }

    /// Returns train text.
    pub fn train_text(&mut self, text: &str) {
        let tokens = tokenize_words(text);
        self.train_tokens(&tokens);
    }

    /// Returns train documents.
    pub fn train_documents<I, S>(&mut self, documents: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for document in documents {
            self.train_text(document.as_ref());
        }
    }

    /// Returns train tokens.
    pub fn train_tokens(&mut self, tokens: &[String]) {
        if tokens.len() < self.order {
            return;
        }
        *self
            .starts
            .entry(tokens[..self.order].to_vec())
            .or_insert(0) += 1;

        for index in self.order..tokens.len() {
            let context = tokens[index - self.order..index].to_vec();
            let next = tokens[index].clone();
            *self
                .transitions
                .entry(context)
                .or_default()
                .entry(next)
                .or_insert(0) += 1;
        }
    }

    /// Returns predict next.
    pub fn predict_next<'a>(
        &self,
        context: impl IntoIterator<Item = &'a str>,
        limit: usize,
    ) -> Result<Vec<MarkovPrediction>> {
        let tokens = context
            .into_iter()
            .map(|token| token.to_lowercase())
            .collect::<Vec<_>>();
        self.predict_next_tokens(&tokens, limit)
    }

    /// Returns predict next tokens.
    pub fn predict_next_tokens(
        &self,
        context: &[String],
        limit: usize,
    ) -> Result<Vec<MarkovPrediction>> {
        if limit == 0 {
            return Err(invalid_argument(
                "prediction limit must be greater than zero",
            ));
        }
        let context = self.context_suffix(context)?;
        let Some(next_counts) = self.transitions.get(&context) else {
            return Ok(Vec::new());
        };
        let total = next_counts.values().sum::<usize>().max(1) as f32;
        let mut predictions = next_counts
            .iter()
            .map(|(token, count)| MarkovPrediction {
                token: token.clone(),
                count: *count,
                probability: *count as f32 / total,
            })
            .collect::<Vec<_>>();
        predictions.sort_by(|left, right| {
            right
                .count
                .cmp(&left.count)
                .then_with(|| left.token.cmp(&right.token))
        });
        predictions.truncate(limit);
        Ok(predictions)
    }

    /// Returns generate.
    pub fn generate(&self, seed: &[&str], max_tokens: usize) -> Result<MarkovGeneration> {
        let seed = seed
            .iter()
            .map(|token| token.to_lowercase())
            .collect::<Vec<_>>();
        self.generate_from_tokens(&seed, max_tokens)
    }

    /// Returns generate from tokens.
    pub fn generate_from_tokens(
        &self,
        seed: &[String],
        max_tokens: usize,
    ) -> Result<MarkovGeneration> {
        if max_tokens == 0 {
            return Err(invalid_argument(
                "generation token count must be greater than zero",
            ));
        }
        if self.starts.is_empty() {
            return Ok(MarkovGeneration {
                tokens: Vec::new(),
                text: String::new(),
            });
        }

        let mut tokens = if seed.len() >= self.order {
            seed.to_vec()
        } else if seed.is_empty() {
            self.best_start()
                .ok_or_else(|| invalid_argument("markov model has no start states"))?
        } else {
            self.best_start_matching(seed)
                .unwrap_or_else(|| seed.to_vec())
        };

        while tokens.len() < max_tokens {
            let context = match self.context_suffix(&tokens) {
                Ok(context) => context,
                Err(_) => break,
            };
            let Some(next_counts) = self.transitions.get(&context) else {
                break;
            };
            let Some(next) = best_next(next_counts) else {
                break;
            };
            tokens.push(next);
        }

        Ok(MarkovGeneration {
            text: tokens.join(" "),
            tokens,
        })
    }

    /// Returns perplexity.
    pub fn perplexity(&self, text: &str) -> Result<f32> {
        let tokens = tokenize_words(text);
        if tokens.len() <= self.order {
            return Err(invalid_argument(
                "perplexity text must contain more tokens than the markov order",
            ));
        }

        let mut log_probability = 0.0_f32;
        let mut steps = 0;
        for index in self.order..tokens.len() {
            let context = tokens[index - self.order..index].to_vec();
            let next = &tokens[index];
            let Some(next_counts) = self.transitions.get(&context) else {
                return Ok(f32::INFINITY);
            };
            let total = next_counts.values().sum::<usize>() as f32;
            let count = next_counts.get(next).copied().unwrap_or(0);
            if count == 0 {
                return Ok(f32::INFINITY);
            }
            log_probability += (count as f32 / total).ln();
            steps += 1;
        }

        Ok((-log_probability / steps as f32).exp())
    }

    fn context_suffix(&self, context: &[String]) -> Result<Vec<String>> {
        if context.len() < self.order {
            return Err(invalid_argument(format!(
                "context must contain at least {} token(s)",
                self.order
            )));
        }
        Ok(context[context.len() - self.order..].to_vec())
    }

    fn best_start(&self) -> Option<Vec<String>> {
        self.starts
            .iter()
            .max_by(|(left_context, left_count), (right_context, right_count)| {
                left_count
                    .cmp(right_count)
                    .then_with(|| right_context.cmp(left_context))
            })
            .map(|(context, _)| context.clone())
    }

    fn best_start_matching(&self, seed: &[String]) -> Option<Vec<String>> {
        self.starts
            .iter()
            .filter(|(context, _)| context.starts_with(seed))
            .max_by(|(left_context, left_count), (right_context, right_count)| {
                left_count
                    .cmp(right_count)
                    .then_with(|| right_context.cmp(left_context))
            })
            .map(|(context, _)| context.clone())
    }
}

impl Default for MarkovChain {
    fn default() -> Self {
        Self::new(2).expect("default markov order is valid")
    }
}

fn best_next(counts: &BTreeMap<String, usize>) -> Option<String> {
    counts
        .iter()
        .max_by(|(left_token, left_count), (right_token, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_token.cmp(left_token))
        })
        .map(|(token, _)| token.clone())
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_next_tokens_from_markov_context() {
        let mut chain = MarkovChain::new(2).unwrap();
        chain.train_text("the quick brown fox the quick blue bird the quick blue sky");

        let predictions = chain.predict_next(["the", "quick"], 2).unwrap();
        assert_eq!(predictions[0].token, "blue");
        assert_eq!(predictions[0].count, 2);
        assert!((predictions[0].probability - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn uses_suffix_context_for_predictions() {
        let mut chain = MarkovChain::new(2).unwrap();
        chain.train_text("rust cargo build rust cargo test");

        let predictions = chain
            .predict_next_tokens(
                &[
                    "ignore".to_string(),
                    "rust".to_string(),
                    "cargo".to_string(),
                ],
                2,
            )
            .unwrap();

        assert_eq!(
            predictions
                .iter()
                .map(|prediction| prediction.token.as_str())
                .collect::<Vec<_>>(),
            vec!["build", "test"]
        );
    }

    #[test]
    fn generates_deterministic_text() {
        let mut chain = MarkovChain::new(1).unwrap();
        chain.train_text("rust builds crates rust builds tools");

        let generated = chain.generate(&["rust"], 4).unwrap();
        assert_eq!(generated.tokens, vec!["rust", "builds", "crates", "rust"]);
    }

    #[test]
    fn computes_perplexity_for_seen_text() {
        let mut chain = MarkovChain::new(1).unwrap();
        chain.train_text("a b a b a b");

        let perplexity = chain.perplexity("a b a b").unwrap();
        assert!(perplexity.is_finite());
        assert!(perplexity <= 1.1);
    }

    #[test]
    fn returns_infinite_perplexity_for_unseen_transition() {
        let mut chain = MarkovChain::new(1).unwrap();
        chain.train_text("a b a b");

        assert_eq!(chain.perplexity("a c").unwrap(), f32::INFINITY);
    }

    #[test]
    fn validates_order_limits_and_context_lengths() {
        assert!(MarkovChain::new(0).is_err());

        let mut chain = MarkovChain::new(2).unwrap();
        chain.train_text("rust cargo build");

        assert!(chain.predict_next(["rust", "cargo"], 0).is_err());
        assert!(chain.predict_next(["rust"], 1).is_err());
        assert!(chain.generate(&["rust", "cargo"], 0).is_err());
        assert!(chain.perplexity("rust cargo").is_err());
    }

    #[test]
    fn trains_from_owned_documents_and_tracks_transitions() {
        let mut chain = MarkovChain::new(2).unwrap();
        assert!(chain.is_empty());

        chain.train_documents([
            "rust cargo builds crates".to_string(),
            "rust cargo runs tests".to_string(),
        ]);

        assert!(!chain.is_empty());
        assert_eq!(chain.total_transitions(), 4);
    }
}
