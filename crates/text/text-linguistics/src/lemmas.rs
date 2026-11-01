//! UD-derived lemma lookup for German and Spanish.
//!
//! Tables are built by `scripts/build_ud_lemma_tables.py` from the training splits
//! of UD_German-GSD 2.17 (CC BY-SA 4.0) and UD_Spanish-AnCora 2.17 (CC BY 4.0):
//! suffix edit rules per UPOS plus an exception list for the training words the
//! rules miss. The rule application here mirrors that script.

use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use crate::morphology::PosTag;

const MAX_SUFFIX: usize = 7;

/// Lemmatizer predicting a lemma from a surface form and its UPOS.
#[derive(Debug, Default)]
pub struct UdLemmatizer {
    capitalized: BTreeSet<String>,
    keep_form: BTreeSet<String>,
    rules: HashMap<(String, String), (usize, String)>,
    exceptions: HashMap<(String, String), String>,
}

impl UdLemmatizer {
    /// The lemmatizer for the primary subtag of `language`, if a table ships for it.
    pub fn for_language(language: &str) -> Option<&'static Self> {
        static DE: OnceLock<UdLemmatizer> = OnceLock::new();
        static ES: OnceLock<UdLemmatizer> = OnceLock::new();
        let primary = language
            .trim()
            .split(['-', '_'])
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        match primary.as_str() {
            "de" => Some(DE.get_or_init(|| Self::parse(include_str!("../data/lemmas/de.tsv")))),
            "es" => Some(ES.get_or_init(|| Self::parse(include_str!("../data/lemmas/es.tsv")))),
            _ => None,
        }
    }

    /// Parses a table in the format written by `scripts/build_ud_lemma_tables.py`.
    pub fn parse(table: &str) -> Self {
        let mut lemmatizer = Self::default();
        for line in table.lines() {
            if line.starts_with('#') {
                continue;
            }
            let cols = line.split('\t').collect::<Vec<_>>();
            match cols.as_slice() {
                ["C", upos] => {
                    lemmatizer.capitalized.insert((*upos).to_string());
                }
                ["K", upos] => {
                    lemmatizer.keep_form.insert((*upos).to_string());
                }
                ["R", upos, suffix, strip, append] => {
                    if let Ok(strip) = strip.parse() {
                        lemmatizer.rules.insert(
                            ((*upos).to_string(), (*suffix).to_string()),
                            (strip, (*append).to_string()),
                        );
                    }
                }
                ["E", form, upos, lemma] => {
                    lemmatizer.exceptions.insert(
                        ((*form).to_string(), (*upos).to_string()),
                        (*lemma).to_string(),
                    );
                }
                _ => {}
            }
        }
        lemmatizer
    }

    /// Lemma for `form` tagged `upos`.
    pub fn lemmatize(&self, form: &str, upos: PosTag) -> String {
        let upos = upos.as_ud();
        let lower = form.to_lowercase();
        if let Some(lemma) = self.exceptions.get(&(lower.clone(), upos.to_string())) {
            return lemma.clone();
        }
        if self.keep_form.contains(upos) {
            return form.to_string();
        }
        let chars = lower.chars().collect::<Vec<_>>();
        for length in (0..=MAX_SUFFIX.min(chars.len())).rev() {
            let suffix = chars[chars.len() - length..].iter().collect::<String>();
            let Some((strip, append)) = self.rules.get(&(upos.to_string(), suffix)) else {
                continue;
            };
            if *strip > chars.len() {
                continue;
            }
            let mut candidate = chars[..chars.len() - strip].iter().collect::<String>();
            candidate.push_str(append);
            if !candidate.is_empty() {
                return self.recase(candidate, upos);
            }
        }
        self.recase(lower, upos)
    }

    fn recase(&self, lemma: String, upos: &str) -> String {
        if !self.capitalized.contains(upos) {
            return lemma;
        }
        let mut chars = lemma.chars();
        match chars.next() {
            Some(first) => first.to_uppercase().chain(chars).collect(),
            None => lemma,
        }
    }
}
