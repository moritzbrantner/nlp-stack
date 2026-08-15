use std::collections::{BTreeMap, BTreeSet};

use text_core::{Token, TokenKind};

#[derive(Debug, Clone, PartialEq)]
/// Data type for lemma.
pub struct Lemma {
    /// The token index value.
    pub token_index: usize,
    /// The value value.
    pub value: String,
    /// Language tag for this value.
    pub language: Option<String>,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Data type for lemma options.
pub struct LemmaOptions {
    /// Language tag for this value.
    pub language: Option<String>,
    /// The lowercase proper nouns value.
    pub lowercase_proper_nouns: bool,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for lemmatization result.
pub struct LemmatizationResult {
    /// Language tag for this value.
    pub language: Option<String>,
    /// The lemmas value.
    pub lemmas: Vec<Lemma>,
}

/// Returns lemmatize tokens.
pub fn lemmatize_tokens(
    tokens: &[Token],
    language: Option<&str>,
    options: &LemmaOptions,
) -> LemmatizationResult {
    let language = options
        .language
        .as_deref()
        .or(language)
        .map(ToString::to_string);
    let lemmas = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            let (value, confidence) = heuristic_lemma(token, language.as_deref(), options);
            Lemma {
                token_index: index,
                value,
                language: language.clone(),
                confidence,
            }
        })
        .collect();
    LemmatizationResult { language, lemmas }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Variants describing morph feature.
pub enum MorphFeature {
    /// The number sing variant.
    NumberSing,
    /// The number plur variant.
    NumberPlur,
    /// The person1 variant.
    Person1,
    /// The person2 variant.
    Person2,
    /// The person3 variant.
    Person3,
    /// The tense past variant.
    TensePast,
    /// The tense pres variant.
    TensePres,
    /// The tense fut variant.
    TenseFut,
    /// The aspect prog variant.
    AspectProg,
    /// The aspect perf variant.
    AspectPerf,
    /// The mood imp variant.
    MoodImp,
    /// The verb form fin variant.
    VerbFormFin,
    /// The verb form inf variant.
    VerbFormInf,
    /// The case nom variant.
    CaseNom,
    /// The case acc variant.
    CaseAcc,
    /// The case dat variant.
    CaseDat,
    /// The gender masc variant.
    GenderMasc,
    /// The gender fem variant.
    GenderFem,
    /// The gender neut variant.
    GenderNeut,
    /// The definiteness def variant.
    DefinitenessDef,
    /// The definiteness ind variant.
    DefinitenessInd,
    /// The polarity neg variant.
    PolarityNeg,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for morph tag set.
pub struct MorphTagSet {
    /// The features value.
    pub features: BTreeSet<MorphFeature>,
}

impl MorphTagSet {
    /// Creates a new value.
    pub fn new() -> Self {
        Self {
            features: BTreeSet::new(),
        }
    }

    /// Returns insert.
    pub fn insert(&mut self, feature: MorphFeature) {
        self.features.insert(feature);
    }
}

impl Default for MorphTagSet {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for morph annotation.
pub struct MorphAnnotation {
    /// The token index value.
    pub token_index: usize,
    /// The lemma value.
    pub lemma: Option<String>,
    /// The tags value.
    pub tags: MorphTagSet,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing pos tag.
pub enum PosTag {
    /// The adj variant.
    Adj,
    /// The adp variant.
    Adp,
    /// The adv variant.
    Adv,
    /// The aux variant.
    Aux,
    /// The cconj variant.
    Cconj,
    /// The det variant.
    Det,
    /// The intj variant.
    Intj,
    /// The noun variant.
    Noun,
    /// The num variant.
    Num,
    /// The part variant.
    Part,
    /// The pron variant.
    Pron,
    /// The propn variant.
    Propn,
    /// The punct variant.
    Punct,
    /// The sconj variant.
    Sconj,
    /// The sym variant.
    Sym,
    /// The verb variant.
    Verb,
    /// The x variant.
    X,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for pos annotation.
pub struct PosAnnotation {
    /// The token index value.
    pub token_index: usize,
    /// The tag value.
    pub tag: PosTag,
    /// Confidence score for this value.
    pub confidence: f32,
    /// The reason value.
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Data type for pos tagging options.
pub struct PosTaggingOptions {
    /// Language tag for this value.
    pub language: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Data type for pos tagger.
pub struct PosTagger {
    /// The options value.
    pub options: PosTaggingOptions,
}

impl PosTagger {
    /// Returns tag tokens.
    pub fn tag_tokens(&self, tokens: &[Token], lemmas: &LemmatizationResult) -> Vec<PosAnnotation> {
        tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let lemma = lemmas.lemmas.get(index).map(|lemma| lemma.value.as_str());
                heuristic_pos(token, lemma, index, index == 0)
            })
            .collect()
    }
}

/// Returns annotate morphology.
pub fn annotate_morphology(
    tokens: &[Token],
    lemmas: &LemmatizationResult,
    pos: &[PosAnnotation],
) -> Vec<MorphAnnotation> {
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            let mut tags = MorphTagSet::new();
            let lemma = lemmas.lemmas.get(index).map(|lemma| lemma.value.clone());
            let pos_tag = pos
                .get(index)
                .map(|annotation| annotation.tag)
                .unwrap_or(PosTag::X);
            let normalized = token.normalized.as_str();
            match pos_tag {
                PosTag::Pron => {
                    if ["i", "me", "my", "we", "us", "our"].contains(&normalized) {
                        tags.insert(MorphFeature::Person1);
                    } else if ["you", "your"].contains(&normalized) {
                        tags.insert(MorphFeature::Person2);
                    } else {
                        tags.insert(MorphFeature::Person3);
                    }
                    if ["they", "them", "their", "we", "us", "our"].contains(&normalized) {
                        tags.insert(MorphFeature::NumberPlur);
                    } else {
                        tags.insert(MorphFeature::NumberSing);
                    }
                    if ["he", "him", "his"].contains(&normalized) {
                        tags.insert(MorphFeature::GenderMasc);
                    }
                    if ["she", "her", "hers"].contains(&normalized) {
                        tags.insert(MorphFeature::GenderFem);
                    }
                    if ["it", "its"].contains(&normalized) {
                        tags.insert(MorphFeature::GenderNeut);
                    }
                    tags.insert(MorphFeature::CaseNom);
                }
                PosTag::Det => {
                    if ["the", "this", "that", "these", "those"].contains(&normalized) {
                        tags.insert(MorphFeature::DefinitenessDef);
                    } else {
                        tags.insert(MorphFeature::DefinitenessInd);
                    }
                }
                PosTag::Noun | PosTag::Propn => {
                    if normalized.ends_with('s') && !normalized.ends_with("ss") {
                        tags.insert(MorphFeature::NumberPlur);
                    } else {
                        tags.insert(MorphFeature::NumberSing);
                    }
                }
                PosTag::Verb | PosTag::Aux => {
                    if normalized.ends_with("ed")
                        || ["was", "were", "did", "had"].contains(&normalized)
                    {
                        tags.insert(MorphFeature::TensePast);
                    } else if normalized == "will" || normalized == "shall" {
                        tags.insert(MorphFeature::TenseFut);
                    } else {
                        tags.insert(MorphFeature::TensePres);
                    }
                    if normalized.ends_with("ing") {
                        tags.insert(MorphFeature::AspectProg);
                    }
                    if ["have", "has", "had"].contains(&normalized) {
                        tags.insert(MorphFeature::AspectPerf);
                    }
                    if normalized == "to" {
                        tags.insert(MorphFeature::VerbFormInf);
                    } else {
                        tags.insert(MorphFeature::VerbFormFin);
                    }
                }
                _ => {}
            }
            if ["not", "never", "no"].contains(&normalized) {
                tags.insert(MorphFeature::PolarityNeg);
            }
            MorphAnnotation {
                token_index: index,
                lemma,
                tags,
                confidence: if lemmas.language.as_deref() == Some("en") {
                    0.75
                } else {
                    0.4
                },
            }
        })
        .collect()
}

fn heuristic_lemma(token: &Token, language: Option<&str>, options: &LemmaOptions) -> (String, f32) {
    let normalized = if token.kind == TokenKind::Punctuation {
        return (token.text.clone(), 1.0);
    } else if matches!(
        token.kind,
        TokenKind::Url
            | TokenKind::Email
            | TokenKind::Mention
            | TokenKind::Hashtag
            | TokenKind::Number
    ) {
        return (token.normalized.clone(), 1.0);
    } else if options.lowercase_proper_nouns {
        token.text.to_lowercase()
    } else {
        token.normalized.clone()
    };

    if language != Some("en") {
        return (normalized, 0.35);
    }

    let irregular = BTreeMap::from([
        ("was", "be"),
        ("were", "be"),
        ("is", "be"),
        ("are", "be"),
        ("am", "be"),
        ("has", "have"),
        ("had", "have"),
        ("did", "do"),
        ("does", "do"),
        ("went", "go"),
        ("gone", "go"),
        ("children", "child"),
        ("men", "man"),
        ("women", "woman"),
        ("mice", "mouse"),
        ("geese", "goose"),
    ]);
    if let Some(lemma) = irregular.get(normalized.as_str()) {
        return ((*lemma).to_string(), 0.95);
    }
    if normalized.ends_with("ies") && normalized.len() > 4 {
        return (format!("{}y", &normalized[..normalized.len() - 3]), 0.85);
    }
    if normalized.ends_with("ing") && normalized.len() > 5 {
        return (
            normalized
                .trim_end_matches("ing")
                .trim_end_matches(char::is_whitespace)
                .to_string(),
            0.8,
        );
    }
    if normalized.ends_with("ed") && normalized.len() > 4 {
        return (normalized.trim_end_matches("ed").to_string(), 0.8);
    }
    if normalized.ends_with("es") && normalized.len() > 4 {
        let candidate = normalized.trim_end_matches("es");
        if candidate.ends_with('h')
            || candidate.ends_with('s')
            || candidate.ends_with('x')
            || candidate.ends_with('z')
            || candidate.ends_with("ch")
            || candidate.ends_with("sh")
        {
            return (candidate.to_string(), 0.78);
        }
    }
    if normalized.ends_with('s') && normalized.len() > 3 && !normalized.ends_with("ss") {
        return (normalized.trim_end_matches('s').to_string(), 0.75);
    }
    (normalized, 0.7)
}

fn heuristic_pos(
    token: &Token,
    lemma: Option<&str>,
    token_index: usize,
    is_sentence_initial: bool,
) -> PosAnnotation {
    let normalized = token.normalized.as_str();
    let lemma = lemma.unwrap_or(normalized);
    let (tag, confidence, reason) = match token.kind {
        TokenKind::Punctuation => (PosTag::Punct, 1.0, "punctuation token".to_string()),
        TokenKind::Number => (PosTag::Num, 1.0, "numeric token".to_string()),
        TokenKind::Url | TokenKind::Email => (PosTag::Sym, 1.0, "symbolic token".to_string()),
        TokenKind::Mention | TokenKind::Hashtag => (PosTag::Propn, 0.9, "social token".to_string()),
        TokenKind::Other => (PosTag::X, 0.5, "unclassified token".to_string()),
        TokenKind::Word => {
            if ["the", "a", "an", "this", "that", "these", "those"].contains(&normalized) {
                (PosTag::Det, 0.95, "determiner lexicon".to_string())
            } else if [
                "he", "she", "they", "it", "we", "i", "you", "me", "us", "them",
            ]
            .contains(&normalized)
            {
                (PosTag::Pron, 0.95, "pronoun lexicon".to_string())
            } else if ["and", "or", "but"].contains(&normalized) {
                (PosTag::Cconj, 0.95, "coordinating conjunction".to_string())
            } else if ["because", "if", "while", "although", "that"].contains(&normalized) {
                (PosTag::Sconj, 0.9, "subordinating conjunction".to_string())
            } else if [
                "in", "on", "at", "for", "from", "to", "with", "by", "over", "under",
            ]
            .contains(&normalized)
            {
                (PosTag::Adp, 0.95, "adposition lexicon".to_string())
            } else if [
                "is", "are", "was", "were", "be", "been", "being", "am", "have", "has", "had",
                "do", "does", "did", "will", "shall",
            ]
            .contains(&normalized)
            {
                (PosTag::Aux, 0.9, "auxiliary lexicon".to_string())
            } else if normalized.ends_with("ly") {
                (PosTag::Adv, 0.8, "adverb suffix".to_string())
            } else if normalized.ends_with("ing")
                || normalized.ends_with("ed")
                || normalized.ends_with("es")
                || ["go", "make", "say", "present", "launch", "visit"].contains(&lemma)
            {
                (PosTag::Verb, 0.8, "verbal morphology".to_string())
            } else if normalized.ends_with("ous")
                || normalized.ends_with("ive")
                || normalized.ends_with("al")
                || normalized.ends_with("ful")
            {
                (PosTag::Adj, 0.75, "adjective suffix".to_string())
            } else if token
                .text
                .chars()
                .next()
                .is_some_and(|ch| ch.is_uppercase())
                && ![
                    "the", "a", "an", "this", "that", "these", "those", "and", "but", "or", "in",
                    "on", "at", "for", "from", "to", "with", "by",
                ]
                .contains(&normalized)
                && (!is_sentence_initial || token.text.chars().skip(1).any(|ch| ch.is_lowercase()))
            {
                (PosTag::Propn, 0.7, "capitalized token".to_string())
            } else {
                (PosTag::Noun, 0.65, "noun fallback".to_string())
            }
        }
    };
    PosAnnotation {
        token_index,
        tag,
        confidence,
        reason,
    }
}
