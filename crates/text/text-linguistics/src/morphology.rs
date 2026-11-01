use std::collections::BTreeMap;

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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// UD morphological features (`FEATS`), keyed by feature name (`Case`, `Number`, ...).
pub struct MorphTagSet {
    /// Feature name to value, e.g. `Number` -> `Sing`.
    pub features: BTreeMap<String, String>,
}

impl MorphTagSet {
    /// Creates an empty feature set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets feature `name` to `value`.
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.features.insert(name.into(), value.into());
    }

    /// The value of feature `name`.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.features.get(name).map(String::as_str)
    }

    /// Whether feature `name` has `value`.
    pub fn has(&self, name: &str, value: &str) -> bool {
        self.get(name) == Some(value)
    }

    /// Whether no feature is set.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// CoNLL-U `FEATS` notation (`Case=Nom|Number=Sing`, `_` when empty), with
    /// features in UD's case-insensitive alphabetical order (`Number` before `NumType`).
    pub fn to_ud_string(&self) -> String {
        if self.features.is_empty() {
            return "_".to_string();
        }
        let mut features = self.features.iter().collect::<Vec<_>>();
        features.sort_by_key(|(name, _)| name.to_lowercase());
        features
            .into_iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join("|")
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

impl PosTag {
    /// Every UPOS tag.
    pub const ALL: &'static [Self] = &[
        Self::Adj,
        Self::Adp,
        Self::Adv,
        Self::Aux,
        Self::Cconj,
        Self::Det,
        Self::Intj,
        Self::Noun,
        Self::Num,
        Self::Part,
        Self::Pron,
        Self::Propn,
        Self::Punct,
        Self::Sconj,
        Self::Sym,
        Self::Verb,
        Self::X,
    ];

    /// The Universal Dependencies UPOS label (`NOUN`, `PROPN`, ...).
    pub fn as_ud(self) -> &'static str {
        match self {
            Self::Adj => "ADJ",
            Self::Adp => "ADP",
            Self::Adv => "ADV",
            Self::Aux => "AUX",
            Self::Cconj => "CCONJ",
            Self::Det => "DET",
            Self::Intj => "INTJ",
            Self::Noun => "NOUN",
            Self::Num => "NUM",
            Self::Part => "PART",
            Self::Pron => "PRON",
            Self::Propn => "PROPN",
            Self::Punct => "PUNCT",
            Self::Sconj => "SCONJ",
            Self::Sym => "SYM",
            Self::Verb => "VERB",
            Self::X => "X",
        }
    }

    /// Parses a UD UPOS label.
    pub fn from_ud(label: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|tag| tag.as_ud() == label)
    }
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
                        tags.insert("Person", "1");
                    } else if ["you", "your"].contains(&normalized) {
                        tags.insert("Person", "2");
                    } else {
                        tags.insert("Person", "3");
                    }
                    if ["they", "them", "their", "we", "us", "our"].contains(&normalized) {
                        tags.insert("Number", "Plur");
                    } else {
                        tags.insert("Number", "Sing");
                    }
                    if ["he", "him", "his"].contains(&normalized) {
                        tags.insert("Gender", "Masc");
                    }
                    if ["she", "her", "hers"].contains(&normalized) {
                        tags.insert("Gender", "Fem");
                    }
                    if ["it", "its"].contains(&normalized) {
                        tags.insert("Gender", "Neut");
                    }
                    tags.insert("Case", "Nom");
                }
                PosTag::Det => {
                    if ["the", "this", "that", "these", "those"].contains(&normalized) {
                        tags.insert("Definite", "Def");
                    } else {
                        tags.insert("Definite", "Ind");
                    }
                }
                PosTag::Noun | PosTag::Propn => {
                    if normalized.ends_with('s') && !normalized.ends_with("ss") {
                        tags.insert("Number", "Plur");
                    } else {
                        tags.insert("Number", "Sing");
                    }
                }
                PosTag::Verb | PosTag::Aux => {
                    if normalized.ends_with("ed")
                        || ["was", "were", "did", "had"].contains(&normalized)
                    {
                        tags.insert("Tense", "Past");
                    } else if normalized == "will" || normalized == "shall" {
                        tags.insert("Tense", "Fut");
                    } else {
                        tags.insert("Tense", "Pres");
                    }
                    if normalized.ends_with("ing") {
                        tags.insert("Aspect", "Prog");
                    }
                    if ["have", "has", "had"].contains(&normalized) {
                        tags.insert("Aspect", "Perf");
                    }
                    if normalized == "to" {
                        tags.insert("VerbForm", "Inf");
                    } else {
                        tags.insert("VerbForm", "Fin");
                    }
                }
                _ => {}
            }
            if ["not", "never", "no"].contains(&normalized) {
                tags.insert("Polarity", "Neg");
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
