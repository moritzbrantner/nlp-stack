use std::collections::BTreeMap;

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

#[derive(Debug, Clone, PartialEq)]
/// Data type for lemmatization result.
pub struct LemmatizationResult {
    /// Language tag for this value.
    pub language: Option<String>,
    /// The lemmas value.
    pub lemmas: Vec<Lemma>,
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

impl serde::Serialize for PosTag {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_ud())
    }
}

impl serde::Serialize for MorphTagSet {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.features.serialize(serializer)
    }
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
