use text_core::{Sentence, TextSpan, Token};

use crate::{PosAnnotation, PosTag};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing chunk kind.
pub enum ChunkKind {
    /// The noun phrase variant.
    NounPhrase,
    /// The verb phrase variant.
    VerbPhrase,
    /// The prep phrase variant.
    PrepPhrase,
    /// The adjective phrase variant.
    AdjectivePhrase,
    /// The adverb phrase variant.
    AdverbPhrase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for phrase chunk.
pub struct PhraseChunk {
    /// The kind value.
    pub kind: ChunkKind,
    /// The sentence index value.
    pub sentence_index: usize,
    /// The token start value.
    pub token_start: usize,
    /// The token end value.
    pub token_end: usize,
    /// The head token index value.
    pub head_token_index: usize,
    /// Text content for this value.
    pub text: String,
    /// The span value.
    pub span: TextSpan,
}

/// Returns chunk phrases.
pub fn chunk_phrases(
    text: &str,
    sentences: &[Sentence],
    tokens: &[Token],
    pos: &[PosAnnotation],
) -> Vec<PhraseChunk> {
    let sentence_ranges = sentence_token_ranges(sentences, tokens);
    let mut chunks = Vec::new();
    for (sentence_index, token_range) in sentence_ranges.into_iter().enumerate() {
        let indices = (token_range.0..token_range.1).collect::<Vec<_>>();
        let mut cursor = 0;
        while cursor < indices.len() {
            let token_index = indices[cursor];
            let tag = pos
                .get(token_index)
                .map(|annotation| annotation.tag)
                .unwrap_or(PosTag::X);

            let maybe_chunk = if matches!(
                tag,
                PosTag::Det | PosTag::Adj | PosTag::Noun | PosTag::Propn | PosTag::Pron
            ) {
                let mut end = cursor;
                while end + 1 < indices.len() {
                    let next_tag = pos
                        .get(indices[end + 1])
                        .map(|annotation| annotation.tag)
                        .unwrap_or(PosTag::X);
                    if matches!(
                        next_tag,
                        PosTag::Adj | PosTag::Noun | PosTag::Propn | PosTag::Pron
                    ) {
                        end += 1;
                    } else {
                        break;
                    }
                }
                let start_index = indices[cursor];
                let end_index = indices[end];
                Some((ChunkKind::NounPhrase, start_index, end_index, end_index))
            } else if matches!(tag, PosTag::Aux | PosTag::Verb | PosTag::Adv) {
                let mut end = cursor;
                while end + 1 < indices.len() {
                    let next_tag = pos
                        .get(indices[end + 1])
                        .map(|annotation| annotation.tag)
                        .unwrap_or(PosTag::X);
                    if matches!(
                        next_tag,
                        PosTag::Aux | PosTag::Verb | PosTag::Adv | PosTag::Part
                    ) {
                        end += 1;
                    } else {
                        break;
                    }
                }
                let start_index = indices[cursor];
                let end_index = indices[end];
                let head = indices[cursor..=end]
                    .iter()
                    .copied()
                    .find(|index| matches!(pos[*index].tag, PosTag::Verb | PosTag::Aux))
                    .unwrap_or(start_index);
                Some((ChunkKind::VerbPhrase, start_index, end_index, head))
            } else if tag == PosTag::Adp {
                let start_index = token_index;
                let mut end = cursor;
                while end + 1 < indices.len() {
                    let next_tag = pos
                        .get(indices[end + 1])
                        .map(|annotation| annotation.tag)
                        .unwrap_or(PosTag::X);
                    if matches!(
                        next_tag,
                        PosTag::Det | PosTag::Adj | PosTag::Noun | PosTag::Propn | PosTag::Pron
                    ) {
                        end += 1;
                    } else {
                        break;
                    }
                }
                let end_index = indices[end];
                Some((ChunkKind::PrepPhrase, start_index, end_index, token_index))
            } else {
                None
            };

            if let Some((kind, start_index, end_index, head_index)) = maybe_chunk {
                let span = TextSpan::from_byte_range(
                    text,
                    tokens[start_index].span.byte_start,
                    tokens[end_index].span.byte_end,
                )
                .expect("token spans define valid UTF-8 byte boundaries");
                chunks.push(PhraseChunk {
                    kind,
                    sentence_index,
                    token_start: start_index,
                    token_end: end_index + 1,
                    head_token_index: head_index,
                    text: text[span.byte_start..span.byte_end].to_string(),
                    span,
                });
                cursor = indices
                    .iter()
                    .position(|index| *index == end_index)
                    .map(|position| position + 1)
                    .unwrap_or(indices.len());
            } else {
                cursor += 1;
            }
        }
    }
    chunks
}

macro_rules! ud_relations {
    ($($variant:ident => $label:literal,)*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        /// A Universal Dependencies v2 relation, without its language-specific subtype.
        pub enum DependencyRelation {
            $(#[doc = concat!("UD `", $label, "`.")] $variant,)*
        }

        impl DependencyRelation {
            /// Every UD v2 universal relation.
            pub const ALL: &'static [Self] = &[$(Self::$variant,)*];

            /// The UD label (`nsubj`).
            pub fn as_ud(self) -> &'static str {
                match self {
                    $(Self::$variant => $label,)*
                }
            }
        }
    };
}

ud_relations! {
    Acl => "acl",
    Advcl => "advcl",
    Advmod => "advmod",
    Amod => "amod",
    Appos => "appos",
    Aux => "aux",
    Case => "case",
    Cc => "cc",
    Ccomp => "ccomp",
    Clf => "clf",
    Compound => "compound",
    Conj => "conj",
    Cop => "cop",
    Csubj => "csubj",
    Dep => "dep",
    Det => "det",
    Discourse => "discourse",
    Dislocated => "dislocated",
    Expl => "expl",
    Fixed => "fixed",
    Flat => "flat",
    Goeswith => "goeswith",
    Iobj => "iobj",
    List => "list",
    Mark => "mark",
    Nmod => "nmod",
    Nsubj => "nsubj",
    Nummod => "nummod",
    Obj => "obj",
    Obl => "obl",
    Orphan => "orphan",
    Parataxis => "parataxis",
    Punct => "punct",
    Reparandum => "reparandum",
    Root => "root",
    Vocative => "vocative",
    Xcomp => "xcomp",
}

impl DependencyRelation {
    /// Parses a UD universal relation label (`nsubj`, not `nsubj:pass`).
    pub fn from_ud(label: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|relation| relation.as_ud() == label)
    }

    /// Splits a full UD DEPREL (`nsubj:pass`) into relation and subtype. Unknown
    /// relations become [`DependencyRelation::Dep`].
    pub fn parse_ud_deprel(label: &str) -> (Self, Option<String>) {
        let (base, subtype) = match label.split_once(':') {
            Some((base, subtype)) => (base, Some(subtype.to_string())),
            None => (label, None),
        };
        (Self::from_ud(base).unwrap_or(Self::Dep), subtype)
    }

    /// Formats the relation with an optional subtype as a UD DEPREL (`nsubj:pass`).
    pub fn to_ud_deprel(self, subtype: Option<&str>) -> String {
        match subtype {
            Some(subtype) => format!("{}:{subtype}", self.as_ud()),
            None => self.as_ud().to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for dependency node.
pub struct DependencyNode {
    /// The token index value.
    pub token_index: usize,
    /// The head token index value.
    pub head_token_index: Option<usize>,
    /// The relation value.
    pub relation: DependencyRelation,
    /// Language-specific UD subtype (`pass` in `nsubj:pass`).
    pub subtype: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for dependency edge.
pub struct DependencyEdge {
    /// The head token index value.
    pub head_token_index: usize,
    /// The dependent token index value.
    pub dependent_token_index: usize,
    /// The relation value.
    pub relation: DependencyRelation,
    /// Language-specific UD subtype (`pass` in `nsubj:pass`).
    pub subtype: Option<String>,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for dependency tree.
pub struct DependencyTree {
    /// The sentence index value.
    pub sentence_index: usize,
    /// The root token index value.
    pub root_token_index: Option<usize>,
    /// The nodes value.
    pub nodes: Vec<DependencyNode>,
    /// The edges value.
    pub edges: Vec<DependencyEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for dependency parser.
pub struct DependencyParser;

impl DependencyParser {
    /// Parses parse document.
    pub fn parse_document(
        &self,
        sentences: &[Sentence],
        tokens: &[Token],
        pos: &[PosAnnotation],
    ) -> Vec<DependencyTree> {
        sentence_token_ranges(sentences, tokens)
            .into_iter()
            .enumerate()
            .map(|(sentence_index, (start, end))| {
                self.parse_sentence(sentence_index, tokens, pos, start, end)
            })
            .collect()
    }

    fn parse_sentence(
        &self,
        sentence_index: usize,
        tokens: &[Token],
        pos: &[PosAnnotation],
        start: usize,
        end: usize,
    ) -> DependencyTree {
        let indices = (start..end).collect::<Vec<_>>();
        let root_token_index = indices
            .iter()
            .copied()
            .find(|index| matches!(pos[*index].tag, PosTag::Verb | PosTag::Aux))
            .or_else(|| {
                indices
                    .iter()
                    .copied()
                    .find(|index| matches!(pos[*index].tag, PosTag::Noun | PosTag::Propn))
            })
            .or_else(|| indices.first().copied());

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        for token_index in indices.iter().copied() {
            let tag = pos
                .get(token_index)
                .map(|annotation| annotation.tag)
                .unwrap_or(PosTag::X);
            let (head, relation, confidence) = if Some(token_index) == root_token_index {
                (None, DependencyRelation::Root, 1.0)
            } else if matches!(tag, PosTag::Det) {
                let head = nearest_following(tokens, pos, token_index, end, |tag| {
                    matches!(tag, PosTag::Noun | PosTag::Propn | PosTag::Pron)
                })
                .or(root_token_index);
                (head, DependencyRelation::Det, 0.8)
            } else if matches!(tag, PosTag::Adj) {
                let head = nearest_following(tokens, pos, token_index, end, |tag| {
                    matches!(tag, PosTag::Noun | PosTag::Propn)
                })
                .or(root_token_index);
                (head, DependencyRelation::Amod, 0.7)
            } else if matches!(tag, PosTag::Adp) {
                let head = nearest_following(tokens, pos, token_index, end, |tag| {
                    matches!(tag, PosTag::Noun | PosTag::Propn | PosTag::Pron)
                })
                .or(root_token_index);
                (head, DependencyRelation::Case, 0.7)
            } else if matches!(tag, PosTag::Aux) {
                (root_token_index, DependencyRelation::Aux, 0.85)
            } else if matches!(tag, PosTag::Adv) {
                (root_token_index, DependencyRelation::Advmod, 0.7)
            } else if matches!(tag, PosTag::Cconj | PosTag::Sconj) {
                (root_token_index, DependencyRelation::Cc, 0.6)
            } else if matches!(tag, PosTag::Noun | PosTag::Propn | PosTag::Pron) {
                if let Some(root) = root_token_index {
                    if token_index < root {
                        (Some(root), DependencyRelation::Nsubj, 0.75)
                    } else {
                        (Some(root), DependencyRelation::Obj, 0.7)
                    }
                } else {
                    (None, DependencyRelation::Dep, 0.4)
                }
            } else {
                (root_token_index, DependencyRelation::Dep, 0.4)
            };
            nodes.push(DependencyNode {
                token_index,
                head_token_index: head,
                relation,
                subtype: None,
            });
            if let Some(head) = head {
                edges.push(DependencyEdge {
                    head_token_index: head,
                    dependent_token_index: token_index,
                    relation,
                    subtype: None,
                    confidence,
                });
            }
        }

        DependencyTree {
            sentence_index,
            root_token_index,
            nodes,
            edges,
        }
    }
}

pub(crate) fn sentence_token_ranges(
    sentences: &[Sentence],
    tokens: &[Token],
) -> Vec<(usize, usize)> {
    sentences
        .iter()
        .map(|sentence| {
            let start = tokens
                .iter()
                .position(|token| token.span.byte_start >= sentence.span.byte_start)
                .unwrap_or(tokens.len());
            let end = tokens[start..]
                .iter()
                .position(|token| token.span.byte_end > sentence.span.byte_end)
                .map(|offset| start + offset)
                .unwrap_or(tokens.len());
            (start, end.max(start))
        })
        .collect()
}

fn nearest_following(
    _tokens: &[Token],
    pos: &[PosAnnotation],
    token_index: usize,
    sentence_end: usize,
    matcher: impl Fn(PosTag) -> bool,
) -> Option<usize> {
    ((token_index + 1)..sentence_end).find(|index| {
        pos.get(*index)
            .map(|annotation| matcher(annotation.tag))
            .unwrap_or(false)
    })
}
