use std::collections::{BTreeMap, BTreeSet};

use text_core::Result;
use text_core::{Sentence, TextSpan, Token};

use crate::local_models::{RawPrediction, SequenceLabeler};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Variants describing entity type.
pub enum EntityType {
    /// The person variant.
    Person,
    /// The organization variant.
    Organization,
    /// The location variant.
    Location,
    /// The product variant.
    Product,
    /// The date variant.
    Date,
    /// The amount variant.
    Amount,
    /// The law variant.
    Law,
    /// The work variant.
    Work,
    /// The event variant.
    Event,
    /// The misc variant.
    Misc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for entity mention span.
pub struct EntityMentionSpan {
    /// Text content for this value.
    pub text: String,
    /// The span value.
    pub span: TextSpan,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for named entity.
pub struct NamedEntity {
    /// Identifier for this value.
    pub id: String,
    /// The entity type value.
    pub entity_type: EntityType,
    /// The mention value.
    pub mention: EntityMentionSpan,
    /// The normalized value.
    pub normalized: String,
    /// The sentence index value.
    pub sentence_index: usize,
    /// The token start value.
    pub token_start: usize,
    /// The token end value.
    pub token_end: usize,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for canonical entity.
pub struct CanonicalEntity {
    /// Identifier for this value.
    pub id: String,
    /// The entity type value.
    pub entity_type: EntityType,
    /// The canonical name value.
    pub canonical_name: String,
    /// The aliases value.
    pub aliases: Vec<String>,
    /// The mentions value.
    pub mentions: Vec<NamedEntity>,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for entity linking options.
pub struct EntityLinkingOptions {
    /// The lowercase keys value.
    pub lowercase_keys: bool,
}

impl Default for EntityLinkingOptions {
    fn default() -> Self {
        Self {
            lowercase_keys: true,
        }
    }
}

/// Returns named entities from a model-backed BIO token labeler.
pub fn extract_named_entities_with_labeler(
    text: &str,
    sentences: &[Sentence],
    tokens: &[Token],
    labeler: &mut dyn SequenceLabeler,
) -> Result<Vec<NamedEntity>> {
    let predictions = labeler.label_text(text)?;
    Ok(named_entities_from_model_predictions(
        text,
        sentences,
        tokens,
        &predictions,
    ))
}

/// Returns named entities from token classification predictions.
///
/// Sub-word predictions are aggregated onto the caller's word `tokens` first: the
/// earliest sub-word prediction inside a word labels the whole word (the Hugging Face
/// `aggregation_strategy="first"` rule), so entity spans never cut through a word.
/// Words are then grouped into entities by their BIO labels; an outside word or a
/// word without any prediction ends the current entity.
pub fn named_entities_from_model_predictions(
    text: &str,
    sentences: &[Sentence],
    tokens: &[Token],
    predictions: &[RawPrediction],
) -> Vec<NamedEntity> {
    let mut pieces = predictions
        .iter()
        .filter_map(model_piece)
        .collect::<Vec<_>>();
    pieces.sort_by_key(|piece| (piece.byte_start, piece.byte_end));

    let mut entities = Vec::new();
    let mut current = Vec::<ModelEntityToken>::new();
    for token in tokens {
        let word = pieces
            .iter()
            .find(|piece| byte_ranges_overlap(piece.byte_start, piece.byte_end, token.span))
            .and_then(|first| {
                let (prefix, entity_type) = first.label?;
                Some(ModelEntityToken {
                    entity_type,
                    prefix,
                    byte_start: token.span.byte_start,
                    byte_end: token.span.byte_end,
                    score: first.score,
                })
            });
        let starts_new = match &word {
            None => true,
            Some(word) => {
                word.prefix == ModelEntityPrefix::Begin
                    || current
                        .last()
                        .is_some_and(|last| last.entity_type != word.entity_type)
            }
        };
        if starts_new && !current.is_empty() {
            if let Some(entity) =
                build_model_entity(entities.len(), text, sentences, tokens, &current)
            {
                entities.push(entity);
            }
            current.clear();
        }
        if let Some(word) = word {
            current.push(word);
        }
    }
    if !current.is_empty() {
        if let Some(entity) = build_model_entity(entities.len(), text, sentences, tokens, &current)
        {
            entities.push(entity);
        }
    }

    entities
}

/// Returns canonicalize entities.
pub fn canonicalize_entities(
    entities: &[NamedEntity],
    options: &EntityLinkingOptions,
) -> Vec<CanonicalEntity> {
    let mut grouped = BTreeMap::<(EntityType, String), Vec<NamedEntity>>::new();
    for entity in entities {
        let key = if options.lowercase_keys {
            entity.normalized.to_lowercase()
        } else {
            entity.normalized.clone()
        };
        grouped
            .entry((entity.entity_type, key))
            .or_default()
            .push(entity.clone());
    }
    grouped
        .into_iter()
        .enumerate()
        .map(|(index, ((entity_type, key), mentions))| CanonicalEntity {
            id: format!("entity-{index}"),
            entity_type,
            canonical_name: mentions
                .first()
                .map(|mention| mention.mention.text.clone())
                .unwrap_or(key.clone()),
            aliases: mentions
                .iter()
                .map(|mention| mention.mention.text.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            mentions,
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for coref mention.
pub struct CorefMention {
    /// Text content for this value.
    pub text: String,
    /// The sentence index value.
    pub sentence_index: usize,
    /// The token start value.
    pub token_start: usize,
    /// The token end value.
    pub token_end: usize,
    /// The entity type value.
    pub entity_type: Option<EntityType>,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// A coreference cluster. No producer fills this yet: the heuristic resolver was
/// removed (#186) and coreference requests fail until a model backend exists.
pub struct CorefCluster {
    /// Identifier for this value.
    pub id: String,
    /// The canonical text value.
    pub canonical_text: String,
    /// The entity type value.
    pub entity_type: Option<EntityType>,
    /// The mentions value.
    pub mentions: Vec<CorefMention>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing relation type.
pub enum RelationType {
    /// The action variant.
    Action,
    /// The attribution variant.
    Attribution,
    /// The temporal variant.
    Temporal,
    /// The causal variant.
    Causal,
    /// The location variant.
    Location,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for event argument.
pub struct EventArgument {
    /// The role value.
    pub role: String,
    /// Text content for this value.
    pub text: String,
    /// Source span for this argument.
    pub span: TextSpan,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// A relation triple. No producer fills this yet: the heuristic extractor was
/// removed (#186) and relation requests fail until a model backend exists.
pub struct RelationTriple {
    /// Source sentence containing this relation.
    pub sentence_index: usize,
    /// The subject value.
    pub subject: String,
    /// Source span for the subject.
    pub subject_span: TextSpan,
    /// The relation value.
    pub relation: String,
    /// Source span for the relation predicate.
    pub relation_span: TextSpan,
    /// The object value.
    pub object: String,
    /// Source span for the object.
    pub object_span: TextSpan,
    /// The relation type value.
    pub relation_type: RelationType,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// An extracted event. No producer fills this yet: the heuristic extractor was
/// removed (#186) and event requests fail until a model backend exists.
pub struct ExtractedEvent {
    /// The sentence index value.
    pub sentence_index: usize,
    /// The predicate value.
    pub predicate: String,
    /// Source span for the predicate.
    pub predicate_span: TextSpan,
    /// The lemma value.
    pub lemma: String,
    /// The relation type value.
    pub relation_type: RelationType,
    /// The arguments value.
    pub arguments: Vec<EventArgument>,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelEntityPrefix {
    Begin,
    Inside,
}

#[derive(Debug, Clone, PartialEq)]
struct ModelEntityToken {
    entity_type: EntityType,
    prefix: ModelEntityPrefix,
    byte_start: usize,
    byte_end: usize,
    score: f32,
}

/// One sub-word prediction; `label` is `None` for outside (`O`) or unknown labels.
struct ModelPiece {
    label: Option<(ModelEntityPrefix, EntityType)>,
    byte_start: usize,
    byte_end: usize,
    score: f32,
}

fn model_piece(prediction: &RawPrediction) -> Option<ModelPiece> {
    let label = prediction.label.as_deref()?;
    let byte_start = prediction.attributes.get("byte_start")?.parse().ok()?;
    let byte_end = prediction.attributes.get("byte_end")?.parse().ok()?;
    if byte_start >= byte_end {
        return None;
    }
    Some(ModelPiece {
        label: parse_model_entity_label(label),
        byte_start,
        byte_end,
        score: prediction.score.unwrap_or(0.5),
    })
}

fn parse_model_entity_label(label: &str) -> Option<(ModelEntityPrefix, EntityType)> {
    let normalized = label
        .strip_prefix("LABEL_")
        .unwrap_or(label)
        .trim()
        .to_ascii_uppercase()
        .replace('_', "-");
    if normalized == "O" || normalized == "OUTSIDE" {
        return None;
    }
    let (prefix, kind) = if let Some(kind) = normalized.strip_prefix("B-") {
        (ModelEntityPrefix::Begin, kind)
    } else if let Some(kind) = normalized.strip_prefix("I-") {
        (ModelEntityPrefix::Inside, kind)
    } else {
        (ModelEntityPrefix::Begin, normalized.as_str())
    };
    let entity_type = match kind {
        "PER" | "PERSON" => EntityType::Person,
        "ORG" | "ORGANIZATION" => EntityType::Organization,
        "LOC" | "LOCATION" | "GPE" => EntityType::Location,
        "PROD" | "PRODUCT" => EntityType::Product,
        "DATE" | "TIME" => EntityType::Date,
        "MONEY" | "PERCENT" | "QUANTITY" | "AMOUNT" => EntityType::Amount,
        "LAW" => EntityType::Law,
        "WORK" | "WORK-OF-ART" => EntityType::Work,
        "EVENT" => EntityType::Event,
        "MISC" | "MISCELLANEOUS" => EntityType::Misc,
        _ => return None,
    };
    Some((prefix, entity_type))
}

fn build_model_entity(
    index: usize,
    text: &str,
    sentences: &[Sentence],
    tokens: &[Token],
    parts: &[ModelEntityToken],
) -> Option<NamedEntity> {
    let first = parts.first()?;
    let last = parts.last()?;
    let span = TextSpan::from_byte_range(text, first.byte_start, last.byte_end).ok()?;
    let token_start = tokens
        .iter()
        .position(|token| byte_ranges_overlap(span.byte_start, span.byte_end, token.span))?;
    let token_end = tokens
        .iter()
        .rposition(|token| byte_ranges_overlap(span.byte_start, span.byte_end, token.span))?
        + 1;
    let mention_text = text[span.byte_start..span.byte_end].to_string();
    let sentence_index = sentences
        .iter()
        .position(|sentence| spans_overlap(sentence.span, span))
        .unwrap_or(0);
    let confidence = parts.iter().map(|part| part.score).sum::<f32>() / parts.len() as f32;
    Some(NamedEntity {
        id: format!("mention-model-{index}"),
        entity_type: first.entity_type,
        mention: EntityMentionSpan {
            text: mention_text.clone(),
            span,
        },
        normalized: mention_text.to_lowercase(),
        sentence_index,
        token_start,
        token_end,
        confidence,
    })
}

fn spans_overlap(left: TextSpan, right: TextSpan) -> bool {
    left.byte_start < right.byte_end && right.byte_start < left.byte_end
}

fn byte_ranges_overlap(byte_start: usize, byte_end: usize, span: TextSpan) -> bool {
    byte_start < span.byte_end && span.byte_start < byte_end
}
