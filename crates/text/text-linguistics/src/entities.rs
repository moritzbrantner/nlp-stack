use std::collections::{BTreeMap, BTreeSet};

use text_core::Result;
use text_core::{Sentence, TextSpan, Token, TokenKind};

use crate::local_models::{RawPrediction, SequenceLabeler};
use crate::syntax::sentence_token_ranges;
use crate::{DependencyRelation, DependencyTree, LemmatizationResult, PosAnnotation, PosTag};

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

/// Returns extract named entities.
pub fn extract_named_entities(
    text: &str,
    sentences: &[Sentence],
    tokens: &[Token],
    pos: &[PosAnnotation],
) -> Vec<NamedEntity> {
    let sentence_ranges = sentence_token_ranges(sentences, tokens);
    let mut entities = Vec::new();
    let mut next_id = 0_usize;

    for (sentence_index, (start, end)) in sentence_ranges.into_iter().enumerate() {
        let mut cursor = start;
        while cursor < end {
            let tag = pos
                .get(cursor)
                .map(|annotation| annotation.tag)
                .unwrap_or(PosTag::X);
            let token = &tokens[cursor];
            if token.text.starts_with('$') && token.text[1..].chars().any(|ch| ch.is_ascii_digit())
            {
                entities.push(build_entity(
                    next_id,
                    EntityType::Amount,
                    text,
                    sentence_index,
                    cursor..(cursor + 1),
                    tokens,
                    0.9,
                ));
                next_id += 1;
                cursor += 1;
                continue;
            }
            if is_date_token(token) {
                entities.push(build_entity(
                    next_id,
                    EntityType::Date,
                    text,
                    sentence_index,
                    cursor..(cursor + 1),
                    tokens,
                    0.85,
                ));
                next_id += 1;
                cursor += 1;
                continue;
            }
            if matches!(tag, PosTag::Propn) {
                let start_index = cursor;
                let mut end_index = cursor + 1;
                while end_index < end {
                    let next_tag = pos
                        .get(end_index)
                        .map(|annotation| annotation.tag)
                        .unwrap_or(PosTag::X);
                    if matches!(next_tag, PosTag::Propn | PosTag::Noun) {
                        end_index += 1;
                    } else {
                        break;
                    }
                }
                let entity_type = classify_capitalized_entity(&tokens[start_index..end_index]);
                entities.push(build_entity(
                    next_id,
                    entity_type,
                    text,
                    sentence_index,
                    start_index..end_index,
                    tokens,
                    0.7,
                ));
                next_id += 1;
                cursor = end_index;
                continue;
            }
            cursor += 1;
        }
    }

    entities
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
pub fn named_entities_from_model_predictions(
    text: &str,
    sentences: &[Sentence],
    tokens: &[Token],
    predictions: &[RawPrediction],
) -> Vec<NamedEntity> {
    let mut model_tokens = predictions
        .iter()
        .filter_map(model_entity_token)
        .collect::<Vec<_>>();
    model_tokens.sort_by_key(|token| (token.byte_start, token.byte_end));

    let mut entities = Vec::new();
    let mut current = Vec::<ModelEntityToken>::new();
    for token in model_tokens {
        let starts_new = token.prefix == ModelEntityPrefix::Begin
            || current
                .last()
                .map(|last| last.entity_type != token.entity_type)
                .unwrap_or(false);
        if starts_new && !current.is_empty() {
            if let Some(entity) =
                build_model_entity(entities.len(), text, sentences, tokens, &current)
            {
                entities.push(entity);
            }
            current.clear();
        }
        current.push(token);
    }
    if !current.is_empty() {
        if let Some(entity) = build_model_entity(entities.len(), text, sentences, tokens, &current)
        {
            entities.push(entity);
        }
    }

    entities
}

/// Returns model entities plus deterministic rule entities not covered by the model.
pub fn merge_model_and_heuristic_entities(
    mut model_entities: Vec<NamedEntity>,
    heuristic_entities: Vec<NamedEntity>,
) -> Vec<NamedEntity> {
    if model_entities.is_empty() {
        return heuristic_entities;
    }
    for entity in heuristic_entities {
        if !matches!(
            entity.entity_type,
            EntityType::Date | EntityType::Amount | EntityType::Law
        ) {
            continue;
        }
        if model_entities
            .iter()
            .any(|model_entity| spans_overlap(model_entity.mention.span, entity.mention.span))
        {
            continue;
        }
        model_entities.push(entity);
    }
    model_entities
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
/// Data type for coref cluster.
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Data type for coref resolver.
pub struct CorefResolver {
    /// The speaker aware value.
    pub speaker_aware: bool,
}

impl CorefResolver {
    /// Returns resolve.
    pub fn resolve(
        &self,
        tokens: &[Token],
        canonical_entities: &[CanonicalEntity],
    ) -> Vec<CorefCluster> {
        let mut clusters = canonical_entities
            .iter()
            .map(|entity| CorefCluster {
                id: entity.id.clone(),
                canonical_text: entity.canonical_name.clone(),
                entity_type: Some(entity.entity_type),
                mentions: entity
                    .mentions
                    .iter()
                    .map(|mention| CorefMention {
                        text: mention.mention.text.clone(),
                        sentence_index: mention.sentence_index,
                        token_start: mention.token_start,
                        token_end: mention.token_end,
                        entity_type: Some(mention.entity_type),
                        confidence: mention.confidence,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();

        for (index, token) in tokens.iter().enumerate() {
            if token.kind != TokenKind::Word {
                continue;
            }
            let normalized = token.normalized.as_str();
            let pronoun_type = match normalized {
                "he" | "him" | "his" | "she" | "her" | "hers" | "they" | "them" | "their" => {
                    Some(EntityType::Person)
                }
                "it" | "its" => Some(EntityType::Misc),
                _ => None,
            };
            let Some(entity_type) = pronoun_type else {
                continue;
            };
            if let Some(cluster) = clusters.iter_mut().rev().find(|cluster| {
                cluster.entity_type == Some(entity_type)
                    || cluster.entity_type == Some(EntityType::Organization)
            }) {
                cluster.mentions.push(CorefMention {
                    text: token.text.clone(),
                    sentence_index: 0,
                    token_start: index,
                    token_end: index + 1,
                    entity_type: Some(entity_type),
                    confidence: 0.55,
                });
            }
        }

        clusters
    }
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
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for relation triple.
pub struct RelationTriple {
    /// The subject value.
    pub subject: String,
    /// The relation value.
    pub relation: String,
    /// The object value.
    pub object: String,
    /// The relation type value.
    pub relation_type: RelationType,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for extracted event.
pub struct ExtractedEvent {
    /// The sentence index value.
    pub sentence_index: usize,
    /// The predicate value.
    pub predicate: String,
    /// The lemma value.
    pub lemma: String,
    /// The relation type value.
    pub relation_type: RelationType,
    /// The arguments value.
    pub arguments: Vec<EventArgument>,
    /// Confidence score for this value.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for event extractor.
pub struct EventExtractor;

impl EventExtractor {
    /// Returns extract.
    pub fn extract(
        &self,
        trees: &[DependencyTree],
        tokens: &[Token],
        lemmas: &LemmatizationResult,
    ) -> (Vec<ExtractedEvent>, Vec<RelationTriple>) {
        let mut events = Vec::new();
        let mut relations = Vec::new();
        for tree in trees {
            let Some(root_index) = tree.root_token_index else {
                continue;
            };
            let predicate = tokens[root_index].text.clone();
            let lemma = lemmas
                .lemmas
                .get(root_index)
                .map(|lemma| lemma.value.clone())
                .unwrap_or_else(|| tokens[root_index].normalized.clone());
            let subject = tree
                .edges
                .iter()
                .find(|edge| edge.relation == DependencyRelation::Nsubj)
                .map(|edge| tokens[edge.dependent_token_index].text.clone());
            let object = tree
                .edges
                .iter()
                .find(|edge| {
                    matches!(
                        edge.relation,
                        DependencyRelation::Obj | DependencyRelation::Obl
                    )
                })
                .map(|edge| tokens[edge.dependent_token_index].text.clone());
            let relation_type =
                if ["say", "announce", "report", "present"].contains(&lemma.as_str()) {
                    RelationType::Attribution
                } else if ["because", "cause"].contains(&lemma.as_str()) {
                    RelationType::Causal
                } else if ["visit", "go", "travel"].contains(&lemma.as_str()) {
                    RelationType::Location
                } else {
                    RelationType::Action
                };
            let mut arguments = Vec::new();
            if let Some(subject) = subject.clone() {
                arguments.push(EventArgument {
                    role: "subject".to_string(),
                    text: subject,
                    confidence: 0.75,
                });
            }
            if let Some(object) = object.clone() {
                arguments.push(EventArgument {
                    role: "object".to_string(),
                    text: object,
                    confidence: 0.7,
                });
            }
            if !arguments.is_empty() {
                events.push(ExtractedEvent {
                    sentence_index: tree.sentence_index,
                    predicate: predicate.clone(),
                    lemma: lemma.clone(),
                    relation_type,
                    arguments: arguments.clone(),
                    confidence: 0.7,
                });
            }
            if let (Some(subject), Some(object)) = (subject, object) {
                relations.push(RelationTriple {
                    subject,
                    relation: lemma,
                    object,
                    relation_type,
                    confidence: 0.7,
                });
            }
        }
        (events, relations)
    }
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

fn model_entity_token(prediction: &RawPrediction) -> Option<ModelEntityToken> {
    let label = prediction.label.as_deref()?;
    let (prefix, entity_type) = parse_model_entity_label(label)?;
    let byte_start = prediction.attributes.get("byte_start")?.parse().ok()?;
    let byte_end = prediction.attributes.get("byte_end")?.parse().ok()?;
    if byte_start >= byte_end {
        return None;
    }
    Some(ModelEntityToken {
        entity_type,
        prefix,
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
    if last.byte_end > text.len()
        || !text.is_char_boundary(first.byte_start)
        || !text.is_char_boundary(last.byte_end)
    {
        return None;
    }
    let token_start = tokens
        .iter()
        .position(|token| byte_ranges_overlap(first.byte_start, last.byte_end, token.span))?;
    let token_end = tokens
        .iter()
        .rposition(|token| byte_ranges_overlap(first.byte_start, last.byte_end, token.span))?
        + 1;
    let char_start = text[..first.byte_start].chars().count();
    let char_end = text[..last.byte_end].chars().count();
    let span = TextSpan {
        byte_start: first.byte_start,
        byte_end: last.byte_end,
        char_start,
        char_end,
    };
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

fn byte_ranges_overlap(byte_start: usize, byte_end: usize, span: TextSpan) -> bool {
    byte_start < span.byte_end && span.byte_start < byte_end
}

fn spans_overlap(left: TextSpan, right: TextSpan) -> bool {
    left.byte_start < right.byte_end && right.byte_start < left.byte_end
}

fn is_date_token(token: &Token) -> bool {
    let normalized = token.normalized.as_str();
    normalized.chars().any(|ch| ch.is_ascii_digit())
        || matches!(
            normalized,
            "january"
                | "february"
                | "march"
                | "april"
                | "may"
                | "june"
                | "july"
                | "august"
                | "september"
                | "october"
                | "november"
                | "december"
        )
}

fn classify_capitalized_entity(tokens: &[Token]) -> EntityType {
    let normalized = tokens
        .iter()
        .map(|token| token.normalized.as_str())
        .collect::<Vec<_>>();
    if normalized
        .iter()
        .any(|token| matches!(*token, "inc" | "corp" | "ltd" | "gmbh" | "ag"))
    {
        EntityType::Organization
    } else if normalized.iter().any(|token| {
        is_date_token(&Token {
            text: (*token).to_string(),
            normalized: (*token).to_string(),
            span: TextSpan {
                byte_start: 0,
                byte_end: token.len(),
                char_start: 0,
                char_end: token.chars().count(),
            },
            kind: TokenKind::Word,
        })
    }) {
        EntityType::Date
    } else if tokens.len() == 1 {
        EntityType::Person
    } else {
        EntityType::Organization
    }
}

fn build_entity(
    index: usize,
    entity_type: EntityType,
    text: &str,
    sentence_index: usize,
    token_range: std::ops::Range<usize>,
    tokens: &[Token],
    confidence: f32,
) -> NamedEntity {
    let token_start = token_range.start;
    let token_end = token_range.end;
    let span = TextSpan {
        byte_start: tokens[token_start].span.byte_start,
        byte_end: tokens[token_end - 1].span.byte_end,
        char_start: tokens[token_start].span.char_start,
        char_end: tokens[token_end - 1].span.char_end,
    };
    let mention_text = text[span.byte_start..span.byte_end].to_string();
    NamedEntity {
        id: format!("mention-{index}"),
        entity_type,
        mention: EntityMentionSpan {
            text: mention_text.clone(),
            span,
        },
        normalized: mention_text.to_lowercase(),
        sentence_index,
        token_start,
        token_end,
        confidence,
    }
}
