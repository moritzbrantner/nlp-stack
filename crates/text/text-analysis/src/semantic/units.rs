use text_core::{split_paragraphs, split_sentence_spans, TextDocument, TextSpan};

use super::corpus::SemanticCorpusItem;
use super::{ConversationTurn, SemanticAnalysisOptions, SemanticUnit, SemanticUnitKind};

pub(super) fn document_units(
    document: &TextDocument<'_>,
    options: &SemanticAnalysisOptions,
) -> Vec<SemanticUnit> {
    let sentences = split_sentence_spans(document.text, &options.processing);
    let paragraphs = split_paragraphs(document.text);
    let document_unit_id = format!("{}:document", document.id);

    let mut units = Vec::new();
    for (sentence_index, sentence) in sentences.into_iter().enumerate() {
        let parent_id = paragraphs
            .iter()
            .enumerate()
            .find(|(_, paragraph)| contains_span(paragraph.span, sentence.span))
            .map(|(paragraph_index, _)| format!("{}:paragraph:{paragraph_index}", document.id));
        units.push(SemanticUnit {
            id: format!("{}:sentence:{sentence_index}", document.id),
            source_id: document.id.to_string(),
            kind: SemanticUnitKind::Sentence,
            parent_id,
            sequence_index: sentence_index,
            span: sentence.span,
            speaker: None,
            start_seconds: None,
            end_seconds: None,
            text: sentence.text,
            embedding: Vec::new(),
        });
    }

    for (paragraph_index, paragraph) in paragraphs.into_iter().enumerate() {
        units.push(SemanticUnit {
            id: format!("{}:paragraph:{paragraph_index}", document.id),
            source_id: document.id.to_string(),
            kind: SemanticUnitKind::Paragraph,
            parent_id: Some(document_unit_id.clone()),
            sequence_index: paragraph_index,
            span: paragraph.span,
            speaker: None,
            start_seconds: None,
            end_seconds: None,
            text: paragraph.text,
            embedding: Vec::new(),
        });
    }

    units.push(SemanticUnit {
        id: document_unit_id,
        source_id: document.id.to_string(),
        kind: SemanticUnitKind::Document,
        parent_id: None,
        sequence_index: 0,
        span: span_for_text(document.text),
        speaker: None,
        start_seconds: None,
        end_seconds: None,
        text: document.text.to_string(),
        embedding: Vec::new(),
    });

    units
}

pub(super) fn corpus_units(
    items: &[SemanticCorpusItem<'_>],
    options: &SemanticAnalysisOptions,
) -> Vec<SemanticUnit> {
    let mut units = Vec::new();
    let mut sentence_sequence_index = 0usize;

    for item in items {
        let document = TextDocument::new(item.id, item.text);
        let author = item.author.map(str::trim).map(ToString::to_string);
        let mut item_units = document_units(&document, options);
        for unit in &mut item_units {
            unit.speaker = author.clone();
            if unit.kind == SemanticUnitKind::Sentence {
                unit.sequence_index = sentence_sequence_index;
                sentence_sequence_index += 1;
            }
        }
        units.extend(item_units);
    }

    units
}

pub(super) fn conversation_units(
    turns: &[ConversationTurn<'_>],
    options: &SemanticAnalysisOptions,
) -> Vec<SemanticUnit> {
    let mut units = Vec::new();
    for (turn_index, turn) in turns.iter().enumerate() {
        let turn_unit_id = format!("{}:turn", turn.id);
        units.push(SemanticUnit {
            id: turn_unit_id.clone(),
            source_id: turn.id.to_string(),
            kind: SemanticUnitKind::SpeakerTurn,
            parent_id: None,
            sequence_index: turn_index,
            span: span_for_text(turn.text),
            speaker: turn.speaker.map(ToString::to_string),
            start_seconds: turn.start_seconds,
            end_seconds: turn.end_seconds,
            text: turn.text.to_string(),
            embedding: Vec::new(),
        });

        for (sentence_index, sentence) in split_sentence_spans(turn.text, &options.processing)
            .into_iter()
            .enumerate()
        {
            units.push(SemanticUnit {
                id: format!("{}:sentence:{sentence_index}", turn.id),
                source_id: turn.id.to_string(),
                kind: SemanticUnitKind::Sentence,
                parent_id: Some(turn_unit_id.clone()),
                sequence_index: sentence_index,
                span: sentence.span,
                speaker: turn.speaker.map(ToString::to_string),
                start_seconds: turn.start_seconds,
                end_seconds: turn.end_seconds,
                text: sentence.text,
                embedding: Vec::new(),
            });
        }
    }
    units
}

fn contains_span(outer: TextSpan, inner: TextSpan) -> bool {
    outer.byte_start <= inner.byte_start && outer.byte_end >= inner.byte_end
}

fn span_for_text(text: &str) -> TextSpan {
    TextSpan {
        byte_start: 0,
        byte_end: text.len(),
        char_start: 0,
        char_end: text.chars().count(),
    }
}
