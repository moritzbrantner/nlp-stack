#![doc = include_str!("../README.md")]

pub mod contracts;
pub mod operations;
pub mod surface;

pub use contracts::{
    AsTextSegmentContract, IntoTextDocumentContract, TextAnnotationSpan, TextDocumentContract,
    TextProvenance, TextSegmentContract, TextSourceRef, TimebaseContract, TimestampContract,
};
pub use media_core::{AnalysisEvent, DetectError, Result, Timebase, Timestamp};

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Copy)]
/// Borrowed text segment at the boundary between text producers and analyzers.
pub struct TextSegment<'a> {
    /// Stable segment index within its source.
    pub segment_index: u64,
    /// Optional media timestamp.
    pub timestamp: Option<Timestamp>,
    /// UTF-8 text content.
    pub text: &'a str,
    /// Optional language tag.
    pub language: Option<&'a str>,
    /// Whether the producer considers this segment final.
    pub is_final: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Owned text segment for source and pipeline boundaries.
pub struct OwnedTextSegment {
    /// Stable segment index within its source.
    pub segment_index: u64,
    /// Optional media timestamp.
    pub timestamp: Option<Timestamp>,
    /// UTF-8 text content.
    pub text: String,
    /// Optional language tag.
    pub language: Option<String>,
    /// Whether the producer considers this segment final.
    pub is_final: bool,
}

impl OwnedTextSegment {
    /// Creates a final segment without optional timing or language metadata.
    pub fn new(segment_index: u64, text: impl Into<String>) -> Self {
        Self {
            segment_index,
            timestamp: None,
            text: text.into(),
            language: None,
            is_final: true,
        }
    }

    /// Adds a media timestamp.
    pub fn timestamp(mut self, timestamp: Timestamp) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Adds a language tag.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets whether this segment is final.
    pub fn finality(mut self, is_final: bool) -> Self {
        self.is_final = is_final;
        self
    }

    /// Borrows this value as a segment.
    pub fn as_segment(&self) -> TextSegment<'_> {
        TextSegment {
            segment_index: self.segment_index,
            timestamp: self.timestamp,
            text: &self.text,
            language: self.language.as_deref(),
            is_final: self.is_final,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
/// Aggregate result returned when a text pipeline finishes.
pub struct TextAnalysisResult {
    /// Events emitted during the pipeline run.
    pub events: Vec<AnalysisEvent>,
    /// Number of segments processed.
    pub segments_processed: u64,
}

#[derive(Debug, Clone, PartialEq)]
/// Incremental result returned after one text segment.
pub struct TextAnalysis {
    /// The processed segment index.
    pub segment_index: u64,
    /// Events emitted for this segment.
    pub events: Vec<AnalysisEvent>,
    /// Number of segments processed in the current run.
    pub segments_processed: u64,
}

/// Analyzer interface for text segments.
pub trait TextAnalyzer {
    /// Returns the analyzer name.
    fn name(&self) -> &str;

    /// Processes one borrowed segment.
    fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>>;

    /// Finishes a run and optionally emits delayed events.
    fn finish(&mut self, _last_segment_index: Option<u64>) -> Result<Vec<AnalysisEvent>> {
        Ok(Vec::new())
    }
}

/// Stateful composition of text analyzers.
pub struct TextPipeline {
    analyzers: Vec<Box<dyn TextAnalyzer>>,
    state: TextPipelineState,
}

#[derive(Debug, Default, Clone)]
struct TextPipelineState {
    events: Vec<AnalysisEvent>,
    last_segment_index: Option<u64>,
    segments_processed: u64,
    finished: bool,
}

impl TextPipeline {
    /// Creates a pipeline builder.
    pub fn builder() -> TextPipelineBuilder {
        TextPipelineBuilder::default()
    }

    /// Processes one owned segment.
    pub fn process_segment(&mut self, segment: OwnedTextSegment) -> Result<TextAnalysis> {
        if self.state.finished {
            return Err(DetectError::InvalidArgument(
                "cannot process text segments after finish_analysis; call reset first".to_string(),
            ));
        }
        let segment_ref = segment.as_segment();
        self.state.last_segment_index = Some(segment_ref.segment_index);

        let mut events = Vec::new();
        for analyzer in &mut self.analyzers {
            let mut new_events = analyzer.process_segment(&segment_ref)?;
            events.append(&mut new_events);
        }
        self.state.events.extend(events.iter().cloned());
        self.state.segments_processed += 1;

        Ok(TextAnalysis {
            segment_index: segment_ref.segment_index,
            events,
            segments_processed: self.state.segments_processed,
        })
    }

    /// Finishes the current analysis run.
    pub fn finish_analysis(&mut self) -> Result<TextAnalysisResult> {
        if !self.state.finished {
            let mut events = Vec::new();
            for analyzer in &mut self.analyzers {
                let mut new_events = analyzer.finish(self.state.last_segment_index)?;
                events.append(&mut new_events);
            }
            self.state.events.extend(events);
            self.state.finished = true;
        }
        Ok(TextAnalysisResult {
            events: self.state.events.clone(),
            segments_processed: self.state.segments_processed,
        })
    }

    /// Resets the pipeline for another run.
    pub fn reset(&mut self) {
        self.state = TextPipelineState::default();
    }

    /// Returns all events emitted in the current run.
    pub fn events(&self) -> &[AnalysisEvent] {
        &self.state.events
    }

    /// Returns the number of processed segments.
    pub fn segments_processed(&self) -> u64 {
        self.state.segments_processed
    }
}

#[derive(Default)]
/// Builder for a text analysis pipeline.
pub struct TextPipelineBuilder {
    analyzers: Vec<Box<dyn TextAnalyzer>>,
}

impl TextPipelineBuilder {
    /// Adds an analyzer.
    pub fn analyzer<A: TextAnalyzer + 'static>(mut self, analyzer: A) -> Self {
        self.analyzers.push(Box::new(analyzer));
        self
    }

    /// Builds a non-empty pipeline.
    pub fn build(self) -> Result<TextPipeline> {
        if self.analyzers.is_empty() {
            return Err(DetectError::InvalidArgument(
                "at least one text analyzer is required".to_string(),
            ));
        }
        Ok(TextPipeline {
            analyzers: self.analyzers,
            state: TextPipelineState::default(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Borrowed text document used as the lightweight boundary between text crates.
pub struct TextDocument<'a> {
    /// Stable caller-supplied document id.
    pub id: &'a str,
    /// UTF-8 document body.
    pub text: &'a str,
    /// Optional BCP-47-style language hint such as `en`.
    pub language: Option<&'a str>,
    /// Optional media timestamp when this document came from a timed segment.
    pub timestamp: Option<Timestamp>,
}

impl<'a> TextDocument<'a> {
    /// Creates a new value.
    pub fn new(id: &'a str, text: &'a str) -> Self {
        Self {
            id,
            text,
            language: None,
            timestamp: None,
        }
    }

    /// Builds this value from segment identifier.
    pub fn from_segment_id(id: &'a str, segment: &TextSegment<'a>) -> Self {
        Self {
            id,
            text: segment.text,
            language: segment.language,
            timestamp: segment.timestamp,
        }
    }

    /// Builds this value from stream segment.
    pub fn from_stream_segment(stream_id: &str, segment: &TextSegment<'_>) -> OwnedTextDocument {
        OwnedTextDocument::from_stream_segment(stream_id, segment)
    }

    /// Builds a document using `stream_id` directly as the document id.
    ///
    /// Prefer [`TextDocument::from_segment_id`] when the id has already been
    /// chosen, or [`TextDocument::from_stream_segment`] when converting a
    /// stream segment into the canonical `stream_id:segment_index` id.
    pub fn from_segment(stream_id: &'a str, segment: &TextSegment<'a>) -> Self {
        Self {
            id: stream_id,
            text: segment.text,
            language: segment.language,
            timestamp: segment.timestamp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Owned text document for storage, serialization, and cross-thread workflows.
pub struct OwnedTextDocument {
    /// Stable caller-supplied document id.
    pub id: String,
    /// UTF-8 document body.
    pub text: String,
    /// Optional BCP-47-style language hint such as `en`.
    pub language: Option<String>,
    /// Optional media timestamp when this document came from a timed segment.
    pub timestamp: Option<Timestamp>,
}

impl OwnedTextDocument {
    /// Creates a new value.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            language: None,
            timestamp: None,
        }
    }

    /// Returns language.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Returns timestamp.
    pub fn timestamp(mut self, timestamp: Timestamp) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Builds this value from segment identifier.
    pub fn from_segment_id(id: impl Into<String>, segment: &OwnedTextSegment) -> Self {
        let segment = segment.as_segment();
        Self {
            id: id.into(),
            text: segment.text.to_string(),
            language: segment.language.map(ToString::to_string),
            timestamp: segment.timestamp,
        }
    }

    /// Builds this value from stream segment.
    pub fn from_stream_segment(stream_id: &str, segment: &TextSegment<'_>) -> Self {
        Self {
            id: segment_document_id(stream_id, segment.segment_index),
            text: segment.text.to_string(),
            language: segment.language.map(ToString::to_string),
            timestamp: segment.timestamp,
        }
    }

    /// Builds this value from owned stream segment.
    pub fn from_owned_stream_segment(stream_id: &str, segment: &OwnedTextSegment) -> Self {
        Self::from_stream_segment(stream_id, &segment.as_segment())
    }

    /// Builds a document using the supplied `stream_id` directly as the id.
    ///
    /// Prefer [`OwnedTextDocument::from_segment_id`] when the id has already
    /// been chosen, or [`OwnedTextDocument::from_owned_stream_segment`] for the
    /// canonical `stream_id:segment_index` id.
    pub fn from_segment(stream_id: impl Into<String>, segment: &OwnedTextSegment) -> Self {
        let segment = segment.as_segment();
        Self {
            id: stream_id.into(),
            text: segment.text.to_string(),
            language: segment.language.map(ToString::to_string),
            timestamp: segment.timestamp,
        }
    }

    /// Borrows this value as a document.
    pub fn as_document(&self) -> TextDocument<'_> {
        TextDocument {
            id: &self.id,
            text: &self.text,
            language: self.language.as_deref(),
            timestamp: self.timestamp,
        }
    }
}

/// Returns segment document identifier.
pub fn segment_document_id(stream_id: &str, segment_index: u64) -> String {
    format!("{stream_id}:{segment_index}")
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// Data type for text stats.
pub struct TextStats {
    /// The bytes value.
    pub bytes: usize,
    /// The chars value.
    pub chars: usize,
    /// The words value.
    pub words: usize,
    /// The lines value.
    pub lines: usize,
    /// The sentences value.
    pub sentences: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Half-open byte and character span into a UTF-8 text buffer.
pub struct TextSpan {
    /// Inclusive byte offset.
    pub byte_start: usize,
    /// Exclusive byte offset.
    pub byte_end: usize,
    /// Inclusive Unicode scalar index.
    pub char_start: usize,
    /// Exclusive Unicode scalar index.
    pub char_end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Data type for annotation identifier.
pub struct AnnotationId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// Confidence score normalized into the inclusive range `0.0..=1.0`.
pub struct AnnotationConfidence(f32);

impl AnnotationConfidence {
    /// Creates a clamped confidence score; non-finite inputs become `0.0`.
    pub fn new(value: f32) -> Self {
        let value = if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self(value)
    }

    /// Returns the normalized confidence value.
    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for AnnotationConfidence {
    fn default() -> Self {
        Self(1.0)
    }
}

impl From<f32> for AnnotationConfidence {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Variants describing annotation provenance.
pub enum AnnotationProvenance {
    /// The heuristic variant.
    Heuristic,
    /// The tokenizer variant.
    Tokenizer,
    /// The ONNX variant.
    Onnx,
    /// The candle variant.
    Candle,
    /// The cuda oxide variant.
    CudaOxide,
    /// The external variant.
    External,
    /// The derived variant.
    Derived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for text span ref.
pub struct TextSpanRef {
    /// Identifier for this value.
    pub id: AnnotationId,
    /// The span value.
    pub span: TextSpan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for text boundary options.
pub struct TextBoundaryOptions {
    /// The lowercase value.
    pub lowercase: bool,
    /// The normalize unicode value.
    pub normalize_unicode: bool,
    /// The include numbers value.
    pub include_numbers: bool,
    /// The include punctuation value.
    pub include_punctuation: bool,
    /// The min chars value.
    pub min_chars: usize,
}

impl Default for TextBoundaryOptions {
    fn default() -> Self {
        Self {
            lowercase: true,
            normalize_unicode: true,
            include_numbers: true,
            include_punctuation: false,
            min_chars: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for word segment.
pub struct WordSegment {
    /// Text content for this value.
    pub text: String,
    /// The normalized value.
    pub normalized: String,
    /// The span value.
    pub span: TextSpan,
    /// The kind value.
    pub kind: TokenKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for grapheme span.
pub struct GraphemeSpan {
    /// Text content for this value.
    pub text: String,
    /// The span value.
    pub span: TextSpan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for script profile.
pub struct ScriptProfile {
    /// The scripts value.
    pub scripts: BTreeMap<String, usize>,
    /// The digits value.
    pub digits: usize,
    /// The whitespace value.
    pub whitespace: usize,
    /// The punctuation value.
    pub punctuation: usize,
    /// The other value.
    pub other: usize,
    /// The dominant script value.
    pub dominant_script: Option<String>,
    /// The is mixed value.
    pub is_mixed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for token.
pub struct Token {
    /// Text content for this value.
    pub text: String,
    /// The normalized value.
    pub normalized: String,
    /// The span value.
    pub span: TextSpan,
    /// The kind value.
    pub kind: TokenKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Variants describing token kind.
pub enum TokenKind {
    /// The word variant.
    Word,
    /// The number variant.
    Number,
    /// The URL variant.
    Url,
    /// The email variant.
    Email,
    /// The mention variant.
    Mention,
    /// The hashtag variant.
    Hashtag,
    /// The punctuation variant.
    Punctuation,
    /// The other variant.
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for sentence.
pub struct Sentence {
    /// Text content for this value.
    pub text: String,
    /// The span value.
    pub span: TextSpan,
    /// The token count value.
    pub token_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for paragraph.
pub struct Paragraph {
    /// Text content for this value.
    pub text: String,
    /// The span value.
    pub span: TextSpan,
    /// The sentence count value.
    pub sentence_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for canonical token.
pub struct CanonicalToken {
    /// Identifier for this value.
    pub id: AnnotationId,
    /// The span value.
    pub span: TextSpanRef,
    /// Text content for this value.
    pub text: String,
    /// The normalized value.
    pub normalized: String,
    /// The kind value.
    pub kind: TokenKind,
    /// The sentence identifier value.
    pub sentence_id: AnnotationId,
    /// The paragraph identifier value.
    pub paragraph_id: Option<AnnotationId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for annotated sentence.
pub struct AnnotatedSentence {
    /// Identifier for this value.
    pub id: AnnotationId,
    /// The span value.
    pub span: TextSpanRef,
    /// Text content for this value.
    pub text: String,
    /// The token start value.
    pub token_start: usize,
    /// The token end value.
    pub token_end: usize,
    /// The token identifiers value.
    pub token_ids: Vec<AnnotationId>,
    /// The paragraph identifier value.
    pub paragraph_id: Option<AnnotationId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for annotated paragraph.
pub struct AnnotatedParagraph {
    /// Identifier for this value.
    pub id: AnnotationId,
    /// The span value.
    pub span: TextSpanRef,
    /// Text content for this value.
    pub text: String,
    /// The sentence start value.
    pub sentence_start: usize,
    /// The sentence end value.
    pub sentence_end: usize,
    /// The sentence identifiers value.
    pub sentence_ids: Vec<AnnotationId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for text annotation graph.
pub struct TextAnnotationGraph {
    /// Text content for this value.
    pub text: String,
    /// The provenance value.
    pub provenance: AnnotationProvenance,
    /// Confidence score for this value.
    pub confidence: AnnotationConfidence,
    /// The tokens value.
    pub tokens: Vec<CanonicalToken>,
    /// The sentences value.
    pub sentences: Vec<AnnotatedSentence>,
    /// The paragraphs value.
    pub paragraphs: Vec<AnnotatedParagraph>,
}

impl TextAnnotationGraph {
    /// Returns token.
    pub fn token(&self, id: AnnotationId) -> Option<&CanonicalToken> {
        self.tokens.iter().find(|token| token.id == id)
    }

    /// Returns sentence.
    pub fn sentence(&self, id: AnnotationId) -> Option<&AnnotatedSentence> {
        self.sentences.iter().find(|sentence| sentence.id == id)
    }

    /// Returns paragraph.
    pub fn paragraph(&self, id: AnnotationId) -> Option<&AnnotatedParagraph> {
        self.paragraphs.iter().find(|paragraph| paragraph.id == id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for text processing options.
pub struct TextProcessingOptions {
    /// Language tag for this value.
    pub language: Option<String>,
    /// The lowercase value.
    pub lowercase: bool,
    /// The normalize unicode value.
    pub normalize_unicode: bool,
    /// The keep apostrophes value.
    pub keep_apostrophes: bool,
    /// The include punctuation value.
    pub include_punctuation: bool,
}

impl Default for TextProcessingOptions {
    fn default() -> Self {
        Self {
            language: None,
            lowercase: true,
            normalize_unicode: true,
            keep_apostrophes: true,
            include_punctuation: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// Data type for detailed text stats.
pub struct DetailedTextStats {
    /// The basic value.
    pub basic: TextStats,
    /// The paragraphs value.
    pub paragraphs: usize,
    /// The tokens value.
    pub tokens: usize,
    /// The unique tokens value.
    pub unique_tokens: usize,
    /// The average words per sentence value.
    pub average_words_per_sentence: f32,
    /// The average chars per word value.
    pub average_chars_per_word: f32,
}

/// Returns text stats.
pub fn text_stats(text: &str) -> TextStats {
    TextStats {
        bytes: text.len(),
        chars: text.chars().count(),
        words: tokenize_words(text).len(),
        lines: text.lines().count(),
        sentences: split_sentences(text).len(),
    }
}

/// Returns normalize whitespace.
pub fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Returns normalize text.
pub fn normalize_text(text: &str, options: &TextProcessingOptions) -> String {
    let normalized = if options.normalize_unicode {
        text.nfkc().collect::<String>()
    } else {
        text.to_string()
    };
    if options.lowercase {
        normalized.to_lowercase()
    } else {
        normalized
    }
}

/// Returns tokenize words.
pub fn tokenize_words(text: &str) -> Vec<String> {
    let options = TextProcessingOptions::default();
    tokenize(text, &options)
        .into_iter()
        .filter(|token| {
            matches!(
                token.kind,
                TokenKind::Word
                    | TokenKind::Number
                    | TokenKind::Url
                    | TokenKind::Email
                    | TokenKind::Mention
                    | TokenKind::Hashtag
            )
        })
        .map(|token| token.normalized)
        .collect()
}

/// Returns word counts.
pub fn word_counts(text: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for token in tokenize_words(text) {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

/// Returns split sentences.
pub fn split_sentences(text: &str) -> Vec<String> {
    split_sentence_spans(text, &TextProcessingOptions::default())
        .into_iter()
        .map(|sentence| normalize_whitespace(&sentence.text))
        .collect()
}

/// Returns segment words.
pub fn segment_words(text: &str, options: &TextBoundaryOptions) -> Vec<WordSegment> {
    let processing = TextProcessingOptions {
        lowercase: options.lowercase,
        normalize_unicode: options.normalize_unicode,
        include_punctuation: options.include_punctuation,
        ..TextProcessingOptions::default()
    };
    let mut segments = Vec::<WordSegment>::new();
    for (byte_start, segment) in UnicodeSegmentation::split_word_bound_indices(text) {
        if segment.chars().all(char::is_whitespace) {
            continue;
        }
        let kind = classify_word_segment(segment);
        let keep = match kind {
            TokenKind::Word
            | TokenKind::Url
            | TokenKind::Email
            | TokenKind::Mention
            | TokenKind::Hashtag => true,
            TokenKind::Number => options.include_numbers,
            TokenKind::Punctuation => options.include_punctuation,
            TokenKind::Other => segment.chars().any(char::is_alphanumeric),
        };
        if !keep || segment.chars().count() < options.min_chars {
            continue;
        }
        let byte_end = byte_start + segment.len();
        let current = WordSegment {
            text: segment.to_string(),
            normalized: normalize_text(segment, &processing),
            span: span_for(text, byte_start, byte_end),
            kind,
        };
        if let Some(previous) = segments.last_mut() {
            if previous.kind == TokenKind::Word
                && current.kind == TokenKind::Word
                && previous.span.byte_end == current.span.byte_start
            {
                previous.text.push_str(&current.text);
                previous.normalized = normalize_text(&previous.text, &processing);
                previous.span.byte_end = current.span.byte_end;
                previous.span.char_end = current.span.char_end;
                continue;
            }
        }
        segments.push(current);
    }
    segments
}

/// Returns segment graphemes.
pub fn segment_graphemes(text: &str) -> Vec<GraphemeSpan> {
    UnicodeSegmentation::grapheme_indices(text, true)
        .map(|(byte_start, grapheme)| {
            let byte_end = byte_start + grapheme.len();
            GraphemeSpan {
                text: grapheme.to_string(),
                span: span_for(text, byte_start, byte_end),
            }
        })
        .collect()
}

/// Returns detect script profile.
pub fn detect_script_profile(text: &str) -> ScriptProfile {
    let mut scripts = BTreeMap::<String, usize>::new();
    let mut digits = 0;
    let mut whitespace = 0;
    let mut punctuation = 0;
    let mut other = 0;

    for ch in text.chars() {
        if ch.is_whitespace() {
            whitespace += 1;
        } else if ch.is_numeric() {
            digits += 1;
        } else if is_sentence_or_symbol_punctuation(ch) || ch.is_ascii_punctuation() {
            punctuation += 1;
        } else if let Some(script) = script_name(ch) {
            *scripts.entry(script.to_string()).or_insert(0) += 1;
        } else {
            other += 1;
        }
    }

    let dominant_script = scripts
        .iter()
        .max_by(|left, right| left.1.cmp(right.1).then_with(|| right.0.cmp(left.0)))
        .map(|(script, _)| script.clone());
    let is_mixed = scripts.len() > 1;

    ScriptProfile {
        scripts,
        digits,
        whitespace,
        punctuation,
        other,
        dominant_script,
        is_mixed,
    }
}

/// Returns tokenize.
pub fn tokenize(text: &str, options: &TextProcessingOptions) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut byte_index = 0;

    while byte_index < text.len() {
        let ch = next_char(text, byte_index);
        if ch.is_whitespace() {
            byte_index += ch.len_utf8();
            continue;
        }

        let (byte_end, kind) = if starts_url(text, byte_index) {
            (consume_until_whitespace(text, byte_index), TokenKind::Url)
        } else if ch == '@' {
            let end = consume_prefixed_word(text, byte_index);
            if end > byte_index + ch.len_utf8() {
                (end, TokenKind::Mention)
            } else {
                (byte_index + ch.len_utf8(), TokenKind::Other)
            }
        } else if ch == '#' {
            let end = consume_prefixed_word(text, byte_index);
            if end > byte_index + ch.len_utf8() {
                (end, TokenKind::Hashtag)
            } else {
                (byte_index + ch.len_utf8(), TokenKind::Other)
            }
        } else if ch.is_ascii_digit() {
            (consume_number(text, byte_index), TokenKind::Number)
        } else if is_word_char(ch, options.keep_apostrophes) {
            let mut end = consume_word_like(text, byte_index, options.keep_apostrophes);
            let candidate_end =
                trim_trailing_token_punctuation(text, byte_index, end, TokenKind::Email);
            let candidate = &text[byte_index..candidate_end];
            let kind = if is_email(candidate) {
                end = candidate_end;
                TokenKind::Email
            } else {
                end = consume_plain_word(text, byte_index, options.keep_apostrophes);
                TokenKind::Word
            };
            (end, kind)
        } else if is_sentence_or_symbol_punctuation(ch) {
            (byte_index + ch.len_utf8(), TokenKind::Punctuation)
        } else {
            (byte_index + ch.len_utf8(), TokenKind::Other)
        };

        let byte_end = trim_trailing_token_punctuation(text, byte_index, byte_end, kind);
        if byte_end == byte_index {
            byte_index += ch.len_utf8();
            continue;
        }
        if kind != TokenKind::Punctuation || options.include_punctuation {
            let raw = &text[byte_index..byte_end];
            tokens.push(Token {
                text: raw.to_string(),
                normalized: normalize_text(raw, options),
                span: span_for(text, byte_index, byte_end),
                kind,
            });
        }
        byte_index = byte_end;
    }

    tokens
}

/// Returns split sentence spans.
pub fn split_sentence_spans(text: &str, options: &TextProcessingOptions) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let chars = text.char_indices().collect::<Vec<_>>();

    for (position, (byte_index, ch)) in chars.iter().copied().enumerate() {
        if !is_sentence_terminator(ch) {
            continue;
        }
        if ch == '.' && is_abbreviation_boundary(text, byte_index) {
            continue;
        }
        if ch == '.'
            && previous_char(&chars, position).is_some_and(|value| value.is_ascii_digit())
            && next_char_from_indices(&chars, position).is_some_and(|value| value.is_ascii_digit())
        {
            continue;
        }
        if next_char_from_indices(&chars, position).is_some_and(is_sentence_terminator) {
            continue;
        }

        let end = byte_index + ch.len_utf8();
        push_sentence(text, start, end, options, &mut sentences);
        start = end;
    }

    push_sentence(text, start, text.len(), options, &mut sentences);
    sentences
}

/// Returns split paragraphs.
pub fn split_paragraphs(text: &str) -> Vec<Paragraph> {
    let mut paragraphs = Vec::new();
    let mut paragraph_start = None;
    let mut last_non_blank_end = 0;
    let mut line_start = 0;

    for line in text.split_inclusive('\n') {
        let line_end = line_start + line.len();
        let line_without_newline = line.trim_end_matches(['\r', '\n']);
        if line_without_newline.trim().is_empty() {
            if let Some(start) = paragraph_start.take() {
                push_paragraph(text, start, last_non_blank_end, &mut paragraphs);
            }
        } else {
            let content_start =
                line_start + (line_without_newline.len() - line_without_newline.trim_start().len());
            paragraph_start.get_or_insert(content_start);
            last_non_blank_end = line_start + line_without_newline.trim_end().len();
        }
        line_start = line_end;
    }

    if let Some(start) = paragraph_start {
        push_paragraph(text, start, last_non_blank_end, &mut paragraphs);
    }

    paragraphs
}

/// Builds annotation graph.
pub fn build_annotation_graph(text: &str, options: &TextProcessingOptions) -> TextAnnotationGraph {
    let tokens = tokenize(text, options);
    let sentences = split_sentence_spans(text, options);
    let paragraphs = split_paragraphs(text);
    build_annotation_graph_from_parts(text, &tokens, &sentences, &paragraphs)
}

/// Builds annotation graph from parts.
pub fn build_annotation_graph_from_parts(
    text: &str,
    tokens: &[Token],
    sentences: &[Sentence],
    paragraphs: &[Paragraph],
) -> TextAnnotationGraph {
    let token_id_offset = 0;
    let sentence_id_offset = tokens.len();
    let paragraph_id_offset = tokens.len() + sentences.len();

    let annotated_paragraphs = paragraphs
        .iter()
        .enumerate()
        .map(|(index, paragraph)| {
            let id = AnnotationId(paragraph_id_offset + index);
            let sentence_indices = sentences
                .iter()
                .enumerate()
                .filter(|(_, sentence)| span_is_inside(sentence.span, paragraph.span))
                .map(|(sentence_index, _)| sentence_index)
                .collect::<Vec<_>>();
            let sentence_ids = sentence_indices
                .iter()
                .map(|sentence_index| AnnotationId(sentence_id_offset + sentence_index))
                .collect::<Vec<_>>();
            let sentence_start = sentence_indices.first().copied().unwrap_or(sentences.len());
            let sentence_end = sentence_indices
                .last()
                .map(|index| index + 1)
                .unwrap_or(sentence_start);

            AnnotatedParagraph {
                id,
                span: TextSpanRef {
                    id,
                    span: paragraph.span,
                },
                text: paragraph.text.clone(),
                sentence_start,
                sentence_end,
                sentence_ids,
            }
        })
        .collect::<Vec<_>>();

    let annotated_sentences = sentences
        .iter()
        .enumerate()
        .map(|(sentence_index, sentence)| {
            let id = AnnotationId(sentence_id_offset + sentence_index);
            let token_indices = tokens
                .iter()
                .enumerate()
                .filter(|(_, token)| span_is_inside(token.span, sentence.span))
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let paragraph_id = annotated_paragraphs
                .iter()
                .find(|paragraph| span_is_inside(sentence.span, paragraph.span.span))
                .map(|paragraph| paragraph.id);
            AnnotatedSentence {
                id,
                span: TextSpanRef {
                    id,
                    span: sentence.span,
                },
                text: sentence.text.clone(),
                token_start: token_indices.first().copied().unwrap_or(tokens.len()),
                token_end: token_indices
                    .last()
                    .map(|index| index + 1)
                    .unwrap_or(tokens.len()),
                token_ids: token_indices
                    .into_iter()
                    .map(|index| AnnotationId(token_id_offset + index))
                    .collect(),
                paragraph_id,
            }
        })
        .collect::<Vec<_>>();

    let annotated_tokens = tokens
        .iter()
        .enumerate()
        .map(|(token_index, token)| {
            let id = AnnotationId(token_id_offset + token_index);
            let sentence_id = annotated_sentences
                .iter()
                .find(|sentence| span_is_inside(token.span, sentence.span.span))
                .map(|sentence| sentence.id)
                .unwrap_or(AnnotationId(sentence_id_offset));
            let paragraph_id = annotated_paragraphs
                .iter()
                .find(|paragraph| span_is_inside(token.span, paragraph.span.span))
                .map(|paragraph| paragraph.id);
            CanonicalToken {
                id,
                span: TextSpanRef {
                    id,
                    span: token.span,
                },
                text: token.text.clone(),
                normalized: token.normalized.clone(),
                kind: token.kind,
                sentence_id,
                paragraph_id,
            }
        })
        .collect();

    TextAnnotationGraph {
        text: text.to_string(),
        provenance: AnnotationProvenance::Tokenizer,
        confidence: AnnotationConfidence::default(),
        tokens: annotated_tokens,
        sentences: annotated_sentences,
        paragraphs: annotated_paragraphs,
    }
}

/// Returns detailed text stats.
pub fn detailed_text_stats(text: &str, options: &TextProcessingOptions) -> DetailedTextStats {
    let basic = text_stats(text);
    let paragraphs = split_paragraphs(text).len();
    let tokens = tokenize(text, options);
    let unique_tokens = tokens
        .iter()
        .filter(|token| token.kind != TokenKind::Punctuation)
        .map(|token| token.normalized.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let chars_in_words = tokens
        .iter()
        .filter(|token| matches!(token.kind, TokenKind::Word | TokenKind::Number))
        .map(|token| token.text.chars().count())
        .sum::<usize>();
    DetailedTextStats {
        basic,
        paragraphs,
        tokens: tokens.len(),
        unique_tokens,
        average_words_per_sentence: if basic.sentences == 0 {
            0.0
        } else {
            basic.words as f32 / basic.sentences as f32
        },
        average_chars_per_word: if basic.words == 0 {
            0.0
        } else {
            chars_in_words as f32 / basic.words as f32
        },
    }
}

fn next_char(text: &str, byte_index: usize) -> char {
    text[byte_index..]
        .chars()
        .next()
        .expect("byte_index must be inside text")
}

fn span_for(text: &str, byte_start: usize, byte_end: usize) -> TextSpan {
    TextSpan {
        byte_start,
        byte_end,
        char_start: text[..byte_start].chars().count(),
        char_end: text[..byte_end].chars().count(),
    }
}

fn span_is_inside(inner: TextSpan, outer: TextSpan) -> bool {
    inner.byte_start >= outer.byte_start && inner.byte_end <= outer.byte_end
}

fn starts_url(text: &str, byte_index: usize) -> bool {
    let tail = &text[byte_index..];
    tail.starts_with("http://") || tail.starts_with("https://") || tail.starts_with("www.")
}

fn classify_word_segment(segment: &str) -> TokenKind {
    if starts_url(segment, 0) {
        TokenKind::Url
    } else if is_email(segment) {
        TokenKind::Email
    } else if segment.starts_with('@') && segment[1..].chars().any(char::is_alphanumeric) {
        TokenKind::Mention
    } else if segment.starts_with('#') && segment[1..].chars().any(char::is_alphanumeric) {
        TokenKind::Hashtag
    } else if segment
        .chars()
        .all(|ch| ch.is_numeric() || matches!(ch, '.' | ',' | ':' | '/' | '-'))
    {
        TokenKind::Number
    } else if segment.chars().all(is_sentence_or_symbol_punctuation) {
        TokenKind::Punctuation
    } else if segment.chars().any(char::is_alphanumeric) {
        TokenKind::Word
    } else {
        TokenKind::Other
    }
}

fn consume_until_whitespace(text: &str, byte_start: usize) -> usize {
    let mut end = byte_start;
    for (offset, ch) in text[byte_start..].char_indices() {
        if ch.is_whitespace() {
            break;
        }
        end = byte_start + offset + ch.len_utf8();
    }
    end
}

fn consume_prefixed_word(text: &str, byte_start: usize) -> usize {
    let base = byte_start + next_char(text, byte_start).len_utf8();
    let mut end = base;
    for (offset, ch) in text[base..].char_indices() {
        if ch.is_alphanumeric() || ch == '_' || ch == '-' {
            end = base + offset + ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

fn consume_number(text: &str, byte_start: usize) -> usize {
    let mut end = byte_start;
    for (offset, ch) in text[byte_start..].char_indices() {
        if ch.is_ascii_digit() || matches!(ch, '.' | ',' | ':' | '/' | '-') {
            end = byte_start + offset + ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

fn consume_word_like(text: &str, byte_start: usize, keep_apostrophes: bool) -> usize {
    let mut end = byte_start;
    for (offset, ch) in text[byte_start..].char_indices() {
        if is_word_char(ch, keep_apostrophes) || matches!(ch, '@' | '.' | '_' | '-' | '+') {
            end = byte_start + offset + ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

fn consume_plain_word(text: &str, byte_start: usize, keep_apostrophes: bool) -> usize {
    let mut end = byte_start;
    for (offset, ch) in text[byte_start..].char_indices() {
        if is_word_char(ch, keep_apostrophes) {
            end = byte_start + offset + ch.len_utf8();
        } else {
            break;
        }
    }
    end
}

fn is_word_char(ch: char, keep_apostrophes: bool) -> bool {
    ch.is_alphanumeric() || (keep_apostrophes && is_apostrophe(ch))
}

fn is_apostrophe(ch: char) -> bool {
    matches!(ch, '\'' | '’')
}

fn is_email(candidate: &str) -> bool {
    let Some((local, domain)) = candidate.split_once('@') else {
        return false;
    };
    !local.is_empty() && domain.contains('.') && !domain.starts_with('.') && !domain.ends_with('.')
}

fn is_sentence_or_symbol_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '…' | '。' | '！' | '？' | '،' | '؛' | '¿' | '¡' | '«' | '»'
        )
}

fn trim_trailing_token_punctuation(
    text: &str,
    byte_start: usize,
    mut byte_end: usize,
    kind: TokenKind,
) -> usize {
    if !matches!(kind, TokenKind::Url | TokenKind::Email | TokenKind::Number) {
        return byte_end;
    }
    while byte_end > byte_start {
        let Some(ch) = text[..byte_end].chars().next_back() else {
            break;
        };
        if matches!(ch, '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}') {
            byte_end -= ch.len_utf8();
        } else {
            break;
        }
    }
    byte_end
}

fn is_sentence_terminator(ch: char) -> bool {
    matches!(ch, '.' | '?' | '!' | '…' | '。' | '！' | '？')
}

fn is_abbreviation_boundary(text: &str, period_byte_index: usize) -> bool {
    let prefix = &text[..period_byte_index];
    let word_start = prefix
        .char_indices()
        .rev()
        .find_map(|(index, ch)| (!ch.is_alphabetic()).then_some(index + ch.len_utf8()))
        .unwrap_or(0);
    let word = &text[word_start..period_byte_index];
    if word.is_empty() {
        return false;
    }
    let normalized = word.to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "mr" | "mrs"
            | "ms"
            | "dr"
            | "prof"
            | "sr"
            | "jr"
            | "st"
            | "vs"
            | "etc"
            | "e.g"
            | "i.e"
            | "u.s"
            | "u.k"
    ) || (word.chars().count() == 1
        && word
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase()))
}

fn script_name(ch: char) -> Option<&'static str> {
    let value = ch as u32;
    match value {
        0x0041..=0x007A | 0x00C0..=0x024F | 0x1E00..=0x1EFF => Some("Latin"),
        0x0370..=0x03FF | 0x1F00..=0x1FFF => Some("Greek"),
        0x0400..=0x052F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => Some("Cyrillic"),
        0x0590..=0x05FF => Some("Hebrew"),
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF => Some("Arabic"),
        0x0900..=0x097F => Some("Devanagari"),
        0x3040..=0x309F => Some("Hiragana"),
        0x30A0..=0x30FF | 0x31F0..=0x31FF => Some("Katakana"),
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => Some("Han"),
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => Some("Hangul"),
        _ if ch.is_alphabetic() => Some("Other"),
        _ => None,
    }
}

fn previous_char(chars: &[(usize, char)], position: usize) -> Option<char> {
    position
        .checked_sub(1)
        .and_then(|index| chars.get(index).map(|(_, ch)| *ch))
}

fn next_char_from_indices(chars: &[(usize, char)], position: usize) -> Option<char> {
    chars.get(position + 1).map(|(_, ch)| *ch)
}

fn push_sentence(
    text: &str,
    byte_start: usize,
    byte_end: usize,
    options: &TextProcessingOptions,
    sentences: &mut Vec<Sentence>,
) {
    if byte_start >= byte_end {
        return;
    }
    let raw = &text[byte_start..byte_end];
    let leading = raw.len() - raw.trim_start().len();
    let trailing = raw.trim_end().len();
    let start = byte_start + leading;
    let end = byte_start + trailing;
    if start >= end {
        return;
    }
    let sentence_text = text[start..end].to_string();
    let token_count = tokenize(&sentence_text, options).len();
    sentences.push(Sentence {
        text: sentence_text,
        span: span_for(text, start, end),
        token_count,
    });
}

fn push_paragraph(text: &str, byte_start: usize, byte_end: usize, paragraphs: &mut Vec<Paragraph>) {
    if byte_start >= byte_end {
        return;
    }
    let paragraph_text = text[byte_start..byte_end].to_string();
    paragraphs.push(Paragraph {
        sentence_count: split_sentence_spans(&paragraph_text, &TextProcessingOptions::default())
            .len(),
        text: paragraph_text,
        span: span_for(text, byte_start, byte_end),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_document_ids_include_stream_and_index() {
        assert_eq!(segment_document_id("subs", 7), "subs:7");
    }

    #[test]
    fn tokenizes_and_counts_words() {
        let counts = word_counts("Hello, hello world.");
        assert_eq!(counts.get("hello"), Some(&2));
        assert_eq!(counts.get("world"), Some(&1));
    }

    #[test]
    fn computes_text_stats() {
        let stats = text_stats("One sentence. Two words!");
        assert_eq!(stats.sentences, 2);
        assert_eq!(stats.words, 4);
    }

    #[test]
    fn tokenizes_unicode_words_with_offsets() {
        let tokens = tokenize("Hi café 東京", &TextProcessingOptions::default());
        assert_eq!(tokens[1].text, "café");
        assert_eq!(tokens[1].span.byte_start, 3);
        assert_eq!(tokens[1].span.char_start, 3);
        assert_eq!(tokens[2].text, "東京");
    }

    #[test]
    fn classifies_common_token_patterns() {
        let tokens = tokenize(
            "Mail a@b.com. #rust @team https://example.com 3.14",
            &TextProcessingOptions::default(),
        );
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Email));
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Hashtag));
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Mention));
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Url));
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Number));
    }

    #[test]
    fn apostrophe_behavior_is_configurable() {
        let keep = TextProcessingOptions::default();
        assert_eq!(tokenize_words("Don't stop"), vec!["don't", "stop"]);

        let split = TextProcessingOptions {
            keep_apostrophes: false,
            ..TextProcessingOptions::default()
        };
        let tokens = tokenize("Don't", &split)
            .into_iter()
            .map(|token| token.normalized)
            .collect::<Vec<_>>();
        assert_eq!(tokens, vec!["don", "t"]);
        assert_eq!(tokenize("Don't", &keep)[0].normalized, "don't");
    }

    #[test]
    fn splits_sentences_with_decimals_ellipses_and_multilingual_marks() {
        let sentences = split_sentences("Dr. Smith wrote pi is 3.14. Wait... Really？ Yes!");
        assert_eq!(
            sentences,
            vec!["Dr. Smith wrote pi is 3.14.", "Wait...", "Really？", "Yes!"]
        );
    }

    #[test]
    fn segments_graphemes_with_byte_and_char_spans() {
        let graphemes = segment_graphemes("e\u{301}👍🏽a");
        assert_eq!(graphemes.len(), 3);
        assert_eq!(graphemes[0].text, "e\u{301}");
        assert_eq!(graphemes[0].span.byte_start, 0);
        assert_eq!(graphemes[0].span.byte_end, 3);
        assert_eq!(graphemes[0].span.char_start, 0);
        assert_eq!(graphemes[0].span.char_end, 2);
        assert_eq!(graphemes[1].text, "👍🏽");
        assert_eq!(graphemes[1].span.char_start, 2);
        assert_eq!(graphemes[1].span.char_end, 4);
    }

    #[test]
    fn segments_words_with_unicode_boundaries() {
        let segments = segment_words("Café 東京 42!", &TextBoundaryOptions::default());
        let texts = segments
            .into_iter()
            .map(|segment| segment.normalized)
            .collect::<Vec<_>>();
        assert_eq!(texts, vec!["café", "東京", "42"]);
    }

    #[test]
    fn profiles_scripts() {
        let profile = detect_script_profile("Hello 東京 123!");
        assert_eq!(profile.scripts.get("Latin"), Some(&5));
        assert_eq!(profile.scripts.get("Han"), Some(&2));
        assert_eq!(profile.digits, 3);
        assert_eq!(profile.dominant_script.as_deref(), Some("Latin"));
        assert!(profile.is_mixed);
    }

    #[test]
    fn splits_paragraphs_on_blank_lines() {
        let paragraphs = split_paragraphs("First paragraph.\nStill first.\n\nSecond.");
        assert_eq!(paragraphs.len(), 2);
        assert_eq!(paragraphs[0].sentence_count, 2);
        assert_eq!(paragraphs[1].text, "Second.");
    }

    #[test]
    fn detailed_stats_include_derived_counts() {
        let stats = detailed_text_stats("One sentence.\n\nTwo words here.", &Default::default());
        assert_eq!(stats.paragraphs, 2);
        assert_eq!(stats.basic.sentences, 2);
        assert!(stats.average_words_per_sentence > 0.0);
    }

    #[test]
    fn builds_annotation_graph_with_stable_cross_references() {
        let graph = build_annotation_graph(
            "Alice launched the API.\n\nBerlin hosted the event.",
            &TextProcessingOptions::default(),
        );
        assert_eq!(graph.tokens.len(), 8);
        assert_eq!(graph.sentences.len(), 2);
        assert_eq!(graph.paragraphs.len(), 2);
        assert_eq!(graph.provenance, AnnotationProvenance::Tokenizer);
        assert!(graph.confidence.get() > 0.0);

        let first_sentence = &graph.sentences[0];
        let first_token = graph.token(first_sentence.token_ids[0]).unwrap();
        assert_eq!(first_token.text, "Alice");
        assert_eq!(first_token.sentence_id, first_sentence.id);
        assert_eq!(first_token.paragraph_id, first_sentence.paragraph_id);

        let second_paragraph = &graph.paragraphs[1];
        assert_eq!(second_paragraph.sentence_ids.len(), 1);
        let sentence = graph.sentence(second_paragraph.sentence_ids[0]).unwrap();
        assert_eq!(sentence.text, "Berlin hosted the event.");
    }
}
