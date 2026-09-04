use std::{collections::BTreeMap, fmt};

use crate::{OwnedTextSegment, TextSegment};
use media_core::{Timebase, Timestamp};
use runtime_core::{MobileCapability, OperationId, OperationMetadata, RuntimeCapabilities};
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::{segment_document_id, OwnedTextDocument, TextDocument, TextSpan};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TimebaseContract {
    pub num: i32,
    pub den: i32,
}

impl From<Timebase> for TimebaseContract {
    fn from(value: Timebase) -> Self {
        Self {
            num: value.num,
            den: value.den,
        }
    }
}

impl From<TimebaseContract> for Timebase {
    fn from(value: TimebaseContract) -> Self {
        Self::new(value.num, value.den)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TimestampContract {
    pub pts: i64,
    pub timebase: TimebaseContract,
}

impl TimestampContract {
    pub fn seconds(self) -> f64 {
        Timestamp::from(self).seconds()
    }
}

impl From<Timestamp> for TimestampContract {
    fn from(value: Timestamp) -> Self {
        Self {
            pts: value.pts,
            timebase: value.timebase.into(),
        }
    }
}

impl From<TimestampContract> for Timestamp {
    fn from(value: TimestampContract) -> Self {
        Self::new(value.pts, value.timebase.into())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TextSourceRef {
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub source_kind: Option<String>,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub media_timestamp: Option<TimestampContract>,
    #[serde(default)]
    pub duration_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TextProvenance {
    #[serde(default)]
    pub crate_name: Option<String>,
    #[serde(default)]
    pub operation: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub runtime: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TextAnnotationSpan {
    pub span: TextSpan,
    #[serde(default)]
    pub token_start: Option<usize>,
    #[serde(default)]
    pub token_end: Option<usize>,
    #[serde(default)]
    pub source_segment_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TextDocumentContract {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub timestamp: Option<TimestampContract>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
}

impl TextDocumentContract {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            language: None,
            timestamp: None,
            attributes: BTreeMap::new(),
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
        }
    }

    pub fn from_segment_contract(segment: &TextSegmentContract) -> Self {
        segment.to_text_document_contract()
    }

    pub fn to_text_segment_contract(&self, segment_index: u64) -> TextSegmentContract {
        TextSegmentContract::from_document_contract(self, segment_index)
    }
}

pub trait IntoTextDocumentContract {
    fn into_text_document_contract(self) -> TextDocumentContract;
}

impl IntoTextDocumentContract for TextDocument<'_> {
    fn into_text_document_contract(self) -> TextDocumentContract {
        TextDocumentContract {
            id: self.id.to_string(),
            text: self.text.to_string(),
            language: self.language.map(ToString::to_string),
            timestamp: self.timestamp.map(Into::into),
            attributes: BTreeMap::new(),
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
        }
    }
}

impl IntoTextDocumentContract for OwnedTextDocument {
    fn into_text_document_contract(self) -> TextDocumentContract {
        TextDocumentContract {
            id: self.id,
            text: self.text,
            language: self.language,
            timestamp: self.timestamp.map(Into::into),
            attributes: BTreeMap::new(),
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
        }
    }
}

impl IntoTextDocumentContract for &OwnedTextDocument {
    fn into_text_document_contract(self) -> TextDocumentContract {
        self.as_document().into_text_document_contract()
    }
}

impl From<TextDocument<'_>> for TextDocumentContract {
    fn from(value: TextDocument<'_>) -> Self {
        value.into_text_document_contract()
    }
}

impl From<OwnedTextDocument> for TextDocumentContract {
    fn from(value: OwnedTextDocument) -> Self {
        value.into_text_document_contract()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TextSegmentContract {
    #[serde(default)]
    pub stream_id: Option<String>,
    pub segment_index: u64,
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub timestamp: Option<TimestampContract>,
    #[serde(default)]
    pub duration_seconds: Option<f64>,
    pub is_final: bool,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
}

impl TextSegmentContract {
    pub fn new(segment_index: u64, text: impl Into<String>) -> Self {
        Self {
            stream_id: None,
            segment_index,
            text: text.into(),
            language: None,
            timestamp: None,
            duration_seconds: None,
            is_final: true,
            attributes: BTreeMap::new(),
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
        }
    }

    pub fn document_id(&self) -> Option<String> {
        self.stream_id
            .as_deref()
            .map(|stream_id| segment_document_id(stream_id, self.segment_index))
    }

    pub fn to_owned_text_segment(&self) -> OwnedTextSegment {
        let mut segment =
            OwnedTextSegment::new(self.segment_index, self.text.clone()).finality(self.is_final);
        if let Some(language) = &self.language {
            segment = segment.language(language.clone());
        }
        if let Some(timestamp) = self.timestamp {
            segment = segment.timestamp(timestamp.into());
        }
        segment
    }

    pub fn to_text_document_contract(&self) -> TextDocumentContract {
        TextDocumentContract {
            id: self
                .document_id()
                .unwrap_or_else(|| self.segment_index.to_string()),
            text: self.text.clone(),
            language: self.language.clone(),
            timestamp: self.timestamp,
            attributes: self.attributes.clone(),
            source: self.source.clone().or_else(|| {
                (self.timestamp.is_some() || self.duration_seconds.is_some()).then(|| {
                    TextSourceRef {
                        source_id: self.stream_id.clone(),
                        source_kind: Some("text_segment".to_string()),
                        uri: None,
                        media_timestamp: self.timestamp,
                        duration_seconds: self.duration_seconds,
                    }
                })
            }),
            provenance: self.provenance.clone(),
            annotations: self.annotations.clone(),
        }
    }

    pub fn from_document_contract(document: &TextDocumentContract, segment_index: u64) -> Self {
        Self {
            stream_id: None,
            segment_index,
            text: document.text.clone(),
            language: document.language.clone(),
            timestamp: document.timestamp.or_else(|| {
                document
                    .source
                    .as_ref()
                    .and_then(|source| source.media_timestamp)
            }),
            duration_seconds: document
                .source
                .as_ref()
                .and_then(|source| source.duration_seconds),
            is_final: true,
            attributes: document.attributes.clone(),
            source: document.source.clone(),
            provenance: document.provenance.clone(),
            annotations: document.annotations.clone(),
        }
    }
}

pub trait AsTextSegmentContract {
    fn as_text_segment_contract(&self) -> TextSegmentContract;
}

impl AsTextSegmentContract for TextSegment<'_> {
    fn as_text_segment_contract(&self) -> TextSegmentContract {
        TextSegmentContract {
            stream_id: None,
            segment_index: self.segment_index,
            text: self.text.to_string(),
            language: self.language.map(ToString::to_string),
            timestamp: self.timestamp.map(Into::into),
            duration_seconds: None,
            is_final: self.is_final,
            attributes: BTreeMap::new(),
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
        }
    }
}

impl AsTextSegmentContract for OwnedTextSegment {
    fn as_text_segment_contract(&self) -> TextSegmentContract {
        self.as_segment().as_text_segment_contract()
    }
}

impl From<TextSegment<'_>> for TextSegmentContract {
    fn from(value: TextSegment<'_>) -> Self {
        value.as_text_segment_contract()
    }
}

impl From<OwnedTextSegment> for TextSegmentContract {
    fn from(value: OwnedTextSegment) -> Self {
        value.as_text_segment_contract()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TextStatisticsRequest {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TextStatisticsResult {
    pub byte_count: usize,
    pub character_count: usize,
    pub word_count: usize,
    pub line_count: usize,
    pub sentence_count: usize,
}

pub fn text_statistics_metadata() -> OperationMetadata {
    OperationMetadata {
        id: OperationId::new("text.statistics"),
        name: "Text statistics".to_string(),
        description: Some("Counts bytes, characters, words, lines, and sentences.".to_string()),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities {
            native: true,
            server: true,
            wasm: true,
            mobile: MobileCapability::Wasm,
            requirements: Vec::new(),
            max_recommended_input_bytes: Some(1_000_000),
        },
    }
}

/// Explicit UTF-16 code-unit offsets derived from a canonical UTF-8 byte span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Utf16Span {
    /// Inclusive UTF-16 code-unit offset.
    pub start: usize,
    /// Exclusive UTF-16 code-unit offset.
    pub end: usize,
}

/// Explicit grapheme-cluster offsets derived from a canonical UTF-8 byte span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphemeOffsetSpan {
    /// Inclusive grapheme-cluster offset.
    pub start: usize,
    /// Exclusive grapheme-cluster offset.
    pub end: usize,
}

/// Errors raised when a canonical UTF-8 byte span cannot be converted safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextSpanConversionError {
    /// The range is reversed or lies outside the source text.
    InvalidByteRange {
        byte_start: usize,
        byte_end: usize,
        text_length: usize,
    },
    /// One of the offsets splits a UTF-8 scalar value.
    NonCharacterBoundary { byte_start: usize, byte_end: usize },
}

impl fmt::Display for TextSpanConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidByteRange {
                byte_start,
                byte_end,
                text_length,
            } => write!(
                formatter,
                "invalid UTF-8 byte range {byte_start}..{byte_end} for text length {text_length}"
            ),
            Self::NonCharacterBoundary {
                byte_start,
                byte_end,
            } => write!(
                formatter,
                "UTF-8 byte range {byte_start}..{byte_end} does not align to character boundaries"
            ),
        }
    }
}

impl std::error::Error for TextSpanConversionError {}

impl TextSpan {
    /// Constructs a canonical span from a validated UTF-8 byte range.
    ///
    /// Legacy scalar offsets are populated only for compatibility. Callers own
    /// byte ranges; alternate coordinate systems are derived at their boundary.
    pub fn from_byte_range(
        text: &str,
        byte_start: usize,
        byte_end: usize,
    ) -> Result<Self, TextSpanConversionError> {
        let candidate = Self {
            byte_start,
            byte_end,
            char_start: 0,
            char_end: 0,
        };
        candidate.validate_byte_range(text)?;
        Ok(Self {
            byte_start,
            byte_end,
            char_start: text[..byte_start].chars().count(),
            char_end: text[..byte_end].chars().count(),
        })
    }

    /// Validates the canonical byte range against the supplied UTF-8 text.
    pub fn validate_byte_range(self, text: &str) -> Result<(), TextSpanConversionError> {
        if self.byte_start > self.byte_end || self.byte_end > text.len() {
            return Err(TextSpanConversionError::InvalidByteRange {
                byte_start: self.byte_start,
                byte_end: self.byte_end,
                text_length: text.len(),
            });
        }
        if !text.is_char_boundary(self.byte_start) || !text.is_char_boundary(self.byte_end) {
            return Err(TextSpanConversionError::NonCharacterBoundary {
                byte_start: self.byte_start,
                byte_end: self.byte_end,
            });
        }
        Ok(())
    }

    /// Converts the canonical byte range to UTF-16 code-unit offsets.
    ///
    /// Legacy `char_start`/`char_end` fields are intentionally ignored: byte
    /// offsets are the source of truth and alternate coordinates are derived at
    /// the boundary that needs them.
    pub fn to_utf16(self, text: &str) -> Result<Utf16Span, TextSpanConversionError> {
        self.validate_byte_range(text)?;
        Ok(Utf16Span {
            start: text[..self.byte_start].encode_utf16().count(),
            end: text[..self.byte_end].encode_utf16().count(),
        })
    }

    /// Converts the canonical byte range to grapheme-cluster offsets.
    ///
    /// Legacy `char_start`/`char_end` fields are intentionally ignored.
    pub fn to_grapheme(self, text: &str) -> Result<GraphemeOffsetSpan, TextSpanConversionError> {
        self.validate_byte_range(text)?;
        Ok(GraphemeOffsetSpan {
            start: text[..self.byte_start].graphemes(true).count(),
            end: text[..self.byte_end].graphemes(true).count(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_construction_derives_legacy_scalars_from_bytes() {
        let text = "e\u{301}👍🏽a";
        let span = TextSpan::from_byte_range(text, 3, 11).unwrap();

        assert_eq!((span.byte_start, span.byte_end), (3, 11));
        assert_eq!((span.char_start, span.char_end), (2, 4));
    }

    #[test]
    fn span_conversions_use_bytes_as_the_source_of_truth() {
        let text = "e\u{301}👍🏽a";
        let span = TextSpan {
            byte_start: 3,
            byte_end: 11,
            char_start: 99,
            char_end: 100,
        };

        assert_eq!(span.to_utf16(text).unwrap(), Utf16Span { start: 2, end: 6 });
        assert_eq!(
            span.to_grapheme(text).unwrap(),
            GraphemeOffsetSpan { start: 1, end: 2 }
        );
    }

    #[test]
    fn span_conversions_reject_non_utf8_boundaries() {
        let text = "é";
        let span = TextSpan {
            byte_start: 1,
            byte_end: 2,
            char_start: 0,
            char_end: 1,
        };

        assert_eq!(
            span.to_utf16(text),
            Err(TextSpanConversionError::NonCharacterBoundary {
                byte_start: 1,
                byte_end: 2,
            })
        );
    }

    #[test]
    fn span_conversions_reject_reversed_ranges() {
        let text = "abc";
        let span = TextSpan {
            byte_start: 2,
            byte_end: 1,
            char_start: 2,
            char_end: 1,
        };

        assert_eq!(
            span.to_grapheme(text),
            Err(TextSpanConversionError::InvalidByteRange {
                byte_start: 2,
                byte_end: 1,
                text_length: 3,
            })
        );
    }
}
