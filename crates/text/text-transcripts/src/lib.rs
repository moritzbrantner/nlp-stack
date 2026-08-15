#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;
use text_core::{
    tokenize, tokenize_words, AnalysisEvent, OwnedTextSegment, TextAnalyzer, TextProcessingOptions,
    TextSegment, Timestamp, TokenKind,
};

use thiserror::Error;
pub mod contracts;
pub use contracts::{
    text_segment_contract_with_source, TranscriptCharContract, TranscriptSegmentContract,
    TranscriptWordContract, TranscriptionContract,
};

#[derive(Debug, Error)]
/// Variants describing transcription error.
pub enum TranscriptionError {
    #[error("I/O error: {0}")]
    /// The I/O variant.
    Io(#[from] std::io::Error),
    #[error("invalid transcript JSON: {0}")]
    /// The JSON variant.
    Json(#[from] serde_json::Error),
    #[error("invalid transcript: {0}")]
    /// The invalid transcript variant.
    InvalidTranscript(String),
    #[error("{0}")]
    /// The detect variant.
    Detect(#[from] text_core::DetectError),
}

/// Type alias for result.
pub type Result<T> = std::result::Result<T, TranscriptionError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing transcript format.
pub enum TranscriptFormat {
    /// The plain variant.
    Plain,
    /// The lines variant.
    Lines,
    /// The whisper JSON variant.
    WhisperJson,
    /// The srt variant.
    Srt,
    /// The web vtt variant.
    WebVtt,
}

impl TranscriptFormat {
    /// Infers a transcript format from a file extension.
    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension
            .trim()
            .trim_start_matches('.')
            .to_ascii_lowercase()
            .as_str()
        {
            "txt" | "text" => Some(Self::Plain),
            "lines" => Some(Self::Lines),
            "json" | "whisper" | "whisperjson" | "whisper-json" => Some(Self::WhisperJson),
            "srt" => Some(Self::Srt),
            "vtt" | "webvtt" | "web-vtt" => Some(Self::WebVtt),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for transcript segment.
pub struct TranscriptSegment {
    /// The index value.
    pub index: u64,
    /// The start seconds value.
    pub start_seconds: Option<f64>,
    /// The end seconds value.
    pub end_seconds: Option<f64>,
    /// Text content for this value.
    pub text: String,
    /// Language tag for this value.
    pub language: Option<String>,
    /// The speaker value.
    pub speaker: Option<String>,
    /// Confidence score for this value.
    pub confidence: Option<f32>,
    /// The is final value.
    pub is_final: bool,
}

impl TranscriptSegment {
    /// Returns metadata.
    pub fn metadata(&self) -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::new();
        insert_optional(&mut metadata, "language", self.language.as_deref());
        insert_optional(&mut metadata, "speaker", self.speaker.as_deref());
        insert_optional_number(&mut metadata, "start_seconds", self.start_seconds);
        insert_optional_number(&mut metadata, "end_seconds", self.end_seconds);
        insert_optional_display(&mut metadata, "confidence", self.confidence);
        metadata
    }

    /// Returns metadata with source.
    pub fn metadata_with_source(&self, source: impl Into<String>) -> BTreeMap<String, String> {
        let mut metadata = self.metadata();
        let source = source.into();
        if !source.is_empty() {
            metadata.insert("source".to_string(), source);
        }
        metadata
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for transcription result.
pub struct TranscriptionResult {
    /// Text content for this value.
    pub text: Option<String>,
    /// Language tag for this value.
    pub language: Option<String>,
    /// The segments value.
    pub segments: Vec<TranscriptSegment>,
    /// The source value.
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
/// Optional word-level transcript timing.
pub struct TranscriptWord {
    /// Word text.
    pub text: String,
    /// Optional start time in seconds.
    pub start_seconds: Option<f64>,
    /// Optional end time in seconds.
    pub end_seconds: Option<f64>,
    /// Optional confidence.
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone)]
/// Options for subtitle text normalization.
pub struct SubtitleNormalizationOptions {
    /// Strip WebVTT/SRT markup used for subtitle rendering.
    pub strip_markup: bool,
    /// Decode common HTML entities used in captions.
    pub decode_basic_entities: bool,
    /// Collapse all whitespace runs into one ASCII space.
    pub collapse_whitespace: bool,
}

impl Default for SubtitleNormalizationOptions {
    fn default() -> Self {
        Self {
            strip_markup: true,
            decode_basic_entities: true,
            collapse_whitespace: true,
        }
    }
}

/// Normalizes subtitle cue text without parsing timing blocks.
pub fn normalize_subtitle_text(text: &str, options: SubtitleNormalizationOptions) -> String {
    let mut normalized = text.to_string();
    if options.strip_markup {
        normalized = strip_subtitle_markup(&normalized);
    }
    if options.decode_basic_entities {
        normalized = decode_basic_entities(&normalized);
    }
    if options.collapse_whitespace {
        normalized = collapse_whitespace(&normalized);
    }
    normalized
}

#[derive(Debug, Default, Clone)]
/// Transcript-specific deterministic analyzer.
pub struct TranscriptHeuristicAnalyzer;

impl TextAnalyzer for TranscriptHeuristicAnalyzer {
    fn name(&self) -> &str {
        "transcript_heuristics"
    }

    fn process_segment(
        &mut self,
        segment: &TextSegment<'_>,
    ) -> text_core::Result<Vec<AnalysisEvent>> {
        let mut events = Vec::new();
        let text = segment.text.trim();
        if text.ends_with(['?', '؟', '？']) {
            events.push(event_at(self.name(), "speech:question", segment.timestamp));
        }
        if has_token_kind(text, TokenKind::Url) {
            events.push(event_at(self.name(), "speech:url", segment.timestamp));
        }
        if has_token_kind(text, TokenKind::Number) {
            events.push(event_at(self.name(), "speech:number", segment.timestamp));
        }
        if tokenize_words(text).len() >= 30 {
            events.push(event_at(
                self.name(),
                "speech:long_segment",
                segment.timestamp,
            ));
        }
        Ok(events)
    }
}

#[derive(Debug, Deserialize)]
struct WhisperOutput {
    text: Option<String>,
    language: Option<String>,
    #[serde(default)]
    segments: Vec<WhisperSegment>,
}

#[derive(Debug, Deserialize)]
struct WhisperSegment {
    id: Option<u64>,
    start: Option<f64>,
    end: Option<f64>,
    text: String,
    #[serde(default)]
    avg_logprob: Option<f32>,
    #[serde(default)]
    no_speech_prob: Option<f32>,
}

/// Parses parse whisper JSON.
pub fn parse_whisper_json(bytes: &[u8]) -> Result<TranscriptionResult> {
    let parsed: WhisperOutput = serde_json::from_slice(bytes)?;
    let segments = parsed
        .segments
        .into_iter()
        .enumerate()
        .map(|(index, segment)| TranscriptSegment {
            index: segment.id.unwrap_or(index as u64),
            start_seconds: segment.start,
            end_seconds: segment.end,
            text: segment.text.trim().to_string(),
            language: parsed.language.clone(),
            speaker: None,
            confidence: whisper_confidence(segment.avg_logprob, segment.no_speech_prob),
            is_final: true,
        })
        .collect::<Vec<_>>();
    Ok(TranscriptionResult {
        text: parsed.text.map(|text| text.trim().to_string()),
        language: parsed.language,
        segments,
        source: None,
    })
}

/// Normalizes an existing transcription contract.
pub fn normalize_transcription_contract(
    contract: TranscriptionContract,
) -> Result<TranscriptionContract> {
    contract.normalized()
}

/// Builds and normalizes a transcription contract from imported transcript segments.
pub fn normalize_imported_segments(
    source: Option<String>,
    language: Option<String>,
    segments: Vec<TranscriptSegmentContract>,
) -> Result<TranscriptionContract> {
    TranscriptionContract::from_segments(source, language, segments)
}

/// Parses WhisperX JSON into the shared transcription contract.
pub fn parse_whisperx_json(bytes: &[u8]) -> Result<TranscriptionContract> {
    let value: Value = serde_json::from_slice(bytes)?;
    let object = value.as_object().ok_or_else(|| {
        TranscriptionError::InvalidTranscript("WhisperX JSON must be an object".to_string())
    })?;
    let language = object
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_string);
    let text = object
        .get("text")
        .and_then(Value::as_str)
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty());
    let source = object
        .get("source")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut attributes = unknown_attributes(
        object,
        &["language", "text", "source", "segments", "word_segments"],
    );
    if let Some(count) = object.get("segment_count").and_then(Value::as_u64) {
        attributes.insert("segment_count".to_string(), count.to_string());
    }

    let mut segments = object
        .get("segments")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            TranscriptionError::InvalidTranscript(
                "WhisperX JSON must include a segments array".to_string(),
            )
        })?
        .iter()
        .enumerate()
        .map(|(index, segment)| whisperx_segment(segment, index as u64, language.clone()))
        .collect::<Result<Vec<_>>>()?;

    if segments.iter().all(|segment| segment.words.is_empty()) {
        if let Some(words) = object.get("word_segments").and_then(Value::as_array) {
            attach_flat_whisperx_words(&mut segments, words)?;
        }
    }

    let mut contract = TranscriptionContract {
        text,
        language,
        segments,
        source,
        attributes,
    }
    .normalized()?;
    for segment in &mut contract.segments {
        if segment.speaker.is_none() {
            segment.speaker = infer_segment_speaker(&segment.words);
        }
    }
    contract.validate_strict()?;
    Ok(contract)
}

/// Parses parse srt.
pub fn parse_srt(text: &str) -> Result<TranscriptionResult> {
    parse_subtitle_blocks(text, TranscriptFormat::Srt)
}

/// Parses parse webvtt.
pub fn parse_webvtt(text: &str) -> Result<TranscriptionResult> {
    parse_subtitle_blocks(text, TranscriptFormat::WebVtt)
}

/// Parses parse plain lines.
pub fn parse_plain_lines(text: &str) -> TranscriptionResult {
    let segments = text
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = line.trim();
            (!line.is_empty()).then(|| TranscriptSegment {
                index: index as u64,
                start_seconds: None,
                end_seconds: None,
                text: line.to_string(),
                language: None,
                speaker: None,
                confidence: None,
                is_final: true,
            })
        })
        .collect::<Vec<_>>();
    TranscriptionResult {
        text: Some(
            segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        language: None,
        segments,
        source: None,
    }
}

/// Returns format srt.
pub fn format_srt(segments: &[TranscriptSegment]) -> String {
    let mut output = String::new();
    for (index, segment) in segments.iter().enumerate() {
        let start = segment.start_seconds.unwrap_or(0.0);
        let end = segment
            .end_seconds
            .unwrap_or_else(|| (start + 2.0).max(start));
        output.push_str(&(index + 1).to_string());
        output.push('\n');
        output.push_str(&format_srt_timestamp(start));
        output.push_str(" --> ");
        output.push_str(&format_srt_timestamp(end.max(start)));
        output.push('\n');
        output.push_str(segment.text.trim());
        output.push_str("\n\n");
    }
    output
}

/// Returns format webvtt.
pub fn format_webvtt(segments: &[TranscriptSegment]) -> String {
    let mut output = String::from("WEBVTT\n\n");
    for (index, segment) in segments.iter().enumerate() {
        if index > 0 {
            output.push('\n');
        }
        let start = segment.start_seconds.unwrap_or(0.0);
        let end = segment
            .end_seconds
            .unwrap_or_else(|| (start + 2.0).max(start));
        output.push_str(&format_webvtt_timestamp(start));
        output.push_str(" --> ");
        output.push_str(&format_webvtt_timestamp(end.max(start)));
        output.push('\n');
        output.push_str(segment.text.trim());
        output.push('\n');
    }
    output
}

/// Writes srt.
pub fn write_srt(path: impl AsRef<Path>, segments: &[TranscriptSegment]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format_srt(segments))?;
    Ok(())
}

/// Returns format srt timestamp.
pub fn format_srt_timestamp(seconds: f64) -> String {
    let total_millis = (seconds.max(0.0) * 1_000.0).round() as u64;
    let millis = total_millis % 1_000;
    let total_seconds = total_millis / 1_000;
    let secs = total_seconds % 60;
    let total_minutes = total_seconds / 60;
    let minutes = total_minutes % 60;
    let hours = total_minutes / 60;
    format!("{hours:02}:{minutes:02}:{secs:02},{millis:03}")
}

fn format_webvtt_timestamp(seconds: f64) -> String {
    format_srt_timestamp(seconds).replace(',', ".")
}

/// Returns segment to owned text segment.
pub fn segment_to_owned_text_segment(segment: &TranscriptSegment) -> OwnedTextSegment {
    text_core::TextSegmentContract::from(TranscriptSegmentContract::from(segment))
        .to_owned_text_segment()
}

/// Parses a transcript file by inferring the format from its extension.
pub fn parse_transcript_file(path: impl AsRef<Path>) -> Result<TranscriptionResult> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            TranscriptionError::InvalidTranscript("transcript file missing extension".to_string())
        })?;
    let format = TranscriptFormat::from_extension(extension).ok_or_else(|| {
        TranscriptionError::InvalidTranscript(format!(
            "unsupported transcript file extension `{extension}`"
        ))
    })?;
    let bytes = fs::read(path)?;
    let mut parsed = parse_transcript_bytes(&bytes, format)?;
    parsed.source = Some(path.to_string_lossy().into_owned());
    Ok(parsed)
}

/// Parses and normalizes a transcript file into the stable transcript contract.
pub fn parse_normalized_transcript_file(
    path: impl AsRef<Path>,
    options: SubtitleNormalizationOptions,
) -> Result<TranscriptionContract> {
    let parsed = parse_transcript_file(path)?;
    let mut contract = TranscriptionContract::from(parsed);
    contract.segments = contract
        .segments
        .into_iter()
        .filter_map(|mut segment| {
            segment.text = normalize_subtitle_text(&segment.text, options.clone());
            (!segment.text.is_empty()).then_some(segment)
        })
        .collect();
    contract.text = contract
        .text
        .as_deref()
        .map(|text| normalize_subtitle_text(text, options))
        .filter(|text| !text.is_empty());
    contract.normalized()
}

fn parse_transcript_bytes(bytes: &[u8], format: TranscriptFormat) -> Result<TranscriptionResult> {
    match format {
        TranscriptFormat::Plain | TranscriptFormat::Lines => {
            Ok(parse_plain_lines(&String::from_utf8_lossy(bytes)))
        }
        TranscriptFormat::WhisperJson => parse_whisper_json(bytes),
        TranscriptFormat::Srt => parse_srt(&String::from_utf8_lossy(bytes)),
        TranscriptFormat::WebVtt => parse_webvtt(&String::from_utf8_lossy(bytes)),
    }
}

fn whisperx_segment(
    value: &Value,
    fallback_index: u64,
    language: Option<String>,
) -> Result<TranscriptSegmentContract> {
    let object = value.as_object().ok_or_else(|| {
        TranscriptionError::InvalidTranscript("WhisperX segment must be an object".to_string())
    })?;
    let mut segment = TranscriptSegmentContract::new(
        object
            .get("id")
            .or_else(|| object.get("index"))
            .and_then(Value::as_u64)
            .unwrap_or(fallback_index),
        object
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    );
    segment.start_seconds = number_field(object, &["start", "start_seconds", "startSeconds"]);
    segment.end_seconds = number_field(object, &["end", "end_seconds", "endSeconds"]);
    segment.language = object
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or(language);
    segment.speaker = object
        .get("speaker")
        .or_else(|| object.get("speaker_label"))
        .or_else(|| object.get("speakerLabel"))
        .and_then(Value::as_str)
        .map(str::to_string);
    segment.confidence = confidence_field(
        object,
        &["confidence", "score", "avg_logprob", "no_speech_prob"],
    );
    segment.words = object
        .get("words")
        .or_else(|| object.get("word_segments"))
        .and_then(Value::as_array)
        .map(|words| {
            words
                .iter()
                .map(whisperx_word)
                .collect::<Result<Vec<TranscriptWordContract>>>()
        })
        .transpose()?
        .unwrap_or_default();
    segment.chars = object
        .get("chars")
        .or_else(|| object.get("characters"))
        .and_then(Value::as_array)
        .map(|chars| {
            chars
                .iter()
                .map(whisperx_char)
                .collect::<Result<Vec<TranscriptCharContract>>>()
        })
        .transpose()?
        .unwrap_or_default();
    if segment.speaker.is_none() {
        segment.speaker = infer_segment_speaker(&segment.words);
    }
    segment.attributes = unknown_attributes(
        object,
        &[
            "id",
            "index",
            "start",
            "start_seconds",
            "startSeconds",
            "end",
            "end_seconds",
            "endSeconds",
            "text",
            "language",
            "speaker",
            "speaker_label",
            "speakerLabel",
            "confidence",
            "score",
            "avg_logprob",
            "no_speech_prob",
            "words",
            "word_segments",
            "chars",
            "characters",
        ],
    );
    Ok(segment)
}

fn whisperx_word(value: &Value) -> Result<TranscriptWordContract> {
    let object = value.as_object().ok_or_else(|| {
        TranscriptionError::InvalidTranscript("WhisperX word must be an object".to_string())
    })?;
    Ok(TranscriptWordContract {
        text: object
            .get("word")
            .or_else(|| object.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        start_seconds: number_field(object, &["start", "start_seconds", "startSeconds"]),
        end_seconds: number_field(object, &["end", "end_seconds", "endSeconds"]),
        confidence: confidence_field(object, &["confidence", "score", "probability"]),
        speaker: object
            .get("speaker")
            .or_else(|| object.get("speaker_label"))
            .or_else(|| object.get("speakerLabel"))
            .and_then(Value::as_str)
            .map(str::to_string),
        attributes: unknown_attributes(
            object,
            &[
                "word",
                "text",
                "start",
                "start_seconds",
                "startSeconds",
                "end",
                "end_seconds",
                "endSeconds",
                "confidence",
                "score",
                "probability",
                "speaker",
                "speaker_label",
                "speakerLabel",
            ],
        ),
    })
}

fn whisperx_char(value: &Value) -> Result<TranscriptCharContract> {
    let object = value.as_object().ok_or_else(|| {
        TranscriptionError::InvalidTranscript("WhisperX char must be an object".to_string())
    })?;
    Ok(TranscriptCharContract {
        character: object
            .get("char")
            .or_else(|| object.get("character"))
            .or_else(|| object.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        start_seconds: number_field(object, &["start", "start_seconds", "startSeconds"]),
        end_seconds: number_field(object, &["end", "end_seconds", "endSeconds"]),
        confidence: confidence_field(object, &["confidence", "score", "probability"]),
        attributes: unknown_attributes(
            object,
            &[
                "char",
                "character",
                "text",
                "start",
                "start_seconds",
                "startSeconds",
                "end",
                "end_seconds",
                "endSeconds",
                "confidence",
                "score",
                "probability",
            ],
        ),
    })
}

fn attach_flat_whisperx_words(
    segments: &mut [TranscriptSegmentContract],
    words: &[Value],
) -> Result<()> {
    for value in words {
        let word = whisperx_word(value)?;
        let midpoint = match (word.start_seconds, word.end_seconds) {
            (Some(start), Some(end)) => Some((start + end) * 0.5),
            (Some(start), None) => Some(start),
            (None, Some(end)) => Some(end),
            (None, None) => None,
        };
        let Some(segment) = segments.iter_mut().find(|segment| {
            midpoint
                .zip(segment.start_seconds.zip(segment.end_seconds))
                .map(|(midpoint, (start, end))| midpoint >= start && midpoint <= end)
                .unwrap_or(false)
        }) else {
            continue;
        };
        segment.words.push(word);
    }
    for segment in segments {
        if segment.speaker.is_none() {
            segment.speaker = infer_segment_speaker(&segment.words);
        }
    }
    Ok(())
}

fn infer_segment_speaker(words: &[TranscriptWordContract]) -> Option<String> {
    let mut scores: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    for word in words {
        let Some(speaker) = word
            .speaker
            .as_deref()
            .filter(|speaker| !speaker.is_empty())
        else {
            continue;
        };
        let duration = word
            .start_seconds
            .zip(word.end_seconds)
            .map(|(start, end)| (end - start).max(0.0))
            .filter(|duration| duration.is_finite() && *duration > 0.0)
            .unwrap_or(1.0);
        let entry = scores.entry(speaker).or_insert((0.0, 0));
        entry.0 += duration;
        entry.1 += 1;
    }
    scores
        .into_iter()
        .max_by(|left, right| {
            left.1
                 .0
                .total_cmp(&right.1 .0)
                .then(left.1 .1.cmp(&right.1 .1))
                .then_with(|| right.0.cmp(left.0))
        })
        .map(|(speaker, _)| speaker.to_string())
}

fn unknown_attributes(
    object: &serde_json::Map<String, Value>,
    known_fields: &[&str],
) -> BTreeMap<String, String> {
    object
        .iter()
        .filter(|(key, _)| !known_fields.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), json_attribute(value)))
        .collect()
}

fn json_attribute(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

fn number_field(object: &serde_json::Map<String, Value>, names: &[&str]) -> Option<f64> {
    names
        .iter()
        .find_map(|name| object.get(*name).and_then(Value::as_f64))
        .filter(|value| value.is_finite())
}

fn confidence_field(object: &serde_json::Map<String, Value>, names: &[&str]) -> Option<f32> {
    for name in names {
        let Some(value) = object.get(*name).and_then(Value::as_f64) else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        return match *name {
            "avg_logprob" => Some(value.exp().clamp(0.0, 1.0) as f32),
            "no_speech_prob" => Some((1.0 - value).clamp(0.0, 1.0) as f32),
            _ => Some(value.clamp(0.0, 1.0) as f32),
        };
    }
    None
}

fn strip_subtitle_markup(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '<' {
            output.push(ch);
            continue;
        }

        let mut tag = String::new();
        let mut closed = false;
        for tag_ch in chars.by_ref() {
            if tag_ch == '>' {
                closed = true;
                break;
            }
            tag.push(tag_ch);
        }

        if !closed {
            output.push('<');
            output.push_str(&tag);
            break;
        }

        let tag = tag.trim();
        if tag.is_empty() {
            continue;
        }

        if is_subtitle_timestamp(tag) {
            output.push(' ');
        } else if is_subtitle_tag(tag) {
            continue;
        } else {
            output.push('<');
            output.push_str(tag);
            output.push('>');
        }
    }
    output
}

fn is_subtitle_tag(tag: &str) -> bool {
    let tag = tag.trim_start_matches('/');
    let name = tag
        .split(|ch: char| ch.is_whitespace() || ch == '.')
        .next()
        .unwrap_or_default();
    matches!(name, "b" | "c" | "i" | "lang" | "rt" | "ruby" | "u" | "v")
}

fn is_subtitle_timestamp(value: &str) -> bool {
    let value = value.replace(',', ".");
    let parts = value.split(':').collect::<Vec<_>>();
    let [hours, minutes, seconds] = parts.as_slice() else {
        return false;
    };
    is_two_digits(hours)
        && is_two_digits(minutes)
        && seconds.len() == 6
        && seconds.as_bytes().get(2) == Some(&b'.')
        && seconds[..2].bytes().all(|byte| byte.is_ascii_digit())
        && seconds[3..].bytes().all(|byte| byte.is_ascii_digit())
}

fn is_two_digits(value: &str) -> bool {
    value.len() == 2 && value.bytes().all(|byte| byte.is_ascii_digit())
}

fn decode_basic_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn parse_subtitle_blocks(text: &str, format: TranscriptFormat) -> Result<TranscriptionResult> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut segments = Vec::new();

    for block in normalized.split("\n\n") {
        let lines = block
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        if lines.is_empty()
            || (format == TranscriptFormat::WebVtt && lines[0].starts_with("WEBVTT"))
        {
            continue;
        }

        let time_line_index = lines
            .iter()
            .position(|line| line.contains("-->"))
            .ok_or_else(|| {
                TranscriptionError::InvalidTranscript(
                    "subtitle block missing timestamp".to_string(),
                )
            })?;
        let (start_seconds, end_seconds) = parse_timestamp_range(lines[time_line_index])?;
        let text_lines = &lines[time_line_index + 1..];
        if text_lines.is_empty() {
            continue;
        }
        let index = if time_line_index > 0 {
            lines[0].parse::<u64>().unwrap_or(segments.len() as u64)
        } else {
            segments.len() as u64
        };
        segments.push(TranscriptSegment {
            index,
            start_seconds: Some(start_seconds),
            end_seconds: Some(end_seconds),
            text: text_lines.join(" "),
            language: None,
            speaker: None,
            confidence: None,
            is_final: true,
        });
    }

    Ok(TranscriptionResult {
        text: Some(
            segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        language: None,
        segments,
        source: None,
    })
}

fn parse_timestamp_range(line: &str) -> Result<(f64, f64)> {
    let Some((start, end_with_settings)) = line.split_once("-->") else {
        return Err(TranscriptionError::InvalidTranscript(
            "timestamp range missing -->".to_string(),
        ));
    };
    let end = end_with_settings
        .split_whitespace()
        .next()
        .unwrap_or(end_with_settings);
    Ok((parse_timestamp(start.trim())?, parse_timestamp(end.trim())?))
}

fn parse_timestamp(value: &str) -> Result<f64> {
    let value = value.replace(',', ".");
    let pieces = value.split(':').collect::<Vec<_>>();
    let seconds = match pieces.as_slice() {
        [minutes, seconds] => {
            Some(parse_timestamp_component(minutes)? * 60.0 + parse_timestamp_component(seconds)?)
        }
        [hours, minutes, seconds] => Some(
            parse_timestamp_component(hours)? * 3600.0
                + parse_timestamp_component(minutes)? * 60.0
                + parse_timestamp_component(seconds)?,
        ),
        _ => None,
    };
    seconds.ok_or_else(|| {
        TranscriptionError::InvalidTranscript(format!("invalid timestamp `{value}`"))
    })
}

fn parse_timestamp_component(value: &str) -> Result<f64> {
    value.parse::<f64>().map_err(|_| {
        TranscriptionError::InvalidTranscript(format!("invalid timestamp component `{value}`"))
    })
}

fn whisper_confidence(avg_logprob: Option<f32>, no_speech_prob: Option<f32>) -> Option<f32> {
    avg_logprob.or(no_speech_prob.map(|probability| 1.0 - probability))
}

fn insert_optional(metadata: &mut BTreeMap<String, String>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        metadata.insert(key.to_string(), value.to_string());
    }
}

fn insert_optional_number(metadata: &mut BTreeMap<String, String>, key: &str, value: Option<f64>) {
    insert_optional_display(metadata, key, value);
}

fn insert_optional_display<T: fmt::Display>(
    metadata: &mut BTreeMap<String, String>,
    key: &str,
    value: Option<T>,
) {
    if let Some(value) = value {
        metadata.insert(key.to_string(), value.to_string());
    }
}

fn has_token_kind(text: &str, kind: TokenKind) -> bool {
    tokenize(text, &TextProcessingOptions::default())
        .into_iter()
        .any(|token| token.kind == kind)
}

fn event_at(analyzer: &str, label: &str, timestamp: Option<Timestamp>) -> AnalysisEvent {
    let event = AnalysisEvent::new(analyzer, label);
    if let Some(timestamp) = timestamp {
        event.at_timestamp(timestamp)
    } else {
        event
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_whisper_json() {
        let parsed = parse_whisper_json(
            br#"{"text":"hello world","language":"en","segments":[{"id":7,"start":0.0,"end":1.5,"text":" hello"}]}"#,
        )
        .unwrap();

        assert_eq!(parsed.text.as_deref(), Some("hello world"));
        assert_eq!(parsed.language.as_deref(), Some("en"));
        assert_eq!(parsed.segments[0].index, 7);
        assert_eq!(parsed.segments[0].start_seconds, Some(0.0));
    }

    #[test]
    fn parses_whisperx_segment_chars_and_preserves_unknown_fields() {
        let parsed = parse_whisperx_json(
            br#"{
                "text": "hi",
                "language": "en",
                "segments": [{
                    "id": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "hi",
                    "chars": [
                        {"char": "h", "start": 1.1, "end": 1.2, "score": 0.9, "extra": "kept"},
                        {"char": "i", "start": 1.2, "end": 1.3, "score": 0.8}
                    ]
                }]
            }"#,
        )
        .unwrap();

        let chars = &parsed.segments[0].chars;
        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0].character, "h");
        assert_eq!(chars[0].start_seconds, Some(1.1));
        assert_eq!(chars[0].end_seconds, Some(1.2));
        assert_eq!(chars[0].confidence, Some(0.9));
        assert_eq!(
            chars[0].attributes.get("extra").map(String::as_str),
            Some("kept")
        );
    }

    #[test]
    fn normalization_keeps_chars_and_strict_validation_checks_bounds() {
        let mut segment = TranscriptSegmentContract::new(0, " hello ");
        segment.start_seconds = Some(0.0);
        segment.end_seconds = Some(1.0);
        segment.chars.push(TranscriptCharContract {
            character: "h".to_string(),
            start_seconds: Some(0.1),
            end_seconds: Some(0.2),
            confidence: Some(0.5),
            attributes: BTreeMap::new(),
        });

        let normalized = TranscriptionContract::new(vec![segment])
            .normalized()
            .unwrap();
        assert_eq!(normalized.segments[0].text, "hello");
        assert_eq!(normalized.segments[0].chars.len(), 1);
        normalized.validate_strict().unwrap();

        let mut invalid = normalized.clone();
        invalid.segments[0].chars[0].end_seconds = Some(1.1);
        let error = invalid.validate_strict().unwrap_err().to_string();
        assert!(error.contains("transcript char end_seconds"));
    }

    #[test]
    fn parses_srt() {
        let parsed = parse_srt("1\n00:00:01,000 --> 00:00:02,500\nHello\n\n").unwrap();
        assert_eq!(parsed.segments.len(), 1);
        assert_eq!(parsed.segments[0].start_seconds, Some(1.0));
        assert_eq!(parsed.segments[0].end_seconds, Some(2.5));
    }

    #[test]
    fn formats_srt() {
        let text = format_srt(&[
            TranscriptSegment {
                index: 0,
                start_seconds: Some(1.25),
                end_seconds: Some(3.5),
                text: "Hello".to_string(),
                language: None,
                speaker: None,
                confidence: None,
                is_final: true,
            },
            TranscriptSegment {
                index: 1,
                start_seconds: Some(63.0),
                end_seconds: Some(65.125),
                text: "World".to_string(),
                language: None,
                speaker: None,
                confidence: None,
                is_final: true,
            },
        ]);

        assert_eq!(
            text,
            "1\n00:00:01,250 --> 00:00:03,500\nHello\n\n2\n00:01:03,000 --> 00:01:05,125\nWorld\n\n"
        );
    }

    #[test]
    fn formats_webvtt() {
        let text = format_webvtt(&[TranscriptSegment {
            index: 0,
            start_seconds: Some(1.0),
            end_seconds: Some(0.5),
            text: "Hello.".to_string(),
            language: None,
            speaker: None,
            confidence: None,
            is_final: true,
        }]);

        assert_eq!(text, "WEBVTT\n\n00:00:01.000 --> 00:00:01.000\nHello.\n");
    }

    #[test]
    fn parses_webvtt() {
        let parsed = parse_webvtt("WEBVTT\n\ncue\n00:00:03.000 --> 00:00:04.250\nHi\n").unwrap();
        assert_eq!(parsed.segments.len(), 1);
        assert_eq!(parsed.segments[0].text, "Hi");
        assert_eq!(parsed.segments[0].start_seconds, Some(3.0));
    }

    #[test]
    fn parses_plain_lines() {
        let parsed = parse_plain_lines("one\n\ntwo\n");
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.text.as_deref(), Some("one\ntwo"));
    }

    #[test]
    fn converts_segment_timestamp() {
        let segment = TranscriptSegment {
            index: 2,
            start_seconds: Some(1.25),
            end_seconds: Some(2.0),
            text: "hello".to_string(),
            language: Some("en".to_string()),
            speaker: None,
            confidence: None,
            is_final: true,
        };
        let owned = segment_to_owned_text_segment(&segment);
        assert_eq!(owned.segment_index, 2);
        assert_eq!(owned.timestamp.unwrap().seconds(), 1.25);
        assert_eq!(owned.language.as_deref(), Some("en"));
    }

    #[test]
    fn transcript_segment_metadata_preserves_optional_fields() {
        let segment = TranscriptSegment {
            index: 2,
            start_seconds: Some(1.25),
            end_seconds: Some(2.0),
            text: "hello".to_string(),
            language: Some("en".to_string()),
            speaker: Some("speaker-1".to_string()),
            confidence: Some(0.75),
            is_final: true,
        };

        let metadata = segment.metadata_with_source("fixture.srt");

        assert_eq!(metadata["language"], "en");
        assert_eq!(metadata["speaker"], "speaker-1");
        assert_eq!(metadata["start_seconds"], "1.25");
        assert_eq!(metadata["end_seconds"], "2");
        assert_eq!(metadata["confidence"], "0.75");
        assert_eq!(metadata["source"], "fixture.srt");
    }

    #[test]
    fn transcript_heuristic_analyzer_emits_speech_events() {
        let segment = TextSegment {
            segment_index: 0,
            timestamp: None,
            text: "Visit https://example.com at 3?",
            language: None,
            is_final: true,
        };
        let mut analyzer = TranscriptHeuristicAnalyzer;

        let labels = analyzer
            .process_segment(&segment)
            .unwrap()
            .into_iter()
            .map(|event| event.label)
            .collect::<Vec<_>>();

        assert!(labels.iter().any(|label| label == "speech:question"));
        assert!(labels.iter().any(|label| label == "speech:url"));
        assert!(labels.iter().any(|label| label == "speech:number"));
    }

    #[test]
    fn srt_webvtt_plain_round_trip() {
        let srt = "1\n00:00:00,000 --> 00:00:01,000\nHello world\n";
        let parsed = parse_srt(srt).unwrap();
        let formatted = format_srt(&parsed.segments);
        assert!(formatted.contains("Hello world"));

        let webvtt =
            parse_webvtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world\n").unwrap();
        let plain = parse_plain_lines("Hello world\n");

        assert_eq!(parsed.segments[0].text, webvtt.segments[0].text);
        assert_eq!(plain.text.as_deref(), Some("Hello world"));
    }

    #[test]
    fn normalizes_subtitle_markup_entities_and_whitespace() {
        let normalized = normalize_subtitle_text(
            "<v Speaker>Hello&nbsp; <c.yellow>Rust</c> &amp; friends <00:00:01.000>\nnow",
            SubtitleNormalizationOptions::default(),
        );

        assert_eq!(normalized, "Hello Rust & friends now");
    }

    #[test]
    fn infers_transcript_format_from_extension() {
        assert_eq!(
            TranscriptFormat::from_extension(".vtt"),
            Some(TranscriptFormat::WebVtt)
        );
        assert_eq!(
            TranscriptFormat::from_extension("SRT"),
            Some(TranscriptFormat::Srt)
        );
        assert_eq!(
            TranscriptFormat::from_extension("json"),
            Some(TranscriptFormat::WhisperJson)
        );
        assert_eq!(TranscriptFormat::from_extension("csv"), None);
    }

    #[test]
    fn parses_and_normalizes_transcript_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.vtt");
        fs::write(
            &path,
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<c>Hello&nbsp;file</c>\n",
        )
        .unwrap();

        let parsed =
            parse_normalized_transcript_file(&path, SubtitleNormalizationOptions::default())
                .unwrap();

        assert_eq!(parsed.source.as_deref(), Some(path.to_str().unwrap()));
        assert_eq!(parsed.text.as_deref(), Some("Hello file"));
        assert_eq!(parsed.segments[0].text, "Hello file");
    }

    #[test]
    fn builds_text_segment_contract_with_source() {
        let mut segment = TranscriptSegmentContract::new(7, "hello source");
        segment.start_seconds = Some(1.25);
        segment.end_seconds = Some(2.5);
        segment.language = Some("en".to_string());

        let contract = text_segment_contract_with_source(
            &segment,
            "stream-1",
            "caption_manual",
            "https://example.test/video",
        );

        assert_eq!(contract.stream_id.as_deref(), Some("stream-1"));
        assert_eq!(contract.segment_index, 7);
        assert_eq!(contract.language.as_deref(), Some("en"));
        assert_eq!(contract.duration_seconds, Some(1.25));
        let source = contract.source.unwrap();
        assert_eq!(source.source_id.as_deref(), Some("stream-1"));
        assert_eq!(source.source_kind.as_deref(), Some("caption_manual"));
        assert_eq!(source.uri.as_deref(), Some("https://example.test/video"));
        assert_eq!(source.duration_seconds, Some(1.25));
    }

    #[test]
    fn preserves_unknown_markup_conservatively() {
        let normalized = normalize_subtitle_text(
            "look <custom value>here</custom>",
            SubtitleNormalizationOptions::default(),
        );

        assert_eq!(normalized, "look <custom value>here</custom>");
    }

    #[test]
    fn decodes_basic_entities() {
        let normalized = normalize_subtitle_text(
            "&amp; &lt; &gt; &quot; &#39; &apos; &nbsp;",
            SubtitleNormalizationOptions {
                strip_markup: false,
                decode_basic_entities: true,
                collapse_whitespace: true,
            },
        );

        assert_eq!(normalized, "& < > \" ' '");
    }

    #[test]
    fn parse_and_normalize_round_trip_for_subtitles() {
        let srt = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n<c>Hello&nbsp;SRT</c>\n").unwrap();
        let vtt =
            parse_webvtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<v Speaker>Hello&nbsp;VTT\n")
                .unwrap();

        let options = SubtitleNormalizationOptions::default();
        assert_eq!(
            normalize_subtitle_text(&srt.segments[0].text, options.clone()),
            "Hello SRT"
        );
        assert_eq!(
            normalize_subtitle_text(&vtt.segments[0].text, options),
            "Hello VTT"
        );
    }
}
