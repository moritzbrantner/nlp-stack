use media_core::{
    TimedTextCharContract, TimedTextContract, TimedTextSegmentContract, TimedTextWordContract,
};

use crate::{
    TranscriptCharContract, TranscriptSegmentContract, TranscriptWordContract,
    TranscriptionContract,
};

/// Converts neutral media-timeline text into the richer NLP transcript contract.
///
/// This is an explicit adapter boundary: media-core owns interchange data;
/// text-transcripts owns transcript validation, parsing, formatting, and NLP-facing behavior.
pub fn timed_text_to_transcription(value: TimedTextContract) -> crate::Result<TranscriptionContract> {
    TranscriptionContract {
        text: value.text,
        language: value.language,
        segments: value
            .segments
            .into_iter()
            .map(timed_segment_to_transcript)
            .collect(),
        source: value.source,
        attributes: value.attributes,
    }
    .normalized()
}

/// Converts an NLP transcript into neutral media-timeline text for downstream
/// audio, visual, storage, or application consumers.
pub fn transcription_to_timed_text(
    value: TranscriptionContract,
) -> media_core::Result<TimedTextContract> {
    TimedTextContract {
        text: value.text,
        language: value.language,
        segments: value
            .segments
            .into_iter()
            .map(transcript_segment_to_timed)
            .collect(),
        source: value.source,
        attributes: value.attributes,
    }
    .normalized()
}

fn timed_segment_to_transcript(value: TimedTextSegmentContract) -> TranscriptSegmentContract {
    TranscriptSegmentContract {
        index: value.index,
        start_seconds: value.start_seconds,
        end_seconds: value.end_seconds,
        text: value.text,
        language: value.language,
        speaker: value.speaker,
        confidence: value.confidence,
        is_final: value.is_final,
        words: value.words.into_iter().map(timed_word_to_transcript).collect(),
        chars: value.chars.into_iter().map(timed_char_to_transcript).collect(),
        attributes: value.attributes,
    }
}

fn timed_word_to_transcript(value: TimedTextWordContract) -> TranscriptWordContract {
    TranscriptWordContract {
        text: value.text,
        start_seconds: value.start_seconds,
        end_seconds: value.end_seconds,
        confidence: value.confidence,
        speaker: value.speaker,
        attributes: value.attributes,
    }
}

fn timed_char_to_transcript(value: TimedTextCharContract) -> TranscriptCharContract {
    TranscriptCharContract {
        character: value.character,
        start_seconds: value.start_seconds,
        end_seconds: value.end_seconds,
        confidence: value.confidence,
        attributes: value.attributes,
    }
}

fn transcript_segment_to_timed(value: TranscriptSegmentContract) -> TimedTextSegmentContract {
    TimedTextSegmentContract {
        index: value.index,
        start_seconds: value.start_seconds,
        end_seconds: value.end_seconds,
        text: value.text,
        language: value.language,
        speaker: value.speaker,
        confidence: value.confidence,
        is_final: value.is_final,
        words: value.words.into_iter().map(transcript_word_to_timed).collect(),
        chars: value.chars.into_iter().map(transcript_char_to_timed).collect(),
        attributes: value.attributes,
    }
}

fn transcript_word_to_timed(value: TranscriptWordContract) -> TimedTextWordContract {
    TimedTextWordContract {
        text: value.text,
        start_seconds: value.start_seconds,
        end_seconds: value.end_seconds,
        confidence: value.confidence,
        speaker: value.speaker,
        attributes: value.attributes,
    }
}

fn transcript_char_to_timed(value: TranscriptCharContract) -> TimedTextCharContract {
    TimedTextCharContract {
        character: value.character,
        start_seconds: value.start_seconds,
        end_seconds: value.end_seconds,
        confidence: value.confidence,
        attributes: value.attributes,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn neutral_and_nlp_contracts_round_trip_without_losing_interchange_data() {
        let neutral = TimedTextContract {
            text: Some("hello world".to_string()),
            language: Some("en".to_string()),
            segments: vec![TimedTextSegmentContract {
                index: 7,
                start_seconds: Some(1.0),
                end_seconds: Some(2.0),
                text: "hello world".to_string(),
                language: Some("en".to_string()),
                speaker: Some("speaker-a".to_string()),
                confidence: Some(0.9),
                is_final: true,
                words: vec![TimedTextWordContract {
                    text: "hello".to_string(),
                    start_seconds: Some(1.0),
                    end_seconds: Some(1.4),
                    confidence: Some(0.8),
                    speaker: Some("speaker-a".to_string()),
                    attributes: BTreeMap::from([("token".to_string(), "1".to_string())]),
                }],
                chars: vec![TimedTextCharContract {
                    character: "h".to_string(),
                    start_seconds: Some(1.0),
                    end_seconds: Some(1.05),
                    confidence: Some(0.7),
                    attributes: BTreeMap::new(),
                }],
                attributes: BTreeMap::from([("segment".to_string(), "7".to_string())]),
            }],
            source: Some("clip.wav".to_string()),
            attributes: BTreeMap::from([("producer".to_string(), "audio".to_string())]),
        };

        let nlp = timed_text_to_transcription(neutral.clone()).unwrap();
        let round_trip = transcription_to_timed_text(nlp).unwrap();

        assert_eq!(round_trip, neutral);
    }
}
