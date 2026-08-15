//! Library-owned runtime surface for `text-transcripts`.

use runtime_core::{
    structured_surface_value, OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation,
    SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;
use text_core::TextSegmentContract;

use crate::{
    format_srt, format_webvtt, normalize_transcription_contract, parse_plain_lines, parse_srt,
    parse_webvtt, parse_whisper_json, parse_whisperx_json, TranscriptionContract,
    TranscriptionResult,
};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Inspect package metadata",
                "Transcript parsing and ASR command adapters for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "transcripts.parse",
                "Parse transcript",
                "Parses plain text, Whisper JSON, SRT, or WebVTT into the transcript contract.",
                serde_json::json!({"format": "srt", "content": "1\n00:00:01,000 --> 00:00:02,000\nHello.\n"}),
            ),
            operation(
                "transcripts.normalize",
                "Normalize transcript",
                "Normalizes transcript contract text, segments, words, and confidence.",
                serde_json::json!({"segments": [{"index": 0, "text": " hello ", "isFinal": true}]}),
            ),
            operation(
                "transcripts.importWhisperX",
                "Import WhisperX JSON",
                "Parses existing WhisperX JSON output into the normalized transcript contract without running external tools.",
                serde_json::json!({"content": "{\"segments\":[{\"start\":0.0,\"end\":1.0,\"text\":\"Hello.\",\"words\":[{\"word\":\"Hello\",\"start\":0.0,\"end\":0.8,\"score\":0.9}]}]}"}),
            ),
            operation(
                "transcripts.formatSrt",
                "Format SRT",
                "Formats a transcript contract as SRT text.",
                serde_json::json!({"segments": [{"index": 0, "startSeconds": 1.0, "endSeconds": 2.0, "text": "Hello.", "isFinal": true}]}),
            ),
            operation(
                "transcripts.formatWebVtt",
                "Format WebVTT",
                "Formats a transcript contract as WebVTT text.",
                serde_json::json!({"segments": [{"index": 0, "startSeconds": 1.0, "endSeconds": 2.0, "text": "Hello.", "isFinal": true}]}),
            ),
            operation(
                "transcripts.toTextSegments",
                "Convert to text segments",
                "Converts transcript segments into shared text segment contracts and document records.",
                serde_json::json!({"streamId": "transcript-1", "segments": [{"index": 0, "startSeconds": 1.0, "endSeconds": 2.0, "text": "Hello.", "language": "en", "speaker": "A", "confidence": 0.9, "isFinal": true}]}),
            ),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    let mut operation = runtime_core::surface_operation(id, name, description, example_request);
    if id == "transcripts.toTextSegments" {
        runtime_core::attach_landscape_contract(
            &mut operation,
            runtime_core::landscape::LandscapeOperationContract::new(
                runtime_core::landscape::LandscapeFunction::new(
                    "text.transcripts.toTextSegments",
                    env!("CARGO_PKG_NAME"),
                )
                .input(
                    runtime_core::landscape::LandscapePort::new(
                        "segments",
                        runtime_core::landscape::well_known::text_transcript_segment(),
                    )
                    .many(),
                )
                .output(
                    runtime_core::landscape::LandscapePort::new(
                        "segments",
                        runtime_core::landscape::well_known::text_segment(),
                    )
                    .many(),
                )
                .output(
                    runtime_core::landscape::LandscapePort::new(
                        "documents",
                        runtime_core::landscape::well_known::text_document(),
                    )
                    .many(),
                ),
            ),
        );
    }
    operation
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "transcripts.parse" => parse_value(parse_input(request.input)?)?,
        "transcripts.normalize" => normalize_value(parse_input(request.input)?)?,
        "transcripts.importWhisperX" => import_whisperx_value(parse_input(request.input)?)?,
        "transcripts.formatSrt" => format_srt_value(parse_input(request.input)?)?,
        "transcripts.formatWebVtt" => format_webvtt_value(parse_input(request.input)?)?,
        "transcripts.toTextSegments" => to_text_segments_value(parse_input(request.input)?)?,
        operation => {
            return Err(runtime_core::SurfaceError::unsupported_operation(
                operation,
                env!("CARGO_PKG_NAME"),
            )
            .to_error_string())
        }
    };
    let value = annotated_value(&operation, value);
    Ok(SurfaceResponse {
        operation,
        value,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    })
}

fn describe_value(input: serde_json::Value) -> serde_json::Value {
    let surface = package_surface();
    serde_json::json!({
        "library": surface.library,
        "version": surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface.operations.iter().map(|operation| operation.id.as_str()).collect::<Vec<_>>(),
        "input": input
    })
}

fn annotated_value(operation: &OperationId, value: serde_json::Value) -> serde_json::Value {
    let (title, message, summary) = match operation.as_str() {
        "describe" => (
            "Package surface metadata",
            "Inspected the text-transcripts package operations and runtime support.",
            serde_json::json!({
                "status": "ok",
                "operationCount": value["operationCount"]
            }),
        ),
        "transcripts.parse" => (
            "Transcript parse result",
            "Parsed transcript content into the normalized transcript contract.",
            serde_json::json!({
                "status": "ok",
                "segmentCount": value["segments"].as_array().map(Vec::len).unwrap_or(0),
                "hasText": value["text"].as_str().map(|text| !text.is_empty()).unwrap_or(false)
            }),
        ),
        "transcripts.normalize" => (
            "Transcript normalization result",
            "Normalized transcript contract text, segments, words, and confidence values.",
            serde_json::json!({
                "status": "ok",
                "segmentCount": value["segments"].as_array().map(Vec::len).unwrap_or(0),
                "hasText": value["text"].as_str().map(|text| !text.is_empty()).unwrap_or(false)
            }),
        ),
        "transcripts.importWhisperX" => (
            "WhisperX import result",
            "Imported WhisperX JSON into the normalized transcript contract.",
            serde_json::json!({
                "status": "ok",
                "segmentCount": value["segments"].as_array().map(Vec::len).unwrap_or(0),
                "hasText": value["text"].as_str().map(|text| !text.is_empty()).unwrap_or(false)
            }),
        ),
        "transcripts.formatSrt" => (
            "SRT formatting result",
            "Formatted a normalized transcript contract as SRT text.",
            serde_json::json!({
                "status": "ok",
                "bytes": value["srt"].as_str().map(str::len).unwrap_or(0)
            }),
        ),
        "transcripts.formatWebVtt" => (
            "WebVTT formatting result",
            "Formatted a normalized transcript contract as WebVTT text.",
            serde_json::json!({
                "status": "ok",
                "bytes": value["webVtt"].as_str().map(str::len).unwrap_or(0)
            }),
        ),
        "transcripts.toTextSegments" => (
            "Text segment conversion result",
            "Converted normalized transcript segments into shared text segment and document contracts.",
            serde_json::json!({
                "status": "ok",
                "segmentCount": value["segments"].as_array().map(Vec::len).unwrap_or(0),
                "documentCount": value["documents"].as_array().map(Vec::len).unwrap_or(0),
                "streamId": value["streamId"]
            }),
        ),
        _ => (
            "Transcript result",
            "Ran a text-transcripts package operation.",
            serde_json::json!({"status": "ok"}),
        ),
    };
    structured_surface_value(operation, title, message, summary, value)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ParseRequest {
    format: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImportContentRequest {
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToTextSegmentsRequest {
    stream_id: Option<String>,
    #[serde(flatten)]
    contract: TranscriptionContract,
}

fn parse_value(request: ParseRequest) -> Result<serde_json::Value, String> {
    let result = match request.format.as_str() {
        "plain" | "lines" => parse_plain_lines(&request.content),
        "whisperJson" | "whisper_json" | "whisper-json" => {
            parse_whisper_json(request.content.as_bytes()).map_err(|error| error.to_string())?
        }
        "srt" => parse_srt(&request.content).map_err(|error| error.to_string())?,
        "webVtt" | "webvtt" | "web-vtt" => {
            parse_webvtt(&request.content).map_err(|error| error.to_string())?
        }
        other => return Err(format!("unsupported transcript format `{other}`")),
    };
    let mut contract = TranscriptionContract::from(result)
        .normalized()
        .map_err(|error| error.to_string())?;
    let joined = contract.joined_text();
    if !joined.is_empty() {
        contract.text = Some(joined);
    }
    Ok(serde_json::json!(contract))
}

fn normalize_value(contract: TranscriptionContract) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!(normalize_transcription_contract(
        contract
    )
    .map_err(|error| error.to_string())?))
}

fn import_whisperx_value(request: ImportContentRequest) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!(parse_whisperx_json(
        request.content.as_bytes()
    )
    .map_err(|error| error.to_string())?))
}

fn format_srt_value(contract: TranscriptionContract) -> Result<serde_json::Value, String> {
    let normalized = contract.normalized().map_err(|error| error.to_string())?;
    let result = TranscriptionResult::from(normalized);
    Ok(serde_json::json!({ "srt": format_srt(&result.segments) }))
}

fn format_webvtt_value(contract: TranscriptionContract) -> Result<serde_json::Value, String> {
    let normalized = contract.normalized().map_err(|error| error.to_string())?;
    let result = TranscriptionResult::from(normalized);
    Ok(serde_json::json!({ "webVtt": format_webvtt(&result.segments) }))
}

fn to_text_segments_value(request: ToTextSegmentsRequest) -> Result<serde_json::Value, String> {
    let normalized = request
        .contract
        .normalized()
        .map_err(|error| error.to_string())?;
    let segments = normalized
        .segments
        .iter()
        .map(|segment| {
            let mut text_segment = TextSegmentContract::from(segment);
            if let Some(stream_id) = &request.stream_id {
                text_segment.stream_id = Some(stream_id.clone());
            }
            text_segment
        })
        .collect::<Vec<_>>();
    let documents = segments
        .iter()
        .filter_map(|segment| {
            segment.document_id().map(|id| {
                serde_json::json!({
                    "id": id,
                    "text": segment.text
                })
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "streamId": request.stream_id,
        "segments": segments,
        "documents": documents
    }))
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    runtime_core::parse_surface_input(None, input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_transcript_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"transcripts.parse".to_string()));
        assert!(ids.contains(&"transcripts.importWhisperX".to_string()));
        assert!(ids.contains(&"transcripts.formatSrt".to_string()));
        assert!(ids.contains(&"transcripts.formatWebVtt".to_string()));
        assert!(ids.contains(&"transcripts.toTextSegments".to_string()));
    }

    #[test]
    fn parse_operation_normalizes_plain_lines() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("transcripts.parse"),
            input: serde_json::json!({"format": "lines", "content": "hello\n\nworld\n"}),
        })
        .expect("parse");
        assert_eq!(response.value["text"], "hello world");
        assert_eq!(response.value["segments"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn malformed_input_returns_typed_error_string() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("transcripts.parse"),
            input: serde_json::json!({"format": "srt"}),
        })
        .expect_err("invalid request");
        assert!(error.contains("invalid request"));
    }

    #[test]
    fn format_webvtt_operation_returns_cue_text() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("transcripts.formatWebVtt"),
            input: serde_json::json!({"segments": [{"index": 0, "startSeconds": 1.0, "endSeconds": 2.0, "text": "Hello.", "isFinal": true}]}),
        })
        .expect("format webvtt");
        let webvtt = response.value["result"]["webVtt"].as_str().unwrap();
        assert!(webvtt.starts_with("WEBVTT"));
        assert!(webvtt.contains("Hello."));
    }

    #[test]
    fn to_text_segments_sets_stream_and_documents() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("transcripts.toTextSegments"),
            input: serde_json::json!({
                "streamId": "transcript-1",
                "segments": [{
                    "index": 0,
                    "startSeconds": 1.0,
                    "endSeconds": 2.0,
                    "text": "Hello.",
                    "language": "en",
                    "speaker": "A",
                    "confidence": 0.9,
                    "isFinal": true
                }]
            }),
        })
        .expect("to text segments");
        assert_eq!(
            response.value["result"]["segments"][0]["streamId"],
            "transcript-1"
        );
        assert_eq!(
            response.value["result"]["documents"][0]["id"],
            "transcript-1:0"
        );
    }
}
