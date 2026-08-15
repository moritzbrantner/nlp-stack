//! Library-owned runtime surface for `text-core`.

use runtime_core::{
    describe_surface_response, structured_surface_response, surface_operation, PackageSurface,
    RuntimeCapabilities, SurfaceOperation, SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;

use crate::{
    detailed_text_stats, detect_script_profile, normalize_text, normalize_whitespace,
    operations::analyze_text_statistics, segment_graphemes, segment_words, split_paragraphs,
    split_sentence_spans, tokenize, TextBoundaryOptions, TextProcessingOptions,
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
                "Shared text documents, tokenization, spans, and statistics for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "text.statistics",
                "Text statistics",
                "Counts bytes, characters, words, lines, and sentences.",
                serde_json::json!({"text": "Hello world. Again."}),
            ),
            operation(
                "text.normalize",
                "Normalize text",
                "Normalizes Unicode, casing, and whitespace with before/after statistics.",
                serde_json::json!({"text": "  Hello   WORLD  ", "lowercase": true, "normalizeWhitespace": true}),
            ),
            operation(
                "text.tokenize",
                "Tokenize text",
                "Returns span-aware tokens, script profile, and detailed text statistics.",
                serde_json::json!({"text": "Hello, Berlin 2026.", "includePunctuation": true}),
            ),
            operation(
                "text.boundaries",
                "Text boundaries",
                "Returns Unicode-safe word, sentence, paragraph, and grapheme boundaries.",
                serde_json::json!({"text": "Hello world. Second paragraph."}),
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
    let mut operation = surface_operation(id, name, description, example_request);
    if id == "text.tokenize" {
        runtime_core::attach_landscape_contract(
            &mut operation,
            runtime_core::landscape::LandscapeOperationContract::new(
                runtime_core::landscape::LandscapeFunction::new(
                    "text.core.tokenize",
                    env!("CARGO_PKG_NAME"),
                )
                .input(runtime_core::landscape::LandscapePort::new(
                    "document",
                    runtime_core::landscape::well_known::text_document(),
                ))
                .output(
                    runtime_core::landscape::LandscapePort::new(
                        "segments",
                        runtime_core::landscape::well_known::text_segment(),
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
        "describe" => return Ok(describe_surface_response(&package_surface(), request)),
        "text.statistics" => {
            let result = analyze_text_statistics(parse_input(request.input)?);
            serde_json::to_value(result).map_err(|error| error.to_string())?
        }
        "text.normalize" => normalize_value(parse_input(request.input)?)?,
        "text.tokenize" => tokenize_value(parse_input(request.input)?)?,
        "text.boundaries" => boundaries_value(parse_input(request.input)?)?,
        operation => {
            return Err(runtime_core::SurfaceError::unsupported_operation(
                operation,
                env!("CARGO_PKG_NAME"),
            )
            .to_error_string());
        }
    };
    Ok(structured_surface_response(
        operation.clone(),
        workflow_title(operation.as_str()),
        workflow_message(operation.as_str()),
        workflow_summary(operation.as_str(), &value),
        value,
    ))
}

fn workflow_title(operation: &str) -> &'static str {
    match operation {
        "text.statistics" => "Text statistics",
        "text.normalize" => "Normalized text",
        "text.tokenize" => "Tokenized text",
        "text.boundaries" => "Text boundaries",
        _ => "Text core result",
    }
}

fn workflow_message(operation: &str) -> &'static str {
    match operation {
        "text.statistics" => {
            "Computed deterministic byte, character, word, line, and sentence statistics."
        }
        "text.normalize" => "Normalized text with explicit before and after statistics.",
        "text.tokenize" => {
            "Tokenized the supplied text with spans, script profile, and text statistics."
        }
        "text.boundaries" => {
            "Extracted Unicode-safe word, sentence, paragraph, and grapheme boundaries."
        }
        _ => "Ran a text-core package operation.",
    }
}

fn workflow_summary(operation: &str, value: &serde_json::Value) -> serde_json::Value {
    match operation {
        "text.statistics" => serde_json::json!({
            "status": "ok",
            "words": value["value"]["wordCount"],
            "sentences": value["value"]["sentenceCount"]
        }),
        "text.normalize" => serde_json::json!({
            "status": "ok",
            "inputWords": value["before"]["basic"]["words"],
            "outputWords": value["after"]["basic"]["words"]
        }),
        "text.tokenize" => serde_json::json!({
            "status": "ok",
            "tokenCount": value["tokens"].as_array().map(Vec::len).unwrap_or(0),
            "dominantScript": value["scriptProfile"]["dominantScript"]
        }),
        "text.boundaries" => serde_json::json!({
            "status": "ok",
            "wordCount": value["words"].as_array().map(Vec::len).unwrap_or(0),
            "sentenceCount": value["sentences"].as_array().map(Vec::len).unwrap_or(0),
            "paragraphCount": value["paragraphs"].as_array().map(Vec::len).unwrap_or(0)
        }),
        _ => serde_json::json!({"status": "ok"}),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NormalizeRequest {
    text: String,
    #[serde(default = "default_true")]
    lowercase: bool,
    #[serde(default)]
    strip_diacritics: bool,
    #[serde(default = "default_true")]
    normalize_whitespace: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TokenizeRequest {
    text: String,
    #[serde(default)]
    include_whitespace: bool,
    #[serde(default)]
    include_punctuation: bool,
    #[serde(default = "default_true")]
    lowercase: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoundariesRequest {
    text: String,
    #[serde(default = "default_true")]
    keep_apostrophes: bool,
}

fn normalize_value(request: NormalizeRequest) -> Result<serde_json::Value, String> {
    let before = detailed_text_stats(&request.text, &TextProcessingOptions::default());
    let mut normalized = normalize_text(
        &request.text,
        &TextProcessingOptions {
            lowercase: request.lowercase,
            ..TextProcessingOptions::default()
        },
    );
    if request.strip_diacritics {
        normalized = normalized.chars().filter(|ch| ch.is_ascii()).collect();
    }
    if request.normalize_whitespace {
        normalized = normalize_whitespace(&normalized);
    }
    let after = detailed_text_stats(&normalized, &TextProcessingOptions::default());
    Ok(serde_json::json!({
        "text": normalized,
        "before": before,
        "after": after
    }))
}

fn tokenize_value(request: TokenizeRequest) -> Result<serde_json::Value, String> {
    let options = TextProcessingOptions {
        lowercase: request.lowercase,
        include_punctuation: request.include_punctuation || request.include_whitespace,
        ..TextProcessingOptions::default()
    };
    Ok(serde_json::json!({
        "tokens": tokenize(&request.text, &options),
        "scriptProfile": detect_script_profile(&request.text),
        "stats": detailed_text_stats(&request.text, &options)
    }))
}

fn boundaries_value(request: BoundariesRequest) -> Result<serde_json::Value, String> {
    let processing = TextProcessingOptions {
        keep_apostrophes: request.keep_apostrophes,
        include_punctuation: true,
        ..TextProcessingOptions::default()
    };
    let boundary_options = TextBoundaryOptions {
        include_punctuation: false,
        ..TextBoundaryOptions::default()
    };
    Ok(serde_json::json!({
        "words": segment_words(&request.text, &boundary_options),
        "sentences": split_sentence_spans(&request.text, &processing),
        "paragraphs": split_paragraphs(&request.text),
        "graphemes": segment_graphemes(&request.text)
    }))
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    runtime_core::parse_surface_input(None, input)
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_core::OperationId;

    #[test]
    fn package_surface_lists_text_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();

        assert!(ids.contains(&"text.statistics".to_string()));
        assert!(ids.contains(&"text.tokenize".to_string()));
    }

    #[test]
    fn tokenization_operation_returns_stable_tokens() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("text.tokenize"),
            input: serde_json::json!({"text": "Hello Berlin.", "includePunctuation": true}),
        })
        .expect("tokenize");

        assert_eq!(response.value["tokens"][0]["normalized"], "hello");
        assert_eq!(response.value["stats"]["basic"]["words"], 2);
    }

    #[test]
    fn malformed_input_returns_typed_error_string() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("text.statistics"),
            input: serde_json::json!({"missing": true}),
        })
        .expect_err("invalid request");

        assert!(error.contains("invalid request"));
    }
}
