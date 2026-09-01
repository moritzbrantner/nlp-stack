use jobs_core::OperationResult;
use runtime_core::{
    cli::{self, CliAdapterMetadata},
    describe_surface_response, structured_surface_response, surface_operation, PackageSurface,
    Diagnostic, DiagnosticSeverity, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::Deserialize;

/// Wrapped library crate name.
pub const LIBRARY_CRATE: &str = "text-core";
/// Adapter surface kind.
pub const SURFACE_KIND: &str = "cli";
/// Rust import path for the wrapped crate.
pub const LIBRARY_IMPORT: &str = "use text_core";
/// Companion server package name.
pub const SERVER_PACKAGE: &str = "text-core-server";
/// Companion React app package name.
pub const APP_PACKAGE: &str = "text-core-app";
/// Companion WASM package name.
pub const WASM_PACKAGE: &str = "text-core-wasm";

const METADATA: CliAdapterMetadata = CliAdapterMetadata {
    library_crate: LIBRARY_CRATE,
    surface_kind: SURFACE_KIND,
    library_import: LIBRARY_IMPORT,
    server_package: SERVER_PACKAGE,
    app_package: APP_PACKAGE,
    wasm_package: WASM_PACKAGE,
};

pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: "moenarch-text-core".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Inspect package metadata",
                "Shared text documents, tokenization, spans, and statistics.",
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
    surface_operation(id, name, description, example_request)
}

pub fn package_metadata_json() -> String {
    cli::package_metadata_json(METADATA, package_surface())
}

pub fn command_schema_json() -> String {
    cli::command_schema_json()
}

pub fn run_operation(operation: &str, input: serde_json::Value) -> Result<SurfaceResponse, String> {
    cli::run_wrapped_operation(operation, input, run_surface_operation)
}

/// Runs the compatibility transport surface from the CLI boundary.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&package_surface(), request)),
        "text.statistics" => statistics_value(parse_input(request.input)?)?,
        "text.normalize" => normalize_value(parse_input(request.input)?)?,
        "text.tokenize" => tokenize_value(parse_input(request.input)?)?,
        "text.boundaries" => boundaries_value(parse_input(request.input)?)?,
        operation => {
            return Err(runtime_core::SurfaceError::unsupported_operation(
                operation,
                LIBRARY_CRATE,
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
struct StatisticsRequest {
    text: String,
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

fn statistics_value(request: StatisticsRequest) -> Result<serde_json::Value, String> {
    let stats = text_core::text_stats(&request.text);
    let mut diagnostics = Vec::new();
    if request.text.is_empty() {
        diagnostics.push(Diagnostic::new(
            DiagnosticSeverity::Warning,
            "text.empty",
            "Input text is empty.",
        ));
    }
    let result = OperationResult {
        value: Some(serde_json::json!({
            "byteCount": stats.bytes,
            "characterCount": stats.chars,
            "wordCount": stats.words,
            "lineCount": stats.lines,
            "sentenceCount": stats.sentences,
        })),
        diagnostics,
        artifacts: Vec::new(),
    };
    serde_json::to_value(result).map_err(|error| error.to_string())
}

fn normalize_value(request: NormalizeRequest) -> Result<serde_json::Value, String> {
    let before =
        text_core::detailed_text_stats(&request.text, &text_core::TextProcessingOptions::default());
    let mut normalized = text_core::normalize_text(
        &request.text,
        &text_core::TextProcessingOptions {
            lowercase: request.lowercase,
            ..text_core::TextProcessingOptions::default()
        },
    );
    if request.strip_diacritics {
        normalized = normalized.chars().filter(|ch| ch.is_ascii()).collect();
    }
    if request.normalize_whitespace {
        normalized = text_core::normalize_whitespace(&normalized);
    }
    let after =
        text_core::detailed_text_stats(&normalized, &text_core::TextProcessingOptions::default());
    Ok(serde_json::json!({"text": normalized, "before": before, "after": after}))
}

fn tokenize_value(request: TokenizeRequest) -> Result<serde_json::Value, String> {
    let options = text_core::TextProcessingOptions {
        lowercase: request.lowercase,
        include_punctuation: request.include_punctuation || request.include_whitespace,
        ..text_core::TextProcessingOptions::default()
    };
    Ok(serde_json::json!({
        "tokens": text_core::tokenize(&request.text, &options),
        "scriptProfile": text_core::detect_script_profile(&request.text),
        "stats": text_core::detailed_text_stats(&request.text, &options),
    }))
}

fn boundaries_value(request: BoundariesRequest) -> Result<serde_json::Value, String> {
    let processing = text_core::TextProcessingOptions {
        keep_apostrophes: request.keep_apostrophes,
        include_punctuation: true,
        ..text_core::TextProcessingOptions::default()
    };
    let boundary_options = text_core::TextBoundaryOptions {
        include_punctuation: false,
        ..text_core::TextBoundaryOptions::default()
    };
    Ok(serde_json::json!({
        "words": text_core::segment_words(&request.text, &boundary_options),
        "sentences": text_core::split_sentence_spans(&request.text, &processing),
        "paragraphs": text_core::split_paragraphs(&request.text),
        "graphemes": text_core::segment_graphemes(&request.text),
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

    #[test]
    fn metadata_mentions_wrapped_library() {
        let metadata = package_metadata_json();
        assert!(metadata.contains(LIBRARY_CRATE));
        assert!(metadata.contains(SURFACE_KIND));
    }

    #[test]
    fn statistics_preserves_the_operation_result_envelope() {
        let response = run_surface_operation(SurfaceRequest {
            operation: runtime_core::OperationId::new("text.statistics"),
            input: serde_json::json!({"text": "one two"}),
        })
        .unwrap();

        assert_eq!(response.value["value"]["wordCount"], 2);
        assert_eq!(response.value["diagnostics"], serde_json::json!([]));
        assert_eq!(response.value["artifacts"], serde_json::json!([]));
    }

    #[test]
    fn statistics_preserves_empty_input_diagnostics() {
        let response = run_surface_operation(SurfaceRequest {
            operation: runtime_core::OperationId::new("text.statistics"),
            input: serde_json::json!({"text": ""}),
        })
        .unwrap();

        assert_eq!(response.value["diagnostics"][0]["code"], "text.empty");
    }
}
