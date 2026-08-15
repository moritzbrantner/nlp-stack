//! Library-owned runtime surface for `text-model-runtime`.

use runtime_core::{
    structured_surface_value, OperationId, PackageSurface, RuntimeCapabilities, RuntimeRequirement,
    SurfaceExecutionMode, SurfaceExecutionPlan, SurfaceOperation, SurfaceRequest, SurfaceResponse,
    SurfaceSideEffect,
};
use serde::Deserialize;

use std::path::PathBuf;

use crate::{
    softmax, validate_text_model_bundle, validate_tokenizer_bundle, TextModelBundleCheck,
    TextModelCapability, TokenizedText,
};
#[cfg(feature = "local-onnx")]
use crate::{OnnxQuestionAnswerer, QuestionAnsweringBackend, QuestionAnsweringDecodeOptions};
#[cfg(feature = "model-bundles")]
use model_runtime::{resolve_or_download_bundle, ModelBundleResolveOptions, ModelPreset};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation("describe", "Inspect package metadata", "Shared tokenizer and native text model runtime traits for video-analysis.", serde_json::json!({"includeOperations": true})),
            operation("runtime.tokenizeSummary", "Tokenize summary", "Builds a deterministic TokenizedText-like whitespace-token summary without downloads or native runtime execution.", serde_json::json!({"text": "Rust text runtime", "maxTokens": 8})),
            operation("runtime.bundleCheck", "Check model bundle", "Validates required local model bundle files without downloads or native runtime execution.", serde_json::json!({"modelId": "demo-tokenizer", "capability": "tokenizer", "bundleRoot": ".model-runtime/demo-tokenizer", "requiredFiles": ["tokenizer.json"]})),
            model_operation("runtime.downloadBundle", "Download model bundle", "Downloads and materializes a Hugging Face model preset bundle through model-runtime.", serde_json::json!({"modelId": "roberta-base-squad2-onnx", "bundleRoot": ".model-runtime", "autoDownload": true})),
            model_operation("runtime.onnxQaProbe", "Probe ONNX QA", "Auto-downloads the default RoBERTa SQuAD2 ONNX bundle when needed and runs one local extractive QA inference.", serde_json::json!({"question": "What language is reliable?", "context": "Rust is reliable for systems programming.", "topK": 1})),
            operation("runtime.tokenizerProbe", "Probe tokenizer bundle", "Validates a local tokenizer file and performs a sample tokenization only when the tokenizers feature is available.", serde_json::json!({"modelId": "demo-tokenizer", "tokenizerPath": ".model-runtime/demo-tokenizer/tokenizer.json", "sample": "Rust text runtime"})),
            operation("runtime.softmax", "Normalize logits", "Normalizes logits with a stable softmax helper.", serde_json::json!({"logits": [0.0, 1.0]})),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    runtime_core::surface_operation(id, name, description, example_request)
}

fn model_operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    let mut operation = runtime_core::surface_operation_with_execution_plan(
        id,
        name,
        description,
        example_request,
        model_execution_plan(id),
    );
    operation.wasm_supported = false;
    operation.server_supported = true;
    operation
}

fn model_execution_plan(operation: &str) -> SurfaceExecutionPlan {
    SurfaceExecutionPlan {
        operation: OperationId::new(operation),
        mode: SurfaceExecutionMode::InMemory,
        side_effects: vec![
            SurfaceSideEffect::ReadsFiles,
            SurfaceSideEffect::WritesFiles,
            SurfaceSideEffect::Network,
        ],
        cancellable: false,
        progress_unit: Some("files".to_string()),
        expected_artifacts: Vec::new(),
        requirements: vec![
            RuntimeRequirement {
                name: "onnxruntime".to_string(),
                description: Some("ONNX Runtime CPU execution".to_string()),
                required: true,
            },
            RuntimeRequirement {
                name: "local-filesystem".to_string(),
                description: Some("Readable and writable .model-runtime bundle root".to_string()),
                required: true,
            },
            RuntimeRequirement {
                name: "hugging-face-access".to_string(),
                description: Some("Network access for first-run model downloads".to_string()),
                required: true,
            },
        ],
        max_recommended_input_bytes: Some(1_048_576),
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "runtime.tokenizeSummary" => tokenize_summary_value(parse_input(request.input)?)?,
        "runtime.bundleCheck" => bundle_check_value(parse_input(request.input)?)?,
        "runtime.downloadBundle" => download_bundle_value(parse_input(request.input)?)?,
        "runtime.onnxQaProbe" => onnx_qa_probe_value(parse_input(request.input)?)?,
        "runtime.tokenizerProbe" => tokenizer_probe_value(parse_input(request.input)?)?,
        "runtime.softmax" => softmax_value(parse_input(request.input)?)?,
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
            "Inspected the text-model-runtime package operations and runtime support.",
            serde_json::json!({
                "status": "ok",
                "operationCount": value["operationCount"]
            }),
        ),
        "runtime.tokenizeSummary" => (
            "Tokenizer summary result",
            "Built a deterministic whitespace-token summary without downloads or native model execution.",
            serde_json::json!({
                "status": "ok",
                "backend": value["backend"],
                "tokenCount": value["tokens"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "runtime.bundleCheck" => (
            "Model bundle readiness result",
            "Validated required local model bundle files without downloads or native runtime execution.",
            serde_json::json!({
                "status": "ok",
                "modelId": value["modelId"],
                "loadable": value["loadable"],
                "missingFileCount": value["missingFiles"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "runtime.downloadBundle" => (
            "Model bundle download result",
            "Resolved or materialized a model bundle through model-runtime.",
            serde_json::json!({
                "status": "ok",
                "modelId": value["modelId"],
                "bundlePath": value["bundlePath"]
            }),
        ),
        "runtime.onnxQaProbe" => (
            "ONNX QA probe result",
            "Ran or attempted a server-only local ONNX question-answering probe.",
            serde_json::json!({
                "status": "ok",
                "modelId": value["modelId"],
                "answerCount": value["answers"].as_array().map(Vec::len).unwrap_or(0),
                "runtime": value["runtime"]
            }),
        ),
        "runtime.tokenizerProbe" => (
            "Tokenizer probe result",
            "Validated a local tokenizer bundle and reported whether a sample tokenization could run.",
            serde_json::json!({
                "status": "ok",
                "modelId": value["report"]["modelId"],
                "loadable": value["report"]["loadable"],
                "ran": value["run"].is_object()
            }),
        ),
        "runtime.softmax" => (
            "Softmax support result",
            "Normalized logits with the shared stable softmax helper.",
            serde_json::json!({
                "status": "ok",
                "probabilityCount": value["probabilities"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        _ => (
            "Text model runtime result",
            "Ran a text-model-runtime package operation.",
            serde_json::json!({"status": "ok"}),
        ),
    };
    structured_surface_value(operation, title, message, summary, value)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TokenizeSummaryRequest {
    text: String,
    max_tokens: Option<usize>,
    #[serde(default = "default_truncation")]
    truncation: String,
}

#[derive(Debug, Deserialize)]
struct SoftmaxRequest {
    logits: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BundleCheckRequest {
    model_id: String,
    capability: TextModelCapability,
    bundle_root: PathBuf,
    required_files: Vec<String>,
    required_feature: Option<String>,
    required_setup: Option<String>,
    smoke_operation: Option<String>,
    #[serde(default = "default_supported")]
    supported: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct LocalModelRequest {
    model_id: Option<String>,
    bundle_root: Option<PathBuf>,
    auto_download: Option<bool>,
    download_progress: Option<bool>,
    cache_dir: Option<PathBuf>,
    hf_token: Option<String>,
    max_retries: Option<usize>,
    overwrite: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct OnnxQaProbeRequest {
    question: String,
    context: String,
    top_k: Option<usize>,
    max_answer_len: Option<usize>,
    min_score: Option<f32>,
    #[serde(flatten)]
    model: LocalModelRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TokenizerProbeRequest {
    model_id: String,
    tokenizer_path: PathBuf,
    sample: String,
}

fn tokenize_summary_value(request: TokenizeSummaryRequest) -> Result<serde_json::Value, String> {
    let mut pieces = whitespace_tokens(&request.text);
    if let Some(max_tokens) = request.max_tokens {
        match request.truncation.as_str() {
            "none" if pieces.len() > max_tokens => {}
            "none" => {}
            "head" | "onlyFirst" | "only_first" => pieces.truncate(max_tokens),
            "tail" => {
                if pieces.len() > max_tokens {
                    pieces = pieces[pieces.len() - max_tokens..].to_vec();
                }
            }
            other => return Err(format!("unsupported truncation strategy `{other}`")),
        }
    }
    let tokenized = TokenizedText {
        input_ids: (0..pieces.len()).map(|index| index as i64 + 1).collect(),
        attention_mask: vec![1; pieces.len()],
        token_type_ids: None,
        offsets: pieces
            .iter()
            .map(|(_, start, end)| Some((*start, *end)))
            .collect(),
    };
    Ok(serde_json::json!({
        "backend": "deterministic-whitespace",
        "tokens": pieces.into_iter().map(|(token, start, end)| serde_json::json!({
            "text": token,
            "byteStart": start,
            "byteEnd": end
        })).collect::<Vec<_>>(),
        "tokenized": tokenized
    }))
}

fn softmax_value(request: SoftmaxRequest) -> Result<serde_json::Value, String> {
    runtime_core::require_non_empty("runtime.softmax", "logits", &request.logits)?;
    Ok(serde_json::json!({ "probabilities": softmax(&request.logits) }))
}

fn bundle_check_value(request: BundleCheckRequest) -> Result<serde_json::Value, String> {
    require_non_empty_string("runtime.bundleCheck", "modelId", &request.model_id)?;
    require_non_empty_path("runtime.bundleCheck", "bundleRoot", &request.bundle_root)?;
    runtime_core::require_non_empty(
        "runtime.bundleCheck",
        "requiredFiles",
        &request.required_files,
    )?;
    for file in &request.required_files {
        require_non_empty_string("runtime.bundleCheck", "requiredFiles[]", file)?;
    }

    let mut check = TextModelBundleCheck::new(
        request.model_id,
        request.capability,
        request.bundle_root,
        request.required_files,
    );
    if let Some(feature) = request.required_feature {
        check = check.required_feature(feature);
    }
    if let Some(setup) = request.required_setup {
        check = check.required_setup(setup);
    }
    if let Some(operation) = request.smoke_operation {
        check = check.smoke_operation(operation);
    }
    if !request.supported {
        check = check.reference_only();
    }

    serde_json::to_value(validate_text_model_bundle(check)).map_err(|error| error.to_string())
}

fn download_bundle_value(request: LocalModelRequest) -> Result<serde_json::Value, String> {
    #[cfg(feature = "model-bundles")]
    {
        let (model_id, spec, options) = model_resolve_parts(request)?;
        let bundle =
            resolve_or_download_bundle(&spec, &options).map_err(|error| error.to_string())?;
        Ok(serde_json::json!({
            "modelId": model_id,
            "repoId": bundle.manifest.repo_id,
            "bundlePath": bundle.root,
            "manifestPath": bundle.manifest_path(),
            "runtime": "model-runtime"
        }))
    }
    #[cfg(not(feature = "model-bundles"))]
    {
        Ok(serde_json::json!({
            "executed": false,
            "serverOnly": true,
            "wasmSupported": false,
            "serverSupported": true,
            "modelId": request.model_id.unwrap_or_else(|| "roberta-base-squad2-onnx".to_string()),
            "bundlePath": request.bundle_root.unwrap_or_else(|| PathBuf::from(".model-runtime")),
            "runtime": "model-runtime",
            "diagnostics": ["runtime.downloadBundle requires the `model-bundles` feature"]
        }))
    }
}

fn onnx_qa_probe_value(request: OnnxQaProbeRequest) -> Result<serde_json::Value, String> {
    require_non_empty_string("runtime.onnxQaProbe", "question", &request.question)?;
    require_non_empty_string("runtime.onnxQaProbe", "context", &request.context)?;
    let top_k = request.top_k.unwrap_or(3).max(1);
    #[cfg(feature = "local-onnx")]
    {
        let (model_id, spec, options) = model_resolve_parts(request.model)?;
        let bundle =
            resolve_or_download_bundle(&spec, &options).map_err(|error| error.to_string())?;
        let bundle_path = bundle.root.clone();
        let manifest_path = bundle.manifest_path();
        let mut qa = OnnxQuestionAnswerer::from_bundle(bundle)
            .map_err(|error| error.to_string())?
            .with_decode_options(QuestionAnsweringDecodeOptions {
                max_answer_len: request.max_answer_len.unwrap_or(30),
                top_k,
                min_score: request.min_score,
            });
        let answers = qa
            .answer(&request.question, &request.context)
            .map_err(|error| error.to_string())?;
        Ok(serde_json::json!({
            "loadReport": {
                "loadable": true,
                "bundlePresent": true,
                "manifestPath": manifest_path
            },
            "answers": answers,
            "modelId": model_id,
            "repoId": spec.repo_id_value(),
            "bundlePath": bundle_path,
            "runtime": "onnx"
        }))
    }
    #[cfg(not(feature = "local-onnx"))]
    {
        let _ = top_k;
        Ok(serde_json::json!({
            "executed": false,
            "serverOnly": true,
            "wasmSupported": false,
            "serverSupported": true,
            "loadReport": {
                "loadable": false,
                "bundlePresent": false,
                "diagnostics": ["runtime.onnxQaProbe is server-only and requires the `local-onnx` feature"]
            },
            "answers": [],
            "modelId": request.model.model_id.unwrap_or_else(|| "roberta-base-squad2-onnx".to_string()),
            "bundlePath": request.model.bundle_root.unwrap_or_else(|| PathBuf::from(".model-runtime")),
            "runtime": "onnx"
        }))
    }
}

#[cfg(feature = "model-bundles")]
fn model_resolve_parts(
    request: LocalModelRequest,
) -> Result<
    (
        String,
        model_runtime::HuggingFaceModelSpec,
        ModelBundleResolveOptions,
    ),
    String,
> {
    let model_id = request.model_id.unwrap_or_else(|| {
        ModelPreset::OnnxCommunityRobertaBaseSquad2
            .as_str()
            .to_string()
    });
    let preset = model_id
        .parse::<ModelPreset>()
        .map_err(|error| error.to_string())?;
    let mut options = ModelBundleResolveOptions::default();
    if let Some(bundle_root) = request.bundle_root {
        options.bundle_root = bundle_root;
    }
    if let Some(auto_download) = request.auto_download {
        options.auto_download = auto_download;
    }
    if let Some(download_progress) = request.download_progress {
        options.download_progress = download_progress;
    }
    options.cache_dir = request.cache_dir;
    options.hf_token = request.hf_token;
    if let Some(max_retries) = request.max_retries {
        options.max_retries = max_retries;
    }
    if let Some(overwrite) = request.overwrite {
        options.overwrite = overwrite;
    }
    Ok((model_id, preset.spec(), options))
}

#[cfg(not(feature = "model-bundles"))]
#[allow(dead_code)]
fn model_resolve_parts(request: LocalModelRequest) -> Result<(String, (), ()), String> {
    let _ = request;
    Err("model bundle resolution requires the `model-bundles` feature".to_string())
}

fn tokenizer_probe_value(request: TokenizerProbeRequest) -> Result<serde_json::Value, String> {
    require_non_empty_string("runtime.tokenizerProbe", "modelId", &request.model_id)?;
    require_non_empty_path(
        "runtime.tokenizerProbe",
        "tokenizerPath",
        &request.tokenizer_path,
    )?;
    require_non_empty_string("runtime.tokenizerProbe", "sample", &request.sample)?;

    let (report, run) =
        validate_tokenizer_bundle(request.model_id, request.tokenizer_path, &request.sample);
    Ok(serde_json::json!({
        "report": report,
        "run": run
    }))
}

fn whitespace_tokens(text: &str) -> Vec<(String, usize, usize)> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(byte_start) = start.take() {
                tokens.push((text[byte_start..index].to_string(), byte_start, index));
            }
        } else {
            start.get_or_insert(index);
        }
    }
    if let Some(byte_start) = start {
        tokens.push((text[byte_start..].to_string(), byte_start, text.len()));
    }
    tokens
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    runtime_core::parse_surface_input(None, input)
}

fn default_truncation() -> String {
    "head".to_string()
}

fn default_supported() -> bool {
    true
}

fn require_non_empty_string(operation: &str, field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!(
            "invalid request for `{operation}`: `{field}` must not be empty"
        ));
    }
    Ok(())
}

fn require_non_empty_path(
    operation: &str,
    field: &str,
    value: &std::path::Path,
) -> Result<(), String> {
    if value.as_os_str().is_empty() {
        return Err(format!(
            "invalid request for `{operation}`: `{field}` must not be empty"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_runtime_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"runtime.tokenizeSummary".to_string()));
        assert!(ids.contains(&"runtime.bundleCheck".to_string()));
        assert!(ids.contains(&"runtime.tokenizerProbe".to_string()));
        assert!(ids.contains(&"runtime.softmax".to_string()));
    }

    #[test]
    fn bundle_check_reports_missing_files_without_downloads() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("runtime.bundleCheck"),
            input: serde_json::json!({
                "modelId": "missing-tokenizer",
                "capability": "tokenizer",
                "bundleRoot": ".model-runtime/missing-tokenizer",
                "requiredFiles": ["tokenizer.json"]
            }),
        })
        .expect("bundle check");
        assert_eq!(response.value["loadable"], false);
        assert_eq!(response.value["missingFiles"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn softmax_operation_normalizes_logits() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("runtime.softmax"),
            input: serde_json::json!({"logits": [0.0, 1.0]}),
        })
        .expect("softmax");
        assert_eq!(response.value["probabilities"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn malformed_input_returns_typed_error_string() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("runtime.softmax"),
            input: serde_json::json!({"logits": "bad"}),
        })
        .expect_err("invalid request");
        assert!(error.contains("invalid request"));
    }
}
