//! Library-owned runtime surface for `text-generation-linguistics`.

use runtime_core::{
    structured_surface_value, OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation,
    SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;
use text_generation::{MarkovChain, MarkovInputMode, TextSynthesisOptions};
use text_linguistics::{analyze_text, LinguisticAnalysisOptions};

use crate::{
    analysis_tokens, synthesize_from_analysis, terms_from_analysis, LinguisticMarkovTraining,
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
                "Adapters from text-linguistics analysis outputs into text-generation.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "generationLinguistics.analysisTerms",
                "Analysis terms",
                "Analyzes text and converts linguistic signals into weighted term prompts.",
                serde_json::json!({"text": "Alice presented the tokenizer roadmap in Berlin."}),
            ),
            operation(
                "generationLinguistics.synthesizeFromAnalysis",
                "Synthesize from analysis",
                "Analyzes text and synthesizes a deterministic document from linguistic terms.",
                serde_json::json!({"id": "analysis-doc", "text": "Alice presented the tokenizer roadmap in Berlin."}),
            ),
            operation(
                "generationLinguistics.trainAnalysis",
                "Train analysis",
                "Analyzes text and trains a transient Markov chain from linguistic tokens.",
                serde_json::json!({"text": "Scene transitions follow visual changes.", "mode": "lemma", "order": 2}),
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
    runtime_core::surface_operation(id, name, description, example_request)
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "generationLinguistics.analysisTerms" => analysis_terms_value(parse_input(request.input)?)?,
        "generationLinguistics.synthesizeFromAnalysis" => {
            synthesize_value(parse_input(request.input)?)?
        }
        "generationLinguistics.trainAnalysis" => train_value(parse_input(request.input)?)?,
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
            "Inspected the text-generation-linguistics package operations and runtime support.",
            serde_json::json!({
                "status": "ok",
                "operationCount": value["operationCount"]
            }),
        ),
        "generationLinguistics.analysisTerms" => (
            "Linguistic term extraction result",
            "Analyzed text and converted linguistic signals into weighted term prompts.",
            serde_json::json!({
                "status": "ok",
                "termCount": value["terms"].as_array().map(Vec::len).unwrap_or(0),
                "entityCount": value["entities"].as_array().map(Vec::len).unwrap_or(0),
                "language": value["language"]
            }),
        ),
        "generationLinguistics.synthesizeFromAnalysis" => (
            "Linguistic synthesis result",
            "Analyzed text and synthesized a deterministic document from linguistic terms.",
            serde_json::json!({
                "status": "ok",
                "id": value["value"]["id"],
                "language": value["value"]["language"],
                "assumptionCount": value["trace"]["assumptions"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "generationLinguistics.trainAnalysis" => (
            "Linguistic Markov training result",
            "Analyzed text and trained a transient Markov chain from linguistic tokens.",
            serde_json::json!({
                "status": "ok",
                "mode": value["mode"],
                "order": value["order"],
                "tokenCount": value["tokens"].as_array().map(Vec::len).unwrap_or(0),
                "contexts": value["contexts"]
            }),
        ),
        _ => (
            "Text generation linguistics result",
            "Ran a text-generation-linguistics package operation.",
            serde_json::json!({"status": "ok"}),
        ),
    };
    structured_surface_value(operation, title, message, summary, value)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisTextRequest {
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SynthesizeAnalysisRequest {
    id: Option<String>,
    text: String,
    #[serde(default)]
    options: TextSynthesisOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrainAnalysisRequest {
    text: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_order")]
    order: usize,
}

fn analysis_terms_value(request: AnalysisTextRequest) -> Result<serde_json::Value, String> {
    let analysis = analyze_text(&request.text, &LinguisticAnalysisOptions::heuristic())
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "terms": terms_from_analysis(&analysis),
        "language": analysis.language.primary.map(|prediction| prediction.language),
        "entities": analysis.entities.into_iter().map(|entity| serde_json::json!({
            "id": entity.id,
            "type": format!("{:?}", entity.entity_type),
            "text": entity.mention.text,
            "normalized": entity.normalized,
            "confidence": entity.confidence
        })).collect::<Vec<_>>()
    }))
}

fn synthesize_value(request: SynthesizeAnalysisRequest) -> Result<serde_json::Value, String> {
    let analysis = analyze_text(&request.text, &LinguisticAnalysisOptions::heuristic())
        .map_err(|error| error.to_string())?;
    let generated = synthesize_from_analysis(
        request
            .id
            .unwrap_or_else(|| "analysis-generated".to_string()),
        &analysis,
        request.options,
    )
    .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "value": {
            "id": generated.value.id,
            "text": generated.value.text,
            "language": generated.value.language
        },
        "trace": {
            "sourceType": generated.trace.source_type,
            "targetType": generated.trace.target_type,
            "fidelity": format!("{:?}", generated.trace.fidelity),
            "confidence": generated.trace.confidence,
            "assumptions": generated.trace.assumptions
        }
    }))
}

fn train_value(request: TrainAnalysisRequest) -> Result<serde_json::Value, String> {
    let analysis = analyze_text(&request.text, &LinguisticAnalysisOptions::heuristic())
        .map_err(|error| error.to_string())?;
    let mode = parse_mode(&request.mode)?;
    let mut chain = MarkovChain::new(request.order).map_err(|error| error.to_string())?;
    chain.train_analysis(&analysis, mode);
    Ok(serde_json::json!({
        "order": chain.order(),
        "contexts": chain.contexts(),
        "transitions": chain.total_transitions(),
        "mode": request.mode,
        "tokens": analysis_tokens(&analysis, mode)
    }))
}

fn parse_mode(mode: &str) -> Result<MarkovInputMode, String> {
    match mode {
        "surface" => Ok(MarkovInputMode::Surface),
        "normalized" => Ok(MarkovInputMode::Normalized),
        "lemma" => Ok(MarkovInputMode::Lemma),
        "entityAware" | "entity_aware" | "entity-aware" => Ok(MarkovInputMode::EntityAware),
        other => Err(format!("unsupported Markov input mode `{other}`")),
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    runtime_core::parse_surface_input(None, input)
}

fn default_mode() -> String {
    "lemma".to_string()
}
fn default_order() -> usize {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_generation_linguistics_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"generationLinguistics.analysisTerms".to_string()));
        assert!(ids.contains(&"generationLinguistics.trainAnalysis".to_string()));
    }

    #[test]
    fn analysis_terms_operation_returns_terms() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("generationLinguistics.analysisTerms"),
            input: serde_json::json!({"text": "Alice presented the tokenizer roadmap in Berlin."}),
        })
        .expect("analysis terms");
        assert!(!response.value["terms"].as_array().unwrap().is_empty());
    }

    #[test]
    fn malformed_input_returns_typed_error_string() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("generationLinguistics.trainAnalysis"),
            input: serde_json::json!({"text": "hello", "mode": "missing"}),
        })
        .expect_err("invalid request");
        assert!(error.contains("unsupported Markov input mode"));
    }
}
