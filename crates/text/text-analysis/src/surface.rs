use std::path::PathBuf;

use runtime_core::{
    structured_surface_value, Diagnostic, DiagnosticCode, DiagnosticSeverity, OperationId,
    PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;
use text_core::TextDocument;
use text_embeddings::PoolingStrategy;
use text_lexical::{character_shingle_similarity, token_shingle_similarity};

use crate::{
    analyze_corpus, analyze_text, AnalysisProfile, CorpusAnalysisOptions, DocumentAnalysisOptions,
    EmbeddingDepth, LinguisticDepth, TextAnalysisDiagnostic,
};

pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Inspect package metadata",
                "Unified text analysis orchestration for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "analysis.describe",
                "Inspect analysis metadata",
                "Unified text analysis orchestration for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "analysis.document",
                "Analyze document",
                "Analyzes one text with core, lexical, similarity, linguistic, and embedding sections.",
                serde_json::json!({
                    "id": "doc-1",
                    "text": "Alice presented the tokenizer roadmap in Berlin.",
                    "profile": "deterministic",
                    "keywordLimit": 10,
                    "summarySentences": 3,
                    "ngramSizes": [2, 3],
                    "shingleSizes": [3, 5],
                    "linguistics": {"mode": "heuristicBalanced"},
                    "embedding": {"mode": "hashed", "dimensions": 128, "useIdf": false}
                }),
            ),
            operation(
                "analysis.corpus",
                "Analyze corpus",
                "Analyzes a transient corpus with TF-IDF, BM25, near-duplicate, and semantic neighbor reports.",
                serde_json::json!({
                    "documents": [
                        {"id": "doc-1", "text": "rust text analysis"},
                        {"id": "doc-2", "text": "video scene analysis"}
                    ],
                    "query": "text analysis",
                    "topK": 10,
                    "includeNearDuplicates": true,
                    "includeSemanticNeighbors": true,
                    "embedding": {"mode": "hashed", "dimensions": 128, "useIdf": true}
                }),
            ),
            operation(
                "analysis.similarity",
                "Compare texts",
                "Compares two texts with character or token shingle Jaccard similarity.",
                serde_json::json!({
                    "left": "scene transitions follow motion",
                    "right": "scene transitions follow dialogue",
                    "n": 3,
                    "mode": "token"
                }),
            ),
        ],
    }
}

pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let (value, diagnostics) = match request.operation.as_str() {
        "describe" | "analysis.describe" => (describe_value(request.input), Vec::new()),
        "analysis.document" => {
            let input = parse_input::<DocumentRequest>(request.input)?;
            let options = input.options();
            let report = analyze_text(
                input.id.unwrap_or_else(|| "doc-0".to_string()),
                &input.text,
                &options,
            )
            .map_err(|error| error.to_string())?;
            let diagnostics = runtime_diagnostics(&report.diagnostics);
            (
                serde_json::to_value(report).map_err(|error| error.to_string())?,
                diagnostics,
            )
        }
        "analysis.corpus" => {
            let input = parse_input::<CorpusRequest>(request.input)?;
            let options = input.options();
            let ids = input
                .documents
                .iter()
                .enumerate()
                .map(|(index, document)| {
                    document
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("doc-{index}"))
                })
                .collect::<Vec<_>>();
            let documents = input
                .documents
                .iter()
                .enumerate()
                .map(|(index, document)| TextDocument::new(ids[index].as_str(), &document.text))
                .collect::<Vec<_>>();
            let report = analyze_corpus(documents, &options).map_err(|error| error.to_string())?;
            let diagnostics = runtime_diagnostics(&report.diagnostics);
            (
                serde_json::to_value(report).map_err(|error| error.to_string())?,
                diagnostics,
            )
        }
        "analysis.similarity" => {
            let input = parse_input::<SimilarityRequest>(request.input)?;
            let n = input.n.max(1);
            let value = match input.mode.as_str() {
                "character" | "char" => serde_json::json!({
                    "mode": "character",
                    "n": n,
                    "similarity": character_shingle_similarity(&input.left, &input.right, n)
                        .map_err(|error| error.to_string())?
                }),
                "token" => serde_json::json!({
                    "mode": "token",
                    "n": n,
                    "similarity": token_shingle_similarity(
                        &input.left,
                        &input.right,
                        n,
                        &Default::default()
                    ).map_err(|error| error.to_string())?
                }),
                other => return Err(format!("unsupported similarity mode `{other}`")),
            };
            (value, Vec::new())
        }
        other => {
            return Err(runtime_core::SurfaceError::unsupported_operation(
                other,
                env!("CARGO_PKG_NAME"),
            )
            .to_error_string());
        }
    };
    let value = annotated_value(&operation, value);
    Ok(SurfaceResponse {
        operation,
        value,
        diagnostics,
        artifacts: Vec::new(),
    })
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    let mut operation = runtime_core::surface_operation(id, name, description, example_request);
    if let Some(contract) = landscape_contract(id) {
        runtime_core::attach_landscape_contract(&mut operation, contract);
    }
    operation
}

fn landscape_contract(id: &str) -> Option<runtime_core::landscape::LandscapeOperationContract> {
    match id {
        "analysis.document" => Some(runtime_core::landscape::LandscapeOperationContract::new(
            runtime_core::landscape::LandscapeFunction::new(
                "text.analysis.analyzeDocument",
                env!("CARGO_PKG_NAME"),
            )
            .input(runtime_core::landscape::LandscapePort::new(
                "document",
                runtime_core::landscape::well_known::text_document(),
            ))
            .output(runtime_core::landscape::LandscapePort::new(
                "report",
                runtime_core::landscape::well_known::text_analysis_report(),
            )),
        )),
        _ => None,
    }
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
        "describe" | "analysis.describe" => (
            "Package surface metadata",
            "Inspected the text-analysis package operations and runtime support.",
            serde_json::json!({
                "status": "ok",
                "operationCount": value["operationCount"]
            }),
        ),
        "analysis.document" => (
            "Document analysis result",
            "Analyzed one text with core, lexical, similarity, linguistic, and embedding sections.",
            serde_json::json!({
                "status": "ok",
                "id": value["id"],
                "language": value["language"],
                "tokenCount": value["core"]["tokens"].as_array().map(Vec::len).unwrap_or(0),
                "sentenceCount": value["core"]["sentences"].as_array().map(Vec::len).unwrap_or(0),
                "keywordCount": value["lexical"]["keywords"].as_array().map(Vec::len).unwrap_or(0),
                "entityCount": value["lexical"]["ruleEntities"].as_array().map(Vec::len).unwrap_or(0),
                "embeddingDimensions": value["embedding"]["dimensions"],
                "diagnosticCount": value["diagnostics"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "analysis.corpus" => (
            "Corpus analysis result",
            "Analyzed a transient corpus with lexical statistics, search reports, near duplicates, and semantic neighbors.",
            serde_json::json!({
                "status": "ok",
                "documentCount": value["documents"].as_array().map(Vec::len).unwrap_or(0),
                "termCount": value["termStats"].as_array().map(Vec::len).unwrap_or(0),
                "resultCount": value["tfidfSearch"]
                    .as_array()
                    .or_else(|| value["bm25Search"].as_array())
                    .map(Vec::len)
                    .unwrap_or(0),
                "nearDuplicateCount": value["nearDuplicates"].as_array().map(Vec::len).unwrap_or(0),
                "semanticNeighborCount": value["semanticNeighbors"].as_array().map(Vec::len).unwrap_or(0),
                "diagnosticCount": value["diagnostics"].as_array().map(Vec::len).unwrap_or(0)
            }),
        ),
        "analysis.similarity" => (
            "Text similarity result",
            "Compared two texts with character or token shingle Jaccard similarity.",
            serde_json::json!({
                "status": "ok",
                "mode": value["mode"],
                "n": value["n"],
                "score": value["similarity"]["jaccard"],
                "leftCount": value["similarity"]["leftCount"],
                "rightCount": value["similarity"]["rightCount"],
                "intersectionCount": value["similarity"]["intersectionCount"],
                "unionCount": value["similarity"]["unionCount"],
                "similarity": value["similarity"]
            }),
        ),
        _ => (
            "Text analysis result",
            "Ran a text-analysis package operation.",
            serde_json::json!({"status": "ok"}),
        ),
    };
    structured_surface_value(operation, title, message, summary, value)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DocumentRequest {
    id: Option<String>,
    text: String,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    language_hint: Option<String>,
    #[serde(default)]
    keyword_limit: Option<usize>,
    #[serde(default)]
    summary_sentences: Option<usize>,
    #[serde(default)]
    ngram_sizes: Option<Vec<usize>>,
    #[serde(default)]
    shingle_sizes: Option<Vec<usize>>,
    #[serde(default)]
    include_annotation_graph: Option<bool>,
    #[serde(default)]
    include_sparse_embedding: Option<bool>,
    #[serde(default)]
    linguistics: Option<ModeRequest>,
    #[serde(default)]
    embedding: Option<EmbeddingRequest>,
}

impl DocumentRequest {
    fn options(&self) -> DocumentAnalysisOptions {
        let mut options = if self.profile.as_deref() == Some("modelBacked")
            || self.profile.as_deref() == Some("model-backed")
        {
            DocumentAnalysisOptions::model_backed()
        } else {
            DocumentAnalysisOptions::default()
        };
        options.profile = match self.profile.as_deref() {
            Some("modelBacked" | "model-backed") => AnalysisProfile::ModelBacked,
            _ => AnalysisProfile::Deterministic,
        };
        options.language_hint = self.language_hint.clone();
        if let Some(limit) = self.keyword_limit {
            options.keyword_limit = limit;
        }
        if let Some(sentences) = self.summary_sentences {
            options.summary_sentences = sentences;
        }
        if let Some(sizes) = &self.ngram_sizes {
            options.ngram_sizes = sizes.clone();
        }
        if let Some(sizes) = &self.shingle_sizes {
            options.shingle_sizes = sizes.clone();
        }
        if let Some(include) = self.include_annotation_graph {
            options.include_annotation_graph = include;
        }
        if let Some(include) = self.include_sparse_embedding {
            options.include_sparse_embedding = include;
        }
        if let Some(linguistics) = &self.linguistics {
            options.linguistic_depth = parse_linguistic_depth(linguistics);
        }
        if let Some(embedding) = &self.embedding {
            options.embedding_depth = parse_embedding_depth(embedding);
        }
        options
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CorpusRequest {
    documents: Vec<CorpusDocumentRequest>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    keyword_limit: Option<usize>,
    #[serde(default)]
    tfidf_terms_per_document: Option<usize>,
    #[serde(default)]
    include_near_duplicates: Option<bool>,
    #[serde(default)]
    include_semantic_neighbors: Option<bool>,
    #[serde(default)]
    embedding: Option<EmbeddingRequest>,
}

impl CorpusRequest {
    fn options(&self) -> CorpusAnalysisOptions {
        let mut options = CorpusAnalysisOptions {
            query: self.query.clone(),
            ..CorpusAnalysisOptions::default()
        };
        if let Some(top_k) = self.top_k {
            options.top_k = top_k;
        }
        if let Some(limit) = self.keyword_limit {
            options.document.keyword_limit = limit;
        }
        if let Some(limit) = self.tfidf_terms_per_document {
            options.tfidf_terms_per_document = limit;
        }
        if let Some(include) = self.include_near_duplicates {
            options.include_near_duplicates = include;
        }
        if let Some(include) = self.include_semantic_neighbors {
            options.include_semantic_neighbors = include;
        }
        if let Some(embedding) = &self.embedding {
            options.document.embedding_depth = parse_embedding_depth(embedding);
        }
        options
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CorpusDocumentRequest {
    id: Option<String>,
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModeRequest {
    mode: String,
    #[serde(default)]
    bundle_dir: Option<String>,
    #[serde(default)]
    auto_download: Option<bool>,
    #[serde(default)]
    download_progress: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingRequest {
    mode: String,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    use_idf: Option<bool>,
    #[serde(default)]
    bundle_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SimilarityRequest {
    left: String,
    right: String,
    #[serde(default = "default_similarity_n")]
    n: usize,
    #[serde(default = "default_similarity_mode")]
    mode: String,
}

fn parse_linguistic_depth(request: &ModeRequest) -> LinguisticDepth {
    match request.mode.as_str() {
        "off" => LinguisticDepth::Off,
        "heuristicFast" | "heuristic-fast" | "fast" => LinguisticDepth::HeuristicFast,
        "heuristicRich" | "heuristic-rich" | "rich" => LinguisticDepth::HeuristicRich,
        "localModel" | "local-model" | "model" => LinguisticDepth::LocalModel {
            bundle_dir: request
                .bundle_dir
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(".model-runtime")),
            auto_download: request.auto_download.unwrap_or(false),
            download_progress: request.download_progress.unwrap_or(false),
        },
        _ => LinguisticDepth::HeuristicBalanced,
    }
}

fn parse_embedding_depth(request: &EmbeddingRequest) -> EmbeddingDepth {
    match request.mode.as_str() {
        "off" => EmbeddingDepth::Off,
        "candle" | "candleBundle" | "candle-bundle" => EmbeddingDepth::CandleBundle {
            bundle_dir: request
                .bundle_dir
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(".model-runtime")),
            pooling: PoolingStrategy::Mean,
        },
        "onnx" | "onnxBundle" | "onnx-bundle" => EmbeddingDepth::OnnxBundle {
            bundle_dir: request
                .bundle_dir
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(".model-runtime")),
            pooling: PoolingStrategy::Mean,
        },
        _ => EmbeddingDepth::Hashed {
            dimensions: request.dimensions.unwrap_or(128),
            use_idf: request.use_idf.unwrap_or(false),
        },
    }
}

fn runtime_diagnostics(diagnostics: &[TextAnalysisDiagnostic]) -> Vec<Diagnostic> {
    diagnostics
        .iter()
        .map(|diagnostic| Diagnostic {
            severity: if diagnostic.severity == "error" {
                DiagnosticSeverity::Error
            } else {
                DiagnosticSeverity::Warning
            },
            code: DiagnosticCode(diagnostic.code.clone()),
            message: diagnostic.message.clone(),
            source: diagnostic.source.clone(),
            help: None,
        })
        .collect()
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    runtime_core::parse_surface_input(None, input)
}

fn default_similarity_n() -> usize {
    3
}

fn default_similarity_mode() -> String {
    "token".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_model_request_defaults_do_not_auto_download() {
        let depth = parse_linguistic_depth(&ModeRequest {
            mode: "localModel".to_string(),
            bundle_dir: None,
            auto_download: None,
            download_progress: None,
        });

        match depth {
            LinguisticDepth::LocalModel {
                auto_download,
                download_progress,
                ..
            } => {
                assert!(!auto_download);
                assert!(!download_progress);
            }
            other => panic!("expected local model depth, got {other:?}"),
        }
    }

    #[test]
    fn local_model_request_accepts_explicit_auto_download() {
        let depth = parse_linguistic_depth(&ModeRequest {
            mode: "localModel".to_string(),
            bundle_dir: Some(".model-runtime".to_string()),
            auto_download: Some(true),
            download_progress: Some(true),
        });

        match depth {
            LinguisticDepth::LocalModel {
                auto_download,
                download_progress,
                ..
            } => {
                assert!(auto_download);
                assert!(download_progress);
            }
            other => panic!("expected local model depth, got {other:?}"),
        }
    }
}
