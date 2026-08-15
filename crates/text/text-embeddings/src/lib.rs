#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "model-bundles")]
use std::fs;
#[cfg(any(feature = "onnx", feature = "model-bundles"))]
use std::path::Path;
#[cfg(feature = "tokenizers")]
use std::path::PathBuf;
#[cfg(feature = "onnx")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "onnx")]
use std::time::Instant;

#[cfg(feature = "candle")]
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};
#[cfg(feature = "candle")]
use candle_nn::VarBuilder as CandleVarBuilder;
#[cfg(feature = "candle")]
use candle_transformers::models::{bert as candle_bert, distilbert as candle_distilbert};
use math_sparse_data::SparseVector;
#[cfg(feature = "model-bundles")]
use model_runtime::ModelBundle;
use serde::{Deserialize, Serialize};
#[cfg(feature = "tokenizers")]
use serde_json::Value;
use text_core::{segment_document_id, tokenize_words, AnnotationProvenance, TextDocument};
use text_core::{DetectError, Result, TextSegment};
use text_lexical::{term_counts, CorpusOptions, TfIdfCorpus};
pub use text_model_runtime::TokenizedText;
#[cfg(feature = "tokenizers")]
use text_model_runtime::TokenizerBundle;
use vector_analysis_core::cosine_similarity;
/// Re-exports the dense vector API.
pub use vector_analysis_core::DenseVector;
use vector_analysis_index::{SearchConfig, VectorRecord, VectorSearchIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for text embedding config.
pub struct TextEmbeddingConfig {
    /// The dimensions value.
    pub dimensions: usize,
    /// The use idf value.
    pub use_idf: bool,
}

impl Default for TextEmbeddingConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            use_idf: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Data type for hashed text embedder.
pub struct HashedTextEmbedder {
    /// The config value.
    pub config: TextEmbeddingConfig,
    /// The corpus options value.
    pub corpus_options: CorpusOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Variants describing text embedding backend kind.
#[serde(rename_all = "snake_case")]
pub enum TextEmbeddingBackendKind {
    /// The hashed variant.
    Hashed,
    /// The ONNX variant.
    Onnx,
    /// The candle variant.
    Candle,
    /// The cuda oxide variant.
    CudaOxide,
    /// The external variant.
    External,
    /// The custom variant.
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for text embedding metadata.
pub struct TextEmbeddingMetadata {
    /// The backend value.
    pub backend: TextEmbeddingBackendKind,
    /// The provenance value.
    pub provenance: AnnotationProvenance,
    /// The model name value.
    pub model_name: Option<String>,
    /// The dimensions value.
    pub dimensions: Option<usize>,
}

impl Default for TextEmbeddingMetadata {
    fn default() -> Self {
        Self {
            backend: TextEmbeddingBackendKind::Custom,
            provenance: AnnotationProvenance::Derived,
            model_name: None,
            dimensions: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for embedding model info.
pub struct EmbeddingModelInfo {
    /// The model name value.
    pub model_name: String,
    /// The backend value.
    pub backend: TextEmbeddingBackendKind,
    /// The dimensions value.
    pub dimensions: usize,
    /// Whether vectors are normalized.
    #[serde(default = "default_normalized")]
    pub normalized: bool,
    /// Maximum supported token count when known.
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

impl Default for EmbeddingModelInfo {
    fn default() -> Self {
        Self {
            model_name: "custom".to_string(),
            backend: TextEmbeddingBackendKind::Custom,
            dimensions: 0,
            normalized: true,
            max_tokens: None,
        }
    }
}

fn default_normalized() -> bool {
    true
}

/// Trait for embedding cache hooks.
pub trait EmbeddingCacheHooks {
    /// Returns cache key for text.
    fn cache_key(&self, text: &str) -> Option<String>;
}

/// Trait for text embedding backend implementations.
pub trait TextEmbeddingBackend {
    /// Returns embed text.
    fn embed_text(&self, text: &str) -> Result<DenseVector>;

    /// Returns embed batch.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<DenseVector>> {
        texts.iter().map(|text| self.embed_text(text)).collect()
    }

    /// Returns metadata.
    fn metadata(&self) -> TextEmbeddingMetadata {
        TextEmbeddingMetadata::default()
    }

    /// Returns embed document.
    fn embed_document(&self, document: &TextDocument<'_>) -> Result<DenseVector> {
        self.embed_text(document.text)
    }

    /// Returns model info.
    fn model_info(&self) -> EmbeddingModelInfo {
        let metadata = self.metadata();
        EmbeddingModelInfo {
            model_name: metadata.model_name.unwrap_or_else(|| "custom".to_string()),
            backend: metadata.backend,
            dimensions: metadata.dimensions.unwrap_or(0),
            normalized: true,
            max_tokens: None,
        }
    }

    /// Returns cache hooks.
    fn cache_hooks(&self) -> Option<&dyn EmbeddingCacheHooks> {
        None
    }
}

/// Trait alias-style marker for text embedders used by retrieval indexes.
pub trait TextEmbedderBackend: TextEmbeddingBackend {}

impl<T: TextEmbeddingBackend> TextEmbedderBackend for T {}

/// Trait for sentence embedding backend implementations.
pub trait SentenceEmbedder: TextEmbedderBackend {
    /// Returns embed sentences.
    fn embed_sentences(&self, sentences: &[String]) -> Result<Vec<DenseVector>> {
        sentences
            .iter()
            .map(|sentence| self.embed_text(sentence))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing pooling strategy.
pub enum PoolingStrategy {
    /// Use the first token.
    Cls,
    /// Mean-pool unmasked tokens.
    Mean,
}

#[cfg(feature = "onnx")]
#[derive(Debug)]
/// Data type for native ONNX runner.
pub struct NativeOnnxRunner {
    session: Mutex<runtime_onnx::OnnxSession>,
    model_path: PathBuf,
}

#[cfg(feature = "onnx")]
impl NativeOnnxRunner {
    /// Creates a new value.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let timing_enabled = onnx_timing_enabled();
        let started = timing_enabled.then(Instant::now);
        let session = runtime_onnx::OnnxSession::from_file_with_options(
            &model_path,
            runtime_onnx::OnnxSessionOptions {
                graph_optimization: runtime_onnx::OnnxGraphOptimization::Disable,
                execution_provider: runtime_onnx::OnnxExecutionProvider::Cpu,
            },
        );
        if let Some(started) = started {
            log_onnx_stage_timing(
                "NativeOnnxRunner::new",
                &model_path,
                started.elapsed(),
                session.is_ok(),
            );
        }
        Ok(Self {
            session: Mutex::new(session.map_err(runtime_onnx_error)?),
            model_path,
        })
    }

    fn run_first_f32_output(&self, tokens: &TokenizedText) -> Result<(Vec<f32>, Vec<usize>)> {
        use runtime_onnx::{OnnxRunner, OnnxTensor, OnnxTensorValue};

        let mut session = self
            .session
            .lock()
            .map_err(|_| DetectError::Source("ONNX session mutex was poisoned".to_string()))?;
        let metadata = session.metadata().map_err(runtime_onnx_error)?;
        let input_names = metadata
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect::<Vec<_>>();
        let shape = vec![1, tokens.input_ids.len()];
        let mut inputs = Vec::new();
        if input_names.iter().any(|name| name == "input_ids") {
            inputs.push(runtime_onnx::OnnxNamedTensor {
                name: "input_ids".to_string(),
                tensor: OnnxTensorValue::I64(
                    OnnxTensor::new(shape.clone(), tokens.input_ids.clone())
                        .map_err(runtime_onnx_error)?,
                ),
            });
        }
        if input_names.iter().any(|name| name == "attention_mask") {
            inputs.push(runtime_onnx::OnnxNamedTensor {
                name: "attention_mask".to_string(),
                tensor: OnnxTensorValue::I64(
                    OnnxTensor::new(shape.clone(), tokens.attention_mask.clone())
                        .map_err(runtime_onnx_error)?,
                ),
            });
        }
        if input_names.iter().any(|name| name == "token_type_ids") {
            if let Some(token_type_ids) = &tokens.token_type_ids {
                inputs.push(runtime_onnx::OnnxNamedTensor {
                    name: "token_type_ids".to_string(),
                    tensor: OnnxTensorValue::I64(
                        OnnxTensor::new(shape, token_type_ids.clone())
                            .map_err(runtime_onnx_error)?,
                    ),
                });
            }
        }
        if inputs.is_empty() {
            return Err(invalid_argument(
                "ONNX text model does not expose a supported text input",
            ));
        }
        if onnx_timing_enabled() {
            log_onnx_stage_event("NativeOnnxRunner::session.run", &self.model_path, "start");
        }
        let outputs = session.run(inputs).map_err(runtime_onnx_error)?;
        let output = runtime_onnx::first_f32_output(&outputs).map_err(runtime_onnx_error)?;
        Ok((output.values.clone(), output.shape.clone()))
    }
}

/// Trait for ONNX text embedding runner implementations.
pub trait OnnxTextEmbeddingRunner {
    /// Runs embeddings.
    fn run_embeddings(&self, tokens: &TokenizedText) -> Result<(Vec<f32>, Vec<usize>)>;
}

#[derive(Debug, Clone, Default)]
/// Data type for unavailable ONNX runner.
pub struct UnavailableOnnxRunner;

impl OnnxTextEmbeddingRunner for UnavailableOnnxRunner {
    fn run_embeddings(&self, _tokens: &TokenizedText) -> Result<(Vec<f32>, Vec<usize>)> {
        Err(DetectError::Source(
            "native ONNX execution is unavailable; construct with a runner or enable an executor"
                .to_string(),
        ))
    }
}

#[cfg(feature = "onnx")]
impl OnnxTextEmbeddingRunner for NativeOnnxRunner {
    fn run_embeddings(&self, tokens: &TokenizedText) -> Result<(Vec<f32>, Vec<usize>)> {
        self.run_first_f32_output(tokens)
    }
}

#[cfg(feature = "tokenizers")]
#[derive(Debug, Clone)]
/// Data type for ONNX text embedder.
pub struct OnnxTextEmbedder<R = UnavailableOnnxRunner> {
    tokenizer: TokenizerBundle,
    runner: R,
    pooling: PoolingStrategy,
    normalize: bool,
    model_name: String,
    dimensions: Option<usize>,
    max_tokens: Option<usize>,
}

#[cfg(all(
    feature = "tokenizers",
    feature = "model-bundles",
    not(feature = "onnx")
))]
impl OnnxTextEmbedder<UnavailableOnnxRunner> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self> {
        let info = validate_onnx_bundle(&bundle)?;
        Ok(Self {
            tokenizer: TokenizerBundle::new(info.tokenizer_path),
            runner: UnavailableOnnxRunner,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            model_name: bundle.manifest.name,
            dimensions: embedding_dimensions_from_config_path(&info.config_path)?,
            max_tokens: model_max_tokens_from_config_path(&info.config_path)?,
        })
    }
}

#[cfg(all(feature = "tokenizers", feature = "onnx", feature = "model-bundles"))]
impl OnnxTextEmbedder<NativeOnnxRunner> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self> {
        let info = validate_onnx_bundle(&bundle)?;
        Ok(Self {
            tokenizer: TokenizerBundle::new(info.tokenizer_path),
            runner: NativeOnnxRunner::new(info.model_path)?,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            model_name: bundle.manifest.name,
            dimensions: embedding_dimensions_from_config_path(&info.config_path)?,
            max_tokens: model_max_tokens_from_config_path(&info.config_path)?,
        })
    }
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
impl<R: OnnxTextEmbeddingRunner> OnnxTextEmbedder<R> {
    /// Builds this value from runner.
    pub fn from_runner(bundle: ModelBundle, runner: R) -> Result<Self> {
        let info = validate_onnx_bundle(&bundle)?;
        Ok(Self {
            tokenizer: TokenizerBundle::new(info.tokenizer_path),
            runner,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            model_name: bundle.manifest.name,
            dimensions: embedding_dimensions_from_config_path(&info.config_path)?,
            max_tokens: model_max_tokens_from_config_path(&info.config_path)?,
        })
    }
}

#[cfg(feature = "tokenizers")]
impl<R: OnnxTextEmbeddingRunner> OnnxTextEmbedder<R> {
    /// Returns pooling.
    pub fn pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Normalizes this value.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Returns embed tokenized.
    pub fn embed_tokenized(&self, tokens: &TokenizedText) -> Result<DenseVector> {
        let (values, shape) = self.runner.run_embeddings(tokens)?;
        pool_embedding_output(
            &values,
            &shape,
            &tokens.attention_mask,
            self.pooling,
            self.normalize,
        )
    }
}

#[cfg(feature = "tokenizers")]
impl<R: OnnxTextEmbeddingRunner> TextEmbeddingBackend for OnnxTextEmbedder<R> {
    fn embed_text(&self, text: &str) -> Result<DenseVector> {
        let tokens = self.tokenizer.tokenize(text)?;
        self.embed_tokenized(&tokens)
    }

    fn metadata(&self) -> TextEmbeddingMetadata {
        TextEmbeddingMetadata {
            backend: TextEmbeddingBackendKind::Onnx,
            provenance: AnnotationProvenance::Onnx,
            model_name: Some(self.model_name.clone()),
            dimensions: self.dimensions,
        }
    }

    fn model_info(&self) -> EmbeddingModelInfo {
        EmbeddingModelInfo {
            model_name: self.model_name.clone(),
            backend: TextEmbeddingBackendKind::Onnx,
            dimensions: self.dimensions.unwrap_or(0),
            normalized: self.normalize,
            max_tokens: self.max_tokens.or(self.tokenizer.max_length),
        }
    }
}

#[cfg(feature = "tokenizers")]
impl<R: OnnxTextEmbeddingRunner> SentenceEmbedder for OnnxTextEmbedder<R> {}

#[cfg(feature = "tokenizers")]
#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX bundle info.
pub struct OnnxBundleInfo {
    /// The config path value.
    pub config_path: PathBuf,
    /// The tokenizer path value.
    pub tokenizer_path: PathBuf,
    /// The model path value.
    pub model_path: PathBuf,
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
/// Validates ONNX bundle.
pub fn validate_onnx_bundle(bundle: &ModelBundle) -> Result<OnnxBundleInfo> {
    let config_path = required_bundle_file(bundle, "config.json")?;
    let tokenizer_path = required_bundle_file(bundle, "tokenizer.json")?;
    let model_path = first_bundle_file_with_extension(bundle, "onnx").ok_or_else(|| {
        invalid_argument("ONNX text bundle must contain at least one `.onnx` model file")
    })?;
    Ok(OnnxBundleInfo {
        config_path,
        tokenizer_path,
        model_path,
    })
}

#[cfg(feature = "tokenizers")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum CandleEmbeddingArchitecture {
    Bert,
    DistilBert,
}

#[cfg(feature = "tokenizers")]
#[derive(Debug, Clone)]
/// Data type for candle text embedder.
pub struct CandleTextEmbedder {
    tokenizer: TokenizerBundle,
    config: Value,
    model_paths: Vec<PathBuf>,
    architecture: CandleEmbeddingArchitecture,
    pooling: PoolingStrategy,
    normalize: bool,
    model_name: String,
    dimensions: Option<usize>,
    max_tokens: Option<usize>,
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
impl CandleTextEmbedder {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: ModelBundle) -> Result<Self> {
        let config_path = required_bundle_file(&bundle, "config.json")?;
        let tokenizer_path = required_bundle_file(&bundle, "tokenizer.json")?;
        let config = read_json(&config_path)?;
        let model_paths = bundle_files_with_extension(&bundle, "safetensors");
        if model_paths.is_empty() {
            return Err(invalid_argument(
                "Candle text bundles must contain a `.safetensors` model file",
            ));
        }
        let architecture = embedding_architecture_from_config(&config)?;
        let dimensions = embedding_dimensions_from_config(&config);
        let max_tokens = model_max_tokens_from_config(&config);
        Ok(Self {
            tokenizer: tokenizer_with_model_limit(TokenizerBundle::new(tokenizer_path), &config),
            config,
            model_paths,
            architecture,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            model_name: bundle.manifest.name,
            dimensions,
            max_tokens,
        })
    }
}

#[cfg(feature = "tokenizers")]
impl CandleTextEmbedder {
    /// Returns pooling.
    pub fn pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Normalizes this value.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Returns embed tokenized.
    pub fn embed_tokenized(&self, tokens: &TokenizedText) -> Result<DenseVector> {
        #[cfg(feature = "candle")]
        {
            let (values, shape) =
                run_candle_embedder(&self.config, &self.model_paths, self.architecture, tokens)?;
            pool_embedding_output(
                &values,
                &shape,
                &tokens.attention_mask,
                self.pooling,
                self.normalize,
            )
        }
        #[cfg(not(feature = "candle"))]
        {
            let _ = (&self.config, &self.model_paths, self.architecture, tokens);
            Err(invalid_argument(
                "native Candle execution requires the `candle` feature",
            ))
        }
    }
}

#[cfg(feature = "tokenizers")]
impl TextEmbeddingBackend for CandleTextEmbedder {
    fn embed_text(&self, text: &str) -> Result<DenseVector> {
        let tokens = self.tokenizer.tokenize(text)?;
        self.embed_tokenized(&tokens)
    }

    fn metadata(&self) -> TextEmbeddingMetadata {
        TextEmbeddingMetadata {
            backend: TextEmbeddingBackendKind::Candle,
            provenance: AnnotationProvenance::Candle,
            model_name: Some(self.model_name.clone()),
            dimensions: self.dimensions,
        }
    }

    fn model_info(&self) -> EmbeddingModelInfo {
        EmbeddingModelInfo {
            model_name: self.model_name.clone(),
            backend: TextEmbeddingBackendKind::Candle,
            dimensions: self.dimensions.unwrap_or(0),
            normalized: self.normalize,
            max_tokens: self.max_tokens.or(self.tokenizer.max_length),
        }
    }
}

#[cfg(feature = "tokenizers")]
impl SentenceEmbedder for CandleTextEmbedder {}

impl HashedTextEmbedder {
    /// Creates a new value.
    pub fn new(config: TextEmbeddingConfig, corpus_options: CorpusOptions) -> Result<Self> {
        if config.dimensions == 0 {
            return Err(invalid_argument(
                "text embedding dimensions must be greater than zero",
            ));
        }
        Ok(Self {
            config,
            corpus_options,
        })
    }

    /// Returns embed text.
    pub fn embed_text(&self, text: &str) -> Result<DenseVector> {
        self.embed_text_with_corpus(text, None)
    }

    /// Returns embed document.
    pub fn embed_document(&self, document: &TextDocument<'_>) -> Result<DenseVector> {
        self.embed_text(document.text)
    }

    /// Returns embed text with corpus.
    pub fn embed_text_with_corpus(
        &self,
        text: &str,
        corpus: Option<&TfIdfCorpus>,
    ) -> Result<DenseVector> {
        let counts = term_counts(text, &self.corpus_options);
        self.embed_counts(&counts, corpus)
    }

    /// Returns embed counts.
    pub fn embed_counts(
        &self,
        counts: &BTreeMap<String, usize>,
        corpus: Option<&TfIdfCorpus>,
    ) -> Result<DenseVector> {
        if counts.is_empty() {
            return Err(invalid_argument("text must contain at least one term"));
        }
        let mut values = vec![0.0; self.config.dimensions];
        for (term, count) in counts {
            let hash = stable_hash(term.as_bytes());
            let index = hash as usize % self.config.dimensions;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            let idf = if self.config.use_idf {
                corpus
                    .map(|corpus| corpus.inverse_document_frequency(term))
                    .unwrap_or(1.0)
            } else {
                1.0
            };
            values[index] += sign * (*count as f32).ln_1p() * idf;
        }
        DenseVector::new(values)?.l2_normalized()
    }

    /// Returns embed text sparse.
    pub fn embed_text_sparse(
        &self,
        text: &str,
        corpus: Option<&TfIdfCorpus>,
    ) -> Result<SparseVector> {
        let counts = term_counts(text, &self.corpus_options);
        self.embed_counts_sparse(&counts, corpus)
    }

    /// Returns embed counts sparse.
    pub fn embed_counts_sparse(
        &self,
        counts: &BTreeMap<String, usize>,
        corpus: Option<&TfIdfCorpus>,
    ) -> Result<SparseVector> {
        if counts.is_empty() {
            return Err(invalid_argument("text must contain at least one term"));
        }
        let mut buckets = BTreeMap::<usize, f32>::new();
        for (term, count) in counts {
            let hash = stable_hash(term.as_bytes());
            let index = hash as usize % self.config.dimensions;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            let idf = if self.config.use_idf {
                corpus
                    .map(|corpus| corpus.inverse_document_frequency(term))
                    .unwrap_or(1.0)
            } else {
                1.0
            };
            *buckets.entry(index).or_insert(0.0) += sign * (*count as f32).ln_1p() * idf;
        }
        SparseVector::new(
            self.config.dimensions,
            buckets.keys().copied().collect(),
            buckets.values().copied().collect(),
        )?
        .canonicalized()?
        .normalize_l2()
    }
}

impl TextEmbeddingBackend for HashedTextEmbedder {
    fn embed_text(&self, text: &str) -> Result<DenseVector> {
        HashedTextEmbedder::embed_text(self, text)
    }

    fn embed_document(&self, document: &TextDocument<'_>) -> Result<DenseVector> {
        HashedTextEmbedder::embed_document(self, document)
    }

    fn metadata(&self) -> TextEmbeddingMetadata {
        TextEmbeddingMetadata {
            backend: TextEmbeddingBackendKind::Hashed,
            provenance: AnnotationProvenance::Heuristic,
            model_name: Some("hashed-text-embedder".to_string()),
            dimensions: Some(self.config.dimensions),
        }
    }
}

impl SentenceEmbedder for HashedTextEmbedder {}

#[derive(Debug, Clone, PartialEq)]
/// Data type for semantic match.
pub struct SemanticMatch {
    /// Identifier for this value.
    pub id: String,
    /// Score assigned to this value.
    pub score: f32,
    /// The distance value.
    pub distance: f32,
    /// Metadata associated with this value.
    pub metadata: TextEmbeddingMetadata,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for embedding search index.
pub struct EmbeddingSearchIndex<E> {
    embedder: E,
    vectors: VectorSearchIndex,
}

/// Type alias for semantic index.
pub type SemanticIndex<E> = EmbeddingSearchIndex<E>;

impl<E: TextEmbeddingBackend> EmbeddingSearchIndex<E> {
    /// Creates a new value.
    pub fn new(embedder: E) -> Self {
        Self {
            embedder,
            vectors: VectorSearchIndex::new(),
        }
    }

    /// Returns embedder.
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Returns embedder mut.
    pub fn embedder_mut(&mut self) -> &mut E {
        &mut self.embedder
    }

    /// Returns backend metadata.
    pub fn backend_metadata(&self) -> TextEmbeddingMetadata {
        self.embedder.metadata()
    }

    /// Adds add document to this value.
    pub fn add_document(&mut self, id: impl Into<String>, text: &str) -> Result<()> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(invalid_argument("document id must not be empty"));
        }
        if self.vectors.records().iter().any(|record| record.id == id) {
            return Err(invalid_argument(format!(
                "document id `{id}` already exists"
            )));
        }
        let vector = self.embedder.embed_text(text)?;
        self.vectors.add(VectorRecord::new(id, vector))
    }

    /// Adds add text document to this value.
    pub fn add_text_document(&mut self, document: &TextDocument<'_>) -> Result<()> {
        let id = document.id.to_string();
        if id.trim().is_empty() {
            return Err(invalid_argument("document id must not be empty"));
        }
        if self.vectors.records().iter().any(|record| record.id == id) {
            return Err(invalid_argument(format!(
                "document id `{id}` already exists"
            )));
        }
        let vector = self.embedder.embed_document(document)?;
        self.vectors.add(VectorRecord::new(id, vector))
    }

    /// Adds add text segment to this value.
    pub fn add_text_segment(&mut self, stream_id: &str, segment: &TextSegment<'_>) -> Result<()> {
        validate_stream_id(stream_id)?;
        self.add_document(
            segment_document_id(stream_id, segment.segment_index),
            segment.text,
        )
    }

    /// Returns search.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SemanticMatch>> {
        let metadata = self.embedder.metadata();
        let query = self.embedder.embed_text(query)?;
        let results = self.vectors.search(
            &query,
            SearchConfig {
                limit,
                ..SearchConfig::default()
            },
        )?;
        Ok(results
            .into_iter()
            .map(|result| SemanticMatch {
                id: result.id,
                score: result.score,
                distance: result.distance,
                metadata: metadata.clone(),
            })
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for semantic text index.
pub struct SemanticTextIndex {
    embedder: HashedTextEmbedder,
    corpus: TfIdfCorpus,
    vectors: VectorSearchIndex,
}

impl SemanticTextIndex {
    /// Creates a new value.
    pub fn new(embedder: HashedTextEmbedder) -> Self {
        Self {
            corpus: TfIdfCorpus::new(embedder.corpus_options.clone()),
            embedder,
            vectors: VectorSearchIndex::new(),
        }
    }

    /// Builds this value from documents.
    pub fn from_documents<'a, I>(embedder: HashedTextEmbedder, documents: I) -> Result<Self>
    where
        I: IntoIterator<Item = TextDocument<'a>>,
    {
        let mut index = Self::new(embedder);
        index.add_documents(documents)?;
        Ok(index)
    }

    /// Returns embedder.
    pub fn embedder(&self) -> &HashedTextEmbedder {
        &self.embedder
    }

    /// Returns corpus.
    pub fn corpus(&self) -> &TfIdfCorpus {
        &self.corpus
    }

    /// Returns backend metadata.
    pub fn backend_metadata(&self) -> TextEmbeddingMetadata {
        self.embedder.metadata()
    }

    /// Adds add document to this value.
    pub fn add_document(&mut self, id: impl Into<String>, text: &str) -> Result<()> {
        let id = id.into();
        self.corpus.add_document(id.clone(), text)?;
        if self.embedder.config.use_idf {
            self.rebuild_vectors()
        } else {
            let vector = self
                .embedder
                .embed_text_with_corpus(text, Some(&self.corpus))?;
            self.vectors.add(VectorRecord::new(id, vector))
        }
    }

    /// Adds add text document to this value.
    pub fn add_text_document(&mut self, document: &TextDocument<'_>) -> Result<()> {
        self.add_document(document.id, document.text)
    }

    /// Adds add documents to this value.
    pub fn add_documents<'a, I>(&mut self, documents: I) -> Result<()>
    where
        I: IntoIterator<Item = TextDocument<'a>>,
    {
        let documents = documents
            .into_iter()
            .map(|document| (document.id.to_string(), document.text.to_string()))
            .collect::<Vec<_>>();
        self.validate_new_document_ids(documents.iter().map(|(id, _)| id.as_str()))?;

        for (id, text) in &documents {
            self.corpus.add_document(id.clone(), text)?;
        }
        if self.embedder.config.use_idf {
            self.rebuild_vectors()
        } else {
            for (id, text) in documents {
                let vector = self
                    .embedder
                    .embed_text_with_corpus(&text, Some(&self.corpus))?;
                self.vectors.add(VectorRecord::new(id, vector))?;
            }
            Ok(())
        }
    }

    /// Adds add text segment to this value.
    pub fn add_text_segment(&mut self, stream_id: &str, segment: &TextSegment<'_>) -> Result<()> {
        validate_stream_id(stream_id)?;
        self.add_document(
            segment_document_id(stream_id, segment.segment_index),
            segment.text,
        )
    }

    /// Returns search.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SemanticMatch>> {
        let metadata = self.embedder.metadata();
        let query = self
            .embedder
            .embed_text_with_corpus(query, Some(&self.corpus))?;
        let results = self.vectors.search(
            &query,
            SearchConfig {
                limit,
                ..SearchConfig::default()
            },
        )?;
        Ok(results
            .into_iter()
            .map(|result| SemanticMatch {
                id: result.id,
                score: result.score,
                distance: result.distance,
                metadata: metadata.clone(),
            })
            .collect())
    }

    fn rebuild_vectors(&mut self) -> Result<()> {
        let mut vectors = VectorSearchIndex::new();
        for document in self.corpus.documents() {
            let vector = self
                .embedder
                .embed_counts(&document.term_counts, Some(&self.corpus))?;
            vectors.add(VectorRecord::new(document.id.clone(), vector))?;
        }
        self.vectors = vectors;
        Ok(())
    }

    fn validate_new_document_ids<'a>(&self, ids: impl IntoIterator<Item = &'a str>) -> Result<()> {
        let mut seen = BTreeSet::new();
        for id in ids {
            if id.trim().is_empty() {
                return Err(invalid_argument("document id must not be empty"));
            }
            if self.corpus.document(id).is_some() || !seen.insert(id.to_string()) {
                return Err(invalid_argument(format!(
                    "document id `{id}` already exists"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for cooccurrence config.
pub struct CooccurrenceConfig {
    /// The window size value.
    pub window_size: usize,
    /// The min term len value.
    pub min_term_len: usize,
}

impl Default for CooccurrenceConfig {
    fn default() -> Self {
        Self {
            window_size: 4,
            min_term_len: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for related term.
pub struct RelatedTerm {
    /// The term value.
    pub term: String,
    /// Number of items represented by this value.
    pub count: usize,
    /// Score assigned to this value.
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for cooccurrence graph.
pub struct CooccurrenceGraph {
    /// The config value.
    pub config: CooccurrenceConfig,
    term_counts: BTreeMap<String, usize>,
    pair_counts: BTreeMap<(String, String), usize>,
}

impl Default for CooccurrenceGraph {
    fn default() -> Self {
        Self::new(CooccurrenceConfig::default()).expect("default cooccurrence config is valid")
    }
}

impl CooccurrenceGraph {
    /// Creates a new value.
    pub fn new(config: CooccurrenceConfig) -> Result<Self> {
        if config.window_size == 0 {
            return Err(invalid_argument(
                "cooccurrence window size must be greater than zero",
            ));
        }
        Ok(Self {
            config,
            term_counts: BTreeMap::new(),
            pair_counts: BTreeMap::new(),
        })
    }

    /// Returns term counts.
    pub fn term_counts(&self) -> &BTreeMap<String, usize> {
        &self.term_counts
    }

    /// Returns pair counts.
    pub fn pair_counts(&self) -> &BTreeMap<(String, String), usize> {
        &self.pair_counts
    }

    /// Returns train text.
    pub fn train_text(&mut self, text: &str) {
        let tokens = tokenize_words(text)
            .into_iter()
            .filter(|term| term.chars().count() >= self.config.min_term_len)
            .collect::<Vec<_>>();
        self.train_tokens(&tokens);
    }

    /// Returns train tokens.
    pub fn train_tokens(&mut self, tokens: &[String]) {
        for token in tokens {
            *self.term_counts.entry(token.clone()).or_insert(0) += 1;
        }
        for start in 0..tokens.len() {
            let end = (start + self.config.window_size + 1).min(tokens.len());
            for right in start + 1..end {
                if tokens[start] == tokens[right] {
                    continue;
                }
                let pair = ordered_pair(&tokens[start], &tokens[right]);
                *self.pair_counts.entry(pair).or_insert(0) += 1;
            }
        }
    }

    /// Returns related terms.
    pub fn related_terms(&self, term: &str, limit: usize) -> Vec<RelatedTerm> {
        let normalized = term.to_lowercase();
        let base_count = self
            .term_counts
            .get(&normalized)
            .copied()
            .unwrap_or(0)
            .max(1) as f32;
        let mut related = Vec::new();
        for ((left, right), count) in &self.pair_counts {
            let other = if left == &normalized {
                Some(right)
            } else if right == &normalized {
                Some(left)
            } else {
                None
            };
            if let Some(other) = other {
                let other_count = self.term_counts.get(other).copied().unwrap_or(0).max(1) as f32;
                related.push(RelatedTerm {
                    term: other.clone(),
                    count: *count,
                    score: *count as f32 / (base_count * other_count).sqrt(),
                });
            }
        }
        related.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.count.cmp(&left.count))
                .then_with(|| left.term.cmp(&right.term))
        });
        if limit > 0 {
            related.truncate(limit);
        }
        related
    }
}

/// Returns text similarity.
pub fn text_similarity(left: &str, right: &str, embedder: &HashedTextEmbedder) -> Result<f32> {
    let left = embedder.embed_text(left)?;
    let right = embedder.embed_text(right)?;
    cosine_similarity(left.as_slice(), right.as_slice())
}

fn ordered_pair(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Returns pool embedding output.
pub fn pool_embedding_output(
    values: &[f32],
    shape: &[usize],
    attention_mask: &[i64],
    pooling: PoolingStrategy,
    normalize: bool,
) -> Result<DenseVector> {
    let (sequence, hidden) = match shape {
        [_hidden] => {
            let vector = DenseVector::new(values.to_vec())?;
            return if normalize {
                vector.l2_normalized()
            } else {
                Ok(vector)
            };
        }
        [sequence, hidden] => (*sequence, *hidden),
        [batch, sequence, hidden] if *batch == 1 => (*sequence, *hidden),
        _ => {
            return Err(invalid_argument(format!(
                "unsupported embedding output shape `{shape:?}`"
            )));
        }
    };
    if sequence == 0 || hidden == 0 || values.len() != sequence * hidden {
        return Err(invalid_argument(
            "embedding output shape does not match values",
        ));
    }

    let pooled = match pooling {
        PoolingStrategy::Cls => values[..hidden].to_vec(),
        PoolingStrategy::Mean => {
            let mut pooled = vec![0.0_f32; hidden];
            let mut count = 0.0_f32;
            for token_index in 0..sequence {
                if attention_mask.get(token_index).copied().unwrap_or(1) == 0 {
                    continue;
                }
                count += 1.0;
                let offset = token_index * hidden;
                for dimension in 0..hidden {
                    pooled[dimension] += values[offset + dimension];
                }
            }
            if count <= f32::EPSILON {
                return Err(invalid_argument(
                    "mean pooling requires at least one unmasked token",
                ));
            }
            for value in &mut pooled {
                *value /= count;
            }
            pooled
        }
    };
    let vector = DenseVector::new(pooled)?;
    if normalize {
        vector.l2_normalized()
    } else {
        Ok(vector)
    }
}

/// Returns softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f32>();
    if total <= f32::EPSILON || !total.is_finite() {
        return vec![0.0; logits.len()];
    }
    for value in &mut values {
        *value /= total;
    }
    values
}

#[cfg(feature = "model-bundles")]
fn required_bundle_file(bundle: &ModelBundle, remote_path: &str) -> Result<PathBuf> {
    bundle.file_path(remote_path).ok_or_else(|| {
        invalid_argument(format!(
            "model bundle `{}` is missing required file `{remote_path}`",
            bundle.manifest.name
        ))
    })
}

#[cfg(feature = "model-bundles")]
fn first_bundle_file_with_extension(bundle: &ModelBundle, extension: &str) -> Option<PathBuf> {
    bundle
        .manifest
        .files
        .keys()
        .find(|path| {
            Path::new(path).extension().and_then(|value| value.to_str()) == Some(extension)
        })
        .and_then(|path| bundle.file_path(path))
}

#[cfg(feature = "model-bundles")]
fn bundle_files_with_extension(bundle: &ModelBundle, extension: &str) -> Vec<PathBuf> {
    bundle
        .manifest
        .files
        .keys()
        .filter(|path| {
            Path::new(path).extension().and_then(|value| value.to_str()) == Some(extension)
        })
        .filter_map(|path| bundle.file_path(path))
        .collect::<Vec<_>>()
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
fn embedding_architecture_from_config(config: &Value) -> Result<CandleEmbeddingArchitecture> {
    let architectures = architectures_from_config(config);
    if architectures.contains(&"DistilBertModel") {
        return Ok(CandleEmbeddingArchitecture::DistilBert);
    }
    if architectures
        .iter()
        .any(|architecture| matches!(*architecture, "BertModel" | "SentenceTransformer"))
    {
        return Ok(CandleEmbeddingArchitecture::Bert);
    }
    Err(invalid_argument(format!(
        "unsupported Candle text architecture {}; supported: BertModel, DistilBertModel, SentenceTransformer",
        if architectures.is_empty() {
            "<missing>".to_string()
        } else {
            architectures.join(", ")
        },
    )))
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
fn architectures_from_config(config: &Value) -> Vec<&str> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
fn embedding_dimensions_from_config_path(config_path: &Path) -> Result<Option<usize>> {
    Ok(embedding_dimensions_from_config(&read_json(config_path)?))
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
fn embedding_dimensions_from_config(config: &Value) -> Option<usize> {
    config
        .get("hidden_size")
        .or_else(|| config.get("dim"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
fn model_max_tokens_from_config_path(config_path: &Path) -> Result<Option<usize>> {
    Ok(model_max_tokens_from_config(&read_json(config_path)?))
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
fn model_max_tokens_from_config(config: &Value) -> Option<usize> {
    config
        .get("max_position_embeddings")
        .or_else(|| config.get("max_seq_len"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
fn tokenizer_with_model_limit(tokenizer: TokenizerBundle, config: &Value) -> TokenizerBundle {
    match model_max_tokens_from_config(config) {
        Some(max_tokens) => tokenizer.max_length(max_tokens),
        None => tokenizer,
    }
}

#[cfg(all(feature = "tokenizers", feature = "model-bundles"))]
fn read_json(path: &Path) -> Result<Value> {
    let data = fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

#[cfg(feature = "candle")]
fn run_candle_embedder(
    config: &Value,
    model_paths: &[PathBuf],
    architecture: CandleEmbeddingArchitecture,
    tokens: &TokenizedText,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let device = text_model_runtime::candle_device_from_preference()?;
    let vb = candle_var_builder(model_paths, &device)?;
    let prefixes = model_prefix_candidates(config);

    let sequence_output = match architecture {
        CandleEmbeddingArchitecture::Bert => {
            let config: candle_bert::Config =
                serde_json::from_value(config.clone()).map_err(|err| {
                    invalid_argument(format!("failed to parse BERT config for Candle: {err}"))
                })?;
            let model = load_candle_bert_model(&vb, &config, &prefixes)?;
            let input_ids = candle_input_ids(tokens, &device)?;
            let token_type_ids = candle_token_type_ids(tokens, &device)?;
            let attention_mask = candle_attention_mask_keep(tokens, &device)?;
            model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))
                .map_err(candle_error)?
        }
        CandleEmbeddingArchitecture::DistilBert => {
            let config: candle_distilbert::Config = serde_json::from_value(config.clone())
                .map_err(|err| {
                    invalid_argument(format!(
                        "failed to parse DistilBERT config for Candle: {err}"
                    ))
                })?;
            let model = load_candle_distilbert_model(&vb, &config, &prefixes)?;
            let input_ids = candle_input_ids(tokens, &device)?;
            let attention_mask = candle_attention_mask_distil(tokens, &device)?;
            model
                .forward(&input_ids, &attention_mask)
                .map_err(candle_error)?
        }
    };

    let shape = sequence_output.dims().to_vec();
    let values = sequence_output
        .flatten_all()
        .map_err(candle_error)?
        .to_vec1::<f32>()
        .map_err(candle_error)?;
    Ok((values, shape))
}

#[cfg(feature = "candle")]
fn candle_var_builder<'a>(
    model_paths: &'a [PathBuf],
    device: &CandleDevice,
) -> Result<CandleVarBuilder<'a>> {
    let paths = model_paths
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<_>>();
    unsafe { CandleVarBuilder::from_mmaped_safetensors(&paths, CandleDType::F32, device) }
        .map_err(candle_error)
}

#[cfg(feature = "candle")]
fn load_candle_bert_model(
    vb: &CandleVarBuilder<'_>,
    config: &candle_bert::Config,
    prefixes: &[String],
) -> Result<candle_bert::BertModel> {
    let mut last_error = None;
    for prefix in prefixes {
        let model_vb = if prefix.is_empty() {
            vb.clone()
        } else {
            vb.pp(prefix)
        };
        match candle_bert::BertModel::load(model_vb, config) {
            Ok(model) => return Ok(model),
            Err(err) => last_error = Some(err.to_string()),
        }
    }
    Err(DetectError::Source(format!(
        "failed to load Candle BERT model for prefixes [{}]{}",
        prefixes.join(", "),
        last_error.map(|err| format!(": {err}")).unwrap_or_default()
    )))
}

#[cfg(feature = "candle")]
fn load_candle_distilbert_model(
    vb: &CandleVarBuilder<'_>,
    config: &candle_distilbert::Config,
    prefixes: &[String],
) -> Result<candle_distilbert::DistilBertModel> {
    let mut last_error = None;
    for prefix in prefixes {
        let model_vb = if prefix.is_empty() {
            vb.clone()
        } else {
            vb.pp(prefix)
        };
        match candle_distilbert::DistilBertModel::load(model_vb, config) {
            Ok(model) => return Ok(model),
            Err(err) => last_error = Some(err.to_string()),
        }
    }
    Err(DetectError::Source(format!(
        "failed to load Candle DistilBERT model for prefixes [{}]{}",
        prefixes.join(", "),
        last_error.map(|err| format!(": {err}")).unwrap_or_default()
    )))
}

#[cfg(feature = "candle")]
fn model_prefix_candidates(config: &Value) -> Vec<String> {
    let mut prefixes = Vec::new();
    push_unique_string(&mut prefixes, String::new());
    if let Some(model_type) = config.get("model_type").and_then(Value::as_str) {
        push_unique_string(&mut prefixes, model_type.to_string());
    }
    push_unique_string(&mut prefixes, "bert".to_string());
    push_unique_string(&mut prefixes, "distilbert".to_string());
    push_unique_string(&mut prefixes, "0.auto_model".to_string());
    push_unique_string(&mut prefixes, "auto_model".to_string());
    push_unique_string(&mut prefixes, "model".to_string());
    prefixes
}

#[cfg(feature = "candle")]
fn push_unique_string(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

#[cfg(feature = "candle")]
fn candle_input_ids(tokens: &TokenizedText, device: &CandleDevice) -> Result<CandleTensor> {
    let values = tokens
        .input_ids
        .iter()
        .map(|value| {
            u32::try_from(*value).map_err(|_| {
                invalid_argument(format!(
                    "tokenizer produced an out-of-range input id for Candle: {value}"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    CandleTensor::from_vec(values, (1, tokens.input_ids.len()), device).map_err(candle_error)
}

#[cfg(feature = "candle")]
fn candle_token_type_ids(tokens: &TokenizedText, device: &CandleDevice) -> Result<CandleTensor> {
    let values = match &tokens.token_type_ids {
        Some(values) => values
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| {
                    invalid_argument(format!(
                        "tokenizer produced an out-of-range token type id for Candle: {value}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?,
        None => vec![0_u32; tokens.input_ids.len()],
    };
    CandleTensor::from_vec(values, (1, tokens.input_ids.len()), device).map_err(candle_error)
}

#[cfg(feature = "candle")]
fn candle_attention_mask_keep(
    tokens: &TokenizedText,
    device: &CandleDevice,
) -> Result<CandleTensor> {
    let values = tokens
        .attention_mask
        .iter()
        .map(|value| if *value == 0 { 0_u32 } else { 1_u32 })
        .collect::<Vec<_>>();
    CandleTensor::from_vec(values, (1, tokens.attention_mask.len()), device).map_err(candle_error)
}

#[cfg(feature = "candle")]
fn candle_attention_mask_distil(
    tokens: &TokenizedText,
    device: &CandleDevice,
) -> Result<CandleTensor> {
    let values = tokens
        .attention_mask
        .iter()
        .map(|value| if *value == 0 { 1_u8 } else { 0_u8 })
        .collect::<Vec<_>>();
    CandleTensor::from_vec(values, (1, tokens.attention_mask.len()), device).map_err(candle_error)
}

#[cfg(feature = "onnx")]
fn runtime_onnx_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message)
        | runtime_onnx::OnnxRuntimeError::InvalidTensorShape(message) => invalid_argument(message),
        runtime_onnx::OnnxRuntimeError::Io(error) => DetectError::Io(error),
        other => DetectError::Source(other.to_string()),
    }
}

#[cfg(feature = "onnx")]
fn onnx_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("VIDEO_ANALYSIS_ONNX_TIMING")
            .map(|value| {
                let value = value.to_string_lossy();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false)
    })
}

#[cfg(feature = "onnx")]
fn log_onnx_stage_timing(stage: &str, model_path: &Path, elapsed: std::time::Duration, ok: bool) {
    eprintln!(
        "text-embeddings onnx timing: stage={stage} model={} elapsed_ms={} status={}",
        model_path.display(),
        elapsed.as_millis(),
        if ok { "ok" } else { "err" }
    );
}

#[cfg(feature = "onnx")]
fn log_onnx_stage_event(stage: &str, model_path: &Path, event: &str) {
    eprintln!(
        "text-embeddings onnx timing: stage={stage} model={} event={event}",
        model_path.display()
    );
}

#[cfg(feature = "candle")]
fn candle_error(error: candle_core::Error) -> DetectError {
    DetectError::Source(format!("Candle runtime error: {error}"))
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

fn validate_stream_id(stream_id: &str) -> Result<()> {
    if stream_id.trim().is_empty() {
        return Err(invalid_argument("stream id must not be empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeds_and_compares_text() {
        let embedder = HashedTextEmbedder::default();
        let same_topic =
            text_similarity("rust cargo crates", "cargo rust package", &embedder).unwrap();
        let different_topic =
            text_similarity("rust cargo crates", "oranges bananas", &embedder).unwrap();
        assert!(same_topic > different_topic);
    }

    #[test]
    fn validates_embedding_configuration_and_input() {
        assert!(HashedTextEmbedder::new(
            TextEmbeddingConfig {
                dimensions: 0,
                use_idf: false,
            },
            CorpusOptions::default(),
        )
        .is_err());

        let embedder = HashedTextEmbedder::new(
            TextEmbeddingConfig {
                dimensions: 8,
                use_idf: false,
            },
            CorpusOptions::default(),
        )
        .unwrap();

        assert_eq!(embedder.embed_text("rust rust").unwrap().dimensions(), 8);
        assert!(embedder.embed_text("... !!!").is_err());
    }

    #[test]
    fn searches_semantic_index() {
        let mut index = SemanticTextIndex::new(HashedTextEmbedder::default());
        index
            .add_document("rust", "rust cargo crates ownership")
            .unwrap();
        index
            .add_document("fruit", "oranges bananas apples")
            .unwrap();

        let results = index.search("cargo package", 1).unwrap();
        assert_eq!(results[0].id, "rust");
    }

    #[test]
    fn rejects_duplicate_documents_before_vector_insert() {
        let mut index = SemanticTextIndex::new(HashedTextEmbedder::default());
        index.add_document("rust", "rust cargo crates").unwrap();

        assert!(index.add_document("rust", "new text").is_err());
        assert_eq!(index.corpus().len(), 1);
        assert_eq!(index.search("cargo", 1).unwrap()[0].id, "rust");
    }

    #[test]
    fn idf_weighted_index_rebuilds_vectors_after_insert() {
        let embedder = HashedTextEmbedder::new(
            TextEmbeddingConfig {
                dimensions: 64,
                use_idf: true,
            },
            CorpusOptions::default(),
        )
        .unwrap();
        let mut index = SemanticTextIndex::new(embedder);
        index.add_document("rust", "rust cargo crates").unwrap();
        index.add_document("fruit", "orange banana apple").unwrap();

        let results = index.search("cargo", 2).unwrap();
        assert_eq!(results[0].id, "rust");
    }

    #[test]
    fn idf_weighted_batch_ingestion_rebuilds_vectors_once_after_corpus_update() {
        let embedder = HashedTextEmbedder::new(
            TextEmbeddingConfig {
                dimensions: 64,
                use_idf: true,
            },
            CorpusOptions::default(),
        )
        .unwrap();
        let documents = [
            TextDocument::new("rust", "rust cargo crates"),
            TextDocument::new("fruit", "orange banana apple"),
            TextDocument::new("search", "semantic retrieval search"),
        ];

        let index = SemanticTextIndex::from_documents(embedder, documents).unwrap();

        assert_eq!(index.corpus().len(), 3);
        assert_eq!(index.vectors.records().len(), 3);
        assert_eq!(index.search("cargo", 1).unwrap()[0].id, "rust");
    }

    #[test]
    fn finds_related_terms_from_context_windows() {
        let mut graph = CooccurrenceGraph::default();
        graph.train_text("rust cargo build rust cargo test rust ownership");

        let related = graph.related_terms("rust", 2);
        assert_eq!(related[0].term, "cargo");
    }

    #[test]
    fn cooccurrence_graph_filters_short_terms_and_validates_window() {
        assert!(CooccurrenceGraph::new(CooccurrenceConfig {
            window_size: 0,
            min_term_len: 1,
        })
        .is_err());

        let mut graph = CooccurrenceGraph::new(CooccurrenceConfig {
            window_size: 1,
            min_term_len: 4,
        })
        .unwrap();
        graph.train_text("AI rust ML cargo rust");

        assert_eq!(graph.term_counts().get("ai"), None);
        assert_eq!(graph.term_counts().get("ml"), None);
        assert_eq!(
            graph
                .pair_counts()
                .get(&("cargo".to_string(), "rust".to_string())),
            Some(&2)
        );
    }

    #[derive(Debug, Clone)]
    struct TinyEmbedder;

    impl TextEmbeddingBackend for TinyEmbedder {
        fn embed_text(&self, text: &str) -> Result<DenseVector> {
            let has_rust = text.contains("rust") || text.contains("cargo");
            DenseVector::new(if has_rust { [1.0, 0.0] } else { [0.0, 1.0] })
        }

        fn metadata(&self) -> TextEmbeddingMetadata {
            TextEmbeddingMetadata {
                backend: TextEmbeddingBackendKind::Custom,
                provenance: AnnotationProvenance::External,
                model_name: Some("tiny".to_string()),
                dimensions: Some(2),
            }
        }
    }

    #[test]
    fn generic_embedding_index_supports_trait_backends() {
        let mut index = EmbeddingSearchIndex::new(TinyEmbedder);
        index.add_document("rust", "rust cargo").unwrap();
        index.add_document("fruit", "orange banana").unwrap();
        assert!(index.add_document("rust", "duplicate").is_err());

        let results = index.search("cargo", 1).unwrap();
        assert_eq!(results[0].id, "rust");
    }

    #[test]
    fn generic_embedding_index_adds_text_segments_with_generated_ids() {
        let mut index = EmbeddingSearchIndex::new(TinyEmbedder);
        let rust_segment = TextSegment {
            segment_index: 4,
            timestamp: None,
            text: "rust cargo segment",
            language: Some("en"),
            is_final: true,
        };
        let fruit_segment = TextSegment {
            segment_index: 5,
            timestamp: None,
            text: "orange banana segment",
            language: Some("en"),
            is_final: true,
        };
        index.add_text_segment("subs", &rust_segment).unwrap();
        index.add_text_segment("subs", &fruit_segment).unwrap();

        let results = index.search("cargo", 1).unwrap();
        assert_eq!(results[0].id, "subs:4");
        assert!(matches!(
            index.add_text_segment("subs", &rust_segment),
            Err(DetectError::InvalidArgument(message))
                if message.contains("document id `subs:4` already exists")
        ));
        assert!(matches!(
            index.add_text_segment(" ", &rust_segment),
            Err(DetectError::InvalidArgument(message)) if message == "stream id must not be empty"
        ));
    }

    #[test]
    fn semantic_index_adds_text_segments_with_generated_ids() {
        let mut index = SemanticTextIndex::new(HashedTextEmbedder::default());
        let first = TextSegment {
            segment_index: 0,
            timestamp: None,
            text: "rust cargo crates",
            language: Some("en"),
            is_final: true,
        };
        let second = TextSegment {
            segment_index: 1,
            timestamp: None,
            text: "banana citrus apple",
            language: Some("en"),
            is_final: true,
        };
        index.add_text_segment("subs", &first).unwrap();
        index.add_text_segment("subs", &second).unwrap();

        assert_eq!(index.corpus().documents()[0].id, "subs:0");
        assert_eq!(index.corpus().documents()[1].id, "subs:1");
        assert_eq!(index.search("cargo", 1).unwrap()[0].id, "subs:0");
        assert!(matches!(
            index.add_text_segment("", &first),
            Err(DetectError::InvalidArgument(message)) if message == "stream id must not be empty"
        ));
    }

    #[test]
    fn stream_segment_document_constructor_matches_corpus_and_semantic_ids() {
        let segment = TextSegment {
            segment_index: 9,
            timestamp: None,
            text: "rust cargo segment",
            language: Some("en"),
            is_final: true,
        };
        let document = TextDocument::from_stream_segment("subs", &segment);
        let mut corpus = TfIdfCorpus::default();
        corpus.add_text_segment("subs", &segment).unwrap();
        let mut semantic = SemanticTextIndex::new(HashedTextEmbedder::default());
        semantic.add_text_segment("subs", &segment).unwrap();

        assert_eq!(document.id, "subs:9");
        assert_eq!(corpus.documents()[0].id, document.id);
        assert_eq!(semantic.corpus().documents()[0].id, document.id);
    }

    #[test]
    fn hashed_embedder_implements_embedding_backend() {
        fn embed_with_trait(backend: &dyn TextEmbeddingBackend, text: &str) -> Result<DenseVector> {
            backend.embed_text(text)
        }

        let embedder = HashedTextEmbedder::default();
        let direct = embedder.embed_text("rust cargo").unwrap();
        let via_trait = embed_with_trait(&embedder, "rust cargo").unwrap();
        assert_eq!(direct, via_trait);
    }

    #[test]
    fn semantic_matches_include_backend_metadata() {
        let mut index = EmbeddingSearchIndex::new(TinyEmbedder);
        index.add_document("rust", "rust cargo").unwrap();

        let results = index.search("cargo", 1).unwrap();
        assert_eq!(
            results[0].metadata.backend,
            TextEmbeddingBackendKind::Custom
        );
        assert_eq!(
            results[0].metadata.provenance,
            AnnotationProvenance::External
        );
        assert_eq!(results[0].metadata.model_name.as_deref(), Some("tiny"));

        let hashed = HashedTextEmbedder::default().metadata();
        assert_eq!(hashed.backend, TextEmbeddingBackendKind::Hashed);
        assert_eq!(hashed.provenance, AnnotationProvenance::Heuristic);
        assert_eq!(
            hashed.dimensions,
            Some(TextEmbeddingConfig::default().dimensions)
        );
    }

    #[test]
    fn hashed_embedder_can_emit_sparse_vectors() {
        let embedder = HashedTextEmbedder::default();
        let sparse = embedder
            .embed_text_sparse("rust cargo crates", None)
            .unwrap();
        assert_eq!(sparse.dimensions(), embedder.config.dimensions);
        assert!(sparse.nnz() > 0);
    }

    #[test]
    fn hashed_embedder_is_stable_and_normalized() {
        let embedder = HashedTextEmbedder::default();
        let left = embedder.embed_text("rust cargo crates").unwrap();
        let right = embedder.embed_text("rust cargo crates").unwrap();

        assert_eq!(left, right);
        assert!((vector_analysis_core::l2_norm(left.as_slice()).unwrap() - 1.0).abs() < 1.0e-6);
    }
}
