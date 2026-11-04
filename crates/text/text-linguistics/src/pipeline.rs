use std::path::PathBuf;

use crate::discourse::{DiscourseSegment, DocumentOutline, StyleProfile, TopicModel};
use crate::entities::{
    canonicalize_entities, extract_named_entities_with_labeler, CanonicalEntity, CorefCluster,
    EntityLinkingOptions, ExtractedEvent, NamedEntity, RelationTriple,
};
use crate::language::{
    primary_language_subtag, LanguageDecision, LanguageDetectionOptions, LanguageDetector,
    LanguagePrediction, LanguageProfile, SentenceLanguage, LANGUAGE_DETECTOR_PRODUCER,
};
use crate::local_models::{CandleTokenClassifier, SequenceLabeler};
use crate::morphology::{Lemma, LemmatizationResult, MorphAnnotation, PosAnnotation};
use crate::routing::{LinguisticOperation, OperationReport, OperationSupport, SentenceAnalysis};
use crate::syntax::{chunk_phrases, DependencyTree, PhraseChunk};
use crate::tokenization::{
    TokenAlignmentMap, TokenizationMode, TokenizerPolicy, TokenizerRegistry, TokenizerSelection,
};
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use crate::ud_model::UdModel;
use crate::ud_model::UdModelPreset;
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use model_runtime::{HuggingFaceDownloader, ModelBundle, ModelBundleStore};
use text_core::{
    build_annotation_graph_from_parts, split_paragraphs, split_sentence_spans, tokenize,
    AnnotationConfidence, AnnotationProvenance, Sentence, TextAnnotationGraph, TextDocument,
    TextProcessingOptions, Token,
};
use text_core::{DetectError, Result, TextSegment};
#[cfg(feature = "transcripts")]
use text_transcripts::{TranscriptSegment, TranscriptionContract, TranscriptionResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Variants describing analysis profile.
pub enum AnalysisProfile {
    /// The fast variant.
    Fast,
    /// The balanced variant.
    Balanced,
    #[default]
    /// The rich variant.
    Rich,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing entity recognition mode.
pub enum EntityRecognitionMode {
    /// Uses a local model backend. Fails when the model cannot be loaded.
    LocalModel,
    /// Named entity recognition is not requested: every sentence reports the
    /// operation as unsupported and the analysis has no entities. There is no
    /// heuristic fallback.
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Local entity model presets understood by text-linguistics.
pub enum ModelPreset {
    /// English-only `dslim/bert-base-NER` (CoNLL-2003: PER/ORG/LOC/MISC).
    BertBaseNer,
    /// Multilingual `Davlan/distilbert-base-multilingual-cased-ner-hrl` (AFL-3.0:
    /// PER/ORG/LOC/DATE), pinned by commit. The default for English, German and Spanish.
    DistilbertMultilingualNerHrl,
}

/// Pinned Hugging Face commit of `Davlan/distilbert-base-multilingual-cased-ner-hrl`.
pub const MULTILINGUAL_NER_HRL_REVISION: &str = "d421f57d5b1d36b375408588669e9340f9b11a89";

impl ModelPreset {
    /// Every local entity preset.
    pub const ALL: &'static [Self] = &[Self::BertBaseNer, Self::DistilbertMultilingualNerHrl];

    /// Stable preset id.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BertBaseNer => "bert-base-ner",
            Self::DistilbertMultilingualNerHrl => "distilbert-multilingual-ner-hrl",
        }
    }

    /// Hugging Face repository the preset loads.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::BertBaseNer => "dslim/bert-base-NER",
            Self::DistilbertMultilingualNerHrl => {
                "Davlan/distilbert-base-multilingual-cased-ner-hrl"
            }
        }
    }

    /// ISO 639-1 languages this preset is evaluated for (see
    /// `evaluation/corpora/named-entities-multilingual-v1.jsonl`).
    pub fn supported_languages(self) -> &'static [&'static str] {
        match self {
            Self::BertBaseNer => &["en"],
            Self::DistilbertMultilingualNerHrl => &["en", "de", "es"],
        }
    }

    /// Whether the preset supports the primary subtag of `language` (`de-AT` -> `de`).
    pub fn supports_language(self, language: &str) -> bool {
        let primary = primary_language_subtag(language);
        self.supported_languages().contains(&primary.as_str())
    }

    /// The Hugging Face bundle spec this preset downloads.
    #[cfg(all(feature = "candle", feature = "model-bundles"))]
    pub fn spec(self) -> model_runtime::HuggingFaceModelSpec {
        match self {
            Self::BertBaseNer => model_runtime::ModelPreset::BertBaseNer.spec(),
            Self::DistilbertMultilingualNerHrl => model_runtime::HuggingFaceModelSpec::new(
                self.repo_id(),
                model_runtime::ModelTask::TokenClassification,
            )
            .name(self.as_str())
            .revision(MULTILINGUAL_NER_HRL_REVISION)
            .file("config.json")
            .file("tokenizer_config.json")
            .file("vocab.txt")
            .optional_file("special_tokens_map.json")
            .first_available_file(["model.safetensors", "pytorch_model.bin"]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Entity recognition runtime selection.
///
/// The default is [`EntityRecognitionOptions::local_model`]: the local model
/// without downloads, failing when its bundle is missing. Use
/// [`EntityRecognitionOptions::local_model_with_downloads`] for explicit download
/// opt-in, or [`EntityRecognitionOptions::disabled`] to skip the operation.
pub struct EntityRecognitionOptions {
    /// Whether to run the local model or skip the operation.
    pub mode: EntityRecognitionMode,
    /// Directory containing local model bundles when `mode` is `LocalModel`.
    pub bundle_dir: PathBuf,
    /// Local model preset to load. Check [`ModelPreset::supports_language`] before
    /// trusting entities for a language.
    pub preset: ModelPreset,
    /// Whether a missing model bundle may be downloaded.
    pub auto_download: bool,
    /// Whether model download progress should be printed by the downloader.
    pub download_progress: bool,
    /// Maximum retry count for explicit model downloads.
    pub max_retries: usize,
}

impl Default for EntityRecognitionOptions {
    fn default() -> Self {
        Self::local_model()
    }
}

impl EntityRecognitionOptions {
    /// Returns local model options without downloading missing bundles.
    pub fn local_model() -> Self {
        Self {
            mode: EntityRecognitionMode::LocalModel,
            bundle_dir: PathBuf::from(".model-runtime"),
            preset: ModelPreset::DistilbertMultilingualNerHrl,
            auto_download: false,
            download_progress: false,
            max_retries: 1,
        }
    }

    /// Returns local model options that may download missing bundles.
    pub fn local_model_with_downloads() -> Self {
        Self {
            auto_download: true,
            download_progress: true,
            ..Self::local_model()
        }
    }

    /// Skips named entity recognition; sentences report it as unsupported.
    pub fn disabled() -> Self {
        Self {
            mode: EntityRecognitionMode::Disabled,
            ..Self::local_model()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// How lemmas, part of speech and morphological features are produced.
pub enum MorphologyMode {
    /// UD-treebank-trained taggers ([`UdModelPreset`]) plus UD-derived lemma tables
    /// for sentences in a language that has a preset (en, de, es); unsupported
    /// otherwise. Fails when a needed model cannot be loaded.
    LocalModel,
    /// Lemmas, POS and morphology are not requested: every sentence reports them as
    /// unsupported. There is no heuristic fallback.
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Lemma/POS/morphology runtime selection. The default is the local UD models
/// without downloads, failing when a needed bundle is missing.
///
/// Each sentence uses the model for its own language (see
/// [`LinguisticAnalysis::sentence_analyses`]). Pin one language for every sentence
/// with `TextProcessingOptions::language`.
pub struct MorphologyOptions {
    /// Whether to run the UD-trained local models or skip the layer.
    pub mode: MorphologyMode,
    /// Directory containing local model bundles when `mode` is `LocalModel`.
    pub bundle_dir: PathBuf,
    /// Whether missing model bundles may be downloaded.
    pub auto_download: bool,
    /// Whether model download progress should be printed by the downloader.
    pub download_progress: bool,
    /// Maximum retry count for explicit model downloads.
    pub max_retries: usize,
}

impl Default for MorphologyOptions {
    fn default() -> Self {
        Self::local_model()
    }
}

impl MorphologyOptions {
    /// Local UD models without downloading missing bundles.
    pub fn local_model() -> Self {
        Self {
            mode: MorphologyMode::LocalModel,
            bundle_dir: PathBuf::from(".model-runtime"),
            auto_download: false,
            download_progress: false,
            max_retries: 1,
        }
    }

    /// Local UD models that may download missing bundles.
    pub fn local_model_with_downloads() -> Self {
        Self {
            auto_download: true,
            download_progress: true,
            ..Self::local_model()
        }
    }

    /// Skips lemmas, POS and morphology; sentences report them as unsupported.
    pub fn disabled() -> Self {
        Self {
            mode: MorphologyMode::Disabled,
            ..Self::local_model()
        }
    }

    fn ud_model_source(&self, preset: UdModelPreset) -> Option<UdModelSource> {
        (self.mode == MorphologyMode::LocalModel).then(|| UdModelSource {
            preset,
            bundle_dir: self.bundle_dir.clone(),
            auto_download: self.auto_download,
            download_progress: self.download_progress,
            max_retries: self.max_retries,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// How dependency trees are produced.
pub enum DependencyParsingMode {
    /// The biaffine dependency head of the UD-treebank-trained models
    /// ([`UdModelPreset`]) for sentences in a language that has a preset (en, de,
    /// es); unsupported otherwise. Fails when a needed model cannot be loaded.
    LocalModel,
    /// Dependency trees are not requested: every sentence reports the layer as
    /// unsupported. There is no heuristic fallback.
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Dependency parser runtime selection. The default is the local UD models without
/// downloads, failing when a needed bundle is missing.
///
/// When this and [`MorphologyOptions`] select the same UD model from the same
/// bundle directory, each sentence is encoded once for both layers. Routing is per
/// sentence, as for [`MorphologyOptions`].
pub struct DependencyParsingOptions {
    /// Whether to run the UD-trained local models or skip the layer.
    pub mode: DependencyParsingMode,
    /// Directory containing local model bundles when `mode` is `LocalModel`.
    pub bundle_dir: PathBuf,
    /// Whether missing model bundles may be downloaded.
    pub auto_download: bool,
    /// Whether model download progress should be printed by the downloader.
    pub download_progress: bool,
    /// Maximum retry count for explicit model downloads.
    pub max_retries: usize,
}

impl Default for DependencyParsingOptions {
    fn default() -> Self {
        Self::local_model()
    }
}

impl DependencyParsingOptions {
    /// Local UD models without downloading missing bundles.
    pub fn local_model() -> Self {
        Self {
            mode: DependencyParsingMode::LocalModel,
            bundle_dir: PathBuf::from(".model-runtime"),
            auto_download: false,
            download_progress: false,
            max_retries: 1,
        }
    }

    /// Local UD models that may download missing bundles.
    pub fn local_model_with_downloads() -> Self {
        Self {
            auto_download: true,
            download_progress: true,
            ..Self::local_model()
        }
    }

    /// Skips dependency parsing; sentences report it as unsupported.
    pub fn disabled() -> Self {
        Self {
            mode: DependencyParsingMode::Disabled,
            ..Self::local_model()
        }
    }

    fn ud_model_source(&self, preset: UdModelPreset) -> Option<UdModelSource> {
        (self.mode == DependencyParsingMode::LocalModel).then(|| UdModelSource {
            preset,
            bundle_dir: self.bundle_dir.clone(),
            auto_download: self.auto_download,
            download_progress: self.download_progress,
            max_retries: self.max_retries,
        })
    }
}

/// Which UD model to run and how its bundles are loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    not(all(feature = "candle", feature = "model-bundles")),
    allow(dead_code)
)]
struct UdModelSource {
    preset: UdModelPreset,
    bundle_dir: PathBuf,
    auto_download: bool,
    download_progress: bool,
    max_retries: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for linguistic analysis options.
pub struct LinguisticAnalysisOptions {
    /// The processing value.
    pub processing: TextProcessingOptions,
    /// The language detection value.
    pub language_detection: LanguageDetectionOptions,
    /// The tokenizer policy value.
    pub tokenizer_policy: TokenizerPolicy,
    /// Lemma/POS/morphology backend selection.
    pub morphology: MorphologyOptions,
    /// Dependency parser backend selection.
    pub dependency_parsing: DependencyParsingOptions,
    /// The entity linking value.
    pub entity_linking: EntityLinkingOptions,
    /// The entity recognition value.
    pub entity_recognition: EntityRecognitionOptions,
    /// The enable alignment value.
    pub enable_alignment: bool,
    /// Requests coreference resolution. No model backend exists yet, so `true`
    /// fails the analysis instead of running a heuristic resolver.
    pub enable_coreference: bool,
    /// Requests event and relation extraction. No model backend exists yet, so
    /// `true` fails the analysis instead of running heuristic extraction.
    pub enable_events: bool,
    /// Requests discourse segmentation. No model backend exists; `true` fails.
    pub enable_discourse: bool,
    /// Requests topic modelling. No model backend exists; `true` fails.
    pub enable_topics: bool,
    /// Requests a style profile. No model backend exists; `true` fails.
    pub enable_style: bool,
}

impl Default for LinguisticAnalysisOptions {
    fn default() -> Self {
        Self {
            processing: TextProcessingOptions {
                include_punctuation: true,
                ..TextProcessingOptions::default()
            },
            language_detection: LanguageDetectionOptions::default(),
            tokenizer_policy: TokenizerPolicy::default(),
            morphology: MorphologyOptions::default(),
            dependency_parsing: DependencyParsingOptions::default(),
            entity_linking: EntityLinkingOptions::default(),
            entity_recognition: EntityRecognitionOptions::default(),
            enable_alignment: false,
            enable_coreference: false,
            enable_events: false,
            enable_discourse: false,
            enable_topics: false,
            enable_style: false,
        }
    }
}

impl LinguisticAnalysisOptions {
    /// Only the model-free operations: language detection, sentence segmentation and
    /// tokenization. Named
    /// entities, morphology, POS and dependency syntax are disabled and reported as
    /// unsupported for every sentence.
    pub fn without_models() -> Self {
        Self {
            entity_recognition: EntityRecognitionOptions::disabled(),
            morphology: MorphologyOptions::disabled(),
            dependency_parsing: DependencyParsingOptions::disabled(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Configuration for a local linguistic analysis pipeline.
pub struct TextNlpConfig {
    /// Analysis profile controlling the amount of deterministic annotation work.
    pub profile: AnalysisProfile,
    /// Detailed pipeline options.
    pub options: LinguisticAnalysisOptions,
    /// Whether tokenizer/model-backed metadata should influence provenance.
    pub prefer_model_backends: bool,
    /// Optional tokenizer/model family override for explicit model-backed runs.
    pub model_family: Option<String>,
}

impl Default for TextNlpConfig {
    fn default() -> Self {
        Self::rich()
    }
}

impl TextNlpConfig {
    /// Returns fast.
    pub fn fast() -> Self {
        Self {
            profile: AnalysisProfile::Fast,
            options: options_for_profile(AnalysisProfile::Fast),
            prefer_model_backends: false,
            model_family: None,
        }
    }

    /// Returns balanced.
    pub fn balanced() -> Self {
        Self {
            profile: AnalysisProfile::Balanced,
            options: options_for_profile(AnalysisProfile::Balanced),
            prefer_model_backends: false,
            model_family: None,
        }
    }

    /// Returns the rich profile with word tokenization and no subword alignment.
    pub fn rich() -> Self {
        let mut options = options_for_profile(AnalysisProfile::Rich);
        options.tokenizer_policy.mode = TokenizationMode::Word;
        options.enable_alignment = false;
        Self {
            profile: AnalysisProfile::Rich,
            options,
            prefer_model_backends: false,
            model_family: None,
        }
    }

    /// Returns the richer tokenizer/model-backed profile.
    ///
    /// This constructor may require tokenizer or model-bundle setup depending
    /// on enabled features and selected options.
    pub fn rich_with_model_backends() -> Self {
        Self {
            profile: AnalysisProfile::Rich,
            options: options_for_profile(AnalysisProfile::Rich),
            prefer_model_backends: true,
            model_family: Some("default-rich".to_string()),
        }
    }

    /// Builds this value from options.
    pub fn from_options(options: LinguisticAnalysisOptions) -> Self {
        Self {
            profile: AnalysisProfile::Balanced,
            options,
            prefer_model_backends: false,
            model_family: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for text nlp pipeline.
pub struct TextNlpPipeline {
    config: TextNlpConfig,
}

impl Default for TextNlpPipeline {
    fn default() -> Self {
        Self::new(TextNlpConfig::default())
    }
}

impl TextNlpPipeline {
    /// Creates a new value.
    pub fn new(config: TextNlpConfig) -> Self {
        Self { config }
    }

    /// Returns config.
    pub fn config(&self) -> &TextNlpConfig {
        &self.config
    }

    /// Returns analyze text.
    pub fn analyze_text(
        &self,
        text: &str,
    ) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
        analyze_text_with_config(text, &self.config)
    }

    /// Returns analyze text with a model-backed entity labeler.
    pub fn analyze_text_with_entity_labeler(
        &self,
        text: &str,
        entity_labeler: &mut dyn SequenceLabeler,
    ) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
        analyze_text_with_config_and_labeler(text, &self.config, Some(entity_labeler))
    }

    /// Returns analyze document.
    pub fn analyze_document(
        &self,
        document: &TextDocument<'_>,
    ) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
        self.analyze_text(document.text)
    }

    /// Returns analyze segment.
    pub fn analyze_segment(
        &self,
        segment: &TextSegment<'_>,
    ) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
        self.analyze_text(segment.text)
    }

    /// Returns analyze subtitle segments.
    #[cfg(feature = "transcripts")]
    pub fn analyze_subtitle_segments(
        &self,
        segments: &[TranscriptSegment],
    ) -> Result<SubtitleLinguisticAnalysis> {
        let cues = segments
            .iter()
            .cloned()
            .map(|cue| {
                let analysis = self.analyze_text(&cue.text)?;
                Ok(SubtitleCueLinguisticAnalysis { cue, analysis })
            })
            .collect::<Result<Vec<_>>>()?;
        let aggregate = self.analyze_text(&join_subtitle_text(segments, None))?;
        Ok(SubtitleLinguisticAnalysis { cues, aggregate })
    }

    /// Returns analyze transcription.
    #[cfg(feature = "transcripts")]
    pub fn analyze_transcription(
        &self,
        result: &TranscriptionResult,
    ) -> Result<SubtitleLinguisticAnalysis> {
        let cues = result
            .segments
            .iter()
            .cloned()
            .map(|cue| {
                let analysis = self.analyze_text(&cue.text)?;
                Ok(SubtitleCueLinguisticAnalysis { cue, analysis })
            })
            .collect::<Result<Vec<_>>>()?;
        let aggregate = self.analyze_text(&join_subtitle_text(
            &result.segments,
            result.text.as_deref(),
        ))?;
        Ok(SubtitleLinguisticAnalysis { cues, aggregate })
    }

    /// Returns analyze transcription contract.
    #[cfg(feature = "transcripts")]
    pub fn analyze_transcription_contract(
        &self,
        contract: &TranscriptionContract,
    ) -> Result<SubtitleLinguisticAnalysis> {
        let contract = contract
            .clone()
            .normalized()
            .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
        contract
            .validate_strict()
            .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
        let result = TranscriptionResult::from(contract);
        self.analyze_transcription(&result)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for linguistic analysis.
pub struct LinguisticAnalysis {
    /// The profile value.
    pub profile: AnalysisProfile,
    /// The provenance value.
    pub provenance: AnnotationProvenance,
    /// Confidence score for this value.
    pub confidence: AnnotationConfidence,
    /// The graph value.
    pub graph: TextAnnotationGraph,
    /// Language tag for this value.
    pub language: LanguageProfile,
    /// The tokenizer value.
    pub tokenizer: TokenizerSelection,
    /// The sentences value.
    pub sentences: Vec<Sentence>,
    /// Per-sentence language routing: the language each sentence was analyzed in and
    /// the support, producer and reason for each core operation. Parallel to
    /// `sentences`.
    pub sentence_analyses: Vec<SentenceAnalysis>,
    /// The tokens value.
    pub tokens: Vec<Token>,
    /// The alignments value.
    pub alignments: Option<TokenAlignmentMap>,
    /// Lemmas, only for sentences whose morphology report keeps results.
    pub lemmas: Vec<Lemma>,
    /// Morphology, only for sentences whose morphology report keeps results.
    pub morphology: Vec<MorphAnnotation>,
    /// POS tags, only for sentences whose part-of-speech report keeps results.
    pub pos: Vec<PosAnnotation>,
    /// The chunks value.
    pub chunks: Vec<PhraseChunk>,
    /// Dependency trees, only for sentences whose dependency report keeps results.
    pub dependencies: Vec<DependencyTree>,
    /// Entities, only for sentences whose entity report keeps results.
    pub entities: Vec<NamedEntity>,
    /// The canonical entities value.
    pub canonical_entities: Vec<CanonicalEntity>,
    /// The coreference value.
    pub coreference: Vec<CorefCluster>,
    /// The events value.
    pub events: Vec<ExtractedEvent>,
    /// The relations value.
    pub relations: Vec<RelationTriple>,
    /// The discourse value.
    pub discourse: Vec<DiscourseSegment>,
    /// The outline value.
    pub outline: DocumentOutline,
    /// The topics value.
    pub topics: TopicModel,
    /// The style value.
    pub style: StyleProfile,
}

impl LinguisticAnalysis {
    /// Returns token ref.
    pub fn token_ref(&self, token_index: usize) -> Option<&text_core::CanonicalToken> {
        self.graph.tokens.get(token_index)
    }

    /// Returns sentence ref.
    pub fn sentence_ref(&self, sentence_index: usize) -> Option<&text_core::AnnotatedSentence> {
        self.graph.sentences.get(sentence_index)
    }
}

fn options_for_profile(profile: AnalysisProfile) -> LinguisticAnalysisOptions {
    let mut options = LinguisticAnalysisOptions::default();
    match profile {
        AnalysisProfile::Fast => {
            options.tokenizer_policy.mode = TokenizationMode::Word;
            options.enable_alignment = false;
            options.enable_coreference = false;
            options.enable_events = false;
            options.enable_discourse = false;
            options.enable_topics = false;
            options.enable_style = false;
        }
        AnalysisProfile::Balanced => {}
        AnalysisProfile::Rich => {
            options.enable_alignment = true;
        }
    }
    options
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for subtitle cue linguistic analysis.
#[cfg(feature = "transcripts")]
pub struct SubtitleCueLinguisticAnalysis {
    /// The cue value.
    pub cue: TranscriptSegment,
    /// The analysis value.
    pub analysis: LinguisticAnalysis,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for subtitle linguistic analysis.
#[cfg(feature = "transcripts")]
pub struct SubtitleLinguisticAnalysis {
    /// The cues value.
    pub cues: Vec<SubtitleCueLinguisticAnalysis>,
    /// The aggregate value.
    pub aggregate: LinguisticAnalysis,
}

/// Returns analyze document.
pub fn analyze_document(
    document: &TextDocument<'_>,
    options: &LinguisticAnalysisOptions,
) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
    TextNlpPipeline::new(TextNlpConfig::from_options(options.clone())).analyze_document(document)
}

/// Returns analyze segment.
pub fn analyze_segment(
    segment: &TextSegment<'_>,
    options: &LinguisticAnalysisOptions,
) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
    TextNlpPipeline::new(TextNlpConfig::from_options(options.clone())).analyze_segment(segment)
}

/// Returns analyze subtitle segments.
#[cfg(feature = "transcripts")]
pub fn analyze_subtitle_segments(
    segments: &[TranscriptSegment],
    options: &LinguisticAnalysisOptions,
) -> Result<SubtitleLinguisticAnalysis> {
    TextNlpPipeline::new(TextNlpConfig::from_options(options.clone()))
        .analyze_subtitle_segments(segments)
}

/// Returns analyze transcription.
#[cfg(feature = "transcripts")]
pub fn analyze_transcription(
    result: &TranscriptionResult,
    options: &LinguisticAnalysisOptions,
) -> Result<SubtitleLinguisticAnalysis> {
    TextNlpPipeline::new(TextNlpConfig::from_options(options.clone())).analyze_transcription(result)
}

/// Returns analyze transcription contract.
#[cfg(feature = "transcripts")]
pub fn analyze_transcription_contract(
    contract: &TranscriptionContract,
    options: &LinguisticAnalysisOptions,
) -> Result<SubtitleLinguisticAnalysis> {
    TextNlpPipeline::new(TextNlpConfig::from_options(options.clone()))
        .analyze_transcription_contract(contract)
}

/// Returns analyze text.
pub fn analyze_text(
    text: &str,
    options: &LinguisticAnalysisOptions,
) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
    analyze_text_with_config(text, &TextNlpConfig::from_options(options.clone()))
}

/// Returns analyze text with a model-backed entity labeler.
pub fn analyze_text_with_entity_labeler(
    text: &str,
    options: &LinguisticAnalysisOptions,
    entity_labeler: &mut dyn SequenceLabeler,
) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
    analyze_text_with_config_and_labeler(
        text,
        &TextNlpConfig::from_options(options.clone()),
        Some(entity_labeler),
    )
}

fn analyze_text_with_config(
    text: &str,
    config: &TextNlpConfig,
) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
    analyze_text_with_config_and_labeler(text, config, None)
}

fn analyze_text_with_config_and_labeler(
    text: &str,
    config: &TextNlpConfig,
    entity_labeler: Option<&mut dyn SequenceLabeler>,
) -> std::result::Result<LinguisticAnalysis, LinguisticAnalysisError> {
    let options = &config.options;
    if options.enable_coreference {
        return Err(model_required(
            "coreference resolution",
            "set `enable_coreference: false`",
        ));
    }
    if options.enable_events {
        return Err(model_required(
            "event and relation extraction",
            "set `enable_events: false`",
        ));
    }
    for (requested, capability, opt_out) in [
        (
            options.enable_discourse,
            "discourse segmentation",
            "set `enable_discourse: false`",
        ),
        (
            options.enable_topics,
            "topic modelling",
            "set `enable_topics: false`",
        ),
        (
            options.enable_style,
            "style profiling",
            "set `enable_style: false`",
        ),
    ] {
        if requested {
            return Err(model_required(capability, opt_out));
        }
    }
    let sentences = split_sentence_spans(text, &options.processing);
    let language_detector = LanguageDetector {
        options: options.language_detection.clone(),
    };
    let language = if options.language_detection.sentence_level {
        language_detector.detect_sentences(text, &sentences)
    } else {
        language_detector.detect_sentences(text, &[])
    };
    let pinned_language = options
        .processing
        .language
        .as_deref()
        .map(primary_language_subtag)
        .filter(|language| !language.is_empty());
    let routes = sentence_routes(&language, &sentences, pinned_language.as_deref());
    let primary_language = pinned_language.clone().or_else(|| {
        language
            .primary
            .as_ref()
            .map(|prediction| prediction.language.clone())
    });
    let primary_language = primary_language.as_deref();
    let registry = TokenizerRegistry {
        policy: options.tokenizer_policy.clone(),
    };
    let tokenizer = registry.select(
        primary_language,
        Some("linguistic-analysis"),
        config.model_family.as_deref(),
    );

    let tokens = tokenize(text, &options.processing);
    let token_sentences = token_sentence_indices(&sentences, &tokens);
    let alignments = if options.enable_alignment {
        registry.align(text, &tokens, &tokenizer)?
    } else {
        None
    };

    // Each sentence goes to the UD model of its own language, when one is selected.
    let sentence_presets = routes
        .iter()
        .map(|route| {
            route
                .language
                .as_deref()
                .and_then(UdModelPreset::for_language)
        })
        .collect::<Vec<_>>();
    let ud = ud_model_layers(
        &options.morphology,
        &options.dependency_parsing,
        &sentence_presets,
        &sentences,
        &tokens,
    )?;
    // Lemmas, POS, morphology and trees come only from the UD models; sentences
    // without one get none (their reports say why).
    let lemmas = LemmatizationResult {
        language: primary_language.map(ToString::to_string),
        lemmas: ud.lemmas,
    };
    let pos = ud.pos;
    let morphology = ud.morphology;
    let chunks = chunk_phrases(text, &sentences, &tokens, &pos);
    let mut dependencies = ud.dependencies;
    dependencies.sort_by_key(|tree| tree.sentence_index);
    let entity_producer = if entity_labeler.is_some() {
        EntityProducer::CallerLabeler
    } else {
        match options.entity_recognition.mode {
            EntityRecognitionMode::LocalModel => {
                EntityProducer::Preset(options.entity_recognition.preset)
            }
            EntityRecognitionMode::Disabled => EntityProducer::Disabled,
        }
    };
    let used_entity_model = entity_producer != EntityProducer::Disabled;
    // Model entities are returned as the model produced them, never merged with
    // rule-based guesses.
    let entities = match (entity_labeler, entity_producer) {
        (Some(entity_labeler), _) => {
            extract_named_entities_with_labeler(text, &sentences, &tokens, entity_labeler)?
        }
        (None, EntityProducer::Preset(_)) => {
            let mut entity_labeler = local_entity_labeler(&options.entity_recognition)?;
            extract_named_entities_with_labeler(text, &sentences, &tokens, &mut entity_labeler)?
        }
        (None, _) => Vec::new(),
    };
    let sentence_analyses = sentence_analyses(
        routes,
        entity_producer,
        &options.morphology,
        &options.dependency_parsing,
        &sentence_presets,
    );
    // Language-specific layers keep results only for sentences whose report says so;
    // chunks, events, topics and style still see every sentence's internal layers.
    let keeps = |operation: LinguisticOperation, sentence: Option<usize>| {
        sentence
            .and_then(|sentence| sentence_analyses.get(sentence))
            .is_none_or(|analysis| analysis.keeps(operation))
    };
    let token_keeps = |operation: LinguisticOperation, token_index: usize| {
        keeps(
            operation,
            token_sentences.get(token_index).copied().flatten(),
        )
    };
    let entities = entities
        .into_iter()
        .filter(|entity| {
            keeps(
                LinguisticOperation::NamedEntities,
                Some(entity.sentence_index),
            )
        })
        .collect::<Vec<_>>();
    let canonical_entities = canonicalize_entities(&entities, &options.entity_linking);
    // Requests for these layers fail above; no backend fills them yet.
    let coreference = Vec::<CorefCluster>::new();
    let (events, relations) = (Vec::<ExtractedEvent>::new(), Vec::<RelationTriple>::new());
    // Requests for these layers fail above; no backend fills them yet.
    let outline = DocumentOutline {
        segments: Vec::new(),
    };
    let discourse = outline.segments.clone();
    let topics = TopicModel {
        descriptors: Vec::new(),
        clusters: Vec::new(),
    };
    let style = StyleProfile::default();
    let graph =
        build_annotation_graph_from_parts(text, &tokens, &sentences, &split_paragraphs(text));
    let confidence = summarize_analysis_confidence(
        &language,
        &lemmas.lemmas,
        &pos,
        &entities,
        &events,
        alignments.as_ref(),
    );
    let provenance = analysis_provenance(
        &tokenizer,
        alignments.as_ref(),
        config.prefer_model_backends,
        used_entity_model,
    );
    let lemmas = lemmas
        .lemmas
        .into_iter()
        .filter(|lemma| token_keeps(LinguisticOperation::Morphology, lemma.token_index))
        .collect();
    let morphology = morphology
        .into_iter()
        .filter(|annotation| token_keeps(LinguisticOperation::Morphology, annotation.token_index))
        .collect();
    let pos = pos
        .into_iter()
        .filter(|annotation| token_keeps(LinguisticOperation::PartOfSpeech, annotation.token_index))
        .collect();
    let dependencies = dependencies
        .into_iter()
        .filter(|tree| {
            keeps(
                LinguisticOperation::DependencySyntax,
                Some(tree.sentence_index),
            )
        })
        .collect();

    Ok(LinguisticAnalysis {
        profile: config.profile,
        provenance,
        confidence,
        graph,
        language,
        tokenizer,
        sentences,
        sentence_analyses,
        tokens,
        alignments,
        lemmas,
        morphology,
        pos,
        chunks,
        dependencies,
        entities,
        canonical_entities,
        coreference,
        events,
        relations,
        discourse,
        outline,
        topics,
        style,
    })
}

/// The language one sentence is analyzed in.
struct SentenceRoute {
    /// ISO 639-1 primary subtag every language-specific operation uses.
    language: Option<String>,
    /// Whether `language` came from `TextProcessingOptions::language`.
    pinned: bool,
    detection: SentenceLanguage,
}

fn sentence_routes(
    language: &LanguageProfile,
    sentences: &[Sentence],
    pinned: Option<&str>,
) -> Vec<SentenceRoute> {
    sentences
        .iter()
        .enumerate()
        .map(|(index, sentence)| {
            let detection = language
                .sentences
                .get(index)
                .cloned()
                .unwrap_or_else(|| document_level_decision(language, index, sentence));
            let language = match pinned {
                Some(pinned) => Some(pinned.to_string()),
                None => detection
                    .language()
                    .map(|prediction| primary_language_subtag(&prediction.language)),
            };
            SentenceRoute {
                language,
                pinned: pinned.is_some(),
                detection,
            }
        })
        .collect()
}

/// With sentence-level detection off, every sentence takes the document's decision.
fn document_level_decision(
    language: &LanguageProfile,
    sentence_index: usize,
    sentence: &Sentence,
) -> SentenceLanguage {
    let candidates = language
        .primary
        .iter()
        .chain(&language.alternatives)
        .filter(|prediction| prediction.language != "und")
        .map(|prediction| LanguagePrediction {
            reason: format!("document-level detection ({})", prediction.reason),
            ..prediction.clone()
        })
        .collect::<Vec<_>>();
    SentenceLanguage {
        sentence_index,
        span: sentence.span,
        decision: language.decision,
        candidates,
    }
}

/// The sentence each token starts in.
fn token_sentence_indices(sentences: &[Sentence], tokens: &[Token]) -> Vec<Option<usize>> {
    tokens
        .iter()
        .map(|token| {
            let start = token.span.byte_start;
            let index = sentences.partition_point(|sentence| sentence.span.byte_end <= start);
            sentences
                .get(index)
                .filter(|sentence| sentence.span.byte_start <= start)
                .map(|_| index)
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntityProducer {
    Disabled,
    Preset(ModelPreset),
    CallerLabeler,
}

/// Whether this build of text-linguistics can run its local models (the `candle` and
/// `model-bundles` features). Without them every model-backed operation fails.
pub fn local_models_compiled() -> bool {
    cfg!(all(feature = "candle", feature = "model-bundles"))
}

/// The failure for a requested operation that has no model backend. The pipeline
/// never substitutes heuristic output.
fn model_required(capability: &str, opt_out: &str) -> LinguisticAnalysisError {
    LinguisticAnalysisError::ModelRequired {
        capability: capability.to_owned(),
        hint: format!("{capability} has no model backend in text-linguistics yet; {opt_out} to analyze without it"),
    }
}

/// Errors from a linguistic analysis run.
#[derive(Debug)]
pub enum LinguisticAnalysisError {
    /// A requested capability needs a model that this build or configuration does not
    /// provide. No heuristic stand-in is run.
    ModelRequired {
        /// The capability that was requested, e.g. `coreference resolution`.
        capability: String,
        /// How to provide the model or opt out of the capability.
        hint: String,
    },
    /// Input, tokenizer or model execution failed.
    Detect(DetectError),
}

impl std::fmt::Display for LinguisticAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelRequired { hint, .. } => write!(f, "model required: {hint}"),
            Self::Detect(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for LinguisticAnalysisError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelRequired { .. } => None,
            Self::Detect(error) => Some(error),
        }
    }
}

impl From<DetectError> for LinguisticAnalysisError {
    fn from(error: DetectError) -> Self {
        Self::Detect(error)
    }
}

/// Lets callers that report shared detection errors keep using `?`; the model-required
/// case becomes an `InvalidArgument` carrying the same message.
impl From<LinguisticAnalysisError> for DetectError {
    fn from(error: LinguisticAnalysisError) -> Self {
        match error {
            LinguisticAnalysisError::Detect(error) => error,
            model_required @ LinguisticAnalysisError::ModelRequired { .. } => {
                DetectError::InvalidArgument(model_required.to_string())
            }
        }
    }
}

fn sentence_analyses(
    routes: Vec<SentenceRoute>,
    entities: EntityProducer,
    morphology: &MorphologyOptions,
    dependencies: &DependencyParsingOptions,
    sentence_presets: &[Option<UdModelPreset>],
) -> Vec<SentenceAnalysis> {
    use LinguisticOperation as Op;
    use OperationSupport as Support;

    routes
        .into_iter()
        .zip(sentence_presets)
        .map(|(route, &preset)| {
            let language = route.language.as_deref();
            let detection = language_detection_report(&route);
            let segmentation = OperationReport::new(
                Op::SentenceSegmentation,
                Support::Supported,
                Some("text-core:sentence-rules".to_string()),
                "language-neutral sentence boundary rules",
            );
            let tokenization = OperationReport::new(
                Op::Tokenization,
                Support::Supported,
                Some("text-core:word-rules".to_string()),
                "language-neutral word tokenization rules",
            );
            let named_entities = match (entities, language) {
                (EntityProducer::Disabled, _) => OperationReport::new(
                    Op::NamedEntities,
                    Support::Unsupported,
                    None,
                    "not requested (EntityRecognitionMode::Disabled); no heuristic fallback",
                ),
                (EntityProducer::CallerLabeler, _) => OperationReport::new(
                    Op::NamedEntities,
                    Support::Partial,
                    Some("caller-sequence-labeler".to_string()),
                    "caller-supplied labeler; its language support is not declared",
                ),
                (EntityProducer::Preset(_), None) => undecided_report(Op::NamedEntities),
                (EntityProducer::Preset(preset), Some(language))
                    if preset.supports_language(language) =>
                {
                    OperationReport::new(
                        Op::NamedEntities,
                        Support::Supported,
                        Some(format!("ner-model:{}", preset.as_str())),
                        format!("{} is gold-scored for `{language}`", preset.repo_id()),
                    )
                }
                (EntityProducer::Preset(preset), Some(language)) => OperationReport::new(
                    Op::NamedEntities,
                    Support::Unsupported,
                    None,
                    format!(
                        "{} supports {}, not `{language}`",
                        preset.as_str(),
                        preset.supported_languages().join(", ")
                    ),
                ),
            };
            let morphology_from_ud = morphology.mode == MorphologyMode::LocalModel;
            let dependencies_from_ud = dependencies.mode == DependencyParsingMode::LocalModel;
            let operations = vec![
                detection,
                segmentation,
                tokenization,
                named_entities,
                layer_report(
                    Op::Morphology,
                    language,
                    morphology_from_ud,
                    preset,
                    "MorphologyMode::Disabled",
                ),
                layer_report(
                    Op::PartOfSpeech,
                    language,
                    morphology_from_ud,
                    preset,
                    "MorphologyMode::Disabled",
                ),
                layer_report(
                    Op::DependencySyntax,
                    language,
                    dependencies_from_ud,
                    preset,
                    "DependencyParsingMode::Disabled",
                ),
            ];
            SentenceAnalysis {
                sentence_index: route.detection.sentence_index,
                span: route.detection.span,
                language: route.language,
                language_pinned: route.pinned,
                detection: route.detection,
                operations,
            }
        })
        .collect()
}

fn language_detection_report(route: &SentenceRoute) -> OperationReport {
    let operation = LinguisticOperation::LanguageDetection;
    if route.pinned {
        return OperationReport::new(
            operation,
            OperationSupport::Supported,
            Some("caller".to_string()),
            format!(
                "language pinned to `{}` by the caller",
                route.language.as_deref().unwrap_or_default()
            ),
        );
    }
    let detection = &route.detection;
    let producer = Some(LANGUAGE_DETECTOR_PRODUCER.to_string());
    let candidates = detection
        .candidates
        .iter()
        .map(|candidate| format!("{} {:.2}", candidate.language, candidate.confidence))
        .collect::<Vec<_>>()
        .join(", ");
    match detection.decision {
        LanguageDecision::Detected => OperationReport::new(
            operation,
            OperationSupport::Supported,
            producer,
            format!("detected ({candidates})"),
        ),
        LanguageDecision::Ambiguous => OperationReport::new(
            operation,
            OperationSupport::Unknown,
            producer,
            format!("ambiguous, no candidate reached the confidence threshold ({candidates})"),
        ),
        LanguageDecision::TooShort => OperationReport::new(
            operation,
            OperationSupport::Unknown,
            producer,
            "too few words to decide a language",
        ),
        LanguageDecision::UnsupportedScript => OperationReport::new(
            operation,
            OperationSupport::Unknown,
            producer,
            detection
                .candidates
                .first()
                .map(|candidate| candidate.reason.clone())
                .unwrap_or_else(|| "script not covered by the detector".to_string()),
        ),
    }
}

fn undecided_report(operation: LinguisticOperation) -> OperationReport {
    OperationReport::new(
        operation,
        OperationSupport::Unknown,
        None,
        "sentence language undecided; no language-specific producer ran",
    )
}

/// Report for a lemma/morphology, POS or dependency layer. `requested` says whether
/// the layer's UD models are selected; `preset` is the sentence's UD model, if any.
fn layer_report(
    operation: LinguisticOperation,
    language: Option<&str>,
    requested: bool,
    preset: Option<UdModelPreset>,
    disabled_mode: &str,
) -> OperationReport {
    match (requested, language, preset) {
        (false, _, _) => OperationReport::new(
            operation,
            OperationSupport::Unsupported,
            None,
            format!("not requested ({disabled_mode}); no heuristic fallback"),
        ),
        (true, None, _) => undecided_report(operation),
        (true, Some(_), Some(preset)) => OperationReport::new(
            operation,
            OperationSupport::Supported,
            Some(format!("ud-model:{}", preset.as_str())),
            format!("{} (UD-treebank-trained, scored)", preset.repo_id()),
        ),
        (true, Some(language), None) => OperationReport::new(
            operation,
            OperationSupport::Unsupported,
            None,
            format!("no model for `{language}`"),
        ),
    }
}

/// Loaded NER sessions keyed by bundle directory and preset (default device); clones share
/// the weights, so each model is read from disk once per process.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn local_entity_labeler(options: &EntityRecognitionOptions) -> Result<CandleTokenClassifier> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    type Cache = Mutex<HashMap<(PathBuf, &'static str), CandleTokenClassifier>>;
    static LABELERS: OnceLock<Cache> = OnceLock::new();
    let key = (options.bundle_dir.clone(), options.preset.as_str());
    let cache = LABELERS.get_or_init(Default::default);
    if let Some(labeler) = cache.lock().ok().and_then(|cache| cache.get(&key).cloned()) {
        return Ok(labeler);
    }
    let labeler = CandleTokenClassifier::from_bundle(ensure_local_entity_bundle(options)?)?;
    if let Ok(mut cache) = cache.lock() {
        cache.insert(key, labeler.clone());
    }
    Ok(labeler)
}

#[cfg(not(all(feature = "candle", feature = "model-bundles")))]
fn local_entity_labeler(options: &EntityRecognitionOptions) -> Result<CandleTokenClassifier> {
    let _ = options;
    Err(DetectError::InvalidArgument(
        "local entity recognition requires the `candle` and `model-bundles` features".to_string(),
    ))
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
type MorphologyLayers = (
    LemmatizationResult,
    Vec<PosAnnotation>,
    Vec<MorphAnnotation>,
);

/// UD-model annotations for the sentences routed to a UD model. Morphology covers
/// only those sentences' tokens; trees only those sentences.
#[derive(Default)]
struct UdLayers {
    lemmas: Vec<Lemma>,
    pos: Vec<PosAnnotation>,
    morphology: Vec<MorphAnnotation>,
    dependencies: Vec<DependencyTree>,
}

/// The distinct presets of `sentence_presets`, in first-seen order.
fn distinct_presets(sentence_presets: &[Option<UdModelPreset>]) -> Vec<UdModelPreset> {
    let mut presets = Vec::new();
    for preset in sentence_presets.iter().flatten() {
        if !presets.contains(preset) {
            presets.push(*preset);
        }
    }
    presets
}

#[cfg(not(all(feature = "candle", feature = "model-bundles")))]
fn ud_model_layers(
    morphology: &MorphologyOptions,
    dependencies: &DependencyParsingOptions,
    sentence_presets: &[Option<UdModelPreset>],
    _sentences: &[Sentence],
    _tokens: &[Token],
) -> Result<UdLayers> {
    for preset in distinct_presets(sentence_presets) {
        if let Some(source) = morphology
            .ud_model_source(preset)
            .or_else(|| dependencies.ud_model_source(preset))
        {
            return Err(DetectError::InvalidArgument(format!(
                "UD model `{}` requires the `candle` and `model-bundles` features",
                source.preset.as_str()
            )));
        }
    }
    Ok(UdLayers::default())
}

/// One sentence's tokens as the syntactic words the UD model analyzed.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
struct UdSentence<'a> {
    /// Index into the document's sentences.
    sentence_index: usize,
    tokens: Vec<usize>,
    /// Index of each token's first syntactic word.
    first_word: Vec<usize>,
    contractions: Vec<Option<[&'a str; 2]>>,
    words: Vec<crate::ud_model::UdWordAnalysis>,
}

/// Runs each language's UD model once per sentence routed to it, for the layers
/// that select a UD model, sharing the run when both layers select the same model.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn ud_model_layers(
    morphology: &MorphologyOptions,
    dependencies: &DependencyParsingOptions,
    sentence_presets: &[Option<UdModelPreset>],
    sentences: &[Sentence],
    tokens: &[Token],
) -> Result<UdLayers> {
    let mut layers = UdLayers::default();
    for preset in distinct_presets(sentence_presets) {
        let morphology_source = morphology.ud_model_source(preset);
        let dependency_source = dependencies.ud_model_source(preset);
        if morphology_source.is_none() && dependency_source.is_none() {
            continue;
        }
        let routed = (0..sentences.len())
            .filter(|&index| sentence_presets[index] == Some(preset))
            .collect::<Vec<_>>();
        let morphology_run = morphology_source
            .as_ref()
            .map(|source| run_ud_model(source, &routed, sentences, tokens))
            .transpose()?;
        let dependency_run = match &dependency_source {
            Some(source) if morphology_source.as_ref() != Some(source) => {
                Some(run_ud_model(source, &routed, sentences, tokens)?)
            }
            _ => None,
        };
        if dependency_source.is_some() {
            layers.dependencies.extend(ud_dependency_trees(
                dependency_run
                    .as_deref()
                    .or(morphology_run.as_deref())
                    .unwrap_or_default(),
            ));
        }
        if let Some(source) = morphology_source {
            let (lemmas, pos, morphology) = ud_morphology_layers(
                source.preset,
                morphology_run.as_deref().unwrap_or_default(),
                tokens,
            )?;
            layers.lemmas.extend(lemmas.lemmas);
            layers.pos.extend(pos);
            layers.morphology.extend(morphology);
        }
    }
    Ok(layers)
}

/// Analyzes the `routed` sentences with the UD model. Contractions (`zum`, `del`,
/// `don't`) become their two syntactic words.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn run_ud_model<'a>(
    source: &UdModelSource,
    routed: &[usize],
    sentences: &[Sentence],
    tokens: &'a [Token],
) -> Result<Vec<UdSentence<'a>>> {
    use crate::ud_model::split_contraction;

    let model = cached_ud_model(source)?;
    let language = source.preset.language();
    let groups = routed.iter().map(|&sentence_index| {
        let sentence = &sentences[sentence_index];
        let members = (0..tokens.len())
            .filter(|&index| {
                let start = tokens[index].span.byte_start;
                start >= sentence.span.byte_start && start < sentence.span.byte_end
            })
            .collect::<Vec<_>>();
        (sentence_index, members)
    });

    groups
        .map(|(sentence_index, group)| {
            let mut words = Vec::new();
            let mut first_word = Vec::new();
            let mut contractions = Vec::new();
            for &index in &group {
                first_word.push(words.len());
                let text = tokens[index].text.as_str();
                if let Some(parts) = split_contraction(language, text) {
                    words.extend(parts);
                    contractions.push(Some(parts));
                } else {
                    words.push(text);
                    contractions.push(None);
                }
            }
            Ok(UdSentence {
                sentence_index,
                words: model.analyze_words(&words)?,
                tokens: group,
                first_word,
                contractions,
            })
        })
        .collect()
}

/// Dependency trees over tokens from the UD model's word-level trees. A
/// contraction token takes the attachment of whichever of its words is closest to
/// the root (the adposition for `zum`, `del`), so the result stays a tree.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn ud_dependency_trees(run: &[UdSentence]) -> Vec<DependencyTree> {
    use crate::syntax::{DependencyEdge, DependencyNode, DependencyRelation};

    let mut trees = Vec::new();
    for sentence in run {
        let sentence_index = sentence.sentence_index;
        let words = &sentence.words;
        let mut word_token = vec![0; words.len()];
        let spans = (0..sentence.tokens.len())
            .map(|k| {
                let end = sentence
                    .first_word
                    .get(k + 1)
                    .copied()
                    .unwrap_or(words.len());
                sentence.first_word[k]..end
            })
            .collect::<Vec<_>>();
        for (k, span) in spans.iter().enumerate() {
            for word in span.clone() {
                word_token[word] = k;
            }
        }
        // Whether word `ancestor` dominates word `word` (both 0-based).
        let dominates = |ancestor: usize, word: usize| {
            let mut node = words[word].head;
            for _ in 0..words.len() {
                if node == 0 {
                    return false;
                }
                if node - 1 == ancestor {
                    return true;
                }
                node = words[node - 1].head;
            }
            false
        };

        let mut tree = DependencyTree {
            sentence_index,
            root_token_index: None,
            nodes: Vec::new(),
            edges: Vec::new(),
        };
        for (k, span) in spans.iter().enumerate() {
            let token_index = sentence.tokens[k];
            let Some(top) = span.clone().find(|&word| {
                !span
                    .clone()
                    .any(|other| other != word && dominates(other, word))
            }) else {
                continue;
            };
            let analysis = &words[top];
            let head = (analysis.head != 0).then(|| sentence.tokens[word_token[analysis.head - 1]]);
            let (relation, subtype) = DependencyRelation::parse_ud_deprel(&analysis.deprel);
            if head.is_none() {
                tree.root_token_index = Some(token_index);
            }
            tree.nodes.push(DependencyNode {
                token_index,
                head_token_index: head,
                relation,
                subtype: subtype.clone(),
            });
            if let Some(head) = head {
                tree.edges.push(DependencyEdge {
                    head_token_index: head,
                    dependent_token_index: token_index,
                    relation,
                    subtype,
                    confidence: analysis.head_confidence,
                });
            }
        }
        trees.push(tree);
    }
    trees
}

/// Lemma, POS and morphology layers from a UD model run, lemmatized with the
/// UD-derived table. A contraction token keeps its first word's UPOS and lemma. A
/// German/Spanish preposition-article contraction takes the article's features; an
/// English clitic contraction (`don't`, `John's`) the host's.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn ud_morphology_layers(
    preset: UdModelPreset,
    run: &[UdSentence],
    tokens: &[Token],
) -> Result<MorphologyLayers> {
    use crate::lemmas::UdLemmatizer;

    let language = preset.language();
    let lemmatizer = UdLemmatizer::for_language(language).ok_or_else(|| {
        DetectError::InvalidArgument(format!("no UD lemma table for `{language}`"))
    })?;
    let feats_word = usize::from(language != "en");

    let mut lemmas = vec![None; tokens.len()];
    let mut pos = vec![None; tokens.len()];
    let mut morphology = vec![None; tokens.len()];
    for sentence in run {
        let tagged = &sentence.words;
        for ((&index, &word), contraction) in sentence
            .tokens
            .iter()
            .zip(&sentence.first_word)
            .zip(&sentence.contractions)
        {
            let analysis = &tagged[word];
            let (lemma, feats, reason) = match contraction {
                Some(parts) => (
                    lemmatizer.lemmatize(parts[0], analysis.upos),
                    tagged[word + feats_word].feats.clone(),
                    format!(
                        "ud-model:{} contraction {}+{}",
                        preset.as_str(),
                        parts[0],
                        parts[1]
                    ),
                ),
                None => (
                    lemmatizer.lemmatize(&tokens[index].text, analysis.upos),
                    analysis.feats.clone(),
                    format!("ud-model:{}", preset.as_str()),
                ),
            };
            lemmas[index] = Some(Lemma {
                token_index: index,
                value: lemma.clone(),
                language: Some(language.to_string()),
                confidence: analysis.confidence,
            });
            pos[index] = Some(PosAnnotation {
                token_index: index,
                tag: analysis.upos,
                confidence: analysis.confidence,
                reason,
            });
            morphology[index] = Some(MorphAnnotation {
                token_index: index,
                lemma: Some(lemma),
                tags: feats,
                confidence: analysis.confidence,
            });
        }
    }
    Ok((
        LemmatizationResult {
            language: Some(language.to_string()),
            lemmas: lemmas.into_iter().flatten().collect(),
        },
        pos.into_iter().flatten().collect(),
        morphology.into_iter().flatten().collect(),
    ))
}

/// Loaded UD models keyed by bundle directory and preset; each checkpoint is ~1.1 GB.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn cached_ud_model(source: &UdModelSource) -> Result<std::sync::Arc<UdModel>> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    type Cache = Mutex<HashMap<(PathBuf, &'static str), Arc<UdModel>>>;
    static MODELS: OnceLock<Cache> = OnceLock::new();
    let preset = source.preset;
    let key = (source.bundle_dir.clone(), preset.as_str());
    let cache = MODELS.get_or_init(Default::default);
    if let Some(model) = cache.lock().ok().and_then(|cache| cache.get(&key).cloned()) {
        return Ok(model);
    }
    let store = ModelBundleStore::new(&source.bundle_dir).downloader(
        HuggingFaceDownloader::new()
            .progress(source.download_progress)
            .max_retries(source.max_retries),
    );
    let ensure = |spec: model_runtime::HuggingFaceModelSpec| -> Result<ModelBundle> {
        let revision = spec.revision_value().unwrap_or("main").to_string();
        match store.load(&spec.name, &revision) {
            Ok(bundle) => Ok(bundle),
            Err(_) if source.auto_download => store.download(&spec).map_err(model_runtime_error),
            Err(error) => Err(model_runtime_error(error)),
        }
    };
    let bundle = ensure(preset.spec())?;
    let tokenizer = ensure(UdModelPreset::tokenizer_spec())?;
    let model = Arc::new(UdModel::from_bundles(preset, &bundle, &tokenizer)?);
    if let Ok(mut cache) = cache.lock() {
        cache.insert(key, Arc::clone(&model));
    }
    Ok(model)
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn ensure_local_entity_bundle(options: &EntityRecognitionOptions) -> Result<ModelBundle> {
    let spec = options.preset.spec();
    let store = local_model_bundle_store(options);
    let revision = spec.revision_value().unwrap_or("main").to_string();
    if let Ok(bundle) = store.load(&spec.name, &revision) {
        return Ok(bundle);
    }
    if !options.auto_download {
        return store
            .load(&spec.name, &revision)
            .map_err(model_runtime_error);
    }

    let store = local_model_bundle_store(options);
    store.download(&spec).map_err(model_runtime_error)
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn local_model_bundle_store(options: &EntityRecognitionOptions) -> ModelBundleStore {
    ModelBundleStore::new(&options.bundle_dir).downloader(
        HuggingFaceDownloader::new()
            .progress(options.download_progress)
            .max_retries(options.max_retries),
    )
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn model_runtime_error(error: model_runtime::ModelRuntimeError) -> DetectError {
    match error {
        model_runtime::ModelRuntimeError::InvalidArgument(message) => {
            DetectError::InvalidArgument(message)
        }
        model_runtime::ModelRuntimeError::Source(message) => DetectError::Source(message),
        model_runtime::ModelRuntimeError::Io(error) => DetectError::Io(error),
        // Registry 0.1.1 and the pinned workspace source share the stable
        // variants above, while the workspace source also reports cancellation.
        // Its Display text is "model operation cancelled". Preserve that message
        // (and other upstream error details) without weakening the feature checks
        // for either dependency graph.
        #[allow(unreachable_patterns)]
        // Registry has only the stable variants; source mode adds cancellation.
        other => DetectError::Source(other.to_string()),
    }
}

#[cfg(feature = "transcripts")]
fn join_subtitle_text(segments: &[TranscriptSegment], aggregate_text: Option<&str>) -> String {
    if let Some(text) = aggregate_text.filter(|text| !text.trim().is_empty()) {
        return text.to_string();
    }

    segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn summarize_analysis_confidence(
    language: &LanguageProfile,
    lemmas: &[Lemma],
    pos: &[PosAnnotation],
    entities: &[NamedEntity],
    events: &[ExtractedEvent],
    alignments: Option<&TokenAlignmentMap>,
) -> AnnotationConfidence {
    let mut values = Vec::new();
    if let Some(primary) = &language.primary {
        values.push(primary.confidence);
    }
    values.extend(lemmas.iter().map(|lemma| lemma.confidence));
    values.extend(pos.iter().map(|annotation| annotation.confidence));
    values.extend(entities.iter().map(|entity| entity.confidence));
    values.extend(events.iter().map(|event| event.confidence));
    if alignments.is_some() {
        values.push(0.85);
    }
    if values.is_empty() {
        AnnotationConfidence::new(0.0)
    } else {
        AnnotationConfidence::new(values.iter().sum::<f32>() / values.len() as f32)
    }
}

fn analysis_provenance(
    tokenizer: &TokenizerSelection,
    alignments: Option<&TokenAlignmentMap>,
    prefer_model_backends: bool,
    used_entity_model: bool,
) -> AnnotationProvenance {
    // Entities from a sequence-labeling model make the analysis model-derived too.
    if used_entity_model
        || alignments.is_some()
        || (prefer_model_backends && tokenizer.source.is_some())
    {
        AnnotationProvenance::Model
    } else {
        AnnotationProvenance::RuleBased
    }
}
