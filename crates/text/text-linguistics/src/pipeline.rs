#[cfg(all(feature = "candle", feature = "model-bundles"))]
use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::discourse::{
    build_style_profile, build_topic_model, DiscourseSegment, DocumentOutline, SectionClassifier,
    StyleProfile, TopicModel,
};
use crate::entities::{
    canonicalize_entities, extract_named_entities, extract_named_entities_with_labeler,
    merge_model_and_heuristic_entities, CanonicalEntity, CorefCluster, CorefResolver,
    EntityLinkingOptions, EventExtractor, ExtractedEvent, NamedEntity, RelationTriple,
};
use crate::language::{LanguageDetectionOptions, LanguageDetector, LanguageProfile, LexiconStore};
use crate::local_models::{CandleTokenClassifier, SequenceLabeler};
use crate::morphology::{
    annotate_morphology, lemmatize_tokens, Lemma, LemmaOptions, LemmatizationResult,
    MorphAnnotation, PosAnnotation, PosTagger, PosTaggingOptions,
};
use crate::syntax::{chunk_phrases, DependencyParser, DependencyTree, PhraseChunk};
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
    /// Uses a local model backend.
    LocalModel,
    /// Uses deterministic rules only.
    Heuristic,
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

fn primary_language_subtag(language: &str) -> String {
    language
        .trim()
        .split(['-', '_'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Entity recognition runtime selection.
///
/// Defaults are heuristic and no-download. Use [`EntityRecognitionOptions::local_model`]
/// when a local bundle has already been materialized, or
/// [`EntityRecognitionOptions::local_model_with_downloads`] for explicit
/// download opt-in.
pub struct EntityRecognitionOptions {
    /// Whether to use deterministic rules or a local model.
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
        Self::heuristic()
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

    /// Returns heuristic.
    pub fn heuristic() -> Self {
        Self {
            mode: EntityRecognitionMode::Heuristic,
            ..Self::local_model()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// How lemmas, part of speech and morphological features are produced.
pub enum MorphologyMode {
    /// English heuristics for every language.
    Heuristic,
    /// UD-treebank-trained taggers ([`UdModelPreset`]) plus UD-derived lemma tables
    /// for languages that have a preset (de, es); English heuristics otherwise.
    LocalModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Lemma/POS/morphology runtime selection. Defaults are heuristic and no-download.
pub struct MorphologyOptions {
    /// Whether to use heuristics or the UD-trained local models.
    pub mode: MorphologyMode,
    /// Language whose UD model to use; `None` uses the detected primary language.
    pub language: Option<String>,
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
        Self::heuristic()
    }
}

impl MorphologyOptions {
    /// Local UD models without downloading missing bundles.
    pub fn local_model() -> Self {
        Self {
            mode: MorphologyMode::LocalModel,
            language: None,
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

    /// English heuristics only.
    pub fn heuristic() -> Self {
        Self {
            mode: MorphologyMode::Heuristic,
            ..Self::local_model()
        }
    }

    fn ud_model_source(&self, detected: Option<&str>) -> Option<UdModelSource> {
        if self.mode != MorphologyMode::LocalModel {
            return None;
        }
        Some(UdModelSource {
            preset: UdModelPreset::for_language(self.language.as_deref().or(detected)?)?,
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
    /// English POS-pattern heuristics for every language.
    Heuristic,
    /// The biaffine dependency head of the UD-treebank-trained models
    /// ([`UdModelPreset`]) for languages that have a preset (de, es); heuristics otherwise.
    LocalModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Dependency parser runtime selection. Defaults are heuristic and no-download.
///
/// When this and [`MorphologyOptions`] select the same UD model from the same
/// bundle directory, each sentence is encoded once for both layers.
pub struct DependencyParsingOptions {
    /// Whether to use heuristics or the UD-trained local models.
    pub mode: DependencyParsingMode,
    /// Language whose UD model to use; `None` uses the detected primary language.
    pub language: Option<String>,
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
        Self::heuristic()
    }
}

impl DependencyParsingOptions {
    /// Local UD models without downloading missing bundles.
    pub fn local_model() -> Self {
        Self {
            mode: DependencyParsingMode::LocalModel,
            language: None,
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

    /// English heuristics only.
    pub fn heuristic() -> Self {
        Self {
            mode: DependencyParsingMode::Heuristic,
            ..Self::local_model()
        }
    }

    fn ud_model_source(&self, detected: Option<&str>) -> Option<UdModelSource> {
        if self.mode != DependencyParsingMode::LocalModel {
            return None;
        }
        Some(UdModelSource {
            preset: UdModelPreset::for_language(self.language.as_deref().or(detected)?)?,
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
    /// The lemma value.
    pub lemma: LemmaOptions,
    /// The pos value.
    pub pos: PosTaggingOptions,
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
    /// The enable coreference value.
    pub enable_coreference: bool,
    /// The enable events value.
    pub enable_events: bool,
    /// The enable discourse value.
    pub enable_discourse: bool,
    /// The enable topics value.
    pub enable_topics: bool,
    /// The enable style value.
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
            lemma: LemmaOptions::default(),
            pos: PosTaggingOptions::default(),
            morphology: MorphologyOptions::default(),
            dependency_parsing: DependencyParsingOptions::default(),
            entity_linking: EntityLinkingOptions::default(),
            entity_recognition: EntityRecognitionOptions::default(),
            enable_alignment: false,
            enable_coreference: true,
            enable_events: true,
            enable_discourse: true,
            enable_topics: true,
            enable_style: true,
        }
    }
}

impl LinguisticAnalysisOptions {
    /// Returns heuristic.
    pub fn heuristic() -> Self {
        Self {
            entity_recognition: EntityRecognitionOptions::heuristic(),
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

    /// Returns a rich deterministic profile without model-bundle requirements.
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
    pub fn analyze_text(&self, text: &str) -> Result<LinguisticAnalysis> {
        analyze_text_with_config(text, &self.config)
    }

    /// Returns analyze text with a model-backed entity labeler.
    pub fn analyze_text_with_entity_labeler(
        &self,
        text: &str,
        entity_labeler: &mut dyn SequenceLabeler,
    ) -> Result<LinguisticAnalysis> {
        analyze_text_with_config_and_labeler(text, &self.config, Some(entity_labeler))
    }

    /// Returns analyze document.
    pub fn analyze_document(&self, document: &TextDocument<'_>) -> Result<LinguisticAnalysis> {
        self.analyze_text(document.text)
    }

    /// Returns analyze segment.
    pub fn analyze_segment(&self, segment: &TextSegment<'_>) -> Result<LinguisticAnalysis> {
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
    /// The tokens value.
    pub tokens: Vec<Token>,
    /// The alignments value.
    pub alignments: Option<TokenAlignmentMap>,
    /// The lemmas value.
    pub lemmas: Vec<Lemma>,
    /// The morphology value.
    pub morphology: Vec<MorphAnnotation>,
    /// The pos value.
    pub pos: Vec<PosAnnotation>,
    /// The chunks value.
    pub chunks: Vec<PhraseChunk>,
    /// The dependencies value.
    pub dependencies: Vec<DependencyTree>,
    /// The entities value.
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
            options.entity_recognition = EntityRecognitionOptions::heuristic();
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
            options.enable_coreference = true;
            options.enable_events = true;
            options.enable_discourse = true;
            options.enable_topics = true;
            options.enable_style = true;
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
) -> Result<LinguisticAnalysis> {
    TextNlpPipeline::new(TextNlpConfig::from_options(options.clone())).analyze_document(document)
}

/// Returns analyze segment.
pub fn analyze_segment(
    segment: &TextSegment<'_>,
    options: &LinguisticAnalysisOptions,
) -> Result<LinguisticAnalysis> {
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
pub fn analyze_text(text: &str, options: &LinguisticAnalysisOptions) -> Result<LinguisticAnalysis> {
    analyze_text_with_config(text, &TextNlpConfig::from_options(options.clone()))
}

/// Returns analyze text with a model-backed entity labeler.
pub fn analyze_text_with_entity_labeler(
    text: &str,
    options: &LinguisticAnalysisOptions,
    entity_labeler: &mut dyn SequenceLabeler,
) -> Result<LinguisticAnalysis> {
    analyze_text_with_config_and_labeler(
        text,
        &TextNlpConfig::from_options(options.clone()),
        Some(entity_labeler),
    )
}

fn analyze_text_with_config(text: &str, config: &TextNlpConfig) -> Result<LinguisticAnalysis> {
    analyze_text_with_config_and_labeler(text, config, None)
}

fn analyze_text_with_config_and_labeler(
    text: &str,
    config: &TextNlpConfig,
    entity_labeler: Option<&mut dyn SequenceLabeler>,
) -> Result<LinguisticAnalysis> {
    let options = &config.options;
    let language_detector = LanguageDetector {
        options: options.language_detection.clone(),
        lexicons: LexiconStore::default(),
    };
    let language = language_detector.detect_text(text);
    let registry = TokenizerRegistry {
        policy: options.tokenizer_policy.clone(),
    };
    let tokenizer = registry.select(
        language
            .primary
            .as_ref()
            .map(|prediction| prediction.language.as_str()),
        Some("linguistic-analysis"),
        config.model_family.as_deref(),
    );

    let sentences = split_sentence_spans(text, &options.processing);
    let tokens = tokenize(text, &options.processing);
    let alignments = if options.enable_alignment {
        registry.align(text, &tokens, &tokenizer)?
    } else {
        None
    };
    let primary_language = language
        .primary
        .as_ref()
        .map(|prediction| prediction.language.as_str());
    let (ud_morphology, ud_dependencies) = ud_model_layers(
        options.morphology.ud_model_source(primary_language),
        options.dependency_parsing.ud_model_source(primary_language),
        &sentences,
        &tokens,
    )?;
    let (lemmas, pos, morphology) = ud_morphology.unwrap_or_else(|| {
        let lemmas = lemmatize_tokens(&tokens, primary_language, &options.lemma);
        let pos_tagger = PosTagger {
            options: options.pos.clone(),
        };
        let pos = pos_tagger.tag_tokens(&tokens, &lemmas);
        let morphology = annotate_morphology(&tokens, &lemmas, &pos);
        (lemmas, pos, morphology)
    });
    let chunks = chunk_phrases(text, &sentences, &tokens, &pos);
    let dependencies = ud_dependencies
        .unwrap_or_else(|| DependencyParser.parse_document(&sentences, &tokens, &pos));
    let heuristic_entities = extract_named_entities(text, &sentences, &tokens, &pos);
    let used_entity_model = entity_labeler.is_some()
        || options.entity_recognition.mode == EntityRecognitionMode::LocalModel;
    let entities = if let Some(entity_labeler) = entity_labeler {
        entities_with_model_labeler(
            text,
            &sentences,
            &tokens,
            heuristic_entities,
            entity_labeler,
        )?
    } else if options.entity_recognition.mode == EntityRecognitionMode::LocalModel {
        let mut entity_labeler = local_entity_labeler(&options.entity_recognition)?;
        entities_with_model_labeler(
            text,
            &sentences,
            &tokens,
            heuristic_entities,
            &mut entity_labeler,
        )?
    } else {
        heuristic_entities
    };
    let canonical_entities = canonicalize_entities(&entities, &options.entity_linking);
    let coreference = if options.enable_coreference {
        CorefResolver::default().resolve(&tokens, &canonical_entities)
    } else {
        Vec::new()
    };
    let (events, relations) = if options.enable_events {
        EventExtractor.extract(&dependencies, &tokens, &lemmas)
    } else {
        (Vec::new(), Vec::new())
    };
    let outline = if options.enable_discourse {
        SectionClassifier.classify(&sentences)
    } else {
        DocumentOutline {
            segments: Vec::new(),
        }
    };
    let discourse = outline.segments.clone();
    let topics = if options.enable_topics {
        build_topic_model(
            &sentences,
            &tokens,
            &lemmas,
            &chunks,
            language.primary.as_ref(),
        )
    } else {
        TopicModel {
            descriptors: Vec::new(),
            clusters: Vec::new(),
        }
    };
    let style = if options.enable_style {
        build_style_profile(text, &sentences, &tokens, &lemmas.lemmas, &dependencies)
    } else {
        StyleProfile::default()
    };
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

    Ok(LinguisticAnalysis {
        profile: config.profile,
        provenance,
        confidence,
        graph,
        language,
        tokenizer,
        sentences,
        tokens,
        alignments,
        lemmas: lemmas.lemmas,
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

fn entities_with_model_labeler(
    text: &str,
    sentences: &[Sentence],
    tokens: &[Token],
    heuristic_entities: Vec<NamedEntity>,
    entity_labeler: &mut dyn SequenceLabeler,
) -> Result<Vec<NamedEntity>> {
    let model_entities =
        extract_named_entities_with_labeler(text, sentences, tokens, entity_labeler)?;
    Ok(merge_model_and_heuristic_entities(
        model_entities,
        heuristic_entities,
    ))
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn local_entity_labeler(options: &EntityRecognitionOptions) -> Result<CandleTokenClassifier> {
    let bundle = ensure_local_entity_bundle(options)?;
    CandleTokenClassifier::from_bundle(bundle)
}

#[cfg(not(all(feature = "candle", feature = "model-bundles")))]
fn local_entity_labeler(options: &EntityRecognitionOptions) -> Result<CandleTokenClassifier> {
    let _ = options;
    Err(DetectError::InvalidArgument(
        "local entity recognition requires the `candle` and `model-bundles` features".to_string(),
    ))
}

type MorphologyLayers = (
    LemmatizationResult,
    Vec<PosAnnotation>,
    Vec<MorphAnnotation>,
);

/// The UD-model morphology and dependency layers, for whichever of the two selects
/// a UD model.
type UdLayers = (Option<MorphologyLayers>, Option<Vec<DependencyTree>>);

#[cfg(not(all(feature = "candle", feature = "model-bundles")))]
fn ud_model_layers(
    morphology: Option<UdModelSource>,
    dependencies: Option<UdModelSource>,
    _sentences: &[Sentence],
    _tokens: &[Token],
) -> Result<UdLayers> {
    match morphology.or(dependencies) {
        Some(source) => Err(DetectError::InvalidArgument(format!(
            "UD model `{}` requires the `candle` and `model-bundles` features",
            source.preset.as_str()
        ))),
        None => Ok((None, None)),
    }
}

/// One sentence's tokens as the syntactic words the UD model analyzed.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
struct UdSentence {
    /// Index into the document's sentences; `None` for tokens outside every sentence.
    sentence_index: Option<usize>,
    tokens: Vec<usize>,
    /// Index of each token's first syntactic word.
    first_word: Vec<usize>,
    contractions: Vec<Option<[&'static str; 2]>>,
    words: Vec<crate::ud_model::UdWordAnalysis>,
}

/// Runs the UD model once per sentence for the layers that select it, sharing the
/// run when both select the same model.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn ud_model_layers(
    morphology: Option<UdModelSource>,
    dependencies: Option<UdModelSource>,
    sentences: &[Sentence],
    tokens: &[Token],
) -> Result<UdLayers> {
    let morphology_run = morphology
        .as_ref()
        .map(|source| run_ud_model(source, sentences, tokens))
        .transpose()?;
    let dependency_run = match &dependencies {
        Some(source) if morphology.as_ref() != Some(source) => {
            Some(run_ud_model(source, sentences, tokens)?)
        }
        _ => None,
    };
    let trees = dependencies.map(|_| {
        ud_dependency_trees(
            dependency_run
                .as_deref()
                .or(morphology_run.as_deref())
                .unwrap_or_default(),
        )
    });
    let layers = morphology
        .map(|source| {
            ud_morphology_layers(
                source.preset,
                morphology_run.as_deref().unwrap_or_default(),
                tokens,
            )
        })
        .transpose()?;
    Ok((layers, trees))
}

/// Analyzes every sentence with the UD model. Contractions (`zum`, `del`) become
/// their two syntactic words.
#[cfg(all(feature = "candle", feature = "model-bundles"))]
fn run_ud_model(
    source: &UdModelSource,
    sentences: &[Sentence],
    tokens: &[Token],
) -> Result<Vec<UdSentence>> {
    use crate::ud_model::split_contraction;

    let model = cached_ud_model(source)?;
    let language = source.preset.language();
    let mut groups = sentences
        .iter()
        .enumerate()
        .map(|(sentence_index, sentence)| {
            let members = (0..tokens.len())
                .filter(|&index| {
                    let start = tokens[index].span.byte_start;
                    start >= sentence.span.byte_start && start < sentence.span.byte_end
                })
                .collect::<Vec<_>>();
            (Some(sentence_index), members)
        })
        .collect::<Vec<_>>();
    let grouped = groups
        .iter()
        .flat_map(|(_, members)| members)
        .copied()
        .collect::<BTreeSet<_>>();
    let rest = (0..tokens.len())
        .filter(|index| !grouped.contains(index))
        .collect::<Vec<_>>();
    groups.push((None, rest));

    groups
        .into_iter()
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
        let Some(sentence_index) = sentence.sentence_index else {
            continue;
        };
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
/// UD-derived table. A contraction token keeps the adposition's UPOS and lemma and
/// the article's features.
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
                    tagged[word + 1].feats.clone(),
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
        AnnotationProvenance::Heuristic
    }
}
