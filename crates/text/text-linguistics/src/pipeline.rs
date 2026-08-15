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
    annotate_morphology, lemmatize_tokens, Lemma, LemmaOptions, MorphAnnotation, PosAnnotation,
    PosTagger, PosTaggingOptions,
};
use crate::syntax::{chunk_phrases, DependencyParser, DependencyTree, PhraseChunk};
use crate::tokenization::{
    TokenAlignmentMap, TokenizationMode, TokenizerPolicy, TokenizerRegistry, TokenizerSelection,
};
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use jobs_core::BackgroundJobRunner;
#[cfg(all(feature = "candle", feature = "model-bundles"))]
use model_runtime::{
    jobs::spawn_model_download_job, HuggingFaceDownloader, ModelBundle, ModelBundleStore,
};
use text_core::{
    build_annotation_graph_from_parts, split_paragraphs, split_sentence_spans, tokenize,
    AnnotationConfidence, AnnotationProvenance, Sentence, TextAnnotationGraph, TextDocument,
    TextProcessingOptions, Token,
};
use text_core::{AnalysisEvent, DetectError, OwnedTextSegment, Result, TextAnalyzer, TextSegment};
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
    /// BERT base NER token-classification model.
    BertBaseNer,
}

impl ModelPreset {
    /// Stable preset id.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BertBaseNer => "bert-base-ner",
        }
    }

    #[cfg(all(feature = "candle", feature = "model-bundles"))]
    pub(crate) fn spec(self) -> model_runtime::HuggingFaceModelSpec {
        match self {
            Self::BertBaseNer => model_runtime::ModelPreset::BertBaseNer.spec(),
        }
    }
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
    /// Local model preset to load.
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
            preset: ModelPreset::BertBaseNer,
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
    let lemmas = lemmatize_tokens(
        &tokens,
        language
            .primary
            .as_ref()
            .map(|prediction| prediction.language.as_str()),
        &options.lemma,
    );
    let pos_tagger = PosTagger {
        options: options.pos.clone(),
    };
    let pos = pos_tagger.tag_tokens(&tokens, &lemmas);
    let morphology = annotate_morphology(&tokens, &lemmas, &pos);
    let chunks = chunk_phrases(text, &sentences, &tokens, &pos);
    let dependency_parser = DependencyParser;
    let dependencies = dependency_parser.parse_document(&sentences, &tokens, &pos);
    let heuristic_entities = extract_named_entities(text, &sentences, &tokens, &pos);
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
    if options.preset != ModelPreset::BertBaseNer {
        return Err(DetectError::InvalidArgument(format!(
            "unsupported local entity model preset `{}`; expected `{}`",
            options.preset.as_str(),
            ModelPreset::BertBaseNer.as_str()
        )));
    }
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

    let runner = BackgroundJobRunner::default();
    let store = local_model_bundle_store(options);
    let mut handle = spawn_model_download_job(&runner, spec, store).map_job_error()?;
    handle.join_result().map_job_error()
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
    }
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
trait JobResultExt<T> {
    fn map_job_error(self) -> Result<T>;
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
impl<T> JobResultExt<T> for jobs_core::Result<T> {
    fn map_job_error(self) -> Result<T> {
        self.map_err(|err| DetectError::Source(err.to_string()))
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
) -> AnnotationProvenance {
    if alignments.is_some() || (prefer_model_backends && tokenizer.source.is_some()) {
        AnnotationProvenance::Tokenizer
    } else {
        AnnotationProvenance::Heuristic
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for linguistic analyzer.
pub struct LinguisticAnalyzer {
    options: LinguisticAnalysisOptions,
    segment_buffer: Vec<OwnedTextSegment>,
}

impl LinguisticAnalyzer {
    /// Creates a new value.
    pub fn new(options: LinguisticAnalysisOptions) -> Self {
        Self {
            options,
            segment_buffer: Vec::new(),
        }
    }

    fn segment_events(
        &self,
        segment: &TextSegment<'_>,
        analysis: &LinguisticAnalysis,
    ) -> Vec<AnalysisEvent> {
        let mut events = Vec::new();
        if let Some(language) = &analysis.language.primary {
            let mut event =
                AnalysisEvent::new(self.name(), format!("text:language:{}", language.language))
                    .score(language.confidence);
            if let Some(timestamp) = segment.timestamp {
                event = event.at_timestamp(timestamp);
            }
            events.push(event);
        }
        events.extend(analysis.entities.iter().map(|entity| {
            let mut event = AnalysisEvent::new(
                self.name(),
                format!(
                    "text:entity:{:?}:{}",
                    entity.entity_type,
                    entity.normalized.to_lowercase()
                ),
            )
            .score(entity.confidence);
            if let Some(timestamp) = segment.timestamp {
                event = event.at_timestamp(timestamp);
            }
            event
        }));
        events.extend(analysis.events.iter().map(|event_analysis| {
            let mut event = AnalysisEvent::new(
                self.name(),
                format!("text:event:{}", event_analysis.lemma.to_lowercase()),
            )
            .score(event_analysis.confidence);
            if let Some(timestamp) = segment.timestamp {
                event = event.at_timestamp(timestamp);
            }
            event
        }));
        events
    }

    fn document_events(&self) -> Result<Vec<AnalysisEvent>> {
        if self.segment_buffer.is_empty() {
            return Ok(Vec::new());
        }
        let joined = self
            .segment_buffer
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let analysis = analyze_text(&joined, &self.options)?;
        let mut events = Vec::new();
        for topic in &analysis.topics.descriptors {
            events.push(
                AnalysisEvent::new(self.name(), format!("text:topic:{}", topic.label))
                    .score(topic.score),
            );
        }
        for segment in &analysis.discourse {
            events.push(
                AnalysisEvent::new(
                    self.name(),
                    format!("text:discourse:{:?}", segment.kind).to_lowercase(),
                )
                .score(segment.confidence),
            );
        }
        Ok(events)
    }
}

impl TextAnalyzer for LinguisticAnalyzer {
    fn name(&self) -> &str {
        "linguistics"
    }

    fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>> {
        let analysis = analyze_segment(segment, &self.options)?;
        self.segment_buffer.push(
            OwnedTextSegment::new(segment.segment_index, segment.text)
                .finality(segment.is_final)
                .language(segment.language.unwrap_or("und")),
        );
        Ok(self.segment_events(segment, &analysis))
    }

    fn finish(&mut self, _last_segment_index: Option<u64>) -> Result<Vec<AnalysisEvent>> {
        self.document_events()
    }
}
