use super::*;

use text_core::Result;
use text_core::{split_sentence_spans, tokenize, AnnotationProvenance, TextProcessingOptions};
#[cfg(feature = "transcripts")]
use text_transcripts::{
    parse_srt, parse_webvtt, TranscriptSegment, TranscriptSegmentContract, TranscriptionContract,
};

#[test]
fn detects_english_text() {
    let detector = LanguageDetector::default();
    let profile = detector.detect_text("This is a simple English sentence with the roadmap.");
    assert_eq!(profile.primary.unwrap().language, "en");
}

#[test]
fn default_options_require_local_models_without_downloads() {
    let options = LinguisticAnalysisOptions::default();
    assert_eq!(
        options.entity_recognition.mode,
        EntityRecognitionMode::LocalModel
    );
    assert_eq!(options.morphology.mode, MorphologyMode::LocalModel);
    assert_eq!(
        options.dependency_parsing.mode,
        DependencyParsingMode::LocalModel
    );
    assert!(!options.morphology.auto_download);
    assert!(!options.dependency_parsing.auto_download);
    assert!(!options.enable_coreference);
    assert!(!options.enable_events);
    assert_eq!(
        options.entity_recognition.preset,
        ModelPreset::DistilbertMultilingualNerHrl
    );
    assert!(!options.entity_recognition.auto_download);
    assert!(!options.entity_recognition.download_progress);
}

#[test]
fn entity_presets_declare_their_supported_languages() {
    let multilingual = ModelPreset::DistilbertMultilingualNerHrl;
    assert_eq!(multilingual.supported_languages(), &["en", "de", "es"]);
    assert!(multilingual.supports_language("de-AT"));
    assert!(multilingual.supports_language("ES"));
    assert!(!multilingual.supports_language("fr"));
    assert!(ModelPreset::BertBaseNer.supports_language("en_US"));
    assert!(!ModelPreset::BertBaseNer.supports_language("de"));
    assert_eq!(
        multilingual.repo_id(),
        "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    );
    assert_eq!(MULTILINGUAL_NER_HRL_REVISION.len(), 40);
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
#[test]
fn multilingual_preset_spec_is_pinned_by_commit() {
    let spec = ModelPreset::DistilbertMultilingualNerHrl.spec();
    assert_eq!(spec.name, "distilbert-multilingual-ner-hrl");
    assert_eq!(spec.revision_value(), Some(MULTILINGUAL_NER_HRL_REVISION));
}

#[test]
fn token_classifier_session_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CandleTokenClassifier>();
}

#[test]
fn local_model_options_are_explicit_and_do_not_download_by_default() {
    let options = EntityRecognitionOptions::local_model();
    assert_eq!(options.mode, EntityRecognitionMode::LocalModel);
    assert_eq!(options.preset, ModelPreset::DistilbertMultilingualNerHrl);
    assert!(!options.auto_download);
    assert!(!options.download_progress);

    let options = EntityRecognitionOptions::local_model_with_downloads();
    assert_eq!(options.mode, EntityRecognitionMode::LocalModel);
    assert!(options.auto_download);
    assert!(options.download_progress);
}

#[test]
fn default_analysis_does_not_create_model_runtime_state() {
    let bundle_dir = std::env::temp_dir().join(format!(
        "text-linguistics-no-model-write-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(!bundle_dir.exists());

    let mut config = TextNlpConfig::default();
    config.options.entity_recognition.bundle_dir = bundle_dir.clone();
    config.options.morphology.bundle_dir = bundle_dir.clone();
    config.options.dependency_parsing.bundle_dir = bundle_dir.clone();
    // No bundle exists and downloads are off, so the default analysis fails rather
    // than falling back to heuristics.
    assert!(TextNlpPipeline::new(config)
        .analyze_text("Alice visited Berlin.")
        .is_err());

    assert!(
        !bundle_dir.exists(),
        "default linguistic analysis must not materialize model-runtime state"
    );
}

/// `config` with every model-backed layer disabled.
fn without_models(mut config: TextNlpConfig) -> TextNlpConfig {
    let models = LinguisticAnalysisOptions::without_models();
    config.options.entity_recognition = models.entity_recognition;
    config.options.morphology = models.morphology;
    config.options.dependency_parsing = models.dependency_parsing;
    config
}

#[test]
fn coreference_and_event_requests_fail_without_a_model_backend() {
    for (options, capability) in [
        (
            LinguisticAnalysisOptions {
                enable_coreference: true,
                ..LinguisticAnalysisOptions::without_models()
            },
            "coreference resolution",
        ),
        (
            LinguisticAnalysisOptions {
                enable_events: true,
                ..LinguisticAnalysisOptions::without_models()
            },
            "event and relation extraction",
        ),
    ] {
        let error = analyze_text("Alice visited Berlin. She presented it.", &options)
            .expect_err("no model backend exists");
        let message = error.to_string();
        assert!(message.contains("model required"), "{message}");
        assert!(message.contains(capability), "{message}");
        assert_model_required(error, capability);
    }
}

#[test]
fn disabled_model_layers_report_unsupported_and_produce_nothing() {
    let analysis = analyze_text(
        "Alice visited Berlin. She presented the roadmap.",
        &LinguisticAnalysisOptions::without_models(),
    )
    .unwrap();
    assert!(analysis.entities.is_empty());
    assert!(analysis.lemmas.is_empty());
    assert!(analysis.pos.is_empty());
    assert!(analysis.morphology.is_empty());
    assert!(analysis.dependencies.is_empty());
    assert!(analysis.coreference.is_empty());
    assert!(analysis.events.is_empty());
    assert!(analysis.relations.is_empty());
    for sentence in &analysis.sentence_analyses {
        for operation in [
            LinguisticOperation::NamedEntities,
            LinguisticOperation::Morphology,
            LinguisticOperation::PartOfSpeech,
            LinguisticOperation::DependencySyntax,
        ] {
            let report = sentence.operation(operation).unwrap();
            assert_eq!(
                report.support,
                OperationSupport::Unsupported,
                "{operation:?}"
            );
            assert_eq!(report.producer, None);
            assert!(report.reason.contains("not requested"), "{}", report.reason);
        }
    }
}

#[cfg(feature = "external-tests")]
#[test]
#[ignore = "downloads and runs the default Hugging Face NER model"]
fn downloads_default_ner_model_and_runs_local_entities() {
    use std::path::{Path, PathBuf};

    use model_runtime::ModelBundleStore;

    fn assert_nonempty_file(path: impl AsRef<Path>) {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)
            .unwrap_or_else(|err| panic!("expected `{}` metadata: {err}", path.display()));
        assert!(
            metadata.is_file() && metadata.len() > 0,
            "expected `{}` to be a non-empty file",
            path.display()
        );
    }

    let bundle_dir = std::env::var_os("TEXT_LINGUISTICS_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".model-runtime"));
    let mut options = LinguisticAnalysisOptions {
        entity_recognition: EntityRecognitionOptions::local_model_with_downloads(),
        ..LinguisticAnalysisOptions::default()
    };
    options.entity_recognition.bundle_dir = bundle_dir.clone();
    options.entity_recognition.download_progress = false;
    options.entity_recognition.max_retries = 1;

    let analysis =
        analyze_text("Alice works at OpenAI in Berlin.", &options).expect("model-backed analysis");
    assert!(analysis.entities.iter().any(|entity| {
        matches!(
            entity.entity_type,
            EntityType::Person | EntityType::Organization | EntityType::Location
        )
    }));

    let spec = options.entity_recognition.preset.spec();
    let bundle = ModelBundleStore::new(bundle_dir)
        .load(&spec.name, spec.revision_value().unwrap_or("main"))
        .expect("materialized default NER bundle");
    assert_nonempty_file(bundle.manifest_path());
    assert_nonempty_file(
        bundle
            .file_path("model.safetensors")
            .expect("default NER bundle should include safetensors weights"),
    );
}

#[test]
fn detects_mixed_script_text() {
    let detector = LanguageDetector::default();
    let profile = detector.detect_text("Hello 東京 and Berlin");
    assert!(profile.is_mixed || profile.dominant_script.is_some());
}

#[test]
fn falls_back_to_und_for_empty_text() {
    let detector = LanguageDetector::default();
    let profile = detector.detect_text("");
    let primary = profile
        .primary
        .expect("empty text still yields a fallback profile");

    assert_eq!(primary.language, "und");
    assert_eq!(primary.confidence, 0.0);
    assert_eq!(primary.reason, "no text");
    assert_eq!(profile.token_count, 0);
    assert!(profile.alternatives.is_empty());
    assert!(profile.sentences.is_empty());
}

#[test]
fn selects_default_mixed_tokenizer_policy() {
    let registry = TokenizerRegistry::default();
    let selection = registry.select(Some("en"), Some("linguistic-analysis"), None);
    assert_eq!(selection.mode, TokenizationMode::Mixed);
    assert!(selection.source.is_some());
}

#[test]
fn tokenizer_selection_prefers_task_override_and_word_mode_has_no_source() {
    let mut registry = TokenizerRegistry::default();
    let language_source = TokenizerSource::local("/tmp/language-tokenizer.json");
    let family_source = TokenizerSource::local("/tmp/family-tokenizer.json");
    let task_source = TokenizerSource::local("/tmp/task-tokenizer.json");

    registry
        .policy
        .language_overrides
        .insert("en".to_string(), language_source);
    registry
        .policy
        .model_family_overrides
        .insert("bert".to_string(), family_source);
    registry
        .policy
        .task_overrides
        .insert("classification".to_string(), task_source.clone());

    let selection = registry.select(Some("en"), Some("classification"), Some("bert"));
    assert_eq!(selection.source, Some(task_source));
    assert_eq!(selection.reason, "task override for `classification`");

    registry.policy.mode = TokenizationMode::Word;
    let word_only = registry.select(Some("en"), Some("classification"), Some("bert"));
    assert_eq!(word_only.mode, TokenizationMode::Word);
    assert_eq!(word_only.source, None);
}

#[test]
fn aligns_surface_tokens_to_fake_subwords() {
    let text = "don't panic";
    let tokens = tokenize(
        text,
        &TextProcessingOptions {
            include_punctuation: false,
            ..TextProcessingOptions::default()
        },
    );
    let alignment = align_tokenized_text(
        text,
        &tokens,
        TokenizerSelection {
            mode: TokenizationMode::Mixed,
            source: Some(TokenizerSource::default()),
            language: Some("en".to_string()),
            task: None,
            model_family: None,
            reason: "test".to_string(),
        },
        &TokenizedText {
            input_ids: vec![1, 2, 3, 4],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: Some(vec![0, 0, 0, 0]),
            offsets: vec![Some((0, 2)), Some((2, 5)), Some((6, 8)), Some((8, 11))],
        },
    )
    .unwrap();
    assert_eq!(alignment.aligned_tokens.len(), 2);
    assert_eq!(alignment.aligned_tokens[0].subword_indices, vec![0, 1]);
}

#[test]
fn ignores_invalid_subword_offsets_during_alignment() {
    let text = "hello world";
    let tokens = tokenize(
        text,
        &TextProcessingOptions {
            include_punctuation: false,
            ..TextProcessingOptions::default()
        },
    );
    let alignment = align_tokenized_text(
        text,
        &tokens,
        TokenizerSelection {
            mode: TokenizationMode::Mixed,
            source: Some(TokenizerSource::default()),
            language: Some("en".to_string()),
            task: None,
            model_family: None,
            reason: "test".to_string(),
        },
        &TokenizedText {
            input_ids: vec![10, 11, 12, 13],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: None,
            offsets: vec![Some((0, 5)), Some((5, 5)), Some((6, 11)), Some((50, 51))],
        },
    )
    .unwrap();

    assert_eq!(alignment.subwords[0].text.as_deref(), Some("hello"));
    assert_eq!(alignment.subwords[1].span, None);
    assert_eq!(alignment.subwords[2].text.as_deref(), Some("world"));
    assert_eq!(alignment.subwords[3].span, None);
    assert_eq!(alignment.aligned_tokens[0].subword_indices, vec![0]);
    assert_eq!(alignment.aligned_tokens[1].subword_indices, vec![2]);
}

#[cfg(feature = "tokenizers")]
#[test]
fn vocab_tokenizer_honors_cased_tokenizer_config() {
    let dir = tempfile::tempdir().unwrap();
    let vocab_path = dir.path().join("vocab.txt");
    std::fs::write(
        &vocab_path,
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nAlice\nalice\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"do_lower_case": false}"#,
    )
    .unwrap();

    let tokenized = TokenizerBundle::new(vocab_path).tokenize("Alice").unwrap();
    assert_eq!(tokenized.input_ids, vec![2, 5, 3]);
}

/// Caller-supplied POS annotations, as a model would produce them.
fn fixture_pos(tags: &[(usize, PosTag)]) -> Vec<PosAnnotation> {
    tags.iter()
        .map(|&(token_index, tag)| PosAnnotation {
            token_index,
            tag,
            confidence: 1.0,
            reason: "fixture".to_string(),
        })
        .collect()
}

#[test]
fn chunks_phrases_from_supplied_pos_tags() {
    let text = "The new product launches today";
    let tokens = tokenize(
        text,
        &TextProcessingOptions {
            include_punctuation: true,
            ..TextProcessingOptions::default()
        },
    );
    let pos = fixture_pos(&[
        (0, PosTag::Det),
        (1, PosTag::Adj),
        (2, PosTag::Noun),
        (3, PosTag::Verb),
        (4, PosTag::Noun),
    ]);
    let sentences = split_sentence_spans(text, &TextProcessingOptions::default());
    let chunks = chunk_phrases(text, &sentences, &tokens, &pos);
    assert!(chunks
        .iter()
        .any(|chunk| chunk.kind == ChunkKind::NounPhrase && chunk.text == "The new product"));
    assert!(chunks
        .iter()
        .any(|chunk| chunk.kind == ChunkKind::VerbPhrase));

    // Tags are looked up by token index, so a partial tagging (a sentence without a
    // model) never shifts tags onto the wrong tokens.
    let partial = fixture_pos(&[(3, PosTag::Verb)]);
    let chunks = chunk_phrases(text, &sentences, &tokens, &partial);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].kind, ChunkKind::VerbPhrase);
    assert_eq!(chunks[0].text, "launches");
}

#[test]
fn model_labeler_entities_are_returned_unmerged() {
    struct FakeNer;

    impl SequenceLabeler for FakeNer {
        fn label_text(&mut self, _text: &str) -> Result<Vec<RawPrediction>> {
            Ok(vec![
                token_prediction("B-ORG", 0, 6, 0.96),
                token_prediction("B-LOC", 14, 17, 0.94),
                token_prediction("I-LOC", 18, 22, 0.93),
            ])
        }
    }

    let text = "OpenAI opened New York on January 2024.";
    let mut labeler = FakeNer;
    let analysis = analyze_text_with_entity_labeler(
        text,
        &LinguisticAnalysisOptions::without_models(),
        &mut labeler,
    )
    .unwrap();

    // Exactly the model's entities: no rule-based date added for "January 2024".
    let entities = analysis
        .entities
        .iter()
        .map(|entity| (entity.entity_type, entity.mention.text.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(
        entities,
        [
            (EntityType::Organization, "OpenAI"),
            (EntityType::Location, "New York")
        ]
    );
    // Model-labeled entities make the analysis model-derived.
    assert_eq!(analysis.provenance, AnnotationProvenance::Model);
}

struct FixedNer(Vec<RawPrediction>);

impl SequenceLabeler for FixedNer {
    fn label_text(&mut self, _text: &str) -> Result<Vec<RawPrediction>> {
        Ok(self.0.clone())
    }
}

fn model_entities(text: &str, predictions: Vec<RawPrediction>) -> Vec<(EntityType, String)> {
    analyze_text_with_entity_labeler(
        text,
        &LinguisticAnalysisOptions::without_models(),
        &mut FixedNer(predictions),
    )
    .unwrap()
    .entities
    .into_iter()
    .map(|entity| (entity.entity_type, entity.mention.text))
    .collect()
}

#[test]
fn model_entities_take_the_first_sub_word_label_for_the_whole_word() {
    // "Talk with Alan Turing": the trailing "g" piece is labeled O.
    let entities = model_entities(
        "Talk with Alan Turing",
        vec![
            token_prediction("O", 0, 4, 0.99),
            token_prediction("O", 5, 9, 0.99),
            token_prediction("B-PER", 10, 14, 0.99),
            token_prediction("I-PER", 15, 20, 0.98),
            token_prediction("O", 20, 21, 0.6),
        ],
    );
    assert_eq!(
        entities,
        vec![(EntityType::Person, "Alan Turing".to_string())]
    );

    // "Helios conference": three pieces with conflicting labels.
    let entities = model_entities(
        "Helios conference",
        vec![
            token_prediction("B-MISC", 0, 2, 0.7),
            token_prediction("B-ORG", 2, 5, 0.6),
            token_prediction("I-MISC", 5, 6, 0.5),
            token_prediction("O", 7, 17, 0.99),
        ],
    );
    assert_eq!(entities, vec![(EntityType::Misc, "Helios".to_string())]);

    // "ADA LOVELACE": the trailing "LACE" piece is labeled O.
    let entities = model_entities(
        "ADA LOVELACE",
        vec![
            token_prediction("B-ORG", 0, 3, 0.8),
            token_prediction("I-ORG", 4, 8, 0.7),
            token_prediction("O", 8, 12, 0.6),
        ],
    );
    assert_eq!(
        entities,
        vec![(EntityType::Organization, "ADA LOVELACE".to_string())]
    );
}

#[test]
fn model_entities_end_at_outside_words() {
    let entities = model_entities(
        "Alice and Bob",
        vec![
            token_prediction("B-PER", 0, 5, 0.99),
            token_prediction("O", 6, 9, 0.99),
            token_prediction("I-PER", 10, 13, 0.9),
        ],
    );
    assert_eq!(
        entities,
        vec![
            (EntityType::Person, "Alice".to_string()),
            (EntityType::Person, "Bob".to_string()),
        ]
    );
}

#[test]
fn model_mode_does_not_fall_back_to_capitalization_guesses() {
    let entities = model_entities(
        "Welcome to the show.",
        vec![
            token_prediction("O", 0, 7, 0.99),
            token_prediction("O", 8, 10, 0.99),
            token_prediction("O", 11, 14, 0.99),
            token_prediction("O", 15, 19, 0.99),
        ],
    );
    assert!(entities.is_empty(), "{entities:?}");
}

#[test]
#[cfg(feature = "transcripts")]
fn analyzes_subtitle_segments_per_cue_and_in_aggregate() {
    let cues = vec![
        TranscriptSegment {
            index: 0,
            start_seconds: Some(0.0),
            end_seconds: Some(1.0),
            text: "Alice visited Berlin".to_string(),
            language: Some("en".to_string()),
            speaker: Some("narrator".to_string()),
            confidence: Some(0.9),
            is_final: true,
        },
        TranscriptSegment {
            index: 1,
            start_seconds: Some(1.0),
            end_seconds: Some(2.0),
            text: "Sie praesentierte die Roadmap".to_string(),
            language: Some("de".to_string()),
            speaker: None,
            confidence: Some(0.8),
            is_final: true,
        },
    ];

    let analysis =
        analyze_subtitle_segments(&cues, &LinguisticAnalysisOptions::without_models()).unwrap();

    assert_eq!(analysis.cues.len(), 2);
    assert_eq!(analysis.cues[0].cue, cues[0]);
    assert_eq!(analysis.cues[1].cue, cues[1]);
    assert!(!analysis.cues[0].analysis.tokens.is_empty());
    assert!(!analysis.cues[1].analysis.tokens.is_empty());
    assert!(!analysis.aggregate.tokens.is_empty());
}

#[test]
#[cfg(feature = "transcripts")]
fn analyzes_transcription_using_explicit_transcript_text_when_present() {
    let transcription = parse_srt(
        "1\n00:00:00,000 --> 00:00:01,000\nAlice visited Berlin\n\n2\n00:00:01,000 --> 00:00:02,000\nShe presented the roadmap\n",
    )
    .unwrap();

    let analysis =
        analyze_transcription(&transcription, &LinguisticAnalysisOptions::without_models())
            .unwrap();

    assert_eq!(analysis.cues.len(), 2);
    assert!(analysis
        .aggregate
        .tokens
        .iter()
        .any(|token| token.normalized == "roadmap"));
}

#[test]
#[cfg(feature = "transcripts")]
fn analyzes_transcription_falling_back_to_joined_cue_text() {
    let mut transcription = parse_webvtt(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello Berlin\n\n00:00:01.000 --> 00:00:02.000\nHola Madrid\n",
    )
    .unwrap();
    transcription.text = Some("   ".to_string());

    let analysis =
        analyze_transcription(&transcription, &LinguisticAnalysisOptions::without_models())
            .unwrap();

    assert_eq!(analysis.cues.len(), 2);
    assert!(analysis
        .aggregate
        .tokens
        .iter()
        .any(|token| token.normalized == "hello"));
    assert!(analysis
        .aggregate
        .tokens
        .iter()
        .any(|token| token.normalized == "hola"));
}

#[test]
#[cfg(feature = "transcripts")]
fn analyzes_transcription_contracts_directly() {
    let mut first = TranscriptSegmentContract::new(0, "Alice visited Berlin");
    first.start_seconds = Some(0.0);
    first.end_seconds = Some(1.0);
    first.speaker = Some("speaker_a".to_string());
    let mut second = TranscriptSegmentContract::new(1, "Roadmap review");
    second.start_seconds = Some(1.0);
    second.end_seconds = Some(2.0);
    second.speaker = Some("speaker_b".to_string());
    let mut contract = TranscriptionContract::from_segments(
        Some("fixture.srt".to_string()),
        Some("en".to_string()),
        vec![first, second],
    )
    .unwrap();
    contract.text = Some("Alice visited Berlin. Roadmap review.".to_string());

    let analysis =
        analyze_transcription_contract(&contract, &LinguisticAnalysisOptions::without_models())
            .unwrap();

    assert_eq!(analysis.cues.len(), 2);
    assert_eq!(analysis.cues[0].cue.speaker.as_deref(), Some("speaker_a"));
    assert_eq!(analysis.cues[0].cue.start_seconds, Some(0.0));
    assert!(analysis
        .aggregate
        .tokens
        .iter()
        .any(|token| token.normalized == "roadmap"));
}

#[test]
fn discourse_topic_and_style_requests_fail_without_a_model_backend() {
    for (options, capability) in [
        (
            LinguisticAnalysisOptions {
                enable_discourse: true,
                ..LinguisticAnalysisOptions::without_models()
            },
            "discourse segmentation",
        ),
        (
            LinguisticAnalysisOptions {
                enable_topics: true,
                ..LinguisticAnalysisOptions::without_models()
            },
            "topic modelling",
        ),
        (
            LinguisticAnalysisOptions {
                enable_style: true,
                ..LinguisticAnalysisOptions::without_models()
            },
            "style profiling",
        ),
    ] {
        let error = analyze_text("First, we introduce the API.", &options)
            .expect_err("no model backend exists");
        let message = error.to_string();
        assert!(message.contains("model required"), "{message}");
        assert!(message.contains(capability), "{message}");
        assert_model_required(error, capability);
    }
}

#[test]
fn text_nlp_pipeline_exposes_rich_graph_and_profile_metadata() {
    let pipeline = TextNlpPipeline::new(without_models(TextNlpConfig::rich()));
    let analysis = pipeline
        .analyze_text("Alice presented the roadmap in Berlin.")
        .unwrap();

    assert_eq!(analysis.profile, AnalysisProfile::Rich);
    assert_eq!(analysis.graph.tokens.len(), analysis.tokens.len());
    assert_eq!(analysis.provenance, AnnotationProvenance::RuleBased);
    assert!(analysis.confidence.get() > 0.0);
    assert_eq!(analysis.token_ref(0).unwrap().text, "Alice");
}

#[test]
fn fast_profile_disables_heavier_annotations() {
    let pipeline = TextNlpPipeline::new(without_models(TextNlpConfig::fast()));
    let analysis = pipeline
        .analyze_text("Alice presented the roadmap in Berlin.")
        .unwrap();

    assert_eq!(analysis.profile, AnalysisProfile::Fast);
    assert!(analysis.alignments.is_none());
    assert!(analysis.events.is_empty());
    assert!(analysis.discourse.is_empty());
    assert!(analysis.topics.descriptors.is_empty());
}

fn token_prediction(label: &str, byte_start: usize, byte_end: usize, score: f32) -> RawPrediction {
    let mut prediction = RawPrediction {
        kind: Some("token".to_string()),
        label: Some(label.to_string()),
        score: Some(score),
        ..RawPrediction::default()
    };
    prediction
        .attributes
        .insert("byte_start".to_string(), byte_start.to_string());
    prediction
        .attributes
        .insert("byte_end".to_string(), byte_end.to_string());
    prediction
}

#[test]
fn morph_tag_set_writes_ud_feats_order() {
    let feats = parse_ud_feats("NumType=Card|Number=Sing|Gender[psor]=Masc|Gender=Fem");
    assert_eq!(
        feats.to_ud_string(),
        "Gender=Fem|Gender[psor]=Masc|Number=Sing|NumType=Card"
    );
    assert_eq!(MorphTagSet::new().to_ud_string(), "_");
}

#[test]
fn ud_presets_route_by_primary_language_and_split_contractions() {
    assert_eq!(
        UdModelPreset::for_language("de-AT"),
        Some(UdModelPreset::ComboGermanGsd)
    );
    assert_eq!(
        UdModelPreset::for_language("es"),
        Some(UdModelPreset::ComboSpanishAncora)
    );
    assert_eq!(
        UdModelPreset::for_language("en-GB"),
        Some(UdModelPreset::ComboEnglishEwt)
    );
    assert_eq!(UdModelPreset::for_language("fr"), None);
    assert_eq!(split_contraction("de", "Zum"), Some(["zu", "dem"]));
    assert_eq!(split_contraction("es", "del"), Some(["de", "el"]));
    assert_eq!(split_contraction("es", "im"), None);
    assert_eq!(split_contraction("en", "Don't"), Some(["Do", "n't"]));
    assert_eq!(split_contraction("en", "can't"), Some(["ca", "n't"]));
    assert_eq!(split_contraction("en", "won’t"), Some(["wo", "n’t"]));
    assert_eq!(split_contraction("en", "John's"), Some(["John", "'s"]));
    assert_eq!(split_contraction("en", "I'm"), Some(["I", "'m"]));
    assert_eq!(split_contraction("en", "they're"), Some(["they", "'re"]));
    assert_eq!(split_contraction("en", "Cannot"), Some(["Can", "not"]));
    assert_eq!(split_contraction("en", "'s"), None);
    assert_eq!(split_contraction("en", "o'clock"), None);
    assert_eq!(split_contraction("en", "im"), None);
}

#[test]
fn dependency_decoding_breaks_cycles_and_keeps_one_root() {
    // Greedy heads form the cycle 1 <-> 2; the root arc into word 1 resolves it.
    let scores = vec![
        vec![5.0, 0.0, 9.0, 0.0],
        vec![1.0, 9.0, 0.0, 0.0],
        vec![0.0, 0.0, 8.0, 0.0],
    ];
    assert_eq!(decode_dependency_tree(&scores), vec![0, 1, 2]);

    // Two words prefer the root; the stronger root arc wins and the other attaches to it.
    let scores = vec![vec![9.0, 0.0, 1.0], vec![6.0, 1.0, 0.0]];
    assert_eq!(decode_dependency_tree(&scores), vec![0, 1]);
    assert!(decode_dependency_tree(&[]).is_empty());

    // Nested cycles: every word reaches the root exactly once.
    let scores = vec![
        vec![1.0, 0.0, 9.0, 2.0, 0.0],
        vec![0.0, 2.0, 0.0, 9.0, 1.0],
        vec![0.0, 9.0, 1.0, 0.0, 3.0],
        vec![4.0, 0.0, 2.0, 7.0, 0.0],
    ];
    let heads = decode_dependency_tree(&scores);
    assert_eq!(heads.iter().filter(|&&head| head == 0).count(), 1);
    for start in 1..=heads.len() {
        let mut node = start;
        for _ in 0..=heads.len() {
            if node != 0 {
                node = heads[node - 1];
            }
        }
        assert_eq!(node, 0, "{heads:?}");
    }
}

#[test]
fn ud_deprels_parse_into_relation_and_subtype() {
    assert_eq!(
        DependencyRelation::parse_ud_deprel("nsubj:pass"),
        (DependencyRelation::Nsubj, Some("pass".to_string()))
    );
    assert_eq!(
        DependencyRelation::parse_ud_deprel("punct"),
        (DependencyRelation::Punct, None)
    );
    assert_eq!(
        DependencyRelation::parse_ud_deprel("bogus"),
        (DependencyRelation::Dep, None)
    );
    assert_eq!(DependencyRelation::Obl.to_ud_deprel(Some("arg")), "obl:arg");
    for &relation in DependencyRelation::ALL {
        assert_eq!(
            DependencyRelation::from_ud(relation.as_ud()),
            Some(relation)
        );
    }
}

const MIXED_SAMPLE: &str =
    "Sarah says John is late. Er glaubt, dass John nicht kommen wird. María explicó que Ana llegó tarde.";

fn report(
    analysis: &LinguisticAnalysis,
    sentence: usize,
    operation: LinguisticOperation,
) -> &OperationReport {
    analysis.sentence_analyses[sentence]
        .operation(operation)
        .expect("every operation is reported")
}

#[test]
fn routes_each_sentence_by_its_own_language() {
    let analysis =
        analyze_text(MIXED_SAMPLE, &LinguisticAnalysisOptions::without_models()).unwrap();
    let languages = analysis
        .sentence_analyses
        .iter()
        .map(|sentence| sentence.language.as_deref())
        .collect::<Vec<_>>();
    assert_eq!(languages, [Some("en"), Some("de"), Some("es")]);
    assert!(analysis.language.is_mixed);

    // Spans are canonical UTF-8 byte offsets and recover the exact source text.
    for (sentence, record) in analysis.sentences.iter().zip(&analysis.sentence_analyses) {
        assert_eq!(record.span, sentence.span);
        assert_eq!(
            &MIXED_SAMPLE[record.span.byte_start..record.span.byte_end],
            sentence.text
        );
        assert_eq!(record.operations.len(), LinguisticOperation::ALL.len());
    }
    let spanish = &analysis.sentence_analyses[2];
    assert_eq!(
        &MIXED_SAMPLE[spanish.span.byte_start..spanish.span.byte_end],
        "María explicó que Ana llegó tarde."
    );
    assert_eq!(spanish.span.byte_end, MIXED_SAMPLE.len());

    // No sentence gets model-free stand-ins: every model-backed layer is unsupported
    // and empty when its model is not requested, whatever the language.
    for sentence in 0..3 {
        for operation in [
            LinguisticOperation::NamedEntities,
            LinguisticOperation::Morphology,
            LinguisticOperation::PartOfSpeech,
            LinguisticOperation::DependencySyntax,
        ] {
            let report = report(&analysis, sentence, operation);
            assert_eq!(
                report.support,
                OperationSupport::Unsupported,
                "{operation:?}"
            );
            assert!(report.reason.contains("not requested"), "{}", report.reason);
        }
        for operation in [
            LinguisticOperation::LanguageDetection,
            LinguisticOperation::SentenceSegmentation,
            LinguisticOperation::Tokenization,
        ] {
            assert_eq!(
                report(&analysis, sentence, operation).support,
                OperationSupport::Supported
            );
        }
    }
    assert!(analysis.pos.is_empty());
    assert!(analysis.lemmas.is_empty());
    assert!(analysis.dependencies.is_empty());
    assert!(analysis.entities.is_empty());
}

#[test]
fn undecided_and_unsupported_sentences_keep_their_neighbours() {
    // Models are requested, but no sentence routes to a UD model, so none is loaded;
    // a caller labeler stands in for the NER model.
    let text = "Ok. Le chat dort sur le canapé depuis ce matin. Привет, как дела у тебя сегодня?";
    let analysis = analyze_text_with_entity_labeler(
        text,
        &LinguisticAnalysisOptions::default(),
        &mut FixedNer(Vec::new()),
    )
    .unwrap();
    assert_eq!(analysis.sentences.len(), 3);
    let [short, french, russian] = &analysis.sentence_analyses[..] else {
        panic!("three sentence records");
    };
    assert!(analysis.pos.is_empty());

    assert_eq!(short.detection.decision, LanguageDecision::TooShort);
    assert_eq!(short.language, None);
    assert_eq!(
        report(&analysis, 0, LinguisticOperation::LanguageDetection).support,
        OperationSupport::Unknown
    );
    assert_eq!(
        report(&analysis, 0, LinguisticOperation::PartOfSpeech).support,
        OperationSupport::Unknown
    );
    assert_eq!(
        report(&analysis, 0, LinguisticOperation::Tokenization).support,
        OperationSupport::Supported
    );

    assert_eq!(french.language.as_deref(), Some("fr"));
    let french_pos = report(&analysis, 1, LinguisticOperation::PartOfSpeech);
    assert_eq!(french_pos.support, OperationSupport::Unsupported);
    assert_eq!(french_pos.reason, "no model for `fr`");

    assert_eq!(
        russian.detection.decision,
        LanguageDecision::UnsupportedScript
    );
    assert_eq!(russian.language, None);
    assert_eq!(russian.detection.candidates[0].language, "ru");
    assert_eq!(russian.detection.candidates[0].confidence, 0.0);
}

#[test]
fn an_english_name_does_not_make_a_german_sentence_english() {
    let detector = LanguageDetector::default();
    for text in [
        "Er glaubt, dass John nicht kommen wird.",
        "John Smith wohnt seit drei Jahren in München.",
    ] {
        let profile = detector.detect_text(text);
        let sentence = &profile.sentences[0];
        assert_eq!(sentence.decision, LanguageDecision::Detected, "{text}");
        assert_eq!(sentence.language().unwrap().language, "de", "{text}");
    }
}

#[test]
fn ambiguous_sentences_report_their_candidates() {
    let detector = LanguageDetector {
        options: LanguageDetectionOptions {
            min_margin: 0.99,
            ..LanguageDetectionOptions::default()
        },
    };
    let profile = detector.detect_text("Hotel Berlin Taxi.");
    let sentence = &profile.sentences[0];
    assert_eq!(sentence.decision, LanguageDecision::Ambiguous);
    assert!(sentence.language().is_none());
    assert!(!sentence.candidates.is_empty());
    assert!(sentence.candidates.len() <= 4);
}

#[test]
fn a_pinned_language_routes_every_sentence() {
    let mut options = LinguisticAnalysisOptions::without_models();
    options.processing.language = Some("en-US".to_string());
    let analysis = analyze_text(MIXED_SAMPLE, &options).unwrap();
    for sentence in &analysis.sentence_analyses {
        assert_eq!(sentence.language.as_deref(), Some("en"));
        assert!(sentence.language_pinned);
        assert_eq!(
            sentence
                .operation(LinguisticOperation::LanguageDetection)
                .unwrap()
                .producer
                .as_deref(),
            Some("caller")
        );
    }
}

#[test]
fn routing_is_deterministic() {
    let options = LinguisticAnalysisOptions::without_models();
    let first = analyze_text(MIXED_SAMPLE, &options).unwrap();
    let second = analyze_text(MIXED_SAMPLE, &options).unwrap();
    assert_eq!(first.sentence_analyses, second.sentence_analyses);
}

/// No gold-fixture sentence is routed to the wrong language, and sentences labelled
/// undecidable stay undecided. A labelled sentence may come back undecided; those
/// are counted and printed.
#[test]
fn never_misroutes_a_gold_fixture_sentence() {
    let detector = LanguageDetector::default();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../evaluation/corpora");
    let mut checked = 0;
    let mut undecided = Vec::new();
    for corpus in [
        "named-entities-multilingual-v1.jsonl",
        "morphology-multilingual-v1.jsonl",
        "dependencies-multilingual-v1.jsonl",
        "sentence-languages-multilingual-v1.jsonl",
    ] {
        for line in std::fs::read_to_string(root.join(corpus)).unwrap().lines() {
            let case: serde_json::Value = serde_json::from_str(line).unwrap();
            let text = case["text"].as_str().unwrap();
            let profile = detector.detect_text(text);
            let expected = case["sentences"]
                .as_array()
                .map(|sentences| {
                    sentences
                        .iter()
                        .map(|sentence| sentence["language"].as_str().map(ToString::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![case["language"].as_str().map(ToString::to_string)]);
            let detected = profile
                .sentences
                .iter()
                .map(|sentence| {
                    sentence
                        .language()
                        .map(|prediction| prediction.language.clone())
                })
                .collect::<Vec<_>>();
            assert_eq!(detected.len(), expected.len(), "{corpus} {}", case["id"]);
            for (index, (detected, expected)) in detected.iter().zip(&expected).enumerate() {
                match (detected, expected) {
                    (None, Some(_)) => undecided.push(format!("{} #{index}", case["id"])),
                    _ => assert_eq!(detected, expected, "{corpus} {} #{index}", case["id"]),
                }
            }
            if let Some(sentences) = case["sentences"].as_array() {
                let spans = profile
                    .sentences
                    .iter()
                    .map(|sentence| (sentence.span.byte_start, sentence.span.byte_end))
                    .collect::<Vec<_>>();
                let gold = sentences
                    .iter()
                    .map(|sentence| {
                        let start = sentence["byteStart"].as_u64().unwrap() as usize;
                        let end = sentence["byteEnd"].as_u64().unwrap() as usize;
                        assert_eq!(&text[start..end], sentence["text"].as_str().unwrap());
                        (start, end)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(spans, gold, "{} spans", case["id"]);
            }
            checked += expected.len();
        }
    }
    eprintln!(
        "gold sentences: {checked}, undecided though labelled: {} {undecided:?}",
        undecided.len()
    );
    assert!(checked >= 100, "{checked} sentences");
}

#[cfg(feature = "candle")]
#[test]
fn ud_lemmatizer_applies_table_rules_and_exceptions() {
    let de = UdLemmatizer::for_language("de").unwrap();
    assert_eq!(de.lemmatize("Kinder", PosTag::Noun), "Kind");
    assert_eq!(de.lemmatize("spielten", PosTag::Verb), "spielen");
    assert_eq!(de.lemmatize("war", PosTag::Aux), "sein");
    assert_eq!(de.lemmatize("München", PosTag::Propn), "München");
    let es = UdLemmatizer::for_language("es").unwrap();
    assert_eq!(es.lemmatize("niños", PosTag::Noun), "niño");
    assert_eq!(es.lemmatize("jugaban", PosTag::Verb), "jugar");
    assert_eq!(es.lemmatize("las", PosTag::Det), "el");
    let en = UdLemmatizer::for_language("en").unwrap();
    assert_eq!(en.lemmatize("children", PosTag::Noun), "child");
    assert_eq!(en.lemmatize("was", PosTag::Aux), "be");
    assert_eq!(en.lemmatize("ca", PosTag::Aux), "can");
    assert_eq!(en.lemmatize("n't", PosTag::Part), "not");
    assert_eq!(en.lemmatize("I", PosTag::Pron), "I");
    assert!(UdLemmatizer::for_language("fr").is_none());

    let table = UdLemmatizer::parse("C\tNOUN\nR\tNOUN\ter\t2\t\nE\tmänner\tNOUN\tMann\n");
    assert_eq!(table.lemmatize("Lehrer", PosTag::Noun), "Lehr");
    assert_eq!(table.lemmatize("Männer", PosTag::Noun), "Mann");
}

/// Missing models surface as the typed `ModelRequired` variant, not a formatted string.
fn assert_model_required(error: LinguisticAnalysisError, capability: &str) {
    match error {
        LinguisticAnalysisError::ModelRequired {
            capability: required,
            hint,
        } => {
            assert_eq!(required, capability);
            assert!(hint.contains("set `enable_"), "{hint}");
        }
        other => panic!("expected a typed ModelRequired error, got {other:?}"),
    }
}
