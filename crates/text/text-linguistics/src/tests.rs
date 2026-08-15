use super::*;

use text_core::{
    split_sentence_spans, tokenize, AnnotationProvenance, Sentence, TextProcessingOptions,
    TextSpan, Token, TokenKind,
};
use text_core::{OwnedTextSegment, Result, TextAnalyzer};
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
fn default_options_use_heuristic_entity_recognition_without_downloads() {
    let options = LinguisticAnalysisOptions::default();
    assert_eq!(
        options.entity_recognition.mode,
        EntityRecognitionMode::Heuristic
    );
    assert_eq!(options.entity_recognition.preset, ModelPreset::BertBaseNer);
    assert!(!options.entity_recognition.auto_download);
    assert!(!options.entity_recognition.download_progress);
}

#[test]
fn local_model_options_are_explicit_and_do_not_download_by_default() {
    let options = EntityRecognitionOptions::local_model();
    assert_eq!(options.mode, EntityRecognitionMode::LocalModel);
    assert_eq!(options.preset, ModelPreset::BertBaseNer);
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
    TextNlpPipeline::new(config)
        .analyze_text("Alice visited Berlin.")
        .unwrap();

    assert!(
        !bundle_dir.exists(),
        "default linguistic analysis must not materialize model-runtime state"
    );
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
        .load(&spec.name, &spec.revision)
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
    assert_eq!(primary.reason, "script fallback");
    assert_eq!(profile.token_count, 0);
    assert!(profile.alternatives.is_empty());
    assert!(profile.sentence_predictions.is_empty());
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

#[test]
fn lemmatizes_plural_and_inflected_tokens() {
    let tokens = tokenize(
        "Cars were running",
        &TextProcessingOptions {
            include_punctuation: false,
            ..TextProcessingOptions::default()
        },
    );
    let lemmas = lemmatize_tokens(&tokens, Some("en"), &LemmaOptions::default());
    assert_eq!(lemmas.lemmas[0].value, "car");
    assert_eq!(lemmas.lemmas[1].value, "be");
}

#[test]
fn morphology_annotation_tolerates_missing_lemma_and_pos_entries() {
    let tokens = tokenize(
        "They were testing robots",
        &TextProcessingOptions {
            include_punctuation: false,
            ..TextProcessingOptions::default()
        },
    );
    let lemmas = lemmatize_tokens(&tokens, Some("en"), &LemmaOptions::default());
    let mut partial_lemmas = lemmas.clone();
    partial_lemmas.lemmas.truncate(3);
    let pos = PosTagger::default().tag_tokens(&tokens, &lemmas);
    let annotations = annotate_morphology(&tokens, &partial_lemmas, &pos[..3]);

    assert_eq!(annotations.len(), 4);
    assert!(annotations[0]
        .tags
        .features
        .contains(&MorphFeature::Person3));
    assert!(annotations[0]
        .tags
        .features
        .contains(&MorphFeature::NumberPlur));
    assert!(annotations[1]
        .tags
        .features
        .contains(&MorphFeature::TensePast));
    assert_eq!(annotations[3].lemma, None);
    assert!(annotations[3].tags.features.is_empty());
}

#[test]
fn tags_pos_and_chunks_simple_sentence() {
    let text = "The new product launches today";
    let tokens = tokenize(
        text,
        &TextProcessingOptions {
            include_punctuation: true,
            ..TextProcessingOptions::default()
        },
    );
    let lemmas = lemmatize_tokens(&tokens, Some("en"), &LemmaOptions::default());
    let tagger = PosTagger::default();
    let pos = tagger.tag_tokens(&tokens, &lemmas);
    assert!(pos.iter().any(|annotation| annotation.tag == PosTag::Verb));
    let sentences = split_sentence_spans(text, &TextProcessingOptions::default());
    let chunks = chunk_phrases(text, &sentences, &tokens, &pos);
    assert!(chunks
        .iter()
        .any(|chunk| chunk.kind == ChunkKind::NounPhrase));
    assert!(chunks
        .iter()
        .any(|chunk| chunk.kind == ChunkKind::VerbPhrase));
}

#[test]
fn builds_dependency_tree_with_subject_and_object_like_relations() {
    let text = "Alice launched product";
    let tokens = tokenize(
        text,
        &TextProcessingOptions {
            include_punctuation: true,
            ..TextProcessingOptions::default()
        },
    );
    let lemmas = lemmatize_tokens(&tokens, Some("en"), &LemmaOptions::default());
    let mut pos = PosTagger::default().tag_tokens(&tokens, &lemmas);
    pos[0].tag = PosTag::Propn;
    pos[1].tag = PosTag::Verb;
    pos[2].tag = PosTag::Noun;
    let trees = DependencyParser.parse_document(
        &split_sentence_spans(text, &TextProcessingOptions::default()),
        &tokens,
        &pos,
    );
    assert_eq!(trees.len(), 1);
    assert!(trees[0]
        .edges
        .iter()
        .any(|edge| edge.relation == DependencyRelation::Nsubj));
}

#[test]
fn extracts_entities_coreference_and_events() {
    let text = "Alice visited Berlin. She presented the roadmap.";
    let analysis = analyze_text(text, &LinguisticAnalysisOptions::heuristic()).unwrap();
    assert!(analysis
        .entities
        .iter()
        .any(|entity| entity.entity_type == EntityType::Person));
    assert!(!analysis.coreference.is_empty());
    assert!(!analysis.events.is_empty());
}

#[test]
fn model_labeler_entities_replace_capitalization_guesses() {
    struct FakeNer;

    impl SequenceLabeler for FakeNer {
        fn label_text(&mut self, _text: &str) -> Result<Vec<RawPrediction>> {
            Ok(vec![
                token_prediction("B-ORG", 0, 6, 0.96),
                token_prediction("B-LOC", 14, 17, 0.94),
                token_prediction("I-LOC", 18, 22, 0.93),
            ])
        }

        fn runtime_backend(&self) -> TextRuntimeBackend {
            TextRuntimeBackend::External
        }
    }

    let text = "OpenAI opened New York on January 2024.";
    let mut labeler = FakeNer;
    let analysis = analyze_text_with_entity_labeler(
        text,
        &LinguisticAnalysisOptions::heuristic(),
        &mut labeler,
    )
    .unwrap();

    assert!(analysis.entities.iter().any(|entity| {
        entity.entity_type == EntityType::Organization && entity.mention.text == "OpenAI"
    }));
    assert!(analysis.entities.iter().any(|entity| {
        entity.entity_type == EntityType::Location && entity.mention.text == "New York"
    }));
    assert!(analysis
        .entities
        .iter()
        .any(|entity| entity.entity_type == EntityType::Date));
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
        analyze_subtitle_segments(&cues, &LinguisticAnalysisOptions::heuristic()).unwrap();

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
        analyze_transcription(&transcription, &LinguisticAnalysisOptions::heuristic()).unwrap();

    assert_eq!(analysis.cues.len(), 2);
    assert!(!analysis.aggregate.entities.is_empty());
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
        analyze_transcription(&transcription, &LinguisticAnalysisOptions::heuristic()).unwrap();

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
        analyze_transcription_contract(&contract, &LinguisticAnalysisOptions::heuristic()).unwrap();

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
fn rule_entities_live_in_linguistics() {
    let entities = crate::rule_entities(
        "Email hi@example.com and tag #Rust.",
        &crate::EntityRuleSet::default(),
    );
    assert!(entities.iter().any(|entity| entity.kind == "email"));
    assert!(entities.iter().any(|entity| entity.kind == "hashtag"));
}

#[test]
fn classifies_discourse_topics_and_style() {
    let text = "First, we introduce the API. However, the migration is tricky. Finally, the rollout is stable.";
    let analysis = analyze_text(text, &LinguisticAnalysisOptions::heuristic()).unwrap();
    assert!(!analysis.discourse.is_empty());
    assert!(!analysis.topics.descriptors.is_empty());
    assert!(analysis.style.complexity.average_sentence_tokens > 0.0);
}

#[test]
fn analyzer_emits_segment_and_document_events() {
    let mut analyzer = LinguisticAnalyzer::new(LinguisticAnalysisOptions::heuristic());
    let segment = OwnedTextSegment::new(0, "Alice presented the roadmap");
    let events = analyzer.process_segment(&segment.as_segment()).unwrap();
    assert!(events
        .iter()
        .any(|event| event.label.starts_with("text:language:")));
    let final_events = analyzer.finish(Some(0)).unwrap();
    assert!(final_events
        .iter()
        .any(|event| event.label.starts_with("text:topic:")));
}

#[test]
fn extracts_dates_and_amounts_without_pos_annotations() {
    let text = "Launch on January 2024 costs $99.";
    let sentences = vec![Sentence {
        text: text.to_string(),
        span: TextSpan {
            byte_start: 0,
            byte_end: text.len(),
            char_start: 0,
            char_end: text.chars().count(),
        },
        token_count: 6,
    }];
    let tokens = vec![
        Token {
            text: "Launch".to_string(),
            normalized: "launch".to_string(),
            span: TextSpan {
                byte_start: 0,
                byte_end: 6,
                char_start: 0,
                char_end: 6,
            },
            kind: TokenKind::Word,
        },
        Token {
            text: "on".to_string(),
            normalized: "on".to_string(),
            span: TextSpan {
                byte_start: 7,
                byte_end: 9,
                char_start: 7,
                char_end: 9,
            },
            kind: TokenKind::Word,
        },
        Token {
            text: "January".to_string(),
            normalized: "january".to_string(),
            span: TextSpan {
                byte_start: 10,
                byte_end: 17,
                char_start: 10,
                char_end: 17,
            },
            kind: TokenKind::Word,
        },
        Token {
            text: "2024".to_string(),
            normalized: "2024".to_string(),
            span: TextSpan {
                byte_start: 18,
                byte_end: 22,
                char_start: 18,
                char_end: 22,
            },
            kind: TokenKind::Number,
        },
        Token {
            text: "costs".to_string(),
            normalized: "costs".to_string(),
            span: TextSpan {
                byte_start: 23,
                byte_end: 28,
                char_start: 23,
                char_end: 28,
            },
            kind: TokenKind::Word,
        },
        Token {
            text: "$99".to_string(),
            normalized: "$99".to_string(),
            span: TextSpan {
                byte_start: 29,
                byte_end: 32,
                char_start: 29,
                char_end: 32,
            },
            kind: TokenKind::Other,
        },
    ];
    let entities = extract_named_entities(text, &sentences, &tokens, &[]);

    assert!(entities
        .iter()
        .any(|entity| entity.entity_type == EntityType::Date
            && entity.mention.text.to_lowercase().contains("january")));
    assert!(entities.iter().any(
        |entity| entity.entity_type == EntityType::Date && entity.mention.text.contains("2024")
    ));
    assert!(entities
        .iter()
        .any(|entity| entity.entity_type == EntityType::Amount && entity.mention.text == "$99"));
}

#[test]
fn text_nlp_pipeline_exposes_rich_graph_and_profile_metadata() {
    let pipeline = TextNlpPipeline::new(TextNlpConfig {
        options: LinguisticAnalysisOptions::heuristic(),
        ..TextNlpConfig::rich()
    });
    let analysis = pipeline
        .analyze_text("Alice presented the roadmap in Berlin.")
        .unwrap();

    assert_eq!(analysis.profile, AnalysisProfile::Rich);
    assert_eq!(analysis.graph.tokens.len(), analysis.tokens.len());
    assert_eq!(analysis.provenance, AnnotationProvenance::Heuristic);
    assert!(analysis.confidence.get() > 0.0);
    assert_eq!(analysis.token_ref(0).unwrap().text, "Alice");
}

#[test]
fn fast_profile_disables_heavier_annotations() {
    let pipeline = TextNlpPipeline::new(TextNlpConfig::fast());
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
