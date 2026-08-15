use std::path::PathBuf;

use text_classification::{
    analyze_sentiment, classify_text, zero_shot_classify, FallbackPolicy, ModelSelection,
    SentimentRequest, TextClassificationRequest, ZeroShotClassificationRequest,
};
use text_core::Result;
use text_core::{
    build_annotation_graph_from_parts, detailed_text_stats, detect_script_profile,
    split_paragraphs, split_sentence_spans, tokenize, TextDocument, TextProcessingOptions,
};
use text_embeddings::{
    HashedTextEmbedder, TextEmbeddingBackend, TextEmbeddingConfig, TextEmbeddingMetadata,
};
use text_lexical::{
    character_ngram_frequencies, character_shingles, diverse_extractive_summary,
    english_stop_words, extractive_summary, keywords, phrase_keywords, readability_summary,
    rule_entities, sentiment, summarize_text, term_frequencies, token_ngram_frequencies,
    token_shingles, EntityRuleSet, ExtractiveSummaryOptions, KeywordOptions, PhraseKeywordOptions,
    SentimentLexicon,
};
use text_linguistics::{
    EntityRecognitionMode, EntityRecognitionOptions, TextNlpConfig, TextNlpPipeline,
};

use crate::fingerprint::{character_shingle_simhash, token_shingle_simhash};
use crate::stats::enriched_text_stats_from_tokens;
use crate::{
    invalid_argument, ClassificationAnalysisSection, ClassificationDepth, CoreAnalysisSection,
    DocumentAnalysisOptions, DocumentAnalysisReport, EmbeddingAnalysisSection, EmbeddingDepth,
    LexicalAnalysisSection, LinguisticAnalysisSection, LinguisticDepth, NgramFrequencyReport,
    ShingleCountReport, SimilarityAnalysisSection, SparseEmbeddingReport, TextAnalysisDiagnostic,
};

pub fn analyze_text(
    id: impl Into<String>,
    text: &str,
    options: &DocumentAnalysisOptions,
) -> Result<DocumentAnalysisReport> {
    let id = id.into();
    let document = TextDocument::new(&id, text);
    analyze_document(&document, options)
}

pub fn analyze_document(
    document: &TextDocument<'_>,
    options: &DocumentAnalysisOptions,
) -> Result<DocumentAnalysisReport> {
    validate_options(options)?;
    let mut diagnostics = Vec::new();
    let processing = effective_processing(document, options);
    let tokens = tokenize(document.text, &processing);
    let sentences = split_sentence_spans(document.text, &processing);
    let paragraphs = split_paragraphs(document.text);
    let graph = options.include_annotation_graph.then(|| {
        build_annotation_graph_from_parts(document.text, &tokens, &sentences, &paragraphs)
    });
    let core = CoreAnalysisSection {
        stats: detailed_text_stats(document.text, &processing),
        script_profile: detect_script_profile(document.text),
        tokens: tokens.clone(),
        sentences,
        paragraphs,
        annotation_graph: graph,
    };
    let enriched_stats = enriched_text_stats_from_tokens(document.text, &processing, &tokens);
    let lexical = lexical_section(document.text, options)?;
    let similarity = similarity_section(document.text, options, &processing)?;
    let linguistic = linguistic_section(document.text, options, &processing, &mut diagnostics);
    let embedding = embedding_section(document.text, options, None, &mut diagnostics);
    let classification = classification_section(document.text, options, &mut diagnostics);
    let language = document
        .language
        .map(ToString::to_string)
        .or_else(|| options.language_hint.clone())
        .or_else(|| {
            linguistic
                .as_ref()
                .and_then(|section| section.language.get("primary"))
                .and_then(|primary| primary.get("language"))
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string)
        });

    Ok(DocumentAnalysisReport {
        id: document.id.to_string(),
        language,
        core,
        enriched_stats,
        lexical,
        similarity,
        linguistic,
        embedding,
        classification,
        diagnostics,
    })
}

pub(crate) fn classification_section(
    text: &str,
    options: &DocumentAnalysisOptions,
    diagnostics: &mut Vec<TextAnalysisDiagnostic>,
) -> Option<ClassificationAnalysisSection> {
    match options.classification_depth {
        ClassificationDepth::Off => None,
        ClassificationDepth::Imported => {
            diagnostics.push(TextAnalysisDiagnostic::warning(
                "classification_unavailable",
                "imported classification depth requires imported predictions supplied to text-classification directly",
            ));
            None
        }
        ClassificationDepth::Backend => {
            let model = ModelSelection::default();
            let local_model = options.classification_local_model.clone();
            let sentiment = match analyze_sentiment(SentimentRequest {
                text: text.to_string(),
                model: model.clone(),
                imported_predictions: Vec::new(),
                local_model: local_model.clone(),
            }) {
                Ok(response) => response,
                Err(error) => {
                    diagnostics.push(TextAnalysisDiagnostic::warning(
                        "classification_unavailable",
                        error.to_string(),
                    ));
                    return None;
                }
            };
            let classification = match classify_text(TextClassificationRequest {
                text: text.to_string(),
                labels: options.classification_labels.clone(),
                top_k: options.classification_labels.len().max(1),
                multi_label: false,
                model: model.clone(),
                imported_predictions: Vec::new(),
                local_model: local_model.clone(),
            }) {
                Ok(response) => response,
                Err(error) => {
                    diagnostics.push(TextAnalysisDiagnostic::warning(
                        "classification_unavailable",
                        error.to_string(),
                    ));
                    return None;
                }
            };
            let zero_shot = match zero_shot_classify(ZeroShotClassificationRequest {
                text: text.to_string(),
                labels: options.zero_shot_labels.clone(),
                hypothesis_template: "This text is about {}.".to_string(),
                model,
                imported_predictions: Vec::new(),
                local_model,
            }) {
                Ok(response) => response,
                Err(error) => {
                    diagnostics.push(TextAnalysisDiagnostic::warning(
                        "classification_unavailable",
                        error.to_string(),
                    ));
                    return None;
                }
            };
            Some(ClassificationAnalysisSection {
                sentiment,
                classification,
                zero_shot,
            })
        }
        ClassificationDepth::LexicalFallback => {
            let model = ModelSelection {
                fallback_policy: FallbackPolicy::LexicalFallback,
                ..ModelSelection::default()
            };
            let sentiment = match analyze_sentiment(SentimentRequest {
                text: text.to_string(),
                model: model.clone(),
                imported_predictions: Vec::new(),
                local_model: None,
            }) {
                Ok(response) => response,
                Err(error) => {
                    diagnostics.push(TextAnalysisDiagnostic::warning(
                        "classification_unavailable",
                        error.to_string(),
                    ));
                    return None;
                }
            };
            let classification = match classify_text(TextClassificationRequest {
                text: text.to_string(),
                labels: options.classification_labels.clone(),
                top_k: options.classification_labels.len().max(1),
                multi_label: false,
                model: model.clone(),
                imported_predictions: Vec::new(),
                local_model: None,
            }) {
                Ok(response) => response,
                Err(error) => {
                    diagnostics.push(TextAnalysisDiagnostic::warning(
                        "classification_unavailable",
                        error.to_string(),
                    ));
                    return None;
                }
            };
            let zero_shot = match zero_shot_classify(ZeroShotClassificationRequest {
                text: text.to_string(),
                labels: options.zero_shot_labels.clone(),
                hypothesis_template: "This text is about {}.".to_string(),
                model,
                imported_predictions: Vec::new(),
                local_model: None,
            }) {
                Ok(response) => response,
                Err(error) => {
                    diagnostics.push(TextAnalysisDiagnostic::warning(
                        "classification_unavailable",
                        error.to_string(),
                    ));
                    return None;
                }
            };
            Some(ClassificationAnalysisSection {
                sentiment,
                classification,
                zero_shot,
            })
        }
    }
}

pub(crate) fn embedding_section(
    text: &str,
    options: &DocumentAnalysisOptions,
    corpus: Option<&text_lexical::TfIdfCorpus>,
    diagnostics: &mut Vec<TextAnalysisDiagnostic>,
) -> Option<EmbeddingAnalysisSection> {
    match &options.embedding_depth {
        EmbeddingDepth::Off => None,
        EmbeddingDepth::Hashed {
            dimensions,
            use_idf,
        } => {
            let embedder = match HashedTextEmbedder::new(
                TextEmbeddingConfig {
                    dimensions: (*dimensions).max(1),
                    use_idf: *use_idf,
                },
                text_lexical::CorpusOptions::default(),
            ) {
                Ok(embedder) => embedder,
                Err(error) => {
                    diagnostics.push(TextAnalysisDiagnostic::warning(
                        "embedding_unavailable",
                        error.to_string(),
                    ));
                    return None;
                }
            };
            embedding_from_hashed(text, &embedder, corpus, options, diagnostics)
        }
        EmbeddingDepth::CandleBundle {
            bundle_dir,
            pooling,
        } => candle_embedding_section(text, bundle_dir, *pooling, corpus, options, diagnostics),
        EmbeddingDepth::OnnxBundle {
            bundle_dir,
            pooling,
        } => onnx_embedding_section(text, bundle_dir, *pooling, corpus, options, diagnostics),
    }
}

fn lexical_section(
    text: &str,
    options: &DocumentAnalysisOptions,
) -> Result<LexicalAnalysisSection> {
    let stop_words = options
        .language_hint
        .as_deref()
        .map(text_lexical::stop_words_for_language)
        .unwrap_or_else(english_stop_words);
    let keyword_options = KeywordOptions {
        max_terms: options.keyword_limit,
        stop_words: stop_words.clone(),
        ..KeywordOptions::default()
    };
    let phrase_options = PhraseKeywordOptions {
        max_phrases: options.keyword_limit,
        stop_words: stop_words.clone(),
        ..PhraseKeywordOptions::default()
    };
    let summary_options = ExtractiveSummaryOptions {
        max_sentences: options.summary_sentences.max(1),
        stop_words,
        ..ExtractiveSummaryOptions::default()
    };
    Ok(LexicalAnalysisSection {
        summary: summarize_text(text, options.keyword_limit),
        top_terms: term_frequencies(text)
            .into_iter()
            .take(options.keyword_limit)
            .collect(),
        keywords: keywords(text, &keyword_options),
        phrase_keywords: phrase_keywords(text, &phrase_options),
        readability: readability_summary(text, &options.processing),
        sentiment: sentiment(text, &SentimentLexicon::default()),
        extractive_summary: extractive_summary(text, &summary_options)?,
        diverse_extractive_summary: diverse_extractive_summary(text, &summary_options, 0.35)?,
        rule_entities: rule_entities(text, &EntityRuleSet::default()),
    })
}

fn similarity_section(
    text: &str,
    options: &DocumentAnalysisOptions,
    processing: &TextProcessingOptions,
) -> Result<SimilarityAnalysisSection> {
    let mut character_ngram_reports = Vec::new();
    let mut token_ngram_reports = Vec::new();
    for n in &options.ngram_sizes {
        character_ngram_reports.push(NgramFrequencyReport {
            n: *n,
            terms: character_ngram_frequencies(text, *n)?,
        });
        token_ngram_reports.push(NgramFrequencyReport {
            n: *n,
            terms: token_ngram_frequencies(text, *n, processing)?,
        });
    }
    let mut character_shingle_counts = Vec::new();
    let mut token_shingle_counts = Vec::new();
    for n in &options.shingle_sizes {
        character_shingle_counts.push(ShingleCountReport {
            n: *n,
            count: character_shingles(text, *n)?.len(),
        });
        token_shingle_counts.push(ShingleCountReport {
            n: *n,
            count: token_shingles(text, *n, processing)?.len(),
        });
    }
    let fingerprint_n = options.shingle_sizes.first().copied().unwrap_or(3).max(1);
    Ok(SimilarityAnalysisSection {
        character_ngram_frequencies: character_ngram_reports,
        token_ngram_frequencies: token_ngram_reports,
        character_shingle_counts,
        token_shingle_counts,
        token_shingle_simhash: format!(
            "{:016x}",
            token_shingle_simhash(text, fingerprint_n, processing)?
        ),
        character_shingle_simhash: format!(
            "{:016x}",
            character_shingle_simhash(text, fingerprint_n)?
        ),
    })
}

fn linguistic_section(
    text: &str,
    options: &DocumentAnalysisOptions,
    processing: &TextProcessingOptions,
    diagnostics: &mut Vec<TextAnalysisDiagnostic>,
) -> Option<LinguisticAnalysisSection> {
    let depth = effective_linguistic_depth(options);
    let mut config = match depth {
        LinguisticDepth::Off => return None,
        LinguisticDepth::HeuristicFast => TextNlpConfig::fast(),
        LinguisticDepth::HeuristicBalanced => TextNlpConfig::balanced(),
        LinguisticDepth::HeuristicRich => TextNlpConfig::rich(),
        LinguisticDepth::LocalModel { .. } => TextNlpConfig::balanced(),
    };
    config.options.processing = processing.clone();
    if options.language_hint.is_some() {
        config.options.language_detection.sentence_level = false;
    }
    match depth {
        LinguisticDepth::HeuristicFast
        | LinguisticDepth::HeuristicBalanced
        | LinguisticDepth::HeuristicRich => {
            config.options.entity_recognition = EntityRecognitionOptions::heuristic();
        }
        LinguisticDepth::LocalModel {
            bundle_dir,
            auto_download,
            download_progress,
        } => {
            config.options.entity_recognition = EntityRecognitionOptions {
                mode: EntityRecognitionMode::LocalModel,
                bundle_dir,
                auto_download,
                download_progress,
                ..EntityRecognitionOptions::local_model()
            };
        }
        LinguisticDepth::Off => return None,
    }

    match TextNlpPipeline::new(config).analyze_text(text) {
        Ok(analysis) => Some(project_linguistic_analysis(analysis)),
        Err(error) => {
            diagnostics.push(TextAnalysisDiagnostic::warning(
                "linguistics_unavailable",
                error.to_string(),
            ));
            None
        }
    }
}

fn project_linguistic_analysis(
    analysis: text_linguistics::LinguisticAnalysis,
) -> LinguisticAnalysisSection {
    LinguisticAnalysisSection {
        language: serde_json::json!({
            "primary": analysis.language.primary.as_ref().map(|prediction| serde_json::json!({
                "language": prediction.language,
                "confidence": prediction.confidence,
                "script": prediction.script,
                "reason": prediction.reason
            })),
            "dominantScript": analysis.language.dominant_script,
            "isMixed": analysis.language.is_mixed,
            "tokenCount": analysis.language.token_count
        }),
        tokenizer: serde_json::json!({
            "mode": format!("{:?}", analysis.tokenizer.mode),
            "source": analysis.tokenizer.source.map(|source| format!("{source:?}")),
            "reason": analysis.tokenizer.reason
        }),
        lemmas: serde_json::json!({
            "count": analysis.lemmas.len(),
            "items": analysis.lemmas.iter().map(|lemma| serde_json::json!({
                "tokenIndex": lemma.token_index,
                "value": lemma.value,
                "language": lemma.language,
                "confidence": lemma.confidence
            })).collect::<Vec<_>>()
        }),
        morphology: debug_collection(analysis.morphology),
        pos: debug_collection(analysis.pos),
        chunks: debug_collection(analysis.chunks),
        dependencies: debug_collection(analysis.dependencies),
        entities: debug_collection(analysis.entities),
        canonical_entities: debug_collection(analysis.canonical_entities),
        coreference: debug_collection(analysis.coreference),
        events: debug_collection(analysis.events),
        relations: debug_collection(analysis.relations),
        discourse: debug_collection(analysis.discourse),
        outline: serde_json::json!({
            "debug": format!("{:?}", analysis.outline)
        }),
        topics: debug_value(analysis.topics),
        style: debug_value(analysis.style),
    }
}

fn debug_collection<T: std::fmt::Debug>(items: Vec<T>) -> serde_json::Value {
    serde_json::json!({
        "count": items.len(),
        "items": items.into_iter().map(|item| format!("{item:?}")).collect::<Vec<_>>()
    })
}

fn debug_value<T: std::fmt::Debug>(value: T) -> serde_json::Value {
    serde_json::json!({
        "debug": format!("{value:?}")
    })
}

#[allow(dead_code)]
fn embedding_from_backend<E: TextEmbeddingBackend>(
    text: &str,
    embedder: &E,
    _corpus: Option<&text_lexical::TfIdfCorpus>,
    options: &DocumentAnalysisOptions,
    diagnostics: &mut Vec<TextAnalysisDiagnostic>,
) -> Option<EmbeddingAnalysisSection> {
    let vector = match embedder.embed_text(text) {
        Ok(vector) => vector,
        Err(error) => {
            diagnostics.push(TextAnalysisDiagnostic::warning(
                "embedding_unavailable",
                error.to_string(),
            ));
            return None;
        }
    };
    let sparse = if options.include_sparse_embedding {
        match &options.embedding_depth {
            EmbeddingDepth::Hashed {
                dimensions,
                use_idf,
            } => {
                let embedder = HashedTextEmbedder::new(
                    TextEmbeddingConfig {
                        dimensions: (*dimensions).max(1),
                        use_idf: *use_idf,
                    },
                    text_lexical::CorpusOptions::default(),
                )
                .ok()?;
                match embedder.embed_text_sparse(text, _corpus) {
                    Ok(sparse) => Some(SparseEmbeddingReport {
                        dimensions: sparse.dimensions(),
                        indices: sparse.indices().to_vec(),
                        values: sparse.values().to_vec(),
                    }),
                    Err(error) => {
                        diagnostics.push(TextAnalysisDiagnostic::warning(
                            "sparse_embedding_unavailable",
                            error.to_string(),
                        ));
                        None
                    }
                }
            }
            _ => None,
        }
    } else {
        None
    };
    let model = embedder.model_info();
    let provenance = provenance_label(&embedder.metadata());
    Some(EmbeddingAnalysisSection {
        dimensions: vector.dimensions(),
        preview: vector.as_slice().iter().take(16).copied().collect(),
        vector: vector.as_slice().to_vec(),
        sparse,
        model,
        provenance,
    })
}

fn embedding_from_hashed(
    text: &str,
    embedder: &HashedTextEmbedder,
    corpus: Option<&text_lexical::TfIdfCorpus>,
    options: &DocumentAnalysisOptions,
    diagnostics: &mut Vec<TextAnalysisDiagnostic>,
) -> Option<EmbeddingAnalysisSection> {
    let vector = match embedder.embed_text_with_corpus(text, corpus) {
        Ok(vector) => vector,
        Err(error) => {
            diagnostics.push(TextAnalysisDiagnostic::warning(
                "embedding_unavailable",
                error.to_string(),
            ));
            return None;
        }
    };
    let sparse = if options.include_sparse_embedding {
        match embedder.embed_text_sparse(text, corpus) {
            Ok(sparse) => Some(SparseEmbeddingReport {
                dimensions: sparse.dimensions(),
                indices: sparse.indices().to_vec(),
                values: sparse.values().to_vec(),
            }),
            Err(error) => {
                diagnostics.push(TextAnalysisDiagnostic::warning(
                    "sparse_embedding_unavailable",
                    error.to_string(),
                ));
                None
            }
        }
    } else {
        None
    };
    let model = embedder.model_info();
    let provenance = provenance_label(&embedder.metadata());
    Some(EmbeddingAnalysisSection {
        dimensions: vector.dimensions(),
        preview: vector.as_slice().iter().take(16).copied().collect(),
        vector: vector.as_slice().to_vec(),
        sparse,
        model,
        provenance,
    })
}

fn provenance_label(metadata: &TextEmbeddingMetadata) -> String {
    format!("{:?}", metadata.backend).to_ascii_lowercase()
}

fn candle_embedding_section(
    text: &str,
    bundle_dir: &PathBuf,
    _pooling: text_embeddings::PoolingStrategy,
    corpus: Option<&text_lexical::TfIdfCorpus>,
    options: &DocumentAnalysisOptions,
    diagnostics: &mut Vec<TextAnalysisDiagnostic>,
) -> Option<EmbeddingAnalysisSection> {
    #[cfg(all(feature = "candle", feature = "model-bundles"))]
    {
        match model_runtime::ModelBundle::load(bundle_dir)
            .map_err(|error| error.to_string())
            .and_then(|bundle| {
                text_embeddings::CandleTextEmbedder::from_bundle(bundle)
                    .map(|embedder| embedder.pooling(_pooling))
                    .map_err(|error| error.to_string())
            }) {
            Ok(embedder) => embedding_from_backend(text, &embedder, corpus, options, diagnostics),
            Err(error) => {
                diagnostics.push(TextAnalysisDiagnostic::warning(
                    "candle_embedding_unavailable",
                    error,
                ));
                None
            }
        }
    }
    #[cfg(not(all(feature = "candle", feature = "model-bundles")))]
    {
        let _ = (text, bundle_dir, corpus, options);
        diagnostics.push(TextAnalysisDiagnostic::warning(
            "candle_embedding_unavailable",
            "text-analysis was built without the `candle` and `model-bundles` features",
        ));
        None
    }
}

fn onnx_embedding_section(
    text: &str,
    bundle_dir: &PathBuf,
    _pooling: text_embeddings::PoolingStrategy,
    corpus: Option<&text_lexical::TfIdfCorpus>,
    options: &DocumentAnalysisOptions,
    diagnostics: &mut Vec<TextAnalysisDiagnostic>,
) -> Option<EmbeddingAnalysisSection> {
    #[cfg(feature = "onnx")]
    {
        match model_runtime::ModelBundle::load(bundle_dir)
            .map_err(|error| error.to_string())
            .and_then(|bundle| {
                text_embeddings::OnnxTextEmbedder::from_bundle(bundle)
                    .map(|embedder| embedder.pooling(_pooling))
                    .map_err(|error| error.to_string())
            }) {
            Ok(embedder) => embedding_from_backend(text, &embedder, corpus, options, diagnostics),
            Err(error) => {
                diagnostics.push(TextAnalysisDiagnostic::warning(
                    "onnx_embedding_unavailable",
                    error,
                ));
                None
            }
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (text, bundle_dir, corpus, options);
        diagnostics.push(TextAnalysisDiagnostic::warning(
            "onnx_embedding_unavailable",
            "text-analysis was built without the `onnx` feature",
        ));
        None
    }
}

fn effective_linguistic_depth(options: &DocumentAnalysisOptions) -> LinguisticDepth {
    if matches!(options.profile, crate::AnalysisProfile::ModelBacked)
        && matches!(options.linguistic_depth, LinguisticDepth::HeuristicBalanced)
    {
        LinguisticDepth::LocalModel {
            bundle_dir: PathBuf::from(".model-runtime"),
            auto_download: false,
            download_progress: false,
        }
    } else {
        options.linguistic_depth.clone()
    }
}

fn effective_processing(
    document: &TextDocument<'_>,
    options: &DocumentAnalysisOptions,
) -> TextProcessingOptions {
    let mut processing = options.processing.clone();
    if processing.language.is_none() {
        processing.language = options
            .language_hint
            .clone()
            .or_else(|| document.language.map(ToString::to_string));
    }
    processing
}

fn validate_options(options: &DocumentAnalysisOptions) -> Result<()> {
    if options.ngram_sizes.contains(&0) {
        return Err(invalid_argument("ngram sizes must be greater than zero"));
    }
    if options.shingle_sizes.contains(&0) {
        return Err(invalid_argument("shingle sizes must be greater than zero"));
    }
    if options.keyword_limit == 0 {
        return Err(invalid_argument("keyword limit must be greater than zero"));
    }
    Ok(())
}
