use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use text_core::{Result, TextDocument, TextSpan};
use text_embeddings::{
    DenseVector, EmbeddingModelInfo, HashedTextEmbedder, TextEmbeddingBackend,
    TextEmbeddingBackendKind, TextEmbeddingConfig,
};
use text_lexical::{term_frequencies, CorpusOptions, TermFrequency, TfIdfCorpus};

use crate::invalid_argument;

use super::derive::build_report;
use super::units::corpus_units;
use super::{
    SemanticAnalysisOptions, SemanticAnalysisReport, SemanticCluster, SemanticUnit,
    SemanticUnitKind, SpeakerConceptShare,
};

const DEFAULT_HASHED_SEMANTIC_DIMENSIONS: usize = 512;
const DEFAULT_MIN_CONCEPT_UNITS: usize = 2;
const CONCEPT_KEY_TERM_LIMIT: usize = 4;

/// One attributed text item supplied to corpus-level semantic analysis.
#[derive(Debug, Clone, Copy)]
pub struct SemanticCorpusItem<'a> {
    pub id: &'a str,
    pub author: Option<&'a str>,
    pub text: &'a str,
    pub source: Option<&'a str>,
    pub timestamp_millis: Option<i64>,
}

impl<'a> SemanticCorpusItem<'a> {
    /// Creates a corpus item without source or timestamp metadata.
    pub fn new(id: &'a str, author: Option<&'a str>, text: &'a str) -> Self {
        Self {
            id,
            author,
            text,
            source: None,
            timestamp_millis: None,
        }
    }

    /// Attaches a caller-owned source label or locator.
    pub fn with_source(mut self, source: &'a str) -> Self {
        self.source = Some(source);
        self
    }

    /// Attaches a caller-owned timestamp in Unix milliseconds.
    pub fn with_timestamp_millis(mut self, timestamp_millis: i64) -> Self {
        self.timestamp_millis = Some(timestamp_millis);
        self
    }
}

/// Corpus-level semantic analysis options.
#[derive(Debug, Clone)]
pub struct SemanticCorpusAnalysisOptions {
    pub semantic: SemanticAnalysisOptions,
    pub top_terms: usize,
    /// Minimum sentence-unit support required before a cluster is presented as a corpus concept.
    pub min_concept_units: usize,
}

impl Default for SemanticCorpusAnalysisOptions {
    fn default() -> Self {
        Self {
            semantic: SemanticAnalysisOptions::default(),
            top_terms: 20,
            min_concept_units: DEFAULT_MIN_CONCEPT_UNITS,
        }
    }
}

/// Lexical evidence aggregated over a whole corpus or one attributed author.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusLexicalProfile {
    pub item_count: usize,
    pub word_count: usize,
    pub unique_terms: usize,
    pub lexical_diversity: f32,
    pub top_terms: Vec<TermFrequency>,
}

/// Source metadata retained independently from semantic-unit spans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusSource {
    pub id: String,
    pub author: Option<String>,
    pub source: Option<String>,
    pub timestamp_millis: Option<i64>,
}

/// One representative passage for a corpus-level deterministic concept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusPassage {
    pub unit_id: String,
    pub source_id: String,
    pub author: Option<String>,
    pub source: Option<String>,
    pub timestamp_millis: Option<i64>,
    pub span: TextSpan,
    pub text: String,
}

/// Evidence summary retaining where a corpus-level concept came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusConceptEvidence {
    pub cluster_id: String,
    /// Deterministic corpus-derived key-term label; this is evidence, not a model-generated name.
    pub label: String,
    pub key_terms: Vec<String>,
    pub coherence: f32,
    pub member_unit_count: usize,
    pub source_item_count: usize,
    pub author_count: usize,
    pub source_item_ids: Vec<String>,
    pub authors: Vec<String>,
    pub representative: SemanticCorpusPassage,
}

/// Corpus profile for one attributed author.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusAuthorProfile {
    pub author: String,
    pub item_count: usize,
    pub semantic_unit_count: usize,
    pub lexical: SemanticCorpusLexicalProfile,
    pub concepts: Vec<SpeakerConceptShare>,
}

/// Deterministic lexical and semantic aggregation over many attributed texts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticCorpusReport {
    pub item_count: usize,
    pub author_count: usize,
    pub sources: Vec<SemanticCorpusSource>,
    pub lexical: SemanticCorpusLexicalProfile,
    pub authors: Vec<SemanticCorpusAuthorProfile>,
    /// Supported recurring concepts only. Low-support clusters remain available in `semantic`.
    pub concepts: Vec<SemanticCorpusConceptEvidence>,
    /// Number of sentence units assigned only to clusters below `min_concept_units` support.
    pub non_concept_unit_count: usize,
    pub semantic: SemanticAnalysisReport,
}

/// Analyzes an attributed corpus with a deterministic corpus-aware hashed TF-IDF baseline.
///
/// This baseline is intentionally local and reproducible. It should not be presented as equivalent
/// to a learned sentence-embedding model; callers that own such a model can use
/// [`analyze_corpus_semantics_with`] instead.
pub fn analyze_corpus_semantics(
    items: &[SemanticCorpusItem<'_>],
    options: &SemanticCorpusAnalysisOptions,
) -> Result<SemanticCorpusReport> {
    super::validate_options(&options.semantic)?;
    validate_corpus(items, options)?;

    let units = corpus_units(items, &options.semantic);
    let corpus_options = semantic_embedding_corpus_options();
    let tfidf = semantic_tfidf_corpus(&units, corpus_options.clone())?;
    let embedder = CorpusAwareHashedTextEmbedder {
        embedder: HashedTextEmbedder::new(
            TextEmbeddingConfig {
                dimensions: DEFAULT_HASHED_SEMANTIC_DIMENSIONS,
                use_idf: true,
            },
            corpus_options,
        )?,
        corpus: tfidf,
    };

    analyze_corpus_semantics_prepared(items, options, units, &embedder)
}

/// Analyzes an attributed corpus with a caller-supplied embedding backend.
pub fn analyze_corpus_semantics_with<E: TextEmbeddingBackend + ?Sized>(
    items: &[SemanticCorpusItem<'_>],
    options: &SemanticCorpusAnalysisOptions,
    embedder: &E,
) -> Result<SemanticCorpusReport> {
    super::validate_options(&options.semantic)?;
    validate_corpus(items, options)?;

    analyze_corpus_semantics_prepared(
        items,
        options,
        corpus_units(items, &options.semantic),
        embedder,
    )
}

fn analyze_corpus_semantics_prepared<E: TextEmbeddingBackend + ?Sized>(
    items: &[SemanticCorpusItem<'_>],
    options: &SemanticCorpusAnalysisOptions,
    units: Vec<SemanticUnit>,
    embedder: &E,
) -> Result<SemanticCorpusReport> {
    let sources = items
        .iter()
        .map(|item| SemanticCorpusSource {
            id: item.id.to_string(),
            author: normalized_author(item.author),
            source: item.source.map(ToString::to_string),
            timestamp_millis: item.timestamp_millis,
        })
        .collect::<Vec<_>>();

    let semantic = build_report(
        items.iter().map(|item| item.id.to_string()).collect(),
        SemanticUnitKind::Sentence,
        units,
        &options.semantic,
        embedder,
    )?;

    let lexical = lexical_profile(items.iter().copied(), options.top_terms);
    let authors = author_profiles(
        items,
        &semantic,
        options.top_terms,
        options.min_concept_units,
    );
    let concepts = concept_evidence(&semantic, &sources, options.min_concept_units);
    let non_concept_unit_count = semantic
        .clusters
        .iter()
        .filter(|cluster| cluster.member_unit_ids.len() < options.min_concept_units)
        .map(|cluster| cluster.member_unit_ids.len())
        .sum();

    Ok(SemanticCorpusReport {
        item_count: items.len(),
        author_count: authors.len(),
        sources,
        lexical,
        authors,
        concepts,
        non_concept_unit_count,
        semantic,
    })
}

fn validate_corpus(
    items: &[SemanticCorpusItem<'_>],
    options: &SemanticCorpusAnalysisOptions,
) -> Result<()> {
    if items.is_empty() {
        return Err(invalid_argument(
            "semantic corpus must contain at least one item",
        ));
    }
    if options.top_terms == 0 {
        return Err(invalid_argument(
            "semantic corpus top_terms must be greater than zero",
        ));
    }
    if options.min_concept_units == 0 {
        return Err(invalid_argument(
            "semantic corpus min_concept_units must be greater than zero",
        ));
    }

    let mut seen = BTreeSet::new();
    for item in items {
        if item.id.trim().is_empty() {
            return Err(invalid_argument(
                "semantic corpus item id must not be empty",
            ));
        }
        if !seen.insert(item.id) {
            return Err(invalid_argument(format!(
                "duplicate semantic corpus item id `{}`",
                item.id
            )));
        }
        if item.text.trim().is_empty() {
            return Err(invalid_argument(format!(
                "semantic corpus item `{}` text must not be empty",
                item.id
            )));
        }
        if item.author.is_some_and(|author| author.trim().is_empty()) {
            return Err(invalid_argument(format!(
                "semantic corpus item `{}` author must not be empty when present",
                item.id
            )));
        }
        if item.source.is_some_and(|source| source.trim().is_empty()) {
            return Err(invalid_argument(format!(
                "semantic corpus item `{}` source must not be empty when present",
                item.id
            )));
        }
    }
    Ok(())
}

fn semantic_embedding_corpus_options() -> CorpusOptions {
    CorpusOptions {
        min_term_len: 2,
        ..CorpusOptions::default()
    }
}

fn semantic_tfidf_corpus(units: &[SemanticUnit], options: CorpusOptions) -> Result<TfIdfCorpus> {
    let documents = units
        .iter()
        .filter(|unit| unit.kind == SemanticUnitKind::Sentence)
        .map(|unit| TextDocument::new(unit.id.as_str(), unit.text.as_str()))
        .collect::<Vec<_>>();
    TfIdfCorpus::from_documents(documents, options)
}

struct CorpusAwareHashedTextEmbedder {
    embedder: HashedTextEmbedder,
    corpus: TfIdfCorpus,
}

impl TextEmbeddingBackend for CorpusAwareHashedTextEmbedder {
    fn embed_text(&self, text: &str) -> Result<DenseVector> {
        self.embedder
            .embed_text_with_corpus(text, Some(&self.corpus))
    }

    fn model_info(&self) -> EmbeddingModelInfo {
        EmbeddingModelInfo {
            model_name: "hashed-tfidf-sentence-baseline".to_string(),
            backend: TextEmbeddingBackendKind::Hashed,
            dimensions: self.embedder.config.dimensions,
            normalized: true,
            max_tokens: None,
        }
    }
}

fn lexical_profile<'a, I>(items: I, top_terms: usize) -> SemanticCorpusLexicalProfile
where
    I: IntoIterator<Item = SemanticCorpusItem<'a>>,
{
    let items = items.into_iter().collect::<Vec<_>>();
    let text = items
        .iter()
        .map(|item| item.text)
        .collect::<Vec<_>>()
        .join("\n");
    let mut terms = term_frequencies(&text);
    let word_count = terms.iter().map(|term| term.count).sum::<usize>();
    let unique_terms = terms.len();
    let lexical_diversity = if word_count == 0 {
        0.0
    } else {
        unique_terms as f32 / word_count as f32
    };
    terms.truncate(top_terms);
    SemanticCorpusLexicalProfile {
        item_count: items.len(),
        word_count,
        unique_terms,
        lexical_diversity,
        top_terms: terms,
    }
}

fn author_profiles(
    items: &[SemanticCorpusItem<'_>],
    semantic: &SemanticAnalysisReport,
    top_terms: usize,
    min_concept_units: usize,
) -> Vec<SemanticCorpusAuthorProfile> {
    let mut items_by_author = BTreeMap::<String, Vec<SemanticCorpusItem<'_>>>::new();
    for item in items {
        let Some(author) = normalized_author(item.author) else {
            continue;
        };
        items_by_author.entry(author).or_default().push(*item);
    }
    let semantic_by_author = semantic
        .speaker_profiles
        .iter()
        .map(|profile| (profile.speaker.as_str(), profile))
        .collect::<BTreeMap<_, _>>();
    let supported_cluster_ids = semantic
        .clusters
        .iter()
        .filter(|cluster| cluster.member_unit_ids.len() >= min_concept_units)
        .map(|cluster| cluster.id.as_str())
        .collect::<BTreeSet<_>>();

    items_by_author
        .into_iter()
        .map(|(author, author_items)| {
            let semantic_profile = semantic_by_author.get(author.as_str()).copied();
            SemanticCorpusAuthorProfile {
                author,
                item_count: author_items.len(),
                semantic_unit_count: semantic_profile.map_or(0, |profile| profile.unit_count),
                lexical: lexical_profile(author_items.iter().copied(), top_terms),
                concepts: semantic_profile
                    .map(|profile| {
                        profile
                            .concepts
                            .iter()
                            .filter(|concept| {
                                supported_cluster_ids.contains(concept.cluster_id.as_str())
                            })
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default(),
            }
        })
        .collect()
}

fn concept_evidence(
    semantic: &SemanticAnalysisReport,
    sources: &[SemanticCorpusSource],
    min_concept_units: usize,
) -> Vec<SemanticCorpusConceptEvidence> {
    let units = semantic
        .units
        .iter()
        .map(|unit| (unit.id.as_str(), unit))
        .collect::<BTreeMap<_, _>>();
    let source_by_id = sources
        .iter()
        .map(|source| (source.id.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    let primary_units = semantic
        .units
        .iter()
        .filter(|unit| unit.kind == semantic.primary_unit_kind)
        .collect::<Vec<_>>();
    let document_frequency = term_document_frequency(&primary_units);
    let total_primary_units = primary_units.len();

    let mut concepts = semantic
        .clusters
        .iter()
        .filter(|cluster| cluster.member_unit_ids.len() >= min_concept_units)
        .filter_map(|cluster| {
            let representative = units
                .get(cluster.representative_unit_id.as_str())
                .copied()?;
            let source = source_by_id.get(representative.source_id.as_str()).copied();
            let mut source_item_ids = BTreeSet::new();
            let mut authors = BTreeSet::new();
            for member_id in &cluster.member_unit_ids {
                let Some(unit) = units.get(member_id.as_str()).copied() else {
                    continue;
                };
                source_item_ids.insert(unit.source_id.clone());
                if let Some(author) = &unit.speaker {
                    authors.insert(author.clone());
                }
            }
            let key_terms =
                concept_key_terms(cluster, &units, &document_frequency, total_primary_units);
            let label = if key_terms.is_empty() {
                cluster.id.clone()
            } else {
                key_terms
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" · ")
            };
            Some(SemanticCorpusConceptEvidence {
                cluster_id: cluster.id.clone(),
                label,
                key_terms,
                coherence: cluster.mean_similarity,
                member_unit_count: cluster.member_unit_ids.len(),
                source_item_count: source_item_ids.len(),
                author_count: authors.len(),
                source_item_ids: source_item_ids.into_iter().collect(),
                authors: authors.into_iter().collect(),
                representative: passage(representative, source),
            })
        })
        .collect::<Vec<_>>();

    concepts.sort_by(|left, right| {
        right
            .member_unit_count
            .cmp(&left.member_unit_count)
            .then_with(|| right.coherence.total_cmp(&left.coherence))
            .then_with(|| left.cluster_id.cmp(&right.cluster_id))
    });
    concepts
}

fn term_document_frequency(primary_units: &[&SemanticUnit]) -> BTreeMap<String, usize> {
    let mut document_frequency = BTreeMap::new();
    for unit in primary_units {
        for term in term_frequencies(&unit.text) {
            *document_frequency.entry(term.term).or_default() += 1;
        }
    }
    document_frequency
}

fn concept_key_terms(
    cluster: &SemanticCluster,
    units: &BTreeMap<&str, &SemanticUnit>,
    document_frequency: &BTreeMap<String, usize>,
    total_primary_units: usize,
) -> Vec<String> {
    let mut cluster_counts = BTreeMap::<String, usize>::new();
    for member_id in &cluster.member_unit_ids {
        let Some(unit) = units.get(member_id.as_str()).copied() else {
            continue;
        };
        for term in term_frequencies(&unit.text) {
            *cluster_counts.entry(term.term).or_default() += term.count;
        }
    }

    let mut scored = cluster_counts
        .into_iter()
        .filter_map(|(term, count)| {
            if term.chars().count() < 2 || !term.chars().any(|character| character.is_alphabetic())
            {
                return None;
            }
            let document_count = document_frequency.get(&term).copied().unwrap_or(0);
            if total_primary_units >= 5 && document_count * 5 >= total_primary_units * 4 {
                return None;
            }
            let idf =
                (((total_primary_units + 1) as f32) / ((document_count + 1) as f32)).ln() + 1.0;
            let score = (1.0 + (count as f32).ln()) * idf * idf;
            Some((term, count, score))
        })
        .collect::<Vec<_>>();
    scored.sort_by(
        |(left_term, left_count, left_score), (right_term, right_count, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| right_count.cmp(left_count))
                .then_with(|| left_term.cmp(right_term))
        },
    );
    scored
        .into_iter()
        .take(CONCEPT_KEY_TERM_LIMIT)
        .map(|(term, _, _)| term)
        .collect()
}

fn passage(unit: &SemanticUnit, source: Option<&SemanticCorpusSource>) -> SemanticCorpusPassage {
    SemanticCorpusPassage {
        unit_id: unit.id.clone(),
        source_id: unit.source_id.clone(),
        author: unit.speaker.clone(),
        source: source.and_then(|source| source.source.clone()),
        timestamp_millis: source.and_then(|source| source.timestamp_millis),
        span: unit.span,
        text: unit.text.clone(),
    }
}

fn normalized_author(author: Option<&str>) -> Option<String> {
    author.map(str::trim).map(ToString::to_string)
}
