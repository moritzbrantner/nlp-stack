#![doc = include_str!("../README.md")]

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};

use math_sparse_data::{CooMatrix, CsrMatrix, SparseVector};
use serde::{Deserialize, Serialize};
use text_core::{
    segment_document_id, tokenize, TextAnnotationSpan, TextDocument, TextDocumentContract,
    TextProcessingOptions, TextProvenance, TextSegmentContract, TextSourceRef, TimestampContract,
    TokenKind,
};
use text_core::{DetectError, Result, TextSegment};

const TEXT_CORPUS_SNAPSHOT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for corpus options.
pub struct CorpusOptions {
    /// The processing value.
    pub processing: TextProcessingOptions,
    /// The min term len value.
    pub min_term_len: usize,
    /// The stop words value.
    pub stop_words: BTreeSet<String>,
    /// The max terms per document value.
    pub max_terms_per_document: Option<usize>,
}

impl Default for CorpusOptions {
    fn default() -> Self {
        Self {
            processing: TextProcessingOptions::default(),
            min_term_len: 1,
            stop_words: BTreeSet::new(),
            max_terms_per_document: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for user-facing lexical text corpus.
pub struct TextCorpus {
    /// Options used when deriving lexical indexes from this corpus.
    pub options: CorpusOptions,
    /// Documents contained in this corpus.
    pub documents: Vec<TextCorpusDocument>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for a text corpus document.
pub struct TextCorpusDocument {
    /// Identifier for this value.
    pub id: String,
    /// Text content for this value.
    pub text: String,
    /// Language tag for this value.
    pub language: Option<String>,
    /// Optional timestamp associated with this document.
    #[serde(default)]
    pub timestamp: Option<TimestampContract>,
    /// Optional rich source reference.
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    /// Provenance records for this document.
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    /// Annotation spans associated with this document.
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
    /// Metadata associated with this value.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Serializable snapshot of a deterministic lexical corpus index.
pub struct TextCorpusSnapshot {
    /// Snapshot schema version.
    pub schema_version: u32,
    /// Options used to build the index.
    pub options: CorpusOptions,
    /// Indexed documents in corpus order.
    pub documents: Vec<TextCorpusSnapshotDocument>,
    /// Aggregate corpus statistics.
    pub stats: CorpusStats,
    /// Sorted vocabulary.
    pub vocabulary: Vec<String>,
    /// Number of documents containing each term.
    pub document_frequency: BTreeMap<String, usize>,
    /// Total collection count for each term.
    pub collection_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Serializable snapshot entry for one corpus document.
pub struct TextCorpusSnapshotDocument {
    /// Identifier for this value.
    pub id: String,
    /// Language tag for this value.
    pub language: Option<String>,
    /// Optional timestamp associated with this document.
    #[serde(default)]
    pub timestamp: Option<TimestampContract>,
    /// Optional rich source reference.
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    /// Provenance records for this document.
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    /// Annotation spans associated with this document.
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
    /// Metadata associated with this value.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// Total terms in this document.
    pub total_terms: usize,
    /// Per-term counts in this document.
    pub term_counts: BTreeMap<String, usize>,
}

impl TextCorpusDocument {
    /// Creates a new value.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            language: None,
            timestamp: None,
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}

impl TextCorpus {
    /// Creates a new value.
    pub fn new(options: CorpusOptions) -> Self {
        Self {
            options,
            documents: Vec::new(),
        }
    }

    /// Builds this value from texts.
    pub fn from_texts<I, S>(texts: I, options: CorpusOptions) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut corpus = Self::new(options);
        for (index, text) in texts.into_iter().enumerate() {
            corpus.add_document(TextCorpusDocument::new(
                format!("doc-{index}"),
                text.as_ref(),
            ))?;
        }
        Ok(corpus)
    }

    /// Builds this value from corpus documents.
    pub fn from_documents<I>(documents: I, options: CorpusOptions) -> Result<Self>
    where
        I: IntoIterator<Item = TextCorpusDocument>,
    {
        let mut corpus = Self::new(options);
        for document in documents {
            corpus.add_document(document)?;
        }
        Ok(corpus)
    }

    /// Builds this value from portable text document contracts.
    pub fn from_document_contracts<I, D>(documents: I, options: CorpusOptions) -> Result<Self>
    where
        I: IntoIterator<Item = D>,
        D: Borrow<TextDocumentContract>,
    {
        let mut corpus = Self::new(options);
        for document in documents {
            corpus.add_document_contract(document.borrow())?;
        }
        Ok(corpus)
    }

    /// Builds this value from portable text segment contracts.
    pub fn from_segment_contracts<I, S>(segments: I, options: CorpusOptions) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Borrow<TextSegmentContract>,
    {
        let mut corpus = Self::new(options);
        for segment in segments {
            corpus.add_segment_contract(segment.borrow())?;
        }
        Ok(corpus)
    }

    /// Adds document to this value.
    pub fn add_document(&mut self, document: TextCorpusDocument) -> Result<usize> {
        validate_document_id(&document.id)?;
        if self
            .documents
            .iter()
            .any(|existing| existing.id == document.id)
        {
            return Err(duplicate_document_id(&document.id));
        }

        self.documents.push(document);
        Ok(self.documents.len() - 1)
    }

    /// Adds text document to this value.
    pub fn add_text_document(&mut self, document: &TextDocument<'_>) -> Result<usize> {
        self.add_document(TextCorpusDocument {
            id: document.id.to_string(),
            text: document.text.to_string(),
            language: document.language.map(ToString::to_string),
            timestamp: document.timestamp.map(Into::into),
            source: document.timestamp.map(|timestamp| TextSourceRef {
                source_id: None,
                source_kind: Some("text_document".to_string()),
                uri: None,
                media_timestamp: Some(timestamp.into()),
                duration_seconds: None,
            }),
            provenance: Vec::new(),
            annotations: Vec::new(),
            metadata: BTreeMap::new(),
        })
    }

    /// Adds portable text document contract to this value.
    pub fn add_document_contract(&mut self, document: &TextDocumentContract) -> Result<usize> {
        self.add_document(text_corpus_document_from_contract(document))
    }

    /// Adds portable text segment contract to this value.
    pub fn add_segment_contract(&mut self, segment: &TextSegmentContract) -> Result<usize> {
        self.add_document(text_corpus_document_from_segment_contract(segment))
    }

    /// Replaces an existing document.
    pub fn replace_document(&mut self, document: TextCorpusDocument) -> Result<TextCorpusDocument> {
        validate_document_id(&document.id)?;
        let Some(index) = self
            .documents
            .iter()
            .position(|existing| existing.id == document.id)
        else {
            return Err(invalid_argument(format!(
                "document `{}` was not found",
                document.id
            )));
        };

        Ok(std::mem::replace(&mut self.documents[index], document))
    }

    /// Removes document from this value.
    pub fn remove_document(&mut self, id: &str) -> Option<TextCorpusDocument> {
        let index = self
            .documents
            .iter()
            .position(|document| document.id == id)?;
        Some(self.documents.remove(index))
    }

    /// Returns document.
    pub fn document(&self, id: &str) -> Option<&TextCorpusDocument> {
        self.documents.iter().find(|document| document.id == id)
    }

    /// Returns len.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Returns whether is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Returns borrowed text documents.
    pub fn as_text_documents(&self) -> Vec<TextDocument<'_>> {
        self.text_documents().collect()
    }

    /// Returns an iterator over borrowed text documents.
    pub fn text_documents(&self) -> impl Iterator<Item = TextDocument<'_>> + '_ {
        self.documents.iter().map(|document| TextDocument {
            id: &document.id,
            text: &document.text,
            language: document.language.as_deref(),
            timestamp: document.timestamp.map(Into::into),
        })
    }

    /// Returns full-fidelity portable text document contracts.
    pub fn text_document_contracts(&self) -> Vec<TextDocumentContract> {
        self.documents
            .iter()
            .map(|document| TextDocumentContract {
                id: document.id.clone(),
                text: document.text.clone(),
                language: document.language.clone(),
                timestamp: document.timestamp,
                attributes: document.metadata.clone(),
                source: document.source.clone(),
                provenance: document.provenance.clone(),
                annotations: document.annotations.clone(),
            })
            .collect()
    }

    /// Builds a TF-IDF scoring corpus from this value.
    pub fn to_tfidf_corpus(&self) -> Result<TfIdfCorpus> {
        TfIdfCorpus::from_documents(self.text_documents(), self.options.clone())
    }

    /// Builds a BM25 scoring corpus from this value.
    pub fn to_bm25_corpus(&self, options: Bm25Options) -> Result<Bm25Corpus> {
        Bm25Corpus::from_documents(self.text_documents(), options)
    }

    /// Returns a reproducible lexical corpus snapshot.
    pub fn snapshot(&self) -> Result<TextCorpusSnapshot> {
        let mut snapshot = self.to_tfidf_corpus()?.snapshot();
        for snapshot_document in &mut snapshot.documents {
            if let Some(document) = self.document(&snapshot_document.id) {
                snapshot_document.language = document.language.clone();
                snapshot_document.timestamp = document.timestamp;
                snapshot_document.source = document.source.clone();
                snapshot_document.provenance = document.provenance.clone();
                snapshot_document.annotations = document.annotations.clone();
                snapshot_document.metadata = document.metadata.clone();
            }
        }
        Ok(snapshot)
    }
}

impl TextCorpusSnapshot {
    /// Validates this snapshot for deterministic reconstruction.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != TEXT_CORPUS_SNAPSHOT_SCHEMA_VERSION {
            return Err(invalid_argument(format!(
                "unsupported text corpus snapshot schema version {}",
                self.schema_version
            )));
        }

        let mut ids = BTreeSet::new();
        let mut document_frequency = BTreeMap::new();
        let mut collection_counts = BTreeMap::new();
        let mut total_terms = 0;

        for document in &self.documents {
            validate_document_id(&document.id)?;
            if !ids.insert(document.id.clone()) {
                return Err(duplicate_document_id(&document.id));
            }

            let mut document_total_terms = 0;
            for (term, count) in &document.term_counts {
                if *count == 0 {
                    return Err(invalid_argument(format!(
                        "term `{term}` in document `{}` has zero count",
                        document.id
                    )));
                }
                document_total_terms += count;
                *document_frequency.entry(term.clone()).or_insert(0) += 1;
                *collection_counts.entry(term.clone()).or_insert(0) += count;
            }
            if document.total_terms != document_total_terms {
                return Err(invalid_argument(format!(
                    "document `{}` total_terms does not match term counts",
                    document.id
                )));
            }
            total_terms += document.total_terms;
        }

        if self.document_frequency != document_frequency {
            return Err(invalid_argument(
                "document frequency does not match documents",
            ));
        }
        if self.collection_counts != collection_counts {
            return Err(invalid_argument("collection counts do not match documents"));
        }

        let vocabulary = collection_counts.keys().cloned().collect::<Vec<_>>();
        if self.vocabulary != vocabulary {
            return Err(invalid_argument("vocabulary does not match documents"));
        }

        let stats = CorpusStats {
            documents: self.documents.len(),
            total_terms,
            unique_terms: collection_counts.len(),
            average_terms_per_document: if self.documents.is_empty() {
                0.0
            } else {
                total_terms as f32 / self.documents.len() as f32
            },
        };
        if self.stats.documents != stats.documents
            || self.stats.total_terms != stats.total_terms
            || self.stats.unique_terms != stats.unique_terms
            || (self.stats.average_terms_per_document - stats.average_terms_per_document).abs()
                > f32::EPSILON
        {
            return Err(invalid_argument("corpus stats do not match documents"));
        }

        Ok(())
    }

    /// Reconstructs a TF-IDF corpus from this snapshot.
    pub fn to_tfidf_corpus(&self) -> Result<TfIdfCorpus> {
        tfidf_from_snapshot_parts(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for indexed document.
pub struct IndexedDocument {
    /// Identifier for this value.
    pub id: String,
    /// The term counts value.
    pub term_counts: BTreeMap<String, usize>,
    /// The total terms value.
    pub total_terms: usize,
}

impl IndexedDocument {
    /// Returns unique terms.
    pub fn unique_terms(&self) -> usize {
        self.term_counts.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// Data type for corpus stats.
pub struct CorpusStats {
    /// The documents value.
    pub documents: usize,
    /// The total terms value.
    pub total_terms: usize,
    /// The unique terms value.
    pub unique_terms: usize,
    /// The average terms per document value.
    pub average_terms_per_document: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for corpus term stats.
pub struct CorpusTermStats {
    /// The term value.
    pub term: String,
    /// The collection count value.
    pub collection_count: usize,
    /// The document count value.
    pub document_count: usize,
    /// The collection frequency value.
    pub collection_frequency: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for tf idf term.
pub struct TfIdfTerm {
    /// The term value.
    pub term: String,
    /// Number of items represented by this value.
    pub count: usize,
    /// The term frequency value.
    pub term_frequency: f32,
    /// The inverse document frequency value.
    pub inverse_document_frequency: f32,
    /// Score assigned to this value.
    pub score: f32,
    /// The document count value.
    pub document_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for document search result.
pub struct DocumentSearchResult {
    /// Identifier for this value.
    pub id: String,
    /// Score assigned to this value.
    pub score: f32,
    /// The matched terms value.
    pub matched_terms: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for bm25 options.
pub struct Bm25Options {
    /// The k1 value.
    pub k1: f32,
    /// The b value.
    pub b: f32,
    /// The min term len value.
    pub min_term_len: usize,
    /// The stop words value.
    pub stop_words: BTreeSet<String>,
}

impl Default for Bm25Options {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_term_len: 1,
            stop_words: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for bm25 search result.
pub struct Bm25SearchResult {
    /// Identifier for this value.
    pub id: String,
    /// Score assigned to this value.
    pub score: f32,
    /// The matched terms value.
    pub matched_terms: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for sparse term matrix.
pub struct SparseTermMatrix {
    /// The vocabulary value.
    pub vocabulary: Vec<String>,
    /// The matrix value.
    pub matrix: CsrMatrix,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for tf idf corpus.
pub struct TfIdfCorpus {
    options: CorpusOptions,
    documents: Vec<IndexedDocument>,
    document_frequency: BTreeMap<String, usize>,
    collection_counts: BTreeMap<String, usize>,
    total_terms: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for bm25 corpus.
pub struct Bm25Corpus {
    options: Bm25Options,
    documents: Vec<IndexedDocument>,
    document_frequency: BTreeMap<String, usize>,
    total_terms: usize,
}

impl Default for Bm25Corpus {
    fn default() -> Self {
        Self::new(Bm25Options::default())
    }
}

impl Bm25Corpus {
    /// Creates a new value.
    pub fn new(options: Bm25Options) -> Self {
        Self {
            options,
            documents: Vec::new(),
            document_frequency: BTreeMap::new(),
            total_terms: 0,
        }
    }

    /// Builds this value from texts.
    pub fn from_texts<I, S>(texts: I, options: Bm25Options) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut corpus = Self::new(options);
        for (index, text) in texts.into_iter().enumerate() {
            corpus.add_document(format!("doc-{index}"), text.as_ref())?;
        }
        Ok(corpus)
    }

    /// Builds this value from documents.
    pub fn from_documents<'a, I>(documents: I, options: Bm25Options) -> Result<Self>
    where
        I: IntoIterator<Item = TextDocument<'a>>,
    {
        let mut corpus = Self::new(options);
        for document in documents {
            corpus.add_text_document(&document)?;
        }
        Ok(corpus)
    }

    /// Builds this value from text segments.
    pub fn from_text_segments<'a, I>(
        stream_id: &str,
        segments: I,
        options: Bm25Options,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = TextSegment<'a>>,
    {
        validate_stream_id(stream_id)?;
        let mut corpus = Self::new(options);
        for segment in segments {
            corpus.add_text_segment(stream_id, &segment)?;
        }
        Ok(corpus)
    }

    /// Returns options.
    pub fn options(&self) -> &Bm25Options {
        &self.options
    }

    /// Returns len.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Returns whether is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Returns documents.
    pub fn documents(&self) -> &[IndexedDocument] {
        &self.documents
    }

    /// Adds add text document to this value.
    pub fn add_text_document(&mut self, document: &TextDocument<'_>) -> Result<usize> {
        self.add_document(document.id, document.text)
    }

    /// Adds add text segment to this value.
    pub fn add_text_segment(
        &mut self,
        stream_id: &str,
        segment: &TextSegment<'_>,
    ) -> Result<usize> {
        validate_stream_id(stream_id)?;
        self.add_document(
            segment_document_id(stream_id, segment.segment_index),
            segment.text,
        )
    }

    /// Adds add document to this value.
    pub fn add_document(&mut self, id: impl Into<String>, text: &str) -> Result<usize> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(invalid_argument("document id must not be empty"));
        }
        if self.documents.iter().any(|document| document.id == id) {
            return Err(invalid_argument(format!(
                "document id `{id}` already exists"
            )));
        }
        if self.options.k1 <= 0.0 || !self.options.k1.is_finite() {
            return Err(invalid_argument(
                "BM25 k1 must be finite and greater than zero",
            ));
        }
        if !(0.0..=1.0).contains(&self.options.b) || !self.options.b.is_finite() {
            return Err(invalid_argument(
                "BM25 b must be finite and between 0 and 1",
            ));
        }

        let term_counts = bm25_term_counts(text, &self.options);
        let indexed = IndexedDocument {
            id,
            total_terms: term_counts.values().sum(),
            term_counts,
        };
        for term in indexed.term_counts.keys() {
            *self.document_frequency.entry(term.clone()).or_insert(0) += 1;
        }
        self.total_terms += indexed.total_terms;
        self.documents.push(indexed);
        Ok(self.documents.len() - 1)
    }

    /// Returns search.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Bm25SearchResult>> {
        if limit == 0 {
            return Err(invalid_argument("search limit must be greater than zero"));
        }
        if self.documents.is_empty() {
            return Ok(Vec::new());
        }
        let query_counts = bm25_term_counts(query, &self.options);
        if query_counts.is_empty() {
            return Ok(Vec::new());
        }

        let average_document_len = self.total_terms as f32 / self.documents.len() as f32;
        let mut results = Vec::new();
        for document in &self.documents {
            if document.total_terms == 0 {
                continue;
            }
            let mut score = 0.0_f32;
            let mut matched_terms = 0;
            for (term, query_count) in &query_counts {
                let Some(term_count) = document.term_counts.get(term).copied() else {
                    continue;
                };
                matched_terms += 1;
                let idf = self.inverse_document_frequency(term);
                let tf = term_count as f32;
                let length_factor = 1.0 - self.options.b
                    + self.options.b * document.total_terms as f32
                        / average_document_len.max(f32::EPSILON);
                let numerator = tf * (self.options.k1 + 1.0);
                let denominator = tf + self.options.k1 * length_factor;
                score += *query_count as f32 * idf * numerator / denominator;
            }
            if matched_terms > 0 {
                results.push(Bm25SearchResult {
                    id: document.id.clone(),
                    score,
                    matched_terms,
                });
            }
        }
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.matched_terms.cmp(&left.matched_terms))
                .then_with(|| left.id.cmp(&right.id))
        });
        results.truncate(limit);
        Ok(results)
    }

    /// Returns inverse document frequency.
    pub fn inverse_document_frequency(&self, term: &str) -> f32 {
        let documents = self.documents.len() as f32;
        let document_count = self.document_frequency.get(term).copied().unwrap_or(0) as f32;
        (1.0 + (documents - document_count + 0.5) / (document_count + 0.5)).ln()
    }

    /// Returns sparse term matrix.
    pub fn sparse_term_matrix(&self) -> Result<SparseTermMatrix> {
        build_sparse_term_matrix(&self.documents)
    }
}

impl Default for TfIdfCorpus {
    fn default() -> Self {
        Self::new(CorpusOptions::default())
    }
}

impl TfIdfCorpus {
    /// Creates a new value.
    pub fn new(options: CorpusOptions) -> Self {
        Self {
            options,
            documents: Vec::new(),
            document_frequency: BTreeMap::new(),
            collection_counts: BTreeMap::new(),
            total_terms: 0,
        }
    }

    /// Builds this value from texts.
    pub fn from_texts<I, S>(texts: I, options: CorpusOptions) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut corpus = Self::new(options);
        for (index, text) in texts.into_iter().enumerate() {
            corpus.add_document(format!("doc-{index}"), text.as_ref())?;
        }
        Ok(corpus)
    }

    /// Builds this value from documents.
    pub fn from_documents<'a, I>(documents: I, options: CorpusOptions) -> Result<Self>
    where
        I: IntoIterator<Item = TextDocument<'a>>,
    {
        let mut corpus = Self::new(options);
        for document in documents {
            corpus.add_text_document(&document)?;
        }
        Ok(corpus)
    }

    /// Builds this value from text segments.
    pub fn from_text_segments<'a, I>(
        stream_id: &str,
        segments: I,
        options: CorpusOptions,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = TextSegment<'a>>,
    {
        validate_stream_id(stream_id)?;
        let mut corpus = Self::new(options);
        for segment in segments {
            corpus.add_text_segment(stream_id, &segment)?;
        }
        Ok(corpus)
    }

    /// Returns options.
    pub fn options(&self) -> &CorpusOptions {
        &self.options
    }

    /// Returns len.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Returns whether is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Returns documents.
    pub fn documents(&self) -> &[IndexedDocument] {
        &self.documents
    }

    /// Returns document.
    pub fn document(&self, id: &str) -> Option<&IndexedDocument> {
        self.documents.iter().find(|document| document.id == id)
    }

    /// Adds add text document to this value.
    pub fn add_text_document(&mut self, document: &TextDocument<'_>) -> Result<usize> {
        self.add_document(document.id, document.text)
    }

    /// Adds add text segment to this value.
    pub fn add_text_segment(
        &mut self,
        stream_id: &str,
        segment: &TextSegment<'_>,
    ) -> Result<usize> {
        validate_stream_id(stream_id)?;
        self.add_document(
            segment_document_id(stream_id, segment.segment_index),
            segment.text,
        )
    }

    /// Adds add document to this value.
    pub fn add_document(&mut self, id: impl Into<String>, text: &str) -> Result<usize> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(invalid_argument("document id must not be empty"));
        }
        if self.documents.iter().any(|document| document.id == id) {
            return Err(invalid_argument(format!(
                "document id `{id}` already exists"
            )));
        }

        let term_counts = term_counts(text, &self.options);
        let mut indexed = IndexedDocument {
            id,
            total_terms: term_counts.values().sum(),
            term_counts,
        };
        if let Some(max_terms) = self.options.max_terms_per_document {
            indexed.term_counts = retain_top_counts(&indexed.term_counts, max_terms);
            indexed.total_terms = indexed.term_counts.values().sum();
        }

        for (term, count) in &indexed.term_counts {
            *self.collection_counts.entry(term.clone()).or_insert(0) += count;
            *self.document_frequency.entry(term.clone()).or_insert(0) += 1;
        }
        self.total_terms += indexed.total_terms;
        self.documents.push(indexed);
        Ok(self.documents.len() - 1)
    }

    /// Returns stats.
    pub fn stats(&self) -> CorpusStats {
        CorpusStats {
            documents: self.documents.len(),
            total_terms: self.total_terms,
            unique_terms: self.collection_counts.len(),
            average_terms_per_document: if self.documents.is_empty() {
                0.0
            } else {
                self.total_terms as f32 / self.documents.len() as f32
            },
        }
    }

    /// Returns sparse term matrix.
    pub fn sparse_term_matrix(&self) -> Result<SparseTermMatrix> {
        build_sparse_term_matrix(&self.documents)
    }

    /// Returns a reproducible lexical corpus snapshot.
    pub fn snapshot(&self) -> TextCorpusSnapshot {
        TextCorpusSnapshot {
            schema_version: TEXT_CORPUS_SNAPSHOT_SCHEMA_VERSION,
            options: self.options.clone(),
            documents: self
                .documents
                .iter()
                .map(|document| TextCorpusSnapshotDocument {
                    id: document.id.clone(),
                    language: None,
                    timestamp: None,
                    source: None,
                    provenance: Vec::new(),
                    annotations: Vec::new(),
                    metadata: BTreeMap::new(),
                    total_terms: document.total_terms,
                    term_counts: document.term_counts.clone(),
                })
                .collect(),
            stats: self.stats(),
            vocabulary: self.collection_counts.keys().cloned().collect(),
            document_frequency: self.document_frequency.clone(),
            collection_counts: self.collection_counts.clone(),
        }
    }

    /// Returns term stats.
    pub fn term_stats(&self, limit: usize) -> Vec<CorpusTermStats> {
        let total = self.total_terms.max(1) as f32;
        let mut terms = self
            .collection_counts
            .iter()
            .map(|(term, count)| CorpusTermStats {
                term: term.clone(),
                collection_count: *count,
                document_count: self.document_count(term),
                collection_frequency: *count as f32 / total,
            })
            .collect::<Vec<_>>();
        terms.sort_by(|left, right| {
            right
                .collection_count
                .cmp(&left.collection_count)
                .then_with(|| right.document_count.cmp(&left.document_count))
                .then_with(|| left.term.cmp(&right.term))
        });
        truncate_limit(&mut terms, limit);
        terms
    }

    /// Returns document tfidf.
    pub fn document_tfidf(&self, id: &str, limit: usize) -> Result<Vec<TfIdfTerm>> {
        let document = self
            .document(id)
            .ok_or_else(|| invalid_argument(format!("document `{id}` was not indexed")))?;
        Ok(self.tfidf_for_counts(&document.term_counts, document.total_terms, limit))
    }

    /// Returns text tfidf.
    pub fn text_tfidf(&self, text: &str, limit: usize) -> Vec<TfIdfTerm> {
        let counts = term_counts(text, &self.options);
        let total_terms = counts.values().sum();
        self.tfidf_for_counts(&counts, total_terms, limit)
    }

    /// Returns search.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<DocumentSearchResult>> {
        if limit == 0 {
            return Err(invalid_argument("search limit must be greater than zero"));
        }
        if self.documents.is_empty() {
            return Ok(Vec::new());
        }
        let query_counts = term_counts(query, &self.options);
        if query_counts.is_empty() {
            return Ok(Vec::new());
        }

        let query_weights = self.weight_map(&query_counts, query_counts.values().sum());
        let query_norm = l2_norm(query_weights.values().copied());
        if query_norm <= f32::EPSILON {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        for document in &self.documents {
            let document_weights = self.weight_map(&document.term_counts, document.total_terms);
            let document_norm = l2_norm(document_weights.values().copied());
            if document_norm <= f32::EPSILON {
                continue;
            }

            let mut dot = 0.0;
            let mut matched_terms = 0;
            for (term, query_weight) in &query_weights {
                if let Some(document_weight) = document_weights.get(term) {
                    dot += query_weight * document_weight;
                    matched_terms += 1;
                }
            }
            if matched_terms == 0 {
                continue;
            }
            results.push(DocumentSearchResult {
                id: document.id.clone(),
                score: dot / (query_norm * document_norm),
                matched_terms,
            });
        }

        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.matched_terms.cmp(&left.matched_terms))
                .then_with(|| left.id.cmp(&right.id))
        });
        results.truncate(limit);
        Ok(results)
    }

    /// Returns document count.
    pub fn document_count(&self, term: &str) -> usize {
        self.document_frequency.get(term).copied().unwrap_or(0)
    }

    /// Returns inverse document frequency.
    pub fn inverse_document_frequency(&self, term: &str) -> f32 {
        let documents = self.documents.len() as f32;
        let document_count = self.document_count(term) as f32;
        ((documents + 1.0) / (document_count + 1.0)).ln() + 1.0
    }

    fn tfidf_for_counts(
        &self,
        counts: &BTreeMap<String, usize>,
        total_terms: usize,
        limit: usize,
    ) -> Vec<TfIdfTerm> {
        let denominator = total_terms.max(1) as f32;
        let mut terms = counts
            .iter()
            .map(|(term, count)| {
                let term_frequency = *count as f32 / denominator;
                let inverse_document_frequency = self.inverse_document_frequency(term);
                TfIdfTerm {
                    term: term.clone(),
                    count: *count,
                    term_frequency,
                    inverse_document_frequency,
                    score: term_frequency * inverse_document_frequency,
                    document_count: self.document_count(term),
                }
            })
            .collect::<Vec<_>>();
        terms.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.count.cmp(&left.count))
                .then_with(|| left.term.cmp(&right.term))
        });
        truncate_limit(&mut terms, limit);
        terms
    }

    fn weight_map(
        &self,
        counts: &BTreeMap<String, usize>,
        total_terms: usize,
    ) -> BTreeMap<String, f32> {
        self.tfidf_for_counts(counts, total_terms, 0)
            .into_iter()
            .map(|term| (term.term, term.score))
            .collect()
    }
}

/// Returns terms.
pub fn terms(text: &str, options: &CorpusOptions) -> Vec<String> {
    tokenize(text, &options.processing)
        .into_iter()
        .filter(|token| {
            matches!(
                token.kind,
                TokenKind::Word
                    | TokenKind::Number
                    | TokenKind::Url
                    | TokenKind::Email
                    | TokenKind::Mention
                    | TokenKind::Hashtag
            )
        })
        .map(|token| token.normalized)
        .filter(|term| term.chars().count() >= options.min_term_len)
        .filter(|term| !options.stop_words.contains(term))
        .collect()
}

/// Returns term counts.
pub fn term_counts(text: &str, options: &CorpusOptions) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for term in terms(text, options) {
        *counts.entry(term).or_insert(0) += 1;
    }
    counts
}

/// Returns sparse term vector.
pub fn sparse_term_vector(
    counts: &BTreeMap<String, usize>,
    vocabulary: &[String],
) -> Result<SparseVector> {
    let index_by_term = vocabulary
        .iter()
        .enumerate()
        .map(|(index, term)| (term.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (term, count) in counts {
        if let Some(index) = index_by_term.get(term.as_str()) {
            indices.push(*index);
            values.push(*count as f32);
        }
    }
    SparseVector::new(vocabulary.len().max(1), indices, values)
}

fn bm25_term_counts(text: &str, options: &Bm25Options) -> BTreeMap<String, usize> {
    term_counts(
        text,
        &CorpusOptions {
            min_term_len: options.min_term_len,
            stop_words: options.stop_words.clone(),
            ..CorpusOptions::default()
        },
    )
}

fn text_corpus_document_from_contract(document: &TextDocumentContract) -> TextCorpusDocument {
    let mut metadata = document.attributes.clone();
    if let Some(timestamp) = document.timestamp {
        metadata.insert(
            "timestamp_seconds".to_string(),
            timestamp.seconds().to_string(),
        );
    }
    TextCorpusDocument {
        id: document.id.clone(),
        text: document.text.clone(),
        language: document.language.clone(),
        timestamp: document.timestamp,
        source: document.source.clone(),
        provenance: document.provenance.clone(),
        annotations: document.annotations.clone(),
        metadata,
    }
}

fn text_corpus_document_from_segment_contract(segment: &TextSegmentContract) -> TextCorpusDocument {
    let mut metadata = segment.attributes.clone();
    if let Some(timestamp) = segment.timestamp {
        metadata.insert(
            "timestamp_seconds".to_string(),
            timestamp.seconds().to_string(),
        );
    }
    if let Some(duration_seconds) = segment.duration_seconds {
        metadata.insert("duration_seconds".to_string(), duration_seconds.to_string());
    }
    TextCorpusDocument {
        id: segment
            .document_id()
            .unwrap_or_else(|| segment.segment_index.to_string()),
        text: segment.text.clone(),
        language: segment.language.clone(),
        timestamp: segment.timestamp,
        source: segment.source.clone().or_else(|| {
            (segment.timestamp.is_some() || segment.duration_seconds.is_some()).then(|| {
                TextSourceRef {
                    source_id: segment.stream_id.clone(),
                    source_kind: Some("text_segment".to_string()),
                    uri: None,
                    media_timestamp: segment.timestamp,
                    duration_seconds: segment.duration_seconds,
                }
            })
        }),
        provenance: segment.provenance.clone(),
        annotations: segment.annotations.clone(),
        metadata,
    }
}

fn tfidf_from_snapshot_parts(snapshot: &TextCorpusSnapshot) -> Result<TfIdfCorpus> {
    snapshot.validate()?;
    Ok(TfIdfCorpus {
        options: snapshot.options.clone(),
        documents: snapshot
            .documents
            .iter()
            .map(|document| IndexedDocument {
                id: document.id.clone(),
                term_counts: document.term_counts.clone(),
                total_terms: document.total_terms,
            })
            .collect(),
        document_frequency: snapshot.document_frequency.clone(),
        collection_counts: snapshot.collection_counts.clone(),
        total_terms: snapshot.stats.total_terms,
    })
}

fn validate_document_id(id: &str) -> Result<()> {
    if id.trim().is_empty() {
        return Err(invalid_argument("document id must not be empty"));
    }
    Ok(())
}

fn duplicate_document_id(id: &str) -> DetectError {
    invalid_argument(format!("document id `{id}` already exists"))
}

fn retain_top_counts(counts: &BTreeMap<String, usize>, limit: usize) -> BTreeMap<String, usize> {
    let mut ranked = counts.iter().collect::<Vec<_>>();
    ranked.sort_by(|(left_term, left_count), (right_term, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_term.cmp(right_term))
    });
    ranked
        .into_iter()
        .take(limit)
        .map(|(term, count)| (term.clone(), *count))
        .collect()
}

fn truncate_limit<T>(values: &mut Vec<T>, limit: usize) {
    if limit > 0 {
        values.truncate(limit);
    }
}

fn build_sparse_term_matrix(documents: &[IndexedDocument]) -> Result<SparseTermMatrix> {
    let vocabulary = documents
        .iter()
        .flat_map(|document| document.term_counts.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let index_by_term = vocabulary
        .iter()
        .enumerate()
        .map(|(index, term)| (term.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut entries = Vec::new();
    for (row, document) in documents.iter().enumerate() {
        for (term, count) in &document.term_counts {
            if let Some(col) = index_by_term.get(term.as_str()) {
                entries.push((row, *col, *count as f32));
            }
        }
    }
    let matrix =
        CooMatrix::new(documents.len().max(1), vocabulary.len().max(1), entries)?.to_csr()?;
    Ok(SparseTermMatrix { vocabulary, matrix })
}

fn l2_norm(values: impl IntoIterator<Item = f32>) -> f32 {
    values
        .into_iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
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
    fn text_corpus_from_document_contract_preserves_metadata() {
        let mut document = TextDocumentContract::new("doc-1", "rust cargo crates");
        document.language = Some("en".to_string());
        document
            .attributes
            .insert("source".to_string(), "fixture".to_string());
        document.timestamp = Some(text_core::TimestampContract {
            pts: 1500,
            timebase: text_core::TimebaseContract { num: 1, den: 1000 },
        });

        let corpus =
            TextCorpus::from_document_contracts([document], CorpusOptions::default()).unwrap();
        let indexed = corpus.document("doc-1").unwrap();

        assert_eq!(indexed.language.as_deref(), Some("en"));
        assert_eq!(indexed.metadata.get("source").unwrap(), "fixture");
        assert_eq!(indexed.metadata.get("timestamp_seconds").unwrap(), "1.5");
        assert!(!indexed.metadata.contains_key("language"));
    }

    #[test]
    fn text_corpus_from_segment_contract_uses_canonical_id_and_metadata() {
        let mut segment = TextSegmentContract::new(7, "scene transcript text");
        segment.stream_id = Some("subs".to_string());
        segment.language = Some("en".to_string());
        segment.duration_seconds = Some(2.25);
        segment
            .attributes
            .insert("speaker".to_string(), "A".to_string());
        segment.timestamp = Some(text_core::TimestampContract {
            pts: 2500,
            timebase: text_core::TimebaseContract { num: 1, den: 1000 },
        });

        let corpus =
            TextCorpus::from_segment_contracts([segment], CorpusOptions::default()).unwrap();
        let indexed = corpus.document("subs:7").unwrap();

        assert_eq!(indexed.text, "scene transcript text");
        assert_eq!(indexed.language.as_deref(), Some("en"));
        assert_eq!(indexed.metadata.get("speaker").unwrap(), "A");
        assert_eq!(indexed.metadata.get("timestamp_seconds").unwrap(), "2.5");
        assert_eq!(indexed.metadata.get("duration_seconds").unwrap(), "2.25");
        assert!(!indexed.metadata.contains_key("language"));
    }

    #[test]
    fn text_corpus_segment_contract_without_stream_id_uses_segment_index() {
        let segment = TextSegmentContract::new(3, "unscoped segment");

        let corpus =
            TextCorpus::from_segment_contracts([segment], CorpusOptions::default()).unwrap();

        assert!(corpus.document("3").is_some());
    }

    #[test]
    fn text_corpus_to_tfidf_matches_direct_from_documents() {
        let documents = vec![
            TextCorpusDocument::new("rust", "rust cargo crates ownership"),
            TextCorpusDocument::new("video", "video scenes reports frames"),
        ];
        let corpus = TextCorpus::from_documents(documents, CorpusOptions::default()).unwrap();
        let direct = TfIdfCorpus::from_documents(
            [
                TextDocument::new("rust", "rust cargo crates ownership"),
                TextDocument::new("video", "video scenes reports frames"),
            ],
            CorpusOptions::default(),
        )
        .unwrap();
        let derived = corpus.to_tfidf_corpus().unwrap();

        assert_eq!(derived.documents(), direct.documents());
        assert_eq!(
            derived.search("cargo rust", 1).unwrap(),
            direct.search("cargo rust", 1).unwrap()
        );
    }

    #[test]
    fn text_corpus_replace_document_updates_converted_tfidf() {
        let mut corpus = TextCorpus::from_texts(
            ["rust cargo ownership", "video scene detection"],
            CorpusOptions::default(),
        )
        .unwrap();

        let old = corpus
            .replace_document(TextCorpusDocument::new("doc-1", "cargo compiler build"))
            .unwrap();
        let tfidf = corpus.to_tfidf_corpus().unwrap();

        assert_eq!(old.text, "video scene detection");
        assert_eq!(tfidf.search("compiler cargo", 1).unwrap()[0].id, "doc-1");
        assert!(tfidf.search("scene detection", 5).unwrap().is_empty());
    }

    #[test]
    fn text_corpus_remove_document_removes_converted_document() {
        let mut corpus = TextCorpus::from_texts(
            ["rust cargo ownership", "video scene detection"],
            CorpusOptions::default(),
        )
        .unwrap();

        let removed = corpus.remove_document("doc-0").unwrap();
        let tfidf = corpus.to_tfidf_corpus().unwrap();

        assert_eq!(removed.id, "doc-0");
        assert_eq!(corpus.len(), 1);
        assert!(tfidf.document("doc-0").is_none());
        assert_eq!(tfidf.search("scene detection", 1).unwrap()[0].id, "doc-1");
    }

    #[test]
    fn text_corpus_rejects_duplicate_ids() {
        let documents = vec![
            TextCorpusDocument::new("dup", "rust cargo"),
            TextCorpusDocument::new("dup", "video scene"),
        ];

        let error = TextCorpus::from_documents(documents, CorpusOptions::default()).unwrap_err();

        assert!(
            matches!(error, DetectError::InvalidArgument(message) if message.contains("document id `dup` already exists"))
        );
    }

    #[test]
    fn text_corpus_rejects_empty_ids() {
        let error = TextCorpus::from_documents(
            [TextCorpusDocument::new("  ", "rust cargo")],
            CorpusOptions::default(),
        )
        .unwrap_err();

        assert!(
            matches!(error, DetectError::InvalidArgument(message) if message == "document id must not be empty")
        );
    }

    #[test]
    fn tfidf_snapshot_validates_and_round_trips() {
        let corpus = TfIdfCorpus::from_texts(
            ["rust cargo cargo", "video scene report"],
            CorpusOptions::default(),
        )
        .unwrap();

        let snapshot = corpus.snapshot();
        snapshot.validate().unwrap();
        let json = serde_json::to_string(&snapshot).unwrap();
        let decoded: TextCorpusSnapshot = serde_json::from_str(&json).unwrap();
        let restored = decoded.to_tfidf_corpus().unwrap();

        assert_eq!(restored.documents(), corpus.documents());
        assert_eq!(restored.stats(), corpus.stats());
        assert_eq!(
            restored.search("cargo rust", 1).unwrap(),
            corpus.search("cargo rust", 1).unwrap()
        );
    }

    #[test]
    fn text_corpus_snapshot_preserves_document_metadata() {
        let mut document = TextCorpusDocument::new("doc", "rust cargo");
        document.language = Some("en".to_string());
        document
            .metadata
            .insert("source".to_string(), "contract".to_string());
        let corpus = TextCorpus::from_documents([document], CorpusOptions::default()).unwrap();

        let snapshot = corpus.snapshot().unwrap();

        snapshot.validate().unwrap();
        assert_eq!(snapshot.documents[0].language.as_deref(), Some("en"));
        assert_eq!(
            snapshot.documents[0].metadata.get("source").unwrap(),
            "contract"
        );
    }

    #[test]
    fn snapshot_validation_rejects_inconsistent_counts() {
        let corpus = TfIdfCorpus::from_texts(["rust cargo"], CorpusOptions::default()).unwrap();
        let mut snapshot = corpus.snapshot();
        snapshot.documents[0]
            .term_counts
            .insert("rust".to_string(), 0);

        let error = snapshot.validate().unwrap_err();

        assert!(
            matches!(error, DetectError::InvalidArgument(message) if message.contains("zero count"))
        );
    }

    #[test]
    fn indexes_corpus_statistics() {
        let mut corpus = TfIdfCorpus::default();
        corpus.add_document("a", "rust text text").unwrap();
        corpus.add_document("b", "video text").unwrap();

        let stats = corpus.stats();
        assert_eq!(stats.documents, 2);
        assert_eq!(stats.total_terms, 5);
        assert_eq!(stats.unique_terms, 3);
        assert_eq!(corpus.document_count("text"), 2);
    }

    #[test]
    fn filters_terms_with_processing_options() {
        let mut options = CorpusOptions {
            min_term_len: 4,
            ..CorpusOptions::default()
        };
        options.stop_words.insert("this".to_string());

        let terms = terms("This Rust #AI 2026 contact@example.com", &options);

        assert_eq!(terms, vec!["rust", "2026", "contact@example.com"]);
    }

    #[test]
    fn retains_most_frequent_terms_after_counting() {
        let mut corpus = TfIdfCorpus::new(CorpusOptions {
            max_terms_per_document: Some(2),
            ..CorpusOptions::default()
        });

        corpus
            .add_document("doc", "alpha beta beta gamma gamma gamma")
            .unwrap();
        let document = corpus.document("doc").unwrap();

        assert_eq!(document.total_terms, 5);
        assert_eq!(document.term_counts.get("gamma"), Some(&3));
        assert_eq!(document.term_counts.get("beta"), Some(&2));
        assert_eq!(document.term_counts.get("alpha"), None);
    }

    #[test]
    fn rejects_invalid_document_and_search_inputs() {
        let mut corpus = TfIdfCorpus::default();

        assert!(corpus.add_document("  ", "text").is_err());
        corpus.add_document("doc", "rust cargo").unwrap();
        assert!(corpus.add_document("doc", "duplicate").is_err());
        assert!(corpus.search("rust", 0).is_err());
        assert!(corpus.document_tfidf("missing", 10).is_err());
    }

    #[test]
    fn ranks_tfidf_terms() {
        let mut corpus = TfIdfCorpus::default();
        corpus.add_document("a", "rust rust text").unwrap();
        corpus.add_document("b", "text video").unwrap();

        let terms = corpus.document_tfidf("a", 2).unwrap();
        assert_eq!(terms[0].term, "rust");
        assert!(terms[0].score > terms[1].score);
    }

    #[test]
    fn searches_documents_with_tfidf_cosine() {
        let mut corpus = TfIdfCorpus::default();
        corpus
            .add_document("rust", "rust crates cargo ownership")
            .unwrap();
        corpus
            .add_document("video", "frames ffmpeg scene detection")
            .unwrap();

        let results = corpus.search("cargo crate rust", 1).unwrap();
        assert_eq!(results[0].id, "rust");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn bm25_ranks_documents_and_rejects_duplicates() {
        let mut corpus = Bm25Corpus::default();
        corpus
            .add_document("rust", "rust cargo crates rust ownership")
            .unwrap();
        corpus
            .add_document("video", "frames ffmpeg scene detection")
            .unwrap();

        assert!(corpus.add_document("rust", "duplicate").is_err());
        let results = corpus.search("rust cargo", 2).unwrap();
        assert_eq!(results[0].id, "rust");
        assert!(results[0].score > 0.0);
        assert!(corpus.search("!!!", 5).unwrap().is_empty());
        assert!(corpus.search("rust", 0).is_err());
    }

    #[test]
    fn bm25_honors_text_document_and_filters() {
        let mut stop_words = BTreeSet::new();
        stop_words.insert("the".to_string());
        let mut corpus = Bm25Corpus::new(Bm25Options {
            min_term_len: 4,
            stop_words,
            ..Bm25Options::default()
        });
        let document = TextDocument::new("doc", "the reliable reliable rust code");
        corpus.add_text_document(&document).unwrap();
        assert_eq!(corpus.documents()[0].term_counts.get("the"), None);
        assert_eq!(corpus.search("reliable", 1).unwrap()[0].id, "doc");
    }

    #[test]
    fn tfidf_from_texts_matches_incremental_indexing_and_assigns_ids() {
        let texts = ["rust text text", "video text", "audio scene transcript"];

        let from_texts = TfIdfCorpus::from_texts(texts, CorpusOptions::default()).unwrap();

        let mut incremental = TfIdfCorpus::default();
        incremental.add_document("doc-0", "rust text text").unwrap();
        incremental.add_document("doc-1", "video text").unwrap();
        incremental
            .add_document("doc-2", "audio scene transcript")
            .unwrap();

        assert_eq!(from_texts.stats(), incremental.stats());
        assert_eq!(from_texts.documents(), incremental.documents());
        assert_eq!(from_texts.documents()[0].id, "doc-0");
        assert_eq!(from_texts.documents()[1].id, "doc-1");
        assert_eq!(from_texts.documents()[2].id, "doc-2");
    }

    #[test]
    fn tfidf_from_documents_preserves_ids_and_supports_search() {
        let documents = vec![
            TextDocument::new("rust", "rust cargo crates ownership"),
            TextDocument::new("video", "video scenes reports frames"),
        ];

        let corpus = TfIdfCorpus::from_documents(documents, CorpusOptions::default()).unwrap();

        assert_eq!(corpus.document("rust").unwrap().id, "rust");
        assert_eq!(corpus.document("video").unwrap().unique_terms(), 4);
        assert_eq!(corpus.search("cargo rust", 1).unwrap()[0].id, "rust");
        assert!(!corpus.document_tfidf("video", 3).unwrap().is_empty());
    }

    #[test]
    fn tfidf_indexes_text_segments_incrementally() {
        let mut corpus = TfIdfCorpus::default();
        let segments = vec![
            TextSegment {
                segment_index: 0,
                timestamp: None,
                text: "rust cargo ownership",
                language: Some("en"),
                is_final: true,
            },
            TextSegment {
                segment_index: 1,
                timestamp: None,
                text: "video scene detection",
                language: Some("en"),
                is_final: true,
            },
            TextSegment {
                segment_index: 2,
                timestamp: None,
                text: "cargo crates and compiler",
                language: Some("en"),
                is_final: true,
            },
        ];
        for segment in &segments {
            corpus.add_text_segment("subs", segment).unwrap();
        }

        assert_eq!(corpus.documents()[0].id, "subs:0");
        assert_eq!(corpus.documents()[1].id, "subs:1");
        assert_eq!(corpus.documents()[2].id, "subs:2");
        assert_eq!(corpus.search("compiler cargo", 1).unwrap()[0].id, "subs:2");
        assert!(matches!(
            corpus.add_text_segment("subs", &segments[0]),
            Err(DetectError::InvalidArgument(message))
                if message.contains("document id `subs:0` already exists")
        ));
    }

    #[test]
    fn tfidf_from_text_segments_validates_stream_id() {
        let segments = vec![TextSegment {
            segment_index: 0,
            timestamp: None,
            text: "rust cargo",
            language: None,
            is_final: true,
        }];

        let error =
            TfIdfCorpus::from_text_segments("   ", segments, CorpusOptions::default()).unwrap_err();
        assert!(
            matches!(error, DetectError::InvalidArgument(message) if message == "stream id must not be empty")
        );
    }

    #[test]
    fn tfidf_batch_construction_honors_options() {
        let mut stop_words = BTreeSet::new();
        stop_words.insert("skip".to_string());
        let corpus = TfIdfCorpus::from_texts(
            ["skip alpha beta beta gamma gamma gamma"],
            CorpusOptions {
                min_term_len: 5,
                stop_words,
                max_terms_per_document: Some(2),
                ..CorpusOptions::default()
            },
        )
        .unwrap();

        let document = &corpus.documents()[0];
        assert_eq!(document.id, "doc-0");
        assert_eq!(document.total_terms, 4);
        assert_eq!(document.term_counts.get("gamma"), Some(&3));
        assert_eq!(document.term_counts.get("alpha"), Some(&1));
        assert_eq!(document.term_counts.get("beta"), None);
        assert_eq!(document.term_counts.get("skip"), None);
    }

    #[test]
    fn tfidf_batch_construction_accepts_empty_inputs_and_text() {
        let empty =
            TfIdfCorpus::from_texts(std::iter::empty::<&str>(), CorpusOptions::default()).unwrap();
        assert!(empty.is_empty());

        let corpus = TfIdfCorpus::from_texts([""], CorpusOptions::default()).unwrap();
        assert_eq!(corpus.len(), 1);
        assert_eq!(corpus.documents()[0].total_terms, 0);
    }

    #[test]
    fn tfidf_from_documents_reuses_duplicate_id_validation() {
        let documents = vec![
            TextDocument::new("dup", "rust cargo"),
            TextDocument::new("dup", "video scene"),
        ];

        let error = TfIdfCorpus::from_documents(documents, CorpusOptions::default()).unwrap_err();
        assert!(
            matches!(error, DetectError::InvalidArgument(message) if message.contains("document id `dup` already exists"))
        );
    }

    #[test]
    fn bm25_from_texts_builds_searchable_corpus_and_assigns_ids() {
        let corpus = Bm25Corpus::from_texts(
            [
                "rust cargo crates rust ownership",
                "frames ffmpeg scene detection",
            ],
            Bm25Options::default(),
        )
        .unwrap();

        assert_eq!(corpus.documents()[0].id, "doc-0");
        assert_eq!(corpus.documents()[1].id, "doc-1");
        assert_eq!(corpus.search("rust cargo", 1).unwrap()[0].id, "doc-0");
    }

    #[test]
    fn bm25_from_documents_preserves_ids_and_matches_incremental_behavior() {
        let documents = vec![
            TextDocument::new("rust", "rust cargo crates rust ownership"),
            TextDocument::new("video", "frames ffmpeg scene detection"),
        ];
        let from_documents =
            Bm25Corpus::from_documents(documents.clone(), Bm25Options::default()).unwrap();

        let mut incremental = Bm25Corpus::default();
        for document in documents {
            incremental.add_text_document(&document).unwrap();
        }

        assert_eq!(from_documents.documents(), incremental.documents());
        assert_eq!(
            from_documents.search("rust cargo", 1).unwrap()[0].id,
            "rust"
        );
    }

    #[test]
    fn bm25_indexes_text_segments_and_searches_generated_ids() {
        let segments = vec![
            TextSegment {
                segment_index: 0,
                timestamp: None,
                text: "guten tag berlin",
                language: Some("de"),
                is_final: true,
            },
            TextSegment {
                segment_index: 1,
                timestamp: None,
                text: "rust cargo crates ownership",
                language: Some("en"),
                is_final: true,
            },
            TextSegment {
                segment_index: 2,
                timestamp: None,
                text: "cargo build pipeline",
                language: Some("en"),
                is_final: true,
            },
        ];
        let corpus =
            Bm25Corpus::from_text_segments("subs", segments.clone(), Bm25Options::default())
                .unwrap();

        assert_eq!(corpus.documents()[0].id, "subs:0");
        assert_eq!(corpus.documents()[1].id, "subs:1");
        assert_eq!(corpus.documents()[2].id, "subs:2");
        assert_eq!(corpus.search("cargo ownership", 1).unwrap()[0].id, "subs:1");

        let mut incremental = Bm25Corpus::default();
        for segment in &segments {
            incremental.add_text_segment("subs", segment).unwrap();
        }
        assert_eq!(corpus.documents(), incremental.documents());
    }

    #[test]
    fn bm25_text_segment_ingestion_rejects_empty_stream_id() {
        let segment = TextSegment {
            segment_index: 0,
            timestamp: None,
            text: "rust cargo",
            language: None,
            is_final: true,
        };
        let error = Bm25Corpus::default()
            .add_text_segment(" ", &segment)
            .unwrap_err();
        assert!(
            matches!(error, DetectError::InvalidArgument(message) if message == "stream id must not be empty")
        );
    }

    #[test]
    fn bm25_batch_construction_honors_options_and_empty_inputs() {
        let mut stop_words = BTreeSet::new();
        stop_words.insert("skip".to_string());
        let corpus = Bm25Corpus::from_texts(
            ["skip beta beta gamma gamma"],
            Bm25Options {
                min_term_len: 5,
                stop_words,
                ..Bm25Options::default()
            },
        )
        .unwrap();

        assert_eq!(corpus.documents()[0].term_counts.get("skip"), None);
        assert_eq!(corpus.documents()[0].term_counts.get("beta"), None);
        assert_eq!(corpus.documents()[0].term_counts.get("gamma"), Some(&2));

        let empty = Bm25Corpus::from_documents(
            std::iter::empty::<TextDocument<'static>>(),
            Bm25Options::default(),
        )
        .unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn bm25_from_documents_reuses_duplicate_id_validation() {
        let documents = vec![
            TextDocument::new("dup", "rust cargo"),
            TextDocument::new("dup", "video scene"),
        ];

        let error = Bm25Corpus::from_documents(documents, Bm25Options::default()).unwrap_err();
        assert!(
            matches!(error, DetectError::InvalidArgument(message) if message.contains("document id `dup` already exists"))
        );
    }

    #[test]
    fn sparse_term_outputs_round_trip() {
        let corpus =
            TfIdfCorpus::from_texts(["rust cargo", "cargo build"], CorpusOptions::default())
                .unwrap();
        let sparse = corpus.sparse_term_matrix().unwrap();
        assert_eq!(sparse.matrix.rows(), 2);
        assert!(sparse.vocabulary.iter().any(|term| term == "cargo"));
        let vector =
            sparse_term_vector(&corpus.documents()[0].term_counts, &sparse.vocabulary).unwrap();
        assert_eq!(
            vector
                .to_dense()
                .iter()
                .filter(|value| **value > 0.0)
                .count(),
            2
        );
    }
}
