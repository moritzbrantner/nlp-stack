//! Deterministic test fixtures. Nothing here is a semantic embedding model.
//!
//! [`HashedFixtureEmbedder`] hashes term counts into a fixed number of buckets. It is
//! stable and dependency-free, which makes it useful for tests that need
//! reproducible vectors, but its similarities are lexical overlap, not meaning.
//! Production callers must supply a real [`TextEmbeddingBackend`].

use std::collections::{BTreeMap, BTreeSet};

use math_sparse_data::SparseVector;
use text_core::{segment_document_id, AnnotationProvenance, Result, TextDocument, TextSegment};
use text_lexical::{term_counts, CorpusOptions, TfIdfCorpus};
use vector_analysis_index::{SearchConfig, VectorRecord, VectorSearchIndex};

use crate::{
    invalid_argument, validate_stream_id, DenseVector, SemanticMatch, SentenceEmbedder,
    TextEmbeddingBackend, TextEmbeddingBackendKind, TextEmbeddingConfig, TextEmbeddingMetadata,
};

#[derive(Debug, Clone, Default, PartialEq)]
/// Deterministic feature-hashing embedder for tests and fixtures.
///
/// Not a semantic model: it hashes term counts into buckets. Never use it as a
/// production default.
pub struct HashedFixtureEmbedder {
    /// The config value.
    pub config: TextEmbeddingConfig,
    /// The corpus options value.
    pub corpus_options: CorpusOptions,
}

impl HashedFixtureEmbedder {
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

impl TextEmbeddingBackend for HashedFixtureEmbedder {
    fn embed_text(&self, text: &str) -> Result<DenseVector> {
        HashedFixtureEmbedder::embed_text(self, text)
    }

    fn embed_document(&self, document: &TextDocument<'_>) -> Result<DenseVector> {
        HashedFixtureEmbedder::embed_document(self, document)
    }

    fn metadata(&self) -> TextEmbeddingMetadata {
        TextEmbeddingMetadata {
            backend: TextEmbeddingBackendKind::HashedFixture,
            provenance: AnnotationProvenance::Heuristic,
            model_name: Some("hashed-fixture-embedder".to_string()),
            dimensions: Some(self.config.dimensions),
        }
    }
}

impl SentenceEmbedder for HashedFixtureEmbedder {}

#[derive(Debug, Clone, PartialEq)]
/// TF-IDF index over [`HashedFixtureEmbedder`] vectors, for tests and fixtures.
pub struct HashedFixtureTextIndex {
    embedder: HashedFixtureEmbedder,
    corpus: TfIdfCorpus,
    pub(crate) vectors: VectorSearchIndex,
}

impl HashedFixtureTextIndex {
    /// Creates a new value.
    pub fn new(embedder: HashedFixtureEmbedder) -> Self {
        Self {
            corpus: TfIdfCorpus::new(embedder.corpus_options.clone()),
            embedder,
            vectors: VectorSearchIndex::new(),
        }
    }

    /// Builds this value from documents.
    pub fn from_documents<'a, I>(embedder: HashedFixtureEmbedder, documents: I) -> Result<Self>
    where
        I: IntoIterator<Item = TextDocument<'a>>,
    {
        let mut index = Self::new(embedder);
        index.add_documents(documents)?;
        Ok(index)
    }

    /// Returns embedder.
    pub fn embedder(&self) -> &HashedFixtureEmbedder {
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

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
