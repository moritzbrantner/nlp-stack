//! The in-memory hybrid retrieval index and its query-time ranking.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use text_core::{tokenize, Result as CoreResult, TextProcessingOptions};
use text_embeddings::{EmbeddingModelInfo, TextEmbedderBackend};
use text_lexical::{Bm25Corpus, Bm25Options, CorpusOptions};
use vector_analysis_index::{
    SerializableVectorRecord, VectorRecord, VectorRecordMetadata, VectorSearchIndex,
};

use crate::ingestion::{chunk_document, validate_document, validate_ingestion_options};
use crate::query::{
    normalize_query, requires_post_filter, search_filter_to_vector_filter, validate_query,
};
use crate::rerank::rerank_search_results;
use crate::{
    invalid_argument, DocumentChunk, IngestionOptions, RerankExecutionContext, SearchDocument,
    SearchFilter, SearchQuery, SearchResult,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for ingest report.
pub struct IngestReport {
    /// The documents received value.
    pub documents_received: usize,
    /// The documents replaced value.
    pub documents_replaced: usize,
    /// The documents skipped value.
    pub documents_skipped: usize,
    /// The chunks indexed value.
    pub chunks_indexed: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for retrieval index.
pub struct RetrievalIndex<B> {
    embedder: B,
    corpus_options: CorpusOptions,
    bm25_options: Bm25Options,
    pub(crate) chunks: BTreeMap<String, DocumentChunk>,
    raw_text_by_chunk_id: BTreeMap<String, String>,
    document_chunks: BTreeMap<String, Vec<String>>,
    vectors: VectorSearchIndex,
    bm25: Bm25Corpus,
}

impl<B: TextEmbedderBackend> RetrievalIndex<B> {
    /// Creates a new value.
    pub fn new(embedder: B) -> Self {
        Self::with_options(embedder, CorpusOptions::default(), Bm25Options::default())
    }

    /// Returns this value with options.
    pub fn with_options(
        embedder: B,
        corpus_options: CorpusOptions,
        bm25_options: Bm25Options,
    ) -> Self {
        Self {
            embedder,
            bm25: Bm25Corpus::new(bm25_options.clone()),
            corpus_options,
            bm25_options,
            chunks: BTreeMap::new(),
            raw_text_by_chunk_id: BTreeMap::new(),
            document_chunks: BTreeMap::new(),
            vectors: VectorSearchIndex::new(),
        }
    }

    /// Builds this value from parts.
    pub fn from_parts(
        embedder: B,
        corpus_options: CorpusOptions,
        bm25_options: Bm25Options,
        chunks: Vec<DocumentChunk>,
        raw_text_by_chunk_id: BTreeMap<String, String>,
        vector_records: Vec<SerializableVectorRecord>,
    ) -> CoreResult<Self> {
        let mut index = Self::with_options(embedder, corpus_options, bm25_options);
        for chunk in chunks {
            index.insert_chunk_state(chunk);
        }
        index.raw_text_by_chunk_id = raw_text_by_chunk_id;
        index.vectors = VectorSearchIndex::import_records(vector_records)?;
        index.rebuild_bm25()?;
        index.validate_vector_state()?;
        Ok(index)
    }

    /// Returns embedder.
    pub fn embedder(&self) -> &B {
        &self.embedder
    }

    /// Returns embedder info.
    pub fn embedder_info(&self) -> EmbeddingModelInfo {
        self.embedder.model_info()
    }

    /// Returns corpus options.
    pub fn corpus_options(&self) -> &CorpusOptions {
        &self.corpus_options
    }

    /// Returns bm25 options.
    pub fn bm25_options(&self) -> &Bm25Options {
        &self.bm25_options
    }

    /// Returns chunks.
    pub fn chunks(&self) -> Vec<&DocumentChunk> {
        self.chunks.values().collect()
    }

    /// Returns chunks iter.
    pub fn chunks_iter(&self) -> impl Iterator<Item = &DocumentChunk> {
        self.chunks.values()
    }

    /// Returns chunk.
    pub fn chunk(&self, chunk_id: &str) -> Option<&DocumentChunk> {
        self.chunks.get(chunk_id)
    }

    /// Returns raw text.
    pub fn raw_text(&self, chunk_id: &str) -> Option<&str> {
        self.raw_text_by_chunk_id.get(chunk_id).map(String::as_str)
    }

    /// Returns vector records.
    pub fn vector_records(&self) -> Vec<SerializableVectorRecord> {
        self.vectors.export_records()
    }

    /// Returns ingest documents.
    pub fn ingest_documents(
        &mut self,
        docs: &[SearchDocument],
        options: &IngestionOptions,
    ) -> CoreResult<IngestReport> {
        validate_ingestion_options(options)?;

        let mut replaced = 0;
        let mut skipped = 0;
        for document in docs {
            validate_document(document)?;
            if self.document_chunks.contains_key(&document.id) {
                self.remove_document(&document.id);
                replaced += 1;
            }

            let chunks = chunk_document(document, options, &self.corpus_options.processing)?;
            if chunks.is_empty() {
                skipped += 1;
                continue;
            }

            for chunk in chunks {
                if options.store_raw_text {
                    self.raw_text_by_chunk_id
                        .insert(chunk.chunk_id.clone(), chunk.text.clone());
                }
                self.insert_chunk_state(chunk);
            }
        }

        self.rebuild_indices()?;
        Ok(IngestReport {
            documents_received: docs.len(),
            documents_replaced: replaced,
            documents_skipped: skipped,
            chunks_indexed: self.chunks.len(),
        })
    }

    /// Returns search.
    pub fn search(&self, query: &SearchQuery) -> CoreResult<Vec<SearchResult>> {
        self.search_with_context(query, &mut RerankExecutionContext::default())
    }

    /// Returns search, optionally reranking the initial candidates with context.
    pub fn search_with_context(
        &self,
        query: &SearchQuery,
        context: &mut RerankExecutionContext<'_>,
    ) -> CoreResult<Vec<SearchResult>> {
        validate_query(query)?;
        self.ensure_non_empty()?;

        let normalized_query = normalize_query(&query.text, &self.corpus_options.processing)?;
        let filter = query.filter.as_ref();
        let overfetch = query
            .hybrid
            .rerank_window
            .max(query.top_k)
            .max(query.top_k * 4);
        let post_filter_limit = if filter.is_some_and(requires_post_filter) {
            self.chunks.len()
        } else {
            overfetch
        };

        let semantic_hits = if query.hybrid.semantic_weight > f32::EPSILON {
            let query_vector = self.embedder.embed_text(&normalized_query)?;
            self.validate_query_dimensions(query_vector.dimensions())?;
            self.vectors.search_filtered(
                query_vector.as_slice(),
                post_filter_limit,
                filter.map(search_filter_to_vector_filter).as_ref(),
            )?
        } else {
            Vec::new()
        };
        let lexical_hits = if query.hybrid.lexical_weight > f32::EPSILON {
            self.bm25.search(&normalized_query, post_filter_limit)?
        } else {
            Vec::new()
        };

        let semantic_scores = semantic_hits
            .iter()
            .filter(|hit| self.matches_filter(hit.id.as_str(), filter))
            .map(|hit| (hit.id.as_str().to_string(), hit.score))
            .collect::<BTreeMap<_, _>>();
        let lexical_scores = lexical_hits
            .into_iter()
            .filter(|hit| self.matches_filter(&hit.id, filter))
            .map(|hit| (hit.id, hit.score))
            .collect::<BTreeMap<_, _>>();

        let normalized_semantic = normalize_scores(&semantic_scores);
        let normalized_lexical = normalize_scores(&lexical_scores);

        let mut merged_ids = normalized_semantic.keys().cloned().collect::<BTreeSet<_>>();
        merged_ids.extend(normalized_lexical.keys().cloned());

        let mut results = merged_ids
            .into_iter()
            .filter_map(|chunk_id| {
                let chunk = self.chunks.get(&chunk_id)?;
                let semantic_score = normalized_semantic.get(&chunk_id).copied().unwrap_or(0.0);
                let lexical_score = normalized_lexical.get(&chunk_id).copied().unwrap_or(0.0);
                let score = semantic_score * query.hybrid.semantic_weight
                    + lexical_score * query.hybrid.lexical_weight;
                Some(SearchResult {
                    chunk_id: chunk.chunk_id.clone(),
                    document_id: chunk.document_id.clone(),
                    score,
                    semantic_score,
                    lexical_score,
                    snippet: self.build_snippet(chunk, &normalized_query),
                    metadata: chunk.metadata.clone(),
                })
            })
            .collect::<Vec<_>>();

        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.semantic_score.total_cmp(&left.semantic_score))
                .then_with(|| right.lexical_score.total_cmp(&left.lexical_score))
                .then_with(|| left.chunk_id.cmp(&right.chunk_id))
        });
        if query.hybrid.rerank {
            rerank_search_results(&query.text, &mut results, query.top_k, context)?;
        } else {
            results.truncate(query.top_k);
        }
        Ok(results)
    }

    /// Returns related chunks.
    pub fn related_chunks(&self, chunk_id: &str, top_k: usize) -> CoreResult<Vec<SearchResult>> {
        if top_k == 0 {
            return Err(invalid_argument("search limit must be greater than zero"));
        }
        self.ensure_non_empty()?;

        let query_record = self
            .vectors
            .records()
            .iter()
            .find(|record| record.id == chunk_id)
            .ok_or_else(|| invalid_argument(format!("chunk `{chunk_id}` was not indexed")))?;
        let hits = self
            .vectors
            .search_filtered(query_record.vector.as_slice(), top_k + 1, None)?;

        let mut results = hits
            .into_iter()
            .filter(|hit| hit.id.as_str() != chunk_id)
            .filter_map(|hit| {
                let chunk = self.chunks.get(hit.id.as_str())?;
                Some(SearchResult {
                    chunk_id: chunk.chunk_id.clone(),
                    document_id: chunk.document_id.clone(),
                    score: hit.score,
                    semantic_score: hit.score,
                    lexical_score: 0.0,
                    snippet: self.build_snippet(chunk, &chunk.text),
                    metadata: chunk.metadata.clone(),
                })
            })
            .collect::<Vec<_>>();
        results.truncate(top_k);
        Ok(results)
    }

    fn insert_chunk_state(&mut self, chunk: DocumentChunk) {
        self.document_chunks
            .entry(chunk.document_id.clone())
            .or_default()
            .push(chunk.chunk_id.clone());
        self.chunks.insert(chunk.chunk_id.clone(), chunk);
    }

    fn remove_document(&mut self, document_id: &str) {
        if let Some(chunk_ids) = self.document_chunks.remove(document_id) {
            for chunk_id in chunk_ids {
                self.chunks.remove(&chunk_id);
                self.raw_text_by_chunk_id.remove(&chunk_id);
            }
        }
    }

    fn rebuild_indices(&mut self) -> CoreResult<()> {
        self.rebuild_bm25()?;
        if self.chunks.is_empty() {
            self.vectors = VectorSearchIndex::new();
            return Ok(());
        }

        let texts = self
            .chunks
            .values()
            .map(|chunk| {
                self.raw_text_by_chunk_id
                    .get(&chunk.chunk_id)
                    .map(String::as_str)
                    .unwrap_or(chunk.text.as_str())
            })
            .collect::<Vec<_>>();
        let vectors = self.embedder.embed_batch(&texts)?;
        if vectors.len() != self.chunks.len() {
            return Err(invalid_argument(
                "embedder returned a different number of vectors than input texts",
            ));
        }
        self.validate_embedded_dimensions(vectors.iter().map(|vector| vector.dimensions()))?;

        let mut index = VectorSearchIndex::new();
        for (chunk, vector) in self.chunks.values().zip(vectors) {
            index.add(VectorRecord::with_payload(
                chunk.chunk_id.clone(),
                vector,
                VectorRecordMetadata {
                    tags: metadata_tags(&chunk.metadata),
                    metadata: chunk.metadata.clone(),
                },
            ))?;
        }
        self.vectors = index;
        Ok(())
    }

    fn rebuild_bm25(&mut self) -> CoreResult<()> {
        let mut bm25 = Bm25Corpus::new(self.bm25_options.clone());
        for chunk in self.chunks.values() {
            bm25.add_document(chunk.chunk_id.clone(), &chunk.text)?;
        }
        self.bm25 = bm25;
        Ok(())
    }

    fn validate_embedded_dimensions(
        &self,
        vector_dimensions: impl IntoIterator<Item = usize>,
    ) -> CoreResult<()> {
        let expected = self.embedder.model_info().dimensions;
        if expected == 0 {
            return Ok(());
        }
        if vector_dimensions
            .into_iter()
            .any(|dimensions| dimensions != expected)
        {
            return Err(invalid_argument(format!(
                "embedder produced vectors that did not match advertised dimensions {expected}"
            )));
        }
        Ok(())
    }

    fn validate_vector_state(&self) -> CoreResult<()> {
        let chunk_ids = self.chunks.keys().cloned().collect::<BTreeSet<_>>();
        let vector_ids = self
            .vectors
            .records()
            .iter()
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>();
        if chunk_ids != vector_ids {
            return Err(invalid_argument(
                "persisted vector records and chunk catalog do not describe the same ids",
            ));
        }
        if let Some(dimensions) = self.vectors.dimensions() {
            let expected = self.embedder.model_info().dimensions;
            if expected > 0 && expected != dimensions {
                return Err(invalid_argument(format!(
                    "embedder dimensions {expected} did not match persisted index dimensions {dimensions}"
                )));
            }
        }
        Ok(())
    }

    fn validate_query_dimensions(&self, dimensions: usize) -> CoreResult<()> {
        if let Some(index_dimensions) = self.vectors.dimensions() {
            if dimensions != index_dimensions {
                return Err(invalid_argument(format!(
                    "query vector dimensions {dimensions} did not match index dimensions {index_dimensions}"
                )));
            }
        }
        Ok(())
    }

    fn matches_filter(&self, chunk_id: &str, filter: Option<&SearchFilter>) -> bool {
        let Some(filter) = filter else {
            return true;
        };
        let Some(chunk) = self.chunks.get(chunk_id) else {
            return false;
        };
        if !filter.document_ids.is_empty() && !filter.document_ids.contains(&chunk.document_id) {
            return false;
        }
        if !filter.metadata_contains.iter().all(|(key, needle)| {
            chunk
                .metadata
                .get(key)
                .is_some_and(|value| value.contains(needle))
        }) {
            return false;
        }
        let tags = metadata_tags(&chunk.metadata);
        filter
            .required_tags
            .iter()
            .all(|tag| tags.iter().any(|candidate| candidate == tag))
            && filter
                .metadata_equals
                .iter()
                .all(|(key, value)| chunk.metadata.get(key) == Some(value))
    }

    fn build_snippet(&self, chunk: &DocumentChunk, normalized_query: &str) -> String {
        let text = self
            .raw_text_by_chunk_id
            .get(&chunk.chunk_id)
            .map(String::as_str)
            .unwrap_or(chunk.text.as_str());
        build_snippet(text, normalized_query)
    }

    fn ensure_non_empty(&self) -> CoreResult<()> {
        if self.chunks.is_empty() {
            return Err(invalid_argument("search index is empty"));
        }
        Ok(())
    }
}

fn metadata_tags(metadata: &BTreeMap<String, String>) -> Vec<String> {
    metadata
        .get("tags")
        .map(|tags| {
            tags.split([',', ';'])
                .flat_map(|group| group.split_whitespace())
                .map(str::trim)
                .filter(|tag| !tag.is_empty())
                .map(ToString::to_string)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn normalize_scores(scores: &BTreeMap<String, f32>) -> BTreeMap<String, f32> {
    if scores.is_empty() {
        return BTreeMap::new();
    }
    let min = scores.values().copied().fold(f32::INFINITY, f32::min);
    let max = scores.values().copied().fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() <= f32::EPSILON {
        return scores
            .keys()
            .cloned()
            .map(|id| (id, 1.0))
            .collect::<BTreeMap<_, _>>();
    }
    scores
        .iter()
        .map(|(id, score)| (id.clone(), (score - min) / (max - min)))
        .collect()
}

fn build_snippet(text: &str, normalized_query: &str) -> String {
    let terms = tokenize(normalized_query, &TextProcessingOptions::default())
        .into_iter()
        .map(|token| token.normalized)
        .collect::<Vec<_>>();
    let text_lower = text.to_lowercase();
    let match_start = terms.iter().filter_map(|term| text_lower.find(term)).min();

    let snippet = match match_start {
        Some(byte_index) => {
            let start = text[..byte_index]
                .char_indices()
                .rev()
                .nth(60)
                .map(|(index, _)| index)
                .unwrap_or(0);
            let end = text[byte_index..]
                .char_indices()
                .nth(160)
                .map(|(index, _)| byte_index + index)
                .unwrap_or(text.len());
            text[start..end].trim()
        }
        None => text
            .char_indices()
            .nth(160)
            .map(|(index, _)| &text[..index])
            .unwrap_or(text)
            .trim(),
    };
    snippet.to_string()
}
