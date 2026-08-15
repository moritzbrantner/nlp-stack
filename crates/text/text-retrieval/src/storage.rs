#![doc = include_str!("../README.md")]

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use text_embeddings::{EmbeddingModelInfo, TextEmbedderBackend};
use thiserror::Error;
use vector_analysis_index::SerializableVectorRecord;

use crate::{DocumentChunk, RetrievalIndex};

#[derive(Debug, Error)]
/// Variants describing storage error.
pub enum StorageError {
    #[error("I/O error: {0}")]
    /// The I/O variant.
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    /// The JSON variant.
    Json(String),
    #[error("invalid manifest: {0}")]
    /// The invalid manifest variant.
    InvalidManifest(String),
    #[error("invalid retrieval state: {0}")]
    /// The invalid state variant.
    InvalidState(String),
}

/// Type alias for result.
pub type Result<T> = std::result::Result<T, StorageError>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for retrieval file.
pub struct RetrievalFile {
    /// Filesystem path for this value.
    pub path: String,
    /// The records value.
    pub records: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for retrieval manifest.
pub struct RetrievalManifest {
    /// The schema version value.
    pub schema_version: u32,
    /// The chunk count value.
    pub chunk_count: u64,
    /// The vector count value.
    pub vector_count: u64,
    /// The dimensions value.
    pub dimensions: Option<usize>,
    /// The embedder value.
    pub embedder: EmbeddingModelInfo,
    /// The chunks file value.
    pub chunks_file: RetrievalFile,
    /// The vectors file value.
    pub vectors_file: RetrievalFile,
    /// The corpus file value.
    pub corpus_file: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for persisted corpus metadata.
pub struct PersistedCorpusMetadata {
    /// The corpus options value.
    pub corpus_options: text_lexical::CorpusOptions,
    /// The bm25 options value.
    pub bm25_options: text_lexical::Bm25Options,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for persisted chunk record.
pub struct PersistedChunkRecord {
    /// The chunk value.
    pub chunk: DocumentChunk,
    /// The raw text value.
    pub raw_text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for persisted search index.
pub struct PersistedSearchIndex {
    /// The manifest value.
    pub manifest: RetrievalManifest,
    /// The corpus value.
    pub corpus: PersistedCorpusMetadata,
    /// The chunks value.
    pub chunks: Vec<PersistedChunkRecord>,
    /// The vectors value.
    pub vectors: Vec<SerializableVectorRecord>,
}

impl PersistedSearchIndex {
    /// Builds this value from index.
    pub fn from_index<B: TextEmbedderBackend>(index: &RetrievalIndex<B>) -> Self {
        let chunks = index
            .chunks_iter()
            .map(|chunk| PersistedChunkRecord {
                chunk: chunk.clone(),
                raw_text: index.raw_text(&chunk.chunk_id).map(ToString::to_string),
            })
            .collect::<Vec<_>>();
        let vectors = index.vector_records();
        let embedder = index.embedder_info();

        Self {
            manifest: RetrievalManifest {
                schema_version: 1,
                chunk_count: chunks.len() as u64,
                vector_count: vectors.len() as u64,
                dimensions: vectors.first().map(|record| record.vector.len()),
                embedder,
                chunks_file: RetrievalFile {
                    path: "chunks.jsonl".to_string(),
                    records: chunks.len() as u64,
                },
                vectors_file: RetrievalFile {
                    path: "vectors.jsonl".to_string(),
                    records: vectors.len() as u64,
                },
                corpus_file: "corpus.json".to_string(),
            },
            corpus: PersistedCorpusMetadata {
                corpus_options: index.corpus_options().clone(),
                bm25_options: index.bm25_options().clone(),
            },
            chunks,
            vectors,
        }
    }

    /// Returns save to path.
    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        fs::create_dir_all(path)?;
        write_json(path.join("manifest.json"), &self.manifest)?;
        write_json(path.join(&self.manifest.corpus_file), &self.corpus)?;
        write_jsonl(path.join(&self.manifest.chunks_file.path), &self.chunks)?;
        write_jsonl(path.join(&self.manifest.vectors_file.path), &self.vectors)?;
        Ok(())
    }

    /// Returns load from path.
    pub fn load_from_path(path: &Path) -> Result<Self> {
        let manifest = read_json::<RetrievalManifest>(path.join("manifest.json"))?;
        let corpus = read_json::<PersistedCorpusMetadata>(path.join(&manifest.corpus_file))?;
        let chunks = read_jsonl::<PersistedChunkRecord>(path.join(&manifest.chunks_file.path))?;
        let vectors =
            read_jsonl::<SerializableVectorRecord>(path.join(&manifest.vectors_file.path))?;

        if chunks.len() as u64 != manifest.chunk_count {
            return Err(StorageError::InvalidManifest(format!(
                "manifest expected {} chunks, loaded {}",
                manifest.chunk_count,
                chunks.len()
            )));
        }
        if vectors.len() as u64 != manifest.vector_count {
            return Err(StorageError::InvalidManifest(format!(
                "manifest expected {} vectors, loaded {}",
                manifest.vector_count,
                vectors.len()
            )));
        }
        if let Some(dimensions) = manifest.dimensions {
            if vectors
                .iter()
                .any(|record| record.vector.len() != dimensions)
            {
                return Err(StorageError::InvalidManifest(format!(
                    "persisted vectors did not all match manifest dimension {dimensions}"
                )));
            }
        }

        Ok(Self {
            manifest,
            corpus,
            chunks,
            vectors,
        })
    }

    /// Consumes this value into an index.
    pub fn into_index<B: TextEmbedderBackend>(self, embedder: B) -> Result<RetrievalIndex<B>> {
        validate_embedder_compatibility(&self.manifest.embedder, &embedder.model_info())?;
        let raw_text_by_chunk_id = self
            .chunks
            .iter()
            .filter_map(|record| {
                record
                    .raw_text
                    .as_ref()
                    .map(|raw_text| (record.chunk.chunk_id.clone(), raw_text.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let chunks = self
            .chunks
            .into_iter()
            .map(|record| record.chunk)
            .collect::<Vec<_>>();

        RetrievalIndex::from_parts(
            embedder,
            self.corpus.corpus_options,
            self.corpus.bm25_options,
            chunks,
            raw_text_by_chunk_id,
            self.vectors,
        )
        .map_err(|err| StorageError::InvalidState(err.to_string()))
    }

    /// Returns load with embedder.
    pub fn load_with_embedder<B: TextEmbedderBackend>(
        path: &Path,
        embedder: B,
    ) -> Result<RetrievalIndex<B>> {
        Self::load_from_path(path)?.into_index(embedder)
    }

    /// Imports this legacy JSON/JSONL snapshot into a SQLite-backed `text-index`.
    #[cfg(feature = "sqlite")]
    pub fn import_into_sqlite_path(self, path: &Path) -> Result<text_index::IndexMutationReport> {
        use text_index::{
            IndexBuildOptions, IndexDocument, IndexDocumentMetadata, StoredVector, TextIndexStore,
        };

        let mut store = text_index::SqliteIndexStore::open(path, true)
            .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        let mut vectors_by_chunk = self
            .vectors
            .into_iter()
            .map(|record| {
                let chunk_id = record.id.as_str().to_string();
                (
                    chunk_id.clone(),
                    StoredVector {
                        chunk_id,
                        backend: format!("{:?}", self.manifest.embedder.backend),
                        model_name: self.manifest.embedder.model_name.clone(),
                        dimensions: record.vector.len(),
                        vector: record.vector,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut chunks_by_document = BTreeMap::<String, Vec<DocumentChunk>>::new();
        for record in self.chunks {
            chunks_by_document
                .entry(record.chunk.document_id.clone())
                .or_default()
                .push(record.chunk);
        }

        let mut total = text_index::IndexMutationReport {
            documents_received: chunks_by_document.len(),
            documents_replaced: 0,
            documents_removed: 0,
            documents_skipped: 0,
            chunks_indexed: 0,
            vectors_indexed: 0,
        };
        for (document_id, chunks) in chunks_by_document {
            let Some(first) = chunks.first() else {
                total.documents_skipped += 1;
                continue;
            };
            let document = IndexDocument {
                id: document_id.clone(),
                title: first.metadata.get("__title").cloned(),
                body: chunks
                    .iter()
                    .map(|chunk| chunk.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
                language: first.metadata.get("language").cloned(),
                metadata: IndexDocumentMetadata {
                    attributes: first.metadata.clone(),
                    source: first.source.clone(),
                    provenance: first.provenance.clone(),
                    annotations: first.annotations.clone(),
                },
                analysis_attachments: Vec::new(),
                semantic_facets: Vec::new(),
            };
            let index_chunks = chunks
                .iter()
                .map(text_index::IndexChunk::from)
                .collect::<Vec<_>>();
            let vectors = index_chunks
                .iter()
                .filter_map(|chunk| vectors_by_chunk.remove(&chunk.id))
                .collect::<Vec<_>>();
            let report = store
                .upsert_document_state(
                    document,
                    index_chunks,
                    vectors,
                    &IndexBuildOptions::default(),
                )
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
            total.documents_replaced += report.documents_replaced;
            total.chunks_indexed += report.chunks_indexed;
            total.vectors_indexed += report.vectors_indexed;
        }
        Ok(total)
    }
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), value).map_err(json_error)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T> {
    let file = File::open(path)?;
    serde_json::from_reader(BufReader::new(file)).map_err(json_error)
}

fn write_jsonl<T: Serialize>(path: impl AsRef<Path>, values: &[T]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for value in values {
        let line = serde_json::to_string(value).map_err(json_error)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str::<T>(&line)
            .map_err(|err| StorageError::Json(format!("line {}: {err}", line_index + 1)))?;
        values.push(value);
    }
    Ok(values)
}

fn validate_embedder_compatibility(
    persisted: &EmbeddingModelInfo,
    current: &EmbeddingModelInfo,
) -> Result<()> {
    if !persisted.model_name.is_empty()
        && !current.model_name.is_empty()
        && persisted.model_name != current.model_name
    {
        return Err(StorageError::InvalidState(format!(
            "persisted embedder `{}` did not match provided embedder `{}`",
            persisted.model_name, current.model_name
        )));
    }
    if persisted.dimensions > 0
        && current.dimensions > 0
        && persisted.dimensions != current.dimensions
    {
        return Err(StorageError::InvalidState(format!(
            "persisted embedder dimensions {} did not match provided embedder dimensions {}",
            persisted.dimensions, current.dimensions
        )));
    }
    Ok(())
}

fn json_error(error: serde_json::Error) -> StorageError {
    StorageError::Json(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{HybridConfig, IngestionOptions, SearchDocument, SearchQuery};
    use tempfile::tempdir;
    use text_embeddings::{
        DenseVector, HashedTextEmbedder, TextEmbeddingBackend, TextEmbeddingBackendKind,
        TextEmbeddingConfig, TextEmbeddingMetadata,
    };
    use text_lexical::CorpusOptions;

    use super::*;

    fn embedder() -> HashedTextEmbedder {
        HashedTextEmbedder::new(
            TextEmbeddingConfig {
                dimensions: 32,
                use_idf: true,
            },
            CorpusOptions::default(),
        )
        .unwrap()
    }

    #[derive(Debug, Clone)]
    struct NamedEmbedder {
        name: String,
        dimensions: usize,
    }

    impl TextEmbeddingBackend for NamedEmbedder {
        fn embed_text(&self, _text: &str) -> text_core::Result<DenseVector> {
            DenseVector::new(vec![1.0; self.dimensions])
        }

        fn metadata(&self) -> TextEmbeddingMetadata {
            TextEmbeddingMetadata {
                backend: TextEmbeddingBackendKind::Custom,
                model_name: Some(self.name.clone()),
                dimensions: Some(self.dimensions),
                ..TextEmbeddingMetadata::default()
            }
        }
    }

    #[test]
    fn persisted_index_round_trips_with_manifest_validation() {
        let mut index = RetrievalIndex::new(embedder());
        index
            .ingest_documents(
                &[
                    SearchDocument {
                        id: "doc-1".to_string(),
                        title: Some("Rust Search".to_string()),
                        body: "Rust cargo crates enable semantic search over documentation."
                            .to_string(),
                        metadata: BTreeMap::from([("lang".to_string(), "en".to_string())]),
                        source: None,
                        provenance: Vec::new(),
                        annotations: Vec::new(),
                    },
                    SearchDocument {
                        id: "doc-2".to_string(),
                        title: None,
                        body: "Music playlists and recommendation notes.".to_string(),
                        metadata: BTreeMap::from([("lang".to_string(), "en".to_string())]),
                        source: None,
                        provenance: Vec::new(),
                        annotations: Vec::new(),
                    },
                ],
                &IngestionOptions::default(),
            )
            .unwrap();

        let dir = tempdir().unwrap();
        let persisted = PersistedSearchIndex::from_index(&index);
        persisted.save_to_path(dir.path()).unwrap();

        let loaded = PersistedSearchIndex::load_with_embedder(dir.path(), embedder()).unwrap();
        let query = SearchQuery {
            text: "rust search docs".to_string(),
            top_k: 2,
            filter: None,
            hybrid: HybridConfig::default(),
        };

        assert_eq!(
            loaded.search(&query).unwrap(),
            index.search(&query).unwrap()
        );
    }

    #[test]
    fn malformed_manifest_is_rejected() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("manifest.json"),
            r#"{"schema_version":1,"chunk_count":2,"vector_count":1,"dimensions":2,"embedder":{"model_name":"hashed-text-embedder","backend":"hashed","dimensions":32},"chunks_file":{"path":"chunks.jsonl","records":0},"vectors_file":{"path":"vectors.jsonl","records":0},"corpus_file":"corpus.json"}"#,
        )
        .unwrap();
        fs::write(dir.path().join("corpus.json"), r#"{"corpus_options":{"processing":{"language":null,"lowercase":true,"normalize_unicode":true,"keep_apostrophes":true,"include_punctuation":false},"min_term_len":1,"stop_words":[],"max_terms_per_document":null},"bm25_options":{"k1":1.2,"b":0.75,"min_term_len":1,"stop_words":[]}}"#).unwrap();
        fs::write(dir.path().join("chunks.jsonl"), "").unwrap();
        fs::write(dir.path().join("vectors.jsonl"), "").unwrap();

        let err = PersistedSearchIndex::load_from_path(dir.path()).unwrap_err();
        assert!(
            matches!(err, StorageError::InvalidManifest(message) if message.contains("chunks"))
        );
    }

    #[test]
    fn loading_rejects_incompatible_embedder_name_or_dimensions() {
        let mut index = RetrievalIndex::new(NamedEmbedder {
            name: "persisted".to_string(),
            dimensions: 4,
        });
        index
            .ingest_documents(
                &[SearchDocument {
                    id: "doc-1".to_string(),
                    title: None,
                    body: "rust cargo crates".to_string(),
                    metadata: BTreeMap::new(),
                    source: None,
                    provenance: Vec::new(),
                    annotations: Vec::new(),
                }],
                &IngestionOptions::default(),
            )
            .unwrap();

        let dir = tempdir().unwrap();
        PersistedSearchIndex::from_index(&index)
            .save_to_path(dir.path())
            .unwrap();

        let wrong_name = PersistedSearchIndex::load_with_embedder(
            dir.path(),
            NamedEmbedder {
                name: "other".to_string(),
                dimensions: 4,
            },
        )
        .unwrap_err();
        assert!(
            matches!(wrong_name, StorageError::InvalidState(message) if message.contains("embedder"))
        );

        let wrong_dimensions = PersistedSearchIndex::load_with_embedder(
            dir.path(),
            NamedEmbedder {
                name: "persisted".to_string(),
                dimensions: 8,
            },
        )
        .unwrap_err();
        assert!(
            matches!(wrong_dimensions, StorageError::InvalidState(message) if message.contains("dimensions"))
        );
    }
}
