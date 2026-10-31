//! Search documents, their chunks, and conversions from `text-core` contracts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use text_core::{
    TextAnnotationSpan, TextDocument, TextDocumentContract, TextProvenance, TextSegmentContract,
    TextSourceRef,
};
use text_index::{IndexDocument, IndexDocumentMetadata};
use text_lexical::{TextCorpus, TextCorpusDocument};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for search document.
pub struct SearchDocument {
    /// Identifier for this value.
    pub id: String,
    /// The title value.
    #[serde(default)]
    pub title: Option<String>,
    /// The body value.
    pub body: String,
    /// Metadata associated with this value.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// Optional rich source reference.
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    /// Provenance records for this document.
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    /// Annotation spans associated with this document.
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
}

impl SearchDocument {
    /// Creates a new value.
    pub fn new(id: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: None,
            body: body.into(),
            metadata: BTreeMap::new(),
            source: None,
            provenance: Vec::new(),
            annotations: Vec::new(),
        }
    }

    /// Builds this value from text document.
    pub fn from_text_document(document: &TextDocument<'_>) -> Self {
        let mut metadata = BTreeMap::new();
        if let Some(language) = document.language {
            metadata.insert("language".to_string(), language.to_string());
        }
        if let Some(timestamp) = document.timestamp {
            metadata.insert(
                "timestamp_seconds".to_string(),
                timestamp.seconds().to_string(),
            );
        }
        Self {
            id: document.id.to_string(),
            title: None,
            body: document.text.to_string(),
            metadata,
            source: document.timestamp.map(|timestamp| TextSourceRef {
                source_id: None,
                source_kind: Some("text_document".to_string()),
                uri: None,
                media_timestamp: Some(timestamp.into()),
                duration_seconds: None,
            }),
            provenance: Vec::new(),
            annotations: Vec::new(),
        }
    }

    /// Builds this value from a portable text document contract.
    pub fn from_text_document_contract(document: &TextDocumentContract) -> Self {
        let mut metadata = document.attributes.clone();
        if let Some(language) = &document.language {
            metadata.insert("language".to_string(), language.clone());
        }
        if let Some(timestamp) = document.timestamp {
            metadata.insert(
                "timestamp_seconds".to_string(),
                timestamp.seconds().to_string(),
            );
        }
        Self {
            id: document.id.clone(),
            title: None,
            body: document.text.clone(),
            metadata,
            source: document.source.clone(),
            provenance: document.provenance.clone(),
            annotations: document.annotations.clone(),
        }
    }

    /// Builds this value from a portable text segment contract.
    pub fn from_text_segment_contract(segment: &TextSegmentContract) -> Self {
        let mut metadata = segment.attributes.clone();
        if let Some(language) = &segment.language {
            metadata.insert("language".to_string(), language.clone());
        }
        if let Some(timestamp) = segment.timestamp {
            metadata.insert(
                "timestamp_seconds".to_string(),
                timestamp.seconds().to_string(),
            );
        }
        if let Some(duration_seconds) = segment.duration_seconds {
            metadata.insert("duration_seconds".to_string(), duration_seconds.to_string());
        }
        Self {
            id: segment
                .document_id()
                .unwrap_or_else(|| segment.segment_index.to_string()),
            title: None,
            body: segment.text.clone(),
            metadata,
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
        }
    }

    /// Builds this value from a lexical corpus document without losing rich metadata.
    pub fn from_text_corpus_document(document: &TextCorpusDocument) -> Self {
        let mut metadata = document.metadata.clone();
        if let Some(language) = &document.language {
            metadata.insert("language".to_string(), language.clone());
        }
        if let Some(timestamp) = document.timestamp {
            metadata.insert(
                "timestamp_seconds".to_string(),
                timestamp.seconds().to_string(),
            );
        }
        if let Some(duration_seconds) = document
            .source
            .as_ref()
            .and_then(|source| source.duration_seconds)
        {
            metadata.insert("duration_seconds".to_string(), duration_seconds.to_string());
        }
        Self {
            id: document.id.clone(),
            title: None,
            body: document.text.clone(),
            metadata,
            source: document.source.clone(),
            provenance: document.provenance.clone(),
            annotations: document.annotations.clone(),
        }
    }

    /// Builds retrieval documents from a lexical corpus.
    pub fn from_text_corpus(corpus: &TextCorpus) -> Vec<Self> {
        corpus
            .documents
            .iter()
            .map(Self::from_text_corpus_document)
            .collect()
    }
}

impl From<&SearchDocument> for IndexDocument {
    fn from(document: &SearchDocument) -> Self {
        let mut attributes = document.metadata.clone();
        if let Some(language) = attributes.get("language").cloned() {
            attributes.entry("language".to_string()).or_insert(language);
        }
        Self {
            id: document.id.clone(),
            title: document.title.clone(),
            body: document.body.clone(),
            language: document.metadata.get("language").cloned(),
            metadata: IndexDocumentMetadata {
                attributes,
                source: document.source.clone(),
                provenance: document.provenance.clone(),
                annotations: document.annotations.clone(),
            },
            analysis_attachments: Vec::new(),
            semantic_facets: Vec::new(),
        }
    }
}

pub trait IntoSearchDocument {
    fn into_search_document(self) -> SearchDocument;
}

impl IntoSearchDocument for TextDocument<'_> {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_document(&self)
    }
}

impl IntoSearchDocument for &TextDocument<'_> {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_document(self)
    }
}

impl IntoSearchDocument for TextDocumentContract {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_document_contract(&self)
    }
}

impl IntoSearchDocument for &TextDocumentContract {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_document_contract(self)
    }
}

impl IntoSearchDocument for TextSegmentContract {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_segment_contract(&self)
    }
}

impl IntoSearchDocument for &TextSegmentContract {
    fn into_search_document(self) -> SearchDocument {
        SearchDocument::from_text_segment_contract(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for document chunk.
pub struct DocumentChunk {
    /// The chunk identifier value.
    pub chunk_id: String,
    /// The document identifier value.
    pub document_id: String,
    /// Text content for this value.
    pub text: String,
    /// The ordinal value.
    pub ordinal: usize,
    /// Metadata associated with this value.
    pub metadata: BTreeMap<String, String>,
    /// Optional rich source reference.
    #[serde(default)]
    pub source: Option<TextSourceRef>,
    /// Provenance records for this chunk.
    #[serde(default)]
    pub provenance: Vec<TextProvenance>,
    /// Annotation spans associated with this chunk.
    #[serde(default)]
    pub annotations: Vec<TextAnnotationSpan>,
}

impl DocumentChunk {
    /// Converts this retrieval chunk into a portable document contract.
    pub fn to_text_document_contract(&self) -> TextDocumentContract {
        TextDocumentContract {
            id: self.chunk_id.clone(),
            text: self.text.clone(),
            language: self.metadata.get("language").cloned(),
            timestamp: self
                .source
                .as_ref()
                .and_then(|source| source.media_timestamp),
            attributes: self.metadata.clone(),
            source: self.source.clone(),
            provenance: self.provenance.clone(),
            annotations: self.annotations.clone(),
        }
    }

    /// Converts this retrieval chunk into a portable segment contract.
    pub fn to_text_segment_contract(&self) -> TextSegmentContract {
        TextSegmentContract {
            stream_id: Some(self.document_id.clone()),
            segment_index: self.ordinal as u64,
            text: self.text.clone(),
            language: self.metadata.get("language").cloned(),
            timestamp: self
                .source
                .as_ref()
                .and_then(|source| source.media_timestamp),
            duration_seconds: self
                .source
                .as_ref()
                .and_then(|source| source.duration_seconds),
            is_final: true,
            attributes: self.metadata.clone(),
            source: self.source.clone(),
            provenance: self.provenance.clone(),
            annotations: self.annotations.clone(),
        }
    }
}

impl From<&DocumentChunk> for text_index::IndexChunk {
    fn from(chunk: &DocumentChunk) -> Self {
        text_index::IndexChunk {
            id: chunk.chunk_id.clone(),
            document_id: chunk.document_id.clone(),
            ordinal: chunk.ordinal,
            text: chunk.text.clone(),
            byte_start: 0,
            byte_end: chunk.text.len(),
            token_start: None,
            token_end: None,
            metadata: chunk.metadata.clone(),
            source: chunk.source.clone(),
            provenance: chunk.provenance.clone(),
            annotations: chunk.annotations.clone(),
            semantic_facets: Vec::new(),
        }
    }
}
