use text_embeddings::TextEmbedderBackend;
use text_index::IndexSearchResult;
use text_retrieval::{RetrievalIndex, SearchResult};

use crate::{AnswerCitation, AnswerPrediction, CitedAnswer, TextSpanRef};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RetrievedContextChunk {
    pub(crate) document_id: String,
    pub(crate) chunk_id: String,
    pub(crate) full_text: String,
    pub(crate) snippet: String,
    pub(crate) score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RetrievedContext {
    pub(crate) chunks: Vec<RetrievedContextChunk>,
    pub(crate) context_text: String,
}

impl RetrievedContext {
    pub(crate) fn from_index_results(results: &[IndexSearchResult]) -> Self {
        Self::from_chunks(
            results
                .iter()
                .map(|result| RetrievedContextChunk {
                    document_id: result.document_id.clone(),
                    chunk_id: result.chunk_id.clone(),
                    full_text: result.chunk.text.clone(),
                    snippet: result.snippet.clone(),
                    score: result.score,
                })
                .collect(),
        )
    }

    pub(crate) fn from_search_results<B: TextEmbedderBackend>(
        results: &[SearchResult],
        index: &RetrievalIndex<B>,
    ) -> Self {
        Self::from_chunks(
            results
                .iter()
                .map(|result| RetrievedContextChunk {
                    document_id: result.document_id.clone(),
                    chunk_id: result.chunk_id.clone(),
                    full_text: index
                        .raw_text(&result.chunk_id)
                        .unwrap_or(result.snippet.as_str())
                        .to_string(),
                    snippet: result.snippet.clone(),
                    score: result.score,
                })
                .collect(),
        )
    }

    pub(crate) fn cited_answer_from_prediction(&self, answer: AnswerPrediction) -> CitedAnswer {
        let citations = self.citations_for_answer(&answer.answer);
        CitedAnswer {
            answer: answer.answer,
            score: answer.score,
            span: answer.span,
            citations,
        }
    }

    fn from_chunks(chunks: Vec<RetrievedContextChunk>) -> Self {
        let mut context = Self {
            chunks,
            context_text: String::new(),
        };
        context.context_text = context
            .chunks
            .iter()
            .map(|chunk| chunk.full_text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        context
    }

    fn citations_for_answer(&self, answer: &str) -> Vec<AnswerCitation> {
        let mut citations = self
            .chunks
            .iter()
            .filter_map(|chunk| {
                let span = find_span(&chunk.snippet, answer);
                if !chunk_contains_answer(chunk, answer) {
                    return None;
                }
                Some(AnswerCitation {
                    document_id: chunk.document_id.clone(),
                    chunk_id: chunk.chunk_id.clone(),
                    snippet: chunk.snippet.clone(),
                    score: chunk.score,
                    span,
                })
            })
            .collect::<Vec<_>>();
        if citations.is_empty() {
            citations = self
                .chunks
                .iter()
                .take(1)
                .map(|chunk| AnswerCitation {
                    document_id: chunk.document_id.clone(),
                    chunk_id: chunk.chunk_id.clone(),
                    snippet: chunk.snippet.clone(),
                    score: chunk.score,
                    span: None,
                })
                .collect();
        }
        citations
    }
}

fn chunk_contains_answer(chunk: &RetrievedContextChunk, answer: &str) -> bool {
    let answer = answer.trim();
    !answer.is_empty() && (chunk.snippet.contains(answer) || chunk.full_text.contains(answer))
}

fn find_span(text: &str, needle: &str) -> Option<TextSpanRef> {
    let needle = needle.trim();
    if needle.is_empty() {
        return None;
    }
    text.find(needle)
        .map(|byte_start| byte_span_to_ref(text, byte_start, byte_start + needle.len()))
}

fn byte_span_to_ref(text: &str, byte_start: usize, byte_end: usize) -> TextSpanRef {
    let char_start = text[..byte_start].chars().count();
    let char_end = char_start + text[byte_start..byte_end].chars().count();
    TextSpanRef {
        byte_start,
        byte_end,
        char_start,
        char_end,
    }
}
