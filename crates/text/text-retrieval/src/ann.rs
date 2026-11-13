//! Optional external HNSW candidate selection for unfiltered semantic hits (#119).
//!
//! Compiled only with the `hnsw` feature. The graph is Foundation's
//! `vector_analysis_index::hnsw::CosineHnswIndex`; `hnsw_rs` types stay below
//! that boundary. It is built lazily from the exact index's records on the first
//! eligible query and dropped whenever the exact index is rebuilt, so the exact
//! [`VectorSearchIndex`] remains the source of truth and the fallback.

use std::fmt;
use std::sync::OnceLock;

use text_core::Result as CoreResult;
use text_embeddings::DenseVector;
use vector_analysis_index::hnsw::{CosineHnswConfig, CosineHnswIndex, CosineHnswSearchConfig};
use vector_analysis_index::{VectorHit, VectorRecordId, VectorSearchIndex};

/// Below this many chunks exact search is used even with the feature enabled.
///
/// Measured on the hashed-fixture retrieval workload (see the crate README):
/// exact cosine scan stays cheaper than an HNSW query plus its amortized build
/// up to a few thousand chunks, and is exact.
pub(crate) const HNSW_MIN_CHUNKS: usize = 4_096;

/// Minimum HNSW candidate width; also the number of candidates re-scored exactly.
const MIN_EF_SEARCH: usize = 64;

/// Derived, lazily built HNSW graph over the exact index's records.
///
/// It is cache state, not index state: clones start empty and equality ignores it.
#[derive(Default)]
pub(crate) struct SemanticAnn {
    graph: OnceLock<CosineHnswIndex>,
}

impl SemanticAnn {
    /// Returns the `limit` nearest unfiltered hits, ranked like the exact path.
    pub(crate) fn search(
        &self,
        vectors: &VectorSearchIndex,
        query: &DenseVector,
        limit: usize,
    ) -> CoreResult<Vec<VectorHit>> {
        let graph = match self.graph.get() {
            Some(graph) => graph,
            None => {
                let graph = CosineHnswIndex::from_records(
                    CosineHnswConfig::default(),
                    vectors.records().iter().cloned(),
                )?;
                self.graph.get_or_init(|| graph)
            }
        };
        let report = graph.search(
            query,
            CosineHnswSearchConfig {
                limit,
                ef_search: limit.saturating_mul(2).max(MIN_EF_SEARCH),
            },
        )?;
        Ok(report
            .results
            .into_iter()
            .map(|result| VectorHit {
                id: VectorRecordId::from(result.id),
                distance: result.distance,
                score: result.score,
            })
            .collect())
    }
}

impl Clone for SemanticAnn {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for SemanticAnn {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl fmt::Debug for SemanticAnn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemanticAnn")
            .field("built", &self.graph.get().is_some())
            .finish()
    }
}
