# text-retrieval

Soft-legacy semantic and hybrid retrieval compatibility for
`moritzbrantner-video-analysis`.

Default builds are deterministic and local-first. Transcript integration is
feature-gated, and native model execution stays outside the default dependency
closure.

For the high-level text workflow, see
[`docs/TEXT_WORKSPACE_GUIDE.md`](../../../docs/TEXT_WORKSPACE_GUIDE.md). For a
lower-level walkthrough of how `RetrievalIndex` relates to `TextCorpus`, lexical
scoring, hashed semantic search, and corpus analysis reports, see
[`docs/TEXT_CORPUS_GUIDE.md`](../../../docs/TEXT_CORPUS_GUIDE.md).

## Highlights

- Soft-legacy compatibility `RetrievalIndex` APIs for existing package consumers
- Legacy deterministic text chunking with token overlap
- Exact semantic retrieval over `moritzbrantner-vector-analysis-index`
- BM25 lexical retrieval over `moritzbrantner-text-lexical`
- Hybrid weighted ranking with metadata filters
- Related-content lookup, reranking, and persisted snapshot import helpers

## Stable contract

The stable surface is chunk construction, `SearchDocument`,
`TextDocumentContract`/`TextSegmentContract` ingestion, retrieval request/result
types, metadata filters, snapshot planning, and persistence DTOs. New durable
indexing/search work should use `moritzbrantner-text-index`; this crate keeps
older callers and reranking workflows working while they migrate.

## Quality and limits

Hybrid score calibration and ranking quality are best-effort. Persistence helper
types are stable DTOs, but default package-surface operations plan or build
in-memory indexes and do not write files.

## Package surface

- Primary workflow: `retrieval.search` builds a transient in-memory retrieval
  index and searches it.
- Workflow operations: `retrieval.chunk`, `retrieval.search`,
  `retrieval.rerank`, and `retrieval.snapshotPlan`.
- Debug operations: `describe` inspects package metadata and operation support.
- Runtime support: pure Rust, available through library, CLI, server, and WASM
  wrappers.
- Sample output includes `title`, `message`, `summary`, `result`, and
  operation-specific fields such as `chunks`, `report`, `mode`, `results`, or
  snapshot planning details.
- Package-surface operations do not write persistence artifacts or run native
  model inference; `retrieval.snapshotPlan` plans in-memory persistence work but
  does not write files.
- CLI/server/WASM/app package surfaces are request-scoped compatibility
  operations. They do not create server-side retrieval sessions or open durable
  index handles.

## Related crates

- `moritzbrantner-text-embeddings`
- `moritzbrantner-text-lexical`
- `moritzbrantner-vector-analysis-index`
