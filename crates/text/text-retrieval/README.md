# text-retrieval

Soft-legacy semantic and hybrid retrieval compatibility for
`moritzbrantner-video-analysis`.

Default builds are deterministic and local-first. Transcript integration is
feature-gated, and native model execution stays outside the default dependency
closure.

For the high-level text workflow, see
[`docs/TEXT_WORKSPACE_GUIDE.md`](../../../docs/TEXT_WORKSPACE_GUIDE.md). For a
lower-level walkthrough of how `RetrievalIndex` relates to `TextCorpus`, lexical
scoring, model-backed semantic search, and corpus analysis reports, see
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

These text-specific request/result types are deliberately richer than the
backend-neutral `moenarch-corpus-core` retrieval seam in Foundation. Foundation
owns portable corpus identity and minimal cross-domain retrieval interchange;
this crate continues to own text query interpretation, BM25/semantic/hybrid
ranking, richer filters, candidate windows, snippets, facets, score breakdowns,
related-text behavior, and reranking. Do not expand Foundation merely to make the
two APIs field-for-field identical.

See [`docs/FOUNDATION_RETRIEVAL_BOUNDARY.md`](../../../docs/FOUNDATION_RETRIEVAL_BOUNDARY.md)
for the lossless mapping, dependency rule, and extraction criteria.

## Optional external HNSW (`hnsw` feature)

With the `hnsw` feature, unfiltered semantic hits (`search` without a filter and
`related_chunks`) over at least 4,096 chunks are selected by Foundation's
`vector_analysis_index::hnsw::CosineHnswIndex` (external `hnsw_rs`, #119)
instead of a full exact scan. Candidates are re-scored with exact cosine and
ordered by `(distance, id)`, so scores and tie order keep exact semantics. The
graph is built lazily on the first eligible query and dropped whenever the
exact index is rebuilt (ingest, replace, snapshot import); clones start without
it. Filtered queries, smaller corpora and default builds keep the exact
`VectorSearchIndex`, which is always the source of truth and the fallback.
There is no runtime backend switch: the Cargo feature plus the corpus-size
threshold are the whole seam.

Measured with the deterministic hashed-fixture workload (128 dims, 48 topics,
one chunk per document, related-chunk top-10, 200 queries, one thread; release
build on a shared, loaded 8-core laptop, so treat times as indicative):

| chunks | exact query | HNSW ef=64 query | HNSW recall@10 | HNSW build | LSH best recall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.62 ms | 0.24 ms | 1.000 | 0.35 s | 0.469 |
| 4,096 | 2.63 ms | 0.64 ms | 1.000 | 5.9 s | 0.507 |
| 16,384 | 11.2 ms | 0.97 ms | 0.9995 | 30 s | 0.310 |

The build amortizes after roughly 900-3,000 queries, so the feature suits
long-lived, query-heavy indexes; the first eligible query pays the build. Below
4,096 chunks an exact query stays under ~3 ms and exact, so the graph is not
built. Reproduce with:

```bash
cargo run --release -p moenarch-text-retrieval --features hnsw --example ann_three_way [sizes...]
```

Real embedding models (Candle/ONNX) were not measured; the hashed fixture is a
lexical stand-in. `text-index` still rebuilds an exact index per query from its
store and is not wired to HNSW.

## Quality and limits

Hybrid score calibration and ranking quality are best-effort. Persistence helper
types are stable DTOs, but default package-surface operations plan or build
in-memory indexes and do not write files.

## Package surface

- Primary workflow: `retrieval.search`, which needs an embedding model and fails
  with `missing_dependency` at the surface (`RetrievalIndex` embeds every
  chunk; it has no lexical-only mode).
- `retrieval.rerank` without imported scores uses the rule-based lexical-overlap
  scorer; results carry `runtime: "rule_based"`.
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

- `moritzbrantner-text-index`
- `moritzbrantner-text-embeddings`
- `moritzbrantner-text-lexical`
- `moritzbrantner-vector-analysis-index`
- `moenarch-corpus-core` (portable corpus/retrieval interchange; no committed dependency yet)
