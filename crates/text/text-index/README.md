# moritzbrantner-text-index

Durable local text indexing and hybrid search for the text package family.

This crate owns the long-term text index boundary: deterministic chunking,
generic ingestion from text contracts and `TextCorpus`, in-memory indexing,
SQLite-backed persistence behind the `sqlite` feature, lexical/semantic/hybrid
search, metadata filters, semantic facets, and analysis attachment persistence.
The default build is deterministic and no-network; hashed embeddings are the
default semantic backend.

Default package-surface operations stay memory-backed and side-effect free.
CLI/server callers may request SQLite explicitly with a path and `commit: true`;
WASM/default builds report SQLite as unsupported instead of pretending to
persist.

## Highlights

- Durable text index boundary for text package consumers.
- Deterministic chunking and ingestion from text contracts and `TextCorpus`.
- Memory-backed default package-surface operations that do not write artifacts.
- Optional SQLite persistence behind the `sqlite` feature.
- Hybrid lexical and semantic search with hashed embeddings by default.

## Stable contract

The stable surface includes `IndexDocument`, `IndexQuery`, `TextIndex`, store
abstractions, mutation and search reports, snapshot planning, and runtime
package-surface request and response behavior. Package-surface operations
preserve the structured `title`, `message`, `summary`, and `result` response
shape across library, CLI, server, WASM, and app adapters.

## Quality and limits

Default package-surface operations are memory-backed and side-effect free.
SQLite writes require an explicit `backend: "sqlite"`, `path`, and
`commit: true`. WASM and default builds report SQLite as unsupported instead of
pretending persistence is available. Hashed embeddings are deterministic and
local-first; they are not model-quality semantic embeddings.

## Example

```rust,no_run
use text_index::{IndexDocument, IndexQuery, MemoryTextIndex};

let mut index = MemoryTextIndex::new_memory()?;
let documents = vec![
    IndexDocument::new("doc-1", "Rust package surfaces expose stable adapters."),
    IndexDocument::new("doc-2", "Durable text indexes support hybrid search."),
];

index.upsert_documents(&documents)?;
let results = index.search(&IndexQuery::new("hybrid search", 5))?;

assert_eq!(results[0].document_id, "doc-2");
# Ok::<(), text_index::TextIndexError>(())
```

Required phrases can be attached to a query when the caller needs exact passage
constraints after hybrid scoring:

```rust,no_run
use text_index::{IndexDocument, IndexQuery, MemoryTextIndex};

let mut index = MemoryTextIndex::new_memory()?;
index.upsert_documents(&[
    IndexDocument::new("doc-1", "Climate policy needs public funding."),
    IndexDocument::new("doc-2", "Climate policy mentions funding separately."),
])?;

let mut query = IndexQuery::new("climate policy public funding", 5);
query.required_phrases = vec!["public funding".to_string()];

let results = index.search(&query)?;
assert_eq!(results[0].matched_phrases, vec!["public funding"]);
# Ok::<(), text_index::TextIndexError>(())
```

## Package surface

- Primary workflow: `index.search` builds or opens the requested backend and
  searches it.
- Workflow operations: `index.build`, `index.addDocuments`, `index.search`, and
  `index.snapshotPlan`.
- Debug and inspection operations: `describe`, `index.open`, and
  `index.inspect`.
- Runtime support: pure Rust defaults are available through library, CLI,
  server, WASM, and app package surfaces.
- Default package-surface requests use an in-memory backend. Durable writes are
  explicit and require SQLite support plus `commit: true`.

## Related crates

- `moritzbrantner-text-core`
- `moritzbrantner-text-lexical`
- `moritzbrantner-text-embeddings`
- `moritzbrantner-text-retrieval`
