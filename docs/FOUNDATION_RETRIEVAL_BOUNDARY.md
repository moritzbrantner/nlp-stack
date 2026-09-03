# Foundation retrieval boundary

`nlp-stack` and `moenarch-foundation` intentionally expose retrieval at different abstraction levels.

`moenarch-foundation::moenarch-corpus-core` is the cross-domain interchange layer. It names corpus items, carries provenance, accepts a small backend-neutral retrieval request, and returns ranked item references. It must remain usable by text, audio, image, video, document, and other corpus consumers without importing NLP policy.

`nlp-stack` remains the semantic owner of text indexing and text retrieval. Its richer APIs are not duplicate Foundation contracts: they express text-specific behavior that the portable corpus contract deliberately does not standardize.

## Ownership

| Concern | Owner |
| --- | --- |
| corpus source/asset/segment/representation identity and provenance | `moenarch-corpus-core` |
| portable retrieval input, intent, limit, equality filters, ranked item references | `moenarch-corpus-core` |
| text chunking and text index materialization/storage/mutation | `text-index` |
| text lexical/semantic/hybrid query interpretation and ranking | `text-retrieval` |
| NLP candidate windows, score weights/decomposition, facets, snippets, richer filters, sorting and reranking | `text-retrieval` |
| text embedding generation/model policy | `text-embeddings` |
| exact in-memory dense-vector lookup mechanics | `moenarch-vector-analysis-index` |

The Foundation vector index is an implementation primitive used by text retrieval; it is not a competing corpus/search contract.

## Portable mapping

When a real corpus consumer requires an adapter, use the narrow lossless mapping first:

| Foundation request | NLP mapping |
| --- | --- |
| text + `Lexical` | full-text retrieval |
| text + `Semantic` | semantic retrieval |
| text + `Hybrid` | hybrid retrieval |
| corpus item + `Similarity` | related-content lookup when the item maps to an indexed text chunk |
| dense vector + `Similarity`/`Semantic` | vector-backed retrieval when vector dimensions/metadata are compatible |
| equality metadata filters | NLP metadata-equality filters |

For results, Foundation `rank` comes from the final NLP result ordering. `raw_score` may carry the NLP combined score, but it remains backend-defined and must not acquire cross-engine calibration semantics. A result should identify a Foundation segment only when the adapter can map the NLP chunk to an actual corpus segment; it must not invent corpus identities from transient ranking state.

## What does not move to Foundation

The following stay in `nlp-stack` unless independent non-NLP consumers demonstrate a genuinely shared requirement:

- BM25 and other text-specific lexical behavior;
- tokenizer/chunking policy;
- semantic/full-text weights and score decomposition;
- embedding generation, model/backend selection and vector normalization policy;
- metadata-contains and tag filters;
- candidate windows, facets, snippets and presentation-oriented sorting;
- text reranking and related text semantics;
- text index persistence policy and text-specific schemas.

Likewise, Foundation should not grow a second text index just because `corpus-core` exposes `Retriever`.

## What may move downward later

Generic mechanics such as stable top-k selection, rank fusion, reusable filter algebra, index lifecycle contracts, or backend capability descriptions are candidates for Foundation only after at least one non-NLP consumer needs substantially the same semantics. Extraction should preserve the NLP behavior first and remove the local implementation only after the shared primitive is proven.

This is the same direction already expressed by the NLP architecture rule that a future general search/index owner may receive generic implementation once real reuse proves it; `corpus-core` establishes the interchange boundary, not automatic ownership of every search mechanism.

## Dependency and migration rule

Do not add a pairwise `foundation-nlp` bridge crate for mechanical conversions. Keep a small adapter in the first consuming application or capability boundary. Promote conversion code only after multiple independent consumers require it.

`moenarch-corpus-core` is currently a post-extraction Foundation package without publication authorization. Therefore this repository must not add a committed registry dependency that would make normal registry-only CI depend on an unpublished crate. The existing `.coding-tooling.source-deps.json` pin also predates `corpus-core`; advance that exact Foundation revision only in a change that verifies all currently patched Foundation packages together.

The first Rust-level dogfood adapter should therefore be a deliberate migration slice: authorize or otherwise establish a repository-valid dependency path, advance the exact source-development revision with its existing canary, then implement and test the narrow mapping above without changing NLP ranking behavior.

## Compatibility rule for future changes

Changes to `text-index` or `text-retrieval` do not need to mirror Foundation field-for-field. Changes to `corpus-core` must remain representable without importing NLP concepts. If either side needs a richer shared contract, demonstrate the requirement in a real non-NLP consumer before broadening Foundation.
