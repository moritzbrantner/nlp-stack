# NLP stack architecture

This document records the target architecture for `nlp-stack`. It is the durable result of the August 2026 architecture review of the extraction-era workspace.

The current repository still contains many packages, adapters, application surfaces, and compatibility seams copied from `rust-packages`. Those are migration inputs, not architectural invariants. Implementation issues may remove or rename them as described here.

## Repository ownership

`nlp-stack` is the canonical owner of NLP source, architecture, tests, issues, versions, and releases for packages that belong to this domain. Historical copies in `rust-packages` are compatibility/provenance material only and must not become a second source of truth.

Canonical ownership does not authorize publication. Publishing crates, creating tags/releases, deleting historical source, or migrating consumers still requires an explicit release or migration task.

## Architectural principles

### Organize around semantic capabilities

Packages exist for meaningful NLP capabilities, not because an implementation technique, transport, backend, or pairwise integration can be packaged separately.

The intended capability set is deliberately small. The durable seams are approximately:

- `text-core`: small text kernel;
- `text-lexical`: lexical/statistical primitives;
- `text-linguistics`: language, morphology, syntax, entities, coreference, relations, discourse, and related linguistic operations;
- `text-classification`: classification, sentiment, and zero-shot classification;
- `text-embeddings`: text embeddings and direct pairwise embedding similarity;
- `text-index`: index materialization, storage, and update;
- `text-retrieval`: query-time retrieval, ranking, fusion, filtering, and reranking;
- `text-extractive-qa`: question plus supplied context to answer span(s);
- `text-markov`: the current concrete Markov prediction/generation implementation;
- `text-analysis`: a small library of proven reusable cross-capability recipes;
- `nlp-package-registry`: an outermost aggregate discovery/dispatch composition root.

This list describes architectural roles, not a promise that every item must forever remain a separately published crate.

### Start concrete; extract under pressure

Do not introduce an abstraction merely because a future implementation might need it. Shared crates, backend packages, contract packages, compatibility layers, adapters, and recipes are extracted when real reuse, dependency weight, compile cost, or release lifecycle proves the need.

A package name must describe what the package actually guarantees today. In particular, the current Markov implementation should become `text-markov`; a generic `text-generation` capability should appear only when a real second generation mode or consumer proves the abstraction. Likewise, the current extractive QA implementation should become `text-extractive-qa`; generic question answering is a future abstraction, not a current claim.

### Typed domain APIs are the source of truth

Domain crates expose ordinary typed Rust APIs and typed domain data. They do not own generic string operation identifiers, JSON dispatch, transport DTOs, workflow messages, or `PackageSurface` machinery.

Transport/discovery layers may map external operation names onto typed APIs. The mapping belongs in the aggregate registry or another boundary adapter, not in the capability implementation.

Do not build a dynamic plugin system until a real runtime plugin requirement exists.

## `text-core`

`text-core` is a kernel, not a framework. It owns universal text-domain primitives and cheap deterministic operations such as:

- borrowed and owned documents/segments;
- text spans;
- normalization;
- Unicode-safe segmentation;
- basic tokenization and statistics;
- small universal identifiers or provenance vocabulary when genuinely cross-capability.

It does not own:

- generic analyzer/pipeline lifecycle frameworks;
- `PackageSurface` or JSON dispatch;
- job/runtime orchestration;
- model runtime selection;
- media timestamps or media analysis events;
- index/retrieval behavior;
- capability-specific result graphs.

`text-core` must not depend on `media-core` merely because some text originates from transcripts. Plain text is media-agnostic.

### Data contracts

A data contract is a stable typed shape; it does not need a Rust name ending in `Contract`.

Keep the useful Rust distinction between a borrowed view and an owned serializable value, for example `TextDocument<'a>` and `OwnedTextDocument`. Do not maintain a third parallel `TextDocumentContract` / `TextSegmentContract` hierarchy that mirrors the same data.

The universal document shape should remain small: identity, text, language hint, and similarly universal text facts. An optional extension/attachment container may preserve unknown external metadata across interchange, but capability semantics must not be encoded primarily as stringly-typed extension keys.

Classification results, entities, embeddings, retrieval results, and similar outputs remain explicit capability-owned types.

UTF-8 half-open byte offsets are the canonical span coordinate system. When a boundary requires another coordinate system, use explicitly typed conversion helpers such as `Utf16Span` or `GraphemeSpan`; do not store multiple synchronized offset systems in the core span.

Core provenance is semantic rather than implementation-specific. Values such as observed, heuristic, model-derived, derived, or imported may be universal. Concrete execution facts such as ONNX, Candle, a device, or a model identifier belong to the producing capability's execution metadata.

Errors are capability-local and typed. `text-core` must not re-export a media-wide error enum, and higher capabilities should compose lower-level errors explicitly rather than share one universal NLP error dumping ground.

## Runtime and model execution

`text-model-runtime` is not a durable NLP layer. Remove it as a distinct architectural package.

Capabilities use generic foundation facilities such as `model-runtime` and `runtime-onnx` directly and own their task-specific tokenizer/model glue. Shared NLP runtime helpers should be extracted only after real duplication demonstrates a stable common seam.

Semantic requests are backend-neutral. Model/backend/device/download/cache/retry credentials and resource policy live in a separate execution configuration or context. The same semantic request contract should work with heuristics, local models, ONNX, Candle, imported predictions, or future remote execution.

Backend packaging is hybrid. Lightweight/default implementations may live with the semantic capability. Heavy backends may become separate implementation crates when dependency weight, compile time, platform constraints, or independent release lifecycle creates actual pressure. Do not pre-split every backend.

## Capability boundaries

### Lexical

`text-lexical` owns lexical/statistical behavior: terms and frequencies, n-grams, stemming, stopwords, readability, and other genuinely lexical primitives.

Semantic ownership wins over implementation style:

- sentiment belongs to classification;
- query ranking/BM25 behavior that exists for search belongs to indexing/retrieval;
- summarization belongs to a composition/generation concern rather than lexical merely because an implementation is deterministic.

### Linguistics

Keep `text-linguistics` as one crate initially, but expose useful operations independently: language detection, morphology, syntax, entities, coreference, relations, discourse, and similar operations.

A full Fast/Balanced/Rich analysis is a convenience recipe, not the fundamental API. Split linguistics into multiple packages later only if dependency or lifecycle pressure justifies it.

### Embeddings

`text-embeddings` owns text-to-vector transformation, batching, pooling/normalization, embedding metadata, and direct pairwise similarity.

It does not own collection indexes or collection search. Persistent/materialized indexes belong to `text-index`; query-time collection ranking belongs to `text-retrieval`.

### Index versus retrieval

Keep `text-index` and `text-retrieval` distinct for now.

A useful rule is: if an operation can happen without a query, it probably belongs to indexing; if it exists because a query is being answered, it probably belongs to retrieval.

`text-index` owns materialization/storage/update. `text-retrieval` owns query interpretation, ranking, fusion, reranking, filtering, and retrieval result contracts, accessing indexes through a narrow reader seam.

If a future general search/index project proves a better owner for the generic index layer, move the generic implementation downward then rather than building that abstraction now.

### Extractive QA

`text-extractive-qa` answers a question from supplied context(s). It does not own retrieval.

Retrieval-backed QA/RAG is composition: retrieval produces chunks, a recipe/application maps chunks to QA contexts, extractive QA returns answer spans, and the composition layer maps provenance/citations because it knows how retrieved chunks relate to sources.

### Transcripts and timed text

Neutral timed-text interchange belongs below both speech/audio applications and NLP. For now, place the minimal neutral timed-text contract in `moenarch-foundation::media-core`; extract a dedicated `timed-text-contracts` crate later only if that area becomes large enough to justify it.

`text-core` remains media-agnostic. Audio/speech and NLP may both consume the neutral timed-text contract; neither domain owns the other.

Current `text-transcripts` functionality must be split along this boundary. Neutral transcript/timed-text contracts and generic formatting/parsing are foundation/media concerns. NLP-specific transcript enrichment may remain in the NLP domain only when it is actually linguistic/semantic analysis.

## Composition and bridge crates

Pairwise bridge crates are a smell by default. Prefer contract-mediated composition over a combinatorial matrix such as `generation-linguistics`, `retrieval-embeddings`, and similar pair packages.

A bridge crate is justified only when there is substantial reusable semantic transformation that naturally belongs to neither endpoint. Mere conversion or orchestration should stay in the consuming application or a proven recipe.

`text-analysis` is not an orchestration engine. Recipes begin in the consuming application. Promote a recipe into `text-analysis` only after at least two independent consumers need substantially the same stateless composition. It must not own model acquisition/runtime policy, persistence, indexing infrastructure, or deployment lifecycle.

## Registry and adapters

`nlp-package-registry` is an outermost leaf composition root. It may depend on many capabilities so aggregate CLI/server/WASM boundaries can discover and dispatch them, but semantic/domain crates must never depend on the registry.

The preferred Rust API is direct typed capability usage, not registry dispatch.

Adapters are earned by an actual independent consumption/deployment requirement. Do not automatically create a CLI, server, WASM package, or frontend for every capability.

The browser/demo default is one NLP workbench/showcase that exposes useful capabilities. Keep a capability-specific application only when it has genuinely distinct UX needs. Browser smoke fixtures may remain narrow automated fixtures without becoming maintained mini-products.

## Publication and compatibility

Publication is hybrid. Genuinely reusable capability crates may be independently published. Internal recipes, registry glue, backend adapters, workbench support, and other implementation seams do not become public packages merely because they are crates in the workspace.

An optional umbrella crate may re-export stable capability crates for convenience, but it must not become another semantic owner.

Published sibling libraries use normal semver-compatible dependency requirements by default. Application/build reproducibility belongs to `Cargo.lock` and release evidence. Exact sibling pins are exceptional and require a concrete compatibility reason.

The existing 0.1.x graph is pre-1.0 and may be simplified. Compatibility is consumer-driven: preserve or deliberately migrate APIs used by real consumers, but do not maintain ceremonial wrappers for unused adapters, bridge crates, or obsolete runtime layers. Known consumers should be verified alongside the slice that affects them.

## Evaluation

Quality evaluation is repository-level development/evidence infrastructure under `evaluation/`, not a production runtime crate.

Evaluation may contain reusable deterministic metrics, versioned corpora, baseline predictions, and reports. Production capabilities must not depend on evaluation code. If a metric implementation later proves useful across multiple domains such as search, recommenders, audio, and vision, extract a generic evaluation foundation then.

## Migration order

Migrate bottom-up rather than performing a big-bang rewrite:

1. make repository ownership and this architecture authoritative;
2. establish required foundation boundaries, especially neutral timed text;
3. land repository-level evaluation evidence needed to protect refactors;
4. simplify `text-core` into the media/runtime-agnostic text kernel;
5. remove `text-model-runtime` and move task-specific execution glue to semantic owners;
6. enforce lexical, linguistics, classification, embeddings, index, retrieval, QA, and generation boundaries;
7. remove pairwise bridge crates and keep only proven recipes in `text-analysis`;
8. move erased operation/JSON/transport concerns outward to the registry/adapters;
9. collapse ceremonial focused adapters/apps/WASM surfaces into the aggregate boundaries and single workbench where appropriate;
10. update publication metadata, semver requirements, known consumers, and release plans for the smaller public surface.

Each implementation issue should be independently reviewable and should verify any known consumer it changes. If implementation uncovers a new architectural choice not resolved here, return that choice to an architecture review instead of inventing another abstraction inside the implementation task.
