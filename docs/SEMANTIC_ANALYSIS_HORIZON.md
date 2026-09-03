# Semantic analysis implementation horizon

This document defines the implementation direction for semantic analysis in `nlp-stack`.

The goal is not to equate meaning with one embedding vector. The stack should combine multiple typed signals over multiple text scales so longer documents and conversations can be represented as semantic units, neighborhoods, concepts, trajectories, and eventually richer linguistic graphs.

## Architectural rule

Do not introduce a `text-semantics` mega-crate merely to collect everything described here.

Ownership remains capability-driven:

- `text-core` owns text structure, stable spans, segmentation, and universal text primitives;
- `text-embeddings` owns text-to-vector transformation and direct vector similarity;
- `text-linguistics` owns entities, coreference, relations, events, discourse, and other linguistic semantics;
- `text-index` / `text-retrieval` own materialized search state and query-time retrieval behavior;
- generic vector/index algorithms belong in `moenarch-foundation` when they are not NLP-specific;
- reusable cross-capability semantic recipes may live in `text-analysis` only while they remain typed, stateless composition.

Semantic request/result contracts stay independent of model backend, device, download, cache, credential, and retry policy.

## Implemented first batch

Issues #34–#36 establish the first deterministic semantic-map baseline.

### S1: multi-scale semantic units

Represent meaning-bearing units at multiple levels:

- sentence;
- paragraph;
- document;
- conversation speaker turn, with sentence children.

Each unit retains stable identity, source identity, hierarchy, sequence order, UTF-8 span, optional speaker/timing provenance, text, and an embedding vector.

The library provides a deterministic hashed baseline and accepts any existing `TextEmbeddingBackend` for real model-backed embeddings.

### S2: semantic neighborhood graph

For the primary sequence (sentences for documents, speaker turns for conversations), compute exact pairwise cosine similarity and expose a deterministic undirected k-nearest-neighbor graph.

The first implementation is intentionally exact and O(n^2). This provides a transparent baseline. If scale becomes a real problem, index-backed exact/approximate kNN can replace the implementation behind the same result shape.

### S3: concepts, hotspots, and trajectory

Derive deterministic higher-level structure from the similarity matrix:

- connected-component concept clustering under an explicit similarity threshold;
- a medoid/representative unit for every concept;
- cluster cohesion;
- per-unit semantic shift against the previous unit;
- activation against the concept representative;
- hotspot coverage, persistence, mean activation, and peak position;
- per-speaker concept shares for conversations.

Every primary unit belongs to exactly one concept, including singleton concepts. The baseline does not ask an LLM to name clusters.

## Next horizons

### H4: large-scale semantic neighborhoods

Move generic kNN/index mechanics downward when scale demonstrates the need. Benchmark exact pairwise computation against foundation vector indexes and approximate-neighbor methods while preserving deterministic test fixtures and result semantics.

### H5: linguistic semantic graph

Join the semantic-map units to `text-linguistics` outputs:

- entities and canonical entities;
- coreference chains;
- events;
- typed relations;
- discourse links;
- topic signals.

This produces a graph with several edge kinds rather than pretending cosine similarity alone is meaning.

### H6: conversation dynamics

Add speaker-aware measures that can be derived from ordered turn embeddings and concept membership:

- semantic convergence/divergence between speakers;
- topic ownership and hand-off;
- recurring unresolved concepts;
- concept introduction and adoption;
- parallel conversational threads.

These measures must be phrased as observable semantic-structure signals, not psychological judgments about participants.

### H7: evaluation and model quality

Add versioned evaluation data for:

- semantic textual similarity correlation;
- clustering stability/quality on labelled corpora;
- topic-shift boundary quality;
- multilingual behavior;
- model size/latency/memory across native and browser/WASM paths.

Hashed embeddings remain a deterministic interoperability fixture, not a semantic-quality benchmark.

### H8: workbench visualization

Expose the typed report in the NLP workbench with views such as:

- semantic timeline/heat map;
- concept hotspot lanes;
- neighborhood graph;
- per-speaker concept distribution;
- drill-down from concept to source spans/turns;
- model comparison using the same semantic-map contracts.

Dimensionality-reduction plots may be useful exploratory views, but they must not be presented as lossless semantic geometry.

### H9: optional model-assisted interpretation

Only after the deterministic structures are useful on their own, allow optional model-assisted interpretation such as:

- human-readable names for deterministic clusters;
- claim/proposition extraction;
- entailment/contradiction classification;
- concise semantic summaries grounded in cluster members and source spans.

The model-generated interpretation is an annotation over the semantic structure, not the structure's source of truth.

## Design constraints

- Prefer typed Rust contracts over JSON/string operation identifiers in the capability layer.
- Keep model/backend execution policy separate from semantic requests.
- Preserve source spans and provenance through every aggregation step.
- Make thresholds and approximation choices explicit.
- Keep deterministic tie-breaking so reports are reproducible.
- Do not infer intent, personality, truth, agreement, or psychological state from vector distance alone.
- Promote generic algorithms downward only after cross-domain reuse proves their owner.
