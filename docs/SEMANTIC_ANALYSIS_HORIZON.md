# Semantic analysis implementation horizon

This document defines the implementation direction for semantic analysis in `nlp-stack`.

The goal is not to equate meaning with one embedding vector. The stack should combine multiple typed signals over multiple text scales so longer documents and conversations can be represented as semantic units, neighborhoods, concepts, trajectories, and richer linguistic graphs.

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

Issues #34–#36 established the first deterministic semantic-map baseline.

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

The production baseline remains intentionally exact and O(n^2). This provides a transparent reference behavior rather than silently replacing the result with an approximate algorithm.

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

## H4–H9 baseline slices

Issues #38–#43 continue the horizon with independently useful baseline slices. These slices establish contracts and evidence paths; they do not imply that each horizon is complete.

### H4: large-scale semantic neighborhoods

Implemented baseline:

- exact semantic-neighborhood behavior remains the source-of-truth implementation;
- `SemanticNeighborhoodEvidence` compares that behavior with the existing Foundation-backed `EmbeddingSearchIndex` over the same stored vectors;
- parity evidence reports shared/exact-only/indexed-only edges and maximum similarity delta;
- a Criterion benchmark exercises the comparison path.

This is deliberately an observation path, not an automatic index switch. Next work is to measure crossover points on larger corpora and evaluate approximate-neighbor indexes only when their ownership and accuracy tradeoffs are explicit.

### H5: linguistic semantic graph

Implemented baseline:

- `SemanticLinguisticGraph` composes semantic units/concepts with existing `text-linguistics` outputs;
- typed nodes cover entity mentions, canonical entities, coreference clusters/mentions, events/arguments, relations/endpoints, discourse segments, and topics;
- typed edges retain distinctions such as concept membership, semantic neighbor, mention-to-canonical, coreference, event arguments, relation subject/object, discourse transition, and topic membership;
- source spans are retained wherever the underlying linguistic output exposes them.

`text-analysis` performs graph projection only. Extraction remains owned by `text-linguistics`. Next work should improve explicit span/provenance on event/relation outputs themselves rather than reconstructing information in the composition layer.

### H6: conversation dynamics

Implemented baseline derives observable structure from ordered speaker turns:

- adjacent cross-speaker similarity by speaker pair, including first/last/mean similarity and delta;
- deterministic concept introduction;
- first adoption of an introduced concept by another speaker;
- concept hand-offs across adjacent speaker/topic changes;
- concepts that recur after intervening turns.

These are semantic-structure measurements, not psychological judgments. Next work includes stronger topic-ownership definitions, parallel-thread tracking, and evidence for any notion of an "unresolved" concept before such a label is exposed.

### H7: evaluation and model quality

Implemented baseline:

- versioned semantic-textual-similarity and topic-shift smoke corpora under the existing evaluation framework;
- semantic-similarity reports with Spearman correlation and mean absolute error;
- topic-shift reports with exact-index precision/recall/F1;
- evaluator tests for both report types.

No semantic-quality threshold is blocking CI yet. The smoke fixtures validate contracts/evaluators only. Hashed embeddings remain a deterministic interoperability fixture, not a semantic-quality benchmark.

Next work includes labelled clustering data, larger STS/topic-shift datasets with provenance, multilingual evaluation, and native/browser model latency/memory evidence.

### H8: workbench visualization

Implemented baseline:

- `analysis.semantic-map` is exposed through the existing `text-analysis` package surface and therefore the existing WASM adapter;
- the text-analysis workbench includes a semantic-map preset;
- result views expose semantic concepts, timeline, hotspots, neighborhood edges, linguistic graph nodes/edges, source-carrying semantic units, and optional H4 parity counts.

This first workbench slice uses the existing structured result renderer rather than introducing a bespoke graph/heat-map library. Next work can add purpose-built timeline/hotspot lanes, graph interaction, source-span drill-down, conversation input, speaker distributions, and model comparison while retaining the same report contracts.

Dimensionality-reduction plots may be useful exploratory views, but they must not be presented as lossless semantic geometry.

### H9: optional model-assisted interpretation

Implemented baseline:

- `SemanticInterpretationBackend` is a caller-supplied typed seam;
- backends receive deterministic cluster membership, representative units, and source members;
- optional labels/summaries/confidence become `SemanticConceptInterpretation` annotations with explicit backend/model provenance;
- interpretation validation does not mutate cluster membership, graph structure, spans, or trajectories.

The capability layer adds no hosted inference, credential, download, retry, or caching policy. Next work may add proposition/claim annotations and entailment/contradiction annotations, but only as grounded annotations over deterministic source evidence.

The model-generated interpretation is an annotation over the semantic structure, not the structure's source of truth.

## Design constraints

- Prefer typed Rust contracts over JSON/string operation identifiers in the capability layer.
- Keep model/backend execution policy separate from semantic requests.
- Preserve source spans and provenance through every aggregation step.
- Make thresholds and approximation choices explicit.
- Keep deterministic tie-breaking so reports are reproducible.
- Do not infer intent, personality, truth, agreement, or psychological state from vector distance alone.
- Promote generic algorithms downward only after cross-domain reuse proves their owner.
