# ADR 0013: Source-relative, corpus-local semantic maps

- Status: Accepted
- Date: 2026-10-07
- Scope: `nlp-stack` semantic analysis and its downstream epistemic seam

## Context

The repository already exposes a useful deterministic semantic-analysis baseline: source-carrying units, embedding neighborhoods, clusters, trajectories, hotspots, speaker profiles, and a composed linguistic graph. The workbench currently presents several of those report sections under the name "semantic map."

That baseline is valuable, but the term "semantic map" has become overloaded. An analysis cluster is not automatically a durable concept identity, a visualization is not the domain model, and an NLP observation must not silently become a source-independent assertion about the world.

The downstream `belief-lab` repository already treats `nlp-stack` as an evidence producer. The seam needs to remain explicit as both systems grow toward richer semantic exploration and a later world model.

## Decision

### Semantic maps are source-relative

`nlp-stack` describes what supplied material expresses, mentions, relates, groups, or resembles. Its semantic objects retain source provenance.

A semantic-map object may therefore mean "these passages express a related theme" or "this source contains this relation." It does not mean that the corresponding proposition is true in the world.

Truth assessment, support/contradiction across sources, and belief semantics remain downstream epistemic concerns.

### Semantic identity is corpus-local

A durable semantic concept belongs to one corpus namespace. Two independently analyzed corpora do not silently share concept identity, even when their content is highly similar.

Cross-corpus analysis may emit explicit correspondence evidence such as similarity or possible equivalence. It does not rewrite local identities into a global concept namespace.

### Regions and concepts are distinct

A **semantic region** is an analysis-derived grouping produced by a particular analysis revision, model/configuration, and threshold policy.

A **concept** is a durable corpus-local semantic identity that may be supported by one or more regions across re-analysis.

The current `SemanticCluster` baseline should be understood as region-like analysis output until a concrete migration introduces durable concept identity. Existing public names do not require an immediate repository-wide rename.

Optional interpretation may label or summarize regions, but it does not make the interpretation the source of truth for region membership.

### Durable concepts evolve through reconciliation

When a corpus is re-analyzed, concept identity should be reconciled rather than regenerated blindly.

A future reconciliation capability receives the previous semantic map plus current analysis evidence and produces a new map revision. It preserves concept IDs only when continuity is justified and records lineage when identity changes, including splits, merges, replacements, or uncertain correspondence.

The exact reconciliation algorithm, thresholds, and storage schema are deliberately not fixed by this ADR.

### Corpus owners persist maps

`nlp-stack` owns typed semantic-map contracts, semantic analysis, and reusable reconciliation behavior. It does not own a central semantic-map database or corpus lifecycle.

The corpus/application owner persists its map revisions and lineage. A caller that wants durable reconciliation supplies the previous map state to the NLP capability and persists the returned result.

This keeps corpus authority with repositories such as corpus applications and document systems rather than creating a second source of persistence inside `nlp-stack`.

### Views are projections over the map

Timeline, hotspot, graph, dimensionality-reduction, speaker, corpus-comparison, and other visualizations are views over semantic-map state. No individual visualization defines the semantic map.

The existing Pages/workbench report remains a valid baseline surface, but its current panels are not the long-term semantic identity model.

### The epistemic seam remains downstream

`nlp-stack` may export provenance-bearing references to source-relative semantic objects. Downstream consumers may resolve bounded source content when needed.

`belief-lab` owns the later step from source-relative semantic observations toward normalized propositions, support/contradiction, judgments, derivations, and beliefs. A future world model belongs to that epistemic layer rather than being smuggled into NLP clustering or entity/relation extraction.

## Consequences

- Existing `SemanticAnalysisReport`, clustering, trajectory, hotspot, and graph behavior remains useful and does not require a big-bang rewrite.
- Future semantic-map APIs should make corpus scope and map revision explicit when durable identity is introduced.
- Source spans and producer/model/config provenance must survive aggregation and reconciliation.
- Cross-corpus comparison produces correspondence evidence, not global identity.
- `nlp-stack` remains stateless with respect to corpus persistence.
- `belief-lab` may consume references to semantic-map objects without copying embeddings, graph payloads, or corpus storage authority.
- Product UX can evolve from report inspection toward map navigation without changing epistemic ownership.

## Deferred

This ADR intentionally does not create implementation issues for:

- the durable map storage schema;
- the reconciliation algorithm and thresholds;
- migration or renaming of existing `SemanticCluster` APIs;
- a bespoke semantic-map UI;
- cross-corpus equivalence adjudication;
- world-model proposition normalization in `belief-lab`.

Those become implementation work only when a concrete slice requires them.
