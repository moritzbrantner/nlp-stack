# NLP stack architecture

This document records the target architecture for `nlp-stack`. It is the durable result of the August 2026 architecture review of the extraction-era workspace.

The current repository still contains many packages, adapters, application surfaces, and compatibility seams copied from `rust-packages`. Those are migration inputs, not architectural invariants. Implementation issues may remove or rename them as described here.

## Repository ownership

`nlp-stack` is the canonical owner of NLP source, architecture, tests, issues, versions, and releases for packages that belong to this domain. Historical copies in `rust-packages` are compatibility/provenance material only and must not become a second source of truth.

The machine-readable purpose, owned capabilities, known consumers, and excluded authorities (corpus persistence and product workflow stay with consuming applications; neutral timed-text and media timing stay in `moenarch-foundation`) live in `repository_boundary` of `docs/repository-split/package-ownership.json`. `scripts/check_repository_boundaries.py` enforces that declaration, the intra-NLP dependency direction below, and that no application repository becomes a capability implementation dependency.

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

`text-model-runtime` is not a durable NLP layer and has been removed (#17). Tokenizer loading, Candle/ONNX model glue, and backend traits now live in the capability that uses them: embeddings, linguistics (token classification and subword alignment), classification (sequence and zero-shot), extractive QA, and retrieval (reranker trait). Candle-backed models run on the CPU unless the caller passes an explicit device with `on_device`; there is no process-wide device preference.

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

## Semantic maps and epistemic boundary

[ADR 0013](adr/0013-source-relative-corpus-local-semantic-maps.md) defines semantic maps as source-relative and corpus-local. Analysis-derived similarity clusters are semantic regions; durable concept identities are a separate reconciled layer that may persist across map revisions.

`nlp-stack` owns typed semantic analysis and reusable reconciliation behavior, but not corpus persistence. Corpus/application owners persist map revisions and lineage and supply prior state when reconciliation is required. Cross-corpus analysis may report correspondence evidence without creating global concept identity.

Semantic-map outputs describe what supplied material expresses or relates; they do not assert world truth. Downstream epistemic consumers such as `belief-lab` may reference these source-relative objects when normalizing propositions and evaluating support, contradiction, and belief.

### Linguistic evidence versus claim extraction

**Current owner-approved decision (2026-10-13, `/steer` Question 1: A; supersedes earlier answer B):** `nlp-stack` is responsible for **linguistic extraction only**. It may report text structure, entities, grammatical and semantic relations, quotations, reporting/belief constructions, explicit source-relative attribution structure, and exact source spans with provenance. Those observations describe what a source says, **not** whether any embedded statement should become a reusable claim or whether a named person really holds a belief.

**Downstream ownership:** [`proposition-api`](https://github.com/moritzbrantner/proposition-api) owns generic extraction of source-grounded **claim/proposition candidates** from textual/linguistic evidence, including evidence links and reported attribution where relevant. [`philosophy-extractor`](https://github.com/moritzbrantner/philosophy-extractor) owns philosophical claim interpretation, argument reconstruction and formalization. [`belief-lab`](https://github.com/moritzbrantner/belief-lab) owns person-belief attribution, judgments and conviction-related semantics. None of these responsibilities implies objective truth assessment by NLP.

**Question 2 — nested linguistic attribution (owner-approved B, 2026-10-13; answers the question of flat versus nested reported speech):** Preserve **nested reporting/belief relationships** rather than flattening independent relations that lose scope. For example, from “Sarah reports that John believes P,” expose the *linguistic* structure **Sarah reports [John believes [text P]]**, with source-relative evidence and exact recoverable text spans for the reporting construction, embedded belief construction and embedded content. It records what the source *reports*, not whether Sarah endorses P, whether John truly believes P, or whether P is true. This is linguistic evidence supplied to downstream claim extractors, not an NLP claim-candidate/proposition API.

**Question 3 — contextual pronoun coreference (owner-approved B, 2026-10-13; answers whether to resolve pronouns across sentences):** Link a pronoun to an earlier textual referent **when the supplied context supports the link**, and retain the source passages/spans and uncertainty of that linguistic observation. For example, “Sarah interviewed John. He said the river was polluted.” should support a source-relative link from “He” to “John”; the output must retain both mentions as evidence. In “John spoke with Mark. He disagreed,” where the text does not justify a unique antecedent, leave “He” **unresolved** rather than inventing a confident person link. A missing or ambiguous antecedent is not a failure that licenses a guessed identity.

This is **within-source linguistic coreference**, not person identity resolution, a verified speaker/believer assignment, or proposition/claim extraction. Resolved pronouns may enrich the nested reporting evidence of Question 2, but they do not assert that a referenced person actually holds or endorses any embedded statement. Avoid treating heuristic recency matches or uncalibrated numeric scores as proof of an unambiguous referent.

**Question 4 — scoped negation and uncertainty (owner-approved B, 2026-10-13):** Represent **which text-bearing linguistic construction** a source's negation, hypothetical marking or uncertainty applies to. Preserve the operator/cue, its target construction, their source spans and the nesting of the source's expression. For example, **“Sarah says [John does not believe P]”** negates John's reported belief; **“Sarah does not say [John believes P]”** negates Sarah's saying. They must not collapse into the same relation or a flat negative keyword on P. Similarly, “John might leave” and “If John leaves, Sarah will call” must retain the relevant uncertain/hypothetical source scope without upgrading that content to a fact.

The result is **source-relative linguistic scope**, not a logical proof, truth judgment, proposition identity, verified person-belief inference or fully general formal semantics. Keep the outer/inner scope from Question 2, and use correct mentions and ambiguity handling from Question 3 when applicable. A missing or ambiguous scope must remain unknown rather than inventing a definitive interpretation. Extracting only the negation word without its scoped target is insufficient; full logic/counterfactual evaluation is not approved by this decision.

**Question 5 — sentence-level multilingual analysis (owner-approved B, 2026-10-13):** When a document or conversation contains sentences in different languages, detect the language **per sentence** and run **supported** language-specific linguistic analysis in the context of each sentence's detected language, rather than silently applying the document's primary language to every sentence. Preserve each sentence's original text, source identity, exact UTF-8 half-open spans, detected language and any detection/analysis uncertainty. A document containing “Sarah says John is late.” and “Er glaubt, dass John nicht kommen wird.” must retain the English/German sentence distinction; a language or operation without verified support must produce a clear **unsupported, partial or unknown** result, not plausible-looking invented entities, syntax or negation scopes.

Sentence-level processing does not authorize automatic translation, a universally supported language catalog, or full **within-sentence code-switching**. Do not infer cross-language pronoun, quotation, negation or reporting links as established facts merely because independent sentence analyses exist; cross-language relationship resolution remains a separate product decision. Reuse `text-linguistics` and the existing typed provenance/span contracts. Maintain the distinction between **language-detection confidence**, **linguistic extraction confidence**, and the source's own uncertainty. This decision specifies the user-observable outcome, not a new crate, model, backend or transport surface.

**Question 6 — initial language guarantee (owner-approved B, 2026-10-13):** English (`en`), German (`de`) and Spanish (`es`) are the **initial first-class target languages** for `nlp-stack` linguistic analysis. The product must ultimately provide **the same explicitly defined core set of linguistic operations** for all three, with evidence-based language-sensitive execution, original source provenance and inspectable limitations. Do not substitute a document-level language guess for a sentence-level result, silently apply an English-only model to German/Spanish, or present heuristic vocabulary coverage as proof of dependable linguistic analysis.

A capability outside the agreed common core may remain language-dependent and must report unsupported/partial/uncertain output accurately. Additional languages (including French, which has lexical resources) may be supported opportunistically but are **not part of this initial guarantee**. This is a target product contract, **not a claim that the current implementation or evaluated models already satisfy it**. The choice does not authorize automatic translation or intra-sentence code-switching.

**Question 7 — common capability floor (answered 2026-10-13):** Which linguistic operations must be guaranteed across English, German and Spanish initially? Earlier, the question was open pending the owner's answer. Sentence-level detection/routing and unsupported-state reporting from Question 5 remain a separately demonstrable outcome.

**Question 7 answer — B: standard linguistic analysis (owner-approved, 2026-10-13):** The **same initial guaranteed core for English (`en`), German (`de`) and Spanish (`es`)** includes **language detection, sentence segmentation, tokenization, named entity recognition, morphology (including appropriate inflection/lemma information), part-of-speech tagging and dependency syntax**. For each language, those outputs must reflect the original sentence's linguistic context with inspectable source spans and producer/model provenance. A convenient operation name, heuristic guess, lexicon entry, model download or fixture alone does **not** establish that the promised capability is actually supported. Missing or unverified core operations are gaps against this target; expose them honestly while implementation proceeds, and do not declare three-language parity complete until all approved core operations have independently verified language-appropriate behavior in all three languages.

Nested reporting (Question 2), contextual pronoun coreference (Question 3) and scoped negation/uncertainty (Question 4) may have their own language-specific availability and are **not** part of the initial three-language parity guarantee. Intra-sentence code-switching and cross-language relation inference remain outside Question 7. Preserve Question 5's source-relative spans, explicit unsupported/error states and language-aware per-sentence routing; no new general-purpose NLP crate, backend policy or model choice is authorized by this decision.

**Question 8 — interactive source-text inspection (owner-approved B, 2026-10-13):** The **original source text remains central** in the NLP workbench's linguistic analysis experience. Users can select a **word or sentence directly in the displayed text** and inspect the corresponding source-grounded linguistic annotations in a **compact, contextual detail panel**: sentence language, lemma/morphology, part-of-speech, named entities and grammatical/dependency relationships wherever genuinely available. The selected text and its contextual annotations remain visually linked, and changing selection updates the panel without switching to a separate table-oriented or graph-first workflow. Provide accessible keyboard/touch selection alongside pointer interaction; keep the existing workbench and other analysis results available rather than building a second NLP app.

This is a presentation decision over typed, provenance-bearing results from `text-linguistics` and the analysis adapters, **not** authorization for a JavaScript NLP extractor or inference of missing annotations. Display unsupported, ambiguous, missing or partial operation results **as such**, together with applicable source span and model/producer provenance. Source offsets must select the correct original word/sentence even in German/Spanish UTF-8 text; selection must not modify the source, alter linguistic evidence, conflate source uncertainty with extraction uncertainty, or turn reported statements into established beliefs. Interactive dependency trees may be added later as an optional view, but **are not the primary UX**; standalone result tables are not the required inspection workflow. This view must not claim English/German/Spanish feature parity until the Question 7 guarantee is independently verified.

**Question 9 — local reviewable annotation corrections (owner-approved B, 2026-10-13):** Within the source-text-first linguistic inspector (Question 8), users may **correct an inaccurate linguistic annotation** (for example, a word's lemma or part-of-speech tag) while viewing the corresponding original source passage and the model's unmodified result. The **user correction is an explicitly separate, reviewable local overlay**, not an overwrite of the producer/model output. Users can distinguish the original value from their replacement, inspect which word/sentence/source span it applies to, and revert an edit in the current analysis.

**Export:** Provide an explicit, usable export of local corrections with their original values, user-supplied replacement values, exact source identity/span and original producer/model provenance, distinguishable as user-made annotations. Preserve enough source/result revision identity to avoid treating a correction as applying to unrelated or changed input; surface an appropriate warning or invalidate stale overlays after source changes or reanalysis rather than silently assigning them to different tokens. Keep unsupported/missing analysis visible rather than fabricating a model prediction.

**Scope boundary (Question 9, as clarified by Question 10):** Corrections are **local to the active workbench analysis** and exportable; this decision does not authorize automatic retraining, changing upstream source text, silent model-output mutation, server persistence, durable correction history, **automatic** reuse during ordinary reanalysis, corpus-wide propagation, import/synchronization or a new corpus database. Persistent correction lifecycle belongs to a separate later decision (the unselected Question 9 option C). Correction/export is a user-facing review affordance; it must not turn linguistic annotations into claim/proposition extraction or verified person beliefs.

**Question 10 — explicit correction-driven reanalysis (owner-approved B, 2026-10-13):** After a user makes a local linguistic annotation correction (Question 9), the workbench should offer an **explicit user-triggered action** to recompute **affected dependent linguistic analyses** using those corrected annotations **where the underlying capability supports a correction-aware execution path**. For example, when a corrected part-of-speech tag changes the syntactic interpretation of a sentence, supported dependency analysis may be rerun with that corrected input. The act of editing an annotation does **not** trigger automatic recomputation, and recomputation must not arbitrarily revisit unrelated text or pretend that a model supports a correction input it cannot consume.

**Original and derived evidence:** Preserve the immutable original source, original model annotations and the user's explicit correction. Store the recomputed output **separately within the active analysis** as a derived result with a trace to the correction(s), original input/result, affected operation, producer/model/method and source spans where known. The inspector must clearly distinguish **original model output, user correction and subsequent derived linguistic result**; on unsupported operations, errors, cancellation, or ambiguous dependencies, retain the originals and show why reanalysis was unavailable/partial instead of manufacturing downstream updates. Derived analysis is not a more authoritative ground truth, and must not be relabeled as independent model corroboration or a verified person's belief.

**Lifetime:** This is one explicitly initiated, correction-aware computation within the current local workbench session, **not** automatic reapplication of corrections when a document is replaced, ordinary analysis is rerun, or a different corpus is selected. Export of user corrections under Question 9 remains required, but broader derived-result export formats, long-lived storage, universal incremental dependency engines, training and downstream epistemic propagation are not approved by this decision. Implement only the bounded supported dependent path rather than a speculative generic workflow graph.



**Still unresolved:** Intra-sentence code-switching and cross-language relationship resolution, implicit attribution beyond source-supported pronoun coreference, broad dialogue/speaker inference, optional flattened convenience views, claim-to-claim relations and downstream `proposition-api` representation remain separate product questions. Do not flatten nested reporting scope, invent person beliefs or add new semantic owners.

**Superseded history:** Question 1 was originally answered **B** on 2026-10-13, assigning claim-candidate extraction to `nlp-stack`. The owner subsequently revised it to **A** and assigned generic claim extraction to `proposition-api`. The original B wording remains inspectable in [the earlier documentation commit](https://github.com/moritzbrantner/nlp-stack/commit/6198114b8490f94b19e58ba7a111487e585649b7); it is not current product intent.

Timeline, hotspot, graph, dimensionality-reduction, speaker, and comparison surfaces are projections over semantic-map state rather than the canonical identity model. Existing report and workbench structures remain valid baseline behavior and may migrate incrementally.

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
