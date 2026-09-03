import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-analysis-wasm";

const sampleText =
  "Alice presented the tokenizer roadmap in Berlin during the release review. Rust text crates extract keywords, entities, and transcript evidence with deterministic local features. Semantic search and lexical statistics help editors find the strongest report passages.";

const semanticSampleText =
  "Alice presented the semantic search roadmap in Berlin. Semantic retrieval uses embeddings to find related passages. Bob asked how the vector index scales to larger corpora. Alice explained that exact similarity remains the deterministic baseline. Tomatoes grow in garden soil. Healthy soil supports tomato roots.";

const packageAppConfig: PackageAppConfig = {
  library: "text-analysis",
  title: "Text Analysis",
  description: "Unified text analysis orchestration for video-analysis.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-analysis",
    standaloneRoute: "",
  },
  defaultOperation: "analysis.document",
  defaultPresetId: "document-deterministic",
  featuredOperations: ["analysis.document", "analysis.semantic-map", "analysis.corpus", "analysis.similarity", "analysis.describe", "describe"],
  workbench: {
    layout: "focused",
    sidePanels: {
      runtime: false,
      models: false,
      files: false,
      support: false,
    },
    inputChrome: "compact",
    showLandscapeContract: false,
    inputFields: {
      "analysis.document": ["text", "keywordLimit", "summarySentences"],
      "analysis.semantic-map": ["text", "neighborsPerUnit", "neighborThreshold", "clusterThreshold", "includeLinguisticGraph", "includeNeighborhoodEvidence"],
      "analysis.corpus": ["query", "topK", "includeNearDuplicates", "includeSemanticNeighbors"],
      "analysis.similarity": ["left", "right", "n", "mode"],
      "analysis.describe": [],
      describe: [],
    },
  },
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run document, semantic-map, corpus, and text-similarity analysis workflows.",
      operations: ["analysis.document", "analysis.semantic-map", "analysis.corpus", "analysis.similarity"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect package metadata and operation support.",
      operations: ["analysis.describe", "describe"],
    },
  ],
  presets: [
    {
      id: "document-deterministic",
      label: "Document: deterministic report",
      operation: "analysis.document",
      description: "Deterministic report analysis with non-empty keywords, entity hints, sentence counts, and hashed embeddings.",
      input: {
        id: "release-report-berlin",
        text: sampleText,
        profile: "deterministic",
        keywordLimit: 10,
        summarySentences: 3,
        ngramSizes: [2, 3],
        shingleSizes: [3, 5],
        linguistics: { mode: "heuristicBalanced" },
        embedding: { mode: "hashed", dimensions: 128, useIdf: false },
      },
    },
    {
      id: "semantic-map",
      label: "Semantic map: concepts and trajectory",
      operation: "analysis.semantic-map",
      description: "Build deterministic semantic units, concept clusters, trajectory and hotspots, then project existing linguistic evidence onto the same units.",
      input: {
        id: "semantic-map-berlin",
        text: semanticSampleText,
        neighborsPerUnit: 4,
        neighborThreshold: 0.25,
        clusterThreshold: 0.6,
        includeLinguisticGraph: true,
        includeNeighborhoodEvidence: true,
      },
    },
    {
      id: "document-model-backed",
      label: "Document: model-backed fallback",
      operation: "analysis.document",
      description: "Model-capable request with local bundle paths optional, no auto-download, and diagnostics expected when native support is unavailable.",
      input: {
        id: "release-report-model-fallback",
        text: sampleText,
        profile: "modelBacked",
        keywordLimit: 10,
        summarySentences: 3,
        linguistics: { mode: "localModel", bundleDir: ".model-runtime/text-analysis/ner", autoDownload: false, downloadProgress: false },
        embedding: { mode: "candleBundle", bundleDir: ".model-runtime/text-analysis/embeddings" },
      },
    },
    {
      id: "corpus",
      label: "Corpus: transcript retrieval",
      operation: "analysis.corpus",
      description: "Transient transcript corpus with retrieval hits, near-duplicate checks, and semantic neighbors.",
      input: {
        documents: [
          { id: "report-berlin", text: "Alice presents the Berlin release report with tokenizer roadmap evidence and transcript retrieval notes." },
          { id: "transcript-berlin-a", text: "Alice says the tokenizer roadmap links rust text analysis to semantic transcript search." },
          { id: "transcript-berlin-b", text: "Alice says the tokenizer roadmap links rust text analysis to semantic transcript search for editors." },
          { id: "editorial-note", text: "The playlist discussion covers music pacing and visual transitions." },
        ],
        query: "tokenizer transcript search evidence",
        topK: 10,
        includeNearDuplicates: true,
        includeSemanticNeighbors: true,
        embedding: { mode: "hashed", dimensions: 128, useIdf: true },
      },
    },
    {
      id: "similarity",
      label: "Similarity: transcript overlap",
      operation: "analysis.similarity",
      description: "Compare two transcript-style passages with overlapping token shingles and scalar overlap counts.",
      input: {
        left: "Alice presents tokenizer roadmap evidence for semantic transcript search in the Berlin release report.",
        right: "Alice presents tokenizer roadmap evidence for transcript retrieval in the Berlin editor report.",
        n: 2,
        mode: "token",
      },
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-analysis",
    primaryOperations: {
      "analysis.document": {
        title: "Document analysis",
        summaryFields: ["tokenCount", "sentenceCount", "keywordCount", "entityCount", "embeddingDimensions", "diagnosticCount"],
        listFields: ["lexical.keywords", "lexical.ruleEntities", "diagnostics"],
        objectFields: ["core", "lexical", "linguistic", "embedding", "classification"],
        explanation: () => "The orchestrator ran core statistics, lexical keyword extraction, linguistic projections, deterministic embeddings, and retrieval-oriented diagnostics for one document.",
      },
      "analysis.semantic-map": {
        title: "Semantic map",
        summaryFields: ["unitCount", "conceptCount", "neighborCount", "hotspotCount", "graphNodeCount", "graphEdgeCount", "neighborhoodSharedEdgeCount", "neighborhoodExactOnlyEdgeCount", "neighborhoodIndexedOnlyEdgeCount"],
        listFields: ["semantic.clusters", "semantic.timeline", "semantic.hotspots", "semantic.neighbors", "linguisticGraph.nodes", "linguisticGraph.edges"],
        objectFields: ["semantic.embeddingModel", "semantic.conversationDynamics", "neighborhoodEvidence"],
        explanation: () => "The semantic map keeps source units and deterministic concept structure as the primary evidence, adds typed linguistic graph projections, and can compare exact neighbors with the existing vector-index implementation without silently changing algorithms.",
      },
      "analysis.corpus": {
        title: "Corpus analysis",
        summaryFields: ["documentCount", "resultCount", "nearDuplicateCount", "semanticNeighborCount"],
        listFields: ["results", "nearDuplicates", "semanticNeighbors", "documents"],
        objectFields: ["embedding", "diagnostics"],
        explanation: () => "The app built a transient corpus from the sample documents, searched it, and added near-duplicate or semantic-neighbor sections when requested.",
      },
      "analysis.similarity": {
        title: "Text similarity",
        summaryFields: ["mode", "n", "score", "intersectionCount", "unionCount"],
        objectFields: ["similarity", "result"],
        explanation: () => "The run compared two transcript passages with the selected shingle mode and reports scalar overlap counts for the selected n-gram size.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
