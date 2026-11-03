import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-embeddings-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-embeddings",
  title: "Text Embeddings",
  description: "Lightweight semantic text embeddings and search for video-analysis.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-embeddings",
    standaloneRoute: "",
  },
  defaultOperation: "embeddings.embed",
  featuredOperations: ["embeddings.embed", "embeddings.backends", "embeddings.similarity", "embeddings.semanticSearch", "embeddings.relatedTerms", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run deterministic embedding, similarity, semantic search, and related-term workflows.",
      operations: ["embeddings.embed", "embeddings.similarity", "embeddings.semanticSearch", "embeddings.relatedTerms"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect package metadata, operation support, and backend availability.",
      operations: ["embeddings.backends", "describe"],
    },
  ],
  presets: [
    {
      id: "embed",
      label: "Embed transcript passages",
      operation: "embeddings.embed",
      description: "Embed text with a configured embedding model; without one the operation reports model required.",
      input: {
        texts: [
          "Rust text analysis extracts transcript evidence for editors.",
          "Semantic search ranks captions by learned sentence embeddings.",
          "Scene notes and subtitles can share a reusable vector space.",
        ],
      },
    },
    {
      id: "similarity",
      label: "Compare semantic similarity",
      operation: "embeddings.similarity",
      description: "Compare two passages with a configured embedding model; without one the operation reports model required.",
      input: {
        left: "Rust packages rank transcript search results with sentence embeddings.",
        right: "Caption retrieval uses deterministic vectors to compare related text.",
      },
    },
    {
      id: "semantic-search",
      label: "Search caption corpus",
      operation: "embeddings.semanticSearch",
      description: "Search with a configured embedding model; without one the operation reports model required.",
      input: {
        documents: [
          { id: "doc-1", text: "Rust text analysis extracts transcript keywords and entities." },
          { id: "doc-2", text: "Video scene reports summarize camera motion and shot boundaries." },
          { id: "doc-3", text: "Caption retrieval ranks semantic neighbors for editorial review." },
        ],
        query: "transcript semantic search",
        topK: 3,
      },
    },
    {
      id: "related-terms",
      label: "Find related terms",
      operation: "embeddings.relatedTerms",
      description: "Score co-occurring terms from local text.",
      input: {
        text: "rust text analysis supports transcript search text embeddings semantic search transcript retrieval",
        term: "transcript",
        windowSize: 4,
        limit: 8,
      },
    },
    {
      id: "backends",
      label: "Inspect backend catalog",
      operation: "embeddings.backends",
      description: "List feature-gated embedding model backends without loading models.",
      input: {},
    },
  ],
  benchmarkScenarios: [
    {
      id: "related-terms",
      label: "Related Terms",
      operation: "embeddings.relatedTerms",
      input: { text: "rust text analysis supports transcript search text embeddings", term: "text" },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["relatedTerms"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-embeddings",
    primaryOperations: {
      "embeddings.embed": {
        title: "Text embeddings",
        summaryFields: ["embeddingCount", "dimensions"],
        listFields: ["embeddings"],
        objectFields: ["model"],
        explanation: () => "The configured embedding model converted each input text into a fixed-size vector.",
      },
      "embeddings.backends": {
        title: "Embedding backends",
        summaryFields: ["defaultBackend", "backendCount"],
        listFields: ["backends"],
        objectFields: ["result"],
        explanation: () => "The backend catalog shows the feature-gated native embedding backends without loading model bundles; none is available by default.",
      },
      "embeddings.similarity": {
        title: "Embedding similarity",
        summaryFields: ["similarity", "dimensions"],
        objectFields: ["model", "result"],
        explanation: () => "The selected texts were embedded with the same deterministic model and compared with vector similarity.",
      },
      "embeddings.semanticSearch": {
        title: "Semantic search",
        summaryFields: ["resultCount", "dimensions"],
        listFields: ["results"],
        objectFields: ["model"],
        explanation: () => "The app built a transient in-memory semantic index from the sample documents and returned the nearest matches for the query.",
      },
      "embeddings.relatedTerms": {
        title: "Related terms",
        summaryFields: ["term", "relatedTermCount"],
        listFields: ["relatedTerms"],
        explanation: () => "The co-occurrence graph scored terms that appeared near the requested term in the local sample text.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
