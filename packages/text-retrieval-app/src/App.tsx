import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-retrieval-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-retrieval",
  title: "Text Retrieval",
  description: "Library-first semantic and hybrid retrieval for video-analysis.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-retrieval",
    standaloneRoute: "",
  },
  defaultOperation: "retrieval.search",
  featuredOperations: ["retrieval.search", "retrieval.chunk", "retrieval.rerank", "retrieval.snapshotPlan", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run transient retrieval chunking, search, reranking, and snapshot planning workflows.",
      operations: ["retrieval.search", "retrieval.chunk", "retrieval.rerank", "retrieval.snapshotPlan"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect package metadata and operation support.",
      operations: ["describe"],
    },
  ],
  presets: [
    {
      id: "chunk",
      label: "Chunk transcript documents",
      operation: "retrieval.chunk",
      description: "Chunk documents into retrievable transcript passages.",
      input: {
        documents: [
          {
            id: "caption-1",
            title: "Tokenizer roadmap",
            body: "Alice presented the tokenizer roadmap in Berlin. The team linked Rust packages to transcript search and scene reports.",
            metadata: { source: "srt", episode: "demo" },
          },
        ],
        options: { strategy: "TokenWindow", ingestion: { chunk_tokens: 14, chunk_overlap_tokens: 3, store_raw_text: true } },
      },
    },
    {
      id: "full-text-search",
      label: "Full-text search",
      operation: "retrieval.search",
      description: "Build and search a transient full-text retrieval index.",
      input: {
        documents: [
          { id: "doc-1", title: "Rust transcript search", body: "rust text retrieval supports transcript keyword search", metadata: { type: "caption" } },
          { id: "doc-2", title: "Scene reports", body: "video scene reports summarize shot boundaries", metadata: { type: "scene" } },
          { id: "doc-3", title: "Editorial evidence", body: "caption retrieval and chunks help editors find evidence", metadata: { type: "review" } },
        ],
        query: "transcript keyword search",
        mode: "fullText",
        topK: 3,
      },
    },
    {
      id: "hybrid-search",
      label: "Hybrid search",
      operation: "retrieval.search",
      description: "Build and search a transient in-memory hybrid retrieval index.",
      input: {
        documents: [
          { id: "doc-1", title: "Rust transcript search", body: "rust text retrieval supports transcript keyword search", metadata: { type: "caption" } },
          { id: "doc-2", title: "Scene reports", body: "video scene reports summarize shot boundaries", metadata: { type: "scene" } },
          { id: "doc-3", title: "Editorial evidence", body: "caption retrieval and chunks help editors find semantic evidence", metadata: { type: "review" } },
        ],
        query: "semantic transcript evidence",
        mode: "hybrid",
        topK: 3,
        dimensions: 96,
      },
    },
    {
      id: "rerank",
      label: "Rerank candidate passages",
      operation: "retrieval.rerank",
      description: "Rerank query/document pairs with lexical overlap.",
      input: {
        query: "rust transcript retrieval",
        documents: [
          "Rust transcript retrieval ranks caption passages for review.",
          "Video scene summaries describe camera motion.",
          "Caption keyword search helps editors find evidence.",
        ],
        topK: 3,
      },
    },
    {
      id: "snapshot-plan",
      label: "Plan retrieval snapshot",
      operation: "retrieval.snapshotPlan",
      description: "Build a transient index and preview persistence manifest records without writing files.",
      input: {
        documents: [
          { id: "doc-1", title: "Rust transcript search", body: "rust text retrieval supports transcript keyword search", metadata: { type: "caption" } },
          { id: "doc-2", title: "Scene reports", body: "video scene reports summarize shot boundaries", metadata: { type: "scene" } },
        ],
        dimensions: 96,
        previewLimit: 3,
      },
    },
  ],
  benchmarkScenarios: [
    {
      id: "chunk",
      label: "Chunk",
      operation: "retrieval.chunk",
      input: { text: "Rust text retrieval chunks transcript content for search. ".repeat(20), maxChunkTokens: 24, overlapTokens: 4 },
      iterations: 100,
      warmupIterations: 5,
      outputCountPath: ["chunks"],
    },
    {
      id: "full-text-search",
      label: "Full-text Search",
      operation: "retrieval.search",
      input: {
        documents: [
          { id: "doc-1", body: "rust text retrieval" },
          { id: "doc-2", body: "video scene reports" },
          { id: "doc-3", body: "transcript search and chunks" },
        ],
        query: "text search",
        mode: "fullText",
      },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["results"],
    },
    {
      id: "hybrid-search",
      label: "Hybrid Search",
      operation: "retrieval.search",
      input: {
        documents: [
          { id: "doc-1", body: "rust text retrieval" },
          { id: "doc-2", body: "video scene reports" },
          { id: "doc-3", body: "transcript search and chunks" },
        ],
        query: "text search",
        mode: "hybrid",
      },
      iterations: 60,
      warmupIterations: 5,
      outputCountPath: ["results"],
    },
    {
      id: "snapshot-plan",
      label: "Snapshot Plan",
      operation: "retrieval.snapshotPlan",
      input: {
        documents: [
          { id: "doc-1", body: "rust text retrieval" },
          { id: "doc-2", body: "video scene reports" },
          { id: "doc-3", body: "transcript search and chunks" },
        ],
        dimensions: 64,
        previewLimit: 3,
      },
      iterations: 60,
      warmupIterations: 5,
      outputCountPath: ["files"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-retrieval",
    primaryOperations: {
      "retrieval.chunk": {
        title: "Document chunking",
        summaryFields: ["documentCount", "chunkCount"],
        listFields: ["chunks"],
        objectFields: ["report"],
        explanation: () => "The chunker split input documents into overlapping search passages and reported how many chunks would be indexed.",
      },
      "retrieval.search": {
        title: "Retrieval search",
        summaryFields: ["mode", "indexedChunks", "resultCount"],
        listFields: ["results"],
        objectFields: ["report", "metadata"],
        explanation: () => "The app built a transient index, selected full-text, semantic, or hybrid ranking, then returned scored matches with document metadata.",
      },
      "retrieval.rerank": {
        title: "Document reranking",
        summaryFields: ["query", "resultCount"],
        listFields: ["results"],
        objectFields: ["result"],
        explanation: () => "The reranker sorted candidate passages with caller-supplied scores when present or deterministic lexical overlap otherwise.",
      },
      "retrieval.snapshotPlan": {
        title: "Retrieval snapshot plan",
        summaryFields: ["chunkCount", "vectorCount", "dimensions", "fileCount"],
        listFields: ["files", "chunksPreview", "vectorsPreview"],
        objectFields: ["manifest", "corpus", "report"],
        explanation: () => "The snapshot planner builds the same transient index as search, then returns manifest and preview records for persistence files without touching the filesystem.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
