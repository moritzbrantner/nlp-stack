import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-index-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-index",
  title: "Text Index",
  description: "Durable local text indexing and hybrid search.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-index",
    standaloneRoute: "",
  },
  defaultPresetId: "hybrid-search",
  presets: [
    {
      id: "build-index",
      label: "Build index",
      operation: "index.build",
      description: "Build a transient in-memory Text Index from sample documents.",
      input: {
        documents: [
          { id: "doc-1", body: "Rust text indexes combine chunks, vectors, and facets." },
          { id: "doc-2", body: "Transient browser runs avoid durable SQLite writes." },
        ],
        options: { chunkingStrategy: "tokenWindow", chunkTokens: 16, chunkOverlapTokens: 0, storeRawText: true },
        dimensions: 64,
      },
    },
    {
      id: "add-documents",
      label: "Add documents",
      operation: "index.addDocuments",
      description: "Exercise the package-surface add-documents workflow on an in-memory index.",
      input: {
        documents: [
          { id: "doc-1", body: "Add document workflows share the same deterministic index builder." },
          { id: "doc-2", body: "Package surface operations stay side-effect free by default." },
        ],
        dimensions: 64,
      },
    },
    {
      id: "hybrid-search",
      label: "Hybrid search",
      operation: "index.search",
      description: "Build a transient in-memory Text Index and run hybrid search.",
      input: {
        documents: [
          { id: "report-1", title: "Multimodal report", body: "The report cites transcript segments about durable hybrid search.", language: "en", metadata: { attributes: { source: "report", kind: "analysis" } } },
          { id: "caption-1", title: "Transcript", body: "Alice describes semantic facets and timestamped evidence.", language: "en", metadata: { attributes: { source: "transcript", speaker: "alice" } } },
          { id: "note-1", title: "Playlist", body: "Music recommendations and editorial notes.", language: "en", metadata: { attributes: { source: "notes" } } },
        ],
        query: { text: "semantic transcript evidence", topK: 3, explain: true },
        options: { chunkingStrategy: "tokenWindow", chunkTokens: 18, chunkOverlapTokens: 0, storeRawText: true, commit: false },
        dimensions: 96,
      },
    },
    {
      id: "inspect",
      label: "Inspect index",
      operation: "index.inspect",
      description: "Inspect transient index counts without writing SQLite files.",
      input: {
        documents: [
          { id: "doc-1", body: "Durable text index inspection reports chunk and vector counts." },
          { id: "doc-2", body: "Semantic facets remain attached to searchable chunks." },
        ],
        dimensions: 64,
      },
    },
    {
      id: "snapshot-plan",
      label: "Snapshot plan",
      operation: "index.snapshotPlan",
      description: "Plan index snapshot metadata without persistence side effects.",
      input: {
        documents: [
          { id: "doc-1", body: "Snapshot plans describe text index files without writing them." },
          { id: "doc-2", body: "SQLite writes require commit true and a path outside browser builds." },
        ],
        dimensions: 64,
      },
    },
    {
      id: "remove-documents",
      label: "Remove documents plan",
      operation: "index.removeDocuments",
      description: "Preview a side-effect-free document removal plan for a durable index.",
      input: {
        backend: "memory",
        documentIds: ["doc-1", "doc-2"],
      },
    },
  ],
  benchmarkScenarios: [
    {
      id: "hybrid-search",
      label: "Hybrid Search",
      operation: "index.search",
      input: {
        documents: [
          { id: "doc-1", body: "rust text indexing search" },
          { id: "doc-2", body: "video scene reports" },
          { id: "doc-3", body: "transcript semantic facets" },
        ],
        query: { text: "text search", topK: 3 },
      },
      iterations: 60,
      warmupIterations: 5,
      outputCountPath: ["results"],
    },
    {
      id: "inspect",
      label: "Inspect",
      operation: "index.inspect",
      input: {
        documents: [
          { id: "doc-1", body: "rust text indexing search" },
          { id: "doc-2", body: "video scene reports" },
        ],
      },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["chunkCount"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-index",
    primaryOperations: {
      "index.search": {
        title: "Index search",
        summaryFields: ["status", "resultCount"],
        listFields: ["results"],
        objectFields: ["result"],
        explanation: () => "The app built a transient Text Index, combined lexical and semantic candidates, and returned scored chunks with score breakdowns.",
      },
      "index.build": {
        title: "Index build",
        summaryFields: ["status", "documentsReceived", "documentsReplaced", "chunkCount", "vectorCount"],
        objectFields: ["report", "inspect", "result"],
        explanation: () => "The app built a side-effect-free transient Text Index from the sample documents and reported the indexed document, chunk, and vector counts.",
      },
      "index.addDocuments": {
        title: "Add documents",
        summaryFields: ["status", "documentsReceived", "documentsReplaced", "chunkCount", "vectorCount"],
        objectFields: ["report", "inspect", "result"],
        explanation: () => "The app exercised the add-documents surface against an in-memory index so the preview stays deterministic and browser-safe.",
      },
      "index.removeDocuments": {
        title: "Remove documents",
        summaryFields: ["status", "documentCount", "commitRequiredForDurableWrites"],
        objectFields: ["report", "inspect", "result"],
        explanation: () => "The app returns a removal plan for browser/default execution and avoids durable writes unless a committed backend is selected explicitly.",
      },
      "index.inspect": {
        title: "Index inspection",
        summaryFields: ["documentCount", "chunkCount", "vectorCount", "facetCount"],
        objectFields: ["result"],
        explanation: () => "The inspector reports bounded index counts without opening durable browser-side SQLite storage.",
      },
      "index.snapshotPlan": {
        title: "Index snapshot plan",
        summaryFields: ["backend", "documentCount", "chunkCount", "vectorCount"],
        objectFields: ["result"],
        explanation: () => "The snapshot planner describes the transient index state and side-effect policy without writing files.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
