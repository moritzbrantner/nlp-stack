import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-lexical-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-lexical",
  title: "Text Lexical",
  description: "Lexical text features, corpus statistics, TF-IDF, and BM25 for video-analysis.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-lexical",
    standaloneRoute: "",
  },
  defaultOperation: "lexical.analyze",
  featuredOperations: ["lexical.analyze", "lexical.keywords", "lexical.corpusSearch", "lexical.corpusStats", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run lexical analysis, keyword extraction, transient corpus search, and corpus statistics.",
      operations: ["lexical.analyze", "lexical.keywords", "lexical.corpusSearch", "lexical.corpusStats"],
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
      id: "lexical-analysis",
      label: "Analyze release note",
      operation: "lexical.analyze",
      description: "Compute deterministic lexical features for a short text.",
      input: {
        text: "Rust crates make transcript analysis reliable. Editors search captions, inspect keywords, and compare scene summaries before publishing.",
        maxTerms: 10,
      },
    },
    {
      id: "keywords",
      label: "Extract transcript keywords",
      operation: "lexical.keywords",
      description: "Rank deterministic lexical keywords.",
      input: {
        text: "Transcript search highlights transcript evidence, caption keywords, Rust analysis crates, and retrieval workflows.",
        limit: 8,
      },
    },
    {
      id: "corpus-search",
      label: "Search support corpus",
      operation: "lexical.corpusSearch",
      description: "Search a transient BM25 corpus.",
      input: {
        documents: [
          { id: "doc-1", text: "rust text analysis supports transcript keyword extraction" },
          { id: "doc-2", text: "video scene analysis summarizes shot boundaries" },
          { id: "doc-3", text: "caption retrieval and lexical search help editors find evidence" },
        ],
        query: "transcript keyword search",
        mode: "bm25",
      },
    },
    {
      id: "corpus-stats",
      label: "Inspect corpus statistics",
      operation: "lexical.corpusStats",
      description: "Build a transient TF-IDF corpus and report term, document, and sparse-matrix statistics.",
      input: {
        documents: [
          { id: "doc-1", text: "rust text analysis supports transcript keyword extraction" },
          { id: "doc-2", text: "video scene analysis summarizes shot boundaries" },
          { id: "doc-3", text: "caption retrieval and lexical search help editors find evidence" },
        ],
        documentId: "doc-1",
        limit: 8,
        includeSparseMatrix: true,
      },
    },
  ],
  benchmarkScenarios: [
    {
      id: "keywords",
      label: "Keywords",
      operation: "lexical.keywords",
      input: { text: "Rust text analysis supports transcript retrieval and lexical search. ".repeat(24), maxTerms: 12 },
      iterations: 100,
      warmupIterations: 5,
      outputCountPath: ["keywords"],
    },
    {
      id: "corpus-search",
      label: "Corpus Search",
      operation: "lexical.corpusSearch",
      input: {
        documents: [
          { id: "doc-1", text: "rust text analysis" },
          { id: "doc-2", text: "video scene analysis" },
          { id: "doc-3", text: "transcript retrieval and search" },
        ],
        query: "text search",
        mode: "bm25",
      },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["results"],
    },
    {
      id: "corpus-stats",
      label: "Corpus Stats",
      operation: "lexical.corpusStats",
      input: {
        documents: [
          { id: "doc-1", text: "rust text analysis" },
          { id: "doc-2", text: "video scene analysis" },
          { id: "doc-3", text: "transcript retrieval and search" },
        ],
        documentId: "doc-1",
        limit: 8,
      },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["terms"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-lexical",
    primaryOperations: {
      "lexical.analyze": {
        title: "Lexical analysis",
        summaryFields: ["keywordCount", "entityCount", "wordCount", "sentenceCount"],
        listFields: ["keywords", "entities"],
        objectFields: ["readability", "sentiment", "summary", "statistics"],
        explanation: () => "The lexical pipeline computed keywords, readability, sentiment-style cues, rule entities, and a compact summary from the sample text.",
      },
      "lexical.keywords": {
        title: "Keyword ranking",
        summaryFields: ["keywordCount"],
        listFields: ["keywords"],
        explanation: () => "The keyword extractor ranked terms by deterministic lexical frequency and weighting, without loading an external model.",
      },
      "lexical.corpusSearch": {
        title: "Lexical corpus search",
        summaryFields: ["resultCount", "mode"],
        listFields: ["results"],
        objectFields: ["corpus", "metadata"],
        explanation: () => "The app built a transient TF-IDF or BM25 index from the sample corpus and returned the highest scoring documents.",
      },
      "lexical.corpusStats": {
        title: "Corpus statistics",
        summaryFields: ["documentCount", "uniqueTerms", "termCount", "documentTfidfCount"],
        listFields: ["terms", "documentTfidf"],
        objectFields: ["stats", "sparseMatrix", "result"],
        explanation: () => "The app built a transient TF-IDF corpus and reported corpus, term, document, and sparse matrix statistics without writing artifacts.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
