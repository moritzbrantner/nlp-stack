import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-question-answering-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-question-answering",
  title: "Text Question Answering",
  description: "Question answering APIs, imported span postprocessing, and deterministic fallback handling.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-question-answering",
    standaloneRoute: "",
  },
  defaultOperation: "qa.answer",
  featuredOperations: ["qa.answer", "qa.answerWithIndex", "qa.answerWithRetrieval", "qa.answerBatch", "qa.models", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run extractive question-answering and document QA workflows.",
      operations: ["qa.answer", "qa.answerWithIndex", "qa.answerBatch"],
    },
    {
      id: "support",
      label: "Support",
      description: "Run compatibility paths that remain available for existing consumers.",
      operations: ["qa.answerWithRetrieval"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect package metadata and model catalog helpers.",
      operations: ["qa.models", "describe"],
    },
  ],
  presets: [
    {
      id: "local-model-answer",
      label: "Run local QA",
      operation: "qa.answer",
      description: "Use the default local RoBERTa SQuAD2 ONNX model on the server.",
      input: {
        question: "Who presented the tokenizer roadmap?",
        context: "Alice presented the tokenizer roadmap in Berlin while Bob reviewed transcript retrieval evidence.",
        topK: 2,
      },
    },
    {
      id: "lexical-style-imported-answer",
      label: "Review lexical-style answer",
      operation: "qa.answer",
      description: "Demonstrate the fallback-style path by supplying deterministic imported spans.",
      input: {
        question: "What helps editors find evidence?",
        context: "Caption retrieval helps editors find evidence in long transcripts.",
        importedPredictions: [
          { text: "Caption retrieval", score: 0.88, attributes: { byte_start: "0", byte_end: "17" } },
        ],
      },
    },
    {
      id: "index-answer",
      label: "Answer from index",
      operation: "qa.answerWithIndex",
      description: "Build a deterministic Text Index and return cited answers.",
      input: {
        question: "What language has ownership?",
        documents: [
          {
            id: "doc-rust",
            body: "Rust has ownership and deterministic package workflows.",
            language: "en",
            metadata: { attributes: { kind: "language" } },
          },
          { id: "doc-python", body: "Python has a large package ecosystem.", language: "en" },
        ],
        indexOptions: {
          chunkingStrategy: "tokenWindow",
          chunkTokens: 16,
          chunkOverlapTokens: 0,
          storeRawText: true,
        },
        topKChunks: 2,
        topKAnswers: 1,
        localModel: { autoDownload: false },
        fallbackPolicy: "heuristicIfUnavailable",
      },
    },
    {
      id: "compatibility-retrieval-answer",
      label: "Compatibility retrieval answer",
      operation: "qa.answerWithRetrieval",
      description: "Use the deprecated compatibility retrieval path and return cited answers.",
      input: {
        question: "What language has ownership?",
        documents: [
          { id: "doc-rust", body: "Rust has ownership and deterministic package workflows." },
          { id: "doc-python", body: "Python has a large package ecosystem." },
        ],
        topKChunks: 2,
        topKAnswers: 1,
        localModel: { autoDownload: false },
        fallbackPolicy: "heuristicIfUnavailable",
      },
    },
    {
      id: "batch-answer",
      label: "Answer a batch",
      operation: "qa.answerBatch",
      description: "Run multiple imported-span QA requests with item-level results.",
      input: {
        requests: [
          {
            question: "Who presented the roadmap?",
            context: "Alice presented the roadmap.",
            importedPredictions: [{ text: "Alice", score: 0.94 }],
          },
        ],
      },
    },
  ],
  benchmarkScenarios: [
    {
      id: "imported-span",
      label: "Imported Span",
      operation: "qa.answer",
      input: {
        question: "What is reliable?",
        context: "Rust is reliable for deterministic text package benchmarks.",
        importedPredictions: [{ text: "Rust", score: 0.9, attributes: { byte_start: "0", byte_end: "4" } }],
      },
      iterations: 100,
      warmupIterations: 5,
      outputCountPath: ["answers"],
    },
    {
      id: "index-answer",
      label: "Indexed Answer",
      operation: "qa.answerWithIndex",
      input: {
        question: "What language has ownership?",
        documents: [{ id: "doc-rust", body: "Rust has ownership and deterministic package workflows.", language: "en" }],
        indexOptions: {
          chunkingStrategy: "tokenWindow",
          chunkTokens: 16,
          chunkOverlapTokens: 0,
          storeRawText: true,
        },
        topKChunks: 2,
        topKAnswers: 1,
        localModel: { autoDownload: false },
        fallbackPolicy: "heuristicIfUnavailable",
      },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["answers"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-question-answering",
    primaryOperations: {
      "qa.answer": {
        title: "Question answering",
        summaryFields: ["answerCount"],
        listFields: ["answers"],
        objectFields: ["model", "result"],
        explanation: () => "The current browser-safe workflow postprocesses supplied extractive span predictions and reports question, answer, score, span, and runtime metadata.",
      },
      "qa.answerWithIndex": {
        title: "Indexed QA",
        summaryFields: ["retrievedChunkCount", "answerCount"],
        listFields: ["answers", "retrievedChunks"],
        objectFields: ["result"],
        explanation: () => "The workflow builds a deterministic in-memory Text Index from supplied documents and returns cited answers.",
      },
      "qa.answerWithRetrieval": {
        title: "Compatibility Retrieval QA",
        summaryFields: ["retrievedChunkCount", "answerCount"],
        listFields: ["answers", "retrievedChunks"],
        objectFields: ["result"],
        explanation: () => "The support path builds a deterministic in-memory compatibility retrieval index from supplied documents and returns cited answers through the same retrieved context behavior as indexed QA.",
      },
      "qa.answerBatch": {
        title: "Batch QA",
        summaryFields: ["successes", "failures"],
        listFields: ["results"],
        objectFields: ["result"],
        explanation: () => "The workflow runs imported-span QA requests in order and preserves item-level successes and validation errors.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
