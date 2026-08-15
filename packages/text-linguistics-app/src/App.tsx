import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-linguistics-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-linguistics",
  title: "Text Linguistics",
  description: "Local model-backed linguistic analysis pipeline for video-analysis.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-linguistics",
    standaloneRoute: "",
  },
  defaultOperation: "linguistics.analyze",
  featuredOperations: ["linguistics.analyze", "linguistics.entities", "linguistics.language", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run deterministic linguistic analysis, entity extraction, and focused language detection.",
      operations: ["linguistics.analyze", "linguistics.entities", "linguistics.language"],
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
      id: "fast-analysis",
      label: "Fast linguistic pass",
      operation: "linguistics.analyze",
      description: "Run the fast deterministic linguistic profile.",
      input: { text: "Alice presented the tokenizer roadmap in Berlin.", profile: "fast" },
    },
    {
      id: "balanced-analysis",
      label: "Balanced linguistic pass",
      operation: "linguistics.analyze",
      description: "Run balanced token, lemma, POS, entity, and relation extraction.",
      input: {
        text: "Alice presented the tokenizer roadmap in Berlin. Bob reviewed transcript retrieval evidence for the release notes.",
        profile: "balanced",
      },
    },
    {
      id: "rich-analysis",
      label: "Rich linguistic report",
      operation: "linguistics.analyze",
      description: "Run the richest deterministic linguistic projection exposed by the surface.",
      input: {
        text: "Alice presented the tokenizer roadmap in Berlin. The editorial team connected transcript retrieval, scene summaries, and Rust package APIs.",
        profile: "rich",
      },
    },
    {
      id: "entities",
      label: "Extract named entities",
      operation: "linguistics.entities",
      description: "Extract entities and relations from a short sentence.",
      input: { text: "Alice presented the tokenizer roadmap in Berlin while Bob reviewed transcript retrieval in Paris." },
    },
    {
      id: "language",
      label: "Detect language",
      operation: "linguistics.language",
      description: "Detect language and script signals without running the full NLP pipeline.",
      input: { text: "This is a simple English sentence.", sentenceLevel: true, maxAlternatives: 3 },
    },
  ],
  benchmarkScenarios: [
    {
      id: "fast-analysis",
      label: "Fast Analysis",
      operation: "linguistics.analyze",
      input: { text: "Alice presented the tokenizer roadmap in Berlin.", profile: "fast" },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["entities"],
    },
    {
      id: "balanced-analysis",
      label: "Balanced Analysis",
      operation: "linguistics.analyze",
      input: { text: "Alice presented the tokenizer roadmap in Berlin. The team discussed Rust search workflows.", profile: "balanced" },
      iterations: 60,
      warmupIterations: 5,
      outputCountPath: ["entities"],
    },
    {
      id: "rich-analysis",
      label: "Rich Analysis",
      operation: "linguistics.analyze",
      input: { text: "Alice presented the tokenizer roadmap in Berlin. Bob reviewed transcript retrieval evidence.", profile: "rich" },
      iterations: 40,
      warmupIterations: 3,
      outputCountPath: ["entities"],
    },
    {
      id: "language",
      label: "Language",
      operation: "linguistics.language",
      input: { text: "This is a simple English sentence.", sentenceLevel: true, maxAlternatives: 3 },
      iterations: 100,
      warmupIterations: 5,
      outputCountPath: ["alternatives"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-linguistics",
    primaryOperations: {
      "linguistics.analyze": {
        title: "Linguistic analysis",
        summaryFields: ["profile", "language", "tokenCount", "entityCount", "relationCount", "eventCount"],
        listFields: ["tokens", "lemmas", "posTags", "entities", "relations", "events", "topics"],
        objectFields: ["language", "style", "syntax", "model"],
        explanation: () => "The WASM path runs the deterministic linguistic pipeline; the overview server can also expose model catalog metadata without forcing optional native model execution.",
      },
      "linguistics.entities": {
        title: "Entity extraction",
        summaryFields: ["entityCount", "relationCount", "eventCount"],
        listFields: ["entities", "canonicalEntities", "relations", "events"],
        objectFields: ["language", "model"],
        explanation: () => "The entity workflow focuses the linguistic projection on named entities, canonical forms, relations, and event-style facts.",
      },
      "linguistics.language": {
        title: "Language detection",
        summaryFields: ["language", "confidence", "isMixed", "tokenCount"],
        listFields: ["alternatives", "sentencePredictions"],
        objectFields: ["primary", "result"],
        explanation: () => "The language workflow runs the focused detector only, returning primary language, script signals, alternatives, and optional sentence-level predictions.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
