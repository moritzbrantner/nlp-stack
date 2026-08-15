import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-generation-linguistics-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-generation-linguistics",
  title: "Text Generation Linguistics",
  description: "Adapters from text-linguistics analysis outputs into text-generation.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-generation-linguistics",
    standaloneRoute: "",
  },
  defaultOperation: "generationLinguistics.synthesizeFromAnalysis",
  featuredOperations: [
    "generationLinguistics.synthesizeFromAnalysis",
    "generationLinguistics.analysisTerms",
    "generationLinguistics.trainAnalysis",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run linguistic term extraction, synthesis, and Markov training workflows.",
      operations: [
        "generationLinguistics.synthesizeFromAnalysis",
        "generationLinguistics.analysisTerms",
        "generationLinguistics.trainAnalysis",
      ],
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
      id: "analysis-terms",
      label: "Extract analysis terms",
      operation: "generationLinguistics.analysisTerms",
      description: "Analyze text and convert linguistic signals into weighted terms.",
      input: {
        text: "Alice presented the tokenizer roadmap in Berlin while Bob connected transcript retrieval with scene summaries.",
      },
    },
    {
      id: "synthesize",
      label: "Synthesize from analysis",
      operation: "generationLinguistics.synthesizeFromAnalysis",
      description: "Analyze text and synthesize a deterministic document from linguistic terms.",
      input: {
        id: "analysis-doc",
        text: "Alice presented the tokenizer roadmap in Berlin while Bob connected transcript retrieval with scene summaries.",
      },
    },
    {
      id: "train",
      label: "Train analysis Markov chain",
      operation: "generationLinguistics.trainAnalysis",
      description: "Train a transient Markov chain from linguistic tokens.",
      input: {
        text: "Scene transitions follow visual changes. Transcript retrieval follows caption evidence. Rust packages connect analysis and generation.",
        mode: "lemma",
        order: 2,
      },
    },
  ],
  benchmarkScenarios: [
    {
      id: "synthesize-from-analysis",
      label: "Synthesize",
      operation: "generationLinguistics.synthesizeFromAnalysis",
      input: { id: "bench-analysis", text: "Alice presented the tokenizer roadmap in Berlin. Rust workflows summarize transcript evidence." },
      iterations: 50,
      warmupIterations: 3,
      outputCountPath: ["terms"],
    },
    {
      id: "analysis-terms",
      label: "Terms",
      operation: "generationLinguistics.analysisTerms",
      input: { text: "Scene transitions follow visual changes and transcript cues." },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["terms"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-generation-linguistics",
    primaryOperations: {
      "generationLinguistics.analysisTerms": {
        title: "Linguistic term extraction",
        summaryFields: ["termCount", "entityCount", "language"],
        listFields: ["terms", "entities"],
        explanation: () => "The adapter analyzed the text with deterministic linguistic features and converted entities/tokens into weighted generation terms.",
      },
      "generationLinguistics.synthesizeFromAnalysis": {
        title: "Synthesis from analysis",
        summaryFields: ["id", "language", "assumptionCount"],
        listFields: ["trace.assumptions"],
        objectFields: ["value", "trace"],
        explanation: () => "The synthesis workflow extracted linguistic terms and generated deterministic text, keeping the inversion assumptions visible in the trace.",
      },
      "generationLinguistics.trainAnalysis": {
        title: "Analysis Markov training",
        summaryFields: ["mode", "order", "tokenCount", "contexts"],
        listFields: ["tokens"],
        objectFields: ["result"],
        explanation: () => "The Markov workflow trained on surface, normalized, lemma, or entity-aware tokens derived from the linguistic analysis.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
