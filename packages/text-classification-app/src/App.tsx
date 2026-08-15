import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-classification-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-classification",
  title: "Text Classification",
  description: "Text classification APIs, runtime selection, imported prediction handling, and deterministic fallbacks.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-classification",
    standaloneRoute: "",
  },
  defaultOperation: "classification.classify",
  featuredOperations: ["classification.classify", "classification.sentiment", "classification.zeroShot", "classification.models", "classification.schema", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run classification, sentiment, and zero-shot workflows.",
      operations: ["classification.classify", "classification.sentiment", "classification.zeroShot"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect package metadata, schema metadata, and model catalog helpers.",
      operations: ["classification.models", "classification.schema", "describe"],
    },
  ],
  presets: [
    {
      id: "classify",
      label: "Classify reliability feedback",
      operation: "classification.classify",
      description: "Classify text with lexical fallback behavior.",
      input: {
        text: "The Rust transcript workflow is reliable, fast, and clear enough for editorial review.",
        labels: ["positive", "negative", "technical feedback"],
        model: { fallbackPolicy: "lexical_fallback" },
      },
    },
    {
      id: "sentiment",
      label: "Analyze support sentiment",
      operation: "classification.sentiment",
      description: "Score positive and negative sentiment with deterministic lexical fallback.",
      input: {
        text: "The caption search results were accurate and the reviewer felt confident approving the cut.",
        model: { fallbackPolicy: "lexical_fallback" },
      },
    },
    {
      id: "zero-shot",
      label: "Zero-shot topic labels",
      operation: "classification.zeroShot",
      description: "Score candidate labels for short text.",
      input: {
        text: "A Rust package ranks transcript passages for semantic search and editorial evidence review.",
        labels: ["software engineering", "sports recap", "music metadata", "legal transcript"],
        model: { fallbackPolicy: "lexical_fallback" },
      },
    },
    {
      id: "schema",
      label: "Inspect classification schema",
      operation: "classification.schema",
      description: "Inspect task schemas, registered presets, and model metadata.",
      input: {},
    },
  ],
  benchmarkScenarios: [
    {
      id: "lexical-classify",
      label: "Lexical Classify",
      operation: "classification.classify",
      input: { text: "rust text analysis is reliable and fast", labels: ["positive", "negative"], model: { fallbackPolicy: "lexical_fallback" } },
      iterations: 100,
      warmupIterations: 5,
      outputCountPath: ["predictions"],
    },
    {
      id: "sentiment",
      label: "Sentiment",
      operation: "classification.sentiment",
      input: { text: "the transcript workflow is reliable", model: { fallbackPolicy: "lexical_fallback" } },
      iterations: 100,
      warmupIterations: 5,
      outputCountPath: ["predictions"],
    },
    {
      id: "zero-shot",
      label: "Zero-shot",
      operation: "classification.zeroShot",
      input: { text: "rust text retrieval", labels: ["code", "sports", "music"], model: { fallbackPolicy: "lexical_fallback" } },
      iterations: 80,
      warmupIterations: 5,
      outputCountPath: ["predictions"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-classification",
    primaryOperations: {
      "classification.classify": {
        title: "Text classification",
        summaryFields: ["predictionCount", "topLabel", "topScore", "fallbackUsed"],
        listFields: ["predictions", "labels"],
        objectFields: ["model", "metadata", "result"],
        explanation: () => "The classifier used imported predictions when supplied; otherwise the configured lexical fallback scored each candidate label.",
      },
      "classification.sentiment": {
        title: "Sentiment analysis",
        summaryFields: ["predictionCount", "topLabel", "topScore", "fallbackUsed"],
        listFields: ["predictions"],
        objectFields: ["model", "metadata", "result"],
        explanation: () => "The sentiment workflow maps the text onto sentiment labels and reports the fallback/model metadata that governed the run.",
      },
      "classification.zeroShot": {
        title: "Zero-shot classification",
        summaryFields: ["predictionCount", "topLabel", "topScore", "fallbackUsed"],
        listFields: ["predictions", "labels"],
        objectFields: ["model", "metadata", "result"],
        explanation: () => "The zero-shot path scores the supplied labels directly, so the result explains which candidate matched the text best.",
      },
      "classification.schema": {
        title: "Classification schema",
        summaryFields: ["taskCount", "modelCount", "presetCount"],
        listFields: ["tasks", "models", "registeredPresets"],
        objectFields: ["result"],
        explanation: () => "The schema view reports available task schemas, presets, and model catalog metadata without running classification.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
