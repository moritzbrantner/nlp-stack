import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-model-runtime-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-model-runtime",
  title: "Text Model Runtime",
  description: "Shared tokenizer and native text model runtime traits for video-analysis.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-model-runtime",
    standaloneRoute: "",
  },
  defaultOperation: "runtime.onnxQaProbe",
  featuredOperations: [
    "runtime.onnxQaProbe",
    "runtime.downloadBundle",
    "runtime.tokenizeSummary",
    "runtime.bundleCheck",
    "runtime.tokenizerProbe",
    "runtime.softmax",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run deterministic tokenizer and local bundle readiness workflows.",
      operations: ["runtime.onnxQaProbe", "runtime.downloadBundle", "runtime.tokenizeSummary", "runtime.bundleCheck", "runtime.tokenizerProbe"],
    },
    {
      id: "support",
      label: "Support",
      description: "Run reusable runtime support helpers.",
      operations: ["runtime.softmax"],
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
      id: "onnx-qa-probe",
      label: "Probe local QA",
      operation: "runtime.onnxQaProbe",
      description: "Resolve the default RoBERTa SQuAD2 ONNX bundle and run one local QA inference on the server.",
      input: {
        question: "What language is reliable?",
        context: "Rust is reliable for systems programming.",
        topK: 1,
      },
    },
    {
      id: "tokenize-summary",
      label: "Summarize tokenizer offsets",
      operation: "runtime.tokenizeSummary",
      description: "Build a deterministic whitespace-token summary.",
      input: { text: "Rust text runtime summarizes token offsets for transcript package surfaces.", maxTokens: 12 },
    },
    {
      id: "softmax",
      label: "Normalize logits",
      operation: "runtime.softmax",
      description: "Normalize support logits into a probability distribution.",
      input: { logits: [0.1, 0.3, 1.2, -0.7, 2.4, 0.0] },
    },
    {
      id: "bundle-check",
      label: "Check tokenizer bundle files",
      operation: "runtime.bundleCheck",
      description: "Validate required local tokenizer files without downloads.",
      input: {
        modelId: "demo-tokenizer",
        capability: "tokenizer",
        bundleRoot: ".model-runtime/demo-tokenizer",
        requiredFiles: ["tokenizer.json"],
      },
    },
    {
      id: "tokenizer-probe",
      label: "Probe tokenizer bundle",
      operation: "runtime.tokenizerProbe",
      description: "Report tokenizer load/run readiness for a local tokenizer file.",
      input: {
        modelId: "demo-tokenizer",
        tokenizerPath: ".model-runtime/demo-tokenizer/tokenizer.json",
        sample: "Rust text runtime",
      },
    },
  ],
  benchmarkScenarios: [
    {
      id: "tokenizer-summary",
      label: "Tokenizer Summary",
      operation: "runtime.tokenizeSummary",
      input: { text: "Rust text runtime summarizes token offsets for package surfaces.", maxTokens: 12 },
      iterations: 120,
      warmupIterations: 5,
      outputCountPath: ["tokens"],
    },
    {
      id: "softmax",
      label: "Softmax",
      operation: "runtime.softmax",
      input: { logits: [0.1, 0.3, 1.2, -0.7, 2.4, 0.0] },
      iterations: 200,
      warmupIterations: 10,
      outputCountPath: ["probabilities"],
    },
    {
      id: "bundle-check",
      label: "Bundle Check",
      operation: "runtime.bundleCheck",
      input: {
        modelId: "demo-tokenizer",
        capability: "tokenizer",
        bundleRoot: ".model-runtime/demo-tokenizer",
        requiredFiles: ["tokenizer.json"],
      },
      iterations: 100,
      warmupIterations: 5,
      outputCountPath: ["missingFiles"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-model-runtime",
    primaryOperations: {
      "runtime.tokenizeSummary": {
        title: "Tokenizer summary",
        summaryFields: ["tokenCount", "maxTokens", "truncated"],
        listFields: ["tokens"],
        objectFields: ["offsets", "metadata", "result"],
        explanation: () => "The runtime helper split text deterministically and reported token offsets/counts without invoking an optional native model runtime.",
      },
      "runtime.bundleCheck": {
        title: "Bundle readiness",
        summaryFields: ["loadable", "missingFileCount"],
        listFields: ["missingFiles", "presentFiles", "requiredFiles"],
        objectFields: ["report", "result"],
        explanation: () => "The runtime helper checked local bundle files only and reported whether the configured model assets are present.",
      },
      "runtime.tokenizerProbe": {
        title: "Tokenizer probe",
        summaryFields: ["loadable", "ran"],
        listFields: ["diagnostics"],
        objectFields: ["report", "run", "result"],
        explanation: () => "The runtime helper inspected a local tokenizer path and only runs sample tokenization when the optional tokenizer feature is available.",
      },
      "runtime.softmax": {
        title: "Softmax probabilities",
        summaryFields: ["probabilityCount", "maxProbability"],
        listFields: ["probabilities"],
        objectFields: ["result"],
        explanation: () => "The support helper normalized logits with a stable softmax pass so downstream model wrappers can present probabilities.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
