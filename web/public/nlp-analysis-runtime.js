import { createAnalysisWorkerRuntime } from "./nlp-analysis-client.js";

// WASM initialization and all analysis run off the UI thread.
const ready = Promise.resolve({ createSession: createAnalysisWorkerRuntime });
globalThis.nlpStackTextAnalysis = { ready };
globalThis.dispatchEvent(new Event("nlp-stack-text-analysis-ready"));
