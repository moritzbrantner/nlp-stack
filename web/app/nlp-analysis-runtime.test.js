// @vitest-environment node
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SemanticCorpusPanel } from "../../packages/nlp-app-ui/dist/package-surface/SemanticCorpusPanels.js";
import * as coreWasm from "../public/wasm/moenarch_text_core_wasm.js";
import * as wasm from "../public/wasm/moenarch_text_analysis_wasm.js";

// CI builds these assets before unit tests. Exercise the actual Rust wire contract.
await wasm.default({
  module_or_path: readFileSync(new URL("../public/wasm/moenarch_text_analysis_wasm_bg.wasm", import.meta.url)),
});

await coreWasm.default({
  module_or_path: readFileSync(new URL("../public/wasm/moenarch_text_core_wasm_bg.wasm", import.meta.url)),
});

let runtime;
let workerRequests;
let failModel;

beforeEach(async () => {
  vi.resetModules();
  workerRequests = 0;
  failModel = false;
  vi.stubGlobal("Worker", class {
    listeners = new Map();
    addEventListener(type, listener) { this.listeners.set(type, listener); }
    postMessage({ id, texts }) {
      workerRequests += 1;
      const data = failModel ? { id, error: "Model unavailable" } : {
        id, modelName: "fixture/model", dimensions: 2, vectors: texts.map(() => [1, 0]),
      };
      queueMicrotask(() => this.listeners.get("message")({ data }));
    }
    terminate() {}
  });
  const { createTextAnalysisRuntime } = await import("../public/nlp-semantic-runtime.js");
  runtime = createTextAnalysisRuntime(wasm, coreWasm);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function request(operation, evidence = {}) {
  return {
    operation,
    input: {
      text: "Cats sleep.",
      items: [{ id: "a", text: "Cats sleep." }, { id: "b", text: "Cats sleep." }],
      includeLinguisticGraph: false,
      ...evidence,
    },
  };
}

describe.each(["analysis.semantic-map", "analysis.semantic-corpus"])("%s browser evidence", (operation) => {
  it.each([null, "not an array", {}])("preserves invalid imported embeddings for Rust rejection: %j", async (importedEmbeddings) => {
    const input = request(operation, { importedEmbeddings });
    expect(() => wasm.runOperation(input)).toThrow();
    await expect(runtime.runOperation(input)).rejects.toThrow();
    expect(workerRequests).toBe(0);
  });

  it.each([{}, "not a model", 123])("preserves invalid model metadata for Rust rejection: %j", async (embeddingModel) => {
    const input = request(operation, { embeddingModel });
    expect(() => wasm.runOperation(input)).toThrow();
    await expect(runtime.runOperation(input)).rejects.toThrow();
    expect(workerRequests).toBe(0);
  });

  it("preserves caller-supplied embeddings and model identity", async () => {
    const response = await runtime.runOperation(request(operation, {
      importedEmbeddings: [{ text: "Cats sleep.", vector: [0, 1] }],
      embeddingModel: { name: "caller/model", dimensions: 2 },
    }));
    expect(response.value.result.semantic.embeddingModel.model_name).toBe("caller/model");
    expect(workerRequests).toBe(0);
  });

  it("computes model evidence when the embedding list is empty", async () => {
    const response = await runtime.runOperation(request(operation, { importedEmbeddings: [] }));
    expect(response.value.result.semantic.embeddingModel.model_name).toBe("fixture/model");
    expect(workerRequests).toBeGreaterThan(0);
  });

  it("falls back to local analysis when the model fails", async () => {
    failModel = true;
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const response = await runtime.runOperation(request(operation));
    expect(response.value.result.semantic.embeddingModel.backend).toBe("hashed");
  });
});

it("renders the model identity from the real Rust corpus response", async () => {
  const response = await runtime.runOperation(request("analysis.semantic-corpus"));
  const html = renderToStaticMarkup(createElement(SemanticCorpusPanel, { response }));
  expect(html).toContain("fixture/model");
  expect(html).toContain("Embedding evidence:");
});

it.each(["analysis.document", "analysis.semantic-map", "analysis.semantic-corpus"])("bounds repeated sentences before model work for %s", async (operation) => {
  const text = "Cats sleep. ".repeat(513);
  await expect(runtime.runOperation(request(operation, {
    text, items: [{ id: "a", text }],
  }))).rejects.toThrow(/512 sentences/);
  expect(workerRequests).toBe(0);
});

it("counts repeated sentences across corpus items even with supplied embeddings", async () => {
  await expect(runtime.runOperation(request("analysis.semantic-corpus", {
    items: [{ id: "a", text: "Cats sleep. ".repeat(256) }, { id: "b", text: "Cats sleep. ".repeat(257) }],
    importedEmbeddings: [{ text: "Cats sleep.", vector: [1, 0] }],
  }))).rejects.toThrow(/512 sentences/);
  expect(workerRequests).toBe(0);
});

it("bounds total UTF-8 source bytes across corpus items", async () => {
  await expect(runtime.runOperation(request("analysis.semantic-corpus", {
    items: [{ id: "a", text: "é".repeat(65_537) }, { id: "b", text: "é".repeat(65_537) }],
  }))).rejects.toThrow(/256 KiB/);
  expect(workerRequests).toBe(0);
});

it("accepts exactly 512 repeated sentences without truncation", async () => {
  const response = await runtime.runOperation(request("analysis.semantic-map", {
    text: "Cats sleep. ".repeat(512),
  }));
  expect(response.value.result.semantic.timeline).toHaveLength(512);
});
