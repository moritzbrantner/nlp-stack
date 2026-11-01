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
    constructor(url) { this.url = String(url); }
    addEventListener(type, listener) { this.listeners.set(type, listener); }
    postMessage({ id, texts, segments }) {
      workerRequests += 1;
      let data;
      const ner = this.url.includes("browser-ner-worker.js");
      if (failModel) {
        data = ner ? { type: "error", id, stage: "model", message: "Model unavailable" } : { id, error: "Model unavailable" };
      } else if (ner) {
        // Labels a leading "Alice" in each segment.
        const labelled = segments.filter((segment) => segment.text.startsWith("Alice"));
        data = {
          type: "result",
          id,
          entities: labelled.map((segment) => ({
            segmentId: segment.id, label: "PER", text: "Alice", score: 0.99, start: 0, end: 5,
            byteStart: segment.byteStart, byteEnd: segment.byteStart + 5,
          })),
          segments: segments.map((segment) => ({
            id: segment.id, entityCount: Number(segment.text.startsWith("Alice")), truncated: false,
          })),
        };
      } else {
        data = { id, modelName: "fixture/model", dimensions: 2, vectors: texts.map(() => [1, 0]) };
      }
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

  it("fails instead of substituting heuristic semantic analysis when the model fails", async () => {
    failModel = true;
    await expect(runtime.runOperation(request(operation))).rejects.toThrow("Model unavailable");
  });
});

function documentRequest(text) {
  return {
    operation: "analysis.document",
    input: {
      id: "doc",
      text,
      linguistics: { mode: "heuristicFast" },
      embedding: { mode: "off" },
      namedEntities: { mode: "huggingFace" },
    },
  };
}

it("routes Hugging Face named entities by each sentence's detected language", async () => {
  const text = "Alice works in Berlin with her team. Alice wohnt seit drei Jahren in Kreuzberg.";
  const response = await runtime.runOperation(documentRequest(text));
  const { namedEntityModel, namedEntities, linguistic } = response.value.result;

  expect(linguistic.sentences.map((sentence) => sentence.language)).toEqual(["en", "de"]);
  expect(namedEntityModel).toMatchObject({
    name: "Xenova/distilbert-base-multilingual-cased-ner-hrl",
    revision: "c2a4dbf593c57f47004c5bc2d3770d311aee9c43",
    supportedLanguages: ["en", "de", "es"],
    sentences: [
      { sentenceIndex: 0, language: "en", languageSource: "detected", status: "completed", entityCount: 1 },
      { sentenceIndex: 1, language: "de", languageSource: "detected", status: "completed", entityCount: 1 },
    ],
  });
  // One model request per language.
  expect(workerRequests).toBe(2);
  const secondStart = new TextEncoder().encode("Alice works in Berlin with her team. ").length;
  expect(namedEntities).toEqual([
    expect.objectContaining({ text: "Alice", kind: "PER", language: "en", sentenceIndex: 0, span: { byte_start: 0, byte_end: 5 } }),
    expect.objectContaining({ text: "Alice", kind: "PER", language: "de", sentenceIndex: 1, span: { byte_start: secondStart, byte_end: secondStart + 5 } }),
  ]);
});

it("reports undecided and unsupported sentences without running the model on them", async () => {
  const response = await runtime.runOperation(documentRequest("Ok. Le chat dort sur le canapé depuis ce matin."));
  expect(response.value.result.namedEntityModel.sentences).toEqual([
    expect.objectContaining({ language: null, status: "unknown-language", entityCount: 0 }),
    expect.objectContaining({ language: "fr", status: "unsupported-language", entityCount: 0 }),
  ]);
  expect(response.value.result.namedEntities).toEqual([]);
  expect(workerRequests).toBe(0);
});

it("fails document analysis instead of falling back to heuristic entities when NER fails", async () => {
  failModel = true;
  await expect(runtime.runOperation({
    operation: "analysis.document",
    input: {
      id: "doc",
      text: "Alice works in Berlin with her team.",
      linguistics: { mode: "heuristicFast" },
      embedding: { mode: "off" },
      namedEntities: { mode: "huggingFace" },
    },
  })).rejects.toThrow("Model unavailable");
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
