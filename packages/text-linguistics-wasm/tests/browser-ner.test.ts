import { expect, test } from "bun:test";
import { projectBrowserNerSegments } from "../browser-ner-core.js";

type Prediction = { word: string; entity: string; score: number };

const p = (word: string, entity: string, score = 0.9): Prediction => ({ word, entity, score });

const fixtures: Record<string, Prediction[]> = {
  "Talk with Alan Turing.": [
    p("Talk", "O"),
    p("with", "O"),
    p("Alan", "B-PER", 0.99),
    p("Turin", "I-PER", 0.97),
    p("##g", "O", 0.5),
    p(".", "O"),
  ],
  "María visitó Madrid ayer.": [
    p("Mar", "B-PER", 0.8),
    p("##ía", "O"),
    p("visit", "O"),
    p("##ó", "O"),
    p("Madrid", "B-LOC", 0.95),
    p("ayer", "O"),
    p(".", "O"),
  ],
};

class FakeWorker {
  static created = 0;
  static failNextLoad = false;
  static hang = false;
  terminated = false;
  listeners = new Map<string, (event: any) => void>();
  constructor() {
    FakeWorker.created += 1;
  }
  addEventListener(type: string, listener: (event: any) => void) {
    this.listeners.set(type, listener);
  }
  postMessage(message: any) {
    if (FakeWorker.hang) return;
    const emit = (data: any) => queueMicrotask(() => !this.terminated && this.listeners.get("message")?.({ data }));
    emit({ type: "progress", id: message.id, detail: { status: "progress", file: "onnx/model_quantized.onnx", progress: 50 } });
    if (FakeWorker.failNextLoad) {
      FakeWorker.failNextLoad = false;
      emit({ type: "error", id: message.id, stage: "model", message: "network error fetching model" });
      return;
    }
    if (message.type === "prepare") {
      emit({ type: "ready", id: message.id });
      return;
    }
    const predictions = message.segments.map((segment: any) => fixtures[segment.text] ?? []);
    emit({ type: "result", id: message.id, ...projectBrowserNerSegments(message.segments, predictions) });
  }
  terminate() {
    this.terminated = true;
  }
}

async function adapter() {
  const { createBrowserNer } = await import("../browser-ner.js");
  return createBrowserNer({ createWorker: () => new FakeWorker() as unknown as Worker });
}

test("importing the adapter performs no network or worker requests", async () => {
  const originalFetch = globalThis.fetch;
  let requests = 0;
  globalThis.fetch = (() => {
    requests += 1;
    throw new Error("unexpected fetch");
  }) as unknown as typeof fetch;
  try {
    FakeWorker.created = 0;
    const module = await import("../browser-ner.js");
    const ner = module.createBrowserNer({ createWorker: () => new FakeWorker() as unknown as Worker });
    expect(ner.capabilities().model.revision).toMatch(/^[0-9a-f]{40}$/);
    expect(requests).toBe(0);
    expect(FakeWorker.created).toBe(0);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("sub-word predictions become whole-word entities with exact source spans", async () => {
  const { entitiesFromTokenPredictions } = await import("../browser-ner.js");
  expect(entitiesFromTokenPredictions("Talk with Alan Turing.", fixtures["Talk with Alan Turing."]).entities).toEqual([
    { label: "PER", text: "Alan Turing", start: 10, end: 21, score: 0.98 },
  ]);
  const helios = entitiesFromTokenPredictions("Helios conference", [
    p("He", "B-MISC", 0.7),
    p("##lio", "B-ORG", 0.6),
    p("##s", "I-MISC", 0.5),
    p("conference", "O"),
  ]);
  expect(helios.entities.map(({ label, text }) => [label, text])).toEqual([["MISC", "Helios"]]);
  const unknown = entitiesFromTokenPredictions("Alice ☃ Bob", [p("Alice", "B-PER"), p("[UNK]", "O"), p("Bob", "B-PER")]);
  expect(unknown.entities.map(({ text }) => text)).toEqual(["Alice", "Bob"]);
});

test("analyze returns provenance and absolute UTF-8 spans for supported languages", async () => {
  FakeWorker.created = 0;
  const ner = await adapter();
  const result = await ner.analyze({
    language: "en-US",
    segments: [
      { id: "cue-1", text: "Talk with Alan Turing.", byteStart: 100 },
      { id: "cue-2", text: "María visitó Madrid ayer.", byteStart: 200 },
    ],
  });
  expect(result.status).toBe("completed");
  expect(result.language).toBe("en");
  expect(result.producer.module).toBe("browser-ner");
  expect(result.model.id).toBe("Xenova/distilbert-base-multilingual-cased-ner-hrl");
  expect(result.model.supportedLanguages).toEqual(["en", "de", "es"]);
  const source = new TextEncoder().encode("María visitó Madrid ayer.");
  const madrid = result.entities.find((entity) => entity.text === "Madrid")!;
  expect(madrid).toMatchObject({ segmentId: "cue-2", label: "LOC" });
  expect(new TextDecoder().decode(source.slice(madrid.byteStart - 200, madrid.byteEnd - 200))).toBe("Madrid");
  expect(result.entities.find((entity) => entity.label === "PER" && entity.segmentId === "cue-2")?.text).toBe("María");
  expect(result.entities.find((entity) => entity.segmentId === "cue-1")).toMatchObject({
    text: "Alan Turing",
    byteStart: 110,
    byteEnd: 121,
  });
  expect(FakeWorker.created).toBe(1);
});

test("German and Spanish segments are labeled by the multilingual model", async () => {
  FakeWorker.created = 0;
  const ner = await adapter();
  for (const language of ["de-DE", "es"]) {
    const result = await ner.analyze({ language, segments: [{ id: "s", text: "María visitó Madrid ayer." }] });
    expect(result.status).toBe("completed");
    expect(result.entities.map(({ label, text }) => [label, text])).toContainEqual(["LOC", "Madrid"]);
  }
  expect(FakeWorker.created).toBe(1);
});

test("the declared languages match the gold NER fixtures, whose spans are exact", async () => {
  const { BROWSER_NER_MODEL } = await import("../browser-ner.js");
  const corpus = await Bun.file(new URL("../../../evaluation/corpora/named-entities-multilingual-v1.jsonl", import.meta.url)).text();
  const cases = corpus.split("\n").filter(Boolean).map((line) => JSON.parse(line));
  expect([...new Set(cases.map((item) => item.language))].sort()).toEqual([...BROWSER_NER_MODEL.supportedLanguages].sort());
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  for (const item of cases) {
    const bytes = encoder.encode(item.text);
    for (const entity of item.entities) {
      expect(decoder.decode(bytes.slice(entity.byteStart, entity.byteEnd))).toBe(entity.text);
      expect(["PER", "ORG", "LOC"]).toContain(entity.label);
    }
  }
});

test("unsupported or unknown languages are reported without loading the model", async () => {
  FakeWorker.created = 0;
  const ner = await adapter();
  for (const language of ["fr", null]) {
    const result = await ner.analyze({ language, segments: [{ text: "Anna besuchte München." }] });
    expect(result.status).toBe("unsupported-language");
    expect(result.entities).toEqual([]);
  }
  expect(FakeWorker.created).toBe(0);
});

test("model load failures surface as errors and a retry succeeds", async () => {
  const ner = await adapter();
  const progress: unknown[] = [];
  FakeWorker.failNextLoad = true;
  await expect(ner.prepare({ onProgress: (event) => progress.push(event) })).rejects.toMatchObject({
    name: "BrowserNerError",
    stage: "model",
  });
  expect(progress.length).toBeGreaterThan(0);
  await expect(ner.prepare()).resolves.toMatchObject({ model: { id: "Xenova/distilbert-base-multilingual-cased-ner-hrl" } });
});

const fakeTransformersUrl = new URL("./fixtures/fake-transformers.js", import.meta.url).href;
let workerInstances = 0;

/** Loads a fresh copy of the real worker module against the fake Transformers.js module. */
async function startWorker() {
  let listener: ((event: any) => void) | undefined;
  const waiting = new Map<number, (data: any) => void>();
  (globalThis as any).self = {
    addEventListener: (_type: string, fn: (event: any) => void) => (listener = fn),
    postMessage: (data: any) => data.type !== "progress" && waiting.get(data.id)?.(data),
  };
  await import(`../browser-ner-worker.js?instance=${++workerInstances}`);
  let id = 0;
  return (type: string, segments?: unknown[]) =>
    new Promise<any>((resolve) => {
      waiting.set(++id, resolve);
      listener!({ data: { type, id, segments, transformersModuleUrl: fakeTransformersUrl } });
    });
}

test("a cached model loads offline at the pinned revision without any startup request", async () => {
  const { BROWSER_NER_MODEL } = await import("../browser-ner.js");
  const transformers = await import(fakeTransformersUrl);
  transformers.state.cache.clear();
  transformers.restart({ online: true });
  expect(await (await startWorker())("prepare")).toMatchObject({ type: "ready" });
  expect(transformers.state.requests.length).toBeGreaterThan(0);
  for (const url of transformers.state.requests) {
    expect(url).toContain(`/resolve/${BROWSER_NER_MODEL.revision}/`);
  }

  transformers.restart({ online: false });
  const result = await (await startWorker())("analyze", [{ id: "s", text: "Talk with Alan Turing.", byteStart: 0 }]);
  expect(result).toMatchObject({ type: "result", segments: [{ id: "s", entityCount: 0 }] });
  expect(transformers.state.requests).toEqual([]);

  transformers.state.cache.clear();
  transformers.restart({ online: false });
  const cold = await (await startWorker())("prepare");
  expect(cold).toMatchObject({ type: "error", stage: "model" });
  expect(cold.message).toContain("download failed (offline?)");
});

test("a canceled request rejects and never completes; the next request uses a fresh worker", async () => {
  FakeWorker.created = 0;
  const ner = await adapter();
  const controller = new AbortController();
  FakeWorker.hang = true;
  const canceled = ner.analyze({ language: "en", segments: [{ text: "Talk with Alan Turing." }] }, { signal: controller.signal });
  controller.abort();
  await expect(canceled).rejects.toMatchObject({ name: "AbortError" });
  FakeWorker.hang = false;
  const retried = await ner.analyze({ language: "en", segments: [{ text: "Talk with Alan Turing." }] });
  expect(retried.entities.map((entity) => entity.text)).toEqual(["Alan Turing"]);
  expect(FakeWorker.created).toBe(2);
});
