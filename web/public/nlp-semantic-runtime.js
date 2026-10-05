import { assertAnalysisTextBudget, assertAnalysisSentenceBudget } from "./nlp-analysis-budget.js";

const semanticEmbeddingModel = "onnx-community/all-MiniLM-L6-v2-ONNX";
const semanticModelAttemptTimeoutMs = 10_000;
const semanticEmbeddingBatchSize = 32;
const namedEntityModel = "onnx-community/distilbert-NER-ONNX";
const namedEntityModelAttemptTimeoutMs = 45_000;
let semanticWorkerInstance = null;
let semanticWorkerRequestSequence = 0;
const pendingSemanticWorkerRequests = new Map();
let entityWorkerInstance = null;
let entityWorkerRequestSequence = 0;
const pendingEntityWorkerRequests = new Map();

function fromWasmValue(value) {
  if (value instanceof Map) {
    return Object.fromEntries(
      Array.from(value.entries(), ([key, entry]) => [key, fromWasmValue(entry)]),
    );
  }

  if (Array.isArray(value)) {
    return value.map(fromWasmValue);
  }

  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, fromWasmValue(entry)]),
    );
  }

  return value;
}

function surfaceResult(response) {
  const value = response?.value;
  return value?.result ?? value ?? {};
}

function semanticUnitTexts(documentResult, sourceText) {
  const core = documentResult?.core ?? {};
  const sentenceTexts = Array.isArray(core.sentences)
    ? core.sentences.map((sentence) => sentence?.text).filter((text) => typeof text === "string" && text.length > 0)
    : [];
  const paragraphTexts = Array.isArray(core.paragraphs)
    ? core.paragraphs.map((paragraph) => paragraph?.text).filter((text) => typeof text === "string" && text.length > 0)
    : [];
  return Array.from(new Set([...sentenceTexts, ...paragraphTexts, sourceText]));
}

function hasSuppliedSemanticEvidence(input) {
  // Preserve caller-owned metadata and malformed evidence for Rust validation.
  // An empty list without model metadata still means no supplied embeddings.
  return Object.hasOwn(input, "embeddingModel")
    || (Object.hasOwn(input, "importedEmbeddings")
      && (!Array.isArray(input.importedEmbeddings) || input.importedEmbeddings.length > 0));
}

function semanticWorkerError(error) {
  return error instanceof Error ? error : new Error(String(error));
}

function resetSemanticWorker(error) {
  const worker = semanticWorkerInstance;
  semanticWorkerInstance = null;
  worker?.terminate();

  const failure = semanticWorkerError(error);
  for (const pending of pendingSemanticWorkerRequests.values()) {
    pending.reject(failure);
  }
  pendingSemanticWorkerRequests.clear();
}

function semanticWorker() {
  if (semanticWorkerInstance) {
    return semanticWorkerInstance;
  }
  if (typeof Worker === "undefined") {
    throw new Error("Web Workers are unavailable in this browser.");
  }

  const worker = new Worker(new URL("./nlp-semantic-worker.js", import.meta.url), { type: "module" });
  semanticWorkerInstance = worker;
  worker.addEventListener("message", (event) => {
    const id = event.data?.id;
    const pending = pendingSemanticWorkerRequests.get(id);
    if (!pending) {
      return;
    }
    pendingSemanticWorkerRequests.delete(id);
    if (event.data?.error) {
      pending.reject(new Error(event.data.error));
    } else {
      pending.resolve(event.data);
    }
  });
  worker.addEventListener("error", (event) => {
    resetSemanticWorker(event.message || "Semantic model worker failed.");
  });
  return worker;
}

function requestSemanticEmbeddings(texts) {
  const id = ++semanticWorkerRequestSequence;
  const promise = new Promise((resolve, reject) => {
    let worker;
    try {
      worker = semanticWorker();
    } catch (error) {
      reject(semanticWorkerError(error));
      return;
    }

    pendingSemanticWorkerRequests.set(id, { resolve, reject });
    try {
      worker.postMessage({ id, texts });
    } catch (error) {
      pendingSemanticWorkerRequests.delete(id);
      reject(semanticWorkerError(error));
    }
  });
  return { id, promise };
}

function resetEntityWorker(error) {
  const worker = entityWorkerInstance;
  entityWorkerInstance = null;
  worker?.terminate();

  const failure = semanticWorkerError(error);
  for (const pending of pendingEntityWorkerRequests.values()) {
    pending.reject(failure);
  }
  pendingEntityWorkerRequests.clear();
}

function entityWorker() {
  if (entityWorkerInstance) return entityWorkerInstance;
  if (typeof Worker === "undefined") {
    throw new Error("Web Workers are unavailable in this browser.");
  }

  const worker = new Worker(new URL("./nlp-entity-worker.js", import.meta.url), { type: "module" });
  entityWorkerInstance = worker;
  worker.addEventListener("message", (event) => {
    const id = event.data?.id;
    const pending = pendingEntityWorkerRequests.get(id);
    if (!pending) return;
    pendingEntityWorkerRequests.delete(id);
    if (event.data?.error) pending.reject(new Error(event.data.error));
    else pending.resolve(event.data);
  });
  worker.addEventListener("error", (event) => {
    resetEntityWorker(event.message || "Named entity model worker failed.");
  });
  return worker;
}

function requestNamedEntities(segments) {
  const id = ++entityWorkerRequestSequence;
  const promise = new Promise((resolve, reject) => {
    let worker;
    try {
      worker = entityWorker();
    } catch (error) {
      reject(semanticWorkerError(error));
      return;
    }
    pendingEntityWorkerRequests.set(id, { resolve, reject });
    try {
      worker.postMessage({ id, segments });
    } catch (error) {
      pendingEntityWorkerRequests.delete(id);
      reject(semanticWorkerError(error));
    }
  });
  return { id, promise };
}

async function modelNamedEntities(documentResult) {
  const sentences = Array.isArray(documentResult?.core?.sentences) ? documentResult.core.sentences : [];
  const segments = sentences
    .filter((sentence) => typeof sentence?.text === "string" && sentence.text.length > 0)
    .map((sentence) => ({
      text: sentence.text,
      byteStart: Number(sentence?.span?.byte_start ?? 0),
    }));
  if (segments.length === 0) {
    return { modelName: namedEntityModel, entities: [] };
  }

  let timeoutId;
  const request = requestNamedEntities(segments);
  const deadline = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      const failure = new Error(`Named entity model attempt exceeded ${namedEntityModelAttemptTimeoutMs}ms.`);
      resetEntityWorker(failure);
      reject(failure);
    }, namedEntityModelAttemptTimeoutMs);
  });
  try {
    const result = await Promise.race([request.promise, deadline]);
    if (!Array.isArray(result?.entities)) {
      throw new Error(`Hugging Face NER model ${namedEntityModel} returned invalid entity evidence.`);
    }
    return {
      modelName: result.modelName || namedEntityModel,
      entities: result.entities,
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

async function runModelBackedDocumentAnalysis(wasm, request) {
  const response = fromWasmValue(wasm.runOperation(request));
  const result = surfaceResult(response);
  const namedEntities = await modelNamedEntities(result);
  result.namedEntities = namedEntities.entities;
  result.namedEntityModel = { name: namedEntities.modelName };
  return response;
}

function semanticModelDeadlineError() {
  return new Error(`Semantic model attempt exceeded ${semanticModelAttemptTimeoutMs}ms.`);
}

async function withinModelAttemptDeadline(request, timeoutMs = semanticModelAttemptTimeoutMs) {
  let timeoutId;
  const boundedTimeoutMs = Math.max(1, Math.min(timeoutMs, semanticModelAttemptTimeoutMs));
  const deadline = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      const failure = semanticModelDeadlineError();
      resetSemanticWorker(failure);
      reject(failure);
    }, boundedTimeoutMs);
  });
  try {
    return await Promise.race([request.promise, deadline]);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function modelEmbeddingEvidence(texts) {
  if (!Array.isArray(texts) || texts.length === 0) {
    throw new Error("Semantic model analysis requires at least one text unit.");
  }

  const deadlineAt = Date.now() + semanticModelAttemptTimeoutMs;
  const vectors = [];
  let dimensions = null;
  let modelName = null;

  for (let offset = 0; offset < texts.length; offset += semanticEmbeddingBatchSize) {
    const remainingMs = deadlineAt - Date.now();
    if (remainingMs <= 0) {
      const failure = semanticModelDeadlineError();
      resetSemanticWorker(failure);
      throw failure;
    }

    const batchTexts = texts.slice(offset, offset + semanticEmbeddingBatchSize);
    const modelResult = await withinModelAttemptDeadline(
      requestSemanticEmbeddings(batchTexts),
      remainingMs,
    );
    const batchVectors = modelResult?.vectors;
    const batchDimensions = modelResult?.dimensions;
    const batchModelName = modelResult?.modelName || semanticEmbeddingModel;
    if (
      !Array.isArray(batchVectors)
      || batchVectors.length !== batchTexts.length
      || !Array.isArray(batchVectors[0])
      || !Number.isInteger(batchDimensions)
      || batchDimensions <= 0
      || batchVectors.some((vector) => !Array.isArray(vector) || vector.length !== batchDimensions)
    ) {
      throw new Error(`Hugging Face semantic model ${semanticEmbeddingModel} returned an unexpected embedding shape.`);
    }
    if (dimensions !== null && dimensions !== batchDimensions) {
      throw new Error(`Hugging Face semantic model ${semanticEmbeddingModel} changed embedding dimensions between batches.`);
    }
    if (modelName !== null && modelName !== batchModelName) {
      throw new Error(`Hugging Face semantic model identity changed between embedding batches.`);
    }

    dimensions = batchDimensions;
    modelName = batchModelName;
    vectors.push(...batchVectors);
  }

  return {
    importedEmbeddings: texts.map((text, index) => ({ text, vector: vectors[index] })),
    embeddingModel: {
      name: modelName || semanticEmbeddingModel,
      dimensions,
      maxTokens: 256,
    },
  };
}

function documentSemanticUnits(wasm, id, text) {
  const segmentationResponse = fromWasmValue(
    wasm.runOperation({
      operation: "analysis.document",
      input: {
        id,
        text,
        profile: "deterministic",
        linguistics: { mode: "off" },
        embedding: { mode: "off" },
      },
    }),
  );
  const result = surfaceResult(segmentationResponse);
  return {
    texts: semanticUnitTexts(result, text),
    sentenceCount: result.core?.sentences?.length ?? 0,
  };
}

async function computeModelBackedSemanticMap(wasm, request, texts) {
  const input = request?.input ?? {};
  const evidence = await modelEmbeddingEvidence(texts);

  return fromWasmValue(
    wasm.runOperation({
      ...request,
      input: {
        ...input,
        ...evidence,
      },
    }),
  );
}

async function runModelBackedSemanticMap(wasm, request) {
  const input = request?.input ?? {};
  const text = typeof input.text === "string" ? input.text : "";
  const units = text.trim() ? documentSemanticUnits(wasm, input.id ?? "semantic-doc", text) : null;
  assertAnalysisSentenceBudget(units?.sentenceCount ?? 0);
  if (!units || hasSuppliedSemanticEvidence(input)) {
    return fromWasmValue(wasm.runOperation(request));
  }

  return computeModelBackedSemanticMap(wasm, request, units.texts);
}

function validSemanticCorpusItems(input) {
  const items = input?.items;
  if (!Array.isArray(items) || items.length === 0) {
    return null;
  }
  const ids = new Set();
  for (const item of items) {
    if (
      typeof item?.id !== "string"
      || !item.id.trim()
      || typeof item?.text !== "string"
      || !item.text.trim()
      || ids.has(item.id)
    ) {
      return null;
    }
    ids.add(item.id);
  }
  return items;
}

async function computeModelBackedSemanticCorpus(wasm, request, texts) {
  const input = request?.input ?? {};
  const evidence = await modelEmbeddingEvidence(texts);

  return fromWasmValue(
    wasm.runOperation({
      ...request,
      input: {
        ...input,
        ...evidence,
      },
    }),
  );
}

async function runModelBackedSemanticCorpus(wasm, request) {
  const input = request?.input ?? {};
  const items = validSemanticCorpusItems(input);
  let sentenceCount = 0;
  const texts = new Set();
  for (const item of items ?? []) {
    const units = documentSemanticUnits(wasm, item.id, item.text);
    sentenceCount += units.sentenceCount;
    // Count occurrences before deduplicating embedding inputs.
    assertAnalysisSentenceBudget(sentenceCount);
    for (const text of units.texts) texts.add(text);
  }
  if (!items || hasSuppliedSemanticEvidence(input)) {
    return fromWasmValue(wasm.runOperation(request));
  }

  return computeModelBackedSemanticCorpus(wasm, request, Array.from(texts));
}

export function createTextAnalysisRuntime(wasm, coreWasm) {
  return {
    packageSurface: () => fromWasmValue(wasm.packageSurface()),
    runOperation: async (request) => {
      assertAnalysisTextBudget(request);
      const input = request?.input;
      const texts = request?.operation === "analysis.semantic-corpus" && Array.isArray(input?.items)
        ? input.items.map((item) => item?.text)
        : [input?.text];
      let sentenceCount = 0;
      for (const text of texts) {
        if (typeof text !== "string") continue;
        // Count with the same Rust splitter before lexical, linguistic or semantic work.
        const statistics = surfaceResult(fromWasmValue(coreWasm.runOperation({
          operation: "text.statistics", input: { text },
        })));
        sentenceCount += statistics.sentenceCount;
        assertAnalysisSentenceBudget(sentenceCount);
      }
      if (request?.operation === "analysis.document" && input?.namedEntities?.mode === "huggingFace") {
        return runModelBackedDocumentAnalysis(wasm, request);
      }
      if (request?.operation === "analysis.semantic-map") {
        return runModelBackedSemanticMap(wasm, request);
      }
      if (request?.operation === "analysis.semantic-corpus") {
        return runModelBackedSemanticCorpus(wasm, request);
      }
      return fromWasmValue(wasm.runOperation(request));
    },
  };
}
