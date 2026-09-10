const runtimeReadyEvent = "nlp-stack-text-analysis-ready";
const runtimeErrorEvent = "nlp-stack-text-analysis-error";
const semanticEmbeddingModel = "onnx-community/all-MiniLM-L6-v2-ONNX";
const semanticModelAttemptTimeoutMs = 10_000;
const semanticEmbeddingBatchSize = 32;
let semanticWorkerInstance = null;
let semanticWorkerRequestSequence = 0;
const pendingSemanticWorkerRequests = new Map();

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

function hasImportedSemanticEmbeddings(input) {
  return Array.isArray(input?.importedEmbeddings) && input.importedEmbeddings.length > 0;
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

function hashedSemanticFallback(wasm, request, error) {
  console.warn(
    `Falling back to local hashed semantic analysis for ${request?.operation ?? "semantic analysis"} because ${semanticEmbeddingModel} was unavailable within the interactive analysis budget.`,
    error,
  );
  return fromWasmValue(wasm.runOperation(request));
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

function documentSemanticUnitTexts(wasm, id, text) {
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
  return semanticUnitTexts(surfaceResult(segmentationResponse), text);
}

async function computeModelBackedSemanticMap(wasm, request, text) {
  const input = request?.input ?? {};
  const texts = documentSemanticUnitTexts(wasm, input.id ?? "semantic-doc", text);
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
  if (!text.trim() || hasImportedSemanticEmbeddings(input)) {
    return fromWasmValue(wasm.runOperation(request));
  }

  try {
    return await computeModelBackedSemanticMap(wasm, request, text);
  } catch (error) {
    return hashedSemanticFallback(wasm, request, error);
  }
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

async function computeModelBackedSemanticCorpus(wasm, request, items) {
  const input = request?.input ?? {};
  const texts = Array.from(new Set(items.flatMap((item) =>
    documentSemanticUnitTexts(wasm, item.id, item.text),
  )));
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
  if (!items || hasImportedSemanticEmbeddings(input)) {
    return fromWasmValue(wasm.runOperation(request));
  }

  try {
    return await computeModelBackedSemanticCorpus(wasm, request, items);
  } catch (error) {
    return hashedSemanticFallback(wasm, request, error);
  }
}

const ready = import(new URL("./wasm/moenarch_text_analysis_wasm.js", import.meta.url).href)
  .then(async (wasm) => {
    await wasm.default();
    return {
      packageSurface: () => fromWasmValue(wasm.packageSurface()),
      runOperation: (request) => {
        if (request?.operation === "analysis.semantic-map") {
          return runModelBackedSemanticMap(wasm, request);
        }
        if (request?.operation === "analysis.semantic-corpus") {
          return runModelBackedSemanticCorpus(wasm, request);
        }
        return fromWasmValue(wasm.runOperation(request));
      },
    };
  })
  .catch((error) => {
    globalThis.dispatchEvent(new Event(runtimeErrorEvent));
    throw error;
  });

globalThis.nlpStackTextAnalysis = { ready };
globalThis.dispatchEvent(new Event(runtimeReadyEvent));
