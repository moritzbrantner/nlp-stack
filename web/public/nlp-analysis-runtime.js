const runtimeReadyEvent = "nlp-stack-text-analysis-ready";
const runtimeErrorEvent = "nlp-stack-text-analysis-error";
const semanticEmbeddingModel = "onnx-community/all-MiniLM-L6-v2-ONNX";
const semanticModelAttemptTimeoutMs = 10_000;
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

async function withinModelAttemptDeadline(request) {
  let timeoutId;
  const deadline = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      pendingSemanticWorkerRequests.delete(request.id);
      reject(new Error(`Semantic model attempt exceeded ${semanticModelAttemptTimeoutMs}ms.`));
    }, semanticModelAttemptTimeoutMs);
  });
  try {
    return await Promise.race([request.promise, deadline]);
  } finally {
    clearTimeout(timeoutId);
  }
}

function hashedSemanticMapFallback(wasm, request, error) {
  console.warn(
    `Falling back to the local hashed semantic map because ${semanticEmbeddingModel} was unavailable within the interactive analysis budget.`,
    error,
  );
  return fromWasmValue(wasm.runOperation(request));
}

async function computeModelBackedSemanticMap(wasm, request, text) {
  const input = request?.input ?? {};
  const segmentationResponse = fromWasmValue(
    wasm.runOperation({
      operation: "analysis.document",
      input: {
        id: input.id ?? "semantic-doc",
        text,
        profile: "deterministic",
        linguistics: { mode: "off" },
        embedding: { mode: "off" },
      },
    }),
  );
  const texts = semanticUnitTexts(surfaceResult(segmentationResponse), text);
  const modelResult = await withinModelAttemptDeadline(requestSemanticEmbeddings(texts));
  const vectors = modelResult?.vectors;
  const dimensions = modelResult?.dimensions;
  if (
    !Array.isArray(vectors)
    || vectors.length !== texts.length
    || !Array.isArray(vectors[0])
    || !Number.isInteger(dimensions)
    || dimensions <= 0
    || vectors.some((vector) => !Array.isArray(vector) || vector.length !== dimensions)
  ) {
    throw new Error(`Hugging Face semantic model ${semanticEmbeddingModel} returned an unexpected embedding shape.`);
  }

  return fromWasmValue(
    wasm.runOperation({
      ...request,
      input: {
        ...input,
        importedEmbeddings: texts.map((unitText, index) => ({
          text: unitText,
          vector: vectors[index],
        })),
        embeddingModel: {
          name: modelResult.modelName || semanticEmbeddingModel,
          dimensions,
          maxTokens: 256,
        },
      },
    }),
  );
}

async function runModelBackedSemanticMap(wasm, request) {
  const input = request?.input ?? {};
  const text = typeof input.text === "string" ? input.text : "";
  if (!text.trim()) {
    return fromWasmValue(wasm.runOperation(request));
  }

  try {
    return await computeModelBackedSemanticMap(wasm, request, text);
  } catch (error) {
    return hashedSemanticMapFallback(wasm, request, error);
  }
}

const ready = import(new URL("./wasm/moenarch_text_analysis_wasm.js", import.meta.url).href)
  .then(async (wasm) => {
    await wasm.default();
    return {
      packageSurface: () => fromWasmValue(wasm.packageSurface()),
      runOperation: (request) => request?.operation === "analysis.semantic-map"
        ? runModelBackedSemanticMap(wasm, request)
        : fromWasmValue(wasm.runOperation(request)),
    };
  })
  .catch((error) => {
    globalThis.dispatchEvent(new Event(runtimeErrorEvent));
    throw error;
  });

globalThis.nlpStackTextAnalysis = { ready };
globalThis.dispatchEvent(new Event(runtimeReadyEvent));
