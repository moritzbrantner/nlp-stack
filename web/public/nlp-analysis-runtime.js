const runtimeReadyEvent = "nlp-stack-text-analysis-ready";
const runtimeErrorEvent = "nlp-stack-text-analysis-error";
const transformersModuleUrl = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const semanticEmbeddingModel = "onnx-community/all-MiniLM-L6-v2-ONNX";
const semanticModelAttemptTimeoutMs = 10_000;
let semanticExtractorPromise = null;

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

async function semanticExtractor() {
  if (!semanticExtractorPromise) {
    semanticExtractorPromise = import(transformersModuleUrl)
      .then(({ pipeline }) => pipeline("feature-extraction", semanticEmbeddingModel, { dtype: "q4" }))
      .catch((error) => {
        semanticExtractorPromise = null;
        throw error;
      });
  }
  return semanticExtractorPromise;
}

async function withinModelAttemptDeadline(operation) {
  let timeoutId;
  const deadline = new Promise((_, reject) => {
    timeoutId = setTimeout(
      () => reject(new Error(`Semantic model attempt exceeded ${semanticModelAttemptTimeoutMs}ms.`)),
      semanticModelAttemptTimeoutMs,
    );
  });
  try {
    return await Promise.race([operation, deadline]);
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
  const extractor = await semanticExtractor();
  const output = await extractor(texts, { pooling: "mean", normalize: true });
  const vectors = output.tolist();
  if (!Array.isArray(vectors) || vectors.length !== texts.length || !Array.isArray(vectors[0])) {
    throw new Error(`Hugging Face semantic model ${semanticEmbeddingModel} returned an unexpected embedding shape.`);
  }
  const dimensions = vectors[0].length;

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
          name: semanticEmbeddingModel,
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
    return await withinModelAttemptDeadline(computeModelBackedSemanticMap(wasm, request, text));
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
