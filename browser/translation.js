const DEFAULT_TRANSFORMERS_MODULE_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const DEFAULT_BROWSER_TRANSLATION_MODEL_ID = "onnx-community/opus-mt-de-en";
const DEFAULT_BROWSER_TRANSLATION_DTYPE = "q4";
const DEFAULT_SOURCE_LANGUAGE = "de";
const DEFAULT_TARGET_LANGUAGE = "en";
const DEFAULT_MAX_NEW_TOKENS = 256;
const BROWSER_TRANSLATION_RUNTIME_ID = "nlp-stack-transformers-js-webgpu-translation";

let transformersModulePromise;

export function browserTranslationCapabilities() {
  return {
    runtime: BROWSER_TRANSLATION_RUNTIME_ID,
    requiredAcceleration: "webgpu",
    modelProvisioning: "browser-cache",
    defaultModelId: DEFAULT_BROWSER_TRANSLATION_MODEL_ID,
    defaultDtype: DEFAULT_BROWSER_TRANSLATION_DTYPE,
    defaultSourceLanguage: DEFAULT_SOURCE_LANGUAGE,
    defaultTargetLanguage: DEFAULT_TARGET_LANGUAGE,
    input: {
      kind: "text-segments",
      identity: ["string", "number"],
    },
    features: {
      translation: true,
      orderedSegments: true,
      callerOwnedTiming: true,
    },
    fallbacks: {
      server: false,
      python: false,
      cpu: false,
    },
  };
}

export function createBrowserTranslationAdapter(options = {}) {
  const loadPipeline = options.loadPipeline ?? loadBrowserTranslationPipeline;
  const webGpuAvailable = options.webGpuAvailable ?? supportsBrowserTranslation;
  const pipelineCache = new Map();

  async function translateSegments(segments, request = {}) {
    const normalizedSegments = normalizeSegments(segments);
    const config = normalizeRequest(request);
    if (normalizedSegments.length === 0) {
      return translationResult(normalizedSegments, [], config);
    }

    if (!(await webGpuAvailable())) {
      throw new Error(
        "WebGPU is required for browser translation. No CPU, server, or Python fallback is used.",
      );
    }

    emitProgress(request, {
      stage: "translate",
      message: `Preparing browser translation for ${normalizedSegments.length} segment${normalizedSegments.length === 1 ? "" : "s"}…`,
      detail: {
        modelId: config.modelId,
        sourceLanguage: config.sourceLanguage,
        targetLanguage: config.targetLanguage,
      },
    });

    const pipeline = await requirePipeline(config, request);
    const texts = normalizedSegments.map((segment) => segment.text);
    let output;
    try {
      output = await pipeline(texts.length === 1 ? texts[0] : texts, {
        max_new_tokens: config.maxNewTokens,
      });
    } catch (error) {
      throw new Error(`Browser translation failed: ${formatError(error)}`);
    }

    const translations = normalizeTranslationOutput(output, normalizedSegments.length);
    emitProgress(request, {
      stage: "translate",
      message: `Translated ${translations.length} segment${translations.length === 1 ? "" : "s"} locally.`,
      detail: {
        modelId: config.modelId,
        sourceLanguage: config.sourceLanguage,
        targetLanguage: config.targetLanguage,
        status: "done",
      },
    });
    return translationResult(normalizedSegments, translations, config);
  }

  async function requirePipeline(config, request) {
    const key = `${config.modelId}\n${config.dtype}`;
    let promise = pipelineCache.get(key);
    if (!promise) {
      emitProgress(request, {
        stage: "model",
        message: `Loading browser translation model ${config.modelId}…`,
        detail: { modelId: config.modelId, dtype: config.dtype, status: "loading" },
      });
      promise = Promise.resolve(
        loadPipeline({
          modelId: config.modelId,
          dtype: config.dtype,
          onProgress: (detail) =>
            emitProgress(request, {
              stage: "model",
              message: `Loading browser translation model ${config.modelId}…`,
              detail,
            }),
        }),
      ).catch((error) => {
        pipelineCache.delete(key);
        throw new Error(`Unable to load browser translation model: ${formatError(error)}`);
      });
      pipelineCache.set(key, promise);
    } else {
      emitProgress(request, {
        stage: "model",
        message: `Reusing cached browser translation model ${config.modelId}.`,
        detail: { modelId: config.modelId, dtype: config.dtype, status: "cached" },
      });
    }
    return promise;
  }

  return {
    capabilities: browserTranslationCapabilities,
    supports: webGpuAvailable,
    translateSegments,
  };
}

export async function supportsBrowserTranslation() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return false;
  }
  const adapter = await navigator.gpu.requestAdapter();
  return adapter !== null;
}

const defaultBrowserTranslationAdapter = createBrowserTranslationAdapter();

export async function translateBrowserSegments(segments, options = {}) {
  return defaultBrowserTranslationAdapter.translateSegments(segments, options);
}

async function loadBrowserTranslationPipeline({ modelId, dtype, onProgress }) {
  const transformers = await importTransformers();
  return transformers.pipeline("translation", modelId, {
    device: "webgpu",
    dtype,
    progress_callback: onProgress,
  });
}

function importTransformers() {
  transformersModulePromise ??= import(
    /* @vite-ignore */ DEFAULT_TRANSFORMERS_MODULE_URL
  );
  return transformersModulePromise;
}

function normalizeRequest(request) {
  const modelId = nonEmptyStringOrDefault(
    request.modelId,
    DEFAULT_BROWSER_TRANSLATION_MODEL_ID,
    "modelId",
  );
  const dtype = nonEmptyStringOrDefault(
    request.dtype,
    DEFAULT_BROWSER_TRANSLATION_DTYPE,
    "dtype",
  );
  const sourceLanguage = nonEmptyStringOrDefault(
    request.sourceLanguage,
    DEFAULT_SOURCE_LANGUAGE,
    "sourceLanguage",
  );
  const targetLanguage = nonEmptyStringOrDefault(
    request.targetLanguage,
    DEFAULT_TARGET_LANGUAGE,
    "targetLanguage",
  );
  const maxNewTokens = request.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;
  if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens <= 0) {
    throw new RangeError("Browser translation maxNewTokens must be a positive integer.");
  }
  if (sourceLanguage === targetLanguage) {
    throw new RangeError("Browser translation source and target languages must differ.");
  }
  return { modelId, dtype, sourceLanguage, targetLanguage, maxNewTokens };
}

function normalizeSegments(segments) {
  if (!Array.isArray(segments)) {
    throw new TypeError("Browser translation requires an array of text segments.");
  }
  return segments.map((segment, index) => {
    if (!segment || typeof segment !== "object") {
      throw new TypeError(`Browser translation segment ${index} must be an object.`);
    }
    if (typeof segment.id !== "string" && typeof segment.id !== "number") {
      throw new TypeError(`Browser translation segment ${index} requires a string or number id.`);
    }
    const text = typeof segment.text === "string" ? segment.text.trim() : "";
    if (!text) {
      throw new TypeError(`Browser translation segment ${index} requires non-empty text.`);
    }
    return { id: segment.id, text };
  });
}

function normalizeTranslationOutput(output, expectedCount) {
  const items = Array.isArray(output) ? output : [output];
  const normalized = items.map((item) => {
    const candidate = Array.isArray(item) ? item[0] : item;
    const text = candidate?.translation_text;
    if (typeof text !== "string" || text.trim().length === 0) {
      throw new Error("Browser translation returned an empty or malformed translation.");
    }
    return text.trim();
  });
  if (normalized.length !== expectedCount) {
    throw new Error(
      `Browser translation returned ${normalized.length} result(s) for ${expectedCount} segment(s).`,
    );
  }
  return normalized;
}

function translationResult(segments, translations, config) {
  return {
    segments: segments.map((segment, index) => ({
      id: segment.id,
      text: translations[index] ?? segment.text,
    })),
    sourceLanguage: config.sourceLanguage,
    targetLanguage: config.targetLanguage,
    attributes: {
      runtime: BROWSER_TRANSLATION_RUNTIME_ID,
      acceleration: "webgpu",
      modelId: config.modelId,
      dtype: config.dtype,
      modelProvisioning: "browser-cache",
    },
  };
}

function nonEmptyStringOrDefault(value, fallback, name) {
  if (value === undefined || value === null) return fallback;
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError(`Browser translation ${name} must be a non-empty string.`);
  }
  return value.trim();
}

function emitProgress(options, update) {
  if (typeof options.onProgress === "function") {
    options.onProgress(update);
  }
}

function formatError(error) {
  return error instanceof Error && error.message ? error.message : String(error);
}
