import { describe, expect, test } from "bun:test";

import {
  browserTranslationCapabilities,
  createBrowserTranslationAdapter,
} from "./translation.js";

describe("browser translation adapter", () => {
  test("advertises explicit WebGPU-only translation", () => {
    expect(browserTranslationCapabilities()).toEqual({
      runtime: "nlp-stack-transformers-js-webgpu-translation",
      requiredAcceleration: "webgpu",
      modelProvisioning: "browser-cache",
      defaultModelId: "onnx-community/opus-mt-de-en",
      defaultDtype: "q4",
      defaultSourceLanguage: "de",
      defaultTargetLanguage: "en",
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
    });
  });

  test("preserves segment identity and order while caching the model pipeline", async () => {
    let loads = 0;
    const progress = [];
    const adapter = createBrowserTranslationAdapter({
      webGpuAvailable: async () => true,
      loadPipeline: async ({ modelId, dtype, onProgress }) => {
        loads += 1;
        expect(modelId).toBe("onnx-community/opus-mt-de-en");
        expect(dtype).toBe("q4");
        onProgress({ status: "ready", modelId });
        return async (input, options) => {
          expect(options.max_new_tokens).toBe(64);
          const texts = Array.isArray(input) ? input : [input];
          return texts.map((text) => [{ translation_text: `EN: ${text}` }]);
        };
      },
    });

    const request = {
      sourceLanguage: "de",
      targetLanguage: "en",
      maxNewTokens: 64,
      onProgress: (update) => progress.push(update),
    };
    const first = await adapter.translateSegments(
      [
        { id: 7, text: "Guten Morgen" },
        { id: "segment-b", text: "Wie geht es dir?" },
      ],
      request,
    );
    const second = await adapter.translateSegments([{ id: 8, text: "Danke" }], request);

    expect(loads).toBe(1);
    expect(first.segments).toEqual([
      { id: 7, text: "EN: Guten Morgen" },
      { id: "segment-b", text: "EN: Wie geht es dir?" },
    ]);
    expect(second.segments).toEqual([{ id: 8, text: "EN: Danke" }]);
    expect(first.attributes).toEqual({
      runtime: "nlp-stack-transformers-js-webgpu-translation",
      acceleration: "webgpu",
      modelId: "onnx-community/opus-mt-de-en",
      dtype: "q4",
      modelProvisioning: "browser-cache",
    });
    expect(progress.some((update) => update.stage === "model" && update.detail?.status === "ready")).toBe(true);
    expect(progress.some((update) => update.stage === "model" && update.detail?.status === "cached")).toBe(true);
    expect(progress.some((update) => update.stage === "translate" && update.detail?.status === "done")).toBe(true);
  });

  test("keeps different model configurations in different cache entries", async () => {
    let loads = 0;
    const adapter = createBrowserTranslationAdapter({
      webGpuAvailable: async () => true,
      loadPipeline: async () => {
        loads += 1;
        return async () => [{ translation_text: "ok" }];
      },
    });

    await adapter.translateSegments([{ id: 1, text: "eins" }], {
      modelId: "model-a",
      dtype: "q4",
      sourceLanguage: "de",
      targetLanguage: "en",
    });
    await adapter.translateSegments([{ id: 2, text: "zwei" }], {
      modelId: "model-a",
      dtype: "fp16",
      sourceLanguage: "de",
      targetLanguage: "en",
    });
    await adapter.translateSegments([{ id: 3, text: "drei" }], {
      modelId: "model-b",
      dtype: "q4",
      sourceLanguage: "de",
      targetLanguage: "en",
    });

    expect(loads).toBe(3);
  });

  test("fails closed when WebGPU is unavailable", async () => {
    const adapter = createBrowserTranslationAdapter({
      webGpuAvailable: async () => false,
      loadPipeline: async () => {
        throw new Error("must not load");
      },
    });

    await expect(
      adapter.translateSegments([{ id: 1, text: "Hallo" }]),
    ).rejects.toThrow("No CPU, server, or Python fallback");
  });

  test("rejects malformed inputs and malformed model outputs", async () => {
    const adapter = createBrowserTranslationAdapter({
      webGpuAvailable: async () => true,
      loadPipeline: async () => async () => [{ generated_text: "wrong field" }],
    });

    await expect(adapter.translateSegments([{ id: 1, text: "Hallo" }], {
      sourceLanguage: "en",
      targetLanguage: "en",
    })).rejects.toThrow("source and target languages must differ");
    await expect(adapter.translateSegments([{ text: "missing id" }])).rejects.toThrow(
      "requires a string or number id",
    );
    await expect(adapter.translateSegments([{ id: 1, text: "Hallo" }])).rejects.toThrow(
      "empty or malformed translation",
    );
  });

  test("empty input is a deterministic no-op without loading a model", async () => {
    let loads = 0;
    const adapter = createBrowserTranslationAdapter({
      webGpuAvailable: async () => true,
      loadPipeline: async () => {
        loads += 1;
        return async () => [];
      },
    });

    const result = await adapter.translateSegments([]);
    expect(loads).toBe(0);
    expect(result.segments).toEqual([]);
  });
});
