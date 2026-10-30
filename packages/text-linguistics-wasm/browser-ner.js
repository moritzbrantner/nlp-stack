// Consumer-facing browser NER adapter. Importing this module performs no network or model
// requests; the worker and model are created only by an explicit prepare()/analyze().
import {
  browserNerCapabilities,
  normalizeBrowserNerSegments,
  primaryLanguage,
  supportsBrowserNerLanguage,
} from "./browser-ner-core.js";

export {
  BROWSER_NER_MODEL,
  BROWSER_NER_PRODUCER,
  BROWSER_NER_TRANSFORMERS_MODULE_URL,
  browserNerCapabilities,
  entitiesFromTokenPredictions,
  supportsBrowserNerLanguage,
} from "./browser-ner-core.js";

function defaultCreateWorker() {
  if (typeof Worker === "undefined") {
    throw new Error("Web Workers are unavailable in this environment.");
  }
  return new Worker(new URL("./browser-ner-worker.js", import.meta.url), { type: "module" });
}

function abortError(signal) {
  const reason = signal?.reason;
  if (reason instanceof Error) return reason;
  return new DOMException("Browser NER request was canceled.", "AbortError");
}

export class BrowserNerError extends Error {
  constructor(message, stage) {
    super(message);
    this.name = "BrowserNerError";
    this.stage = stage;
  }
}

/**
 * Creates a browser NER adapter that owns one lazily spawned worker.
 *
 * Cancelling any request (via its `signal`) or calling `dispose()` terminates the worker
 * and rejects every in-flight request; the next request spawns a fresh worker (the model
 * is then reused from the browser cache), so a retry after cancel or failure works.
 */
export function createBrowserNer(options = {}) {
  const createWorker = options.createWorker ?? defaultCreateWorker;
  const transformersModuleUrl = options.transformersModuleUrl;
  let worker = null;
  let sequence = 0;
  const pending = new Map();

  function reset(error) {
    const current = worker;
    worker = null;
    current?.terminate();
    for (const request of pending.values()) request.reject(error);
    pending.clear();
  }

  function ensureWorker() {
    if (worker) return worker;
    const created = createWorker();
    created.addEventListener("message", (event) => {
      const data = event.data ?? {};
      const request = pending.get(data.id);
      if (!request) return;
      if (data.type === "progress") {
        request.onProgress?.({ stage: "model", detail: data.detail });
        return;
      }
      pending.delete(data.id);
      if (data.type === "error") request.reject(new BrowserNerError(data.message, data.stage));
      else request.resolve(data);
    });
    created.addEventListener("error", (event) => {
      event.preventDefault?.();
      reset(new BrowserNerError(event.message || "Browser NER worker failed.", "worker"));
    });
    worker = created;
    return created;
  }

  function send(type, payload, { signal, onProgress } = {}) {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(abortError(signal));
        return;
      }
      const id = ++sequence;
      const onAbort = () => {
        if (!pending.has(id)) return;
        reset(abortError(signal));
      };
      const settle = (fn) => (value) => {
        signal?.removeEventListener("abort", onAbort);
        fn(value);
      };
      pending.set(id, { resolve: settle(resolve), reject: settle(reject), onProgress });
      signal?.addEventListener("abort", onAbort, { once: true });
      try {
        ensureWorker().postMessage({ type, id, transformersModuleUrl, ...payload });
      } catch (error) {
        pending.get(id)?.reject(error instanceof Error ? error : new Error(String(error)));
        pending.delete(id);
      }
    });
  }

  function result(status, language, extra) {
    return {
      status,
      language,
      ...browserNerCapabilities(),
      ...extra,
    };
  }

  return {
    capabilities: browserNerCapabilities,
    supportsLanguage: supportsBrowserNerLanguage,

    /** Explicitly downloads/loads the pinned model, reporting Transformers.js progress. */
    async prepare(requestOptions = {}) {
      await send("prepare", {}, requestOptions);
      return browserNerCapabilities();
    },

    /**
     * Labels caller segments. `language` is required: unsupported or unknown languages
     * return `status: "unsupported-language"` without loading the model.
     */
    async analyze({ language, segments }, requestOptions = {}) {
      const normalized = normalizeBrowserNerSegments(segments);
      const primary = primaryLanguage(language);
      if (!supportsBrowserNerLanguage(primary)) {
        return result("unsupported-language", primary, {
          entities: [],
          segments: normalized.map(({ id }) => ({ id, entityCount: 0, truncated: false })),
        });
      }
      if (normalized.length === 0) {
        return result("completed", primary, { entities: [], segments: [] });
      }
      const response = await send("analyze", { segments: normalized }, requestOptions);
      return result("completed", primary, { entities: response.entities, segments: response.segments });
    },

    /** Terminates the worker and rejects any in-flight request. */
    dispose() {
      reset(new DOMException("Browser NER adapter was disposed.", "AbortError"));
    },
  };
}
