import { assertAnalysisTextBudget } from "./nlp-analysis-budget.js";

export function createAnalysisWorkerRuntime() {
  let worker = null;
  let sequence = 0;
  const pending = new Map();

  function reset(error) {
    worker?.terminate();
    worker = null;
    for (const task of pending.values()) task.finish(error);
    pending.clear();
  }

  function getWorker() {
    if (worker) return worker;
    const instance = new Worker(new URL("./nlp-analysis-worker.js", import.meta.url), { type: "module" });
    worker = instance;
    instance.addEventListener("message", ({ data }) => {
      if (worker !== instance) return;
      const task = pending.get(data?.id);
      if (!task) return;
      task.finish(data.error ? new Error(data.error) : null, data.value);
    });
    instance.addEventListener("error", (event) => {
      if (worker === instance) reset(new Error(event.message || "Analysis worker failed."));
    });
    instance.addEventListener("messageerror", () => {
      if (worker === instance) reset(new Error("Unable to read the analysis worker response."));
    });
    return instance;
  }

  function call(method, request, signal) {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(new DOMException("Analysis cancelled.", "AbortError"));
        return;
      }
      if (method === "runOperation") assertAnalysisTextBudget(request);
      const id = ++sequence;
      const abort = () => reset(new DOMException("Analysis cancelled.", "AbortError"));
      // Bounds worker startup, queued work and operations that never reply.
      const timer = setTimeout(() => reset(new Error("Analysis exceeded 60 seconds. Choose a shorter passage or fewer documents.")), 60_000);
      pending.set(id, {
        finish(error, value) {
          clearTimeout(timer);
          signal?.removeEventListener("abort", abort);
          pending.delete(id);
          if (error) reject(error);
          else resolve(value);
        },
      });
      signal?.addEventListener("abort", abort, { once: true });
      try {
        getWorker().postMessage({ id, method, request });
      } catch (error) {
        reset(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  return {
    dispose: () => reset(new DOMException("Analysis cancelled.", "AbortError")),
    packageSurface: () => call("packageSurface"),
    runOperation: (request, { signal } = {}) => call("runOperation", request, signal),
  };
}
