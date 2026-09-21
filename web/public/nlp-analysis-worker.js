import { createTextAnalysisRuntime } from "./nlp-semantic-runtime.js";

let runtime;
async function getRuntime() {
  if (!runtime) {
    const wasm = await import("./wasm/moenarch_text_analysis_wasm.js");
    const coreWasm = await import("./wasm/moenarch_text_core_wasm.js");
    await wasm.default();
    await coreWasm.default();
    runtime = createTextAnalysisRuntime(wasm, coreWasm);
  }
  return runtime;
}

// Serialize requests so their similarity matrices and model attempts do not overlap.
// The page owns cancellation by terminating this worker and its nested model worker.
let queue = Promise.resolve();
self.addEventListener("message", ({ data }) => {
  queue = queue.then(async () => {
    try {
      const analysis = await getRuntime();
      const value = data.method === "packageSurface"
        ? analysis.packageSurface()
        : await analysis.runOperation(data.request);
      self.postMessage({ id: data.id, value });
    } catch (error) {
      self.postMessage({ id: data.id, error: error instanceof Error ? error.message : String(error) });
    }
  });
});
