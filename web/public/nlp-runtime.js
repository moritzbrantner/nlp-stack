const runtimeReadyEvent = "nlp-stack-text-core-ready";
const runtimeErrorEvent = "nlp-stack-text-core-error";

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

const ready = import(new URL("./wasm/moenarch_text_core_wasm.js", import.meta.url).href)
  .then(async (wasm) => {
    await wasm.default();
    return {
      packageSurface: () => fromWasmValue(wasm.packageSurface()),
      runOperation: (request) => fromWasmValue(wasm.runOperation(request)),
    };
  })
  .catch((error) => {
    globalThis.dispatchEvent(new Event(runtimeErrorEvent));
    throw error;
  });

globalThis.nlpStackTextCore = { ready };
globalThis.dispatchEvent(new Event(runtimeReadyEvent));
